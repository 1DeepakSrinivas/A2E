//! Starts the inference server for a model and context size and records the load.

use crate::{
    config::Config,
    inference::Server,
    measure::{Monitor, Resources},
    upload::{
        payload::{self, LoadRecord},
        Resume, Uploader,
    },
};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::{
    path::Path,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

/// Result of a successful model load.
pub(super) struct Loaded {
    pub(super) load_ms: f64,
    pub(super) resources: Resources,
    pub(super) template: String,
    pub(super) props: Value,
}
/// Longest wait for a cancelled call to leave its slot before the server is restarted.
pub(super) const SETTLE: Duration = Duration::from_secs(60);
pub(super) struct LoadRequest<'a> {
    pub(super) config: &'a Config,
    pub(super) model: &'a crate::config::Model,
    pub(super) ctx: usize,
    pub(super) group: &'a Path,
    pub(super) study: &'a str,
    pub(super) cancel: &'a Arc<AtomicBool>,
}
/// Starts the inference server and records the load, failed or not, under a new load ID.
pub(super) fn launch(
    x: &LoadRequest,
    resume: &mut Resume,
    uploader: &mut Uploader,
) -> Result<(Result<(Server, Loaded)>, String)> {
    warn_if_tight(x.config, x.model);
    let launched = (|| -> Result<(Server, Loaded)> {
        let mut server = Server::spawn(x.config, x.model, x.ctx, x.group)?;
        let monitor = Monitor::start(server.pid(), &x.config.runtime, x.cancel.clone())?;
        let ready = server.ready(x.config.runtime.startup_seconds, x.cancel);
        let resources = monitor.finish();
        if resources.memory_exceeded {
            bail!("model_load_memory_budget");
        }
        let mut props = ready?;
        let template = props["chat_template"]
            .as_str()
            .context("missing model template")?
            .to_owned();
        std::fs::write(x.group.join("chat_template.jinja"), &template)?;
        if let Some(p) = props.as_object_mut() {
            p.remove("chat_template");
        }
        let load_ms = server.started.elapsed().as_secs_f64() * 1000.;
        Ok((
            server,
            Loaded {
                load_ms,
                resources,
                template,
                props,
            },
        ))
    })();
    let load_id = resume.next_load_id(x.study)?;
    let no_resources = Resources::default();
    let error = launched.as_ref().err().map(ToString::to_string);
    let record = match &launched {
        Ok((server, l)) => LoadRecord {
            id: &load_id,
            study: x.study,
            model: &x.model.id,
            context: x.ctx,
            load_ms: l.load_ms,
            resources: &l.resources,
            template: Some(&l.template),
            server_version: Some(&server.version),
            props: Some(&l.props),
            error: None,
        },
        Err(_) => LoadRecord {
            id: &load_id,
            study: x.study,
            model: &x.model.id,
            context: x.ctx,
            load_ms: 0.,
            resources: &no_resources,
            template: None,
            server_version: None,
            props: None,
            error: error.as_deref(),
        },
    };
    uploader.submit(payload::load(&record))?;
    Ok((launched, load_id))
}
/// Warns when the model file cannot stay in the page cache next to the reserve.
/// A mapped model that does not fit is paged from storage on every token.
pub(super) fn warn_if_tight(config: &Config, model: &crate::config::Model) {
    let Ok(file) = std::fs::metadata(config.model_path(model)) else {
        return;
    };
    let available = crate::measure::available_memory();
    let needed = file.len() + config.runtime.reserve_mb * 1024 * 1024;
    if available > 0 && needed > available {
        eprintln!(
            "warning: {} needs {} MiB for its weights plus the {} MiB reserve, before the context cache, but {} MiB is available; expect paging and call timeouts",
            model.id,
            file.len() >> 20,
            config.runtime.reserve_mb,
            available >> 20
        );
    }
}
