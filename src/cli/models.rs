//! `models add`: look up a Hugging Face GGUF, download it, and write a pinned entry.

use a2e::{config::Config, hub};
use anyhow::{bail, Result};
use clap::Args;
use serde_json::{json, Value};
use std::{io::Write, path::Path};

#[derive(Args)]
pub struct AddArgs {
    /// Hugging Face repository, with an optional quant: `unsloth/Qwen3-0.6B-GGUF:Q4_K_M`
    pub reference: String,
    /// Name for the model (default: the repository name)
    #[arg(long)]
    pub id: Option<String>,
    /// File name in the repository, when the quant does not pick one
    #[arg(long)]
    pub file: Option<String>,
    /// Tool-call format: functiongemma, qwen3, lfm2 or jinja (default: chosen from the repository name)
    #[arg(long)]
    pub adapter: Option<String>,
    /// Write the entry to models.toml instead of local.toml
    #[arg(long)]
    pub shared: bool,
}

pub fn add(config_path: &Path, a: &AddArgs) -> Result<Value> {
    let config = Config::load(config_path)?;
    let (repo, quant) = hub::parse_ref(&a.reference);
    let found = hub::resolve(&repo, quant.as_deref(), a.file.as_deref())?;
    let id = a.id.clone().unwrap_or_else(|| hub::default_id(&repo));
    if config.models.iter().any(|m| m.id == id) {
        bail!("a model named {id} already exists; choose another with --id");
    }
    let adapter = a
        .adapter
        .clone()
        .unwrap_or_else(|| hub::guess_adapter(&repo).to_string());
    a2e::adapter::Adapter::new(&adapter)?;
    let adv = &config.settings.advanced;
    let dest = config
        .resolve(&config.settings.host.models_dir)
        .join(&found.file.name);
    let hash = hub::download(
        &found,
        &dest,
        adv.download_timeout_s,
        adv.download_max_bytes,
    )?;
    let entry = hub::model_entry(&found, &id, &adapter, Some(hash.clone()), true);
    let target = config_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(if a.shared {
            "models.toml"
        } else {
            "local.toml"
        });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&target)?;
    write!(file, "\n[[models]]\n{}", toml::to_string(&entry)?)?;
    let size = std::fs::metadata(&dest)?.len();
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let needed = size + config.settings.limits.reserve_mb * 1024 * 1024;
    let warning = (system.available_memory() > 0 && needed > system.available_memory()).then(|| {
        format!(
            "the file ({} MiB) plus limits.reserve_mb is more than the {} MiB of memory available now",
            size >> 20,
            system.available_memory() >> 20
        )
    });
    Ok(json!({
        "id": id, "adapter": adapter, "quant": found.quant, "source": found.repo,
        "revision": found.revision, "file": found.file.name, "sha256": hash,
        "written_to": target, "warning": warning,
        "next": format!("a2e run smoke --model {id}"),
    }))
}

/// Resolves `hf:repo:quant` into an unpinned model entry added to `config`; returns its id.
pub fn unpinned(config: &mut Config, reference: &str) -> Result<String> {
    let (repo, quant) = hub::parse_ref(reference);
    let found = hub::resolve(&repo, quant.as_deref(), None)?;
    let id = format!("{}-{}", hub::default_id(&repo), found.quant.to_lowercase());
    let adapter = hub::guess_adapter(&repo);
    config.add_model(hub::model_entry(
        &found,
        &id,
        adapter,
        found.file.sha256.clone(),
        false,
    ));
    Ok(id)
}

/// Downloads the files of unpinned models that are not on disk yet.
pub fn fetch_unpinned(config: &Config, ids: &[String]) -> Result<()> {
    let adv = &config.settings.advanced;
    for id in ids {
        let m = config.model(id)?;
        if m.pinned {
            continue;
        }
        let path = config.model_path(m);
        if !path.exists() {
            eprintln!("downloading {} from {}", m.file, m.source);
            hub::download(
                &hub::found_from(m),
                &path,
                adv.download_timeout_s,
                adv.download_max_bytes,
            )?;
        }
    }
    Ok(())
}
