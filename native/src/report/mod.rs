// report/mod.rs
// Study report: Markdown tables and SVG plots from the SQLite results
// Aggregation by model, variant, and context; missing data is reported as n/a

mod markdown;
mod plots;
mod svg;

use crate::store::{cluster_interval, Store};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

/// Attempts that share model, variant, and context; `idx` points into `Report::rows`.
pub struct Group {
    pub model: String,
    pub variant: String,
    pub context: u64,
    pub idx: Vec<usize>,
    pub scored: usize,
    pub passes: usize,
    pub pending: usize,
    pub rate: Option<f64>,
    pub ci: Option<[f64; 2]>,
}
/// Resource series of the run drawn in the timeline plot.
pub struct Timeline {
    pub run_id: i64,
    pub label: String,
    pub seconds: Vec<f64>,
    pub harness_mb: Vec<f64>,
    pub inference_mb: Vec<f64>,
    pub temperature_c: Vec<Option<f64>>,
}
pub struct FailedAttempt {
    pub id: i64,
    pub case_id: String,
    pub model: String,
    pub variant: String,
    pub status: String,
    pub reason: String,
    pub class: String,
    pub checks: Vec<String>,
    pub calls: Vec<String>,
}
pub struct Report {
    pub study: String,
    pub created_at: String,
    pub definition: Value,
    pub rows: Vec<Value>,
    pub groups: Vec<Group>,
    pub models: Vec<String>,
    pub loads: Vec<Value>,
    pub judge_calls: Value,
    pub calibration: Option<bool>,
    pub failed: Vec<FailedAttempt>,
    pub timeline: Option<Timeline>,
    pub memory_limit_mb: Option<f64>,
}

/// Values of a numeric column; missing and null cells are skipped.
pub fn column(rows: &[&Value], key: &str) -> Vec<f64> {
    rows.iter().filter_map(|r| r[key].as_f64()).collect()
}
pub fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}
/// Nearest-rank percentile of unsorted values.
pub fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    Some(s[((s.len() as f64 * p).ceil().max(1.) as usize - 1).min(s.len() - 1)])
}
fn flag(v: &Value) -> bool {
    v.as_i64() == Some(1)
}
impl Group {
    pub fn rows<'a>(&self, all: &'a [Value]) -> Vec<&'a Value> {
        self.idx.iter().map(|i| &all[*i]).collect()
    }
    pub fn label(&self, many_contexts: bool) -> String {
        if many_contexts {
            format!("{} @{}", self.variant, self.context)
        } else {
            self.variant.clone()
        }
    }
}
impl Report {
    pub fn many_contexts(&self) -> bool {
        self.groups
            .iter()
            .map(|g| g.context)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1
    }
    pub fn model_index(&self, model: &str) -> usize {
        self.models.iter().position(|m| m == model).unwrap_or(0)
    }
}
/// Models and variants in the order the suite lists them.
fn suite_order(definition: &Value, key: &str, field: &str) -> Vec<String> {
    definition["suite"][key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| match field {
            "" => v.as_str().map(String::from),
            f => v[f].as_str().map(String::from),
        })
        .collect()
}
fn group_rows(rows: &[Value], definition: &Value) -> (Vec<Group>, Vec<String>) {
    let models = suite_order(definition, "models", "");
    let variants = suite_order(definition, "variants", "name");
    let position =
        |list: &[String], x: &str| list.iter().position(|m| m == x).unwrap_or(usize::MAX);
    let mut map = BTreeMap::<(String, String, u64), Vec<usize>>::new();
    for (i, r) in rows.iter().enumerate() {
        map.entry((
            r["model"].as_str().unwrap_or("").into(),
            r["variant"].as_str().unwrap_or("").into(),
            r["context"].as_u64().unwrap_or(0),
        ))
        .or_default()
        .push(i);
    }
    let mut groups = map
        .into_iter()
        .map(|((model, variant, context), idx)| {
            let mut per_task = BTreeMap::<&str, Vec<f64>>::new();
            let (mut scored, mut passes, mut pending) = (0, 0, 0);
            for i in &idx {
                let r = &rows[*i];
                if flag(&r["pending"]) {
                    pending += 1;
                    continue;
                }
                scored += 1;
                let pass = flag(&r["passed"]);
                passes += usize::from(pass);
                per_task
                    .entry(r["case_id"].as_str().unwrap_or(""))
                    .or_default()
                    .push(f64::from(u8::from(pass)));
            }
            let rates = per_task
                .values()
                .map(|a| a.iter().sum::<f64>() / a.len() as f64)
                .collect::<Vec<_>>();
            Group {
                model,
                variant,
                context,
                idx,
                scored,
                passes,
                pending,
                rate: (scored > 0).then(|| passes as f64 / scored as f64),
                ci: cluster_interval(&rates),
            }
        })
        .collect::<Vec<_>>();
    groups.sort_by_key(|g| {
        (
            position(&models, &g.model),
            position(&variants, &g.variant),
            g.context,
        )
    });
    let mut order = models;
    for g in &groups {
        if !order.contains(&g.model) {
            order.push(g.model.clone());
        }
    }
    (groups, order)
}
fn load(store: &Store, study: &str, run: Option<i64>) -> Result<Report> {
    let head = store.query(
        "SELECT created_at,definition FROM studies WHERE id=?",
        [study],
    )?;
    let head = head.first().context("study not found")?;
    let definition: Value = serde_json::from_str(head["definition"].as_str().unwrap_or("{}"))?;
    let rows = store
        .study_rows(study)?
        .into_iter()
        .filter(|r| r["phase"] == "measured" && r["state"] == "complete")
        .collect::<Vec<_>>();
    let (groups, models) = group_rows(&rows, &definition);
    let loads = store.query(
        "SELECT model,context,load_ms,peak_bytes,error FROM model_loads WHERE study=? ORDER BY id",
        [study],
    )?;
    let judge_calls = store
        .query(
            "SELECT count(*) AS calls,sum(ok) AS ok,sum(prompt_tokens) AS prompt_tokens,sum(completion_tokens) AS completion_tokens,sum(cost) AS cost,avg(wall_ms) AS mean_wall_ms FROM judge_calls WHERE run_id IN (SELECT id FROM runs WHERE study=?)",
            [study],
        )?
        .pop()
        .unwrap_or(Value::Null);
    let calibration = store
        .query(
            "SELECT json_extract(j.response,'$.calibration.passed') AS passed FROM run_judgments rj JOIN judgments j ON j.cache_key=rj.cache_key JOIN runs r ON r.id=rj.run_id WHERE r.study=? LIMIT 1",
            [study],
        )?
        .first()
        .and_then(|r| r["passed"].as_i64())
        .map(|p| p == 1);
    let mut failed = Vec::new();
    for r in rows
        .iter()
        .filter(|r| !flag(&r["passed"]) && !flag(&r["pending"]))
        .take(40)
    {
        let id = r["id"].as_i64().unwrap_or_default();
        let text = |sql: &str, key: &str| -> Result<Vec<String>> {
            Ok(store
                .query(sql, [id])?
                .iter()
                .filter_map(|c| c[key].as_str().map(String::from))
                .collect())
        };
        failed.push(FailedAttempt {
            id,
            case_id: r["case_id"].as_str().unwrap_or("").into(),
            model: r["model"].as_str().unwrap_or("").into(),
            variant: r["variant"].as_str().unwrap_or("").into(),
            status: r["status"].as_str().unwrap_or("").into(),
            reason: r["reason"].as_str().unwrap_or("").into(),
            class: r["failure_class"].as_str().unwrap_or("").into(),
            checks: text(
                "SELECT label FROM checks WHERE run_id=? AND passed=0 ORDER BY idx",
                "label",
            )?,
            calls: text(
                "SELECT tool || CASE WHEN ok=1 THEN '' ELSE ' (failed)' END AS call FROM tool_calls WHERE run_id=? ORDER BY idx",
                "call",
            )?,
        });
    }
    let timeline = timeline(store, &rows, run)?;
    Ok(Report {
        study: study.into(),
        created_at: head["created_at"].as_str().unwrap_or("").into(),
        memory_limit_mb: definition["runtime"]["memory_limit_mb"].as_f64(),
        definition,
        rows,
        groups,
        models,
        loads,
        judge_calls,
        calibration,
        failed,
        timeline,
    })
}
/// Samples of the requested run, or of the slowest measured run that has any.
fn timeline(store: &Store, rows: &[Value], run: Option<i64>) -> Result<Option<Timeline>> {
    let chosen = match run {
        Some(id) => rows.iter().find(|r| r["id"].as_i64() == Some(id)),
        None => rows
            .iter()
            .filter(|r| r["sample_count"].as_i64().unwrap_or(0) > 0)
            .max_by(|a, b| {
                a["wall_ms"]
                    .as_f64()
                    .unwrap_or(0.)
                    .total_cmp(&b["wall_ms"].as_f64().unwrap_or(0.))
            }),
    };
    let Some(r) = chosen else {
        if run.is_some() {
            bail!("run is not a measured attempt of this study");
        }
        return Ok(None);
    };
    let id = r["id"].as_i64().unwrap_or_default();
    let samples = store.query(
        "SELECT t_ms,harness_bytes,inference_bytes,temperature_c FROM samples WHERE run_id=? ORDER BY idx",
        [id],
    )?;
    if samples.is_empty() {
        return Ok(None);
    }
    let mb = |s: &Value, k: &str| s[k].as_f64().unwrap_or(0.) / 1_048_576.;
    Ok(Some(Timeline {
        run_id: id,
        label: format!(
            "{} · {} · {}",
            r["case_id"].as_str().unwrap_or(""),
            r["model"].as_str().unwrap_or(""),
            r["variant"].as_str().unwrap_or("")
        ),
        seconds: samples
            .iter()
            .map(|s| s["t_ms"].as_f64().unwrap_or(0.) / 1000.)
            .collect(),
        harness_mb: samples.iter().map(|s| mb(s, "harness_bytes")).collect(),
        inference_mb: samples.iter().map(|s| mb(s, "inference_bytes")).collect(),
        temperature_c: samples
            .iter()
            .map(|s| s["temperature_c"].as_f64())
            .collect(),
    }))
}
/// Writes `report.md` and `plots/*.svg`; the newest study is used unless one is named.
pub fn generate(
    store: &Store,
    study: Option<&str>,
    out: Option<&Path>,
    default_root: &Path,
    run: Option<i64>,
) -> Result<Value> {
    let found = store.query(
        "SELECT id FROM studies WHERE id LIKE ?||'%' ORDER BY created_at DESC, rowid DESC LIMIT 1",
        [study.unwrap_or("")],
    )?;
    let id = found
        .first()
        .and_then(|r| r["id"].as_str())
        .context("no study found; run a suite first")?
        .to_owned();
    let report = load(store, &id, run)?;
    let dir = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_root.join("reports").join(&id[..12.min(id.len())]));
    let plot_dir = dir.join("plots");
    std::fs::create_dir_all(&plot_dir)?;
    let mut written = Vec::new();
    for (name, svg) in plots::all(&report) {
        std::fs::write(plot_dir.join(&name), svg)?;
        written.push(name);
    }
    std::fs::write(dir.join("report.md"), markdown::render(&report, &written))?;
    Ok(
        json!({"study":id,"report":dir.join("report.md"),"plots":written,"attempts":report.rows.len()}),
    )
}
