// store/summary.rs
// Grouped success summary with a task-clustered bootstrap, and CSV export
// Reads the flat rows from Store::rows

use anyhow::Result;
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write, path::Path};

fn flag(v: &Value) -> bool {
    v.as_i64() == Some(1)
}
/// Measured attempts grouped by experiment condition.
pub fn summary(rows: &[Value]) -> Value {
    let mut groups: BTreeMap<String, Vec<&Value>> = Default::default();
    for r in rows
        .iter()
        .filter(|r| r["phase"] == "measured" && r["state"] == "complete")
    {
        let k = json!({"study":r["study"],"model":r["model"],"variant":r["variant"],"context":r["context"],"category":r["category"]}).to_string();
        groups.entry(k).or_default().push(r);
    }
    let mut out = Vec::new();
    for (key, rs) in groups {
        let mut success = 0;
        let mut unscored = 0;
        let mut statuses = BTreeMap::<String, usize>::new();
        let mut times = Vec::new();
        let mut calls = 0;
        let mut tool_calls = 0;
        let mut peak_bytes = None;
        let mut retrieved = 0;
        let mut retrieval_scored = 0;
        let mut per_task = BTreeMap::<String, Vec<f64>>::new();
        for r in &rs {
            if flag(&r["pending"]) {
                unscored += 1;
            } else {
                let pass = flag(&r["passed"]);
                success += usize::from(pass);
                per_task
                    .entry(r["case_id"].as_str().unwrap_or("").into())
                    .or_default()
                    .push(if pass { 1. } else { 0. });
            }
            *statuses
                .entry(r["status"].as_str().unwrap_or("unknown").into())
                .or_default() += 1;
            if let Some(t) = r["wall_ms"].as_f64() {
                times.push(t);
            }
            calls += r["model_calls"].as_u64().unwrap_or(0);
            tool_calls += r["tool_invocations"].as_u64().unwrap_or(0);
            if let Some(bytes) = r["combined_peak_bytes"].as_u64().filter(|n| *n > 0) {
                peak_bytes = Some(peak_bytes.map_or(bytes, |previous: u64| previous.max(bytes)));
            }
            if let Some(found) = r["retrieval_success"].as_i64() {
                retrieval_scored += 1;
                retrieved += found;
            }
        }
        times.sort_by(f64::total_cmp);
        let rates = per_task
            .values()
            .map(|a| a.iter().sum::<f64>() / a.len() as f64)
            .collect::<Vec<_>>();
        out.push(json!({
            "group": serde_json::from_str::<Value>(&key).expect("group key was serialized as JSON"),
            "attempts": rs.len(),
            "scored": rs.len() - unscored,
            "successes": success,
            "unscored": unscored,
            "success_rate": (rs.len() > unscored).then(|| success as f64 / (rs.len() - unscored) as f64),
            "task_bootstrap_95": cluster_interval(&rates),
            "unique_tasks": rates.len(),
            "retrieval_successes": retrieved,
            "retrieval_scored": retrieval_scored,
            "model_calls": calls,
            "tool_invocations": tool_calls,
            "combined_peak_bytes": peak_bytes,
            "median_wall_ms": times.get(times.len() / 2),
            "p95_wall_ms": times.get((times.len() as f64 * 0.95).ceil().max(1.) as usize - 1),
            "statuses": statuses
        }));
    }
    json!({"groups":out,"interval_method":"Percentile bootstrap over task-level mean success, 2000 seeded resamples; unavailable below 2 unique tasks. Repeats are clustered by task."})
}
/// Percentile bootstrap interval over per-task success rates.
pub fn cluster_interval(rates: &[f64]) -> Option<[f64; 2]> {
    if rates.len() < 2 {
        return None;
    }
    let mut seed = 42u64;
    let mut means = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sum = 0.;
        for _ in rates {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            sum += rates[(seed >> 32) as usize % rates.len()];
        }
        means.push(sum / rates.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    Some([means[50], means[1949]])
}
/// One CSV row per attempt with every flat column.
pub fn export_csv(rows: &[Value], out: &Path) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
    let columns = rows
        .first()
        .and_then(Value::as_object)
        .map(|o| o.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    writeln!(w, "{}", columns.join(","))?;
    for r in rows {
        let cells = columns.iter().map(|k| match &r[k] {
            Value::Null => String::new(),
            Value::String(s) => format!("\"{}\"", s.replace('"', "\"\"")),
            v => v.to_string(),
        });
        writeln!(w, "{}", cells.collect::<Vec<_>>().join(","))?;
    }
    Ok(())
}
