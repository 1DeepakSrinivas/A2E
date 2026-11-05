// report/markdown.rs
// Markdown report: tables from the grouped attempts, plot links, failed-case appendix
// Every missing value prints as n/a, never as zero

use super::{column, mean, percentile, Group, Report};
use serde_json::Value;
use std::{collections::BTreeMap, fmt::Write};

const NA: &str = "n/a";
const CATEGORIES: [&str; 8] = [
    "simple",
    "multiple",
    "irrelevance",
    "missing_param",
    "chain",
    "recovery",
    "robustness",
    "web",
];
const MB: f64 = 1_048_576.;

fn cell(v: Option<f64>, digits: usize) -> String {
    v.map_or(NA.into(), |v| format!("{v:.digits$}"))
}
fn percent(v: Option<f64>) -> String {
    v.map_or(NA.into(), |v| format!("{:.1}%", v * 100.))
}
fn escape(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}
fn table(out: &mut String, headers: &[&str], rows: &[Vec<String>]) {
    let _ = writeln!(out, "| {} |", headers.join(" | "));
    let _ = writeln!(out, "|{}|", vec!["---"; headers.len()].join("|"));
    for r in rows {
        let _ = writeln!(
            out,
            "| {} |",
            r.iter().map(|c| escape(c)).collect::<Vec<_>>().join(" | ")
        );
    }
    out.push('\n');
}
fn label(r: &Report, g: &Group) -> String {
    format!("{} · {}", g.model, g.label(r.many_contexts()))
}
fn text<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    path.iter().fold(v, |a, k| &a[*k]).as_str().unwrap_or(NA)
}
fn plot(out: &mut String, written: &[String], name: &str, alt: &str) {
    if written.iter().any(|w| w == name) {
        let _ = writeln!(out, "![{alt}](plots/{name})\n");
    }
}
fn sum(rows: &[&Value], key: &str) -> Option<f64> {
    let v = column(rows, key);
    (!v.is_empty()).then(|| v.iter().sum())
}

pub fn render(r: &Report, written: &[String]) -> String {
    let mut out = String::new();
    header(&mut out, r);
    headline(&mut out, r, written);
    categories(&mut out, r, written);
    failures(&mut out, r, written);
    latency(&mut out, r, written);
    calls(&mut out, r, written);
    memory(&mut out, r, written);
    thermal(&mut out, r);
    judge(&mut out, r);
    appendix(&mut out, r);
    out
}

fn header(out: &mut String, r: &Report) {
    let d = &r.definition;
    let _ = writeln!(
        out,
        "# Study report {}\n",
        &r.study[..12.min(r.study.len())]
    );
    let _ = writeln!(
        out,
        "Created {}. Measured attempts in this report: {}.\n",
        r.created_at,
        r.rows.len()
    );
    let binary = text(d, &["binary_sha256"]);
    let rows = vec![
        vec!["Suite".into(), text(d, &["suite", "name"]).into()],
        vec!["Split".into(), text(d, &["suite", "split"]).into()],
        vec!["Device".into(), text(d, &["device"]).into()],
        vec!["Kernel".into(), text(d, &["kernel"]).into()],
        vec!["Architecture".into(), text(d, &["arch"]).into()],
        vec![
            "Total memory".into(),
            format!(
                "{} MB",
                cell(d["total_memory_bytes"].as_f64().map(|b| b / MB), 0)
            ),
        ],
        vec![
            "Memory limit".into(),
            format!("{} MB", cell(r.memory_limit_mb, 0)),
        ],
        vec![
            "llama.cpp commit".into(),
            text(d, &["runtime", "llama_commit"]).into(),
        ],
        vec![
            "Runtime version".into(),
            text(d, &["runtime_version"]).into(),
        ],
        vec![
            "Binary SHA-256".into(),
            binary[..binary.len().min(16)].into(),
        ],
        vec!["Study id".into(), r.study.clone()],
    ];
    table(out, &["Field", "Value"], &rows);
    let models = r
        .models
        .iter()
        .map(|m| {
            let load = r.loads.iter().rfind(|l| l["model"] == m.as_str());
            vec![
                m.clone(),
                text(d, &["artifacts", m, "sha256"])
                    .chars()
                    .take(16)
                    .collect(),
                cell(
                    load.and_then(|l| l["load_ms"].as_f64()).map(|v| v / 1000.),
                    1,
                ),
                cell(
                    load.and_then(|l| l["peak_bytes"].as_f64()).map(|v| v / MB),
                    0,
                ),
            ]
        })
        .collect::<Vec<_>>();
    table(
        out,
        &["Model", "SHA-256", "Load seconds", "Load peak MB"],
        &models,
    );
}

fn headline(out: &mut String, r: &Report, written: &[String]) {
    out.push_str("## Headline\n\n");
    out.push_str("Success is the share of scored attempts that passed every check. Attempts waiting for a judge are counted as pending and left out of the rate. The interval is a bootstrap over tasks, so repeats of one task count once.\n\n");
    let rows = r
        .groups
        .iter()
        .map(|g| {
            let wall = column(&g.rows(&r.rows), "wall_ms");
            vec![
                g.model.clone(),
                g.variant.clone(),
                g.context.to_string(),
                g.idx.len().to_string(),
                g.scored.to_string(),
                g.pending.to_string(),
                percent(g.rate),
                g.ci.map_or(NA.into(), |[a, b]| {
                    format!("{:.1}% to {:.1}%", a * 100., b * 100.)
                }),
                cell(percentile(&wall, 0.5).map(|v| v / 1000.), 2),
            ]
        })
        .collect::<Vec<_>>();
    table(
        out,
        &[
            "Model",
            "Variant",
            "Context",
            "Attempts",
            "Scored",
            "Pending",
            "Success",
            "95% interval",
            "Median wall s",
        ],
        &rows,
    );
    plot(
        out,
        written,
        "success_by_model.svg",
        "Success rate by model and variant",
    );
}

fn categories(out: &mut String, r: &Report, written: &[String]) {
    out.push_str("## Success by category\n\n");
    let mut counts = BTreeMap::<(&str, usize), (usize, usize)>::new();
    for (gi, g) in r.groups.iter().enumerate() {
        for row in g.rows(&r.rows) {
            if row["pending"].as_i64() == Some(1) {
                continue;
            }
            let c = counts
                .entry((row["category"].as_str().unwrap_or(""), gi))
                .or_default();
            c.1 += 1;
            c.0 += usize::from(row["passed"].as_i64() == Some(1));
        }
    }
    let labels = r.groups.iter().map(|g| label(r, g)).collect::<Vec<_>>();
    let mut headers = vec!["Category"];
    headers.extend(labels.iter().map(String::as_str));
    let rows = CATEGORIES
        .iter()
        .filter(|c| counts.keys().any(|(k, _)| k == *c))
        .map(|c| {
            let mut row = vec![c.to_string()];
            for gi in 0..r.groups.len() {
                row.push(match counts.get(&(*c, gi)) {
                    Some((p, n)) if *n > 0 => format!("{p}/{n}"),
                    _ => NA.into(),
                });
            }
            row
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        out.push_str("No scored attempts.\n\n");
    } else {
        table(out, &headers, &rows);
    }
    plot(
        out,
        written,
        "category_heatmap.svg",
        "Success rate by category",
    );
}

fn failures(out: &mut String, r: &Report, written: &[String]) {
    out.push_str("## Failure classes\n\n");
    let mut totals = BTreeMap::<String, usize>::new();
    for row in &r.rows {
        if let Some(c) = row["failure_class"].as_str() {
            *totals.entry(c.into()).or_default() += 1;
        }
    }
    if totals.is_empty() {
        out.push_str("No failed attempts.\n\n");
        return;
    }
    let classes = totals.keys().cloned().collect::<Vec<_>>();
    let mut headers = vec!["Condition"];
    headers.extend(classes.iter().map(String::as_str));
    let rows = r
        .groups
        .iter()
        .map(|g| {
            let mut row = vec![label(r, g)];
            for c in &classes {
                let n = g
                    .rows(&r.rows)
                    .iter()
                    .filter(|x| x["failure_class"] == c.as_str())
                    .count();
                row.push(n.to_string());
            }
            row
        })
        .collect::<Vec<_>>();
    table(out, &headers, &rows);
    plot(
        out,
        written,
        "failure_classes.svg",
        "Failure classes per condition",
    );
}

fn latency(out: &mut String, r: &Report, written: &[String]) {
    out.push_str("## Latency\n\n");
    out.push_str("Times are per attempt. Tokens per second is generated tokens over decode time. Harness share is the part of wall time spent outside the model, the tools, and parsing.\n\n");
    let rows = r
        .groups
        .iter()
        .map(|g| {
            let rows = g.rows(&r.rows);
            let wall = column(&rows, "wall_ms");
            let tokens = sum(&rows, "generated_tokens");
            let decode = sum(&rows, "decode_ms");
            let share = match (sum(&rows, "harness_other_ms"), sum(&rows, "wall_ms")) {
                (Some(a), Some(b)) if b > 0. => Some(a / b),
                _ => None,
            };
            vec![
                label(r, g),
                cell(percentile(&wall, 0.5).map(|v| v / 1000.), 2),
                cell(percentile(&wall, 0.95).map(|v| v / 1000.), 2),
                cell(mean(&column(&rows, "prefill_ms")), 0),
                cell(mean(&column(&rows, "decode_ms")), 0),
                cell(mean(&column(&rows, "tokenize_ms")), 0),
                match (tokens, decode) {
                    (Some(t), Some(d)) if d > 0. => cell(Some(t / (d / 1000.)), 1),
                    _ => NA.into(),
                },
                percent(share),
            ]
        })
        .collect::<Vec<_>>();
    table(
        out,
        &[
            "Condition",
            "Median s",
            "p95 s",
            "Prefill ms",
            "Decode ms",
            "Tokenize ms",
            "Tokens per s",
            "Harness share",
        ],
        &rows,
    );
    plot(out, written, "latency.svg", "Median and p95 wall time");
}

fn calls(out: &mut String, r: &Report, written: &[String]) {
    out.push_str("## Calls and tokens\n\n");
    out.push_str("Means per attempt.\n\n");
    let rows = r
        .groups
        .iter()
        .map(|g| {
            let rows = g.rows(&r.rows);
            let m = |k: &str, d: usize| cell(mean(&column(&rows, k)), d);
            vec![
                label(r, g),
                m("model_calls", 1),
                m("tool_invocations", 1),
                m("valid_tool_calls", 1),
                m("argument_errors", 2),
                m("repairs", 2),
                m("prompt_tokens", 0),
                m("generated_tokens", 0),
                m("cached_tokens", 0),
            ]
        })
        .collect::<Vec<_>>();
    table(
        out,
        &[
            "Condition",
            "Model calls",
            "Tool calls",
            "Valid calls",
            "Argument errors",
            "Repairs",
            "Prompt tokens",
            "Generated tokens",
            "Cached tokens",
        ],
        &rows,
    );
    plot(
        out,
        written,
        "tokens_vs_wall.svg",
        "Generated tokens against wall time",
    );
}

fn memory(out: &mut String, r: &Report, written: &[String]) {
    out.push_str("## Memory\n\n");
    out.push_str("Peak is the highest combined harness and inference resident memory in any sample. Available minimum is the lowest system memory seen.\n\n");
    let rows = r
        .groups
        .iter()
        .map(|g| {
            let rows = g.rows(&r.rows);
            let peak = column(&rows, "combined_peak_bytes")
                .into_iter()
                .filter(|b| *b > 0.)
                .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))));
            let available = column(&rows, "available_min_bytes")
                .into_iter()
                .filter(|b| *b > 0.)
                .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.min(b))));
            let exceeded = column(&rows, "memory_exceeded");
            vec![
                label(r, g),
                cell(peak.map(|b| b / MB), 0),
                cell(r.memory_limit_mb, 0),
                cell(available.map(|b| b / MB), 0),
                if exceeded.is_empty() {
                    NA.into()
                } else {
                    format!("{:.0}", exceeded.iter().sum::<f64>())
                },
            ]
        })
        .collect::<Vec<_>>();
    table(
        out,
        &[
            "Condition",
            "Peak MB",
            "Limit MB",
            "Available min MB",
            "Attempts over limit",
        ],
        &rows,
    );
    plot(out, written, "memory.svg", "Peak memory against the limit");
}

fn thermal(out: &mut String, r: &Report) {
    out.push_str("## Temperature and energy\n\n");
    out.push_str("Devices without a thermal sensor or a power meter show n/a.\n\n");
    let rows = r
        .groups
        .iter()
        .map(|g| {
            let rows = g.rows(&r.rows);
            let peak = column(&rows, "peak_temperature_c")
                .into_iter()
                .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))));
            let throttled = rows
                .iter()
                .filter_map(|x| x["throttled"].as_str())
                .filter(|t| !t.is_empty() && *t != "0x0")
                .count();
            let reported = rows.iter().filter(|x| x["throttled"].is_string()).count();
            let scope = rows
                .iter()
                .find_map(|x| x["power_scope"].as_str())
                .unwrap_or(NA);
            vec![
                label(r, g),
                cell(peak, 1),
                if reported == 0 {
                    NA.into()
                } else {
                    throttled.to_string()
                },
                cell(sum(&rows, "energy_j"), 1),
                scope.into(),
            ]
        })
        .collect::<Vec<_>>();
    table(
        out,
        &[
            "Condition",
            "Peak °C",
            "Throttled attempts",
            "Energy J",
            "Power scope",
        ],
        &rows,
    );
    if let Some(t) = &r.timeline {
        let _ = writeln!(
            out,
            "![Resources over run {}](plots/resource_timeline.svg)\n",
            t.run_id
        );
    }
}

fn judge(out: &mut String, r: &Report) {
    let judged = r.rows.iter().filter(|x| !x["verdict"].is_null()).count();
    let pending = r
        .rows
        .iter()
        .filter(|x| x["pending"].as_i64() == Some(1))
        .count();
    if judged == 0 && pending == 0 {
        return;
    }
    out.push_str("## Judge scores\n\n");
    let calibration = match r.calibration {
        Some(true) => "passed",
        Some(false) => "not passed, so verdicts are provisional",
        None => NA,
    };
    let _ = writeln!(
        out,
        "Calibration: {calibration}. Unscored attempts: {pending}.\n"
    );
    let mut verdicts = BTreeMap::<String, [usize; 2]>::new();
    for row in &r.rows {
        if let Some(v) = row["verdict"].as_str() {
            verdicts.entry(v.into()).or_default()[0] += 1;
        }
    }
    let rows = verdicts
        .into_iter()
        .map(|(v, n)| vec![v, n[0].to_string()])
        .collect::<Vec<_>>();
    table(out, &["Verdict", "Attempts"], &rows);
    let j = &r.judge_calls;
    let rows = vec![vec![
        j["calls"].as_i64().map_or(NA.into(), |n| n.to_string()),
        j["ok"].as_i64().map_or(NA.into(), |n| n.to_string()),
        cell(j["prompt_tokens"].as_f64(), 0),
        cell(j["completion_tokens"].as_f64(), 0),
        cell(j["cost"].as_f64(), 4),
        cell(j["mean_wall_ms"].as_f64(), 0),
    ]];
    table(
        out,
        &[
            "Judge calls",
            "Valid",
            "Prompt tokens",
            "Completion tokens",
            "Cost",
            "Mean wall ms",
        ],
        &rows,
    );
}

fn appendix(out: &mut String, r: &Report) {
    out.push_str("## Failed attempts\n\n");
    if r.failed.is_empty() {
        out.push_str("No failed attempts.\n");
        return;
    }
    let total = r
        .rows
        .iter()
        .filter(|x| x["passed"].as_i64() != Some(1) && x["pending"].as_i64() != Some(1))
        .count();
    let _ = writeln!(
        out,
        "Showing {} of {total} failed attempts.\n",
        r.failed.len()
    );
    for f in &r.failed {
        let _ = writeln!(
            out,
            "### Run {}: {} ({} · {})\n",
            f.id, f.case_id, f.model, f.variant
        );
        let _ = writeln!(
            out,
            "- Status: {}. Class: {}. Reason: {}.",
            f.status, f.class, f.reason
        );
        let _ = writeln!(
            out,
            "- Failed checks: {}",
            if f.checks.is_empty() {
                NA.into()
            } else {
                f.checks.join("; ")
            }
        );
        let _ = writeln!(
            out,
            "- Tool calls: {}\n",
            if f.calls.is_empty() {
                "none".into()
            } else {
                f.calls.join(", ")
            }
        );
    }
}
