//! Maps studies, model loads and finished attempts to Worker request bodies.
//!
//! Large resource series are split into separate sample requests.

use super::client::Request;
use crate::{
    case::Case,
    config::Variant,
    measure::{Resources, Sample},
    runner::{Outcome, Status},
    store::{tag, Phase},
};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

const BODY_LIMIT: usize = 2_000_000;
const SAMPLE_CHUNK: usize = 5_000;

/// Identity of one attempt.
pub struct RunMeta<'a> {
    pub study: &'a str,
    pub job: &'a str,
    pub attempt: u32,
    pub phase: Phase,
    pub model: &'a str,
    pub context: usize,
    pub variant: &'a Variant,
    pub repeat: usize,
    pub case: &'a Case,
    pub load_id: Option<&'a str>,
    pub trace: &'a Path,
}
/// One model load: timing, memory, and the server's template.
pub struct LoadRecord<'a> {
    pub id: &'a str,
    pub study: &'a str,
    pub model: &'a str,
    pub context: usize,
    pub load_ms: f64,
    pub resources: &'a Resources,
    pub template: Option<&'a str>,
    pub server_version: Option<&'a str>,
    pub props: Option<&'a Value>,
    pub error: Option<&'a str>,
}
pub fn run_id(study: &str, job: &str, attempt: u32) -> String {
    format!(
        "{}-{}-{attempt}",
        &study[..12.min(study.len())],
        &job[..16.min(job.len())]
    )
}
fn sample_rows(r: &Resources) -> Vec<Value> {
    r.series
        .iter()
        .enumerate()
        .map(|(i, s): (usize, &Sample)| {
            json!({"idx":i,"t_ms":s.t_ms,"harness_bytes":s.harness_bytes,"inference_bytes":s.inference_bytes,
                "available_bytes":s.available_bytes,"cpu_ms":s.cpu_ms,"temperature_c":s.temperature_c,
                "throttled":s.throttled,"power_w":s.power_w,"minor_faults":s.minor_faults,"major_faults":s.major_faults})
        })
        .collect()
}
/// Keeps samples inline when the body is small, otherwise returns them as follow-up requests.
fn spill(body: &mut Value, key: &str, samples: Vec<Value>, path: &str) -> Vec<Request> {
    let size = body.to_string().len() + Value::Array(samples.clone()).to_string().len();
    if size <= BODY_LIMIT {
        body[key] = Value::Array(samples);
        return Vec::new();
    }
    body[key] = json!([]);
    samples
        .chunks(SAMPLE_CHUNK)
        .map(|c| Request {
            method: "POST".into(),
            path: path.into(),
            body: json!({"samples":c}),
        })
        .collect()
}
pub fn study(id: &str, definition: &Value, cases: &[Case]) -> Result<Request> {
    let rows = cases
        .iter()
        .map(|c| {
            Ok(
                json!({"id":c.id,"hash":crate::canonical_hash(c)?,"category":tag(&c.category),
                "split":tag(&c.split),"tags":c.tags,"definition":c}),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Request {
        method: "PUT".into(),
        path: format!("/v1/studies/{id}"),
        body: json!({"definition":definition,"cases":rows}),
    })
}
pub fn load(l: &LoadRecord) -> Vec<Request> {
    let mut body = json!({"id":l.id,"study":l.study,"model":l.model,"context":l.context,"load_ms":l.load_ms,
        "peak_bytes":l.resources.inference_peak_bytes,"template_sha256":l.template.map(|t| crate::digest(t.as_bytes())),
        "template":l.template,"server_version":l.server_version,"props":l.props,"error":l.error});
    let extra = spill(
        &mut body,
        "samples",
        sample_rows(l.resources),
        &format!("/v1/loads/{}/samples", l.id),
    );
    let mut out = vec![Request {
        method: "POST".into(),
        path: "/v1/loads".into(),
        body,
    }];
    out.extend(extra);
    out
}
pub fn run(m: &RunMeta, out: &Outcome) -> Result<Vec<Request>> {
    let id = run_id(m.study, m.job, m.attempt);
    let r = &out.resources;
    let state = if out.status == Status::Interrupted {
        "interrupted"
    } else {
        "complete"
    };
    let mut run = json!({"id":id,"study":m.study,"job":m.job,"attempt":m.attempt,"state":state,"phase":tag(&m.phase),
        "model":m.model,"context":m.context,"variant":m.variant.name,"variant_config":m.variant,"mode":tag(&m.variant.mode),
        "repeat_index":m.repeat,"case_id":m.case.id,"category":tag(&m.case.category),"split":tag(&m.case.split),
        "load_id":m.load_id,"trace":m.trace.to_string_lossy(),
        "status":tag(&out.status),"reason":out.reason,"failure_class":out.failure_class.as_ref().map(tag),
        "milestone_score":out.milestone_score,"attempted_calls":out.attempted_calls,"abstained":out.abstained,
        "expected_calls":m.case.scope.expected_calls,"answer":out.answer,"evidence":out.evidence,"final_state":out.final_state,"retrieval_success":out.retrieval_success});
    let counts = json!({"model_calls":out.model_calls,"tool_invocations":out.tool_invocations,
        "valid_tool_calls":out.valid_tool_calls,"argument_errors":out.argument_errors,"repairs":out.repairs,
        "prompt_tokens":out.prompt_tokens,"generated_tokens":out.generated_tokens,"cached_tokens":out.cached_tokens});
    let timing = json!({"prefill_ms":out.prefill_ms,"decode_ms":out.decode_ms,"http_ms":out.http_ms,
        "template_ms":out.template_ms,"tokenize_ms":out.tokenize_ms,"parse_ms":out.parse_ms,"validate_ms":out.validate_ms,
        "tool_ms":out.tool_ms,"tool_setup_ms":out.tool_setup_ms,"wall_ms":out.wall_ms,"harness_other_ms":out.harness_other_ms});
    let resources = json!({"harness_peak_bytes":r.harness_peak_bytes,"inference_peak_bytes":r.inference_peak_bytes,
        "combined_peak_bytes":r.combined_peak_bytes,"available_min_bytes":r.available_min_bytes,"cpu_time_ms":r.cpu_time_ms,
        "minor_faults":r.minor_faults,"major_faults":r.major_faults,"peak_temperature_c":r.peak_temperature_c,
        "throttled":r.throttled,"energy_j":r.energy_j,"power_scope":r.power_scope,"sample_count":r.sample_count,
        "memory_exceeded":r.memory_exceeded});
    for part in [counts, timing, resources] {
        if let (Some(a), Value::Object(b)) = (run.as_object_mut(), part) {
            a.extend(b);
        }
    }
    let turns: Vec<Value> = out
        .turns
        .iter()
        .map(|t| {
            json!({"idx":t.idx,"visible_tools":t.visible_tools,"template_ms":t.template_ms,"tokenize_ms":t.tokenize_ms,
                "http_ms":t.http_ms,"prefill_ms":t.prefill_ms,"decode_ms":t.decode_ms,"prompt_tokens":t.prompt_tokens,
                "cached_tokens":t.cached_tokens,"generated_tokens":t.generated_tokens,"parse_ms":t.parse_ms,
                "parse_status":tag(&t.parse_status),"leading_text":t.leading_text,"parse_error":t.parse_error,"raw_text":t.raw_text})
        })
        .collect();
    let calls: Vec<Value> = out
        .tool_log
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({"idx":i,"turn":c.turn,"tool":c.tool,"arguments":c.arguments,"valid":c.valid,"ok":c.ok,
                "validate_ms":c.validate_ms,"ms":c.ms,"output_bytes":c.output_bytes,"output":c.output,"output_truncated":c.output_truncated,"error":c.error})
        })
        .collect();
    let checks: Vec<Value> = out
        .checks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({"idx":i,"kind":c.kind,"label":c.label,"passed":c.passed,"failure_class":c.class.as_ref().map(tag)})
        })
        .collect();
    let mut body = json!({"run":run,"turns":turns,"tool_calls":calls,"checks":checks});
    let extra = spill(
        &mut body,
        "samples",
        sample_rows(r),
        &format!("/v1/runs/{id}/samples"),
    );
    let mut reqs = vec![Request {
        method: "POST".into(),
        path: "/v1/runs".into(),
        body,
    }];
    reqs.extend(extra);
    Ok(reqs)
}
