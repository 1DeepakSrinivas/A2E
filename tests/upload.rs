//! Result upload: payload shape against the Worker schema, outbox ordering, retries, resume

use a2e::{
    case::Case,
    config::{Suite, Variant},
    measure::Sample,
    record::{ParseStatus, ToolEntry, Turn},
    runner::{Outcome, Status},
    store::Phase,
    upload::{
        payload::{self, LoadRecord, RunMeta},
        Client, Request, Resume, Uploader,
    },
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

const STUDY: &str = "0123456789abcdef0123456789abcdef";
const JOB: &str = "fedcba9876543210fedcba9876543210";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn case() -> Case {
    let path = repo().join("configs/suites/smoke.toml");
    Suite::load(&path)
        .unwrap()
        .load_cases(&path)
        .unwrap()
        .into_iter()
        .find(|t| t.id == "chain-copy")
        .unwrap()
}
fn sample(t: f64) -> Sample {
    Sample {
        t_ms: t,
        harness_bytes: 1,
        inference_bytes: 2,
        available_bytes: Some(3),
        cpu_ms: Some(4),
        temperature_c: Some(50.),
        throttled: None,
        power_w: None,
        minor_faults: Some(0),
        major_faults: Some(0),
    }
}
fn outcome(samples: usize) -> Outcome {
    let mut o = Outcome {
        status: Status::Success,
        answer: Some("done".into()),
        ..Outcome::default()
    };
    o.turns = vec![Turn {
        idx: 0,
        parse_status: ParseStatus::Call,
        ..Turn::default()
    }];
    o.tool_log = vec![ToolEntry {
        turn: 0,
        tool: "read_file".into(),
        arguments: json!({"path":"a"}),
        valid: true,
        ok: true,
        validate_ms: 0.1,
        ms: 0.2,
        output_bytes: 3,
        output: Some("{\"content\":\"hi\"}".into()),
        output_truncated: false,
        error: None,
    }];
    o.checks = vec![a2e::eval::CheckResult {
        kind: "file_equals".into(),
        label: "copy".into(),
        passed: true,
        class: None,
    }];
    o.resources.series = (0..samples).map(|i| sample(i as f64)).collect();
    o
}
fn run_requests(c: &Case, v: &Variant, attempt: u32, samples: usize) -> Vec<Request> {
    let meta = RunMeta {
        study: STUDY,
        job: JOB,
        attempt,
        phase: Phase::Measured,
        model: "m",
        context: 1024,
        variant: v,
        repeat: 0,
        case: c,
        load_id: Some("0123456789ab-load-1"),
        trace: Path::new("trace.jsonl"),
    };
    payload::run(&meta, &outcome(samples)).unwrap()
}

#[test]
fn run_payload_carries_the_scoring_fields() {
    let reqs = run_requests(&case(), &Variant::default(), 1, 1);
    let run = &reqs[0].body["run"];
    for key in [
        "milestone_score",
        "attempted_calls",
        "abstained",
        "expected_calls",
    ] {
        assert!(run.get(key).is_some(), "{key}");
    }
}

struct MockWorker {
    url: String,
    seen: Arc<Mutex<Vec<(String, String, Value)>>>,
    statuses: Arc<Mutex<Vec<u16>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}
impl MockWorker {
    fn new(statuses: Vec<u16>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let statuses = Arc::new(Mutex::new(statuses));
        let stop = Arc::new(AtomicBool::new(false));
        let (log, queue, done) = (seen.clone(), statuses.clone(), stop.clone());
        let handle = thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
                stream.set_nonblocking(false).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut parts = first.split_whitespace();
                let (method, path) = (
                    parts.next().unwrap_or("").to_owned(),
                    parts.next().unwrap_or("").to_owned(),
                );
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                        length = v.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                log.lock().unwrap().push((
                    method,
                    path,
                    serde_json::from_slice(&body).unwrap_or(Value::Null),
                ));
                let status = {
                    let mut q = queue.lock().unwrap();
                    if q.is_empty() {
                        200
                    } else {
                        q.remove(0)
                    }
                };
                let reply = if status < 300 {
                    r#"{"inserted":1}"#
                } else {
                    r#"{"error":{"code":"x","message":"no"}}"#
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        Self {
            url,
            seen,
            statuses,
            stop,
            handle: Some(handle),
        }
    }
    fn client(&self) -> Client {
        Client::new(&self.url, "token").unwrap()
    }
    fn paths(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|(m, p, _)| format!("{m} {p}"))
            .collect()
    }
}
impl Drop for MockWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let _ = &self.statuses;
    }
}

fn schema_columns(sql: &str) -> BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> {
    let mut tables: BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();
    let mut current: Option<(String, BTreeSet<String>, BTreeSet<String>)> = None;
    for line in sql.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("CREATE TABLE ") {
            let name = rest.split('(').next().unwrap().to_owned();
            current = Some((name, BTreeSet::new(), BTreeSet::new()));
            continue;
        }
        if let Some(rest) = t.strip_prefix("ALTER TABLE ") {
            let mut words = rest.split_whitespace();
            let (table, column) = (words.next().unwrap_or(""), words.nth(2).unwrap_or(""));
            if let Some((all, _)) = tables.get_mut(table) {
                all.insert(column.to_owned());
            }
            continue;
        }
        let Some((_, all, required)) = current.as_mut() else {
            continue;
        };
        let word = t.split_whitespace().next().unwrap_or("");
        let is_column = word
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && !word.is_empty();
        if is_column {
            all.insert(word.to_owned());
            let upper = t.to_uppercase();
            if (upper.contains("NOT NULL") || upper.contains("PRIMARY KEY"))
                && !upper.contains("DEFAULT")
            {
                required.insert(word.to_owned());
            }
        }
        if t.ends_with(';') {
            let (n, a, r) = current.take().unwrap();
            tables.insert(n, (a, r));
        }
    }
    tables
}
fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().unwrap().keys().cloned().collect()
}
fn check_row(
    tables: &BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)>,
    table: &str,
    row: &Value,
    omit: &[&str],
) {
    let (all, required) = &tables[table];
    let sent = keys(row);
    for k in &sent {
        assert!(all.contains(k), "{table}.{k} is not a column");
        assert!(!omit.contains(&k.as_str()), "{table}.{k} must be omitted");
    }
    for k in required {
        if !omit.contains(&k.as_str()) {
            assert!(sent.contains(k), "{table}.{k} is required");
        }
    }
}

#[test]
fn payload_fields_match_the_worker_schema() {
    let dir = repo().join("a2e-results/migrations");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: submodule a2e-results is not checked out");
        return;
    };
    let mut files: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    files.sort();
    let sql: String = files
        .iter()
        .filter_map(|f| std::fs::read_to_string(f).ok())
        .collect::<Vec<_>>()
        .join("\n");
    let tables = schema_columns(&sql);
    let reqs = run_requests(&case(), &Variant::default(), 1, 3);
    let body = &reqs[0].body;
    check_row(&tables, "runs", &body["run"], &[]);
    check_row(&tables, "turns", &body["turns"][0], &["run_id"]);
    check_row(&tables, "tool_calls", &body["tool_calls"][0], &["run_id"]);
    check_row(&tables, "checks", &body["checks"][0], &["run_id"]);
    assert_eq!(body["tool_calls"][0]["output"], "{\"content\":\"hi\"}");
    check_row(
        &tables,
        "samples",
        &body["samples"][0],
        &["run_id", "load_id"],
    );
    let resources = Default::default();
    let load = payload::load(&LoadRecord {
        id: "0123456789ab-load-1",
        study: STUDY,
        model: "m",
        context: 1024,
        load_ms: 1.,
        resources: &resources,
        template: Some("t"),
        server_version: Some("v"),
        props: Some(&json!({"a":1})),
        error: None,
    });
    check_row(
        &tables,
        "model_loads",
        &{
            let mut b = load[0].body.clone();
            b.as_object_mut().unwrap().remove("samples");
            b
        },
        &[],
    );
    let study = payload::study(STUDY, &json!({}), &[case()]).unwrap();
    check_row(&tables, "cases", &study.body["cases"][0], &["study"]);
}
#[test]
fn run_ids_follow_the_api_form_and_children_have_no_run_id() {
    let reqs = run_requests(&case(), &Variant::default(), 2, 1);
    let id = format!("{}-{}-2", &STUDY[..12], &JOB[..16]);
    assert_eq!(payload::run_id(STUDY, JOB, 2), id);
    assert_eq!(reqs[0].path, "/v1/runs");
    assert_eq!(reqs[0].body["run"]["id"], id);
    assert_eq!(reqs[0].body["run"]["state"], "complete");
    assert!(reqs[0].body["turns"][0].get("run_id").is_none());
}
#[test]
fn large_series_are_sent_as_separate_sample_requests() {
    let reqs = run_requests(&case(), &Variant::default(), 1, 12_000);
    assert_eq!(reqs.len(), 4);
    assert_eq!(reqs[0].body["samples"], json!([]));
    let id = payload::run_id(STUDY, JOB, 1);
    for r in &reqs[1..] {
        assert_eq!(r.path, format!("/v1/runs/{id}/samples"));
    }
    let total: usize = reqs[1..]
        .iter()
        .map(|r| r.body["samples"].as_array().unwrap().len())
        .sum();
    assert_eq!(total, 12_000);
}
#[test]
fn requests_post_in_order_and_leave_the_outbox() {
    let dir = tempfile::tempdir().unwrap();
    let worker = MockWorker::new(vec![]);
    let mut up = Uploader::with_client(dir.path(), Some(worker.client())).unwrap();
    up.submit(vec![payload::study(STUDY, &json!({}), &[case()]).unwrap()])
        .unwrap();
    up.submit(run_requests(&case(), &Variant::default(), 1, 2))
        .unwrap();
    assert_eq!(
        worker.paths(),
        [
            format!("PUT /v1/studies/{STUDY}"),
            "POST /v1/runs".to_string()
        ]
    );
    assert_eq!(up.status().unwrap().pending, 0);
}
#[test]
fn failed_posts_stay_pending_and_resend_in_order_later() {
    let dir = tempfile::tempdir().unwrap();
    let worker = MockWorker::new(vec![503]);
    let mut up = Uploader::with_client(dir.path(), Some(worker.client())).unwrap();
    up.submit(vec![payload::study(STUDY, &json!({}), &[case()]).unwrap()])
        .unwrap();
    up.submit(run_requests(&case(), &Variant::default(), 1, 2))
        .unwrap();
    assert_eq!(
        worker.paths().len(),
        1,
        "no more posts after a retryable failure"
    );
    assert_eq!(up.status().unwrap().pending, 2);
    let flush = up.flush().unwrap();
    assert_eq!((flush.sent, flush.pending), (2, 0));
    assert_eq!(
        worker.paths()[1..],
        [
            format!("PUT /v1/studies/{STUDY}"),
            "POST /v1/runs".to_string()
        ]
    );
}
#[test]
fn rejected_requests_move_aside_and_do_not_block_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let worker = MockWorker::new(vec![400]);
    let mut up = Uploader::with_client(dir.path(), Some(worker.client())).unwrap();
    up.submit(vec![payload::study(STUDY, &json!({}), &[case()]).unwrap()])
        .unwrap();
    up.submit(run_requests(&case(), &Variant::default(), 1, 2))
        .unwrap();
    let status = up.status().unwrap();
    assert_eq!((status.pending, status.rejected), (0, 1));
    assert_eq!(worker.paths().len(), 2);
}
#[test]
fn without_credentials_requests_wait_in_the_outbox() {
    let dir = tempfile::tempdir().unwrap();
    let mut up = Uploader::with_client(dir.path(), None).unwrap();
    up.submit(run_requests(&case(), &Variant::default(), 1, 2))
        .unwrap();
    assert_eq!(up.status().unwrap().pending, 1);
    assert!(up.flush().unwrap().stopped.is_some());
}
#[test]
fn local_mode_keeps_requests_pending_regardless_of_credentials() {
    let dir = tempfile::tempdir().unwrap();
    let mut uploader = Uploader::with_client(&dir.path().join("results"), None).unwrap();
    uploader
        .submit(run_requests(&case(), &Variant::default(), 1, 2))
        .unwrap();
    std::fs::write(
        dir.path().join(".env"),
        "A2E_RESULTS_ENV=missing-results.env\n",
    )
    .unwrap();
    for (url, token) in [
        ("", ""),
        ("http://localhost:1", ""),
        ("http://localhost:1", "token"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_a2e"))
            .current_dir(dir.path())
            .args(["--config"])
            .arg(repo().join("configs/models.toml"))
            .arg("upload")
            .env("A2E_LOCAL", "1")
            .env("WORKER_URL", url)
            .env("WRITE_TOKEN", token)
            .env("A2E_OUTPUT_DIR", dir.path().join("results"))
            .env_remove("A2E_RESULTS_ENV")
            .env_remove("A2E_HOST_ID")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let status: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(status["pending"], 1);
        assert_eq!(status["sent"], 0);
        assert_eq!(status["stopped"], "uploads are disabled in local mode");
    }
}
#[test]
fn resume_skips_finished_jobs_and_numbers_attempts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resume.jsonl");
    let mut r = Resume::open(&path).unwrap();
    assert!(r.should_run(STUDY, JOB, false));
    assert_eq!(r.next_attempt(STUDY, JOB), 1);
    r.record_job(STUDY, JOB, 1, "interrupted", Some("interrupted"))
        .unwrap();
    assert!(r.should_run(STUDY, JOB, false));
    r.record_job(STUDY, JOB, 2, "complete", Some("model_failure"))
        .unwrap();
    assert!(!r.should_run(STUDY, JOB, false));
    assert!(r.should_run(STUDY, JOB, true));
    assert_eq!(r.next_load_id(STUDY).unwrap(), "0123456789ab-load-1");
    drop(r);
    let mut r = Resume::open(&path).unwrap();
    assert!(!r.should_run(STUDY, JOB, false));
    assert_eq!(r.next_attempt(STUDY, JOB), 3);
    assert_eq!(r.next_load_id(STUDY).unwrap(), "0123456789ab-load-2");
    r.record_job(STUDY, "other", 1, "complete", Some("success"))
        .unwrap();
    assert!(!r.should_run(STUDY, "other", true));
}
