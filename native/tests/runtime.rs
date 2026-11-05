//! Integration tests for the agent loop and real tool subprocesses
//! Scripted inference, resource limits, retrieval checks, and run recovery

mod support;
use a2e_native::{
    adapter::{Adapter, Call},
    config::{Config, Manifest, Runtime, Suite, Task, Variant},
    inference::Inference,
    runner::{task_run, TaskExecution},
    sdk::{ToolHost, ToolProcess},
    store::Store,
    tools::Builtins,
};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};
use support::{call, MockServer};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}
fn manifest() -> Manifest {
    let mut m = Manifest::load(&repo().join("configs/native/tools/standard.toml")).unwrap();
    m.executable = env!("CARGO_BIN_EXE_a2e-native").into();
    m
}
fn suite() -> Suite {
    Suite::load(&repo().join("configs/native/suites/smoke.toml")).unwrap()
}
fn task() -> Task {
    let s = suite();
    s.load_tasks(&repo().join("configs/native/suites/smoke.toml"))
        .unwrap()
        .into_iter()
        .find(|t| t.id.starts_with("copy-"))
        .unwrap()
}
fn execute(server: &MockServer, t: &Task, v: &Variant, s: &Suite) -> a2e_native::runner::Outcome {
    let dir = tempfile::tempdir().unwrap();
    let client = Inference::new(server.url.clone()).unwrap();
    let runtime = Runtime {
        memory_limit_mb: 1_000_000,
        reserve_mb: 0,
        power: "none".into(),
        ..Runtime::default()
    };
    task_run(TaskExecution {
        client: &client,
        adapter: Adapter::Qwen,
        task: t,
        suite: s,
        variant: v,
        manifest: &manifest(),
        ctx: 1024,
        dir: dir.path(),
        cancel: Arc::new(AtomicBool::new(false)),
        server_pid: std::process::id(),
        runtime: &runtime,
    })
    .unwrap()
}

#[test]
fn dependency_chain_checks_final_state_in_both_modes() {
    for mode in ["autonomous", "guided"] {
        let server = MockServer::new(
            vec![
                call("read_file", json!({"path":"reading.txt"})),
                call(
                    "write_file",
                    json!({"path":"copy.txt","content":"Reading: 17"}),
                ),
                call("final_answer", json!({"answer":"copied"})),
            ],
            100,
            Duration::ZERO,
        );
        let out = execute(
            &server,
            &task(),
            &Variant {
                mode: mode.into(),
                ..Default::default()
            },
            &suite(),
        );
        assert_eq!(out.status, "success");
        assert_eq!(out.model_calls, 3);
        assert_eq!(out.tool_invocations, 2);
        assert_eq!(out.generated_tokens, Some(36));
        let seen = server.seen.lock().unwrap();
        let last = seen
            .iter()
            .rfind(|v| v["path"] == "/apply-template")
            .unwrap();
        assert!(last["body"]["messages"].to_string().contains("Reading: 17"));
        assert!(!last["body"].to_string().contains("file_eq"));
    }
}
#[test]
fn alternative_valid_action_sequence_passes() {
    let t = task();
    let state = json!({"files":{"copy.txt":"Reading: 17"}});
    let calls = [
        Call {
            name: "read_file".into(),
            arguments: json!({"path":"reading.txt"}),
        },
        Call {
            name: "read_file".into(),
            arguments: json!({"path":"reading.txt"}),
        },
        Call {
            name: "write_file".into(),
            arguments: json!({"path":"copy.txt","content":"Reading: 17"}),
        },
    ];
    assert_eq!(
        a2e_native::runner::checks(&t, &state, &calls, "done").unwrap()[0]["passed"],
        true
    );
}
#[test]
fn bounded_repair_counts_failed_model_calls() {
    let server = MockServer::new(
        vec![
            "<tool_call>broken".into(),
            call("read_file", json!({"path":"reading.txt"})),
            call(
                "write_file",
                json!({"path":"copy.txt","content":"Reading: 17"}),
            ),
            "done".into(),
        ],
        100,
        Duration::ZERO,
    );
    let out = execute(
        &server,
        &task(),
        &Variant {
            repairs: 1,
            ..Default::default()
        },
        &suite(),
    );
    assert_eq!(out.status, "success");
    assert_eq!(out.repairs, 1);
    assert_eq!(out.model_calls, 4);
}
#[test]
fn invalid_arguments_are_not_executed() {
    let server = MockServer::new(
        vec![call("write_file", json!({"path":"copy.txt"}))],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.status, "model_failure", "{:?}", out.reason);
    assert_eq!(out.argument_errors, 1);
    assert_eq!(out.tool_invocations, 0);
}
#[test]
fn context_budget_prevents_generation() {
    let server = MockServer::new(vec![], 900, Duration::ZERO);
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.status, "resource_limit");
    assert_eq!(out.model_calls, 0);
    assert_eq!(out.reason.as_deref(), Some("context_budget"));
}
#[test]
fn repeated_actions_stop_before_exhausting_all_calls() {
    let server = MockServer::new(
        vec![call("read_file", json!({"path":"reading.txt"})); 8],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.reason.as_deref(), Some("repeated_action"));
    assert_eq!(out.model_calls, 3);
}
#[test]
fn timeout_keeps_failed_call_cost_and_missing_timings() {
    let server = MockServer::new(vec!["done".into()], 100, Duration::from_millis(1100));
    let mut s = suite();
    s.budget.call_seconds = 1;
    let out = execute(&server, &task(), &Variant::default(), &s);
    assert_ne!(out.status, "success");
    assert_eq!(out.model_calls, 1);
    assert!(out.http_ms >= 900.);
    assert_eq!(out.generated_tokens, None);
}
#[test]
fn real_tool_process_preserves_state_and_exits_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let mut p = ToolProcess::spawn(&manifest(), dir.path(), false).unwrap();
    let pid = p.pid();
    p.request("init", "", json!({"fixtures":{}}), Duration::from_secs(2))
        .unwrap();
    assert!(
        p.request(
            "call",
            "write_file",
            json!({"path":"a.txt","content":"saved"}),
            Duration::from_secs(2)
        )
        .unwrap()
        .ok
    );
    assert_eq!(
        p.request(
            "call",
            "read_file",
            json!({"path":"a.txt"}),
            Duration::from_secs(2)
        )
        .unwrap()
        .output["content"],
        "saved"
    );
    drop(p);
    // SAFETY: signal zero checks whether the child was reaped.
    assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
}
#[test]
fn timeout_kills_stalled_tool_and_does_not_hang_drop() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = manifest();
    m.executable = "/bin/sleep".into();
    m.args = vec!["30".into()];
    m.timeout_seconds = 1;
    let mut p = ToolProcess::spawn(&m, dir.path(), false).unwrap();
    let start = std::time::Instant::now();
    assert!(p
        .request("init", "", json!({}), Duration::from_millis(20))
        .is_err());
    drop(p);
    assert!(start.elapsed() < Duration::from_secs(2));
}
#[test]
fn output_limit_rejects_unbounded_tool_stream() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = manifest();
    m.output_limit_bytes = 256;
    let mut p = ToolProcess::spawn(&m, dir.path(), false).unwrap();
    p.request(
        "init",
        "",
        json!({"fixtures":{"files":{"a.txt":"x".repeat(3000)}}}),
        Duration::from_secs(2),
    )
    .unwrap();
    assert!(p
        .request(
            "call",
            "read_file",
            json!({"path":"a.txt"}),
            Duration::from_secs(2)
        )
        .is_err());
}
#[test]
fn store_recovers_interrupted_attempt_without_double_counting() {
    let dir = tempfile::tempdir().unwrap();
    let st = Store::open(&dir.path().join("native.db")).unwrap();
    st.study("study", &json!({})).unwrap();
    let meta = json!({"model":"test","variant":{"name":"baseline"},"context":1024,"task":{"id":"t","family":"local","required_calls":1,"tools":["read_file"]}});
    st.begin("study", "job", "measured", &meta, Path::new("trace"))
        .unwrap();
    assert_eq!(st.recover().unwrap(), 1);
    assert!(st.should_run("study", "job", false).unwrap());
    let id = st
        .begin("study", "job", "measured", &meta, Path::new("trace2"))
        .unwrap();
    st.finish(
        id,
        &json!({"status":"success","wall_ms":10,"model_calls":1}),
    )
    .unwrap();
    assert!(!st.should_run("study", "job", false).unwrap());
    let rows = st.rows().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        a2e_native::store::summary(&rows)["groups"][0]["successes"],
        1
    );
    st.freeze_judge("study", "first").unwrap();
    assert!(st.freeze_judge("study", "different").is_err());
}
#[test]
fn native_store_refuses_legacy_database() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("old.db");
    let c = rusqlite::Connection::open(&p).unwrap();
    c.execute("CREATE TABLE turns(id INTEGER)", []).unwrap();
    assert!(Store::open(&p).is_err());
}
#[test]
fn every_shipped_suite_validates_and_has_disjoint_splits() {
    let cfg = Config::load(&repo().join("configs/native/models.toml")).unwrap();
    let m = manifest();
    for entry in std::fs::read_dir(repo().join("configs/native/suites")).unwrap() {
        let path = entry.unwrap().path();
        let s = Suite::load(&path).unwrap();
        let tasks = s.load_tasks(&path).unwrap();
        assert!(a2e_native::runner::plan(&cfg, &s, &tasks, &m).is_ok());
        assert!(tasks.iter().all(|t| t.split == s.split));
    }
}
#[test]
fn fixture_queries_generalize_and_fetch_is_task_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = Builtins::new(dir.path().into(), false).unwrap();
    tools.initialize(json!({"fixtures":{"pages":[{"id":"source","url":"https://example.org/specs","title":"Device memory requirements","text":"The device requires 1 GB memory."}]}})).unwrap();
    assert!(tools
        .call("fetch_page", json!({"source_id":"source"}))
        .is_err());
    for q in ["memory requirement", "device memory", "requirements device"] {
        assert_eq!(
            tools.call("web_search", json!({"query":q})).unwrap()["results"][0]["source_id"],
            "source"
        );
    }
    assert_eq!(
        tools
            .call("web_search", json!({"query":"penguin"}))
            .unwrap()["results"],
        json!([])
    );
    let page = tools
        .call("fetch_page", json!({"source_id":"source"}))
        .unwrap();
    assert_eq!(page["text"], "The device requires 1 GB memory.");
}
#[test]
fn destination_validation_blocks_local_addresses_and_redirect_targets() {
    for url in [
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://localhost/",
        "https://a:b@example.org/",
        "file:///etc/passwd",
        "https://example.org:1234/",
    ] {
        assert!(a2e_native::tools::validate_url(url).is_err(), "{url}");
    }
    assert!(a2e_native::tools::validate_url("https://example.org/docs").is_ok());
}
#[test]
fn file_paths_and_html_extraction_keep_tool_data_bounded() {
    let dir = tempfile::tempdir().unwrap();
    for p in ["../escape", "/etc/passwd", "a/../b", ""] {
        assert!(a2e_native::tools::safe_path(dir.path(), p).is_err());
    }
    let text = a2e_native::tools::html_text(
        "<html><script>bad()</script><style>bad</style><p>Visible facts</p></html>",
    );
    assert_eq!(text, "Visible facts");
    assert!(a2e_native::read_bounded(std::io::Cursor::new(vec![0; 1025]), 1024).is_err());
}
