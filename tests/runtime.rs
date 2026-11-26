//! Integration tests for the agent loop with in-process tools
//! Scripted inference, resource limits, retrieval checks, and run recovery

mod support;
use a2e::{
    adapter::Call,
    case::{Case, FinalAnswer},
    config::{Config, Mode, Selection, Suite, Variant},
    eval::{evaluate, FailureClass, Observation},
    runner::{select_tools, Status},
    tools::Builtins,
};
use serde_json::json;
use std::time::Duration;
use support::{call, catalog, execute, golden_script, repo, use_sandbox, MockServer};

fn suite() -> Suite {
    let config = Config::load(&repo().join("configs/models.toml")).unwrap();
    Suite::load(&repo().join("configs/suites/smoke.toml"))
        .unwrap()
        .resolve(&config, &Selection::default())
        .unwrap()
}
fn task() -> Case {
    let s = suite();
    s.load_cases(&repo().join("configs/suites/smoke.toml"))
        .unwrap()
        .into_iter()
        .find(|t| t.id == "chain-copy")
        .unwrap()
}
#[test]
fn dependency_chain_checks_final_state_in_both_modes() {
    for mode in [Mode::Autonomous, Mode::Guided] {
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
                mode,
                ..Default::default()
            },
            &suite(),
        );
        assert_eq!(out.status, Status::Success);
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
    let observation = Observation {
        calls: &calls,
        attempted: &calls,
        state: &state,
        answer: "done",
    };
    assert!(evaluate(&t, &observation).iter().all(|c| c.passed));
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
    assert_eq!(out.status, Status::Success);
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
    assert_eq!(out.status, Status::ModelFailure, "{:?}", out.reason);
    assert_eq!(out.argument_errors, 1);
    assert_eq!(out.tool_invocations, 0);
}
#[test]
fn a_schema_error_is_fed_back_and_the_model_can_recover() {
    let t = task();
    let mut script = vec![call("write_file", json!({"path":"copy.txt"}))];
    script.extend(golden_script(&t));
    let server = MockServer::new(script, 100, Duration::ZERO);
    let out = execute(&server, &t, &Variant::default(), &suite());
    assert_eq!(out.status, Status::Success, "{:?}", out.reason);
    assert_eq!(out.argument_errors, 1);
    let seen = server.seen.lock().unwrap();
    let last = seen
        .iter()
        .rfind(|r| r["path"] == "/apply-template")
        .unwrap();
    assert!(last["body"].to_string().contains("error"));
}
#[test]
fn an_attempt_cut_short_by_the_context_budget_still_has_checks() {
    let t = task();
    let server = MockServer::new(golden_script(&t), 2000, Duration::ZERO);
    let out = execute(&server, &t, &Variant::default(), &suite());
    assert_eq!(out.status, Status::ResourceLimit, "{:?}", out.reason);
    assert!(!out.checks.is_empty());
    assert!(out.milestone_score.is_some());
}
#[test]
fn context_budget_prevents_generation() {
    let server = MockServer::new(vec![], 900, Duration::ZERO);
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.status, Status::ResourceLimit);
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
    assert_ne!(out.status, Status::Success);
    assert_eq!(out.model_calls, 1);
    assert!(out.http_ms >= 900.);
    assert_eq!(out.generated_tokens, None);
    // This is what the stall counter in `run` counts
    assert!(a2e::runner::no_call_returned(&out));
}
#[test]
fn completed_calls_do_not_count_as_a_stall_and_prompt_tokens_are_not_doubled() {
    let t = task();
    let server = MockServer::new(golden_script(&t), 100, Duration::ZERO);
    let out = execute(&server, &t, &Variant::default(), &suite());
    assert_eq!(out.status, Status::Success, "{:?}", out.reason);
    assert!(!a2e::runner::no_call_returned(&out));
    assert_eq!(out.prompt_tokens, Some(100 * out.model_calls));
    assert_eq!(out.cached_tokens, Some(0));
    let client = a2e::inference::Inference::new(server.url.clone()).unwrap();
    assert!(client.settle(Duration::from_secs(2)));
}
#[test]
fn slow_tool_past_the_budget_stops_the_attempt() {
    let mut t = task();
    t.fixtures.faults =
        vec![
            serde_json::from_value(json!({"tool":"read_file","mode":"slow_ms","ms":1200})).unwrap(),
        ];
    let server = MockServer::new(
        vec![call("read_file", json!({"path":"reading.txt"}))],
        100,
        Duration::ZERO,
    );
    let mut s = suite();
    s.budget.tool_seconds = 1;
    let out = execute(&server, &t, &Variant::default(), &s);
    assert_eq!(out.status, Status::ResourceLimit, "{:?}", out.reason);
    assert!(out.reason.as_deref().unwrap().contains("tool_timeout"));
    assert!(out.tool_ms >= 1000.);
}
#[test]
fn oversize_tool_result_stops_the_attempt() {
    let mut t = task();
    t.fixtures
        .files
        .insert("reading.txt".into(), "x".repeat(3000));
    let server = MockServer::new(
        vec![call("read_file", json!({"path":"reading.txt"}))],
        100,
        Duration::ZERO,
    );
    let mut s = suite();
    s.budget.tool_output_bytes = 256;
    let out = execute(&server, &t, &Variant::default(), &s);
    assert_eq!(out.status, Status::ResourceLimit, "{:?}", out.reason);
    assert!(out.reason.as_deref().unwrap().contains("output_limit"));
}
#[test]
fn tool_setup_time_is_recorded_and_stays_outside_tool_calls() {
    let server = MockServer::new(
        vec![call("final_answer", json!({"answer":"nothing to do"}))],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert!(out.tool_setup_ms > 0.);
    assert_eq!(out.tool_ms, 0.);
}
#[test]
fn harness_overhead_stays_a_small_share_of_wall_time() {
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
        Duration::from_millis(60),
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.status, Status::Success);
    assert!(out.wall_ms >= 180.);
    assert!(
        out.harness_other_ms < out.wall_ms * 0.1,
        "other {} of wall {}",
        out.harness_other_ms,
        out.wall_ms
    );
}
#[test]
fn every_shipped_suite_validates_and_has_disjoint_splits() {
    use_sandbox();
    let cfg = Config::load(&repo().join("configs/models.toml")).unwrap();
    let m = catalog();
    for entry in std::fs::read_dir(repo().join("configs/suites")).unwrap() {
        let path = entry.unwrap().path();
        let s = Suite::load(&path)
            .unwrap()
            .resolve(&cfg, &Selection::default())
            .unwrap();
        let cases = s.load_cases(&path).unwrap();
        assert!(a2e::runner::plan(&cfg, &s, &cases, &m).is_ok());
        assert!(cases.iter().all(|c| c.split == s.split));
    }
}
#[test]
fn fixture_queries_generalize_and_fetch_is_task_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = Builtins::new(dir.path().into(), false).unwrap();
    tools.initialize(&serde_json::from_value(json!({"pages":[{"id":"source","url":"https://example.org/specs","title":"Device memory requirements","text":"The device requires 1 GB memory."}]})).unwrap()).unwrap();
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
        assert!(a2e::tools::validate_url(url).is_err(), "{url}");
    }
    assert!(a2e::tools::validate_url("https://example.org/docs").is_ok());
}
#[test]
fn file_paths_and_html_extraction_keep_tool_data_bounded() {
    let dir = tempfile::tempdir().unwrap();
    for p in ["../escape", "/etc/passwd", "a/../b", ""] {
        assert!(a2e::tools::safe_path(dir.path(), p).is_err());
    }
    let text = a2e::tools::html_text(
        "<html><script>bad()</script><style>bad</style><p>Visible facts</p></html>",
    );
    assert_eq!(text, "Visible facts");
    assert!(a2e::read_bounded(std::io::Cursor::new(vec![0; 1025]), 1024).is_err());
}

#[test]
fn shipped_cases_validate_against_the_standard_catalog() {
    let registry = catalog();
    let registry = registry
        .tools
        .iter()
        .map(|t| t.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let cases = a2e::case::load_all(&[repo().join("configs/cases")]).unwrap();
    assert!(cases.len() >= 48);
    for c in &cases {
        c.validate(&registry).unwrap();
    }
}
#[test]
fn guided_plan_marks_cases_without_a_workflow_not_applicable() {
    let cfg = Config::load(&repo().join("configs/models.toml")).unwrap();
    let mut chain = task();
    chain.stages.clear();
    let single = {
        let mut c = task();
        c.id = "single".into();
        c.category = a2e::case::Category::Simple;
        c.stages.clear();
        c
    };
    chain.id = "chain".into();
    let mut s = suite();
    s.variants = vec![
        Variant::default(),
        Variant {
            name: "guided".into(),
            mode: Mode::Guided,
            ..Default::default()
        },
    ];
    let plan = a2e::runner::plan(&cfg, &s, &[chain, single], &catalog()).unwrap();
    let groups = plan["groups"].as_array().unwrap();
    let guided = groups
        .iter()
        .find(|g| g["variant"]["name"] == "guided")
        .unwrap();
    assert_eq!(
        (guided["cases"].as_u64(), guided["not_applicable"].as_u64()),
        (Some(1), Some(1))
    );
    let auto = groups
        .iter()
        .find(|g| g["variant"]["name"] == "baseline")
        .unwrap();
    assert_eq!(auto["not_applicable"], 0);
}
#[test]
fn distractors_extend_autonomous_tools_but_not_guided_stages() {
    let mut c = task();
    let m = catalog();
    let base = select_tools(&c, &m, &Variant::default(), &[], 0)
        .unwrap()
        .len();
    c.tools.distractors = 3;
    let auto = select_tools(&c, &m, &Variant::default(), &[], 0).unwrap();
    assert_eq!(auto.len(), base + 3);
    let guided = Variant {
        mode: Mode::Guided,
        ..Default::default()
    };
    let stages = c.stage_plan().unwrap();
    let names = select_tools(&c, &m, &guided, &stages, 0).unwrap();
    assert_eq!(names.len(), stages[0].tools.len() + 1);
}
#[test]
fn state_only_case_passes_without_a_final_answer_when_optional() {
    let mut c = task();
    c.scope.final_answer = FinalAnswer::Optional;
    let script = vec![
        call("read_file", json!({"path":"reading.txt"})),
        call(
            "write_file",
            json!({"path":"copy.txt","content":"Reading: 17"}),
        ),
    ];
    let server = MockServer::new(
        [script.clone(), vec!["<tool_call>x".into(); 8]].concat(),
        100,
        Duration::ZERO,
    );
    let mut s = suite();
    s.budget.model_calls = 2;
    let out = execute(&server, &c, &Variant::default(), &s);
    assert_eq!(out.status, Status::Success, "{:?}", out.reason);
    assert!(out.answer.is_none());
    c.scope.final_answer = FinalAnswer::Required;
    let server = MockServer::new(script, 100, Duration::ZERO);
    let out = execute(&server, &c, &Variant::default(), &s);
    assert_eq!(out.status, Status::ResourceLimit);
    assert_eq!(out.failure_class, Some(FailureClass::ResourceLimit));
}
#[test]
fn failed_attempts_report_a_failure_class_and_check_results() {
    let server = MockServer::new(
        vec![
            call("write_file", json!({"path":"copy.txt","content":"wrong"})),
            call("final_answer", json!({"answer":"done"})),
        ],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.reason.as_deref(), Some("outcome_check_failed"));
    assert_eq!(out.failure_class, Some(FailureClass::StateMismatch));
    assert!(out.checks.iter().any(|c| !c.passed));
}
#[test]
fn answering_without_the_required_tools_is_no_tool_use() {
    let server = MockServer::new(
        vec![call("final_answer", json!({"answer":"done"}))],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.failure_class, Some(FailureClass::NoToolUse));
}
#[test]
fn stopping_after_a_tool_error_is_tool_error_abandoned() {
    let server = MockServer::new(
        vec![
            call("read_file", json!({"path":"missing.txt"})),
            call("final_answer", json!({"answer":"could not read it"})),
        ],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &task(), &Variant::default(), &suite());
    assert_eq!(out.failure_class, Some(FailureClass::ToolErrorAbandoned));
}
#[test]
fn catalog_and_builtin_registry_list_the_same_tools() {
    let mut declared = catalog()
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect::<Vec<_>>();
    declared.sort();
    let registered = Builtins::names();
    assert_eq!(declared, registered);
}
#[test]
fn sql_file_seeds_resolve_relative_to_the_case_and_validate() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("seed.sql"),
        "CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);",
    )
    .unwrap();
    let case = |sql: &str| {
        format!(
            "id = \"db-1\"\ncategory = \"simple\"\nsplit = \"dev\"\n[scope]\nprompt = \"Count rows.\"\n[tools]\nallowed = [\"db_query\"]\n[fixtures.db]\n{sql}\n[[check]]\nkind = \"call\"\ntool = \"db_query\"\n"
        )
    };
    let path = dir.path().join("db-1.toml");
    std::fs::write(&path, case("sql_file = \"seed.sql\"")).unwrap();
    let m = catalog();
    let registry = m.tools.iter().map(|t| t.name.as_str()).collect();
    let cases = a2e::case::load_all(std::slice::from_ref(&path)).unwrap();
    cases[0].validate(&registry).unwrap();
    assert!(cases[0]
        .fixtures
        .db
        .as_ref()
        .unwrap()
        .sql
        .as_ref()
        .unwrap()
        .contains("INSERT"));
    std::fs::write(&path, case("sql_file = \"missing.sql\"")).unwrap();
    assert!(a2e::case::load_all(std::slice::from_ref(&path)).is_err());
    std::fs::write(&path, case("sql = \"NOT SQL\"")).unwrap();
    let bad = a2e::case::load_all(&[path]).unwrap();
    assert!(bad[0].validate(&registry).is_err());
}
#[test]
fn shipped_database_seeds_load() {
    for name in ["inventory", "sensors"] {
        let sql =
            std::fs::read_to_string(repo().join(format!("configs/fixtures/{name}.sql"))).unwrap();
        a2e::tools::check_seed(&sql).unwrap();
    }
}
#[test]
fn database_case_runs_through_the_agent_loop() {
    let sql = std::fs::read_to_string(repo().join("configs/fixtures/inventory.sql")).unwrap();
    let case = Case::from_toml(&format!(
        r#"
id = "db-e2e"
category = "simple"
split = "dev"
[scope]
prompt = "How many tools are out of stock?"
[tools]
allowed = ["db_lookup", "db_query"]
[fixtures.db]
sql = """{sql}"""
[[check]]
kind = "call"
tool = "db_lookup"
args = {{ table = "products", where = {{ stock = 0 }} }}
match = "subset"
[[check]]
kind = "answer"
number = {{ value = 1 }}
"#
    ))
    .unwrap();
    let server = MockServer::new(
        vec![
            call(
                "db_lookup",
                json!({"table":"products","where":{"stock":0},"columns":["name"]}),
            ),
            call(
                "final_answer",
                json!({"answer":"1 item (wrench) is out of stock"}),
            ),
        ],
        100,
        Duration::ZERO,
    );
    let out = execute(&server, &case, &Variant::default(), &suite());
    assert_eq!(
        out.status,
        Status::Success,
        "{:?} {:?}",
        out.reason,
        out.checks
    );
    assert_eq!(out.tool_invocations, 1);
}

#[test]
fn every_offline_golden_sequence_succeeds_through_the_agent_loop() {
    use_sandbox();
    let cases = a2e::case::load_all(&[repo().join("configs/cases")]).unwrap();
    let s = suite();
    for case in cases.iter().filter(|c| !c.tags.iter().any(|t| t == "live")) {
        let script = golden_script(case);
        let server = MockServer::new(script, 100, Duration::ZERO);
        let out = execute(&server, case, &Variant::default(), &s);
        let want = if case.judge.enabled {
            Status::AwaitingJudge
        } else {
            Status::Success
        };
        assert_eq!(
            out.status, want,
            "{}: {:?} {:?}",
            case.id, out.reason, out.checks
        );
    }
}
#[test]
fn attempt_line_reports_the_failure_class_failed_checks_and_trace() {
    use_sandbox();
    let s = suite();
    let case = task();
    let server = MockServer::new(
        vec![call("final_answer", json!({"answer":"x"}))],
        100,
        Duration::ZERO,
    );
    let v = Variant::default();
    let out = execute(&server, &case, &v, &s);
    assert_ne!(out.status, Status::Success);
    let trace = std::path::Path::new("run/trace.jsonl");
    let meta = a2e::upload::payload::RunMeta {
        study: "s1",
        job: "job",
        attempt: 1,
        phase: a2e::store::Phase::Measured,
        model: "m",
        context: 1024,
        variant: &v,
        repeat: 0,
        case: &case,
        load_id: None,
        trace,
    };
    let line = a2e::runner::attempt_line(&meta, &out);
    assert_eq!(line["case"], case.id);
    assert_eq!(line["trace"], "run/trace.jsonl");
    assert!(line["failure_class"].is_string());
    let failed = line["failed_checks"].as_array().unwrap();
    assert!(!failed.is_empty() && failed.iter().all(|c| c["passed"] == false));
}

#[test]
fn stable_guided_mode_keeps_the_prompt_prefix_and_enforces_stage_order() {
    let t = task();
    let stages = t.stage_plan().unwrap();
    assert!(stages.len() > 1, "needs a multi-stage case");
    let stable = Variant {
        name: "guided_stable".into(),
        mode: Mode::GuidedStable,
        ..Default::default()
    };
    let server = MockServer::new(golden_script(&t), 100, Duration::ZERO);
    let out = execute(&server, &t, &stable, &suite());
    assert_eq!(out.status, Status::Success, "{:?}", out.reason);
    let seen = server.seen.lock().unwrap();
    let rendered = seen
        .iter()
        .filter(|r| r["path"] == "/apply-template")
        .map(|r| r["body"].clone())
        .collect::<Vec<_>>();
    assert!(rendered.len() > 1);
    // Tools, system message and first user message never change between calls
    for r in &rendered[1..] {
        assert_eq!(r["tools"], rendered[0]["tools"]);
        assert_eq!(r["messages"][0], rendered[0]["messages"][0]);
        assert_eq!(r["messages"][1], rendered[0]["messages"][1]);
    }
    let user = rendered[0]["messages"][1]["content"].as_str().unwrap();
    for s in &stages {
        assert!(user.contains(&s.instruction), "{user}");
    }
    drop(seen);
    // A later stage's tool is visible but refused until its stage
    let later = stages[1]
        .tools
        .iter()
        .find(|n| !stages[0].tools.contains(n))
        .unwrap();
    let server = MockServer::new(vec![call(later, json!({}))], 100, Duration::ZERO);
    let out = execute(&server, &t, &stable, &suite());
    assert_eq!(out.argument_errors, 1);
    assert_eq!(out.tool_invocations, 0);
    assert_eq!(
        out.tool_log[0].error.as_deref(),
        Some("tool is not available in this stage")
    );
}

#[test]
fn the_tool_filter_never_hides_a_tool_the_golden_path_needs() {
    let config = Config::load(&repo().join("configs/models.toml")).unwrap();
    let catalog = catalog();
    let mut hidden = Vec::new();
    for entry in std::fs::read_dir(repo().join("configs/suites")).unwrap() {
        let path = entry.unwrap().path();
        let suite = Suite::load(&path)
            .unwrap()
            .resolve(&config, &Selection::default())
            .unwrap();
        for v in suite.variants.iter().filter(|v| v.filter_top_k > 0) {
            for case in suite.load_cases(&path).unwrap() {
                let stages = case.stage_plan().unwrap();
                let shown = select_tools(&case, &catalog, v, &stages, 0).unwrap();
                for g in &case.golden {
                    if !shown.iter().any(|t| t.name == g.tool) {
                        hidden.push(format!("{} {} {}", suite.name, case.id, g.tool));
                    }
                }
            }
        }
    }
    hidden.sort();
    hidden.dedup();
    assert!(hidden.is_empty(), "{hidden:#?}");
}
