//! Shipped benchmark cases: every case must be solvable and not trivially passable
//! Golden calls replay through the real tools and checks

use a2e::{
    adapter::Call,
    case::{load_all, Case, Category},
    config::Catalog,
    eval::{
        checks::{AnswerCheck, FileCheck},
        evaluate, Check, CheckResult, Observation,
    },
    tools::Builtins,
};
use serde_json::{json, Value};
use std::path::PathBuf;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn library() -> Vec<Case> {
    load_all(&[repo().join("configs/cases")]).unwrap()
}
fn catalog() -> Catalog {
    Catalog::load(&repo().join("configs/tools/standard.toml")).unwrap()
}
/// Workspace tree generated once per test process.
fn sandbox() -> &'static std::path::Path {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        a2e::sandbox::init(dir.path(), a2e::sandbox::DEFAULT_SEED, false).unwrap();
        dir
    })
    .path()
}
fn is_live(case: &Case) -> bool {
    case.tags.iter().any(|t| t == "live")
}
/// Runs calls through fresh tools; only calls that succeed count as observed.
fn replay(case: &Case, calls: &[(String, Value)], answer: &str) -> Vec<CheckResult> {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = Builtins::new(dir.path().into(), false).unwrap();
    tools.set_sandbox_base(Some(sandbox().join("base")));
    tools.initialize(&case.fixtures).unwrap();
    let observed = calls
        .iter()
        .filter(|(name, args)| tools.call(name, args.clone()).is_ok())
        .map(|(name, args)| Call {
            name: name.clone(),
            arguments: args.clone(),
        })
        .collect::<Vec<_>>();
    let state = tools.snapshot().unwrap();
    evaluate(
        case,
        &Observation {
            calls: &observed,
            attempted: &observed,
            state: &state,
            answer,
        },
    )
}
fn golden(case: &Case) -> Vec<(String, Value)> {
    case.golden
        .iter()
        .map(|g| {
            (
                g.tool.clone(),
                if g.args.is_null() {
                    json!({})
                } else {
                    g.args.clone()
                },
            )
        })
        .collect()
}
fn failures(results: &[CheckResult]) -> Vec<&str> {
    results
        .iter()
        .filter(|c| !c.passed)
        .map(|c| c.label.as_str())
        .collect()
}
fn offline(case: &Case) -> bool {
    !is_live(case)
}

#[test]
fn library_covers_every_category_with_both_splits() {
    let cases = library();
    let want = [
        (Category::Simple, 16),
        (Category::Multiple, 16),
        (Category::Irrelevance, 10),
        (Category::MissingParam, 10),
        (Category::Chain, 24),
        (Category::Recovery, 12),
        (Category::Robustness, 8),
        (Category::Web, 12),
        (Category::Files, 20),
    ];
    for (category, minimum) in want {
        let of = cases.iter().filter(|c| c.category == category);
        assert!(of.clone().count() >= minimum, "{category:?}");
        for split in [a2e::case::Split::Dev, a2e::case::Split::Heldout] {
            assert!(
                of.clone().any(|c| c.split == split),
                "{category:?} {split:?}"
            );
        }
    }
}
#[test]
fn live_cases_check_the_answer_for_key_facts() {
    let state = json!({});
    for case in library().iter().filter(|c| is_live(c)) {
        let regexes = case
            .checks
            .iter()
            .filter_map(|c| match c {
                a2e::eval::Check::Answer(a) => a.regex.as_deref(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!regexes.is_empty(), "{} checks no answer", case.id);
        assert!(!regexes.contains(&"."), "{} accepts any answer", case.id);
        // The golden answer passes and an empty one does not
        let answer = |text: &str| {
            evaluate(
                case,
                &Observation {
                    calls: &[],
                    attempted: &[],
                    state: &state,
                    answer: text,
                },
            )
            .into_iter()
            .filter(|r| r.kind == "answer")
            .collect::<Vec<_>>()
        };
        let golden = answer(case.golden_answer.as_deref().unwrap_or(""));
        assert!(
            golden.iter().all(|r| r.passed),
            "{}: {:?}",
            case.id,
            failures(&golden)
        );
        assert!(
            answer("I could not find it.").iter().any(|r| !r.passed),
            "{}",
            case.id
        );
    }
}
#[test]
fn golden_arguments_match_the_tool_schemas() {
    let specs = catalog().specs();
    for case in library().iter().filter(|c| offline(c)) {
        for g in &case.golden {
            let spec = &specs[&g.tool];
            let validator = jsonschema::validator_for(&spec.parameters).unwrap();
            let args = if g.args.is_null() {
                json!({})
            } else {
                g.args.clone()
            };
            assert!(validator.is_valid(&args), "{}: {} {args}", case.id, g.tool);
        }
    }
}
#[test]
fn an_empty_attempt_never_passes() {
    for case in library().iter().filter(|c| offline(c)) {
        let results = replay(case, &[], "");
        assert!(
            !failures(&results).is_empty(),
            "{} passes with no work",
            case.id
        );
    }
}
#[test]
fn forbidden_calls_fail_irrelevance_and_missing_parameter_cases() {
    for case in library()
        .iter()
        .filter(|c| matches!(c.category, Category::Irrelevance | Category::MissingParam))
    {
        assert!(case.golden.is_empty(), "{}", case.id);
        let answer = case.golden_answer.as_deref().unwrap();
        assert!(
            failures(&replay(case, &[], answer)).is_empty(),
            "{}",
            case.id
        );
        // Calls are observed directly here, whether or not the tool would accept them
        let observed = case
            .tools
            .allowed
            .iter()
            .map(|t| Call {
                name: t.clone(),
                arguments: json!({}),
            })
            .collect::<Vec<_>>();
        let results = evaluate(
            case,
            &Observation {
                calls: &observed,
                attempted: &observed,
                state: &json!({"files":{}}),
                answer,
            },
        );
        assert!(!failures(&results).is_empty(), "{} allows a call", case.id);
    }
}
#[test]
fn a_wrong_answer_fails_cases_that_check_the_answer() {
    for case in library().iter().filter(|c| offline(c)) {
        let has_answer_check = case.checks.iter().any(|c| {
            matches!(
                c,
                Check::Answer(AnswerCheck {
                    not_contains: None,
                    ..
                })
            )
        });
        if !has_answer_check {
            continue;
        }
        let results = replay(case, &golden(case), "zzz");
        assert!(
            !failures(&results).is_empty(),
            "{} accepts any answer",
            case.id
        );
    }
}
#[test]
fn dropping_the_last_golden_call_fails_cases_with_state_checks() {
    for case in library().iter().filter(|c| offline(c)) {
        let writes = case.checks.iter().any(|c| {
            matches!(
                c,
                Check::File(FileCheck { absent: false, .. }) | Check::State(_)
            )
        });
        let mut calls = golden(case);
        if !writes || calls.len() < 2 {
            continue;
        }
        let last = calls.pop().unwrap();
        // Only mutating final steps are required to matter
        if ![
            "write_file",
            "set_setting",
            "kv_set",
            "todo_add",
            "todo_complete",
            "move_file",
            "delete_file",
            "note_add",
            "append_file",
        ]
        .contains(&last.0.as_str())
        {
            continue;
        }
        let answer = case.golden_answer.as_deref().unwrap_or("");
        assert!(
            !failures(&replay(case, &calls, answer)).is_empty(),
            "{} passes without its last step",
            case.id
        );
    }
}
#[test]
fn guided_workflows_exist_and_use_allowed_tools() {
    for case in library() {
        let stages = case
            .stage_plan()
            .unwrap_or_else(|| panic!("{} has no workflow", case.id));
        assert!(!stages.is_empty());
        for s in stages {
            assert!(
                s.tools.iter().all(|t| case.tools.allowed.contains(t)),
                "{}",
                case.id
            );
        }
    }
}
#[test]
fn judged_cases_name_the_facts_the_judge_looks_for() {
    for case in library().iter().filter(|c| c.judge.enabled) {
        assert!(!case.judge.required_facts.is_empty(), "{}", case.id);
        assert_eq!(case.category, Category::Web, "{}", case.id);
    }
}
#[test]
fn live_cases_need_no_fixtures_and_fixture_web_cases_have_pages() {
    for case in library().iter().filter(|c| c.category == Category::Web) {
        if is_live(case) {
            assert!(case.fixtures.pages.is_empty(), "{}", case.id);
        } else {
            assert!(!case.fixtures.pages.is_empty(), "{}", case.id);
        }
    }
}

#[test]
fn every_judged_case_names_a_rubric_of_the_results_project() {
    let dir = repo().join("a2e-results/rubrics");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: submodule a2e-results is not checked out");
        return;
    };
    let names: Vec<String> = entries
        .filter_map(|e| {
            e.ok()?
                .path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();
    assert!(names.iter().any(|n| n == "retrieval_summary"));
    for case in library().iter().filter(|c| c.judge.enabled) {
        assert!(names.contains(&case.judge.rubric), "{}", case.id);
    }
}
