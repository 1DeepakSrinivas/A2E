//! Guards for the run-mode benchmark suites
//! Plans, coverage of case directories, and offline golden runs of the sandbox cases

mod support;
use a2e::{
    case::Case,
    config::{Config, ContextWindow, Mode, Suite},
    runner::Status,
};
use std::{collections::BTreeSet, path::PathBuf, time::Duration};
use support::{catalog, execute, golden_script, repo, use_sandbox, MockServer};

const SUITES: [&str; 3] = ["test_filesystem", "test_compute", "test_data_web"];

fn path(name: &str) -> PathBuf {
    repo().join(format!("configs/suites/{name}.toml"))
}
fn load(name: &str) -> (Suite, Vec<Case>) {
    use_sandbox();
    let s = Suite::load(&path(name)).unwrap();
    let cases = s.load_cases(&path(name)).unwrap();
    (s, cases)
}

#[test]
fn each_suite_plans_with_the_shipped_models_and_tools() {
    let cfg = Config::load(&repo().join("configs/models.toml")).unwrap();
    for name in SUITES {
        let (s, cases) = load(name);
        assert!(!cases.is_empty(), "{name}");
        a2e::runner::plan(&cfg, &s, &cases, &catalog()).unwrap();
        assert!(cases.iter().all(|c| c.split == s.split), "{name}");
    }
}

#[test]
fn suites_split_every_offline_case_by_family_without_overlap() {
    let mut seen = BTreeSet::new();
    for (name, family) in SUITES.iter().zip(["fs", "compute", "data"]) {
        let (_, cases) = load(name);
        for c in &cases {
            assert!(c.tags.iter().any(|t| t == family), "{name}/{}", c.id);
            assert!(seen.insert(c.id.clone()), "{} is in two suites", c.id);
        }
    }
    let all = a2e::case::load_all(&[repo().join("configs/cases")]).unwrap();
    for c in all
        .iter()
        .filter(|c| c.split == a2e::case::Split::Dev && !c.tags.iter().any(|t| t == "live"))
    {
        assert!(seen.contains(&c.id), "{} is in no test suite", c.id);
    }
}

#[test]
fn filesystem_suite_uses_the_sandbox_and_every_ablation_knob() {
    let (s, cases) = load("test_filesystem");
    assert!(cases.iter().any(|c| c.fixtures.sandbox.is_some()));
    assert!(s.variants.iter().any(|v| v.mode == Mode::Guided));
    assert!(s.variants.iter().any(|v| v.filter_top_k > 0));
    assert!(s
        .variants
        .iter()
        .any(|v| v.context == ContextWindow::Recent));
    assert!(s.variants.iter().any(|v| v.constrained));
    assert!(s.variants.iter().any(|v| v.repairs > 0));
}

#[test]
fn golden_sequences_pass_in_every_suite_variant() {
    for name in SUITES {
        let (s, cases) = load(name);
        for case in &cases {
            for v in &s.variants {
                if v.filter_top_k > 0 || v.mode.staged() {
                    continue;
                }
                let server = MockServer::new(golden_script(case), 100, Duration::ZERO);
                let out = execute(&server, case, v, &s);
                let want = if case.judge.enabled {
                    Status::AwaitingJudge
                } else {
                    Status::Success
                };
                assert_eq!(
                    out.status, want,
                    "{name}/{}/{}: {:?} {:?}",
                    v.name, case.id, out.reason, out.checks
                );
            }
        }
    }
}

#[test]
fn pi_100_suite_is_a_small_slice_of_the_filesystem_suite() {
    let (s, cases) = load("test_pi_100");
    let (_, fs) = load("test_filesystem");
    let fs_ids = fs.iter().map(|c| c.id.as_str()).collect::<BTreeSet<_>>();
    assert_eq!(cases.len(), 16);
    assert!(cases.iter().all(|c| fs_ids.contains(c.id.as_str())));
    assert!(cases.iter().any(|c| c.fixtures.sandbox.is_some()));
    assert!(cases.len() * s.models.len() * s.variants.len() * s.repeats <= 100);
    assert_eq!(s.warmup, 0);
    for case in &cases {
        let server = MockServer::new(golden_script(case), 100, Duration::ZERO);
        let out = execute(&server, case, &s.variants[0], &s);
        assert_eq!(out.status, Status::Success, "{}: {:?}", case.id, out.reason);
    }
}

#[test]
fn pi_ablation_suite_runs_every_knob_on_the_same_cases_as_its_baseline() {
    let (s, cases) = load("test_pi_ablation");
    let (_, pi) = load("test_pi_100");
    let pi_ids = pi.iter().map(|c| c.id.as_str()).collect::<BTreeSet<_>>();
    assert_eq!(cases.len(), 6);
    assert!(cases.iter().all(|c| pi_ids.contains(c.id.as_str())));
    assert!(cases.len() * s.models.len() * s.variants.len() * s.repeats <= 100);
    // The first variant is the baseline the others are compared with
    let base = &s.variants[0];
    assert_eq!(base.name, "autonomous");
    assert!(base.filter_top_k == 0 && !base.constrained && base.repairs == 0);
    assert_eq!(base.context, ContextWindow::Full);
    assert!(s.variants.iter().all(|v| v.mode == Mode::Autonomous));
    assert!(s.variants.iter().any(|v| v.filter_top_k > 0));
    assert!(s
        .variants
        .iter()
        .any(|v| v.context == ContextWindow::Recent));
    assert!(s.variants.iter().any(|v| v.constrained));
    assert!(s.variants.iter().any(|v| v.repairs > 0));
    let cfg = Config::load(&repo().join("configs/models.toml")).unwrap();
    a2e::runner::plan(&cfg, &s, &cases, &catalog()).unwrap();
}
#[test]
fn a_suite_can_list_single_case_files() {
    let (_, cases) = load("test_pi_100");
    let ids = cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>();
    // Listed order is kept, and nothing else in the folder is pulled in
    assert_eq!(ids[..2], ["simple-list-dir", "simple-read-file"]);
    assert_eq!(ids.len(), 16);
}

#[test]
fn pi_guided_suite_compares_both_guided_modes_on_staged_cases() {
    let (s, cases) = load("test_pi_guided");
    let (_, pi) = load("test_pi_100");
    let pi_ids = pi.iter().map(|c| c.id.as_str()).collect::<BTreeSet<_>>();
    assert!(cases.iter().all(|c| pi_ids.contains(c.id.as_str())));
    // Every case runs in every variant, so the comparison is paired
    assert!(cases.iter().all(|c| c.stage_plan().is_some()));
    assert!(cases.iter().any(|c| c.stage_plan().unwrap().len() > 1));
    assert!(cases.len() * s.models.len() * s.variants.len() * s.repeats <= 100);
    let modes = s.variants.iter().map(|v| v.mode).collect::<Vec<_>>();
    assert_eq!(modes, [Mode::Autonomous, Mode::Guided, Mode::GuidedStable]);
    let cfg = Config::load(&repo().join("configs/models.toml")).unwrap();
    a2e::runner::plan(&cfg, &s, &cases, &catalog()).unwrap();
}
