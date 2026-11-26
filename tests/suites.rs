//! Guards for the run-mode benchmark suites
//! Plans, coverage of case directories, and offline golden runs of the sandbox cases

mod support;
use a2e::{
    case::Case,
    config::{Config, ContextWindow, Mode, Selection, Suite},
    runner::Status,
};
use std::{collections::BTreeSet, path::PathBuf, time::Duration};
use support::{catalog, execute, execute_at, golden_script, repo, use_sandbox, MockServer};

const SUITES: [&str; 3] = ["test_filesystem", "test_compute", "test_data_web"];

fn path(name: &str) -> PathBuf {
    repo().join(format!("configs/suites/{name}.toml"))
}
fn load(name: &str) -> (Suite, Vec<Case>) {
    use_sandbox();
    let config = Config::load(&repo().join("configs/models.toml")).unwrap();
    let s = Suite::load(&path(name))
        .unwrap()
        .resolve(&config, &Selection::default())
        .unwrap();
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
fn pi_suite_is_a_slice_of_the_filesystem_suite_with_every_variant_on_every_case() {
    let (s, cases) = load("test_pi");
    let (_, fs) = load("test_filesystem");
    let fs_ids = fs.iter().map(|c| c.id.as_str()).collect::<BTreeSet<_>>();
    assert_eq!(cases.len(), 16);
    assert!(cases.iter().all(|c| fs_ids.contains(c.id.as_str())));
    assert!(cases.iter().any(|c| c.fixtures.sandbox.is_some()));
    assert_eq!((s.warmup, s.repeats), (0, 1));
    // Every case runs in every variant, so each comparison with autonomous is paired
    assert!(cases.iter().all(|c| c.stage_plan().is_some()));
    assert!(cases.iter().any(|c| c.stage_plan().unwrap().len() > 1));
    let base = &s.variants[0];
    assert_eq!(base.name, "autonomous");
    assert!(base.filter_top_k == 0 && !base.constrained && base.repairs == 0);
    assert_eq!(base.context, ContextWindow::Full);
    let modes = s.variants.iter().map(|v| v.mode).collect::<Vec<_>>();
    assert!(modes.contains(&Mode::Guided) && modes.contains(&Mode::GuidedStable));
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
    let (_, cases) = load("test_pi");
    let ids = cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>();
    // Listed order is kept, and nothing else in the folder is pulled in
    assert_eq!(ids[..2], ["simple-list-dir", "simple-read-file"]);
    assert_eq!(ids.len(), 16);
}

/// Every case's golden path finishes in every suite, at each of the suite's real
/// contexts and in each of its variants, with prompt tokens estimated from the
/// rendered prompt. A case that cannot fit its own correct path fails for every model.
#[test]
fn golden_paths_fit_each_suite_context_in_every_variant() {
    use_sandbox();
    let config = Config::load(&repo().join("configs/models.toml")).unwrap();
    let mut failed = Vec::new();
    let mut runs = 0;
    let mut covered = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(repo().join("configs/suites")).unwrap() {
        let path = entry.unwrap().path();
        let s = Suite::load(&path)
            .unwrap()
            .resolve(&config, &Selection::default())
            .unwrap();
        let cases = s.load_cases(&path).unwrap();
        for case in cases.iter().filter(|c| !c.tags.iter().any(|t| t == "live")) {
            covered.insert(case.id.clone());
            for v in &s.variants {
                for &ctx in &s.contexts {
                    runs += 1;
                    let server = MockServer::estimating(golden_script(case));
                    let out = execute_at(&server, case, v, &s, ctx);
                    let want = if case.judge.enabled {
                        Status::AwaitingJudge
                    } else {
                        Status::Success
                    };
                    if out.status != want {
                        failed.push(format!(
                            "{}/{}/{}@{ctx}: {:?}",
                            s.name, v.name, case.id, out.reason
                        ));
                    }
                }
            }
        }
    }
    // Cases no suite lists still run once, at the shared context
    let all = a2e::case::load_all(&[repo().join("configs/cases")]).unwrap();
    let smoke = Suite::load(&path("smoke"))
        .unwrap()
        .resolve(&config, &Selection::default())
        .unwrap();
    for case in all.iter().filter(|c| !c.tags.iter().any(|t| t == "live")) {
        if covered.contains(&case.id) {
            continue;
        }
        runs += 1;
        let server = MockServer::estimating(golden_script(case));
        let out = execute_at(
            &server,
            case,
            &a2e::config::Variant::default(),
            &smoke,
            2048,
        );
        if !matches!(out.status, Status::Success | Status::AwaitingJudge) {
            failed.push(format!("unlisted/{}@2048: {:?}", case.id, out.reason));
        }
    }
    assert!(runs > 0);
    assert!(failed.is_empty(), "{} of {runs}: {failed:#?}", failed.len());
}
