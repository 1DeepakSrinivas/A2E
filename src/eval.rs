// eval.rs
// Outcome checks over observed calls, final tool state, and the answer
// Tolerant argument matching and failure classes for error analysis

use crate::{
    adapter::Call,
    case::{tolerance, Case, Check, MatchMode},
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    MissingCall,
    WrongTool,
    WrongArgs,
    WrongOrder,
    ExtraCall,
    NoFinalAnswer,
    BadAnswer,
    StateMismatch,
    ParseFailure,
    ArgumentError,
    RepeatedAction,
    ResourceLimit,
    Infrastructure,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckResult {
    pub kind: String,
    pub label: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<FailureClass>,
}
pub struct Observation<'a> {
    pub calls: &'a [Call],
    pub state: &'a Value,
    pub answer: &'a str,
}

/// Structural match with numeric tolerance; subset mode ignores extra object keys.
pub fn values_match(expected: &Value, actual: &Value, mode: MatchMode, tol: f64) -> bool {
    match (expected, actual) {
        (Value::Number(e), Value::Number(a)) => {
            e == a
                || e.as_f64()
                    .zip(a.as_f64())
                    .is_some_and(|(e, a)| (e - a).abs() <= tol * e.abs().max(a.abs()).max(1.))
        }
        (Value::Object(e), Value::Object(a)) => {
            (mode == MatchMode::Subset || e.len() == a.len())
                && e.iter()
                    .all(|(k, ev)| a.get(k).is_some_and(|av| values_match(ev, av, mode, tol)))
        }
        (Value::Array(e), Value::Array(a)) => {
            e.len() == a.len()
                && e.iter()
                    .zip(a)
                    .all(|(ev, av)| values_match(ev, av, mode, tol))
        }
        _ => expected == actual,
    }
}
fn numbers(text: &str) -> Vec<f64> {
    let cleaned = text.replace(',', "");
    Regex::new(r"-?\d+(?:\.\d+)?")
        .map(|re| {
            re.find_iter(&cleaned)
                .filter_map(|m| m.as_str().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}
fn found(pattern: &str, text: &str) -> bool {
    Regex::new(pattern).is_ok_and(|re| re.is_match(text))
}
fn urls_match(a: &str, b: &str) -> bool {
    a.trim_end_matches('/') == b.trim_end_matches('/')
}
fn string_list(v: &Value) -> impl Iterator<Item = &str> {
    v.as_array().into_iter().flatten().filter_map(Value::as_str)
}
fn result(kind: &str, label: String, passed: bool, class: FailureClass) -> CheckResult {
    CheckResult {
        kind: kind.into(),
        label,
        passed,
        class: (!passed).then_some(class),
    }
}
fn call_check(c: &Check, calls: &[Call]) -> CheckResult {
    let Check::Call {
        tool,
        args,
        mode,
        min,
        max,
        tol,
    } = c
    else {
        unreachable!("call_check receives call checks")
    };
    let named = calls.iter().filter(|c| &c.name == tool).count();
    let hits = calls
        .iter()
        .filter(|c| &c.name == tool)
        .filter(|c| {
            args.as_ref()
                .is_none_or(|a| values_match(a, &c.arguments, *mode, *tol))
        })
        .count();
    let passed = hits >= min.unwrap_or(1) && max.is_none_or(|m| hits <= m);
    let class = if max.is_some_and(|m| hits > m) {
        FailureClass::ExtraCall
    } else if named > 0 {
        FailureClass::WrongArgs
    } else if calls.is_empty() {
        FailureClass::MissingCall
    } else {
        FailureClass::WrongTool
    };
    result("call", format!("call {tool}"), passed, class)
}
/// Every configured check plus an existence check for each required output file.
pub fn evaluate(case: &Case, obs: &Observation) -> Vec<CheckResult> {
    let files = &obs.state["files"];
    let mut out = Vec::with_capacity(case.checks.len() + case.output.files.len());
    for c in &case.checks {
        out.push(match c {
            Check::Call { .. } => call_check(c, obs.calls),
            Check::NoCall { tool } => result(
                "no_call",
                format!("no_call {tool}"),
                !obs.calls.iter().any(|c| &c.name == tool),
                FailureClass::ExtraCall,
            ),
            Check::CallOrder { tools } => {
                let mut next = tools.iter().peekable();
                for call in obs.calls {
                    if next.peek().is_some_and(|t| **t == call.name) {
                        next.next();
                    }
                }
                result(
                    "call_order",
                    format!("call_order {}", tools.join(" > ")),
                    next.peek().is_none(),
                    FailureClass::WrongOrder,
                )
            }
            Check::CallsAtMost { max } => result(
                "calls_at_most",
                format!("calls_at_most {max}"),
                obs.calls.len() <= *max,
                FailureClass::ExtraCall,
            ),
            Check::File {
                path,
                eq,
                contains,
                regex,
                json_eq,
                absent,
                present,
            } => {
                let body = files.get(path).and_then(Value::as_str);
                let passed = if *absent {
                    files.get(path).is_none()
                } else if *present {
                    files.get(path).is_some()
                } else if let Some(want) = eq {
                    body == Some(want.as_str())
                } else if let Some(part) = contains {
                    body.is_some_and(|b| b.contains(part.as_str()))
                } else if let Some(p) = regex {
                    body.is_some_and(|b| found(p, b))
                } else if let Some(want) = json_eq {
                    body.and_then(|b| serde_json::from_str::<Value>(b).ok())
                        .is_some_and(|v| values_match(want, &v, MatchMode::Exact, tolerance()))
                } else {
                    false
                };
                result(
                    "file",
                    format!("file {path}"),
                    passed,
                    FailureClass::StateMismatch,
                )
            }
            Check::State { path, eq } => result(
                "state",
                format!("state {path}"),
                obs.state
                    .pointer(path)
                    .is_some_and(|v| values_match(eq, v, MatchMode::Exact, tolerance())),
                FailureClass::StateMismatch,
            ),
            Check::Answer {
                contains,
                not_contains,
                regex,
                exact,
                number,
            } => {
                let lower = obs.answer.to_lowercase();
                let passed = if let Some(part) = contains {
                    lower.contains(&part.to_lowercase())
                } else if let Some(part) = not_contains {
                    !lower.contains(&part.to_lowercase())
                } else if let Some(p) = regex {
                    found(p, obs.answer)
                } else if let Some(want) = exact {
                    obs.answer.trim() == want.trim()
                } else if let Some(n) = number {
                    numbers(obs.answer)
                        .iter()
                        .any(|x| (x - n.value).abs() <= n.tol * x.abs().max(n.value.abs()).max(1.))
                } else {
                    false
                };
                result("answer", "answer".into(), passed, FailureClass::BadAnswer)
            }
            Check::FetchedUrl { url } => result(
                "fetched_url",
                format!("fetched_url {url}"),
                string_list(&obs.state["fetched_urls"]).any(|u| urls_match(u, url)),
                FailureClass::MissingCall,
            ),
            Check::FetchedSource { id } => result(
                "fetched_source",
                format!("fetched_source {id}"),
                string_list(&obs.state["fetched_sources"]).any(|s| s == id),
                FailureClass::MissingCall,
            ),
            Check::FetchedCount { min } => result(
                "fetched_count",
                format!("fetched_count {min}"),
                obs.state["fetched_sources"].as_array().map_or(0, Vec::len) >= *min,
                FailureClass::MissingCall,
            ),
        });
    }
    for path in &case.output.files {
        out.push(result(
            "output_file",
            format!("output_file {path}"),
            files.get(path).is_some(),
            FailureClass::StateMismatch,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: Value) -> Call {
        Call {
            name: name.into(),
            arguments: args,
        }
    }
    fn case(checks: &str) -> Case {
        Case::from_toml(&format!(
            "id = \"t\"\ncategory = \"chain\"\nsplit = \"dev\"\n[scope]\nprompt = \"p\"\n[tools]\nallowed = [\"calculate\", \"read_file\"]\n{checks}"
        ))
        .unwrap()
    }
    fn run(checks: &str, calls: &[Call], state: Value, answer: &str) -> Vec<CheckResult> {
        evaluate(
            &case(checks),
            &Observation {
                calls,
                state: &state,
                answer,
            },
        )
    }

    #[test]
    fn integer_and_float_arguments_match_within_tolerance() {
        let checks = "[[check]]\nkind = \"call\"\ntool = \"calculate\"\nargs = { a = 17, b = 6 }";
        for a in [json!(17), json!(17.0)] {
            let r = run(
                checks,
                &[call("calculate", json!({"a":a,"b":6}))],
                json!({}),
                "",
            );
            assert!(r[0].passed);
        }
        let r = run(
            checks,
            &[call("calculate", json!({"a":18,"b":6}))],
            json!({}),
            "",
        );
        assert_eq!(r[0].class, Some(FailureClass::WrongArgs));
    }
    #[test]
    fn subset_mode_allows_extra_arguments_and_exact_rejects_them() {
        let subset = "[[check]]\nkind = \"call\"\ntool = \"read_file\"\nargs = { path = \"a\" }\nmatch = \"subset\"";
        let exact = subset.replace("subset", "exact");
        let calls = [call("read_file", json!({"path":"a","extra":1}))];
        assert!(run(subset, &calls, json!({}), "")[0].passed);
        assert!(!run(&exact, &calls, json!({}), "")[0].passed);
    }
    #[test]
    fn missing_wrong_tool_and_extra_calls_are_classified() {
        let want = "[[check]]\nkind = \"call\"\ntool = \"calculate\"";
        assert_eq!(
            run(want, &[], json!({}), "")[0].class,
            Some(FailureClass::MissingCall)
        );
        assert_eq!(
            run(want, &[call("read_file", json!({}))], json!({}), "")[0].class,
            Some(FailureClass::WrongTool)
        );
        let none = "[[check]]\nkind = \"no_call\"\ntool = \"calculate\"";
        assert_eq!(
            run(none, &[call("calculate", json!({}))], json!({}), "")[0].class,
            Some(FailureClass::ExtraCall)
        );
        let max = "[[check]]\nkind = \"call\"\ntool = \"calculate\"\nmax = 1";
        let twice = [call("calculate", json!({})), call("calculate", json!({}))];
        assert_eq!(
            run(max, &twice, json!({}), "")[0].class,
            Some(FailureClass::ExtraCall)
        );
    }
    #[test]
    fn call_order_is_a_subsequence_check() {
        let order = "[[check]]\nkind = \"call_order\"\ntools = [\"read_file\", \"calculate\"]";
        let ok = [
            call("read_file", json!({})),
            call("read_file", json!({})),
            call("calculate", json!({})),
        ];
        assert!(run(order, &ok, json!({}), "")[0].passed);
        let bad = [call("calculate", json!({})), call("read_file", json!({}))];
        assert_eq!(
            run(order, &bad, json!({}), "")[0].class,
            Some(FailureClass::WrongOrder)
        );
    }
    #[test]
    fn file_checks_cover_each_matcher_and_absence() {
        let state = json!({"files":{"a.txt":"Reading: 17","j.json":"{\"x\":1.0}"}});
        let checks = [
            ("path = \"a.txt\"\neq = \"Reading: 17\"", true),
            ("path = \"a.txt\"\neq = \"reading: 17\"", false),
            ("path = \"a.txt\"\ncontains = \"17\"", true),
            ("path = \"a.txt\"\nregex = \"^Reading: \\\\d+$\"", true),
            ("path = \"j.json\"\njson_eq = { x = 1 }", true),
            ("path = \"gone.txt\"\nabsent = true", true),
            ("path = \"a.txt\"\nabsent = true", false),
            ("path = \"a.txt\"\npresent = true", true),
            ("path = \"gone.txt\"\npresent = true", false),
            ("path = \"gone.txt\"\ncontains = \"x\"", false),
        ];
        for (body, want) in checks {
            let r = run(
                &format!("[[check]]\nkind = \"file\"\n{body}"),
                &[],
                state.clone(),
                "",
            );
            assert_eq!(r[0].passed, want, "{body}");
        }
    }
    #[test]
    fn answer_checks_are_case_insensitive_for_contains_and_numeric_for_numbers() {
        let contains = "[[check]]\nkind = \"answer\"\ncontains = \"READY\"";
        assert!(run(contains, &[], json!({}), "I am ready.")[0].passed);
        let number = "[[check]]\nkind = \"answer\"\nnumber = { value = 1234.5, tol = 0.001 }";
        assert!(run(number, &[], json!({}), "The total is 1,234.5 units")[0].passed);
        assert!(!run(number, &[], json!({}), "The total is 99")[0].passed);
        let not = "[[check]]\nkind = \"answer\"\nnot_contains = \"password\"";
        assert!(!run(not, &[], json!({}), "the Password is x")[0].passed);
    }
    #[test]
    fn state_pointer_and_output_files_are_checked() {
        let state = json!({"devices":{"cooler":{"setting":80}},"files":{"out.txt":"x"}});
        let checks = "[[check]]\nkind = \"state\"\npath = \"/devices/cooler/setting\"\neq = 80\n[output]\nfiles = [\"out.txt\", \"missing.txt\"]";
        let r = run(checks, &[], state, "");
        assert!(r[0].passed && r[1].passed && !r[2].passed);
        assert_eq!(r[2].kind, "output_file");
    }
    #[test]
    fn fetched_urls_ignore_a_trailing_slash() {
        let state = json!({"fetched_urls":["https://a.org/x/"],"fetched_sources":["s1"]});
        let checks = "[[check]]\nkind = \"fetched_url\"\nurl = \"https://a.org/x\"\n[[check]]\nkind = \"fetched_source\"\nid = \"s1\"\n[[check]]\nkind = \"fetched_count\"\nmin = 2";
        let r = run(checks, &[], state, "");
        assert_eq!(
            r.iter().map(|c| c.passed).collect::<Vec<_>>(),
            [true, true, false]
        );
    }
}
