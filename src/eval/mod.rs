//! Outcome checks over observed calls, final tool state, and the answer.
//!
//! Argument matching is tolerant. Failure classes support error analysis.

pub mod check;
pub mod checks;

use crate::{
    adapter::Call,
    case::{Case, MatchMode},
};
pub use check::Check;
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
    NoToolUse,
    ToolErrorAbandoned,
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
    /// Calls whose tool ran successfully.
    pub calls: &'a [Call],
    /// Every parsed call, including schema errors and tool errors.
    pub attempted: &'a [Call],
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
pub(crate) fn numbers(text: &str) -> Vec<f64> {
    let cleaned = text.replace(',', "");
    Regex::new(r"-?\d+(?:\.\d+)?")
        .map(|re| {
            re.find_iter(&cleaned)
                .filter_map(|m| m.as_str().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}
pub(crate) fn found(pattern: &str, text: &str) -> bool {
    Regex::new(pattern).is_ok_and(|re| re.is_match(text))
}
/// Wording that marks an answer as a request for information rather than a result.
const REQUEST: &str = r"(?i)\?|\b(which|what|where|when|who|whom|how (many|much|long|far)|could you|can you|would you|will you|specify|provide|clarify|confirm|tell me|let me know|(i|i'll|i will|i would|i'd|we) need|missing|unclear|unspecified|not (specified|provided|given|stated|mentioned|clear|sure)|(didn't|did not|haven't|have not) (specify|specified|provide|provided|say|said|mention|mentioned|give|given)|no .{0,40}\b(was|were|is|has been) (specified|provided|given|stated|mentioned))\b";
/// True when the answer asks for information and names one of the topics.
/// Topics match whole words, ignoring case and a plural `s`.
pub fn asks_for(topics: &[String], answer: &str) -> bool {
    found(REQUEST, answer)
        && topics.iter().any(|t| {
            found(
                &format!(r"(?i)\b{}(s|es)?\b", regex::escape(t.trim())),
                answer,
            )
        })
}
pub(crate) fn urls_match(a: &str, b: &str) -> bool {
    a.trim_end_matches('/') == b.trim_end_matches('/')
}
pub(crate) fn string_list(v: &Value) -> impl Iterator<Item = &str> {
    v.as_array().into_iter().flatten().filter_map(Value::as_str)
}
pub(crate) fn result(kind: &str, label: String, passed: bool, class: FailureClass) -> CheckResult {
    CheckResult {
        kind: kind.into(),
        label,
        passed,
        class: (!passed).then_some(class),
    }
}
/// Share of milestone checks passed; any failed minefield (`no_call`, `calls_at_most`) zeroes it.
pub fn milestone_score(checks: &[CheckResult]) -> Option<f64> {
    if checks.is_empty() {
        return None;
    }
    let is_mine = |c: &CheckResult| matches!(c.kind.as_str(), "no_call" | "calls_at_most");
    if checks.iter().any(|c| is_mine(c) && !c.passed) {
        return Some(0.);
    }
    let (passed, total) = checks
        .iter()
        .filter(|c| !is_mine(c))
        .fold((0., 0.), |(p, t), c| {
            (p + f64::from(u8::from(c.passed)), t + 1.)
        });
    Some(if total == 0. { 1. } else { passed / total })
}
/// Every configured check plus an existence check for each required output file.
pub fn evaluate(case: &Case, obs: &Observation) -> Vec<CheckResult> {
    let mut out = Vec::with_capacity(case.checks.len() + case.output.files.len());
    out.extend(case.checks.iter().map(|c| c.evaluate(obs)));
    for path in &case.output.files {
        out.push(result(
            "output_file",
            format!("output_file {path}"),
            obs.state["files"].get(path).is_some(),
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
                attempted: calls,
                state: &state,
                answer,
            },
        )
    }

    #[test]
    fn clarifying_questions_pass_in_any_phrasing_and_statements_do_not() {
        let topics = ["file", "path", "name"].map(String::from);
        for ok in [
            "Which file should I read?",
            "I would be happy to read your file. Please provide the path to the file you wish to read.",
            "I need the file name before I can read it.",
            "You did not specify a path.",
            "There are two files, a.txt and b.txt. Which one?",
        ] {
            assert!(asks_for(&topics, ok), "{ok}");
        }
        for bad in [
            "I would recommend reading the file.",
            "The file says alpha.",
            "I cannot assist with reading files.",
            "Which one do you mean?",
            "",
        ] {
            assert!(!asks_for(&topics, bad), "{bad}");
        }
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
    fn failed_calls_count_as_attempts_for_minefields_and_argument_classes() {
        let none = case("[[check]]\nkind = \"no_call\"\ntool = \"calculate\"");
        let attempted = [call("calculate", json!({}))];
        let obs = Observation {
            calls: &[],
            attempted: &attempted,
            state: &json!({}),
            answer: "",
        };
        assert!(!evaluate(&none, &obs)[0].passed);
        let want = case("[[check]]\nkind = \"call\"\ntool = \"calculate\"");
        assert_eq!(
            evaluate(&want, &obs)[0].class,
            Some(FailureClass::WrongArgs)
        );
        let max = case("[[check]]\nkind = \"calls_at_most\"\nmax = 0");
        assert!(!evaluate(&max, &obs)[0].passed);
    }
    #[test]
    fn milestone_score_is_the_passed_share_and_minefields_zero_it() {
        let check = |kind: &str, passed| CheckResult {
            kind: kind.into(),
            label: kind.into(),
            passed,
            class: None,
        };
        assert_eq!(milestone_score(&[]), None);
        let partial = [
            check("file", true),
            check("answer", false),
            check("no_call", true),
        ];
        assert_eq!(milestone_score(&partial), Some(0.5));
        let hit = [check("file", true), check("no_call", false)];
        assert_eq!(milestone_score(&hit), Some(0.));
        assert_eq!(milestone_score(&[check("calls_at_most", true)]), Some(1.));
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
