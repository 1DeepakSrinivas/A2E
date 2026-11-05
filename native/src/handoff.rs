// handoff.rs
// Judge packet builder: only the fields a case lists in judge.handoff
// Prompt, answer, calls, evidence, written files, state diff, and required facts

use crate::{
    case::{Case, Handoff},
    text_limit,
};
use serde_json::{json, Map, Value};

const FILE_BYTES: usize = 4000;
/// State domains compared for the diff; the snapshot also holds web bookkeeping.
const DOMAINS: [&str; 5] = ["files", "devices", "kv", "notes", "todos"];

/// Changed leaves as `{before, after}`; objects recurse, anything else is replaced whole.
fn diff(before: &Value, after: &Value) -> Option<Value> {
    if before == after {
        return None;
    }
    match (before, after) {
        (Value::Object(b), Value::Object(a)) => {
            let mut out = Map::new();
            for key in b.keys().chain(a.keys()) {
                let d = diff(
                    b.get(key).unwrap_or(&Value::Null),
                    a.get(key).unwrap_or(&Value::Null),
                );
                if let Some(d) = d {
                    out.insert(key.clone(), d);
                }
            }
            Some(Value::Object(out))
        }
        _ => Some(json!({"before":before,"after":after})),
    }
}
fn domains(state: &Value) -> Value {
    let mut out = Map::new();
    for key in DOMAINS {
        out.insert(key.into(), state.get(key).cloned().unwrap_or(Value::Null));
    }
    Value::Object(out)
}
/// Output files when the case names them, otherwise every file the run added or changed.
fn written_files(case: &Case, before: &Value, after: &Value) -> Value {
    let files = after["files"].as_object();
    let mut out = Map::new();
    for (path, body) in files.into_iter().flatten() {
        let wanted = if case.output.files.is_empty() {
            before["files"].get(path) != Some(body)
        } else {
            case.output.files.contains(path)
        };
        if wanted {
            out.insert(
                path.clone(),
                json!(text_limit(body.as_str().unwrap_or(""), FILE_BYTES)),
            );
        }
    }
    Value::Object(out)
}
/// Packet for one attempt; `row` carries answer, evidence, calls, and final_state.
pub fn packet(case: &Case, row: &Value) -> Value {
    let mut before = json!({});
    if let Ok(v) = serde_json::to_value(&case.fixtures) {
        before = domains(&v);
    }
    let after = domains(&row["final_state"]);
    let mut out = Map::new();
    for field in &case.judge.handoff {
        let (key, value) = match field {
            Handoff::Prompt => ("prompt", json!(case.scope.prompt)),
            Handoff::Answer => ("answer", row["answer"].clone()),
            Handoff::Calls => ("calls", row["calls"].clone()),
            Handoff::Evidence => ("evidence", row["evidence"].clone()),
            Handoff::Files => ("files", written_files(case, &before, &after)),
            Handoff::StateDiff => (
                "state_diff",
                diff(&before, &after).unwrap_or_else(|| json!({})),
            ),
            Handoff::RequiredFacts => ("required_facts", json!(case.judge.required_facts)),
        };
        out.insert(key.into(), value);
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(handoff: &str) -> Case {
        Case::from_toml(&format!(
            r#"
id = "h-1"
category = "chain"
split = "dev"
[scope]
prompt = "Summarise the page."
[tools]
allowed = ["write_file"]
[fixtures.files]
"keep.txt" = "same"
"edit.txt" = "old"
[[check]]
kind = "calls_at_most"
max = 3
[judge]
enabled = true
handoff = {handoff}
required_facts = ["fact one"]
"#
        ))
        .unwrap()
    }
    fn row() -> Value {
        json!({"answer":"done","evidence":[{"source_id":"s1"}],"calls":[{"tool":"write_file","ok":true}],
               "final_state":{"files":{"keep.txt":"same","edit.txt":"new","out.txt":"fresh"},"kv":{"a":"1"},"devices":{},"notes":[],"todos":[]}})
    }

    #[test]
    fn only_the_listed_fields_are_handed_over() {
        let p = packet(&case(r#"["prompt","answer"]"#), &row());
        let keys = p.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys, ["answer", "prompt"]);
        let p = packet(&case(r#"["required_facts","evidence","calls"]"#), &row());
        assert_eq!(p["required_facts"], json!(["fact one"]));
        assert_eq!(p["evidence"][0]["source_id"], "s1");
        assert_eq!(p["calls"][0]["tool"], "write_file");
        assert!(p.get("answer").is_none());
    }
    #[test]
    fn files_default_to_what_the_run_wrote_or_changed() {
        let p = packet(&case(r#"["files"]"#), &row());
        assert_eq!(p["files"], json!({"edit.txt":"new","out.txt":"fresh"}));
        let mut c = case(r#"["files"]"#);
        c.output.files = vec!["out.txt".into()];
        assert_eq!(packet(&c, &row())["files"], json!({"out.txt":"fresh"}));
    }
    #[test]
    fn state_diff_lists_only_changed_leaves() {
        let d = packet(&case(r#"["state_diff"]"#), &row())["state_diff"].clone();
        assert_eq!(
            d["files"]["edit.txt"],
            json!({"before":"old","after":"new"})
        );
        assert_eq!(
            d["files"]["out.txt"],
            json!({"before":null,"after":"fresh"})
        );
        assert!(d["files"].get("keep.txt").is_none());
        assert_eq!(d["kv"]["a"], json!({"before":null,"after":"1"}));
    }
}
