//! FunctionGemma: `call:name{key:<escape>text<escape>}` between function call markers.

use super::{literal::Literal, Call, ModelFormat};
use anyhow::Result;
use serde_json::Value;

pub struct FunctionGemma;

/// FunctionGemma literal: strings in `<escape>` markers, other values as JSON-like text.
fn gemma_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("<escape>{s}<escape>"),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(gemma_value).collect::<Vec<_>>().join(",")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{k}:{}", gemma_value(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        other => other.to_string(),
    }
}

impl ModelFormat for FunctionGemma {
    fn name(&self) -> &'static str {
        "functiongemma"
    }
    fn stops(&self) -> &'static [&'static str] {
        &["<end_of_turn>", "<start_function_response>"]
    }
    fn call_delimiters(&self) -> (&'static str, &'static str) {
        ("<start_function_call>", "<end_function_call>")
    }
    fn parse_call<'a>(&self, body: &'a str) -> Result<(Call, &'a str)> {
        let mut p = Literal::new(body, false);
        p.expect("call:")?;
        let name = p.identifier()?;
        let arguments = p.value(0)?;
        Ok((Call { name, arguments }, p.s))
    }
    fn grammar(&self) -> &'static str {
        r#"root ::= "<start_function_call>call:" name object "<end_function_call>"
escaped ::= "<escape>" [^<]* "<escape>"
value ::= escaped | number | object | array | "true" | "false" | "null"
object ::= "{" ws (id ws ":" ws value (ws "," ws id ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
"#
    }
    /// Rewrites tool calls from earlier turns that the server template renders as a JSON string.
    fn normalize_prompt(&self, prompt: String) -> String {
        const OPEN: &str = "<start_function_call>call:";
        const CLOSE: &str = "<end_function_call>";
        let mut out = String::with_capacity(prompt.len());
        let mut rest = prompt.as_str();
        while let Some(at) = rest.find(OPEN) {
            let head = at + OPEN.len();
            out.push_str(&rest[..head]);
            rest = &rest[head..];
            let fixed = (|| {
                let (name, body) = rest.split_once('{')?;
                let body = body.trim_start();
                if !body.starts_with('{') {
                    return None;
                }
                let mut values = serde_json::Deserializer::from_str(body).into_iter::<Value>();
                let value = values.next()?.ok()?;
                let after = body[values.byte_offset()..].strip_prefix('}')?;
                after.starts_with(CLOSE).then_some(())?;
                let args = value.as_object()?;
                let fields = args
                    .iter()
                    .map(|(k, v)| format!("{k}:{}", gemma_value(v)))
                    .collect::<Vec<_>>()
                    .join(",");
                Some((format!("{name}{{{fields}}}"), after))
            })();
            if let Some((call, after)) = fixed {
                out.push_str(&call);
                rest = after;
            }
        }
        out.push_str(rest);
        out
    }
}
