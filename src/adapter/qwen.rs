//! Qwen3: a JSON call between `<tool_call>` tags.

use super::{Call, ModelFormat};
use anyhow::{Context, Result};

pub struct Qwen;

impl ModelFormat for Qwen {
    fn name(&self) -> &'static str {
        "qwen3"
    }
    fn stops(&self) -> &'static [&'static str] {
        &["<|im_end|>"]
    }
    fn call_delimiters(&self) -> (&'static str, &'static str) {
        ("<tool_call>", "</tool_call>")
    }
    fn parse_call<'a>(&self, body: &'a str) -> Result<(Call, &'a str)> {
        let mut calls = serde_json::Deserializer::from_str(body).into_iter::<Call>();
        let call = calls.next().context("empty tool call")??;
        Ok((call, &body[calls.byte_offset()..]))
    }
    fn grammar(&self) -> &'static str {
        r#"root ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws "\"" name "\"" ws "," ws "\"arguments\"" ws ":" ws object ws "}" ws "</tool_call>"
value ::= string | number | object | array | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
"#
    }
}
