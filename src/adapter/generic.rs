//! Default format for models without their own adapter: a JSON call between `<tool_call>` tags.
//!
//! Most recent instruction-tuned GGUF models use this form in their chat template. The server
//! renders the tools through that template, so only the call syntax and the end markers are fixed here.

use super::{Call, ModelFormat};
use anyhow::{Context, Result};

pub struct Generic;

impl ModelFormat for Generic {
    fn name(&self) -> &'static str {
        "jinja"
    }
    fn stops(&self) -> &'static [&'static str] {
        &[
            "<|im_end|>",
            "<|eot_id|>",
            "<|end|>",
            "<end_of_turn>",
            "</s>",
        ]
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

#[cfg(test)]
mod tests {
    use super::super::{Action, Adapter};

    #[test]
    fn reads_a_tagged_json_call_and_strips_an_end_marker() {
        let a = Adapter::new("jinja").unwrap();
        let text = "<tool_call>{\"name\":\"get_time\",\"arguments\":{\"zone\":\"UTC\"}}</tool_call><|eot_id|>";
        match a.parse(text).unwrap() {
            Action::Call(c) => {
                assert_eq!(c.name, "get_time");
                assert_eq!(c.arguments["zone"], "UTC");
            }
            other => panic!("{other:?}"),
        }
    }
}
