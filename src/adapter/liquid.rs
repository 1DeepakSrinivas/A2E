//! LFM2: a Python style call, `[name(key='value')]`, between tool call markers.

use super::{literal::Literal, Call, ModelFormat};
use anyhow::{bail, Result};
use serde_json::{Map, Value};

pub struct Liquid;

impl ModelFormat for Liquid {
    fn name(&self) -> &'static str {
        "lfm2"
    }
    fn stops(&self) -> &'static [&'static str] {
        &["<|im_end|>"]
    }
    fn call_delimiters(&self) -> (&'static str, &'static str) {
        ("<|tool_call_start|>", "<|tool_call_end|>")
    }
    fn parse_call<'a>(&self, body: &'a str) -> Result<(Call, &'a str)> {
        let mut p = Literal::new(body, true);
        p.expect("[")?;
        let name = p.identifier()?;
        p.expect("(")?;
        let mut arguments = Map::new();
        if !p.eat(")") {
            loop {
                let k = p.identifier()?;
                p.expect("=")?;
                let v = p.value(0)?;
                if arguments.insert(k, v).is_some() {
                    bail!("duplicate argument");
                }
                if p.eat(")") {
                    break;
                }
                p.expect(",")?;
            }
        }
        p.expect("]")?;
        let call = Call {
            name,
            arguments: Value::Object(arguments),
        };
        Ok((call, p.s))
    }
    fn grammar(&self) -> &'static str {
        r#"root ::= "<|tool_call_start|>[" name "(" ws (id ws "=" ws value (ws "," ws id ws "=" ws value)*)? ws ")]<|tool_call_end|>"
single ::= "'" ([^'\\\x00-\x1F] | "\\" [\\'nr])* "'"
value ::= string | single | number | object | array | "true" | "false" | "null" | "True" | "False" | "None"
object ::= "{" ws ((string | single) ws ":" ws value (ws "," ws (string | single) ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
"#
    }
}
