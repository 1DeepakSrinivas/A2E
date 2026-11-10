// adapter.rs
// Native tool-call formats for FunctionGemma, Qwen, and Liquid
// Prompt messages, generation stops, output parsing, and grammar constraints

use crate::config::ToolSpec;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Adapter {
    FunctionGemma,
    Qwen,
    Liquid,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: Value,
}
#[derive(Debug, PartialEq)]
pub enum Action {
    Call(Call),
    Final(String),
}
impl Adapter {
    pub fn new(name: &str) -> Result<Self> {
        match name {
            "functiongemma" => Ok(Self::FunctionGemma),
            "qwen3" => Ok(Self::Qwen),
            "lfm2" => Ok(Self::Liquid),
            _ => bail!("unknown adapter {name}"),
        }
    }
    pub fn stops(self) -> Vec<&'static str> {
        match self {
            Self::FunctionGemma => vec!["<end_of_turn>", "<start_function_response>"],
            _ => vec!["<|im_end|>"],
        }
    }
    /// Native delimiter for one generated call.
    fn call_end(self) -> &'static str {
        match self {
            Self::Qwen => "</tool_call>",
            Self::FunctionGemma => "<end_function_call>",
            Self::Liquid => "<|tool_call_end|>",
        }
    }
    pub fn generation_stops(self) -> Vec<&'static str> {
        let mut stops = self.stops();
        stops.push(self.call_end());
        stops
    }
    pub fn parse_generation(self, text: &str, stopping_word: Option<&str>) -> Result<Action> {
        // Matched delimiter omitted from llama.cpp output
        if stopping_word == Some(self.call_end()) && !text.ends_with(self.call_end()) {
            self.parse(&format!("{text}{}", self.call_end()))
        } else {
            self.parse(text)
        }
    }
    /// Single action from a native model response.
    pub fn parse(self, text: &str) -> Result<Action> {
        let mut t = text.trim();
        if t.starts_with("<think>") {
            t = t
                .split_once("</think>")
                .context("unfinished thinking block")?
                .1
                .trim();
        }
        for stop in self.stops() {
            t = t.strip_suffix(stop).unwrap_or(t).trim();
        }
        if t.is_empty() {
            bail!("empty_response");
        }
        let (open, close) = match self {
            Self::Qwen => ("<tool_call>", "</tool_call>"),
            Self::FunctionGemma => ("<start_function_call>", "<end_function_call>"),
            Self::Liquid => ("<|tool_call_start|>", "<|tool_call_end|>"),
        };
        if !t.contains(open) {
            if t.contains(close) || t.starts_with('{') || t.starts_with('[') {
                bail!("malformed_tool_call");
            }
            return Ok(Action::Final(t.into()));
        }
        let body = t
            .strip_prefix(open)
            .context("text before tool call")?
            .strip_suffix(close)
            .context("incomplete tool call")?
            .trim();
        let call = match self {
            Self::Qwen => serde_json::from_str::<Call>(body)?,
            Self::FunctionGemma => {
                let mut p = Literal::new(body);
                p.expect("call:")?;
                let name = p.identifier()?;
                let arguments = p.value(0)?;
                p.end()?;
                Call { name, arguments }
            }
            Self::Liquid => {
                let mut p = Literal::new(body);
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
                p.end()?;
                Call {
                    name,
                    arguments: Value::Object(arguments),
                }
            }
        };
        if !call.arguments.is_object() {
            bail!("arguments must be an object");
        }
        if call.name == "final_answer" {
            if call.arguments.as_object().is_none_or(|a| a.len() != 1) {
                bail!("invalid final_answer arguments");
            }
            return Ok(Action::Final(
                call.arguments["answer"]
                    .as_str()
                    .context("final_answer requires answer string")?
                    .into(),
            ));
        }
        Ok(Action::Call(call))
    }
    /// Tool result in the server's accepted message shape.
    pub fn tool_message(self, name: &str, output: &Value) -> Value {
        json!({"role":"tool","name":name,"content":output.to_string()})
    }
    pub fn assistant(call: &Call) -> Value {
        json!({"role":"assistant","content":"","tool_calls":[{"type":"function","function":call}]})
    }
    pub fn template_request(self, messages: &[Value], tools: &[ToolSpec]) -> Value {
        let tools: Vec<_> = tools
            .iter()
            .map(|t| json!({"type":"function","function":t}))
            .collect();
        json!({"messages":messages,"tools":tools,"tool_choice":"auto","parallel_tool_calls":false,"chat_template_kwargs":{"enable_thinking":false}})
    }
    /// Output shape constraints for the native syntax.
    pub fn grammar(self, tools: &[ToolSpec]) -> String {
        let names = tools
            .iter()
            .map(|t| serde_json::to_string(&t.name).expect("strings are JSON serializable"))
            .collect::<Vec<_>>()
            .join(" | ");
        let common = r#"
ws ::= [ \t\n\r]*
number ::= "-"? ([0-9] | [1-9] [0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
id ::= [a-zA-Z_] [a-zA-Z_0-9]*
"#;
        let body = match self {
            Self::Qwen => {
                r#"root ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws "\"" name "\"" ws "," ws "\"arguments\"" ws ":" ws object ws "}" ws "</tool_call>"
value ::= string | number | object | array | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
"#
            }
            Self::FunctionGemma => {
                r#"root ::= "<start_function_call>call:" name object "<end_function_call>"
escaped ::= "<escape>" [^<]* "<escape>"
value ::= escaped | number | object | array | "true" | "false" | "null"
object ::= "{" ws (id ws ":" ws value (ws "," ws id ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
"#
            }
            Self::Liquid => {
                r#"root ::= "<|tool_call_start|>[" name "(" ws (id ws "=" ws value (ws "," ws id ws "=" ws value)*)? ws ")]<|tool_call_end|>"
single ::= "'" ([^'\\\x00-\x1F] | "\\" [\\'nr])* "'"
value ::= string | single | number | object | array | "true" | "false" | "null" | "True" | "False" | "None"
object ::= "{" ws ((string | single) ws ":" ws value (ws "," ws (string | single) ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
"#
            }
        };
        format!("{body}\nname ::= {names}\n{common}")
    }
}
pub fn final_tool() -> ToolSpec {
    ToolSpec {
        name: "final_answer".into(),
        description: "Finish the task and provide the answer with source citations when requested."
            .into(),
        parameters: json!({"type":"object","properties":{"answer":{"type":"string","description":"Final answer"}},"required":["answer"],"additionalProperties":false}),
    }
}

// Literal parser without expression evaluation
struct Literal<'a> {
    s: &'a str,
}

impl<'a> Literal<'a> {
    fn new(s: &'a str) -> Self {
        Self { s }
    }
    fn eat(&mut self, t: &str) -> bool {
        self.s = self.s.trim_start();
        if let Some(s) = self.s.strip_prefix(t) {
            self.s = s;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: &str) -> Result<()> {
        if !self.eat(t) {
            bail!("expected {t}");
        }
        Ok(())
    }
    fn end(&self) -> Result<()> {
        if !self.s.trim().is_empty() {
            bail!("trailing expression or parallel calls");
        }
        Ok(())
    }
    fn identifier(&mut self) -> Result<String> {
        self.s = self.s.trim_start();
        let n = self
            .s
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if n == 0 || self.s.as_bytes()[0].is_ascii_digit() {
            bail!("expected identifier");
        }
        let v = self.s[..n].to_owned();
        self.s = &self.s[n..];
        Ok(v)
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 32 {
            bail!("literal nesting limit");
        }
        self.s = self.s.trim_start();
        if self.eat("<escape>") {
            let (v, tail) = self
                .s
                .split_once("<escape>")
                .context("unclosed escape string")?;
            self.s = tail;
            return Ok(json!(v));
        }
        if self.s.starts_with('"') || self.s.starts_with('\'') {
            let q = self.s.chars().next().context("missing quote")?;
            self.s = &self.s[1..];
            let mut out = String::new();
            loop {
                let c = self.s.chars().next().context("unclosed string")?;
                self.s = &self.s[c.len_utf8()..];
                if c == q {
                    break;
                }
                if c == '\\' {
                    let e = self.s.chars().next().context("invalid escape")?;
                    self.s = &self.s[e.len_utf8()..];
                    match e {
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'b' => out.push('\u{0008}'),
                        'f' => out.push('\u{000c}'),
                        '\\' | '\'' | '"' | '/' => out.push(e),
                        'u' => {
                            if self.s.len() < 4 {
                                bail!("short unicode escape");
                            }
                            let v = u32::from_str_radix(&self.s[..4], 16)?;
                            self.s = &self.s[4..];
                            out.push(char::from_u32(v).context("invalid unicode scalar")?);
                        }
                        _ => bail!("invalid string escape"),
                    }
                } else {
                    out.push(c);
                }
            }
            return Ok(json!(out));
        }
        if self.eat("{") {
            let mut m = Map::new();
            if !self.eat("}") {
                loop {
                    let key = if self.s.trim_start().starts_with(['\'', '"', '<']) {
                        self.value(depth + 1)?
                            .as_str()
                            .context("nonstring key")?
                            .to_string()
                    } else {
                        self.identifier()?
                    };
                    self.expect(":")?;
                    let v = self.value(depth + 1)?;
                    if m.insert(key, v).is_some() {
                        bail!("duplicate key");
                    }
                    if self.eat("}") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            return Ok(Value::Object(m));
        }
        if self.eat("[") {
            let mut a = Vec::new();
            if !self.eat("]") {
                loop {
                    a.push(self.value(depth + 1)?);
                    if self.eat("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            return Ok(Value::Array(a));
        }
        for (token, v) in [
            ("true", json!(true)),
            ("True", json!(true)),
            ("false", json!(false)),
            ("False", json!(false)),
            ("null", Value::Null),
            ("None", Value::Null),
        ] {
            if self.eat(token) {
                return Ok(v);
            }
        }
        let n = self
            .s
            .bytes()
            .take_while(|b| b.is_ascii_digit() || b"-+.eE".contains(b))
            .count();
        if n == 0 {
            bail!("expected literal");
        }
        let v = serde_json::from_str::<Value>(&self.s[..n])?;
        self.s = &self.s[n..];
        if !v.is_number() {
            bail!("expected number");
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_native_format_with_embedded_delimiters() {
        let expected = Action::Call(Call {
            name: "write_file".into(),
            arguments: json!({"path":"a.txt","content":"x,{y}"}),
        });
        for (adapter,raw) in [
            (Adapter::Qwen,r#"<tool_call>{"name":"write_file","arguments":{"path":"a.txt","content":"x,{y}"}}</tool_call>"#),
            (Adapter::FunctionGemma,"<start_function_call>call:write_file{path:<escape>a.txt<escape>,content:<escape>x,{y}<escape>}<end_function_call>"),
            (Adapter::Liquid,"<|tool_call_start|>[write_file(path='a.txt', content='x,{y}')]<|tool_call_end|>"),
        ] {assert_eq!(adapter.parse(raw).unwrap(),expected);}
    }

    #[test]
    fn rejects_expressions_duplicate_keys_and_parallel_calls() {
        for raw in [
            "<|tool_call_start|>[f(a=__import__('os'))]<|tool_call_end|>",
            "<|tool_call_start|>[f(a=1+2)]<|tool_call_end|>",
            "<|tool_call_start|>[f(a=1,a=2)]<|tool_call_end|>",
            "<|tool_call_start|>[f(),g()]<|tool_call_end|>",
            "<|tool_call_start|>[f(a='unfinished)]<|tool_call_end|>",
        ] {
            assert!(Adapter::Liquid.parse(raw).is_err(), "{raw}");
        }
        assert!(Adapter::FunctionGemma
            .parse("<start_function_call>call:f{x:<escape>unclosed}<end_function_call>")
            .is_err());
        assert!(Adapter::Qwen.parse("<think>unfinished").is_err());
    }

    #[test]
    fn final_answer_requires_one_string_argument() {
        assert_eq!(Adapter::Qwen.parse(r#"<tool_call>{"name":"final_answer","arguments":{"answer":"done"}}</tool_call>"#).unwrap(),Action::Final("done".into()));
        assert!(Adapter::Qwen
            .parse(r#"<tool_call>{"name":"final_answer","arguments":{}}</tool_call>"#)
            .is_err());
        assert!(Adapter::Qwen
            .parse(r#"<tool_call>{"name":"final_answer","arguments":{"answer":4}}</tool_call>"#)
            .is_err());
    }

    #[test]
    fn restores_a_native_call_delimiter_removed_by_the_server() {
        let text = "<|tool_call_start|>[read_file(path='notes.txt')]";
        let action = Adapter::Liquid
            .parse_generation(text, Some("<|tool_call_end|>"))
            .unwrap();
        assert_eq!(
            action,
            Action::Call(Call {
                name: "read_file".into(),
                arguments: json!({"path":"notes.txt"}),
            })
        );
        assert!(Adapter::Liquid.parse_generation(text, None).is_err());
    }

    #[test]
    fn excessive_literal_nesting_is_rejected() {
        let text = format!(
            "<|tool_call_start|>[f(x={}1{})]<|tool_call_end|>",
            "[".repeat(40),
            "]".repeat(40)
        );
        assert!(Adapter::Liquid.parse(&text).is_err());
    }
}
