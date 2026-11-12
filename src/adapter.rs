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
/// Version of the rules that turn model text into an action; part of the study id.
pub const PARSE_RULES: u32 = 3;
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
        self.parse_generation_flagged(text, stopping_word)
            .map(|(action, _)| action)
    }
    /// Action plus whether text came before the tool call.
    pub fn parse_generation_flagged(
        self,
        text: &str,
        stopping_word: Option<&str>,
    ) -> Result<(Action, bool)> {
        // Matched delimiter omitted from llama.cpp output
        if stopping_word == Some(self.call_end()) && !text.ends_with(self.call_end()) {
            self.parse_flagged(&format!("{text}{}", self.call_end()))
        } else {
            self.parse_flagged(text)
        }
    }
    /// Single action from a native model response.
    pub fn parse(self, text: &str) -> Result<Action> {
        self.parse_flagged(text).map(|(action, _)| action)
    }
    /// Text before exactly one well-formed call is accepted for every format and flagged.
    pub fn parse_flagged(self, text: &str) -> Result<(Action, bool)> {
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
            return Ok((Action::Final(t.into()), false));
        }
        let at = t.find(open).context("malformed_tool_call")?;
        let leading_text = !t[..at].trim().is_empty();
        let body = t[at + open.len()..]
            .strip_suffix(close)
            .context("incomplete_tool_call")?
            .trim();
        let (call, trailing) = match self {
            Self::Qwen => {
                let mut calls = serde_json::Deserializer::from_str(body).into_iter::<Call>();
                let call = calls.next().context("empty tool call")??;
                (call, &body[calls.byte_offset()..])
            }
            Self::FunctionGemma => {
                let mut p = Literal::new(body, self == Self::Liquid);
                p.expect("call:")?;
                let name = p.identifier()?;
                let arguments = p.value(0)?;
                (Call { name, arguments }, p.s)
            }
            Self::Liquid => {
                let mut p = Literal::new(body, self == Self::Liquid);
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
                (
                    Call {
                        name,
                        arguments: Value::Object(arguments),
                    },
                    p.s,
                )
            }
        };
        let trailing = trailing.trim();
        if trailing
            .strip_prefix(close)
            .is_some_and(|rest| rest.trim_start().starts_with(open))
        {
            bail!("multiple_calls");
        }
        if !trailing.is_empty() {
            bail!("trailing expression or parallel calls");
        }
        if !call.arguments.is_object() {
            bail!("arguments must be an object");
        }
        if call.name == "final_answer" {
            if call.arguments.as_object().is_none_or(|a| a.len() != 1) {
                bail!("invalid final_answer arguments");
            }
            return Ok((
                Action::Final(
                    call.arguments["answer"]
                        .as_str()
                        .context("final_answer requires answer string")?
                        .into(),
                ),
                leading_text,
            ));
        }
        Ok((Action::Call(call), leading_text))
    }
    /// Tool result in the server's accepted message shape.
    pub fn tool_message(self, name: &str, output: &Value) -> Value {
        json!({"role":"tool","name":name,"content":output.to_string()})
    }
    /// Rewrites tool calls from earlier turns that the server template renders as a JSON string.
    pub fn normalize_prompt(self, prompt: String) -> String {
        if self != Self::FunctionGemma {
            return prompt;
        }
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
    python_strings: bool,
}

impl<'a> Literal<'a> {
    fn new(s: &'a str, python_strings: bool) -> Self {
        Self { s, python_strings }
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
                        '\\' | '\'' | '"' => out.push(e),
                        '/' if !self.python_strings => out.push(e),
                        'u' => {
                            if self.s.len() < 4 {
                                bail!("short unicode escape");
                            }
                            let digits = self.s.get(..4).context("invalid unicode escape")?;
                            let v = u32::from_str_radix(digits, 16)?;
                            self.s = &self.s[4..];
                            out.push(char::from_u32(v).context("invalid unicode scalar")?);
                        }
                        // Python preserves unrecognized escapes, including regex classes.
                        _ if self.python_strings
                            && !matches!(
                                e,
                                'a' | 'v' | 'x' | 'U' | 'N' | '0'..='7' | '\n' | '\r'
                            ) =>
                        {
                            out.push('\\');
                            out.push(e);
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
    fn accepts_and_flags_text_before_one_call_in_every_format() {
        let expected = || {
            Action::Call(Call {
                name: "read_file".into(),
                arguments: json!({"path":"a.txt"}),
            })
        };
        for (adapter,raw) in [
            (Adapter::Qwen,r#"Reading it now.<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>"#),
            (Adapter::FunctionGemma,"Reading it now.<start_function_call>call:read_file{path:<escape>a.txt<escape>}<end_function_call>"),
            (Adapter::Liquid,"Reading it now.<|tool_call_start|>[read_file(path='a.txt')]<|tool_call_end|>"),
        ] {
            assert_eq!(adapter.parse_flagged(raw).unwrap(), (expected(), true), "{raw}");
        }
    }

    #[test]
    fn plain_calls_are_not_flagged_and_two_calls_fail_with_a_reason() {
        let one = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
        assert!(!Adapter::Qwen.parse_flagged(one).unwrap().1);
        let two = format!("{one}{one}");
        let error = Adapter::Qwen.parse_flagged(&two).unwrap_err();
        assert_eq!(error.to_string(), "multiple_calls");
        let open = r#"text <tool_call>{"name":"f","arguments":{}}"#;
        let error = Adapter::Qwen.parse_flagged(open).unwrap_err();
        assert_eq!(error.to_string(), "incomplete_tool_call");
    }

    #[test]
    fn opening_markers_inside_arguments_are_literal() {
        for (adapter, raw, value) in [
            (Adapter::Qwen, r#"<tool_call>{"name":"f","arguments":{"text":"<tool_call>"}}</tool_call>"#, "<tool_call>"),
            (Adapter::FunctionGemma, "<start_function_call>call:f{text:<escape><start_function_call><escape>}<end_function_call>", "<start_function_call>"),
            (Adapter::Liquid, "<|tool_call_start|>[f(text='<|tool_call_start|>')]<|tool_call_end|>", "<|tool_call_start|>"),
        ] {
            assert_eq!(
                adapter.parse(raw).unwrap(),
                Action::Call(Call { name: "f".into(), arguments: json!({ "text": value }) }),
            );
            assert_eq!(
                adapter.parse(&format!("{raw}\n{raw}")).unwrap_err().to_string(),
                "multiple_calls",
            );
        }
    }

    #[test]
    fn rewrites_a_json_string_call_rendered_by_the_gemma_template() {
        let rendered = "<start_of_turn>model\n<start_function_call>call:set_setting{                    {\"name\":\"heater\",\"value\":55}}<end_function_call><start_function_response>response:set_setting{value:<escape>{\"setting\":55}<escape>}<end_function_response>";
        let fixed = Adapter::FunctionGemma.normalize_prompt(rendered.into());
        assert!(fixed
            .contains("call:set_setting{name:<escape>heater<escape>,value:55}<end_function_call>"));
        assert!(fixed.ends_with("<end_function_response>"));
        let parsed = Adapter::FunctionGemma
            .parse("<start_function_call>call:set_setting{name:<escape>heater<escape>,value:55}<end_function_call>")
            .unwrap();
        assert!(matches!(parsed, Action::Call(_)));
        let good = "<start_function_call>call:f{a:1}<end_function_call>";
        assert_eq!(Adapter::FunctionGemma.normalize_prompt(good.into()), good);
        assert_eq!(Adapter::Qwen.normalize_prompt(rendered.into()), rendered);
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
    fn liquid_preserves_python_regex_escapes() {
        for pattern in [r"\d{4}-\d{3}", r"\w+\s+\S+", r"\/", r"\D\W"] {
            for quote in ['\'', '"'] {
                let raw = format!("<|tool_call_start|>[regex_extract(pattern={quote}{pattern}{quote}, text='orders')]<|tool_call_end|>");
                assert_eq!(
                    Adapter::Liquid.parse(&raw).unwrap(),
                    Action::Call(Call {
                        name: "regex_extract".into(),
                        arguments: json!({"pattern":pattern,"text":"orders"}),
                    })
                );
            }
        }
        assert!(Adapter::Qwen
            .parse(r#"<tool_call>{"name":"f","arguments":{"x":"\d"}}</tool_call>"#)
            .is_err());
        assert!(Adapter::FunctionGemma
            .parse(r#"<start_function_call>call:f{x:"\d"}<end_function_call>"#)
            .is_err());
    }

    #[test]
    fn malformed_unicode_escape_returns_an_error_without_panicking() {
        for value in [r"\u123", r"\u12🦀", r"\uZZZZ", r"\uD800"] {
            let raw = format!("<|tool_call_start|>[f(x='{value}')]<|tool_call_end|>");
            assert!(Adapter::Liquid.parse(&raw).is_err());
        }
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
