//! Tool call formats for the supported model families.
//!
//! Each format implements `ModelFormat` in its own file. To add one, write that file, then add the
//! format to `FORMATS` and an `Adapter` constant below. The shared parsing rules, prompt
//! requests and grammar assembly live here.

mod functiongemma;
mod generic;
mod liquid;
mod literal;
mod qwen;

use crate::config::ToolSpec;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Call {
    pub name: String,
    /// Small models often write `parameters`, or leave the key out for a tool with no arguments
    #[serde(default, alias = "parameters")]
    pub arguments: Value,
}
#[derive(Debug, PartialEq)]
pub enum Action {
    Call(Call),
    Final(String),
}

/// What differs between model families; everything else is shared in `Adapter`.
pub trait ModelFormat: Sync {
    /// The `adapter` value in `models.toml`.
    fn name(&self) -> &'static str;
    /// Strings that end a turn.
    fn stops(&self) -> &'static [&'static str];
    /// Opening and closing marker of one generated call.
    fn call_delimiters(&self) -> (&'static str, &'static str);
    /// Parses the text between the markers; returns the call and any text left over.
    fn parse_call<'a>(&self, body: &'a str) -> Result<(Call, &'a str)>;
    /// GBNF rules for the call syntax; `name`, `ws`, `string`, `number` and `id` come from `Adapter::grammar`.
    fn grammar(&self) -> &'static str;
    /// Rewrites calls from earlier turns that the server template renders in the wrong shape.
    fn normalize_prompt(&self, prompt: String) -> String {
        prompt
    }
    /// Version of the parsing rules for this format. Bump it when parsing changes; it is part of the study id.
    fn parse_rules(&self) -> u32 {
        3
    }
}

static FORMATS: [&dyn ModelFormat; 4] = [
    &functiongemma::FunctionGemma,
    &qwen::Qwen,
    &liquid::Liquid,
    &generic::Generic,
];

/// A model family's tool call format.
#[derive(Clone, Copy)]
pub struct Adapter(&'static dyn ModelFormat);

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Adapter({})", self.0.name())
    }
}
impl PartialEq for Adapter {
    fn eq(&self, other: &Self) -> bool {
        self.0.name() == other.0.name()
    }
}

/// Versions of every format's parsing rules, recorded in the study definition.
pub fn parse_rules() -> Value {
    FORMATS
        .iter()
        .map(|f| (f.name().to_string(), json!(f.parse_rules())))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

impl Adapter {
    pub const FUNCTIONGEMMA: Adapter = Adapter(&functiongemma::FunctionGemma);
    pub const QWEN3: Adapter = Adapter(&qwen::Qwen);
    pub const LFM2: Adapter = Adapter(&liquid::Liquid);
    pub const JINJA: Adapter = Adapter(&generic::Generic);

    pub fn new(name: &str) -> Result<Self> {
        FORMATS
            .iter()
            .find(|f| f.name() == name)
            .map(|f| Self(*f))
            .with_context(|| {
                let known: Vec<_> = FORMATS.iter().map(|f| f.name()).collect();
                format!("unknown adapter {name}; known: {}", known.join(", "))
            })
    }
    pub fn stops(self) -> Vec<&'static str> {
        self.0.stops().to_vec()
    }
    pub fn generation_stops(self) -> Vec<&'static str> {
        let mut stops = self.stops();
        stops.push(self.0.call_delimiters().1);
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
        let close = self.0.call_delimiters().1;
        // llama.cpp leaves out the delimiter that matched a stop
        if stopping_word == Some(close) && !text.ends_with(close) {
            self.parse_flagged(&format!("{text}{close}"))
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
        for stop in self.0.stops() {
            t = t.strip_suffix(stop).unwrap_or(t).trim();
        }
        if t.is_empty() {
            bail!("empty_response");
        }
        let (open, close) = self.0.call_delimiters();
        if !t.contains(open) {
            // A call written without its tags is still a call when it parses as one
            if t.starts_with('{') && !t.contains(close) {
                if let Ok((call, rest)) = self.0.parse_call(t) {
                    if rest.trim().is_empty() {
                        return Self::action(call, false);
                    }
                }
            }
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
        let (call, trailing) = self.0.parse_call(body)?;
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
        Self::action(call, leading_text)
    }
    /// Checks a parsed call's arguments and turns final_answer into the answer.
    fn action(mut call: Call, leading_text: bool) -> Result<(Action, bool)> {
        match &call.arguments {
            Value::Null => call.arguments = json!({}),
            // Arguments sent as a JSON string holding an object
            Value::String(text) => {
                if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(text) {
                    call.arguments = v;
                }
            }
            _ => {}
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
    pub fn normalize_prompt(self, prompt: String) -> String {
        self.0.normalize_prompt(prompt)
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
        format!("{}\nname ::= {names}\n{common}", self.0.grammar())
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
            (Adapter::QWEN3,r#"<tool_call>{"name":"write_file","arguments":{"path":"a.txt","content":"x,{y}"}}</tool_call>"#),
            (Adapter::FUNCTIONGEMMA,"<start_function_call>call:write_file{path:<escape>a.txt<escape>,content:<escape>x,{y}<escape>}<end_function_call>"),
            (Adapter::LFM2,"<|tool_call_start|>[write_file(path='a.txt', content='x,{y}')]<|tool_call_end|>"),
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
            (Adapter::QWEN3,r#"Reading it now.<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>"#),
            (Adapter::FUNCTIONGEMMA,"Reading it now.<start_function_call>call:read_file{path:<escape>a.txt<escape>}<end_function_call>"),
            (Adapter::LFM2,"Reading it now.<|tool_call_start|>[read_file(path='a.txt')]<|tool_call_end|>"),
        ] {
            assert_eq!(adapter.parse_flagged(raw).unwrap(), (expected(), true), "{raw}");
        }
    }

    #[test]
    fn common_json_call_slips_still_parse() {
        let want = Action::Call(Call {
            name: "read_file".into(),
            arguments: json!({"path":"a.txt"}),
        });
        for raw in [
            r#"<tool_call>{"name":"read_file","parameters":{"path":"a.txt"}}</tool_call>"#,
            r#"<tool_call>{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}</tool_call>"#,
            r#"{"name":"read_file","arguments":{"path":"a.txt"}}"#,
        ] {
            assert_eq!(Adapter::QWEN3.parse(raw).unwrap(), want, "{raw}");
        }
        assert_eq!(
            Adapter::QWEN3
                .parse(r#"<tool_call>{"name":"device_list"}</tool_call>"#)
                .unwrap(),
            Action::Call(Call {
                name: "device_list".into(),
                arguments: json!({}),
            })
        );
        // Bare JSON that is not a call, or not this model's call format, is still refused
        for (adapter, raw) in [
            (Adapter::QWEN3, r#"{"answer": 4}"#),
            (Adapter::QWEN3, r#"{"name":"f","arguments":{}} and more"#),
            (Adapter::LFM2, r#"{"name":"read_file","arguments":{}}"#),
            (
                Adapter::FUNCTIONGEMMA,
                r#"{"name":"read_file","arguments":{}}"#,
            ),
        ] {
            assert_eq!(
                adapter.parse(raw).unwrap_err().to_string(),
                "malformed_tool_call",
                "{raw}"
            );
        }
    }

    #[test]
    fn plain_calls_are_not_flagged_and_two_calls_fail_with_a_reason() {
        let one = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
        assert!(!Adapter::QWEN3.parse_flagged(one).unwrap().1);
        let two = format!("{one}{one}");
        let error = Adapter::QWEN3.parse_flagged(&two).unwrap_err();
        assert_eq!(error.to_string(), "multiple_calls");
        let open = r#"text <tool_call>{"name":"f","arguments":{}}"#;
        let error = Adapter::QWEN3.parse_flagged(open).unwrap_err();
        assert_eq!(error.to_string(), "incomplete_tool_call");
    }

    #[test]
    fn opening_markers_inside_arguments_are_literal() {
        for (adapter, raw, value) in [
            (Adapter::QWEN3, r#"<tool_call>{"name":"f","arguments":{"text":"<tool_call>"}}</tool_call>"#, "<tool_call>"),
            (Adapter::FUNCTIONGEMMA, "<start_function_call>call:f{text:<escape><start_function_call><escape>}<end_function_call>", "<start_function_call>"),
            (Adapter::LFM2, "<|tool_call_start|>[f(text='<|tool_call_start|>')]<|tool_call_end|>", "<|tool_call_start|>"),
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
        let fixed = Adapter::FUNCTIONGEMMA.normalize_prompt(rendered.into());
        assert!(fixed
            .contains("call:set_setting{name:<escape>heater<escape>,value:55}<end_function_call>"));
        assert!(fixed.ends_with("<end_function_response>"));
        let parsed = Adapter::FUNCTIONGEMMA
            .parse("<start_function_call>call:set_setting{name:<escape>heater<escape>,value:55}<end_function_call>")
            .unwrap();
        assert!(matches!(parsed, Action::Call(_)));
        let good = "<start_function_call>call:f{a:1}<end_function_call>";
        assert_eq!(Adapter::FUNCTIONGEMMA.normalize_prompt(good.into()), good);
        assert_eq!(Adapter::QWEN3.normalize_prompt(rendered.into()), rendered);
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
            assert!(Adapter::LFM2.parse(raw).is_err(), "{raw}");
        }
        assert!(Adapter::FUNCTIONGEMMA
            .parse("<start_function_call>call:f{x:<escape>unclosed}<end_function_call>")
            .is_err());
        assert!(Adapter::QWEN3.parse("<think>unfinished").is_err());
    }

    #[test]
    fn liquid_preserves_python_regex_escapes() {
        for pattern in [r"\d{4}-\d{3}", r"\w+\s+\S+", r"\/", r"\D\W"] {
            for quote in ['\'', '"'] {
                let raw = format!("<|tool_call_start|>[regex_extract(pattern={quote}{pattern}{quote}, text='orders')]<|tool_call_end|>");
                assert_eq!(
                    Adapter::LFM2.parse(&raw).unwrap(),
                    Action::Call(Call {
                        name: "regex_extract".into(),
                        arguments: json!({"pattern":pattern,"text":"orders"}),
                    })
                );
            }
        }
        assert!(Adapter::QWEN3
            .parse(r#"<tool_call>{"name":"f","arguments":{"x":"\d"}}</tool_call>"#)
            .is_err());
        assert!(Adapter::FUNCTIONGEMMA
            .parse(r#"<start_function_call>call:f{x:"\d"}<end_function_call>"#)
            .is_err());
    }

    #[test]
    fn malformed_unicode_escape_returns_an_error_without_panicking() {
        for value in [r"\u123", r"\u12🦀", r"\uZZZZ", r"\uD800"] {
            let raw = format!("<|tool_call_start|>[f(x='{value}')]<|tool_call_end|>");
            assert!(Adapter::LFM2.parse(&raw).is_err());
        }
    }

    #[test]
    fn final_answer_requires_one_string_argument() {
        assert_eq!(Adapter::QWEN3.parse(r#"<tool_call>{"name":"final_answer","arguments":{"answer":"done"}}</tool_call>"#).unwrap(),Action::Final("done".into()));
        assert!(Adapter::QWEN3
            .parse(r#"<tool_call>{"name":"final_answer","arguments":{}}</tool_call>"#)
            .is_err());
        assert!(Adapter::QWEN3
            .parse(r#"<tool_call>{"name":"final_answer","arguments":{"answer":4}}</tool_call>"#)
            .is_err());
    }

    #[test]
    fn restores_a_native_call_delimiter_removed_by_the_server() {
        let text = "<|tool_call_start|>[read_file(path='notes.txt')]";
        let action = Adapter::LFM2
            .parse_generation(text, Some("<|tool_call_end|>"))
            .unwrap();
        assert_eq!(
            action,
            Action::Call(Call {
                name: "read_file".into(),
                arguments: json!({"path":"notes.txt"}),
            })
        );
        assert!(Adapter::LFM2.parse_generation(text, None).is_err());
    }

    #[test]
    fn excessive_literal_nesting_is_rejected() {
        let text = format!(
            "<|tool_call_start|>[f(x={}1{})]<|tool_call_end|>",
            "[".repeat(40),
            "]".repeat(40)
        );
        assert!(Adapter::LFM2.parse(&text).is_err());
    }
}
