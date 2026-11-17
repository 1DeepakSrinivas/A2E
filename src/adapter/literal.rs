//! Literal parser for tool call arguments, without expression evaluation.

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

pub(super) struct Literal<'a> {
    pub(super) s: &'a str,
    python_strings: bool,
}

impl<'a> Literal<'a> {
    pub(super) fn new(s: &'a str, python_strings: bool) -> Self {
        Self { s, python_strings }
    }
    pub(super) fn eat(&mut self, t: &str) -> bool {
        self.s = self.s.trim_start();
        if let Some(s) = self.s.strip_prefix(t) {
            self.s = s;
            true
        } else {
            false
        }
    }
    pub(super) fn expect(&mut self, t: &str) -> Result<()> {
        if !self.eat(t) {
            bail!("expected {t}");
        }
        Ok(())
    }
    pub(super) fn identifier(&mut self) -> Result<String> {
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
    pub(super) fn value(&mut self, depth: usize) -> Result<Value> {
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
