// rust_tool.rs
// Minimal external tool built with the Rust SDK
// Uppercase conversion and JSON Lines dispatch over stdin and stdout

use a2e_native::sdk::{serve, ToolHost};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

struct Uppercase;
impl ToolHost for Uppercase {
    fn initialize(&mut self, _: Value) -> Result<Value> {
        Ok(json!({}))
    }
    fn snapshot(&mut self) -> Result<Value> {
        Ok(json!({}))
    }
    fn call(&mut self, name: &str, args: Value) -> Result<Value> {
        if name != "uppercase" {
            bail!("unknown tool");
        }
        Ok(json!({"text": args["text"].as_str().context("text is required")?.to_uppercase()}))
    }
}
fn main() -> Result<()> {
    serve(&mut Uppercase)
}
