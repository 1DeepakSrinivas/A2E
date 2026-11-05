// tools/misc.rs
// Arithmetic and simulated device tools
// Two-operand calculation and bounded device settings

use super::{string, Registry, World};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

fn calculate(_: &mut World, a: &Value) -> Result<Value> {
    let x = a["a"].as_f64().context("a must be numeric")?;
    let y = a["b"].as_f64().context("b must be numeric")?;
    let v = match string(a, "operation")? {
        "add" => x + y,
        "subtract" => x - y,
        "multiply" => x * y,
        "divide" if y != 0. => x / y,
        _ => bail!("invalid operation or division by zero"),
    };
    if !v.is_finite() {
        bail!("nonfinite result");
    }
    Ok(json!({"result":v}))
}
fn device_list(w: &mut World, _: &Value) -> Result<Value> {
    Ok(json!({"devices":w.devices.keys().collect::<Vec<_>>()}))
}
fn device_status(w: &mut World, a: &Value) -> Result<Value> {
    Ok(w.devices
        .get(string(a, "name")?)
        .context("unknown device")?
        .clone())
}
fn set_setting(w: &mut World, a: &Value) -> Result<Value> {
    let value = a["value"].as_i64().context("setting must be an integer")?;
    if !(0..=100).contains(&value) {
        bail!("setting outside permitted range");
    }
    let d = w
        .devices
        .get_mut(string(a, "name")?)
        .context("unknown device")?;
    d["setting"] = json!(value);
    Ok(json!({"setting":value}))
}
pub fn register(r: &mut Registry) {
    r.insert("calculate", calculate);
    r.insert("device_list", device_list);
    r.insert("device_status", device_status);
    r.insert("set_setting", set_setting);
}
