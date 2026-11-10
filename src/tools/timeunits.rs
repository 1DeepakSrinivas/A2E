// tools/timeunits.rs
// Fixture-clock date tools and table-driven unit conversion
// Proleptic Gregorian days since 1970-01-01; results are deterministic per task

use super::{string, Registry, World};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

/// Days since 1970-01-01 for a civil date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}
/// Civil date for days since 1970-01-01.
pub(super) fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}
/// Days since the epoch for `YYYY-MM-DD`, ignoring any time suffix.
pub fn parse_date(s: &str) -> Result<i64> {
    let head = s.get(..10).context("date must look like YYYY-MM-DD")?;
    let mut parts = head.split('-').map(str::parse::<i64>);
    let (Some(Ok(y)), Some(Ok(m)), Some(Ok(d)), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        bail!("date must look like YYYY-MM-DD");
    };
    let days = days_from_civil(y, m, d);
    if civil_from_days(days) != (y, m, d) {
        bail!("invalid calendar date");
    }
    Ok(days)
}
fn format_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}
fn now(w: &mut World, _: &Value) -> Result<Value> {
    Ok(json!({"now":w.clock}))
}
fn date_add(_: &mut World, a: &Value) -> Result<Value> {
    let days = a["days"].as_i64().context("days must be an integer")?;
    if days.abs() > 36500 {
        bail!("days outside permitted range");
    }
    Ok(json!({"date":format_date(parse_date(string(a, "date")?)? + days)}))
}
fn date_diff(_: &mut World, a: &Value) -> Result<Value> {
    Ok(json!({"days":parse_date(string(a, "end")?)? - parse_date(string(a, "start")?)?}))
}
fn weekday(_: &mut World, a: &Value) -> Result<Value> {
    const NAMES: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    let days = parse_date(string(a, "date")?)?;
    Ok(json!({"weekday":NAMES[(days + 4).rem_euclid(7) as usize]}))
}

#[derive(Clone, Copy, PartialEq)]
enum Dimension {
    Length,
    Mass,
    Data,
    Temperature,
}
/// Dimension and factor to the base unit (m, kg, B); temperature units have no factor.
fn unit(name: &str) -> Option<(Dimension, f64)> {
    use Dimension::*;
    Some(match name.to_lowercase().as_str() {
        "m" => (Length, 1.),
        "km" => (Length, 1000.),
        "cm" => (Length, 0.01),
        "mm" => (Length, 0.001),
        "mi" => (Length, 1609.344),
        "yd" => (Length, 0.9144),
        "ft" => (Length, 0.3048),
        "in" => (Length, 0.0254),
        "kg" => (Mass, 1.),
        "g" => (Mass, 0.001),
        "lb" => (Mass, 0.453_592_37),
        "oz" => (Mass, 0.028_349_523_125),
        "b" => (Data, 1.),
        "kb" => (Data, 1e3),
        "mb" => (Data, 1e6),
        "gb" => (Data, 1e9),
        "kib" => (Data, 1024.),
        "mib" => (Data, 1_048_576.),
        "gib" => (Data, 1_073_741_824.),
        "c" | "f" | "k" => (Temperature, 1.),
        _ => return None,
    })
}
fn to_kelvin(v: f64, unit: &str) -> f64 {
    match unit {
        "c" => v + 273.15,
        "f" => (v - 32.) * 5. / 9. + 273.15,
        _ => v,
    }
}
fn from_kelvin(v: f64, unit: &str) -> f64 {
    match unit {
        "c" => v - 273.15,
        "f" => (v - 273.15) * 9. / 5. + 32.,
        _ => v,
    }
}
fn unit_convert(_: &mut World, a: &Value) -> Result<Value> {
    let value = a["value"].as_f64().context("value must be numeric")?;
    let (from, to) = (
        string(a, "from")?.to_lowercase(),
        string(a, "to")?.to_lowercase(),
    );
    let (Some((d1, f1)), Some((d2, f2))) = (unit(&from), unit(&to)) else {
        bail!("unknown unit");
    };
    if d1 != d2 {
        bail!("cannot convert between different kinds of unit");
    }
    let result = if d1 == Dimension::Temperature {
        from_kelvin(to_kelvin(value, &from), &to)
    } else {
        value * f1 / f2
    };
    if !result.is_finite() {
        bail!("nonfinite result");
    }
    Ok(json!({"result":(result * 1e6).round() / 1e6}))
}
pub fn register(r: &mut Registry) {
    r.insert("now", now);
    r.insert("date_add", date_add);
    r.insert("date_diff", date_diff);
    r.insert("weekday", weekday);
    r.insert("unit_convert", unit_convert);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_support::host;

    #[test]
    fn civil_conversion_round_trips_across_leap_years() {
        for days in [-800_000, -1, 0, 59, 60, 11_016, 20_513, 800_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(parse_date("1970-01-01").unwrap(), 0);
        assert_eq!(parse_date("2024-02-29").unwrap(), 19_782);
        assert!(parse_date("2023-02-29").is_err());
        assert!(parse_date("2023-13-01").is_err());
        assert!(parse_date("not a date").is_err());
    }
    #[test]
    fn date_tools_add_diff_and_name_weekdays() {
        let (mut h, _dir) = host(json!({"clock":"2026-03-01T09:00:00Z"}));
        assert_eq!(
            h.call("now", json!({})).unwrap()["now"],
            "2026-03-01T09:00:00Z"
        );
        let add = h
            .call("date_add", json!({"date":"2026-02-27","days":3}))
            .unwrap();
        assert_eq!(add["date"], "2026-03-02");
        let diff = h
            .call(
                "date_diff",
                json!({"start":"2026-01-01","end":"2026-03-01"}),
            )
            .unwrap();
        assert_eq!(diff["days"], 59);
        let day = h.call("weekday", json!({"date":"2026-03-01"})).unwrap();
        assert_eq!(day["weekday"], "Sunday");
        assert!(h
            .call("date_add", json!({"date":"2026-01-01","days":10_000_000}))
            .is_err());
    }
    #[test]
    fn unit_conversion_covers_each_dimension_and_rejects_mixes() {
        let (mut h, _dir) = host(json!({}));
        let mut convert = |v: f64, from: &str, to: &str| {
            h.call("unit_convert", json!({"value":v,"from":from,"to":to}))
        };
        assert_eq!(convert(1., "km", "m").unwrap()["result"], 1000.);
        assert_eq!(convert(1., "mi", "km").unwrap()["result"], 1.609344);
        assert_eq!(convert(100., "c", "f").unwrap()["result"], 212.);
        assert_eq!(convert(32., "F", "C").unwrap()["result"], 0.);
        assert_eq!(convert(1., "GiB", "MiB").unwrap()["result"], 1024.);
        assert!(convert(1., "kg", "m").is_err());
        assert!(convert(1., "parsec", "m").is_err());
    }
}
