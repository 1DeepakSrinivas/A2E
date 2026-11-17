//! Text and data tools: JSON lookup, CSV filter, regex extraction, word count.
//!
//! Small bounded inputs; regexes use the linear-time engine with a size cap.

use super::{files::read_text, string, Registry, World};
use crate::text_limit;
use anyhow::{bail, Context, Result};
use regex::RegexBuilder;
use serde_json::{json, Value};

const ROW_LIMIT: usize = 20;

/// Value at a dotted path; numeric segments index arrays.
fn json_get(w: &mut World, a: &Value) -> Result<Value> {
    let doc: Value = serde_json::from_str(&read_text(w, string(a, "path")?)?)
        .context("file is not valid JSON")?;
    let key = string(a, "key")?;
    let mut node = &doc;
    for part in key.split('.').filter(|p| !p.is_empty()) {
        node = match part.parse::<usize>() {
            Ok(i) if node.is_array() => node.get(i),
            _ => node.get(part),
        }
        .with_context(|| format!("key {part} not found"))?;
    }
    Ok(json!({"value":node}))
}
fn compare(cell: &str, op: &str, want: &str) -> Result<bool> {
    let numbers = cell
        .trim()
        .parse::<f64>()
        .ok()
        .zip(want.trim().parse::<f64>().ok());
    Ok(match (op, numbers) {
        ("eq", Some((a, b))) => a == b,
        ("ne", Some((a, b))) => a != b,
        ("gt", Some((a, b))) => a > b,
        ("lt", Some((a, b))) => a < b,
        ("eq", None) => cell.trim() == want.trim(),
        ("ne", None) => cell.trim() != want.trim(),
        ("contains", _) => cell.to_lowercase().contains(&want.to_lowercase()),
        ("gt" | "lt", None) => bail!("gt and lt need numeric values"),
        _ => bail!("unknown operation {op}"),
    })
}
/// Rows whose column satisfies the comparison; plain comma-separated cells.
fn csv_filter(w: &mut World, a: &Value) -> Result<Value> {
    let body = read_text(w, string(a, "path")?)?;
    let mut lines = body.lines();
    let header = lines
        .next()
        .context("empty CSV")?
        .split(',')
        .map(|c| c.trim().to_owned())
        .collect::<Vec<_>>();
    let column = string(a, "column")?;
    let index = header
        .iter()
        .position(|h| h.eq_ignore_ascii_case(column))
        .with_context(|| format!("unknown column {column}"))?;
    let (op, want) = (string(a, "op")?, string(a, "value")?);
    if !["eq", "ne", "gt", "lt", "contains"].contains(&op) {
        bail!("unknown operation {op}");
    }
    let mut rows = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let cells = line
            .split(',')
            .map(|c| c.trim().to_owned())
            .collect::<Vec<_>>();
        if cells
            .get(index)
            .is_some_and(|c| compare(c, op, want).unwrap_or(false))
        {
            rows.push(cells);
        }
    }
    let truncated = rows.len() > ROW_LIMIT;
    rows.truncate(ROW_LIMIT);
    Ok(json!({"header":header,"rows":rows,"truncated":truncated}))
}
/// First capture group per match, or the whole match when there is none.
fn regex_extract(_: &mut World, a: &Value) -> Result<Value> {
    let re = RegexBuilder::new(string(a, "pattern")?)
        .size_limit(1 << 16)
        .build()
        .map_err(|e| anyhow::anyhow!("invalid pattern: {e}"))?;
    let matches = re
        .captures_iter(string(a, "text")?)
        .take(ROW_LIMIT)
        .map(|c| {
            text_limit(
                c.get(1).or_else(|| c.get(0)).map_or("", |m| m.as_str()),
                200,
            )
        })
        .collect::<Vec<_>>();
    Ok(json!({"matches":matches}))
}
fn word_count(_: &mut World, a: &Value) -> Result<Value> {
    let text = string(a, "text")?;
    Ok(
        json!({"words":text.split_whitespace().count(),"lines":text.lines().count(),"characters":text.chars().count()}),
    )
}
pub fn register(r: &mut Registry) {
    r.insert("json_get", json_get);
    r.insert("csv_filter", csv_filter);
    r.insert("regex_extract", regex_extract);
    r.insert("word_count", word_count);
}

#[cfg(test)]
mod tests {
    use crate::tools::test_support::host;
    use serde_json::json;

    #[test]
    fn json_get_follows_object_keys_and_array_indexes() {
        let (mut h, _dir) = host(
            json!({"files":{"c.json":"{\"items\":[{\"name\":\"a\"},{\"name\":\"b\"}],\"n\":3}"}}),
        );
        let get = |h: &mut super::super::Builtins, k: &str| {
            h.call("json_get", json!({"path":"c.json","key":k}))
        };
        assert_eq!(get(&mut h, "items.1.name").unwrap()["value"], "b");
        assert_eq!(get(&mut h, "n").unwrap()["value"], 3);
        assert!(get(&mut h, "items.5").is_err());
    }
    #[test]
    fn csv_filter_compares_numbers_and_text() {
        let csv = "name,qty\nbolt,5\nnut,12\nwasher,40";
        let (mut h, _dir) = host(json!({"files":{"s.csv":csv}}));
        let rows = |h: &mut super::super::Builtins, op: &str, v: &str| {
            h.call(
                "csv_filter",
                json!({"path":"s.csv","column":"qty","op":op,"value":v}),
            )
            .unwrap()["rows"]
                .as_array()
                .unwrap()
                .len()
        };
        assert_eq!(rows(&mut h, "gt", "10"), 2);
        assert_eq!(rows(&mut h, "lt", "10"), 1);
        assert_eq!(rows(&mut h, "eq", "12"), 1);
        let by_name = h
            .call(
                "csv_filter",
                json!({"path":"s.csv","column":"name","op":"contains","value":"NU"}),
            )
            .unwrap();
        assert_eq!(by_name["rows"][0][0], "nut");
        assert!(h
            .call(
                "csv_filter",
                json!({"path":"s.csv","column":"zzz","op":"eq","value":"1"})
            )
            .is_err());
        assert!(h
            .call(
                "csv_filter",
                json!({"path":"s.csv","column":"qty","op":"like","value":"1"})
            )
            .is_err());
    }
    #[test]
    fn regex_extract_prefers_the_first_group_and_rejects_bad_patterns() {
        let (mut h, _dir) = host(json!({}));
        let r = h
            .call(
                "regex_extract",
                json!({"text":"id=7 id=9","pattern":"id=(\\d+)"}),
            )
            .unwrap();
        assert_eq!(r["matches"], json!(["7", "9"]));
        assert!(h
            .call("regex_extract", json!({"text":"x","pattern":"("}))
            .is_err());
    }
    #[test]
    fn word_count_reports_words_lines_and_characters() {
        let (mut h, _dir) = host(json!({}));
        let r = h
            .call("word_count", json!({"text":"one two\nthree"}))
            .unwrap();
        assert_eq!(
            (
                r["words"].as_u64(),
                r["lines"].as_u64(),
                r["characters"].as_u64()
            ),
            (Some(3), Some(2), Some(13))
        );
    }
}
