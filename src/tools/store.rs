//! Task-scoped key-value, note, and todo stores.
//!
//! Small bounded collections with 1-based IDs.

use super::{string, words, Registry, World};
use crate::fixtures::{Fixtures, Note, Todo};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

const MAX_ITEMS: usize = 64;
const MAX_TEXT: usize = 512;

fn text<'a>(a: &'a Value, key: &str) -> Result<&'a str> {
    let s = string(a, key)?;
    if s.len() > MAX_TEXT {
        bail!("{key} exceeds {MAX_TEXT} bytes");
    }
    Ok(s)
}
fn kv_set(w: &mut World, a: &Value) -> Result<Value> {
    let (key, value) = (text(a, "key")?, text(a, "value")?);
    if !w.kv.contains_key(key) && w.kv.len() >= MAX_ITEMS {
        bail!("store limit reached");
    }
    w.kv.insert(key.into(), value.into());
    Ok(json!({"stored":key}))
}
fn kv_get(w: &mut World, a: &Value) -> Result<Value> {
    let key = string(a, "key")?;
    let value =
        w.kv.get(key)
            .with_context(|| format!("unknown key {key}"))?;
    Ok(json!({"key":key,"value":value}))
}
fn kv_delete(w: &mut World, a: &Value) -> Result<Value> {
    let key = string(a, "key")?;
    w.kv.remove(key)
        .with_context(|| format!("unknown key {key}"))?;
    Ok(json!({"deleted":key}))
}
fn kv_list(w: &mut World, _: &Value) -> Result<Value> {
    Ok(json!({"keys":w.kv.keys().collect::<Vec<_>>()}))
}
fn note_json(i: usize, n: &Note) -> Value {
    json!({"id":i + 1,"text":n.text,"tag":n.tag})
}
fn note_add(w: &mut World, a: &Value) -> Result<Value> {
    if w.notes.len() >= MAX_ITEMS {
        bail!("store limit reached");
    }
    w.notes.push(Note {
        text: text(a, "text")?.into(),
        tag: a["tag"].as_str().map(Into::into),
    });
    Ok(json!({"id":w.notes.len()}))
}
fn note_list(w: &mut World, a: &Value) -> Result<Value> {
    let tag = a["tag"].as_str();
    let notes = w
        .notes
        .iter()
        .enumerate()
        .filter(|(_, n)| tag.is_none_or(|t| n.tag.as_deref() == Some(t)))
        .map(|(i, n)| note_json(i, n))
        .collect::<Vec<_>>();
    Ok(json!({"notes":notes}))
}
fn note_search(w: &mut World, a: &Value) -> Result<Value> {
    let q = words(string(a, "query")?);
    let notes = w
        .notes
        .iter()
        .enumerate()
        .filter(|(_, n)| !q.is_disjoint(&words(&n.text)))
        .map(|(i, n)| note_json(i, n))
        .collect::<Vec<_>>();
    Ok(json!({"notes":notes}))
}
fn todo_json(i: usize, t: &Todo) -> Value {
    json!({"id":i + 1,"title":t.title,"done":t.done})
}
fn todo_add(w: &mut World, a: &Value) -> Result<Value> {
    if w.todos.len() >= MAX_ITEMS {
        bail!("store limit reached");
    }
    w.todos.push(Todo {
        title: text(a, "title")?.into(),
        done: false,
    });
    Ok(json!({"id":w.todos.len()}))
}
fn todo_complete(w: &mut World, a: &Value) -> Result<Value> {
    let id = a["id"].as_u64().context("id must be a positive integer")? as usize;
    let todo = id
        .checked_sub(1)
        .and_then(|i| w.todos.get_mut(i))
        .with_context(|| format!("unknown todo {id}"))?;
    todo.done = true;
    Ok(json!({"completed":id}))
}
fn todo_list(w: &mut World, a: &Value) -> Result<Value> {
    let status = a["status"].as_str().unwrap_or("all");
    if !["open", "done", "all"].contains(&status) {
        bail!("status must be open, done, or all");
    }
    let todos = w
        .todos
        .iter()
        .enumerate()
        .filter(|(_, t)| match status {
            "open" => !t.done,
            "done" => t.done,
            _ => true,
        })
        .map(|(i, t)| todo_json(i, t))
        .collect::<Vec<_>>();
    Ok(json!({"todos":todos}))
}
pub fn initialize(w: &mut World, fixtures: &Fixtures) {
    w.kv = fixtures.kv.clone();
    w.notes = fixtures.notes.clone();
    w.todos = fixtures.todos.clone();
}
pub fn snapshot(w: &World, state: &mut Value) {
    state["kv"] = json!(w.kv);
    state["notes"] = json!(w.notes);
    state["todos"] = json!(w.todos);
}
pub fn register(r: &mut Registry) {
    r.insert("kv_set", kv_set);
    r.insert("kv_get", kv_get);
    r.insert("kv_delete", kv_delete);
    r.insert("kv_list", kv_list);
    r.insert("note_add", note_add);
    r.insert("note_list", note_list);
    r.insert("note_search", note_search);
    r.insert("todo_add", todo_add);
    r.insert("todo_complete", todo_complete);
    r.insert("todo_list", todo_list);
}

#[cfg(test)]
mod tests {
    use crate::tools::test_support::host;
    use serde_json::json;

    #[test]
    fn kv_store_round_trips_and_reports_unknown_keys() {
        let (mut h, _dir) = host(json!({"kv":{"mode":"eco"}}));
        assert_eq!(
            h.call("kv_get", json!({"key":"mode"})).unwrap()["value"],
            "eco"
        );
        h.call("kv_set", json!({"key":"level","value":"3"}))
            .unwrap();
        assert_eq!(
            h.call("kv_list", json!({})).unwrap()["keys"],
            json!(["level", "mode"])
        );
        h.call("kv_delete", json!({"key":"mode"})).unwrap();
        assert!(h.call("kv_get", json!({"key":"mode"})).is_err());
        assert_eq!(h.snapshot().unwrap()["kv"], json!({"level":"3"}));
    }
    #[test]
    fn notes_filter_by_tag_and_search_by_words() {
        let (mut h, _dir) = host(json!({"notes":[{"text":"Order bolts on Friday","tag":"buy"}]}));
        h.call("note_add", json!({"text":"Call the supplier","tag":"call"}))
            .unwrap();
        assert_eq!(
            h.call("note_list", json!({"tag":"call"})).unwrap()["notes"][0]["id"],
            2
        );
        assert_eq!(
            h.call("note_list", json!({})).unwrap()["notes"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let found = h
            .call("note_search", json!({"query":"bolt order"}))
            .unwrap();
        assert_eq!(found["notes"][0]["id"], 1);
    }
    #[test]
    fn todos_complete_by_id_and_filter_by_status() {
        let (mut h, _dir) = host(json!({"todos":[{"title":"a"},{"title":"b","done":true}]}));
        let id = h.call("todo_add", json!({"title":"c"})).unwrap()["id"].clone();
        assert_eq!(id, 3);
        h.call("todo_complete", json!({"id":1})).unwrap();
        let open = h.call("todo_list", json!({"status":"open"})).unwrap();
        assert_eq!(open["todos"].as_array().unwrap().len(), 1);
        assert!(h.call("todo_complete", json!({"id":9})).is_err());
        assert!(h.call("todo_complete", json!({"id":0})).is_err());
        assert!(h.call("todo_list", json!({"status":"later"})).is_err());
        assert_eq!(h.snapshot().unwrap()["todos"][0]["done"], true);
    }
}
