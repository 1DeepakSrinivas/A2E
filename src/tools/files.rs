//! Task-scoped file tools: search, read, write, list, append, move, delete, grep.
//!
//! Relative paths only; symlinks denied; bounded reads and writes.

use super::{string, words, Registry, World};
use crate::{read_bounded, text_limit};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};

const READ_LIMIT: usize = 65536;
const WRITE_LIMIT: usize = 4096;
const QUOTA: usize = 262_144;

pub fn safe_path(root: &Path, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        bail!("path must stay within the task directory");
    }
    let mut p = root.to_owned();
    for c in path.components() {
        p.push(c);
        if p.symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink())
        {
            bail!("symlinks are not allowed");
        }
    }
    Ok(p)
}
pub fn read_text(w: &World, name: &str) -> Result<String> {
    if !w.files.contains(name) {
        bail!("unknown task file");
    }
    let bytes = read_bounded(std::fs::File::open(safe_path(&w.root, name)?)?, READ_LIMIT)?;
    Ok(String::from_utf8(bytes)?)
}
fn store(w: &mut World, name: &str, content: &str) -> Result<()> {
    if content.len() > WRITE_LIMIT {
        bail!("file content limit");
    }
    charge(w, content.len())?;
    let p = safe_path(&w.root, name)?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(p, content).map_err(|e| anyhow::anyhow!("cannot write {name}: {e}"))?;
    super::fs::track(w, name);
    Ok(())
}
/// Counts written bytes against the per-attempt quota.
pub fn charge(w: &mut World, bytes: usize) -> Result<()> {
    if w.written + bytes > QUOTA {
        bail!("write quota for this task is used up");
    }
    w.written += bytes;
    Ok(())
}
fn search_files(w: &mut World, a: &Value) -> Result<Value> {
    let q = words(string(a, "query")?);
    let mut out = Vec::new();
    for name in &w.files {
        let body = read_text(w, name)?;
        if !q.is_disjoint(&words(&format!("{name} {body}"))) {
            out.push(name.clone());
        }
        if out.len() == 8 {
            break;
        }
    }
    Ok(json!({"paths":out}))
}
fn read_file(w: &mut World, a: &Value) -> Result<Value> {
    let path = string(a, "path")?;
    let body = read_text(w, path)?;
    Ok(json!({"path":path,"content":text_limit(&body,3000)}))
}
fn write_file(w: &mut World, a: &Value) -> Result<Value> {
    let path = string(a, "path")?;
    store(w, path, string(a, "content")?)?;
    Ok(json!({"written":path}))
}
fn append_file(w: &mut World, a: &Value) -> Result<Value> {
    let path = string(a, "path")?;
    let mut body = if w.files.contains(path) {
        read_text(w, path)?
    } else {
        String::new()
    };
    body.push_str(string(a, "content")?);
    store(w, path, &body)?;
    Ok(json!({"appended":path,"bytes":body.len()}))
}
fn list_dir(w: &mut World, a: &Value) -> Result<Value> {
    let dir = a["path"].as_str().unwrap_or("").trim_matches('/');
    let prefix = if dir.is_empty() {
        String::new()
    } else {
        safe_path(&w.root, dir)?;
        format!("{dir}/")
    };
    let mut entries = std::collections::BTreeSet::new();
    for name in &w.files {
        if let Some(rest) = name.strip_prefix(&prefix) {
            entries.insert(match rest.split_once('/') {
                Some((first, _)) => format!("{first}/"),
                None => rest.to_owned(),
            });
        }
    }
    if entries.is_empty() && !prefix.is_empty() {
        bail!("unknown directory");
    }
    Ok(json!({"entries":entries}))
}
fn move_file(w: &mut World, a: &Value) -> Result<Value> {
    let (from, to) = (string(a, "from")?, string(a, "to")?);
    if !w.files.contains(from) {
        bail!("unknown task file");
    }
    if w.files.contains(to) {
        bail!("destination already exists");
    }
    let (src, dst) = (safe_path(&w.root, from)?, safe_path(&w.root, to)?);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(src, dst)?;
    w.files.remove(from);
    super::fs::track(w, to);
    Ok(json!({"moved":to}))
}
fn delete_file(w: &mut World, a: &Value) -> Result<Value> {
    let path = string(a, "path")?;
    if !w.files.contains(path) {
        bail!("unknown task file");
    }
    let p = safe_path(&w.root, path)?;
    if std::fs::metadata(&p)?.permissions().readonly() {
        bail!("permission denied: {path} is read-only");
    }
    std::fs::remove_file(p)?;
    w.files.remove(path);
    Ok(json!({"deleted":path}))
}
/// Numbered lines that contain the text, ignoring case.
fn grep_file(w: &mut World, a: &Value) -> Result<Value> {
    let path = string(a, "path")?;
    let needle = string(a, "text")?.to_lowercase();
    if needle.is_empty() {
        bail!("text must not be empty");
    }
    let body = read_text(w, path)?;
    let lines = body
        .lines()
        .enumerate()
        .filter(|(_, l)| l.to_lowercase().contains(&needle))
        .take(20)
        .map(|(i, l)| json!({"line":i + 1,"text":text_limit(l,200)}))
        .collect::<Vec<_>>();
    Ok(json!({"path":path,"matches":lines}))
}
pub fn register(r: &mut Registry) {
    r.insert("search_files", search_files);
    r.insert("read_file", read_file);
    r.insert("write_file", write_file);
    r.insert("append_file", append_file);
    r.insert("list_dir", list_dir);
    r.insert("move_file", move_file);
    r.insert("delete_file", delete_file);
    r.insert("grep_file", grep_file);
}

#[cfg(test)]
mod tests {
    use crate::tools::test_support::host;
    use serde_json::json;

    #[test]
    fn list_move_delete_and_append_track_task_files() {
        let (mut h, _dir) =
            host(json!({"files":{"a.txt":"one","docs/b.txt":"two","docs/c.txt":"3"}}));
        let top = h.call("list_dir", json!({})).unwrap();
        assert_eq!(top["entries"], json!(["a.txt", "docs/"]));
        let docs = h.call("list_dir", json!({"path":"docs"})).unwrap();
        assert_eq!(docs["entries"], json!(["b.txt", "c.txt"]));
        assert!(h.call("list_dir", json!({"path":"nope"})).is_err());
        h.call("append_file", json!({"path":"a.txt","content":"+x"}))
            .unwrap();
        assert_eq!(
            h.call("read_file", json!({"path":"a.txt"})).unwrap()["content"],
            "one+x"
        );
        h.call("move_file", json!({"from":"a.txt","to":"z/a.txt"}))
            .unwrap();
        assert!(h.call("read_file", json!({"path":"a.txt"})).is_err());
        assert!(h
            .call("move_file", json!({"from":"docs/b.txt","to":"docs/c.txt"}))
            .is_err());
        h.call("delete_file", json!({"path":"docs/b.txt"})).unwrap();
        let state = h.snapshot().unwrap();
        assert_eq!(
            state["files"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["docs/c.txt", "z/a.txt"]
        );
    }
    #[test]
    fn grep_is_case_insensitive_and_numbers_lines() {
        let (mut h, _dir) = host(json!({"files":{"log.txt":"ok\nERROR disk\nfine\nerror net"}}));
        let r = h
            .call("grep_file", json!({"path":"log.txt","text":"error"}))
            .unwrap();
        assert_eq!(r["matches"][0]["line"], 2);
        assert_eq!(r["matches"][1]["line"], 4);
        assert!(h
            .call("grep_file", json!({"path":"log.txt","text":""}))
            .is_err());
    }
    #[test]
    fn writes_are_bounded_and_paths_stay_inside_the_task() {
        let (mut h, _dir) = host(json!({}));
        let big = "x".repeat(4097);
        assert!(h
            .call("write_file", json!({"path":"a.txt","content":big}))
            .is_err());
        assert!(h
            .call("write_file", json!({"path":"../a.txt","content":"x"}))
            .is_err());
        assert!(h
            .call(
                "append_file",
                json!({"path":"a.txt","content":"x".repeat(3000)})
            )
            .is_ok());
        assert!(h
            .call(
                "append_file",
                json!({"path":"a.txt","content":"x".repeat(3000)})
            )
            .is_err());
    }
}
