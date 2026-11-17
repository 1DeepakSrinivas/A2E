//! Workspace tools: ls, tree, find, grep, cat, head, tail, wc, stat, mkdir, cp, touch, rm, diff.
//!
//! Paths are relative to the task root; results are capped with a note to narrow the query.

use super::{
    files::{charge, safe_path},
    string,
    timeunits::civil_from_days,
    Registry, World,
};
use crate::{read_bounded, text_limit};
use anyhow::{bail, Result};
use regex::RegexBuilder;
use serde_json::{json, Value};
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::BTreeMap,
    path::{Component, Path},
    time::{SystemTime, UNIX_EPOCH},
};

const TEXT_LIMIT: usize = 1 << 20;
const FIND_CAP: usize = 25;
const GREP_CAP: usize = 20;
const TREE_CAP: usize = 60;
const DIFF_CAP: usize = 40;
const DIFF_LINES: usize = 500;
const LINE_CHARS: usize = 200;

/// Registers a file and its parent directories as present and changed.
pub fn track(w: &mut World, name: &str) {
    add_parents(w, name);
    w.files.insert(name.into());
    w.touched.insert(name.into());
}
fn add_parents(w: &mut World, name: &str) {
    let mut end = 0;
    while let Some(i) = name[end..].find('/') {
        end += i;
        w.dirs.insert(name[..end].into());
        end += 1;
    }
}
/// Builds the file and directory index by walking the copied tree.
pub fn index(w: &mut World) -> Result<()> {
    fn go(w: &mut World, dir: &Path, prefix: &str) -> Result<()> {
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let name = format!("{prefix}{}", e.file_name().to_string_lossy());
            if e.file_type()?.is_dir() {
                w.dirs.insert(name.clone());
                go(w, &e.path(), &format!("{name}/"))?;
            } else {
                w.files.insert(name);
            }
        }
        Ok(())
    }
    let root = w.root.clone();
    go(w, &root, "")
}

/// Path with leading "./" and surrounding slashes removed; "" is the root.
fn norm(a: &Value, key: &str) -> Result<String> {
    let p = a[key].as_str().unwrap_or("").trim();
    let p = p.strip_prefix("./").unwrap_or(p).trim_matches('/');
    let p = if p == "." { "" } else { p };
    if Path::new(p)
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        bail!("path must stay within the task directory");
    }
    Ok(p.into())
}
fn flag(a: &Value, key: &str) -> bool {
    a[key].as_bool().unwrap_or(false)
}
fn count(a: &Value, key: &str, default: usize, max: usize) -> usize {
    a[key]
        .as_u64()
        .map_or(default, |n| n as usize)
        .clamp(1, max)
}
fn is_dir(w: &World, p: &str) -> bool {
    p.is_empty() || w.dirs.contains(p)
}
fn exists(w: &World, p: &str) -> bool {
    is_dir(w, p) || w.files.contains(p)
}
fn require_file(w: &World, p: &str) -> Result<()> {
    if w.files.contains(p) {
        Ok(())
    } else if w.dirs.contains(p) {
        bail!("{p} is a directory")
    } else {
        bail!("no such file: {p}")
    }
}
fn require_dir(w: &World, p: &str) -> Result<()> {
    if is_dir(w, p) {
        Ok(())
    } else if w.files.contains(p) {
        bail!("{p} is a file, not a directory")
    } else {
        bail!("no such directory: {p}")
    }
}
/// File text up to 1 MiB; binary files are rejected.
fn text(w: &World, p: &str) -> Result<String> {
    require_file(w, p)?;
    let bytes = read_bounded(std::fs::File::open(safe_path(&w.root, p)?)?, TEXT_LIMIT)?;
    String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("{p} is not a text file"))
}
fn base_name(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}
fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.into()
    } else {
        format!("{dir}/{name}")
    }
}
/// Direct children of a directory, directories flagged.
fn children(w: &World, dir: &str) -> BTreeMap<String, bool> {
    let prefix = if dir.is_empty() {
        String::new()
    } else {
        format!("{dir}/")
    };
    let mut out = BTreeMap::new();
    for (set, is_d) in [(&w.files, false), (&w.dirs, true)] {
        for n in set {
            if let Some(rest) = n.strip_prefix(&prefix) {
                if !rest.is_empty() && !rest.contains('/') {
                    out.insert(rest.to_owned(), is_d);
                }
            }
        }
    }
    out
}
fn mtime(w: &World, p: &str) -> Result<u64> {
    let t = std::fs::metadata(safe_path(&w.root, p)?)?.modified()?;
    Ok(t.duration_since(UNIX_EPOCH)?.as_secs())
}
fn date(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}
fn cap_note(total: usize, cap: usize) -> Option<String> {
    (total > cap).then(|| format!("{} more not shown; narrow your query", total - cap))
}
fn clip(l: &str) -> String {
    text_limit(l, LINE_CHARS)
}

fn ls(w: &mut World, a: &Value) -> Result<Value> {
    let dir = norm(a, "path")?;
    require_dir(w, &dir)?;
    let (all, long) = (flag(a, "all"), flag(a, "long"));
    let mut entries = Vec::new();
    for (name, d) in children(w, &dir) {
        if !all && name.starts_with('.') {
            continue;
        }
        let label = if d { format!("{name}/") } else { name.clone() };
        entries.push(if long && !d {
            let p = join(&dir, &name);
            let size = std::fs::metadata(safe_path(&w.root, &p)?)?.len();
            json!(format!("{label}  {size} bytes  {}", date(mtime(w, &p)?)))
        } else {
            json!(label)
        });
    }
    Ok(json!({"path":dir,"entries":entries}))
}
fn tree(w: &mut World, a: &Value) -> Result<Value> {
    let dir = norm(a, "path")?;
    require_dir(w, &dir)?;
    let depth = count(a, "depth", 2, 3);
    fn go(w: &World, dir: &str, level: usize, depth: usize, out: &mut Vec<String>) {
        for (name, d) in children(w, dir) {
            if name.starts_with('.') {
                continue;
            }
            out.push(format!(
                "{}{name}{}",
                "  ".repeat(level),
                if d { "/" } else { "" }
            ));
            if d && level + 1 < depth {
                go(w, &join(dir, &name), level + 1, depth, out);
            }
        }
    }
    let mut lines = Vec::new();
    go(w, &dir, 0, depth, &mut lines);
    let note = cap_note(lines.len(), TREE_CAP);
    lines.truncate(TREE_CAP);
    Ok(json!({"tree":lines.join("\n"),"note":note}))
}

/// Case-insensitive match of `*` and `?` wildcards.
fn glob(pat: &[char], s: &[char]) -> bool {
    match (pat.first(), s.first()) {
        (None, None) => true,
        (Some('*'), _) => glob(&pat[1..], s) || (!s.is_empty() && glob(pat, &s[1..])),
        (Some('?'), Some(_)) => glob(&pat[1..], &s[1..]),
        (Some(p), Some(c)) => p.eq_ignore_ascii_case(c) && glob(&pat[1..], &s[1..]),
        _ => false,
    }
}
fn find(w: &mut World, a: &Value) -> Result<Value> {
    let dir = norm(a, "path")?;
    require_dir(w, &dir)?;
    let pattern: Vec<char> = a["name"].as_str().unwrap_or("*").chars().collect();
    let kind = a["type"].as_str().unwrap_or("any");
    let min_size = a["min_size"].as_u64().unwrap_or(0);
    let newer = match a["newer_than"].as_str() {
        Some(p) => {
            let p = p.trim_matches('/');
            require_file(w, p)?;
            Some(mtime(w, p)?)
        }
        None => None,
    };
    if !["any", "f", "d"].contains(&kind) {
        bail!("type must be f or d");
    }
    let prefix = if dir.is_empty() {
        String::new()
    } else {
        format!("{dir}/")
    };
    let mut hits = Vec::new();
    for (set, is_d) in [(&w.files, false), (&w.dirs, true)] {
        for p in set.iter().filter(|p| p.starts_with(&prefix)) {
            let name: Vec<char> = base_name(p).chars().collect();
            if (kind == "f" && is_d) || (kind == "d" && !is_d) || !glob(&pattern, &name) {
                continue;
            }
            if !is_d && (min_size > 0 || newer.is_some()) {
                let meta = std::fs::metadata(safe_path(&w.root, p)?)?;
                if meta.len() < min_size || newer.is_some_and(|t| mtime(w, p).unwrap_or(0) <= t) {
                    continue;
                }
            } else if is_d && (min_size > 0 || newer.is_some()) {
                continue;
            }
            hits.push(if is_d { format!("{p}/") } else { p.clone() });
        }
    }
    hits.sort();
    let note = cap_note(hits.len(), FIND_CAP);
    hits.truncate(FIND_CAP);
    Ok(json!({"paths":hits,"note":note}))
}

fn grep(w: &mut World, a: &Value) -> Result<Value> {
    let pattern = string(a, "pattern")?;
    if pattern.is_empty() {
        bail!("pattern must not be empty");
    }
    let literal = !flag(a, "regex");
    let re = RegexBuilder::new(&if literal {
        regex::escape(pattern)
    } else {
        pattern.into()
    })
    .case_insensitive(flag(a, "ignore_case"))
    .size_limit(1 << 16)
    .build()
    .map_err(|e| anyhow::anyhow!("invalid pattern: {e}"))?;
    let path = norm(a, "path")?;
    let targets: Vec<String> = if w.files.contains(&path) {
        vec![path.clone()]
    } else if is_dir(w, &path) {
        if !flag(a, "recursive") {
            bail!(
                "{} is a directory; set recursive to true",
                if path.is_empty() { "." } else { &path }
            );
        }
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };
        w.files
            .iter()
            .filter(|p| p.starts_with(&prefix))
            .cloned()
            .collect()
    } else {
        bail!("no such file or directory: {path}")
    };
    let single = targets.len() == 1 && w.files.contains(&path);
    let files_only = flag(a, "files_only");
    let (mut matches, mut total) = (Vec::new(), 0);
    for t in &targets {
        let body = match text(w, t) {
            Ok(b) => b,
            Err(e) if single => return Err(e),
            Err(_) => continue,
        };
        for (i, l) in body.lines().enumerate() {
            if re.is_match(l) {
                total += 1;
                if files_only {
                    matches.push(t.clone());
                    break;
                }
                if matches.len() < GREP_CAP {
                    matches.push(format!("{t}:{}:{}", i + 1, clip(l)));
                }
            }
        }
    }
    let note = cap_note(total, GREP_CAP);
    matches.truncate(GREP_CAP);
    Ok(json!({"matches":matches,"note":note}))
}

fn cat(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    let body = text(w, &path)?;
    let lines: Vec<&str> = body.lines().collect();
    let start = count(a, "offset", 1, usize::MAX);
    let limit = count(a, "limit", 50, 100);
    let shown: Vec<String> = lines
        .iter()
        .skip(start - 1)
        .take(limit)
        .map(|l| clip(l))
        .collect();
    let last = start - 1 + shown.len();
    Ok(json!({
        "path":path,
        "content":shown.join("\n"),
        "from":start,
        "to":last,
        "total_lines":lines.len(),
        "more":last < lines.len(),
    }))
}
fn head(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    let body = text(w, &path)?;
    let n = count(a, "lines", 10, 50);
    let lines: Vec<String> = body.lines().take(n).map(clip).collect();
    Ok(json!({"path":path,"content":lines.join("\n")}))
}
fn tail(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    let body = text(w, &path)?;
    let n = count(a, "lines", 10, 50);
    let all: Vec<&str> = body.lines().collect();
    let lines: Vec<String> = all[all.len().saturating_sub(n)..]
        .iter()
        .map(|l| clip(l))
        .collect();
    Ok(json!({"path":path,"content":lines.join("\n")}))
}
fn wc(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    let body = text(w, &path)?;
    Ok(json!({
        "path":path,
        "lines":body.lines().count(),
        "words":body.split_whitespace().count(),
        "bytes":body.len(),
    }))
}
fn stat(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    if !exists(w, &path) {
        bail!("no such file or directory: {path}");
    }
    let meta = std::fs::metadata(safe_path(&w.root, &path)?)?;
    Ok(json!({
        "path":path,
        "kind":if is_dir(w, &path) { "directory" } else { "file" },
        "size":if meta.is_dir() { Value::Null } else { json!(meta.len()) },
        "modified":date(meta.modified()?.duration_since(UNIX_EPOCH)?.as_secs()),
        "read_only":meta.permissions().readonly(),
    }))
}

fn mkdir(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    if path.is_empty() || exists(w, &path) {
        bail!(
            "{} already exists",
            if path.is_empty() { "." } else { &path }
        );
    }
    std::fs::create_dir_all(safe_path(&w.root, &path)?)?;
    add_parents(w, &format!("{path}/x"));
    Ok(json!({"created":path}))
}
fn cp(w: &mut World, a: &Value) -> Result<Value> {
    let (from, to) = (norm(a, "from")?, norm(a, "to")?);
    require_file(w, &from)?;
    if to.is_empty() || exists(w, &to) {
        bail!("destination already exists");
    }
    let size = std::fs::metadata(safe_path(&w.root, &from)?)?.len() as usize;
    charge(w, size)?;
    let dst = safe_path(&w.root, &to)?;
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(safe_path(&w.root, &from)?, &dst)?;
    let mut perm = std::fs::metadata(&dst)?.permissions();
    perm.set_mode(perm.mode() | 0o200);
    std::fs::set_permissions(&dst, perm)?;
    track(w, &to);
    Ok(json!({"copied":to}))
}
fn touch(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    if is_dir(w, &path) {
        bail!(
            "{} is a directory",
            if path.is_empty() { "." } else { &path }
        );
    }
    let p = safe_path(&w.root, &path)?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = std::fs::File::options()
        .create(true)
        .append(true)
        .open(&p)
        .map_err(|e| anyhow::anyhow!("cannot touch {path}: {e}"))?;
    f.set_modified(SystemTime::now())?;
    track(w, &path);
    Ok(json!({"touched":path}))
}
fn rm(w: &mut World, a: &Value) -> Result<Value> {
    let path = norm(a, "path")?;
    if path.is_empty() {
        bail!("refusing to remove the root");
    }
    let p = safe_path(&w.root, &path)?;
    if w.files.contains(&path) {
        if std::fs::metadata(&p)?.permissions().readonly() {
            bail!("permission denied: {path} is read-only");
        }
        std::fs::remove_file(p)?;
        w.files.remove(&path);
    } else if w.dirs.contains(&path) {
        if !children(w, &path).is_empty() {
            bail!("{path} is not empty; remove its files first");
        }
        std::fs::remove_dir(p)?;
        w.dirs.remove(&path);
    } else {
        bail!("no such file or directory: {path}");
    }
    Ok(json!({"removed":path}))
}

/// Line diff by longest common subsequence: `-` only in a, `+` only in b.
fn diff(w: &mut World, a: &Value) -> Result<Value> {
    let (pa, pb) = (norm(a, "a")?, norm(a, "b")?);
    let (ta, tb) = (text(w, &pa)?, text(w, &pb)?);
    let (x, y): (Vec<&str>, Vec<&str>) = (ta.lines().collect(), tb.lines().collect());
    if x.len() > DIFF_LINES || y.len() > DIFF_LINES {
        bail!("files are too long to diff; use grep or cat on each");
    }
    let mut t = vec![vec![0usize; y.len() + 1]; x.len() + 1];
    for i in (0..x.len()).rev() {
        for j in (0..y.len()).rev() {
            t[i][j] = if x[i] == y[j] {
                t[i + 1][j + 1] + 1
            } else {
                t[i + 1][j].max(t[i][j + 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < x.len() || j < y.len() {
        if i < x.len() && j < y.len() && x[i] == y[j] {
            i += 1;
            j += 1;
        } else if j == y.len() || (i < x.len() && t[i + 1][j] >= t[i][j + 1]) {
            out.push(format!("-{}: {}", i + 1, clip(x[i])));
            i += 1;
        } else {
            out.push(format!("+{}: {}", j + 1, clip(y[j])));
            j += 1;
        }
    }
    let note = cap_note(out.len(), DIFF_CAP);
    out.truncate(DIFF_CAP);
    Ok(json!({"identical":out.is_empty() && note.is_none(),"changes":out,"note":note}))
}

pub fn register(r: &mut Registry) {
    r.insert("ls", ls);
    r.insert("tree", tree);
    r.insert("find", find);
    r.insert("grep", grep);
    r.insert("cat", cat);
    r.insert("head", head);
    r.insert("tail", tail);
    r.insert("wc", wc);
    r.insert("stat", stat);
    r.insert("mkdir", mkdir);
    r.insert("cp", cp);
    r.insert("touch", touch);
    r.insert("rm", rm);
    r.insert("diff", diff);
}

#[cfg(test)]
mod tests {
    use crate::tools::Builtins;
    use serde_json::{json, Value};

    fn workspace() -> (Builtins, tempfile::TempDir, tempfile::TempDir) {
        let sb = tempfile::tempdir().unwrap();
        crate::sandbox::init(sb.path(), crate::sandbox::DEFAULT_SEED, false).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut h = Builtins::new(dir.path().into(), false).unwrap();
        h.set_sandbox_base(Some(sb.path().join("base")));
        h.initialize(&serde_json::from_value(json!({"sandbox":"workspace"})).unwrap())
            .unwrap();
        (h, dir, sb)
    }
    fn strings(v: &Value) -> Vec<&str> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect()
    }

    #[test]
    fn ls_hides_dotfiles_until_all_is_set() {
        let (mut h, _d, _s) = workspace();
        let plain = h.call("ls", json!({"path":"home"})).unwrap();
        assert!(!strings(&plain["entries"]).contains(&".hidden_notes"));
        let all = h.call("ls", json!({"path":"home","all":true})).unwrap();
        assert!(strings(&all["entries"]).contains(&".hidden_notes"));
        let long = h
            .call("ls", json!({"path":"archive/old","long":true}))
            .unwrap();
        assert!(strings(&long["entries"])
            .iter()
            .any(|e| e.contains("bytes") && e.contains("2026-")));
        assert!(h.call("ls", json!({"path":"nope"})).is_err());
        assert!(h.call("ls", json!({"path":"../x"})).is_err());
    }
    #[test]
    fn find_matches_globs_and_caps_results() {
        let (mut h, _d, _s) = workspace();
        let r = h
            .call("find", json!({"name":"report*","type":"f"}))
            .unwrap();
        assert_eq!(r["paths"].as_array().unwrap().len(), 3);
        let many = h.call("find", json!({"name":"*.txt"})).unwrap();
        assert_eq!(many["paths"].as_array().unwrap().len(), 25);
        assert!(many["note"].as_str().unwrap().contains("narrow"));
        let big = h
            .call("find", json!({"min_size":100000,"type":"f"}))
            .unwrap();
        assert!(strings(&big["paths"]).contains(&"logs/big/service-2026-01.log"));
    }
    #[test]
    fn grep_searches_recursively_and_reports_path_line_text() {
        let (mut h, _d, _s) = workspace();
        assert!(h
            .call("grep", json!({"pattern":"passcode","path":"logs"}))
            .is_err());
        let r = h
            .call(
                "grep",
                json!({"pattern":"passcode","path":"logs","recursive":true}),
            )
            .unwrap();
        assert!(r["matches"][0]
            .as_str()
            .unwrap()
            .starts_with("logs/big/service-2026-01.log:"));
        let files = h
            .call(
                "grep",
                json!({"pattern":"REVENUE","recursive":true,"ignore_case":true,"files_only":true}),
            )
            .unwrap();
        assert_eq!(files["matches"].as_array().unwrap().len(), 4);
        let errs = h
            .call("grep", json!({"pattern":"ERROR","path":"logs/app.log"}))
            .unwrap();
        assert_eq!(errs["matches"].as_array().unwrap().len(), 3);
    }
    #[test]
    fn cat_windows_and_reads_past_the_read_file_cap() {
        let (mut h, _d, _s) = workspace();
        assert!(h
            .call("read_file", json!({"path":"data/export_full.csv"}))
            .is_err());
        let r = h
            .call(
                "cat",
                json!({"path":"data/export_full.csv","offset":5,"limit":3}),
            )
            .unwrap();
        assert_eq!(r["from"], 5);
        assert_eq!(r["to"], 7);
        assert_eq!(r["total_lines"], 9001);
        assert_eq!(r["more"], true);
        let t = h
            .call(
                "tail",
                json!({"path":"logs/big/service-2026-01.log","lines":1}),
            )
            .unwrap();
        assert!(t["content"].as_str().unwrap().contains("passcode 7731"));
        assert!(h.call("cat", json!({"path":"data/blob.bin"})).is_err());
        assert!(h.call("cat", json!({"path":"data"})).is_err());
    }
    #[test]
    fn wc_stat_and_diff_describe_files() {
        let (mut h, _d, _s) = workspace();
        let w = h.call("wc", json!({"path":"data/inventory.csv"})).unwrap();
        assert_eq!(w["lines"], 6);
        let s = h
            .call("stat", json!({"path":"home/.config/app/locked.conf"}))
            .unwrap();
        assert_eq!(s["read_only"], true);
        let d = h
            .call(
                "diff",
                json!({"a":"archive/old/report_final.txt","b":"archive/old/report_final_v2.txt"}),
            )
            .unwrap();
        assert_eq!(d["identical"], false);
        assert_eq!(d["changes"].as_array().unwrap().len(), 2);
    }
    #[test]
    fn write_tools_change_state_and_respect_read_only_files() {
        let (mut h, _d, _s) = workspace();
        h.call("mkdir", json!({"path":"scratch/out/deep"})).unwrap();
        assert!(h.call("mkdir", json!({"path":"scratch/out"})).is_err());
        h.call(
            "cp",
            json!({"from":"data/inventory.csv","to":"scratch/out/inv.csv"}),
        )
        .unwrap();
        assert!(h
            .call(
                "cp",
                json!({"from":"data/inventory.csv","to":"scratch/out/inv.csv"})
            )
            .is_err());
        h.call("touch", json!({"path":"scratch/new.txt"})).unwrap();
        assert!(h.call("rm", json!({"path":"scratch/out"})).is_err());
        h.call("rm", json!({"path":"scratch/out/deep"})).unwrap();
        // Root ignores file modes, so the read-only checks only hold for other users.
        if unsafe { libc::geteuid() } != 0 {
            assert!(h
                .call("rm", json!({"path":"home/.config/app/locked.conf"}))
                .is_err());
            let err = h
                .call(
                    "write_file",
                    json!({"path":"home/.config/app/locked.conf","content":"x"}),
                )
                .unwrap_err();
            assert!(err.to_string().contains("cannot write"));
        }
        let s = h.snapshot().unwrap();
        assert!(s["files"]["scratch/out/inv.csv"]
            .as_str()
            .unwrap()
            .contains("sprocket-77"));
        assert!(s["files"]["data/inventory.csv"].is_null());
        assert_eq!(s["files"]["scratch/new.txt"], "");
        h.call("rm", json!({"path":"scratch/new.txt"})).unwrap();
        assert!(h.snapshot().unwrap()["files"]
            .get("scratch/new.txt")
            .is_none());
    }
    #[test]
    fn tree_lists_visible_entries_to_the_requested_depth() {
        let (mut h, _d, _s) = workspace();
        let t = h
            .call("tree", json!({"path":"projects","depth":1}))
            .unwrap();
        assert_eq!(t["tree"], "notes-cli/\nsensor-fw/\nwebapp/");
        let deep = h.call("tree", json!({"depth":3})).unwrap();
        assert!(deep["note"].as_str().is_some());
    }
    #[test]
    fn sandbox_cases_need_a_base_and_a_known_profile() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Builtins::new(dir.path().into(), false).unwrap();
        let fx = |p: &str| serde_json::from_value(json!({"sandbox":p})).unwrap();
        assert!(h.initialize(&fx("workspace")).is_err());
        assert!(h.initialize(&fx("other")).is_err());
    }
}
