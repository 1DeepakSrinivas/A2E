// sandbox/mod.rs
// Deterministic workspace tree for file tasks: generate, hash, verify, and copy per attempt
// The base image is read-only input; each attempt works on its own copy

mod facts;
mod filler;

use crate::digest;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

pub const VERSION: u32 = 1;
pub const DEFAULT_SEED: u64 = 1;
/// 2026-01-01T00:00:00Z in seconds since the epoch; file ages count from here.
const BASE_EPOCH: u64 = 1_767_225_600;

pub enum Kind {
    Dir,
    File {
        data: Vec<u8>,
        age_secs: u64,
        readonly: bool,
    },
}
pub struct Entry {
    pub path: String,
    pub kind: Kind,
}
impl Entry {
    pub fn dir(path: &str) -> Self {
        Self {
            path: path.into(),
            kind: Kind::Dir,
        }
    }
    pub fn file(path: &str, data: impl Into<Vec<u8>>, age_days: u64) -> Self {
        Self {
            path: path.into(),
            kind: Kind::File {
                data: data.into(),
                age_secs: age_days * 86_400,
                readonly: false,
            },
        }
    }
    pub fn readonly(mut self) -> Self {
        if let Kind::File { readonly, .. } = &mut self.kind {
            *readonly = true;
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub seed: u64,
    pub files: usize,
    pub dirs: usize,
    pub bytes: u64,
    pub sha256: String,
}

/// Every entry of the workspace for a seed; planted facts are fixed, filler follows the seed.
pub fn build(seed: u64) -> Vec<Entry> {
    let mut entries = facts::planted();
    entries.extend(filler::generate(seed, &facts::taken(&entries)));
    entries
}

/// Writes `<out>/base` and `<out>/manifest.json`; an existing base needs `force`.
pub fn init(out: &Path, seed: u64, force: bool) -> Result<Manifest> {
    let base = out.join("base");
    if base.exists() {
        ensure!(force, "{} exists; pass --force to rebuild", base.display());
        fs::remove_dir_all(&base)?;
    }
    let staging = out.join("base.tmp");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    for e in build(seed) {
        write_entry(&staging, &e)?;
    }
    fs::rename(&staging, &base)?;
    let manifest = measure(&base, seed)?;
    fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(manifest)
}
fn write_entry(root: &Path, e: &Entry) -> Result<()> {
    let p = root.join(&e.path);
    match &e.kind {
        Kind::Dir => fs::create_dir_all(&p)?,
        Kind::File {
            data,
            age_secs,
            readonly,
        } => {
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&p, data)?;
            fs::File::options().write(true).open(&p)?.set_modified(
                SystemTime::UNIX_EPOCH + Duration::from_secs(BASE_EPOCH + age_secs),
            )?;
            if *readonly {
                let mut perm = fs::metadata(&p)?.permissions();
                perm.set_readonly(true);
                fs::set_permissions(&p, perm)?;
            }
        }
    }
    Ok(())
}

/// Counts and content hash of a tree; file times and modes are not hashed.
pub fn measure(base: &Path, seed: u64) -> Result<Manifest> {
    let mut lines = BTreeMap::new();
    let (mut files, mut dirs, mut bytes) = (0, 0, 0);
    walk(base, base, &mut |rel, path, is_dir| {
        if is_dir {
            dirs += 1;
            lines.insert(rel, "d".to_string());
        } else {
            files += 1;
            let data = fs::read(path)?;
            bytes += data.len() as u64;
            lines.insert(rel, format!("f {} {}", data.len(), digest(&data)));
        }
        Ok(())
    })?;
    let listing: String = lines.iter().map(|(k, v)| format!("{k}\t{v}\n")).collect();
    Ok(Manifest {
        version: VERSION,
        seed,
        files,
        dirs,
        bytes,
        sha256: digest(listing.as_bytes()),
    })
}
fn walk(
    root: &Path,
    dir: &Path,
    visit: &mut dyn FnMut(String, &Path, bool) -> Result<()>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        ensure!(!kind.is_symlink(), "symlink in sandbox: {}", path.display());
        let rel = path.strip_prefix(root)?.to_string_lossy().into_owned();
        visit(rel, &path, kind.is_dir())?;
        if kind.is_dir() {
            walk(root, &path, visit)?;
        }
    }
    Ok(())
}

pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let p = dir.join("manifest.json");
    let text = fs::read_to_string(&p)
        .with_context(|| format!("{} is missing; run scripts/init_sandbox.sh", p.display()))?;
    Ok(serde_json::from_str(&text)?)
}
/// Recomputes the base hash and compares it with the manifest.
pub fn verify(dir: &Path) -> Result<Manifest> {
    let want = read_manifest(dir)?;
    ensure!(
        want.version == VERSION,
        "sandbox version {} differs from generator version {VERSION}; rebuild with --force",
        want.version
    );
    let got = measure(&dir.join("base"), want.seed)?;
    if got != want {
        bail!(
            "sandbox base does not match its manifest (hash {} vs {}); rebuild with --force",
            got.sha256,
            want.sha256
        );
    }
    Ok(got)
}

/// Copies the base tree into `dst`, keeping file times and read-only flags.
pub fn copy_into(base: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(base)? {
        let entry = entry?;
        let (from, to): (PathBuf, PathBuf) = (entry.path(), dst.join(entry.file_name()));
        let meta = entry.metadata()?;
        if meta.is_dir() {
            copy_into(&from, &to)?;
        } else {
            fs::write(&to, fs::read(&from)?)?;
            fs::File::options()
                .write(true)
                .open(&to)?
                .set_modified(meta.modified()?)?;
            fs::set_permissions(&to, meta.permissions())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_tree_and_a_new_seed_changes_filler() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let m1 = init(a.path(), 7, false).unwrap();
        let m2 = init(b.path(), 7, false).unwrap();
        assert_eq!(m1, m2);
        let c = tempfile::tempdir().unwrap();
        assert_ne!(init(c.path(), 8, false).unwrap().sha256, m1.sha256);
    }
    #[test]
    fn tree_is_about_three_hundred_files_and_two_megabytes() {
        let dir = tempfile::tempdir().unwrap();
        let m = init(dir.path(), DEFAULT_SEED, false).unwrap();
        assert!((250..=400).contains(&m.files), "{} files", m.files);
        assert!(
            (1_000_000..=3_000_000).contains(&m.bytes),
            "{} bytes",
            m.bytes
        );
        assert!(m.dirs > 40);
    }
    #[test]
    fn verify_detects_edits_and_init_refuses_to_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        init(dir.path(), DEFAULT_SEED, false).unwrap();
        verify(dir.path()).unwrap();
        assert!(init(dir.path(), DEFAULT_SEED, false).is_err());
        fs::write(dir.path().join("base/scratch/new.txt"), "x").unwrap();
        assert!(verify(dir.path()).is_err());
        init(dir.path(), DEFAULT_SEED, true).unwrap();
        verify(dir.path()).unwrap();
    }
    #[test]
    fn planted_needles_appear_in_exactly_one_file() {
        for (needle, path) in facts::needles() {
            let hits: Vec<_> = build(DEFAULT_SEED)
                .into_iter()
                .filter(|e| {
                    matches!(&e.kind, Kind::File { data, .. }
                    if String::from_utf8_lossy(data).contains(needle))
                })
                .map(|e| e.path)
                .collect();
            assert_eq!(hits, [path], "{needle}");
        }
    }
    #[test]
    fn copy_keeps_content_times_and_read_only_flags() {
        let dir = tempfile::tempdir().unwrap();
        init(dir.path(), DEFAULT_SEED, false).unwrap();
        let copy = tempfile::tempdir().unwrap();
        copy_into(&dir.path().join("base"), copy.path()).unwrap();
        let want = measure(&dir.path().join("base"), DEFAULT_SEED).unwrap();
        assert_eq!(measure(copy.path(), DEFAULT_SEED).unwrap(), want);
        let p = "home/.config/app/locked.conf";
        let (a, b) = (
            fs::metadata(dir.path().join("base").join(p)).unwrap(),
            fs::metadata(copy.path().join(p)).unwrap(),
        );
        assert!(b.permissions().readonly());
        assert_eq!(a.modified().unwrap(), b.modified().unwrap());
    }
    #[test]
    fn tree_has_no_symlinks_and_includes_hidden_empty_and_binary_entries() {
        let entries = build(DEFAULT_SEED);
        let has = |p: &str| entries.iter().any(|e| e.path == p);
        assert!(has("home/.hidden_notes") && has("scratch") && has("data/blob.bin"));
        assert!(entries.iter().any(|e| e.path.contains(' ')));
    }
}
