//! Shared modules for the evaluation CLI.
//!
//! Artifact hashing, bounded reads, and UTF-8-safe text limits.

pub mod adapter;
pub mod case;
pub mod config;
pub mod dotenv;
pub mod error;
pub mod eval;
mod fixtures;
pub mod hub;
pub mod inference;
pub mod measure;
mod process;
pub mod progress;
pub mod record;
pub mod runner;
pub mod sandbox;
pub mod settings;
pub mod store;
pub mod tools;
pub mod upload;

use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Read;

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn hash_file(path: &std::path::Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn canonical_hash(value: &impl Serialize) -> Result<String> {
    // serde_json's default map sorts object keys, including nested objects.
    Ok(digest(&serde_json::to_vec(&serde_json::to_value(value)?)?))
}
pub fn read_bounded(mut r: impl Read, max: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    r.by_ref().take(max as u64 + 1).read_to_end(&mut out)?;
    if out.len() > max {
        return Err(
            error::HarnessError::OutputLimit(format!("response exceeds {max} bytes")).into(),
        );
    }
    Ok(out)
}
pub fn text_limit(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.into();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[truncated]", &s[..end])
}
