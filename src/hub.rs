//! Hugging Face lookups and downloads, so any single-file GGUF can become a pinned model.

use crate::config::Model;
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::{
    io::{Read, Write},
    path::Path,
    time::Duration,
};

/// Quant labels in the order they are preferred when none is named.
const PREFERRED: [&str; 5] = ["Q4_K_M", "Q4_0", "Q5_K_M", "Q8_0", "Q4_K_S"];

#[derive(Clone, Debug, PartialEq)]
pub struct File {
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// One model file at one commit of a repository.
#[derive(Clone, Debug)]
pub struct Found {
    pub repo: String,
    pub revision: String,
    pub file: File,
    pub quant: String,
}

/// Splits `repo`, `repo:quant` or `hf:repo:quant`.
pub fn parse_ref(text: &str) -> (String, Option<String>) {
    let text = text.strip_prefix("hf:").unwrap_or(text);
    match text.split_once(':') {
        Some((repo, quant)) if !quant.is_empty() => (repo.to_string(), Some(quant.to_string())),
        _ => (text.trim_end_matches(':').to_string(), None),
    }
}

pub fn is_ref(text: &str) -> bool {
    text.starts_with("hf:")
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".into())
        .trim_end_matches('/')
        .to_string()
}

fn client(seconds: u64) -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(seconds))
        .build()?)
}

fn authorized(r: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
    match std::env::var("HF_TOKEN") {
        Ok(token) if !token.is_empty() => r.bearer_auth(token),
        _ => r,
    }
}

fn is_shard(name: &str) -> bool {
    regex::Regex::new(r"-\d{5}-of-\d{5}\.gguf$")
        .map(|r| r.is_match(name))
        .unwrap_or(false)
}

/// The quant label in a file name, such as `Q4_K_M`.
pub fn quant_of(name: &str) -> String {
    regex::Regex::new(r"(?i)(IQ\d_[A-Z0-9_]+|Q\d(?:_[A-Z0-9]+)+|BF16|F16|F32)")
        .ok()
        .and_then(|r| r.find(name).map(|m| m.as_str().to_uppercase()))
        .unwrap_or_else(|| "unknown".into())
}

/// Chooses one GGUF from a repository listing.
pub fn pick<'a>(files: &'a [File], quant: Option<&str>, file: Option<&str>) -> Result<&'a File> {
    let ggufs: Vec<&File> = files
        .iter()
        .filter(|f| f.name.ends_with(".gguf"))
        .filter(|f| !f.name.contains("mmproj") && !f.name.contains("imatrix"))
        .collect();
    if let Some(wanted) = file {
        let found = ggufs.iter().find(|f| f.name == wanted);
        if let Some(f) = found {
            if is_shard(&f.name) {
                bail!(
                    "{wanted} is one part of a sharded model; sharded models are not supported yet"
                );
            }
            return Ok(f);
        }
        bail!("{wanted} is not a GGUF file in this repository");
    }
    let single: Vec<&File> = ggufs
        .iter()
        .copied()
        .filter(|f| !is_shard(&f.name))
        .collect();
    if single.is_empty() {
        if ggufs.is_empty() {
            bail!("no GGUF files in this repository");
        }
        bail!("only sharded GGUF files found; sharded models are not supported yet");
    }
    let names = || {
        single
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(q) = quant {
        let q = q.to_lowercase();
        let hits: Vec<&&File> = single
            .iter()
            .filter(|f| f.name.to_lowercase().contains(&q))
            .collect();
        return match hits.as_slice() {
            [one] => Ok(**one),
            [] => bail!("no file matches quant {q}. Files: {}", names()),
            many => bail!(
                "quant {q} matches several files; name one with --file. Files: {}",
                many.iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
    }
    if let [one] = single.as_slice() {
        return Ok(one);
    }
    for label in PREFERRED {
        let hits: Vec<&&File> = single
            .iter()
            .filter(|f| f.name.to_uppercase().contains(label))
            .collect();
        if let [one] = hits.as_slice() {
            return Ok(**one);
        }
    }
    bail!(
        "several GGUF files and none is a clear default; add :<quant> or --file. Files: {}",
        names()
    )
}

/// Looks up the newest commit of `repo` and the file to use.
pub fn resolve(repo: &str, quant: Option<&str>, file: Option<&str>) -> Result<Found> {
    let url = format!("{}/api/models/{repo}?blobs=true", endpoint());
    let response = authorized(client(30)?.get(&url))
        .send()
        .with_context(|| format!("look up {repo} on Hugging Face"))?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        bail!("{repo} needs access. Accept its terms on huggingface.co and set HF_TOKEN");
    }
    let info: Value = response
        .error_for_status()
        .with_context(|| format!("{repo} was not found"))?
        .json()?;
    let revision = info["sha"]
        .as_str()
        .context("Hugging Face reply has no commit")?
        .to_string();
    let files: Vec<File> = info["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some(File {
                name: s["rfilename"].as_str()?.to_string(),
                size: s["size"]
                    .as_u64()
                    .or_else(|| s["lfs"]["size"].as_u64())
                    .unwrap_or(0),
                sha256: s["lfs"]["sha256"].as_str().map(str::to_string),
            })
        })
        .collect();
    let chosen = pick(&files, quant, file)?.clone();
    let quant = quant_of(&chosen.name);
    Ok(Found {
        repo: repo.to_string(),
        revision,
        file: chosen,
        quant,
    })
}

/// Downloads to `dest`, resuming a partial file, and checks the published hash when there is one.
/// Returns the file's SHA-256.
pub fn download(found: &Found, dest: &Path, seconds: u64, max_bytes: u64) -> Result<String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if dest.exists() {
        let have = crate::hash_file(dest)?;
        if found.file.sha256.as_ref().is_none_or(|s| *s == have) {
            return Ok(have);
        }
    }
    let part = dest.with_extension("gguf.part");
    let resume = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint(),
        found.repo,
        found.revision,
        found.file.name
    );
    let mut request = authorized(client(seconds)?.get(url));
    if resume > 0 {
        request = request.header("Range", format!("bytes={resume}-"));
    }
    let mut response = request.send()?.error_for_status()?;
    let append = resume > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let mut out = if append {
        std::fs::OpenOptions::new().append(true).open(&part)?
    } else {
        std::fs::File::create(&part)?
    };
    let mut total = if append { resume } else { 0 };
    let bar = indicatif::ProgressBar::with_draw_target(
        Some(found.file.size.max(total)),
        indicatif::ProgressDrawTarget::stderr_with_hz(2),
    );
    bar.set_position(total);
    bar.set_message(found.file.name.clone());
    if let Ok(style) = indicatif::ProgressStyle::with_template(
        "{msg} {bar:30} {bytes}/{total_bytes} {bytes_per_sec} {eta} left",
    ) {
        bar.set_style(style);
    }
    let mut buffer = [0u8; 65536];
    loop {
        let n = response.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > max_bytes {
            bail!("file exceeds advanced.download_max_bytes");
        }
        out.write_all(&buffer[..n])?;
        bar.set_position(total);
    }
    out.sync_all()?;
    bar.finish_and_clear();
    let hash = crate::hash_file(&part)?;
    if let Some(expected) = &found.file.sha256 {
        if *expected != hash {
            // A bad partial file would fail again on resume
            let _ = std::fs::remove_file(&part);
            bail!("download hash mismatch for {}", found.file.name);
        }
    }
    std::fs::rename(&part, dest)?;
    Ok(hash)
}

/// A `Found` rebuilt from a stored model entry, so its file can be downloaded again.
pub fn found_from(m: &Model) -> Found {
    Found {
        repo: m.source.clone(),
        revision: m.revision.clone(),
        file: File {
            name: m.file.clone(),
            size: 0,
            sha256: m.sha256.clone(),
        },
        quant: m.quant.clone(),
    }
}

/// Default model id: the repository name without a `-GGUF` suffix, in lower case.
pub fn default_id(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo).to_lowercase();
    name.strip_suffix("-gguf").unwrap_or(&name).to_string()
}

/// A family adapter when the repository name matches one, else the generic `jinja` format.
pub fn guess_adapter(repo: &str) -> &'static str {
    let r = repo.to_lowercase();
    if r.contains("functiongemma") {
        "functiongemma"
    } else if r.contains("qwen3") {
        "qwen3"
    } else if r.contains("lfm2") {
        "lfm2"
    } else {
        "jinja"
    }
}

pub fn model_entry(
    found: &Found,
    id: &str,
    adapter: &str,
    sha256: Option<String>,
    pinned: bool,
) -> Model {
    Model {
        id: id.to_string(),
        adapter: adapter.to_string(),
        quant: found.quant.clone(),
        source: found.repo.clone(),
        revision: found.revision.clone(),
        file: found.file.name.clone(),
        sha256,
        settings: None,
        pinned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    fn file(name: &str) -> File {
        File {
            name: name.into(),
            size: 1,
            sha256: None,
        }
    }

    #[test]
    fn refs_split_into_repo_and_quant() {
        assert_eq!(
            parse_ref("a/b:Q4_K_M"),
            ("a/b".into(), Some("Q4_K_M".into()))
        );
        assert_eq!(parse_ref("hf:a/b"), ("a/b".into(), None));
        assert!(is_ref("hf:a/b") && !is_ref("qwen3-0.6b"));
    }

    #[test]
    fn picks_the_named_quant_or_a_sensible_default() {
        let files = vec![
            file("m-Q8_0.gguf"),
            file("m-Q4_K_M.gguf"),
            file("mmproj-m.gguf"),
            file("README.md"),
        ];
        assert_eq!(pick(&files, None, None).unwrap().name, "m-Q4_K_M.gguf");
        assert_eq!(
            pick(&files, Some("q8_0"), None).unwrap().name,
            "m-Q8_0.gguf"
        );
        assert!(pick(&files, Some("Q2_K"), None).is_err());
        assert_eq!(
            pick(&files, None, Some("m-Q8_0.gguf")).unwrap().name,
            "m-Q8_0.gguf"
        );
        assert_eq!(quant_of("m-Q4_K_M.gguf"), "Q4_K_M");
    }

    #[test]
    fn sharded_files_are_refused_with_a_reason() {
        let files = vec![file("m-00001-of-00002.gguf"), file("m-00002-of-00002.gguf")];
        let e = pick(&files, None, None).unwrap_err().to_string();
        assert!(e.contains("sharded"));
    }

    #[test]
    fn ids_and_adapters_come_from_the_repo_name() {
        assert_eq!(default_id("unsloth/Qwen3-0.6B-GGUF"), "qwen3-0.6b");
        assert_eq!(guess_adapter("unsloth/Qwen3-0.6B-GGUF"), "qwen3");
        assert_eq!(guess_adapter("x/Llama-3.2-1B-GGUF"), "jinja");
    }

    /// Serves one repository listing and one file, with resume, on a local port.
    fn serve(body: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut range = 0usize;
                loop {
                    let mut h = String::new();
                    reader.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                    if let Some(r) = h.to_lowercase().strip_prefix("range: bytes=") {
                        range = r.trim().trim_end_matches('-').parse().unwrap();
                    }
                }
                let hash = crate::digest(body);
                let (status, payload): (&str, Vec<u8>) = if line.contains("/api/models/") {
                    let info = serde_json::json!({
                        "sha": "0123456789012345678901234567890123456789",
                        "siblings": [{"rfilename":"m-Q4_K_M.gguf","size":body.len(),"lfs":{"sha256":hash,"size":body.len()}}]
                    });
                    ("200 OK", info.to_string().into_bytes())
                } else if range > 0 {
                    ("206 Partial Content", body[range..].to_vec())
                } else {
                    ("200 OK", body.to_vec())
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&payload);
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn resolves_downloads_resumes_and_verifies_against_a_local_server() {
        let body: &'static [u8] = b"GGUF-test-model-bytes-0123456789";
        std::env::set_var("HF_ENDPOINT", serve(body));
        let found = resolve("o/m-GGUF", Some("Q4_K_M"), None).unwrap();
        assert_eq!(found.revision.len(), 40);
        assert_eq!(found.quant, "Q4_K_M");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(&found.file.name);
        std::fs::write(dest.with_extension("gguf.part"), &body[..10]).unwrap();
        let hash = download(&found, &dest, 30, 1 << 20).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(hash, crate::digest(body));
        // A second call finds the verified file and downloads nothing
        assert_eq!(download(&found, &dest, 30, 1 << 20).unwrap(), hash);
        // A wrong published hash is rejected and the partial file removed
        let mut bad = found.clone();
        bad.file.sha256 = Some("0".repeat(64));
        let other = dir.path().join("other.gguf");
        assert!(download(&bad, &other, 30, 1 << 20).is_err());
        assert!(!other.exists());
        assert!(!other.with_extension("gguf.part").exists());
    }
}
