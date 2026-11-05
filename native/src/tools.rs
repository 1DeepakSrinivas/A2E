// tools.rs
// Built-in tools for local files, arithmetic, simulated devices, and web retrieval
// Task-scoped fixtures, bounded excerpts, and live destination checks

use crate::{read_bounded, sdk::ToolHost, text_limit};
use anyhow::{bail, Context, Result};
use reqwest::{blocking::Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    net::{IpAddr, SocketAddr, ToSocketAddrs},
    path::{Component, Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Page {
    pub id: String,
    pub url: String,
    pub title: String,
    pub text: String,
}
pub fn words(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_lowercase().trim_end_matches('s').to_string())
        .collect()
}
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
pub fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => {
            let o = v.octets();
            !v.is_private()
                && !v.is_loopback()
                && !v.is_link_local()
                && !v.is_unspecified()
                && !v.is_multicast()
                && !v.is_broadcast()
                && o[0] != 0
                && o[0] < 224
                && !(o[0] == 100 && (64..=127).contains(&o[1]))
                && !(o[0] == 192 && o[1] == 0)
                && !(o[0] == 198 && (18..=19).contains(&o[1]))
        }
        IpAddr::V6(v) => {
            let s = v.segments();
            if let Some(v) = v.to_ipv4_mapped() {
                return public_ip(v.into());
            }
            !v.is_loopback()
                && !v.is_unspecified()
                && (s[0] & 0xe000) == 0x2000
                && !(s[0] == 0x2001 && (s[1] == 0xdb8 || s[1] < 0x200))
        }
    }
}
/// Allowed URL schemes, ports, and public destinations.
pub fn validate_url(raw: &str) -> Result<Url> {
    let u = Url::parse(raw)?;
    if !["https", "http"].contains(&u.scheme())
        || u.host_str().is_none()
        || !u.username().is_empty()
        || u.password().is_some()
        || !matches!(u.port_or_known_default(), Some(80 | 443))
    {
        bail!("unsupported web destination");
    }
    let host = u
        .host_str()
        .context("missing host")?
        .trim_matches(['[', ']']);
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        bail!("local destination denied");
    }
    if let Ok(ip) = host.parse() {
        if !public_ip(ip) {
            bail!("nonpublic destination denied");
        }
    }
    Ok(u)
}
fn resolve_public(u: &Url) -> Result<Vec<SocketAddr>> {
    let host = u
        .host_str()
        .context("missing host")?
        .trim_matches(['[', ']'])
        .to_owned();
    let port = u.port_or_known_default().context("missing port")?;
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let r = (host.as_str(), port)
            .to_socket_addrs()
            .map(|a| a.collect::<Vec<_>>());
        let _ = tx.send(r);
    });
    let addresses = rx
        .recv_timeout(Duration::from_secs(3))
        .context("DNS timeout")??;
    if addresses.is_empty() || addresses.iter().any(|a| !public_ip(a.ip())) {
        bail!("nonpublic DNS result denied");
    }
    Ok(addresses)
}
pub fn html_text(html: &str) -> String {
    let doc = scraper::Html::parse_document(html);
    let mut parts = Vec::new();
    for node in doc.tree.nodes() {
        if let scraper::node::Node::Text(t) = node.value() {
            if node.ancestors().any(|a| {
                a.value().as_element().is_some_and(|e| {
                    ["script", "style", "noscript", "nav", "footer"].contains(&e.name())
                })
            }) {
                continue;
            }
            let clean = t.text.split_whitespace().collect::<Vec<_>>().join(" ");
            if !clean.is_empty() {
                parts.push(clean);
            }
        }
    }
    parts.join("\n")
}
pub struct Builtins {
    root: PathBuf,
    files: BTreeSet<String>,
    devices: BTreeMap<String, Value>,
    pages: Vec<Page>,
    returned: BTreeMap<String, Page>,
    evidence: BTreeMap<String, Page>,
    live: bool,
    requests: usize,
    http: Client,
    faults: BTreeMap<String, usize>,
}
impl Builtins {
    pub fn new(root: PathBuf, live: bool) -> Result<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            files: BTreeSet::new(),
            devices: BTreeMap::new(),
            pages: Vec::new(),
            returned: BTreeMap::new(),
            evidence: BTreeMap::new(),
            live,
            requests: 0,
            http: Client::builder()
                .timeout(Duration::from_secs(12))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            faults: BTreeMap::new(),
        })
    }
    fn take_request(&mut self) -> Result<()> {
        self.requests += 1;
        if self.requests > 8 {
            bail!("web request budget exhausted");
        }
        Ok(())
    }
    /// Stable source IDs across search and retrieval.
    fn search(&mut self, query: &str) -> Result<Value> {
        let pages = if self.live {
            self.take_request()?;
            let key =
                std::env::var("BRAVE_SEARCH_API_KEY").context("BRAVE_SEARCH_API_KEY is missing")?;
            let response = self
                .http
                .get("https://api.search.brave.com/res/v1/web/search")
                .query(&[("q", query), ("count", "3")])
                .header("X-Subscription-Token", key)
                .send()?
                .error_for_status()?;
            let data: Value = serde_json::from_slice(&read_bounded(response, 262144)?)?;
            data["web"]["results"]
                .as_array()
                .into_iter()
                .flatten()
                .take(3)
                .filter_map(|v| {
                    let url = v["url"].as_str()?;
                    validate_url(url).ok()?;
                    Some(Page {
                        id: format!("src_{}", &crate::digest(url.as_bytes())[..12]),
                        url: url.into(),
                        title: text_limit(v["title"].as_str().unwrap_or(""), 160),
                        text: text_limit(v["description"].as_str().unwrap_or(""), 300),
                    })
                })
                .collect::<Vec<_>>()
        } else {
            let q = words(query);
            let mut matches = self
                .pages
                .iter()
                .filter_map(|p| {
                    let w = words(&format!("{} {}", p.title, p.text));
                    let score = q.intersection(&w).count();
                    (score > 0).then_some((score, p.clone()))
                })
                .collect::<Vec<_>>();
            matches.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
            matches.into_iter().take(3).map(|(_, p)| p).collect()
        };
        let mut out = Vec::new();
        for p in pages {
            out.push(json!({"source_id":p.id,"url":p.url,"title":p.title,"snippet":text_limit(&p.text,240)}));
            self.returned.insert(p.id.clone(), p);
        }
        Ok(json!({"results":out}))
    }
    /// Page excerpt from a task-scoped search result.
    fn fetch(&mut self, id: &str) -> Result<Value> {
        let mut page = self
            .returned
            .get(id)
            .cloned()
            .context("source must come from this task's search results")?;
        if self.live {
            let mut url = validate_url(&page.url)?;
            let mut response = None;
            for _ in 0..4 {
                self.take_request()?;
                let addresses = resolve_public(&url)?;
                // DNS pinning across validation and connection
                let client = Client::builder()
                    .no_proxy()
                    .timeout(Duration::from_secs(12))
                    .redirect(reqwest::redirect::Policy::none())
                    .resolve_to_addrs(url.host_str().context("missing host")?, &addresses)
                    .build()?;
                let r = client
                    .get(url.clone())
                    .header("User-Agent", "A2E-native/0.1")
                    .send()?;
                if r.status().is_redirection() {
                    let next = r
                        .headers()
                        .get("location")
                        .context("redirect without location")?
                        .to_str()?;
                    url = validate_url(url.join(next)?.as_str())?;
                    continue;
                }
                response = Some(r.error_for_status()?);
                break;
            }
            let r = response.context("redirect limit")?;
            let mime = r
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_lowercase();
            if !mime.starts_with("text/html") && !mime.starts_with("text/plain") {
                bail!("only HTML and plain text pages are supported");
            }
            let bytes = read_bounded(r, 262144)?;
            let body = String::from_utf8_lossy(&bytes);
            page.text = if mime.starts_with("text/html") {
                html_text(&body)
            } else {
                body.into_owned()
            };
            page.url = url.to_string();
        }
        page.text = text_limit(&page.text, 3000);
        self.evidence.insert(id.into(), page.clone());
        Ok(
            json!({"source_id":page.id,"url":page.url,"title":page.title,"text":page.text,"truncated":page.text.ends_with("[truncated]")}),
        )
    }
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing string {key}"))
}
impl ToolHost for Builtins {
    fn initialize(&mut self, args: Value) -> Result<Value> {
        let f = &args["fixtures"];
        self.files.clear();
        self.returned.clear();
        self.evidence.clear();
        self.requests = 0;
        for (name, v) in f["files"].as_object().into_iter().flatten() {
            let p = safe_path(&self.root, name)?;
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, v.as_str().context("fixture file must be text")?)?;
            self.files.insert(name.clone());
        }
        self.devices = serde_json::from_value(f.get("devices").cloned().unwrap_or(json!({})))?;
        self.pages = serde_json::from_value(f.get("pages").cloned().unwrap_or(json!([])))?;
        self.faults = serde_json::from_value(f.get("fail_once").cloned().unwrap_or(json!({})))?;
        Ok(json!({"ready":true}))
    }
    fn call(&mut self, name: &str, a: Value) -> Result<Value> {
        if let Some(n) = self.faults.get_mut(name) {
            if *n > 0 {
                *n -= 1;
                bail!("temporary fixture error; retry is permitted");
            }
        }
        match name {
            "search_files" => {
                let q = words(string(&a, "query")?);
                let mut out = Vec::new();
                for name in &self.files {
                    let p = safe_path(&self.root, name)?;
                    let body = String::from_utf8(read_bounded(std::fs::File::open(p)?, 65536)?)?;
                    if !q.is_disjoint(&words(&format!("{name} {body}"))) {
                        out.push(name.clone());
                    }
                    if out.len() == 8 {
                        break;
                    }
                }
                Ok(json!({"paths":out}))
            }
            "read_file" => {
                let path = string(&a, "path")?;
                if !self.files.contains(path) {
                    bail!("unknown task file");
                }
                let body = String::from_utf8(read_bounded(
                    std::fs::File::open(safe_path(&self.root, path)?)?,
                    65536,
                )?)?;
                Ok(json!({"path":path,"content":text_limit(&body,3000)}))
            }
            "write_file" => {
                let path = string(&a, "path")?;
                let content = string(&a, "content")?;
                if content.len() > 4096 {
                    bail!("file content limit");
                }
                let p = safe_path(&self.root, path)?;
                if let Some(p) = p.parent() {
                    std::fs::create_dir_all(p)?;
                }
                std::fs::write(p, content)?;
                self.files.insert(path.into());
                Ok(json!({"written":path}))
            }
            "calculate" => {
                let x = a["a"].as_f64().context("a must be numeric")?;
                let y = a["b"].as_f64().context("b must be numeric")?;
                let v = match string(&a, "operation")? {
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
            "device_list" => Ok(json!({"devices":self.devices.keys().collect::<Vec<_>>()})),
            "device_status" => Ok(self
                .devices
                .get(string(&a, "name")?)
                .context("unknown device")?
                .clone()),
            "set_setting" => {
                let value = a["value"].as_i64().context("setting must be an integer")?;
                if !(0..=100).contains(&value) {
                    bail!("setting outside permitted range");
                }
                let d = self
                    .devices
                    .get_mut(string(&a, "name")?)
                    .context("unknown device")?;
                d["setting"] = json!(value);
                Ok(json!({"setting":value}))
            }
            "web_search" => self.search(string(&a, "query")?),
            "fetch_page" => self.fetch(string(&a, "source_id")?),
            _ => bail!("unknown tool {name}"),
        }
    }
    fn snapshot(&mut self) -> Result<Value> {
        let mut files = BTreeMap::new();
        for name in &self.files {
            let p = safe_path(&self.root, name)?;
            let bytes = read_bounded(std::fs::File::open(p)?, 65536)?;
            files.insert(name.clone(), String::from_utf8(bytes)?);
        }
        Ok(
            json!({"files":files,"devices":self.devices,"fetched_sources":self.evidence.keys().collect::<Vec<_>>(),"fetched_urls":self.evidence.values().map(|page| &page.url).collect::<Vec<_>>(),"web_requests":self.requests}),
        )
    }
}
