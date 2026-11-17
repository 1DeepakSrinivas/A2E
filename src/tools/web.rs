//! Search and page retrieval from fixtures or live Brave search.
//!
//! Task-scoped source IDs, bounded excerpts, and public-destination checks.

use super::words;
use crate::{fixtures::Page, read_bounded, text_limit};
use anyhow::{bail, Context, Result};
use reqwest::{blocking::Client, Url};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    net::{IpAddr, SocketAddr, ToSocketAddrs},
    time::Duration,
};

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
        .recv_timeout(Duration::from_secs(
            crate::settings::active().advanced.dns_timeout_s,
        ))
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
pub struct Web {
    pages: Vec<Page>,
    returned: BTreeMap<String, Page>,
    evidence: BTreeMap<String, Page>,
    live: bool,
    requests: usize,
}
impl Web {
    pub fn new(live: bool) -> Result<Self> {
        Ok(Self {
            pages: Vec::new(),
            returned: BTreeMap::new(),
            evidence: BTreeMap::new(),
            live,
            requests: 0,
        })
    }
    pub fn load(&mut self, pages: Vec<Page>) {
        self.pages = pages;
        self.returned.clear();
        self.evidence.clear();
        self.requests = 0;
    }
    pub fn snapshot(&self) -> Value {
        json!({"fetched_sources":self.evidence.keys().collect::<Vec<_>>(),"fetched_urls":self.evidence.values().map(|page| &page.url).collect::<Vec<_>>(),"web_requests":self.requests})
    }
    fn take_request(&mut self) -> Result<()> {
        self.requests += 1;
        if self.requests > 8 {
            bail!("web request budget exhausted");
        }
        Ok(())
    }
    /// Stable source IDs across search and retrieval.
    pub fn search(&mut self, query: &str) -> Result<Value> {
        let pages = if self.live {
            self.take_request()?;
            let key =
                std::env::var("BRAVE_SEARCH_API_KEY").context("BRAVE_SEARCH_API_KEY is missing")?;
            let response = Client::builder()
                .timeout(Duration::from_secs(
                    crate::settings::active().advanced.web_timeout_s,
                ))
                .redirect(reqwest::redirect::Policy::none())
                .build()?
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
    pub fn fetch(&mut self, id: &str) -> Result<Value> {
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
                    .timeout(Duration::from_secs(
                        crate::settings::active().advanced.web_timeout_s,
                    ))
                    .redirect(reqwest::redirect::Policy::none())
                    .resolve_to_addrs(url.host_str().context("missing host")?, &addresses)
                    .build()?;
                let r = client
                    .get(url.clone())
                    .header("User-Agent", "A2E/0.1")
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
pub fn web_search(w: &mut super::World, a: &Value) -> Result<Value> {
    w.web.search(super::string(a, "query")?)
}
pub fn fetch_page(w: &mut super::World, a: &Value) -> Result<Value> {
    w.web.fetch(super::string(a, "source_id")?)
}
pub fn initialize(w: &mut super::World, fixtures: &crate::fixtures::Fixtures) {
    w.web.load(fixtures.pages.clone());
}
pub fn register(r: &mut super::Registry) {
    r.insert("web_search", web_search);
    r.insert("fetch_page", fetch_page);
}
