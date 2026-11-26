//! Local mock llama.cpp server for runtime tests
//! Scripted completions, request capture, and tool-call fixtures

use a2e::{
    adapter::Adapter,
    case::Case,
    config::{Catalog, Suite, Variant},
    inference::Inference,
    runner::{task_run, Outcome, TaskExecution},
    settings::Settings,
};
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

/// Characters per token for the estimating server. Below the usual 3.5 to 4 of
/// JSON-heavy prompts, so the estimate errs toward a larger prompt.
#[allow(dead_code)]
pub const CHARS_PER_TOKEN: f64 = 3.2;

pub struct MockServer {
    pub url: String,
    #[allow(dead_code)]
    pub seen: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}
impl MockServer {
    /// A server whose `/tokenize` estimates the prompt length from its characters,
    /// so the context budget is exercised. `new` always reports `prompt_tokens`.
    #[allow(dead_code)]
    pub fn estimating(outputs: Vec<String>) -> Self {
        Self::new(outputs, 0, Duration::ZERO)
    }
    pub fn new(outputs: Vec<String>, prompt_tokens: usize, delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server bind");
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let done = stop.clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let records = seen.clone();
        let index = AtomicUsize::new(0);
        let handle = thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(_) => break,
                };
                // Accepted sockets inherit the listener's non-blocking flag on macOS
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                if reader.read_line(&mut first).is_err() {
                    continue;
                }
                let path = first.split_whitespace().nth(1).unwrap_or("");
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                        length = v.trim().parse().unwrap();
                    }
                }
                let mut bytes = vec![0; length];
                if reader.read_exact(&mut bytes).is_err() {
                    continue;
                }
                let body: Value = serde_json::from_slice(&bytes).unwrap_or(json!({}));
                records
                    .lock()
                    .unwrap()
                    .push(json!({"path":path,"body":body}));
                let response = match path {
                    "/health" => json!({"status":"ok"}),
                    "/props" => json!({"chat_template":"mock template"}),
                    "/apply-template" => json!({"prompt":body.to_string()}),
                    "/tokenize" => {
                        let n = if prompt_tokens == 0 {
                            let chars = body["content"].as_str().unwrap_or("").chars().count();
                            (chars as f64 / CHARS_PER_TOKEN).ceil() as usize
                        } else {
                            prompt_tokens
                        };
                        json!({"tokens":vec![1;n]})
                    }
                    "/slots" => json!([{"id":0,"is_processing":false}]),
                    "/slots/0?action=erase" => {
                        index.store(0, Ordering::Relaxed);
                        json!({"id_slot":0})
                    }
                    "/completion" => {
                        thread::sleep(delay);
                        let i = index.fetch_add(1, Ordering::Relaxed);
                        json!({"content":outputs.get(i).cloned().unwrap_or_else(||"done".into()),"stop_type":"eos","tokens_evaluated":prompt_tokens,"tokens_cached":prompt_tokens+12,"timings":{"prompt_n":prompt_tokens,"cache_n":0,"prompt_ms":2.0,"predicted_n":12,"predicted_ms":4.0}})
                    }
                    _ => json!({}),
                };
                let response = response.to_string();
                let _=write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",response.len(),response);
            }
        });
        Self {
            url,
            seen,
            stop,
            handle: Some(handle),
        }
    }
}
impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect(self.url.trim_start_matches("http://"));
        if let Some(h) = self.handle.take() {
            h.join().unwrap();
        }
    }
}
pub fn call(name: &str, args: Value) -> String {
    format!(
        "<tool_call>{}</tool_call>",
        json!({"name":name,"arguments":args})
    )
}

#[allow(dead_code)]
pub fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
#[allow(dead_code)]
pub fn catalog() -> Catalog {
    Catalog::load(&repo().join("configs/tools/standard.toml")).unwrap()
}
/// Generates the workspace tree once and points `A2E_SANDBOX` at it.
#[allow(dead_code)]
pub fn use_sandbox() {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let dir = DIR.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        a2e::sandbox::init(dir.path(), a2e::sandbox::DEFAULT_SEED, false).unwrap();
        dir
    });
    std::env::set_var("A2E_SANDBOX", dir.path());
}
/// Runs one attempt of a case against the mock server.
#[allow(dead_code)]
pub fn execute(server: &MockServer, t: &Case, v: &Variant, s: &Suite) -> Outcome {
    execute_at(server, t, v, s, 1024)
}
/// Runs one attempt at the given context size.
#[allow(dead_code)]
pub fn execute_at(server: &MockServer, t: &Case, v: &Variant, s: &Suite, ctx: usize) -> Outcome {
    let dir = tempfile::tempdir().unwrap();
    let client = Inference::new(server.url.clone()).unwrap();
    let mut runtime = Settings::default();
    runtime.limits.memory_mb = 1_000_000;
    runtime.limits.reserve_mb = 0;
    runtime.host.power = "none".into();
    task_run(TaskExecution {
        client: &client,
        adapter: Adapter::QWEN3,
        case: t,
        suite: s,
        variant: v,
        catalog: &catalog(),
        ctx,
        dir: dir.path(),
        cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        server_pid: std::process::id(),
        runtime: &runtime,
    })
    .unwrap()
}
/// Scripted model output that replays a case's golden calls and answer.
#[allow(dead_code)]
pub fn golden_script(case: &Case) -> Vec<String> {
    let mut script = case
        .golden
        .iter()
        .map(|g| {
            call(
                &g.tool,
                if g.args.is_null() {
                    json!({})
                } else {
                    g.args.clone()
                },
            )
        })
        .collect::<Vec<_>>();
    script.push(call(
        "final_answer",
        json!({"answer":case.golden_answer.clone().unwrap_or_default()}),
    ));
    script
}
