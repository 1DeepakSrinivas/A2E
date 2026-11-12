//! Local mock llama.cpp server for runtime tests
//! Scripted completions, request capture, and tool-call fixtures

use a2e::{
    adapter::Adapter,
    case::Case,
    config::{Catalog, Runtime, Suite, Variant},
    inference::Inference,
    runner::{task_run, Outcome, TaskExecution},
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

pub struct MockServer {
    pub url: String,
    #[allow(dead_code)]
    pub seen: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}
impl MockServer {
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
                    "/tokenize" => json!({"tokens":vec![1;prompt_tokens]}),
                    "/slots/0?action=erase" => {
                        index.store(0, Ordering::Relaxed);
                        json!({"id_slot":0})
                    }
                    "/completion" => {
                        thread::sleep(delay);
                        let i = index.fetch_add(1, Ordering::Relaxed);
                        json!({"content":outputs.get(i).cloned().unwrap_or_else(||"done".into()),"stop_type":"eos","tokens_cached":0,"timings":{"prompt_n":prompt_tokens,"prompt_ms":2.0,"predicted_n":12,"predicted_ms":4.0}})
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
    let dir = tempfile::tempdir().unwrap();
    let client = Inference::new(server.url.clone()).unwrap();
    let runtime = Runtime {
        memory_limit_mb: 1_000_000,
        reserve_mb: 0,
        power: "none".into(),
        ..Runtime::default()
    };
    task_run(TaskExecution {
        client: &client,
        adapter: Adapter::Qwen,
        case: t,
        suite: s,
        variant: v,
        catalog: &catalog(),
        ctx: 1024,
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
