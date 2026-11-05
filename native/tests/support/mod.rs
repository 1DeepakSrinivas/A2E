//! Local mock llama.cpp server for runtime tests
//! Scripted completions, request capture, and tool-call fixtures

use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

pub struct MockServer {
    pub url: String,
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
