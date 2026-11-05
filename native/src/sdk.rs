// sdk.rs
// Versioned JSON Lines protocol for tool executables
// Tool host dispatch, bounded process I/O, deadlines, and cleanup

use crate::config::Manifest;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::process::CommandExt;
use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub const PROTOCOL: u32 = 1;
pub const MAX_REQUEST: usize = 262144;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub protocol: u32,
    pub id: u64,
    pub method: String,
    pub name: String,
    pub arguments: Value,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub protocol: u32,
    pub id: u64,
    pub ok: bool,
    pub output: Value,
    pub error: Option<String>,
}

/// Bounded read of one protocol message.
pub fn line(reader: &mut impl BufRead, max: usize) -> Result<Option<Vec<u8>>> {
    let mut out = Vec::new();
    loop {
        let b = reader.fill_buf()?;
        if b.is_empty() {
            return if out.is_empty() {
                Ok(None)
            } else {
                bail!("incomplete protocol line")
            };
        }
        let n = b
            .iter()
            .position(|c| *c == b'\n')
            .map(|i| i + 1)
            .unwrap_or(b.len());
        if out.len() + n > max {
            bail!("output_limit: protocol line too large");
        }
        let done = b[n - 1] == b'\n';
        out.extend_from_slice(&b[..n]);
        reader.consume(n);
        if done {
            return Ok(Some(out));
        }
    }
}

pub trait ToolHost {
    fn initialize(&mut self, fixtures: Value) -> Result<Value>;
    fn call(&mut self, name: &str, arguments: Value) -> Result<Value>;
    fn snapshot(&mut self) -> Result<Value>;
}
/// JSON lines dispatcher for tool authors.
pub fn serve(host: &mut impl ToolHost) -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout().lock();
    while let Some(bytes) = line(&mut reader, MAX_REQUEST)? {
        let r: Request = serde_json::from_slice(&bytes)?;
        let result = if r.protocol != PROTOCOL {
            Err(anyhow::anyhow!("protocol mismatch"))
        } else {
            match r.method.as_str() {
                "init" => host.initialize(r.arguments),
                "call" => host.call(&r.name, r.arguments),
                "snapshot" => host.snapshot(),
                _ => Err(anyhow::anyhow!("unknown method")),
            }
        };
        let reply = match result {
            Ok(v) => Response {
                protocol: PROTOCOL,
                id: r.id,
                ok: true,
                output: v,
                error: None,
            },
            Err(e) => Response {
                protocol: PROTOCOL,
                id: r.id,
                ok: false,
                output: Value::Null,
                error: Some(e.to_string()),
            },
        };
        serde_json::to_writer(&mut out, &reply)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    Ok(())
}

pub fn kill_group(child: &mut Child) {
    // Process group includes managed descendants
    // SAFETY: Numeric process-group ID from the managed child
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}
pub struct ToolProcess {
    child: Child,
    input: ChildStdin,
    replies: mpsc::Receiver<Result<Response>>,
    sequence: u64,
    timeout: Duration,
    reader: Option<thread::JoinHandle<()>>,
    healthy: bool,
    pub startup_ms: f64,
}
impl ToolProcess {
    pub fn spawn(m: &Manifest, root: &std::path::Path, live: bool) -> Result<Self> {
        let start = Instant::now();
        let executable = if m.executable == "@self" {
            std::env::current_exe()?
        } else {
            crate::config::expand_path(&m.executable)
        };
        let mut child = Command::new(executable)
            .args(&m.args)
            .env("A2E_TOOL_ROOT", root)
            .env("A2E_WEB_MODE", if live { "live" } else { "fixture" })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(std::fs::File::create(
                root.join("tools.stderr"),
            )?))
            .process_group(0)
            .spawn()?;
        let input = child.stdin.take().context("tool stdin")?;
        let stdout = child.stdout.take().context("tool stdout")?;
        let (tx, replies) = mpsc::sync_channel(0);
        let max = m.output_limit_bytes;
        let reader = thread::spawn(move || {
            let mut b = BufReader::new(stdout);
            loop {
                let r = match line(&mut b, max) {
                    Ok(Some(v)) => serde_json::from_slice(&v).map_err(Into::into),
                    Ok(None) => break,
                    Err(e) => Err(e),
                };
                let stop = r.is_err();
                if tx.send(r).is_err() || stop {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            input,
            replies,
            sequence: 0,
            timeout: Duration::from_secs(m.timeout_seconds),
            reader: Some(reader),
            healthy: true,
            startup_ms: start.elapsed().as_secs_f64() * 1000.,
        })
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    /// One correlated tool request with a shared read/write deadline.
    pub fn request(
        &mut self,
        method: &str,
        name: &str,
        args: Value,
        remaining: Duration,
    ) -> Result<Response> {
        if !self.healthy {
            bail!("tool process unavailable");
        }
        self.sequence += 1;
        let request = Request {
            protocol: PROTOCOL,
            id: self.sequence,
            method: method.into(),
            name: name.into(),
            arguments: args,
        };
        let mut bytes = serde_json::to_vec(&request)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_REQUEST {
            bail!("request_limit");
        }
        // Nonblocking writes for the shared deadline
        use std::os::fd::AsRawFd;
        let fd = self.input.as_raw_fd();
        // SAFETY: Open descriptor owned by child stdin
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
        let deadline = Instant::now() + self.timeout.min(remaining);
        let mut offset = 0;
        while offset < bytes.len() {
            if Instant::now() >= deadline {
                self.stop();
                bail!("tool_timeout");
            }
            match self.input.write(&bytes[offset..]) {
                Ok(0) => {
                    self.stop();
                    bail!("tool pipe closed");
                }
                Ok(n) => offset += n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2))
                }
                Err(e) => {
                    self.stop();
                    return Err(e.into());
                }
            }
        }
        let result = self
            .replies
            .recv_timeout(deadline.saturating_duration_since(Instant::now()));
        match result {
            Ok(Ok(r)) if r.protocol == PROTOCOL && r.id == self.sequence => Ok(r),
            Ok(Ok(_)) => {
                self.stop();
                bail!("tool protocol or request id mismatch")
            }
            Ok(Err(e)) => {
                self.stop();
                Err(e)
            }
            Err(_) => {
                self.stop();
                bail!("tool_timeout_or_exit")
            }
        }
    }
    pub fn stop(&mut self) {
        if self.healthy {
            kill_group(&mut self.child);
            self.healthy = false;
        }
    }
}
impl Drop for ToolProcess {
    fn drop(&mut self) {
        self.stop();
        let (_, replacement) = mpsc::channel();
        let old = std::mem::replace(&mut self.replies, replacement);
        drop(old);
        if let Some(t) = self.reader.take() {
            let _ = t.join();
        }
    }
}
pub fn success_output(r: Response) -> Result<Value> {
    if !r.ok {
        bail!("{}", r.error.unwrap_or_else(|| "tool failed".into()));
    }
    Ok(r.output)
}
pub fn init_args(fixtures: Value) -> Value {
    json!({"fixtures":fixtures})
}
