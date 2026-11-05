// measure.rs
// Device preflight and resource sampling during model loads and tasks
// Process memory, CPU time, Pi thermals, and optional Mac power readings

use crate::config::Runtime;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::process::CommandExt;
use std::{
    io::{BufReader, BufWriter, Write},
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use sysinfo::{Pid, ProcessesToUpdate, System};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Resources {
    pub harness_peak_bytes: u64,
    pub tool_peak_bytes: u64,
    pub inference_peak_bytes: u64,
    pub combined_peak_bytes: u64,
    pub available_min_bytes: Option<u64>,
    pub cpu_time_ms: Option<u64>,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
    pub peak_temperature_c: Option<f64>,
    pub throttled: Option<String>,
    pub energy_j: Option<f64>,
    pub power_scope: String,
    pub samples: u64,
    pub memory_exceeded: bool,
    pub telemetry_error: Option<String>,
}
/// Device and swap preflight for the baseline.
pub fn machine(r: &Runtime) -> Result<Value> {
    let mut s = System::new();
    s.refresh_memory();
    let model = std::fs::read_to_string("/proc/device-tree/model").unwrap_or_default();
    let detected = if model.contains("Raspberry Pi 3 Model B Plus") {
        "pi3bplus"
    } else if std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64" {
        "mac"
    } else {
        "unsupported"
    };
    let device = if r.device == "auto" {
        detected
    } else {
        &r.device
    };
    if !["pi3bplus", "mac"].contains(&device) {
        bail!("unsupported device; expected Pi 3B+ or Apple Silicon Mac");
    }
    if device != detected {
        bail!("configured device does not match detected hardware");
    }
    if device == "pi3bplus" && (std::env::consts::ARCH != "aarch64" || s.total_swap() != 0) {
        bail!("Pi baseline requires 64-bit Linux with swap disabled");
    }
    Ok(
        json!({"device":device,"arch":std::env::consts::ARCH,"os":System::long_os_version(),"total_memory_bytes":s.total_memory(),"available_memory_bytes":s.available_memory(),"swap_total_bytes":s.total_swap(),"kernel":System::kernel_version(),"cpu_threads":std::thread::available_parallelism().map(usize::from).ok()}),
    )
}
pub fn fault_counts(pid: u32) -> Option<(u64, u64)> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = s.rsplit_once(')')?.1.split_whitespace().collect::<Vec<_>>();
    Some((rest.get(7)?.parse().ok()?, rest.get(9)?.parse().ok()?))
}
#[cfg(target_os = "macos")]
fn cpu_time_ms(pid: u32) -> Option<u64> {
    // SAFETY: Integer-only structure with valid zero state
    let mut task: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
    // SAFETY: Writable buffer sized for proc_taskinfo
    let bytes = unsafe {
        libc::proc_pidinfo(
            pid as i32,
            libc::PROC_PIDTASKINFO,
            0,
            (&mut task as *mut libc::proc_taskinfo).cast(),
            size,
        )
    };
    (bytes == size).then_some((task.pti_total_user + task.pti_total_system) / 1_000_000)
}
#[cfg(target_os = "linux")]
fn cpu_time_ms(pid: u32) -> Option<u64> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields = text
        .rsplit_once(')')?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    let user = fields.get(11)?.parse::<u64>().ok()?;
    let system = fields.get(12)?.parse::<u64>().ok()?;
    // SAFETY: Valid clock-tick selector
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    (ticks > 0).then(|| (user + system) * 1000 / ticks as u64)
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn cpu_time_ms(_pid: u32) -> Option<u64> {
    None
}
fn pi_thermal() -> (Option<f64>, Option<String>) {
    let temp = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|v| v / 1000.);
    let throttle = if temp.is_some() {
        Command::new("vcgencmd")
            .arg("get_throttled")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    } else {
        None
    };
    (temp, throttle)
}
struct Power {
    child: std::process::Child,
    latest: Arc<Mutex<Option<(Instant, f64)>>>,
    reader: Option<thread::JoinHandle<()>>,
}
impl Power {
    fn start() -> Result<Self> {
        let mut child = Command::new("powermetrics")
            .args(["--samplers", "cpu_power,gpu_power", "-i", "500"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()?;
        let stdout = child.stdout.take().expect("stdout was configured as piped");
        let latest = Arc::new(Mutex::new(None));
        let shared = latest.clone();
        let reader = thread::spawn(move || {
            let mut b = BufReader::new(stdout);
            let mut watts = 0.;
            let mut seen = false;
            while let Ok(Some(line)) = crate::sdk::line(&mut b, 8192) {
                let s = String::from_utf8_lossy(&line);
                if s.starts_with("*** Sampled") {
                    if seen {
                        *shared.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some((Instant::now(), watts));
                    }
                    watts = 0.;
                    seen = false;
                }
                if ["CPU Power:", "GPU Power:", "ANE Power:"]
                    .iter()
                    .any(|p| s.starts_with(p))
                {
                    if let Some(n) = s
                        .split(':')
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        watts += n / 1000.;
                        seen = true;
                    }
                }
            }
        });
        Ok(Self {
            child,
            latest,
            reader: Some(reader),
        })
    }
}
impl Drop for Power {
    fn drop(&mut self) {
        crate::sdk::kill_group(&mut self.child);
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}
pub struct Monitor {
    stop: Arc<AtomicBool>,
    pub exceeded: Arc<AtomicBool>,
    summary: Arc<Mutex<Resources>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Monitor {
    /// Background sampling for one loading or task interval.
    pub fn start(
        pid: u32,
        tool_pid: Option<u32>,
        r: &Runtime,
        path: &Path,
        cancel: Arc<AtomicBool>,
    ) -> Result<Self> {
        // SAFETY: No memory preconditions
        let is_root = unsafe { libc::geteuid() } == 0;
        let power = if r.power != "none" && std::env::consts::OS == "macos" && is_root {
            Power::start().ok()
        } else {
            None
        };
        if r.power == "required" && power.is_none() {
            bail!("power_required: supported telemetry unavailable");
        }
        let file = std::fs::File::create(path)?;
        let stop = Arc::new(AtomicBool::new(false));
        let exceeded = Arc::new(AtomicBool::new(false));
        let summary = Arc::new(Mutex::new(Resources {
            power_scope: if power.is_some() {
                "powermetrics CPU+GPU+ANE; software-reported estimate; excludes DRAM/display/PSU"
                    .into()
            } else {
                "unavailable".into()
            },
            ..Default::default()
        }));
        let done = stop.clone();
        let over = exceeded.clone();
        let stats = summary.clone();
        let r = r.clone();
        let handle = thread::spawn(move || {
            let mut writer = BufWriter::new(file);
            let start = Instant::now();
            let mut system = System::new();
            let own = std::process::id();
            let pids = [own, tool_pid.unwrap_or(own), pid].map(Pid::from_u32);
            let mut previous_power: Option<(f64, f64)> = None;
            let mut energy = 0.;
            let mut energy_intervals = 0;
            let mut initial_cpu = None;
            let mut initial_faults = None;
            while !done.load(Ordering::Relaxed) {
                system.refresh_memory();
                system.refresh_processes(ProcessesToUpdate::Some(&pids), true);
                let memory = |p: u32| {
                    system
                        .process(Pid::from_u32(p))
                        .map(|p| p.memory())
                        .unwrap_or(0)
                };
                let (h, t, m) = (memory(own), tool_pid.map(memory).unwrap_or(0), memory(pid));
                let cpu = pids
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .iter()
                    .map(|p| cpu_time_ms(p.as_u32()))
                    .collect::<Option<Vec<_>>>()
                    .map(|v| v.iter().sum::<u64>());
                let initial = *initial_cpu.get_or_insert(cpu);
                let cpu_elapsed = cpu.zip(initial).map(|(a, b)| a.saturating_sub(b));
                let faults = fault_counts(pid);
                let first = *initial_faults.get_or_insert(faults);
                let (temp, throttled) = pi_thermal();
                let elapsed = start.elapsed().as_secs_f64();
                let watts = power
                    .as_ref()
                    .and_then(|p| *p.latest.lock().unwrap_or_else(|e| e.into_inner()))
                    .filter(|(at, _)| at.elapsed() < Duration::from_secs(2))
                    .map(|(_, w)| w);
                if let (Some((at, old)), Some(w)) = (previous_power, watts) {
                    if elapsed - at <= 2. {
                        energy += (elapsed - at) * (old + w) / 2.;
                        energy_intervals += 1;
                    }
                }
                previous_power = watts.map(|w| (elapsed, w));
                let exceeded = h + t + m > r.memory_limit_mb * 1024 * 1024
                    || system.available_memory() < r.reserve_mb * 1024 * 1024;
                let sample = json!({"seconds":elapsed,"harness_bytes":h,"tool_bytes":t,"inference_bytes":m,"available_bytes":system.available_memory(),"cpu_ms":cpu_elapsed,"temperature_c":temp,"throttled":throttled,"power_w":watts,"faults":faults});
                let write_result = serde_json::to_writer(&mut writer, &sample)
                    .map_err(std::io::Error::other)
                    .and_then(|_| writer.write_all(b"\n"))
                    .and_then(|_| writer.flush());
                {
                    let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                    s.harness_peak_bytes = s.harness_peak_bytes.max(h);
                    s.tool_peak_bytes = s.tool_peak_bytes.max(t);
                    s.inference_peak_bytes = s.inference_peak_bytes.max(m);
                    s.combined_peak_bytes = s.combined_peak_bytes.max(h + t + m);
                    s.available_min_bytes = Some(
                        s.available_min_bytes
                            .map_or(system.available_memory(), |a| {
                                a.min(system.available_memory())
                            }),
                    );
                    s.cpu_time_ms = cpu_elapsed;
                    s.minor_faults = faults.zip(first).map(|(a, b)| a.0.saturating_sub(b.0));
                    s.major_faults = faults.zip(first).map(|(a, b)| a.1.saturating_sub(b.1));
                    s.peak_temperature_c = match (s.peak_temperature_c, temp) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (a, b) => a.or(b),
                    };
                    if throttled.is_some() {
                        s.throttled = throttled;
                    }
                    s.energy_j = (energy_intervals > 0).then_some(energy);
                    s.samples += 1;
                    s.memory_exceeded |= exceeded;
                    if let Err(e) = write_result {
                        s.telemetry_error = Some(e.to_string());
                    }
                }
                if exceeded || cancel.load(Ordering::Relaxed) {
                    over.store(exceeded, Ordering::Relaxed);
                    // SAFETY: Process-group IDs from managed children
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                        if let Some(p) = tool_pid {
                            libc::kill(-(p as i32), libc::SIGKILL);
                        }
                    }
                    break;
                }
                thread::park_timeout(Duration::from_millis(r.sample_ms));
            }
            drop(power);
        });
        Ok(Self {
            stop,
            exceeded,
            summary,
            thread: Some(handle),
        })
    }
    pub fn finish(mut self) -> Resources {
        self.shutdown();
        self.summary
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}
impl Drop for Monitor {
    fn drop(&mut self) {
        self.shutdown();
    }
}
