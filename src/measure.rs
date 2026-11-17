// measure.rs
// Device preflight and resource sampling during model loads and tasks
// Process memory, CPU time, Pi thermals, and optional Mac power readings

use crate::config::Runtime;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::process::CommandExt;
use std::{
    io::BufReader,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use sysinfo::{Pid, ProcessesToUpdate, System};

/// One resource reading; `t_ms` counts from the start of monitoring.
#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub t_ms: f64,
    pub harness_bytes: u64,
    pub inference_bytes: u64,
    pub available_bytes: Option<u64>,
    pub cpu_ms: Option<u64>,
    pub temperature_c: Option<f64>,
    pub throttled: Option<String>,
    pub power_w: Option<f64>,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
}
/// Samples kept per interval; summary statistics still cover every sample.
pub const MAX_SAMPLES: usize = 20_000;
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Resources {
    pub harness_peak_bytes: u64,
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
    pub sample_count: u64,
    pub memory_exceeded: bool,
    #[serde(skip)]
    pub series: Vec<Sample>,
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
const PI_MODEL_SCOPE: &str = "modeled: Joular rbp3b+1.3 polynomial of whole-system CPU usage; not measured, error unknown on this unit";
// Joular rbp3b+1.3 CPU model (ODbL): watts = intercept + sum(c[i] * usage^(i+1)), usage in 0 to 1
const PI_MODEL_INTERCEPT: f64 = 2.484_396_997_449_118_2;
const PI_MODEL_COEFFICIENTS: [f64; 9] = [
    2.933542,
    -150.400134,
    2278.690310,
    -15008.559279,
    51537.315529,
    -98756.887779,
    106478.929766,
    -60432.910139,
    14053.677709,
];
fn pi_model_watts(usage: f64) -> f64 {
    let u = usage.clamp(0., 1.);
    PI_MODEL_INTERCEPT
        + PI_MODEL_COEFFICIENTS
            .iter()
            .rev()
            .fold(0., |acc, c| (acc + c) * u)
}
fn is_pi3bplus() -> bool {
    std::fs::read_to_string("/proc/device-tree/model")
        .is_ok_and(|m| m.contains("Raspberry Pi 3 Model B Plus"))
}
fn pi_thermal() -> (Option<f64>, Option<String>) {
    let temp = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|v| v / 1000.)
        .or_else(|| {
            Command::new("vcgencmd")
                .arg("measure_temp")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .strip_prefix("temp=")?
                        .trim_end_matches("'C")
                        .parse()
                        .ok()
                })
        });
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
/// Available system memory in bytes; 0 when the platform does not report it.
pub fn available_memory() -> u64 {
    let mut s = System::new();
    s.refresh_memory();
    s.available_memory()
}
/// Names for the bits of a Pi `get_throttled` reading; empty for a clean or foreign reading.
pub fn throttle_flags(v: &str) -> Vec<&'static str> {
    let v = v.trim().to_ascii_lowercase();
    let v = v.strip_prefix("throttled=").unwrap_or(&v);
    let Some(bits) = v
        .strip_prefix("0x")
        .and_then(|h| u32::from_str_radix(h, 16).ok())
    else {
        return Vec::new();
    };
    [
        (0, "under-voltage now"),
        (1, "frequency capped now"),
        (2, "throttled now"),
        (3, "soft temperature limit now"),
        (16, "under-voltage since boot"),
        (17, "frequency capped since boot"),
        (18, "throttled since boot"),
        (19, "soft temperature limit since boot"),
    ]
    .into_iter()
    .filter(|(bit, _)| bits & (1 << bit) != 0)
    .map(|(_, name)| name)
    .collect()
}
/// Problems that make timings from this device unrepresentative; shown by `doctor` and before a run.
pub fn preflight_warnings(r: &Runtime) -> Vec<String> {
    let mut out = Vec::new();
    if let (_, Some(t)) = pi_thermal() {
        let flags = throttle_flags(&t);
        if !flags.is_empty() {
            out.push(format!(
                "{t} ({}); the firmware slows the CPU under these conditions, so fix the power supply or cooling and reboot before measuring",
                flags.join(", ")
            ));
        }
    }
    let mut s = System::new();
    s.refresh_memory();
    let total_mb = s.total_memory() >> 20;
    if total_mb > 0 && r.memory_limit_mb > total_mb {
        out.push(format!(
            "memory_limit_mb = {} exceeds the {total_mb} MiB of physical memory, so the memory guard cannot trip",
            r.memory_limit_mb
        ));
    }
    out
}
/// True for a reading with no throttling: `throttled=0x0` on the Pi, `Nominal` pressure on the Mac.
pub fn throttle_clean(v: &str) -> bool {
    let v = v.trim().to_ascii_lowercase();
    let v = v.strip_prefix("throttled=").unwrap_or(&v);
    v.is_empty() || v == "0x0" || v == "nominal"
}
const POWERMETRICS: &str = "/usr/bin/powermetrics";
const POWERMETRICS_ARGS: [&str; 4] = ["--samplers", "cpu_power,gpu_power,thermal", "-i", "500"];
pub const TELEMETRY_SETUP: &str = "./setup.sh telemetry";
fn is_root() -> bool {
    // SAFETY: No memory preconditions
    unsafe { libc::geteuid() == 0 }
}
/// Checks that powermetrics can run now, directly as root or through a passwordless sudo rule.
pub fn power_access() -> Result<()> {
    if std::env::consts::OS != "macos" {
        bail!("power sampling is only supported on macOS");
    }
    if is_root() {
        return Ok(());
    }
    let allowed = Command::new("sudo")
        .args(["-n", "-l", "--", POWERMETRICS])
        .args(POWERMETRICS_ARGS)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !allowed {
        bail!(
            "powermetrics needs root; run `{TELEMETRY_SETUP}` once, or start the runner with sudo"
        );
    }
    Ok(())
}
/// What this device can report during a run; shown by `doctor`.
pub fn telemetry() -> Value {
    let (temp, throttle) = pi_thermal();
    let power = if std::env::consts::OS == "macos" {
        match power_access() {
            Ok(()) => json!("ok"),
            Err(e) => json!(format!("unavailable: {e}")),
        }
    } else {
        json!("unavailable")
    };
    json!({
        "temperature": if temp.is_some() { "ok" } else { "unavailable" },
        "throttle_flags": if throttle.is_some() || (std::env::consts::OS == "macos" && power_access().is_ok()) { "ok" } else { "unavailable" },
        "power": power,
    })
}
struct Power {
    root: bool,
    child: std::process::Child,
    latest: Arc<Mutex<Option<(Instant, f64)>>>,
    pressure: Arc<Mutex<Option<String>>>,
    reader: Option<thread::JoinHandle<()>>,
}
impl Power {
    fn start() -> Result<Self> {
        power_access()?;
        let root = is_root();
        let mut command = if root {
            Command::new(POWERMETRICS)
        } else {
            let mut c = Command::new("sudo");
            c.args(["-n", "--", POWERMETRICS]);
            c
        };
        let mut child = command
            .args(POWERMETRICS_ARGS)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()?;
        let stdout = child.stdout.take().expect("stdout was configured as piped");
        let latest = Arc::new(Mutex::new(None));
        let shared = latest.clone();
        let pressure = Arc::new(Mutex::new(None));
        let level = pressure.clone();
        let reader = thread::spawn(move || {
            let mut b = BufReader::new(stdout);
            let mut watts = 0.;
            let mut seen = false;
            while let Ok(Some(line)) = crate::process::line(&mut b, 8192) {
                let s = String::from_utf8_lossy(&line);
                if let Some(level) = s.strip_prefix("Current pressure level:") {
                    *pressure.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(level.trim().to_string());
                }
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
            root,
            child,
            latest,
            pressure: level,
            reader: Some(reader),
        })
    }
}
impl Drop for Power {
    fn drop(&mut self) {
        if self.root {
            crate::process::kill_group(&mut self.child);
        } else {
            // sudo relays SIGTERM to the root child; SIGKILL would orphan it
            // SAFETY: Numeric PID of the managed sudo child
            unsafe {
                libc::kill(self.child.id() as i32, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline && !matches!(self.child.try_wait(), Ok(Some(_))) {
                thread::sleep(Duration::from_millis(50));
            }
            crate::process::kill_group(&mut self.child);
        }
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
    pub fn start(pid: u32, r: &Runtime, cancel: Arc<AtomicBool>) -> Result<Self> {
        let power = if r.power != "none" && std::env::consts::OS == "macos" {
            match Power::start() {
                Ok(p) => Some(p),
                Err(e) if r.power == "required" => bail!("power_required: {e}"),
                Err(e) => {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| eprintln!("warning: power and thermal unavailable: {e}"));
                    None
                }
            }
        } else {
            None
        };
        if r.power == "required" && power.is_none() {
            bail!("power_required: supported telemetry unavailable");
        }
        let modeled = power.is_none() && r.power == "auto" && is_pi3bplus();
        let stop = Arc::new(AtomicBool::new(false));
        let exceeded = Arc::new(AtomicBool::new(false));
        let summary = Arc::new(Mutex::new(Resources {
            power_scope: if power.is_some() {
                "powermetrics CPU+GPU+ANE; software-reported estimate; excludes DRAM/display/PSU"
                    .into()
            } else if modeled {
                PI_MODEL_SCOPE.into()
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
            let start = Instant::now();
            let mut system = System::new();
            let own = std::process::id();
            let pids = [own, pid].map(Pid::from_u32);
            let mut previous_power: Option<(f64, f64)> = None;
            let mut energy = 0.;
            let mut energy_intervals = 0;
            let mut initial_cpu = None;
            let mut initial_faults = None;
            let mut cpu_primed = false;
            loop {
                let last = done.load(Ordering::Relaxed);
                system.refresh_memory();
                system.refresh_processes(ProcessesToUpdate::Some(&pids), true);
                // CPU usage needs two refreshes, so the first sample has no modeled power
                let usage = modeled.then(|| {
                    system.refresh_cpu_usage();
                    let ready = std::mem::replace(&mut cpu_primed, true);
                    ready.then(|| f64::from(system.global_cpu_usage()) / 100.)
                });
                let memory = |p: u32| {
                    system
                        .process(Pid::from_u32(p))
                        .map(|p| p.memory())
                        .unwrap_or(0)
                };
                let (h, m) = (memory(own), memory(pid));
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
                let (temp, pi_throttled) = pi_thermal();
                let throttled = pi_throttled.or_else(|| {
                    power
                        .as_ref()
                        .and_then(|p| p.pressure.lock().unwrap_or_else(|e| e.into_inner()).clone())
                });
                let elapsed = start.elapsed().as_secs_f64();
                let watts = power
                    .as_ref()
                    .and_then(|p| *p.latest.lock().unwrap_or_else(|e| e.into_inner()))
                    .filter(|(at, _)| at.elapsed() < Duration::from_secs(2))
                    .map(|(_, w)| w)
                    .or_else(|| usage.flatten().map(pi_model_watts));
                if let (Some((at, old)), Some(w)) = (previous_power, watts) {
                    if elapsed - at <= 2. {
                        energy += (elapsed - at) * (old + w) / 2.;
                        energy_intervals += 1;
                    }
                }
                previous_power = watts.map(|w| (elapsed, w));
                let available = Some(system.available_memory()).filter(|a| *a > 0);
                let exceeded = h + m > r.memory_limit_mb * 1024 * 1024
                    || available.is_some_and(|a| a < r.reserve_mb * 1024 * 1024);
                {
                    let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                    s.harness_peak_bytes = s.harness_peak_bytes.max(h);
                    s.inference_peak_bytes = s.inference_peak_bytes.max(m);
                    s.combined_peak_bytes = s.combined_peak_bytes.max(h + m);
                    s.available_min_bytes = match (s.available_min_bytes, available) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    };
                    s.cpu_time_ms = cpu_elapsed;
                    s.minor_faults = faults.zip(first).map(|(a, b)| a.0.saturating_sub(b.0));
                    s.major_faults = faults.zip(first).map(|(a, b)| a.1.saturating_sub(b.1));
                    s.peak_temperature_c = match (s.peak_temperature_c, temp) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (a, b) => a.or(b),
                    };
                    let (minor, major) = (s.minor_faults, s.major_faults);
                    if s.series.len() < MAX_SAMPLES {
                        s.series.push(Sample {
                            t_ms: elapsed * 1000.,
                            harness_bytes: h,
                            inference_bytes: m,
                            available_bytes: available,
                            cpu_ms: cpu_elapsed,
                            temperature_c: temp,
                            throttled: throttled.clone(),
                            power_w: watts,
                            minor_faults: minor,
                            major_faults: major,
                        });
                    }
                    if throttled.is_some() && s.throttled.as_deref().is_none_or(throttle_clean) {
                        s.throttled = throttled;
                    }
                    s.energy_j = (energy_intervals > 0).then_some(energy);
                    s.sample_count += 1;
                    s.memory_exceeded |= exceeded;
                }
                if exceeded || cancel.load(Ordering::Relaxed) {
                    over.store(exceeded, Ordering::Relaxed);
                    // SAFETY: Process-group ID from the managed inference server
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                    }
                    break;
                }
                if last {
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
        std::mem::take(&mut *self.summary.lock().unwrap_or_else(|e| e.into_inner()))
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_model_matches_the_published_endpoints() {
        assert!((pi_model_watts(0.) - 2.484397).abs() < 1e-6);
        assert!((pi_model_watts(1.) - 5.274).abs() < 1e-3);
        assert_eq!(pi_model_watts(-1.), pi_model_watts(0.));
        assert_eq!(pi_model_watts(2.), pi_model_watts(1.));
    }

    #[test]
    fn a_short_interval_still_gets_a_start_and_an_end_sample() {
        let runtime = Runtime {
            power: "none".into(),
            sample_ms: 5_000,
            ..Runtime::default()
        };
        let mut child = Command::new("sleep").arg("5").spawn().unwrap();
        let monitor =
            Monitor::start(child.id(), &runtime, Arc::new(AtomicBool::new(false))).unwrap();
        thread::sleep(Duration::from_millis(50));
        let resources = monitor.finish();
        let _ = child.kill();
        let _ = child.wait();
        assert!(resources.sample_count >= 2);
        assert!(resources.combined_peak_bytes > 0);
        assert_eq!(resources.series.len() as u64, resources.sample_count);
    }

    #[test]
    fn clean_throttle_readings_cover_both_platforms() {
        for clean in ["throttled=0x0", "0x0", "Nominal", ""] {
            assert!(throttle_clean(clean), "{clean}");
        }
        assert_eq!(
            throttle_flags("throttled=0x50005"),
            [
                "under-voltage now",
                "throttled now",
                "under-voltage since boot",
                "throttled since boot"
            ]
        );
        assert!(throttle_flags("throttled=0x0").is_empty());
        assert!(throttle_flags("Heavy").is_empty());
        for hot in ["throttled=0x50005", "Heavy"] {
            assert!(!throttle_clean(hot), "{hot}");
        }
    }
}
