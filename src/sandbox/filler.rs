//! Seeded noise files around the planted facts: projects, logs, notes, mail, data, archive, home.
//!
//! Vocabulary avoids every planted needle; a test checks each needle appears once.

use super::Entry;
use std::collections::BTreeSet;

const WORDS: &[&str] = &[
    "alpha", "bridge", "cache", "delta", "engine", "filter", "garden", "harbor", "index", "jungle",
    "kernel", "ledger", "marble", "nickel", "orbit", "packet", "quartz", "relay", "signal",
    "tunnel", "uplink", "vector", "window", "yarrow", "zephyr", "anchor", "beacon", "copper",
    "dynamo", "ember", "falcon", "gantry", "hollow", "ivory", "jigsaw", "lantern", "meadow",
    "nimbus", "oyster", "pebble", "ripple", "summit", "thread", "umber", "valley", "willow",
    "basalt", "cobalt", "drizzle", "estuary",
];
const PHRASES: &[&str] = &[
    "request handled in {n} ms",
    "cache refreshed with {n} entries",
    "heartbeat ok",
    "queue length {n}",
    "config reloaded",
    "connection pool at {n} percent",
    "background job {n} finished",
    "metrics flushed",
];
const MODULES: &[&str] = &["core", "net", "io", "util", "store", "ui", "sched", "codec"];
const SUBDIRS: &[&str] = &["handlers", "models", "tests", "helpers"];

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn word(&mut self) -> &'static str {
        WORDS[self.below(WORDS.len())]
    }
}

fn sentence(r: &mut Rng) -> String {
    let n = 5 + r.below(8);
    let mut s = (0..n).map(|_| r.word()).collect::<Vec<_>>().join(" ");
    s.push('.');
    s
}
/// Prose paragraphs until the text reaches about `bytes`.
fn prose(r: &mut Rng, bytes: usize) -> String {
    let mut out = String::new();
    while out.len() < bytes {
        out.push_str(&sentence(r));
        out.push(if r.below(5) == 0 { '\n' } else { ' ' });
    }
    out.push('\n');
    out
}
/// Pseudo source lines until the text reaches about `bytes`.
fn code(r: &mut Rng, bytes: usize, comment: &str) -> String {
    let mut out = format!("{comment} {}\n", sentence(r));
    while out.len() < bytes {
        let (a, b) = (r.word(), r.word());
        out.push_str(&format!(
            "fn {a}_{b}(x: {}) -> {} {{ {comment} {} }}\n",
            r.below(100),
            r.below(100),
            sentence(r)
        ));
    }
    out
}
pub fn log_line(r: &mut Rng, day: usize) -> String {
    let level = ["INFO", "INFO", "INFO", "DEBUG", "WARN"][r.below(5)];
    let phrase = PHRASES[r.below(PHRASES.len())].replace("{n}", &r.below(900).to_string());
    format!(
        "2026-01-{day:02} {:02}:{:02}:{:02} {level} worker-{} {phrase}",
        r.below(24),
        r.below(60),
        r.below(60),
        1 + r.below(4)
    )
}
/// Log lines, one per entry, until the text reaches about `bytes`.
pub fn log_lines(r: &mut Rng, bytes: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut size = 0;
    while size < bytes {
        let l = log_line(r, 1 + lines.len() % 28);
        size += l.len() + 1;
        lines.push(l);
    }
    lines
}
fn csv(r: &mut Rng, bytes: usize) -> String {
    let mut out = String::from("id,name,qty,price\n");
    let mut id = 1;
    while out.len() < bytes {
        out.push_str(&format!(
            "{id},{}-{},{},{}.{:02}\n",
            r.word(),
            r.below(90),
            r.below(500),
            r.below(90),
            r.below(100)
        ));
        id += 1;
    }
    out
}
fn json(r: &mut Rng, items: usize) -> String {
    let rows = (0..items)
        .map(|i| {
            format!(
                "  {{\"id\": {i}, \"label\": \"{}\", \"value\": {}}}",
                r.word(),
                r.below(1000)
            )
        })
        .collect::<Vec<_>>();
    format!("[\n{}\n]\n", rows.join(",\n"))
}

pub fn generate(seed: u64, taken: &BTreeSet<String>) -> Vec<Entry> {
    let mut r = Rng::new(seed);
    let mut out = Vec::new();
    let mut add = |e: Entry| {
        if !taken.contains(&e.path) {
            out.push(e);
        }
    };

    for (name, ext, comment) in [
        ("webapp", "js", "//"),
        ("sensor-fw", "c", "//"),
        ("notes-cli", "rs", "//"),
    ] {
        add(Entry::file(
            &format!("projects/{name}/README.md"),
            format!("# {name}\n\n{}", prose(&mut r, 600)),
            20,
        ));
        for i in 0..25 {
            let m = MODULES[i % MODULES.len()];
            let s = SUBDIRS[r.below(SUBDIRS.len())];
            let file = format!("{}_{}.{ext}", r.word(), i);
            let size = 1500 + r.below(3000);
            add(Entry::file(
                &format!("projects/{name}/src/{m}/{s}/{file}"),
                code(&mut r, size, comment),
                30 + r.below(60) as u64,
            ));
        }
    }

    for n in 1..=5 {
        let body = log_lines(&mut r, 90_000).join("\n") + "\n";
        add(Entry::file(
            &format!("logs/app.log.{n}"),
            body,
            150 - 10 * n,
        ));
    }

    for month in 1..=6 {
        for n in 1..=6 {
            let w = r.word();
            let size = 800 + r.below(1500);
            add(Entry::file(
                &format!("notes/2026-{month:02}/note-{n:02}-{w}.txt"),
                prose(&mut r, size),
                (month * 25 + n) as u64,
            ));
        }
    }

    for n in 1..=40 {
        let (a, b) = (r.word(), r.word());
        let size = 800 + r.below(1500);
        add(Entry::file(
            &format!("mail/inbox/{n:04}-{a}-{b}.eml"),
            format!(
                "From: {a}@example.org\nSubject: {b} update\n\n{}",
                prose(&mut r, size)
            ),
            (n * 3) as u64,
        ));
    }
    for n in 1..=10 {
        let size = 600 + r.below(1000);
        add(Entry::file(
            &format!("mail/sent/{n:04}-{}.eml", r.word()),
            format!("To: {}@example.org\n\n{}", r.word(), prose(&mut r, size)),
            (n * 9) as u64,
        ));
    }

    for n in 1..=20 {
        let w = r.word();
        let size = 2500 + r.below(3000);
        add(Entry::file(
            &format!("docs/{w}-guide-{n:02}.md"),
            format!("# {w} guide\n\n{}", prose(&mut r, size)),
            (n * 7) as u64,
        ));
    }

    for m in 1..=12 {
        let size = 15_000 + r.below(10_000);
        add(Entry::file(
            &format!("data/metrics_2026_{m:02}.csv"),
            csv(&mut r, size),
            (m * 12) as u64,
        ));
    }
    for n in 1..=6 {
        let items = 10 + r.below(20);
        add(Entry::file(
            &format!("data/records_{n}.json"),
            json(&mut r, items),
            (n * 20) as u64,
        ));
    }

    for year in [2024, 2025] {
        for q in 1..=4 {
            let size = 2000 + r.below(2500);
            add(Entry::file(
                &format!("archive/{year}/Q{q}/summary.txt"),
                prose(&mut r, size),
                (q * 5) as u64,
            ));
        }
    }
    for n in 1..=24 {
        let size = 2000 + r.below(3000);
        add(Entry::file(
            &format!("archive/misc/file_{n:02}.txt"),
            prose(&mut r, size),
            n as u64,
        ));
    }

    for n in 1..=20 {
        let size = 3000 + r.below(6000);
        add(Entry::file(
            &format!("home/downloads/{}-{n}.txt", r.word()),
            prose(&mut r, size),
            (n * 4) as u64,
        ));
    }
    for n in 1..=15 {
        let size = 4000 + r.below(6000);
        add(Entry::file(
            &format!("home/documents/{}_{n}.txt", r.word()),
            prose(&mut r, size),
            (n * 6) as u64,
        ));
    }
    out
}

pub fn fixed_log(seed: u64, bytes: usize) -> Vec<String> {
    log_lines(&mut Rng::new(seed), bytes)
}
