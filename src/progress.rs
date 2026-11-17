//! Live progress on stderr. Output falls back to plain lines when stderr is piped or `NO_COLOR` is set.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;

pub struct Progress {
    bar: Option<ProgressBar>,
}

impl Progress {
    /// No bar; lines print as they arrive.
    pub fn plain() -> Self {
        Self { bar: None }
    }

    /// A bar for `total` attempts, redrawn at most once a second to keep the run quiet.
    pub fn new(total: u64) -> Self {
        if !std::io::stderr().is_terminal() || std::env::var_os("NO_COLOR").is_some() {
            return Self::plain();
        }
        let bar = ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr_with_hz(1));
        if let Ok(style) = ProgressStyle::with_template(
            "{bar:30} {pos}/{len} {msg} | {elapsed} elapsed, {eta} left",
        ) {
            bar.set_style(style);
        }
        Self { bar: Some(bar) }
    }

    pub fn set_stage(&self, stage: impl Into<String>) {
        if let Some(bar) = &self.bar {
            bar.set_message(stage.into());
        }
    }

    pub fn advance(&self) {
        if let Some(bar) = &self.bar {
            bar.inc(1);
        }
    }

    /// Prints one result line to stdout without breaking the bar.
    pub fn line(&self, text: &str) {
        match &self.bar {
            Some(bar) => bar.suspend(|| println!("{text}")),
            None => println!("{text}"),
        }
    }

    pub fn finish(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }
}
