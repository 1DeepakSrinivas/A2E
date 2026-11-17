//! Bounded line reads and process-group cleanup for managed children.
//!
//! Used by the llama-server launcher and the power sampler.

use anyhow::{bail, Result};
use std::{io::BufRead, process::Child};

/// Bounded read of one newline-terminated line.
pub fn line(reader: &mut impl BufRead, max: usize) -> Result<Option<Vec<u8>>> {
    let mut out = Vec::new();
    loop {
        let b = reader.fill_buf()?;
        if b.is_empty() {
            return if out.is_empty() {
                Ok(None)
            } else {
                bail!("incomplete line")
            };
        }
        let n = b
            .iter()
            .position(|c| *c == b'\n')
            .map(|i| i + 1)
            .unwrap_or(b.len());
        if out.len() + n > max {
            return Err(crate::error::HarnessError::OutputLimit("line too large".into()).into());
        }
        let done = b[n - 1] == b'\n';
        out.extend_from_slice(&b[..n]);
        reader.consume(n);
        if done {
            return Ok(Some(out));
        }
    }
}
/// Kills the child's process group, then reaps the child.
pub fn kill_group(child: &mut Child) {
    // SAFETY: Numeric process-group ID from the managed child
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}
