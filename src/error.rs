//! Errors the harness raises itself. Their text is the stable reason code stored on an attempt.

use crate::runner::Status;
use std::fmt;

#[derive(Debug)]
pub enum HarnessError {
    Interrupted,
    TaskTimeout,
    MemoryBudget,
    ModelLoadMemoryBudget,
    OutputLimit(String),
    ModelStalled { attempts: usize },
}

impl HarnessError {
    /// Attempt status this error maps to.
    pub fn status(&self) -> Status {
        match self {
            Self::Interrupted => Status::Interrupted,
            Self::ModelStalled { .. } => Status::InfrastructureFailure,
            _ => Status::ResourceLimit,
        }
    }
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interrupted => f.write_str("interrupted"),
            Self::TaskTimeout => f.write_str("task_timeout"),
            Self::MemoryBudget => f.write_str("memory_budget"),
            Self::ModelLoadMemoryBudget => f.write_str("model_load_memory_budget"),
            Self::OutputLimit(what) => write!(f, "output_limit: {what}"),
            Self::ModelStalled { attempts } => write!(
                f,
                "model_stalled: skipped after {attempts} attempts with no completed model call"
            ),
        }
    }
}

impl std::error::Error for HarnessError {}
