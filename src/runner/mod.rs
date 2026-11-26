//! Experiment planning and serial case execution.
//!
//! `orchestrate` plans and runs a study, `task` runs one case, `attempt` holds the model loop,
//! `load` starts the inference server and `outcome` defines what an attempt reports.

mod attempt;
mod load;
mod orchestrate;
mod outcome;
mod task;

pub use attempt::{select_tools, system_message, TaskExecution, Trace};
pub use orchestrate::{attempt_line, plan, preflight, run, RunOptions};
pub use outcome::{classify, classify_error, no_call_returned, Outcome, Status};
pub use task::task_run;
