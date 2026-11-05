// store/record.rs
// Batched write of one finished attempt: headline row, turns, calls, checks, samples
// One transaction after the attempt's wall time is fixed

use super::{tag, Store};
use crate::{measure::Resources, runner::Outcome};
use anyhow::Result;
use rusqlite::{named_params, params, Transaction};

pub enum SampleOwner {
    Run(i64),
    Load(i64),
}
/// Resource series for a run or a model load.
pub fn samples(tx: &Transaction, owner: SampleOwner, r: &Resources) -> Result<()> {
    let (run, load) = match owner {
        SampleOwner::Run(id) => (Some(id), None),
        SampleOwner::Load(id) => (None, Some(id)),
    };
    let mut insert = tx.prepare_cached("INSERT INTO samples(run_id,load_id,idx,t_ms,harness_bytes,inference_bytes,available_bytes,cpu_ms,temperature_c,throttled,power_w,minor_faults,major_faults) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")?;
    for (i, s) in r.series.iter().enumerate() {
        insert.execute(params![
            run,
            load,
            i,
            s.t_ms,
            s.harness_bytes,
            s.inference_bytes,
            s.available_bytes,
            s.cpu_ms,
            s.temperature_c,
            s.throttled,
            s.power_w,
            s.minor_faults,
            s.major_faults
        ])?;
    }
    Ok(())
}
impl Store {
    pub fn finish(&self, id: i64, out: &Outcome) -> Result<()> {
        let tx = self.db.unchecked_transaction()?;
        let r = &out.resources;
        tx.execute(
            "UPDATE runs SET state=:state,status=:status,reason=:reason,failure_class=:class,answer=:answer,evidence=:evidence,final_state=:final_state,retrieval_success=:retrieval,
             model_calls=:model_calls,tool_invocations=:tool_invocations,valid_tool_calls=:valid,argument_errors=:argument_errors,repairs=:repairs,
             prompt_tokens=:prompt_tokens,generated_tokens=:generated_tokens,cached_tokens=:cached_tokens,
             prefill_ms=:prefill_ms,decode_ms=:decode_ms,http_ms=:http_ms,template_ms=:template_ms,tokenize_ms=:tokenize_ms,parse_ms=:parse_ms,validate_ms=:validate_ms,tool_ms=:tool_ms,tool_setup_ms=:tool_setup_ms,wall_ms=:wall_ms,harness_other_ms=:other_ms,
             harness_peak_bytes=:harness_peak,inference_peak_bytes=:inference_peak,combined_peak_bytes=:combined_peak,available_min_bytes=:available_min,cpu_time_ms=:cpu,minor_faults=:minor,major_faults=:major,
             peak_temperature_c=:temperature,throttled=:throttled,energy_j=:energy,power_scope=:power_scope,sample_count=:sample_count,memory_exceeded=:exceeded
             WHERE id=:id",
            named_params! {
                ":id": id,
                ":state": if out.status == crate::runner::Status::Interrupted { "interrupted" } else { "complete" },
                ":status": tag(&out.status),
                ":reason": out.reason,
                ":class": out.failure_class.as_ref().map(tag),
                ":answer": out.answer,
                ":evidence": serde_json::to_string(&out.evidence)?,
                ":final_state": out.final_state.as_ref().map(ToString::to_string),
                ":retrieval": out.retrieval_success,
                ":model_calls": out.model_calls,
                ":tool_invocations": out.tool_invocations,
                ":valid": out.valid_tool_calls,
                ":argument_errors": out.argument_errors,
                ":repairs": out.repairs,
                ":prompt_tokens": out.prompt_tokens,
                ":generated_tokens": out.generated_tokens,
                ":cached_tokens": out.cached_tokens,
                ":prefill_ms": out.prefill_ms,
                ":decode_ms": out.decode_ms,
                ":http_ms": out.http_ms,
                ":template_ms": out.template_ms,
                ":tokenize_ms": out.tokenize_ms,
                ":parse_ms": out.parse_ms,
                ":validate_ms": out.validate_ms,
                ":tool_ms": out.tool_ms,
                ":tool_setup_ms": out.tool_setup_ms,
                ":wall_ms": out.wall_ms,
                ":other_ms": out.harness_other_ms,
                ":harness_peak": r.harness_peak_bytes,
                ":inference_peak": r.inference_peak_bytes,
                ":combined_peak": r.combined_peak_bytes,
                ":available_min": r.available_min_bytes,
                ":cpu": r.cpu_time_ms,
                ":minor": r.minor_faults,
                ":major": r.major_faults,
                ":temperature": r.peak_temperature_c,
                ":throttled": r.throttled,
                ":energy": r.energy_j,
                ":power_scope": r.power_scope,
                ":sample_count": r.sample_count,
                ":exceeded": r.memory_exceeded,
            },
        )?;
        {
            let mut turn = tx.prepare_cached("INSERT INTO turns(run_id,idx,visible_tools,template_ms,tokenize_ms,http_ms,prefill_ms,decode_ms,prompt_tokens,cached_tokens,generated_tokens,parse_ms,parse_status,parse_error,raw_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")?;
            for t in &out.turns {
                turn.execute(params![
                    id,
                    t.idx,
                    t.visible_tools,
                    t.template_ms,
                    t.tokenize_ms,
                    t.http_ms,
                    t.prefill_ms,
                    t.decode_ms,
                    t.prompt_tokens,
                    t.cached_tokens,
                    t.generated_tokens,
                    t.parse_ms,
                    tag(&t.parse_status),
                    t.parse_error,
                    t.raw_text
                ])?;
            }
            let mut call = tx.prepare_cached("INSERT INTO tool_calls(run_id,idx,turn,tool,arguments,valid,ok,validate_ms,ms,output_bytes,error) VALUES (?,?,?,?,?,?,?,?,?,?,?)")?;
            for (i, c) in out.tool_log.iter().enumerate() {
                call.execute(params![
                    id,
                    i,
                    c.turn,
                    c.tool,
                    c.arguments.to_string(),
                    c.valid,
                    c.ok,
                    c.validate_ms,
                    c.ms,
                    c.output_bytes,
                    c.error
                ])?;
            }
            let mut check = tx.prepare_cached("INSERT INTO checks(run_id,idx,kind,label,passed,failure_class) VALUES (?,?,?,?,?,?)")?;
            for (i, c) in out.checks.iter().enumerate() {
                check.execute(params![
                    id,
                    i,
                    c.kind,
                    c.label,
                    c.passed,
                    c.class.as_ref().map(tag)
                ])?;
            }
        }
        samples(&tx, SampleOwner::Run(id), r)?;
        tx.commit()?;
        Ok(())
    }
}
