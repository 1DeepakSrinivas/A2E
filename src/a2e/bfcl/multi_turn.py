from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from a2e.agent.grammar import call_schema
from a2e.agent.loop import TurnRecord, model_step
from a2e.agent.prompt import bfcl_system_message, render_tools, tool_response_message
from a2e.bfcl.convert import call_string, tool_spec
from a2e.bfcl.load import Case
from a2e.bfcl.score import CaseResult, release_instances, score_multi
from a2e.tools.base import ToolResult

MAX_STEPS = 20  # upstream MAXIMUM_STEP_LIMIT per user turn


@dataclass
class MultiTurnTrace:
    turns: list[TurnRecord] = field(default_factory=list)  # every model call, in order
    user_turn_of: list[int] = field(default_factory=list)  # user-turn index of each entry in `turns`
    per_turn_steps: list[list[list[str]]] = field(default_factory=list)
    status: str = "done"  # done | force_terminated | server_error
    error: str | None = None
    unseen_calls: list[str] = field(default_factory=list)  # held-out functions called before being revealed


def run_multi_turn(client, case: Case, n_predict: int = 256, cache_prompt: bool = True,
                   grammar: bool = False, max_steps: int = MAX_STEPS) -> MultiTurnTrace:
    """Drive one BFCL multi-turn case against BFCL's own local backends (real tool execution, timed as tool-wait).

    Per user turn the model is called repeatedly until it answers with no call (or max_steps is hit, which
    force-terminates the case as upstream does). Held-out functions are revealed at their turn by
    appending a user message with their docs plus upstream's default sentence (as upstream's prompting mode
    does), so the prompt only ever grows by appending and the prefix cache keeps working.

    Like the official runner, any method of the involved classes may be executed, including held-out ones
    before they are revealed (a few upstream ground truths need that); such calls are listed in
    `unseen_calls` so analysis can separate them.
    """
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    functions = list(case.functions)
    runnable = {d["name"] for d in functions} | {d["name"] for ds in case.held_out.values() for d in ds}
    specs = [tool_spec(d) for d in functions]
    messages = [bfcl_system_message(specs)]
    trace = MultiTurnTrace()
    run_name = f"a2erun{uuid.uuid4().hex[:10]}"
    schema_cache: dict[tuple, dict] = {}
    try:
        for t, user_msgs in enumerate(case.turns):
            if t in case.held_out:  # upstream: the reveal turn has no user message of its own
                functions += case.held_out[t]
                specs = [tool_spec(d) for d in functions]
                user_msgs = [{"role": "user", "content": render_tools([tool_spec(d) for d in case.held_out[t]])
                              + "\n" + DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
            messages += user_msgs
            visible = {d["name"] for d in functions}
            names = tuple(sorted(visible))
            schema = None
            if grammar:
                schema = schema_cache.setdefault(names, call_schema(specs, max_calls=8, min_calls=0))
            steps: list[list[str]] = []
            trace.per_turn_steps.append(steps)
            for step in range(max_steps + 1):
                if step == max_steps:
                    trace.status = "force_terminated"
                    return trace
                turn, stamps = model_step(client, messages, step, n_predict, cache_prompt, schema, allow_empty=True)
                trace.turns.append(turn)
                trace.user_turn_of.append(t)
                if stamps is None:
                    trace.status, trace.error = "server_error", turn.error
                    return trace
                t0, t1, t2, t3 = stamps
                strings, errors = [], []
                for c in turn.calls:
                    try:
                        strings.append(call_string(c.name, c.arguments, runnable))
                        errors.append(None)
                        if c.name not in visible:
                            trace.unseen_calls.append(c.name)
                    except ValueError as e:
                        errors.append(str(e))
                turn.validation_errors = errors
                t4 = time.perf_counter()
                executed = iter(execute_multi_turn_func_call(
                    strings, case.entry["initial_config"], case.entry["involved_classes"], run_name, case.id,
                    long_context=False, is_evaL_run=False)[0] if strings else [])
                turn.results = [ToolResult(False, f"error: invalid call: {e}") if e
                                else ToolResult(not str(r).startswith("Error during execution"), str(r))
                                for e, r in ((e, None if e else next(executed)) for e in errors)]
                turn.finalize(t0, t1, t2, t3, t4, time.perf_counter())
                if not turn.calls:  # model answered without a call: this user turn is over
                    messages.append({"role": "assistant", "content": turn.text})
                    break
                if strings:
                    steps.append(strings)
                messages += [{"role": "assistant", "content": turn.text},
                             tool_response_message(turn.calls, turn.results)]
        return trace
    finally:
        release_instances(run_name)


def score_trace(case: Case, trace: MultiTurnTrace) -> CaseResult:
    if trace.status == "server_error":
        return CaseResult(case.id, case.category, False, "a2e:server_error", trace.error)
    res = score_multi(case, trace.per_turn_steps, forced_quit=trace.status == "force_terminated")
    res.extra["unseen_calls"] = list(trace.unseen_calls)
    return res
