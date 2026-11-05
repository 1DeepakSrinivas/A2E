from __future__ import annotations

from a2e.agent.grammar import call_schema
from a2e.agent.harness import MAX_RETRIES, Harness, feedback_message, retry_reminder
from a2e.agent.loop import TurnRecord, model_step
from a2e.agent.prompt import bfcl_system_message, render_tools
from a2e.bfcl.convert import tool_spec
from a2e.bfcl.load import Case
from a2e.bfcl.score import CaseResult, score_single


def run_single_turn_turns(client, case: Case, n_predict: int = 256, cache_prompt: bool = True,
                          grammar: bool = False, count_tokens=None,
                          harness: Harness | None = None) -> tuple[CaseResult, list[TurnRecord]]:
    """One answer per case, with every model call it took. No tool execution, so validate/tool-wait are zero.

    With grammar on, the schema allows an empty array so the model can still decline (irrelevance cases).
    With `recovery` set to feedback or retry, an unreadable reply is asked again, at most MAX_RETRIES more times.
    The last reply is the one that is scored.
    """
    h = harness or Harness()
    specs = [tool_spec(d) for d in case.functions]
    messages = [bfcl_system_message(specs, h.prompt_format)] + [m for turn in case.turns[:1] for m in turn]
    schema = call_schema(specs, max_calls=8, min_calls=0) if grammar else None
    turns: list[TurnRecord] = []
    remind = False
    for attempt in range(1 + (MAX_RETRIES if h.recovery != "none" else 0)):
        shown = messages + [retry_reminder()] if remind else messages
        turn, stamps = model_step(client, shown, attempt, n_predict, cache_prompt, schema, allow_empty=True)
        turns.append(turn)
        if stamps is None:
            return CaseResult(case.id, case.category, False, "a2e:server_error", turn.error), turns
        t0, t1, t2, t3 = stamps
        turn.finalize(t0, t1, t2, t3, t3, t3)
        if not turn.parse_error or h.recovery == "none":
            break
        if h.recovery == "feedback":
            messages = messages + [{"role": "assistant", "content": turn.text}, feedback_message(turn.parse_error)]
        else:
            remind = True
    res = score_single(case, turn.calls, turn.parse_error)
    res.extra["schema_tokens"] = count_tokens(render_tools(specs)) if count_tokens else None
    if len(turns) > 1:
        res.extra["recovery_attempts"] = len(turns) - 1
    return res, turns


def run_single_turn(client, case: Case, n_predict: int = 256, cache_prompt: bool = True,
                    grammar: bool = False, count_tokens=None,
                    harness: Harness | None = None) -> tuple[CaseResult, TurnRecord]:
    """Like run_single_turn_turns but returns only the last model call."""
    res, turns = run_single_turn_turns(client, case, n_predict, cache_prompt, grammar, count_tokens, harness)
    return res, turns[-1]
