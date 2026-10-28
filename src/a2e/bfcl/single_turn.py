from __future__ import annotations

from a2e.agent.grammar import call_schema
from a2e.agent.loop import TurnRecord, model_step
from a2e.agent.prompt import bfcl_system_message, render_tools
from a2e.bfcl.convert import tool_spec
from a2e.bfcl.load import Case
from a2e.bfcl.score import CaseResult, score_single


def run_single_turn(client, case: Case, n_predict: int = 256, cache_prompt: bool = True,
                    grammar: bool = False, count_tokens=None) -> tuple[CaseResult, TurnRecord]:
    """One model call per case. No tool execution, so validate/tool-wait are zero for this arm.

    With grammar on, the schema allows an empty array so the model can still decline (irrelevance cases).
    """
    specs = [tool_spec(d) for d in case.functions]
    messages = [bfcl_system_message(specs)] + [m for turn in case.turns[:1] for m in turn]
    schema = call_schema(specs, max_calls=8, min_calls=0) if grammar else None
    turn, stamps = model_step(client, messages, 0, n_predict, cache_prompt, schema, allow_empty=True)
    if stamps is None:
        return CaseResult(case.id, case.category, False, "a2e:server_error", turn.error), turn
    t0, t1, t2, t3 = stamps
    turn.finalize(t0, t1, t2, t3, t3, t3)
    res = score_single(case, turn.calls, turn.parse_error)
    res.extra["schema_tokens"] = count_tokens(render_tools(specs)) if count_tokens else None
    return res, turn
