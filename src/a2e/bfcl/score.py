from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from a2e.agent.parse import ToolCall
from a2e.bfcl.convert import to_ast_output
from a2e.bfcl.load import AST_CATEGORIES, MULTI_TURN_CATEGORIES, RELEVANCE_CATEGORIES, Case


@dataclass
class CaseResult:
    case_id: str
    category: str
    valid: bool
    error_type: str | None = None
    detail: Any = None
    extra: dict = field(default_factory=dict)


def score_single(case: Case, calls: list[ToolCall], parse_error: str | None) -> CaseResult:
    """Single-turn scoring: the official ast_checker, or BFCL's relevance rule (a decoded, non-empty call)."""
    has_call = bool(calls) and parse_error is None
    expect = RELEVANCE_CATEGORIES.get(case.category)
    if expect is not None:
        ok = has_call == expect
        kind = "relevance_error:no_call" if expect else "irrelevance_error:call_made"
        return CaseResult(case.id, case.category, ok, None if ok else kind)
    if not has_call:
        return CaseResult(case.id, case.category, False, f"a2e:no_call:{parse_error or 'empty'}")
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    try:
        res = ast_checker(case.functions, to_ast_output(calls), case.ground_truth, Language.PYTHON,
                          case.category, "a2e")
    except Exception as e:  # noqa: BLE001 - malformed model output can trip the checker; that is a wrong answer
        return CaseResult(case.id, case.category, False, f"a2e:checker_exception:{type(e).__name__}", str(e))
    return CaseResult(case.id, case.category, bool(res["valid"]), res.get("error_type"), res.get("error"))


def release_instances(run_name: str) -> None:
    """Drop backend instances the upstream executor caches in module globals under this run name."""
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mtu
    for k in [k for k in vars(mtu) if k.startswith(run_name)]:
        delattr(mtu, k)


def score_multi(case: Case, per_turn_steps: list[list[list[str]]], forced_quit: bool = False) -> CaseResult:
    """Multi-turn scoring with the official multi_turn_checker.

    per_turn_steps[turn][step] is the list of call strings executed in that step (empty steps omitted).
    A fresh run name keeps upstream's cached instances from leaking state between scorings.
    Also reports the (unused upstream) irrelevance check for empty-ground-truth turns as extra.irrelevance_valid.
    """
    gt = case.ground_truth
    if forced_quit or len(per_turn_steps) != len(gt):
        return CaseResult(case.id, case.category, False, "multi_turn:force_terminated")
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
        multi_turn_irrelevance_checker,
    )
    run_name = f"a2escore{uuid.uuid4().hex[:10]}"
    try:
        res = multi_turn_checker(per_turn_steps, gt, case.entry, case.category, run_name)
        irr = multi_turn_irrelevance_checker(per_turn_steps, gt)["valid"]
    except Exception as e:  # noqa: BLE001
        return CaseResult(case.id, case.category, False, f"a2e:checker_exception:{type(e).__name__}", str(e))
    finally:
        release_instances(run_name)
    return CaseResult(case.id, case.category, bool(res["valid"]), res.get("error_type"),
                      res.get("error_message"), {"irrelevance_valid": irr})
