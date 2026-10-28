from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from a2e.agent.parse import ToolCall


def norm(v: Any) -> Any:
    """Normalise for comparison: strings are case/space-insensitive, numbers compare by value."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v.strip().casefold())
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


@dataclass
class ExpectedCall:
    name: str
    args: dict | Callable[[list[str]], dict]  # callable gets the outputs of earlier tool calls
    compare_keys: list[str] | None = None  # None = every expected key; else only these (free-text args)
    squash_ws: bool = False  # drop all whitespace before comparing (calculator expressions)
    rel_tol: float | None = None  # numbers copied from earlier tool output may be rounded by the model
    check: Callable[[dict, list[str]], bool] | None = None  # extra rule on (actual args, earlier tool outputs)

    def resolve(self, results: list[str]) -> dict:
        return self.args(results) if callable(self.args) else self.args


@dataclass
class StepScore:
    name_ok: bool = False
    args_ok: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.name_ok and self.args_ok


@dataclass
class ChainScore:
    steps: list[StepScore] = field(default_factory=list)
    finished: bool = False  # reached final_answer
    state_ok: bool | None = None  # final tool state matches the oracle's (families that define a state check)

    @property
    def success(self) -> bool:
        return self.finished and all(s.ok for s in self.steps) and self.state_ok is not False

    @property
    def n_correct(self) -> int:
        return sum(s.ok for s in self.steps)


def score_call(exp: ExpectedCall, results: list[str], actual: ToolCall) -> StepScore:
    try:
        want = exp.resolve(results)
    except Exception as e:  # noqa: BLE001 - earlier step failed so this one cannot be derived
        return StepScore(note=f"unresolvable: {type(e).__name__}")
    if actual.name != exp.name:
        return StepScore(note=f"wrong tool {actual.name!r}")
    if not isinstance(actual.arguments, dict):
        return StepScore(name_ok=True, note="arguments not an object")
    keys = exp.compare_keys if exp.compare_keys is not None else list(want)
    for k in keys:
        a, w = actual.arguments.get(k), want.get(k)
        if exp.squash_ws and isinstance(a, str) and isinstance(w, str):
            a, w = re.sub(r"\s+", "", a), re.sub(r"\s+", "", w)
        if exp.rel_tol is not None and _numbers(a, w):
            if not math.isclose(float(a), float(w), rel_tol=exp.rel_tol, abs_tol=exp.rel_tol):
                return StepScore(name_ok=True, note=f"arg {k!r}: got {a!r}, want {w!r}")
            continue
        if norm(a) != norm(w):
            return StepScore(name_ok=True, note=f"arg {k!r}: got {a!r}, want {w!r}")
    if exp.check is not None:
        try:
            if not exp.check(actual.arguments, results):
                return StepScore(name_ok=True, note="check failed")
        except Exception as e:  # noqa: BLE001 - a malformed argument or earlier result fails the rule
            return StepScore(name_ok=True, note=f"check failed: {type(e).__name__}")
    return StepScore(True, True)


def _numbers(a, w) -> bool:
    try:
        float(a), float(w)
    except (TypeError, ValueError):
        return False
    return not isinstance(a, bool) and not isinstance(w, bool)
