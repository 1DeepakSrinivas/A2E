from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from a2e.agent.loop import TurnRecord
from a2e.agent.prompt import render_tools
from a2e.bfcl.convert import tool_spec
from a2e.bfcl.load import Case, load_cases, load_subset
from a2e.bfcl.multi_turn import MAX_STEPS, run_multi_turn, score_trace
from a2e.bfcl.replay import ReplayClient
from a2e.bfcl.score import CaseResult
from a2e.bfcl.single_turn import run_single_turn
from a2e.store import Store


@dataclass
class CaseOutcome:
    case: Case
    result: CaseResult
    turns: list[TurnRecord]
    texts: list[str]  # raw model output per call, in order (what rescoring replays)
    status: str = "done"
    user_turn_of: list[int] = field(default_factory=list)


def safe_token_counter(client) -> Callable[[str], int] | None:
    """client.tokenize, or None if the server has no working /tokenize (schema_tokens is then left empty)."""
    try:
        client.tokenize("probe")
    except Exception:  # noqa: BLE001 - optional feature
        return None
    return client.tokenize


def run_case(client, case: Case, *, grammar=False, cache_prompt=True, n_predict=256, max_steps=MAX_STEPS,
             count_tokens=None) -> CaseOutcome:
    if case.is_multi_turn:
        tr = run_multi_turn(client, case, n_predict, cache_prompt, grammar, max_steps)
        res = score_trace(case, tr)
        res.extra["schema_tokens"] = (count_tokens(render_tools([tool_spec(d) for d in case.functions]))
                                      if count_tokens else None)
        return CaseOutcome(case, res, tr.turns, [t.text for t in tr.turns], tr.status, tr.user_turn_of)
    res, turn = run_single_turn(client, case, n_predict, cache_prompt, grammar, count_tokens)
    return CaseOutcome(case, res, [turn], [turn.text], "server_error" if turn.error else "done")


def case_row(o: CaseOutcome, run_id: int) -> dict:
    return {"run_id": run_id, "case_id": o.case.id, "category": o.case.category, "valid": int(o.result.valid),
            "error_type": o.result.error_type, "status": o.status, "n_user_turns": len(o.case.turns),
            "n_model_calls": len(o.turns), "schema_tokens": o.result.extra.get("schema_tokens"),
            "wall_ms": sum(t.wall_ms for t in o.turns),
            "extra_json": json.dumps({k: v for k, v in o.result.extra.items() if k != "schema_tokens"})}


def raw_record(o: CaseOutcome, run_id: int) -> str:
    return json.dumps({"run_id": run_id, "case_id": o.case.id, "category": o.case.category, "status": o.status,
                       "texts": o.texts})


def run_category(client, category: str, store: Store, run_id: int, *, n: int | None = None, seed: int = 0,
                 raw_path: str | Path | None = None, **kw) -> list[CaseOutcome]:
    """Run (a fixed-seed subsample of) one category, writing `turns` and `cases` rows and raw outputs."""
    cases = load_subset(category, n, seed)  # validates the category before anything is written
    outcomes = []
    raw = open(raw_path, "a") if raw_path else None
    try:
        for case in cases:
            o = run_case(client, case, **kw)
            outcomes.append(o)
            for t in o.turns:
                store.insert("turns", t.row(run_id, case.id))
            store.insert("cases", case_row(o, run_id))
            if raw:
                raw.write(raw_record(o, run_id) + "\n")
                raw.flush()
    finally:
        if raw:
            raw.close()
    return outcomes


def rescore(raw_path: str | Path) -> list[CaseResult]:
    """Re-score recorded outputs with the vendored checkers (no model needed)."""
    cases: dict[str, dict[str, Case]] = {}
    out = []
    for line in Path(raw_path).read_text().splitlines():
        rec = json.loads(line)
        by_id = cases.setdefault(rec["category"], {c.id: c for c in load_cases(rec["category"])})
        case = by_id[rec["case_id"]]
        if rec["status"] == "server_error":
            out.append(CaseResult(case.id, case.category, False, "a2e:server_error"))
            continue
        out.append(run_case(ReplayClient(rec["texts"]), case).result)
    return out
