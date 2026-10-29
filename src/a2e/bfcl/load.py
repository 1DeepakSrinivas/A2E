from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from a2e.bfcl.paths import DATA

AST_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple")
# expected behaviour for the no-call tests: irrelevance must NOT call, relevance MUST call
RELEVANCE_CATEGORIES = {"irrelevance": False, "live_irrelevance": False, "live_relevance": True}
MULTI_TURN_CATEGORIES = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param")
ALL_CATEGORIES = AST_CATEGORIES + tuple(RELEVANCE_CATEGORIES) + MULTI_TURN_CATEGORIES

# class name -> function doc file, copied from upstream constants (kept in sync by importing it lazily)


@dataclass
class Case:
    id: str
    category: str
    turns: list[list[dict]]  # per user turn, the messages to append
    functions: list[dict]  # original BFCL function docs (the ast checker needs these, not converted ones)
    ground_truth: list | None = None
    entry: dict = field(default_factory=dict)  # raw upstream entry (multi-turn checker reads it)
    held_out: dict[int, list[dict]] = field(default_factory=dict)  # turn index -> docs revealed at that turn

    @property
    def is_multi_turn(self) -> bool:
        return self.category in MULTI_TURN_CATEGORIES

    @property
    def expects_call(self) -> bool | None:
        """For relevance categories: True/False; None otherwise."""
        return RELEVANCE_CATEGORIES.get(self.category)

    @property
    def stratum(self) -> str:
        """Subsampling stratum: turn count for multi-turn, schema-size bucket (function count) otherwise."""
        if self.is_multi_turn:
            return f"turns={len(self.turns)}"
        n = len(self.functions)
        return "fn=1" if n == 1 else "fn=2-3" if n <= 3 else "fn=4+"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _func_docs(class_name: str) -> list[dict]:
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    return _read_jsonl(DATA / "multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name])


def load_cases(category: str) -> list[Case]:
    if category not in ALL_CATEGORIES:
        raise ValueError(f"unknown category {category!r}; choose from {ALL_CATEGORIES}")
    rows = _read_jsonl(DATA / f"BFCL_v4_{category}.json")
    gt_path = DATA / "possible_answer" / f"BFCL_v4_{category}.json"
    truth = {r["id"]: r["ground_truth"] for r in _read_jsonl(gt_path)} if gt_path.exists() else {}
    cases = []
    for r in rows:
        functions, held = list(r.get("function", [])), {}
        if category in MULTI_TURN_CATEGORIES:  # mirrors upstream populate_test_cases_with_predefined_functions
            functions = [d for c in r["involved_classes"] for d in _func_docs(c)]
            for turn, names in r.get("missed_function", {}).items():
                held[int(turn)] = []
                for name in names:
                    for i, doc in enumerate(functions):
                        if doc["name"] == name:
                            held[int(turn)].append(functions.pop(i))
                            break
        cases.append(Case(r["id"], category, r["question"], functions, truth.get(r["id"]), r, held))
    return cases


def stratified_sample(cases: list[Case], n: int, seed: int = 0) -> list[Case]:
    """n cases, proportional across strata (largest remainder), fixed seed, original file order kept.

    The same (category, n, seed) gives the same ids on every machine, so Pi and Mac run identical cases.
    """
    if n >= len(cases):
        return list(cases)
    groups: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        groups[c.stratum].append(c)
    exact = {k: n * len(v) / len(cases) for k, v in groups.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    chosen: set[str] = set()
    for k, v in groups.items():
        rng = random.Random(f"{seed}/{v[0].category}/{k}")
        chosen.update(c.id for c in rng.sample(v, min(alloc[k], len(v))))
    return [c for c in cases if c.id in chosen]


def load_subset(category: str, n: int | None = None, seed: int = 0) -> list[Case]:
    cases = load_cases(category)
    return cases if n is None else stratified_sample(cases, n, seed)
