"""Custom tool-call chains over the real local tools, with an oracle for every step.

Each task fixes its `family` (and so its tool set), the number of expected calls (`depth`), and for note
tasks the number of arguments per call. Later steps may depend on earlier tool output (e.g. the doc id
returned by search), which is why expected args can be callables over the executed results.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from a2e.tasks.scoring import ExpectedCall

FAMILY_TOOLS = {
    "calc": ["calculator", "final_answer"],
    "notes": ["note_write", "note_list", "final_answer"],
    "retrieval": ["search_docs", "doc_read", "calculator", "note_write", "note_list", "final_answer"],
}
NOTE_FIELDS = ["title", "body", "tags", "priority", "due"]
_WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "gamma", "harbor"]


@dataclass
class ChainTask:
    task_id: str
    family: str
    prompt: str
    steps: list[ExpectedCall]
    depth: int

    @property
    def tool_names(self) -> list[str]:
        return FAMILY_TOOLS[self.family]


def _calc(rng: random.Random, tid: str, depth: int) -> ChainTask:
    exprs = [f"{rng.randint(2, 99)} {rng.choice('+-*')} {rng.randint(2, 99)}" for _ in range(depth)]
    prompt = ("Use the calculator tool once for each of these expressions, in order, one call per turn, "
              "then give the final answer: " + "; ".join(exprs))
    return ChainTask(tid, "calc", prompt, [ExpectedCall("calculator", {"expression": e}, squash_ws=True)
                                           for e in exprs], depth)


def _notes(rng: random.Random, tid: str, depth: int, n_args: int) -> ChainTask:
    n_writes = max(depth - 1, 1) if depth > 1 else 1
    steps, parts = [], []
    for k in range(n_writes):
        vals = {"title": f"{rng.choice(_WORDS)}{rng.randint(10, 99)}", "body": f"{rng.choice(_WORDS)} {rng.choice(_WORDS)}",
                "tags": f"{rng.choice(_WORDS)},{rng.choice(_WORDS)}", "priority": rng.randint(1, 5),
                "due": f"2026-{rng.randint(1, 12):02d}-{rng.randint(10, 28)}"}
        args = {f: vals[f] for f in NOTE_FIELDS[:n_args]}
        steps.append(ExpectedCall("note_write", args))
        parts.append("note_write with " + ", ".join(f"{f}={v!r}" for f, v in args.items()))
    if depth > 1:
        steps.append(ExpectedCall("note_list", {}))
        parts.append("note_list")
    prompt = "Do these in order, one tool call per turn, then give the final answer: " + "; ".join(parts)
    return ChainTask(tid, "notes", prompt, steps, depth)


def _retrieval(rng: random.Random, tid: str, depth: int, corpus: list[dict]) -> ChainTask:
    d = rng.choice(corpus)
    pop = lambda r: int(re.search(r"population is (\d+)", r[1]).group(1))  # noqa: E731
    steps = [
        ExpectedCall("search_docs", {"query": d["name"]}, compare_keys=["query"]),
        ExpectedCall("doc_read", lambda r: {"doc_id": json.loads(r[0])[0]["doc_id"]}),
        ExpectedCall("calculator", lambda r: {"expression": f"{pop(r)}*2"}, squash_ws=True),
        ExpectedCall("note_write", lambda r: {"title": d["name"]}, compare_keys=["title"]),
        ExpectedCall("note_list", {}),
    ][:depth]
    todo = ["search the documents for " + d["name"], "read the best matching document"]
    todo += ["calculate double its population with the calculator",
             f"save a note titled {d['name']}", "list all notes"]
    prompt = ("Do these in order, one tool call per turn, then give the final answer: "
              + "; ".join(todo[:depth]))
    return ChainTask(tid, "retrieval", prompt, steps, depth)


def generate(family: str, n: int, depth: int, n_args: int = 2, seed: int = 0,
             corpus: list[dict] | None = None) -> list[ChainTask]:
    """n deterministic tasks. Same arguments always give the same tasks."""
    if family == "retrieval":
        if not 2 <= depth <= 5 or not corpus:
            raise ValueError("retrieval needs depth 2-5 and a corpus")
    elif not 1 <= depth <= 5:
        raise ValueError("depth must be 1-5")
    if not 1 <= n_args <= 5:
        raise ValueError("n_args must be 1-5")
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}/{family}/{depth}/{n_args}/{i}")
        tid = f"{family}-d{depth}-a{n_args}-{i}"
        out.append(_calc(rng, tid, depth) if family == "calc" else
                   _notes(rng, tid, depth, n_args) if family == "notes" else
                   _retrieval(rng, tid, depth, corpus))
    return out
