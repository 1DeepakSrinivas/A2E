"""Custom tool-call chains over the real local tools, with an oracle for every step.

Each task fixes its `family` (and so its tool set), the number of expected calls (`depth`), and for note
tasks the number of arguments per call. Later steps may depend on earlier tool output (e.g. the doc id
returned by search), which is why expected args can be callables over the executed results.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from a2e.tasks.scoring import ExpectedCall

# Core tools each family shows the model (the rest of the tool list is distractors, see agent/schema_gen.py).
FAMILY_TOOLS = {
    "calc": ["calculator", "final_answer"],
    "notes": ["note_write", "note_list", "final_answer"],
    "retrieval": ["search_docs", "doc_read", "calculator", "note_write", "note_list", "final_answer"],
    "fs_search": ["find", "cd", "cat", "grep", "wc", "sort", "final_answer"],
    "fs_organize": ["mkdir", "mv", "cd", "ls", "final_answer"],
    "math": ["mean", "standard_deviation", "round_number", "max_value", "percentage", "final_answer"],
    "fs_math": ["wc", "add", "percentage", "round_number", "final_answer"],
    "web": ["search_engine_query", "fetch_url_content", "note_write", "note_list", "final_answer"],
}
# Allowed chain depth per family (number of expected tool calls before final_answer).
FAMILY_DEPTH = {"calc": (1, 5), "notes": (1, 5), "retrieval": (2, 5), "fs_search": (2, 5), "fs_organize": (2, 5),
                "math": (1, 5), "fs_math": (3, 5), "web": (2, 4)}
# Where the tools come from: A2E's own tools, BFCL's simulated classes, or live web search (+ A2E notes).
FAMILY_ENV = {"calc": "builtin", "notes": "builtin", "retrieval": "builtin", "fs_search": "bfcl",
              "fs_organize": "bfcl", "math": "bfcl", "fs_math": "bfcl", "web": "web"}
FAMILY_CLASSES = {"fs_search": ["GorillaFileSystem"], "fs_organize": ["GorillaFileSystem"], "math": ["MathAPI"],
                  "fs_math": ["GorillaFileSystem", "MathAPI"]}
NOTE_FIELDS = ["title", "body", "tags", "priority", "due"]
_WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "gamma", "harbor"]


@dataclass
class ChainTask:
    task_id: str
    family: str
    prompt: str
    steps: list[ExpectedCall]
    depth: int
    env: dict = field(default_factory=dict)  # scenario id, BFCL config and starting folder, for BFCL/web families
    state_check: Callable | None = None  # fn(registry) -> bool: final tool state equals the oracle's

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


# ------------------------------------------------------------------ BFCL file system and math families
@lru_cache(maxsize=1)
def fs_scenarios() -> tuple[dict, ...]:
    """Small BFCL file-system starting states, cherry-picked by scripts/curate_fs_scenarios.py."""
    data = json.loads(resources.files("a2e.data").joinpath("fs_scenarios.json").read_text())
    return tuple(data["scenarios"])


def _file_content(sc: dict, path: list[str], name: str) -> str:
    node = sc["config"]["root"][sc["root"]]
    for d in path:
        node = node["contents"][d]
    return node["contents"][name]["content"]


def _dir_names(sc: dict, path: list[str]) -> set[str]:
    node = sc["config"]["root"][sc["root"]]
    for d in path:
        node = node["contents"][d]
    return set(node["contents"])


def _env(sc: dict, family: str, cwd: list[str]) -> dict:
    return {"scenario": sc["id"], "config": sc["config"], "cwd": cwd, "classes": FAMILY_CLASSES[family]}


def _join(parts: list[str]) -> str:
    return ", then ".join(parts)


def _result(r: str, key: str = "result"):
    return json.loads(r)[key]


def _fs_search(rng: random.Random, tid: str, depth: int) -> ChainTask:
    cands = [(sc, f) for sc in fs_scenarios() for f in sc["files"] if len(f["path"]) <= depth - 2]
    sc, f = rng.choice(cands)
    name, k = f["name"], len(f["path"])
    words = [w for w in re.findall(r"[A-Za-z]{4,}", _file_content(sc, f["path"], name))] or ["the"]
    word = rng.choice(words)
    steps = [ExpectedCall("find", {"name": name}, compare_keys=["name"],
                          check=lambda a, r: a.get("path", ".") in (".", "./", "/", ""))]
    steps += [ExpectedCall("cd", {"folder": d}) for d in f["path"]]
    ops = [(ExpectedCall("cat", {"file_name": name}), f"show the contents of '{name}'"),
           (ExpectedCall("grep", {"file_name": name, "pattern": word}), f"show the lines that contain '{word}'"),
           (ExpectedCall("wc", {"file_name": name, "mode": "w"}), "count its words"),
           (ExpectedCall("sort", {"file_name": name}), "sort its lines")][: depth - 1 - k]
    steps += [e for e, _ in ops]
    text = [f"find the file named '{name}' starting from the current folder"]
    if k:
        text.append("go into the folder it is in, one folder level per call")
    text += [t for _, t in ops]
    prompt = "Using the file system tools, " + _join(text) + ". Make one tool call per turn, then give the final answer."
    return ChainTask(tid, "fs_search", prompt, steps, depth, _env(sc, "fs_search", []))


def _fs_organize(rng: random.Random, tid: str, depth: int) -> ChainTask:
    cands = [(sc, f) for sc in fs_scenarios() for f in sc["files"]]
    sc, f = rng.choice(cands)
    here = _dir_names(sc, f["path"])
    new = rng.choice([d for d in ("archive", "backup", "sorted", "reviewed", "old_files") if d not in here])
    name = f["name"]
    renamed = f"final_{name}"
    plan = [(ExpectedCall("mkdir", {"dir_name": new}), f"create a folder named '{new}'"),
            (ExpectedCall("mv", {"source": name, "destination": new}), f"move '{name}' into '{new}'"),
            (ExpectedCall("cd", {"folder": new}), f"go into '{new}'"),
            (ExpectedCall("ls", {}, compare_keys=[]), "list what is there"),
            (ExpectedCall("mv", {"source": name, "destination": renamed}), f"rename '{name}' to '{renamed}'")][:depth]
    steps = [e for e, _ in plan]
    folder = f["path"][-1] if f["path"] else sc["root"]
    prompt = (f"You are in the folder '{folder}'. Using the file system tools, " + _join([t for _, t in plan])
              + ". Make one tool call per turn, then give the final answer.")
    env = _env(sc, "fs_organize", list(f["path"]))

    def state_check(reg, env=env, steps=steps) -> bool:
        from a2e.tools.bfcl_tools import bfcl_registry
        oracle = bfcl_registry(["GorillaFileSystem"], env["config"], cwd=env["cwd"])
        for e in steps:
            oracle.execute(e.name, e.resolve([]))
        return oracle.instances["GorillaFileSystem"].root == reg.instances["GorillaFileSystem"].root

    return ChainTask(tid, "fs_organize", prompt, steps, depth, env, state_check)


def _math(rng: random.Random, tid: str, depth: int) -> ChainTask:
    nums = rng.sample(range(10, 100), 5)
    tol = 1e-3
    plan = [(ExpectedCall("mean", {"numbers": nums}), "compute their mean"),
            (ExpectedCall("standard_deviation", {"numbers": nums}), "compute their standard deviation"),
            (ExpectedCall("round_number", lambda r: {"number": _result(r[1]), "decimal_places": 2}, rel_tol=tol),
             "round the standard deviation to 2 decimal places"),
            (ExpectedCall("max_value", {"numbers": nums}), "find the largest number"),
            (ExpectedCall("percentage", lambda r: {"part": _result(r[0]), "whole": _result(r[3])}, rel_tol=tol),
             "work out what percentage the mean is of the largest number")][:depth]
    prompt = (f"For the numbers {nums}, using the math tools, " + _join([t for _, t in plan])
              + ". Make one tool call per turn, then give the final answer.")
    return ChainTask(tid, "math", prompt, [e for e, _ in plan], depth, {"classes": ["MathAPI"]})


def _fs_math(rng: random.Random, tid: str, depth: int) -> ChainTask:
    dirs = {}
    for sc in fs_scenarios():
        for f in sc["files"]:
            dirs.setdefault((sc["id"], tuple(f["path"])), (sc, []))[1].append(f)
    sc, files = rng.choice([v for v in dirs.values() if len(v[1]) >= 2])
    fa, fb = rng.sample(files, 2)
    tol = 1e-3
    counts = lambda r: (_result(r[0], "count"), _result(r[1], "count"))  # noqa: E731

    def add_ok(a, r):
        return sorted([float(a["a"]), float(a["b"])]) == sorted(float(x) for x in counts(r))

    plan = [(ExpectedCall("wc", {"file_name": fa["name"], "mode": "w"}), f"count the words in '{fa['name']}'"),
            (ExpectedCall("wc", {"file_name": fb["name"], "mode": "w"}), f"count the words in '{fb['name']}'"),
            (ExpectedCall("add", {}, compare_keys=[], check=add_ok), "add the two counts"),
            (ExpectedCall("percentage", lambda r: {"part": counts(r)[0], "whole": _result(r[2])}, rel_tol=tol),
             f"work out what percentage of the total comes from '{fa['name']}'"),
            (ExpectedCall("round_number", lambda r: {"number": _result(r[3]), "decimal_places": 1}, rel_tol=tol),
             "round that percentage to 1 decimal place")][:depth]
    folder = fa["path"][-1] if fa["path"] else sc["root"]
    prompt = (f"You are in the folder '{folder}'. Using the file system and math tools, " + _join([t for _, t in plan])
              + ". Make one tool call per turn, then give the final answer.")
    return ChainTask(tid, "fs_math", prompt, [e for e, _ in plan], depth, _env(sc, "fs_math", list(fa["path"])))


# ------------------------------------------------------------------ live web search
WEB_QUESTIONS = [
    "In which year was the Eiffel Tower completed?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the chemical symbol for gold?",
    "How many bones are in the adult human body?",
    "What is the capital city of Australia?",
    "Which planet in our solar system has the most moons?",
    "What is the boiling point of water at sea level in degrees Fahrenheit?",
    "Who painted the Mona Lisa?",
    "What is the longest river in South America?",
    "In which year did the first human land on the Moon?",
    "What is the largest ocean on Earth?",
    "Who developed the theory of general relativity?",
    "What is the tallest mountain in Africa?",
    "How many players are on the field for one team in a football (soccer) match?",
    "What is the hardest natural mineral?",
    "Which element has the atomic number 1?",
]


def _search_ok(a, r) -> bool:
    return isinstance(a.get("keywords"), str) and bool(a["keywords"].strip())


def _fetch_ok(a, r) -> bool:
    hits = json.loads(r[0])
    return isinstance(hits, list) and a.get("url") in {h.get("href") for h in hits}


def _web(rng: random.Random, tid: str, depth: int) -> ChainTask:
    q = rng.choice(WEB_QUESTIONS)
    plan = [(ExpectedCall("search_engine_query", {}, compare_keys=[], check=_search_ok), "search the web"),
            (ExpectedCall("fetch_url_content", {}, compare_keys=[], check=_fetch_ok), "open one of the results"),
            (ExpectedCall("note_write", {"title": "web-answer"}, compare_keys=["title"]),
             "save a note titled 'web-answer' with the answer"),
            (ExpectedCall("note_list", {}), "list your notes")][:depth]
    prompt = (f"Answer this question using web search: {q} " + _join([t for _, t in plan]).capitalize()
              + ". Make one tool call per turn, then give the final answer.")
    return ChainTask(tid, "web", prompt, [e for e, _ in plan], depth, {"question": q, "live": True})


def generate(family: str, n: int, depth: int, n_args: int = 2, seed: int = 0,
             corpus: list[dict] | None = None) -> list[ChainTask]:
    """n deterministic tasks. Same arguments always give the same tasks (web answers come from the live web)."""
    if family not in FAMILY_TOOLS:
        raise ValueError(f"unknown family {family!r}")
    lo, hi = FAMILY_DEPTH[family]
    if not lo <= depth <= hi:
        raise ValueError(f"{family} needs depth {lo}-{hi}")
    if family == "retrieval" and not corpus:
        raise ValueError("retrieval needs a corpus")
    if not 1 <= n_args <= 5:
        raise ValueError("n_args must be 1-5")
    make = {"calc": lambda r, t: _calc(r, t, depth), "notes": lambda r, t: _notes(r, t, depth, n_args),
            "retrieval": lambda r, t: _retrieval(r, t, depth, corpus), "fs_search": lambda r, t: _fs_search(r, t, depth),
            "fs_organize": lambda r, t: _fs_organize(r, t, depth), "math": lambda r, t: _math(r, t, depth),
            "fs_math": lambda r, t: _fs_math(r, t, depth), "web": lambda r, t: _web(r, t, depth)}[family]
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}/{family}/{depth}/{n_args}/{i}")
        out.append(make(rng, f"{family}-d{depth}-a{n_args}-{i}"))
    return out
