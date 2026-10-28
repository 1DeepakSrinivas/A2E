"""Real local tools: calculator, note file ops, document search over a local corpus, reminders."""
from __future__ import annotations

import ast
import json
import operator
import random
import re
from pathlib import Path

from a2e.tools.base import Registry, Tool

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calc(expression: str) -> str:
    """Arithmetic only: numbers, + - * / // % ** and parentheses. No names, calls or attributes."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")

    v = ev(ast.parse(expression.strip(), mode="eval"))
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "_", title.lower()).strip("_")[:64]
    if not s:
        raise ValueError("invalid title")
    return s


_SYL = ["ka", "re", "vo", "lin", "dar", "mu", "tel", "or", "san", "bri", "quel", "ha", "zi", "fen", "lo"]
_EXPORTS = ["timber", "salt", "copper", "wool", "grain", "glass", "tea"]


def make_corpus(root: Path, n: int = 30, seed: int = 0) -> list[dict]:
    """Deterministic synthetic settlement reports written as real files. Returns metadata for oracles."""
    rng = random.Random(seed)
    (root / "corpus").mkdir(parents=True, exist_ok=True)
    meta, names = [], set()
    while len(meta) < n:
        name = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))).capitalize()
        if name in names:
            continue
        names.add(name)
        i = len(meta)
        d = {"doc_id": f"doc_{i:03d}", "name": name, "population": rng.randint(1000, 99999),
             "founded": rng.randint(1100, 1950), "area": rng.randint(5, 900), "export": rng.choice(_EXPORTS)}
        text = (f"{name} is a settlement founded in {d['founded']}. Its population is {d['population']}. "
                f"It covers {d['area']} square kilometers. Its main export is {d['export']}.")
        (root / "corpus" / f"{d['doc_id']}.txt").write_text(f"{name}\n{text}")
        meta.append(d)
    return meta


def build_registry(workdir: Path) -> Registry:
    workdir = Path(workdir)
    notes, corpus = workdir / "notes", workdir / "corpus"
    notes.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)

    def note_write(title, body="", tags="", priority=0, due=""):
        (notes / f"{slug(title)}.json").write_text(json.dumps(
            {"title": title, "body": body, "tags": tags, "priority": priority, "due": due}))
        return f"saved note {title!r}"

    def note_read(title):
        return (notes / f"{slug(title)}.json").read_text()

    def note_list():
        return json.dumps(sorted(json.loads(p.read_text())["title"] for p in notes.glob("*.json")))

    def search_docs(query, top_k=3):
        words = re.findall(r"[a-z0-9]+", query.lower())
        hits = []
        for p in sorted(corpus.glob("doc_*.txt")):
            title, _, text = p.read_text().partition("\n")
            low = text.lower()
            score = sum(low.count(w) + 3 * (w == title.lower()) for w in words)
            if score:
                hits.append({"doc_id": p.stem, "title": title, "score": score})
        hits.sort(key=lambda h: (-h["score"], h["doc_id"]))
        return json.dumps(hits[:top_k])

    def doc_read(doc_id):
        if not re.fullmatch(r"doc_\d{3}", doc_id):
            raise ValueError("bad doc_id")
        return (corpus / f"{doc_id}.txt").read_text()

    def set_reminder(text, minutes):
        if minutes < 1:
            raise ValueError("minutes must be >= 1")
        f = workdir / "reminders.json"
        items = json.loads(f.read_text()) if f.exists() else []
        items.append({"text": text, "minutes": minutes})
        f.write_text(json.dumps(items))
        return f"reminder #{len(items)} set for +{minutes} min"

    S, I = {"type": "string"}, {"type": "integer"}

    def obj(props, req=()):
        return {"type": "object", "properties": props, "required": list(req)}

    return Registry([
        Tool("calculator", "Evaluate an arithmetic expression.", obj({"expression": S}, ["expression"]), safe_calc),
        Tool("note_write", "Save a note.", obj({"title": S, "body": S, "tags": S, "priority": I, "due": S},
                                               ["title"]), note_write),
        Tool("note_read", "Read a note by title.", obj({"title": S}, ["title"]), note_read),
        Tool("note_list", "List all note titles.", obj({}), note_list),
        Tool("search_docs", "Search documents by keywords.", obj({"query": S, "top_k": I}, ["query"]), search_docs),
        Tool("doc_read", "Read a document by id.", obj({"doc_id": S}, ["doc_id"]), doc_read),
        Tool("set_reminder", "Set a reminder.", obj({"text": S, "minutes": I}, ["text", "minutes"]), set_reminder),
        Tool("final_answer", "Give the final answer and finish.", obj({"answer": S}, ["answer"]),
             lambda answer: answer),
    ])
