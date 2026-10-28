"""The tools a custom task runs against: A2E's own tools, BFCL's simulated classes, or live web search."""
from __future__ import annotations

from pathlib import Path

from a2e.tasks.custom import FAMILY_ENV, FAMILY_TOOLS, ChainTask
from a2e.tools.base import Registry, Tool
from a2e.tools.builtin import build_registry, make_corpus

FINAL = Tool("final_answer", "Give the final answer and finish.",
             {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
             lambda answer: answer)


def family_core_specs(family: str) -> list[dict]:
    """Specs of the tools a family shows the model, in FAMILY_TOOLS order (no live instances needed)."""
    from a2e.bfcl.convert import tool_spec
    from a2e.tools.bfcl_tools import class_docs
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        builtin = {s["name"]: s for s in build_registry(Path(td)).specs()}
    bfcl = {d["name"]: tool_spec(d) for c in ("GorillaFileSystem", "MathAPI", "WebSearchAPI") for d in class_docs(c)}
    return [builtin.get(n) or bfcl[n] for n in FAMILY_TOOLS[family]]


def build_task_registry(task: ChainTask, workdir: Path, seed: int = 0, web=None) -> Registry:
    """A fresh registry for one task. Builtin families get their own note folder and document collection."""
    kind = FAMILY_ENV[task.family]
    if kind == "builtin":
        make_corpus(workdir, n=30, seed=seed)
        return build_registry(workdir)
    if kind == "bfcl":
        from a2e.tools.bfcl_tools import bfcl_registry
        return bfcl_registry(task.env["classes"], task.env.get("config"), extra=[FINAL], cwd=task.env.get("cwd"))
    from a2e.tools.web import WebSearch, web_tools
    from a2e.settings import WebSettings
    ws = WebSearch(web or WebSettings())
    notes = build_registry(workdir)
    reg = Registry(web_tools(ws) + [notes.tools["note_write"], notes.tools["note_list"], FINAL])
    reg.web = ws
    return reg
