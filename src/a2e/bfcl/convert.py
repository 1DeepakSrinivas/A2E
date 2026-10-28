from __future__ import annotations

import keyword
import math
import re

from a2e.agent.grammar import SUPPORTED_KEYWORDS
from a2e.agent.parse import ToolCall

# BFCL docs use Python-ish type names; map them to JSON Schema. Unknown/`any` types become unconstrained.
_TYPE = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer",
         "string": "string", "boolean": "boolean", "array": "array", "tuple": "array"}
_IDENT = re.compile(r"^[A-Za-z_]\w*$")
MAX_CALL_CHARS = 20_000


def schema_from_bfcl(node):
    """BFCL parameter doc -> JSON Schema using only keywords llama-server's converter is assumed to support."""
    if not isinstance(node, dict):
        return node
    out = {}
    t = node.get("type")
    if isinstance(t, str) and t in _TYPE:
        out["type"] = _TYPE[t]
    for k in SUPPORTED_KEYWORDS - {"type", "properties", "items"}:
        if k in node:
            out[k] = node[k]
    if "properties" in node:
        out["properties"] = {k: schema_from_bfcl(v) for k, v in node["properties"].items()}
    if "items" in node:
        out["items"] = schema_from_bfcl(node["items"])
    return out


def tool_spec(doc: dict) -> dict:
    """Prompt-side spec for one BFCL function doc (drops upstream's extra `response` field)."""
    return {"name": doc["name"], "description": doc.get("description", ""),
            "parameters": schema_from_bfcl(doc.get("parameters", {"type": "dict", "properties": {}}))}


def to_ast_output(calls: list[ToolCall]) -> list[dict]:
    """ToolCalls -> the `[{name: {arg: value}}]` shape the official ast_checker consumes."""
    out = []
    for c in calls:
        if not isinstance(c.arguments, dict):
            raise ValueError(f"arguments of {c.name!r} are not an object")
        out.append({c.name: c.arguments})
    return out


def _lit(v, depth: int = 0) -> str:
    """Python literal for a JSON value. Anything else is rejected, so no model text becomes code."""
    if depth > 20:
        raise ValueError("too deeply nested")
    if v is None or isinstance(v, (bool, str)):
        return repr(v)
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite number")
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_lit(x, depth + 1) for x in v) + "]"
    if isinstance(v, dict):
        if not all(isinstance(k, str) for k in v):
            raise ValueError("non-string key")
        return "{" + ", ".join(f"{_lit(k)}: {_lit(x, depth + 1)}" for k, x in v.items()) + "}"
    raise ValueError(f"unsupported value type {type(v).__name__}")


def call_string(name: str, args, allowed: set[str]) -> str:
    """`name(k=v, ...)` for BFCL's multi-turn executor, which eval()s it.

    Only whitelisted function names, identifier keyword names, and repr() of JSON literals are emitted;
    raises ValueError for anything else. Model output must never reach the executor any other way.
    """
    if name not in allowed or not _IDENT.match(name):
        raise ValueError(f"unknown function {name!r}")
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    parts = []
    for k, v in args.items():
        if not isinstance(k, str) or not _IDENT.match(k) or keyword.iskeyword(k):
            raise ValueError(f"bad argument name {k!r}")
        parts.append(f"{k}={_lit(v)}")
    s = f"{name}({', '.join(parts)})"
    if len(s) > MAX_CALL_CHARS:
        raise ValueError("call too long")
    return s
