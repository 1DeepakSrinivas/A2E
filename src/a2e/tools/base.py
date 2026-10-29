from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

MAX_OUTPUT_CHARS = 1000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema (object)
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ToolResult:
    ok: bool
    output: str
    exec_ms: float = 0.0


class Registry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self, names: list[str] | None = None) -> list[dict]:
        return [self.tools[n].spec() for n in (names or self.tools)]

    def validate(self, name: str, args: Any) -> str | None:
        """None if the call is valid, else a short error string fed back to the model."""
        tool = self.tools.get(name)
        if tool is None:
            return f"unknown tool {name!r}"
        if not isinstance(args, dict):
            return "arguments must be an object"
        errs = sorted(Draft202012Validator(tool.parameters).iter_errors(args), key=lambda e: list(e.path))
        return "; ".join(e.message for e in errs[:3]) or None

    def execute(self, name: str, args: dict) -> ToolResult:
        """Run a validated call. Tool exceptions become error results; exec time is measured."""
        t0 = time.perf_counter()
        try:
            out, ok = str(self.tools[name].fn(**args)), True
        except Exception as e:  # noqa: BLE001 - tool failures are data for the model, not harness errors
            out, ok = f"error: {type(e).__name__}: {e}", False
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "...[truncated]"
        return ToolResult(ok, out, (time.perf_counter() - t0) * 1000)
