"""BFCL's simulated tools (Berkeley Function Calling Leaderboard) as ordinary A2E tools.

The classes are the vendored, unmodified BFCL backends (third_party/bfcl_eval). Each public method becomes a `Tool`
whose JSON schema comes from BFCL's own function documents. A registry holds fresh instances, so every task starts
from its own state. Results are turned into text the way BFCL's executor does: dicts as JSON, anything else as str().
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache

import a2e.bfcl  # noqa: F401  (puts the vendored package on the path)
from a2e.bfcl.convert import tool_spec
from a2e.bfcl.paths import DATA
from a2e.tools.base import Registry, Tool

DOC_FILES = {"GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json",
             "MessageAPI": "message_api.json", "TwitterAPI": "posting_api.json", "TicketAPI": "ticket_api.json",
             "TradingBot": "trading_bot.json", "TravelAPI": "travel_booking.json",
             "VehicleControlAPI": "vehicle_control.json", "WebSearchAPI": "web_search.json"}
EXECUTABLE = ("GorillaFileSystem", "MathAPI")


@lru_cache(maxsize=None)
def class_docs(class_name: str) -> tuple[dict, ...]:
    path = DATA / "multi_turn_func_doc" / DOC_FILES[class_name]
    return tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def as_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):  # e.g. mpmath or Decimal values
            return str(result)
    return str(result)


def new_instance(class_name: str, scenario: dict | None = None):
    if class_name == "GorillaFileSystem":
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import GorillaFileSystem
        inst = GorillaFileSystem()
        inst._load_scenario(copy.deepcopy(scenario or {}))
        return inst
    if class_name == "MathAPI":
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.math_api import MathAPI
        return MathAPI()
    raise ValueError(f"{class_name} is not executable in A2E (only {EXECUTABLE})")


def class_tools(class_name: str, instance) -> list[Tool]:
    tools = []
    for doc in class_docs(class_name):
        method = getattr(instance, doc["name"])
        spec = tool_spec(doc)
        tools.append(Tool(spec["name"], spec["description"], spec["parameters"],
                          (lambda m: lambda **kw: as_text(m(**kw)))(method)))
    return tools


class ClassRegistry(Registry):
    """A registry whose tools are bound to live BFCL instances (kept in `.instances` for state checks)."""

    def __init__(self, tools: list[Tool], instances: dict):
        super().__init__(tools)
        self.instances = instances


def bfcl_registry(classes: list[str], scenario: dict | None = None, extra: list[Tool] | None = None,
                  cwd: list[str] | None = None) -> ClassRegistry:
    """Fresh instances of `classes`, all their documented methods as tools, plus `extra` tools.

    `cwd` (GorillaFileSystem) moves into those folders after loading, one level at a time, as `cd` would.
    """
    instances, tools = {}, []
    for c in classes:
        inst = new_instance(c, scenario if c == "GorillaFileSystem" else None)
        if c == "GorillaFileSystem":
            for d in cwd or []:
                inst.cd(folder=d)
        instances[c] = inst
        tools += class_tools(c, inst)
    return ClassRegistry(tools + list(extra or []), instances)


@lru_cache(maxsize=None)
def distractor_pool() -> tuple[dict, ...]:
    """Prompt specs of every BFCL function document A2E ships (9 classes), used to pad tool lists realistically."""
    return tuple(tool_spec(d) for c in DOC_FILES for d in class_docs(c))
