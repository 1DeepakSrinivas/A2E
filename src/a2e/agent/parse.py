from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.S)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: Any  # dict when well formed; kept as-is otherwise so validation can report it


@dataclass
class ParseResult:
    calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None  # empty | no_json | malformed_json | bad_shape


def _as_call(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):  # OpenAI-style JSON-encoded arguments
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return ToolCall(obj["name"], args)


def _first_json(block: str) -> tuple[Any, str | None]:
    """First JSON array/object in the block (anything before it, such as prose or a fence, is skipped)."""
    starts = [i for i in (block.find("["), block.find("{")) if i >= 0]
    if not starts:
        return None, "no_json"
    try:
        return json.JSONDecoder().raw_decode(block, min(starts))[0], None
    except ValueError:  # includes output truncated by n_predict; do not scan into the broken value
        return None, "malformed_json"


def parse_calls(text: str) -> ParseResult:
    """Extract tool calls from model output.

    Thinking blocks are removed first (a truncated one removes everything after it). Each <tool_call> block
    contributes its own JSON value; without tags the first JSON array/object in the text is used. A value is
    one call (object) or several (array). Errors: empty | no_json | malformed_json | bad_shape.
    """
    t = _UNCLOSED_THINK.sub("", _THINK.sub("", text)).strip()
    if not t:
        return ParseResult(error="empty")
    blocks = re.findall(r"<tool_call>(.*?)(?:</tool_call>|\Z)", t, re.S) or [t]
    calls: list[ToolCall] = []
    for b in blocks:
        obj, err = _first_json(b)
        if err:
            return ParseResult(error=err)
        for item in obj if isinstance(obj, list) else [obj]:
            c = _as_call(item)
            if c is None:
                return ParseResult(error="bad_shape")
            calls.append(c)
    return ParseResult(calls) if calls else ParseResult(error="bad_shape")
