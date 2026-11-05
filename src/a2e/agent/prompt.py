from __future__ import annotations

import json

SYSTEM = (
    "You are an assistant that completes tasks by calling tools.\n"
    "Available tools{header}:\n<tools>\n{tools}\n</tools>\n"
    'Respond ONLY with a JSON array of tool calls, for example [{{"name": "<tool name>", "arguments": {{...}}}}]. '
    "Call one tool at a time unless calls are independent. After a tool result arrives, call the next tool. "
    "When the task is complete, call final_answer."
)


# How the tool list is shown to the model. Only the rendering changes; the reply format and the parser do not.
PROMPT_FORMATS = ("json_schema", "compact", "natural")
HEADERS = {"json_schema": " (JSON schemas)",
           "compact": " (name(arguments) - description; a ? after an argument name marks it optional)",
           "natural": ""}


def render_tools(specs: list[dict]) -> str:
    """The baseline rendering: one compact JSON schema per line. Padding and token counts use this form."""
    return "\n".join(json.dumps(s, separators=(",", ":")) for s in specs)


def _props(spec: dict) -> tuple[dict, set]:
    params = spec.get("parameters") or {}
    return params.get("properties") or {}, set(params.get("required") or [])


def render_compact(specs: list[dict]) -> str:
    lines = []
    for s in specs:
        props, req = _props(s)
        args = ", ".join(f"{n}{'' if n in req else '?'}: {p.get('type', 'any')}" for n, p in props.items())
        lines.append(f"{s['name']}({args}) - {s.get('description', '')}".rstrip(" -"))
    return "\n".join(lines)


def render_natural(specs: list[dict]) -> str:
    lines = []
    for s in specs:
        props, req = _props(s)
        text = f"- {s['name']}: {s.get('description', '').rstrip('.')}."
        if props:
            args = "; ".join(f"{n} ({p.get('type', 'any')}, {'required' if n in req else 'optional'})"
                             + (f": {p['description']}" if p.get("description") else "") for n, p in props.items())
            text += f" Arguments: {args}."
        else:
            text += " It takes no arguments."
        lines.append(text)
    return "\n".join(lines)


def render_tools_as(specs: list[dict], fmt: str = "json_schema") -> str:
    if fmt not in PROMPT_FORMATS:
        raise ValueError(f"unknown prompt format {fmt!r}; choose from {PROMPT_FORMATS}")
    return {"json_schema": render_tools, "compact": render_compact, "natural": render_natural}[fmt](specs)


def system_message(specs: list[dict], fmt: str = "json_schema") -> dict:
    return {"role": "system", "content": SYSTEM.format(header=HEADERS[fmt], tools=render_tools_as(specs, fmt))}


def tool_response_message(calls, results) -> dict:
    lines = [json.dumps({"name": c.name, "result": r.output}) for c, r in zip(calls, results)]
    return {"role": "user", "content": "<tool_response>\n" + "\n".join(lines) + "\n</tool_response>"}


SYSTEM_BFCL = (
    "You are an assistant that fulfils requests by calling tools.\n"
    "Available tools{header}:\n<tools>\n{tools}\n</tools>\n"
    'Respond ONLY with a JSON array of tool calls, for example [{{"name": "<tool name>", "arguments": {{...}}}}]. '
    "Put every call the request needs in one array. "
    "If none of the tools is relevant to the request, or a required argument is missing, respond with an empty array []. "
    "After tool results arrive, make any further calls the request needs; when it is fully handled, respond with []. "
    "Additional tools may be introduced later in the conversation; you may use them once they are provided."
)


def bfcl_system_message(specs: list[dict], fmt: str = "json_schema") -> dict:
    return {"role": "system", "content": SYSTEM_BFCL.format(header=HEADERS[fmt], tools=render_tools_as(specs, fmt))}
