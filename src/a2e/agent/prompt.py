from __future__ import annotations

import json

SYSTEM = (
    "You are an assistant that completes tasks by calling tools.\n"
    "Available tools (JSON schemas):\n<tools>\n{tools}\n</tools>\n"
    'Respond ONLY with a JSON array of tool calls, for example [{{"name": "<tool name>", "arguments": {{...}}}}]. '
    "Call one tool at a time unless calls are independent. After a tool result arrives, call the next tool. "
    "When the task is complete, call final_answer."
)


def render_tools(specs: list[dict]) -> str:
    return "\n".join(json.dumps(s, separators=(",", ":")) for s in specs)


def system_message(specs: list[dict]) -> dict:
    return {"role": "system", "content": SYSTEM.format(tools=render_tools(specs))}


def tool_response_message(calls, results) -> dict:
    lines = [json.dumps({"name": c.name, "result": r.output}) for c, r in zip(calls, results)]
    return {"role": "user", "content": "<tool_response>\n" + "\n".join(lines) + "\n</tool_response>"}
