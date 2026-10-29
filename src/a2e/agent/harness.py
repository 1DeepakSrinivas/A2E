"""Harness variants: the parts of the agent loop around the model that can change energy per successful task.

A `Harness` has four independent choices. The default of each is today's behavior, so a default `Harness()` gives
exactly the prompts, turns and job keys the loop produced before this module existed.

prompt_format  How the tool list is shown. `json_schema` (default), `compact` (name, typed arguments, one-line
               description) or `natural` (plain sentences). The reply format and the parser stay the same.
context        What the model sees of earlier tool results. `full` (default), `mask` (tool results older than the
               last MASK_KEEP are replaced by a stub) or `truncate` (each tool result is cut to TRUNCATE_CHARS).
               The stored history is never changed; only the messages sent to the model are.
recovery       What happens when a reply cannot be read as tool calls. `none` (default) ends the task. `feedback`
               keeps the bad reply and adds a message saying what was wrong. `retry` drops the bad reply and asks
               again with a one-line reminder that is not kept, at most MAX_RETRIES times in a row.
               (Calls that parse but fail validation already get an error result back in every mode.)
stopping       When to give up. `fixed` (default) uses the suite's turn limits. `tight` caps the turns at the
               number a correct run needs plus TIGHT_SLACK. `early_stop` ends the task after two identical calls in
               a row or two failed turns in a row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from a2e.agent.prompt import PROMPT_FORMATS

CONTEXTS = ("full", "mask", "truncate")
RECOVERIES = ("none", "feedback", "retry")
STOPPINGS = ("fixed", "tight", "early_stop")
FIELDS = ("prompt_format", "context", "recovery", "stopping")
MASK_KEEP = 2  # most recent tool results kept in full by `mask`
TRUNCATE_CHARS = 400  # longest tool result kept by `truncate`
MAX_RETRIES = 2  # consecutive unreadable replies `retry` will re-ask
TIGHT_SLACK = 2  # extra turns `tight` allows beyond what a correct run needs
EARLY_STOP_REPEATS = 2  # identical calls in a row, or failed turns in a row, that end a task under `early_stop`

MASKED = "<tool_response>\n[earlier tool output removed]\n</tool_response>"
RETRY_REMINDER = "Reply with only a JSON array of tool calls."


@dataclass(frozen=True)
class Harness:
    prompt_format: str = "json_schema"
    context: str = "full"
    recovery: str = "none"
    stopping: str = "fixed"

    def __post_init__(self):
        for name, allowed in (("prompt_format", PROMPT_FORMATS), ("context", CONTEXTS), ("recovery", RECOVERIES),
                              ("stopping", STOPPINGS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"unknown {name} {getattr(self, name)!r}; choose from {allowed}")

    @property
    def is_default(self) -> bool:
        return self == Harness()

    def non_default(self) -> dict:
        """Only the fields that differ from the default. This is what goes into a job's config, so jobs of the
        default harness keep exactly the keys they had before harness variants existed."""
        base = Harness()
        return {f: getattr(self, f) for f in FIELDS if getattr(self, f) != getattr(base, f)}

    @classmethod
    def from_job(cls, job: dict) -> "Harness":
        return cls(**{f: job[f] for f in FIELDS if f in job})


def feedback_message(error: str) -> dict:
    return {"role": "user", "content": f"Your last reply could not be read as tool calls ({error}). "
                                       "Reply with only a JSON array of tool calls."}


def retry_reminder() -> dict:
    return {"role": "user", "content": RETRY_REMINDER}


def call_signature(calls) -> tuple:
    """What a turn asked for, for spotting a model that repeats itself."""
    return tuple((c.name, json.dumps(c.arguments, sort_keys=True, default=str)) for c in calls)


def is_tool_response(msg: dict) -> bool:
    return msg.get("role") == "user" and str(msg.get("content", "")).startswith("<tool_response>")


def apply_context(messages: list[dict], h: Harness) -> list[dict]:
    """The messages to send to the model. Returns the list unchanged for `full`; otherwise a copy."""
    if h.context == "full":
        return messages
    idx = [i for i, m in enumerate(messages) if is_tool_response(m)]
    out = list(messages)
    if h.context == "mask":
        for i in idx[:-MASK_KEEP] if MASK_KEEP else idx:
            out[i] = {**messages[i], "content": MASKED}
    else:  # truncate
        for i in idx:
            text = messages[i]["content"]
            if len(text) > TRUNCATE_CHARS:
                out[i] = {**messages[i], "content": text[:TRUNCATE_CHARS] + "\n[truncated]\n</tool_response>"}
    return out


def turn_budget(h: Harness, limit: int, needed: int | None) -> int:
    """Model calls allowed. `needed` is how many a correct run takes (None when unknown)."""
    if h.stopping == "tight" and needed is not None:
        return max(1, min(limit, needed + TIGHT_SLACK))
    return limit


class StopWatch:
    """Tracks repeated calls and failed turns for `early_stop`."""

    def __init__(self, h: Harness):
        self.on = h.stopping == "early_stop"
        self.last: tuple | None = None
        self.same = self.failed = 0

    def update(self, calls, failed: bool) -> bool:
        """Record one turn; True when the task should end now."""
        if not self.on:
            return False
        sig = call_signature(calls)
        self.same = self.same + 1 if sig and sig == self.last else (1 if sig else 0)
        self.last = sig or self.last
        self.failed = self.failed + 1 if failed else 0
        return self.same >= EARLY_STOP_REPEATS or self.failed >= EARLY_STOP_REPEATS
