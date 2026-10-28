from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from a2e.agent.grammar import call_schema
from a2e.agent.parse import ToolCall, parse_calls
from a2e.agent.prompt import system_message, tool_response_message
from a2e.client import Completion
from a2e.tasks.scoring import ChainScore, ExpectedCall, score_call
from a2e.tools.base import Registry, ToolResult


@dataclass
class TurnRecord:
    step: int
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    parse_error: str | None = None
    validation_errors: list[str | None] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    completion: Completion | None = None
    template_ms: float = 0.0
    parse_ms: float = 0.0
    validate_ms: float = 0.0
    tool_wait_ms: float = 0.0
    wall_ms: float = 0.0
    windows: dict[str, tuple[float, float]] = field(default_factory=dict)  # perf_counter, for energy integration
    error: str | None = None

    @property
    def harness_ms(self) -> float:
        model = self.completion.wall_ms if self.completion else 0.0
        return max(self.wall_ms - model - self.parse_ms - self.validate_ms - self.tool_wait_ms, 0.0)

    def row(self, run_id: int, task_id: str, correct: bool | None = None) -> dict:
        c = self.completion
        return {"run_id": run_id, "task_id": task_id, "step": self.step,
                "prefill_ms": c.prefill_ms if c else None, "decode_ms": c.decode_ms if c else None,
                "parse_ms": self.parse_ms, "validate_ms": self.validate_ms, "tool_wait_ms": self.tool_wait_ms,
                "harness_ms": self.harness_ms, "wall_ms": self.wall_ms,
                "prompt_tokens": c.prompt_tokens if c else None, "cached_tokens": c.cached_tokens if c else None,
                "gen_tokens": c.gen_tokens if c else None, "correct": None if correct is None else int(correct)}


@dataclass
class Trace:
    turns: list[TurnRecord] = field(default_factory=list)
    status: str = ""  # final_answer | parse_error | max_turns | server_error
    final_answer: str | None = None

    def score(self, steps: list[ExpectedCall]) -> ChainScore:
        """Score turn k's first call against expected step k, resolving later steps from real tool output."""
        results: list[str] = []
        out = ChainScore(finished=self.status == "final_answer")
        for k, exp in enumerate(steps):
            turn = self.turns[k] if k < len(self.turns) else None
            if turn is None or not turn.calls:
                from a2e.tasks.scoring import StepScore
                out.steps.append(StepScore(note="missing step"))
                continue
            out.steps.append(score_call(exp, results, turn.calls[0]))
            results.append(turn.results[0].output if turn.results else "")
        return out


class AgentLoop:
    """Multi-turn tool-calling loop that times every stage of every turn.

    Stages per turn: template (harness), model call (prefill + decode from server timings), parse,
    validate (schema check of the calls), tool-wait (real tool execution). The prompt only ever grows
    by appended messages so the schema prefix stays cacheable.
    """

    def __init__(self, client, registry: Registry, specs: list[dict], grammar: bool = False,
                 n_predict: int = 128, cache_prompt: bool = True):
        self.client, self.registry, self.specs = client, registry, specs
        self.n_predict, self.cache_prompt = n_predict, cache_prompt
        self.schema = call_schema(specs) if grammar else None

    def run(self, user_prompt: str, max_turns: int = 6) -> Trace:
        messages = [system_message(self.specs), {"role": "user", "content": user_prompt}]
        trace = Trace(status="max_turns")
        for step in range(max_turns):
            turn = TurnRecord(step)
            trace.turns.append(turn)
            t0 = time.perf_counter()
            try:
                prompt = self.client.apply_template(messages)
                t1 = time.perf_counter()
                turn.completion = self.client.complete(prompt, self.n_predict, json_schema=self.schema,
                                                       cache_prompt=self.cache_prompt)
            except httpx.HTTPError as e:
                turn.error = f"{type(e).__name__}: {e}"
                turn.wall_ms = (time.perf_counter() - t0) * 1000
                trace.status = "server_error"
                return trace
            t2 = time.perf_counter()
            turn.template_ms = (t1 - t0) * 1000
            turn.text = turn.completion.text
            parsed = parse_calls(turn.text)
            t3 = time.perf_counter()
            turn.calls, turn.parse_error = parsed.calls, parsed.error
            turn.validation_errors = [self.registry.validate(c.name, c.arguments) for c in turn.calls]
            t4 = time.perf_counter()
            for c, err in zip(turn.calls, turn.validation_errors):
                turn.results.append(ToolResult(False, f"error: invalid call: {err}") if err
                                    else self.registry.execute(c.name, c.arguments))
            t5 = time.perf_counter()
            turn.parse_ms, turn.validate_ms, turn.tool_wait_ms = (t3 - t2) * 1000, (t4 - t3) * 1000, (t5 - t4) * 1000
            turn.wall_ms = (t5 - t0) * 1000
            c = turn.completion
            turn.windows = {"prefill": (t1, t1 + c.prefill_ms / 1000), "decode": (t1 + c.prefill_ms / 1000, t2),
                            "parse": (t2, t3), "validate": (t3, t4), "tool": (t4, t5)}
            if turn.parse_error:
                trace.status = "parse_error"
                return trace
            done = next((r for call, r, e in zip(turn.calls, turn.results, turn.validation_errors)
                         if call.name == "final_answer" and e is None), None)
            if done is not None:
                trace.status, trace.final_answer = "final_answer", done.output
                return trace
            messages += [{"role": "assistant", "content": turn.text},
                         tool_response_message(turn.calls, turn.results)]
        return trace
