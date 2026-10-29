"""Plain text generation (no tools): the baseline the tool-calling numbers are compared against."""
from __future__ import annotations

import json
import random
from importlib import resources

from a2e.agent.loop import TurnRecord, model_step


def plain_prompts(n: int, seed: int) -> list[tuple[str, str]]:
    prompts = json.loads(resources.files("a2e.data").joinpath("plain_prompts.json").read_text())
    idx = random.Random(f"{seed}/plain").sample(range(len(prompts)), min(n, len(prompts)))
    return [(f"plain-{i}", prompts[i]) for i in sorted(idx)]


def run_plain(client, prompt: str, n_predict: int, cache_prompt: bool) -> TurnRecord:
    """One generation of exactly n_predict tokens (ignore_eos), so every run decodes the same length."""
    turn, stamps = model_step(client, [{"role": "user", "content": prompt}], 0, n_predict, cache_prompt, None,
                              ignore_eos=True)
    if stamps:
        t0, t1, t2, t3 = stamps
        turn.finalize(t0, t1, t2, t2, t2, t2)  # no parse/validate/tool stage for plain generation
        turn.calls, turn.parse_error = [], None
    return turn
