from __future__ import annotations

import random
from collections.abc import Callable

from a2e.agent.prompt import render_tools

_VERBS = ["get", "list", "fetch", "update", "delete", "create", "find", "check", "send", "convert"]
_NOUNS = ["weather", "invoice", "contact", "ticket", "route", "stock", "playlist", "reservation", "inventory",
          "calendar", "shipment", "profile", "forecast", "payment", "thermostat", "device"]
_WORDS = "the a of for to with from this that record item value optional identifier returned details".split()


class SchemaTooLarge(ValueError):
    """The required tools alone already exceed the target schema length."""


def _distractor(rng: random.Random, taken: set[str]) -> dict:
    while True:
        name = f"{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}"
        if name not in taken:
            break
        name = f"{name}_{rng.randint(2, 99)}"
        if name not in taken:
            break
    taken.add(name)
    props = {f"field_{i}": {"type": rng.choice(["string", "integer"]),
                            "description": " ".join(rng.choices(_WORDS, k=4))} for i in range(rng.randint(1, 3))}
    return {"name": name, "description": " ".join(rng.choices(_WORDS, k=rng.randint(4, 8))),
            "parameters": {"type": "object", "properties": props, "required": [next(iter(props))]}}


def pad_tools(core: list[dict], target_tokens: int, count: Callable[[str], int], seed: int = 0,
              tolerance: float = 0.05, pool: list[dict] | None = None) -> tuple[list[dict], int]:
    """Core tools plus distractor tools so the rendered schema is within `tolerance` of target_tokens.

    `count` measures text in model tokens (Client.tokenize). With a `pool` of real tool specs (e.g. BFCL functions),
    distractors are taken from it in a seeded order, skipping any that would overshoot. Seeded synthetic distractors
    then close the remaining gap; only their descriptions are lengthened or shortened, real specs are never edited.
    Order is shuffled deterministically so required tools are not always first. Same inputs always give the same
    schema, keeping the prompt prefix identical across tasks (prefix-cache friendly).
    """
    lo, hi = target_tokens * (1 - tolerance), target_tokens * (1 + tolerance)
    if count(render_tools(core)) > hi:
        raise SchemaTooLarge(f"core tools need {count(render_tools(core))} tokens > target {target_tokens}")
    rng = random.Random(seed)
    taken = {s["name"] for s in core}
    real: list[dict] = []
    synth: list[dict] = []

    def total() -> int:
        return count(render_tools(core + real + synth))

    if pool:
        candidates = [p for p in pool if p["name"] not in taken]
        random.Random(f"{seed}/pool").shuffle(candidates)
        for spec in candidates:
            if total() >= lo:
                break
            real.append(spec)
            if total() > hi:
                real.pop()
                continue
            taken.add(spec["name"])
    while total() < lo:
        synth.append(_distractor(rng, taken))
        if total() > hi:  # overshot: shrink its description, then drop it if still too big
            words = synth[-1]["description"].split()
            while len(words) > 1 and total() > hi:
                words.pop()
                synth[-1]["description"] = " ".join(words)
            if total() > hi:
                synth.pop()
                break
    i = 0
    while synth and total() < lo:  # absorb the remaining gap in the last synthetic distractor
        synth[-1]["description"] += " " + _WORDS[i % len(_WORDS)]
        i += 1
    if total() > hi or total() < lo:
        raise SchemaTooLarge("could not land inside the tolerance band")
    specs = core + real + synth
    random.Random(seed + 1).shuffle(specs)
    return specs, count(render_tools(specs))
