from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class Completion:
    text: str
    prefill_ms: float
    decode_ms: float
    prompt_tokens: int
    cached_tokens: int
    gen_tokens: int
    wall_ms: float
    raw: dict

    @property
    def decode_tok_s(self) -> float:
        return self.gen_tokens / (self.decode_ms / 1000) if self.decode_ms else 0.0

    @property
    def prefill_tok_s(self) -> float:
        fresh = self.prompt_tokens - self.cached_tokens
        return fresh / (self.prefill_ms / 1000) if self.prefill_ms else 0.0


def parse_timings(body: dict, wall_ms: float) -> Completion:
    """Map a llama-server /completion response onto stage timings.

    `timings` carries prompt_n/prompt_ms (uncached prompt tokens only), predicted_n/predicted_ms
    and cache_n (tokens reused from the prefix cache). Total prompt size = prompt_n + cache_n.
    """
    t = body.get("timings") or {}
    cached = int(t.get("cache_n", 0))
    return Completion(
        text=body.get("content", ""),
        prefill_ms=float(t.get("prompt_ms", 0.0)),
        decode_ms=float(t.get("predicted_ms", 0.0)),
        prompt_tokens=int(t.get("prompt_n", 0)) + cached,
        cached_tokens=cached,
        gen_tokens=int(t.get("predicted_n", 0)),
        wall_ms=wall_ms,
        raw=body,
    )


class Client:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def complete(self, prompt: str, n_predict: int, grammar: str | None = None,
                 cache_prompt: bool = True, temperature: float = 0.0, seed: int = 0) -> Completion:
        payload = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": cache_prompt,
                   "temperature": temperature, "seed": seed}
        if grammar:
            payload["grammar"] = grammar
        t0 = time.perf_counter()
        r = self.http.post("/completion", json=payload)
        wall = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return parse_timings(r.json(), wall)

    def slots(self) -> list[dict]:
        return self.http.get("/slots").json()

    def reset_cache(self) -> None:
        self.http.post("/slots/0?action=erase").raise_for_status()
