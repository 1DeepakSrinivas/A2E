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


class CacheResetError(RuntimeError):
    """Neither the slot erase nor the flush prompt could clear the prefix cache."""


# A prompt that shares no text with any task prompt, so completing it makes a one-slot server drop the cached prefix.
FLUSH_PROMPT = "a2e cache flush 0x7f3a9c21 zebra quartz lantern 4417 unrelated filler text"


class Client:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def complete(self, prompt: str, n_predict: int, grammar: str | None = None,
                 cache_prompt: bool = True, temperature: float = 0.0, seed: int = 0,
                 json_schema: dict | None = None, ignore_eos: bool = False) -> Completion:
        payload = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": cache_prompt,
                   "temperature": temperature, "seed": seed}
        if ignore_eos:
            payload["ignore_eos"] = True  # decode exactly n_predict tokens (plain-generation baseline)
        if grammar:
            payload["grammar"] = grammar
        if json_schema:
            payload["json_schema"] = json_schema  # llama-server converts it to a grammar
        t0 = time.perf_counter()
        r = self.http.post("/completion", json=payload)
        wall = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return parse_timings(r.json(), wall)

    def slots(self) -> list[dict]:
        return self.http.get("/slots").json()

    def reset_cache(self) -> str:
        """Make the next prompt start with a cold prefix cache. Returns how: `erase` (the server's slot erase) or
        `flush` (the erase failed, so one unrelated one-token prompt replaces the cached prefix). Raises
        CacheResetError when neither works, because a warm cache would make the next measurements too fast."""
        try:
            self.http.post("/slots/0?action=erase").raise_for_status()
            return "erase"
        except httpx.HTTPError as first:
            try:
                self.complete(FLUSH_PROMPT, 1, cache_prompt=True)
                return "flush"
            except httpx.HTTPError as second:
                raise CacheResetError(f"could not clear the prefix cache: erase failed ({first}); "
                                      f"flush failed ({second})") from second

    def apply_template(self, messages: list[dict]) -> str:
        """Render messages with the model's own chat template (role markup only, no tools)."""
        r = self.http.post("/apply-template", json={"messages": messages})
        r.raise_for_status()
        return r.json()["prompt"]

    def tokenize(self, text: str) -> int:
        r = self.http.post("/tokenize", json={"content": text})
        r.raise_for_status()
        return len(r.json()["tokens"])
