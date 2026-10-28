"""In-process stand-in for Client: scripted model outputs, ChatML-ish template, word-count tokenizer,
prefix-cache emulation. Fast and deterministic for loop/sweep tests."""
from __future__ import annotations

import httpx

from a2e.client import Completion


class StubClient:
    def __init__(self, outputs, ms_per_prompt_token=0.05, ms_per_gen_token=0.1, fail_at: int | None = None):
        self.outputs, self.fail_at = outputs, fail_at
        self.mp, self.mg = ms_per_prompt_token, ms_per_gen_token
        self.prompts: list[str] = []
        self.calls: list[dict] = []
        self._last: list[str] = []

    def apply_template(self, messages):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"

    def tokenize(self, text):
        return max(len(text) // 4, 1)  # ~4 chars per token

    def complete(self, prompt, n_predict, grammar=None, json_schema=None, cache_prompt=True, **kw):
        i = len(self.prompts)
        if self.fail_at == i:
            raise httpx.HTTPStatusError("boom", request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(400))
        self.prompts.append(prompt)
        self.calls.append({"json_schema": json_schema, "grammar": grammar, "cache_prompt": cache_prompt,
                           "n_predict": n_predict})
        text = self.outputs(prompt, i) if callable(self.outputs) else self.outputs[min(i, len(self.outputs) - 1)]
        words = prompt.split()
        cached = 0
        if cache_prompt:
            for a, b in zip(self._last, words):
                if a != b:
                    break
                cached += 1
        self._last = words
        gen = max(len(text.split()), 1)
        fresh = len(words) - cached
        return Completion(text=text, prefill_ms=fresh * self.mp, decode_ms=gen * self.mg,
                          prompt_tokens=len(words), cached_tokens=cached, gen_tokens=gen,
                          wall_ms=fresh * self.mp + gen * self.mg + 0.5, raw={})
