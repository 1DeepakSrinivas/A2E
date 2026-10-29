from __future__ import annotations

from a2e.client import Completion


class ReplayClient:
    """Feeds recorded model outputs back through the harness so a run can be rescored without the model."""

    def __init__(self, texts: list[str]):
        self.texts, self.i = list(texts), 0

    def apply_template(self, messages: list[dict]) -> str:
        return "\n".join(m["content"] for m in messages)

    def tokenize(self, text: str) -> int:
        return len(text.split())

    def complete(self, prompt: str, n_predict: int, json_schema=None, cache_prompt: bool = True, **kw) -> Completion:
        text = self.texts[self.i] if self.i < len(self.texts) else ""
        self.i += 1
        return Completion(text, 0.0, 0.0, len(prompt.split()), 0, len(text.split()), 0.0, {})
