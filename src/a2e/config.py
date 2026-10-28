from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel

Device = Literal["pi5", "mac"]
Quant = Literal["q8_0", "q4_0"]
KVQuant = Literal["f16", "q8_0"]

# Approximate parameter counts (billions), used for the Pi memory-feasibility rule.
MODEL_PARAMS_B: dict[str, float] = {
    "qwen3-4b": 4.0,
    "qwen3-8b": 8.0,
    "xlam2-3b": 3.0,
    "xlam2-8b": 8.0,
    "phi4-mini": 3.8,
}


class RunConfig(BaseModel):
    model: str
    quant: Quant
    kv_quant: KVQuant = "f16"
    schema_len: Literal[256, 1024, 2048] = 1024
    decode_len: Literal[128, 512] = 128
    chain_depth: int = 1
    grammar: bool = False
    prefix_cache: bool = True
    device: Device
    cooler: bool | None = None  # Pi only; decided after pilot
    governor: str | None = None  # Pi only, e.g. ondemand/performance


def feasible(cfg: RunConfig) -> tuple[bool, str]:
    """Pi 5 (8GB) rule: 8B models run Q4_0 only; Q8_0 is for <=4B models."""
    if cfg.device == "pi5" and cfg.quant == "q8_0" and MODEL_PARAMS_B.get(cfg.model, 0) > 4.0:
        return False, "not run: memory"
    return True, ""


def factorial(models, quants, kv_quants, schema_lens, decode_lens, grammars, devices):
    """Yield (config, feasible, reason) for the full Model x Quant x SeqLen x Device grid."""
    for m, q, kv, s, d, g, dev in itertools.product(
        models, quants, kv_quants, schema_lens, decode_lens, grammars, devices
    ):
        cfg = RunConfig(model=m, quant=q, kv_quant=kv, schema_len=s, decode_len=d, grammar=g, device=dev)
        ok, why = feasible(cfg)
        yield cfg, ok, why
