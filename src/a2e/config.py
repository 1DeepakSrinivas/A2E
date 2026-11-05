from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel

Device = Literal["pi5", "mac"]
Quant = Literal["q5_k_m", "q6_k", "iq3_m"]
KVQuant = Literal["f16", "q8_0"]


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


def factorial(models, quants, kv_quants, schema_lens, decode_lens, grammars, devices):
    """Yield a config for every cell of the Model x Quant x KV x SeqLen x Decode x Grammar x Device grid."""
    for m, q, kv, s, d, g, dev in itertools.product(
        models, quants, kv_quants, schema_lens, decode_lens, grammars, devices
    ):
        yield RunConfig(model=m, quant=q, kv_quant=kv, schema_len=s, decode_len=d, grammar=g, device=dev)
