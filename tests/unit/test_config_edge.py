import pytest
from pydantic import ValidationError

from a2e.config import RunConfig, feasible


@pytest.mark.parametrize("field,bad", [("quant", "q5_k_m"), ("kv_quant", "q4_0"), ("schema_len", 512),
                                       ("decode_len", 256), ("device", "android")])
def test_invalid_values_rejected(field, bad):
    kwargs = dict(model="qwen3-4b", quant="q4_0", device="pi5")
    kwargs[field] = bad
    with pytest.raises(ValidationError):
        RunConfig(**kwargs)


def test_unknown_model_is_not_silently_feasible():
    ok, why = feasible(RunConfig(model="mystery-70b", quant="q8_0", device="pi5"))
    assert not ok and "unknown" in why


def test_defaults():
    c = RunConfig(model="qwen3-4b", quant="q4_0", device="mac")
    assert (c.kv_quant, c.grammar, c.prefix_cache, c.chain_depth) == ("f16", False, True, 1)
    assert c.cooler is None and c.governor is None


def test_config_json_roundtrip():
    c = RunConfig(model="gemma3-4b", quant="q4_0", kv_quant="q8_0", device="pi5", governor="performance")
    assert RunConfig.model_validate_json(c.model_dump_json()) == c
