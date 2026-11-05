"""Model x Quant x KV x SeqLen x Decode x Grammar x Device: structural checks on the full grid."""
import itertools
from collections import Counter

import pytest

from a2e.config import factorial
from a2e.server import build_args

MODELS = ["qwen3-4b-instruct-2507", "granite-3-1-2b-instruct"]
GRID = dict(models=MODELS, quants=["q5_k_m", "q6_k", "iq3_m"], kv_quants=["f16", "q8_0"], schema_lens=[256, 1024, 2048],
            decode_lens=[128, 512], grammars=[False, True], devices=["pi5", "mac"])
ROWS = list(factorial(**GRID))


def test_grid_size():
    assert len(ROWS) == 2 * 3 * 2 * 3 * 2 * 2 * 2 == 288


def test_all_cells_unique():
    keys = [c.model_dump_json() for c in ROWS]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("axis,values", [("quant", {"q5_k_m", "q6_k", "iq3_m"}), ("kv_quant", {"f16", "q8_0"}),
                                         ("schema_len", {256, 1024, 2048}), ("decode_len", {128, 512}),
                                         ("grammar", {False, True}), ("device", {"pi5", "mac"})])
def test_every_axis_level_is_covered(axis, values):
    assert {getattr(c, axis) for c in ROWS} == values


def test_balanced_between_grammar_on_off():
    n = Counter(c.grammar for c in ROWS)
    assert n[True] == n[False]


@pytest.mark.parametrize("device", ["pi5", "mac"])
def test_every_feasible_cell_builds_server_args(device):
    for c in ROWS:
        if c.device != device:
            continue
        a = build_args(c, "bin", f"{c.model}-{c.quant}.gguf")
        assert a[0] == "bin" and ("--cache-type-k" in a) == (c.kv_quant != "f16")
        assert a[a.index("-ngl") + 1] == ("99" if device == "mac" else "0")


def test_pairs_for_ablation_exist():
    """Every grammar-on cell has a grammar-off twin differing only in `grammar` (needed for overhead)."""
    off = {c.model_copy(update={"grammar": True}).model_dump_json() for c in ROWS if not c.grammar}
    on = {c.model_dump_json() for c in ROWS if c.grammar}
    assert on == off
