"""Model x Quant x KV x SeqLen x Decode x Grammar x Device: structural checks on the full grid."""
import itertools
from collections import Counter

import pytest

from a2e.config import MODEL_PARAMS_B, factorial
from a2e.server import build_args

MODELS = list(MODEL_PARAMS_B)
GRID = dict(models=MODELS, quants=["q8_0", "q4_0"], kv_quants=["f16", "q8_0"], schema_lens=[256, 1024, 2048],
            decode_lens=[128, 512], grammars=[False, True], devices=["pi5", "mac"])
ROWS = list(factorial(**GRID))


def test_grid_size():
    assert len(ROWS) == 5 * 2 * 2 * 3 * 2 * 2 * 2 == 480


def test_all_cells_unique():
    keys = [c.model_dump_json() for c, _, _ in ROWS]
    assert len(set(keys)) == len(keys)


def test_mac_cells_all_feasible():
    assert all(ok for c, ok, _ in ROWS if c.device == "mac")


def test_pi_infeasible_cells_are_exactly_8b_q8():
    for c, ok, why in ROWS:
        if c.device != "pi5":
            continue
        expect_bad = c.quant == "q8_0" and MODEL_PARAMS_B[c.model] > 4.0
        assert ok == (not expect_bad), (c.model, c.quant)
        assert (why == "not run: memory") == (not ok)


def test_infeasible_count():
    bad = [c for c, ok, _ in ROWS if not ok]
    big = sum(1 for p in MODEL_PARAMS_B.values() if p > 4.0)  # 8B models, Q8_0, Pi only
    per_model = 2 * 3 * 2 * 2  # kv x schema x decode x grammar
    assert len(bad) == big * per_model
    assert Counter(c.device for c in bad) == {"pi5": len(bad)}


@pytest.mark.parametrize("axis,values", [("quant", {"q8_0", "q4_0"}), ("kv_quant", {"f16", "q8_0"}),
                                         ("schema_len", {256, 1024, 2048}), ("decode_len", {128, 512}),
                                         ("grammar", {False, True}), ("device", {"pi5", "mac"})])
def test_every_axis_level_is_covered(axis, values):
    assert {getattr(c, axis) for c, _, _ in ROWS} == values


def test_balanced_between_grammar_on_off():
    n = Counter(c.grammar for c, _, _ in ROWS)
    assert n[True] == n[False]


@pytest.mark.parametrize("device", ["pi5", "mac"])
def test_every_feasible_cell_builds_server_args(device):
    for c, ok, _ in ROWS:
        if c.device != device or not ok:
            continue
        a = build_args(c, "bin", f"{c.model}-{c.quant}.gguf")
        assert a[0] == "bin" and ("--cache-type-k" in a) == (c.kv_quant != "f16")
        assert a[a.index("-ngl") + 1] == ("99" if device == "mac" else "0")


def test_pairs_for_ablation_exist():
    """Every grammar-on cell has a grammar-off twin differing only in `grammar` (needed for overhead)."""
    off = {c.model_copy(update={"grammar": True}).model_dump_json() for c, _, _ in ROWS if not c.grammar}
    on = {c.model_dump_json() for c, _, _ in ROWS if c.grammar}
    assert on == off
