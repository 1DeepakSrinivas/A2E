import pytest
from jsonschema import Draft202012Validator

from a2e.agent.grammar import SUPPORTED_KEYWORDS
from a2e.agent.prompt import render_tools
from a2e.agent.schema_gen import SchemaTooLarge, pad_tools
from a2e.tasks.custom import FAMILY_TOOLS
from a2e.tools.builtin import build_registry

count = lambda t: max(len(t) // 4, 1)  # noqa: E731


@pytest.fixture
def reg(tmp_path):
    return build_registry(tmp_path)


@pytest.mark.parametrize("family", list(FAMILY_TOOLS))
@pytest.mark.parametrize("target", [256, 1024, 2048])
def test_lands_within_tolerance_or_core_too_big(reg, family, target):
    from a2e.tasks.envs import family_core_specs
    core = family_core_specs(family)
    if count(render_tools(core)) > target * 1.05:
        with pytest.raises(SchemaTooLarge):
            pad_tools(core, target, count)
        return
    specs, n = pad_tools(core, target, count)
    assert abs(n - target) / target <= 0.05, (family, target, n)
    assert n == count(render_tools(specs))


def test_deterministic_per_seed_and_varies_across_seeds(reg):
    core = reg.specs(FAMILY_TOOLS["calc"])
    a, _ = pad_tools(core, 1024, count, seed=1)
    b, _ = pad_tools(core, 1024, count, seed=1)
    c, _ = pad_tools(core, 1024, count, seed=2)
    assert a == b and a != c


def test_core_kept_names_unique_and_no_collisions(reg):
    core = reg.specs(FAMILY_TOOLS["retrieval"])
    specs, _ = pad_tools(core, 2048, count)
    names = [s["name"] for s in specs]
    assert len(set(names)) == len(names) and {s["name"] for s in core} <= set(names)
    assert len(specs) > len(core)


def test_core_position_is_shuffled_not_always_first(reg):
    core = reg.specs(FAMILY_TOOLS["calc"])
    firsts = {pad_tools(core, 1024, count, seed=s)[0][0]["name"] for s in range(8)}
    assert len(firsts) > 1


def test_distractors_are_valid_and_in_supported_subset(reg):
    core = reg.specs(["calculator"])
    specs, _ = pad_tools(core, 1024, count)
    for s in specs:
        Draft202012Validator.check_schema(s["parameters"])
        for p in s["parameters"]["properties"].values():
            assert set(p) <= SUPPORTED_KEYWORDS


def test_too_small_target_raises(reg):
    with pytest.raises(SchemaTooLarge):
        pad_tools(reg.specs(), 20, count)


def test_tight_tolerance(reg):
    _, n = pad_tools(reg.specs(["calculator", "final_answer"]), 1024, count, tolerance=0.01)
    assert abs(n - 1024) / 1024 <= 0.01


def test_works_with_word_level_counter(reg):
    """A coarser counter (words) still converges; guards against loops that assume fine granularity."""
    words = lambda t: len(t.replace(",", " ").replace(":", " ").split())  # noqa: E731
    _, n = pad_tools(reg.specs(["calculator", "final_answer"]), 300, words, tolerance=0.1)
    assert abs(n - 300) / 300 <= 0.1


@pytest.mark.parametrize("target", [512, 1024, 2048])
def test_real_bfcl_pool_fills_first_and_real_specs_are_not_edited(reg, target):
    from a2e.tools.bfcl_tools import distractor_pool
    core = reg.specs(FAMILY_TOOLS["calc"])
    pool = list(distractor_pool())
    specs, n = pad_tools(core, target, count, seed=4, pool=pool)
    assert abs(n - target) / target <= 0.05
    by_name = {p["name"]: p for p in pool}
    real = [s for s in specs if s["name"] in by_name]
    assert len(real) >= 3 and all(s == by_name[s["name"]] for s in real)  # untouched BFCL docs
    synthetic = [s for s in specs if s["name"] not in by_name and s not in core]
    assert len(synthetic) <= 2  # only to close the last gap
    assert pad_tools(core, target, count, seed=4, pool=pool) == (specs, n)


def test_pool_never_duplicates_core_names(reg):
    from a2e.tools.bfcl_tools import distractor_pool
    from a2e.tasks.envs import family_core_specs
    core = family_core_specs("fs_search")
    specs, _ = pad_tools(core, 2048, count, seed=1, pool=list(distractor_pool()))
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)) and {"find", "cd", "cat"} <= set(names)
