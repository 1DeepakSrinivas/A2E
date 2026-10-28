from collections import Counter

import pytest

import a2e.bfcl  # noqa: F401
from a2e.bfcl.load import (AST_CATEGORIES, ALL_CATEGORIES, MULTI_TURN_CATEGORIES, RELEVANCE_CATEGORIES,
                           load_cases, load_subset, stratified_sample)

COUNTS = {"simple_python": 400, "multiple": 200, "parallel": 200, "parallel_multiple": 200, "irrelevance": 240,
          "live_irrelevance": 884, "live_relevance": 16, "multi_turn_base": 200, "multi_turn_miss_func": 200,
          "multi_turn_miss_param": 200}


@pytest.mark.parametrize("cat", ALL_CATEGORIES)
def test_category_sizes_ids_and_ground_truth(cat):
    cases = load_cases(cat)
    assert len(cases) == COUNTS[cat]
    assert len({c.id for c in cases}) == len(cases)
    has_gt = [c.ground_truth is not None for c in cases]
    assert all(has_gt) == (cat in AST_CATEGORIES + MULTI_TURN_CATEGORIES)
    assert not any(has_gt) == (cat in RELEVANCE_CATEGORIES)
    assert all(c.turns for c in cases)
    # the only empty turns are miss_func reveal turns, which carry no user message of their own
    for c in cases:
        assert all(t or i in c.held_out for i, t in enumerate(c.turns)), c.id


def test_single_turn_cases_have_one_turn_and_functions():
    for cat in AST_CATEGORIES:
        for c in load_cases(cat):
            assert len(c.turns) == 1 and c.functions and c.expects_call is None


def test_relevance_expectations():
    assert load_cases("irrelevance")[0].expects_call is False
    assert load_cases("live_relevance")[0].expects_call is True


def test_multi_turn_functions_come_from_docs_and_gt_aligned_with_turns():
    for cat in MULTI_TURN_CATEGORIES:
        for c in load_cases(cat):
            assert c.functions and len(c.ground_truth) == len(c.turns)
            assert all("response" in d or "parameters" in d for d in c.functions)


def test_held_out_functions_removed_from_visible_list():
    miss = load_cases("multi_turn_miss_func")
    assert all(c.held_out for c in miss)
    for c in miss:
        visible = {d["name"] for d in c.functions}
        hidden = {d["name"] for ds in c.held_out.values() for d in ds}
        assert hidden and not (visible & hidden), c.id
    assert not any(c.held_out for c in load_cases("multi_turn_base"))
    assert not any(c.held_out for c in load_cases("multi_turn_miss_param"))


def test_unknown_category_rejected():
    with pytest.raises(ValueError, match="unknown category"):
        load_cases("simple_java")


def test_subsample_is_deterministic_sized_and_keeps_file_order():
    a = load_subset("multiple", 30, seed=1)
    b = load_subset("multiple", 30, seed=1)
    assert [c.id for c in a] == [c.id for c in b] and len(a) == 30
    order = {c.id: i for i, c in enumerate(load_cases("multiple"))}
    idx = [order[c.id] for c in a]
    assert idx == sorted(idx)


def test_subsample_seed_changes_selection_and_categories_are_independent():
    s1 = {c.id for c in load_subset("parallel", 40, seed=1)}
    s2 = {c.id for c in load_subset("parallel", 40, seed=2)}
    assert s1 != s2 and len(s1 & s2) < 40


def test_subsample_larger_or_none_returns_everything_and_zero_returns_nothing():
    assert len(load_subset("live_relevance", 100)) == 16 and len(load_subset("live_relevance")) == 16
    assert load_subset("live_relevance", 0) == []


def test_subsample_is_stratified_proportionally():
    cases = load_cases("multiple")  # 164 cases with 2-3 functions, 36 with 4+
    got = Counter(c.stratum for c in stratified_sample(cases, 20, seed=0))
    assert got["fn=2-3"] in (16, 17) and got["fn=4+"] in (3, 4) and sum(got.values()) == 20
    mt = load_cases("multi_turn_base")
    full, sub = Counter(c.stratum for c in mt), Counter(c.stratum for c in stratified_sample(mt, 50, seed=0))
    for k, n in full.items():
        assert abs(sub[k] - 50 * n / 200) <= 1, (k, sub[k])


@pytest.mark.parametrize("n", [1, 7, 50, 199])
def test_subsample_exact_size_for_many_n(n):
    assert len(stratified_sample(load_cases("multiple"), n, seed=3)) == n
