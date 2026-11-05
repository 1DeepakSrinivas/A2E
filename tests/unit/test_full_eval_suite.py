"""The shipped full_eval suite: 50 tasks per model, quant and grammar setting, spread over every difficulty and horizon."""
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from a2e.bfcl.load import MULTI_TURN_CATEGORIES
from a2e.settings import load_settings
from a2e.suite import expand, load_suite

REPO = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def suite():
    return load_suite(REPO / "configs" / "suites" / "full_eval.toml")


@pytest.fixture(scope="module")
def jobs(suite):
    return expand(suite, load_settings(REPO / "configs", local=REPO / "configs" / "__none__.toml"))


def by_config(jobs):
    out = defaultdict(list)
    for j in jobs:
        out[(j["device"], j["model"], j["quant"], j["grammar"])].append(j)
    return out


def test_every_configuration_gets_exactly_50_tasks(jobs):
    groups = by_config(jobs)
    assert len(groups) == 2 * 4 * 2  # devices x (model, quant) pairs x grammar
    assert {sum(j["n"] for j in g) for g in groups.values()} == {50}


def test_one_measured_run_and_no_warmup(suite):
    assert (suite.repeats.warmup, suite.repeats.measured) == (0, 1)


def test_job_counts(jobs):
    per_config = Counter(len(g) for g in by_config(jobs).values())
    assert per_config == {21: 16}  # 8 BFCL categories + 13 custom cells per configuration
    assert len(jobs) == 336 and len({j["job_key"] for j in jobs}) == 336 and not any(j["skip_reason"] for j in jobs)


def test_both_grammar_settings_and_both_quants_and_two_models(suite):
    assert suite.axes.grammar == [False, True]
    assert len(suite.matrix.model) == 2 and len(suite.matrix.quant) == 3  # 3 quants in all, 2 per model


def test_bfcl_covers_single_and_multi_turn_and_irrelevance(jobs):
    cats = {j["category"] for j in jobs if j["kind"] == "bfcl"}
    assert {"simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"} <= cats
    assert set(MULTI_TURN_CATEGORIES) <= cats
    assert sum(j["n"] for j in jobs if j["kind"] == "bfcl" and j["device"] == "mac" and j["model"] == "qwen3-4b-instruct-2507"
               and j["quant"] == "q5_k_m" and j["grammar"] is False) == 24


def test_custom_chains_cover_horizons_2_to_5_and_every_offline_family(jobs):
    custom = [j for j in jobs if j["kind"] == "custom"]
    assert {j["depth"] for j in custom} == {2, 3, 4, 5}
    assert {j["family"] for j in custom} == {"calc", "notes", "retrieval", "fs_search", "fs_organize", "math", "fs_math"}


def test_the_live_web_family_is_left_out(jobs):
    assert "web" not in {j.get("family") for j in jobs}


def test_context_is_large_enough_for_multi_turn(suite):
    assert suite.server.ctx >= 16384


def test_placeholders_are_marked_for_editing():
    text = (REPO / "configs" / "suites" / "full_eval.toml").read_text()
    assert text.count("EDIT") >= 2 and "scripts/report_full_eval.sh" in text
