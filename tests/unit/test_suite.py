import json
from pathlib import Path

import pytest

from a2e.settings import ModelEntry, Settings, load_settings
from a2e.suite import (AXES_FOR_KIND, ConfigError, expand, job_key, load_suite, lock_entry, plan_summary,
                       server_key)

REPO = Path(__file__).resolve().parents[2]
SUITES = sorted((REPO / "configs" / "suites").glob("*.toml"))


def settings():
    return Settings(models={k: ModelEntry(name=k, params_b=v) for k, v in
                            {"qwen3-4b": 4.0, "gemma3-4b": 4.0, "big-8b": 8.0}.items()})


def suite(tmp_path, body, name="t"):
    p = tmp_path / "s.toml"
    p.write_text(f'[suite]\nname = "{name}"\n' + body)
    return load_suite(p)


BASE = """
[matrix]
device = ["pi5", "mac"]
model = ["qwen3-4b", "big-8b"]
quant = ["q4_0", "q8_0"]
[axes]
grammar = [false, true]
decode_len = [128, 512]
schema_len = [256, 1024]
"""


@pytest.mark.parametrize("path", SUITES, ids=[p.stem for p in SUITES])
def test_every_shipped_suite_parses_and_expands(path):
    s = load_suite(path)
    jobs = expand(s, load_settings(REPO / "configs", local=REPO / "configs" / "__none__.toml"))
    assert jobs and len({j["job_key"] for j in jobs}) == len(jobs)


def test_shipped_suites_match_the_lock_file():
    lock = json.loads((REPO / "configs" / "suites.lock.json").read_text())
    st = load_settings(REPO / "configs", local=REPO / "configs" / "__none__.toml")
    current = {load_suite(p).suite.name: lock_entry(load_suite(p), expand(load_suite(p), st)) for p in SUITES}
    assert current == lock, "a suite (or the expansion code) changed: run `a2e suite-lock` if that was intended"


def test_suite_by_name_resolves_to_configs_suites():
    assert load_suite("smoke").suite.name == "smoke"
    with pytest.raises(ConfigError, match="not found"):
        load_suite("no-such-suite")


@pytest.mark.parametrize("body,msg", [
    (BASE + '[[tasks]]\nkind = "plain"\nbogus = 1\n', "bogus"),
    (BASE + 'extra_table = 1\n[[tasks]]\nkind = "plain"\n', "extra_table"),
    (BASE + '[[tasks]]\nkind = "custom"\nfamily = "dance"\ndepth = 2\n', "unknown family"),
    (BASE + '[[tasks]]\nkind = "custom"\nfamily = "retrieval"\ndepth = 1\n', "depth 2 to 5"),
    (BASE + '[[tasks]]\nkind = "custom"\nfamily = "calc"\ndepth = 6\n', "depth 1 to 5"),
    (BASE + '[[tasks]]\nkind = "custom"\nfamily = "notes"\ndepth = 2\nn_args = 6\n', "n_args"),
    (BASE + '[[tasks]]\nkind = "custom"\nfamily = "calc"\n', "need family and depth"),
    (BASE + '[[tasks]]\nkind = "bfcl"\ncategory = "simple_java"\n', "unknown BFCL"),
    (BASE + '[[tasks]]\nkind = "bfcl"\n', "need category"),
    (BASE + '[[tasks]]\nkind = "plain"\ncategory = "simple_python"\n', "only applies"),
    (BASE + '[[tasks]]\nkind = "bfcl"\ncategory = "multi_turn_base"\n', "ctx >= 16384"),
    (BASE.replace('["pi5", "mac"]', '["android"]') + '[[tasks]]\nkind = "plain"\n', "device"),
    (BASE.replace("[128, 512]", "[8]") + '[[tasks]]\nkind = "plain"\n', ">= 16"),
    (BASE, "tasks"),
    ("[[tasks]]\nkind = 'plain'\n", "matrix"),
    ("this is = = not toml", r"s\.toml"),
])
def test_invalid_suites_are_rejected_with_a_message(tmp_path, body, msg):
    with pytest.raises(ConfigError, match=msg):
        suite(tmp_path, body)


def test_each_kind_only_expands_the_axes_it_uses(tmp_path):
    s = suite(tmp_path, BASE + '[[tasks]]\nkind = "plain"\nn = 2\n[[tasks]]\nkind = "custom"\nfamily = "calc"\n'
              'depth = [2, 3]\n[[tasks]]\nkind = "bfcl"\ncategory = ["simple_python", "multiple"]\nn = 5\n')
    jobs = expand(s, settings())
    cells = 2 * 2 * 2  # device x model x quant
    by = lambda k: [j for j in jobs if j["kind"] == k]  # noqa: E731
    assert len(by("plain")) == cells * 2  # decode_len only
    assert len(by("custom")) == cells * 2 * 2 * 2 * 2  # grammar x decode x schema x depth
    assert len(by("bfcl")) == cells * 2 * 2 * 2  # grammar x decode x category
    for j in jobs:
        for axis in ("grammar", "schema_len"):
            assert (axis in j) == (axis in AXES_FOR_KIND[j["kind"]])


def test_task_axis_overrides_suite_axes(tmp_path):
    s = suite(tmp_path, BASE + '[[tasks]]\nkind = "plain"\ndecode_len = [64]\n')
    assert {j["decode_len"] for j in expand(s, settings())} == {64}


def test_duplicate_task_entries_run_once(tmp_path):
    s = suite(tmp_path, BASE + '[[tasks]]\nkind = "plain"\nn = 2\n[[tasks]]\nkind = "plain"\nn = 2\n')
    assert len(expand(s, settings())) == 2 * 2 * 2 * 2


def test_jobs_are_grouped_so_each_model_loads_once(tmp_path):
    s = suite(tmp_path, BASE + '[[tasks]]\nkind = "plain"\n[[tasks]]\nkind = "custom"\nfamily = "calc"\ndepth = 2\n')
    keys = [server_key(j) for j in expand(s, settings())]
    transitions = sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    assert transitions == len(set(keys)) - 1


def test_expansion_is_deterministic_and_machine_independent(tmp_path):
    s = suite(tmp_path, BASE + '[[tasks]]\nkind = "plain"\n')
    a = expand(s, settings())
    other = settings()
    other.threads, other.models_dir, other.llama_server = 4, "/elsewhere", "/x"
    assert a == expand(s, settings()) == expand(s, other)


def test_job_key_changes_with_any_field():
    cfg = {"model": "m", "quant": "q4_0", "grammar": False, "n": 5}
    keys = {job_key(cfg)} | {job_key({**cfg, k: v}) for k, v in
                             [("model", "m2"), ("quant", "q8_0"), ("grammar", True), ("n", 6), ("new", 1)]}
    assert len(keys) == 6 and job_key(dict(reversed(list(cfg.items())))) == job_key(cfg)


def test_suite_hash_ignores_formatting_and_tracks_values(tmp_path):
    body = BASE + '[[tasks]]\nkind = "plain"\n'
    h1 = suite(tmp_path, body).hash
    h2 = suite(tmp_path, "# a comment\n" + body.replace('kind = "plain"', 'kind="plain"') + "# trailing\n").hash
    h3 = suite(tmp_path, body.replace("[128, 512]", "[128]")).hash
    assert h1 == h2 != h3


def test_feasibility_rules(tmp_path):
    s = suite(tmp_path, BASE + '[[tasks]]\nkind = "plain"\ngovernor = ["", "performance"]\n')
    jobs = expand(s, settings())
    mem = [j for j in jobs if j["skip_reason"] and "memory" in j["skip_reason"]]
    assert mem and all(j["device"] == "pi5" and j["quant"] == "q8_0" and j["model"] == "big-8b" for j in mem)
    gov = [j for j in jobs if j["skip_reason"] and "governor" in j["skip_reason"]]
    assert gov and all(j["device"] == "mac" and j["governor"] for j in gov)
    assert not any(j["skip_reason"] for j in jobs if j["device"] == "pi5" and j["model"] == "qwen3-4b"
                   and not j["governor"])


def test_unknown_model_fails_at_expansion(tmp_path):
    s = suite(tmp_path, BASE.replace('"big-8b"', '"mystery"') + '[[tasks]]\nkind = "plain"\n')
    with pytest.raises(KeyError, match="mystery"):
        expand(s, settings())


def test_plan_summary_counts(tmp_path):
    s = suite(tmp_path, BASE + '[repeats]\nwarmup = 1\nmeasured = 2\n[[tasks]]\nkind = "plain"\n')
    p = plan_summary(expand(s, settings()))
    assert p["jobs"] == 16 and p["skipped"] == 2 and p["runnable"] == 14 and p["runs"] == 14 * 3
    assert p["server_loads"] == 7 and p["by_kind"]["plain"] == 14
