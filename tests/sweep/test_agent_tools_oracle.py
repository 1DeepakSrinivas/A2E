"""New tool families (BFCL file system and math, live web search): the oracle passes every generated task,
and corrupted steps or a wrong final state fail. Web uses a local fake of the Brave API."""
import json

import pytest
from jsonschema import Draft202012Validator

from a2e.agent.grammar import call_schema
from a2e.agent.loop import AgentLoop
from a2e.agent.schema_gen import pad_tools
from a2e.settings import WebSettings
from a2e.tasks.custom import FAMILY_DEPTH, fs_scenarios, generate
from a2e.tasks.envs import build_task_registry, family_core_specs
from a2e.tools.bfcl_tools import distractor_pool
from tests.fakes.fake_brave import KEY, FakeBrave
from tests.oracle import oracle_outputs_any
from tests.stubs import StubClient

pytest.importorskip("mpmath")
NEW = ["fs_search", "fs_organize", "math", "fs_math", "web"]
CASES = [(f, d) for f in NEW for d in range(FAMILY_DEPTH[f][0], FAMILY_DEPTH[f][1] + 1)]


@pytest.fixture(scope="module")
def web():
    import os
    fake = FakeBrave()
    os.environ["A2E_TEST_BRAVE"] = KEY
    yield WebSettings(api_key_env="A2E_TEST_BRAVE", base_url=fake.base)
    fake.close()


def run(task, tmp, web, outs=None):
    outs = outs or oracle_outputs_any(task, tmp / "oracle", web)
    reg = build_task_registry(task, tmp / "run", 1, web)
    tr = AgentLoop(StubClient(outs), reg, family_core_specs(task.family)).run(task.prompt, max_turns=len(task.steps) + 2)
    sc = tr.score(task.steps)
    if task.state_check:
        sc.state_ok = task.state_check(reg)
    return tr, sc, reg


@pytest.mark.parametrize("family,depth", CASES)
def test_oracle_solves_every_generated_task(tmp_path, web, family, depth):
    tasks = [t for seed in range(4) for t in generate(family, 6, depth, seed=seed)]
    for k, t in enumerate(tasks):
        tr, sc, _ = run(t, tmp_path / str(k), web)
        assert tr.status == "final_answer" and sc.success, (t.task_id, [s.note for s in sc.steps], sc.state_ok)
        assert len(t.steps) == depth


def test_generated_tasks_cover_many_curated_scenarios():
    used = {t.env["scenario"] for f in ("fs_search", "fs_organize", "fs_math") for s in range(10)
            for t in generate(f, 10, 3, seed=s)}
    assert len(used) >= len(fs_scenarios()) // 2


@pytest.mark.parametrize("family", NEW)
def test_corrupting_any_step_fails_that_step(tmp_path, web, family):
    depth = FAMILY_DEPTH[family][1]
    task = generate(family, 1, depth, seed=3)[0]
    for k, exp in enumerate(task.steps):
        outs = oracle_outputs_any(task, tmp_path / f"o{k}", web)
        bad = {"name": "ls", "arguments": {}} if exp.name != "ls" else {"name": "pwd", "arguments": {}}
        if family in ("math",):
            bad = {"name": "min_value", "arguments": {"numbers": [1, 2]}}
        if family == "web":
            bad = {"name": "note_list", "arguments": {}} if exp.name != "note_list" else \
                {"name": "search_engine_query", "arguments": {"keywords": "x"}}
        outs[k] = json.dumps([bad])
        tr, sc, _ = run(task, tmp_path / f"r{k}", web, outs)
        assert all(s.ok for s in sc.steps[:k]) and not sc.steps[k].ok and not sc.success, (family, k)


def test_wrong_values_fail_even_with_the_right_tool(tmp_path, web):
    t = generate("math", 1, 5, seed=1)[0]
    outs = oracle_outputs_any(t, tmp_path / "o", web)
    call = json.loads(outs[2])[0]
    call["arguments"]["number"] += 5
    outs[2] = json.dumps([call])
    assert not run(t, tmp_path / "r", web, outs)[1].steps[2].ok
    t = generate("fs_math", 1, 3, seed=2)[0]
    outs = oracle_outputs_any(t, tmp_path / "o2", web)
    call = json.loads(outs[2])[0]
    call["arguments"]["a"] += 1
    outs[2] = json.dumps([call])
    assert not run(t, tmp_path / "r2", web, outs)[1].steps[2].ok


def test_rounded_copied_numbers_are_accepted_within_tolerance(tmp_path, web):
    t = generate("math", 1, 3, seed=5)[0]
    outs = oracle_outputs_any(t, tmp_path / "o", web)
    call = json.loads(outs[2])[0]
    call["arguments"]["number"] = round(call["arguments"]["number"], 4)
    outs[2] = json.dumps([call])
    assert run(t, tmp_path / "r", web, outs)[1].success


def test_fs_organize_state_check_catches_extra_damage(tmp_path, web):
    t = generate("fs_organize", 1, 2, seed=0)[0]
    outs = oracle_outputs_any(t, tmp_path / "o", web)
    other = next(n for n in json.loads(build_task_registry(t, tmp_path / "x", 1).execute("ls", {}).output)
                 ["current_directory_content"] if n != json.loads(outs[1])[0]["arguments"]["source"]
                 and n != json.loads(outs[0])[0]["arguments"]["dir_name"]) if False else None
    rm = {"name": "mkdir", "arguments": {"dir_name": "unexpected_extra_folder"}}
    outs = outs[:-1] + [json.dumps([rm]), outs[-1]]
    tr, sc, _ = run(t, tmp_path / "r", web, outs)
    assert all(s.ok for s in sc.steps) and sc.state_ok is False and not sc.success


def test_web_fetch_must_use_a_url_from_the_search(tmp_path, web):
    t = generate("web", 1, 2, seed=0)[0]
    outs = oracle_outputs_any(t, tmp_path / "o", web)
    outs[1] = json.dumps([{"name": "fetch_url_content", "arguments": {"url": "http://169.254.169.254/latest"}}])
    tr, sc, reg = run(t, tmp_path / "r", web, outs)
    assert not sc.steps[1].ok and "only URLs returned" in tr.turns[1].results[0].output


@pytest.mark.parametrize("family", NEW)
@pytest.mark.parametrize("target", [1024, 2048])
def test_grammar_admits_every_oracle_call_with_bfcl_distractors(tmp_path, web, family, target):
    count = lambda t: max(len(t) // 4, 1)  # noqa: E731
    try:
        specs, _ = pad_tools(family_core_specs(family), target, count, seed=2, pool=list(distractor_pool()))
    except Exception:  # core too large for this approximate count: the runner skips such jobs
        pytest.skip("core tools exceed target")
    v = Draft202012Validator(call_schema(specs))
    t = generate(family, 1, FAMILY_DEPTH[family][1], seed=0)[0]
    for out in oracle_outputs_any(t, tmp_path / "o", web):
        assert v.is_valid(json.loads(out)), out


@pytest.mark.parametrize("family", NEW)
def test_same_family_gives_identical_tool_prefix_across_tasks(tmp_path, web, family):
    """Prefix caching only helps if the tool text is byte-identical across a run's tasks."""
    count = lambda t: max(len(t) // 4, 1)  # noqa: E731
    specs, _ = pad_tools(family_core_specs(family), 2048, count, seed=7, pool=list(distractor_pool()))
    prompts = []
    for k, t in enumerate(generate(family, 2, FAMILY_DEPTH[family][0], seed=3)):
        client = StubClient(oracle_outputs_any(t, tmp_path / f"o{k}", web))
        AgentLoop(client, build_task_registry(t, tmp_path / f"r{k}", 1, web), specs).run(t.prompt, max_turns=1)
        prompts.append(client.prompts[0])
    assert prompts[0].split("</tools>")[0] == prompts[1].split("</tools>")[0] and prompts[0] != prompts[1]
