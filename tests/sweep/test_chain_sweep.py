"""Sweeps over task family x depth x arg count x schema length x grammar, driven by the oracle model."""
import json
import itertools

import pytest
from jsonschema import Draft202012Validator

from a2e.agent.grammar import call_schema
from a2e.agent.loop import AgentLoop
from a2e.agent.prompt import render_tools
from a2e.agent.schema_gen import SchemaTooLarge, pad_tools
from a2e.tasks.custom import FAMILY_ENV, FAMILY_TOOLS, generate

BUILTIN = [f for f in FAMILY_TOOLS if FAMILY_ENV[f] == "builtin"]  # new families: test_agent_tools_oracle.py
from tests.oracle import make_env, oracle_outputs
from tests.stubs import StubClient

CASES = ([("calc", d, 1) for d in range(1, 6)] + [("notes", d, a) for d in range(1, 6) for a in range(1, 6)]
         + [("retrieval", d, 1) for d in range(2, 6)])


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("sw"))[1]


def run_oracle(task, tmp, reg, specs=None, grammar=False):
    client = StubClient(oracle_outputs(task, tmp))
    lp = AgentLoop(client, reg, specs or reg.specs(task.tool_names), grammar=grammar)
    return lp.run(task.prompt, max_turns=len(task.steps) + 2), client


@pytest.mark.parametrize("family,depth,n_args", CASES)
def test_oracle_solves_every_chain(tmp_path, corpus, family, depth, n_args):
    reg, _ = make_env(tmp_path, "run")
    for k, task in enumerate(generate(family, 3, depth, n_args, seed=5, corpus=corpus)):
        tr, _ = run_oracle(task, tmp_path / str(k), reg)
        sc = tr.score(task.steps)
        assert tr.status == "final_answer" and sc.success and sc.n_correct == len(task.steps), task.task_id
        assert len(tr.turns) == len(task.steps) + 1


@pytest.mark.parametrize("family,depth", [("calc", 3), ("calc", 5), ("notes", 4), ("retrieval", 3), ("retrieval", 5)])
def test_corrupting_step_k_fails_exactly_that_step_and_not_earlier_ones(tmp_path, corpus, family, depth):
    reg, _ = make_env(tmp_path, "run")
    task = generate(family, 1, depth, 2, seed=9, corpus=corpus)[0]
    for k in range(depth):
        outs = oracle_outputs(task, tmp_path / f"o{k}")
        bad = "note_list" if task.steps[k].name != "note_list" else "calculator"
        outs[k] = json.dumps([{"name": bad, "arguments": {}}])
        tr = AgentLoop(StubClient(outs), reg, reg.specs(task.tool_names)).run(task.prompt, max_turns=depth + 2)
        sc = tr.score(task.steps)
        assert all(s.ok for s in sc.steps[:k]) and not sc.steps[k].ok and not sc.success, (family, depth, k)


@pytest.mark.parametrize("stop_at", [0, 1, 2])
def test_finishing_early_is_a_failure_with_missing_steps(tmp_path, stop_at):
    reg, _ = make_env(tmp_path, "run")
    task = generate("calc", 1, 4, seed=2)[0]
    outs = oracle_outputs(task, tmp_path / "o")
    outs = outs[:stop_at] + [outs[-1]]
    tr = AgentLoop(StubClient(outs), reg, reg.specs(task.tool_names)).run(task.prompt)
    sc = tr.score(task.steps)
    assert sc.finished and not sc.success and sc.n_correct == stop_at
    assert "wrong tool 'final_answer'" in sc.steps[stop_at].note  # it did call something, just the wrong tool
    assert all("missing" in s.note for s in sc.steps[stop_at + 1:])


def count(text):
    return StubClient([]).tokenize(text)


SCHEMA_CASES = [(f, t) for f in BUILTIN for t in (256, 1024, 2048)]


@pytest.mark.parametrize("family,target", SCHEMA_CASES)
def test_padded_schema_still_solves_chains_and_grammar_admits_oracle(tmp_path, corpus, family, target):
    reg, _ = make_env(tmp_path, "run")
    core = reg.specs(FAMILY_TOOLS[family])
    try:
        specs, n = pad_tools(core, target, count, seed=3)
    except SchemaTooLarge:
        pytest.skip(f"{family} core tools exceed {target} (approx tokens); skip this cell in real sweeps")
    assert abs(n - target) / target <= 0.05
    depth = 3
    task = generate(family, 1, depth, 2, seed=1, corpus=corpus)[0]
    validator = Draft202012Validator(call_schema(specs))
    for out in oracle_outputs(task, tmp_path / "o"):
        assert validator.is_valid(json.loads(out)), out  # the grammar must not forbid a correct call
    tr, client = run_oracle(task, tmp_path / "o2", reg, specs=specs, grammar=True)
    assert tr.score(task.steps).success
    assert all(c["json_schema"] == call_schema(specs) for c in client.calls)


@pytest.mark.parametrize("family", BUILTIN)
def test_same_tools_give_identical_prefix_across_tasks(tmp_path, corpus, family):
    """Prefix caching only helps if the schema text is byte-identical across tasks in a session."""
    reg, _ = make_env(tmp_path, "run")
    specs, _ = pad_tools(reg.specs(FAMILY_TOOLS[family]), 1024, count, seed=7)
    tasks = generate(family, 2, 3 if family == "retrieval" else 2, 2, seed=3, corpus=corpus)
    prompts = []
    for k, t in enumerate(tasks):
        client = StubClient(oracle_outputs(t, tmp_path / f"o{k}"))
        AgentLoop(client, reg, specs).run(t.prompt, max_turns=2)
        prompts.append(client.prompts[0])
    marker = "</tools>"
    assert prompts[0].split(marker)[0] == prompts[1].split(marker)[0]
    assert prompts[0] != prompts[1]


def test_prefix_cache_hit_rate_grows_over_turns_and_drops_when_disabled(tmp_path, corpus):
    reg, _ = make_env(tmp_path, "run")
    specs, _ = pad_tools(reg.specs(FAMILY_TOOLS["calc"]), 1024, count)
    task = generate("calc", 1, 4, seed=0)[0]
    rates = {}
    for cache in (True, False):
        client = StubClient(oracle_outputs(task, tmp_path / f"o{cache}"))
        tr = AgentLoop(client, reg, specs, cache_prompt=cache).run(task.prompt)
        rates[cache] = [t.completion.cached_tokens / t.completion.prompt_tokens for t in tr.turns]
    assert rates[True][0] == 0 and all(r > 0.8 for r in rates[True][1:])
    assert all(r == 0 for r in rates[False])


@pytest.mark.parametrize("target", [256, 1024, 2048])
def test_schema_prefill_share_grows_with_schema_length(tmp_path, target):
    """The first-turn prefill cost should scale with the padded schema, with grammar off."""
    reg, _ = make_env(tmp_path, "run")
    specs, _ = pad_tools(reg.specs(["calculator", "final_answer"]), target, count)
    client = StubClient(['[{"name":"final_answer","arguments":{"answer":"x"}}]'])
    t = AgentLoop(client, reg, specs).run("what is 1+1").turns[0]
    assert t.completion.prompt_tokens >= len(render_tools(specs).split()) * 0.9


def test_all_prompts_unique_across_sweep(tmp_path, corpus):
    prompts = {t.prompt for fam, d, a in CASES for t in generate(fam, 3, d, a, seed=0, corpus=corpus)}
    assert len(prompts) > 0.9 * len(CASES) * 3
