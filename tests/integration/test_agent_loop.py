import json

import pytest

from a2e.agent.grammar import call_schema
from a2e.agent.loop import AgentLoop
from a2e.store import Store
from a2e.tasks.custom import generate
from tests.oracle import make_env, oracle_outputs
from tests.stubs import StubClient


def call(name, **args):
    return json.dumps([{"name": name, "arguments": args}])


FINAL = call("final_answer", answer="done")


@pytest.fixture
def env(tmp_path):
    reg, meta = make_env(tmp_path)
    return reg, meta, tmp_path


def loop(reg, outputs, tools=None, **kw):
    client = StubClient(outputs)
    return AgentLoop(client, reg, reg.specs(tools), **kw), client


def test_happy_path_calc_chain(env):
    reg, _, tmp = env
    task = generate("calc", 1, 2, seed=1)[0]
    lp, client = loop(reg, oracle_outputs(task, tmp), task.tool_names)
    tr = lp.run(task.prompt)
    assert tr.status == "final_answer" and tr.final_answer == "done" and len(tr.turns) == 3
    assert tr.score(task.steps).success
    assert all(t.results[0].ok for t in tr.turns)


def test_parse_error_stops_loop(env):
    reg, _, _ = env
    lp, _ = loop(reg, ["I think the answer is 5."])
    tr = lp.run("hi")
    assert tr.status == "parse_error" and len(tr.turns) == 1 and tr.turns[0].parse_error == "no_json"
    assert not tr.turns[0].calls and not tr.turns[0].results


def test_truncated_output_is_malformed_json(env):
    reg, _, _ = env
    tr = loop(reg, ['[{"name": "calculator", "arguments": {"expression": "1+'])[0].run("x")
    assert tr.status == "parse_error" and tr.turns[0].parse_error == "malformed_json"


def test_invalid_args_are_fed_back_to_the_model(env):
    reg, _, _ = env
    lp, client = loop(reg, [call("calculator"), FINAL])
    tr = lp.run("x")
    assert tr.turns[0].validation_errors[0] and not tr.turns[0].results[0].ok
    assert "invalid call" in client.prompts[1]  # the model sees the error on the next turn
    assert tr.status == "final_answer"


def test_unknown_tool_reported(env):
    reg, _, _ = env
    tr = loop(reg, [call("ghost"), FINAL])[0].run("x")
    assert "unknown tool" in tr.turns[0].validation_errors[0]


def test_tool_runtime_error_fed_back_and_loop_continues(env):
    reg, _, _ = env
    lp, client = loop(reg, [call("calculator", expression="1/0"), FINAL])
    tr = lp.run("x")
    assert tr.turns[0].validation_errors == [None] and not tr.turns[0].results[0].ok
    assert "ZeroDivisionError" in client.prompts[1] and tr.status == "final_answer"


def test_extra_unknown_argument_becomes_tool_error_not_crash(env):
    reg, _, _ = env
    tr = loop(reg, [call("calculator", expression="1+1", precision=2), FINAL])[0].run("x")
    assert tr.turns[0].validation_errors == [None]  # schema allows extras...
    assert not tr.turns[0].results[0].ok and "unexpected keyword" in tr.turns[0].results[0].output  # ...tool does not


def test_max_turns(env):
    reg, _, _ = env
    tr = loop(reg, [call("calculator", expression="1+1")])[0].run("x", max_turns=4)
    assert tr.status == "max_turns" and len(tr.turns) == 4


def test_server_error_recorded(env):
    reg, _, _ = env
    client = StubClient([call("calculator", expression="1+1")], fail_at=1)
    tr = AgentLoop(client, reg, reg.specs()).run("x")
    assert tr.status == "server_error" and len(tr.turns) == 2
    assert tr.turns[-1].error and tr.turns[-1].completion is None and tr.turns[-1].row(1, "t")["prefill_ms"] is None


def test_parallel_calls_in_one_turn(env):
    reg, _, _ = env
    two = json.dumps([{"name": "calculator", "arguments": {"expression": "2*3"}},
                      {"name": "calculator", "arguments": {"expression": "4*5"}}])
    lp, client = loop(reg, [two, FINAL])
    tr = lp.run("x")
    assert [r.output for r in tr.turns[0].results] == ["6", "20"]
    assert '"result": "6"' in client.prompts[1] and '"result": "20"' in client.prompts[1]


def test_final_answer_alongside_other_calls_ends_loop(env):
    reg, _, _ = env
    both = json.dumps([{"name": "calculator", "arguments": {"expression": "1+1"}},
                       {"name": "final_answer", "arguments": {"answer": "2"}}])
    tr = loop(reg, [both])[0].run("x")
    assert tr.status == "final_answer" and tr.final_answer == "2"


def test_invalid_final_answer_does_not_end_loop(env):
    reg, _, _ = env
    tr = loop(reg, [call("final_answer"), FINAL])[0].run("x")
    assert len(tr.turns) == 2 and tr.status == "final_answer"


def test_stage_timings_populated_and_windows_contiguous(env):
    reg, _, tmp = env
    task = generate("retrieval", 1, 4, corpus=env[1])[0]
    tr = loop(reg, oracle_outputs(task, tmp), task.tool_names)[0].run(task.prompt)
    for t in tr.turns:
        assert min(t.parse_ms, t.validate_ms, t.tool_wait_ms, t.template_ms, t.harness_ms) >= 0
        assert t.wall_ms > 0 and t.completion.prefill_ms > 0 and t.completion.decode_ms > 0
        w = t.windows
        assert w["parse"][1] == w["validate"][0] and w["validate"][1] == w["tool"][0]
        assert w["parse"][0] <= w["parse"][1] <= w["tool"][1]
    assert sum(t.tool_wait_ms for t in tr.turns) > 0


def test_tool_wait_reflects_slow_tool(env):
    import time

    from a2e.tools.base import Registry, Tool
    slow = Tool("slow", "x", {"type": "object", "properties": {}}, lambda: time.sleep(0.05) or "ok")
    fin = env[0].tools["final_answer"]
    reg = Registry([slow, fin])
    tr = AgentLoop(StubClient([call("slow"), FINAL]), reg, reg.specs()).run("x")
    assert tr.turns[0].tool_wait_ms >= 50 and tr.turns[0].results[0].exec_ms >= 50


def test_history_only_appends_so_prefix_is_cacheable(env):
    reg, _, tmp = env
    task = generate("notes", 1, 4, 2)[0]
    lp, client = loop(reg, oracle_outputs(task, tmp), task.tool_names)
    tr = lp.run(task.prompt)
    for a, b in zip(client.prompts, client.prompts[1:]):
        assert b.startswith(a)
    assert tr.turns[0].completion.cached_tokens == 0
    for prev, t in zip(client.prompts, tr.turns[1:]):
        assert t.completion.cached_tokens >= 0.9 * len(prev.split())


def test_assistant_history_is_raw_model_text(env):
    reg, _, _ = env
    odd = "```json\n" + call("calculator", expression="1+1") + "\n```"
    lp, client = loop(reg, [odd, FINAL])
    lp.run("x")
    assert odd in client.prompts[1]


@pytest.mark.parametrize("grammar", [False, True])
@pytest.mark.parametrize("cache_prompt", [False, True])
def test_request_options_pass_through(env, grammar, cache_prompt):
    reg, _, _ = env
    specs = reg.specs(["calculator", "final_answer"])
    client = StubClient([call("calculator", expression="1+1"), FINAL])
    AgentLoop(client, reg, specs, grammar=grammar, n_predict=77, cache_prompt=cache_prompt).run("x")
    for c in client.calls:
        assert c["n_predict"] == 77 and c["cache_prompt"] is cache_prompt
        assert c["json_schema"] == (call_schema(specs) if grammar else None)


def test_turn_rows_fit_store_schema(env, tmp_path):
    reg, _, tmp = env
    task = generate("calc", 1, 2)[0]
    tr = loop(reg, oracle_outputs(task, tmp), task.tool_names)[0].run(task.prompt)
    store = Store(tmp_path / "t.db")
    rid = store.insert("runs", {"config_json": "{}", "device": "mac", "phase": "measured", "started_at": 0.0})
    sc = tr.score(task.steps)
    for k, t in enumerate(tr.turns):
        store.insert("turns", t.row(rid, task.task_id, sc.steps[k].ok if k < len(sc.steps) else None))
    assert store.db.execute("SELECT count(*), sum(correct) FROM turns").fetchone() == (3, 2)


def test_partial_credit_when_second_step_wrong(env):
    reg, _, tmp = env
    task = generate("calc", 1, 3, seed=4)[0]
    outs = oracle_outputs(task, tmp)
    outs[1] = call("calculator", expression="0+0")
    sc = loop(reg, outs, task.tool_names)[0].run(task.prompt).score(task.steps)
    assert [s.ok for s in sc.steps] == [True, False, True] and not sc.success and sc.finished
