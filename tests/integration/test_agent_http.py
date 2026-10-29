"""AgentLoop through the real HTTP client against the fake server."""
import pytest

from a2e.agent.loop import AgentLoop
from a2e.client import Client
from tests.oracle import make_env


@pytest.fixture
def reg(tmp_path):
    return make_env(tmp_path)[0]


def test_apply_template_and_tokenize_endpoints(fake_server):
    c = Client(fake_server().url)
    p = c.apply_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert p.startswith("<|im_start|>system\ns<|im_end|>") and p.endswith("<|im_start|>assistant\n")
    assert c.tokenize("a b c") == 3


def test_loop_over_http_reports_parse_error_but_populates_timings(fake_server, reg):
    c = Client(fake_server().url)  # fake model answers "xxxx...", which is not a tool call
    tr = AgentLoop(c, reg, reg.specs(["calculator", "final_answer"]), n_predict=20).run("add 1 and 1")
    t = tr.turns[0]
    assert tr.status == "parse_error" and t.parse_error == "no_json"
    assert t.template_ms > 0 and t.completion.prefill_ms > 0 and t.completion.decode_ms > 0
    assert t.completion.prompt_tokens > 10 and t.wall_ms >= t.completion.wall_ms


def test_loop_over_http_server_error(fake_server, reg):
    c = Client(fake_server("--fail-completion", "500").url)
    tr = AgentLoop(c, reg, reg.specs()).run("x")
    assert tr.status == "server_error" and "500" in tr.turns[0].error


def test_json_schema_triggers_constrained_decode_cost(fake_server, reg):
    c = Client(fake_server("--grammar-overhead", "2.0").url)
    specs = reg.specs(["calculator", "final_answer"])
    off = AgentLoop(c, reg, specs, grammar=False, n_predict=100).run("x").turns[0].completion
    on = AgentLoop(c, reg, specs, grammar=True, n_predict=100).run("x").turns[0].completion
    assert on.decode_ms == pytest.approx(off.decode_ms * 2.0, rel=0.01)


def test_connection_refused_is_server_error(fake_server, reg):
    srv = fake_server()
    c = Client(srv.url)
    srv.stop()
    tr = AgentLoop(c, reg, reg.specs()).run("x")
    assert tr.status == "server_error"
