"""Agent loop against a real llama-server: checks the endpoints and behaviours the stubs assume."""
import pytest

from a2e.agent.loop import AgentLoop
from a2e.agent.schema_gen import pad_tools
from a2e.client import Client
from a2e.tasks.custom import FAMILY_TOOLS, generate
from a2e.tools.builtin import build_registry, make_corpus

from .conftest import needs_llama

pytestmark = needs_llama


@pytest.fixture
def reg(tmp_path):
    make_corpus(tmp_path, 10)
    return build_registry(tmp_path)


def test_apply_template_endpoint_renders_roles(real_server):
    c = Client(real_server().url)
    p = c.apply_template([{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}])
    assert "SYS" in p and "USR" in p and p.index("SYS") < p.index("USR"), "chat template did not render both roles"


def test_tokenize_endpoint_counts_tokens(real_server):
    c = Client(real_server().url)
    assert c.tokenize("hello world") >= 2
    assert c.tokenize("hello world " * 50) > c.tokenize("hello world")


def test_padded_schema_hits_target_with_real_tokenizer(real_server, reg):
    c = Client(real_server().url)
    core = reg.specs(FAMILY_TOOLS["calc"])
    for target in (256, 1024):
        _, n = pad_tools(core, target, c.tokenize)
        assert abs(n - target) / target <= 0.05


def test_grammar_on_always_yields_parseable_calls(real_server, reg):
    """json_schema must be honoured end to end: constrained output parses and validates on every first turn."""
    c = Client(real_server().url)
    specs = reg.specs(FAMILY_TOOLS["calc"])
    for t in generate("calc", 3, 1, seed=0):
        turn = AgentLoop(c, reg, specs, grammar=True, n_predict=96).run(t.prompt, max_turns=1).turns[0]
        assert turn.parse_error is None and turn.calls, f"constrained output did not parse: {turn.text!r}"
        assert turn.validation_errors == [None]


def test_grammar_off_vs_on_report_both_costs(real_server, reg):
    c = Client(real_server().url)
    specs = reg.specs(FAMILY_TOOLS["calc"])
    t = generate("calc", 1, 1, seed=0)[0]
    for g in (False, True):
        turn = AgentLoop(c, reg, specs, grammar=g, n_predict=96).run(t.prompt, max_turns=1).turns[0]
        print(f"grammar={g}: decode {turn.completion.decode_tok_s:.1f} tok/s, text={turn.text!r}")
        assert turn.completion.decode_ms > 0


def test_multi_turn_reuses_schema_prefix(real_server, reg):
    """Turn 2+ should reuse the cached prefix. If not, the template re-renders earlier turns differently
    (e.g. Qwen3 strips earlier thinking), which is itself a finding worth reporting."""
    c = Client(real_server().url)
    specs, _ = pad_tools(reg.specs(FAMILY_TOOLS["calc"]), 512, c.tokenize)
    task = generate("calc", 1, 3, seed=1)[0]
    tr = AgentLoop(c, reg, specs, grammar=True, n_predict=96).run(task.prompt, max_turns=3)
    if len(tr.turns) < 2:
        pytest.skip("model finished in one turn")
    later = [t.completion.cached_tokens / t.completion.prompt_tokens for t in tr.turns[1:]]
    assert max(later) > 0.5, f"prefix cache hit rates on later turns: {later}"


def test_depth1_calc_task_end_to_end(real_server, reg):
    c = Client(real_server().url)
    t = generate("calc", 1, 1, seed=3)[0]
    tr = AgentLoop(c, reg, reg.specs(t.tool_names), grammar=True, n_predict=96).run(t.prompt)
    print("status", tr.status, "score", tr.score(t.steps).steps)
    assert tr.turns[0].calls, "no tool call produced even with grammar"
