import json
import os

import httpx
import pytest

pytest.importorskip("mpmath")

from a2e.settings import WebSettings  # noqa: E402
from a2e.tasks.custom import FAMILY_DEPTH, FAMILY_ENV, FAMILY_TOOLS, fs_scenarios, generate  # noqa: E402
from a2e.tasks.envs import family_core_specs  # noqa: E402
from a2e.tools.bfcl_tools import as_text, bfcl_registry, class_docs, distractor_pool  # noqa: E402
from a2e.tools.web import WebSearch, html_to_text, web_tools  # noqa: E402


# ---------------------------------------------------------------- BFCL adapter
@pytest.mark.parametrize("cls,n", [("GorillaFileSystem", 18), ("MathAPI", 17)])
def test_every_documented_method_becomes_a_callable_tool(cls, n):
    reg = bfcl_registry([cls], fs_scenarios()[0]["config"])
    names = [d["name"] for d in class_docs(cls)]
    assert len(names) == n and set(names) <= set(reg.tools)
    for name in names:
        assert callable(reg.tools[name].fn) and reg.tools[name].parameters["type"] == "object"


def test_results_are_text_like_upstream():
    assert as_text({"a": 1}) == '{"a": 1}' and as_text(None) == "None" and as_text("x") == "x"
    import mpmath
    assert "mpf" in as_text({"result": mpmath.mpf(1)})  # non-JSON values fall back to str()


def test_each_registry_has_its_own_state():
    sc = fs_scenarios()[0]
    a, b = bfcl_registry(["GorillaFileSystem"], sc["config"]), bfcl_registry(["GorillaFileSystem"], sc["config"])
    a.execute("mkdir", {"dir_name": "only_in_a"})
    assert "only_in_a" in a.execute("ls", {}).output and "only_in_a" not in b.execute("ls", {}).output


def test_cwd_moves_into_the_starting_folder():
    sc = next(s for s in fs_scenarios() if any(f["path"] for f in s["files"]))
    f = next(f for f in sc["files"] if f["path"])
    reg = bfcl_registry(["GorillaFileSystem"], sc["config"], cwd=f["path"])
    assert f["name"] in reg.execute("ls", {"a": True}).output


def test_distractor_pool_is_every_shipped_bfcl_function():
    pool = distractor_pool()
    assert len(pool) == 130 and all(set(p) == {"name", "description", "parameters"} for p in pool)


# ---------------------------------------------------------------- curated scenarios
def test_curation_script_reproduces_the_committed_file():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "curate_fs_scenarios.py"
    spec = importlib.util.spec_from_file_location("curate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.curate() == json.loads(mod.OUT.read_text())


def test_curated_scenarios_respect_the_small_model_limits():
    data = json.loads((__import__("importlib").resources.files("a2e.data") / "fs_scenarios.json").read_text())
    lim = data["limits"]
    assert data["kept"] == len(data["scenarios"]) == 25 and data["distinct_scenarios_in_bfcl"] == 52
    for s in data["scenarios"]:
        assert lim["min_files"] <= len(s["files"]) <= lim["max_files"] and s["depth"] <= lim["max_depth"]
        assert sum(f["chars"] for f in s["files"]) <= lim["max_chars"] and s["sources"]
        assert list(s["config"]["root"]) == [s["root"]]  # only the folder BFCL loads


# ---------------------------------------------------------------- families
@pytest.mark.parametrize("family", ["fs_search", "fs_organize", "math", "fs_math", "web"])
def test_new_families_are_deterministic_and_sized_by_depth(family):
    lo, hi = FAMILY_DEPTH[family]
    for d in range(lo, hi + 1):
        a, b = generate(family, 4, d, seed=7), generate(family, 4, d, seed=7)
        assert [t.prompt for t in a] == [t.prompt for t in b] and all(len(t.steps) == d for t in a)
        assert {s.name for t in a for s in t.steps} <= set(FAMILY_TOOLS[family])
    with pytest.raises(ValueError):
        generate(family, 1, hi + 1)


def test_family_metadata_is_consistent():
    assert set(FAMILY_TOOLS) == set(FAMILY_DEPTH) == set(FAMILY_ENV)
    for f in FAMILY_TOOLS:
        assert [s["name"] for s in family_core_specs(f)] == FAMILY_TOOLS[f]


# ---------------------------------------------------------------- web search (mocked HTTP)
def mock_web(handler, key="k"):
    os.environ["A2E_UNIT_BRAVE"] = key
    return WebSearch(WebSettings(api_key_env="A2E_UNIT_BRAVE", base_url="https://search.test"),
                     http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_missing_key_is_a_clear_error():
    os.environ.pop("A2E_NO_SUCH_KEY", None)
    with pytest.raises(RuntimeError, match="A2E_NO_SUCH_KEY"):
        WebSearch(WebSettings(api_key_env="A2E_NO_SUCH_KEY"))


def test_search_maps_brave_results_to_bfcl_shape_and_sends_key_and_region():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(dict(req.url.params), token=req.headers.get("X-Subscription-Token"))
        return httpx.Response(200, json={"web": {"results": [
            {"title": "T", "url": "https://a.test/x", "description": "<b>snip</b>"}]}})

    ws = mock_web(handler)
    out = ws.search_engine_query("eiffel tower", max_results=50, region="fr-fr")
    assert out == [{"title": "T", "href": "https://a.test/x", "body": "snip"}]
    assert seen == {"q": "eiffel tower", "count": "10", "country": "FR", "token": "k"}


def test_fetch_only_allows_urls_from_this_tasks_search():
    def handler(req):
        if req.url.host == "search.test":
            return httpx.Response(200, json={"web": {"results": [{"title": "", "url": "https://a.test/p"}]}})
        return httpx.Response(200, text="<script>x()</script><p>Hello</p>")

    ws = mock_web(handler)
    assert "only URLs returned" in ws.fetch_url_content("https://a.test/p")["error"]
    ws.search_engine_query("q")
    assert ws.fetch_url_content("https://a.test/p", mode="truncate") == {"content": "Hello"}
    assert "only URLs returned" in ws.fetch_url_content("http://127.0.0.1/admin")["error"]
    assert "Unsupported" in ws.fetch_url_content("https://a.test/p", mode="pdf")["error"]


def test_http_errors_become_tool_errors_and_are_recorded():
    ws = mock_web(lambda req: httpx.Response(429, json={}))
    out = ws.search_engine_query("q")
    assert "error" in out and ws.errors and "HTTPStatusError" in ws.errors[0]


def test_page_text_is_capped_and_stripped():
    def handler(req):
        if req.url.host == "search.test":
            return httpx.Response(200, json={"web": {"results": [{"url": "https://a.test/p"}]}})
        return httpx.Response(200, text="<style>.a{}</style><div>" + "word " * 5000 + "</div>")

    ws = mock_web(handler)
    ws.search_engine_query("q")
    assert len(ws.fetch_url_content("https://a.test/p", "markdown")["content"]) == ws.s.max_page_chars
    assert html_to_text("<noscript>x</noscript><p>a</p><p>b</p>") == "a\nb"


def test_web_tools_use_bfcl_names_and_return_json():
    ws = mock_web(lambda req: httpx.Response(200, json={"web": {"results": []}}))
    tools = {t.name: t for t in web_tools(ws)}
    assert set(tools) == {"search_engine_query", "fetch_url_content"}
    assert json.loads(tools["search_engine_query"].fn(keywords="x")) == []


# ---------------------------------------------------------------- suite validation
def test_suite_accepts_new_families_and_checks_their_depths(tmp_path):
    from a2e.suite import ConfigError, load_suite
    base = '[suite]\nname = "t"\n[matrix]\ndevice = "pi5"\nmodel = "qwen3-4b-instruct-2507"\nquant = "q5_k_m"\n'
    p = tmp_path / "s.toml"
    p.write_text(base + '[[tasks]]\nkind = "custom"\nfamily = ["fs_search", "math", "web"]\ndepth = 2\n'
                 'distractors = "synthetic"\n')
    assert load_suite(p).tasks[0].distractors == "synthetic"
    p.write_text(base + '[[tasks]]\nkind = "custom"\nfamily = "web"\ndepth = 5\n')
    with pytest.raises(ConfigError, match="depth 2 to 4"):
        load_suite(p)
    p.write_text(base + '[[tasks]]\nkind = "custom"\nfamily = "fs_math"\ndepth = 2\n')
    with pytest.raises(ConfigError, match="depth 3 to 5"):
        load_suite(p)
    p.write_text(base + '[[tasks]]\nkind = "custom"\nfamily = "calc"\ndepth = 2\ndistractors = "fake"\n')
    with pytest.raises(ConfigError):
        load_suite(p)
