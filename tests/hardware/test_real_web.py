"""Live Brave Search API: checks the endpoint, header and response fields tools/web.py assumes.
Needs BRAVE_SEARCH_API_KEY and network access. Uses one or two searches of your quota."""
import os

import pytest

from a2e.settings import WebSettings
from a2e.tools.web import WebSearch

pytestmark = pytest.mark.skipif(not os.environ.get("BRAVE_SEARCH_API_KEY"), reason="set BRAVE_SEARCH_API_KEY")


def test_real_search_returns_bfcl_shaped_results():
    ws = WebSearch(WebSettings())
    out = ws.search_engine_query("Eiffel Tower completed year", max_results=3)
    assert isinstance(out, list) and out, f"no results or an error: {out} {ws.errors}"
    assert all(set(r) == {"title", "href", "body"} and r["href"].startswith("http") for r in out)


def test_real_fetch_of_a_returned_url_gives_text():
    ws = WebSearch(WebSettings())
    hits = ws.search_engine_query("Pride and Prejudice author", max_results=3)
    assert isinstance(hits, list) and hits
    pages = [ws.fetch_url_content(h["href"], mode="truncate") for h in hits]
    assert any(p.get("content") for p in pages), f"no page could be fetched: {ws.errors}"
