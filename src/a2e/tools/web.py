"""Live web search with BFCL's function names and arguments, served by the Brave Search API.

`search_engine_query(keywords, max_results=10, region="wt-wt")` returns BFCL's result shape, a list of
{"title", "href", "body"}. `fetch_url_content(url, mode="raw")` returns {"content": ...}, and only fetches URLs that a
search in the same task returned, so the model cannot make the device fetch arbitrary or local-network addresses.
Network and API errors come back as {"error": ...} so the agent sees them like any tool error.

The Brave endpoint (`GET {base_url}/res/v1/web/search`, header `X-Subscription-Token`, JSON `web.results[]` with
`title`, `url`, `description`) is from memory; tests/hardware/test_real_web.py checks it with a real key.
"""
from __future__ import annotations

import json
from html.parser import HTMLParser

import httpx

from a2e.bfcl.convert import tool_spec
from a2e.settings import WebSettings
from a2e.tools.base import Tool
from a2e.tools.bfcl_tools import class_docs

UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
      "A2E-benchmark")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    return "\n".join(p.parts)


class WebSearch:
    def __init__(self, settings: WebSettings, http: httpx.Client | None = None):
        key = settings.api_key()
        if not key:
            raise RuntimeError(f"web search needs a Brave API key in the {settings.api_key_env} environment variable")
        self.s, self.key = settings, key
        self.http = http or httpx.Client(timeout=settings.timeout_s, follow_redirects=True)
        self.returned: set[str] = set()  # URLs the search gave this task
        self.errors: list[str] = []

    def search_engine_query(self, keywords: str, max_results: int = 10, region: str = "wt-wt"):
        params = {"q": keywords, "count": max(1, min(int(max_results or 10), self.s.max_results))}
        if region and region != "wt-wt":
            params["country"] = region.split("-")[0].upper()
        try:
            r = self.http.get(f"{self.s.base_url}/res/v1/web/search", params=params,
                              headers={"X-Subscription-Token": self.key, "Accept": "application/json"})
            r.raise_for_status()
            items = (r.json().get("web") or {}).get("results") or []
        except (httpx.HTTPError, ValueError) as e:
            self.errors.append(f"search: {type(e).__name__}: {e}")
            return {"error": f"search failed: {e}"}
        out = [{"title": i.get("title", ""), "href": i.get("url", ""), "body": html_to_text(i.get("description", ""))}
               for i in items[: params["count"]]]
        self.returned.update(x["href"] for x in out if x["href"])
        return out

    def fetch_url_content(self, url: str, mode: str = "raw"):
        if url not in self.returned:
            return {"error": "fetch_url_content: only URLs returned by search_engine_query in this task can be fetched"}
        if mode not in ("raw", "markdown", "truncate"):
            return {"error": f"Unsupported mode: {mode}"}
        try:
            r = self.http.get(url, headers={"User-Agent": UA, "Accept": "text/html,*/*;q=0.8"})
            r.raise_for_status()
        except httpx.HTTPError as e:
            self.errors.append(f"fetch: {type(e).__name__}: {e}")
            return {"error": f"fetch failed: {e}"}
        text = r.text if mode == "raw" else html_to_text(r.text)
        return {"content": text[: self.s.max_page_chars]}


def web_tools(ws: WebSearch) -> list[Tool]:
    tools = []
    for doc in class_docs("WebSearchAPI"):
        spec = tool_spec(doc)
        fn = getattr(ws, doc["name"])
        tools.append(Tool(spec["name"], spec["description"], spec["parameters"],
                          (lambda f: lambda **kw: json.dumps(f(**kw)))(fn)))
    return tools
