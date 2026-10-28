"""In-process stand-in for the Brave Search API plus the pages its results point to."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

KEY = "test-key"


class FakeBrave:
    def __init__(self, n_results: int = 3, search_status: int = 200, page_status: int = 200):
        self.n_results, self.search_status, self.page_status = n_results, search_status, page_status
        self.requests: list[tuple[str, dict]] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def _send(self, code, body, ctype="application/json"):
                out = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_GET(self):
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                fake.requests.append((u.path, {**q, "token": self.headers.get("X-Subscription-Token")}))
                if u.path == "/res/v1/web/search":
                    if self.headers.get("X-Subscription-Token") != KEY:
                        return self._send(401, json.dumps({"error": "bad token"}))
                    if fake.search_status != 200:
                        return self._send(fake.search_status, json.dumps({"error": "injected"}))
                    n = min(int(q.get("count", 10)), fake.n_results)
                    res = [{"title": f"Result {i} for {q.get('q', '')}", "url": f"{fake.base}/page/{i}",
                            "description": f"<strong>Snippet</strong> {i}"} for i in range(n)]
                    return self._send(200, json.dumps({"web": {"results": res}}))
                if u.path.startswith("/page/"):
                    if fake.page_status != 200:
                        return self._send(fake.page_status, "nope", "text/plain")
                    html = ("<html><head><style>.x{}</style><script>var a=1;</script></head><body>"
                            f"<h1>Page {u.path.rsplit('/', 1)[-1]}</h1><p>The answer is 42.</p></body></html>")
                    return self._send(200, html, "text/html")
                return self._send(404, "{}")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
