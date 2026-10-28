import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from a2e.client import Client, parse_timings
from a2e.config import RunConfig
from a2e.server import build_args

BODY = {"content": "ok", "timings": {"prompt_n": 100, "prompt_ms": 500.0, "cache_n": 900,
                                     "predicted_n": 20, "predicted_ms": 1000.0}}


def test_parse_timings_splits_cached_prompt():
    c = parse_timings(BODY, wall_ms=1600)
    assert c.prompt_tokens == 1000 and c.cached_tokens == 900
    assert c.prefill_tok_s == 200.0 and c.decode_tok_s == 20.0


def test_client_against_mock_server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps(BODY).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = Client(f"http://127.0.0.1:{srv.server_port}")
    r = c.complete("hi", 16, grammar='root ::= "x"', cache_prompt=False)
    srv.shutdown()
    assert r.text == "ok" and seen["grammar"] and seen["cache_prompt"] is False


def test_build_args_device_and_kv():
    mac = build_args(RunConfig(model="qwen3-4b", quant="q4_0", kv_quant="q8_0", device="mac"), "bin", "m.gguf")
    assert mac[mac.index("-ngl") + 1] == "99" and "--cache-type-k" in mac
    pi = build_args(RunConfig(model="qwen3-4b", quant="q4_0", device="pi5"), "bin", "m.gguf")
    assert pi[pi.index("-ngl") + 1] == "0" and "--cache-type-k" not in pi
