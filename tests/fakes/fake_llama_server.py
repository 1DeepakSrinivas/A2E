#!/usr/bin/env python3
"""Stand-in for llama-server: /health, /completion (with timings + prefix cache), /slots.

Accepts and ignores real llama-server flags so it can be launched through a2e.server.LlamaServer.
Failure injection: --health-delay, --exit-code, --fail-completion, --bad-json, --no-timings, --reject-erase.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--health-delay", type=float, default=0.0)
    ap.add_argument("--exit-code", type=int, default=None)
    ap.add_argument("--fail-completion", type=int, default=0, help="HTTP status to return")
    ap.add_argument("--ms-per-prompt-token", type=float, default=0.2)
    ap.add_argument("--ms-per-gen-token", type=float, default=1.0)
    ap.add_argument("--grammar-overhead", type=float, default=1.25, help="decode slowdown with grammar")
    ap.add_argument("--no-timings", action="store_true")
    ap.add_argument("--reject-erase", action="store_true", help="answer POST /slots/0?action=erase with HTTP 501")
    ap.add_argument("--bad-json", action="store_true")
    ap.add_argument("--rss-mb", type=int, default=0)
    args, _ = ap.parse_known_args()

    if args.exit_code is not None:
        raise SystemExit(args.exit_code)

    ballast = bytearray(args.rss_mb * 2**20)
    for i in range(0, len(ballast), 4096):  # touch pages so they count toward RSS
        ballast[i] = 1
    started = time.monotonic()
    state = {"last": []}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, raw=False):
            out = body.encode() if raw else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            if self.path == "/health":
                ok = time.monotonic() - started >= args.health_delay
                self._send(200 if ok else 503, {"status": "ok" if ok else "loading model"})
            elif self.path == "/slots":
                self._send(200, [{"id": 0, "n_ctx": 8192, "is_processing": False}])
            else:
                self._send(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/apply-template":
                text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in body["messages"])
                # like Qwen3's template: with enable_thinking false the assistant turn starts with an empty think block
                think_off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
                return self._send(200, {"prompt": text + "<|im_start|>assistant\n"
                                        + ("<think>\n\n</think>\n\n" if think_off else "")})
            if self.path == "/tokenize":
                return self._send(200, {"tokens": [0] * len(body.get("content", "").split())})
            if self.path.startswith("/slots/0"):
                if args.reject_erase:
                    return self._send(501, {"error": "slot erase not supported"})
                state["last"] = []
                return self._send(200, {"id_slot": 0})
            if args.fail_completion:
                return self._send(args.fail_completion, {"error": "injected"})
            words = body.get("prompt", "").split()
            cached = 0
            if body.get("cache_prompt", True):
                for a, b in zip(state["last"], words):
                    if a != b:
                        break
                    cached += 1
                cached = min(cached, max(len(words) - 1, 0))  # llama.cpp re-evaluates >=1 token
            state["last"] = words
            fresh = len(words) - cached
            gen = int(body.get("n_predict", 16))
            prompt_ms = fresh * args.ms_per_prompt_token
            per_tok = args.ms_per_gen_token * (args.grammar_overhead if (body.get("grammar") or body.get("json_schema")) else 1.0)
            pred_ms = gen * per_tok
            time.sleep((prompt_ms + pred_ms) / 1000)
            if args.bad_json:
                return self._send(200, "{not json", raw=True)
            out = {"content": "x" * gen}
            if not args.no_timings:
                out["timings"] = {"prompt_n": fresh, "prompt_ms": prompt_ms, "cache_n": cached,
                                  "predicted_n": gen, "predicted_ms": pred_ms}
            self._send(200, out)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
