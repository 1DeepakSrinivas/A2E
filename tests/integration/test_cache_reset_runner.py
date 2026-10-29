"""The runner against a server whose slot erase is rejected: it flushes, records it, and fails the job if it cannot."""
import json

import pytest

from a2e.runner import Runner, RunnerHooks
from a2e.store import Store
from tests.rig import FAKE, SMALL, instruments, make_settings, write_suite

pytest.importorskip("mpmath")
BODY = SMALL.replace('["mac", "pi5"]', '["mac"]')


def settings_with(tmp, *flags):
    s = make_settings(tmp)
    script = tmp / "llama-server"
    import sys
    script.write_text(f'#!/usr/bin/env bash\nexec {sys.executable} {FAKE} {" ".join(flags)} "$@"\n')
    script.chmod(0o755)
    return s


def run(tmp, *flags):
    s = settings_with(tmp, *flags)
    events = []
    r = Runner(write_suite(tmp, BODY), s, store=Store(s.db), log=events.append,
               hooks=RunnerHooks(instruments=instruments(6.0)))
    return r, r.run(), events


def resets(store):
    return {json.loads(r["server_json"]).get("cache_reset") for r in
            store.query("SELECT server_json FROM runs WHERE task_kind != 'bench'")}


def test_working_erase_is_recorded(tmp_path):
    r, out, events = run(tmp_path)
    assert out["counts"].get("done") and not out["counts"].get("failed")
    assert resets(r.store) == {"erase"}
    assert not [e for e in events if e["event"] == "cache_reset"]


def test_rejected_erase_falls_back_to_flush_and_says_so(tmp_path):
    r, out, events = run(tmp_path, "--reject-erase")
    assert out["counts"].get("done") and not out["counts"].get("failed")
    assert resets(r.store) == {"flush"}
    assert any(e["event"] == "cache_reset" and e["how"] == "flush" for e in events)


def test_flushed_runs_start_cold(tmp_path):
    r, _, _ = run(tmp_path, "--reject-erase")
    first = r.store.query("SELECT t.cached_tokens FROM turns t JOIN runs ru ON ru.run_id = t.run_id "
                          "WHERE ru.task_kind = 'plain' ORDER BY t.turn_id LIMIT 1")
    assert first[0]["cached_tokens"] == 0


def test_a_job_is_failed_when_neither_erase_nor_flush_works(tmp_path):
    r, out, events = run(tmp_path, "--reject-erase", "--fail-completion", "500")
    assert not out["counts"].get("done") and out["counts"].get("failed")
    reasons = [row["error"] for row in r.store.query("SELECT error FROM jobs WHERE status = 'failed'")]
    assert reasons and all("no cold prefix cache" in x for x in reasons)
    assert r.store.scalar("SELECT count(*) FROM runs") == 0  # nothing was measured on a warm cache
