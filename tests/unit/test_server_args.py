from a2e.config import RunConfig
from a2e.server import LlamaServer, build_args


def cfg(**kw):
    return RunConfig(**{"model": "qwen3-4b-instruct-2507", "quant": "q5_k_m", "device": "mac", **kw})


def test_flash_attn_default_and_override():
    a = build_args(cfg(), "bin", "m.gguf")
    assert a[a.index("--flash-attn") + 1] == "on"
    a = build_args(cfg(), "bin", "m.gguf", flash_attn="auto")
    assert a[a.index("--flash-attn") + 1] == "auto"


def test_kv_quant_sets_both_k_and_v():
    a = build_args(cfg(kv_quant="q8_0"), "bin", "m.gguf")
    assert a[a.index("--cache-type-k") + 1] == "q8_0" and a[a.index("--cache-type-v") + 1] == "q8_0"


def test_ctx_threads_port():
    a = build_args(cfg(device="pi5"), "bin", "m.gguf", port=9999, ctx=4096, threads=4)
    assert a[a.index("--port") + 1] == "9999" and a[a.index("-c") + 1] == "4096"
    assert a[a.index("-t") + 1] == "4" and a[a.index("-ngl") + 1] == "0"


def test_jinja_always_on_for_tool_templates():
    assert "--jinja" in build_args(cfg(), "bin", "m.gguf")


def test_stop_without_start_is_safe(tmp_path):
    s = LlamaServer(["true"], log_path=tmp_path / "x.log")
    s.stop()
    s.stop()


def test_slot_save_path_enables_slot_actions():
    a = build_args(cfg(), "bin", "m.gguf", slot_save_path="/tmp/slots")
    assert a[a.index("--slot-save-path") + 1] == "/tmp/slots"
    assert "--slot-save-path" not in build_args(cfg(), "bin", "m.gguf")
