import json

import pytest

from a2e.bench import bench_args, parse_bench_json, parse_perf_report


def test_bench_args_per_device_and_options():
    pi = bench_args("lb", "m.gguf", n_prompt=512, n_gen=0, reps=3, device="pi5", kv_quant="f16", flash_attn="on",
                    threads=4)
    assert pi[:9] == ["lb", "-m", "m.gguf", "-p", "512", "-n", "0", "-r", "3"]
    assert pi[pi.index("-ngl") + 1] == "0" and pi[pi.index("-fa") + 1] == "1" and pi[pi.index("-t") + 1] == "4"
    assert "-ctk" not in pi and pi[pi.index("-o") + 1] == "json"
    mac = bench_args("lb", "m", n_prompt=0, n_gen=128, reps=5, device="mac", kv_quant="q8_0", flash_attn="off",
                     threads=None)
    assert mac[mac.index("-ngl") + 1] == "99" and mac[mac.index("-fa") + 1] == "0"
    assert mac[mac.index("-ctk") + 1] == "q8_0" and mac[mac.index("-ctv") + 1] == "q8_0" and "-t" not in mac


def test_parse_bench_json_skips_leading_log_text():
    text = "ggml_init: something\n" + json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4, "stddev_ts": 2.0,
                                                   "avg_ns": 4_000_000, "n_threads": 4}])
    (r,) = parse_bench_json(text, "pp", 512, 0)
    assert (r.test, r.n_prompt, r.n_gen, r.avg_tok_s, r.sd_tok_s, r.avg_ms, r.n_threads) == ("pp", 512, 0, 123.4, 2.0,
                                                                                              4.0, 4)


def test_parse_bench_json_tolerates_missing_fields_and_rejects_no_json():
    (r,) = parse_bench_json('[{"avg_ts": 9}]', "tg", 0, 64)
    assert r.n_gen == 64 and r.avg_ms is None and r.n_threads is None
    with pytest.raises(ValueError):
        parse_bench_json("no output", "pp", 1, 0)


def test_parse_perf_report_ranks_symbols_and_ignores_noise():
    text = ("# Overhead  Shared Object  Symbol\n#\n"
            "     3.10%  libc.so.6          [.] __memcpy_generic\n"
            "    41.20%  libggml-cpu.so     [.] ggml_vec_dot_q4_0_q8_0\n"
            "     8.00%  [kernel.kallsyms]  [k] clear_page_erms\n"
            "not a row\n    12.50%  libggml-cpu.so     [.] ggml_compute_forward_mul_mat\n")
    rows = parse_perf_report(text, top=3)
    assert [r["symbol"] for r in rows] == ["ggml_vec_dot_q4_0_q8_0", "ggml_compute_forward_mul_mat", "clear_page_erms"]
    assert rows[0] == {"rank": 1, "pct": 41.2, "dso": "libggml-cpu.so", "symbol": "ggml_vec_dot_q4_0_q8_0"}
    assert parse_perf_report("") == []


def test_suite_bench_task_validation(tmp_path):
    from a2e.suite import ConfigError, load_suite
    base = '[suite]\nname = "b"\n[matrix]\ndevice = "pi5"\nmodel = "qwen3-4b"\nquant = "q4_0"\n'
    p = tmp_path / "b.toml"
    p.write_text(base + '[[tasks]]\nkind = "bench"\n')
    with pytest.raises(ConfigError, match="n_prompt"):
        load_suite(p)
    p.write_text(base + '[[tasks]]\nkind = "plain"\nperf = true\n')
    with pytest.raises(ConfigError, match="only apply to bench"):
        load_suite(p)
    p.write_text(base + '[[tasks]]\nkind = "bench"\nn_prompt = [0]\n')
    with pytest.raises(ConfigError, match=">= 1"):
        load_suite(p)
    p.write_text(base + '[[tasks]]\nkind = "bench"\nn_prompt = [256, 1024]\nn_gen = 128\nperf = true\nreps = 2\n')
    assert load_suite(p).tasks[0].n_gen == [128]
