# Historical suites

`bfcl_mini_gemma.toml` preserves the Gemma 3 4B Q4_0 experiment introduced on
`benchmarks/bfcl-mini-gemma`. The later `infra/two-model-set` change retired that
model and quantization, so this file is excluded from active suite discovery.

To reproduce the original experiment, use commit `3e603df` in a separate checkout
with the matching model files. The current Python suites use the newer Qwen3
Instruct and Granite model registry.
