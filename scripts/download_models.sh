#!/usr/bin/env bash
# Download the four GGUF files of the study (about 7 GB) into <models_dir>/<model>/<model>-<quant>.gguf.
#   scripts/download_models.sh [models_dir]      default: $A2E_MODELS_DIR or ~/models
# Needs `huggingface-cli` (pip install -U "huggingface_hub[cli]"). Sources are listed in configs/models.toml.
set -euo pipefail

DEST="${1:-${A2E_MODELS_DIR:-$HOME/models}}"

fetch() {  # fetch <repo> <file in repo> <model name> <quant>
  local repo="$1" remote="$2" name="$3" quant="$4"
  local dir="$DEST/$name" out="$DEST/$name/$name-$quant.gguf"
  mkdir -p "$dir"
  if [ -f "$out" ]; then echo "have $out"; return; fi
  huggingface-cli download "$repo" "$remote" --local-dir "$dir"
  mv "$dir/$remote" "$out"
  echo "wrote $out"
}

Q=bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF
fetch "$Q" Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf qwen3-4b-instruct-2507 q5_k_m   # 2.89 GB
fetch "$Q" Qwen_Qwen3-4B-Instruct-2507-IQ3_M.gguf  qwen3-4b-instruct-2507 iq3_m    # 1.96 GB

G=bartowski/granite-3.1-2b-instruct-GGUF
fetch "$G" granite-3.1-2b-instruct-Q6_K.gguf  granite-3-1-2b-instruct q6_k         # 2.08 GB
fetch "$G" granite-3.1-2b-instruct-IQ3_M.gguf granite-3-1-2b-instruct iq3_m        # 1.17 GB
