#!/usr/bin/env bash
# Vendor the official BFCL v4 checker subset from a clone of ShishirPatil/gorilla at the pinned commit.
# Upstream files are copied byte-for-byte; the only non-upstream file is the model_config stub.
# Usage: scripts/sync_bfcl.sh [path-to-gorilla-clone]     (default /home/user/shishirpatil/gorilla)
set -euo pipefail
PIN="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
SRC_REPO="${1:-/home/user/shishirpatil/gorilla}"
SRC="$SRC_REPO/berkeley-function-call-leaderboard/bfcl_eval"
DST="$(cd "$(dirname "$0")/.." && pwd)/third_party/bfcl_eval"

head=$(git -C "$SRC_REPO" rev-parse HEAD)
[ "$head" = "$PIN" ] || { echo "clone is at $head, expected pinned $PIN" >&2; exit 1; }

CATS=(simple_python multiple parallel parallel_multiple irrelevance live_relevance live_irrelevance
      multi_turn_base multi_turn_miss_func multi_turn_miss_param)
BACKENDS=(gorilla_file_system math_api message_api posting_api ticket_api trading_bot travel_booking vehicle_control long_context)
DOCS=(gorilla_file_system math_api message_api posting_api ticket_api trading_bot travel_booking vehicle_control web_search)

FILES=(__init__.py constants/__init__.py constants/enums.py constants/type_mappings.py constants/default_prompts.py
       constants/executable_backend_config.py
       eval_checker/__init__.py eval_checker/ast_eval/__init__.py eval_checker/ast_eval/ast_checker.py
       eval_checker/ast_eval/type_convertor/__init__.py eval_checker/ast_eval/type_convertor/java_type_converter.py
       eval_checker/ast_eval/type_convertor/js_type_converter.py
       eval_checker/multi_turn_eval/__init__.py eval_checker/multi_turn_eval/multi_turn_checker.py
       eval_checker/multi_turn_eval/multi_turn_utils.py eval_checker/multi_turn_eval/func_source_code/__init__.py)
for b in "${BACKENDS[@]}"; do FILES+=("eval_checker/multi_turn_eval/func_source_code/$b.py"); done
for c in "${CATS[@]}"; do
  FILES+=("data/BFCL_v4_$c.json")
  if [ -f "$SRC/data/possible_answer/BFCL_v4_$c.json" ]; then FILES+=("data/possible_answer/BFCL_v4_$c.json"); fi
done
for d in "${DOCS[@]}"; do FILES+=("data/multi_turn_func_doc/$d.json"); done

rm -rf "$DST"; mkdir -p "$DST"
for f in "${FILES[@]}"; do mkdir -p "$DST/$(dirname "$f")"; cp "$SRC/$f" "$DST/$f"; done
cp "$SRC_REPO/LICENSE" "$DST/LICENSE"
echo "$PIN" > "$DST/PIN"

# The only non-upstream file: replaces the 2240-line module that imports every LLM SDK. ast_checker only reads
# MODEL_CONFIG_MAPPING[name].underscore_to_dot, and we pass function names through unchanged.
cat > "$DST/constants/model_config.py" <<'STUB_END'
# NOT UPSTREAM. A2E stub for bfcl_eval.constants.model_config (see ../NOTICE).
from collections import defaultdict
from types import SimpleNamespace

MODEL_CONFIG_MAPPING = defaultdict(lambda: SimpleNamespace(underscore_to_dot=False))
STUB_END

cat > "$DST/NOTICE" <<NOTICE_END
This directory vendors a subset of the Berkeley Function Calling Leaderboard (BFCL) evaluator and data.
Source:  https://github.com/ShishirPatil/gorilla  (berkeley-function-call-leaderboard/bfcl_eval)
Commit:  $PIN
License: Apache License 2.0 (see LICENSE). Copyright holders: the Gorilla / BFCL authors (UC Berkeley Sky Lab).

All files listed in MANIFEST.sha256 are copied unmodified from that commit.
The single file that is NOT upstream is constants/model_config.py (a stub); it is not in the manifest.
Regenerate with scripts/sync_bfcl.sh. Do not edit vendored files by hand.
NOTICE_END

(cd "$DST" && find . -type f ! -name MANIFEST.sha256 ! -name NOTICE ! -name PIN ! -name LICENSE \
   ! -path ./constants/model_config.py | sort | xargs sha256sum > MANIFEST.sha256)
echo "vendored $(wc -l < "$DST/MANIFEST.sha256") files at $PIN into $DST"
