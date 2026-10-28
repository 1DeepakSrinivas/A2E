#!/usr/bin/env python3
"""Cherry-pick small BFCL file-system starting states for the agent's file-system task families.

BFCL's multi-turn data targets every model size. Small models on a Pi need small worlds, so this keeps only the
distinct GorillaFileSystem `initial_config`s that fit these limits (also written into the output file):
  2 to 6 files, directory depth at most 3, at most 800 characters of file content in total.
Per-file line counts are stored so a task family can pick only files that suit it (e.g. line counting).
Run from the repository root:  uv run python scripts/curate_fs_scenarios.py
The output, src/a2e/data/fs_scenarios.json, is committed; a test checks that this script reproduces it exactly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import a2e.bfcl  # noqa: E402,F401  (vendored BFCL on the path)
from a2e.bfcl.load import MULTI_TURN_CATEGORIES, load_cases  # noqa: E402

LIMITS = {"min_files": 2, "max_files": 6, "max_depth": 3, "max_chars": 800}
OUT = REPO / "src" / "a2e" / "data" / "fs_scenarios.json"


def walk(node: dict, path: list[str], files: list[dict], depth: list[int]) -> None:
    depth[0] = max(depth[0], len(path) + 1)
    for name, item in (node.get("contents") or {}).items():
        if item.get("type") == "file":
            text = item.get("content", "")
            files.append({"path": path, "name": name, "chars": len(text),
                          "lines": sum(1 for line in text.splitlines() if line.strip())})
        else:
            walk(item, path + [name], files, depth)


def curate() -> dict:
    seen: dict[str, dict] = {}
    for cat in MULTI_TURN_CATEGORIES:
        for case in load_cases(cat):
            cfg = case.entry["initial_config"].get("GorillaFileSystem")
            if not cfg or "root" not in cfg:
                continue
            key = json.dumps(cfg, sort_keys=True)
            if key in seen:
                seen[key]["sources"].append(case.id)
                continue
            root_name = next(iter(cfg["root"]))
            files: list[dict] = []
            depth = [1]
            walk(cfg["root"][root_name], [], files, depth)  # upstream loads only the first root entry
            # keep only the root folder BFCL loads (the first entry), in BFCL's own key order: ls and find
            # list entries in that order, and json sort_keys would change which root comes first
            used = {"root": {root_name: cfg["root"][root_name]}}
            seen[key] = {"config": used, "root": root_name, "files": files, "depth": depth[0], "sources": [case.id]}
    keep = []
    for s in seen.values():
        n, chars = len(s["files"]), sum(f["chars"] for f in s["files"])
        if not (LIMITS["min_files"] <= n <= LIMITS["max_files"]) or s["depth"] > LIMITS["max_depth"] \
                or chars > LIMITS["max_chars"]:
            continue
        sid = "fs_" + hashlib.sha256(json.dumps(s["config"], sort_keys=True).encode()).hexdigest()[:8]
        keep.append({"id": sid, **s})
    keep.sort(key=lambda s: s["id"])
    return {"limits": LIMITS, "distinct_scenarios_in_bfcl": len(seen), "kept": len(keep), "scenarios": keep}


if __name__ == "__main__":
    data = curate()
    OUT.write_text(json.dumps(data, indent=1) + "\n")  # no sort_keys: order is part of the scenario
    print(f"kept {data['kept']} of {data['distinct_scenarios_in_bfcl']} BFCL file-system scenarios -> {OUT}")
