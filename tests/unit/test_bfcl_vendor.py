import hashlib
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import a2e.bfcl  # noqa: F401
from a2e.bfcl import paths

ROOT = paths.BFCL_ROOT
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sync_bfcl.sh"
NON_UPSTREAM = {"LICENSE", "NOTICE", "PIN", "MANIFEST.sha256", "constants/model_config.py"}
UPSTREAM_CLONE = Path("/home/user/shishirpatil/gorilla")


def manifest():
    out = {}
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, path = line.split("  ", 1)
        out[path.removeprefix("./")] = digest
    return out


def test_manifest_hashes_match_files():
    m = manifest()
    assert len(m) >= 40
    for path, digest in m.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, f"{path} was modified"


def test_only_the_documented_files_are_not_upstream():
    on_disk = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert on_disk - set(manifest()) == NON_UPSTREAM


def test_pin_file_matches_sync_script_and_notice():
    pin = (ROOT / "PIN").read_text().strip()
    assert re.search(rf'^PIN="{pin}"$', SCRIPT.read_text(), re.M)
    assert pin in (ROOT / "NOTICE").read_text() and re.fullmatch(r"[0-9a-f]{40}", pin)


def test_apache_license_and_attribution_present():
    lic = (ROOT / "LICENSE").read_text()
    assert "Apache License" in lic and "Version 2.0" in lic
    assert "github.com/ShishirPatil/gorilla" in (ROOT / "NOTICE").read_text()


def test_stub_model_config_leaves_names_unchanged():
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    assert MODEL_CONFIG_MAPPING["anything/at-all"].underscore_to_dot is False


def test_byte_identical_to_upstream_when_clone_available():
    if not (UPSTREAM_CLONE / ".git").exists():
        pytest.skip("no upstream clone")
    head = subprocess.run(["git", "-C", str(UPSTREAM_CLONE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != (ROOT / "PIN").read_text().strip():
        pytest.skip("clone is not at the pinned commit")
    up = UPSTREAM_CLONE / "berkeley-function-call-leaderboard" / "bfcl_eval"
    for path in manifest():
        assert (up / path).read_bytes() == (ROOT / path).read_bytes(), path


def _heavy_after(code: str) -> str:
    prog = ("import sys; sys.path.insert(0, %r)\n%s\n"
            "heavy=[m for m in ('openai','anthropic','torch','tree_sitter','numpy','boto3','cohere','mistralai',"
            "'google','faiss','sentence_transformers','bs4','requests') if m in sys.modules]\nprint(heavy)"
            % (str(paths.THIRD_PARTY), code))
    return subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, text=True).stdout.strip()


def test_checkers_import_without_any_llm_sdk_or_heavy_dependency():
    code = ("import bfcl_eval.eval_checker.ast_eval.ast_checker\n"
            "import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker\n")
    assert _heavy_after(code) == "[]"


def test_backends_import_without_heavy_dependencies():
    pytest.importorskip("mpmath", reason="needs `uv sync --extra bfcl`")
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
    mods = [m for k, m in CLASS_FILE_PATH_MAPPING.items() if k not in ("WebSearchAPI",) and not k.startswith("MemoryAPI")]
    code = "import importlib\n" + "\n".join(f"importlib.import_module({m!r})" for m in mods)
    assert _heavy_after(code) == "[]"


def test_conflicting_bfcl_eval_is_refused(monkeypatch):
    fake = types.ModuleType("bfcl_eval")
    fake.__file__ = "/somewhere/else/bfcl_eval/__init__.py"
    monkeypatch.setitem(sys.modules, "bfcl_eval", fake)
    with pytest.raises(RuntimeError, match="already imported"):
        paths.ensure_on_path()


def test_data_files_for_every_scoped_category_exist():
    for cat in ("simple_python", "multiple", "parallel", "parallel_multiple", "multi_turn_base",
                "multi_turn_miss_func", "multi_turn_miss_param"):
        assert (paths.DATA / f"BFCL_v4_{cat}.json").exists() and (paths.DATA / "possible_answer" / f"BFCL_v4_{cat}.json").exists()
