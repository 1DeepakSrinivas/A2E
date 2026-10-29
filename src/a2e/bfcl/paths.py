from __future__ import annotations

import sys
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
BFCL_ROOT = THIRD_PARTY / "bfcl_eval"
DATA = BFCL_ROOT / "data"


def ensure_on_path() -> None:
    """Make `import bfcl_eval` resolve to the vendored subset (not a full pip install of the package).

    Mixing the two would silently use the real model_config instead of our stub, so a conflict is an error.
    """
    loaded = sys.modules.get("bfcl_eval")
    if loaded is not None:
        if Path(loaded.__file__).resolve().parent != BFCL_ROOT:
            raise RuntimeError(f"a different bfcl_eval is already imported from {loaded.__file__}")
        return
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))
