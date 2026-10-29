"""Named test groups from configs/tests.toml, turned into pytest command lines."""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from a2e.settings import CONFIG_DIR, REPO


@dataclass
class Group:
    name: str
    description: str = ""
    markers: str | None = None
    keyword: str | None = None
    paths: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)

    def missing_env(self) -> list[str]:
        return [v for v in self.env if not os.environ.get(v)]

    def command(self, extra: list[str] | None = None) -> list[str]:
        cmd = [sys.executable, "-m", "pytest", "-ra"]
        if self.markers:
            cmd += ["-m", self.markers]
        if self.keyword:
            cmd += ["-k", self.keyword]
        cmd += [str(REPO / p) for p in self.paths]
        return cmd + list(extra or [])


def load_groups(path: str | Path | None = None) -> dict[str, Group]:
    data = tomllib.loads(Path(path or CONFIG_DIR / "tests.toml").read_text())
    return {k: Group(name=k, **v) for k, v in data.get("groups", {}).items()}
