"""Suite files: TOML descriptions of what to run, expanded into a fixed, ordered list of jobs.

A job is one measured configuration: one model file on one device with one set of server settings, one task
(plain prompts, a custom chain family, or a BFCL category) and one value for every axis that task uses
(grammar, prefix cache, decode length, schema length, governor). Each job has a key: a hash of its canonical
JSON. The same suite file always gives the same jobs and keys, which is what makes runs resumable and repeatable.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from a2e.agent.harness import CONTEXTS, FIELDS as HARNESS_FIELDS, RECOVERIES, STOPPINGS, Harness
from a2e.agent.prompt import PROMPT_FORMATS
from a2e.bfcl.load import ALL_CATEGORIES, MULTI_TURN_CATEGORIES
from a2e.settings import CONFIG_DIR, Settings
from a2e.tasks.custom import FAMILY_DEPTH, FAMILY_TOOLS

MIN_CTX_MULTI_TURN = 16384
PI_Q8_MAX_PARAMS_B = 4.0
AXES_FOR_KIND = {
    "plain": ("prefix_cache", "decode_len", "governor"),
    "custom": ("grammar", "prefix_cache", "decode_len", "schema_len", "governor"),
    "bfcl": ("grammar", "prefix_cache", "decode_len", "governor"),
    "bench": ("governor",),
}
HARNESS_KINDS = ("custom", "bfcl")  # the kinds that call tools; plain prompts and bench jobs have no harness
KIND_ORDER = {"plain": 0, "custom": 1, "bfcl": 2, "bench": 3}  # bench last: it needs the server stopped
SERVER_FIELDS = ("device", "model", "quant", "kv_quant", "ctx", "flash_attn", "threads", "thinking")


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _listify(v):
    return v if isinstance(v, list) else [v]


class Meta(_Strict):
    name: str
    description: str = ""
    seed: int = 0


class Repeats(_Strict):
    warmup: int = Field(1, ge=0)
    measured: int = Field(3, ge=1)


class Server(_Strict):
    ctx: int = Field(8192, ge=512)
    flash_attn: str = "on"
    threads: int = Field(0, ge=0)  # 0 = use [machine] threads from local.toml, or llama.cpp's default
    # "off" starts llama-server with --chat-template-kwargs {"enable_thinking": false}, so reasoning models such as
    # Qwen3 answer directly instead of spending the token budget on a thinking block. Models whose chat template has
    # no such variable are unaffected.
    thinking: Literal["default", "off"] = "default"


class RunOpts(_Strict):
    idle_seconds: float = Field(20.0, ge=0)  # idle power baseline after each model load
    sample_hz: float = Field(2.0, gt=0)
    cooldown_c: float = Field(0.0, ge=0)  # wait until the SoC is below this temperature before each job (0 = off)
    cooldown_timeout_s: float = Field(600.0, ge=0)
    max_turns: int = Field(8, ge=1)  # custom chains: model calls per task before giving up
    max_steps: int = Field(20, ge=1)  # BFCL multi-turn: model calls per user turn (upstream uses 20)


class Matrix(_Strict):
    device: list[Literal["pi5", "mac"]]
    model: list[str]
    quant: list[Literal["q4_0", "q8_0", "q3_k_m"]]
    kv_quant: list[Literal["f16", "q8_0"]] = ["f16"]

    _l = field_validator("device", "model", "quant", "kv_quant", mode="before")(_listify)


class Axes(_Strict):
    grammar: list[bool] = [False]
    prefix_cache: list[bool] = [True]
    decode_len: list[int] = [256]
    schema_len: list[int] = [1024]
    governor: list[str] = [""]  # "" leaves the governor as it is

    _l = field_validator("grammar", "prefix_cache", "decode_len", "schema_len", "governor", mode="before")(_listify)

    @field_validator("decode_len", "schema_len")
    @classmethod
    def _positive(cls, v):
        if any(x < 16 for x in v):
            raise ValueError("lengths must be >= 16")
        return v


class HarnessVariant(_Strict):
    """One named harness setting. Fields left out keep the default. Applies to custom and bfcl tasks only."""
    name: str = Field(min_length=1)
    prompt_format: Literal[PROMPT_FORMATS] = "json_schema"
    context: Literal[CONTEXTS] = "full"
    recovery: Literal[RECOVERIES] = "none"
    stopping: Literal[STOPPINGS] = "fixed"

    @property
    def harness(self) -> Harness:
        return Harness(**{f: getattr(self, f) for f in HARNESS_FIELDS})


class Task(_Strict):
    kind: Literal["plain", "custom", "bfcl", "bench"]
    n: int | None = Field(None, ge=1)  # prompts / chains / BFCL cases per run; BFCL: all when unset
    family: list[str] = []
    depth: list[int] = []
    n_args: list[int] = [2]
    distractors: Literal["bfcl", "synthetic"] = "bfcl"  # custom: pad the tool list with real BFCL functions or made-up ones
    category: list[str] = []
    n_prompt: list[int] = []  # bench: prompt sizes for the prefill test
    n_gen: list[int] = []  # bench: token counts for the decode test
    perf: bool = False  # bench: also record top kernels with Linux perf (Pi)
    reps: int = Field(5, ge=1)  # bench: llama-bench repetitions per test
    # per-task axis overrides
    grammar: list[bool] | None = None
    prefix_cache: list[bool] | None = None
    decode_len: list[int] | None = None
    schema_len: list[int] | None = None
    governor: list[str] | None = None

    _l = field_validator("family", "depth", "n_args", "category", "n_prompt", "n_gen", "grammar", "prefix_cache",
                         "decode_len", "schema_len", "governor", mode="before")(lambda v: v if v is None else _listify(v))

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "custom":
            if not self.family or not self.depth:
                raise ValueError("custom tasks need family and depth")
            for f in self.family:
                if f not in FAMILY_TOOLS:
                    raise ValueError(f"unknown family {f!r}; choose from {sorted(FAMILY_TOOLS)}")
                lo, hi = FAMILY_DEPTH[f]
                if any(not lo <= d <= hi for d in self.depth):
                    raise ValueError(f"family {f} needs depth {lo} to {hi}")
            if any(not 1 <= a <= 5 for a in self.n_args):
                raise ValueError("n_args must be 1 to 5")
        if self.kind == "bfcl":
            if not self.category:
                raise ValueError("bfcl tasks need category")
            bad = [c for c in self.category if c not in ALL_CATEGORIES]
            if bad:
                raise ValueError(f"unknown BFCL categories {bad}; choose from {list(ALL_CATEGORIES)}")
        if self.kind == "bench":
            if not self.n_prompt and not self.n_gen:
                raise ValueError("bench tasks need n_prompt and/or n_gen")
            if any(x < 1 for x in self.n_prompt + self.n_gen):
                raise ValueError("n_prompt and n_gen must be >= 1")
        elif self.n_prompt or self.n_gen or self.perf:
            raise ValueError("n_prompt, n_gen and perf only apply to bench tasks")
        if self.kind != "custom" and (self.family or self.depth):
            raise ValueError("family/depth only apply to custom tasks")
        if self.kind != "bfcl" and self.category:
            raise ValueError("category only applies to bfcl tasks")
        return self


class Suite(_Strict):
    suite: Meta
    repeats: Repeats = Repeats()
    server: Server = Server()
    run: RunOpts = RunOpts()
    matrix: Matrix
    axes: Axes = Axes()
    harness: list[HarnessVariant] = []  # [[harness]] tables; none means only the default harness
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def _harness_unique(self):
        names = [v.name for v in self.harness]
        if len(set(names)) != len(names):
            raise ValueError("harness names must be unique")
        settings = [v.harness for v in self.harness]
        if len(set(settings)) != len(settings):
            raise ValueError("two [[harness]] entries have the same settings")
        return self

    def canonical(self) -> dict:
        d = json.loads(self.model_dump_json())
        if not d["harness"]:  # keeps the hash of every suite written before harness variants existed
            del d["harness"]
        if d["server"]["thinking"] == "default":  # likewise for suites written before the thinking switch
            del d["server"]["thinking"]
        return d

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True).encode()).hexdigest()[:16]


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = CONFIG_DIR / "suites" / f"{path}.toml"
    try:
        suite = Suite.model_validate(tomllib.loads(p.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"suite file not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValueError) as e:
        raise ConfigError(f"{p}: {e}") from None
    multi = any(c in MULTI_TURN_CATEGORIES for t in suite.tasks for c in t.category)
    if multi and suite.server.ctx < MIN_CTX_MULTI_TURN:
        raise ConfigError(f"{p}: BFCL multi-turn categories need [server] ctx >= {MIN_CTX_MULTI_TURN}")
    return suite


def job_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def server_key(cfg: dict) -> tuple:
    return tuple(cfg.get(k, "default" if k == "thinking" else None) for k in SERVER_FIELDS)


def feasibility(cfg: dict, settings: Settings) -> str | None:
    """None if the job can run; otherwise the reason it is skipped (stored with the job)."""
    params = settings.model(cfg["model"]).params_b
    if cfg["device"] == "pi5" and cfg["quant"] == "q8_0" and params > PI_Q8_MAX_PARAMS_B:
        return "not run: memory (Pi 5 8GB runs Q8_0 only for models <= 4B)"
    if cfg.get("governor") and cfg["device"] != "pi5":
        return "not run: governor settings apply to the Pi only"
    return None


def expand(suite: Suite, settings: Settings) -> list[dict]:
    """All jobs of a suite, in run order (grouped so each model loads once per device)."""
    for m in suite.matrix.model:
        settings.model(m)  # unknown models fail early with a clear message
    jobs = []
    m = suite.matrix
    base = {"seed": suite.suite.seed, "warmup": suite.repeats.warmup, "measured": suite.repeats.measured,
            "ctx": suite.server.ctx, "flash_attn": suite.server.flash_attn,
            "threads": suite.server.threads}  # 0 = machine default; local.toml threads apply at launch only
    if suite.server.thinking != "default":
        base["thinking"] = suite.server.thinking  # stored only when set, so old jobs keep their keys
    for dev, model, quant, kv in itertools.product(m.device, m.model, m.quant, m.kv_quant):
        cell = {**base, "device": dev, "model": model, "quant": quant, "kv_quant": kv}
        for task in suite.tasks:
            axes = AXES_FOR_KIND[task.kind]
            axis_values = [getattr(task, a) if getattr(task, a) is not None else getattr(suite.axes, a) for a in axes]
            if task.kind == "plain":
                grid = [{"n": task.n or 6}]
            elif task.kind == "custom":
                grid = [{"family": f, "depth": d, "n_args": a, "n": task.n or 5, "distractors": task.distractors}
                        for f in task.family for d in task.depth for a in task.n_args]
            elif task.kind == "bfcl":
                grid = [{"category": c, "n": task.n} for c in task.category]
            else:
                grid = [{"n_prompt": task.n_prompt, "n_gen": task.n_gen, "perf": task.perf, "reps": task.reps}]
            variants = suite.harness if suite.harness and task.kind in HARNESS_KINDS else [None]
            for combo, g, hv in itertools.product(itertools.product(*axis_values), grid, variants):
                cfg = {**cell, "kind": task.kind, **dict(zip(axes, combo)), **g}
                if hv is not None and not hv.harness.is_default:
                    # only non-default fields are stored, so the default harness keeps its old job keys
                    cfg.update({"harness": hv.name, **hv.harness.non_default()})
                cfg["job_key"] = job_key(cfg)
                cfg["skip_reason"] = feasibility(cfg, settings)
                jobs.append(cfg)
    seen, unique = set(), []
    for j in jobs:  # identical jobs from overlapping task entries run once
        if j["job_key"] not in seen:
            seen.add(j["job_key"])
            unique.append(j)
    unique.sort(key=lambda j: (server_key(j), KIND_ORDER[j["kind"]],
                               json.dumps({k: v for k, v in j.items() if k != "job_key"}, sort_keys=True)))
    return unique


def plan_summary(jobs: list[dict]) -> dict:
    runnable = [j for j in jobs if not j["skip_reason"]]
    per_kind = {k: sum(1 for j in runnable if j["kind"] == k) for k in KIND_ORDER}
    return {"jobs": len(jobs), "runnable": len(runnable), "skipped": len(jobs) - len(runnable),
            "server_loads": len({server_key(j) for j in runnable if j["kind"] != "bench"}),  # bench runs llama-bench
            "by_kind": per_kind,
            "runs": sum(j["warmup"] + j["measured"] for j in runnable),
            "by_device": {d: sum(1 for j in runnable if j["device"] == d) for d in ("pi5", "mac")}}


def lock_entry(suite: Suite, jobs: list[dict]) -> dict:
    keys = "".join(j["job_key"] for j in jobs)
    return {"suite_hash": suite.hash, "jobs": len(jobs), "job_keys_sha": hashlib.sha256(keys.encode()).hexdigest()[:16]}
