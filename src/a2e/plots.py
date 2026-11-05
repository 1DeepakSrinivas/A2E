"""Figures for the harness question: does the harness change success per joule, or is the system LLM-bound?

Each figure is written as a PNG with the table it draws next to it as a CSV (the table view of the same data).
Colors follow the project's validated categorical order (blue, orange, aqua, yellow); three light-mode colors fall
below 3:1 contrast on the surface, so every figure carries direct labels and a CSV. Needs `uv sync --extra plots`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from a2e import efficiency as E
from a2e.store import Store

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898780", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BAND = "#e9e8e4"  # the equivalence margin: a neutral band, not a series


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("plots need matplotlib: run `uv sync --extra plots`") from e
    return plt


def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _fig(plt, w, h, title, subtitle):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    fig.text(0.02, 0.97, title, color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.97 - 0.045 * (6 / h) * 1.6, subtitle, color=INK2, fontsize=9, va="top")
    return fig


def _axes(fig, h, w, top_in=1.0, bottom_in=0.8, left_in=1.5, right_in=0.9):
    """An axes placed by inch margins, so the plot fills the figure whatever its height."""
    return fig.add_axes([left_in / w, bottom_in / h, 1 - (left_in + right_in) / w, 1 - (top_in + bottom_in) / h])


def _legend(fig, plt, items, x0=0.04, y=0.04, step=0.27):
    for k, (label, color) in enumerate(items):
        fig.patches.append(plt.Rectangle((x0 + k * step, y - 0.012), 0.012, 0.024, transform=fig.transFigure, color=color))
        fig.text(x0 + k * step + 0.018, y, label, color=INK2, fontsize=8, va="center")


def _pct_ticks(ax, lo, hi):
    """Tick marks at round percentage changes on a log-change axis."""
    cand = [p for p in (-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 75, 100, 150, 200)
            if lo <= math.log1p(p / 100) <= hi]
    if len(cand) < 3:
        cand = sorted({round((math.exp(v) - 1) * 100) for v in np.linspace(lo, hi, 5)})
    ax.set_xticks([math.log1p(p / 100) for p in cand], labels=[f"{p:+d}%" if p else "0%" for p in cand])


def _save(fig, path: Path, table: pd.DataFrame) -> Path:
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    table.to_csv(path.with_suffix(".csv"), index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _pct(v):
    return f"{(math.exp(v) - 1) * 100:+.0f}%"


# ------------------------------------------------------------------------------------------------ 1. forest
def forest(store: Store, out: Path, exp_id=None, margin: float = E.DEFAULT_MARGIN) -> Path | None:
    by = ["device"] if store.df("SELECT DISTINCT device FROM runs").shape[0] > 1 else []
    eff = E.harness_effect(store, exp_id, margin=margin, by=by)
    if eff.empty:
        return None
    plt = _plt()
    groups = [("all devices", eff)] if not by else [(d, g) for d, g in eff.groupby("device")]
    rows = max(len(g) for _, g in groups)
    fig = _fig(plt, 4.6 * len(groups) + 1.2, 1.9 + 0.42 * rows, "Joules per success under each harness, against the baseline",
               f"Ratio to baseline, 90% interval. Grey band: equivalent within {margin:.0%}. Right of the band is costlier.")
    axes = fig.subplots(1, len(groups), squeeze=False)[0]
    fig.subplots_adjust(left=0.13 if len(groups) == 1 else 0.10, right=0.80 if len(groups) == 1 else 0.93,
                        top=1 - 0.5 / (1.9 + 0.42 * rows) * 1.0 - 0.14, bottom=0.14, wspace=0.62)
    lo_b, hi_b = 1 / (1 + margin), 1 + margin
    for ax, (name, g) in zip(axes, groups):
        g = g.reset_index(drop=True)
        _style(ax, "x")
        ax.axvspan(lo_b, hi_b, color=BAND, zorder=0)
        ax.axvline(1.0, color=MUTED, linewidth=1, zorder=1)
        finite = g[np.isfinite(g["hi"]) & np.isfinite(g["lo"])]
        span = max([hi_b, *finite["hi"].tolist(), 1 / lo_b]) * 1.15
        lo_ax = min([lo_b, *finite["lo"].tolist()]) / 1.1
        for i, r in g.iterrows():
            y = len(g) - 1 - i
            if np.isfinite(r["ratio"]):
                ax.plot([max(r["lo"], lo_ax), min(r["hi"], span)], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
                ax.plot(r["ratio"], y, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            elif math.isinf(r["ratio"]):
                ax.annotate("never succeeds", (span, y), ha="right", va="center", color=INK2, fontsize=8)
            ax.text(1.01, y, r["verdict"], transform=ax.get_yaxis_transform(), color=INK2, fontsize=8, va="center")
        ax.set_yticks(range(len(g)), labels=list(reversed(g["harness"])), color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(lo_ax, span)
        ticks = [t for t in (0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3) if lo_ax <= t <= span]
        ax.set_xticks(ticks, labels=[f"{t:g}x" for t in ticks])
        ax.minorticks_off()
        ax.set_title(name, loc="left", color=INK, fontsize=10)
    return _save(fig, out / "harness_forest.png", eff)


# ------------------------------------------------------------------------------------------------ 2. decomposition
def decomposition(store: Store, out: Path, exp_id=None) -> Path | None:
    d = E.decompose(store, exp_id)
    if d.empty:
        return None
    plt = _plt()
    d = d.reset_index(drop=True)
    d["effect_of_success"] = -d["d_log_success"]  # lower success raises joules per success
    terms = [("d_log_turns", "Calls per task", BLUE), ("d_log_j_per_turn", "Joules per call", ORANGE),
             ("effect_of_success", "Success (lower costs more)", AQUA)]
    h_in, w_in = 1.7 + 0.62 * len(d), 7.6
    fig = _fig(plt, w_in, h_in, "Where a harness changes joules per success",
               "Change against the baseline, split into the three terms that multiply to joules per success. Dot: total.")
    ax = _axes(fig, h_in, w_in, top_in=1.0, bottom_in=0.85, left_in=1.4, right_in=0.8)
    _style(ax, "x")
    h = 0.22
    for i, r in d.iterrows():
        y = len(d) - 1 - i
        for k, (col, _, color) in enumerate(terms):
            ax.barh(y + (1 - k) * (h + 0.03), r[col], height=h, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.plot(r["d_log_total"], y, "o", color=INK, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(1.01, y, _pct(r["d_log_total"]), transform=ax.get_yaxis_transform(), color=INK, fontsize=9, va="center")
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_yticks(range(len(d)), labels=list(reversed(d["harness"])), color=INK2)
    lo, hi = ax.get_xlim()
    _pct_ticks(ax, lo, hi)
    ax.set_xlim(lo, hi)
    _legend(fig, plt, [(label, color) for _, label, color in terms], x0=0.19, y=0.05, step=0.28)
    return _save(fig, out / "harness_decomposition.png", d)


# ------------------------------------------------------------------------------------------------ 3. frontier
def frontier(store: Store, out: Path, exp_id=None) -> Path | None:
    e = E.efficiency(store, exp_id, by=["device", "harness"])
    if e.empty:
        return None
    plt = _plt()
    e = E.pareto_front(e, within=["device"])
    devices = sorted(e["device"].unique())
    fig = _fig(plt, 4.6 * len(devices) + 0.8, 4.4, "Success against energy for each harness",
               "Up and to the left is better. Line joins the harnesses no other harness beats on both.")
    axes = fig.subplots(1, len(devices), squeeze=False)[0]
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14, wspace=0.28)
    for ax, dev in zip(axes, devices):
        g = e[e["device"] == dev].dropna(subset=["j_per_task", "success_rate"])
        _style(ax, "both")
        front = g[g["on_front"]].sort_values("j_per_task")
        ax.step(front["j_per_task"], front["success_rate"], where="post", color=MUTED, linewidth=1, zorder=1)
        ax.margins(0.12)
        # labels sit at staggered heights above and below their points, joined by a thin line, so near points stay legible
        slots = [(0, 16), (0, -22), (0, 32), (0, -38)]
        for k, (_, r) in enumerate(g.sort_values("j_per_task").iterrows()):
            base = r["harness"] == E.BASELINE
            ax.plot(r["j_per_task"], r["success_rate"], "o", color=BLUE if base else INK2, markersize=9 if base else 7,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(r["harness"], (r["j_per_task"], r["success_rate"]), xytext=slots[k % len(slots)],
                        textcoords="offset points", ha="center", va="center", color=INK if base else INK2, fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 5})
        ax.margins(x=0.1, y=0.32)
        ax.set_xlabel("joules per task", color=INK2, fontsize=9)
        ax.set_ylabel("success rate", color=INK2, fontsize=9)
        ax.set_title(dev, loc="left", color=INK, fontsize=10)
    return _save(fig, out / "harness_frontier.png", e)


# ------------------------------------------------------------------------------------------------ 4. variance
def variance(store: Store, out: Path, exp_id=None) -> Path | None:
    v = E.variance_share(store, exp_id)
    v = v.dropna(subset=["omega2"]) if not v.empty else v
    if v.empty:
        return None
    plt = _plt()
    v = v.sort_values("omega2").reset_index(drop=True)
    fig = _fig(plt, 6.4, 1.9 + 0.5 * len(v), "How much of the variation in joules per success each factor explains",
               "Share of variance (omega squared) after removing task difficulty, with 90% interval.")
    h_in, w_in = 1.9 + 0.5 * len(v), 6.4
    ax = _axes(fig, h_in, w_in, top_in=1.0, bottom_in=0.55, left_in=1.1, right_in=0.5)
    _style(ax, "x")
    for i, r in v.iterrows():
        ax.barh(i, r["omega2"], height=0.38, color=BLUE, zorder=2)
        ax.plot([r["omega2_lo"], r["omega2_hi"]], [i, i], color=INK, linewidth=1.5, zorder=3)
        ax.text(max(r["omega2_hi"], r["omega2"]) + 0.015, i, f"{r['omega2']:.0%}", color=INK, fontsize=9, va="center")
    ax.axvline(0.10, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.10, len(v) - 0.38, " 10% line", color=INK2, fontsize=8, va="center")
    ax.set_ylim(-0.6, len(v) - 0.2)
    ax.set_yticks(range(len(v)), labels=list(v["factor"]), color=INK2)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    return _save(fig, out / "harness_variance.png", v)


# ------------------------------------------------------------------------------------------------ 5. stage energy
def stage_energy(store: Store, out: Path, exp_id=None) -> Path | None:
    s = E.stage_energy(store, exp_id, by=["device", "harness"])
    if s.empty:
        return None
    plt = _plt()
    s = s.reset_index(drop=True)
    parts = [("prefill_share", "Prefill", BLUE), ("decode_share", "Decode", ORANGE),
             ("tool_wait_share", "Tool wait", AQUA), ("other_share", "Other", YELLOW)]
    h_in, w_in = 1.7 + 0.3 * len(s), 7.6
    fig = _fig(plt, w_in, h_in, "Where each turn's energy goes",
               "Share of attributed energy per harness. Right: the harness's own code (templating, parsing, validating).")
    ax = _axes(fig, h_in, w_in, top_in=1.0, bottom_in=0.85, left_in=1.6, right_in=1.2)
    _style(ax, "x")
    labels = [f"{r['device']} {r['harness']}" for _, r in s.iterrows()]
    for i, r in s.iterrows():
        y, left = len(s) - 1 - i, 0.0
        for col, _, color in parts:
            w = max(float(r[col]) if pd.notna(r[col]) else 0.0, 0.0)
            ax.barh(y, w, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
            left += w
        if "harness_code_share" in s.columns and pd.notna(r.get("harness_code_share")):
            ax.text(1.03, y, f"{r['harness_code_share']:.1%}", transform=ax.get_yaxis_transform(), color=INK2,
                    fontsize=8, va="center")
    if "harness_code_share" in s.columns:
        ax.text(1.03, len(s) - 0.35, "harness code", transform=ax.get_yaxis_transform(), color=INK2, fontsize=8)
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.set_yticks(range(len(s)), labels=list(reversed(labels)), color=INK2, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    _legend(fig, plt, [(label, color) for _, label, color in parts], x0=0.21, y=0.045, step=0.16)
    return _save(fig, out / "harness_stage_energy.png", s)


# ------------------------------------------------------------------------------------------------ 6. noise floor
def noise(store: Store, out: Path, exp_id=None, margin: float = E.DEFAULT_MARGIN) -> Path | None:
    n = E.noise_floor(store, exp_id)
    if n.empty:
        return None
    plt = _plt()
    n = n.sort_values("sd_log").reset_index(drop=True)
    chk = E.margin_check(n, margin)
    fig = _fig(plt, 6.4, 3.6, "Run-to-run noise of the baseline harness",
               f"Each dot is one cell. Dashed line: half the margin ({np.log1p(margin) / 2:.3f}). "
               f"The median should sit below it.")
    ax = _axes(fig, 3.6, 6.4, top_in=1.1, bottom_in=0.6, left_in=0.9, right_in=0.4)
    _style(ax, "y")
    ax.plot(range(len(n)), n["sd_log"], "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(np.log1p(margin) / 2, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(chk["median_sd_log"], color=BLUE, linewidth=1)
    ax.text(len(n) - 1, chk["median_sd_log"], f" median {chk['median_sd_log']:.3f}", color=INK2, fontsize=8, va="bottom",
            ha="right")
    ax.set_ylabel("standard deviation of log joules per success", color=INK2, fontsize=9)
    ax.set_xlabel("cells, sorted by noise", color=INK2, fontsize=9)
    ax.set_xticks([])
    ax.set_ylim(bottom=0)
    return _save(fig, out / "harness_noise.png", n)


def make_figures(store: Store, out_dir: str | Path, exp_id=None, margin: float = E.DEFAULT_MARGIN) -> list[Path]:
    """Write every figure that has data. Returns the PNG paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = [forest(store, out, exp_id, margin), decomposition(store, out, exp_id), frontier(store, out, exp_id),
            variance(store, out, exp_id), stage_energy(store, out, exp_id), noise(store, out, exp_id, margin)]
    return [p for p in made if p is not None]
