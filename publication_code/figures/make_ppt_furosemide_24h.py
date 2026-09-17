# -*- coding: utf-8 -*-
"""Standalone PowerPoint export of five-panel figure panel E.

Final 24 h furosemide plasma cycle, CKD-5 day 30:
  dashed  — TID 120 mg/day (40 mg × 3)
  dotted  — QD  240 mg/day (240 mg × 1)
  solid   — TID 240 mg/day (80 mg × 3)

Reads existing chronic_oral_cache_*.npz files (no ODE re-run).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ROOT = (
    os.path.abspath(os.path.join(_PARENT, ".."))
    if os.path.basename(_PARENT) == "publication_code"
    else _PARENT
)
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIG, exist_ok=True)

RUN_DATE = "20260917"
MODE = "ckd5"
N_DAYS = 30
KI = "0p35"

# Match the five-panel / screenshot style (no-drug omitted: it is ~0).
TRACES = (
    ("TID 120 mg/day (40 mg × 3)", 40.0, 3, "--", "0.35", 2.0),
    ("QD 240 mg/day (240 mg × 1)", 240.0, 1, ":", "0.15", 2.0),
    ("TID 240 mg/day (80 mg × 3)", 80.0, 3, "-", "0.0", 2.6),
)


def _cache_path(dose: float, n_doses: int) -> str:
    return os.path.join(
        _RESULTS,
        f"chronic_oral_cache_Ki{KI}_dose{dose:.4g}mg_x{n_doses}pd_"
        f"{N_DAYS}days_{MODE}_{RUN_DATE}.npz",
    )


def _last_day(t, conc):
    t = np.asarray(t)
    y = np.asarray(conc)
    t0 = 29.0 * 24.0
    m = (t >= t0) & (t <= t0 + 24.0)
    return t[m] - t0, y[m]


def plot(with_legend: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    for t_mark in (8.0, 16.0):
        ax.axvline(t_mark, color="0.82", lw=0.9, zorder=0)

    for label, dose, n, ls, c, lw in TRACES:
        path = _cache_path(dose, n)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=False)
        th, yh = _last_day(d["t_all"], d["drg_all"])
        ax.plot(th, yh, ls=ls, color=c, lw=lw, label=label, zorder=3)
        if yh.size > 1:
            trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            mean = float(trap(yh, th) / max(th[-1] - th[0], 1e-12))
            ax.axhline(mean, color=c, ls=":", lw=0.9, alpha=0.85, zorder=1)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=12, length=4, width=0.9)
    ax.set_xlabel("Time within final 24-h dosing cycle (h)", fontsize=13)
    ax.set_ylabel("Furosemide plasma\nconcentration (mg/L)", fontsize=13)
    ax.text(
        0.04, 0.93, "E", transform=ax.transAxes,
        fontsize=16, fontweight="bold", va="top",
    )
    if with_legend:
        ax.legend(
            loc="upper right", frameon=False, fontsize=10,
            handlelength=2.8, borderaxespad=0.2,
        )
    fig.tight_layout()
    return fig


def _save(fig: plt.Figure, stem: str) -> None:
    for folder in (_RESULTS, _PUB_FIG):
        for ext, dpi in (("png", 400), ("pdf", 400), ("svg", 400)):
            p = os.path.join(folder, f"{stem}.{ext}")
            fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"  saved {p}", flush=True)


def main() -> None:
    fig = plot(with_legend=False)
    _save(fig, "chronic_oral_furosemide_24h_cycle_ppt")
    plt.close(fig)
    fig = plot(with_legend=True)
    _save(fig, "chronic_oral_furosemide_24h_cycle_ppt_legend")
    plt.close(fig)


if __name__ == "__main__":
    main()
