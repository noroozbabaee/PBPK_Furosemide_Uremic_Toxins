# -*- coding: utf-8 -*-
"""Supplementary Ki × Vmax ODE figure — single IV dose, not manuscript Fig 7.

Manuscript Fig 7 is the analytical chronic oral map
(heatmap_ki_vmax_analytical.pdf), which is not in this git tree.

This plot uses heatmap_ki_vmax_engine.compute_heatmaps (default
SCENARIO=single_iv): 24 h no-drug vs 24 h IV 120 mg / 2 h. Colour is
%Δ of C_tox*_cen at t = 24 h, not ΔC_ss. No Costes +30% contour
(Section 2.3: a single dose cannot reproduce that signal).

For the chronic ODE grid: HEATMAP_SCENARIO=chronic_oral.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
_ANALYSIS_DIR = os.path.join(_THIS_DIR, "..", "analysis")
for _p in (_SRC_DIR, _ANALYSIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from heatmap_ki_vmax_engine import (  # noqa: E402
    compute_heatmaps, Ki_grid, Vmax_grid, N_KI, N_VMAX, DOSE_MG, INFUSION_H,
)

CURRENT_KI = 0.35
CURRENT_SCALE = 0.05
PANELS = [
    {"toxin": "IS", "panel": "A"},
    {"toxin": "pCS", "panel": "B"},
    {"toxin": "HA", "panel": "C"},
]


def _set_times() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next(
        (n for n in ("Times New Roman", "TimesNewRoman", "Nimbus Roman",
                     "Liberation Serif", "Times") if n in available),
        "serif",
    )
    plt.rcParams.update({
        "font.family": chosen,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


print("=" * 65)
print("Supplementary IV Ki×Vmax (not Fig 7). Engine: heatmap_ki_vmax_engine")
print("Do not reuse heatmap_cache_correct.pkl (no pKa in that key).")
print("=" * 65)
try:
    import constants as K
    K.VERBOSE = False
except Exception:
    pass

grids = compute_heatmaps(scenario="single_iv", use_cache=True)
_set_times()

fig = plt.figure(figsize=(6.27, 2.55), dpi=300)
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    height_ratios=[1.0, 0.06],
    hspace=0.12, wspace=0.18,
    left=0.10, right=0.98, top=0.90, bottom=0.22,
)

for col, cfg in enumerate(PANELS):
    toxin = cfg["toxin"]
    pct_2d = grids[toxin]
    ax = fig.add_subplot(gs[0, col])
    ax_cb = fig.add_subplot(gs[1, col])
    vmax = float(np.nanpercentile(pct_2d, 98)) if np.any(~np.isnan(pct_2d)) else 1.0
    if vmax < 1.0:
        vmax = float(np.nanmax(pct_2d)) + 0.5
    norm = Normalize(vmin=0.0, vmax=vmax)
    im = ax.pcolormesh(
        Ki_grid, Vmax_grid, pct_2d,
        cmap="YlOrRd", norm=norm, shading="auto", rasterized=True,
    )
    if np.nanmax(pct_2d) >= 10.0:
        try:
            cs10 = ax.contour(
                Ki_grid, Vmax_grid, pct_2d, levels=[10.0],
                colors=["#444444"], linewidths=0.6, linestyles=[":"],
            )
            ax.clabel(cs10, fmt="+10%", fontsize=6, colors="#444444")
        except Exception:
            pass
    ax.scatter(
        [CURRENT_KI], [CURRENT_SCALE],
        s=40, marker="*", color="white", edgecolors="black",
        linewidths=0.5, zorder=10,
    )
    ax.set_xscale("log")
    ax.set_xlim(Ki_grid[0], Ki_grid[-1])
    ax.set_ylim(Vmax_grid[0], Vmax_grid[-1])
    ax.set_xlabel(r"$K_i$ (mg/L)", labelpad=2)
    if col == 0:
        ax.set_ylabel(
            r"OAT1 $V_{\mathrm{max}}$ / healthy",
            labelpad=2,
        )
    else:
        ax.set_ylabel("")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: (f"{x:.2f}" if x < 0.1 else f"{x:.1f}" if x < 1 else f"{x:.0f}")
    ))
    ax.set_title(cfg["panel"], fontweight="bold", loc="left", fontsize=8, pad=2)
    ax.text(0.97, 0.95, toxin, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    cbar = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cbar.set_label(r"$\Delta C$ at 24 h (%)", fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6)

for ext in ("png", "pdf"):
    p = os.path.join(RESULTS_DIR, f"heatmap_ki_vmax_iv_supplement.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"[saved] {p}")
plt.close(fig)
print("Grid %d×%d | %.0f mg IV / %.0f h | star Ki=%.2f, Vmax=%.2f healthy"
      % (N_KI, N_VMAX, DOSE_MG, INFUSION_H, CURRENT_KI, CURRENT_SCALE))
