# -*- coding: utf-8 -*-
"""
heatmap_ki_vmax_from_cache.py
===============================
Publication-quality Ki x Vmax sensitivity figure (1 row, 2 panels: A and B),
with a horizontal colourbar below each panel.

RELATIONSHIP TO heatmap_ki_vmax_sensitivity_ode.py
----------------------------------------------------
This script does NOT contain its own copy of the ODE simulation logic.
It imports compute_heatmap() directly from
analysis/heatmap_ki_vmax_sensitivity_ode.py, which is the single source of
truth for how a Ki x Vmax grid point is computed (500h CKD-5 warm-up, 24h
no-drug run, 24h with-drug run, %change extraction).

An earlier version of this script carried its own duplicate copy of that
simulation logic (_run_point, _force_ki, _set_vmax, load_grid), with cache
keys and physics IDENTICAL to heatmap_ki_vmax_sensitivity_ode.py's
compute_heatmap(). Maintaining two independent copies of the same physics
is a real correctness hazard for an open-science codebase: a bug fix
applied to one copy would silently not apply to the other, and the two
could drift apart without anyone noticing until results stopped matching.
This version removes that duplication. The only code that remains here
is genuinely DIFFERENT from heatmap_ki_vmax_sensitivity_ode.py: the figure
layout (GridSpec with a horizontal colourbar row below each heatmap panel,
versus the simpler inline plt.colorbar() in the sensitivity script).

Both scripts read/write the SAME cache file (results/heatmap_cache_correct.pkl)
using the SAME cache-key format, so running either script first will let the
other reuse its cached grid points without recomputation.

USAGE
-----
Run heatmap_ki_vmax_sensitivity_ode.py first (or let this script trigger it
on demand via compute_heatmap, which will run any missing grid points and
cache them) to populate results/heatmap_cache_correct.pkl, then run this
script to produce the publication-styled 2-panel figure.
"""

from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in figures/. The core PBPK engine lives in ../src/, and
# the sibling simulation script (with compute_heatmap) lives in ../analysis/.
_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR       = os.path.join(_THIS_DIR, "..", "src")
_ANALYSIS_DIR  = os.path.join(_THIS_DIR, "..", "analysis")
for _p in (_SRC_DIR, _ANALYSIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Import the single source of truth for grid computation ─────────────────
# compute_heatmap() handles caching internally (results/heatmap_cache_correct.pkl)
# and will run any missing grid points on demand.
from heatmap_ki_vmax_sensitivity_ode import (
    compute_heatmap, Ki_grid, Vmax_grid, N_KI, N_VMAX,
    DOSE_MG, INFUSION_H,
)

# ── Figure-specific configuration (genuinely distinct from the sim script) ──
CURRENT_KI    = 0.35
CURRENT_SCALE = 0.05

PANELS = [
    {"toxin": "IS",  "panel": "A", "cmap": "YlOrRd", "clinical": 30.0},
    {"toxin": "pCS", "panel": "B", "cmap": "YlOrRd", "clinical": 30.0},
]


# ── Load both grids ───────────────────────────────────────────────────────
print("="*65)
print("Ki x Vmax combined figure — loading grids via compute_heatmap()")
print("(shared cache with heatmap_ki_vmax_sensitivity_ode.py: "
      f"{os.path.join(RESULTS_DIR, 'heatmap_cache_correct.pkl')})")
print("="*65)
try:
    import constants as K; K.VERBOSE = False
except Exception:
    pass

grids = {}
for p in PANELS:
    print(f"\n-- {p['toxin']} --")
    grids[p["toxin"]] = compute_heatmap(p["toxin"], use_cache=True)


# ── Figure ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         13,
    "axes.linewidth":    1.0,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  5,
    "ytick.major.size":  5,
})

n_panels = len(PANELS)

# GridSpec: heatmap row + colourbar row, 2 columns
fig = plt.figure(figsize=(15, 9), dpi=300)
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(
    2, n_panels, figure=fig,
    height_ratios=[1.0, 0.055],
    hspace=0.22,
    wspace=0.32,
    left=0.08, right=0.97,
    top=0.93, bottom=0.12,
)

for col, cfg in enumerate(PANELS):
    toxin  = cfg["toxin"]
    panel  = cfg["panel"]
    cmap   = cfg["cmap"]
    clin   = cfg["clinical"]
    pct_2d = grids[toxin]

    ax    = fig.add_subplot(gs[0, col])
    ax_cb = fig.add_subplot(gs[1, col])

    vmin = 0.0
    vmax = float(np.nanpercentile(pct_2d, 98))
    if vmax < 1.0: vmax = float(np.nanmax(pct_2d)) + 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ── Heatmap ───────────────────────────────────────────────────────────
    im = ax.pcolormesh(Ki_grid, Vmax_grid, pct_2d,
                       cmap=cmap, norm=norm,
                       shading="auto", rasterized=True)

    # ── Clinical +30% contour ─────────────────────────────────────────────
    if np.nanmax(pct_2d) >= clin:
        try:
            cs = ax.contour(Ki_grid, Vmax_grid, pct_2d,
                            levels=[clin], colors=["#1A1A8C"],
                            linewidths=2.5, linestyles=["--"])
            ax.clabel(cs, fmt=f"+{clin:.0f}% (clinical)",
                      fontsize=10, colors="#1A1A8C",
                      inline=True, inline_spacing=6)
        except: pass

    # ── +10% guide contour ────────────────────────────────────────────────
    if np.nanmax(pct_2d) >= 10.0:
        try:
            cs10 = ax.contour(Ki_grid, Vmax_grid, pct_2d,
                              levels=[10.0], colors=["#444444"],
                              linewidths=0.9, linestyles=[":"], alpha=0.8)
            ax.clabel(cs10, fmt="+10%", fontsize=9,
                      colors="#444444", inline=True, inline_spacing=4)
        except: pass

    # ── Current model star ────────────────────────────────────────────────
    ax.scatter([CURRENT_KI], [CURRENT_SCALE],
               s=300, marker="*",
               color="white", edgecolors="black",
               linewidths=1.4, zorder=10,
               label=(f"Current model\n"
                      f"$K_i={CURRENT_KI}$ mg/L, "
                      f"scale$={CURRENT_SCALE}$"))

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlim(Ki_grid[0],   Ki_grid[-1])
    ax.set_ylim(Vmax_grid[0], Vmax_grid[-1])
    ax.set_xlabel(r"$K_i$ — OAT1 inhibition constant (mg/L)",
                  fontsize=13, labelpad=5)
    if col == 0:
        ax.set_ylabel(
            r"CKD-5 OAT1 $V_{max}$ scale" + "\n" +
            r"(fraction of healthy $V_{max}$)",
            fontsize=12, labelpad=5)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x,_: (f"{x:.2f}" if x<0.1
                     else f"{x:.1f}" if x<1
                     else f"{x:.0f}")))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.tick_params(labelsize=11)

    # ── Panel letter only ─────────────────────────────────────────────────
    ax.set_title(panel, fontweight="bold", loc="left",
                 fontsize=17, pad=7)

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(fontsize=9.5, frameon=True, loc="upper right",
              framealpha=0.92, edgecolor="#aaaaaa",
              handletextpad=0.3, borderpad=0.5)

    # ── Colourbar below — fully outside heatmap ───────────────────────────
    cbar = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cbar.set_label(rf"$\Delta C_{{ss}}$ {toxin} (%)",
                   fontsize=12, labelpad=5)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    tick_step = max(2, int(np.floor(vmax/6/2)*2))
    cb_ticks  = np.arange(0, vmax+tick_step, tick_step)
    cbar.set_ticks(cb_ticks[cb_ticks <= vmax*1.01])

# ── Shared suptitle ───────────────────────────────────────────────────────
fig.suptitle(
    r"Sensitivity of uremic toxin accumulation to $K_i$ and"
    r" CKD-5 OAT1 $V_{max}$ scale"
    "\n"
    f"Acute IV {DOSE_MG:.0f}\u202fmg furosemide over "
    f"{INFUSION_H:.0f}\u202fh  |  CKD-5  |  "
    f"{N_KI}\u00d7{N_VMAX} ODE grid  |  "
    "Blue dashed = +30% (Costes-Albrespic 2025)",
    fontsize=11, fontweight="bold", y=0.997,
)

# ── Save ──────────────────────────────────────────────────────────────────
for ext in ("png","pdf"):
    p = os.path.join(RESULTS_DIR, f"heatmap_ki_vmax_combined.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[saved] {p}")
plt.close(fig)
print("Done.")