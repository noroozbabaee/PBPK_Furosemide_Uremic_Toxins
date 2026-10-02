# -*- coding: utf-8 -*-
"""Manuscript Figure 7 — analytical Ki × Vmax, chronic oral CKD-5.

Same 3-panel look as Figure 8: A) IS, B) HA, C) pCS; one shared
colourbar. Black star = model
point (Ki=0.35 mg/L, Vmax=0.05 × healthy OAT1). No contours, no
Costes/Deltombe dashed lines, no legend.

Not the IV ODE map (figures/heatmap_ki_vmax_from_cache.py).

Y-axis is the basolateral OAT1 Vmax as a fraction of the *healthy*
value (it replaces the CKD-5 scale of 0.05; it is not “× CKD-5 Vmax”).

Math (heatmap_analytical_formulas.mm_pct):
  Cu_furo = Css_plasma × fu_p   (no extra /BP)
  CL_GFR  = Q_GFR × fu_p / BP
  CL_OAT  scales with Vmax / 0.05; CL_R = CL_GFR + CL_OAT
  no-drug Css_blood scales as 1/CL_R
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ROOT = (
    os.path.abspath(os.path.join(_PARENT, ".."))
    if os.path.basename(_PARENT) == "publication_code"
    else _PARENT
)
_ANALYSIS = os.path.join(_ROOT, "analysis")
_SRC = os.path.join(_ROOT, "src")
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIG, exist_ok=True)
for _p in (_SRC, _ANALYSIS, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from heatmap_analytical_formulas import (  # noqa: E402
    BP, FU_P, mm_pct, f_oat1_from_gfr, q_gfr_ckd5,
)
from make_figure8_fu_heatmap import _load_live, _style, _panel_label  # noqa: E402

N_KI, N_VMAX = 80, 80
KI_GRID = np.logspace(np.log10(0.05), np.log10(5.0), N_KI)
VMAX_GRID = np.linspace(0.01, 0.30, N_VMAX)
VMAX_REF = 0.05
KI_STAR = 0.35
PANELS = ("IS", "HA", "pCS")


def _grid_pct(toxin: str, live: dict, css_furo: float) -> np.ndarray:
    css_ref = {"IS": live["IS_blood"], "HA": live["HA_blood"],
               "pCS": live["pCS_blood"]}[toxin]
    cl_r_ref = live["CL_R"][toxin]
    fu = FU_P[toxin]
    bp = BP[toxin]
    qg = live["Q_GFR"]
    cl_gfr = qg * fu / bp
    cl_oat_ref = max(cl_r_ref - cl_gfr, 0.0)
    z = np.empty((len(VMAX_GRID), len(KI_GRID)))
    for i, vmax in enumerate(VMAX_GRID):
        cl_oat = cl_oat_ref * (float(vmax) / VMAX_REF)
        cl_r = cl_gfr + cl_oat
        f_oat = cl_oat / cl_r if cl_r > 1e-12 else 0.0
        css_blood = css_ref * cl_r_ref / cl_r if cl_r > 1e-12 else css_ref
        for j, ki in enumerate(KI_GRID):
            z[i, j] = mm_pct(
                css_blood, cl_r, f_oat, toxin, css_furo, ki=float(ki),
            )["pct"]
    return z


def main() -> None:
    _style()
    live = _load_live()
    css_furo = live["Css_furo_plasma"]
    print("Figure 7 analytical Ki × Vmax (chronic oral, not IV ODE)")
    print(f"Css_furo = {css_furo:.3f} mg/L plasma")
    print(f"Q_GFR CKD-5 = {live['Q_GFR']:.3f} L/h")
    for t in PANELS:
        fo = f_oat1_from_gfr(live["CL_R"][t], FU_P[t], BP[t], live["Q_GFR"])
        print(f"  {t}: CL_R={live['CL_R'][t]:.4f}  F_OAT1(at V=0.05)={fo:.3f}")

    fig = plt.figure(figsize=(11.2, 3.55), dpi=300)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.28, left=0.07, right=0.90, top=0.88, bottom=0.18,
    )
    grids = []
    for toxin in PANELS:
        z = _grid_pct(toxin, live, css_furo)
        print(f"  {toxin} map: {np.nanmin(z):+.2f} to {np.nanmax(z):+.2f} %")
        grids.append((toxin, z))
    zmax = max(float(np.nanmax(z)) for _t, z in grids)
    vmax = float(np.ceil(max(zmax, 1.0) / 5.0) * 5.0)
    print(f"  SHARED colorbar for A, B, C: 0 to {vmax:.0f} %")
    norm = Normalize(vmin=0.0, vmax=vmax)

    axes = []
    im0 = None
    for col, (toxin, z) in enumerate(grids):
        ax = fig.add_subplot(gs[0, col])
        ax.grid(False)
        im = ax.pcolormesh(
            KI_GRID, VMAX_GRID, z,
            cmap="YlOrRd",
            norm=norm,
            shading="gouraud",
            rasterized=True,
        )
        im.set_clim(0.0, vmax)
        if im0 is None:
            im0 = im
        axes.append(ax)
        ax.scatter(
            [KI_STAR], [VMAX_REF],
            s=55, marker="*", color="black", zorder=5,
        )
        ax.set_xscale("log")
        ax.set_xlim(KI_GRID[0], KI_GRID[-1])
        ax.set_ylim(VMAX_GRID[0], VMAX_GRID[-1])
        ax.set_xlabel(r"$K_i$ (mg/L)")
        if col == 0:
            ax.set_ylabel(r"OAT1 $V_{\max}$ / healthy")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: (
                f"{x:.2f}" if x < 0.1 else f"{x:.1f}" if x < 1 else f"{x:.0f}"
            )
        ))
        _panel_label(ax, col, toxin)

    cbar = fig.colorbar(im0, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\Delta C_{\mathrm{ss}}$ (%)")
    cbar.ax.grid(False)
    cbar.set_ticks(np.linspace(0.0, vmax, 6))

    name = "heatmap_ki_vmax_analytical"
    for folder in (_RESULTS, _PUB_FIG):
        for ext in ("pdf", "png"):
            path = os.path.join(folder, f"{name}.{ext}")
            fig.savefig(
                path, dpi=300, bbox_inches="tight", pad_inches=0.04,
                facecolor="white", edgecolor="none",
            )
            print(f"[saved] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
