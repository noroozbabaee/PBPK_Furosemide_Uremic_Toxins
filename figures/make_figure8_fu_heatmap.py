# -*- coding: utf-8 -*-
"""Manuscript Figure 8 — analytical furosemide fu_p × toxin fu_p (IS, pCS, HA).

Same closed-form as heatmap_analytical_formulas.py / analytical_ss.
Style matches analysis/plot_clearance_flux_decomposition.py:
  no grid, no contours, no axhline/axvline, no legend, no suptitle;
  top/right spines off; panel letters A–C; caption lives in LaTeX.

Does not run the ODE. Outputs:
  results/heatmap_fu_analytical_3panels.pdf
  publication_code/Figures/heatmap_fu_analytical_3panels.pdf
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIG, exist_ok=True)
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)

from heatmap_analytical_formulas import (  # noqa: E402
    BP, FU_P, css_furo_theoretical, mm_pct, CSS_PLASMA_TABLE1,
)

# Table 1 CKD-5 residual CL and typical OAT1 fractions (live ODE if you
# re-extract; these match the analytical maps used in the paper draft).
LIVE = {
    "IS":  {"CL_R": 0.20, "F_OAT1": 0.631},
    "HA":  {"CL_R": 0.49, "F_OAT1": 0.70},
    "pCS": {"CL_R": 0.08, "F_OAT1": 0.562},
}
CL_FURO = 1.70
CSS_FURO = css_furo_theoretical(CL_FURO)

N = 80
FU_DRUG = np.linspace(0.01, 0.13, N)
FU_MULT = np.linspace(0.30, 2.00, N)
PANELS = ("IS", "pCS", "HA")  # paper order A IS, B pCS, C HA


def _flux_style() -> None:
    """Same rc as plot_flux_panels (flux decomposition)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "TimesNewRoman", "Liberation Serif",
            "Nimbus Roman", "Times", "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.8,
    })


def _grid_pct(toxin: str) -> np.ndarray:
    css_blood = CSS_PLASMA_TABLE1[toxin] * BP[toxin]
    cl_r = LIVE[toxin]["CL_R"]
    f_oat = LIVE[toxin]["F_OAT1"]
    fu0 = FU_P[toxin]
    z = np.empty((len(FU_MULT), len(FU_DRUG)))
    for i, m in enumerate(FU_MULT):
        for j, fu_d in enumerate(FU_DRUG):
            z[i, j] = mm_pct(
                css_blood, cl_r, f_oat, toxin, CSS_FURO,
                fu_p_furo=float(fu_d), fu_p_tox=float(fu0 * m),
            )["pct"]
    return z


def main() -> None:
    _flux_style()
    grids = {t: _grid_pct(t) for t in PANELS}
    print(f"Css_furo = {CSS_FURO:.3f} mg/L blood  (F=0.60, 240 mg/d, CL={CL_FURO})")
    for t in PANELS:
        g = grids[t]
        print(f"  {t}: {np.nanmin(g):+.2f} to {np.nanmax(g):+.2f} %")

    # Same canvas width as flux 2×2 (12 in); one row of three panels.
    fig = plt.figure(figsize=(12, 3.9), dpi=300)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.22,
    )
    vmax = max(float(np.nanmax(grids[t])) for t in PANELS)
    vmax = max(vmax, 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    letters = "ABC"

    for col, toxin in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, col])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        z = grids[toxin]
        im = ax.pcolormesh(
            FU_DRUG, FU_MULT, z,
            cmap="YlOrRd", norm=norm,
            shading="gouraud",
            rasterized=True,
        )
        ax.set_xlim(FU_DRUG[0], FU_DRUG[-1])
        ax.set_ylim(FU_MULT[0], FU_MULT[-1])
        ax.set_xlabel(r"Furosemide $f_{u,p}$")
        if col == 0:
            ax.set_ylabel("Toxin $f_{u,p}$ multiplier\n"
                          r"($\times$ Deltombe CKD-5 base)")
        ax.text(
            0.02, 0.95, letters[col],
            transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left",
        )
        ax.text(
            0.97, 0.95, toxin,
            transform=ax.transAxes,
            fontsize=12, va="top", ha="right",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta \overline{C}$ (%)")
        cbar.ax.grid(False)
        cbar.outline.set_linewidth(0.8)

    name = "heatmap_fu_analytical_3panels"
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
