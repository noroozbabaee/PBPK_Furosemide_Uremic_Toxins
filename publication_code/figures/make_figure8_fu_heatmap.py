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
_SRC = os.path.join(_ROOT, "src")
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIG, exist_ok=True)
for _p in (_SRC, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from heatmap_analytical_formulas import (  # noqa: E402
    BP, FU_P, css_furo_theoretical, mm_pct, CSS_PLASMA_TABLE1,
    CL_R_TABLE1, f_oat1_from_gfr, q_gfr_ckd5,
)

N = 80
FU_DRUG = np.linspace(0.01, 0.13, N)
FU_MULT = np.linspace(0.30, 2.00, N)
PANELS = ("IS", "pCS", "HA")


def _scalar(cache, key, default=None):
    if key not in cache.files:
        return default
    return float(np.asarray(cache[key]).reshape(-1)[0])


def _load_live():
    """Prefer a CKD-5 chronic NPZ (CL_FURO, CL_R, blood baselines)."""
    qg = q_gfr_ckd5()
    live = {
        "CL_FURO": 1.70,
        "source": "Table 1 + Q_GFR (no chronic NPZ)",
        "IS_blood": CSS_PLASMA_TABLE1["IS"] * BP["IS"],
        "HA_blood": CSS_PLASMA_TABLE1["HA"] * BP["HA"],
        "pCS_blood": CSS_PLASMA_TABLE1["pCS"] * BP["pCS"],
        "CL_R": dict(CL_R_TABLE1),
        "Q_GFR": qg,
    }
    search = []
    for folder in (_RESULTS, os.path.join(_ROOT, "publication_code", "results")):
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.endswith(".npz") and "chronic" in name.lower():
                    search.append(os.path.join(folder, name))
    search.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for path in search:
        try:
            z = np.load(path, allow_pickle=False)
        except Exception:
            continue
        is0 = _scalar(z, "IS_base")
        if is0 is None or not (8.0 <= is0 <= 80.0):
            continue
        cl = _scalar(z, "CL_FURO")
        if cl is None or cl < 0.2:
            continue
        live["CL_FURO"] = cl
        live["source"] = os.path.basename(path)
        live["IS_blood"] = is0
        live["pCS_blood"] = _scalar(z, "pCS_base", live["pCS_blood"])
        live["HA_blood"] = _scalar(z, "HA_base", live["HA_blood"])
        for tox, key in (("IS", "CL_R_IS"), ("HA", "CL_R_HA"), ("pCS", "CL_R_pCS")):
            v = _scalar(z, key)
            if v is not None and v > 0:
                live["CL_R"][tox] = v
        print(f"  live PK from {live['source']}")
        print(f"  CL_FURO={live['CL_FURO']:.4f} L/h plasma  "
              f"IS_base_blood={live['IS_blood']:.3f}")
        break
    else:
        print(f"  [warn] {live['source']}")
    return live


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


def _grid_pct(toxin: str, live: dict, css_furo: float) -> np.ndarray:
    css_blood = {"IS": live["IS_blood"], "HA": live["HA_blood"],
                 "pCS": live["pCS_blood"]}[toxin]
    cl_r0 = live["CL_R"][toxin]
    fu0 = FU_P[toxin]
    bp = BP[toxin]
    qg = live["Q_GFR"]
    cl_gfr0 = qg * fu0 / bp
    cl_oat0 = max(cl_r0 - cl_gfr0, 0.0)
    z = np.empty((len(FU_MULT), len(FU_DRUG)))
    for i, m in enumerate(FU_MULT):
        fu_t = float(fu0 * m)
        cl_gfr = qg * fu_t / bp
        cl_r = cl_gfr + cl_oat0
        f_oat = cl_oat0 / cl_r if cl_r > 1e-12 else 0.0
        for j, fu_d in enumerate(FU_DRUG):
            z[i, j] = mm_pct(
                css_blood, cl_r, f_oat, toxin, css_furo,
                fu_p_furo=float(fu_d), fu_p_tox=fu_t,
            )["pct"]
    return z


def main() -> None:
    _flux_style()
    live = _load_live()
    css_furo = css_furo_theoretical(live["CL_FURO"])
    print(f"Css_furo = {css_furo:.3f} mg/L plasma  "
          f"(F=0.60, 240 mg/d, CL={live['CL_FURO']:.4f})")
    print(f"Q_GFR CKD-5 = {live['Q_GFR']:.3f} L/h")
    for t in PANELS:
        fo = f_oat1_from_gfr(live["CL_R"][t], FU_P[t], BP[t], live["Q_GFR"])
        print(f"  {t}: CL_R={live['CL_R'][t]:.4f}  F_OAT1(base)={fo:.3f}")
    grids = {t: _grid_pct(t, live, css_furo) for t in PANELS}
    for t in PANELS:
        g = grids[t]
        print(f"  {t} map: {np.nanmin(g):+.2f} to {np.nanmax(g):+.2f} %")

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
