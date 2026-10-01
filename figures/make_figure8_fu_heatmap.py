# -*- coding: utf-8 -*-
"""Manuscript Figure 8 — original 3-panel furosemide fu × toxin fu.

Layout matches the paper screenshot: A IS, B HA, C pCS; y-axis is toxin
fu_p after multiplication (not a shared ×Deltombe axis). Black star =
model point (fu_furo=0.05, Deltombe toxin fu).

No contours, no Deltombe/Costes dashed lines, no legend.

Math: Cu_furo = Css_plasma × fu_p (no extra /BP);
F_OAT1 from Q_GFR and CL_R.
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
    BP, BP_FURO, FU_P, FU_P_FURO, css_furo_theoretical, mm_pct,
    CSS_PLASMA_TABLE1, CL_R_TABLE1, f_oat1_from_gfr, q_gfr_ckd5,
)

N = 80
# Same window as the original figure (caption 0.01–0.15; y ≈ 0.4–2.5 × base).
FU_DRUG = np.linspace(0.01, 0.15, N)
FU_MULT = np.linspace(0.40, 2.50, N)
PANELS = ("IS", "HA", "pCS")
STAR_FU_FURO = FU_P_FURO


def _scalar(cache, key, default=None):
    if key not in getattr(cache, "files", ()):
        return default
    return float(np.asarray(cache[key]).reshape(-1)[0])


def _npz_score(path: str) -> tuple:
    name = os.path.basename(path).lower()
    prefer = 0
    if "dose80mg_x3" in name or "x3pd" in name and "80" in name:
        prefer = 2
    elif "ckd5" in name:
        prefer = 1
    return (prefer, os.path.getmtime(path))


def _load_live():
    qg = q_gfr_ckd5()
    live = {
        "CL_FURO": 1.70,
        "source": "Table 1 + Q_GFR (no chronic NPZ)",
        "IS_blood": CSS_PLASMA_TABLE1["IS"] * BP["IS"],
        "HA_blood": CSS_PLASMA_TABLE1["HA"] * BP["HA"],
        "pCS_blood": CSS_PLASMA_TABLE1["pCS"] * BP["pCS"],
        "CL_R": dict(CL_R_TABLE1),
        "Q_GFR": qg,
        "Css_furo_plasma": 1.68,
    }
    search = []
    for folder in (_RESULTS, os.path.join(_ROOT, "publication_code", "results")):
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.endswith(".npz") and "chronic" in name.lower():
                    search.append(os.path.join(folder, name))
    search.sort(key=_npz_score, reverse=True)
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
        ode_css = _scalar(z, "anal_ode_Css_furo")
        drg = _scalar(z, "drg_avg")
        if ode_css is not None and ode_css > 0.05:
            live["Css_furo_plasma"] = ode_css
        elif drg is not None and drg > 0.05:
            # ODE C_ven is blood; formula needs plasma.
            live["Css_furo_plasma"] = drg / BP_FURO if drg > 2.5 else drg
        print(f"  live PK from {live['source']}")
        print(f"  CL_FURO={live['CL_FURO']:.4f} L/h plasma  "
              f"IS_base_blood={live['IS_blood']:.3f}")
        break
    else:
        print(f"  [warn] {live['source']}")
    return live


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "TimesNewRoman", "Liberation Serif",
            "Nimbus Roman", "Times", "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })


def _grid_pct(toxin: str, live: dict, css_furo: float, y_fu: np.ndarray) -> np.ndarray:
    css_blood = {"IS": live["IS_blood"], "HA": live["HA_blood"],
                 "pCS": live["pCS_blood"]}[toxin]
    cl_r0 = live["CL_R"][toxin]
    fu0 = FU_P[toxin]
    bp = BP[toxin]
    qg = live["Q_GFR"]
    cl_gfr0 = qg * fu0 / bp
    cl_oat0 = max(cl_r0 - cl_gfr0, 0.0)
    z = np.empty((len(y_fu), len(FU_DRUG)))
    for i, fu_t in enumerate(y_fu):
        cl_gfr = qg * fu_t / bp
        cl_r = cl_gfr + cl_oat0
        f_oat = cl_oat0 / cl_r if cl_r > 1e-12 else 0.0
        for j, fu_d in enumerate(FU_DRUG):
            z[i, j] = mm_pct(
                css_blood, cl_r, f_oat, toxin, css_furo,
                fu_p_furo=float(fu_d), fu_p_tox=float(fu_t),
            )["pct"]
    return z


def main() -> None:
    _style()
    live = _load_live()
    css_furo = live["Css_furo_plasma"]
    if css_furo is None:
        css_furo = css_furo_theoretical(live["CL_FURO"])
    print(f"Css_furo = {css_furo:.3f} mg/L plasma  "
          f"(map input; CL={live['CL_FURO']:.4f})")
    print(f"Q_GFR CKD-5 = {live['Q_GFR']:.3f} L/h")
    for t in PANELS:
        fo = f_oat1_from_gfr(live["CL_R"][t], FU_P[t], BP[t], live["Q_GFR"])
        print(f"  {t}: CL_R={live['CL_R'][t]:.4f}  F_OAT1(base)={fo:.3f}  "
              f"fu_p base={FU_P[t]:.3f}")

    fig = plt.figure(figsize=(11.2, 3.55), dpi=300)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.38, left=0.07, right=0.98, top=0.88, bottom=0.18,
    )
    letters = "ABC"

    for col, toxin in enumerate(PANELS):
        y_fu = FU_P[toxin] * FU_MULT
        z = _grid_pct(toxin, live, css_furo, y_fu)
        print(f"  {toxin} map: {np.nanmin(z):+.2f} to {np.nanmax(z):+.2f} %")
        ax = fig.add_subplot(gs[0, col])
        ax.grid(False)
        vmax = max(float(np.nanmax(z)), 1.0)
        im = ax.pcolormesh(
            FU_DRUG, y_fu, z,
            cmap="YlOrRd",
            norm=Normalize(vmin=0.0, vmax=vmax),
            shading="gouraud",
            rasterized=True,
        )
        ax.scatter(
            [STAR_FU_FURO], [FU_P[toxin]],
            s=55, marker="*", color="black", zorder=5,
        )
        ax.set_xlim(FU_DRUG[0], FU_DRUG[-1])
        ax.set_ylim(y_fu[0], y_fu[-1])
        ax.set_xlabel(r"Furosemide $f_{u,p}$")
        if col == 0:
            ax.set_ylabel(r"Toxin $f_{u,p}$ (after multiplication)")
        ax.text(
            0.03, 0.97, f"{letters[col]}\n{toxin}",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta C_{\mathrm{ss}}$ (%)")
        cbar.ax.grid(False)

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
