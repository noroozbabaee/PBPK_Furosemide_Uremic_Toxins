# -*- coding: utf-8 -*-
"""
combine_bar_figures_vector.py

Builds Figure 3: shared 2×2 vector figure
  A/B = Healthy (concentration, clearance)
  C/D = CKD-5 (concentration, clearance)

Fixes relative to the broken Windows copy:
  1. Import pipelines via module aliases (no UnboundLocalError).
  2. Call the live APIs (``axes=``, ``show_legend=``, CKD needs
     ``idx_map`` and ``clearance_fn``). Do not use ``axes_pair`` /
     ``draw_legend`` / ``lit_label`` — those kwargs do not exist.
  3. One four-entry legend (no Healthy/CKD in the legend text).
     Row labels carry Healthy vs CKD-5.
  4. Panel D has no dialysis-session clearance diamonds (Aurélie).
     Panel B keeps healthy literature clearance error bars.
  5. Writes to repo ``results/`` and ``publication_code/Figures/``.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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

import plot_healthy_lit_comparison_3scenarios as _healthy_mod  # noqa: E402
import plot_ckd_lit_comparison_3scenarios as _ckd_mod  # noqa: E402
from state_vector import IDX  # noqa: E402
import constants as K  # noqa: E402

K.VERBOSE = False

# Table 1 clinical plasma (EUTox); healthy renal CL from the cited sources.
# CKD-5 clinical CL is omitted (dialysis session ≠ residual CL_R).
LIT_CONC_HEALTHY = {
    "IS":  {"value": 0.54, "range": (0.34, 0.74)},
    "HA":  {"value": 3.00, "range": (1.00, 5.00)},
    "pCS": {"value": 1.70, "range": (1.10, 2.30)},
}
LIT_CLEAR_HEALTHY = {
    "IS":  {"value": 3.80,  "range": (3.48, 4.12)},
    "HA":  {"value": 27.80, "range": (23.30, 32.30)},
    "pCS": {"value": 1.60,  "range": (1.00, 2.20)},
}
LIT_CONC_CKD = {
    "IS":  {"value": 37.0, "range": (10.5, 63.5)},
    "HA":  {"value": 109.0, "range": (44.4, 173.6)},
    "pCS": {"value": 23.0, "range": (6.1, 39.9)},
}
# Placeholder only — not drawn on Panel D.
LIT_CLEAR_CKD_UNUSED = {
    "IS":  {"value": 1.6,  "range": (1.38, 1.82)},
    "HA":  {"value": 6.06, "range": (4.96, 7.16)},
    "pCS": {"value": 0.1,  "range": (0.0, 0.63)},
}


def _strip_errorbar_artists(ax) -> None:
    """Drop ErrorbarContainer artists (literature diamonds) from one axes."""
    for cont in list(ax.containers):
        if isinstance(cont, ErrorbarContainer):
            try:
                cont.remove()
            except NotImplementedError:
                for artist in list(cont.get_children()):
                    artist.remove()


def _row_label(ax, text: str) -> None:
    ax.annotate(
        text,
        xy=(-0.22, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )


def main() -> None:
    print("=== Running healthy pipeline ===", flush=True)
    fig_t_h, fig_b_h, solutions_healthy = _healthy_mod.run_healthy_and_plot()
    plt.close(fig_t_h)
    plt.close(fig_b_h)

    print("\n=== Running CKD pipeline ===", flush=True)
    fig_t_c, fig_b_c, solutions_ckd = _ckd_mod.run_ckd_and_plot()
    plt.close(fig_t_c)
    plt.close(fig_b_c)

    scenario_names_healthy = [
        "Healthy No Furosemide",
        "Healthy With Furosemide (No Interaction)",
        "Healthy With Furosemide",
    ]
    colors_healthy = {
        scenario_names_healthy[0]: "#4d4d4d",
        scenario_names_healthy[1]: "#9a9a9a",
        scenario_names_healthy[2]: "#1a1a1a",
    }
    scenario_names_ckd = [
        "CKD No Furosemide",
        "CKD With Furosemide (No Interaction)",
        "CKD With Furosemide",
    ]
    colors_ckd = {
        scenario_names_ckd[0]: "#4d4d4d",
        scenario_names_ckd[1]: "#9a9a9a",
        scenario_names_ckd[2]: "#1a1a1a",
    }

    toxins = ["IS", "HA", "pCS"]
    toxin_states = ["C_tox1_cen", "C_tox2_cen", "C_tox3_cen"]
    ae_states = ["Ae_t1", "Ae_t2", "Ae_t3"]
    window = (0.0, 12.0)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.38, wspace=0.30,
        left=0.12, right=0.97, top=0.96, bottom=0.14,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    _healthy_mod.create_figure_bars_generic(
        solutions=solutions_healthy,
        window=window,
        scenario_names=scenario_names_healthy,
        colors=colors_healthy,
        toxins=toxins,
        toxin_states=toxin_states,
        ae_states=ae_states,
        lit_conc=LIT_CONC_HEALTHY,
        lit_clear=LIT_CLEAR_HEALTHY,
        fig=fig,
        axes=(ax_A, ax_B),
        panel_letters=("A", "B"),
        show_legend=False,
        apply_tight_layout=False,
    )
    _ckd_mod.create_figure_bars_generic(
        solutions=solutions_ckd,
        idx_map=IDX,
        window=window,
        scenario_names=scenario_names_ckd,
        colors=colors_ckd,
        toxins=toxins,
        toxin_states=toxin_states,
        ae_states=ae_states,
        lit_conc=LIT_CONC_CKD,
        lit_clear=LIT_CLEAR_CKD_UNUSED,
        clearance_fn=_ckd_mod._clearance_auc_window,
        fig=fig,
        axes=(ax_C, ax_D),
        panel_letters=("C", "D"),
        show_legend=False,
        apply_tight_layout=False,
    )

    # CKD module already omits Panel D literature CL; strip anyway if present.
    _strip_errorbar_artists(ax_D)

    _row_label(ax_A, "Healthy")
    _row_label(ax_C, "CKD-5")

    bar_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#4d4d4d",
                  edgecolor="black", linewidth=0.6),
        Rectangle((0, 0), 1, 1, facecolor="#9a9a9a",
                  edgecolor="black", linewidth=0.6),
        Rectangle((0, 0), 1, 1, facecolor="#1a1a1a",
                  edgecolor="black", linewidth=0.6),
    ]
    lit_handle = Line2D(
        [], [], color="black", marker="D", linestyle="none",
        markersize=8, markerfacecolor="black", markeredgecolor="black",
    )
    fig.legend(
        bar_handles + [lit_handle],
        [
            "No Furosemide",
            "With Furosemide (No Interaction)",
            "With Furosemide",
            "Literature",
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.8,
        handlelength=2.2,
        handletextpad=0.6,
    )

    for folder in (_RESULTS, _PUB_FIG):
        for ext in ("pdf", "png"):
            path = os.path.join(folder, f"bars_healthy_ckd_combined.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[saved] {path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
