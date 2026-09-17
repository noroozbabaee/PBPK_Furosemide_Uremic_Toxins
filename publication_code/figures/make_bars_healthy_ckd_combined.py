# -*- coding: utf-8 -*-
"""Build Figure 3: Healthy (A/B) + CKD-5 (C/D) bars with one 4-entry legend.

Row labels carry Healthy vs CKD-5. Legend is only:

  No Furosemide
  With Furosemide (No Interaction)
  With Furosemide
  Literature

Requires the two three-scenario ODE runs (imported from analysis/).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
# Nested under publication_code/figures → still use the repo src/analysis/results
if os.path.basename(_PARENT) == "publication_code":
    _ROOT = os.path.abspath(os.path.join(_PARENT, ".."))
else:
    _ROOT = _PARENT
_ANALYSIS = os.path.join(_ROOT, "analysis")
_SRC = os.path.join(_ROOT, "src")
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIGURES = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIGURES, exist_ok=True)
for _p in (_SRC, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_healthy_lit_comparison_3scenarios import (  # noqa: E402
    create_figure_healthy_bars,
    run_healthy_and_plot,
)
from plot_ckd_lit_comparison_3scenarios import (  # noqa: E402
    create_figure_bars_generic as create_ckd_bars,
    run_ckd_and_plot,
    TOXINS,
    TOXIN_STATES,
    AE_STATES,
    LIT_CONC_CKD,
    LIT_CLEAR_CKD,
    _clearance_auc_window,
)
from state_vector import IDX  # noqa: E402


def build_combined_bars_figure(solutions_healthy, solutions_ckd, window_h, window_c) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.12,
                        hspace=0.32, wspace=0.28)

    create_figure_healthy_bars(
        solutions_healthy,
        window=window_h,
        fig=fig,
        axes=(axes[0, 0], axes[0, 1]),
        show_legend=False,
        panel_letters=("A", "B"),
        apply_tight_layout=False,
    )
    ckd_names = [
        "CKD No Furosemide",
        "CKD With Furosemide (No Interaction)",
        "CKD With Furosemide",
    ]
    ckd_colors = {
        ckd_names[0]: "#4d4d4d",
        ckd_names[1]: "#9a9a9a",
        ckd_names[2]: "#1a1a1a",
    }
    create_ckd_bars(
        solutions=solutions_ckd,
        idx_map=IDX,
        window=window_c,
        scenario_names=ckd_names,
        colors=ckd_colors,
        toxins=TOXINS,
        toxin_states=TOXIN_STATES,
        ae_states=AE_STATES,
        lit_conc=LIT_CONC_CKD,
        lit_clear=LIT_CLEAR_CKD,
        clearance_fn=_clearance_auc_window,
        fig=fig,
        axes=(axes[1, 0], axes[1, 1]),
        show_legend=False,
        panel_letters=("C", "D"),
        apply_tight_layout=False,
    )

    for ax, tag in ((axes[0, 0], "Healthy"), (axes[1, 0], "CKD-5")):
        ax.annotate(
            tag,
            xy=(-0.22, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=15,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=2.3,
        handlelength=2.4,
    )
    return fig


def main() -> None:
    _fig_t_h, _fig_b_h, sol_h = run_healthy_and_plot(
        window=(0.0, 12.0),
        t_ss=1500.0,
        t_sim=12.0,
        dt_ss=0.5,
        dt_sim=0.02,
    )
    plt.close(_fig_t_h)
    plt.close(_fig_b_h)

    _fig_t_c, _fig_b_c, sol_c = run_ckd_and_plot(
        window=(0.0, 12.0),
        t_pre_h=2100.0,
        t_main_h=12.0,
        dt_pre=0.5,
        dt_main=0.02,
    )
    plt.close(_fig_t_c)
    plt.close(_fig_b_c)

    fig = build_combined_bars_figure(
        sol_h, sol_c, window_h=(0.0, 12.0), window_c=(0.0, 12.0),
    )
    stem = os.path.join(_RESULTS, "bars_healthy_ckd_combined")
    for ext in ("png", "pdf"):
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved {p}")
        pub = os.path.join(_PUB_FIGURES, f"bars_healthy_ckd_combined.{ext}")
        fig.savefig(pub, dpi=300, bbox_inches="tight")
        print(f"  Saved {pub}")
    plt.close(fig)


if __name__ == "__main__":
    main()
