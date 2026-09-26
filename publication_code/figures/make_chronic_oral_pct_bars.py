# -*- coding: utf-8 -*-
"""Three-panel %change bars for CKD-5 chronic oral regimens.

A IS · B p-CS · C HA
Groups: TID 120 / QD 240 / TID 240 mg/day
Bars: Total, Bound, Unbound (same %Δ when fu_p is constant)

Times New Roman, same spine/legend style as make_chronic_oral_five_panel.py.
Panel C annotation “No clinical HA data” is larger than A/B body text.

Writes results/chronic_oral_pct_bars.{png,pdf}
      publication_code/Figures/chronic_oral_pct_bars.{png,pdf}
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib import font_manager
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

import compare_dosing_regimens_grey_HA_ADDED as cmp  # noqa: E402
import dose_response_tid_vs_qd as gen  # noqa: E402


def _set_times_new_roman() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next(
        (
            name
            for name in (
                "Times New Roman",
                "TimesNewRoman",
                "Nimbus Roman",
                "Liberation Serif",
                "Times",
            )
            if name in available
        ),
        "serif",
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                chosen,
                "Times New Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _save_fig(fig, path: str, dpi: int = 300) -> None:
    """Crop the PDF/PNG to the ink (axes + labels + legend), small pad only."""
    fig.set_facecolor("white")
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


XTICKS = ("TID 120 mg/day", "QD 240 mg/day", "TID 240 mg/day")
BAR_COLORS = ("0.12", "0.38", "0.62")  # Total, Bound, Unbound
BAR_LABELS = ("Total", "Bound", "Unbound")


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11, pad=3)


def _draw_panel(ax, pcts, letter, clin=None, ha_note=False, note_size=16):
    x = np.arange(len(pcts), dtype=float)
    w = 0.22
    if clin is not None:
        mean, lo, hi = clin
        ax.axhspan(lo, hi, color="0.90", zorder=0)
        ax.axhline(mean, color="0.15", ls="--", lw=1.4, zorder=1)

    for i, pct in enumerate(pcts):
        for k, col in enumerate(BAR_COLORS):
            xpos = x[i] + (k - 1) * w
            bar = ax.bar(xpos, pct, w * 0.92, color=col, zorder=2)
            if k != 1:
                continue
            fs = 12 if ha_note else 10
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                max(pct + 1.5, 4.0),
                f"{pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=fs,
                fontfamily="serif",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS, fontsize=11)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 65)
    ax.set_ylabel("% change vs\nno-drug baseline", fontsize=11, labelpad=8)
    _style(ax)
    ax.text(
        0.02,
        0.97,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        fontfamily="serif",
    )
    if ha_note:
        ax.text(
            0.50,
            0.72,
            "No clinical HA data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=note_size,
            fontweight="normal",
            fontfamily="serif",
            color="0.15",
        )


def plot_pct_bars(rows: list[dict]) -> plt.Figure:
    _set_times_new_roman()
    order = {n: i for i, (n, *_rest) in enumerate(cmp.REGIMENS)}
    rows = sorted(rows, key=lambda r: order[r["name"]])
    is_pct = [r["pct_IS"] for r in rows]
    pcs_pct = [r["pct_pCS"] for r in rows]
    ha_pct = [r["pct_HA"] for r in rows]

    fig = plt.figure(figsize=(14.2, 5.0))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        wspace=0.38,
        left=0.06,
        right=0.995,
        top=0.96,
        bottom=0.16,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    _draw_panel(
        axA,
        is_pct,
        "A",
        clin=(gen.CLIN_IS_PCT, gen.CLIN_IS_LO, gen.CLIN_IS_HI),
    )
    _draw_panel(
        axB,
        pcs_pct,
        "B",
        clin=(gen.CLIN_pCS_PCT, gen.CLIN_pCS_LO, gen.CLIN_pCS_HI),
    )
    _draw_panel(axC, ha_pct, "C", clin=None, ha_note=True, note_size=16)

    legend_handles = [
        Patch(facecolor=c, edgecolor="none", label=lab)
        for c, lab in zip(BAR_COLORS, BAR_LABELS)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="0.15",
            ls="--",
            lw=1.4,
            label="Clinical average (Costes-Albrespic 2025)",
        )
    )
    legend_handles.append(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="0.90",
            edgecolor="none",
            label="95% confidence interval",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.3,
        columnspacing=1.4,
        handlelength=2.2,
        prop={"family": "serif", "size": 11},
    )
    return fig


def _write_results_table(rows: list[dict]) -> None:
    order = {n: i for i, (n, *_rest) in enumerate(cmp.REGIMENS)}
    rows = sorted(rows, key=lambda r: order[r["name"]])
    lines = [
        "CKD-5 chronic oral % change vs no-drug baseline",
        f"IS_base (all arms) = {rows[0]['IS_base']:.3f} mg/L  (must be CKD-5, not ~0.54)",
        f"Clinical IS  {gen.CLIN_IS_PCT:+.1f}%  (95% CI {gen.CLIN_IS_LO:.1f} to {gen.CLIN_IS_HI:.1f})",
        f"Clinical pCS {gen.CLIN_pCS_PCT:+.1f}%  (95% CI {gen.CLIN_pCS_LO:.1f} to {gen.CLIN_pCS_HI:.1f})",
        "Clinical HA  none",
        "",
        f"{'Regimen':<16} {'file':<52} {'IS %':>8} {'pCS %':>8} {'HA %':>8}",
        "-" * 92,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:<16} {r['fname']:<52} "
            f"{r['pct_IS']:>+7.1f}% {r['pct_pCS']:>+7.1f}% {r['pct_HA']:>+7.1f}%"
        )
    text = "\n".join(lines) + "\n"
    print("\n" + text, flush=True)
    for folder in (_RESULTS, _PUB_FIG):
        p = os.path.join(folder, "chronic_oral_pct_bars.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  Saved {p}", flush=True)


def main() -> None:
    rows = []
    for name, dose, n, *_rest in cmp.REGIMENS:
        rows.append(cmp.load_regimen(name, dose, n))
    cmp._assert_same_phenotype(rows)
    _write_results_table(rows)
    fig = plot_pct_bars(rows)
    stem = "chronic_oral_pct_bars"
    for folder in (_RESULTS, _PUB_FIG):
        for ext in ("png", "pdf"):
            p = os.path.join(folder, f"{stem}.{ext}")
            _save_fig(fig, p)
            print(f"  Saved {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print("\n[MISSING CACHE]", exc, file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, ValueError) as exc:
        print("\n[REFUSED]", exc, file=sys.stderr)
        sys.exit(1)
