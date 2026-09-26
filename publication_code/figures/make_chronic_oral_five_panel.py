# -*- coding: utf-8 -*-
"""Five-panel chronic oral CKD-5 figure (paper layout).

A–C: 30-day IS / pCS / HA plasma
D:   furosemide, final 3-day window
E:   furosemide, final 24 h cycle

Traces: TID 120 mg/day (40×3), QD 240 mg/day, TID 240 mg/day (80×3),
        no-drug CKD-5 baseline.

Uses analysis/dose_response_tid_vs_qd.py for the ODE; writes
publication_code/Figures/chronic_oral_five_panel.pdf
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import numpy as np


def _set_times_new_roman() -> None:
    """Prefer Times New Roman (Windows); Liberation Serif on Linux."""
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ROOT = os.path.abspath(os.path.join(_PARENT, "..")) if os.path.basename(_PARENT) == "publication_code" else _PARENT
_ANALYSIS = os.path.join(_ROOT, "analysis")
_SRC = os.path.join(_ROOT, "src")
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIG, exist_ok=True)
for _p in (_SRC, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import constants as K  # noqa: E402
K.VERBOSE = False

import dose_response_tid_vs_qd as gen  # noqa: E402
import compare_dosing_regimens_grey_HA_ADDED as cmp  # noqa: E402

K.VERBOSE = False


JOBS = (
    # label, dose, n_doses, times, linestyle, color, lw
    ("TID 120 mg/day", 40.0, 3, [0.0, 8.0, 16.0], "--", "0.35", 1.3),
    ("QD 240 mg/day", 240.0, 1, [0.0], ":", "0.15", 1.3),
    ("TID 240 mg/day", 80.0, 3, [0.0, 8.0, 16.0], "-", "0.0", 1.8),
    ("No-drug CKD-5 baseline", 0.0, 0, [], "-", "0.45", 1.2),
)


def _arrays_from_cache(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    need = ("t_all", "IS_all", "HA_all", "pCS_all", "drg_all")
    missing = [k for k in need if k not in d.files]
    if missing:
        raise FileNotFoundError(f"{os.path.basename(path)} missing {missing}")
    print(f"  [CACHE] {os.path.basename(path)}", flush=True)
    return {k: d[k] for k in need}


def _nodrug_from_sibling() -> dict | None:
    """Flat no-drug traces from a CKD-5 40×3 cache if dose0 NPZ is absent."""
    try:
        path = cmp._find_cache(40.0, 3)
    except FileNotFoundError:
        return None
    d = np.load(path, allow_pickle=False)
    t = np.asarray(d["t_all"])
    is0 = float(np.asarray(d["IS_base"]).flat[0]) if "IS_base" in d.files else float(d["IS_all"][0])
    ha0 = float(np.asarray(d["HA_base"]).flat[0]) if "HA_base" in d.files else float(d["HA_all"][0])
    pcs0 = float(np.asarray(d["pCS_base"]).flat[0]) if "pCS_base" in d.files else float(d["pCS_all"][0])
    print(f"  [CACHE] no-drug from baseline of {os.path.basename(path)}", flush=True)
    return dict(
        t_all=t,
        IS_all=np.full_like(t, is0, dtype=float),
        HA_all=np.full_like(t, ha0, dtype=float),
        pCS_all=np.full_like(t, pcs0, dtype=float),
        drg_all=np.zeros_like(t, dtype=float),
    )


def _cache_file(dose: float, n_doses: int) -> str:
    ki = f"{gen.Ki:.4g}".replace(".", "p")
    dose_str = f"{dose:.4g}"
    return os.path.join(
        _RESULTS,
        f"chronic_oral_cache_Ki{ki}_dose{dose_str}mg_x{n_doses}pd_"
        f"{gen.N_DAYS}days_{gen.MODE}_{gen.RUN_DATE}.npz",
    )


def _load_or_run(ic_ss, live_pk, dose, n_doses, times):
    K.VERBOSE = False
    saved = dict(
        DOSE_PER_ADMIN=gen.DOSE_PER_ADMIN,
        N_DOSES_PER_DAY=gen.N_DOSES_PER_DAY,
        DOSE_INTERVAL_H=gen.DOSE_INTERVAL_H,
        DOSE_TIMES_H=list(gen.DOSE_TIMES_H),
    )
    gen.DOSE_PER_ADMIN = float(dose)
    gen.N_DOSES_PER_DAY = int(n_doses)
    gen.DOSE_TIMES_H = list(times)
    gen.DOSE_INTERVAL_H = gen.TAU_H if n_doses <= 1 else gen.TAU_H / float(n_doses)

    try:
        return _arrays_from_cache(cmp._find_cache(float(dose), int(n_doses)))
    except FileNotFoundError:
        pass

    if int(n_doses) == 0:
        nodrug = _nodrug_from_sibling()
        if nodrug is not None:
            for k, v in saved.items():
                setattr(gen, k, v)
            return nodrug

    path = _cache_file(dose, n_doses)
    if gen._cache_is_valid(path, live_pk):
        d = np.load(path, allow_pickle=False)
        for k, v in saved.items():
            setattr(gen, k, v)
        return {k: d[k] for k in ("t_all", "IS_all", "HA_all", "pCS_all", "drg_all")}

    print(f"  simulating {n_doses}×{dose:g} mg ...", flush=True)
    t0 = time.time()
    oral = None if n_doses == 0 else gen.build_oral_profile(dose)
    t, IS, HA, pCS, drg = gen._run_simulation(ic_ss, live_pk, dose, n_doses, times, oral)
    print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)
    fp = gen._build_fingerprint(live_pk)
    save = dict(t_all=t, IS_all=IS, HA_all=HA, pCS_all=pCS, drg_all=drg,
                IS_base=np.array([ic_ss.get("C_tox1_cen", 0.0)]),
                pCS_base=np.array([ic_ss.get("C_tox3_cen", 0.0)]),
                HA_base=np.array([ic_ss.get("C_tox2_cen", 0.0)]))
    save.update({k: np.array([v]) for k, v in fp.items()})
    np.savez(path, **save)
    print(f"  [CACHE] {path}", flush=True)
    for k, v in saved.items():
        setattr(gen, k, v)
    return dict(t_all=t, IS_all=IS, HA_all=HA, pCS_all=pCS, drg_all=drg)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, pad=3)


def plot(runs) -> plt.Figure:
    _set_times_new_roman()
    # Extra width + large wspace so y-labels of A/B/C and D/E do not collide.
    fig = plt.figure(figsize=(14.2, 8.0))
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        hspace=0.50,
        wspace=0.72,
        left=0.06,
        right=0.995,
        top=0.98,
        bottom=0.10,
    )
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])
    axC = fig.add_subplot(gs[0, 4:6])
    axD = fig.add_subplot(gs[1, 0:3])
    axE = fig.add_subplot(gs[1, 3:6])

    top = (
        (axA, "IS_all", "IS plasma\nconcentration (mg/L)", "A"),
        (axB, "pCS_all", "p-CS plasma\nconcentration (mg/L)", "B"),
        (axC, "HA_all", "HA plasma\nconcentration (mg/L)", "C"),
    )
    for ax, key, ylab, letter in top:
        for label, data, ls, c, lw in runs:
            t_d = np.asarray(data["t_all"]) / 24.0
            ax.plot(t_d, data[key], ls=ls, color=c, lw=lw, label=label)
        ax.set_xlabel("Time (days)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11, labelpad=8)
        ax.set_xlim(0, 30)
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

    t_lo, t_hi = 27.0 * 24.0, 30.0 * 24.0
    for label, data, ls, c, lw in runs:
        t = np.asarray(data["t_all"])
        m = (t >= t_lo) & (t <= t_hi)
        axD.plot(t[m] / 24.0, data["drg_all"][m], ls=ls, color=c, lw=lw, label=label)
    axD.set_xlabel("Time (days), final 3-day window", fontsize=11)
    axD.set_ylabel("Furosemide plasma\nconcentration (mg/L)", fontsize=11, labelpad=8)
    axD.set_xlim(27, 30)
    _style(axD)
    axD.text(
        0.02,
        0.97,
        "D",
        transform=axD.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        fontfamily="serif",
    )

    for label, data, ls, c, lw in runs:
        t = np.asarray(data["t_all"])
        t0 = 29.0 * 24.0
        m = (t >= t0) & (t <= t0 + 24.0)
        axE.plot(t[m] - t0, data["drg_all"][m], ls=ls, color=c, lw=lw, label=label)
    axE.set_xlabel("Time within final 24-h dosing cycle (h)", fontsize=11)
    axE.set_ylabel("Furosemide plasma\nconcentration (mg/L)", fontsize=11, labelpad=8)
    axE.set_xlim(0, 24)
    _style(axE)
    axE.text(
        0.02,
        0.97,
        "E",
        transform=axE.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        fontfamily="serif",
    )

    handles, labels = axA.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.4,
        columnspacing=1.8,
        handlelength=2.6,
        prop={"family": "serif", "size": 11},
    )
    return fig


def _all_dose_caches_present() -> bool:
    for _label, dose, n, _times, _ls, _c, _lw in JOBS:
        if int(n) == 0:
            continue
        try:
            cmp._find_cache(float(dose), int(n))
        except FileNotFoundError:
            return False
    return True


def main() -> None:
    print("Five-panel chronic oral figure (CKD-5)", flush=True)
    if _all_dose_caches_present():
        print("  Existing 40×3, 240×1, 80×3 caches found — skip ODE warm-up.", flush=True)
        ic_ss, live_pk = {}, {}
    else:
        print("  Cache missing — SS warm-up + 30-day arms.", flush=True)
        ic_ss = gen.warmup()
        live_pk = gen.extract_baseline_pk(ic_ss)
    runs = []
    for label, dose, n, times, ls, c, lw in JOBS:
        data = _load_or_run(ic_ss, live_pk, dose, n, times)
        runs.append((label, data, ls, c, lw))
    fig = plot(runs)
    stem = "chronic_oral_five_panel"
    for folder in (_RESULTS, _PUB_FIG):
        for ext in ("png", "pdf"):
            p = os.path.join(folder, f"{stem}.{ext}")
            _save_fig(fig, p)
            print(f"  Saved {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
