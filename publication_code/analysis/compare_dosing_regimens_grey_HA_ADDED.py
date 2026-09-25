#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare CKD-5 chronic oral regimens only — never mix healthy / acute files.

Arms (same phenotype, 30-day chronic carry-over, Ki encoded in the filename):

  TID_120mg  =  40 mg × 3 / day   (120 mg/day)
  QD_240mg   = 240 mg × 1 / day   (240 mg/day)
  TID_240mg  =  80 mg × 3 / day   (240 mg/day)

Each arm is loaded from its own ``chronic_oral_cache_*_ckd5_*.npz``.
A file is rejected if it is missing ``_ckd5_``, if MODE_CODE is not CKD-5,
if IS_base looks healthy (~0.5 mg/L), or if the stored dose clock does not
match the regimen slot. All three IS_base values must agree.

This is the check for mixing a healthy/single-dose 240 mg QD cache
(IS_base ≈ 0.54 mg/L, drg_avg ≈ 1 mg/L) with CKD-5 TID chronic caches
(IS_base ≈ 20 mg/L).
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ROOT = (
    os.path.abspath(os.path.join(_PARENT, ".."))
    if os.path.basename(_PARENT) == "publication_code"
    else _PARENT
)
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIG, exist_ok=True)

# CKD-5 no-drug IS is tens of mg/L; healthy IS is ~0.4–0.7 mg/L.
CKD_IS_BASE_MIN = 8.0
CKD_IS_BASE_MAX = 80.0
N_DAYS = 30
TAU_H = 24.0

REGIMENS = (
    # name, dose_per_admin, n_doses, linestyle, grey, lw
    ("TID_120mg", 40.0, 3, "--", "0.45", 1.5),
    ("QD_240mg", 240.0, 1, ":", "0.15", 1.6),
    ("TID_240mg", 80.0, 3, "-", "0.0", 2.0),
)


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _f(cache, key, default=None):
    if key not in cache.files:
        return default
    return float(np.asarray(cache[key]).flat[0])


def _find_cache(dose: float, n_doses: int) -> str:
    """Newest CKD-5 chronic cache for this dose × frequency. No healthy files."""
    dose_str = f"{dose:.4g}"
    pattern = os.path.join(
        _RESULTS,
        f"chronic_oral_cache_*_dose{dose_str}mg_x{n_doses}pd_{N_DAYS}days_ckd5_*.npz",
    )
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No CKD-5 chronic cache for {dose:g} mg × {n_doses}/day.\n"
            f"  looked for: {pattern}\n"
            "  Do not substitute a healthy or single-dose NPZ."
        )
    return hits[-1]


def _last_day_avg(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t0 = (N_DAYS - 1) * TAU_H
    m = t >= t0
    if m.sum() < 3:
        raise ValueError("last-day window too short — not a 30-day chronic series")
    dt = float(t[m][-1] - t[m][0])
    if dt < 12.0:
        raise ValueError(f"last-day span {dt:.2f} h is not a full chronic cycle")
    return _trapz(y[m], t[m]) / dt, t[m], y[m]


def load_regimen(name: str, dose: float, n_doses: int) -> dict:
    path = _find_cache(dose, n_doses)
    cache = np.load(path, allow_pickle=False)
    fname = os.path.basename(path)
    if "_ckd5_" not in fname.lower():
        raise RuntimeError(f"{name}: refused {fname} (filename is not CKD-5 chronic)")

    mode = _f(cache, "MODE_CODE")
    if mode is not None and abs(mode - 5.0) > 1e-9:
        raise RuntimeError(
            f"{name}: MODE_CODE={mode} in {fname} — healthy/other phenotype, not CKD-5"
        )

    stored_dose = _f(cache, "DOSE_PER_ADMIN")
    stored_n = _f(cache, "DOSING_CODE")
    if stored_dose is not None and abs(stored_dose - dose) > 1e-6:
        raise RuntimeError(
            f"{name}: file dose {stored_dose} mg ≠ slot {dose} mg ({fname})"
        )
    if stored_n is not None and abs(stored_n - n_doses) > 1e-6:
        raise RuntimeError(
            f"{name}: file n_doses {stored_n} ≠ slot {n_doses} ({fname})"
        )

    n_days = _f(cache, "N_DAYS")
    if n_days is not None and abs(n_days - N_DAYS) > 1e-6:
        raise RuntimeError(f"{name}: N_DAYS={n_days} in {fname}, expected {N_DAYS}")

    carry = _f(cache, "CHRONIC_CARRYOVER")
    if carry is not None and abs(carry - 1.0) > 1e-9:
        raise RuntimeError(
            f"{name}: CHRONIC_CARRYOVER={carry} in {fname} — daily wipe / acute run"
        )

    if "HA_all" not in cache.files:
        raise RuntimeError(f"{name}: {fname} has no HA_all — not a chronic HA cache")

    IS_base = _f(cache, "IS_base")
    if IS_base is None:
        raise RuntimeError(f"{name}: missing IS_base in {fname}")
    if not (CKD_IS_BASE_MIN <= IS_base <= CKD_IS_BASE_MAX):
        raise RuntimeError(
            f"{name}: IS_base={IS_base:.3f} mg/L in {fname} is not CKD-5.\n"
            f"  Healthy / acute IS_base is ~0.4–0.7 mg/L. "
            f"CKD-5 chronic IS_base is ~{CKD_IS_BASE_MIN:.0f}–{CKD_IS_BASE_MAX:.0f} mg/L.\n"
            "  This is the mix-up: do not plot that file next to CKD-5 TID caches."
        )

    t = np.asarray(cache["t_all"], dtype=float)
    IS = np.asarray(cache["IS_all"], dtype=float)
    HA = np.asarray(cache["HA_all"], dtype=float)
    pCS = np.asarray(cache["pCS_all"], dtype=float)
    drg = np.asarray(cache["drg_all"], dtype=float)
    if t[-1] < (N_DAYS - 0.5) * TAU_H:
        raise RuntimeError(
            f"{name}: series ends at t={t[-1]:.1f} h — not a {N_DAYS}-day chronic run"
        )

    IS_avg, _, _ = _last_day_avg(t, IS)
    HA_avg, _, _ = _last_day_avg(t, HA)
    pCS_avg, _, _ = _last_day_avg(t, pCS)
    drg_avg, _, _ = _last_day_avg(t, drg)

    HA_base = _f(cache, "HA_base")
    pCS_base = _f(cache, "pCS_base")
    if HA_base is None or HA_base <= 0:
        HA_base = float(HA[0])
    if pCS_base is None or pCS_base <= 0:
        pCS_base = _f(cache, "pCS_base") or float(pCS[0])

    rec = dict(
        name=name,
        path=path,
        fname=fname,
        dose=dose,
        n_doses=n_doses,
        IS_base=IS_base,
        HA_base=float(HA_base),
        pCS_base=float(pCS_base),
        IS_avg=IS_avg,
        HA_avg=HA_avg,
        pCS_avg=pCS_avg,
        drg_avg=drg_avg,
        pct_IS=(IS_avg - IS_base) / IS_base * 100.0,
        pct_HA=(HA_avg - float(HA_base)) / max(float(HA_base), 1e-9) * 100.0,
        pct_pCS=(pCS_avg - float(pCS_base)) / max(float(pCS_base), 1e-9) * 100.0,
        t=t,
        IS=IS,
        HA=HA,
        pCS=pCS,
        drg=drg,
    )
    print(
        f"[OK] {name}: {dose}mg x{n_doses}/day, {N_DAYS} days, "
        f"IS_base={IS_base:.3f}  file={fname}"
    )
    return rec


def _assert_same_phenotype(rows: list[dict]) -> None:
    bases = [r["IS_base"] for r in rows]
    mid = float(np.median(bases))
    for r in rows:
        rel = abs(r["IS_base"] - mid) / max(mid, 1e-9)
        if rel > 0.15:
            raise RuntimeError(
                f"IS_base mismatch: {r['name']} has {r['IS_base']:.3f} vs group "
                f"{mid:.3f} ({r['fname']}). These are not the same phenotype. "
                "Do not combine healthy/acute NPZ with CKD-5 chronic NPZ."
            )


def plot_rows(rows: list[dict], styles: dict) -> None:
    fig = plt.figure(figsize=(11.0, 7.2))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.09, right=0.98, top=0.90, bottom=0.14)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    panels = (
        (axes[0], "IS", "IS plasma (mg/L)"),
        (axes[1], "HA", "HA plasma (mg/L)"),
        (axes[2], "pCS", "pCS plasma (mg/L)"),
        (axes[3], "drg", "Furosemide plasma (mg/L)"),
    )
    for ax, key, ylab in panels:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for r in rows:
            ls, c, lw = styles[r["name"]]
            ax.plot(r["t"] / 24.0, r[key], ls=ls, color=c, lw=lw, label=r["name"])
        ax.set_xlim(0, N_DAYS)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(ylab)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "CKD-5 chronic oral only  ·  TID 120 vs QD 240 vs TID 240 mg/day",
        fontsize=11,
    )
    stem = "dosing_comparison_timecourse"
    for folder in (_RESULTS, _PUB_FIG):
        for ext in ("png", "pdf"):
            p = os.path.join(folder, f"{stem}.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[saved] {p}")
    plt.close(fig)


def main() -> None:
    print("=" * 65)
    rows = []
    styles = {}
    for name, dose, n, ls, c, lw in REGIMENS:
        styles[name] = (ls, c, lw)
        rows.append(load_regimen(name, dose, n))
    _assert_same_phenotype(rows)
    print("=" * 65)
    print(f"{'Regimen':<14} {'IS %D':>8} {'HA %D':>8} {'pCS %D':>8} {'drg_avg':>9}  IS_base")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['name']:<14} {r['pct_IS']:>+7.1f}% {r['pct_HA']:>+7.1f}% "
            f"{r['pct_pCS']:>+7.1f}% {r['drg_avg']:9.3f}  {r['IS_base']:.3f}"
        )
    print("=" * 65)
    print("Plotting regimens:", [r["name"] for r in rows])
    plot_rows(rows, styles)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("\n[REFUSED] Comparison aborted — files are not the same study arm.", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)
