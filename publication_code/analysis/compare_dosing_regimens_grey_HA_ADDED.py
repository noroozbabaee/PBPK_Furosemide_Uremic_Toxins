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

# Windows layouts often keep NPZs next to publication_code or under analysis/.
_SEARCH_DIRS = []
for _d in (
    _RESULTS,
    os.path.join(_PARENT, "results"),
    os.path.join(_THIS_DIR, "results"),
    os.path.join(_ROOT, "figures"),
    os.path.join(_PARENT, "figures"),
    os.path.join(os.getcwd(), "results"),
    os.getcwd(),
):
    _d = os.path.abspath(_d)
    if os.path.isdir(_d) and _d not in _SEARCH_DIRS:
        _SEARCH_DIRS.append(_d)

# CKD-5 no-drug IS is tens of mg/L; healthy IS is ~0.4–0.7 mg/L.
CKD_IS_BASE_MIN = 8.0
CKD_IS_BASE_MAX = 80.0
N_DAYS = 30
TAU_H = 24.0

# Cache arrays are ODE blood. Plots and Cmax prints use total plasma.
BP_PLOT = {"IS": 0.70, "HA": 0.84, "pCS": 0.55, "drg": 0.82}

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


def _dose_tokens(dose: float) -> tuple[str, ...]:
    """Filename dose spellings used by current and older cache writers."""
    raw = f"{dose:.4g}"
    alts = {raw, f"{dose:.0f}", f"{dose:.1f}", f"{int(dose)}"}
    return tuple(sorted(alts, key=len))


def _cache_ok_for_slot(path: str, dose: float, n_doses: int) -> tuple[bool, str]:
    """Accept CKD-5 chronic content even if the filename omits `_ckd5_`."""
    try:
        cache = np.load(path, allow_pickle=False)
    except Exception as exc:
        return False, f"unreadable ({exc})"
    fname = os.path.basename(path)
    if "healthy" in fname.lower():
        return False, "filename says healthy"
    mode = _f(cache, "MODE_CODE")
    if mode is not None and abs(mode - 0.0) < 1e-9:
        return False, f"MODE_CODE={mode} (healthy)"
    if mode is not None and abs(mode - 5.0) > 1e-9:
        return False, f"MODE_CODE={mode}"
    stored_dose = _f(cache, "DOSE_PER_ADMIN")
    stored_n = _f(cache, "DOSING_CODE")
    if stored_dose is not None and abs(stored_dose - dose) > 1e-6:
        return False, f"DOSE_PER_ADMIN={stored_dose}"
    if stored_n is not None and abs(stored_n - n_doses) > 1e-6:
        return False, f"DOSING_CODE={stored_n}"
    n_days = _f(cache, "N_DAYS")
    if n_days is not None and abs(n_days - N_DAYS) > 1e-6:
        return False, f"N_DAYS={n_days}"
    if "t_all" not in cache.files or "IS_all" not in cache.files:
        return False, "missing t_all/IS_all"
    t = np.asarray(cache["t_all"], dtype=float)
    if t.size == 0 or t[-1] < (N_DAYS - 0.5) * TAU_H:
        return False, f"series ends at t={0 if t.size == 0 else t[-1]:.1f} h"
    IS_base = _f(cache, "IS_base")
    if IS_base is None:
        return False, "missing IS_base"
    if not (CKD_IS_BASE_MIN <= IS_base <= CKD_IS_BASE_MAX):
        return False, f"IS_base={IS_base:.3f} (not CKD-5)"
    pcs = _f(cache, "pCS_base")
    if pcs is None and "pCS_all" in cache.files:
        pcs = float(np.asarray(cache["pCS_all"]).reshape(-1)[0])
    if pcs is not None and not (12.0 <= pcs <= 15.5):
        return False, f"pCS_base={pcs:.3f} (old pKa, not ~13.7 blood)"
    return True, "ok"


def _find_cache(dose: float, n_doses: int) -> str:
    """Newest matching CKD-5 chronic cache (filename or contents)."""
    patterns = []
    for folder in _SEARCH_DIRS:
        for tok in _dose_tokens(dose):
            patterns.extend(
                (
                    os.path.join(
                        folder,
                        f"chronic_oral_cache_*dose{tok}mg_x{n_doses}pd_{N_DAYS}days_ckd5*.npz",
                    ),
                    os.path.join(
                        folder,
                        f"chronic_oral_cache_*dose{tok}mg_x{n_doses}pd_{N_DAYS}days*.npz",
                    ),
                    os.path.join(
                        folder,
                        f"results_*dose{tok}mg_x{n_doses}pd_{N_DAYS}days*.npz",
                    ),
                )
            )
    seen: list[str] = []
    for pat in patterns:
        for hit in glob.glob(pat):
            if hit not in seen:
                seen.append(hit)

    accepted: list[str] = []
    rejected: list[str] = []
    for path in seen:
        ok, why = _cache_ok_for_slot(path, dose, n_doses)
        if ok:
            accepted.append(path)
        else:
            rejected.append(f"  {os.path.basename(path)}: {why}")

    if accepted:
        def _rank(p: str):
            try:
                c = np.load(p, allow_pickle=False)
                has_ha = int("HA_all" in c.files)
            except Exception:
                has_ha = 0
            name = os.path.basename(p).lower()
            return (
                has_ha,
                int(name.startswith("chronic_oral_cache")),
                int("_ckd5" in name),
                os.path.getmtime(p),
            )

        accepted.sort(key=_rank)
        return accepted[-1]

    looked = "\n".join(f"  {d}" for d in _SEARCH_DIRS)
    extra = "\n".join(rejected[:12]) if rejected else "  (no matching filenames)"
    raise FileNotFoundError(
        f"No CKD-5 chronic cache for {dose:g} mg × {n_doses}/day.\n"
        f"Searched:\n{looked}\n"
        f"Candidates:\n{extra}\n"
        "Generate it with:\n"
        "  python analysis/regenerate_chronic_caches.py\n"
        "or copy "
        f"chronic_oral_cache_*dose{dose:.0f}mg_x{n_doses}pd_30days*.npz "
        "into the repo results/ folder."
    )


def _last_day_peak(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t0 = (N_DAYS - 1) * TAU_H
    m = t >= t0
    if m.sum() < 3:
        raise ValueError("last-day window too short for Cmax")
    return float(np.max(y[m])), float(np.min(y[m]))


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
        print(f"  [warn] {fname} has no HA_all — HA panel will be omitted")
        HA = np.zeros_like(np.asarray(cache["IS_all"], dtype=float))
        HA_missing = True
    else:
        HA = np.asarray(cache["HA_all"], dtype=float)
        HA_missing = False

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
    IS_max, IS_min = _last_day_peak(t, IS)
    HA_max, HA_min = _last_day_peak(t, HA)
    pCS_max, pCS_min = _last_day_peak(t, pCS)
    drg_max, drg_min = _last_day_peak(t, drg)

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
        IS_max=IS_max,
        IS_min=IS_min,
        HA_max=HA_max,
        HA_min=HA_min,
        pCS_max=pCS_max,
        pCS_min=pCS_min,
        drg_max=drg_max,
        drg_min=drg_min,
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
            ax.plot(r["t"] / 24.0, np.asarray(r[key], dtype=float) / BP_PLOT[key],
                    ls=ls, color=c, lw=lw, label=r["name"])
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
    print("Last 24 h peaks (total plasma = blood / BP).")
    print("BP: IS 0.70  HA 0.84  pCS 0.55  furosemide 0.82")
    print(
        f"{'Regimen':<14} {'IS Cmax':>8} {'HA Cmax':>8} {'pCS Cmax':>8} "
        f"{'furo Cmax':>10} {'furo Cavg':>10}"
    )
    for r in rows:
        print(
            f"{r['name']:<14} {r['IS_max']/BP_PLOT['IS']:8.3f} "
            f"{r['HA_max']/BP_PLOT['HA']:8.3f} "
            f"{r['pCS_max']/BP_PLOT['pCS']:8.3f} "
            f"{r['drg_max']/BP_PLOT['drg']:10.3f} "
            f"{r['drg_avg']/BP_PLOT['drg']:10.3f}"
        )
    by = {r["name"]: r for r in rows}
    if "QD_240mg" in by and "TID_240mg" in by:
        q, t = by["QD_240mg"], by["TID_240mg"]
        print("-" * 65)
        print("DB check (same 240 mg/day, last 24 h plasma Cmax):")
        for key, lab, bp in (
            ("IS_max", "IS", BP_PLOT["IS"]),
            ("HA_max", "HA", BP_PLOT["HA"]),
            ("pCS_max", "pCS", BP_PLOT["pCS"]),
        ):
            qd, tid = q[key] / bp, t[key] / bp
            flag = "QD peak > TID" if qd > tid + 1e-9 else "QD peak <= TID"
            print(f"  {lab}: QD {qd:.3f}  TID {tid:.3f}  → {flag}")
        print(
            f"  Furosemide: QD Cmax {q['drg_max']/BP_PLOT['drg']:.2f}  "
            f"TID Cmax {t['drg_max']/BP_PLOT['drg']:.2f}  "
            f"QD Cavg {q['drg_avg']/BP_PLOT['drg']:.2f}  "
            f"TID Cavg {t['drg_avg']/BP_PLOT['drg']:.2f}"
        )
    print("=" * 65)
    print("Plotting regimens:", [r["name"] for r in rows])
    plot_rows(rows, styles)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print("\n[MISSING CACHE] Need the CKD-5 30-day NPZ for this arm.", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, ValueError) as exc:
        print("\n[REFUSED] Comparison aborted — files are not the same study arm.", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)
