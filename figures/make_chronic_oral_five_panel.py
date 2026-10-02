# -*- coding: utf-8 -*-
"""Five-panel chronic oral CKD-5 figure (paper layout).

A–C: 30-day IS / HA / pCS **total plasma** (ODE blood / BP)
D:   furosemide total plasma, final 3-day window
E:   furosemide total plasma, final 24 h cycle

Traces: TID 120 mg/day (40×3), QD 240 mg/day, TID 240 mg/day (80×3),
        no-drug CKD-5 baseline.

Caches must share the post-pKa pCS baseline (~13.7 mg/L blood = 24.99 plasma).
Furosemide leftover is carried across midnight (not zeroed).
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

import compound_parameters as CP  # noqa: E402
import dose_response_tid_vs_qd as gen  # noqa: E402
import compare_dosing_regimens_grey_HA_ADDED as cmp  # noqa: E402

K.VERBOSE = False

# Total plasma = ODE blood / BP (same values as Table 1 / compounds.py).
BP = {
    "IS_all": float(CP.TOX1_BASE.get("BP", 0.70)),
    "HA_all": float(CP.TOX2_BASE.get("BP", 0.84)),
    "pCS_all": float(CP.TOX3_BASE.get("BP", 0.55)),
    "drg_all": float(CP.DRUG_BASE.get("BP", 0.82)),
}

# Post-pKa CKD-5 no-drug blood SS. Old pKa=6.9 caches sit near 17.9.
PCS_BLOOD_OK = (12.0, 15.5)


JOBS = (
    ("TID 120 mg/day", 40.0, 3, [0.0, 8.0, 16.0], "--", "0.35", 1.3),
    ("QD 240 mg/day", 240.0, 1, [0.0], ":", "0.15", 1.3),
    ("TID 240 mg/day", 80.0, 3, [0.0, 8.0, 16.0], "-", "0.0", 1.8),
    ("No-drug CKD-5 baseline", 0.0, 0, [], "-", "0.45", 1.2),
)


CKD_IS_MIN = 8.0
CKD_IS_MAX = 80.0
HEALTHY_IS_MAX = 2.0


def _bp_table() -> dict:
    return dict(BP)


def _to_plasma(key: str, y, bp=None) -> np.ndarray:
    tab = bp or BP
    return np.asarray(y, dtype=float) / tab[key]


def _run_to_plasma(data: dict, bp: dict) -> dict:
    """Copy a run dict with concentration arrays in total plasma."""
    out = dict(data)
    for key in ("IS_all", "HA_all", "pCS_all", "drg_all"):
        out[key] = _to_plasma(key, data[key], bp)
    for bkey, akey in (("IS_base", "IS_all"), ("HA_base", "HA_all"), ("pCS_base", "pCS_all")):
        out[bkey] = np.array([float(np.asarray(data[bkey]).flat[0]) / bp[akey]])
    out["_plasma"] = True
    return out


def _check_baselines_consistent(runs) -> None:
    """All arms must share IS/HA/pCS t=0 (1%). Prints the line the reviewer asked for."""
    keys = ("IS_base", "HA_base", "pCS_base")
    vals = {k: [float(np.asarray(d[k]).flat[0]) for _l, d, *_ in runs] for k in keys}
    ok = True
    for k, xs in vals.items():
        mid = float(np.median(xs))
        for lab, x in zip((r[0] for r in runs), xs):
            if abs(x - mid) / max(abs(mid), 1e-9) > 0.01:
                ok = False
                print(f"  Baselines mismatch: {lab} {k}={x:.3f} vs group {mid:.3f}", flush=True)
    unit = "plasma" if runs[0][1].get("_plasma") else "blood"
    msg = (
        f"  Baselines consistent: IS={vals['IS_base'][0]:.2f}  "
        f"HA={vals['HA_base'][0]:.2f}  pCS={vals['pCS_base'][0]:.2f} mg/L {unit}"
    )
    if not ok:
        raise RuntimeError("Baselines mismatch (stale pKa cache mixed with new).")
    print(msg, flush=True)


def _check_day_boundaries(label: str, data: dict) -> int:
    """Count midnight collapses of furosemide. Returns n_resets."""
    t = np.asarray(data["t_all"], dtype=float)
    y = np.asarray(data["drg_all"], dtype=float)
    n_reset = 0
    t_end = float(t[-1]) if t.size else 0.0
    last_day = int(t_end // 24.0)
    for day in range(1, last_day + 1):
        t_mid = day * 24.0
        before = y[(t < t_mid) & (t >= t_mid - 0.75)]
        after = y[(t >= t_mid) & (t < t_mid + 0.75)]
        if before.size == 0 or after.size == 0:
            continue
        yb, ya = float(before[-1]), float(after[0])
        if yb > 0.4 and ya < 0.25 * yb:
            n_reset += 1
            if n_reset <= 3 or day >= last_day - 2:
                print(
                    f"    RESET {label}  day {day}: {yb:.3f} -> {ya:.3f} mg/L",
                    flush=True,
                )
    print(f"  {label:<28} day-boundary resets: {n_reset}", flush=True)
    return n_reset


def _is_base_of(d) -> float:
    if "IS_base" in getattr(d, "files", ()):
        return float(np.asarray(d["IS_base"]).flat[0])
    if isinstance(d, dict) and "IS_base" in d:
        return float(np.asarray(d["IS_base"]).flat[0])
    is_all = np.asarray(d["IS_all"] if not isinstance(d, dict) else d["IS_all"])
    return float(is_all[0])


def _named_base(d, blood_key: str, arr_key: str) -> float:
    files = getattr(d, "files", ())
    if blood_key in files:
        return float(np.asarray(d[blood_key]).flat[0])
    if isinstance(d, dict) and blood_key in d:
        return float(np.asarray(d[blood_key]).flat[0])
    return float(np.asarray(d[arr_key])[0])


def _refuse_if_not_ckd5(path: str, d, label: str) -> float:
    fname = os.path.basename(path) if path else label
    lower = fname.lower()
    if "healthy" in lower:
        raise RuntimeError(
            f"{label}: refused {fname} — filename is healthy, not CKD-5."
        )
    is_base = _is_base_of(d)
    if is_base <= HEALTHY_IS_MAX:
        raise RuntimeError(
            f"{label}: IS_base={is_base:.3f} mg/L is HEALTHY/acute (~0.54), "
            f"not CKD-5 (~20). File={fname}."
        )
    if not (CKD_IS_MIN <= is_base <= CKD_IS_MAX):
        raise RuntimeError(
            f"{label}: IS_base={is_base:.3f} mg/L is outside CKD-5 "
            f"({CKD_IS_MIN:.0f}–{CKD_IS_MAX:.0f}). File={fname}."
        )
    pcs = _named_base(d, "pCS_base", "pCS_all")
    lo, hi = PCS_BLOOD_OK
    pcs_arr0 = float(np.asarray(d["pCS_all"]).reshape(-1)[0])
    if not (lo <= pcs <= hi) or not (lo <= pcs_arr0 <= hi):
        raise RuntimeError(
            f"{label}: pCS_base={pcs:.3f} pCS_all[0]={pcs_arr0:.3f} mg/L blood "
            f"is not the pKa=−2 SS ({lo:.1f}–{hi:.1f}; expected ~13.7). "
            f"File={fname}. Delete the old pKa=6.9 cache and re-run."
        )
    mode = None
    if hasattr(d, "files") and "MODE_CODE" in d.files:
        mode = float(np.asarray(d["MODE_CODE"]).flat[0])
    if mode is not None and abs(mode - 0.0) < 1e-9:
        raise RuntimeError(f"{label}: MODE_CODE=0 (healthy) in {fname}")
    if mode is not None and abs(mode - 5.0) > 1e-9:
        raise RuntimeError(f"{label}: MODE_CODE={mode} in {fname}, expected CKD-5")
    print(
        f"  [CKD-5] {label}: IS_base={is_base:.3f}  pCS_base={pcs:.3f} mg/L blood"
        f"  file={fname}",
        flush=True,
    )
    return is_base


def _arrays_from_cache(path: str, label: str) -> dict:
    d = np.load(path, allow_pickle=False)
    need = ("t_all", "IS_all", "pCS_all", "drg_all")
    missing = [k for k in need if k not in d.files]
    if missing:
        raise FileNotFoundError(f"{os.path.basename(path)} missing {missing}")
    is_base = _refuse_if_not_ckd5(path, d, label)
    t = np.asarray(d["t_all"])
    out = {k: np.asarray(d[k]) for k in need}
    if "HA_all" in d.files:
        out["HA_all"] = np.asarray(d["HA_all"])
        ha0 = _named_base(d, "HA_base", "HA_all")
    else:
        ha0 = float(CP.TOX2_BASE.get("BP", 0.84)) * 142.86
        if "HA_base" in getattr(d, "files", ()):
            ha0 = float(np.asarray(d["HA_base"]).flat[0])
        out["HA_all"] = np.full_like(t, ha0, dtype=float)
        print(f"  [warn] {os.path.basename(path)} has no HA_all — "
              f"HA panel uses constant {ha0:.2f} mg/L blood", flush=True)
    out["IS_base"] = np.array([is_base])
    out["HA_base"] = np.array([ha0])
    out["pCS_base"] = np.array([_named_base(d, "pCS_base", "pCS_all")])
    return out


def _nodrug_from_sibling() -> dict | None:
    try:
        path = cmp._find_cache(40.0, 3)
    except FileNotFoundError:
        return None
    d = np.load(path, allow_pickle=False)
    _refuse_if_not_ckd5(path, d, "No-drug CKD-5 baseline")
    t = np.asarray(d["t_all"])
    is0 = _is_base_of(d)
    ha0 = _named_base(d, "HA_base", "HA_all")
    pcs0 = _named_base(d, "pCS_base", "pCS_all")
    print(f"  [CACHE] no-drug CKD-5 SS from {os.path.basename(path)}", flush=True)
    return dict(
        t_all=t,
        IS_all=np.full_like(t, is0, dtype=float),
        HA_all=np.full_like(t, ha0, dtype=float),
        pCS_all=np.full_like(t, pcs0, dtype=float),
        drg_all=np.zeros_like(t, dtype=float),
        IS_base=np.array([is0]),
        HA_base=np.array([ha0]),
        pCS_base=np.array([pcs0]),
    )


def _cache_file(dose: float, n_doses: int) -> str:
    ki = f"{gen.Ki:.4g}".replace(".", "p")
    dose_str = f"{dose:.4g}"
    return os.path.join(
        _RESULTS,
        f"chronic_oral_cache_Ki{ki}_dose{dose_str}mg_x{n_doses}pd_"
        f"{gen.N_DAYS}days_{gen.MODE}_{gen.RUN_DATE}.npz",
    )


def _load_or_run(label, ic_ss, live_pk, dose, n_doses, times):
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

    def _restore():
        for k, v in saved.items():
            setattr(gen, k, v)

    # Prefer a globbed cache only if it passes pKa / pCS / CKD guards.
    try:
        found = cmp._find_cache(float(dose), int(n_doses))
        out = _arrays_from_cache(found, label)
        _restore()
        return out
    except FileNotFoundError:
        pass
    except RuntimeError as exc:
        print(f"  [SKIP STALE] {exc}", flush=True)

    if int(n_doses) == 0:
        if ic_ss and "C_tox1_cen" in ic_ss:
            t = np.linspace(0.0, gen.N_DAYS * 24.0, 2001)
            _restore()
            return dict(
                t_all=t,
                IS_all=np.full_like(t, float(ic_ss["C_tox1_cen"])),
                HA_all=np.full_like(t, float(ic_ss["C_tox2_cen"])),
                pCS_all=np.full_like(t, float(ic_ss["C_tox3_cen"])),
                drg_all=np.zeros_like(t),
                IS_base=np.array([float(ic_ss["C_tox1_cen"])]),
                HA_base=np.array([float(ic_ss["C_tox2_cen"])]),
                pCS_base=np.array([float(ic_ss["C_tox3_cen"])]),
            )
        nodrug = _nodrug_from_sibling()
        if nodrug is not None:
            _restore()
            return nodrug

    path = _cache_file(dose, n_doses)
    if live_pk and gen._cache_is_valid(path, live_pk):
        out = _arrays_from_cache(path, label)
        _restore()
        return out

    print(f"  simulating {n_doses}×{dose:g} mg ...", flush=True)
    t0 = time.time()
    oral = None if n_doses == 0 else gen.build_oral_profile(dose)
    t, IS, HA, pCS, drg = gen._run_simulation(ic_ss, live_pk, dose, n_doses, times, oral)
    print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)
    fp = gen._build_fingerprint(live_pk) if live_pk else {}
    save = dict(
        t_all=t, IS_all=IS, HA_all=HA, pCS_all=pCS, drg_all=drg,
        IS_base=np.array([ic_ss.get("C_tox1_cen", float(IS[0]))]),
        pCS_base=np.array([ic_ss.get("C_tox3_cen", float(pCS[0]))]),
        HA_base=np.array([ic_ss.get("C_tox2_cen", float(HA[0]))]),
    )
    save.update({k: np.array([v]) for k, v in fp.items()})
    np.savez(path, **save)
    print(f"  [CACHE] {path}", flush=True)
    _restore()
    return dict(t_all=t, IS_all=IS, HA_all=HA, pCS_all=pCS, drg_all=drg,
                IS_base=save["IS_base"], HA_base=save["HA_base"],
                pCS_base=save["pCS_base"])


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=7, pad=2)


def _panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=8, fontweight="bold", pad=3, fontfamily="serif")


def plot(runs) -> plt.Figure:
    _set_times_new_roman()
    # Journal text width (~6.27 in). Fonts 7–8 pt so they survive column shrink.
    fig = plt.figure(figsize=(6.27, 7.35))
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        hspace=0.42,
        wspace=0.85,
        left=0.10,
        right=0.88,
        top=0.96,
        bottom=0.12,
    )
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])
    axC = fig.add_subplot(gs[0, 4:6])
    axD = fig.add_subplot(gs[1, 0:3])
    axE = fig.add_subplot(gs[1, 3:6])

    nodrug = next(data for lab, data, *_ in runs if "No-drug" in lab)
    already = bool(nodrug.get("_plasma"))
    bases_pl = {
        "IS_all": float(nodrug["IS_base"].flat[0]) if already
        else float(nodrug["IS_base"].flat[0]) / BP["IS_all"],
        "pCS_all": float(nodrug["pCS_base"].flat[0]) if already
        else float(nodrug["pCS_base"].flat[0]) / BP["pCS_all"],
        "HA_all": float(nodrug["HA_base"].flat[0]) if already
        else float(nodrug["HA_base"].flat[0]) / BP["HA_all"],
    }

    top = (
        (axA, "IS_all", "IS plasma\n(mg/L)", "A) IS"),
        (axB, "HA_all", "HA plasma\n(mg/L)", "B) HA"),
        (axC, "pCS_all", "pCS plasma\n(mg/L)", "C) pCS"),
    )
    for ax, key, ylab, letter in top:
        c0 = bases_pl[key]
        ys = []
        for label, data, ls, c, lw in runs:
            t_d = np.asarray(data["t_all"]) / 24.0
            ypl = np.asarray(data[key], dtype=float)
            if not data.get("_plasma"):
                ypl = _to_plasma(key, ypl)
            ys.append(ypl)
            ax.plot(t_d, ypl, ls=ls, color=c, lw=lw, label=label)
        ax.set_xlabel("Time (days)", fontsize=7)
        ax.set_ylabel(ylab, fontsize=7, labelpad=4)
        ax.set_xlim(0, 30)
        ymin = min(float(np.min(y)) for y in ys)
        ymax = max(float(np.max(y)) for y in ys)
        pad = max(0.04 * (ymax - ymin), 0.15)
        ax.set_ylim(ymin - pad, ymax + pad)
        _style(ax)
        ax.spines["right"].set_visible(True)
        ax2 = ax.twinx()
        y0, y1 = ax.get_ylim()
        ax2.set_ylim((y0 - c0) / c0 * 100.0, (y1 - c0) / c0 * 100.0)
        ax2.set_ylabel("% vs baseline", fontsize=7, labelpad=3)
        ax2.tick_params(labelsize=7)
        ax2.spines["top"].set_visible(False)
        _panel_title(ax, letter)

    t_lo, t_hi = 27.0 * 24.0, 30.0 * 24.0
    for label, data, ls, c, lw in runs:
        t = np.asarray(data["t_all"])
        m = (t >= t_lo) & (t <= t_hi)
        yf = np.asarray(data["drg_all"], dtype=float)
        if not data.get("_plasma"):
            yf = _to_plasma("drg_all", yf)
        axD.plot(t[m] / 24.0, yf[m],
                 ls=ls, color=c, lw=lw, label=label)
    for day in (27.0, 28.0, 29.0):
        for h in (0.0, 8.0, 16.0):
            axD.axvline(day + h / 24.0, color="0.82", lw=0.5, zorder=0)
    axD.set_xlabel("Time (days), final 3-day window", fontsize=7)
    axD.set_ylabel("Furosemide plasma\n(mg/L)", fontsize=7, labelpad=4)
    axD.set_xlim(27, 30)
    _style(axD)
    axD.spines["right"].set_visible(False)
    _panel_title(axD, "D)")

    for label, data, ls, c, lw in runs:
        t = np.asarray(data["t_all"])
        t0 = 29.0 * 24.0
        m = (t >= t0) & (t <= t0 + 24.0)
        yf = np.asarray(data["drg_all"], dtype=float)
        if not data.get("_plasma"):
            yf = _to_plasma("drg_all", yf)
        axE.plot(t[m] - t0, yf[m],
                 ls=ls, color=c, lw=lw, label=label)
    for h in (0.0, 8.0, 16.0):
        axE.axvline(h, color="0.82", lw=0.5, zorder=0)
    axE.set_xlabel("Time within final 24-h dosing cycle (h)", fontsize=7)
    axE.set_ylabel("Furosemide plasma\n(mg/L)", fontsize=7, labelpad=4)
    axE.set_xlim(0, 24)
    _style(axE)
    axE.spines["right"].set_visible(False)
    _panel_title(axE, "E)")

    handles, labels = axA.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.2,
        columnspacing=1.4,
        handlelength=2.4,
        prop={"family": "serif", "size": 7},
    )
    return fig


def _all_dose_caches_present() -> bool:
    for _label, dose, n, _times, _ls, _c, _lw in JOBS:
        if int(n) == 0:
            continue
        try:
            path = cmp._find_cache(float(dose), int(n))
            d = np.load(path, allow_pickle=False)
            _refuse_if_not_ckd5(path, d, _label)
        except (FileNotFoundError, RuntimeError):
            return False
    return True


def main() -> None:
    bp = _bp_table()
    print("Five-panel chronic oral figure (CKD-5, total plasma = blood/BP)", flush=True)
    print(
        f"BP  IS={bp['IS_all']:.2f}  HA={bp['HA_all']:.2f}  "
        f"pCS={bp['pCS_all']:.2f}  furo={bp['drg_all']:.2f}",
        flush=True,
    )
    if _all_dose_caches_present():
        print("  Valid pKa=−2 CKD-5 caches found — skip ODE warm-up.", flush=True)
        ic_ss, live_pk = {}, {}
    else:
        print("  Cache missing or stale — SS warm-up + 30-day arms.", flush=True)
        ic_ss = gen.warmup()
        live_pk = gen.extract_baseline_pk(ic_ss)
    runs = []
    for label, dose, n, times, ls, c, lw in JOBS:
        data = _load_or_run(label, ic_ss, live_pk, dose, n, times)
        runs.append((label, data, ls, c, lw))

    print("\n========== CHECKS (paste these three) ==========", flush=True)
    print(
        f"BP  IS={bp['IS_all']:.2f}  HA={bp['HA_all']:.2f}  "
        f"pCS={bp['pCS_all']:.2f}  furo={bp['drg_all']:.2f}",
        flush=True,
    )
    n_resets = sum(
        _check_day_boundaries(lab, data)
        for lab, data, *_ in runs
        if not lab.startswith("No-drug")
    )
    if n_resets:
        print(
            "\n  WARNING: furosemide resets at day boundaries — "
            "fix the chronic driver before trusting these results.\n",
            flush=True,
        )
    runs = [(lab, _run_to_plasma(data, bp), *rest) for lab, data, *rest in runs]
    _check_baselines_consistent(runs)
    nodrug = next(d for lab, d, *_ in runs if "No-drug" in lab)
    pcs_pl = float(np.asarray(nodrug["pCS_base"]).flat[0])
    is_pl = float(np.asarray(nodrug["IS_base"]).flat[0])
    ha_pl = float(np.asarray(nodrug["HA_base"]).flat[0])
    print(
        f"  Plasma t=0  IS={is_pl:.2f}  HA={ha_pl:.2f}  pCS={pcs_pl:.2f} mg/L "
        f"(Table 1: 28.16 / 142.86 / 24.99)",
        flush=True,
    )
    if not (22.0 <= pcs_pl <= 28.0):
        raise RuntimeError(
            f"pCS plasma baseline {pcs_pl:.2f} mg/L is not ~24.99 "
            "(old pKa cache still in the plot)."
        )
    print("================================================\n", flush=True)

    fig = plot(runs)
    stem = "chronic_oral_five_panel"
    for folder in (_RESULTS, _PUB_FIG):
        for ext in ("png", "pdf"):
            p = os.path.join(folder, f"{stem}.{ext}")
            _save_fig(fig, p)
            print(f"  Saved {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print("\n[REFUSED] This figure is CKD-5 chronic oral only.", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)
