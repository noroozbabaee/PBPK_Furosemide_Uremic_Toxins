# -*- coding: utf-8 -*-
"""
heatmap_ki_vmax_engine.py
=========================
Single source of truth for the Ki × Vmax sensitivity grid.
Replaces run_ckd_simulation() / compute_heatmap() in heatmap_ki_vmax_fixed_v4.py.

This is NOT manuscript Fig 7 by default. Fig 7 is
figures/make_figure7_ki_vmax_heatmap.py (heatmap_ki_vmax_analytical.pdf).

What changed vs v4 (same IV physics, same outcome definition):
  1. ROW-WISE: the warm-up and the no-drug reference do not depend on Ki
     (no drug → no inhibition), so they run ONCE per Vmax row.
  2. ALL THREE TOXINS per run: IS, HA and pCS (v4 re-ran IS and pCS separately
     and had no HA).
  3. Vmax scale applied to IS, HA and pCS basolateral OAT1.
  4. STALE-CACHE: fingerprint of live CKD-5 compound dicts including
     pKa-derived TOX*_PASS / DRG_PASS.
  5. Global CKD scales are restored after the grid.

C_tox*_cen is total blood. BP cancels in %Δ. Do not label a single-dose
map ΔC_ss. Vmax axis = fraction of *healthy* baso OAT1 Vmax.

Scenarios:
  "single_iv"    : 120 mg IV / 2 h; %Δ of total conc. at t = SIM_HOURS
                   (aliases: "iv_single")
  "chronic_oral" : dose_response_tid_vs_qd._run_simulation, TID 80×3, 30 d;
                   last-cycle mean vs warm-up baseline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

OUTDIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(OUTDIR, exist_ok=True)
CACHE_FILE = os.path.join(OUTDIR, "heatmap_ki_vmax_engine_cache.pkl")

WARMUP_H = 1000.0
SIM_HOURS = 24.0
DT_WARMUP = 0.5
DT_SIM = 0.1
DOSE_MG = 120.0
INFUSION_H = 2.0

N_KI, N_VMAX = 15, 12
Ki_grid = np.logspace(np.log10(0.05), np.log10(5.0), N_KI)
Vmax_grid = np.linspace(0.01, 0.30, N_VMAX)

TOXINS = {
    "IS": ("tox1", "C_tox1_cen"),
    "HA": ("tox2", "C_tox2_cen"),
    "pCS": ("tox3", "C_tox3_cen"),
}
KI_KEYS = {
    "tox1": "Ki_furo_to_IS_mgL",
    "tox2": "Ki_furo_to_HA_mgL",
    "tox3": "Ki_furo_to_pCS_mgL",
}
_DOSING_KEYS = {"enabled", "IR", "t_start_h", "t_inf_h", "bolus_mg", "inhibit"}
SCENARIO = os.environ.get("HEATMAP_SCENARIO", "single_iv")


def _norm_scenario(scenario: str) -> str:
    if scenario in ("iv_single", "single_iv", "iv"):
        return "single_iv"
    return scenario


def _set_vmax_scale(scale: float) -> None:
    """CKD-5 basolateral OAT1 Vmax as a fraction of healthy, all toxins."""
    import compound_parameters as CP
    for tox in TOXINS:
        CP._CKD_SCALES["per_compound"][tox]["baso"] = float(scale)
    CP.CKD_ADJUSTMENTS = CP._create_ckd_adjustments_dynamic(CP._CKD_SCALES)


def _set_vmax(scale: float) -> None:
    _set_vmax_scale(scale)


def _force_ki(*args) -> None:
    """_force_ki(ki) or _force_ki(C, ki)."""
    if len(args) == 1:
        import compounds as C
        ki = args[0]
    else:
        C, ki = args[0], args[1]
    for attr, key in KI_KEYS.items():
        d = getattr(C, attr, None)
        if isinstance(d, dict):
            d[key] = float(ki)


def _drug_off(C) -> None:
    C.drug.update({
        "enabled": False, "IR": 0.0, "t_start_h": 1e9,
        "t_inf_h": 0.0, "bolus_mg": 0.0, "inhibit": False,
    })


def _drug_single_iv(C) -> None:
    C.drug.update({
        "enabled": True, "IR": DOSE_MG / INFUSION_H,
        "t_inf_h": INFUSION_H, "t_start_h": 0.05,
        "bolus_mg": 0.0, "inhibit": True,
    })
    for attr, _ in TOXINS.values():
        d = getattr(C, attr, None)
        if isinstance(d, dict):
            d["enabled"] = True


def _jsonable(v):
    if isinstance(v, (int, float, np.floating, np.integer)):
        return round(float(v), 10)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in list(v)]
    return str(v)


def parameter_fingerprint(scenario: str) -> str:
    """Hash of CKD-5 compound parameters except the ones the grid varies."""
    import compounds as C
    C.initialize_compounds("ckd5")
    parts = {}
    for attr in ("tox1", "tox2", "tox3", "drug"):
        d = getattr(C, attr, {}) or {}
        parts[attr] = _jsonable({
            k: v for k, v in d.items()
            if k not in _DOSING_KEYS
            and not str(k).startswith("Ki_furo")
            and str(k) != "Vmax_baso_uptake"
        })
    for attr in ("TOX1_PASS", "TOX2_PASS", "TOX3_PASS", "DRG_PASS"):
        parts[attr] = _jsonable(getattr(C, attr, None))
    parts["settings"] = dict(
        scenario=_norm_scenario(scenario),
        warmup=WARMUP_H, sim=SIM_HOURS,
        dose=DOSE_MG, inf=INFUSION_H,
    )
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def _mean_last_cycle(t, y, n_days=30, tau=24.0) -> float:
    mask = t >= (n_days - 1) * tau
    if not np.any(mask):
        return float("nan")
    tt, yy = t[mask], y[mask]
    dt = float(tt[-1] - tt[0])
    if dt <= 0:
        return float(yy[-1])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(yy, tt) / dt)
    return float(np.trapz(yy, tt) / dt)


def _warmup_ic(scale: float) -> dict:
    """Public helper for heatmap_validate_corners.py (one Vmax, no drug)."""
    from simulation import run_sim
    from state_vector import IDX
    import compounds as C
    import constants as K
    K.VERBOSE = False
    _set_vmax_scale(scale)
    C.initialize_compounds("ckd5")
    _drug_off(C)
    sol_ss = run_sim(t_end_h=WARMUP_H, mode="ckd5", dt=DT_WARMUP)
    n = max(10, len(sol_ss.t) // 10)
    for tox, (_, key) in TOXINS.items():
        s = sol_ss.y[IDX[key], -n:]
        drift = abs(s[-1] - s[0]) / max(abs(s[-1]), 1e-12)
        if drift > 0.01:
            print(f"    [warn] scale={scale:.3f}: {tox} drift {drift * 100:.1f}% "
                  f"in last 10% of warm-up — increase WARMUP_H")
    ic = {n: float(sol_ss.y[i, -1]) for n, i in IDX.items()}
    for k in list(ic):
        if k.startswith("Ae_"):
            ic[k] = 0.0
    return ic


def _run_chronic_oral(ic_ss: dict, ki: float, vmax_scale: float) -> Dict[str, float]:
    """TID 80×3, 30 days. Last-cycle mean vs warm-up baseline (blood)."""
    import compounds as C
    import dose_response_tid_vs_qd as gen
    saved_ki = gen.Ki
    saved_restore = gen._restore_ki
    gen.Ki = float(ki)

    def _restore_and_force():
        saved_restore()
        _force_ki(C, ki)

    gen._restore_ki = _restore_and_force
    try:
        _set_vmax_scale(vmax_scale)
        C.initialize_compounds("ckd5")
        _force_ki(C, ki)
        t, IS, HA, pCS, _drg = gen._run_simulation(
            ic_ss, live_pk={"CL_FURO": np.nan},
            dose_per_admin=80.0, n_doses=3,
            dose_times_h=[0.0, 8.0, 16.0],
        )
    finally:
        gen.Ki = saved_ki
        gen._restore_ki = saved_restore
    base = {
        "IS": float(ic_ss.get("C_tox1_cen", np.nan)),
        "HA": float(ic_ss.get("C_tox2_cen", np.nan)),
        "pCS": float(ic_ss.get("C_tox3_cen", np.nan)),
    }
    means = {
        "IS": _mean_last_cycle(t, IS),
        "HA": _mean_last_cycle(t, HA),
        "pCS": _mean_last_cycle(t, pCS),
    }
    return {
        name: (100.0 * (means[name] / base[name] - 1.0)
               if base[name] and base[name] > 1e-9 else float("nan"))
        for name in TOXINS
    }


def run_row(scale: float, ki_values: Sequence[float], scenario: str = "single_iv"
            ) -> Tuple[Dict[str, float], Dict[float, Dict[str, float]]]:
    """Warm-up (+ IV no-drug reference) once, then one drug run per Ki."""
    from simulation import run_sim
    from state_vector import IDX
    import compounds as C
    import constants as K
    K.VERBOSE = False
    scenario = _norm_scenario(scenario)

    _set_vmax_scale(scale)
    C.initialize_compounds("ckd5")
    _drug_off(C)
    sol_ss = run_sim(t_end_h=WARMUP_H, mode="ckd5", dt=DT_WARMUP)
    ic_base = {n: float(sol_ss.y[i, -1]) for n, i in IDX.items()}
    ic_drug = {
        k: (0.0 if k in ("C_ven", "C_peri", "C_bladder", "Ae_drg") else v)
        for k, v in ic_base.items()
    }

    n = max(10, len(sol_ss.t) // 10)
    for tox, (_, key) in TOXINS.items():
        s = sol_ss.y[IDX[key], -n:]
        drift = abs(s[-1] - s[0]) / max(abs(s[-1]), 1e-12)
        if drift > 0.01:
            print(f"    [warn] scale={scale:.3f}: {tox} drift {drift * 100:.1f}% "
                  f"in last 10% of warm-up — increase WARMUP_H")

    if scenario == "chronic_oral":
        base = {tox: float(ic_base[key]) for tox, (_, key) in TOXINS.items()}
        out = {}
        for ki in ki_values:
            out[float(ki)] = _run_chronic_oral(ic_base, float(ki), scale)
        return base, out

    C.initialize_compounds("ckd5")
    _drug_off(C)
    sol_no = run_sim(
        t_end_h=SIM_HOURS, mode="ckd5", dt=DT_SIM,
        custom_initial_conds=ic_base,
    )
    base = {tox: float(sol_no.y[IDX[key], -1]) for tox, (_, key) in TOXINS.items()}

    out = {}
    for ki in ki_values:
        C.initialize_compounds("ckd5")
        _force_ki(C, ki)
        _drug_single_iv(C)
        sol = run_sim(
            t_end_h=SIM_HOURS, mode="ckd5", dt=DT_SIM,
            custom_initial_conds=ic_drug,
        )
        out[float(ki)] = {
            tox: 100.0 * (float(sol.y[IDX[key], -1]) / base[tox] - 1.0)
            for tox, (_, key) in TOXINS.items()
        }
    return base, out


def compute_heatmaps(ki_grid=None, vmax_grid=None,
                     scenario: Optional[str] = None, use_cache: bool = True
                     ) -> Dict[str, np.ndarray]:
    """Returns {"IS": 2D, "HA": 2D, "pCS": 2D}, shape (n_vmax, n_ki)."""
    import compound_parameters as CP
    import constants as K
    K.VERBOSE = False
    scenario = _norm_scenario(scenario or SCENARIO)
    ki_grid = Ki_grid if ki_grid is None else np.asarray(ki_grid, float)
    vmax_grid = Vmax_grid if vmax_grid is None else np.asarray(vmax_grid, float)

    original_scales = copy.deepcopy(CP._CKD_SCALES)
    fp = parameter_fingerprint(scenario)
    print(f"Parameter fingerprint: {fp}  scenario={scenario}")

    cache = {}
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"  Loaded {len(cache)} cached entries")
        except Exception:
            cache = {}

    grids = {tox: np.full((len(vmax_grid), len(ki_grid)), np.nan) for tox in TOXINS}
    t0 = time.time()
    try:
        for i, scale in enumerate(vmax_grid):
            keys = [f"{fp}|{scale:.6f}|{ki:.6f}" for ki in ki_grid]
            todo = [ki for ki, k in zip(ki_grid, keys) if k not in cache]
            if todo:
                print(f"  row {i + 1}/{len(vmax_grid)}  scale={scale:.3f}  "
                      f"{len(todo)} new Ki  ({(time.time() - t0) / 60:.1f} min)")
                _, res = run_row(scale, todo, scenario)
                for ki, r in res.items():
                    cache[f"{fp}|{scale:.6f}|{ki:.6f}"] = r
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(cache, f)
            for j, k in enumerate(keys):
                for tox in TOXINS:
                    grids[tox][i, j] = cache[k][tox]
    finally:
        CP._CKD_SCALES.clear()
        CP._CKD_SCALES.update(original_scales)
        CP.CKD_ADJUSTMENTS = CP._create_ckd_adjustments_dynamic(CP._CKD_SCALES)

    for tox, g in grids.items():
        print(f"  {tox}: {np.nanmin(g):+.2f} % to {np.nanmax(g):+.2f} %")
    return grids


def compute_heatmap(toxin: str, use_cache: bool = True) -> np.ndarray:
    return compute_heatmaps(use_cache=use_cache)[toxin]


if __name__ == "__main__":
    compute_heatmaps()
