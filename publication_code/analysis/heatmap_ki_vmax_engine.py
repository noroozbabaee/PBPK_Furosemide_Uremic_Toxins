# -*- coding: utf-8 -*-
"""Ki × Vmax ODE heatmap engine (IV single-dose or chronic oral).

This is NOT manuscript Fig 7 by default. Fig 7 in the paper is the
analytical chronic map (heatmap_ki_vmax_analytical.pdf), which is not
in this git tree.

SCENARIO
  "iv_single"     1000 h no-drug warm-up once per Vmax row, then 24 h
                  no-drug + 24 h IV 120 mg / 2 h. Outcome: %Δ of total
                  blood C_tox*_cen at t = 24 h (same % as plasma).
  "chronic_oral"  same warm-up, then dose_response_tid_vs_qd._run_simulation
                  TID 80 mg × 3, 30 days. Outcome: last-cycle mean vs
                  warm-up baseline.

Ki does not affect the no-drug warm-up; that run is shared across each
Vmax row. All three toxins are read from every ODE run.

Vmax_scale replaces the CKD-5 basolateral OAT1 scale (default 0.05).
It is a fraction of the *healthy* Vmax, not “relative to CKD-5”.
_CKD_SCALES is restored after each row.

C_tox*_cen is total blood (mg/L). Divide by BP for plasma. %Δ is
identical. Do not call a single-dose map ΔC_ss.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

OUTDIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(OUTDIR, exist_ok=True)

# Default = supplementary IV map. Chronic ODE grid is opt-in (expensive).
SCENARIO = os.environ.get("HEATMAP_SCENARIO", "iv_single")

SIM_HOURS = 24.0
WARMUP_H = 1000.0
DRIFT_WARN = 0.02
DOSE_MG = 120.0
INFUSION_H = 2.0
DT_SIM = 0.1
DT_WARMUP = 0.5

TOXIN_KEYS = {
    "IS": "C_tox1_cen",
    "HA": "C_tox2_cen",
    "pCS": "C_tox3_cen",
}
TOXIN_BP = {"IS": 0.70, "HA": 0.84, "pCS": 0.55}

N_KI = 15
N_VMAX = 12
KI_MIN, KI_MAX = 0.05, 5.0
VMAX_MIN, VMAX_MAX = 0.01, 0.30
Ki_grid = np.logspace(np.log10(KI_MIN), np.log10(KI_MAX), N_KI)
Vmax_grid = np.linspace(VMAX_MIN, VMAX_MAX, N_VMAX)

CACHE_FILE = os.path.join(OUTDIR, "heatmap_ki_vmax_engine.pkl")


def _compound_fingerprint() -> str:
    import compound_parameters as CP
    payload = {}
    for name in ("TOX1_BASE", "TOX2_BASE", "TOX3_BASE", "DRUG_BASE"):
        d = getattr(CP, name, {})
        if not isinstance(d, dict):
            continue
        payload[name] = {
            k: d[k]
            for k in sorted(d)
            if isinstance(d[k], (int, float, str, bool))
        }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _force_ki(ki_value: float) -> None:
    import compounds as C
    for tox, key in (
        (C.tox1, "Ki_furo_to_IS_mgL"),
        (C.tox2, "Ki_furo_to_HA_mgL"),
        (C.tox3, "Ki_furo_to_pCS_mgL"),
    ):
        if isinstance(tox, dict):
            tox[key] = float(ki_value)


def _set_vmax_scale(vmax_scale: float) -> None:
    """Apply the same healthy-fraction scale to IS, HA, and pCS baso Vmax."""
    import compound_parameters as CP
    for tox in ("IS", "HA", "pCS"):
        CP._CKD_SCALES["per_compound"][tox]["baso"] = float(vmax_scale)
    CP.CKD_ADJUSTMENTS = CP._create_ckd_adjustments_dynamic(CP._CKD_SCALES)


def _set_vmax(vmax_scale: float) -> None:
    """Alias used by the validation script."""
    _set_vmax_scale(vmax_scale)


@contextmanager
def vmax_scale_ctx(vmax_scale: float):
    import compound_parameters as CP
    saved = copy.deepcopy(CP._CKD_SCALES)
    try:
        _set_vmax_scale(vmax_scale)
        yield
    finally:
        CP._CKD_SCALES = saved
        CP.CKD_ADJUSTMENTS = CP._create_ckd_adjustments_dynamic(CP._CKD_SCALES)


def _ic_from_sol(sol) -> dict:
    from state_vector import IDX
    ic = {name: float(sol.y[idx, -1]) for name, idx in IDX.items()}
    for k in list(ic):
        if k.startswith("Ae_"):
            ic[k] = 0.0
    return ic


def _zero_drug(ic: dict) -> dict:
    out = dict(ic)
    for k in ("C_ven", "C_peri", "C_bladder", "Ae_drg"):
        if k in out:
            out[k] = 0.0
    return out


def _read_toxins(sol) -> Dict[str, float]:
    from state_vector import IDX
    return {
        name: float(sol.y[IDX[key], -1])
        for name, key in TOXIN_KEYS.items()
        if key in IDX
    }


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


def _warmup_ic(vmax_scale: float) -> dict:
    import constants as K
    import compounds as C
    from simulation import run_sim
    K.VERBOSE = False
    with vmax_scale_ctx(vmax_scale):
        C.initialize_compounds("ckd5")
        _force_ki(0.35)
        C.drug.update({
            "enabled": False, "IR": 0.0,
            "t_start_h": 1e9, "t_inf_h": 0.0,
            "bolus_mg": 0.0, "inhibit": False,
        })
        sol = run_sim(t_end_h=WARMUP_H, mode="ckd5", dt=DT_WARMUP)
        if sol is None:
            raise RuntimeError("warm-up failed")
        # Drift check: last 10% vs previous 10% of the tail.
        n = sol.y.shape[1]
        a, b = max(n // 10, 2), max(n // 5, 4)
        from state_vector import IDX
        for name, key in TOXIN_KEYS.items():
            if key not in IDX:
                continue
            tail = sol.y[IDX[key], -a:]
            prev = sol.y[IDX[key], -b:-a]
            m1, m0 = float(np.mean(tail)), float(np.mean(prev))
            if m0 > 1e-9 and abs(m1 - m0) / m0 > DRIFT_WARN:
                print(f"    [warn] {name} still drifting after {WARMUP_H:.0f} h "
                      f"({100 * (m1 - m0) / m0:+.1f}%)")
        return _ic_from_sol(sol)


def _run_iv_pair(ic_ss: dict, ki: float, vmax_scale: float) -> Dict[str, float]:
    import constants as K
    import compounds as C
    from simulation import run_sim
    K.VERBOSE = False
    with vmax_scale_ctx(vmax_scale):
        C.initialize_compounds("ckd5")
        _force_ki(ki)
        C.drug.update({
            "enabled": False, "IR": 0.0,
            "t_start_h": 1e9, "t_inf_h": 0.0,
            "bolus_mg": 0.0, "inhibit": False,
        })
        sol_no = run_sim(
            t_end_h=SIM_HOURS, mode="ckd5", dt=DT_SIM,
            custom_initial_conds=dict(ic_ss),
        )
        C.initialize_compounds("ckd5")
        _force_ki(ki)
        C.drug.update({
            "enabled": True,
            "IR": DOSE_MG / INFUSION_H,
            "t_inf_h": INFUSION_H,
            "t_start_h": 0.05,
            "bolus_mg": 0.0,
            "inhibit": True,
        })
        for attr in ("tox1", "tox2", "tox3"):
            tox = getattr(C, attr, None)
            if isinstance(tox, dict):
                tox["enabled"] = True
        sol_d = run_sim(
            t_end_h=SIM_HOURS, mode="ckd5", dt=DT_SIM,
            custom_initial_conds=_zero_drug(ic_ss),
        )
    no, yes = _read_toxins(sol_no), _read_toxins(sol_d)
    out = {}
    for name in TOXIN_KEYS:
        c0, c1 = no.get(name, np.nan), yes.get(name, np.nan)
        out[name] = (c1 - c0) / c0 * 100.0 if c0 and c0 > 1e-9 else np.nan
    return out


def _run_chronic_oral(ic_ss: dict, ki: float, vmax_scale: float) -> Dict[str, float]:
    """TID 80×3 for 30 days via the Fig 5 driver."""
    import compounds as C
    import dose_response_tid_vs_qd as gen
    saved_ki = gen.Ki
    saved_restore = gen._restore_ki
    gen.Ki = float(ki)

    def _restore_and_force():
        saved_restore()
        _force_ki(ki)

    gen._restore_ki = _restore_and_force
    try:
        with vmax_scale_ctx(vmax_scale):
            C.initialize_compounds("ckd5")
            _force_ki(ki)
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
    out = {}
    for name in TOXIN_KEYS:
        c0, c1 = base[name], means[name]
        out[name] = (c1 - c0) / c0 * 100.0 if c0 and c0 > 1e-9 else np.nan
    return out


def compute_heatmaps(
    scenario: Optional[str] = None,
    use_cache: bool = True,
    ki_grid=None,
    vmax_grid=None,
) -> Dict[str, np.ndarray]:
    """Return {"IS","HA","pCS"} 2D arrays (n_vmax × n_ki) of %Δ."""
    scenario = scenario or SCENARIO
    ki_grid = Ki_grid if ki_grid is None else np.asarray(ki_grid, float)
    vmax_grid = Vmax_grid if vmax_grid is None else np.asarray(vmax_grid, float)
    n_v, n_k = len(vmax_grid), len(ki_grid)
    fp = _compound_fingerprint()

    cache: dict = {}
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"  Loaded {len(cache)} cached entries ({CACHE_FILE})")
        except Exception:
            cache = {}

    grids = {name: np.full((n_v, n_k), np.nan) for name in TOXIN_KEYS}
    t0 = time.time()
    print(f"\n  scenario={scenario}  {n_v}×{n_k}  fingerprint={fp}")

    for i, scale in enumerate(vmax_grid):
        ic_ss = None
        for j, ki in enumerate(ki_grid):
            key = (
                f"{scenario}_{fp}_ki{ki:.6f}_vm{scale:.6f}"
                f"_w{WARMUP_H:.0f}_t{SIM_HOURS:.0f}"
            )
            if key in cache and isinstance(cache[key], dict):
                for name in TOXIN_KEYS:
                    grids[name][i, j] = cache[key].get(name, np.nan)
                continue
            if ic_ss is None:
                print(f"    warm-up Vmax={scale:.3f} ({WARMUP_H:.0f} h)…")
                ic_ss = _warmup_ic(scale)
            print(f"    [{i + 1}/{n_v} {j + 1}/{n_k}] Ki={ki:.3f}")
            if scenario == "chronic_oral":
                pcts = _run_chronic_oral(ic_ss, ki, scale)
            else:
                pcts = _run_iv_pair(ic_ss, ki, scale)
            cache[key] = pcts
            for name in TOXIN_KEYS:
                grids[name][i, j] = pcts.get(name, np.nan)
            try:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(cache, f)
            except Exception:
                pass

    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return grids


def compute_heatmap(toxin: str, use_cache: bool = True) -> np.ndarray:
    """Back-compat: one toxin from a full three-toxin compute_heatmaps()."""
    return compute_heatmaps(use_cache=use_cache)[toxin]
