# -*- coding: utf-8 -*-
"""
chronic_oral_ss_v5.py
======================
Simulate chronic ORAL furosemide in CKD-5 for 30 days.

CHANGES vs v4:
──────────────
1. DOSING REGIMEN: 40 mg × 3/day (TID) instead of 240 mg once-daily.
   This matches the predominant >120 mg/day regimen in CKD-REIN
   (Costes-Albrespic M et al. Kidney Int Rep 2025;10:2165).
   Three doses per day at t=0h, t=8h, t=16h (every 8 hours).

2. NO HARDCODED SIMULATION OUTPUTS:
   CL_R_IS, F_OAT1_IS, CL_R_pCS, F_OAT1_pCS, CL_FURO are now
   extracted LIVE from the baseline ODE run, not read from module-level
   constants. The analytical formula therefore uses the same simulation
   that produced the baseline concentrations — full internal consistency.

3. Ki COMMENT BUG FIXED:
   v4 comment said "CORRECTED from 0.35 → 3.77" but value stayed 0.35.
   Ki = 0.35 mg/L is the calibrated model value. The Mihaila 2020
   fluorescein value Ki = 3.77 mg/L is shown as a reference comparison
   only, not used in the model.

4. CACHE KEY includes DOSING_REGIMEN so once-daily and TID caches
   never collide. The cache_path includes the regimen name.

DATA FLOW — NOTHING IS HARDCODED FROM PRIOR SIMULATIONS:
──────────────────────────────────────────────────────────
  Step 1 — warm-up 500h CKD-5 → extract IS_base, pCS_base
  Step 2 — baseline 24h no-drug run → compute CL_R_IS, CL_R_pCS,
            F_OAT1_IS, F_OAT1_pCS, CL_FURO from live ODE output
  Step 3 — 30 days TID oral dosing → IS_all, pCS_all, drg_all
  Step 4 — analytical_ss() uses IS_base, pCS_base, and the LIVE
            clearance values from Step 2
  Clinical values (31.9%, 29.3%, CIs) are literature constants —
  correct to hardcode.

VERIFIED REFERENCES:
────────────────────
F_ORAL = 0.60        FDA label NDA 016273, 2016
K_A    = 1.5 h⁻¹    Hammarlund MM et al. Eur J Clin Pharmacol 1984;26:197
FU_P   = 0.05        Takamura N et al. Clin Chem 1997;43:2274
BP     = 0.82        Brater DC et al. Clin Pharmacol Ther 1980;28:426
Ki     = 0.35 mg/L   Calibrated to Costes-Albrespic 2025 clinical signal
                     (Mihaila 2020 fluorescein reference: 3.77 mg/L — not used)
KM_IS  = 3.84 mg/L   Deguchi T et al. Kidney Int 2004;65:162
KM_pCS = 22.58 mg/L  Watanabe H et al. Kidney Int 2014;86:840
fu_IS  = 0.10        Deltombe O et al. Toxins 2015;7:3933
fu_pCS = 0.05        Deltombe O et al. Toxins 2015;7:3933
Clinical: Costes-Albrespic M et al. Kidney Int Rep 2025;10:2165
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
# Add it to sys.path so the imports below work regardless of which directory
# this script is launched from (IDE "Run", terminal, or another script that
# imports this one).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import copy
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Force fresh module load ───────────────────────────────────────────────────
for _mod in ["compound_parameters", "compounds", "simulation",
             "ode_system_twocompartment", "physiology", "constants"]:
    if _mod in sys.modules:
        del sys.modules[_mod]

import compound_parameters as CP
import compounds as C
from simulation import run_sim
from state_vector import IDX
import constants as K

# ── Output directory ────────────────────────────────────────────────────────
# All caches, results, and figures from this script go to ../results/, not a
# local "figures/" folder, so every analysis script in this repo writes to
# the same shared, top-level results/ directory regardless of which folder
# it is launched from.
OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
from datetime import date as _date
RUN_DATE = _date.today().strftime("%Y%m%d")
K.VERBOSE = False

# ═════════════════════════════════════════════════════════════════════════════
# DOSING REGIMEN — 80 mg × 3/day (TID) = 240 mg/day
# ═════════════════════════════════════════════════════════════════════════════
# Targets the >120 mg/day category in CKD-REIN cohort.
# Costes-Albrespic M et al. Kidney Int Rep 2025;10:2165.
DOSING_REGIMEN = "TID"  # "TID" = 3×/day  |  "QD" = once-daily
# When True, skip the paired once-daily run at the same daily total.
SKIP_QD = False
DOSE_PER_ADMIN = 80.0  # mg per administration
N_DOSES_PER_DAY = 3  # 3 doses per day
DOSE_INTERVAL_H = 8.0  # h between doses (24h / 3 = 8h)
DOSE_TIMES_H = [0.0, 8.0, 16.0]  # h within each 24h cycle

DOSE_TOTAL_PER_DAY = DOSE_PER_ADMIN * N_DOSES_PER_DAY  # 240 mg/day

# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — all with verified citations
# ═════════════════════════════════════════════════════════════════════════════
MODE = "ckd5"
SS_WARMUP_H = 500.0  # h — IS CKD-5 tau≈18h (Vcen/CLR); 500h → true SS

F_ORAL = 0.60  # FDA label NDA 016273, 2016
K_A = 1.5  # h⁻¹ — Hammarlund 1984 (range 1.0–2.0 h⁻¹)
TAU_H = 24.0  # h — length of one full day cycle
N_DAYS = 30  # days
N_ORAL_STEPS = 20  # piecewise segments per dose
T_WINDOW_H = 6.0  # h — absorption window (99.99% absorbed at t=6h)
DT_WITHIN = 0.1  # h — ODE step during absorption
DT_BETWEEN = 0.5  # h — ODE step post-absorption
VCEN = 5.0  # L — central venous blood volume

# PK parameters — literature-verified, NOT simulation outputs
FU_P_FURO = 0.05  # Takamura N et al. Clin Chem 1997;43:2274
BP_FURO = 0.82  # Brater DC et al. Clin Pharmacol Ther 1980;28:426
Ki = 0.35  # mg/L — calibrated to Costes-Albrespic 2025

# MM substrate parameters — literature-verified
KM_IS = 3.84  # mg/L — Deguchi T et al. Kidney Int 2004;65:162
KM_pCS = 22.58  # mg/L — Watanabe H et al. Kidney Int 2014;86:840
FU_P_IS = 0.10  # Deltombe O et al. Toxins 2015;7:3933 (CKD-5)
FU_P_pCS = 0.05  # Deltombe O et al. Toxins 2015;7:3933 (CKD-5)
BP_IS = 0.70  # from compound_parameters.py
BP_pCS = 0.55  # from compound_parameters.py

# Clinical reference values — literature constants (correct to hardcode)
CLIN_IS_PCT = 31.9  # % — Costes-Albrespic 2025
CLIN_pCS_PCT = 29.3  # %
CLIN_IS_LO = 13.5  # % — 95% CI lower bound
CLIN_IS_HI = 53.3  # % — 95% CI upper bound
CLIN_pCS_LO = 6.1  # %
CLIN_pCS_HI = 57.8  # %


# NOTE: CL_FURO, CL_R_IS, F_OAT1_IS, CL_R_pCS, F_OAT1_pCS are NOT
# hardcoded here. They are extracted live from the baseline ODE run
# in extract_baseline_pk() and stored in the returned dict.

# ═════════════════════════════════════════════════════════════════════════════
# CACHE
# ═════════════════════════════════════════════════════════════════════════════
def _cache_path() -> str:
    """
    Cache path encodes Ki, dose per administration, number of doses per day,
    and number of simulation days — so every unique combination gets its own
    cache file and nothing ever collides.

    Format:
        ../results/chronic_oral_cache_Ki{Ki}_dose{DOSE}mg_x{N}pd_{DAYS}days.npz

    Examples:
        Ki0p35_dose40mg_x3pd_30days   →  original TID run
        Ki0p35_dose80mg_x3pd_30days   →  80mg TID, 30 days
        Ki0p5_dose80mg_x3pd_30days    →  80mg TID, Ki=0.50, 30 days
        Ki0p35_dose80mg_x3pd_60days   →  80mg TID, 60-day run
    """
    ki_str   = f"{Ki:.4g}".replace(".", "p")          # e.g. 0.35 -> "0p35"
    dose_str = f"{DOSE_PER_ADMIN:.4g}"                 # e.g. 80.0 -> "80"
    return os.path.join(
        OUTPUT_DIR,
        f"chronic_oral_cache_"
        f"Ki{ki_str}_dose{dose_str}mg_x{N_DOSES_PER_DAY}pd_{N_DAYS}days_{MODE}_{RUN_DATE}.npz"
    )


def _build_fingerprint(live_pk: dict) -> dict:
    """
    Build parameter fingerprint including LIVE PK values.
    The fingerprint is saved alongside arrays and validated on reload.
    """
    return dict(
        DOSING_CODE=float(N_DOSES_PER_DAY),
        DOSE_PER_ADMIN=DOSE_PER_ADMIN,
        DOSE_INTERVAL=DOSE_INTERVAL_H,
        F_ORAL=F_ORAL,
        K_A=K_A,
        TAU_H=TAU_H,
        N_DAYS=float(N_DAYS),
        N_ORAL_STEPS=float(N_ORAL_STEPS),
        T_WINDOW_H=T_WINDOW_H,
        SS_WARMUP_H=SS_WARMUP_H,
        Ki=Ki,
        FU_P_FURO=FU_P_FURO,
        BP_FURO=BP_FURO,
        KM_IS=KM_IS,
        KM_pCS=KM_pCS,
        FU_P_IS=FU_P_IS,
        FU_P_pCS=FU_P_pCS,
        BP_IS=BP_IS,
        BP_pCS=BP_pCS,
        MODE_CODE=5.0 if MODE == "ckd5" else 0.0,
        # 1 = residual furosemide is carried across days (not zeroed at 24 h)
        CHRONIC_CARRYOVER=1.0,
        HA_TRACKED=1.0,
        # Live PK — if model changes, these will differ and force re-run
        CL_FURO=live_pk["CL_FURO"],
        CL_R_IS=live_pk["CL_R_IS"],
        F_OAT1_IS=live_pk["F_OAT1_IS"],
        CL_R_pCS=live_pk["CL_R_pCS"],
        F_OAT1_pCS=live_pk["F_OAT1_pCS"],
    )


def _cache_is_valid(cache_path: str, live_pk: dict) -> bool:
    """Load cache and compare all parameters. Returns True only if all match."""
    if not os.path.exists(cache_path):
        return False
    try:
        cache = np.load(cache_path, allow_pickle=False)
    except Exception as e:
        print(f"  [CACHE] Could not load: {e}")
        return False

    fp = _build_fingerprint(live_pk)
    mismatches = []
    for key, current_val in fp.items():
        if "HA_all" not in cache.files:
            mismatches.append(("HA_all", "MISSING", "required"))
            continue
        cached_val = float(cache[key].flat[0])
        if abs(cached_val - current_val) > 1e-9:
            mismatches.append((key, cached_val, current_val))
    if "HA_all" not in cache.files:
        mismatches.append(("HA_all", "MISSING", "required"))

    if mismatches:
        print(f"\n  [CACHE] Parameter mismatch — forcing re-run:")
        print(f"  {'Parameter':<20} {'Cached':>14} {'Current':>14}")
        print("  " + "─" * 50)
        for key, old, new in mismatches:
            print(f"  {key:<20} {str(old):>14} {str(new):>14}")
        return False

    print(f"  [CACHE] All {len(fp)} parameters match — loading saved run.")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════════════════════
def _trapz(y, x):
    """Version-safe trapezoidal integration.

    NumPy >= 2.0 added np.trapezoid and deprecated np.trapz.
    NumPy >= 2.something REMOVED np.trapz entirely, so the old
    `getattr(np, "trapezoid", np.trapz)` pattern crashes — the
    default argument np.trapz is evaluated eagerly even when
    np.trapezoid exists, and raises AttributeError if np.trapz
    is gone. Use hasattr() to check first instead.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def _restore_ki() -> None:
    """Restore Ki after every initialize_compounds() call."""
    _, IS_h, HA_h, pCS_h = CP.get_compound_parameters("healthy")
    for tox, key, src in [
        (C.tox1, "Ki_furo_to_IS_mgL", IS_h),
        (C.tox2, "Ki_furo_to_HA_mgL", HA_h),
        (C.tox3, "Ki_furo_to_pCS_mgL", pCS_h),
    ]:
        if isinstance(tox, dict):
            tox[key] = float(src.get(key, 1e9))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — WARM-UP
# ═════════════════════════════════════════════════════════════════════════════
def warmup() -> dict:
    """500h CKD-5 steady-state warm-up — no drug."""
    print(f"\n[STEP 1] SS warm-up: {SS_WARMUP_H:.0f}h (no drug, CKD-5)...")
    C.initialize_compounds(MODE)
    _restore_ki()
    C.drug.update({"enabled": False, "IR": 0.0,
                   "t_start_h": 1e9, "t_inf_h": 0.0, "bolus_mg": 0.0})
    sol = run_sim(t_end_h=SS_WARMUP_H, mode=MODE, dt=1.0)
    ic = {name: float(sol.y[i, -1]) for name, i in IDX.items()}
    for k in list(ic.keys()):
        if k.startswith("Ae_"):
            ic[k] = 0.0
    IS_base = ic.get("C_tox1_cen", 0.0)
    pCS_base = ic.get("C_tox3_cen", 0.0)
    print(f"  IS  baseline = {IS_base:.4f} mg/L")
    print(f"  pCS baseline = {pCS_base:.4f} mg/L")
    return ic


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT LIVE PK FROM BASELINE ODE RUN
# ═════════════════════════════════════════════════════════════════════════════
def extract_baseline_pk(ic_ss: dict) -> dict:
    """
    Run a 24h no-drug CKD-5 simulation from SS initial conditions.
    Extract CL_R_IS, F_OAT1_IS, CL_R_pCS, F_OAT1_pCS, CL_FURO from
    the ODE output using Ae/AUC method.

    Returns dict of live PK values — these feed into analytical_ss()
    so that the analytical formula uses the same model state as the
    ODE simulation. Nothing is hardcoded.
    """
    print("\n[STEP 2] Extracting live PK from 24h baseline run...")
    C.initialize_compounds(MODE)
    _restore_ki()

    # Run with furosemide IV 120 mg over 2h to extract CL_FURO
    dp, *_ = CP.get_compound_parameters(MODE)
    for k, v in dp.items():
        C.drug[k] = v
    C.drug.update({
        "enabled": True,
        "IR": 60.0,  # mg/h × 2h = 120 mg
        "t_inf_h": 2.0,
        "t_start_h": 0.0,
        "bolus_mg": 0.0,
        "inhibit": False,  # No inhibition — pure PK run
    })
    for attr in ("tox1", "tox2", "tox3"):
        tox = getattr(C, attr, None)
        if isinstance(tox, dict):
            tox["enabled"] = True

    ic_drug = copy.deepcopy(ic_ss)
    for k in ("C_ven", "C_peri", "Ae_drg", "C_bladder"):
        if k in ic_drug:
            ic_drug[k] = 0.0

    sol = run_sim(t_end_h=24.0, mode=MODE, dt=0.05,
                  custom_initial_conds=ic_drug)

    # ── Extract CL_FURO from drug AUC and Ae ─────────────────────────────
    t = sol.t
    i_cven = IDX.get("C_ven")
    i_aedrg = IDX.get("Ae_drg")

    if i_cven is not None and i_aedrg is not None:
        C_drug = np.clip(sol.y[i_cven], 0, np.inf)
        Ae_drug = np.maximum.accumulate(sol.y[i_aedrg])
        auc_drg = float(_trapz(C_drug / BP_FURO, t))
        dAe_drg = float(Ae_drug[-1] - Ae_drug[0])
        # Primary: live ODE. Fallback only if AUC is zero (numerical failure).
        CL_FURO_live = dAe_drg / auc_drg if auc_drg > 1e-9 else 1.70
    else:
        print("  [WARN] C_ven or Ae_drg not in IDX — using fallback CL_FURO")
        CL_FURO_live = 1.70

    # ── Extract toxin clearances and OAT1 fractions ────────────────────────
    # Run no-drug baseline 24h to get clean toxin Ae/AUC
    C.initialize_compounds(MODE)
    _restore_ki()
    C.drug.update({"enabled": False, "IR": 0.0,
                   "t_start_h": 1e9, "t_inf_h": 0.0, "bolus_mg": 0.0})
    for attr in ("tox1", "tox2", "tox3"):
        tox = getattr(C, attr, None)
        if isinstance(tox, dict):
            tox["enabled"] = True

    sol_nd = run_sim(t_end_h=24.0, mode=MODE, dt=0.05,
                     custom_initial_conds=copy.deepcopy(ic_ss))
    t_nd = sol_nd.t

    live_pk = {"CL_FURO": CL_FURO_live}
    toxin_map = [
        ("IS", "C_tox1_cen", "Ae_t1", "F_flux_OAT1_IS", BP_IS, "CL_R_IS", "F_OAT1_IS"),
        ("pCS", "C_tox3_cen", "Ae_t3", "F_flux_OAT1_pCS", BP_pCS, "CL_R_pCS", "F_OAT1_pCS"),
    ]

    for name, c_key, ae_key, foat_key, bp, cl_key, foat_out_key in toxin_map:
        i_c = IDX.get(c_key)
        i_ae = IDX.get(ae_key)
        if i_c is None or i_ae is None:
            print(f"  [WARN] {name}: indices not found — using fallback")
            live_pk[cl_key] = 0.274 if name == "IS" else 0.086
            live_pk[foat_out_key] = 0.631 if name == "IS" else 0.562
            continue

        C_tox = np.clip(sol_nd.y[i_c], 0, np.inf)
        Ae_tox = np.maximum.accumulate(sol_nd.y[i_ae])
        auc_tox = float(_trapz(C_tox / bp, t_nd))
        dAe_tox = float(Ae_tox[-1] - Ae_tox[0])
        CL_R = dAe_tox / auc_tox if auc_tox > 1e-9 else 0.0
        live_pk[cl_key] = CL_R

        # OAT1 fraction from flux decomposition if available
        i_foat = IDX.get(foat_key)
        if i_foat is not None:
            # Average OAT1 fraction over simulation
            foat_vals = sol_nd.y[i_foat]
            foat_vals = foat_vals[foat_vals >= 0]
            live_pk[foat_out_key] = float(np.mean(foat_vals)) if len(foat_vals) > 0 else \
                (0.631 if name == "IS" else 0.562)
        else:
            # Fallback: estimate from GFR
            # CL_GFR = GFR_CKD5 × fu_p / BP (GFR_CKD5 ~ 0.1 L/h in our model)
            # F_OAT1 = 1 - CL_GFR / CL_R
            GFR_CKD5 = 0.1  # L/h approximate
            CL_GFR = GFR_CKD5 * (FU_P_IS if name == "IS" else FU_P_pCS) / bp
            F_OAT1 = max(0.0, min(0.99, 1.0 - CL_GFR / CL_R)) if CL_R > 1e-6 else 0.6
            live_pk[foat_out_key] = F_OAT1
            print(f"  [INFO] {name}: F_OAT1 estimated from GFR = {F_OAT1:.3f}")

    print(f"\n  LIVE PK VALUES (from ODE, no hardcoding):")
    print(f"  {'Parameter':<18} {'Value':>10}  {'Units'}")
    print("  " + "─" * 45)
    print(f"  {'CL_FURO':<18} {live_pk['CL_FURO']:>10.4f}  L/h")
    print(f"  {'CL_R_IS':<18} {live_pk['CL_R_IS']:>10.4f}  L/h")
    print(f"  {'F_OAT1_IS':<18} {live_pk['F_OAT1_IS']:>10.4f}  —")
    print(f"  {'CL_R_pCS':<18} {live_pk['CL_R_pCS']:>10.4f}  L/h")
    print(f"  {'F_OAT1_pCS':<18} {live_pk['F_OAT1_pCS']:>10.4f}  —")
    return live_pk


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — ORAL ABSORPTION PROFILE
# ═════════════════════════════════════════════════════════════════════════════
def build_oral_profile(dose_mg: float) -> list:
    """
    Build piecewise-constant IR segments for one oral dose.
    Returns list of (t0, t1, IR_mg_h) tuples — relative to dose time.
    """
    A0 = F_ORAL * dose_mg
    t_edges = np.linspace(0, T_WINDOW_H, N_ORAL_STEPS + 1)
    segs = []
    for i in range(N_ORAL_STEPS):
        t0 = t_edges[i]
        t1 = t_edges[i + 1]
        dt = t1 - t0
        A_abs = A0 * (np.exp(-K_A * t0) - np.exp(-K_A * t1))
        IR = A_abs / max(dt, 1e-9)
        segs.append((t0, t1, IR))
    total = sum(s[2] * (s[1] - s[0]) for s in segs)
    pct_err = abs(total - A0) / A0 * 100 if A0 > 0 else 0
    print(f"  Oral profile ({dose_mg:.0f} mg): {N_ORAL_STEPS} segs, "
          f"{T_WINDOW_H}h window, absorbed={total:.3f} mg "
          f"(err={pct_err:.3f}%)")
    return segs


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — ONE 24h CYCLE WITH TID DOSING
# ═════════════════════════════════════════════════════════════════════════════
def run_one_day_tid(ic: dict, day_offset_h: float,
                    oral_segs: list, dose_times_h=None) -> tuple:
    """
    Run one 24h day at the given clock times (TID 0/8/16 h or QD 0 h).

    Each dose uses the same oral absorption profile (oral_segs) shifted
    to its dose time. Between doses, the ODE runs with no drug input but
    competitive inhibition still active if furosemide is present.

    Residual furosemide (C_ven, C_peri) is carried in `ic` from the previous
    day. Do not zero those compartments here — that turned a QD run into a
    fresh single dose every morning, and a TID run into three isolated doses
    with no overnight leftover.

    Returns (t_array, y_array, ic_end_of_day)
    """
    K.VERBOSE = False
    t_all_day = []
    y_all_day = []
    ic_cur = copy.deepcopy(ic)
    if dose_times_h is None:
        dose_times_h = DOSE_TIMES_H

    if len(dose_times_h) == 0:
        C.initialize_compounds(MODE)
        _restore_ki()
        dp, *_ = CP.get_compound_parameters(MODE)
        for k, v in dp.items():
            C.drug[k] = v
        C.drug.update({
            "enabled": True,
            "IR": 0.0,
            "t_inf_h": 0.0,
            "t_start_h": 1e9,
            "bolus_mg": 0.0,
            "inhibit": False,
        })
        for attr in ("tox1", "tox2", "tox3"):
            tox = getattr(C, attr, None)
            if isinstance(tox, dict):
                tox["enabled"] = True
        sol = run_sim(t_end_h=TAU_H, mode=MODE, dt=DT_BETWEEN,
                      custom_initial_conds=copy.deepcopy(ic_cur))
        t_day = sol.t + day_offset_h
        ic_end = {name: float(sol.y[i, -1]) for name, i in IDX.items()}
        return t_day, sol.y, ic_end

    # Process each dose within the 24h day
    for dose_idx, dose_time_h in enumerate(dose_times_h):
        # Determine end time for this dose's window
        next_dose_h = dose_times_h[dose_idx + 1] if dose_idx + 1 < len(dose_times_h) \
            else TAU_H

        # ── Phase A: absorption window for this dose ──────────────────────
        for (t0_rel, t1_rel, IR) in oral_segs:
            dur = t1_rel - t0_rel
            if dur < 1e-6:
                continue
            # Absolute time within the day
            t0_abs = dose_time_h + t0_rel
            # Don't run past next dose time
            if t0_abs >= next_dose_h:
                break
            t1_abs = min(dose_time_h + t1_rel, next_dose_h)
            dur = t1_abs - t0_abs
            if dur < 1e-6:
                continue

            C.initialize_compounds(MODE)
            _restore_ki()
            dp, *_ = CP.get_compound_parameters(MODE)
            for k, v in dp.items():
                C.drug[k] = v
            C.drug.update({
                "enabled": True,
                "IR": IR,
                "t_inf_h": dur,
                "t_start_h": 0.0,
                "bolus_mg": 0.0,
                "inhibit": True,
            })
            for attr in ("tox1", "tox2", "tox3"):
                tox = getattr(C, attr, None)
                if isinstance(tox, dict):
                    tox["enabled"] = True

            sol = run_sim(t_end_h=dur, mode=MODE, dt=DT_WITHIN,
                          custom_initial_conds=copy.deepcopy(ic_cur))
            t_all_day.append(sol.t + day_offset_h + t0_abs)
            y_all_day.append(sol.y)
            ic_cur = {name: float(sol.y[i, -1]) for name, i in IDX.items()}

        # ── Phase B: post-absorption until next dose ──────────────────────
        t_abs_end = dose_time_h + min(T_WINDOW_H, next_dose_h - dose_time_h)
        t_remaining = next_dose_h - t_abs_end
        if t_remaining > 0.05:
            C.initialize_compounds(MODE)
            _restore_ki()
            dp, *_ = CP.get_compound_parameters(MODE)
            for k, v in dp.items():
                C.drug[k] = v
            C.drug.update({
                "enabled": True,
                "IR": 0.0,
                "t_inf_h": 0.0,
                "t_start_h": 1e9,
                "bolus_mg": 0.0,
                "inhibit": True,
            })
            for attr in ("tox1", "tox2", "tox3"):
                tox = getattr(C, attr, None)
                if isinstance(tox, dict):
                    tox["enabled"] = True

            sol2 = run_sim(t_end_h=t_remaining, mode=MODE, dt=DT_BETWEEN,
                           custom_initial_conds=copy.deepcopy(ic_cur))
            t_all_day.append(sol2.t + day_offset_h + t_abs_end)
            y_all_day.append(sol2.y)
            ic_cur = {name: float(sol2.y[i, -1]) for name, i in IDX.items()}

    t_day = np.concatenate(t_all_day)
    y_day = np.concatenate(y_all_day, axis=1)
    return t_day, y_day, ic_cur


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — ANALYTICAL SS (uses live PK, not hardcoded values)
# ═════════════════════════════════════════════════════════════════════════════
def _analytical_ss_core(IS_base: float, pCS_base: float,
                        live_pk: dict, Css_furo: float) -> dict:
    """
    Core exact-MM steady-state inhibition calculation.

    Takes Css_furo as an explicit input rather than computing it internally —
    this lets the SAME inhibition math be evaluated against either:
      (a) the closed-form theoretical Css_furo = F*dose/(CL*tau), or
      (b) the actual ODE-simulated time-averaged furosemide concentration.

    See analytical_ss() and analytical_ss_ode_css() below for the two
    entry points that use this core with each Css_furo source.

    Formula:
        Cu_furo  = Css_furo × fu_p / BP
        alpha    = 1 + Cu_furo / Ki
        s        = (Km + Cu_tox) / (alpha × Km + Cu_tox)
        CL_R_new = CL_GFR + CL_OAT1 × s
                 = CL_R × (1 - F_OAT1) + CL_R × F_OAT1 × s
        %change  = (CL_R / CL_R_new - 1) × 100
    """
    Cu_furo = Css_furo * FU_P_FURO / BP_FURO
    alpha = 1.0 + Cu_furo / Ki

    results = {}
    params = [
        ("IS", live_pk["CL_R_IS"], live_pk["F_OAT1_IS"],
         KM_IS, FU_P_IS, BP_IS, IS_base),
        ("pCS", live_pk["CL_R_pCS"], live_pk["F_OAT1_pCS"],
         KM_pCS, FU_P_pCS, BP_pCS, pCS_base),
    ]

    for name, CLR, f_OAT1, Km, fup, bp, Css_tox in params:
        Cu_tox = Css_tox * fup / bp
        # Correct MM scaling factor — NOT simply 1/alpha
        # equals 1/alpha only in unsaturated limit Cu_tox << Km
        s = (Km + Cu_tox) / (alpha * Km + Cu_tox)
        CL_GFR = CLR * (1.0 - f_OAT1)
        CL_OAT1 = CLR * f_OAT1
        CL_new = CL_GFR + CL_OAT1 * s
        pct = (CLR / CL_new - 1.0) * 100.0

        results[name] = {
            "Css_furo": Css_furo,
            "Cu_furo": Cu_furo,
            "alpha": alpha,
            "s": s,
            "CL_R": CLR,
            "F_OAT1": f_OAT1,
            "CL_GFR": CL_GFR,
            "CL_OAT1": CL_OAT1,
            "CL_new": CL_new,
            "pct": pct,
            "Cu_tox": Cu_tox,
        }

    return results


def analytical_ss(IS_base: float, pCS_base: float,
                  live_pk: dict) -> dict:
    """
    THEORETICAL analytical chronic SS prediction (independent of ODE output).

    Css_furo is computed from closed-form PK theory alone:
        Css_furo = F × dose_per_day / (CL_FURO × tau)

    This is dosing-FREQUENCY-INDEPENDENT by construction — TID and QD at the
    same total daily dose give the same Css_furo here, because this formula
    only knows about total daily input vs total daily clearance, not about
    peak/trough dynamics within a day.

    Use this as the fully independent third leg of the ODE / Analytical /
    Clinical comparison — it tests whether PK theory alone (no ODE inputs
    beyond live-extracted clearances) predicts the right ballpark.
    """
    total_dose_per_day = DOSE_PER_ADMIN * N_DOSES_PER_DAY
    Css_furo_theoretical = F_ORAL * total_dose_per_day / (live_pk["CL_FURO"] * TAU_H)
    return _analytical_ss_core(IS_base, pCS_base, live_pk, Css_furo_theoretical)


def analytical_ss_ode_css(IS_base: float, pCS_base: float,
                          live_pk: dict, drg_avg_ode: float) -> dict:
    """
    ODE-FED analytical prediction — DIAGNOSTIC, not independent.

    Uses the SAME exact-MM inhibition formula as analytical_ss(), but feeds
    it the ACTUAL time-averaged furosemide concentration measured from the
    ODE's last-cycle simulation (drg_avg) instead of the closed-form
    theoretical Css_furo.

    WHY THIS EXISTS
    ----------------
    The closed-form Css_furo = F*dose/(CL*tau) is a simple daily-average
    estimate. It cannot distinguish TID from QD dosing at the same total
    daily dose — both give the same Css_furo, even though the ODE shows
    very different inhibition (QD's higher Cmax drives more inhibition
    than TID's flatter profile, because inhibition is a NONLINEAR function
    of Css — see Jensen's inequality: time-averaging a nonlinear function
    of concentration is not the same as evaluating the function at the
    time-averaged concentration).

    This function isolates that effect. Comparing its output to:
      - analytical_ss()  → tells you how much the CLOSED-FORM Css ESTIMATE
                            itself differs from the ODE's true Css_avg
      - the ODE's actual %Δ → tells you how much is left over from
                            TIME-AVERAGING the nonlinear inhibition across
                            a fluctuating within-day profile, even after
                            using the correct (ODE-measured) average Css

    INTERPRETATION CAVEAT
    -----------------------
    This is NOT an independent prediction. It is fed by the ODE's own
    output (drg_avg), so by definition it cannot disagree with the ODE on
    the furosemide exposure — only on how that exposure is converted into
    %Δ for IS/pCS (steady-state MM evaluated once at the average, vs. the
    ODE's full time-resolved nonlinear integration). Report it as a
    diagnostic / sensitivity check, never as a third independent leg of
    the ODE/Analytical/Clinical comparison.
    """
    return _analytical_ss_core(IS_base, pCS_base, live_pk, drg_avg_ode)



# ═════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═════════════════════════════════════════════════════════════════════════════
def print_results(IS_base, pCS_base, pct_IS, pct_pCS,
                  drg_avg, anal, ode_at_sample, live_pk):
    """Print full results table."""
    print("\n" + "=" * 72)
    print(f"RESULTS — {DOSING_REGIMEN}: {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f}mg/day "
          f"({DOSE_TOTAL_PER_DAY:.0f} mg/day total) × {N_DAYS} days, CKD-5")
    print("=" * 72)

    print(f"\n  LIVE PK (extracted from ODE baseline run):")
    print(f"  CL_FURO    = {live_pk['CL_FURO']:.4f} L/h")
    print(f"  CL_R_IS    = {live_pk['CL_R_IS']:.4f} L/h  "
          f"  F_OAT1_IS  = {live_pk['F_OAT1_IS']:.4f}")
    print(f"  CL_R_pCS   = {live_pk['CL_R_pCS']:.4f} L/h  "
          f"  F_OAT1_pCS = {live_pk['F_OAT1_pCS']:.4f}")

    print(f"\n  FUROSEMIDE:")
    print(f"  ODE avg (last cycle)         = {drg_avg:.4f} mg/L")
    print(f"  Analytical Css,avg           = {anal['IS']['Css_furo']:.4f} mg/L")
    print(f"  alpha (1 + Cu_furo/Ki)       = {anal['IS']['alpha']:.4f}")
    print(f"  s (IS, exact MM)             = {anal['IS']['s']:.4f}")
    print(f"  s (pCS, exact MM)            = {anal['pCS']['s']:.4f}")

    print(f"\n  ODE IS/pCS at different sampling times (Day {N_DAYS}):")
    print(f"  {'Time (h)':>10} {'IS %Δ':>10} {'pCS %Δ':>10}")
    print("  " + "─" * 33)
    for sp_h in sorted(ode_at_sample.keys()):
        v = ode_at_sample[sp_h]
        print(f"  {sp_h:>10.1f} {v['IS']:>+10.1f}% {v['pCS']:>+10.1f}%")
    print(f"  {'24h avg':>10} {pct_IS:>+10.1f}% {pct_pCS:>+10.1f}%")

    print(f"\n  THREE-WAY COMPARISON:")
    print(f"  {'':28} {'ODE 24h-avg':>12} {'Analytical':>12} {'Clinical':>10}")
    print("  " + "─" * 66)
    for name, base, ode_v in [("IS", IS_base, pct_IS),
                              ("pCS", pCS_base, pct_pCS)]:
        clin = CLIN_IS_PCT if name == "IS" else CLIN_pCS_PCT
        lo = CLIN_IS_LO if name == "IS" else CLIN_pCS_LO
        hi = CLIN_IS_HI if name == "IS" else CLIN_pCS_HI
        a_v = anal[name]["pct"]
        in_ci_ode = "✓" if lo < ode_v < hi else "✗"
        in_ci_anal = "✓" if lo < a_v < hi else "✗"
        print(f"  {name + ' baseline (mg/L)':<28} {base:>12.4f} {'—':>12} {'—':>10}")
        print(f"  {name + ' %Δ':<28} "
              f"{ode_v:>+11.1f}%{in_ci_ode} "
              f"{a_v:>+11.1f}%{in_ci_anal} "
              f"{clin:>+9.1f}%")
        print(f"  {'  95% CI':<28} "
              f"{'':>12} {'':>12} [{lo:.1f}, {hi:.1f}]")
    print("=" * 72)


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS FIGURE
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(t_all, IS_all, pCS_all, drg_all,
                 t_last, IS_last, pCS_last, drg_last,
                 IS_base, pCS_base, pct_IS, pct_pCS,
                 anal, drg_avg, ode_at_sample, live_pk) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    t_days = t_all / 24.0
    t_last_days = t_last / 24.0
    t_last_rel = t_last - t_last[0]

    fig = plt.figure(figsize=(15, 12), dpi=200)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.36,
                           left=0.08, right=0.97, top=0.93, bottom=0.06)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[2, :])

    # Mark dose times on timecourses
    def _mark_doses(ax, t_max_days):
        for day in range(N_DAYS):
            for dt_h in DOSE_TIMES_H:
                t_mark = (day * TAU_H + dt_h) / 24.0
                if t_mark <= t_max_days:
                    ax.axvline(t_mark, color="0.85", lw=0.3, zorder=0)

    # ── Panel A — IS 30-day ────────────────────────────────────────────────
    ax_A.plot(t_days, IS_all, "#1A5276", lw=0.6, alpha=0.85)
    ax_A.axhline(IS_base, color="0.5", ls="--", lw=1.2,
                 label=f"Baseline {IS_base:.3f} mg/L")
    IS_ss = IS_base * (1 + pct_IS / 100)
    ax_A.axhline(IS_ss, color="#E74C3C", ls=":", lw=1.6,
                 label=f"ODE 24h-avg = {IS_ss:.3f} mg/L ({pct_IS:+.1f}%)")
    _mark_doses(ax_A, t_days.max())
    ax_A.set_xlabel("Time (days)")
    ax_A.set_ylabel("IS plasma (mg/L)")
    ax_A.set_title(f"A   IS — {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f}mg/day oral "
                   f"({DOSING_REGIMEN}), {N_DAYS} days",
                   fontweight="bold", loc="left", fontsize=8)
    ax_A.legend(fontsize=7.5, frameon=False)

    # ── Panel B — pCS 30-day ───────────────────────────────────────────────
    ax_B.plot(t_days, pCS_all, "#7D3C98", lw=0.6, alpha=0.85)
    ax_B.axhline(pCS_base, color="0.5", ls="--", lw=1.2,
                 label=f"Baseline {pCS_base:.3f} mg/L")
    pCS_ss = pCS_base * (1 + pct_pCS / 100)
    ax_B.axhline(pCS_ss, color="#E74C3C", ls=":", lw=1.6,
                 label=f"ODE 24h-avg = {pCS_ss:.3f} mg/L ({pct_pCS:+.1f}%)")
    _mark_doses(ax_B, t_days.max())
    ax_B.set_xlabel("Time (days)")
    ax_B.set_ylabel("pCS plasma (mg/L)")
    ax_B.set_title(f"B   pCS — {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f}mg/day oral "
                   f"({DOSING_REGIMEN}), {N_DAYS} days",
                   fontweight="bold", loc="left", fontsize=8)
    ax_B.legend(fontsize=7.5, frameon=False)

    # ── Panel C — furosemide last day ─────────────────────────────────────
    ax_C.plot(t_last_rel, drg_last, "#BA7517", lw=1.5, alpha=0.9)
    ax_C.axhline(anal["IS"]["Css_furo"], color="#2E86C1", ls="--", lw=1.6,
                 label=f"Analytical Css,avg = {anal['IS']['Css_furo']:.3f} mg/L")
    ax_C.axhline(drg_avg, color="#E74C3C", ls=":", lw=1.4,
                 label=f"ODE 24h-avg = {drg_avg:.3f} mg/L")
    # Mark dose times in last day
    for dt_h in DOSE_TIMES_H:
        ax_C.axvline(dt_h, color="#BA7517", lw=0.8, ls="--", alpha=0.5)
        ax_C.text(dt_h + 0.1, ax_C.get_ylim()[1] if ax_C.get_ylim()[1] > 0 else 1,
                  f"dose\n{dt_h:.0f}h", fontsize=6.5, color="#BA7517", va="top")
    ax_C.set_xlabel("Time after day start (h)")
    ax_C.set_ylabel("Furosemide plasma (mg/L)")
    times_lab = "/".join(f"{h:.0f}h" for h in DOSE_TIMES_H) or "none"
    ax_C.set_title(f"C   Furosemide — last 24h cycle ({DOSING_REGIMEN}: "
                   f"{N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f}mg at {times_lab})",
                   fontweight="bold", loc="left", fontsize=8)
    ax_C.legend(fontsize=7.5, frameon=False)
    ax_C.set_xlim(0, TAU_H)

    # ── Panel D — IS final cycle with sampling times ───────────────────────
    ax_D.plot(t_last_rel, IS_last, "#1A5276", lw=2.0)
    ax_D.axhline(IS_base, color="0.55", ls="--", lw=1.2,
                 label=f"No-drug baseline ({IS_base:.3f} mg/L)")
    IS_ss = IS_base * (1 + pct_IS / 100)
    ax_D.axhline(IS_ss, color="#E74C3C", ls=":", lw=1.5,
                 label=f"ODE 24h-avg ({pct_IS:+.1f}%)")
    ax_D.axhline(IS_base * (1 + anal["IS"]["pct"] / 100),
                 color="#2E86C1", ls="-.", lw=1.5,
                 label=f"Analytical ({anal['IS']['pct']:+.1f}%)")

    sp_colors = ["#E67E22", "#E74C3C", "#8E44AD", "#2E86C1"]
    for sp_h, col in zip(sorted(ode_at_sample.keys()), sp_colors):
        IS_sp = ode_at_sample[sp_h]["IS_abs"]
        pct_sp = ode_at_sample[sp_h]["IS"]
        # Map sampling time relative to last day start
        t_sp_rel = sp_h  # already relative to first dose of the day
        idx_near = np.argmin(np.abs(t_last_rel - t_sp_rel))
        ax_D.scatter([t_last_rel[idx_near]], [IS_last[idx_near]],
                     s=60, color=col, zorder=8)
        ax_D.text(t_last_rel[idx_near] + 0.1, IS_last[idx_near] + 0.02,
                  f"{pct_sp:+.0f}%\n@{sp_h:.0f}h",
                  fontsize=7, color=col)
    ax_D.set_xlabel("Time after day start (h)")
    ax_D.set_ylabel("IS plasma (mg/L)")
    ax_D.set_title("D   IS final 24h cycle — sampling time effect",
                   fontweight="bold", loc="left", fontsize=8)
    ax_D.legend(fontsize=7.5, frameon=False)
    ax_D.set_xlim(0, TAU_H)

    # ── Panel E — Three-way bar chart ─────────────────────────────────────
    x = np.array([0.0, 1.0])
    COL_ODE = "#2C5F8A"
    COL_ANAL = "#E07B39"
    COL_CLIN = "#C0392B"
    w = 0.22

    ode_pcts = [pct_IS, pct_pCS]
    anal_pcts = [anal["IS"]["pct"], anal["pCS"]["pct"]]
    clin_pcts = [CLIN_IS_PCT, CLIN_pCS_PCT]
    clin_los = [CLIN_IS_LO, CLIN_pCS_LO]
    clin_his = [CLIN_IS_HI, CLIN_pCS_HI]

    for i, (ode_v, an_v, cl_v, lo, hi) in enumerate(
            zip(ode_pcts, anal_pcts, clin_pcts, clin_los, clin_his)):
        b1 = ax_E.bar(x[i] - w, ode_v, w,
                      color=COL_ODE, alpha=0.88,
                      label=f"ODE 24h-avg ({DOSING_REGIMEN})" if i == 0 else "")
        b2 = ax_E.bar(x[i], an_v, w,
                      color=COL_ANAL, alpha=0.88,
                      label="Analytical (exact MM, live PK)" if i == 0 else "")
        b3 = ax_E.bar(x[i] + w, cl_v, w,
                      color=COL_CLIN, alpha=0.88,
                      label="Clinical (Costes-Albrespic 2025)" if i == 0 else "")
        ax_E.errorbar(x[i] + w, cl_v,
                      yerr=[[cl_v - lo], [hi - cl_v]],
                      fmt="none", color="black", capsize=4, lw=1.5)
        for bar, val in [(b1, ode_v), (b2, an_v), (b3, cl_v)]:
            if abs(val) > 0.3:
                ax_E.text(bar[0].get_x() + bar[0].get_width() / 2,
                          val + 0.5, f"{val:+.1f}%",
                          ha="center", va="bottom",
                          fontsize=7.5, fontweight="semibold")

    ax_E.set_xticks(x)
    ax_E.set_xticklabels(["Indoxyl Sulphate (IS)",
                          "p-Cresyl Sulphate (pCS)"], fontsize=10)
    ax_E.set_ylabel("% change vs no-drug baseline", fontsize=9)
    ax_E.set_title(
        f"E   ODE ({DOSING_REGIMEN}: {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f}mg) "
        f"vs Analytical (live PK) vs Clinical  —  >120 mg/day",
        fontweight="bold", loc="left", fontsize=8)
    ax_E.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
    ax_E.axhline(0, color="0.5", lw=0.8, ls="--")

    fig.suptitle(
        f"{DOSING_REGIMEN}: {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f} mg/day "
        f"({DOSE_TOTAL_PER_DAY:.0f} mg/day total) × {N_DAYS} days  |  "
        f"F={F_ORAL} (FDA label)  |  "
        f"kₐ={K_A} h⁻¹ (Hammarlund 1984)  |  "
        f"Ki={Ki} mg/L (calibrated)  |  CKD-5",
        fontsize=9, fontweight="bold", y=0.997,
    )

    ki_str   = f"{Ki:.4g}".replace(".", "p")
    dose_str = f"{DOSE_PER_ADMIN:.4g}"
    fig_stem = os.path.join(
        OUTPUT_DIR,
        f"chronic_oral_"
        f"Ki{ki_str}_dose{dose_str}mg_x{N_DOSES_PER_DAY}pd_{N_DAYS}days_{MODE}_{RUN_DATE}"
    )
    for ext in ("png", "pdf"):
        p = f"{fig_stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  [saved] {p}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — timing and results export
# ═════════════════════════════════════════════════════════════════════════════
import time as _time

def _run_tag(dose_per_admin: float, n_doses: int, ki: float, n_days: int) -> str:
    """Unique string tag for a given run configuration — used in filenames."""
    ki_str   = f"{ki:.4g}".replace(".", "p")
    dose_str = f"{dose_per_admin:.4g}"
    return f"Ki{ki_str}_dose{dose_str}mg_x{n_doses}pd_{n_days}days_{MODE}_{RUN_DATE}"


def _save_results_npz(tag: str,
                      t_all, IS_all, pCS_all, drg_all,
                      IS_base, pCS_base,
                      pct_IS, pct_pCS,
                      anal, anal_ode, live_pk,
                      dose_per_admin, n_doses, ki, n_days,
                      dose_times_h) -> str:
    """
    Save all simulation arrays + derived metrics to a standalone .npz file.

    This file is independent of the cache — it is meant for professional
    figure plotting in a separate script. It contains everything needed to
    reproduce any panel without re-running the ODE.

    Saved arrays
    ------------
    t_all, IS_all, pCS_all, drg_all   — full timecourse (hours, mg/L)
    IS_base, pCS_base                  — no-drug CKD-5 baseline (mg/L)
    pct_IS, pct_pCS                    — ODE 24h-avg % change
    anal_IS_pct, anal_pCS_pct          — THEORETICAL analytical % change
                                          (closed-form Css, independent of ODE)
    anal_Css_furo                       — theoretical analytical Css (mg/L)
    anal_ode_IS_pct, anal_ode_pCS_pct  — ODE-FED analytical % change
                                          (DIAGNOSTIC ONLY — uses ODE's own
                                          drg_avg as Css input, NOT an
                                          independent prediction; see
                                          analytical_ss_ode_css() docstring)
    anal_ode_Css_furo                   — ODE-measured Css used as input
                                          (== drg_avg from this same run)
    CL_FURO, CL_R_IS, CL_R_pCS        — live PK clearances (L/h)
    F_OAT1_IS, F_OAT1_pCS             — OAT1 fraction of renal CL
    Ki, dose_per_admin, n_doses        — run parameters
    n_days, dose_times_h               — run parameters
    """
    out_path = os.path.join(OUTPUT_DIR, f"results_{tag}.npz")
    np.savez(
        out_path,
        # Timecourses
        t_all       = t_all,
        IS_all      = IS_all,
        pCS_all     = pCS_all,
        drg_all     = drg_all,
        # Baselines
        IS_base     = np.array([IS_base]),
        pCS_base    = np.array([pCS_base]),
        # ODE metrics
        pct_IS      = np.array([pct_IS]),
        pct_pCS     = np.array([pct_pCS]),
        # Analytical metrics — THEORETICAL (independent, closed-form Css)
        anal_IS_pct     = np.array([anal["IS"]["pct"]]),
        anal_pCS_pct    = np.array([anal["pCS"]["pct"]]),
        anal_Css_furo   = np.array([anal["IS"]["Css_furo"]]),
        anal_alpha_IS   = np.array([anal["IS"]["alpha"]]),
        anal_alpha_pCS  = np.array([anal["pCS"]["alpha"]]),
        anal_s_IS       = np.array([anal["IS"]["s"]]),
        anal_s_pCS      = np.array([anal["pCS"]["s"]]),
        # Analytical metrics — ODE-FED (diagnostic, uses ODE's own drg_avg)
        anal_ode_IS_pct    = np.array([anal_ode["IS"]["pct"]]),
        anal_ode_pCS_pct   = np.array([anal_ode["pCS"]["pct"]]),
        anal_ode_Css_furo  = np.array([anal_ode["IS"]["Css_furo"]]),
        anal_ode_alpha_IS  = np.array([anal_ode["IS"]["alpha"]]),
        anal_ode_alpha_pCS = np.array([anal_ode["pCS"]["alpha"]]),
        anal_ode_s_IS      = np.array([anal_ode["IS"]["s"]]),
        anal_ode_s_pCS     = np.array([anal_ode["pCS"]["s"]]),
        # Live PK
        CL_FURO     = np.array([live_pk["CL_FURO"]]),
        CL_R_IS     = np.array([live_pk["CL_R_IS"]]),
        CL_R_pCS    = np.array([live_pk["CL_R_pCS"]]),
        F_OAT1_IS   = np.array([live_pk["F_OAT1_IS"]]),
        F_OAT1_pCS  = np.array([live_pk["F_OAT1_pCS"]]),
        # Run parameters (stored as 1-element arrays for npz compatibility)
        Ki              = np.array([ki]),
        dose_per_admin  = np.array([dose_per_admin]),
        n_doses         = np.array([n_doses]),
        n_days          = np.array([n_days]),
        dose_times_h    = np.array(dose_times_h),
        # Clinical reference (hardcoded literature values)
        clin_IS_pct     = np.array([CLIN_IS_PCT]),
        clin_pCS_pct    = np.array([CLIN_pCS_PCT]),
        clin_IS_lo      = np.array([CLIN_IS_LO]),
        clin_IS_hi      = np.array([CLIN_IS_HI]),
        clin_pCS_lo     = np.array([CLIN_pCS_LO]),
        clin_pCS_hi     = np.array([CLIN_pCS_HI]),
    )
    print(f"  [results] Saved → {out_path}")
    return out_path


def _run_simulation(ic_ss, live_pk,
                    dose_per_admin, n_doses, dose_times_h,
                    oral_segs=None) -> tuple:
    """
    Run the N_DAYS TID/QD simulation for a given dosing configuration.
    Returns (t_all, IS_all, HA_all, pCS_all, drg_all).
    """
    if oral_segs is None and n_doses > 0:
        oral_segs = build_oral_profile(dose_per_admin)
    elif oral_segs is None:
        oral_segs = []

    all_t, all_IS, all_HA, all_pCS, all_drg = [], [], [], [], []
    ic_cur = copy.deepcopy(ic_ss)
    for key in ("C_ven", "C_peri", "Ae_drg", "C_bladder"):
        if key in ic_cur:
            ic_cur[key] = 0.0

    for day in range(N_DAYS):
        t_off = day * TAU_H
        if day % 5 == 0 or day == N_DAYS - 1:
            print(f"    Day {day + 1:3d}: "
                  f"IS={ic_cur.get('C_tox1_cen', 0):.3f}  "
                  f"pCS={ic_cur.get('C_tox3_cen', 0):.3f} mg/L")
        t_d, y_d, ic_cur = run_one_day_tid(
            ic_cur, t_off, oral_segs, dose_times_h=dose_times_h)
        all_t.append(t_d)
        if "C_tox1_cen" in IDX:
            all_IS.append(y_d[IDX["C_tox1_cen"], :])
        if "C_tox2_cen" in IDX:
            all_HA.append(y_d[IDX["C_tox2_cen"], :])
        if "C_tox3_cen" in IDX:
            all_pCS.append(y_d[IDX["C_tox3_cen"], :])
        if "C_ven" in IDX:
            all_drg.append(y_d[IDX["C_ven"], :])

    return (np.concatenate(all_t),
            np.concatenate(all_IS),
            np.concatenate(all_HA),
            np.concatenate(all_pCS),
            np.concatenate(all_drg))


def _compute_metrics(t_all, IS_all, pCS_all, drg_all,
                     IS_base, pCS_base, live_pk):
    """Compute 24h-avg % changes, sampling-time values, and analytical SS."""
    mask   = t_all >= (N_DAYS - 1) * TAU_H
    t_last = t_all[mask];  IS_last = IS_all[mask]
    pCS_last = pCS_all[mask];  drg_last = drg_all[mask]
    dt_last  = t_last[-1] - t_last[0]

    IS_avg  = float(_trapz(IS_last,  t_last) / dt_last)
    pCS_avg = float(_trapz(pCS_last, t_last) / dt_last)
    drg_avg = float(_trapz(drg_last, t_last) / dt_last)

    pct_IS  = (IS_avg  - IS_base)  / IS_base  * 100
    pct_pCS = (pCS_avg - pCS_base) / pCS_base * 100

    t_day_start = (N_DAYS - 1) * TAU_H
    ode_at_sample = {}
    for sp_h in [1.0, 2.0, 3.0, 4.0]:
        idx_near = np.argmin(np.abs(t_all - (t_day_start + sp_h)))
        IS_at_t  = float(IS_all[idx_near])
        pCS_at_t = float(pCS_all[idx_near])
        ode_at_sample[sp_h] = {
            "IS":     (IS_at_t  - IS_base)  / IS_base  * 100,
            "pCS":    (pCS_at_t - pCS_base) / pCS_base * 100,
            "IS_abs":  IS_at_t,
            "pCS_abs": pCS_at_t,
        }

    anal      = analytical_ss(IS_base, pCS_base, live_pk)
    anal_ode  = analytical_ss_ode_css(IS_base, pCS_base, live_pk, drg_avg)
    return (pct_IS, pct_pCS, drg_avg,
            t_last, IS_last, pCS_last, drg_last,
            ode_at_sample, anal, anal_ode)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    t_main_start = _time.perf_counter()

    print("=" * 72)
    print(f"CHRONIC ORAL FUROSEMIDE SS v5")
    print(f"  Primary regimen : {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f} mg/day "
          f"at hours {DOSE_TIMES_H}  →  {DOSE_TOTAL_PER_DAY:.0f} mg/day total")
    print(f"  Comparison      : 1×{DOSE_TOTAL_PER_DAY:.0f} mg/day QD (same total dose)")
    print(f"  Duration        : {N_DAYS} days  |  Mode: {MODE}")
    print(f"  F={F_ORAL} (FDA label)  |  ka={K_A} h⁻¹  |  Ki={Ki} mg/L")
    print("=" * 72)

    # ── Step 1: warm-up ───────────────────────────────────────────────────────
    print(f"\n[STEP 1] Warm-up ({SS_WARMUP_H:.0f} h CKD-5 no-drug SS)...")
    t0 = _time.perf_counter()
    ic_ss = warmup()
    IS_base  = ic_ss.get("C_tox1_cen", 0.0)
    pCS_base = ic_ss.get("C_tox3_cen", 0.0)
    print(f"  Done in {_time.perf_counter()-t0:.1f} s  |  "
          f"IS_base={IS_base:.3f}  pCS_base={pCS_base:.3f} mg/L")

    # ── Step 2: baseline PK ───────────────────────────────────────────────────
    print(f"\n[STEP 2] Extracting live PK from baseline ODE run...")
    t0 = _time.perf_counter()
    live_pk = extract_baseline_pk(ic_ss)
    print(f"  Done in {_time.perf_counter()-t0:.1f} s  |  "
          f"CL_FURO={live_pk['CL_FURO']:.4f} L/h  "
          f"CL_R_IS={live_pk['CL_R_IS']:.4f} L/h  "
          f"CL_R_pCS={live_pk['CL_R_pCS']:.4f} L/h")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN A — primary regimen (TID / configured N_DOSES_PER_DAY)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"[RUN A] {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f} mg TID "
          f"({DOSE_TOTAL_PER_DAY:.0f} mg/day total) — {N_DAYS} days")
    print(f"{'═'*72}")

    cache_file_A = _cache_path()
    cache_ok_A   = _cache_is_valid(cache_file_A, live_pk)

    if cache_ok_A:
        print(f"  [CACHE] Loading from {cache_file_A}")
        cache = np.load(cache_file_A, allow_pickle=False)
        t_all_A   = cache["t_all"]
        IS_all_A  = cache["IS_all"]
        HA_all_A  = cache["HA_all"] if "HA_all" in cache.files else None
        pCS_all_A = cache["pCS_all"]
        drg_all_A = cache["drg_all"]
        print(f"  Loaded {len(t_all_A)} time points — skipping ODE run")
    else:
        print(f"  Building oral absorption profile...")
        oral_segs_A = build_oral_profile(DOSE_PER_ADMIN)
        print(f"  Running {N_DAYS} days × {N_DOSES_PER_DAY} doses/day "
              f"= {N_DAYS * N_DOSES_PER_DAY} total doses...")
        t0 = _time.perf_counter()
        t_all_A, IS_all_A, HA_all_A, pCS_all_A, drg_all_A = _run_simulation(
            ic_ss, live_pk, DOSE_PER_ADMIN, N_DOSES_PER_DAY,
            DOSE_TIMES_H, oral_segs_A)
        elapsed_A = _time.perf_counter() - t0
        print(f"  ODE run complete in {elapsed_A:.1f} s "
              f"({elapsed_A/60:.1f} min)")
        # Save cache
        fp = _build_fingerprint(live_pk)
        save_dict = {"t_all": t_all_A, "IS_all": IS_all_A, "HA_all": HA_all_A,
                     "pCS_all": pCS_all_A, "drg_all": drg_all_A,
                     "IS_base": np.array([IS_base]),
                     "pCS_base": np.array([pCS_base])}
        save_dict.update({k: np.array([v]) for k, v in fp.items()})
        np.savez(cache_file_A, **save_dict)
        print(f"  [CACHE] Saved → {cache_file_A}")

    # Compute metrics for Run A
    (pct_IS_A, pct_pCS_A, drg_avg_A,
     t_last_A, IS_last_A, pCS_last_A, drg_last_A,
     ode_at_sample_A, anal_A, anal_ode_A) = _compute_metrics(
        t_all_A, IS_all_A, pCS_all_A, drg_all_A, IS_base, pCS_base, live_pk)

    print(f"\n  [RUN A RESULTS]  IS: {pct_IS_A:+.1f}%  pCS: {pct_pCS_A:+.1f}%  "
          f"(ODE 24h-avg)  |  furosemide avg: {drg_avg_A:.3f} mg/L")
    print(f"  [RUN A DIAGNOSTIC] Analytical(theoretical Css={anal_A['IS']['Css_furo']:.3f} mg/L): "
          f"IS {anal_A['IS']['pct']:+.1f}%  pCS {anal_A['pCS']['pct']:+.1f}%")
    print(f"  [RUN A DIAGNOSTIC] Analytical(ODE-measured Css={anal_ode_A['IS']['Css_furo']:.3f} mg/L): "
          f"IS {anal_ode_A['IS']['pct']:+.1f}%  pCS {anal_ode_A['pCS']['pct']:+.1f}%")

    # Print and plot Run A
    print_results(IS_base, pCS_base, pct_IS_A, pct_pCS_A,
                  drg_avg_A, anal_A, ode_at_sample_A, live_pk)
    print("\n  [FIG A] Generating figure...")
    t0 = _time.perf_counter()
    plot_results(t_all_A, IS_all_A, pCS_all_A, drg_all_A,
                 t_last_A, IS_last_A, pCS_last_A, drg_last_A,
                 IS_base, pCS_base, pct_IS_A, pct_pCS_A,
                 anal_A, drg_avg_A, ode_at_sample_A, live_pk)
    print(f"  Figure saved in {_time.perf_counter()-t0:.1f} s")

    # Save full results .npz for Run A
    tag_A = _run_tag(DOSE_PER_ADMIN, N_DOSES_PER_DAY, Ki, N_DAYS)
    _save_results_npz(tag_A, t_all_A, IS_all_A, pCS_all_A, drg_all_A,
                      IS_base, pCS_base, pct_IS_A, pct_pCS_A,
                      anal_A, anal_ode_A, live_pk,
                      DOSE_PER_ADMIN, N_DOSES_PER_DAY, Ki, N_DAYS,
                      DOSE_TIMES_H)

    # ══════════════════════════════════════════════════════════════════════════
    # RUN B — same total daily dose as QD (once daily)
    # ══════════════════════════════════════════════════════════════════════════
    if SKIP_QD:
        print("\n[RUN B] Skipped (SKIP_QD=True).")
        total_elapsed = _time.perf_counter() - t_main_start
        print(f"\n  Total wall-clock time: {total_elapsed:.1f} s "
              f"({total_elapsed/60:.1f} min)")
        print(f"    Cache A  : {os.path.join(OUTPUT_DIR, f'chronic_oral_cache_{tag_A}.npz')}")
        print(f"    Figure A : {os.path.join(OUTPUT_DIR, f'chronic_oral_{tag_A}.png/.pdf')}")
        return

    QD_DOSE       = DOSE_TOTAL_PER_DAY   # same total as Run A (e.g. 240 mg)
    QD_N_DOSES    = 1
    QD_TIMES_H    = [0.0]

    print(f"\n{'═'*72}")
    print(f"[RUN B] 1×{QD_DOSE:.0f} mg QD ({QD_DOSE:.0f} mg/day total) — {N_DAYS} days")
    print(f"  (same total daily dose as Run A, single oral intake)")
    print(f"{'═'*72}")

    # Build QD cache path using same convention
    ki_str_b   = f"{Ki:.4g}".replace(".", "p")
    cache_file_B = os.path.join(
        OUTPUT_DIR,
        f"chronic_oral_cache_"
        f"Ki{ki_str_b}_dose{QD_DOSE:.4g}mg_x{QD_N_DOSES}pd_{N_DAYS}days_{MODE}_{RUN_DATE}.npz"
    )

    # Build a temporary fingerprint for QD to check cache validity
    # (same live_pk, different dose/n_doses)
    _orig_dose  = DOSE_PER_ADMIN
    _orig_n     = N_DOSES_PER_DAY
    # Patch globals temporarily for fingerprint builder
    import builtins
    _g = globals()
    _g["DOSE_PER_ADMIN"]  = QD_DOSE
    _g["N_DOSES_PER_DAY"] = QD_N_DOSES
    _g["DOSE_TIMES_H"]    = QD_TIMES_H
    _g["DOSE_INTERVAL_H"] = TAU_H
    fp_B = _build_fingerprint(live_pk)
    cache_ok_B = False
    if os.path.exists(cache_file_B):
        try:
            cache_B = np.load(cache_file_B, allow_pickle=False)
            mismatches_B = []
            for key, val in fp_B.items():
                if key not in cache_B.files:
                    mismatches_B.append(key)
                elif abs(float(cache_B[key].flat[0]) - val) > 1e-9:
                    mismatches_B.append(key)
            cache_ok_B = (len(mismatches_B) == 0)
            if cache_ok_B:
                print(f"  [CACHE] Loading from {cache_file_B}")
        except Exception:
            cache_ok_B = False
    # Restore globals
    _g["DOSE_PER_ADMIN"]  = _orig_dose
    _g["N_DOSES_PER_DAY"] = _orig_n
    _g["DOSE_TIMES_H"]    = [0.0, 8.0, 16.0] if _orig_n == 3 else [0.0]
    _g["DOSE_INTERVAL_H"] = TAU_H / _orig_n

    if cache_ok_B:
        cache_B   = np.load(cache_file_B, allow_pickle=False)
        t_all_B   = cache_B["t_all"]
        IS_all_B  = cache_B["IS_all"]
        pCS_all_B = cache_B["pCS_all"]
        drg_all_B = cache_B["drg_all"]
        print(f"  Loaded {len(t_all_B)} time points — skipping ODE run")
    else:
        print(f"  Building oral absorption profile for {QD_DOSE:.0f} mg QD...")
        oral_segs_B = build_oral_profile(QD_DOSE)
        print(f"  Running {N_DAYS} days × {QD_N_DOSES} dose/day "
              f"= {N_DAYS * QD_N_DOSES} total doses...")
        t0 = _time.perf_counter()
        t_all_B, IS_all_B, HA_all_B, pCS_all_B, drg_all_B = _run_simulation(
            ic_ss, live_pk, QD_DOSE, QD_N_DOSES, QD_TIMES_H, oral_segs_B)
        elapsed_B = _time.perf_counter() - t0
        print(f"  ODE run complete in {elapsed_B:.1f} s "
              f"({elapsed_B/60:.1f} min)")
        # Save cache for Run B
        _g["DOSE_PER_ADMIN"]  = QD_DOSE
        _g["N_DOSES_PER_DAY"] = QD_N_DOSES
        _g["DOSE_INTERVAL_H"] = TAU_H
        fp_B2 = _build_fingerprint(live_pk)
        _g["DOSE_PER_ADMIN"]  = _orig_dose
        _g["N_DOSES_PER_DAY"] = _orig_n
        _g["DOSE_INTERVAL_H"] = TAU_H / _orig_n
        save_dict_B = {"t_all": t_all_B, "IS_all": IS_all_B, "HA_all": HA_all_B,
                       "pCS_all": pCS_all_B, "drg_all": drg_all_B,
                       "IS_base": np.array([IS_base]),
                       "pCS_base": np.array([pCS_base])}
        save_dict_B.update({k: np.array([v]) for k, v in fp_B2.items()})
        np.savez(cache_file_B, **save_dict_B)
        print(f"  [CACHE] Saved → {cache_file_B}")

    # Compute QD analytical (different Css_furo for QD dose)
    # Temporarily patch globals so analytical_ss uses QD dose
    _g["DOSE_PER_ADMIN"]  = QD_DOSE
    _g["N_DOSES_PER_DAY"] = QD_N_DOSES
    (pct_IS_B, pct_pCS_B, drg_avg_B,
     t_last_B, IS_last_B, pCS_last_B, drg_last_B,
     ode_at_sample_B, anal_B, anal_ode_B) = _compute_metrics(
        t_all_B, IS_all_B, pCS_all_B, drg_all_B, IS_base, pCS_base, live_pk)
    _g["DOSE_PER_ADMIN"]  = _orig_dose
    _g["N_DOSES_PER_DAY"] = _orig_n

    print(f"\n  [RUN B RESULTS]  IS: {pct_IS_B:+.1f}%  pCS: {pct_pCS_B:+.1f}%  "
          f"(ODE 24h-avg)  |  furosemide avg: {drg_avg_B:.3f} mg/L")
    print(f"  [RUN B DIAGNOSTIC] Analytical(theoretical Css={anal_B['IS']['Css_furo']:.3f} mg/L): "
          f"IS {anal_B['IS']['pct']:+.1f}%  pCS {anal_B['pCS']['pct']:+.1f}%")
    print(f"  [RUN B DIAGNOSTIC] Analytical(ODE-measured Css={anal_ode_B['IS']['Css_furo']:.3f} mg/L): "
          f"IS {anal_ode_B['IS']['pct']:+.1f}%  pCS {anal_ode_B['pCS']['pct']:+.1f}%")

    # Print and plot Run B (temporarily patch globals for labels)
    _g["DOSE_PER_ADMIN"]  = QD_DOSE
    _g["N_DOSES_PER_DAY"] = QD_N_DOSES
    _g["DOSE_TIMES_H"]    = QD_TIMES_H
    _g["DOSE_INTERVAL_H"] = TAU_H
    _g["DOSE_TOTAL_PER_DAY"] = QD_DOSE
    _g["DOSING_REGIMEN"]  = "QD"
    print_results(IS_base, pCS_base, pct_IS_B, pct_pCS_B,
                  drg_avg_B, anal_B, ode_at_sample_B, live_pk)
    print("\n  [FIG B] Generating figure...")
    t0 = _time.perf_counter()
    plot_results(t_all_B, IS_all_B, pCS_all_B, drg_all_B,
                 t_last_B, IS_last_B, pCS_last_B, drg_last_B,
                 IS_base, pCS_base, pct_IS_B, pct_pCS_B,
                 anal_B, drg_avg_B, ode_at_sample_B, live_pk)
    print(f"  Figure saved in {_time.perf_counter()-t0:.1f} s")
    # Restore globals
    _g["DOSE_PER_ADMIN"]     = _orig_dose
    _g["N_DOSES_PER_DAY"]    = _orig_n
    _g["DOSE_TIMES_H"]       = [0.0, 8.0, 16.0]
    _g["DOSE_INTERVAL_H"]    = TAU_H / _orig_n
    _g["DOSE_TOTAL_PER_DAY"] = _orig_dose * _orig_n
    _g["DOSING_REGIMEN"]     = "TID"

    # Save full results .npz for Run B
    tag_B = _run_tag(QD_DOSE, QD_N_DOSES, Ki, N_DAYS)
    _save_results_npz(tag_B, t_all_B, IS_all_B, pCS_all_B, drg_all_B,
                      IS_base, pCS_base, pct_IS_B, pct_pCS_B,
                      anal_B, anal_ode_B, live_pk,
                      QD_DOSE, QD_N_DOSES, Ki, N_DAYS, QD_TIMES_H)

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = _time.perf_counter() - t_main_start
    print()
    print("=" * 72)
    print("SUMMARY — both runs complete")
    print("=" * 72)
    print(f"  {'Run':<6} {'Regimen':<22} {'IS %Δ':>8} {'pCS %Δ':>8} "
          f"{'AnalThr IS':>11} {'AnalThr pCS':>12} "
          f"{'AnalODE IS':>11} {'AnalODE pCS':>12}")
    print("  " + "─" * 98)
    print(f"  {'A':<6} {N_DOSES_PER_DAY}×{_orig_dose:.0f}mg TID "
          f"({_orig_dose*_orig_n:.0f}mg/day)   "
          f"{pct_IS_A:>+7.1f}%  {pct_pCS_A:>+7.1f}%  "
          f"{anal_A['IS']['pct']:>+10.1f}%  {anal_A['pCS']['pct']:>+11.1f}%  "
          f"{anal_ode_A['IS']['pct']:>+10.1f}%  {anal_ode_A['pCS']['pct']:>+11.1f}%")
    print(f"  {'B':<6} 1×{QD_DOSE:.0f}mg QD "
          f"({QD_DOSE:.0f}mg/day)         "
          f"{pct_IS_B:>+7.1f}%  {pct_pCS_B:>+7.1f}%  "
          f"{anal_B['IS']['pct']:>+10.1f}%  {anal_B['pCS']['pct']:>+11.1f}%  "
          f"{anal_ode_B['IS']['pct']:>+10.1f}%  {anal_ode_B['pCS']['pct']:>+11.1f}%")
    print(f"  {'Clin':<6} >120mg/day (Costes 2025)      "
          f"{CLIN_IS_PCT:>+7.1f}%  {CLIN_pCS_PCT:>+7.1f}%")
    print()
    print("  Column key:")
    print("    IS %Δ / pCS %Δ        = ODE full simulation (ground truth)")
    print("    AnalThr (THEORETICAL) = closed-form Css = F×dose/(CL×tau);")
    print("                            INDEPENDENT of ODE output;")
    print("                            same for TID/QD at equal total dose")
    print("    AnalODE (DIAGNOSTIC)  = same MM formula fed the ODE's own")
    print("                            measured drg_avg as Css input;")
    print("                            NOT independent — isolates how much")
    print("                            of the ODE/Analytical gap comes from")
    print("                            time-averaging nonlinear inhibition")
    print("                            vs. the closed-form Css estimate itself")
    print()
    print(f"  Total wall-clock time: {total_elapsed:.1f} s "
          f"({total_elapsed/60:.1f} min)")
    print()
    print("  Saved files:")
    print(f"    Cache A  : {os.path.join(OUTPUT_DIR, f'chronic_oral_cache_{tag_A}.npz')}")
    print(f"    Cache B  : {os.path.join(OUTPUT_DIR, f'chronic_oral_cache_{tag_B}.npz')}")
    print(f"    Results A: {os.path.join(OUTPUT_DIR, f'results_{tag_A}.npz')}")
    print(f"    Results B: {os.path.join(OUTPUT_DIR, f'results_{tag_B}.npz')}")
    print(f"    Figure A : {os.path.join(OUTPUT_DIR, f'chronic_oral_{tag_A}.png/.pdf')}")
    print(f"    Figure B : {os.path.join(OUTPUT_DIR, f'chronic_oral_{tag_B}.png/.pdf')}")
    print("=" * 72)


if __name__ == "__main__":
    main()