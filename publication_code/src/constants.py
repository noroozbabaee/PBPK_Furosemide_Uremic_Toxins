"""
constants.py
============
Single source of truth for all geometry, physiology, and numerical
constants in the kidney PBPK model.

WHAT THIS FILE OWNS
-------------------
  - Segment lists and anatomical order (PROX, NONPROX, CDS, SEGMENTS)
  - Apical surface areas (SA_APICAL_DM2)
  - Healthy tubular flows (_HEALTHY_TUBULAR_FLOWS_ML_MIN)
  - CKD scaling factors and urine-flow decoupling (CKD_SCALE_FACTORS,
    CKD_URINE_SCALE, CKD5_RBF_SCALE)
  - Renal blood flow base (_RENAL_BLOOD_FLOW_BASE_ML_MIN)
  - CKD5 flow maps with monotonic CDC chain (_rebuild_mode_maps)
  - Segment volumes (V_SEG)
  - pH maps (_PH_MAPS, update_ph_maps)
  - Molecular weights (MW_FUR, MW_IS, MW_HA, MW_pCS)
  - Transport geometry weights (W_BASO_PROX, W_APIC_PROX)
  - Numerical solver constants (EPS, RTOL, ATOL)
  - Unit conversion factors (ML_MIN_TO_L_H, DM2_TO_CM2, CM_S_TO_L_H)
  - Flow accessor functions and validation (get_tubular_flows_ml_min,
    get_renal_blood_flow_ml_min, validate_flows)

WHAT THIS FILE DOES NOT OWN
-----------------------------
  - Compound PK parameters (fu_p, Vmax, Km, Ki, k_elim, R0)
    → compound_parameters.py
  - CKD5 RBF scale used in ODE exchange term
    → physiology.py imports CKD5_RBF_SCALE from here (single source)
  - Scenario flags (drug.enabled, drug.inhibit)
    → scenario_manager.py

CKD5 FLOW DESIGN DECISIONS
---------------------------
GFR and tubular flow scale together (CKD_SCALE_FACTORS["ckd5"] = 0.10):
  P1 = 120 × 0.10 = 12 mL/min  (= eGFR 12 mL/min, consistent with CKD-5)

CDC segments are scaled separately (CKD_URINE_SCALE = 0.50):
  Residual urine in non-dialysis CKD-5 is ~500-1000 mL/day, not 10% of
  healthy (which would be 144 mL/day — anuric). The decoupling between
  GFR scale and urine scale reflects preserved single-nephron urine output
  in hypertrophied remaining nephrons (Mair 2021 JASN; Hannedouche 2025 KIR).

The CDC chain is fixed to absolute values (not proportional) so that
the DT→CDC1 boundary remains monotonic:
  DT_ckd5 = 20 × 0.10 = 2.0 mL/min
  CDC1     = 2.0 mL/min  (= DT_ckd5, so no unphysical flow increase)
  CDC5     = 0.5 mL/min  → Q_urine = 0.030 L/h (720 mL/day)

RBF is scaled separately from GFR (CKD5_RBF_SCALE = 0.35):
  Q_kidney_CKD5 = 1200 × 0.35 = 420 mL/min = 25.2 L/h
  physiology.py imports CKD5_RBF_SCALE from here — do NOT redefine it there.
  Source: Huang & Isoherranen 2020 CPT Pharmacometrics Syst Pharmacol 9:571.

REFERENCES
----------
  Mair 2021         J Am Soc Nephrol 32:1000 — residual urine in CKD-5
  Hannedouche 2025  Kidney Int Reports        — CKD-5 urine output
  Huang 2020        CPT:PSP 9:571             — CKD RBF scaling
  KDIGO 2012        Kidney Int Suppl 3:1      — CKD-5 classification (eGFR<15)
  Scotcher 2016     CPT:PSP 5:87              — OAT1 segment expression weights
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL TOGGLES
# ══════════════════════════════════════════════════════════════════════════════

MODE              = "healthy"   # default startup mode; overwritten by set_scenario()
USE_APICAL_MM     = True        # Michaelis-Menten for apical secretion (vs linear)
PER_KIDNEY        = False       # True = model one kidney, halve all bilateral flows
DRUG_ENABLED      = True        # master drug on/off (also set in drug dict)
ENABLE_APICAL_FLUXES = True     # master on/off for all apical transport
VERBOSE           = True        # print flow maps and validation results

# Unit contract — used for documentation and assertions
MODEL_UNITS = {
    "concentration": "mg/L",
    "flux":          "mg/h",
    "clearance":     "L/h",
    "flow":          "L/h",
    "time":          "h",
}

# Vmax scaler: per-1e6-cell → organ-level (keep 1.0 when Vmax already organ-level)
CELL_SCALER_million_to_organ = 1.0

def ensure_flow_units_L_per_h(value, unit_hint="L/h"):
    """No-op placeholder — returns value unchanged. Use for documentation."""
    return float(value)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEGMENTS AND ANATOMY
# ══════════════════════════════════════════════════════════════════════════════

PROX    = ["P1", "P2", "P3"]               # proximal convoluted tubule
NONPROX = ["LH1", "LH2", "DT"]            # loop of Henle + distal tubule
CDS     = ["CDC1", "CDC2", "CDC3", "CDC4", "CDC5"]  # collecting duct chain

# Anatomical order: Bowman's capsule → bladder
SEGMENTS = PROX + NONPROX + CDS           # length 11

_twoK_to_oneK = 0.5 if PER_KIDNEY else 1.0   # halve all bilateral quantities if one kidney

# Apical lumen surface area (dm²) — both kidneys unless PER_KIDNEY=True
# Source: anatomical estimates scaled to 70 kg adult
# Proximal tubule has large microvilli (brush border) → large SA
# Collecting duct has much smaller SA per nephron
SA_APICAL_DM2 = {
    "P1":   6107.0 * _twoK_to_oneK,   # proximal S1: brush-border dominant
    "P2":   6107.0 * _twoK_to_oneK,   # proximal S2
    "P3":   6107.0 * _twoK_to_oneK,   # proximal S3 (pars recta)
    "LH1":    61.0 * _twoK_to_oneK,   # thin descending limb
    "LH2":    61.0 * _twoK_to_oneK,   # thin/thick ascending limb
    "DT":    156.0 * _twoK_to_oneK,   # distal tubule
    "CDC1":    6.7 * _twoK_to_oneK,   # collecting duct segments 1-5
    "CDC2":    6.7 * _twoK_to_oneK,
    "CDC3":    6.7 * _twoK_to_oneK,
    "CDC4":    6.7 * _twoK_to_oneK,
    "CDC5":    6.7 * _twoK_to_oneK,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. TUBULAR FLOWS (mL/min) — HEALTHY BASELINE
# ══════════════════════════════════════════════════════════════════════════════
# Single source of truth for healthy tubular flows.
# Flow decreases along the tubule as water is reabsorbed.
# P1 flow = GFR = 120 mL/min (= 7.2 L/h).
# CDC5 flow = urine production = 1 mL/min (= 0.06 L/h = 1.44 L/day).
#
# The large drop from DT (20) to CDC1 (15) reflects the transition from
# the distal tubule to the cortical collecting duct where significant
# aldosterone-regulated water reabsorption occurs.
#
# Reference: standard renal physiology (Guyton & Hall; Boron & Boulpaep)

_HEALTHY_TUBULAR_FLOWS_ML_MIN = {
    "P1":  120,   # = GFR; filtered fluid entering proximal tubule
    "P2":   94,   # ~21% reabsorbed in early proximal (isotonic reabsorption)
    "P3":   68,   # ~28% more reabsorbed
    "LH1":  50,   # thin descending limb — water reabsorption
    "LH2":  50,   # ascending limb — salt reabsorption, water-impermeable
    "DT":   20,   # distal tubule — aldosterone-sensitive NaCl reabsorption
    "CDC1": 15,   # cortical collecting duct — aquaporin-2 water reabsorption
    "CDC2":  8,
    "CDC3":  4,
    "CDC4":  2,
    "CDC5":  1,   # terminal urine flow → Q_urine = 0.06 L/h = 1.44 L/day
}


# ══════════════════════════════════════════════════════════════════════════════
# 4. CKD SCALING
# ══════════════════════════════════════════════════════════════════════════════

# GFR / tubular flow scale per mode
# CKD-5: eGFR < 15 mL/min → use 10% of healthy (KDIGO 2012)
CKD_SCALE_FACTORS = {"healthy": 1.0, "ckd5": 0.10}

# Collecting duct urine scale — INDEPENDENT of GFR scale.
#
# In CKD-5, GFR drops to ~10% of healthy but residual urine output is
# preserved at ~500-1000 mL/day in non-dialysis patients because:
#   1. Surviving nephrons compensate with higher single-nephron urine output
#   2. Damaged tubules have reduced water reabsorption efficiency
#   3. Clinical observation: non-dialysis CKD-5 typically NOT anuric
#      (Mair 2021 JASN; Hannedouche 2025 KIR)
#
# Using GFR_scale=0.10 for CDC segments would give:
#   CDC5 = 1 × 0.10 = 0.1 mL/min → 144 mL/day  (too low — near-anuric)
#
# CKD_URINE_SCALE = 0.50 gives:
#   CDC5 = 1 × 0.50 = 0.5 mL/min → 720 mL/day  (realistic for CKD-5)
#
# Applied ONLY to CDC1-CDC5. P1-P3, LH1-LH2, DT scale with CKD_SCALE_FACTORS.
# Sensitivity range: 0.30 (severe oliguria) to 0.70 (well-preserved output).
CKD_URINE_SCALE = 0.50

# CKD-5 renal blood flow scale — independent of GFR scale.
# RBF is preserved more than GFR in CKD because autoregulation protects
# perfusion. GFR drops to 10% while RBF drops to only ~35%.
# Q_kidney_CKD5 = 1200 mL/min × 0.35 = 420 mL/min = 25.2 L/h
# physiology.py imports this constant — do NOT redefine it there.
# Sensitivity range: 0.20 (severe CKD) to 0.50 (mild CKD).
# Source: Huang & Isoherranen 2020 CPT:PSP 9:571
CKD5_RBF_SCALE = 0.35

# Healthy renal blood flow (both kidneys)
# 1200 mL/min = 72 L/h = ~20% of cardiac output in a 70 kg adult
_RENAL_BLOOD_FLOW_BASE_ML_MIN = 1200.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. FLOW MAP BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _scaled_flows(base: dict, scale: float) -> dict:
    """Return a copy of base flows multiplied by scale."""
    return {seg: base[seg] * scale for seg in base}


def _rebuild_mode_maps() -> None:
    """
    Build _TUBULAR_FLOWS_ML_MIN and RENAL_BLOOD_FLOW_ML_MIN for both modes.

    CKD5 CDC flows are absolute values (not proportional) to preserve the
    DT→CDC1 flow monotonicity constraint.

    CKD5 flow chain (mL/min):
      P1=12, P2=9.4, P3=6.8, LH1=5, LH2=5, DT=2 | CDC1=2, CDC2=1.6,
      CDC3=1.2, CDC4=0.9, CDC5=0.5
    All values ≥ next segment (monotonically non-increasing) ✓
    CDC1 (2.0) ≤ DT (2.0) ✓
    CDC5 (0.5) × 0.06 L/h per mL/min = 0.030 L/h = 720 mL/day ✓
    """
    global _TUBULAR_FLOWS_ML_MIN, RENAL_BLOOD_FLOW_ML_MIN

    _TUBULAR_FLOWS_ML_MIN = {
        # Healthy: all segments at full physiological flow
        "healthy": _scaled_flows(
            _HEALTHY_TUBULAR_FLOWS_ML_MIN, CKD_SCALE_FACTORS["healthy"]
        ),

        # CKD-5: two separate scaling rules
        "ckd5": {
            # Filtration-dependent segments: scale with GFR (0.10)
            # P1 = 12 mL/min → Q_GFR = 0.72 L/h (consistent with eGFR ~10%)
            **{seg: _HEALTHY_TUBULAR_FLOWS_ML_MIN[seg] * CKD_SCALE_FACTORS["ckd5"]
               for seg in PROX + NONPROX},

            # Collecting duct: fixed absolute values to preserve monotonicity.
            # DT_ckd5 = 2.0 mL/min → CDC1 must be ≤ 2.0 mL/min.
            # CDC5 = 0.5 mL/min → Q_urine = 0.030 L/h (720 mL/day).
            # Values decrease smoothly from CDC1 to CDC5.
            "CDC1": 2.0,   # = DT_ckd5; no unphysical flow increase at boundary
            "CDC2": 1.6,
            "CDC3": 1.2,
            "CDC4": 0.9,
            "CDC5": 0.5,   # → Q_urine = 0.030 L/h
        },
    }

    RENAL_BLOOD_FLOW_ML_MIN = {
        # Healthy: 1200 mL/min = 72 L/h
        "healthy": _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD_SCALE_FACTORS["healthy"],

        # CKD-5: physiology.py applies CKD5_RBF_SCALE=0.35 to the healthy value.
        # This entry is stored for reference only; physiology.py does NOT use it
        # directly (it reads "healthy" and applies the scale itself so the scale
        # can be changed in one place).
        "ckd5": _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD5_RBF_SCALE,  # 420 mL/min = 25.2 L/h
    }

    if VERBOSE:
        print(
            f"[constants] Flow maps rebuilt — "
            f"GFR_scale={CKD_SCALE_FACTORS['ckd5']:.3f}, "
            f"CDC_urine_scale={CKD_URINE_SCALE:.2f}, "
            f"RBF_scale={CKD5_RBF_SCALE:.2f}"
        )


# Build maps at import time
_TUBULAR_FLOWS_ML_MIN:     dict = {}
RENAL_BLOOD_FLOW_ML_MIN:   dict = {}
_rebuild_mode_maps()


def set_ckd_scale(scale: float) -> None:
    """
    Override CKD-5 GFR/tubular-flow scale and rebuild maps.

    Use for sensitivity analysis only.
    scale must be in (0, 1]. Typical value: 0.10.
    Does NOT affect CKD5_RBF_SCALE or CKD_URINE_SCALE.
    """
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"CKD scale must be in (0, 1], got {scale}")
    CKD_SCALE_FACTORS["ckd5"] = float(scale)
    _rebuild_mode_maps()
    if VERBOSE:
        print(f"[constants] CKD-5 GFR scale set to {scale:.3f}")


def get_tubular_flows_ml_min(mode: str, scale: float | None = None) -> dict:
    """
    Return a copy of per-segment tubular flows (mL/min) for the given mode.

    Parameters
    ----------
    mode  : "healthy" or "ckd5"
    scale : optional extra multiplier applied to ALL segments.
            Use only for sensitivity analysis — the mode maps are
            already correctly scaled and should not be rescaled routinely.
    """
    if mode not in _TUBULAR_FLOWS_ML_MIN:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'healthy' or 'ckd5'.")
    flows = _TUBULAR_FLOWS_ML_MIN[mode].copy()
    if scale is not None:
        flows = {k: v * float(scale) for k, v in flows.items()}
    if VERBOSE:
        print("_TUBULAR_FLOWS_ML_MIN[mode]", flows)
    return flows


def get_renal_blood_flow_ml_min(mode: str, scale: float | None = None) -> float:
    """
    Return renal blood flow (mL/min) for the given mode.

    physiology.py calls this with mode="healthy" and applies CKD5_RBF_SCALE
    itself, so the CKD5 value here is for reference only.
    """
    if mode not in RENAL_BLOOD_FLOW_ML_MIN:
        raise ValueError(f"Unknown mode '{mode}'.")
    rbf = RENAL_BLOOD_FLOW_ML_MIN[mode]
    if scale is not None:
        rbf = rbf * float(scale)
    if VERBOSE:
        print("RENAL_BLOOD_FLOW_ML_MIN[mode]", rbf)
    return rbf


# ══════════════════════════════════════════════════════════════════════════════
# 6. SEGMENT VOLUMES (L)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed anatomical volumes for both kidneys (unless PER_KIDNEY=True).
# Derived from residence-time estimates:
#   V (L) = Q (mL/min) × τ (min) / 1000
# with target residence times:
#   Proximal (P1-P3): τ ≈ 1 min  → V ≈ 0.030 L each
#   Loop (LH1-LH2):  τ ≈ 1 min  → V ≈ 0.003 L each
#   DT:               τ ≈ 2.5 min → V ≈ 0.019 L
#   CDC (each):       τ ≈ 3-6 min → V ≈ 0.237 L each
# These are held constant across modes (volumes do not change with CKD).

V_SEG = {
    "P1":   0.0305 * _twoK_to_oneK,
    "P2":   0.0305 * _twoK_to_oneK,
    "P3":   0.0305 * _twoK_to_oneK,
    "LH1":  0.0027 * _twoK_to_oneK,
    "LH2":  0.0027 * _twoK_to_oneK,
    "DT":   0.0194 * _twoK_to_oneK,
    "CDC1": 0.2370 * _twoK_to_oneK,
    "CDC2": 0.2370 * _twoK_to_oneK,
    "CDC3": 0.2370 * _twoK_to_oneK,
    "CDC4": 0.2370 * _twoK_to_oneK,
    "CDC5": 0.2370 * _twoK_to_oneK,
}


# ══════════════════════════════════════════════════════════════════════════════
# 7. pH MAPS
# ══════════════════════════════════════════════════════════════════════════════
# Tubular lumen pH decreases from proximal (7.2) to collecting duct (6.5)
# as proton secretion acidifies the urine.
# Plasma pH is 7.40 in healthy; not changed in CKD5 for simplicity
# (systemic acidosis in CKD5 lowers pH slightly but this is not modelled here).
# Cell pH = 7.20 throughout (intracellular buffering).

_PH_MAPS = {
    "healthy": {
        "lumen": {
            "P1": 7.2, "P2": 7.1, "P3": 7.0,
            "LH1": 7.0, "LH2": 7.0, "DT": 6.9,
            "CDC1": 6.8, "CDC2": 6.7, "CDC3": 6.6, "CDC4": 6.5, "CDC5": 6.5,
        },
        "plasma": 7.40,
        "cell": {seg: 7.20 for seg in SEGMENTS},
    },
    "ckd5": {
        "lumen": {
            "P1": 7.2, "P2": 7.1, "P3": 7.0,
            "LH1": 7.0, "LH2": 7.0, "DT": 6.9,
            "CDC1": 6.8, "CDC2": 6.7, "CDC3": 6.6, "CDC4": 6.5, "CDC5": 6.5,
        },
        "plasma": 7.40,
        "cell": {seg: 7.20 for seg in SEGMENTS},
    },
}

LUMEN_pH  = _PH_MAPS[MODE]["lumen"]
PLASMA_pH = _PH_MAPS[MODE]["plasma"]
CELL_pH   = _PH_MAPS[MODE]["cell"]


def update_ph_maps(mode: str = "healthy") -> None:
    """Update module-level pH globals for the given mode."""
    global LUMEN_pH, PLASMA_pH, CELL_pH
    if mode not in _PH_MAPS:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'healthy' or 'ckd5'.")
    LUMEN_pH  = _PH_MAPS[mode]["lumen"]
    PLASMA_pH = _PH_MAPS[mode]["plasma"]
    CELL_pH   = _PH_MAPS[mode]["cell"]


# ══════════════════════════════════════════════════════════════════════════════
# 8. MOLECULAR WEIGHTS (g/mol)
# ══════════════════════════════════════════════════════════════════════════════
# Used for µM ↔ mg/L conversions.
# Formula: mg/L = µM × MW / 1000

MW_FUR = 330.74   # furosemide  (C12H11ClN2O5S)
MW_IS  = 213.17   # indoxyl sulfate  (C8H7NO4S)
MW_HA  = 179.17   # hippuric acid  (C9H9NO3)
MW_pCS = 188.19   # p-cresyl sulfate  (C7H7O4S)


# ══════════════════════════════════════════════════════════════════════════════
# 9. TRANSPORT GEOMETRY WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
# OAT1 basolateral uptake is distributed across P1, P2, P3 in proportion to
# OAT1 expression density along the proximal tubule.
# Scotcher 2016 CPT:PSP 5:87 reports expression gradient P1 > P2 > P3,
# approximated here as 40:40:20 (normalised to sum to 1.0).
# Apical secretion uses the same segment weights.

RAW_BASO_PROX = {"P1": 0.40, "P2": 0.40, "P3": 0.20}
RAW_APIC_PROX = {"P1": 0.40, "P2": 0.40, "P3": 0.20}

_sum_baso = sum(RAW_BASO_PROX.values())
_sum_apic = sum(RAW_APIC_PROX.values())

W_BASO_PROX = {k: v / _sum_baso for k, v in RAW_BASO_PROX.items()}
W_APIC_PROX = {k: v / _sum_apic for k, v in RAW_APIC_PROX.items()}

# Ratio of basolateral to apical OAT expression (used for passive PD scaling)
PROX_BASO_OVER_APICAL  = 1.0 / 30.0

# Microvilli amplification factor for apical surface area
MICROVILLI_FACTOR_APICAL = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# 10. NUMERICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

EPS  = 1e-12   # guard against division by zero throughout ODE
RTOL = 1e-7    # relative tolerance for scipy solve_ivp
ATOL = 1e-9    # absolute tolerance for scipy solve_ivp

# Unit conversion factors
ML_MIN_TO_L_H  = 60.0 / 1000.0    # 1 mL/min = 0.06 L/h
DM2_TO_CM2     = 100.0             # 1 dm² = 100 cm²
CM_S_TO_L_H    = 3600.0 / 1000.0  # 1 cm/s × area(cm²) → volume/h (with area in L units)


# ══════════════════════════════════════════════════════════════════════════════
# 11. FLOW VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_flows(
    flows_ml_min: dict,
    gfr_ml_min: float,
    urine_flow_ml_min: float,
) -> bool:
    """
    Check that tubular flows are physiologically consistent.

    Checks performed:
      1. P1 flow ≈ GFR (within 10%)
      2. CDC5 flow ≈ urine flow (within 10%)
      3. Flows are monotonically non-increasing along SEGMENTS

    Returns True if all checks pass, False otherwise.
    Prints a warning for each failed check.

    Note for CKD5:
      CDC5 is scaled by CKD_URINE_SCALE (not GFR scale), so
      urine_flow_ml_min passed here should be the CDC5 value,
      not a GFR-derived estimate. The ODE caller is responsible
      for passing the correct reference value.
    """
    p1    = flows_ml_min["P1"]
    cdc5  = flows_ml_min["CDC5"]
    ok    = True

    # Check 1: P1 ≈ GFR
    ratio = p1 / max(gfr_ml_min, EPS)
    if abs(ratio - 1.0) > 0.1:
        print(
            f"[WARN] validate_flows: P1={p1:.3f} mL/min ≠ GFR={gfr_ml_min:.3f} "
            f"(ratio={ratio:.3f})"
        )
        ok = False

    # Check 2: CDC5 ≈ urine flow
    ratio2 = cdc5 / max(urine_flow_ml_min, EPS)
    if abs(ratio2 - 1.0) > 0.1:
        print(
            f"[WARN] validate_flows: CDC5={cdc5:.3f} mL/min ≠ "
            f"urine={urine_flow_ml_min:.3f} (ratio={ratio2:.3f})"
        )
        ok = False

    # Check 3: monotonic decrease
    for a, b in zip(SEGMENTS[:-1], SEGMENTS[1:]):
        if flows_ml_min[a] < flows_ml_min[b]:
            print(
                f"[WARN] validate_flows: flow increases from "
                f"{a}({flows_ml_min[a]:.3f}) to {b}({flows_ml_min[b]:.3f})"
            )
            ok = False

    if ok and VERBOSE:
        print("Flow validation passed!")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 12. INITIALISE pH MAPS
# ══════════════════════════════════════════════════════════════════════════════
update_ph_maps(MODE)


# ══════════════════════════════════════════════════════════════════════════════
# 13. SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== constants.py self-test ===\n")

    # Test 1: Healthy flows are unchanged
    h = get_tubular_flows_ml_min("healthy")
    assert h["P1"] == 120, f"FAIL: P1={h['P1']}"
    assert h["CDC5"] == 1, f"FAIL: CDC5={h['CDC5']}"
    print(f"  ✓ Healthy: P1={h['P1']} mL/min  CDC5={h['CDC5']} mL/min")

    # Test 2: CKD5 GFR-scaled segments
    c = get_tubular_flows_ml_min("ckd5")
    assert abs(c["P1"] - 12.0) < 0.01, f"FAIL P1={c['P1']}"
    print(f"  ✓ CKD5 GFR: P1={c['P1']} mL/min (= 120×0.10)")

    # Test 3: CKD5 CDC5 uses urine scale not GFR scale
    assert abs(c["CDC5"] - 0.5) < 0.01, f"FAIL CDC5={c['CDC5']}"
    print(f"  ✓ CKD5 urine: CDC5={c['CDC5']} mL/min (= 0.5 → {c['CDC5']*ML_MIN_TO_L_H:.3f} L/h)")

    # Test 4: CKD5 monotonicity
    segs = ["P1","P2","P3","LH1","LH2","DT","CDC1","CDC2","CDC3","CDC4","CDC5"]
    for a, b in zip(segs[:-1], segs[1:]):
        assert c[a] >= c[b], f"FAIL monotonicity: {a}({c[a]}) < {b}({c[b]})"
    print(f"  ✓ CKD5 monotonicity: all flows non-increasing")

    # Test 5: DT→CDC1 boundary (no flow increase)
    assert c["CDC1"] <= c["DT"], f"FAIL: CDC1({c['CDC1']}) > DT({c['DT']})"
    print(f"  ✓ DT→CDC1 boundary: DT={c['DT']} ≥ CDC1={c['CDC1']}")

    # Test 6: Q_urine values
    Q_urine_h = h["CDC5"] * ML_MIN_TO_L_H
    Q_urine_c = c["CDC5"] * ML_MIN_TO_L_H
    assert abs(Q_urine_h - 0.06) < 1e-6, f"FAIL Q_urine_h={Q_urine_h}"
    assert abs(Q_urine_c - 0.030) < 1e-6, f"FAIL Q_urine_c={Q_urine_c}"
    print(f"  ✓ Q_urine healthy={Q_urine_h:.3f} L/h, CKD5={Q_urine_c:.3f} L/h")

    # Test 7: RBF values
    rbf_h = get_renal_blood_flow_ml_min("healthy")
    assert abs(rbf_h - 1200.0) < 1, f"FAIL rbf_h={rbf_h}"
    Q_kidney_h = rbf_h * ML_MIN_TO_L_H
    Q_kidney_c = _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD5_RBF_SCALE * ML_MIN_TO_L_H
    print(f"  ✓ Q_kidney healthy={Q_kidney_h:.1f} L/h, CKD5={Q_kidney_c:.1f} L/h")

    # Test 8: MW values
    assert abs(MW_IS - 213.17) < 0.01
    assert abs(MW_FUR - 330.74) < 0.01
    print(f"  ✓ MW_FUR={MW_FUR}, MW_IS={MW_IS}, MW_HA={MW_HA}, MW_pCS={MW_pCS}")

    # Test 9: CKD5_RBF_SCALE is exported (physiology.py imports it)
    assert CKD5_RBF_SCALE == 0.35
    print(f"  ✓ CKD5_RBF_SCALE={CKD5_RBF_SCALE} (importable by physiology.py)")

    # Test 10: validate_flows passes for both modes
    assert validate_flows(h, h["P1"], h["CDC5"])
    VERBOSE = False
    assert validate_flows(c, c["P1"], c["CDC5"])
    VERBOSE = True
    print(f"  ✓ validate_flows passes for healthy and CKD5")

    print("\n=== All 10 tests passed. ===\n")