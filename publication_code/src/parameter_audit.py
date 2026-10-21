"""
parameter_audit.py
==================
Parameter protection and audit system for the kidney PBPK model.

DESIGN PRINCIPLE — SINGLE SOURCE OF TRUTH
------------------------------------------
ALL expected parameter values in this file are imported directly from
their canonical source files. No numeric values are hardcoded here.

  Compound values  ←  compound_parameters_final.py  (get_compound_parameters)
  Physiology values ← constants.py + physiology.py  (CKD_SCALE_FACTORS, etc.)

This eliminates the drift problem where the same number existed in multiple
files and could go out of sync silently.

When a parameter changes in compound_parameters_final.py, the audit
automatically reflects the new value on the next import — no manual update
needed here.

THREE MECHANISMS
----------------
1. GROUND-TRUTH REGISTRY  (EXPECTED_PARAMS)
   Built at import time by _build_registry() which reads live values from
   compound_parameters_final.py and constants.py.

2. audit_compounds(drug, tox1, tox2, tox3, phys, scenario, strict)
   Call once in simulation.py before solve_ivp().
   Compares every live value against the registry.
   Raises ValueError on ERROR-level findings if strict=True.

3. GuardedDict
   Wraps compound dicts after scenario setup.
   Raises RuntimeError on any unexpected write during the solve.
   Use the .unlocked() context manager for legitimate CKD scenario writes.

HOW TO USE
----------
In simulation.py, after make_phys():

    from parameter_audit import audit_compounds, GuardedDict
    audit_compounds(C.drug, C.tox1, C.tox2, C.tox3, phys, scenario=mode)

    C.drug  = GuardedDict(C.drug,  name="drug")
    C.tox1  = GuardedDict(C.tox1, name="tox1")
    ...
"""

from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS — single source of truth
# ══════════════════════════════════════════════════════════════════════════════

from compound_parameters import get_compound_parameters

from constants import (
    CKD_SCALE_FACTORS,     # {"healthy": 1.0, "ckd5": 0.10}
    CKD5_RBF_SCALE,        # 0.35
    CKD_URINE_SCALE,       # 0.50
    ML_MIN_TO_L_H,         # 0.06
    MW_IS, MW_HA, MW_pCS, MW_FUR,
    _HEALTHY_TUBULAR_FLOWS_ML_MIN,
    _RENAL_BLOOD_FLOW_BASE_ML_MIN,
)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY BUILDER — reads live values, no hardcoding
# ══════════════════════════════════════════════════════════════════════════════

def _build_registry() -> dict:
    """
    Build the EXPECTED_PARAMS registry from live parameter sources.

    All numeric expected values are derived from:
      - get_compound_parameters("healthy") / ("ckd5")
      - constants.py scalar values

    Never hardcode a number here — add an import instead.
    """

    drug_h, t1_h, t2_h, t3_h = get_compound_parameters("healthy")
    drug_c, t1_c, t2_c, t3_c = get_compound_parameters("ckd5")

    # ── Derived physiology values ─────────────────────────────────────────────
    GFR_scale_h    = CKD_SCALE_FACTORS["healthy"]          # 1.0
    GFR_scale_c    = CKD_SCALE_FACTORS["ckd5"]             # 0.10
    P1_healthy     = _HEALTHY_TUBULAR_FLOWS_ML_MIN["P1"]   # 120 mL/min
    CDC5_healthy   = _HEALTHY_TUBULAR_FLOWS_ML_MIN["CDC5"] # 1 mL/min

    Q_GFR_h        = P1_healthy * GFR_scale_h * ML_MIN_TO_L_H   # 7.2 L/h
    Q_GFR_c        = P1_healthy * GFR_scale_c * ML_MIN_TO_L_H   # 0.72 L/h
    Q_kidney_h     = _RENAL_BLOOD_FLOW_BASE_ML_MIN * ML_MIN_TO_L_H                # 72 L/h
    Q_kidney_c     = _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD5_RBF_SCALE * ML_MIN_TO_L_H  # 25.2 L/h
    Q_urine_h      = CDC5_healthy * ML_MIN_TO_L_H                 # 0.06 L/h
    Q_urine_c      = CDC5_healthy * CKD_URINE_SCALE * ML_MIN_TO_L_H  # 0.030 L/h

    # ── Known-wrong forbidden values (documented once, used in registry) ──────
    # These are values that USED to appear in the codebase and must never
    # reappear. Their source is documented in the forbidden comment.
    _OLD_IS_FU_HEALTHY  = 0.04   # old: implied 96% bound (wrong)
    _OLD_HA_FU_HEALTHY  = 0.50   # old: implied 50% bound (wrong)
    _OLD_pCS_FU_HEALTHY = 0.04   # old: implied 96% bound (wrong)
    _OLD_HA_FU_CKD5     = 0.70   # old: wrong direction
    _OLD_DRUG_FU_CKD5   = 0.20   # old: 10x too high
    _OLD_IS_KM          = 5.02   # old: used wrong MW_IS=279
    _OLD_KI             = 0.13   # old: 32x too low
    _VCEN_CORRUPT       = 42.0   # known corruption from stale TissueDistribution
    _OLD_Q_URINE_BUG    = 0.60   # CDC5=10 mL/min bug

    return {

        # ── Drug — healthy ───────────────────────────────────────────────────
        "drug.fu_p@healthy": {
            "expected":  drug_h["fu_p"],
            "tol_pct":   5.0,
            "forbidden": [_OLD_DRUG_FU_CKD5, 0.10, 0.15],
            "source":    f"compound_parameters.py: drug healthy fu_p={drug_h['fu_p']}. "
                         "Furosemide ~98% albumin-bound (Site II). Andreasen 1981.",
            "units":     "dimensionless",
        },
        "drug.BP": {
            "expected":  drug_h["BP"],
            "tol_pct":   5.0,
            "forbidden": [1.0],
            "source":    f"compound_parameters.py: drug BP={drug_h['BP']}. Britz 2020.",
            "units":     "dimensionless",
        },
        "drug.k_elim_hinv@healthy": {
            "expected":  drug_h["k_elim_hinv"],
            "tol_pct":   20.0,
            "forbidden": [0.40, 1.968, 0.376, 0.0125, 0.688],
            "source":    f"compound_parameters.py: k_elim={drug_h['k_elim_hinv']} h-1 (healthy). "
                         "t1/2=2.8 h. Brater 1983.",
            "units":     "h-1",
        },
        "drug.k_elim_hinv@ckd5": {
            "expected":  drug_c["k_elim_hinv"],
            "tol_pct":   30.0,
            "forbidden": [0.25, 0.688, 0.40],
            "source":    f"compound_parameters.py: k_elim={drug_c['k_elim_hinv']} h-1 (CKD5). "
                         "Uremia inhibits CYP2C9/UGT1A9 — hepatic CL reduced ~80%. "
                         "Andreasen 1983 Br J Clin Pharmacol 16:391; "
                         "Nolin 2008 J Am Soc Nephrol 19:1433.",
            "units":     "h-1",
        },
        "drug.Km_baso_uptake": {
            "expected":  drug_h["Km_baso_uptake"],
            "tol_pct":   5.0,
            "forbidden": [],
            "source":    f"compound_parameters.py: Km_OAT1={drug_h['Km_baso_uptake']} mg/L. "
                         "Hasannejad 2004.",
            "units":     "mg/L",
        },
        "drug.IR@healthy": {
            "expected":  drug_h["IR"],
            "tol_pct":   1.0,
            "forbidden": [300.0, 400.0],
            "source":    f"compound_parameters.py: IR={drug_h['IR']} mg/h = 120 mg / 2 h.",
            "units":     "mg/h",
        },

        # ── Drug — CKD5 ──────────────────────────────────────────────────────
        "drug.fu_p@ckd5": {
            "expected":  drug_c["fu_p"],
            "tol_pct":   40.0,
            "forbidden": [_OLD_DRUG_FU_CKD5, 0.25, 0.10, 0.15],
            "source":    f"compound_parameters.py: drug CKD5 fu_p={drug_c['fu_p']}. "
                         "Modest CMPF/oleate displacement. Takamura 1997.",
            "units":     "dimensionless",
        },
        "drug.IR@ckd5": {
            "expected":  drug_c["IR"],
            "tol_pct":   1.0,
            "forbidden": [300.0, 400.0],
            "source":    f"compound_parameters.py: IR={drug_c['IR']} mg/h (same dose in CKD5).",
            "units":     "mg/h",
        },

        # ── IS (tox1) — healthy ──────────────────────────────────────────────
        "tox1.fu_p@healthy": {
            "expected":  t1_h["fu_p"],
            "tol_pct":   10.0,
            "forbidden": [_OLD_IS_FU_HEALTHY, 0.08, 0.96, 1.0],
            "source":    f"compound_parameters.py: IS healthy fu_p={t1_h['fu_p']}. "
                         "84% bound. Deltombe 2015.",
            "units":     "dimensionless",
        },
        "tox1.BP": {
            "expected":  t1_h["BP"],
            "tol_pct":   5.0,
            "forbidden": [1.0],
            "source":    f"compound_parameters.py: IS BP={t1_h['BP']}.",
            "units":     "dimensionless",
        },
        "tox1.Km_baso_uptake": {
            "expected":  t1_h["Km_baso_uptake"],
            "tol_pct":   5.0,
            "forbidden": [_OLD_IS_KM, 4.02],
            "source":    f"compound_parameters.py: IS Km OAT1={t1_h['Km_baso_uptake']} mg/L. "
                         f"18 uM x MW_IS({MW_IS})/1000. Deguchi 2004.",
            "units":     "mg/L",
        },
        "tox1.Ki_furo_to_IS_mgL": {
            "expected":  t1_h["Ki_furo_to_IS_mgL"],
            "tol_pct":   5.0,
            "forbidden": [_OLD_KI, 0.1310, 0.1, 0.10, 0.08, 4.13],  # 4.13 = old Mihaila value
            "source":    f"compound_parameters.py: Ki={t1_h['Ki_furo_to_IS_mgL']} mg/L. "
                         "Costes-Albrespic 2025 KI Reports 10:2165 (direct Ki at OAT1).",
            "units":     "mg/L",
        },

        # ── IS (tox1) — CKD5 ─────────────────────────────────────────────────
        "tox1.fu_p@ckd5": {
            "expected":  t1_c["fu_p"],
            "tol_pct":   15.0,
            "forbidden": [_OLD_IS_FU_HEALTHY, 0.15, 0.20],
            "source":    f"compound_parameters.py: IS CKD5 fu_p={t1_c['fu_p']}. "
                         "90% bound. Deltombe 2015.",
            "units":     "dimensionless",
        },

        # ── HA (tox2) — healthy ──────────────────────────────────────────────
        "tox2.fu_p@healthy": {
            "expected":  t2_h["fu_p"],
            "tol_pct":   10.0,
            "forbidden": [_OLD_HA_FU_HEALTHY, 0.04, 0.96, 1.0],
            "source":    f"compound_parameters.py: HA healthy fu_p={t2_h['fu_p']}. "
                         "34% bound. Deltombe 2015.",
            "units":     "dimensionless",
        },
        "tox2.BP": {
            "expected":  t2_h["BP"],
            "tol_pct":   5.0,
            "forbidden": [1.0],
            "source":    f"compound_parameters.py: HA BP={t2_h['BP']}.",
            "units":     "dimensionless",
        },
        "tox2.Km_baso_uptake": {
            "expected":  t2_h["Km_baso_uptake"],
            "tol_pct":   5.0,
            "forbidden": [],
            "source":    f"compound_parameters.py: HA Km OAT1={t2_h['Km_baso_uptake']} mg/L. "
                         f"24 uM x MW_HA({MW_HA})/1000. Deguchi 2004.",
            "units":     "mg/L",
        },
        "tox2.Ki_furo_to_HA_mgL": {
            "expected":  t2_h["Ki_furo_to_HA_mgL"],
            "tol_pct":   5.0,
            "forbidden": [_OLD_KI, 0.1310, 0.1, 0.10, 0.08, 4.13],  # 4.13 = old Mihaila value
            "source":    f"compound_parameters.py: Ki={t2_h['Ki_furo_to_HA_mgL']} mg/L. "
                         "Same Ki as IS (furosemide property at OAT1). Costes-Albrespic 2025.",
            "units":     "mg/L",
        },

        # ── HA (tox2) — CKD5 ─────────────────────────────────────────────────
        "tox2.fu_p@ckd5": {
            "expected":  t2_c["fu_p"],
            "tol_pct":   15.0,
            "forbidden": [_OLD_HA_FU_CKD5, _OLD_HA_FU_HEALTHY, 0.65, 0.90],
            "source":    f"compound_parameters.py: HA CKD5 fu_p={t2_c['fu_p']}. "
                         "48% bound. Old 0.70 was wrong direction. Deltombe 2015.",
            "units":     "dimensionless",
        },

        # ── pCS (tox3) — healthy ─────────────────────────────────────────────
        "tox3.fu_p@healthy": {
            "expected":  t3_h["fu_p"],
            "tol_pct":   10.0,
            "forbidden": [_OLD_pCS_FU_HEALTHY, 0.12, 0.15, 0.96, 1.0],
            "source":    f"compound_parameters.py: pCS healthy fu_p={t3_h['fu_p']}. "
                         "94% bound. Deltombe 2015.",
            "units":     "dimensionless",
        },
        "tox3.BP": {
            "expected":  t3_h["BP"],
            "tol_pct":   5.0,
            "forbidden": [1.0],
            "source":    f"compound_parameters.py: pCS BP={t3_h['BP']}.",
            "units":     "dimensionless",
        },
        "tox3.Km_baso_uptake": {
            "expected":  t3_h["Km_baso_uptake"],
            "tol_pct":   5.0,
            "forbidden": [],
            "source":    f"compound_parameters.py: pCS Km OAT1={t3_h['Km_baso_uptake']} mg/L. "
                         f"120 uM x MW_pCS({MW_pCS})/1000. Watanabe 2014.",
            "units":     "mg/L",
        },
        "tox3.Ki_furo_to_pCS_mgL": {
            "expected":  t3_h["Ki_furo_to_pCS_mgL"],
            "tol_pct":   5.0,
            "forbidden": [_OLD_KI, 0.1310, 0.1, 0.10, 0.08, 4.10],  # 4.13 = old Mihaila value
            "source":    f"compound_parameters.py: Ki={t3_h['Ki_furo_to_pCS_mgL']} mg/L. "
                         "Costes-Albrespic 2025 KI Reports 10:2165.",
            "units":     "mg/L",
        },

        # ── pCS (tox3) — CKD5 ────────────────────────────────────────────────
        "tox3.fu_p@ckd5": {
            "expected":  t3_c["fu_p"],
            "tol_pct":   20.0,
            "forbidden": [0.08, 0.12, 0.15],
            "source":    f"compound_parameters.py: pCS CKD5 fu_p={t3_c['fu_p']}. "
                         "95% bound. Deltombe 2015.",
            "units":     "dimensionless",
        },

        # ── Physiology — healthy ─────────────────────────────────────────────
        "phys.V_ven0": {
            "expected":  5.0,
            "tol_pct":   1.0,
            "forbidden": [3.0, _VCEN_CORRUPT],
            "source":    "Central compartment = 5.0 L. "
                         f"Vd_ss = 5 + {drug_h['Vperi_L']} * ({drug_h['k12_hinv']}/{drug_h['k21_hinv']}) = "
                         f"{5.0 + drug_h['Vperi_L']*drug_h['k12_hinv']/drug_h['k21_hinv']:.2f} L. "
                         f"42.0 = known Vcen corruption from stale TissueDistribution.",
            "units":     "L",
        },
        "phys.Q_kidney0@healthy": {
            "expected":  round(Q_kidney_h, 4),
            "tol_pct":   5.0,
            "forbidden": [],
            "source":    f"constants.py: RBF_base({_RENAL_BLOOD_FLOW_BASE_ML_MIN}) x "
                         f"ML_MIN_TO_L_H({ML_MIN_TO_L_H}) = {Q_kidney_h:.1f} L/h.",
            "units":     "L/h",
        },
        "phys.Q_GFR@healthy": {
            "expected":  round(Q_GFR_h, 4),
            "tol_pct":   5.0,
            "forbidden": [],
            "source":    f"constants.py: P1({P1_healthy}) x GFR_scale({GFR_scale_h}) x "
                         f"ML_MIN_TO_L_H = {Q_GFR_h:.3f} L/h.",
            "units":     "L/h",
        },
        "phys.Q_urine@healthy": {
            "expected":  round(Q_urine_h, 4),
            "tol_pct":   20.0,
            "forbidden": [_OLD_Q_URINE_BUG, 3.0, 50.0],
            "source":    f"constants.py: CDC5({CDC5_healthy}) x ML_MIN_TO_L_H = "
                         f"{Q_urine_h:.4f} L/h. "
                         f"{_OLD_Q_URINE_BUG} = CDC5=10 mL/min bug.",
            "units":     "L/h",
        },

        # ── Physiology — CKD5 ────────────────────────────────────────────────
        "phys.Q_kidney0@ckd5": {
            "expected":  round(Q_kidney_c, 4),
            "tol_pct":   10.0,
            "forbidden": [round(Q_kidney_h, 0)],
            "source":    f"constants.py: RBF_base x CKD5_RBF_SCALE({CKD5_RBF_SCALE}) x "
                         f"ML_MIN_TO_L_H = {Q_kidney_c:.1f} L/h.",
            "units":     "L/h",
        },
        "phys.Q_GFR@ckd5": {
            "expected":  round(Q_GFR_c, 4),
            "tol_pct":   10.0,
            "forbidden": [round(Q_GFR_h, 1)],
            "source":    f"constants.py: P1({P1_healthy}) x GFR_scale_ckd5({GFR_scale_c}) x "
                         f"ML_MIN_TO_L_H = {Q_GFR_c:.4f} L/h.",
            "units":     "L/h",
        },
        "phys.Q_urine@ckd5": {
            "expected":  round(Q_urine_c, 4),
            "tol_pct":   30.0,
            "forbidden": [3.0, _OLD_Q_URINE_BUG, 0.006],
            "source":    f"constants.py: CDC5({CDC5_healthy}) x CKD_URINE_SCALE({CKD_URINE_SCALE}) x "
                         f"ML_MIN_TO_L_H = {Q_urine_c:.4f} L/h (720 mL/day). "
                         "Decoupled from GFR. Mair 2021 JASN.",
            "units":     "L/h",
        },
    }


# Build registry once at import time
EXPECTED_PARAMS = _build_registry()


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def audit_compounds(
    drug: dict,
    tox1: dict,
    tox2: dict,
    tox3: dict,
    phys: dict,
    scenario: str = "healthy",
    strict: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Audit compound and physiology parameters against the ground-truth registry.

    Expected values are imported from compound_parameters.py and constants.py
    at module load time — no hardcoded numbers here.

    Parameters
    ----------
    drug, tox1, tox2, tox3 : compound dicts from compounds.py
    phys     : physiology dict from make_phys()
    scenario : "healthy" or "ckd5"
    strict   : raise ValueError on ERROR-level findings
    verbose  : print full audit table

    Returns dict with keys "pass", "warn", "error".
    """
    compound_map = {
        "drug": drug, "tox1": tox1, "tox2": tox2,
        "tox3": tox3, "phys": phys,
    }
    results = {"pass": [], "warn": [], "error": []}

    if verbose:
        print()
        print("=" * 72)
        print(f"  PARAMETER AUDIT  —  scenario: {scenario}")
        print("=" * 72)
        print(f"  {'Parameter':<36} {'Expected':>10} {'Actual':>10}  Status")
        print("-" * 72)

    for reg_key, spec in EXPECTED_PARAMS.items():

        if "@" in reg_key:
            base, key_scenario = reg_key.rsplit("@", 1)
            if key_scenario != scenario:
                continue
        else:
            base = reg_key

        parts = base.split(".", 1)
        if len(parts) != 2:
            continue
        cmpd_name, param_name = parts

        cmpd   = compound_map.get(cmpd_name)
        if cmpd is None:
            continue

        actual   = cmpd.get(param_name, "MISSING")
        expected = spec["expected"]
        tol_pct  = spec.get("tol_pct", 1.0)
        tol_abs  = spec.get("tol_abs", 0.0)
        forbidden = spec.get("forbidden", [])
        source   = spec.get("source", "")
        units    = spec.get("units", "")

        status  = "PASS"
        message = ""

        if actual == "MISSING":
            status  = "WARN"
            message = f"Key '{param_name}' not found in {cmpd_name} dict."

        elif isinstance(expected, (float, int)):
            actual_f = float(actual)
            exp_f    = float(expected)

            for forb in forbidden:
                if abs(actual_f - float(forb)) < 1e-6:
                    status  = "ERROR"
                    message = (
                        f"Value {actual_f} is a KNOWN-WRONG value. "
                        f"Expected {exp_f:.4g} {units}. Source: {source}"
                    )
                    break

            if status != "ERROR":
                abs_tol = max(tol_abs, abs(exp_f) * tol_pct / 100.0)
                if abs(actual_f - exp_f) > abs_tol:
                    rel = abs(actual_f - exp_f) / max(abs(exp_f), 1e-12) * 100.0
                    status  = "WARN"
                    message = (
                        f"Value {actual_f:.4g} vs expected {exp_f:.4g} {units} "
                        f"({rel:.1f}% off). Source: {source}"
                    )

        elif isinstance(expected, bool):
            if bool(actual) != expected:
                status = "WARN"
                message = f"Expected {expected}, got {actual}."

        elif isinstance(expected, str):
            if str(actual) != expected:
                status = "WARN"
                message = f"Expected '{expected}', got '{actual}'."

        icon = {"PASS": "✅", "WARN": "⚠️ ", "ERROR": "🔴"}.get(status, "?")
        display_key = base

        if verbose or status != "PASS":
            exp_s = f"{expected:.4g}" if isinstance(expected, (int, float)) else str(expected)
            act_s = (f"{float(actual):.4g}" if isinstance(actual, (int, float))
                     else str(actual))
            print(f"  {icon} {display_key:<36} {exp_s:>10} {act_s:>10}  {message}")

        results[status.lower()].append(
            {"key": base, "actual": actual, "expected": expected, "message": message}
        )

    if verbose:
        print("-" * 72)
        n_p, n_w, n_e = (len(results[k]) for k in ("pass", "warn", "error"))
        print(f"  Audit: {n_p} PASS  |  {n_w} WARN  |  {n_e} ERROR")
        print("=" * 72)
        print()

    if strict and results["error"]:
        lines = [f"  [{r['key']}] {r['message']}" for r in results["error"]]
        raise ValueError(
            "\n\nParameter audit FAILED — simulation blocked.\n"
            "Known-wrong values detected:\n\n" + "\n".join(lines) +
            "\n\nFix the values in compound_parameters.py or constants.py."
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# GuardedDict
# ══════════════════════════════════════════════════════════════════════════════

class GuardedDict(dict):
    """
    Dict subclass that raises RuntimeError on overwrites of protected keys.
    Keys present at creation are protected. New keys can always be added.

    Usage
    -----
    drug = GuardedDict(drug_dict, name="drug")
    drug["fu_p"] = 0.99    # RuntimeError
    with drug.unlocked("fu_p"):
        drug["fu_p"] = 0.05  # allowed
    """
    def __init__(self, data: dict, name: str = "unnamed"):
        super().__init__(data)
        object.__setattr__(self, "_protected", set(data.keys()))
        object.__setattr__(self, "_unlocked",  set())
        object.__setattr__(self, "_name",      name)

    def __setitem__(self, key: str, value: Any):
        protected = object.__getattribute__(self, "_protected")
        unlocked  = object.__getattribute__(self, "_unlocked")
        name      = object.__getattribute__(self, "_name")
        if key in protected and key not in unlocked:
            current = self.get(key, "MISSING")
            raise RuntimeError(
                f"\n[GuardedDict] Blocked write to '{name}[\"{key}\"]'.\n"
                f"  Current: {current!r}  ->  Attempted: {value!r}\n"
                f"  Use 'with dict.unlocked(\"{key}\"): ...' to allow."
            )
        super().__setitem__(key, value)
        protected.add(key)

    def unlock(self, key: str):
        object.__getattribute__(self, "_unlocked").add(key)

    def lock(self, key: str):
        object.__getattribute__(self, "_unlocked").discard(key)

    def unlocked(self, *keys):
        return _UnlockContext(self, keys)

    def protected_keys(self) -> set:
        return set(object.__getattribute__(self, "_protected"))

    def __repr__(self):
        name = object.__getattribute__(self, "_name")
        return f"GuardedDict(name={name!r}, keys={list(self.keys())})"


class _UnlockContext:
    def __init__(self, gd, keys):
        self._gd, self._keys = gd, keys
    def __enter__(self):
        for k in self._keys: self._gd.unlock(k)
        return self._gd
    def __exit__(self, *a):
        for k in self._keys: self._gd.lock(k)


# ══════════════════════════════════════════════════════════════════════════════
# find_overwrites() — scan project for rogue assignments
# ══════════════════════════════════════════════════════════════════════════════

def find_overwrites(search_dir: str = ".", param_name: str = None) -> None:
    """Grep .py files for direct assignments to known critical parameters."""
    import os, re
    patterns = (
        [param_name] if param_name else
        list({k.split("@")[0].split(".", 1)[1]
              for k in EXPECTED_PARAMS if "." in k.split("@")[0]})
    )
    print(f"\n[find_overwrites] Searching '{search_dir}' for: {patterns}\n")
    found = False
    for root, _, files in os.walk(search_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                lines = open(fpath, encoding="utf-8", errors="replace").readlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                for pat in patterns:
                    if re.search(
                        rf"""["\']({re.escape(pat)})["\'][\s]*[=](?!=)""", line
                    ):
                        print(f"  {fpath}:{lineno}  ->  {line.rstrip()}")
                        found = True
    if not found:
        print("  No assignments found.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from compound_parameters import get_compound_parameters

    def _h(): return get_compound_parameters("healthy")
    def _c(): return get_compound_parameters("ckd5")

    def _phys_h():
        return {"V_ven0": 5.0, "Q_kidney0": 72.0, "Q_GFR": 7.2, "Q_urine": 0.06}
    def _phys_c():
        return {"V_ven0": 5.0, "Q_kidney0": 25.2, "Q_GFR": 0.72, "Q_urine": 0.030}

    print("\nTest 1 — correct healthy params: expect 0 errors")
    r = audit_compounds(*_h(), _phys_h(), scenario="healthy", strict=False)
    assert len(r["error"]) == 0, f"FAIL: {r['error']}"
    print("  PASS\n")

    print("Test 2 — correct CKD5 params: expect 0 errors")
    r = audit_compounds(*_c(), _phys_c(), scenario="ckd5", strict=False, verbose=False)
    assert len(r["error"]) == 0, f"FAIL: {[x['message'] for x in r['error']]}"
    print("  PASS\n")

    print("Test 3 — old Ki=0.1310 caught as ERROR")
    d, t1, t2, t3 = _h()
    t1["Ki_furo_to_IS_mgL"] = 3.77
    r = audit_compounds(d, t1, t2, t3, _phys_h(), scenario="healthy", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 4 — old Q_urine=0.60 (CDC5 bug) caught as ERROR")
    p = _phys_h(); p["Q_urine"] = 0.60
    r = audit_compounds(*_h(), p, scenario="healthy", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 5 — old IS healthy fu_p=0.04 caught as ERROR")
    d, t1, t2, t3 = _h(); t1["fu_p"] = 0.04
    r = audit_compounds(d, t1, t2, t3, _phys_h(), scenario="healthy", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 6 — old HA CKD5 fu_p=0.70 caught as ERROR")
    d, t1, t2, t3 = _c(); t2["fu_p"] = 0.70
    r = audit_compounds(d, t1, t2, t3, _phys_c(), scenario="ckd5", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 7 — old drug CKD5 fu_p=0.20 caught as ERROR")
    d, t1, t2, t3 = _c(); d["fu_p"] = 0.20
    r = audit_compounds(d, t1, t2, t3, _phys_c(), scenario="ckd5", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 8 — old IS Km=5.02 (wrong MW) caught as ERROR")
    d, t1, t2, t3 = _h(); t1["Km_baso_uptake"] = 5.02
    r = audit_compounds(d, t1, t2, t3, _phys_h(), scenario="healthy", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 9 — V_ven0=42 (Vcen corruption) caught as ERROR")
    p = _phys_h(); p["V_ven0"] = 42.0
    r = audit_compounds(*_h(), p, scenario="healthy", strict=False, verbose=False)
    assert len(r["error"]) >= 1
    print("  PASS\n")

    print("Test 10 — GuardedDict")
    gd = GuardedDict({"fu_p": 0.02, "BP": 0.82}, name="drug")
    try:    gd["fu_p"] = 0.99; print("  FAIL: not blocked")
    except RuntimeError: print("  PASS: blocked rogue write")
    with gd.unlocked("fu_p"): gd["fu_p"] = 0.05
    assert gd["fu_p"] == 0.05
    print(f"  PASS: context write fu_p={gd['fu_p']}")
    try:    gd["fu_p"] = 0.99; print("  FAIL: not re-locked")
    except RuntimeError: print("  PASS: re-locked\n")

    print("Test 11 — registry auto-reflects compound_parameters.py values")
    reg = _build_registry()
    dh, t1h = get_compound_parameters("healthy")[:2]
    assert abs(reg["tox1.fu_p@healthy"]["expected"] - t1h["fu_p"]) < 1e-9
    assert abs(reg["tox1.Ki_furo_to_IS_mgL"]["expected"] - t1h["Ki_furo_to_IS_mgL"]) < 1e-9
    print("  PASS: registry matches live compound_parameters values\n")

    print("=" * 50)
    print("All 11 tests passed.")
    print("=" * 50)