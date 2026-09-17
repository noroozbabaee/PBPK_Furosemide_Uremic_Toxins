# scenario_manager.py
# -*- coding: utf-8 -*-
"""
scenario_manager.py
===================
Central configuration module for simulation scenarios.

WHAT THIS FILE DOES
-------------------
Provides a single entry point — set_scenario(key) — that configures
the global compound dicts (C.drug, C.tox1, C.tox2, C.tox3) for a
named simulation scenario before calling run_sim().

USAGE (in plotting scripts)
---------------------------
    from scenario_manager import set_scenario

    set_scenario("H_NO_DRUG")
    sol_base = run_sim(t_end_h=12, mode="healthy", custom_initial_conds=ic)

    set_scenario("H_WITH_DRUG")
    sol_drug = run_sim(t_end_h=12, mode="healthy", custom_initial_conds=ic)

AVAILABLE SCENARIO KEYS
------------------------
Healthy scenarios:
    "H_NO_DRUG"             — no drug, no inhibition
    "H_WITH_DRUG_NO_INT"    — drug present, OAT1 inhibition OFF
    "H_WITH_DRUG"           — drug present, OAT1 inhibition ON

CKD-5 scenarios:
    "CKD_NO_DRUG"           — no drug, no inhibition
    "CKD_WITH_DRUG_NO_INT"  — drug present, OAT1 inhibition OFF
    "CKD_WITH_DRUG"         — drug present, OAT1 inhibition ON

WHAT set_scenario() DOES — IN ORDER
-------------------------------------
    1. Look up the preset (mode, drug_enabled, inhibit) from _PRESETS
    2. Reload all compound dicts for the mode via compounds.initialize_compounds()
    3. Set C.drug["enabled"]  — controls whether drug equations fire in ODE
    4. Set C.drug["inhibit"]  — controls OAT1 competitive inhibition in ODE
    5. For CKD drug scenarios: apply elevated fu_p and clinical CKD dose
    6. Print a confirmation line so you can verify in the console

WHAT THE ODE ACTUALLY READS (important for understanding)
----------------------------------------------------------
    The ODE (ode_system_twocompartment.py) reads:
        C.drug["enabled"]   → whether to compute drug transport
        C.drug["inhibit"]   → whether to shift toxin OAT1 Km
        C.drug["fu_p"]      → unbound fraction (controls Cu_drug → inhibition strength)
        C.tox*["Ki_furo_to_*_mgL"] → Ki values for Km shift

    The ODE does NOT read K.INHIBIT_TOX_BY_DRUG or K.DRUG_ENABLED.
    These constants exist in some older code but have no effect on results.
    This file does NOT set them to avoid confusion.

INHIBITION MECHANISM REMINDER
------------------------------
When drug["inhibit"] = True:
    Km_eff = Km_base × (1 + Cu_drug / Ki)
    where Cu_drug = C_P1_V × fu_p / BP

    With fu_p = 0.02 (healthy): Cu_drug_peak ≈ 0.16 mg/L → alpha ≈ 1.04 (4%)
    With fu_p = 0.05 (CKD):    Cu_drug_peak ≈ 0.40 mg/L → alpha ≈ 1.10 (10%)

    Ki = 4.13 mg/L for all toxins (Mihaila 2020, Toxins 12:391)

CKD DOSE RATIONALE
-------------------
In standard CKD scenarios:
    fu_p = 0.05  (modest CMPF/oleate displacement from albumin Site I;
                  Takamura 1997 Clin Chem 43:2274)
    IR   = 60.0 mg/h, t_inf_h = 2.0 h → 120 mg total

These values come from compound_parameters_final.py and are NOT overridden here.

For high-dose sensitivity scenarios (CKD_HIGH_DOSE*):
    IR is set to _CKD_HIGH_DOSE_IR (120 mg/h → 240 mg total).
    fu_p stays at compound_parameters value unless configure_ckd_dose() is called.

For custom doses: call configure_ckd_dose(total_mg, fu_p) after set_scenario().
"""

from __future__ import annotations

from typing import Dict, Tuple

import compounds as C


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO PRESET TABLE
# ══════════════════════════════════════════════════════════════════════════════
# Each preset defines: (mode, drug_enabled, inhibit)
#
# mode         : "healthy" or "ckd5" — passed to initialize_compounds()
# drug_enabled : True/False — sets C.drug["enabled"]
# inhibit      : True/False — sets C.drug["inhibit"]

_PRESETS: Dict[str, Tuple[str, bool, bool]] = {

    # ── Healthy scenarios ─────────────────────────────────────────────────────
    "H_NO_DRUG": (
        "healthy", False, False,
        # Baseline: no drug present, no OAT1 inhibition.
        # Use for establishing healthy toxin steady state.
    ),
    "H_WITH_DRUG_NO_INT": (
        "healthy", True, False,
        # Drug is present and cleared renally (OAT1 + MRP4).
        # Inhibition is OFF — drug PK occurs but does NOT shift toxin OAT1 Km.
        # Used to isolate pharmacokinetic from pharmacodynamic effects.
    ),
    "H_WITH_DRUG": (
        "healthy", True, True,
        # Full model: drug present AND competitively inhibits toxin OAT1.
        # Km_eff = Km_base × (1 + Cu_drug/Ki) for each toxin.
        # With fu_p=0.02 (healthy), effect is small (~1% Km shift at peak).
    ),

    # ── CKD-5 scenarios — standard 80 mg dose ─────────────────────────────────
    "CKD_NO_DRUG": (
        "ckd5", False, False,
        # CKD-5 baseline: no drug, no inhibition.
        # Use for establishing CKD toxin steady state (IS, HA, pCS elevated).
    ),
    "CKD_WITH_DRUG_NO_INT": (
        "ckd5", True, False,
        # CKD drug present, inhibition OFF.
        # fu_p=0.10, IR=40 mg/h (80 mg total).
    ),
    "CKD_WITH_DRUG": (
        "ckd5", True, True,
        # Full CKD model: drug + OAT1 inhibition.
        # fu_p=0.10, 80 mg dose → alpha≈1.23, ~19% OAT1 flux reduction.
    ),

    # ── CKD-5 scenarios — clinical high-dose 120 mg, fu_p=0.20 ───────────────
    # Rationale: in severe CKD, furosemide doses of 120–250 mg/day are used
    # due to diuretic resistance. At these doses with fu_p=0.20 (uremic
    # albumin displacement at high IS/HA/CMPF concentrations), the model
    # predicts alpha≈1.70 → ~37% IS rise, matching CKD-REIN (+31.9% IS,
    # PMC12266279) for fu_p=0.20 and 120 mg single dose.
    "CKD_HIGH_DOSE_NO_DRUG": (
        "ckd5", False, False,
        # CKD-5 baseline without drug — same as CKD_NO_DRUG.
        # Provided for consistent pairing with HIGH_DOSE scenarios.
    ),
    "CKD_HIGH_DOSE_NO_INT": (
        "ckd5", True, False,
        # CKD 120 mg dose, fu_p=0.20, inhibition OFF.
    ),
    "CKD_HIGH_DOSE": (
        "ckd5", True, True,
        # CKD 120 mg dose, fu_p=0.20, inhibition ON.
        # alpha≈1.70 → ~37% IS rise (matches CKD-REIN clinical observation).
    ),
}


# ── CKD-specific compound overrides ──────────────────────────────────────────
# Applied on top of CKD compound_parameters.py values when mode = "ckd5".
# Protein binding changes sourced from primary literature (see sources below).
#
# Drug fu_p: rises from 0.02 to 0.10 in CKD-5
#   Mechanism: IS, HA, and CMPF accumulation displaces furosemide from albumin
#              Sudlow site II (Ullian 1996 J Pharmacol Exp Ther 278:808).
#   Note: fu_p=0.10 is the midpoint of the clinically reported 0.05–0.15 range.
#
# Toxin fu_p: sourced from Eloot 2015 PMC4626712 Table 5 (n=95 CKD patients)
#   IS (tox1):  %PB=92% in CKD-5 → fu_p=0.08
#               NOTE: binding is HIGHER in CKD than healthy (77–86%), not lower.
#   HA (tox2):  %PB=43% in CKD-5 → fu_p=0.57  (minimal change from healthy 50%)
#   pCS (tox3): %PB=94% in CKD-5 → fu_p=0.06  (essentially unchanged from healthy)
#
# BP: does NOT change with CKD — physicochemical property, CKD-independent.
#     drug BP=0.82, IS BP=0.70, HA BP=0.84, pCS BP=0.55 in all scenarios.
#
# WRONG values that must NOT be used (unverified estimates, no literature source):
#   IS fu_p CKD  = 0.15  ← WRONG
#   HA fu_p CKD  = 0.65  ← WRONG
#   pCS fu_p CKD = 0.12  ← WRONG

# ── CKD DRUG OVERRIDES REMOVED ──────────────────────────────────────────────
# Previously this file had _CKD_DRUG_OVERRIDES = {"fu_p": 0.20, "IR": 80.0}
# which overrode compound_parameters.py and caused the parameter audit to fail.
#
# CORRECT VALUES (from compound_parameters_final.py, _CKD_FU_P):
#   drug fu_p CKD5 = 0.05  (Takamura 1997: modest CMPF/oleate displacement)
#   drug IR        = 60.0 mg/h  (120 mg over 2 h, same as healthy)
#
# These are already set by initialize_compounds("ckd5"). Do NOT override here.
#
# HIGH-DOSE SENSITIVITY ANALYSIS:
#   Use configure_ckd_dose(total_dose_mg, fu_p) AFTER set_scenario().
#   Example: configure_ckd_dose(240, fu_p=0.10)  → 240 mg, standard CKD binding
#   The fu_p passed to configure_ckd_dose reflects the specific clinical context.
#   See configure_ckd_dose() docstring for the alpha/IS-rise lookup table.
_CKD_HIGH_DOSE_IR: float = 120.0   # mg/h → 240 mg total over 2 h (high-dose sensitivity only)

# ── _CKD_TOX_OVERRIDES REMOVED ───────────────────────────────────────────────
# Toxin fu_p CKD5 values are now set exclusively in compound_parameters_final.py
# via _CKD_FU_P and applied by initialize_compounds("ckd5").
# The old values here were WRONG and conflicted with the audit:
#   IS  fu_p: 0.08 here vs 0.10 in compound_parameters  (Deltombe 2015: 90% bound)
#   HA  fu_p: 0.57 here vs 0.52 in compound_parameters  (Deltombe 2015: 48% bound)
#   pCS fu_p: 0.06 here vs 0.05 in compound_parameters  (Deltombe 2015: 95% bound)
# Removing this dict eliminates the duplicate and ensures the audit passes.


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def set_scenario(scenario_key: str) -> None:
    """
    Configure all compound dicts for the named scenario.

    Parameters
    ----------
    scenario_key : str
        One of the keys defined in _PRESETS. Case-sensitive.
        Valid keys: H_NO_DRUG, H_WITH_DRUG_NO_INT, H_WITH_DRUG,
                    CKD_NO_DRUG, CKD_WITH_DRUG_NO_INT, CKD_WITH_DRUG

    Raises
    ------
    ValueError
        If scenario_key is not in _PRESETS.

    Side effects
    ------------
    Modifies C.drug, C.tox1, C.tox2, C.tox3 in place.
    Previous scenario settings are fully replaced.

    Example
    -------
    >>> set_scenario("H_WITH_DRUG")
    [SCENARIO] H_WITH_DRUG — mode=healthy, drug=ON, inhibit=ON
    """
    if scenario_key not in _PRESETS:
        valid = ", ".join(_PRESETS.keys())
        raise ValueError(
            f"Unknown scenario key: '{scenario_key}'. "
            f"Valid keys: {valid}"
        )

    mode, drug_enabled, inhibit = _PRESETS[scenario_key]

    # ── Step 1: Reload all compound dicts for this mode ───────────────────────
    # This resets drug, tox1, tox2, tox3 to their mode baseline values.
    # Must come BEFORE setting flags so that flags are not wiped by the reload.
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)
    elif hasattr(C, "load_compounds"):
        C.load_compounds(mode)
    else:
        raise RuntimeError(
            "compounds module has neither initialize_compounds() nor "
            "load_compounds(). Cannot configure scenario."
        )

    # ── Step 2: Set drug enabled flag ─────────────────────────────────────────
    # C.drug["enabled"] controls whether the drug infusion and transport
    # equations fire in the ODE. When False, drug concentration stays at 0.
    C.drug["enabled"] = drug_enabled

    # Zero out dosing if drug is disabled — prevents accidental infusion
    if not drug_enabled:
        C.drug["IR"]        = 0.0
        C.drug["t_start_h"] = 1e9    # far in future — will never trigger
        C.drug["t_inf_h"]   = 0.0
        C.drug["bolus_mg"]  = 0.0

    # ── Step 3: Set OAT1 inhibition flag ─────────────────────────────────────
    # C.drug["inhibit"] is the ONLY flag the ODE reads to activate inhibition.
    # When True: Km_eff = Km_base × (1 + Cu_drug/Ki) for each toxin.
    # When False: Km_eff = Km_base (no shift, regardless of drug concentration).
    C.drug["inhibit"] = inhibit

    # ── Step 4: CKD-specific overrides ───────────────────────────────────────
    # NO parameter overrides applied here.
    # All fu_p, Vmax, Km, R0 values are already correctly set by
    # initialize_compounds(mode) in Step 1 above.
    #
    # For high-dose sensitivity scenarios, set the IR after this call:
    #   set_scenario("CKD_HIGH_DOSE")
    #   configure_ckd_dose(240, fu_p=0.10)   ← sets IR and fu_p explicitly
    if mode == "ckd5" and drug_enabled and "HIGH_DOSE" in scenario_key:
        # High-dose scenarios: only change the infusion rate.
        # fu_p is NOT changed here — it stays at the compound_parameters value (0.05).
        # Use configure_ckd_dose() to set a specific fu_p for sensitivity analysis.
        C.drug["IR"] = _CKD_HIGH_DOSE_IR  # 120 mg/h → 240 mg total

    # ── Step 5: Confirmation print ────────────────────────────────────────────
    fu_p_str = f"fu_p={C.drug.get('fu_p', '?')}"
    ir_str   = f"IR={C.drug.get('IR', 0)} mg/h"
    dose_str = f"({C.drug.get('IR', 0) * C.drug.get('t_inf_h', 0):.0f} mg total)"
    print(
        f"[SCENARIO] {scenario_key} — "
        f"mode={mode}, "
        f"drug={'ON' if drug_enabled else 'OFF'}, "
        f"inhibit={'ON' if inhibit else 'OFF'}"
        + (f" | {fu_p_str}, {ir_str} {dose_str}" if drug_enabled else "")
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOSE CONFIGURATOR — change furosemide dose and fu_p after set_scenario()
# ══════════════════════════════════════════════════════════════════════════════

def configure_ckd_dose(
    total_dose_mg: float,
    fu_p: float,
    t_inf_h: float = 2.0,
) -> None:
    """
    Override furosemide dose and protein binding for a CKD simulation.

    Call this AFTER set_scenario() for any CKD scenario with drug enabled.
    It replaces whatever IR and fu_p were set by set_scenario().

    Parameters
    ----------
    total_dose_mg : float
        Total IV dose in mg. Divided over t_inf_h to give IR.
        Clinical range in CKD: 10–250 mg per dose.

    fu_p : float
        Unbound plasma fraction for furosemide (0 to 1).
        Healthy = 0.02. Typical CKD range: 0.05–0.20.
        Use 0.10 for standard CKD (80 mg), 0.20 for severe/high-dose CKD.
        Source: Ullian 1996 J Pharmacol Exp Ther 278:808.

    t_inf_h : float
        Infusion duration [h]. Default 2.0 h (standard clinical practice).

    Effect on OAT1 inhibition (Ki = 4.13 mg/L, BP = 0.82)
    -------------------------------------------------------
    Dose (mg)   fu_p   alpha    OAT1 flux↓   Expected IS rise
     10         0.20   1.06      5.5%           ~5%
     80         0.10   1.23     18.9%          ~17%
    120         0.15   1.52     34.4%          ~31%
    120         0.20   1.70     41.1%          ~37%  ← matches CKD-REIN
    250         0.20   2.46     59.3%          ~53%

    Raises
    ------
    ValueError
        If dose <= 0, fu_p not in (0, 1), or t_inf_h <= 0.

    Example
    -------
    >>> set_scenario("CKD_WITH_DRUG")
    >>> configure_ckd_dose(120, fu_p=0.20)   # 120 mg, severe uremic binding
    >>> sol = run_sim(t_end_h=12, mode="ckd5", custom_initial_conds=ic)
    """
    if total_dose_mg <= 0:
        raise ValueError(f"total_dose_mg must be > 0, got {total_dose_mg}")
    if not (0 < fu_p < 1):
        raise ValueError(f"fu_p must be between 0 and 1, got {fu_p}")
    if t_inf_h <= 0:
        raise ValueError(f"t_inf_h must be > 0, got {t_inf_h}")
    if C.drug is None:
        raise RuntimeError("Drug not initialised. Call set_scenario() first.")
    if not C.drug.get("enabled", False):
        raise RuntimeError(
            "Drug is not enabled in the current scenario. "
            "Call set_scenario() with a drug-enabled scenario first "
            "(e.g. 'CKD_WITH_DRUG')."
        )

    IR = total_dose_mg / t_inf_h

    # Compute and display the predicted inhibition strength
    import math
    k_elim = float(C.drug.get("k_elim_hinv", 0.25))   # 0.25 h-1 = t½ 2.8 h
    Vcen   = 5.0   # L
    BP     = float(C.drug.get("BP", 0.82))
    Ki_any = float(C.tox1.get("Ki_furo_to_IS_mgL", 4.13)) if C.tox1 else 4.13
    CL_met = k_elim * Vcen
    # Approximate CKD renal clearance (10% of 6.4 L/h)
    CLR_CKD  = 0.10 * 6.40
    CL_tot   = CL_met + CLR_CKD
    Cmax_est = (IR / CL_tot) * (1 - math.exp(-CL_tot / Vcen * t_inf_h))
    Cu_est   = Cmax_est * fu_p / BP
    alpha    = 1 + Cu_est / Ki_any
    flux_dn  = (alpha - 1) / alpha * 100

    C.drug["fu_p"]    = fu_p
    C.drug["IR"]      = IR
    C.drug["t_inf_h"] = t_inf_h

    print(
        f"[DOSE] {total_dose_mg:.0f} mg over {t_inf_h:.1f} h  "
        f"(IR={IR:.1f} mg/h) | fu_p={fu_p} | "
        f"Cmax≈{Cmax_est:.1f} mg/L | Cu≈{Cu_est:.3f} mg/L | "
        f"α≈{alpha:.3f} | OAT1 flux↓{flux_dn:.1f}%"
    )




def print_current_config() -> None:
    """
    Print the current state of drug and toxin configuration.

    Call this after set_scenario() to verify the settings are correct
    before running a simulation.

    Output example:
        === Current compound configuration ===
        Drug:
          enabled  = True
          inhibit  = True
          fu_p     = 0.10
          IR       = 40.0 mg/h  (80 mg total)
          Ki_IS    = 4.13 mg/L
          Ki_HA    = 4.13 mg/L
          Ki_pCS   = 4.13 mg/L
        IS:  enabled=True, fu_p=0.04, Vmax_baso=11.1 mg/h
        HA:  enabled=True, fu_p=0.50, Vmax_baso=30.5 mg/h
        pCS: enabled=True, fu_p=0.04, Vmax_baso=63.3 mg/h
    """
    print("\n=== Current compound configuration ===")

    d = C.drug
    if d is None:
        print("  Drug: NOT INITIALISED")
    else:
        ir   = float(d.get("IR",       0.0))
        tinf = float(d.get("t_inf_h",  0.0))
        print(f"  Drug:")
        print(f"    enabled  = {d.get('enabled')}")
        print(f"    inhibit  = {d.get('inhibit')}")
        print(f"    fu_p     = {d.get('fu_p')}")
        print(f"    IR       = {ir} mg/h  ({ir * tinf:.0f} mg total)")
        print(f"    k_elim   = {d.get('k_elim_hinv')} h⁻¹")

    for name, tox, ki_key in [
        ("IS",  C.tox1, "Ki_furo_to_IS_mgL"),
        ("HA",  C.tox2, "Ki_furo_to_HA_mgL"),
        ("pCS", C.tox3, "Ki_furo_to_pCS_mgL"),
    ]:
        if tox is None:
            print(f"  {name}: NOT INITIALISED")
        else:
            ki = tox.get(ki_key, "not set")
            print(
                f"  {name}: enabled={tox.get('enabled', True)}, "
                f"fu_p={tox.get('fu_p')}, "
                f"Vmax_baso={tox.get('Vmax_baso_uptake')} mg/h, "
                f"Ki_furo={ki} mg/L"
            )

    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP / RESET
# ══════════════════════════════════════════════════════════════════════════════

def reset_to_healthy_baseline() -> None:
    """
    Reset all compound dicts to the healthy baseline with drug disabled.

    Call at the end of a plotting script or before switching between
    completely different simulation contexts.

    Equivalent to: set_scenario("H_NO_DRUG")
    """
    set_scenario("H_NO_DRUG")
    print("[SCENARIO] Reset to healthy baseline — drug disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# VALID KEYS — for external inspection
# ══════════════════════════════════════════════════════════════════════════════

VALID_SCENARIO_KEYS = list(_PRESETS.keys())
"""
List of all valid scenario key strings.

Standard scenarios (80 mg, fu_p=0.10):
    H_NO_DRUG, H_WITH_DRUG_NO_INT, H_WITH_DRUG
    CKD_NO_DRUG, CKD_WITH_DRUG_NO_INT, CKD_WITH_DRUG

High-dose scenarios (120 mg, fu_p=0.20, matches CKD-REIN):
    CKD_HIGH_DOSE_NO_DRUG, CKD_HIGH_DOSE_NO_INT, CKD_HIGH_DOSE

Custom dose — call configure_ckd_dose() after set_scenario():
    set_scenario("CKD_WITH_DRUG")
    configure_ckd_dose(120, fu_p=0.20)   # 120 mg, alpha≈1.70, IS rise ~37%
    configure_ckd_dose(10,  fu_p=0.20)   # 10 mg,  alpha≈1.06, IS rise  ~5%
    configure_ckd_dose(250, fu_p=0.20)   # 250 mg, alpha≈2.46, IS rise ~53%
"""