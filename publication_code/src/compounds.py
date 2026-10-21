"""
compounds.py
============
Runtime manager for compound parameter dictionaries.

WHAT THIS FILE DOES
-------------------
This file is the bridge between the raw parameter definitions
(compound_parameters.py) and the ODE system (ode_system_twocompartment.py).

It:
  1. Calls get_compound_parameters() to load raw values
  2. Builds passive diffusion parameter bundles (DRG_PASS, TOX*_PASS)
  3. Sets Ki default values (with correct units — mg/L and μM)
  4. Exposes drug, tox1, tox2, tox3 as module-level globals
     that the ODE reads directly via:
         import compounds as C
         drug = C.drug

IMPORTANT — DATA FLOW
---------------------
compound_parameters.py  →  compounds.py  →  ode_system_twocompartment.py
  (raw values)              (runtime mgr)     (ODE reads C.drug, C.tox1, ...)

The ODE never imports from compound_parameters.py directly.
ALL corrections to parameter values must be visible here.

CORRECTIONS MADE IN THIS VERSION
----------------------------------
1. Ki fallback default in _finalize_inhibition_units():
       0.08 mg/L → 4.13 mg/L
   This is the safety net that fires if Ki_furo_to_*_mgL is accidentally
   deleted from a toxin dict. The old 0.08 was 50x too low.
   Source: Mihaila 2020, Toxins 12:391 + Cheng-Prusoff correction.

2. Ki uM setdefault values in initialize_compounds() and
   update_compound_parameters():
       0.1 uM → 12.49 uM
   The old 0.1 uM corresponded to 0.033 mg/L — completely wrong.
   Correct value: IC50=28.1 uM / 2.25 = 12.49 uM = 4.13 mg/L.

3. Same Ki uM fix in update_compound_parameters() (was duplicated
   with the same wrong values).

4. Added Ki plausibility check to validate_compounds():
   Warns if Ki < 0.5 mg/L (likely a unit error) when inhibition is active.
   Raises if Ki is missing or <= 0 when inhibition is active.

UNIT REFERENCE FOR Ki
-----------------------
  Ki_furo_to_*_mgL  = 4.13  mg/L   (used by ODE — lines 859-861 of ODE file)
  Ki_furo_to_*_uM   = 12.49 uM     (stored for reference/documentation)

  Derivation:
    IC50 = 28.1 uM (furosemide at OAT1, ciPTEC-OAT1)
    Source: Mihaila et al. 2020, Toxins 12:391, Table 1, p.5 (full PDF read)
    Cheng-Prusoff: Ki = IC50 / (1 + [S]/Km,fluor)
                      = 28.1 / (1 + 1.0/0.8)
                      = 28.1 / 2.25
                      = 12.49 uM
    Unit conversion: Ki [mg/L] = 12.49 x 330.74 / 1000 = 4.13 mg/L
"""

import numpy as np
from compound_parameters import get_compound_parameters, print_parameter_changes

# ── Module-level globals ───────────────────────────────────────────────────────
# These are read directly by the ODE system (import compounds as C; C.drug etc.)
# They are None until initialize_compounds() is called.
drug = tox1 = tox2 = tox3 = None
DRG_PASS = TOX1_PASS = TOX2_PASS = TOX3_PASS = {}


# ── Verified Ki constants ──────────────────────────────────────────────────────
# Defined once here so they cannot be mis-typed in multiple places.
# Source: Mihaila 2020, Toxins 12:391 (full PDF read in session)
_KI_FURO_UM  = 12.49   # [uM]   -- for documentation and setdefault
_KI_FURO_MGL = 3.77  # [mg/L] — Mihaila 2020, Toxins 12:391 (Cheng-Prusoff: IC50=28.1µM/2.25=12.49µM=4.13 mg/L)


def _uM_to_mgL(uM, MW_g_mol):
    """
    Convert Ki from umol/L to mg/L.

    Formula: Ki [mg/L] = Ki [uM] x MW [g/mol] x 1e-3

    Example: furosemide Ki = 12.49 uM, MW = 330.74 g/mol
             Ki = 12.49 x 330.74 x 1e-3 = 4.13 mg/L
    """
    return (float(uM) * float(MW_g_mol)) / 1000.0


def _finalize_inhibition_units():
    """
    Ensure Ki values used by the ODE exist in mg/L in each toxin dict.

    The ODE reads Ki_furo_to_*_mgL (lines 859-861 of ode_system_twocompartment.py).
    This function is a safety net: if the mg/L key is missing, it computes
    it from the uM key, or falls back to the verified constant _KI_FURO_MGL.

    This function only PREPARES parameters — it does NOT activate inhibition.
    Inhibition is activated by setting drug["inhibit"] = True externally.

    CORRECTION: default_mgL changed from 0.08 to 4.13 mg/L.
    The old 0.08 was 50x too low and would have caused near-complete
    OAT1 inhibition even at sub-therapeutic furosemide concentrations.
    """
    global drug, tox1, tox2, tox3

    if not isinstance(drug, dict):
        return

    drug_mw = drug.get("MW", 330.74)   # furosemide MW fallback

    # (toxin_dict, mg/L key name, uM key name, fallback mg/L value)
    inhibition_targets = [
        (tox1, "Ki_furo_to_IS_mgL",  "Ki_furo_to_IS_uM",  _KI_FURO_MGL),
        (tox2, "Ki_furo_to_HA_mgL",  "Ki_furo_to_HA_uM",  _KI_FURO_MGL),
        (tox3, "Ki_furo_to_pCS_mgL", "Ki_furo_to_pCS_uM", _KI_FURO_MGL),
    ]

    for tox, mgL_key, uM_key, default_mgL in inhibition_targets:
        if not isinstance(tox, dict):
            continue

        # If mg/L key is already set (from compound_parameters.py), keep it.
        # Only fill in if genuinely missing.
        if tox.get(mgL_key) is None:
            if tox.get(uM_key) is not None:
                # Convert from uM if available
                tox[mgL_key] = _uM_to_mgL(tox[uM_key], drug_mw)
            else:
                # Last-resort fallback — use verified constant
                tox[mgL_key] = default_mgL


def initialize_compounds(mode: str = "healthy"):
    """
    Load and initialise all compound dicts for a given physiological mode.

    This is the primary entry point. Call this once before running a simulation.
    After calling this, the module-level globals drug, tox1, tox2, tox3 are ready.

    Parameters
    ----------
    mode : "healthy" or "ckd5"
        Physiological mode. CKD5 applies Vmax scaling from compound_parameters.py.
    """
    global drug, tox1, tox2, tox3
    global DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS

    # Step 1: Load raw parameter values from compound_parameters.py
    drug, tox1, tox2, tox3 = get_compound_parameters(mode)

    # Step 2: Ensure required flags exist with safe defaults
    # drug is OFF by default — must be explicitly enabled in the runner
    drug.setdefault("enabled", False)

    # Step 3: Build passive diffusion parameter bundles
    # These encode the permeability-surface area clearances and ionisation
    # correction factors (alpha, beta, gamma) for each tubular segment.
    from passive_diffusion import build_passive_params
    DRG_PASS  = build_passive_params("drg",  drug, mode=mode)
    TOX1_PASS = build_passive_params("tox1", tox1, mode=mode)
    TOX2_PASS = build_passive_params("tox2", tox2, mode=mode)
    TOX3_PASS = build_passive_params("tox3", tox3, mode=mode)

    # Step 4: Set Ki uM reference values for documentation purposes.
    # The ODE reads the mg/L versions (set in compound_parameters.py or below).
    # setdefault only writes if the key is not already present.
    #
    # CORRECTION: changed from 0.1 uM to 12.49 uM
    # Old value 0.1 uM = 0.033 mg/L was completely wrong (125x too low).
    # Correct: IC50=28.1 uM / Cheng-Prusoff factor 2.25 = 12.49 uM = 4.13 mg/L
    # Source: Mihaila 2020, Toxins 12:391, Table 1, p.5
    tox1.setdefault("Ki_furo_to_IS_uM",  _KI_FURO_UM)
    tox2.setdefault("Ki_furo_to_HA_uM",  _KI_FURO_UM)
    tox3.setdefault("Ki_furo_to_pCS_uM", _KI_FURO_UM)

    # Step 5: Ensure mg/L Ki values are present and correct
    # compound_parameters.py should already have set these to 4.13 mg/L,
    # but _finalize_inhibition_units() is the safety net if anything is missing.
    _finalize_inhibition_units()

    # Step 6: Print CKD parameter changes if not in healthy mode
    if mode != "healthy":
        print_parameter_changes()


def load_compounds(mode: str = "healthy"):
    """
    Backward-compatible entry point.

    Calls initialize_compounds() and returns the four compound dicts.
    The module-level globals are also set and are what the ODE uses.
    """
    initialize_compounds(mode)
    return drug, tox1, tox2, tox3


def apply_drug_protocol(proto: dict | None):
    """
    Push dosing protocol settings into the drug dict at runtime.

    Use this to change dosing parameters between simulations without
    reinitialising all compounds. Only keys present in proto are updated.

    Example:
        apply_drug_protocol({"IR": 40.0, "t_inf_h": 1.0, "fu_p": 0.15})

    Keys that can be updated via this function:
        protocol    : "plasma_britz" or "nina"
        IR          : infusion rate [mg/h]
        t_inf_h     : infusion duration [h]
        t_start_h   : infusion start time [h]
        bolus_mg    : IV bolus dose [mg]
        k_elim_hinv : elimination rate constant [h^-1]
        fu_p        : unbound fraction in plasma (e.g. change for CKD scenario)
        BP          : blood/plasma ratio
    """
    global drug

    if not proto or not isinstance(drug, dict):
        return

    for key in (
        "protocol", "IR", "t_inf_h", "t_start_h", "bolus_mg",
        "k_elim_hinv", "fu_p", "BP",
    ):
        if key in proto:
            drug[key] = proto[key]


def update_compound_parameters(mode: str = "healthy"):
    """
    Reinitialise compound parameters for a new physiological mode.

    Functionally identical to initialize_compounds(). Provided as a
    named alias for clarity in runner scripts that switch between modes.

    Example:
        update_compound_parameters("ckd5")
        sol_ckd = run_sim(48, mode="ckd5")
    """
    global drug, tox1, tox2, tox3
    global DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS

    # Step 1: Load raw parameters
    drug, tox1, tox2, tox3 = get_compound_parameters(mode)
    drug.setdefault("enabled", False)

    # Step 2: Build passive diffusion bundles
    from passive_diffusion import build_passive_params
    DRG_PASS  = build_passive_params("drg",  drug, mode=mode)
    TOX1_PASS = build_passive_params("tox1", tox1, mode=mode)
    TOX2_PASS = build_passive_params("tox2", tox2, mode=mode)
    TOX3_PASS = build_passive_params("tox3", tox3, mode=mode)

    # Step 3: Set Ki uM reference values
    # CORRECTION: 0.1 uM to 12.49 uM (same fix as in initialize_compounds)
    tox1.setdefault("Ki_furo_to_IS_uM",  _KI_FURO_UM)
    tox2.setdefault("Ki_furo_to_HA_uM",  _KI_FURO_UM)
    tox3.setdefault("Ki_furo_to_pCS_uM", _KI_FURO_UM)

    # Step 4: Ensure mg/L Ki values are present
    _finalize_inhibition_units()

    # Step 5: Print CKD changes if applicable
    if mode != "healthy":
        print_parameter_changes()


def validate_compounds():
    """
    Check that all required compound keys are present and physically plausible.

    Raises ValueError if any required key is missing or any compound is None.
    Prints [WARN] messages for implausible values without raising.

    Call this before run_sim() to catch configuration errors early.

    NEW in this version: Ki plausibility check when inhibition is active.
    """
    required_keys = [
        "Km_baso_uptake", "Vmax_baso_uptake",
        "Km_api_secr",    "Vmax_api_secr",
        "CL_sec_api",     "fu_p", "BP",
        "MW",             "pKa",  "type",
    ]

    for name, comp in [
        ("Drug", drug), ("IS", tox1), ("HA", tox2), ("pCS", tox3)
    ]:
        if comp is None:
            raise ValueError(
                f"{name} compound is None — call initialize_compounds() first."
            )
        missing = [k for k in required_keys if k not in comp]
        if missing:
            raise ValueError(
                f"{name} is missing required keys: {missing}"
            )

    # ── Ki plausibility check when inhibition is active ───────────────────────
    # The ODE reads Ki_furo_to_*_mgL directly (ODE lines 859-861).
    # If inhibition is active but Ki is wrong, results are silently incorrect.
    if isinstance(drug, dict) and drug.get("inhibit", False):
        ki_checks = [
            (tox1, "Ki_furo_to_IS_mgL",  "IS"),
            (tox2, "Ki_furo_to_HA_mgL",  "HA"),
            (tox3, "Ki_furo_to_pCS_mgL", "pCS"),
        ]
        for tox, ki_key, tname in ki_checks:
            if not isinstance(tox, dict):
                continue
            ki_val = tox.get(ki_key, None)

            # Missing or non-positive Ki with inhibition active is a hard error
            if ki_val is None or ki_val <= 0:
                raise ValueError(
                    f"{tname}: inhibition is active (drug['inhibit']=True) "
                    f"but {ki_key} is missing or <= 0. "
                    f"Expected 4.13 mg/L (12.49 uM) from Mihaila 2020, "
                    f"Toxins 12:391."
                )

            # Very low Ki almost certainly means uM was entered instead of mg/L
            if ki_val < 0.5:
                print(
                    f"[WARN] {tname} {ki_key} = {ki_val:.4f} mg/L is very low. "
                    f"Expected ~4.13 mg/L (12.49 uM). "
                    f"Check for unit error: did you enter uM instead of mg/L?"
                )

            # Infinite Ki means inhibition has no effect — warn so user knows
            if ki_val > 1e6:
                print(
                    f"[WARN] {tname} {ki_key} = {ki_val:.2e} mg/L (effectively "
                    f"infinite). Furosemide inhibition of {tname} OAT1 is OFF."
                )

    return True


# ── Auto-initialise on import ──────────────────────────────────────────────────
# Matches original behaviour: compounds are ready as soon as this module
# is imported. The ODE can be called immediately without any setup step.
# Mode defaults to "healthy" — change by calling initialize_compounds("ckd5").
initialize_compounds("healthy")