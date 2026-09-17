# simulation.py
# -*- coding: utf-8 -*-
"""
simulation.py
=============
Simulation runner and post-run utilities for the kidney PBPK model.

WHAT THIS FILE DOES
-------------------
This is the central entry point for running simulations. It:

  1. Validates compound parameters (via compounds.validate_compounds)
  2. Builds physiology and tubular flow chain (via physiology.make_phys)
  3. Sets initial conditions (via state_vector.initial_conditions)
  4. Calls scipy solve_ivp with the two-compartment ODE
  5. Returns the OdeResult object for downstream analysis and plotting

PUBLIC API
----------
  run_sim(t_end_h, ...)       → OdeResult   (primary entry point)
  extract_results(sol, ...)   → dict         (convenience extractor)
  verify_tissue_distribution(sol, ...) → bool

IMPORT PATH FOR OTHER SCRIPTS
------------------------------
  from simulation import run_sim          ← this is what plotting scripts use
  from simulation import extract_results
  from simulation import verify_tissue_distribution

UNIT CONVENTIONS
----------------
  Concentrations : mg/L
  Flows          : L/h
  Time           : h
  Amounts (Ae)   : mg

ODE SOLVER
----------
  Method : Radau (stiff solver — required for the multi-compartment kidney model)
  rtol   : K.RTOL (default 1e-7)
  atol   : K.ATOL (default 1e-9)
  t_eval : uniform grid of n_steps+1 points from 0 to t_end_h

TWO-COMPARTMENT DRUG DISTRIBUTION
----------------------------------
  The ODE includes a peripheral tissue compartment (C_peri) for the drug.
  Its dynamics are controlled by Vperi_L, k12_hinv, k21_hinv in the drug
  compound dict, and by use_tissue=True/False.
  Toxins do NOT have a peripheral compartment.

CORRECTIONS IN THIS VERSION
-----------------------------
  1. run_sim() is defined and exported in this file.
     Previous version: run_sim was defined in a separate runner file and
     simulation.py did not export it, causing:
       ImportError: cannot import name 'run_sim' from 'simulation'

  2. The ODE import is explicitly from ode_system_twocompartment:
       from ode_system_twocompartment import kidney_rhs
     This ensures the two-compartment ODE is always used regardless of
     any legacy 'ode_system.py' that may exist in the project.

  3. kidney_rhs._printed_debug is reset before each run so that per-run
     debug prints fire correctly for every simulation call.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

# Project modules — import the module, not individual names, to prevent
# stale flags when constants are updated at runtime.
import constants as K
import compounds as C
from physiology import make_phys, flows_L_per_h
from state_vector import initial_conditions, IDX


# ══════════════════════════════════════════════════════════════════════════════
# Primary simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(
    t_end_h: float,
    custom_initial_conds: Optional[Dict[str, float]] = None,
    mode: str = "healthy",
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    **kwargs,
):
    """
    Integrate the kidney PBPK ODE system using SciPy Radau.

    This is the primary entry point for all simulations. Call this from
    plotting scripts, analysis scripts, or interactively.

    Parameters
    ----------
    t_end_h : float
        Simulation end time [h].

    custom_initial_conds : dict or None
        Override specific initial conditions. Keys are state names (from IDX),
        values are concentrations [mg/L] or amounts [mg] for Ae states.
        Any state not in this dict uses the default from initial_conditions().
        Typical use: pass steady-state endpoint from a pre-run as custom_ic.

    mode : str
        Physiological mode. "healthy" or "ckd5".
        Controls GFR, renal blood flow, and Vmax scaling.

    dt : float or None
        Time step for output grid [h]. Used to compute n_steps if n_steps
        is not provided. For smooth curves, use dt=0.02 h. For pre-runs,
        dt=0.5 h is sufficient.

    n_steps : int or None
        Number of time points in the output grid. If both dt and n_steps
        are None, defaults to 10000 steps.

    rtol, atol : float
        Relative and absolute tolerances for the Radau solver.
        Defaults (1e-7, 1e-9) are tighter than scipy defaults for accuracy.

    **kwargs : dict
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    scipy.integrate.OdeResult
        sol.t  : time vector [h], shape (n_steps+1,)
        sol.y  : state matrix [units depend on state], shape (n_states, n_steps+1)
        sol.success : bool
        Use IDX dict to extract specific states: sol.y[IDX["C_ven"]]

    Raises
    ------
    ValueError
        If compound validation fails and cannot be recovered.
    RuntimeError
        If the ODE integration fails.

    Example
    -------
    >>> from simulation import run_sim
    >>> sol = run_sim(48.0, mode="healthy")
    >>> import numpy as np
    >>> from state_vector import IDX
    >>> C_ven = sol.y[IDX["C_ven"]]   # furosemide plasma [mg/L]
    """

    # ── Step 1: Build time grid ────────────────────────────────────────────────
    if n_steps is None:
        if dt is None or dt <= 0.0:
            n_steps = 10000
        else:
            n_steps = max(1, int(math.ceil(t_end_h / dt)))

    t_eval = np.linspace(0.0, t_end_h, n_steps + 1)

    # ── Step 2: Validate compounds ────────────────────────────────────────────
    # validate_compounds() checks for missing required keys ONLY.
    # The caller (e.g. furosemide_pk._apply_drug_params) is responsible for
    # setting C.drug to the correct mode BEFORE calling run_sim().
    #
    # CRITICAL FIX: The old fallback called C.initialize_compounds(mode) on
    # any validation failure. This silently overwrote CKD-5 values that the
    # caller had just set (k_elim=0.12, Vmax_baso=900) with cached healthy
    # values. CKD-5 then ran with healthy parameters — explaining why
    # t_half_CKD5 < t_half_healthy in the output (physically impossible).
    #
    # New behaviour: reinitialise only if C.drug is None/empty (not set at
    # all). If keys are present, trust the caller unconditionally.
    _drug_already_set = isinstance(C.drug, dict) and len(C.drug) > 5
    try:
        C.validate_compounds()
    except ValueError as e:
        if _drug_already_set:
            # Keys present — caller set them intentionally. Keep them.
            print(f"[run_sim] validate_compounds note (caller dict kept): {e}")
        else:
            # C.drug is None or empty — genuinely uninitialised.
            print(f"[run_sim] C.drug empty/None — initialising for mode={mode}")
            if hasattr(C, "initialize_compounds"):
                C.initialize_compounds(mode)
            elif hasattr(C, "load_compounds"):
                C.load_compounds(mode)
            C.validate_compounds()  # raise if still failing

    # ── Step 2b: Parameter audit ───────────────────────────────────────────────
    # CRITICAL FIX: parameter_audit.EXPECTED_PARAMS is built at module import
    # time (_build_registry() at line 377 of parameter_audit.py). If
    # compound_parameters.py was edited after first import (e.g. k_elim
    # changed from 0.05 to 0.12 for CKD-5), the registry still holds the
    # stale expected value. audit_compounds then raises ValueError because
    # C.drug["k_elim_hinv"]=0.12 != expected=0.05 — triggering the
    # reinit fallback that would destroy our CKD-5 dict.
    # importlib.reload forces _build_registry() to re-read the current file.
    import importlib
    import parameter_audit as _pa_mod
    importlib.reload(_pa_mod)                    # rebuild EXPECTED_PARAMS fresh
    from parameter_audit import audit_compounds  # re-import the reloaded symbol

    _phys_for_audit = make_phys(mode)
    # strict=False: audit prints/warns but NEVER raises ValueError.
    # The caller's hard assertions in _apply_drug_params are the safety net.
    # strict=True would re-introduce the overwrite bug if expected values drift.
    _audit_strict = kwargs.get("audit_strict", False)
    audit_compounds(C.drug, C.tox1, C.tox2, C.tox3, _phys_for_audit,
                    scenario=mode, strict=_audit_strict)

    # ── Step 3: Build physiology and flow chain ────────────────────────────────
    phys  = make_phys(mode)
    flows = flows_L_per_h(phys)
    # ── Step 3: Build physiology and flow chain ────────────────────────────────
    phys  = make_phys(mode)
    flows = flows_L_per_h(phys)

    # ── Step 4: Build initial conditions ──────────────────────────────────────
    # Reset cumulative excretion (Ae_*) in custom_initial_conds to zero.
    # This is essential for correct CLR calculation in the main runs.
    #
    # The warm-up simulation accumulates large Ae values (e.g. Ae_t1 ~2×10⁴ mg).
    # If these are carried into the 12h main run, ΔAe over 12h is tiny relative
    # to the accumulated baseline, making CLR = ΔAe/AUC appear near-zero.
    #
    # Resetting Ae to zero at t=0 of each main run gives a clean ΔAe that
    # correctly represents secretion during the 12h window only.
    #
    # Toxin concentrations, drug compartments, and all other state variables
    # are carried over from the warm-up (preserving the CKD steady state).
    if custom_initial_conds is not None:
        for key in list(custom_initial_conds.keys()):
            if key.startswith("Ae_"):
                custom_initial_conds[key] = 0.0

    y0 = initial_conditions(phys, custom_initial_conds=custom_initial_conds)
    if custom_initial_conds is not None:
        # Force override for toxin central concentrations
        for toxin_state in ["C_tox1_cen", "C_tox2_cen", "C_tox3_cen"]:
            if toxin_state in custom_initial_conds and toxin_state in IDX:
                idx = IDX[toxin_state]
                y0[idx] = custom_initial_conds[toxin_state]
                if K.VERBOSE:
                    print(f"  [FORCED] {toxin_state} = {y0[idx]:.4f} mg/L")
    # ── Step 5: Verbose pre-run summary ───────────────────────────────────────
    if K.VERBOSE:
        print(f"\n=== SIMULATION START ===")
        print(
            f"  mode={mode}  |  t_end={t_end_h} h  |  n_steps={n_steps}  |  "
            f"drug.enabled={C.drug.get('enabled', False)}  |  "
            f"drug.inhibit={C.drug.get('inhibit', False)}  |  "
            f"USE_APICAL_MM={K.USE_APICAL_MM}"
        )
        print(
            f"  PHYSIOLOGY: Q_kidney={phys['Q_kidney0']:.2f} L/h  "
            f"GFR={phys['Q_GFR']:.3f} L/h  "
            f"Urine={phys['Q_urine']:.4f} L/h  "
            f"Vcen={phys['V_ven0']:.1f} L"
        )

        if custom_initial_conds:
            nonzero = {k: v for k, v in custom_initial_conds.items() if v != 0.0}
            if nonzero:
                print(f"  CUSTOM IC ({len(nonzero)} non-zero states):")
                for k, v in list(nonzero.items())[:5]:
                    print(f"    {k}: {v:.4g}")
                if len(nonzero) > 5:
                    print(f"    ... and {len(nonzero)-5} more")

        for nm, tox in [("IS", C.tox1), ("HA", C.tox2), ("pCS", C.tox3)]:
            if tox is not None:
                print(
                    f"  {nm}: C0={tox.get('C0', 0):.3f} mg/L  "
                    f"R0={tox.get('R0', 0):.2f} mg/h  "
                    f"fu_p={tox.get('fu_p', 0):.3f}  "
                    f"Vmax_baso={tox.get('Vmax_baso_uptake', 0):.1f} mg/h  "
                    f"Km={tox.get('Km_baso_uptake', 0):.3f} mg/L"
                )

        if C.drug is not None:
            d = C.drug
            print(
                f"  DRUG: enabled={d.get('enabled', False)}  "
                f"fu_p={d.get('fu_p', float('nan')):.3f}  "
                f"BP={d.get('BP', float('nan')):.2f}  "
                f"IR={d.get('IR', 0)} mg/h  "
                f"t0={d.get('t_start_h', 0)} h  "
                f"t_inf={d.get('t_inf_h', 0)} h  "
                f"k_elim={d.get('k_elim_hinv', 0):.4f} h-1  "
                f"Vperi={d.get('Vperi_L', 0)} L  "
                f"k12={d.get('k12_hinv', 0):.4f} h-1  "
                f"k21={d.get('k21_hinv', 0):.4f} h-1  "
                f"use_tissue={d.get('use_tissue', False)}"
            )

        print(
            f"  [DEBUG] drug.enabled={C.drug.get('enabled', False)}  "
            f"IR={C.drug.get('IR', 0)} mg/h  "
            f"t_start={C.drug.get('t_start_h', 0)} h  "
            f"t_inf={C.drug.get('t_inf_h', 0)} h  "
            f"→ infuses from {C.drug.get('t_start_h', 0)} to "
            f"{C.drug.get('t_start_h', 0) + C.drug.get('t_inf_h', 0)} h"
        )

    # ── Step 6: Reset ODE debug flags for this run ────────────────────────────
    # Flags are set on the kidney_rhs function object during integration.
    # Resetting them ensures per-run debug output fires on each call.
    try:
        from ode_system_twocompartment import kidney_rhs as _rhs
        _rhs._printed_debug = False
        _rhs._apical_guarded = False
        # Clear all per-timestep flags
        for attr in list(dir(_rhs)):
            if (
                attr.startswith("_printed_cin_")
                or attr.startswith("_checked_")
                or attr.startswith("_warned_zero_")
                or attr.startswith("_p1_lumen_")
                or attr in ("_flow_debug_done", "_dbg_last_bucket",
                            "_printed_peak_debug", "_tissue_dist")
            ):
                try:
                    delattr(_rhs, attr)
                except AttributeError:
                    pass
    except ImportError:
        pass   # non-fatal — debug output may be duplicated across runs

    # ── Step 7: Integrate ─────────────────────────────────────────────────────
    # Import inside the function so that any monkey-patching of the module
    # between calls is respected (e.g. if the user replaces kidney_rhs).
    from ode_system_twocompartment import kidney_rhs

    try:
        t_start_inf = float(C.drug.get("t_start_h", 0.0))
        t_end_inf = t_start_inf + float(C.drug.get("t_inf_h", 0.0))
        drug_active = (C.drug.get("enabled", False)
                       and float(C.drug.get("IR", 0.0)) > 0.0
                       and 0.0 < t_end_inf < t_end_h)

        if drug_active:
            # Phase 1: t=0 → t_end_inf (infusion on)
            n1 = max(2, int(round(n_steps * t_end_inf / t_end_h)))
            t_eval_1 = np.linspace(0.0, t_end_inf, n1 + 1)

            sol1 = solve_ivp(
                fun=lambda t, y: kidney_rhs(t, y, phys=phys, FLOWS_L_HR=flows),
                t_span=(0.0, t_end_inf),
                y0=y0,
                method="Radau",
                t_eval=t_eval_1,
                rtol=K.RTOL,
                atol=K.ATOL,
            )

            # Phase 2: t=t_end_inf → t_end_h (infusion off, smooth ODE)
            n2 = n_steps - n1
            t_eval_2 = np.linspace(t_end_inf, t_end_h, n2 + 1)
            y1 = sol1.y[:, -1]  # endpoint of phase 1 = initial conditions for phase 2

            sol2 = solve_ivp(
                fun=lambda t, y: kidney_rhs(t, y, phys=phys, FLOWS_L_HR=flows),
                t_span=(t_end_inf, t_end_h),
                y0=y1,
                method="Radau",
                t_eval=t_eval_2,
                rtol=K.RTOL,
                atol=K.ATOL,
            )

            # Merge: avoid duplicating the junction point
            from types import SimpleNamespace
            sol = SimpleNamespace(
                t=np.concatenate([sol1.t, sol2.t[1:]]),
                y=np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1),
                success=sol1.success and sol2.success,
                message=sol1.message if not sol1.success else sol2.message,
            )
        else:
            # No discontinuity — single solve
            sol = solve_ivp(
                fun=lambda t, y: kidney_rhs(t, y, phys=phys, FLOWS_L_HR=flows),
                t_span=(0.0, t_end_h),
                y0=y0,
                method="Radau",
                t_eval=t_eval,
                rtol=K.RTOL,
                atol=K.ATOL,
            )
    except Exception as e:
        raise RuntimeError(
            f"ODE integration failed: {e}\n"
            "Check compound parameters and physiology values."
        ) from e

    # ── Step 8: Post-run checks ───────────────────────────────────────────────
    if not sol.success:
        print(
            f"[WARN] run_sim: solver did not converge. "
            f"Message: {sol.message}. "
            f"Results may be unreliable."
        )

    if K.VERBOSE:
        print(f"\n  Simulation completed: {len(sol.t)} timepoints, "
              f"success={sol.success}")

    # Check drug actually reached plasma (only if drug was enabled)
    if C.drug.get("enabled", False) and "C_ven" in IDX:
        C_ven_max = float(np.nanmax(sol.y[IDX["C_ven"]]))
        if C_ven_max <= 1e-12:
            print(
                "\n[WARN] Drug enabled but C_ven stayed ~0 throughout. "
                "This usually means the infusion gate is not opening. "
                "Check: drug['IR'] > 0, drug['t_start_h'], drug['t_inf_h'], "
                "and that drug['enabled']=True was set BEFORE calling run_sim(). "
                "Also confirm modules use 'import constants as K' not "
                "'from constants import ...' to avoid stale flag reads."
            )
        else:
            print(
                f"\n[OK] Drug reached peak plasma: "
                f"{C_ven_max:.4e} mg/L at t="
                f"{sol.t[np.nanargmax(sol.y[IDX['C_ven']])]:.2f} h"
            )

    return sol


# ══════════════════════════════════════════════════════════════════════════════
# Post-run utilities
# ══════════════════════════════════════════════════════════════════════════════

def extract_results(sol, compound: str = "all") -> dict:
    """
    Extract named concentration and excretion time-series from a solution.

    Convenience wrapper around sol.y[IDX[...]] that returns a clean dict
    with human-readable keys. All arrays have the same length as sol.t.

    Parameters
    ----------
    sol      : OdeResult from run_sim()
    compound : "all", "drug", "IS", "HA", or "pCS"

    Returns
    -------
    dict with keys:
      "time"          : time vector [h]
      "plasma_drug"   : C_ven [mg/L]          (if compound in ["all","drug"])
      "tissue_drug"   : C_peri [mg/L]         (if compound in ["all","drug"])
      "bladder_drug"  : C_bladder [mg/L]      (if compound in ["all","drug"])
      "excreted_drug" : Ae_drg [mg]           (if compound in ["all","drug"])
      "plasma_IS"     : C_tox1_cen [mg/L]     (if compound in ["all","IS"])
      "excreted_IS"   : Ae_t1 [mg]            (if compound in ["all","IS"])
      "plasma_HA"     : C_tox2_cen [mg/L]     (if compound in ["all","HA"])
      "excreted_HA"   : Ae_t2 [mg]            (if compound in ["all","HA"])
      "plasma_pCS"    : C_tox3_cen [mg/L]     (if compound in ["all","pCS"])
      "excreted_pCS"  : Ae_t3 [mg]            (if compound in ["all","pCS"])
    """
    t = sol.t
    y = sol.y
    results: dict = {"time": t}

    if compound in ("all", "drug"):
        if "C_ven"     in IDX: results["plasma_drug"]   = y[IDX["C_ven"]]
        if "C_peri"    in IDX: results["tissue_drug"]   = y[IDX["C_peri"]]
        if "C_bladder" in IDX: results["bladder_drug"]  = y[IDX["C_bladder"]]
        if "Ae_drg"    in IDX: results["excreted_drug"] = y[IDX["Ae_drg"]]

    if compound in ("all", "IS"):
        if "C_tox1_cen" in IDX: results["plasma_IS"]   = y[IDX["C_tox1_cen"]]
        if "Ae_t1"      in IDX: results["excreted_IS"] = y[IDX["Ae_t1"]]

    if compound in ("all", "HA"):
        if "C_tox2_cen" in IDX: results["plasma_HA"]   = y[IDX["C_tox2_cen"]]
        if "Ae_t2"      in IDX: results["excreted_HA"] = y[IDX["Ae_t2"]]

    if compound in ("all", "pCS"):
        if "C_tox3_cen" in IDX: results["plasma_pCS"]   = y[IDX["C_tox3_cen"]]
        if "Ae_t3"      in IDX: results["excreted_pCS"] = y[IDX["Ae_t3"]]

    return results


def verify_tissue_distribution(
    sol,
    t_start_infusion: float = 0.0,
    t_end_infusion:   float = 0.5,
) -> bool:
    """
    Sanity check that the peripheral tissue compartment is behaving correctly.

    Prints diagnostic values during and after the infusion window.
    Returns True if the tissue compartment appears active, False otherwise.

    Expected behaviour
    ------------------
    During infusion:
      C_ven rises to peak, C_peri rises more slowly (distribution lag).

    After infusion:
      C_ven declines rapidly (elimination + redistribution back from tissue).
      C_peri declines more slowly (governed by k21, the return rate constant).
      C_peri / C_ven ratio > 0.5 post-infusion suggests tissue is active.

    Parameters
    ----------
    sol               : OdeResult from run_sim()
    t_start_infusion  : infusion start time [h]
    t_end_infusion    : infusion duration [h]

    Returns
    -------
    bool : True if tissue distribution appears active, False otherwise.
    """
    if "C_ven" not in IDX or "C_peri" not in IDX:
        print("[verify_tissue] C_peri not in state vector — "
              "tissue compartment is disabled.")
        return False

    t      = sol.t
    C_ven  = sol.y[IDX["C_ven"]]
    C_peri = sol.y[IDX["C_peri"]]

    t_inf_end = t_start_infusion + t_end_infusion
    inf_mask  = (t >= t_start_infusion) & (t <= t_inf_end)
    post_mask = t > t_inf_end + 0.5

    print("\n[verify_tissue_distribution]")

    if np.any(inf_mask):
        max_ven  = float(np.max(C_ven[inf_mask]))
        max_peri = float(np.max(C_peri[inf_mask]))
        print(f"  During infusion : max C_ven  = {max_ven:.4f} mg/L")
        print(f"  During infusion : max C_peri = {max_peri:.4f} mg/L")
    else:
        print("  [WARN] No timepoints found within the infusion window.")
        return False

    active = False
    if np.any(post_mask):
        ven_end  = float(C_ven[post_mask][-1])  if C_ven[post_mask].size  > 0 else 0.0
        peri_end = float(C_peri[post_mask][-1]) if C_peri[post_mask].size > 0 else 0.0
        print(f"  End of sim      : C_ven  = {ven_end:.4f} mg/L")
        print(f"  End of sim      : C_peri = {peri_end:.4f} mg/L")

        if ven_end > 1e-9 and peri_end > 1e-9:
            ratio = peri_end / ven_end
            print(f"  C_peri/C_ven ratio = {ratio:.3f}")
            if ratio > 0.5:
                print("  ✓ Tissue distribution is active")
                active = True
            else:
                print("  ⚠ Tissue concentration is very low relative to plasma. "
                      "Check Vperi_L, k12_hinv, k21_hinv in compounds dict.")
        else:
            print("  ⚠ One or both end concentrations are near zero.")
    else:
        print("  [WARN] No post-infusion timepoints found.")

    return active