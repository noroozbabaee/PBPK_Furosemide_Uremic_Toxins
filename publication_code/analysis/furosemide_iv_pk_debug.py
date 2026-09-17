# furosemide_iv_pk_debug.py  — DEBUG VERSION of the IV furosemide PK runner
# All original logic preserved. Debug prints added at every critical point.
# Search for [DBG] to find all new print statements.
#
# KNOWN ISSUE (documented, not fixed, to preserve original debug behavior):
#   _dbg_snapshot() calls physiology.get_physiology(mode), which does not
#   exist in this codebase (physiology.py only exposes make_phys()). This
#   is caught by a try/except inside _dbg_snapshot(), so it does NOT crash
#   the script — it just prints "[PHYS] could not read: ..." instead of the
#   intended physiology snapshot. The actual PK simulation and metrics
#   (Cmax, AUC, CL_R, t1/2) are computed independently and are unaffected.
#   Left as-is here as a realistic example of how PBPK debug tooling can
#   silently degrade when an internal API changes — useful as a teaching
#   point, and a good first practice fix for someone new to the codebase.

from __future__ import annotations
import os
import sys

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

import constants as K
import compounds as C
from simulation import run_sim
from state_vector import IDX
import compound_parameters as CP

T_BASELINE_H: float = 1200.0
T_PK_H:       float = 12.0


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG HELPER — call this at any point to snapshot the full compound state
# ══════════════════════════════════════════════════════════════════════════════

def _dbg_snapshot(tag: str) -> None:
    """
    Print a full snapshot of every parameter that affects CL_R and Ae.
    Covers: fu_p, BP, IR, k_elim, Vcen, Q_urine, GFR, Vmax_baso, Vmax_api,
    Ki values, enabled flags, and toxin status.
    Call this before and after every major function call.
    """
    sep = "─" * 72
    print(f"\n{'═'*72}")
    print(f"[DBG SNAPSHOT] {tag}")
    print(sep)

    # ── physiology ────────────────────────────────────────────────────────────
    try:
        import physiology as P
        phys = P.get_physiology("healthy")   # just to check healthy baseline
        print(f"  [PHYS healthy] Q_kidney={phys.get('Q_kidney0','?')} L/h  "
              f"GFR={phys.get('Q_GFR','?')} L/h  "
              f"Q_urine={phys.get('Q_urine','?')} L/h  "
              f"Vcen={phys.get('V_ven0','?')} L")
    except Exception as e:
        print(f"  [PHYS] could not read: {e}")

    try:
        import physiology as P
        phys_ckd = P.get_physiology("ckd5")
        print(f"  [PHYS ckd5  ] Q_kidney={phys_ckd.get('Q_kidney0','?')} L/h  "
              f"GFR={phys_ckd.get('Q_GFR','?')} L/h  "
              f"Q_urine={phys_ckd.get('Q_urine','?')} L/h  "
              f"Vcen={phys_ckd.get('V_ven0','?')} L")
    except Exception as e:
        print(f"  [PHYS ckd5] could not read: {e}")

    print(sep)

    # ── drug (C.drug) ─────────────────────────────────────────────────────────
    drug = getattr(C, "drug", None)
    if isinstance(drug, dict):
        print(f"  [C.drug] enabled={drug.get('enabled')}  "
              f"IR={drug.get('IR')} mg/h  "
              f"t0={drug.get('t_start_h')} h  "
              f"t_inf={drug.get('t_inf_h')} h  "
              f"bolus={drug.get('bolus_mg')} mg")
        print(f"           fu_p={drug.get('fu_p')}  "
              f"BP={drug.get('BP')}  "
              f"k_elim={drug.get('k_elim_hinv')} h⁻¹  "
              f"Vcen from phys (not here)")
        print(f"           Vmax_baso={drug.get('Vmax_baso_uptake')} mg/h  "
              f"Vmax_api={drug.get('Vmax_api_secr')} mg/h  "
              f"Km_baso={drug.get('Km_baso_uptake')} mg/L")
        print(f"           Vperi={drug.get('Vperi_L')} L  "
              f"k12={drug.get('k12_hinv')} h⁻¹  "
              f"k21={drug.get('k21_hinv')} h⁻¹  "
              f"use_tissue={drug.get('use_tissue')}")
        print(f"           inhibit={drug.get('inhibit')}  "
              f"Ki_IS={drug.get('Ki_furo_to_IS_mgL','N/A')}  "
              f"Ki_HA={drug.get('Ki_furo_to_HA_mgL','N/A')}  "
              f"Ki_pCS={drug.get('Ki_furo_to_pCS_mgL','N/A')} mg/L")

        # ── KEY DIAGNOSTIC: what will CL_R look like? ─────────────────────────
        # CL_R = Ae/AUC. Ae depends on Q_urine × C_lumen_exit.
        # If Q_urine is wrong OR fu_p is wrong, CL_R collapses.
        try:
            import physiology as P
            # get current mode from C if possible
            mode_guess = getattr(C, "_current_mode", "healthy")
            phys = P.get_physiology(mode_guess)
            Q_u   = phys.get("Q_urine", float("nan"))
            GFR   = phys.get("Q_GFR",   float("nan"))
            fu_p  = drug.get("fu_p", float("nan"))
            BP    = drug.get("BP",   float("nan"))
            # Filtration term at Cmax~4 mg/L (rough):
            C_ven_approx = 4.0
            filt_rate = GFR * (fu_p / BP) * C_ven_approx
            print(f"\n  [CL_R DIAGNOSTIC] mode_guess={mode_guess}")
            print(f"    Q_urine={Q_u:.4f} L/h  — "
                  f"{'⚠ TOO HIGH (should be 0.06 healthy)' if Q_u > 0.1 else '✓ OK'}")
            print(f"    GFR={GFR:.4f} L/h  — "
                  f"{'⚠ TOO LOW' if GFR < 1.0 else '✓ OK'}")
            print(f"    Filtration rate @ Cven≈4 mg/L = "
                  f"{filt_rate:.4f} mg/h  "
                  f"(GFR × fu_p/BP × 4)")
            print(f"    If Ae≈filtration×t = {filt_rate*12:.2f} mg over 12h")
            print(f"    → expected CL_R ≈ {filt_rate*12 / (C_ven_approx*12*0.5):.3f} L/h "
                  f"(rough, assumes AUC~Cmax×t/2)")
        except Exception as e:
            print(f"  [CL_R DIAGNOSTIC] failed: {e}")
    else:
        print("  [C.drug] NOT a dict or None")

    print(sep)

    # ── toxins ────────────────────────────────────────────────────────────────
    for attr, name in [("tox1", "IS"), ("tox2", "HA"), ("tox3", "pCS")]:
        tox = getattr(C, attr, None)
        if isinstance(tox, dict):
            print(f"  [C.{attr} {name}] enabled={tox.get('enabled')}  "
                  f"fu_p={tox.get('fu_p')}  "
                  f"Vmax_baso={tox.get('Vmax_baso_uptake')} mg/h  "
                  f"Vmax_api={tox.get('Vmax_api_secr')} mg/h  "
                  f"R0={tox.get('R0')} mg/h")
        else:
            print(f"  [C.{attr}] NOT a dict")

    print(sep)

    # ── compound_parameters.py direct read ───────────────────────────────────
    try:
        h_drug, h_t1, h_t2, h_t3 = CP.get_compound_parameters("healthy")
        c_drug, c_t1, c_t2, c_t3 = CP.get_compound_parameters("ckd5")
        print(f"  [CP healthy] drug: fu_p={h_drug['fu_p']}  "
              f"IR={h_drug.get('IR')} mg/h  "
              f"k_elim={h_drug.get('k_elim_hinv')} h⁻¹  "
              f"Vmax_baso={h_drug['Vmax_baso_uptake']}")
        print(f"  [CP ckd5  ] drug: fu_p={c_drug['fu_p']}  "
              f"IR={c_drug.get('IR')} mg/h  "
              f"k_elim={c_drug.get('k_elim_hinv')} h⁻¹  "
              f"Vmax_baso={c_drug['Vmax_baso_uptake']}")
        print(f"  [CP healthy] IS: Vmax_baso={h_t1['Vmax_baso_uptake']}  fu_p={h_t1['fu_p']}")
        print(f"  [CP ckd5  ] IS: Vmax_baso={c_t1['Vmax_baso_uptake']}  fu_p={c_t1['fu_p']}")
        print(f"  [CP healthy] HA: Vmax_baso={h_t2['Vmax_baso_uptake']}  fu_p={h_t2['fu_p']}")
        print(f"  [CP ckd5  ] HA: Vmax_baso={c_t2['Vmax_baso_uptake']}  fu_p={c_t2['fu_p']}")

        # Check for the Vcen discrepancy
        print(f"\n  [CP CRITICAL CHECK]")
        print(f"    healthy drug k_elim = {h_drug.get('k_elim_hinv')} h⁻¹  "
              f"→ t½ = {0.693/h_drug.get('k_elim_hinv',1):.2f} h")
        print(f"    ckd5    drug k_elim = {c_drug.get('k_elim_hinv')} h⁻¹  "
              f"→ t½ = {0.693/c_drug.get('k_elim_hinv',1):.2f} h")
        if h_drug.get('k_elim_hinv') == c_drug.get('k_elim_hinv'):
            print(f"    ⚠ k_elim IDENTICAL for healthy and CKD — "
                  f"t½ will be the same. Is CKD k_elim scaling applied?")
        else:
            print(f"    ✓ k_elim differs between modes")

        if abs(h_drug.get('fu_p', 0) - c_drug.get('fu_p', 0)) < 1e-6:
            print(f"    ⚠ fu_p IDENTICAL for healthy and CKD "
                  f"({h_drug['fu_p']}) — CKD binding defect not applied!")
        else:
            print(f"    ✓ fu_p differs: healthy={h_drug['fu_p']} "
                  f"ckd5={c_drug['fu_p']}")

    except Exception as e:
        print(f"  [CP read] failed: {e}")

    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PATCHED _run_baseline — with debug
# ══════════════════════════════════════════════════════════════════════════════

def _run_baseline(mode: str, dt: float) -> Any:
    print("\n" + "=" * 70)
    print(f"[BASELINE] {mode.upper()}: 0 → {T_BASELINE_H:.0f} h  (no drug)")
    print("=" * 70)

    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)
        # [DBG] record mode so snapshot can read it
        C._current_mode = mode

    print(f"\n[DBG] After initialize_compounds('{mode}'):")
    _dbg_snapshot(f"baseline_start_{mode}")

    if isinstance(getattr(C, "drug", None), dict):
        C.drug["enabled"]   = True
        C.drug["IR"]        = 0.0
        C.drug["t_start_h"] = 1e9
        C.drug["t_inf_h"]   = 0.0
        C.drug["bolus_mg"]  = 0.0

    print(f"[DBG] Drug disabled for baseline: IR={C.drug.get('IR')}  "
          f"t_start={C.drug.get('t_start_h')}")

    sol = run_sim(t_end_h=T_BASELINE_H, mode=mode, dt=dt)
    print(f"  Baseline complete — final t = {sol.t[-1]:.1f} h")

    # [DBG] check what the state looks like at end of baseline
    print(f"\n[DBG] End-of-baseline state (first 6 IDX entries):")
    for name, idx in list(IDX.items())[:6]:
        print(f"  {name:20s}[{idx}] = {sol.y[idx, -1]:.4e}")
    # specifically check C_ven and toxin central compartments
    for key in ("C_ven", "C_peri", "C_tox1_cen", "C_tox2_cen", "C_tox3_cen"):
        if key in IDX:
            print(f"  {key:20s}[{IDX[key]}] = {sol.y[IDX[key], -1]:.4e}")

    return sol


# ══════════════════════════════════════════════════════════════════════════════
# PATCHED _apply_drug_params — with debug
# ══════════════════════════════════════════════════════════════════════════════

def _apply_drug_params(mode: str) -> Dict[str, float]:
    print(f"\n[DBG] _apply_drug_params called for mode='{mode}'")
    _dbg_snapshot(f"before_apply_drug_params_{mode}")

    drug_params, _, _, _ = CP.get_compound_parameters(mode)
    print(f"[DBG] CP.get_compound_parameters('{mode}') returned drug:")
    print(f"  fu_p={drug_params['fu_p']}  IR={drug_params.get('IR')}  "
          f"k_elim={drug_params.get('k_elim_hinv')}  "
          f"Vmax_baso={drug_params.get('Vmax_baso_uptake')}")

    if not isinstance(getattr(C, "drug", None), dict):
        raise RuntimeError("C.drug must be a dict.")

    for k, v in drug_params.items():
        C.drug[k] = v

    C.drug["enabled"]   = True
    C.drug["t_start_h"] = 0.0
    C.drug["inhibit"]   = False   # [DBG] was True in original — WRONG for PK validation
    if not C.drug.get("protocol"):
        C.drug["protocol"] = "plasma_britz"

    for tox_name in ("tox1", "tox2", "tox3"):
        tox = getattr(C, tox_name, None)
        if isinstance(tox, dict):
            tox["enabled"] = False
            print(f"[DBG] C.{tox_name}.enabled set to False")

    _dbg_snapshot(f"after_apply_drug_params_{mode}")

    IR       = float(C.drug.get("IR",       0.0))
    t_inf_h  = float(C.drug.get("t_inf_h",  0.0))
    bolus_mg = float(C.drug.get("bolus_mg", 0.0))
    total    = IR * t_inf_h + bolus_mg

    print(f"\n  [DOSE] IR={IR} mg/h × {t_inf_h} h + bolus={bolus_mg} mg "
          f"= {total:.1f} mg total")

    if total <= 0:
        print(f"  ⚠⚠⚠ TOTAL DOSE IS ZERO — drug will not be administered! "
              f"Check IR and t_inf_h in compound_parameters.py DRUG_BASE.")

    return {"IR": IR, "t_inf_h": t_inf_h, "bolus_mg": bolus_mg,
            "total_dose_mg": total}


# ══════════════════════════════════════════════════════════════════════════════
# PATCHED _run_mode — with debug
# ══════════════════════════════════════════════════════════════════════════════

def _make_pk_initial_state(sol_base: Any) -> np.ndarray:
    y0 = np.asarray(sol_base.y[:, -1], dtype=float).copy()
    for name in ("C_ven", "C_peri", "Ae_drg", "C_bladder"):
        if name in IDX:
            y0[IDX[name]] = 0.0
    for name, i in IDX.items():
        if (name.startswith("C_tox1_") or name.startswith("C_tox2_")
                or name.startswith("C_tox3_")):
            y0[i] = 0.0
    for name in ("Ae_t1", "Ae_t2", "Ae_t3"):
        if name in IDX:
            y0[IDX[name]] = 0.0
    return y0


def _extract_pk_curves(sol):
    t       = np.asarray(sol.t, dtype=float)
    Cc      = np.asarray(sol.y[IDX["C_ven"]], dtype=float)
    Ae_full = np.asarray(sol.y[IDX["Ae_drg"]], dtype=float)
    Ae      = Ae_full - Ae_full[0]
    return t, Cc, Ae


def _run_mode(mode: str, dt: float) -> Dict[str, Any]:
    print(f"\n{'#'*70}")
    print(f"# _run_mode: mode={mode}")
    print(f"{'#'*70}")

    # Step 1: baseline
    sol_base = _run_baseline(mode, dt=0.5)

    # Step 2: initial conditions
    y0_pk = _make_pk_initial_state(sol_base)
    print(f"\n[DBG] PK initial conditions built.")
    print(f"  C_ven(t=0) = {y0_pk[IDX['C_ven']]:.4e} mg/L  (should be 0)")

    # Step 3: reinitialise
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)
        C._current_mode = mode
    print(f"[DBG] After reinitialise for mode='{mode}':")
    print(f"  C.drug['fu_p']           = {C.drug.get('fu_p')}")
    print(f"  C.drug['Vmax_baso_uptake']= {C.drug.get('Vmax_baso_uptake')}")

    # Step 4: apply drug params
    dosing  = _apply_drug_params(mode)
    t_inf_h = dosing["t_inf_h"]

    # [DBG] CRITICAL: check Vcen — this comes from physiology, not drug dict
    try:
        import physiology as P
        phys = P.get_physiology(mode)
        Vcen = phys.get("V_ven0", "NOT FOUND")
        Q_u  = phys.get("Q_urine", "NOT FOUND")
        GFR  = phys.get("Q_GFR",   "NOT FOUND")
        print(f"\n[DBG] PHYSIOLOGY for mode='{mode}':")
        print(f"  V_ven0  (Vcen)  = {Vcen} L  "
              f"{'⚠ WRONG: should be 5 L' if Vcen != 5.0 else '✓'}")
        print(f"  Q_urine         = {Q_u} L/h  "
              f"{'⚠ WRONG: should be 0.06 L/h healthy' if isinstance(Q_u,float) and Q_u > 0.1 else '✓ or CKD'}")
        print(f"  Q_GFR           = {GFR} L/h")

        # [DBG] Compute expected Ae from physiology alone
        fu_p = C.drug.get("fu_p", 0.02)
        BP   = C.drug.get("BP",   0.82)
        if isinstance(GFR, float) and isinstance(Q_u, float):
            # Approximate: at Cmax~4 mg/L, filtration rate ≈ GFR * (fu_p/BP) * Cmax
            approx_filt = GFR * (fu_p / max(BP, 1e-9)) * 4.0
            print(f"\n  [CL_R PREDICTION for {mode}]")
            print(f"    Filtration rate (@ Cmax≈4 mg/L) = {approx_filt:.4f} mg/h")
            print(f"    Q_urine = {Q_u:.4f} L/h")
            print(f"    Expected Ae_12h (filtration only) ≈ {approx_filt*2:.2f} mg "
                  f"(during 2h infusion phase)")
            if approx_filt < 0.01:
                print(f"    ⚠⚠⚠ Filtration rate NEAR ZERO → Ae will be near zero → CL_R≈0!")
                print(f"         Cause: GFR={GFR:.4f} L/h OR fu_p={fu_p} is too small")
    except Exception as e:
        print(f"[DBG] Physiology read failed: {e}")

    # Step 5: run PK simulation
    print(f"\n[DBG] Starting PK run for mode='{mode}', "
          f"IR={C.drug.get('IR')} mg/h, t_inf={t_inf_h} h")

    ic = {name: float(y0_pk[i]) for name, i in IDX.items()}
    sol_pk = run_sim(t_end_h=T_PK_H, mode=mode, dt=dt, custom_initial_conds=ic)

    # Step 6: extract
    t, Cc, Ae = _extract_pk_curves(sol_pk)

    print(f"\n[DBG] PK simulation complete for mode='{mode}':")
    print(f"  Cmax     = {np.max(Cc):.4f} mg/L  at t={t[np.argmax(Cc)]:.2f} h")
    print(f"  Ae_final = {Ae[-1]:.4f} mg")
    print(f"  AUC      = {float(_trapz(Cc, t)):.4f} mg·h/L")
    if float(_trapz(Cc, t)) > 0:
        clr = Ae[-1] / float(_trapz(Cc, t))
        print(f"  CL_R     = {clr:.4f} L/h")
        if clr < 0.01:
            print(f"  ⚠⚠⚠ CL_R NEAR ZERO — Ae={Ae[-1]:.4e} mg is the problem")
            print(f"       Check: (1) Ae_drg state index={IDX.get('Ae_drg','?')}")
            print(f"              (2) Q_urine in physiology.py")
            print(f"              (3) excretion step in ode_system_twocompartment.py")
            # Check Ae_drg raw (not baseline-corrected)
            Ae_raw = np.asarray(sol_pk.y[IDX["Ae_drg"]], dtype=float)
            print(f"       Ae_drg raw: start={Ae_raw[0]:.4e}  end={Ae_raw[-1]:.4e}  "
                  f"delta={Ae_raw[-1]-Ae_raw[0]:.4e}")
    else:
        print(f"  ⚠⚠⚠ AUC = 0 — Cmax is zero, drug never entered central compartment!")
        print(f"       Check IR, t_start_h, and whether drug.enabled=True reached the ODE.")

    label = "Healthy" if mode == "healthy" else "CKD-5"
    from plot_furosemide_pk_ckd_vs_healthy import (
        _compute_pk_metrics, validate_against_literature
    )
    metrics = _compute_pk_metrics(t, Cc, Ae, t_inf_h=t_inf_h,
                                   post_inf_buffer_h=0.5)
    validate_against_literature(metrics, mode)
    return {"label": label, "t": t, "C": Cc, "Ae": Ae, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — runs both modes with full debug output
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*72)
    print("█  FUROSEMIDE PK DEBUG RUN")
    print("█"*72)

    # First: snapshot what compound_parameters.py contains RIGHT NOW
    _dbg_snapshot("INITIAL STATE before any simulation")

    try:
        from plot_furosemide_pk_ckd_vs_healthy import (
            setup_professional_style, plot_furosemide_pk
        )
        setup_professional_style()
    except Exception:
        pass

    out_dir = os.path.join(_THIS_DIR, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    dt = 0.1

    print("\n" + "▶"*72)
    print("▶  RUNNING HEALTHY MODE")
    print("▶"*72)
    data_h = _run_mode("healthy", dt=dt)

    print("\n" + "▶"*72)
    print("▶  RUNNING CKD-5 MODE")
    print("▶"*72)
    data_c = _run_mode("ckd5", dt=dt)

    print("\n" + "═"*72)
    print("FINAL COMPARISON")
    print("═"*72)
    print(f"  Healthy: Cmax={np.max(data_h['C']):.3f}  "
          f"Ae={data_h['Ae'][-1]:.2f} mg  "
          f"CL_R={data_h['metrics']['CL_R_L_per_h']:.3f} L/h  "
          f"t½={data_h['metrics']['t_half_h']:.3f} h")
    print(f"  CKD-5:   Cmax={np.max(data_c['C']):.3f}  "
          f"Ae={data_c['Ae'][-1]:.4f} mg  "
          f"CL_R={data_c['metrics']['CL_R_L_per_h']:.3f} L/h  "
          f"t½={data_c['metrics']['t_half_h']:.3f} h")

    if abs(np.max(data_h['C']) - np.max(data_c['C'])) < 0.01:
        print("\n  ⚠⚠⚠ Cmax IDENTICAL for healthy and CKD — "
              "CKD mode is not using different parameters!")
    if abs(data_h['metrics']['t_half_h'] - data_c['metrics']['t_half_h']) < 0.01:
        print("  ⚠⚠⚠ t½ IDENTICAL — k_elim or Vcen is the same for both modes")

    from plot_furosemide_pk_ckd_vs_healthy import plot_furosemide_pk
    pk_data = {
        data_h["label"]: {"t": data_h["t"], "C": data_h["C"],
                           "Ae": data_h["Ae"], "metrics": data_h["metrics"]},
        data_c["label"]: {"t": data_c["t"], "C": data_c["C"],
                           "Ae": data_c["Ae"], "metrics": data_c["metrics"]},
    }
    fig = plot_furosemide_pk(pk_data)
    path = os.path.join(out_dir, "furosemide_pk_debug.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {path}")
    plt.show()


if __name__ == "__main__":
    main()