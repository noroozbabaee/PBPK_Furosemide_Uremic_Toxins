"""
================================================================================
FUROSEMIDE PHARMACOKINETICS: HEALTHY vs CKD-5
================================================================================
Simulates furosemide IV infusion in healthy subjects and CKD-5 patients using
the kidney PBPK model, then compares plasma concentration-time curves and
key PK metrics against published literature.

ALL PARAMETERS COME FROM compound_parameters.py - NO HARDCODED VALUES
================================================================================
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
# It is also imported as a LIBRARY by other analysis scripts (e.g.
# furosemide_iv_pk_debug.py imports _compute_pk_metrics and
# validate_against_literature from here), so the bootstrap must run
# correctly whether this file is executed directly or imported.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from typing import Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Version-safe trapezoidal integration ──────────────────────────────────────
try:
    _trapz = np.trapezoid  # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz  # NumPy < 2.0

# ── Force fresh load to avoid stale cache ─────────────────────────────────────
import sys as _sys

for _mod in ["compound_parameters", "compounds"]:
    if _mod in _sys.modules:
        del _sys.modules[_mod]

# ── Project imports ────────────────────────────────────────────────────────────
import constants as K
import compounds as C
from simulation import run_sim
from state_vector import IDX
import compound_parameters as CP

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Simulation timing constants
# ══════════════════════════════════════════════════════════════════════════════

T_BASELINE_H: float = 1200.0  # Baseline warm-up (no drug)
T_PK_H: float = 12.0  # PK observation window


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Baseline simulation (no drug, toxins only)
# ══════════════════════════════════════════════════════════════════════════════

def _run_baseline(mode: str, dt: float):
    """Run baseline without drug to establish steady state."""
    print("\n" + "=" * 70)
    print(f"[BASELINE] {mode.upper()}: 0 → {T_BASELINE_H:.0f} h (no drug)")
    print("=" * 70)

    # Initialize compounds from parameters file
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)

    # Ensure drug is disabled during baseline
    if isinstance(getattr(C, "drug", None), dict):
        C.drug["enabled"] = False
        C.drug["IR"] = 0.0
        C.drug["t_start_h"] = 1e9
        C.drug["t_inf_h"] = 0.0
        C.drug["bolus_mg"] = 0.0

    sol = run_sim(t_end_h=T_BASELINE_H, mode=mode, dt=dt)
    print(f"  Baseline complete")
    return sol


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PK initial conditions (reset drug and toxins to zero)
# ══════════════════════════════════════════════════════════════════════════════

def _make_pk_initial_state(sol_base):
    """Reset drug and toxin compartments to zero for clean PK study."""
    y0 = np.asarray(sol_base.y[:, -1], dtype=float).copy()

    # Reset drug state to zero
    for name in ("C_ven", "C_peri", "Ae_drg", "C_bladder"):
        if name in IDX:
            y0[IDX[name]] = 0.0

    # Reset all toxin compartments to zero
    for name, i in IDX.items():
        if (name.startswith("C_tox1_") or name.startswith("C_tox2_")
                or name.startswith("C_tox3_")):
            y0[i] = 0.0

    # Reset cumulative excretion
    for name in ("Ae_t1", "Ae_t2", "Ae_t3"):
        if name in IDX:
            y0[IDX[name]] = 0.0

    return y0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Apply drug parameters directly from compound_parameters.py
# ══════════════════════════════════════════════════════════════════════════════

def _apply_drug_params(mode: str):
    """
    Load drug parameters from compound_parameters.py and apply to C.drug.
    NO HARDCODED OVERRIDES - everything comes from the parameters file.
    """
    # Get parameters from the source of truth
    drug_params, _, _, _ = CP.get_compound_parameters(mode)

    print(f"[DIAGNOSTIC] Vmax_baso from CP = {drug_params['Vmax_baso_uptake']} mg/h")
    if not isinstance(getattr(C, "drug", None), dict):
        raise RuntimeError("C.drug not initialized. Call C.initialize_compounds(mode) first.")

    # Copy ALL parameters from the source
    for k, v in drug_params.items():
        C.drug[k] = v

    # Only enable drug and set infusion start (these are simulation-specific,
    # not parameter-specific, so they are set here)
    C.drug["enabled"] = True
    C.drug["t_start_h"] = 0.0
    C.drug["inhibit"] = False  # No toxin inhibition during PK validation

    # Disable toxins for clean PK
    for tox_name in ("tox1", "tox2", "tox3"):
        tox = getattr(C, tox_name, None)
        if isinstance(tox, dict):
            tox["enabled"] = False

    print(f"\n  [DOSE] IR={C.drug['IR']} mg/h × {C.drug['t_inf_h']} h = {C.drug['IR'] * C.drug['t_inf_h']:.0f} mg")
    print(f"  [PK]   fu_p={C.drug['fu_p']}, k_elim={C.drug['k_elim_hinv']} h⁻¹")
    print(f"  [PK]   Vmax_baso={C.drug['Vmax_baso_uptake']} mg/h, Km={C.drug['Km_baso_uptake']} mg/L")

    return {
        "IR": C.drug["IR"],
        "t_inf_h": C.drug["t_inf_h"],
        "total_dose_mg": C.drug["IR"] * C.drug["t_inf_h"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PK metrics calculation
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pk_curves(sol):
    t = np.asarray(sol.t, dtype=float)
    Cc = np.asarray(sol.y[IDX["C_ven"]], dtype=float)
    Ae_full = np.asarray(sol.y[IDX["Ae_drg"]], dtype=float)
    Ae = Ae_full - Ae_full[0]
    return t, Cc, Ae


def _compute_pk_metrics(t, Cc, Ae, t_inf_h, post_inf_buffer_h=0.5):
    """Compute standard non-compartmental PK metrics."""
    print("\n[PK METRICS]")

    idx_max = np.argmax(Cc) if Cc.size > 0 else 0
    Cmax = float(Cc[idx_max]) if Cc.size > 0 else np.nan
    Tmax = float(t[idx_max]) if t.size > 0 else np.nan
    print(f"  Cmax   = {Cmax:.4f} mg/L  at  Tmax = {Tmax:.3f} h")

    AUC = float(_trapz(Cc, t)) if t.size > 1 else 0.0
    print(f"  AUC    = {AUC:.4f} mg·h/L  (0 → {t[-1]:.1f} h)")

    Ae_final = float(Ae[-1]) if Ae.size > 0 else 0.0
    CL_R = Ae_final / AUC if AUC > 1e-12 else np.nan
    print(f"  Ae     = {Ae_final:.4f} mg")
    print(f"  CL_R   = {CL_R:.4f} L/h  ({CL_R * 1000 / 60:.2f} mL/min)" if not np.isnan(CL_R) else "  CL_R   = N/A")

    # Terminal half-life
    t_tail_start = float(t_inf_h + post_inf_buffer_h)
    threshold = 0.01 * Cmax if (not np.isnan(Cmax) and Cmax > 0) else 1e-9
    mask = (t >= t_tail_start) & (Cc > threshold)
    n_avail = int(np.sum(mask))

    if n_avail < 5:
        t_half = np.nan
        print(f"  t½     = N/A  (only {n_avail} points)")
    else:
        tt = t[mask]
        cc = Cc[mask]
        n_tail = max(5, len(tt) // 5)
        tt = tt[-n_tail:]
        cc = cc[-n_tail:]

        if np.any(cc <= 0):
            t_half = np.nan
            print("  t½     = N/A  (non-positive concentrations)")
        else:
            slope, _ = np.polyfit(tt, np.log(cc), 1)
            if slope < 0:
                k_el = -slope
                t_half = float(np.log(2) / k_el)
                print(f"  k_el   = {k_el:.4f} h⁻¹  (using {n_tail} points)")
                print(f"  t½     = {t_half:.3f} h  ({t_half * 60:.1f} min)")
            else:
                t_half = np.nan
                print("  t½     = N/A  (non-negative slope)")

    return {
        "Cmax_mg_per_L": Cmax,
        "Tmax_h": Tmax,
        "AUC_mg_h_per_L": AUC,
        "Ae_final_mg": Ae_final,
        "CL_R_L_per_h": CL_R,
        "t_half_h": t_half,
    }


def validate_against_literature(metrics, mode):
    """Compare model predictions with literature."""
    lit = {
        "healthy": {
            "CL_R": (4.0, 7.0, 6.9),
            "t_half": (0.9, 1.5, 1.2),
            "AUC": (3.0, 5.5, 4.0),
        },
        "ckd5": {
            "CL_R": (1.0, 2.0, 1.5),
            "t_half": (1.0, 3.0, 2.0),
            "AUC": (15.0, 30.0, 22.0),
        },
    }

    print("\n" + "=" * 70)
    print(f"LITERATURE VALIDATION — {mode.upper()}  (40 mg IV dose)")
    print("=" * 70)

    clr = metrics["CL_R_L_per_h"]
    lo, hi, typ = lit[mode]["CL_R"]
    if not np.isnan(clr):
        flag = "✓" if lo <= clr <= hi else "⚠"
        print(f"  {flag} CL_R  = {clr:.3f} L/h  (target {typ:.1f}, range {lo}–{hi})")

    th = metrics["t_half_h"]
    lo, hi, typ = lit[mode]["t_half"]
    if not np.isnan(th):
        flag = "✓" if lo <= th <= hi else "⚠"
        print(f"  {flag} t½    = {th:.3f} h  (target {typ:.2f}, range {lo}–{hi})")

    auc = metrics["AUC_mg_h_per_L"]
    lo, hi, typ = lit[mode]["AUC"]
    if not np.isnan(auc) and auc > 0:
        flag = "✓" if lo <= auc <= hi else "⚠"
        print(f"  {flag} AUC   = {auc:.3f} mg·h/L  (target {typ:.1f}, range {lo}–{hi})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Main simulation driver
# ══════════════════════════════════════════════════════════════════════════════

def _run_mode(mode: str, dt: float):
    """Run full baseline + PK for one mode."""

    # Step 1: Baseline (no drug, toxins accumulate)
    sol_base = _run_baseline(mode, dt=0.5)

    # Step 2: Build PK initial conditions
    y0_pk = _make_pk_initial_state(sol_base)

    # Step 3: Reinitialize compounds
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)

    # Step 4: Apply drug parameters
    dosing = _apply_drug_params(mode)
    t_inf_h = dosing["t_inf_h"]


    # ========== ADD THIS DIAGNOSTIC BLOCK ==========
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC: What C.drug actually contains")
    print("=" * 60)
    print(f"  C.drug['Vmax_baso_uptake'] = {C.drug.get('Vmax_baso_uptake', 'MISSING')}")
    print(f"  C.drug['Vmax_OAT1'] = {C.drug.get('Vmax_OAT1', 'MISSING')}")
    print(f"  C.drug['IR'] = {C.drug.get('IR', 'MISSING')}")
    print(f"  C.drug['fu_p'] = {C.drug.get('fu_p', 'MISSING')}")
    print("=" * 60 + "\n")
    # ========== FORCE OVERRIDE ==========
    # if mode == "healthy":
        # # In your force override:
        # C.drug["Vmax_api_secr"] = 3000.0  # was 2400
        # C.drug["Vmax_baso_uptake"] = 4000.0
        # C.drug["Vmax_OAT1"] = 4000.0
        # C.drug["k_elim_hinv"] = 0.25  # back to original
        # print(f"[FORCED] Vmax_baso = {C.drug['Vmax_baso_uptake']} mg/h")
        # print(f"[FORCED] k_elim = {C.drug['k_elim_hinv']} h-1")
    # ==============================================
    # Step 5: PK simulation
    print("\n" + "-" * 70)
    print(f"[PK] {mode.upper()}: 0 → {T_PK_H:.1f} h (infusion 0 → {t_inf_h:.2f} h)")
    print("-" * 70)

    ic = {name: float(y0_pk[i]) for name, i in IDX.items()}
    sol_pk = run_sim(t_end_h=T_PK_H, mode=mode, dt=dt, custom_initial_conds=ic)

    # Step 6: Extract and compute metrics
    t, Cc, Ae = _extract_pk_curves(sol_pk)
    metrics = _compute_pk_metrics(t, Cc, Ae, t_inf_h=t_inf_h)
    validate_against_literature(metrics, mode)

    label = "Healthy" if mode == "healthy" else "CKD-5"
    return {"label": label, "t": t, "C": Cc, "Ae": Ae, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_furosemide_pk(pk_data):
    """Create 3-panel figure."""
    colors = {"Healthy": "#3969AC", "CKD-5": "#C44E52"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_c, ax_ae, ax_bar = axes

    # Panel A: Plasma concentration
    for label in ("Healthy", "CKD-5"):
        ax_c.plot(pk_data[label]["t"], pk_data[label]["C"],
                  label=label, color=colors[label], linewidth=2.5)
    ax_c.set_xlabel("Time (h)")
    ax_c.set_ylabel("Plasma concentration (mg/L)")
    ax_c.grid(True, alpha=0.3)
    ax_c.text(0.02, 0.95, "A", transform=ax_c.transAxes, fontweight="bold")

    # Panel B: Cumulative excretion
    for label in ("Healthy", "CKD-5"):
        ax_ae.plot(pk_data[label]["t"], pk_data[label]["Ae"],
                   label=label, color=colors[label], linewidth=2.5)
    ax_ae.set_xlabel("Time (h)")
    ax_ae.set_ylabel("Cumulative urinary excretion (mg)")
    ax_ae.grid(True, alpha=0.3)
    ax_ae.text(0.02, 0.95, "B", transform=ax_ae.transAxes, fontweight="bold")

    # Panel C: Bar chart
    metrics_keys = ["CL_R_L_per_h", "t_half_h"]
    metrics_labels = ["CL_R (L/h)", "t½ (h)"]
    x = np.arange(len(metrics_keys))
    w = 0.35

    vals_h = [pk_data["Healthy"]["metrics"].get(m, np.nan) for m in metrics_keys]
    vals_c = [pk_data["CKD-5"]["metrics"].get(m, np.nan) for m in metrics_keys]

    ax_bar.bar(x - w / 2, vals_h, w, label="Healthy", color=colors["Healthy"], alpha=0.85)
    ax_bar.bar(x + w / 2, vals_c, w, label="CKD-5", color=colors["CKD-5"], alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics_labels)
    ax_bar.set_ylabel("Value")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.text(0.02, 0.95, "C", transform=ax_bar.transAxes, fontweight="bold")

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main
# ══════════════════════════════════════════════════════════════════════════════
# def main():
#     """Run the healthy simulation only for calibration."""
#     plt.style.use("seaborn-v0_8-whitegrid")
#
#     out_dir = "figures"
#     os.makedirs(out_dir, exist_ok=True)
#
#     dt = 0.1
#
#     print("\n" + "=" * 70)
#     print("  FUROSEMIDE PK: HEALTHY ONLY (Calibration Run)")
#     print("  All parameters from compound_parameters.py")
#     print("=" * 70)
#
#     # Run ONLY healthy mode
#     data_h = _run_mode("healthy", dt=dt)
#     data_c = _run_mode("ckd_5", dt=dt)
#     # Print concentration at key timepoints
#     print("\n" + "=" * 70)
#     print("  PLASMA CONCENTRATION")
#     print("=" * 70)
#     for i, t in enumerate(data_h["t"][::20]):
#         print(f"t={t:.1f}h: Healthy={data_h['C'][i * 20]:.4f} mg/L")
#
#     # Print PK metrics
#     print("\n" + "=" * 70)
#     print("  PK METRICS SUMMARY")
#     print("=" * 70)
#     print(f"  Cmax = {data_h['metrics']['Cmax_mg_per_L']:.2f} mg/L")
#     print(f"  Tmax = {data_h['metrics']['Tmax_h']:.2f} h")
#     print(f"  AUC  = {data_h['metrics']['AUC_mg_h_per_L']:.2f} mg·h/L")
#     print(f"  CL_R = {data_h['metrics']['CL_R_L_per_h']:.2f} L/h")
#     print(f"  t½   = {data_h['metrics']['t_half_h']:.2f} h")
#
#     # Create simple figure (just healthy)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(data_h["t"], data_h["C"], color="#3969AC", linewidth=2.5, label="Healthy")
#     ax.set_xlabel("Time (h)")
#     ax.set_ylabel("Plasma concentration (mg/L)")
#     ax.set_title("Furosemide PK - Healthy")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#
#     # Save figure
#     for ext in ("png", "pdf"):
#         path = os.path.join(out_dir, f"furosemide_pk_healthy.{ext}")
#         fig.savefig(path, dpi=300, bbox_inches="tight")
#         print(f"  Saved {path}")
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
def main():
    """Run the simulation and save figures."""
    plt.style.use("seaborn-v0_8-whitegrid")

    out_dir = os.path.join(_THIS_DIR, "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    dt = 0.1

    print("\n" + "=" * 70)
    print("  FUROSEMIDE PK: HEALTHY vs CKD-5")
    print("  All parameters from compound_parameters.py")
    print("=" * 70)

    # Run both modes
    data_h = _run_mode("healthy", dt=dt)
    data_c = _run_mode("ckd5", dt=dt)

    # Print concentration at key timepoints
    print("\n" + "=" * 70)
    print("  PLASMA CONCENTRATION COMPARISON")
    print("=" * 70)
    for i, t in enumerate(data_h["t"][::20]):
        print(f"t={t:.1f}h: Healthy={data_h['C'][i * 20]:.4f} mg/L, CKD5={data_c['C'][i * 20]:.4f} mg/L")

    # Plot and save
    pk_data = {
        data_h["label"]: {"t": data_h["t"], "C": data_h["C"], "Ae": data_h["Ae"], "metrics": data_h["metrics"]},
        data_c["label"]: {"t": data_c["t"], "C": data_c["C"], "Ae": data_c["Ae"], "metrics": data_c["metrics"]},
    }

    fig = plot_furosemide_pk(pk_data)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"furosemide_pk_healthy_vs_ckd5.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")

    plt.show()


if __name__ == "__main__":
    main()