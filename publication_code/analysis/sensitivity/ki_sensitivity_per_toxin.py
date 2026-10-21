# sa_ki_sensitivity_per_toxin.py
"""
Ki sensitivity (furosemide -> PBUT) in the Kidney PBPK model.

This script is aligned with your validated healthy comparison script:
- Metric matched to your debug prints: concentration at end-of-sim (t = 12 h), not Cmax.
- Steady-state is built with DRUG OFF.
- References match your scenarios:
    1) No Drug (12 h from SS)
    2) With Drug + inhibit ON + DEFAULT Kis  (your "With Drug")
    3) With Drug + inhibit OFF               (your "No Interaction")
    4) With Drug + inhibit ON + Ki=inf (all) (optional "no inhibition effect")

Outputs:
- One figure PER TOXIN (3 separate figures: IS, HA, pCS), each with 3
  subplots (C_end, AUC, CLr vs Ki, semilog x). Saves PNG+PDF if OUTDIR
  is not None.
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/sensitivity/. The core PBPK engine lives in
# ../../src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — required for unattended/server runs;
                        # also avoids plt.show() blocking in non-interactive use
import matplotlib.pyplot as plt

# ── Version-safe trapezoidal integration ───────────────────────────────────
# NumPy >= 2.0 added np.trapezoid and deprecated np.trapz; later NumPy
# releases removed np.trapz entirely. Use hasattr() rather than a bare
# np.trapz fallback, which would raise AttributeError immediately.
def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

import constants as K
import compounds as C
import simulation as sim
from state_vector import IDX

# Optional CKD scales (harmless in healthy mode, safe if you reuse later)
import compound_parameters as CP
CP.set_ckd_vmax_scales(baso_scale=0.10, apical_scale=0.40)

# Try scenario manager (same pathway as validated plots)
try:
    from scenario_manager import set_scenario
except Exception:
    set_scenario = None


# =============================== CONFIG ======================================

MODE = "healthy"

T_SS_H  = 1500.0   # hours: toxin SS (no drug)
T_RUN_H = 12.0     # hours: drug run duration (your t_sim)

DEFAULT_KI = {"IS": 0.08, "HA": 0.08, "pCS": 0.08}

# Include default Ki EXACTLY in grid:
KI_GRID = np.logspace(np.log10(0.003), np.log10(0.3), 16)
KI_GRID = np.unique(np.sort(np.r_[KI_GRID, DEFAULT_KI["IS"], DEFAULT_KI["HA"], DEFAULT_KI["pCS"]]))

# space to report central concentrations
# - "blood_total" matches your debug outputs if you print sol.y[IDX["C_tox*_cen"]] directly
# - "plasma_total" = blood_total / BP
# - "plasma_unbound" = blood_total * fu_p / BP
CLINICAL_SPACE = "blood_total"

CMP = {
    "IS":  {"state_cen": "C_tox1_cen", "Ae": "Ae_t1"},
    "HA":  {"state_cen": "C_tox2_cen", "Ae": "Ae_t2"},
    "pCS": {"state_cen": "C_tox3_cen", "Ae": "Ae_t3"},
}

# Sweep mode:
#   "single": vary Ki for one toxin, keep other two Kis at DEFAULT
#   "all": vary Ki for all toxins simultaneously (Ki_IS=Ki_HA=Ki_pCS=Ki)
SWEEP_MODE = "single"

# Saving:
# - If OUTDIR is None: do not save
# - If OUTDIR is a folder: saves PNG + PDF
OUTDIR = os.path.join(_THIS_DIR, "..", "..", "results", "ki_sensitivity")   # set to None to disable saving


# =============================== UTILITIES ===================================

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def yvec_to_ic_dict(yvec: np.ndarray) -> dict:
    return {name: float(yvec[i]) for name, i in IDX.items()}


def _series(sol, state_name: str):
    if state_name not in IDX:
        near = [k for k in IDX.keys() if k.startswith(state_name.split("_")[0])]
        raise KeyError(f"State '{state_name}' not in IDX. Near: {near[:10]}")
    i = IDX[state_name]
    return sol.y[i, :], sol.t


def _set_drug_flags(enabled: bool, inhibit: bool) -> None:
    drug = getattr(C, "drug", None)
    if isinstance(drug, dict):
        drug["enabled"] = bool(enabled)
        drug["inhibit"] = bool(inhibit)


def _get_tox_dict(which: str) -> dict:
    if which == "IS":
        return C.tox1
    if which == "HA":
        return C.tox2
    if which == "pCS":
        return C.tox3
    raise ValueError(f"Unknown toxin '{which}'")


def _blood_to_plasma_total(C_blood_total: float, tox_dict: dict) -> float:
    BP = _safe_float(tox_dict.get("BP", 1.0), default=1.0)
    if BP == 0:
        BP = 1.0
    return float(C_blood_total) / float(BP)


def _blood_to_plasma_unbound(C_blood_total: float, tox_dict: dict) -> float:
    BP = _safe_float(tox_dict.get("BP", 1.0), default=1.0)
    if BP == 0:
        BP = 1.0
    fu = _safe_float(tox_dict.get("fu_p", 1.0), default=1.0)
    return float(C_blood_total) * float(fu) / float(BP)


def convert_central(C_cen_blood_total: float, which: str) -> float:
    tox = _get_tox_dict(which)
    if CLINICAL_SPACE == "blood_total":
        return float(C_cen_blood_total)
    if CLINICAL_SPACE == "plasma_total":
        return _blood_to_plasma_total(C_cen_blood_total, tox)
    if CLINICAL_SPACE == "plasma_unbound":
        return _blood_to_plasma_unbound(C_cen_blood_total, tox)
    raise ValueError(f"Unknown CLINICAL_SPACE='{CLINICAL_SPACE}'")


def init_model() -> None:
    # mimic your validated scripts (flows / compound loading)
    _ = K.get_tubular_flows_ml_min(MODE)
    _ = K.get_renal_blood_flow_ml_min(MODE)
    C.load_compounds(MODE)


def _set_all_kis(is_ki: float, ha_ki: float, pcs_ki: float) -> None:
    C.tox1["Ki_furo_to_IS_mgL"]  = float(is_ki)
    C.tox2["Ki_furo_to_HA_mgL"]  = float(ha_ki)
    C.tox3["Ki_furo_to_pCS_mgL"] = float(pcs_ki)


def set_kis_default_all() -> None:
    _set_all_kis(DEFAULT_KI["IS"], DEFAULT_KI["HA"], DEFAULT_KI["pCS"])


def set_kis_inf_all() -> None:
    _set_all_kis(np.inf, np.inf, np.inf)


def set_kis_for_sweep(which: str, Ki: float) -> None:
    Ki = float(Ki)

    if SWEEP_MODE == "all":
        _set_all_kis(Ki, Ki, Ki)
        return

    # single-toxin sweep
    is_ki  = DEFAULT_KI["IS"]
    ha_ki  = DEFAULT_KI["HA"]
    pcs_ki = DEFAULT_KI["pCS"]

    if which == "IS":
        is_ki = Ki
    elif which == "HA":
        ha_ki = Ki
    elif which == "pCS":
        pcs_ki = Ki
    else:
        raise ValueError(f"Unknown toxin '{which}'")

    _set_all_kis(is_ki, ha_ki, pcs_ki)


def run_from_ic_dict(t_end_h: float, ic_dict: dict):
    return sim.run_sim(t_end_h, custom_initial_conds=ic_dict, mode=MODE)


# =============================== METRICS =====================================

def metric_c_end(sol, which: str) -> float:
    """Concentration at end-of-simulation (matches your debug: values at t_sim)."""
    cen_state = CMP[which]["state_cen"]
    Ccen_blood, _t = _series(sol, cen_state)
    return float(convert_central(Ccen_blood[-1], which))


def metrics_auc_ae_clr(sol, which: str, window=(0.0, T_RUN_H)):
    """AUC, Ae, CLr over a window using the same central-space conversion."""
    cen_state = CMP[which]["state_cen"]
    ae_state  = CMP[which]["Ae"]

    Ccen_blood, t = _series(sol, cen_state)
    Ae, _         = _series(sol, ae_state)

    Ccen = np.array([convert_central(c, which) for c in Ccen_blood], dtype=float)

    t0, t1 = window
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))

    tt = np.asarray(t[i0:i1], dtype=float)
    cc = np.clip(np.asarray(Ccen[i0:i1], dtype=float), 0.0, np.inf)
    aa = np.maximum.accumulate(np.asarray(Ae[i0:i1], dtype=float))

    if len(tt) >= 2:
        auc = float(_trapz(cc, tt))
        ae  = float(aa[-1] - aa[0])
    else:
        auc, ae = np.nan, np.nan

    clr = ae / max(auc, 1e-12) if np.isfinite(ae) and np.isfinite(auc) else np.nan
    return auc, ae, clr


# =============================== STEADY STATE ================================

def get_ss_ic():
    """Build toxin SS with DRUG OFF."""
    init_model()
    set_kis_default_all()
    _set_drug_flags(False, False)

    # NOTE: an earlier version of this function tried to route through
    # scenario_manager.set_scenario(mode=..., drug_enabled=..., inhibit=...,
    # ckd_scale=..., initial_conditions=...), but that function's actual
    # signature in this codebase is set_scenario(scenario_key: str) — a
    # single named preset like "H_NO_DRUG", not these keyword arguments.
    # That call would raise TypeError immediately. The init_model() +
    # set_kis_default_all() + _set_drug_flags(False, False) calls above
    # already perform the equivalent setup (healthy/CKD-5 mode, drug off,
    # inhibition off, default Ki values), so we go straight to run_sim().
    sol_ss = sim.run_sim(T_SS_H, mode=MODE)

    y_ss = sol_ss.y[:, -1].copy()
    ic_ss = yvec_to_ic_dict(y_ss)

    ss_end = {k: float(convert_central(y_ss[IDX[CMP[k]["state_cen"]]], k)) for k in ["IS", "HA", "pCS"]}
    return ic_ss, ss_end


# =============================== REFERENCES ==================================

def run_reference_cases(ic_ss: dict):
    """Run reference scenarios starting from toxin SS ICs (same IC for fair comparisons)."""
    refs = {}

    # 1) No Drug
    init_model()
    set_kis_default_all()
    _set_drug_flags(False, False)
    refs["no_drug"] = run_from_ic_dict(T_RUN_H, ic_ss)

    # 2) With Drug + Interaction (DEFAULT Kis)
    init_model()
    set_kis_default_all()
    _set_drug_flags(True, True)
    refs["with_drug_default"] = run_from_ic_dict(T_RUN_H, ic_ss)

    # 3) With Drug (No Interaction)
    init_model()
    set_kis_default_all()
    _set_drug_flags(True, False)
    refs["with_drug_no_interaction"] = run_from_ic_dict(T_RUN_H, ic_ss)

    # 4) With Drug + Ki=inf all (no inhibition effect)
    init_model()
    set_kis_inf_all()
    _set_drug_flags(True, True)
    refs["with_drug_ki_inf"] = run_from_ic_dict(T_RUN_H, ic_ss)

    return refs


# =============================== SWEEP =======================================

def sweep_ki(which: str, ic_ss: dict):
    """Sweep Ki and compute C_end, AUC, CLr (all in chosen CLINICAL_SPACE)."""
    Ki_vals  = []
    Cend_vals = []
    AUC_vals = []
    CLr_vals = []

    for Ki in KI_GRID:
        init_model()
        set_kis_for_sweep(which, Ki)
        _set_drug_flags(True, True)

        sol = run_from_ic_dict(T_RUN_H, ic_ss)

        Ki_vals.append(float(Ki))
        Cend_vals.append(metric_c_end(sol, which))

        auc, _ae, clr = metrics_auc_ae_clr(sol, which)
        AUC_vals.append(float(auc))
        CLr_vals.append(float(clr))

    return {
        "Ki":    np.asarray(Ki_vals, dtype=float),
        "C_end": np.asarray(Cend_vals, dtype=float),
        "AUC":   np.asarray(AUC_vals, dtype=float),
        "CLr":   np.asarray(CLr_vals, dtype=float),
    }


# =============================== PLOTTING + SAVING ===========================

def _ylabel_c(which: str) -> str:
    if CLINICAL_SPACE == "blood_total":
        return f"{which}  total blood concentration at 12 h (mg/L)"
    if CLINICAL_SPACE == "plasma_total":
        return f"{which}  total plasma concentration at 12 h (mg/L)"
    return f"{which}  total plasma concentration at 12 h (mg/L)"


def _save_figure(fig: plt.Figure, outdir: str, basename: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{basename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✔ Saved figure: {path}")


def plot_three_panels(curve: dict, which: str, ss_end: dict, refs: dict):
    """
    One figure with 3 subplots:
      1) C_end vs Ki
      2) AUC vs Ki
      3) CLr vs Ki
    + reference horizontal lines on each.
    """
    Ki   = curve["Ki"]
    Cend = curve["C_end"]
    AUC  = curve["AUC"]
    CLr  = curve["CLr"]

    # Reference endpoints/metrics
    c_no_drug = metric_c_end(refs["no_drug"], which)
    c_default = metric_c_end(refs["with_drug_default"], which)
    c_no_int  = metric_c_end(refs["with_drug_no_interaction"], which)
    c_ki_inf  = metric_c_end(refs["with_drug_ki_inf"], which)

    auc_no_drug, _, clr_no_drug = metrics_auc_ae_clr(refs["no_drug"], which)
    auc_default, _, clr_default = metrics_auc_ae_clr(refs["with_drug_default"], which)
    auc_no_int,  _, clr_no_int  = metrics_auc_ae_clr(refs["with_drug_no_interaction"], which)
    auc_ki_inf,  _, clr_ki_inf  = metrics_auc_ae_clr(refs["with_drug_ki_inf"], which)

    # index of exact default Ki
    default_ki = float(DEFAULT_KI[which])
    idx_default = int(np.where(np.isclose(Ki, default_ki, rtol=0, atol=1e-15))[0][0])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True)
    panel_labels = ["A", "B", "C"]

    for ax, lab in zip(axes, panel_labels):
        ax.text(
            0.015, 0.99, lab,  # inside axes (upper-left)
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left"
        )

    # --- Panel 1: C_end ---
    ax = axes[0]
    ax.semilogx(Ki, Cend, marker="o",color="black", linewidth=2, label=f"Sweep ({SWEEP_MODE})")
    # ax.axhline(ss_end[which], linestyle="-.", linewidth=2, label="Model SS (no drug)")
    ax.axhline(c_no_drug, linestyle=":",color="black", linewidth=2, label="No Drug (12h)")
    ax.axhline(c_default, linestyle="--",color="black", linewidth=2, label="With Drug (DEFAULT Kis)")
    # ax.axhline(c_no_int, linestyle=":", linewidth=2, label="With Drug (No Interaction)")
    # ax.axhline(c_ki_inf, linestyle="--", linewidth=2, label="With Drug (Ki=inf all)")
    ax.plot(Ki[idx_default], Cend[idx_default], "o", markersize=9,color="black", label=f"Point Ki={default_ki:g}")
    # ax.set_title("End concentration (t = 12 h)")
    ax.set_xlabel(r"Inhibition constant $K_i$ (mg/L)",fontsize = 16)
    ax.set_ylabel(_ylabel_c(which),fontsize = 16)
    ax.grid(True, which="both", ls=":")

    # --- Panel 2: AUC ---
    ax = axes[1]
    ax.semilogx(Ki, AUC, marker="o", color="black", linewidth=2, label=f"Sweep ({SWEEP_MODE})")
    ax.axhline(auc_no_drug, linestyle=":",color="black", linewidth=2, label="No Furosemide (12h)")
    ax.axhline(auc_default, linestyle="--",color="black", linewidth=2, label="With Furosemide (DEFAULT Kis)")
    # ax.axhline(auc_no_int, linestyle=":", linewidth=2, label="With Drug (No Interaction)")
    # ax.axhline(auc_ki_inf, linestyle="--", linewidth=2, label="With Drug (Ki=inf all)")
    ax.plot(Ki[idx_default], AUC[idx_default], "o", markersize=9, color="black", label=f"Point Ki={default_ki:g}")
    # ax.set_title("AUC (0–12 h)")
    ax.set_xlabel(r"Inhibition constant $K_i$ (mg/L)", fontsize=16)
    ax.set_ylabel("AUC (mg/L·h)",fontsize=16)
    ax.grid(True, which="both", ls=":")

    # --- Panel 3: CLr ---
    ax = axes[2]
    ax.semilogx(Ki, CLr, marker="o", color="black", linewidth=2, label=f"Sweep ({SWEEP_MODE})")
    ax.axhline(clr_no_drug, linestyle=":",color="black", linewidth=2, label="No Furosemide (12h)")
    ax.axhline(clr_default, linestyle="--",color="black", linewidth=2, label="With Furosemide (DEFAULT Kis)")
    # ax.axhline(clr_no_int, linestyle=":", linewidth=2, label="With Drug (No Interaction)")
    # ax.axhline(clr_ki_inf, linestyle="--", linewidth=2, label="With Drug (Ki=inf all)")
    ax.plot(Ki[idx_default], CLr[idx_default], "o", markersize=9, color="black", label=f"Point Ki={default_ki:g}")
    # ax.set_title("Realised CLr (Ae/AUC)")
    ax.set_xlabel(r"Inhibition constant $K_i$ (mg/L)", fontsize=16)
    ax.set_ylabel("CLr (L/h)")
    ax.grid(True, which="both", ls=":")

    # Shared legend (dedupe)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
        leg = ax.get_legend()
        if leg:
            leg.remove()

    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    fig.legend(list(uniq.values()), list(uniq.keys()),
               loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.06))

    # fig.suptitle(f"Ki sensitivity: furosemide → {which} | mode={MODE} | space={CLINICAL_SPACE} | sweep={SWEEP_MODE}",
    #              y=1.14, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Print match check
    print(f"\n=== MATCH CHECK ({which}) ===")
    print(f"SS_end (from SS solve):                 {ss_end[which]:.6g}")
    print(f"No Furosemide (12h from SS):                  {c_no_drug:.6g}")
    print(f"With Furosemide (DEFAULT Kis) (12h):          {c_default:.6g}")
    print(f"With Furosemide (No Interaction) (12h):       {c_no_int:.6g}")
    print(f"With Furosemide (Ki=inf all) (12h):           {c_ki_inf:.6g}")
    print(f"Sweep point at Ki={default_ki:g}:              {Cend[idx_default]:.6g}")

    # Save
    if OUTDIR is not None:
        basename = f"ki_sensitivity_{MODE}_{CLINICAL_SPACE}_{SWEEP_MODE}_{which}"
        _save_figure(fig, OUTDIR, basename)

    plt.close(fig)


# =============================== MAIN ========================================

def main():
    ic_ss, ss_end = get_ss_ic()
    refs = run_reference_cases(ic_ss)

    toxins = ["IS", "HA", "pCS"]   # run all three by default

    for which in toxins:
        curve = sweep_ki(which, ic_ss)
        plot_three_panels(curve, which, ss_end, refs)


if __name__ == "__main__":
    main()
