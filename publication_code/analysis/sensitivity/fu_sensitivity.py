# sa_fu_sensitivity.py
"""
Sensitivity of the kidney PBPK model to plasma unbound fraction (fu_p)
(and fu_cell) scaling for each compound (furosemide, IS, HA, pCS).

Per fu factor:
1) Reset ALL fu to baseline
2) Scale fu_p and fu_cell for the target compound only
3) Run toxin steady-state with DRUG OFF
4) Start drug run from SS but with DRUG states zeroed
5) Run 12 h with DRUG ON + inhibit ON
6) Compute metrics over 0–12 h:
      - AUC (mg/L·h)
      - Ae  (mg)
      - CLr = Ae/AUC (L/h)   (realised clearance, Ae/AUC over same window)
      - Cmax (mg/L)
      - C_end (mg/L)

Figures: absolute + relative, saved as PNG+PDF when OUTDIR is not None.
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
matplotlib.use("Agg")  # headless backend — required for unattended/server runs
import matplotlib.pyplot as plt

import constants as K
import compounds as C
import simulation as sim
from state_vector import IDX

# Optional CKD scaling (harmless in healthy mode)
import compound_parameters as CP
CP.set_ckd_vmax_scales(baso_scale=0.10, apical_scale=0.40)

# Optional scenario manager (we guard against it overwriting fu)
try:
    from scenario_manager import set_scenario
except Exception:
    set_scenario = None

# ── Version-safe trapezoidal integration ───────────────────────────────────
# NumPy >= 2.0 added np.trapezoid and deprecated np.trapz; later NumPy
# releases removed np.trapz entirely. Use hasattr() rather than a bare
# np.trapz fallback, which would raise AttributeError immediately.
def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ------------------------------- config --------------------------------------

MODE   = "healthy"
T_SS_H = 1200.0     # steady-state time (h)
T_RUN  = 12.0       # drug run + metric window (h)

FU_FACTORS = np.logspace(np.log10(0.25), np.log10(4.0), 9)  # 0.25× … 4×

CMP = {
    "tox1": {"label": "Indoxyl sulfate (IS)",   "key": "IS",   "Ccen": "C_tox1_cen", "Ae": "Ae_t1"},
    "tox2": {"label": "Hippuric acid (HA)",     "key": "HA",   "Ccen": "C_tox2_cen", "Ae": "Ae_t2"},
    "tox3": {"label": "p-Cresyl sulfate (pCS)", "key": "pCS",  "Ccen": "C_tox3_cen", "Ae": "Ae_t3"},
    "drug": {"label": "Furosemide",             "key": "DRUG", "Ccen": "C_ven",      "Ae": "Ae_drg"},
}

# Only used as reference line in the toxin C_end panel
CLIN_MEAN = {"IS": 0.51, "HA": 3.00, "pCS": 1.90}

OUTDIR = os.path.join(_THIS_DIR, "..", "..", "results", "fu_sensitivity")   # set None to disable saving


# ---------------------------- low-level utilities ----------------------------

def _set_drug_flags(enabled: bool, inhibit: bool) -> None:
    """Set flags that the ODE should read."""
    if isinstance(getattr(C, "drug", None), dict):
        C.drug["enabled"] = bool(enabled)
        C.drug["inhibit"] = bool(inhibit)

def reset_ode_debug_flags() -> None:
    """Reset one-time debug prints inside ode_system_twocompartment.kidney_rhs
    (if present).

    NOTE: an earlier version of this function imported from a module named
    "ode_system", which does not exist in this codebase (the actual module
    is ode_system_twocompartment.py). The wrong import name meant this
    function silently did nothing (caught by the try/except below) in every
    previous run. Fixed to import from the correct module so the one-time
    debug-print flags inside kidney_rhs are actually reset between sweep
    iterations, giving consistent debug output behavior on every run rather
    than only the first.
    """
    try:
        from ode_system_twocompartment import kidney_rhs
        kidney_rhs._printed_debug = False
        # NOTE: _printed_inhibit_debug is not read anywhere in this
        # codebase's ode_system_twocompartment.py — setting it here is
        # harmless (creates an unused attribute) but has no effect. Kept
        # for forward-compatibility in case a future ODE version adds it.
        kidney_rhs._printed_inhibit_debug = False
        kidney_rhs._apical_guarded = False
        for attr in list(dir(kidney_rhs)):
            if attr.startswith("_printed_cin_") or attr.startswith("_checked_") or attr.startswith("_warned_zero_"):
                try:
                    delattr(kidney_rhs, attr)
                except Exception:
                    pass
    except Exception:
        pass

def _series(sol, name: str):
    if name not in IDX:
        raise KeyError(f"State '{name}' not in IDX.")
    i = IDX[name]
    return np.asarray(sol.y[i, :], float), np.asarray(sol.t, float)

def run_simulation(t_end_h: float, custom_initial_conds=None, mode: str = MODE):
    """Project wrapper (your sim.run_sim uses SciPy Radau and tight tolerances)."""
    return sim.run_sim(t_end_h, custom_initial_conds=custom_initial_conds, mode=mode)

def yvec_to_ic_dict(y: np.ndarray) -> dict:
    return {k: float(y[i]) for k, i in IDX.items()}

def _create_drug_start_conditions(sol_ss):
    """
    From a toxin SS solution, keep toxins at SS but zero ONLY drug-related states:
    - central venous drug C_ven
    - all renal tubular drug states C_{seg}_{V,C,T}
    - bladder C_bladder
    - cumulative excretion Ae_drg
    """
    y0 = np.asarray(sol_ss.y[:, -1], float).copy()

    # central venous drug
    if "C_ven" in IDX:
        y0[IDX["C_ven"]] = 0.0

    # tubular drug states (no prefix -> your drug uses C_{seg}_*)
    for seg in (K.PROX + K.NONPROX + K.CDS):
        for suf in ("V", "C", "T"):
            nm = f"C_{seg}_{suf}"
            if nm in IDX:
                y0[IDX[nm]] = 0.0

    if "C_bladder" in IDX:
        y0[IDX["C_bladder"]] = 0.0
    if "Ae_drg" in IDX:
        y0[IDX["Ae_drg"]] = 0.0

    return yvec_to_ic_dict(y0)


# -------------------------- fu baseline + scaling ----------------------------

FU_BASE = {}

def capture_fu_baselines() -> None:
    """Call AFTER C.load_compounds(MODE)."""
    global FU_BASE
    FU_BASE = {
        "drug": {"fu_p": float(C.drug.get("fu_p", 1.0)), "fu_cell": float(C.drug.get("fu_cell", 1.0))},
        "tox1": {"fu_p": float(C.tox1.get("fu_p", 1.0)), "fu_cell": float(C.tox1.get("fu_cell", 1.0))},
        "tox2": {"fu_p": float(C.tox2.get("fu_p", 1.0)), "fu_cell": float(C.tox2.get("fu_cell", 1.0))},
        "tox3": {"fu_p": float(C.tox3.get("fu_p", 1.0)), "fu_cell": float(C.tox3.get("fu_cell", 1.0))},
    }

def reset_all_fu_to_baseline() -> None:
    if not FU_BASE:
        raise RuntimeError("FU_BASE empty. Call capture_fu_baselines() after C.load_compounds().")
    C.drug["fu_p"], C.drug["fu_cell"] = FU_BASE["drug"]["fu_p"], FU_BASE["drug"]["fu_cell"]
    C.tox1["fu_p"], C.tox1["fu_cell"] = FU_BASE["tox1"]["fu_p"], FU_BASE["tox1"]["fu_cell"]
    C.tox2["fu_p"], C.tox2["fu_cell"] = FU_BASE["tox2"]["fu_p"], FU_BASE["tox2"]["fu_cell"]
    C.tox3["fu_p"], C.tox3["fu_cell"] = FU_BASE["tox3"]["fu_p"], FU_BASE["tox3"]["fu_cell"]

def set_fu_scaled(prefix: str, factor: float) -> None:
    """Scale fu_p and fu_cell from baseline; clamp to [0,1]."""
    if prefix not in FU_BASE:
        raise ValueError(f"Unknown prefix '{prefix}'")
    f = float(factor)

    fu_p0 = float(FU_BASE[prefix]["fu_p"])
    fu_c0 = float(FU_BASE[prefix]["fu_cell"])
    fu_p  = max(0.0, min(1.0, fu_p0 * f))
    fu_c  = max(0.0, min(1.0, fu_c0 * f))

    if prefix == "drug":
        C.drug["fu_p"], C.drug["fu_cell"] = fu_p, fu_c
    elif prefix == "tox1":
        C.tox1["fu_p"], C.tox1["fu_cell"] = fu_p, fu_c
    elif prefix == "tox2":
        C.tox2["fu_p"], C.tox2["fu_cell"] = fu_p, fu_c
    elif prefix == "tox3":
        C.tox3["fu_p"], C.tox3["fu_cell"] = fu_p, fu_c


# ------------------------------ metrics --------------------------------------

def metrics_window(sol, Ccen_name: str, Ae_name: str, window=(0.0, T_RUN)):
    """
    Return: (AUC, Ae, CLr, Cmax, C_end) over window.
    CLr here is realised clearance over the window: Ae/AUC.
    """
    Ccen, t = _series(sol, Ccen_name)
    Ae, _   = _series(sol, Ae_name)

    t0, t1 = window
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    if i1 - i0 < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    tt = t[i0:i1]
    cc = np.clip(Ccen[i0:i1], 0.0, np.inf)
    aa = np.maximum.accumulate(Ae[i0:i1])  # cumulative nondecreasing

    auc  = float(_trapz(cc, tt))
    ae   = float(aa[-1] - aa[0])
    clr  = ae / max(auc, 1e-12) if np.isfinite(auc) else np.nan
    cmax = float(np.max(cc))
    cend = float(cc[-1])

    return auc, ae, clr, cmax, cend


# ------------------------------ core workflow --------------------------------

def _apply_scenario_base(drug_enabled: bool, inhibit: bool):
    """
    Optional scenario hook — DISABLED, always returns None.

    An earlier version of this function called
        scenario_manager.set_scenario(mode=..., drug_enabled=..., inhibit=...,
                                        ckd_scale=..., initial_conditions=...)
    and expected a dict back containing "initial_conditions". This codebase's
    actual set_scenario(scenario_key: str) takes a single named preset string
    (e.g. "H_NO_DRUG") and returns None — it does not accept these keyword
    arguments and does not return initial conditions. Calling it this way
    would raise TypeError immediately.

    This function is kept (rather than removed) because both call sites
    already handle a None return correctly:
        ic = scen["initial_conditions"] if isinstance(scen, dict) ... else None
    Returning None here means custom_initial_conds=None is passed to
    run_simulation(), which is functionally equivalent to what this function
    was trying to achieve — drug enabled/inhibit flags are already applied
    explicitly via _set_drug_flags() right after this call in both call
    sites, and fu scaling is applied explicitly via set_fu_scaled() /
    reset_all_fu_to_baseline(). The scenario_manager indirection added no
    behavior beyond what those explicit calls already do correctly.
    """
    return None

def run_toxin_steady_state(target_prefix: str | None, fu_factor: float):
    """
    Run toxin SS with drug OFF, after applying fu scaling to target_prefix (optional).
    Key rule: if scenario_manager exists, call it FIRST then apply fu scaling,
    so set_scenario cannot wipe the scaling.
    """
    # 0) scenario baseline (may create initial_conditions)
    scen = _apply_scenario_base(drug_enabled=False, inhibit=False)

    # 1) set fu
    reset_all_fu_to_baseline()
    if target_prefix is not None:
        set_fu_scaled(target_prefix, fu_factor)

    # 2) enforce drug flags OFF
    _set_drug_flags(False, False)
    reset_ode_debug_flags()

    # Sanity print (so you can SEE scaling is applied)
    if target_prefix is not None:
        d = FU_BASE[target_prefix]
        if target_prefix == "drug":
            now = (C.drug["fu_p"], C.drug["fu_cell"])
        elif target_prefix == "tox1":
            now = (C.tox1["fu_p"], C.tox1["fu_cell"])
        elif target_prefix == "tox2":
            now = (C.tox2["fu_p"], C.tox2["fu_cell"])
        else:
            now = (C.tox3["fu_p"], C.tox3["fu_cell"])
        print(f"    fu scaling applied to {target_prefix}: "
              f"fu_p {d['fu_p']:.4g}→{now[0]:.4g}, fu_cell {d['fu_cell']:.4g}→{now[1]:.4g}")

    # 3) run
    ic = scen["initial_conditions"] if isinstance(scen, dict) and "initial_conditions" in scen else None
    sol_ss = run_simulation(T_SS_H, custom_initial_conds=ic, mode=MODE)

    y_ss = np.asarray(sol_ss.y[:, -1], float)
    ss_conc = {
        "IS":  float(y_ss[IDX["C_tox1_cen"]]),
        "HA":  float(y_ss[IDX["C_tox2_cen"]]),
        "pCS": float(y_ss[IDX["C_tox3_cen"]]),
    }
    return sol_ss, ss_conc

def run_drug_from_ss(sol_ss, inhibit: bool = True):
    """
    Run 12h drug sim starting from SS but with drug states zeroed.
    """
    _ = _apply_scenario_base(drug_enabled=True, inhibit=inhibit)  # call first (may set IC templates)
    ic_drug = _create_drug_start_conditions(sol_ss)

    _set_drug_flags(True, inhibit)
    reset_ode_debug_flags()

    sol = run_simulation(T_RUN, custom_initial_conds=ic_drug, mode=MODE)

    # quick drug sanity: if enabled, central venous should not remain identically ~0
    if C.drug.get("enabled", False) and "C_ven" in IDX:
        cven = np.asarray(sol.y[IDX["C_ven"]], float)
        if float(np.nanmax(cven)) <= 1e-12:
            print("[WARN] Drug enabled but C_ven stayed ~0. Infusion gate may still read stale flags.")
    return sol

def sweep_fu(prefix: str):
    """
    Sweep fu scaling for ONE compound, returning results + plotting refs.
    """
    meta = CMP[prefix]
    label = meta["label"]
    print(f"\n=== fu sweep: {label} ===")
    print(f"Baseline: fu_p={FU_BASE[prefix]['fu_p']:.4g}, fu_cell={FU_BASE[prefix]['fu_cell']:.4g}")

    results = []

    # baseline factor=1 (no scaling)
    sol_ss0, ss0 = run_toxin_steady_state(target_prefix=None, fu_factor=1.0)
    sol0 = run_drug_from_ss(sol_ss0, inhibit=True)
    auc0, ae0, clr0, cmax0, cend0 = metrics_window(sol0, meta["Ccen"], meta["Ae"])
    results.append({"fu_factor": 1.0, "AUC": auc0, "Ae": ae0, "CLr": clr0, "Cmax": cmax0, "C_end": cend0,
                    "ss_IS": ss0["IS"], "ss_HA": ss0["HA"], "ss_pCS": ss0["pCS"]})

    # sweep other factors
    for factor in FU_FACTORS:
        if np.isclose(factor, 1.0):
            continue

        sol_ss, ss = run_toxin_steady_state(target_prefix=prefix, fu_factor=float(factor))
        sol = run_drug_from_ss(sol_ss, inhibit=True)
        auc, ae, clr, cmax, cend = metrics_window(sol, meta["Ccen"], meta["Ae"])

        results.append({"fu_factor": float(factor), "AUC": auc, "Ae": ae, "CLr": clr, "Cmax": cmax, "C_end": cend,
                        "ss_IS": ss["IS"], "ss_HA": ss["HA"], "ss_pCS": ss["pCS"]})

    results.sort(key=lambda r: r["fu_factor"])

    # relative vs baseline
    base = results[0]
    for r in results:
        for m in ("AUC", "CLr", "Cmax", "C_end"):
            b = base[m]
            r[m + "_rel"] = (r[m] - b) / b if np.isfinite(b) and abs(b) > 1e-12 else np.nan

    refs = {
        "baseline": {m: base[m] for m in ("AUC", "CLr", "Cmax", "C_end")},
        "ss_baseline": {"IS": base["ss_IS"], "HA": base["ss_HA"], "pCS": base["ss_pCS"]},
    }
    return results, refs


# ------------------------------ plotting -------------------------------------

def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def plot_fu_sensitivity(results, prefix: str, refs: dict, outdir: str | None = OUTDIR):
    meta = CMP[prefix]
    label = meta["label"]
    key   = meta["key"]

    f = np.array([r["fu_factor"] for r in results], float)
    AUC  = np.array([r["AUC"] for r in results], float)
    CLr  = np.array([r["CLr"] for r in results], float)
    Cmax = np.array([r["Cmax"] for r in results], float)
    Cend = np.array([r["C_end"] for r in results], float)

    dAUC  = 100.0 * np.array([r["AUC_rel"] for r in results], float)
    dCLr  = 100.0 * np.array([r["CLr_rel"] for r in results], float)
    dCmax = 100.0 * np.array([r["Cmax_rel"] for r in results], float)
    dCend = 100.0 * np.array([r["C_end_rel"] for r in results], float)

    try:
        import plotting
        if hasattr(plotting, "setup_professional_style"):
            plotting.setup_professional_style()
    except Exception:
        pass

    # ABSOLUTE
    fig1, ax = plt.subplots(1, 4, figsize=(15.5, 4.0))
    fig1.suptitle(f"fu sensitivity — {label} (mode={MODE})", y=1.04)

    ax[0].semilogx(f, AUC,  marker="o", lw=2)
    ax[1].semilogx(f, CLr,  marker="o", lw=2)
    ax[2].semilogx(f, Cmax, marker="o", lw=2)
    ax[3].semilogx(f, Cend, marker="o", lw=2)

    ax[0].axhline(refs["baseline"]["AUC"],  ls="--", lw=2, alpha=0.6, label="baseline (×1)")
    ax[1].axhline(refs["baseline"]["CLr"],  ls="--", lw=2, alpha=0.6, label="baseline (×1)")
    ax[2].axhline(refs["baseline"]["Cmax"], ls="--", lw=2, alpha=0.6, label="baseline (×1)")
    ax[3].axhline(refs["baseline"]["C_end"],ls="--", lw=2, alpha=0.6, label="baseline (×1)")

    ax[0].set_title("AUC (0–12 h)");              ax[0].set_ylabel("AUC (mg/L·h)")
    ax[1].set_title("Realised CLr (Ae/AUC)");     ax[1].set_ylabel("CLr (L/h)")
    ax[2].set_title("Cmax (0–12 h)");             ax[2].set_ylabel("Cmax (mg/L)")
    ax[3].set_title("C_end (t=12 h)");            ax[3].set_ylabel("C_end (mg/L)")
    for a in ax:
        a.set_xlabel("fu factor")
        a.grid(True, which="both", ls=":")

    # extra refs for toxins on the C_end axis
    if prefix != "drug":
        ss_val = refs["ss_baseline"].get(key, None)
        if ss_val is not None:
            ax[3].axhline(ss_val, ls="-.", lw=2, alpha=0.7, label="model SS (no drug)")
        clin = CLIN_MEAN.get(key, None)
        if clin is not None:
            ax[3].axhline(clin, ls=":", lw=2.5, alpha=0.7, label="clinical mean")

    # shared legend (dedupe)
    handles, labels_ = [], []
    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles += h; labels_ += l
        if a.get_legend():
            a.get_legend().remove()
    uniq = {}
    for h, l in zip(handles, labels_):
        if l not in uniq:
            uniq[l] = h
    fig1.legend(list(uniq.values()), list(uniq.keys()),
                loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig1.tight_layout(rect=[0, 0, 1, 0.92])

    # RELATIVE
    fig2, ax2 = plt.subplots(1, 4, figsize=(15.5, 4.0))
    fig2.suptitle(f"fu sensitivity — {label} (relative to baseline ×1) (mode={MODE})", y=1.04)

    for a, y, ttl in zip(
        ax2,
        (dAUC, dCLr, dCmax, dCend),
        ("ΔAUC (%)", "ΔCLr (%)", "ΔCmax (%)", "ΔC_end (%)")
    ):
        a.semilogx(f, y, marker="o", lw=2)
        a.axhline(0, ls="--", lw=1, alpha=0.6)
        a.set_title(ttl)
        a.set_xlabel("fu factor")
        a.set_ylabel(ttl)
        a.grid(True, which="both", ls=":")

    fig2.tight_layout()

    # Save
    if outdir is not None:
        outdir = _ensure_outdir(outdir)
        base = os.path.join(outdir, f"fu_sensitivity_{MODE}_{prefix}")
        for ext in ("png", "pdf"):
            fig1.savefig(f"{base}_abs.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
            fig2.savefig(f"{base}_rel.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"✔ Saved: {base}_[abs/rel].[png/pdf]")

    plt.close(fig1)
    plt.close(fig2)
    return fig1, fig2


# --------------------------------- main --------------------------------------

def main():
    print("=== Initializing model ===")
    _ = K.get_tubular_flows_ml_min(MODE)
    _ = K.get_renal_blood_flow_ml_min(MODE)

    C.load_compounds(MODE)
    capture_fu_baselines()

    print("Baseline fu values captured:")
    for p, d in FU_BASE.items():
        print(f"  {p:4s}: fu_p={d['fu_p']:.4g}, fu_cell={d['fu_cell']:.4g}")

    all_results = {}

    for prefix in ["tox1", "tox2", "tox3", "drug"]:
        print("\n" + "=" * 78)
        print(f"Analyzing: {CMP[prefix]['label']}")
        print("=" * 78)

        res, refs = sweep_fu(prefix)
        all_results[prefix] = {"results": res, "refs": refs}

        plot_fu_sensitivity(res, prefix, refs, outdir=OUTDIR)

    return all_results

if __name__ == "__main__":
    all_results = main()
