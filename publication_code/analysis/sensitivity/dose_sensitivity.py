"""
sa_dose_sensitivity.py

Dose-level sensitivity for the drug (furosemide) with inhibition ON:
- Scales infusion rate IR (total dose = IR * t_inf_h + bolus_mg)
- Keeps inhibition ON so increased drug affects toxin PK via competition
- Metrics per toxin: AUC_total, Ae(12h), CLr = Ae / AUC_free, Cmax
- Drug metrics: AUC_drug (total), Cmax_drug

Outputs:
- figures/dose_sens_results.csv
- figures/dose_sensitivity.png (+ .pdf)

Notes:
- CLr uses FREE AUC for toxins: AUC_free = AUC_total * (fu_p / BP)
- This script assumes the model state vector contains:
    C_ven, C_tox{1,2,3}_cen, Ae_drg, Ae_t{1,2,3}
  Adjust CENTER_NAME / AE_NAME if your canonical names differ.
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

import csv
import math
import inspect
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — required for unattended/server runs
import matplotlib.pyplot as plt

import simulation as sim
import compound_parameters as CP
import compounds as Comp
from state_vector import IDX

# ── Version-safe trapezoidal integration ───────────────────────────────────
# NumPy >= 2.0 added np.trapezoid and deprecated np.trapz; later NumPy
# releases removed np.trapz entirely. Use hasattr() rather than a bare
# np.trapz fallback, which would raise AttributeError immediately.
def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

# ----------------------- Config -----------------------
MODE = "healthy"
DURATION_H = 12.0

# Match your earlier behavior: use linear-ish multipliers (or logspace)
DOSE_MULTIPLIERS = np.logspace(-1, 1, 9)  # 0.1× .. 10×

TOXINS = ["IS", "HA", "pCS"]

CENTER_NAME = {
    "drug": "C_ven",         # central drug conc state name
    "IS":   "C_tox1_cen",
    "HA":   "C_tox2_cen",
    "pCS":  "C_tox3_cen",
}

AE_NAME = {
    "drug": "Ae_drg",
    "IS":   "Ae_t1",
    "HA":   "Ae_t2",
    "pCS":  "Ae_t3",
}

# Output — shared top-level results/ folder, not a local "figures/" subfolder
OUTDIR = os.path.join(_THIS_DIR, "..", "..", "results", "dose_sensitivity")
os.makedirs(OUTDIR, exist_ok=True)
CSV_PATH = os.path.join(OUTDIR, "dose_sens_results.csv")
FIG_PNG  = os.path.join(OUTDIR, "dose_sensitivity.png")
FIG_PDF  = os.path.join(OUTDIR, "dose_sensitivity.pdf")

# Debug prints
VERBOSE = True

# ----------------------- Helpers -----------------------
def _extract_t_y(sol):
    """Return (t, y) arrays from SciPy OdeResult or dict output."""
    if sol is None:
        return None, None
    if isinstance(sol, dict):
        return np.asarray(sol.get("t", None)), np.asarray(sol.get("y", None))
    return np.asarray(getattr(sol, "t", None)), np.asarray(getattr(sol, "y", None))


def _series(sol, state_name: str):
    """Return (series, t) for a canonical state name using IDX mapping."""
    t, y = _extract_t_y(sol)
    if t is None or y is None:
        return None, None
    if state_name not in IDX:
        return None, t
    i = IDX[state_name]
    if i < 0 or i >= y.shape[0]:
        return None, t
    return y[i, :], t


def _auc_trapz(Cp, t) -> float:
    if Cp is None or t is None:
        return float("nan")
    Cp = np.maximum(np.asarray(Cp, dtype=float), 0.0)
    return float(_trapz(Cp, t))


def _cmax(Cp) -> float:
    if Cp is None:
        return float("nan")
    return float(np.nanmax(np.asarray(Cp, dtype=float)))


def _interp_at(t: np.ndarray, y: np.ndarray, T: float) -> float:
    """Linear interpolation with edge handling."""
    if t is None or y is None or len(t) == 0:
        return float("nan")
    T = float(T)
    if T <= t[0]:
        return float(y[0])
    if T >= t[-1]:
        return float(y[-1])
    return float(np.interp(T, t, y))


def _Ae_at(sol, ae_state_name: str, T: float) -> float:
    Ae, t = _series(sol, ae_state_name)
    if Ae is None or t is None:
        return float("nan")
    return _interp_at(t, Ae, T)


def _get_toxin_dict(tag: str) -> Dict[str, Any]:
    """Map toxin tag -> live dict in compounds module."""
    if tag == "IS":
        return Comp.tox1
    if tag == "HA":
        return Comp.tox2
    if tag == "pCS":
        return Comp.tox3
    raise KeyError(f"Unknown toxin tag: {tag}")


def _metrics_one(sol, tag: str) -> Tuple[float, float, float, float]:
    """
    Return (AUC_total, Ae12, CLr, Cmax).

    For toxins, CLr = Ae12 / AUC_free, where:
        AUC_free = AUC_total * (fu_p / BP)

    For drug, CLr is not used (returns NaN).
    """
    c_state = CENTER_NAME[tag]
    a_state = AE_NAME.get(tag, None)

    Cp, t = _series(sol, c_state)
    AUC_total = _auc_trapz(Cp, t)
    Cmax = _cmax(Cp)

    Ae12 = float("nan")
    if a_state is not None:
        Ae12 = _Ae_at(sol, a_state, DURATION_H)

    # CLr
    CLr = float("nan")
    if tag in ("IS", "HA", "pCS"):
        tox = _get_toxin_dict(tag)
        fu_p = float(tox.get("fu_p", 1.0))
        BP   = float(tox.get("BP", 1.0)) or 1.0
        AUC_free = AUC_total * (fu_p / BP)

        if (not math.isnan(AUC_free)) and AUC_free > 0 and (not math.isnan(Ae12)):
            CLr = Ae12 / AUC_free

    return AUC_total, Ae12, CLr, Cmax


def _call_run_sim(t_end_h: float, mode: str):
    """Be robust to signature variations of sim.run_sim."""
    sig = inspect.signature(sim.run_sim)
    if "t_end_h" in sig.parameters:
        return sim.run_sim(t_end_h=t_end_h, mode=mode)
    if "duration_h" in sig.parameters:
        return sim.run_sim(duration_h=t_end_h, mode=mode)
    # fallback
    return sim.run_sim(t_end_h)


def _ensure_drug_on_and_inhibiting():
    """
    Force drug and inhibition flags ON on the LIVE compound dicts.
    This must be called after any get_compound_parameters() rebuild.
    """
    # These keys match your earlier scenario logic
    Comp.drug["enabled"] = True
    Comp.drug["inhibit"] = True

    # keep base config aligned as well
    if isinstance(CP.DRUG_BASE, dict):
        CP.DRUG_BASE["enabled"] = True
        CP.DRUG_BASE["inhibit"] = True


def _baseline_dose() -> Tuple[float, float, float, float]:
    """Return (IR0, t_inf, bolus, total_dose0)."""
    IR0    = float(CP.DRUG_BASE.get("IR", 0.0))
    t_inf  = float(CP.DRUG_BASE.get("t_inf_h", 0.0))
    bolus  = float(CP.DRUG_BASE.get("bolus_mg", 0.0))
    return IR0, t_inf, bolus, (IR0 * t_inf + bolus)


def _rebuild_compounds(mode: str):
    """Rebuild live compounds dicts from parameters module."""
    drug_c, is_c, ha_c, pcs_c = CP.get_compound_parameters(mode)
    Comp.drug, Comp.tox1, Comp.tox2, Comp.tox3 = drug_c, is_c, ha_c, pcs_c


def _sanity_check_required_states():
    needed = list(CENTER_NAME.values()) + list(AE_NAME.values())
    missing = [n for n in needed if n not in IDX]
    if missing:
        print(f"[warn] These states are missing from IDX and will yield NaN: {missing}")


# ----------------------- Sweep ------------------------
def sweep_dose(mults: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
    _sanity_check_required_states()

    # Make sure base has drug on + inhibiting
    _rebuild_compounds(MODE)
    _ensure_drug_on_and_inhibiting()

    IR0, t_inf, bolus, dose0 = _baseline_dose()

    rows: List[Dict[str, Any]] = []
    for m in mults:
        m = float(m)

        # 1) Update base IR for this multiplier
        new_IR = IR0 * m
        CP.DRUG_BASE["IR"] = float(new_IR)

        # 2) Rebuild live dicts so run_sim sees the updated IR
        _rebuild_compounds(MODE)

        # 3) Force drug+inhibition flags again (rebuilds often reset)
        _ensure_drug_on_and_inhibiting()

        # 4) Compute total dose
        total_dose = float(new_IR * t_inf + bolus)

        if VERBOSE:
            print(f"\n[SA] mult={m:.3g}  IR={new_IR:.3f} mg/h  dose={total_dose:.3f} mg")
            print(f"     drug enabled={Comp.drug.get('enabled')} inhibit={Comp.drug.get('inhibit')}")
            # Example Ki check (adjust key names if different)
            print(f"     Ki(IS)={Comp.tox1.get('Ki_furo_to_IS_mgL', 'NA')} mg/L")

        # 5) Run simulation
        sol = _call_run_sim(DURATION_H, MODE)

        # 6) Drug metrics
        AUCd, _, _, Cmaxd = _metrics_one(sol, "drug")

        row: Dict[str, Any] = {
            "dose_mult": m,
            "drug_IR": float(new_IR),
            "drug_t_inf": float(t_inf),
            "drug_bolus": float(bolus),
            "drug_total_dose_mg": total_dose,
            "drug_AUC": float(AUCd),
            "drug_Cmax": float(Cmaxd),
        }

        # 7) Toxin metrics
        for tox in TOXINS:
            AUC, Ae12, CLr, Cmax = _metrics_one(sol, tox)
            row[f"{tox}_AUC"] = float(AUC)
            row[f"{tox}_Ae12h"] = float(Ae12)
            row[f"{tox}_CLr"] = float(CLr)
            row[f"{tox}_Cmax"] = float(Cmax)

        rows.append(row)

    # Restore baseline IR to avoid polluting rest of project
    CP.DRUG_BASE["IR"] = float(IR0)
    _rebuild_compounds(MODE)
    _ensure_drug_on_and_inhibiting()

    return rows, float(dose0)


# ----------------------- Save CSV ---------------------
def save_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        print("[warn] no rows to save")
        return

    # Stable header order
    keys = [
        "dose_mult", "drug_IR", "drug_t_inf", "drug_bolus", "drug_total_dose_mg",
        "drug_AUC", "drug_Cmax",
    ]
    for tox in TOXINS:
        keys += [f"{tox}_AUC", f"{tox}_Ae12h", f"{tox}_CLr", f"{tox}_Cmax"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"[saved] {path}")


# ----------------------- Plot -------------------------
def plot_results(rows: List[Dict[str, Any]], dose0: float):
    x = np.asarray([r["drug_total_dose_mg"] for r in rows], dtype=float)

    def series(key: str) -> np.ndarray:
        return np.asarray([r.get(key, np.nan) for r in rows], dtype=float)

    # Find baseline index (closest to dose0)
    i0 = int(np.nanargmin(np.abs(x - dose0)))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)

    # (1) ΔAUC/AUC0 for toxins
    axes[0].set_title("Dose sensitivity – ΔAUC/AUC₀ (toxins)")
    for tox in TOXINS:
        y = series(f"{tox}_AUC")
        base = y[i0]
        if np.isfinite(base) and base > 0:
            dy = y / base - 1.0
        else:
            dy = np.full_like(y, np.nan)
        axes[0].plot(x, dy, marker="o", label=f"{tox} AUC")
    axes[0].axhline(0, ls="--", lw=1)
    axes[0].set_ylabel("ΔAUC / AUC₀", fontsize=12)
    axes[0].legend()

    # (2) CLr toxins
    axes[1].set_title("Dose sensitivity – CLr (toxins)")
    for tox in TOXINS:
        axes[1].plot(x, series(f"{tox}_CLr"), marker="o", label=f"{tox} CLr")
    axes[1].set_ylabel("CLr (L/h)", fontsize=12)
    axes[1].legend()

    # (3) Drug exposure
    axes[2].set_title("Dose sensitivity – Drug exposure")
    axes[2].plot(x, series("drug_AUC"), marker="o", label="Drug AUC")
    axes[2].plot(x, series("drug_Cmax"), marker="o", label="Drug Cmax")
    axes[2].set_xlabel("Total dose (mg)", fontsize=12)
    axes[2].set_ylabel("Exposure", fontsize=12)
    axes[2].legend()

    for ax in axes:
        ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=200)
    fig.savefig(FIG_PDF)
    print(f"[saved] {FIG_PNG}")
    print(f"[saved] {FIG_PDF}")


# ----------------------- Main -------------------------
def main():
    print("=== Dose-level sensitivity (drug dose) ===")
    rows, dose0 = sweep_dose(DOSE_MULTIPLIERS)
    save_csv(rows, CSV_PATH)
    plot_results(rows, dose0)


if __name__ == "__main__":
    main()
