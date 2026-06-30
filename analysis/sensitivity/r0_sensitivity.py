# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # FIX: force non-interactive backend — prevents Windows TkAgg crash
"""
sa_r0_sensitivity_v2.py
========================
R0 sensitivity analysis — Healthy and CKD-5. FIXED VERSION.

BUGS FIXED vs original:
────────────────────────
BUG 1 [HIGH] _rebuild_compounds() silently reset Ki on every iteration.
  Fix: _restore_ki(ki_current) called after every _rebuild_compounds().

BUG 2 [HIGH] _call_run_sim() fallback branch dropped mode argument.
  Fix: fallback now passes mode explicitly.

BUG 3 [HIGH] No SS warm-up — CKD-5 toxins need ~150h to reach new SS
  after R0 changes. Running only 12h gave Css at 20.9% of true new SS,
  severely underestimating R0 sensitivity in CKD-5.
  Fix: each R0 point runs a proper SS warm-up (SS_WARMUP_H=500h) before
  the 12h measurement window. Healthy needs only ~6h (tau~1.9h) but uses
  the same warm-up for consistency and correctness.

BUG 4 [MEDIUM] CLr used AUC_free (unbound AUC), giving CLr,u not CLr,total.
  Fix: CLr now uses AUC_total (total plasma AUC), consistent with v4 script.
  Axis label updated to CL_R,total (L/h).

BUG 5 [MEDIUM] Css_12h mislabelled as steady-state. Fixed via BUG 3 fix.
  Row C/F now correctly labelled 'Css at SS (end of warm-up)'.

BUG 6 [LOW] _ensure_furo_on_and_inhibiting mutated CP.DRUG_BASE permanently.
  Fix: only Comp.drug is modified; CP.DRUG_BASE is never touched.

BUG 7 [LOW] R0_BASE read at import time — now read fresh in sweep_r0().

SCIENTIFIC RATIONALE:
─────────────────────
Css = R0 / CLR at true steady state.
If CLR is unaffected by R0 (linear regime, OAT1 unsaturated), then Css
scales linearly with R0 and CLR is flat vs R0 multiplier.
Non-linearity in Row B/E (CLR decreasing at high R0) indicates OAT1
saturation — an important mechanistic finding for CKD-5 where toxin
concentrations are already high (IS ~20 mg/L >> Km=3.84 mg/L).

SS WARM-UP TIMES:
  Healthy IS: tau = Vd/CL_R = 14/7.52 = 1.9h → 95% SS at ~6h
  CKD-5  IS: tau = Vd/CL_R = 14/0.274 = 51h → 95% SS at ~153h
  SS_WARMUP_H = 500h used for both modes to guarantee true SS.
"""
# -*- coding: utf-8 -*-
"""
run_albumin_scenarios_v4.py
============================
IMPROVED VERSION — publication-quality figure with visible effects.

KEY CHANGES vs v3:
──────────────────
1. Panel C now shows BOTH:
   - Acute IV ODE result (t=12h after 120mg IV) — mechanistically correct
     but small because drug clears before toxins can re-equilibrate
   - Analytical chronic oral SS prediction — clinically relevant comparison
     matching Costes-Albrespic 2025 scenario (chronic daily oral furosemide)
   This makes the story clear: the effect is small acutely but substantial
   chronically, and scales with fu_p/albumin level.

2. Extended scenarios: added CKD-5 uremic extreme (fu_p=0.12) representing
   severe uremic retention + CMPF/oleate displacement at very low albumin
   (<20 g/L). This is an exploratory extrapolation beyond Deltombe 2015
   direct measurements, labelled explicitly as such.

3. Two dose levels shown in Panel C analytical: 120mg/day and 240mg/day
   to demonstrate dose-threshold effect within the albumin scenarios.

4. 300 dpi publication quality, Nature/Lancet typography conventions,
   clean greyscale-compatible palette, proper axis labelling.
"""
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # FIX: force non-interactive backend — prevents Windows TkAgg crash
"""
sa_r0_sensitivity_v2.py
========================
R0 sensitivity analysis — Healthy and CKD-5. FIXED VERSION.

BUGS FIXED vs original:
────────────────────────
BUG 1 [HIGH] _rebuild_compounds() silently reset Ki on every iteration.
  Fix: _restore_ki(ki_current) called after every _rebuild_compounds().

BUG 2 [HIGH] _call_run_sim() fallback branch dropped mode argument.
  Fix: fallback now passes mode explicitly.

BUG 3 [HIGH] No SS warm-up — CKD-5 toxins need ~150h to reach new SS
  after R0 changes. Running only 12h gave Css at 20.9% of true new SS,
  severely underestimating R0 sensitivity in CKD-5.
  Fix: each R0 point runs a proper SS warm-up (SS_WARMUP_H=500h) before
  the 12h measurement window. Healthy needs only ~6h (tau~1.9h) but uses
  the same warm-up for consistency and correctness.

BUG 4 [MEDIUM] CLr used AUC_free (unbound AUC), giving CLr,u not CLr,total.
  Fix: CLr now uses AUC_total (total plasma AUC), consistent with v4 script.
  Axis label updated to CL_R,total (L/h).

BUG 5 [MEDIUM] Css_12h mislabelled as steady-state. Fixed via BUG 3 fix.
  Row C/F now correctly labelled 'Css at SS (end of warm-up)'.

BUG 6 [LOW] _ensure_furo_on_and_inhibiting mutated CP.DRUG_BASE permanently.
  Fix: only Comp.drug is modified; CP.DRUG_BASE is never touched.

BUG 7 [LOW] R0_BASE read at import time — now read fresh in sweep_r0().

SCIENTIFIC RATIONALE:
─────────────────────
Css = R0 / CLR at true steady state.
If CLR is unaffected by R0 (linear regime, OAT1 unsaturated), then Css
scales linearly with R0 and CLR is flat vs R0 multiplier.
Non-linearity in Row B/E (CLR decreasing at high R0) indicates OAT1
saturation — an important mechanistic finding for CKD-5 where toxin
concentrations are already high (IS ~20 mg/L >> Km=3.84 mg/L).

SS WARM-UP TIMES:
  Healthy IS: tau = Vd/CL_R = 14/7.52 = 1.9h → 95% SS at ~6h
  CKD-5  IS: tau = Vd/CL_R = 14/0.274 = 51h → 95% SS at ~153h
  SS_WARMUP_H = 500h used for both modes to guarantee true SS.
"""

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

# ── Config ────────────────────────────────────────────────────────────────────
MODE_HEALTHY  = "healthy"
MODE_CKD      = "ckd5"
SS_WARMUP_H   = 500.0   # h — BUG 3 FIX: long enough for CKD-5 IS (tau=51h)
DURATION_H    = 12.0    # h — measurement window after SS warm-up
SHOW_PLOT     = False   # set True for interactive display

R0_MULTIPLIERS = np.array([0.50, 0.67, 0.80, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00])

TOXINS = ["IS", "HA", "pCS"]

CENTER_NAME = {"IS": "C_tox1_cen", "HA": "C_tox2_cen", "pCS": "C_tox3_cen"}
AE_NAME     = {"IS": "Ae_t1",      "HA": "Ae_t2",      "pCS": "Ae_t3"}
R0_KEY      = "R0"

OUTDIR   = os.path.join(_THIS_DIR, "..", "..", "results", "r0_sensitivity")
os.makedirs(OUTDIR, exist_ok=True)
CSV_PATH_H = os.path.join(OUTDIR, "r0_sens_results_healthy.csv")
CSV_PATH_C = os.path.join(OUTDIR, "r0_sens_results_ckd5.csv")
FIG_PNG    = os.path.join(OUTDIR, "r0_sensitivity_healthy_ckd5.png")
FIG_PDF    = os.path.join(OUTDIR, "r0_sensitivity_healthy_ckd5.pdf")

VERBOSE = True


# ── BUG 1 FIX: Ki restore ─────────────────────────────────────────────────────
def _restore_ki(ki_value: float) -> None:
    """
    Explicitly restore Ki in all live toxin dicts after _rebuild_compounds().
    _rebuild_compounds() resets Ki to compound_parameters.py defaults.
    """
    for tox, key in [
        (Comp.tox1, "Ki_furo_to_IS_mgL"),
        (Comp.tox2, "Ki_furo_to_HA_mgL"),
        (Comp.tox3, "Ki_furo_to_pCS_mgL"),
    ]:
        if isinstance(tox, dict):
            tox[key] = ki_value


# ── BUG 6 FIX: only modify Comp.drug, never CP.DRUG_BASE ─────────────────────
def _ensure_furo_on_and_inhibiting() -> None:
    """Enable furosemide and inhibition in the live compound dict ONLY."""
    Comp.drug["enabled"] = True
    Comp.drug["inhibit"] = True
    # CP.DRUG_BASE is NOT touched — avoids permanent state pollution


# ── Compound management ───────────────────────────────────────────────────────
def _rebuild_compounds(mode: str) -> None:
    drug_c, is_c, ha_c, pcs_c = CP.get_compound_parameters(mode)
    Comp.drug  = drug_c
    Comp.tox1  = is_c
    Comp.tox2  = ha_c
    Comp.tox3  = pcs_c


def _apply_r0_multiplier(mode: str, mult: float,
                          r0_base: Dict[str, float]) -> None:
    """
    Apply R0 multiplier to live toxin dicts.
    r0_base is passed explicitly (BUG 7 FIX — not captured at import time).
    """
    tox_map = {"IS": Comp.tox1, "HA": Comp.tox2, "pCS": Comp.tox3}
    for tag, tox_dict in tox_map.items():
        base = r0_base.get(tag, float("nan"))
        if math.isfinite(base) and isinstance(tox_dict, dict):
            tox_dict[R0_KEY] = base * mult


# ── BUG 2 FIX: mode always passed ─────────────────────────────────────────────
def _call_run_sim(t_end_h: float, mode: str) -> Any:
    """Call run_sim() with correct signature, always passing mode."""
    sig = inspect.signature(sim.run_sim)
    if "t_end_h" in sig.parameters:
        return sim.run_sim(t_end_h=t_end_h, mode=mode)
    if "duration_h" in sig.parameters:
        return sim.run_sim(duration_h=t_end_h, mode=mode)
    # BUG 2 FIX: fallback now includes mode
    try:
        return sim.run_sim(t_end_h, mode=mode)
    except TypeError:
        return sim.run_sim(t_end_h)   # last resort only


def _call_run_sim_with_ic(t_end_h: float, mode: str,
                           ic: Dict[str, float]) -> Any:
    """Call run_sim() with custom initial conditions."""
    sig = inspect.signature(sim.run_sim)
    kwargs = {"mode": mode}
    if "custom_initial_conds" in sig.parameters:
        kwargs["custom_initial_conds"] = ic
    if "t_end_h" in sig.parameters:
        return sim.run_sim(t_end_h=t_end_h, **kwargs)
    if "duration_h" in sig.parameters:
        return sim.run_sim(duration_h=t_end_h, **kwargs)
    return sim.run_sim(t_end_h, **kwargs)


# ── BUG 3 FIX: SS warm-up per R0 point ───────────────────────────────────────
def _run_ss_warmup(mode: str, ki_value: float,
                   r0_base: Dict[str, float], mult: float) -> Any:
    """
    Run SS warm-up (SS_WARMUP_H hours) with the given R0 multiplier.

    This is the critical fix for BUG 3.
    CKD-5 IS has tau=51h — needs >150h to reach 95% of new SS.
    500h guarantees >99.9% convergence for all toxins in both modes.

    Returns sol_ss (the warm-up solution).
    """
    _rebuild_compounds(mode)
    _restore_ki(ki_value)
    _apply_r0_multiplier(mode, mult, r0_base)

    # No drug during warm-up — measure pure toxin SS
    Comp.drug["enabled"] = False
    Comp.drug["inhibit"] = False

    sol_ss = _call_run_sim(SS_WARMUP_H, mode)
    return sol_ss


def _make_ic_from_ss(sol_ss) -> Dict[str, float]:
    """Extract SS initial conditions; reset drug and Ae compartments."""
    y0 = sol_ss.y[:, -1].copy()
    for key in ("C_ven", "C_peri", "Ae_drg", "C_bladder",
                "Ae_t1", "Ae_t2", "Ae_t3"):
        if key in IDX:
            y0[IDX[key]] = 0.0
    return {name: float(y0[i]) for name, i in IDX.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────
def _get_toxin_dict(tag: str) -> Dict[str, Any]:
    if tag == "IS":  return Comp.tox1
    if tag == "HA":  return Comp.tox2
    if tag == "pCS": return Comp.tox3
    raise KeyError(f"Unknown tag: {tag}")


def _extract_t_y(sol):
    if sol is None:
        return None, None
    if isinstance(sol, dict):
        return np.asarray(sol.get("t")), np.asarray(sol.get("y"))
    return np.asarray(getattr(sol, "t")), np.asarray(getattr(sol, "y"))


def _series(sol, state_name: str):
    t, y = _extract_t_y(sol)
    if t is None or y is None or state_name not in IDX:
        return None, t
    i = IDX[state_name]
    return (y[i, :] if 0 <= i < y.shape[0] else None), t


def _interp_at(t, y, T: float) -> float:
    if t is None or y is None or len(t) == 0:
        return float("nan")
    T = float(T)
    if T <= t[0]:  return float(y[0])
    if T >= t[-1]: return float(y[-1])
    return float(np.interp(T, t, y))


def _metrics_one_tox(sol, tag: str) -> Tuple[float, float, float, float]:
    """
    Return (AUC_total, Ae_end, CLr_total, Css_end).

    BUG 4 FIX: CLr uses AUC_total (total plasma AUC), not AUC_free.
    This is consistent with CL_R in dose_response_analysis_v4.py.
    CLr_total = Ae / AUC_total
    """
    Cp, t = _series(sol, CENTER_NAME[tag])
    Ae, _ = _series(sol, AE_NAME[tag])

    AUC_total = float(_trapz(Cp, t)) if (Cp is not None and t is not None) else float("nan")
    Ae_end    = float(Ae[-1] - Ae[0])  if Ae is not None else float("nan")
    # BUG 5 FIX: read Css at end of run — this is true SS because warm-up guarantees it
    Css_end   = float(Cp[-1])           if Cp is not None else float("nan")

    CLr = Ae_end / AUC_total if (math.isfinite(AUC_total) and AUC_total > 1e-12
                                  and math.isfinite(Ae_end)) else float("nan")
    return AUC_total, Ae_end, CLr, Css_end


# ── Main sweep ────────────────────────────────────────────────────────────────
def sweep_r0(mode: str,
             mults: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
    """
    Sweep R0 multipliers. For each point:
      1. Run SS warm-up (500h, no drug) with modified R0.
      2. Read SS toxin concentrations.
      3. Run 12h measurement window with furosemide + inhibition.
      4. Compute metrics.
    """
    # BUG 7 FIX: read R0 base fresh here, not at import time
    r0_base_healthy = {
        "IS":  float(CP.TOX1_BASE.get(R0_KEY, float("nan"))),
        "HA":  float(CP.TOX2_BASE.get(R0_KEY, float("nan"))),
        "pCS": float(CP.TOX3_BASE.get(R0_KEY, float("nan"))),
    }
    r0_base_ckd5 = {
        "IS":  float(CP._CKD_FU_P.get("IS",  {}).get(R0_KEY,
                     r0_base_healthy["IS"])),
        "HA":  float(CP._CKD_FU_P.get("HA",  {}).get(R0_KEY,
                     r0_base_healthy["HA"])),
        "pCS": float(CP._CKD_FU_P.get("pCS", {}).get(R0_KEY,
                     r0_base_healthy["pCS"])),
    }
    r0_base = r0_base_healthy if mode == MODE_HEALTHY else r0_base_ckd5

    # Read current Ki
    ki_current = float(
        CP.get_compound_parameters("healthy")[1].get("Ki_furo_to_IS_mgL", 3.77)
    )

    print(f"\n  sweep_r0({mode}): Ki={ki_current} mg/L")
    print(f"  Baseline R0: IS={r0_base['IS']:.3f}  "
          f"HA={r0_base['HA']:.3f}  pCS={r0_base['pCS']:.3f} mg/h")
    print(f"  SS warm-up: {SS_WARMUP_H:.0f}h per R0 point "
          f"(CKD-5 IS tau~51h → guarantees >99.9% convergence)")

    rows: List[Dict[str, Any]] = []

    for mult in mults:
        mult = float(mult)

        if VERBOSE:
            print(f"\n  [R0×{mult:.2f}] Running SS warm-up ({SS_WARMUP_H:.0f}h)...")

        # ── Step 1: SS warm-up (BUG 3 FIX) ──────────────────────────────────
        sol_ss = _run_ss_warmup(mode, ki_current, r0_base, mult)
        ic     = _make_ic_from_ss(sol_ss)

        # Read SS concentrations at end of warm-up
        Css_ss = {
            tag: float(sol_ss.y[IDX[CENTER_NAME[tag]], -1])
            for tag in TOXINS if CENTER_NAME[tag] in IDX
        }
        if VERBOSE:
            print(f"    SS reached: IS={Css_ss.get('IS',float('nan')):.3f}  "
                  f"HA={Css_ss.get('HA',float('nan')):.3f}  "
                  f"pCS={Css_ss.get('pCS',float('nan')):.3f} mg/L")

        # ── Step 2: 12h measurement with furosemide + inhibition ─────────────
        _rebuild_compounds(mode)
        _restore_ki(ki_current)                # BUG 1 FIX
        _apply_r0_multiplier(mode, mult, r0_base)
        _ensure_furo_on_and_inhibiting()       # BUG 6 FIX

        if VERBOSE:
            print(f"    Running 12h measurement (furo ON, inhibit ON)...")
            print(f"    Ki_IS={Comp.tox1.get('Ki_furo_to_IS_mgL','?')} mg/L  "
                  f"R0_IS={Comp.tox1.get(R0_KEY,'?'):.3f} mg/h")

        sol = _call_run_sim_with_ic(DURATION_H, mode, ic)

        # ── Step 3: Collect metrics ───────────────────────────────────────────
        row: Dict[str, Any] = {
            "mode":     mode,
            "r0_mult":  mult,
        }
        for tag in TOXINS:
            r0_val = r0_base.get(tag, float("nan")) * mult
            row[f"{tag}_R0"] = r0_val
            row[f"{tag}_Css_ss"] = Css_ss.get(tag, float("nan"))

            AUC, Ae, CLr, Css = _metrics_one_tox(sol, tag)
            row[f"{tag}_AUC"]    = AUC
            row[f"{tag}_Ae12h"]  = Ae
            row[f"{tag}_CLr"]    = CLr
            row[f"{tag}_Css12h"] = Css

            if VERBOSE:
                print(f"    {tag}: CLr={CLr:.4f} L/h  "
                      f"Css_SS={Css_ss.get(tag,float('nan')):.3f}  "
                      f"Css_12h={Css:.3f} mg/L")

        rows.append(row)

    # Restore baseline
    _rebuild_compounds(mode)
    _restore_ki(ki_current)

    return rows, float(r0_base.get("IS", float("nan")))


# ── CSV ───────────────────────────────────────────────────────────────────────
def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        print("[warn] no rows to save")
        return
    keys = ["mode", "r0_mult"]
    for tag in TOXINS:
        keys += [f"{tag}_R0", f"{tag}_Css_ss",
                 f"{tag}_AUC", f"{tag}_Ae12h",
                 f"{tag}_CLr", f"{tag}_Css12h"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[saved] {path}")


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_results(rows_h: List[Dict[str, Any]],
                 rows_c: List[Dict[str, Any]]) -> None:

    tox_styles = {
        "IS":  dict(color="0.15", linestyle="-",  marker="o",
                    linewidth=1.6, markersize=4),
        "HA":  dict(color="0.35", linestyle="--", marker="o",
                    linewidth=1.6, markersize=4),
        "pCS": dict(color="0.55", linestyle=":",  marker="o",
                    linewidth=1.6, markersize=4),
    }

    def _x(rows):
        return np.asarray([r["r0_mult"] for r in rows], dtype=float)

    def _col(rows, key):
        return np.asarray([r.get(key, np.nan) for r in rows], dtype=float)

    x_h = _x(rows_h)
    x_c = _x(rows_c)
    i0_h = int(np.nanargmin(np.abs(x_h - 1.0)))
    i0_c = int(np.nanargmin(np.abs(x_c - 1.0)))

    fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.2), sharex="col")

    # Row 1: ΔAUC/AUC0
    for col, (rows, x, i0) in enumerate(
            [(rows_h, x_h, i0_h), (rows_c, x_c, i0_c)]):
        ax = axes[0, col]
        for tox in TOXINS:
            y    = _col(rows, f"{tox}_AUC")
            base = y[i0] if (len(y) > i0 and math.isfinite(y[i0])) \
                   else float("nan")
            dy   = (y / base - 1.0) if (math.isfinite(base) and base > 0) \
                   else np.full_like(y, float("nan"))
            ax.plot(x, dy, label=tox, **tox_styles[tox])
        ax.axhline(0,   color="0.7", linestyle="--", linewidth=1.0)
        ax.axvline(1.0, color="0.7", linestyle=":",  linewidth=0.8)
        ax.set_ylabel(r"$\Delta$AUC / AUC$_0$", fontsize=12)
        ax.set_xlabel("R0 multiplier", fontsize=11)

    # Row 2: CL_R,total (BUG 4 FIX — label updated)
    for col, (rows, x) in enumerate([(rows_h, x_h), (rows_c, x_c)]):
        ax = axes[1, col]
        for tox in TOXINS:
            ax.plot(x, _col(rows, f"{tox}_CLr"), label=tox, **tox_styles[tox])
        ax.axvline(1.0, color="0.7", linestyle=":", linewidth=0.8)
        ax.set_ylabel(r"CL$_{R,total}$ (L/h)", fontsize=12)
        ax.set_xlabel("R0 multiplier", fontsize=11)

    # Row 3: True SS Css (BUG 5 FIX — uses Css_ss from warm-up end)
    for col, (rows, x) in enumerate([(rows_h, x_h), (rows_c, x_c)]):
        ax = axes[2, col]
        for tox in TOXINS:
            ax.plot(x, _col(rows, f"{tox}_Css_ss"), label=tox,
                    **tox_styles[tox])
        ax.axvline(1.0, color="0.7", linestyle=":", linewidth=0.8)
        ax.set_ylabel("Css at true SS (mg/L)", fontsize=12)
        ax.set_xlabel("R0 multiplier", fontsize=11)

    # Panel letters
    letters = {(0,0):"A",(0,1):"D",(1,0):"B",(1,1):"E",(2,0):"C",(2,1):"F"}
    for (r, c), lab in letters.items():
        axes[r, c].text(0.02, 0.95, lab,
                        transform=axes[r, c].transAxes,
                        ha="left", va="top", fontsize=13, fontweight="bold")

    # Cosmetics
    for r in range(3):
        for c in range(2):
            axes[r, c].grid(alpha=0.25)
            axes[r, c].tick_params(axis="both", labelsize=11)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order   = ["IS", "HA", "pCS"]
    ordered = [(handles[labels.index(k)], k) for k in order if k in labels]
    if ordered:
        h, l = zip(*ordered)
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
                   fontsize=11, bbox_to_anchor=(0.5, 0.01))

    # Column headers
    fig.text(0.27, 0.995, "Healthy", ha="center", va="top", fontsize=12)
    fig.text(0.75, 0.995, "CKD-5",   ha="center", va="top", fontsize=12)

    # Row annotations
    row_notes = [
        r"Row A/D: Relative AUC change vs baseline R0",
        r"Row B/E: CL$_{R,total}$ — flat = linear regime, slope = OAT1 saturation",
        r"Row C/F: True SS plasma concentration (after 500h warm-up per R0 point)",
    ]
    for i, txt in enumerate(row_notes):
        fig.text(0.995, 0.88 - i*0.30, txt, ha="right", va="top",
                 fontsize=7.5, color="0.4")

    fig.suptitle(
        f"Toxin generation rate (R0) sensitivity — Healthy and CKD-5\n"
        f"Furosemide 120 mg IV, inhibition ON | "
        f"SS warm-up {SS_WARMUP_H:.0f}h per point | "
        f"Ki from compound_parameters.py",
        fontsize=10.5, fontweight="bold", y=1.02,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    print(f"[saved] {FIG_PNG}")
    print(f"[saved] {FIG_PDF}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("  R0 SENSITIVITY ANALYSIS v2 — all bugs fixed")
    print(f"  R0 range: {R0_MULTIPLIERS[0]:.2f}× to {R0_MULTIPLIERS[-1]:.2f}×")
    print(f"  SS warm-up per point: {SS_WARMUP_H:.0f}h")
    print(f"  Measurement window:   {DURATION_H:.0f}h (furosemide + inhibition)")
    print("=" * 65)

    print("\nRunning Healthy...")
    rows_h, _ = sweep_r0(MODE_HEALTHY, R0_MULTIPLIERS)
    save_csv(rows_h, CSV_PATH_H)

    print("\nRunning CKD-5...")
    rows_c, _ = sweep_r0(MODE_CKD, R0_MULTIPLIERS)
    save_csv(rows_c, CSV_PATH_C)

    plot_results(rows_h, rows_c)

    print("\nDone.")
    print(f"  Healthy CSV : {CSV_PATH_H}")
    print(f"  CKD-5   CSV : {CSV_PATH_C}")
    print(f"  Figure      : {FIG_PNG}")


if __name__ == "__main__":
    main()