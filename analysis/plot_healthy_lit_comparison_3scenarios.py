# plot_healthy_lit_comparison_3S.py
# -*- coding: utf-8 -*-
"""
plot_healthy_lit_comparison_3S.py
=================================
Healthy scenario comparison — three scenarios:

    1) Healthy No Furosemide
    2) Healthy With Furosemide (No Interaction)
    3) Healthy With Furosemide

SIMULATION PROTOCOL
-------------------
Step 1 — Steady-state warm-up (1500 h, no drug):
    Runs the model without furosemide until IS, HA, pCS reach healthy
    steady state. 1500 h (62.5 days) is more than sufficient for all
    three toxins to equilibrate (slowest is pCS, t½_ss ≈ several days).

Step 2 — Extract two sets of initial conditions from the SS endpoint:
    ic_ss_no   : full steady state — used for "No Furosemide" scenario
    ic_ss_drug : same as above but with all drug compartments zeroed —
                 used for both drug scenarios (clean slate for drug)

Step 3 — Three 12 h simulations from the SS initial conditions.

WHY THIS PROTOCOL MATTERS
--------------------------
Starting all three scenarios from the same steady-state initial conditions
ensures that any difference between the curves is caused ONLY by the
presence or absence of furosemide — not by different starting toxin levels.

WHAT THE FIGURES SHOW
----------------------
fig_time (1×3 panels):
    Plasma time-courses of IS, HA, pCS over 12 h.
    All three scenarios start at the same steady-state concentration.
    "With Furosemide" should show a transient rise during infusion
    (reduced OAT1 secretion → toxins accumulate) then return to SS.

fig_bars (1×2 panels):
    Panel A: Final plasma concentration vs literature (healthy values)
    Panel B: Renal clearance (Ae/AUC) vs literature

CORRECTIONS FROM PREVIOUS VERSION
-----------------------------------
1. reset_ode_debug_flags(): changed
       from: from ode_system import kidney_rhs
       to:   from ode_system_twocompartment import kidney_rhs
   The active ODE is always ode_system_twocompartment.kidney_rhs.

2. reset_compounds=False removed from all run_sim() calls.
   run_sim() does not accept this parameter — it was absorbed by **kwargs
   and silently ignored, giving a false sense of control. Scenario
   configuration is now entirely handled by set_scenario() before run_sim().

3. np.trapz replaced with version-safe _trapz wrapper.
   np.trapz was deprecated in NumPy 2.0 and removed in NumPy 2.4.

4. _ic_from_solution_end_zero_drug() now also zeros C_peri (peripheral
   tissue compartment). With use_tissue=True, C_peri is non-zero at
   steady state and must be reset to 0 at the start of drug scenarios.

5. _ensure_figures_dir() outputs to "./figures" (not "PBPK_Github/figures")
   to avoid hardcoded subdirectory that may not exist.

LITERATURE VALUES (healthy subjects)
--------------------------------------
Concentrations [mg/L]:
    IS:  0.50  (range 0.30–0.70)
    HA:  3.00  (range 1.00–5.00)
    pCS: 1.84  (range 1.20–2.50)

Renal clearances [L/h]:
    IS:  3.80  (range 3.44–4.16)
    HA:  27.8  (range 23.3–32.3)
    pCS: 1.60  (range 0.98–2.22)

Sources:
    IS/pCS: Deguchi 2004, Kidney Int 65:162 (kinetics)
    HA:     Deguchi 2004 (OAT1-selective transport)
    Clearances: literature healthy reference values (verify with your target)
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — required for unattended/server runs
import matplotlib.pyplot as plt

from state_vector import IDX
from simulation import run_sim
from scenario_manager import set_scenario

# ── Version-safe trapezoidal integration ──────────────────────────────────────
# np.trapz removed in NumPy 2.4 → use this wrapper everywhere
try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0


# ── Blood-to-plasma ratios for literature comparison ─────────────────────────
# The ODE state variables C_tox*_cen are treated as blood concentrations.
# Literature PBUT concentrations and renal clearances are usually plasma-based.
# Therefore:
#     C_plasma = C_blood / BP
#     CL_R_plasma = ΔAe / ∫C_plasma dt
TOXIN_BP = {
    "IS":  0.70,
    "HA":  0.84,
    "pCS": 0.55,
}


def _to_plasma_conc(C_blood: np.ndarray, toxin_label: str) -> np.ndarray:
    """Convert model blood concentration [mg/L] to plasma concentration [mg/L]."""
    bp = TOXIN_BP[toxin_label]
    return np.asarray(C_blood, dtype=float) / bp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — ODE debug flag reset
# ══════════════════════════════════════════════════════════════════════════════

def reset_ode_debug_flags() -> None:
    """
    Reset all per-run debug flags on kidney_rhs.

    Must be called before each simulation so that debug prints fire
    correctly for each run and do not carry over from the previous one.

    """
    try:
        # CORRECTED: was 'from ode_system import kidney_rhs'
        from ode_system_twocompartment import kidney_rhs

        kidney_rhs._printed_debug    = False
        kidney_rhs._apical_guarded   = False
        kidney_rhs._prox_debug_fired = False   # suppress proximal spam

        # Clear all per-timestep flags from the previous run
        for attr in list(dir(kidney_rhs)):
            if (
                attr.startswith("_printed_cin_")
                or attr.startswith("_checked_")
                or attr.startswith("_warned_zero_")
                or attr.startswith("_p1_lumen_")
                or attr in ("_flow_debug_done", "_dbg_last_bucket",
                            "_printed_peak_debug", "_tissue_dist")
            ):
                try:
                    delattr(kidney_rhs, attr)
                except AttributeError:
                    pass

    except ImportError as e:
        print(f"[WARN] Could not reset ODE debug flags: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Clearance calculation
# ══════════════════════════════════════════════════════════════════════════════

def clearance_ae_over_auc_window(
    sol: Any,
    c_idx: int,
    ae_idx: int,
    window: Tuple[float, float],
    bp: float = 1.0,
) -> float:
    """
    Compute renal clearance as CLR = Ae / AUC over a time window [L/h].

    Parameters
    ----------
    sol    : OdeResult from run_sim()
    c_idx  : row index in sol.y for the toxin's central concentration
    ae_idx : row index in sol.y for the cumulative excreted amount
    window : (t_start, t_end) [h]

    Returns
    -------
    float : CLR [L/h], or np.nan if inputs are invalid or AUC ≈ 0.

    Method
    ------
    CLR = ΔAe [mg] / AUC_plasma [mg/L·h]  →  [L/h]

    If bp is provided, the raw model concentration is converted as:
        C_plasma = C_blood / bp

    np.maximum.accumulate guards against numerical noise in Ae.
    _trapz is used for AUC (version-safe wrapper for np.trapezoid/trapz).
    """
    if sol is None:
        return np.nan

    t0, t1 = window
    if t1 <= t0:
        return np.nan

    t0c = max(float(t0), float(sol.t[0]))
    t1c = min(float(t1), float(sol.t[-1]))
    if t1c <= t0c:
        return np.nan

    i0 = int(np.searchsorted(sol.t, t0c, side="left"))
    i1 = int(np.searchsorted(sol.t, t1c, side="right"))
    if i1 - i0 < 2:
        return np.nan

    t_w  = np.asarray(sol.t[i0:i1],          dtype=float)
    C_w  = np.clip(np.asarray(sol.y[c_idx][i0:i1],  dtype=float), 0.0, np.inf)
    Ae_w = np.maximum.accumulate(np.asarray(sol.y[ae_idx][i0:i1], dtype=float))

    dAe = max(0.0, float(Ae_w[-1] - Ae_w[0]))
    C_plasma_w = C_w / max(float(bp), 1e-12)
    auc = float(_trapz(C_plasma_w, t_w))

    return dAe / auc if auc > 1e-12 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_figures_dir() -> str:
    """
    Create and return the output results directory.

    Uses the shared top-level results/ folder, not a local "figures/"
    subfolder. NOTE: a previous version of this function had the docstring
    claiming output goes to "./figures" while the code actually used the
    doubled, inconsistent path "figures/figures" — fixed here to point
    directly at the package's shared results/ folder, consistent with
    every other script in this codebase.
    """
    outdir = os.path.join(_THIS_DIR, "..", "results")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _dedupe_legend(
    handles: list,
    labels: list,
) -> Tuple[list, list]:
    """Deduplicate legend entries while preserving insertion order."""
    uniq: Dict[str, Any] = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    return list(uniq.values()), list(uniq.keys())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Generic figure builders
# ══════════════════════════════════════════════════════════════════════════════

def create_figure_timecourses_generic(
    solutions:       Mapping[str, Any],
    scenario_names:  List[str],
    panels:          List[Tuple[str, str]],
    colors:          Dict[str, str],
    line_styles:     Dict[str, str],
    legend_fontsize: int = 15,
    legend_bbox:     Tuple[float, float] = (0.5, 1.02),
) -> plt.Figure:
    """
    Build a 1×3 time-course figure (fig_time).

    Each panel shows one toxin's plasma concentration over 12 h for all
    three scenarios. A single shared legend is placed above all panels.

    Parameters
    ----------
    solutions      : {scenario_name: OdeResult}
    scenario_names : ordered list of scenario name strings
    panels         : [(toxin_label, state_name), ...] — one per panel
    colors         : {scenario_name: hex color}
    line_styles    : {scenario_name: linestyle string}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=False, sharey=False)
    legend_handles = None
    legend_labels  = None

    for ax, (tox_label, state_name) in zip(axes, panels):
        s_idx = IDX.get(state_name)
        if s_idx is None:
            ax.set_axis_off()
            continue

        for scen in scenario_names:
            if scen not in solutions:
                raise KeyError(
                    f"Scenario '{scen}' missing from solutions dict. "
                    f"Available: {list(solutions.keys())}"
                )
            sol = solutions[scen]
            ax.plot(
                sol.t, _to_plasma_conc(sol.y[s_idx], tox_label),
                label=scen,
                color=colors.get(scen, "#000000"),
                linestyle=line_styles.get(scen, "-"),
                linewidth=2.8,
            )

        ax.set_xlabel("Time (h)",               fontsize=16)
        ax.set_ylabel("Concentration (mg/L)",   fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.02, 0.95, tox_label,
                transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        leg = ax.get_legend()
        if leg:
            leg.remove()

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=3.0,
            handletextpad=0.8,
            columnspacing=2.0,
            bbox_to_anchor=legend_bbox,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def create_figure_bars_generic(
    solutions:       Mapping[str, Any],
    window:          Tuple[float, float],
    scenario_names:  List[str],
    colors:          Dict[str, str],
    toxins:          List[str],
    toxin_states:    List[str],
    ae_states:       List[str],
    lit_conc:        Dict[str, Dict[str, Any]],
    lit_clear:       Dict[str, Dict[str, Any]],
    legend_fontsize: int = 15,
    legend_bbox:     Tuple[float, float] = (0.5, -0.07),
) -> plt.Figure:
    """
    Build the 1×2 bar figure (fig_bars).

    Panel A: final plasma concentration [mg/L] vs literature
    Panel B: renal clearance CLR=Ae/AUC [L/h] vs literature

    Literature values are shown as error bars (mean ± range) at x_scenario_0
    position so they do not overlap with the bars.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    x       = np.arange(len(toxins))
    width   = 0.22
    offsets = [-width, 0.0, +width]

    # ── Panel A: final concentrations ─────────────────────────────────────────
    finals: Dict[str, List[float]] = {s: [] for s in scenario_names}
    for state in toxin_states:
        s_idx = IDX.get(state)
        for scen in scenario_names:
            sol = solutions.get(scen)
            finals[scen].append(
                float(_to_plasma_conc(np.array([sol.y[s_idx, -1]]), toxins[len(finals[scen])])[0])
                if (sol is not None and s_idx is not None)
                else np.nan
            )

    for i, scen in enumerate(scenario_names):
        axL.bar(
            x + offsets[i], finals[scen], width,
            label=scen,
            color=colors.get(scen, "#777777"),
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )

    lit_vals = [lit_conc[t]["value"]               for t in toxins]
    lit_lo   = [lit_conc[t]["value"] - lit_conc[t]["range"][0] for t in toxins]
    lit_hi   = [lit_conc[t]["range"][1] - lit_conc[t]["value"] for t in toxins]
    axL.errorbar(
        x, lit_vals, yerr=[lit_lo, lit_hi],
        fmt="D", color="black", markerfacecolor="black",
        markersize=6, capsize=5, capthick=2, elinewidth=2,
        label="Literature Healthy", zorder=5,
    )

    axL.set_xlabel("Toxins",                              fontsize=16)
    axL.set_ylabel("Final plasma concentration (mg/L)",   fontsize=16)
    axL.set_xticks(x)
    axL.set_xticklabels(toxins)
    axL.tick_params(axis="both", labelsize=12)
    axL.grid(True, alpha=0.3, axis="y")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.text(0.02, 0.95, "A", transform=axL.transAxes,
             fontsize=16, fontweight="bold", va="top")

    # ── Panel B: renal clearances ─────────────────────────────────────────────
    clears: Dict[str, List[float]] = {s: [] for s in scenario_names}
    for tox_label, c_state, ae_state in zip(toxins, toxin_states, ae_states):
        c_idx  = IDX.get(c_state)
        ae_idx = IDX.get(ae_state)
        bp     = TOXIN_BP[tox_label]
        for scen in scenario_names:
            sol = solutions.get(scen)
            clears[scen].append(
                clearance_ae_over_auc_window(sol, c_idx, ae_idx, window, bp=bp)
                if (sol is not None and c_idx is not None and ae_idx is not None)
                else np.nan
            )

    for i, scen in enumerate(scenario_names):
        axR.bar(
            x + offsets[i], clears[scen], width,
            label=scen,
            color=colors.get(scen, "#777777"),
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )

    litc_vals = [lit_clear[t]["value"]                 for t in toxins]
    litc_lo   = [lit_clear[t]["value"] - lit_clear[t]["range"][0] for t in toxins]
    litc_hi   = [lit_clear[t]["range"][1] - lit_clear[t]["value"] for t in toxins]
    axR.errorbar(
        x, litc_vals, yerr=[litc_lo, litc_hi],
        fmt="D", color="black", markerfacecolor="black",
        markersize=6, capsize=5, capthick=2, elinewidth=2,
        label="Literature Healthy", zorder=5,
    )

    axR.set_xlabel("Toxins",          fontsize=16)
    axR.set_ylabel("Clearance (L/h)", fontsize=16)
    axR.set_xticks(x)
    axR.set_xticklabels(toxins)
    axR.tick_params(axis="both", labelsize=12)
    axR.grid(True, alpha=0.3, axis="y")
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.text(0.02, 0.95, "B", transform=axR.transAxes,
             fontsize=16, fontweight="bold", va="top")

    # ── Shared legend below both panels ───────────────────────────────────────
    for ax in (axL, axR):
        leg = ax.get_legend()
        if leg:
            leg.remove()

    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    handles, labels = _dedupe_legend(h1 + h2, l1 + l2)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=legend_bbox,
        columnspacing=2.3,
        handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Healthy-specific figure wrappers
# ══════════════════════════════════════════════════════════════════════════════

def create_figure_healthy_timecourses(solutions: Mapping[str, Any]) -> plt.Figure:
    """Build fig_time for the healthy three-scenario comparison."""
    scenario_names = [
        "Healthy No Furosemide",
        "Healthy With Furosemide (No Interaction)",
        "Healthy With Furosemide",
    ]
    panels = [
        ("IS",  "C_tox1_cen"),
        ("HA",  "C_tox2_cen"),
        ("pCS", "C_tox3_cen"),
    ]
    colors = {
        scenario_names[0]: "#4d4d4d",   # mid-grey
        scenario_names[1]: "#9a9a9a",   # light-grey
        scenario_names[2]: "#1a1a1a",   # near-black
    }
    line_styles = {
        scenario_names[0]: "-",
        scenario_names[1]: ":",
        scenario_names[2]: "--",
    }
    return create_figure_timecourses_generic(
        solutions=solutions,
        scenario_names=scenario_names,
        panels=panels,
        colors=colors,
        line_styles=line_styles,
        legend_fontsize=15,
        legend_bbox=(0.5, 1.02),
    )


def create_figure_healthy_bars(
    solutions: Mapping[str, Any],
    window: Tuple[float, float] = (0.0, 12.0),
) -> plt.Figure:
    """Build fig_bars for the healthy three-scenario comparison."""
    scenario_names = [
        "Healthy No Furosemide",
        "Healthy With Furosemide (No Interaction)",
        "Healthy With Furosemide",
    ]
    colors = {
        scenario_names[0]: "#4d4d4d",
        scenario_names[1]: "#9a9a9a",
        scenario_names[2]: "#1a1a1a",
    }
    toxins       = ["IS",         "HA",          "pCS"]
    toxin_states = ["C_tox1_cen", "C_tox2_cen",  "C_tox3_cen"]
    ae_states    = ["Ae_t1",      "Ae_t2",        "Ae_t3"]

    # ── Literature values (healthy subjects) ──────────────────────────────────
    # Concentrations [mg/L] — total plasma, healthy reference
    lit_conc = {
        "IS":  {"value": 0.50, "range": (0.30, 0.70)},
        "HA":  {"value": 3.00, "range": (1.00, 5.00)},
        "pCS": {"value": 1.84, "range": (1.20, 2.50)},
    }
    # Renal clearances [L/h] — healthy reference
    # IS:  Deguchi 2004 + healthy volunteer data
    # HA:  high CLR due to OAT1-selective transport (no OAT3)
    # pCS: lower CLR due to high Km at OAT1 (128 μM)
    lit_clear = {
        "IS":  {"value": 3.80,  "range": (2.44,  5.16)},
        "HA":  {"value": 27.80, "range": (23.30, 32.30)},
        "pCS": {"value": 1.60,  "range": (0.98,  2.22)},
    }

    return create_figure_bars_generic(
        solutions=solutions,
        window=window,
        scenario_names=scenario_names,
        colors=colors,
        toxins=toxins,
        toxin_states=toxin_states,
        ae_states=ae_states,
        lit_conc=lit_conc,
        lit_clear=lit_clear,
        legend_fontsize=15,
        legend_bbox=(0.5, -0.07),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Initial condition builders
# ══════════════════════════════════════════════════════════════════════════════

def _ic_from_solution_end(sol: Any) -> Dict[str, float]:
    """
    Extract the final state vector as an initial conditions dict.

    Returns {state_name: value} for every state in IDX.
    Used to start the 12 h scenarios from the 1500 h SS endpoint.
    """
    y_end = sol.y[:, -1]
    return {name: float(y_end[i]) for name, i in IDX.items()}


def _ic_from_solution_end_zero_drug(sol: Any) -> Dict[str, float]:
    """
    Extract the final state vector but zero all drug-related compartments.

    WHY:
    The drug scenarios (With Furosemide) start from the same toxin steady
    state as the No Drug scenario, but with zero drug present in all
    compartments. This isolates the drug effect from any pre-existing
    drug concentration.

    Zeroed states:
    - C_ven             : central plasma drug concentration
    - C_peri            : peripheral tissue drug concentration (ADDED)
                          previously missing — with use_tissue=True,
                          C_peri is non-zero at SS and must be reset
    - C_{seg}_V/C/T     : drug in all kidney vascular/cellular/lumen segments
    - C_bladder         : drug in bladder
    - Ae_drg            : cumulative urinary drug excretion

    Toxin concentrations (C_tox*) are kept at their SS values.
    """
    import constants as K

    ic = _ic_from_solution_end(sol)

    # Central plasma drug
    for key in ("C_ven", "C_peri", "C_bladder", "Ae_drg"):
        if key in ic:
            ic[key] = 0.0

    # Drug in all kidney tubular segments (vascular, cellular, lumen)
    for seg in (K.PROX + K.NONPROX + K.CDS):
        for suff in ("V", "C", "T"):
            key = f"C_{seg}_{suff}"
            if key in ic:
                ic[key] = 0.0

    return ic


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Quantitative output printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_pct_changes(
    solutions: Dict[str, Any],
    window: Tuple[float, float],
) -> None:
    """
    Print exact % changes in toxin concentration and clearance
    between the three scenarios.

    This is the quantitative output for your results section.
    """
    toxins       = ["IS",         "HA",          "pCS"]
    toxin_states = ["C_tox1_cen", "C_tox2_cen",  "C_tox3_cen"]
    ae_states    = ["Ae_t1",      "Ae_t2",        "Ae_t3"]

    base_name   = "Healthy No Furosemide"
    no_int_name = "Healthy With Furosemide (No Interaction)"
    int_name    = "Healthy With Furosemide"

    def _pct(new: float, base: float) -> float:
        return np.nan if abs(base) < 1e-12 else 100.0 * (new - base) / base

    print("\n" + "=" * 65)
    print("  QUANTITATIVE RESULTS — Healthy 3-scenario comparison")
    print("=" * 65)

    for tox, c_state, ae_state in zip(toxins, toxin_states, ae_states):
        c_idx  = IDX[c_state]
        ae_idx = IDX[ae_state]

        bp = TOXIN_BP[tox]

        c_base = float(_to_plasma_conc(np.array([solutions[base_name].y[c_idx, -1]]), tox)[0])
        c_noi  = float(_to_plasma_conc(np.array([solutions[no_int_name].y[c_idx, -1]]), tox)[0])
        c_int  = float(_to_plasma_conc(np.array([solutions[int_name].y[c_idx, -1]]), tox)[0])

        cl_base = clearance_ae_over_auc_window(solutions[base_name],   c_idx, ae_idx, window, bp=bp)
        cl_noi  = clearance_ae_over_auc_window(solutions[no_int_name], c_idx, ae_idx, window, bp=bp)
        cl_int  = clearance_ae_over_auc_window(solutions[int_name],    c_idx, ae_idx, window, bp=bp)

        print(f"\n  {tox}:")
        print(f"    Final plasma C_ss [mg/L]:")
        print(f"      No Drug       = {c_base:.5f}")
        print(f"      Drug, No Int  = {c_noi:.5f}  ({_pct(c_noi, c_base):+.2f}%)")
        print(f"      Drug + Inhib  = {c_int:.5f}  ({_pct(c_int, c_base):+.2f}%)")
        print(f"    Plasma CL_R [L/h]  (Ae/AUC_plasma over {window[0]:.0f}–{window[1]:.0f} h):")
        print(f"      No Drug       = {cl_base:.4f}")
        print(f"      Drug, No Int  = {cl_noi:.4f}  ({_pct(cl_noi, cl_base):+.2f}%)")
        print(f"      Drug + Inhib  = {cl_int:.4f}  ({_pct(cl_int, cl_base):+.2f}%)")

    print("\n" + "=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main simulation driver
# ══════════════════════════════════════════════════════════════════════════════

def run_healthy_and_plot(
    window:  Tuple[float, float] = (0.0, 12.0),
    t_ss:    float = 1500.0,
    t_sim:   float = 12.0,
    dt_ss:   float = 0.5,
    dt_sim:  float = 0.02,
) -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
    """
    Full simulation pipeline: SS warm-up → 3 scenarios → figures.

    Parameters
    ----------
    window  : (t0, t1) [h] for clearance calculation in fig_bars
    t_ss    : steady-state warm-up duration [h] (default 1500 h)
    t_sim   : main simulation duration [h] (default 12 h)
    dt_ss   : time step for SS warm-up [h] (coarse, 0.5 h is fine)
    dt_sim  : time step for main runs [h] (fine, 0.02 h for smooth curves)

    Returns
    -------
    (fig_time, fig_bars, solutions_dict)
    """
    try:
        import plotting
        if hasattr(plotting, "setup_professional_style"):
            plotting.setup_professional_style()
    except Exception:
        pass

    print("\n=== Healthy 3-scenario comparison ===")
    print(f"    SS warm-up: {t_ss:.0f} h  |  Main runs: {t_sim:.0f} h")

    # ── Step 1: Steady-state warm-up (no drug) ────────────────────────────────
    print(f"\n▶ Steady-state warm-up ({t_ss:.0f} h, no drug) ...")
    set_scenario("H_NO_DRUG")
    reset_ode_debug_flags()
    sol_ss = run_sim(t_end_h=t_ss, mode="healthy", dt=dt_ss)

    # ── Step 2: Build initial conditions ─────────────────────────────────────
    ic_no_drug = _ic_from_solution_end(sol_ss)
    ic_drug    = _ic_from_solution_end_zero_drug(sol_ss)

    # ── Step 3: Scenario 1 — No Furosemide ───────────────────────────────────
    print("\n▶ Scenario 1: Healthy No Furosemide (12 h from SS) ...")
    set_scenario("H_NO_DRUG")
    reset_ode_debug_flags()
    sol_no = run_sim(
        t_end_h=t_sim,
        mode="healthy",
        dt=dt_sim,
        custom_initial_conds=ic_no_drug,
    )

    # ── Step 4: Scenario 2 — With Drug, No Interaction ───────────────────────
    print("\n▶ Scenario 2: Healthy With Furosemide (No Interaction) (12 h from SS) ...")
    set_scenario("H_WITH_DRUG_NO_INT")
    reset_ode_debug_flags()
    sol_no_int = run_sim(
        t_end_h=t_sim,
        mode="healthy",
        dt=dt_sim,
        custom_initial_conds=ic_drug,
    )

    # ── Step 5: Scenario 3 — With Drug + Inhibition ───────────────────────────
    print("\n▶ Scenario 3: Healthy With Furosemide (Interaction ON) (12 h from SS) ...")
    set_scenario("H_WITH_DRUG")
    reset_ode_debug_flags()
    sol_drug = run_sim(
        t_end_h=t_sim,
        mode="healthy",
        dt=dt_sim,
        custom_initial_conds=ic_drug,
    )

    # ── Step 6: Assemble results ──────────────────────────────────────────────
    solutions = {
        "Healthy No Furosemide":                    sol_no,
        "Healthy With Furosemide (No Interaction)": sol_no_int,
        "Healthy With Furosemide":                  sol_drug,
    }

    # ── Step 7: Print quantitative results ───────────────────────────────────
    _print_pct_changes(solutions, window)

    # ── Step 8: Build figures ─────────────────────────────────────────────────
    fig_time = create_figure_healthy_timecourses(solutions)
    fig_bars = create_figure_healthy_bars(solutions, window=window)

    return fig_time, fig_bars, solutions


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Script entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig_time, fig_bars, _solutions = run_healthy_and_plot(
        window=(0.0, 12.0),
        t_ss=1500.0,
        t_sim=2.0,
        dt_ss=0.5,
        dt_sim=0.02,
    )

    outdir = _ensure_figures_dir()
    base1 = os.path.join(outdir, "healthy_timecourses_3scenarios_greyscale")
    base2 = os.path.join(outdir, "healthy_bars_3scenarios_greyscale")

    for ext in ("png", "pdf"):
        fig_time.savefig(f"{base1}.{ext}", dpi=300, bbox_inches="tight")
        fig_bars.savefig(f"{base2}.{ext}", dpi=300, bbox_inches="tight")
        print(f"  Saved {base1}.{ext}")
        print(f"  Saved {base2}.{ext}")

    plt.close(fig_time)
    plt.close(fig_bars)
# plot_healthy_lit_comparison_3S.py
# # -*- coding: utf-8 -*-
# """
# plot_healthy_lit_comparison_3S.py
# =================================
# Healthy scenario comparison — three scenarios:
#
#     1) Healthy No Furosemide
#     2) Healthy With Furosemide (No Interaction)
#     3) Healthy With Furosemide
#
# SIMULATION PROTOCOL
# -------------------
# Step 1 — Steady-state warm-up (1500 h, no drug):
#     Runs the model without furosemide until IS, HA, pCS reach healthy
#     steady state. 1500 h (62.5 days) is more than sufficient for all
#     three toxins to equilibrate (slowest is pCS, t½_ss ≈ several days).
#
# Step 2 — Extract two sets of initial conditions from the SS endpoint:
#     ic_ss_no   : full steady state — used for "No Furosemide" scenario
#     ic_ss_drug : same as above but with all drug compartments zeroed —
#                  used for both drug scenarios (clean slate for drug)
#
# Step 3 — Three 12 h simulations from the SS initial conditions.
#
# WHY THIS PROTOCOL MATTERS
# --------------------------
# Starting all three scenarios from the same steady-state initial conditions
# ensures that any difference between the curves is caused ONLY by the
# presence or absence of furosemide — not by different starting toxin levels.
#
# WHAT THE FIGURES SHOW
# ----------------------
# fig_time (1×3 panels):
#     Plasma time-courses of IS, HA, pCS over 12 h.
#     All three scenarios start at the same steady-state concentration.
#     "With Furosemide" should show a transient rise during infusion
#     (reduced OAT1 secretion → toxins accumulate) then return to SS.
#
# fig_bars (1×2 panels):
#     Panel A: Final plasma concentration vs literature (healthy values)
#     Panel B: Renal clearance (Ae/AUC) vs literature
#
# CORRECTIONS FROM PREVIOUS VERSION
# -----------------------------------
# 1. reset_ode_debug_flags(): changed
#        from: from ode_system import kidney_rhs
#        to:   from ode_system_twocompartment import kidney_rhs
#    The active ODE is always ode_system_twocompartment.kidney_rhs.
#
# 2. reset_compounds=False removed from all run_sim() calls.
#    run_sim() does not accept this parameter — it was absorbed by **kwargs
#    and silently ignored, giving a false sense of control. Scenario
#    configuration is now entirely handled by set_scenario() before run_sim().
#
# 3. np.trapz replaced with version-safe _trapz wrapper.
#    np.trapz was deprecated in NumPy 2.0 and removed in NumPy 2.4.
#
# 4. _ic_from_solution_end_zero_drug() now also zeros C_peri (peripheral
#    tissue compartment). With use_tissue=True, C_peri is non-zero at
#    steady state and must be reset to 0 at the start of drug scenarios.
#
# 5. _ensure_figures_dir() outputs to "./figures" (not "PBPK_Github/figures")
#    to avoid hardcoded subdirectory that may not exist.
#
# LITERATURE VALUES (healthy subjects)
# --------------------------------------
# Concentrations [mg/L]:
#     IS:  0.50  (range 0.30–0.70)
#     HA:  3.00  (range 1.00–5.00)
#     pCS: 1.84  (range 1.20–2.50)
#
# Renal clearances [L/h]:
#     IS:  3.80  (range 3.44–4.16)
#     HA:  27.8  (range 23.3–32.3)
#     pCS: 1.60  (range 0.98–2.22)
#
# Sources:
#     IS/pCS: Deguchi 2004, Kidney Int 65:162 (kinetics)
#     HA:     Deguchi 2004 (OAT1-selective transport)
#     Clearances: literature healthy reference values (verify with your target)
# """
#
# from __future__ import annotations
#
# import os
# from typing import Any, Dict, List, Mapping, Tuple
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# from state_vector import IDX
# from simulation import run_sim
# from scenario_manager import set_scenario
#
# # ── Version-safe trapezoidal integration ──────────────────────────────────────
# # np.trapz removed in NumPy 2.4 → use this wrapper everywhere
# try:
#     _trapz = np.trapezoid   # NumPy >= 2.0
# except AttributeError:
#     _trapz = np.trapz       # NumPy < 2.0
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 0 — ODE debug flag reset
# # ══════════════════════════════════════════════════════════════════════════════
#
# def reset_ode_debug_flags() -> None:
#     """
#     Reset all per-run debug flags on kidney_rhs.
#
#     Must be called before each simulation so that debug prints fire
#     correctly for each run and do not carry over from the previous one.
#
#     CORRECTION: imports from ode_system_twocompartment (not ode_system).
#     """
#     try:
#         # CORRECTED: was 'from ode_system import kidney_rhs'
#         from ode_system_twocompartment import kidney_rhs
#
#         kidney_rhs._printed_debug    = False
#         kidney_rhs._apical_guarded   = False
#         kidney_rhs._prox_debug_fired = False   # suppress proximal spam
#
#         # Clear all per-timestep flags from the previous run
#         for attr in list(dir(kidney_rhs)):
#             if (
#                 attr.startswith("_printed_cin_")
#                 or attr.startswith("_checked_")
#                 or attr.startswith("_warned_zero_")
#                 or attr.startswith("_p1_lumen_")
#                 or attr in ("_flow_debug_done", "_dbg_last_bucket",
#                             "_printed_peak_debug", "_tissue_dist")
#             ):
#                 try:
#                     delattr(kidney_rhs, attr)
#                 except AttributeError:
#                     pass
#
#     except ImportError as e:
#         print(f"[WARN] Could not reset ODE debug flags: {e}")
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 1 — Clearance calculation
# # ══════════════════════════════════════════════════════════════════════════════
#
# def clearance_ae_over_auc_window(
#     sol: Any,
#     c_idx: int,
#     ae_idx: int,
#     window: Tuple[float, float],
# ) -> float:
#     """
#     Compute renal clearance as CLR = Ae / AUC over a time window [L/h].
#
#     Parameters
#     ----------
#     sol    : OdeResult from run_sim()
#     c_idx  : row index in sol.y for the toxin's central concentration
#     ae_idx : row index in sol.y for the cumulative excreted amount
#     window : (t_start, t_end) [h]
#
#     Returns
#     -------
#     float : CLR [L/h], or np.nan if inputs are invalid or AUC ≈ 0.
#
#     Method
#     ------
#     CLR = ΔAe [mg] / AUC [mg/L·h]  →  [L/h]
#
#     np.maximum.accumulate guards against numerical noise in Ae.
#     _trapz is used for AUC (version-safe wrapper for np.trapezoid/trapz).
#     """
#     if sol is None:
#         return np.nan
#
#     t0, t1 = window
#     if t1 <= t0:
#         return np.nan
#
#     t0c = max(float(t0), float(sol.t[0]))
#     t1c = min(float(t1), float(sol.t[-1]))
#     if t1c <= t0c:
#         return np.nan
#
#     i0 = int(np.searchsorted(sol.t, t0c, side="left"))
#     i1 = int(np.searchsorted(sol.t, t1c, side="right"))
#     if i1 - i0 < 2:
#         return np.nan
#
#     t_w  = np.asarray(sol.t[i0:i1],          dtype=float)
#     C_w  = np.clip(np.asarray(sol.y[c_idx][i0:i1],  dtype=float), 0.0, np.inf)
#     Ae_w = np.maximum.accumulate(np.asarray(sol.y[ae_idx][i0:i1], dtype=float))
#
#     dAe = max(0.0, float(Ae_w[-1] - Ae_w[0]))
#     auc = float(_trapz(C_w, t_w))
#
#     return dAe / auc if auc > 1e-12 else np.nan
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 2 — Plotting helpers
# # ══════════════════════════════════════════════════════════════════════════════
#
# def _ensure_figures_dir() -> str:
#     """
#     Create and return the output figures directory.
#
#     Uses './figures' relative to the script location.
#     Previously used 'PBPK_Github/figures' (hardcoded subdirectory
#     that may not exist on all machines).
#     """
#     here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
#     outdir = os.path.join(here, "figures/figures")
#     os.makedirs(outdir, exist_ok=True)
#     return outdir
#
#
# def _dedupe_legend(
#     handles: list,
#     labels: list,
# ) -> Tuple[list, list]:
#     """Deduplicate legend entries while preserving insertion order."""
#     uniq: Dict[str, Any] = {}
#     for h, l in zip(handles, labels):
#         if l not in uniq:
#             uniq[l] = h
#     return list(uniq.values()), list(uniq.keys())
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 3 — Generic figure builders
# # ══════════════════════════════════════════════════════════════════════════════
#
# def create_figure_timecourses_generic(
#     solutions:       Mapping[str, Any],
#     scenario_names:  List[str],
#     panels:          List[Tuple[str, str]],
#     colors:          Dict[str, str],
#     line_styles:     Dict[str, str],
#     legend_fontsize: int = 15,
#     legend_bbox:     Tuple[float, float] = (0.5, 1.02),
# ) -> plt.Figure:
#     """
#     Build a 1×3 time-course figure (fig_time).
#
#     Each panel shows one toxin's plasma concentration over 12 h for all
#     three scenarios. A single shared legend is placed above all panels.
#
#     Parameters
#     ----------
#     solutions      : {scenario_name: OdeResult}
#     scenario_names : ordered list of scenario name strings
#     panels         : [(toxin_label, state_name), ...] — one per panel
#     colors         : {scenario_name: hex color}
#     line_styles    : {scenario_name: linestyle string}
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=False, sharey=False)
#     legend_handles = None
#     legend_labels  = None
#
#     for ax, (tox_label, state_name) in zip(axes, panels):
#         s_idx = IDX.get(state_name)
#         if s_idx is None:
#             ax.set_axis_off()
#             continue
#
#         for scen in scenario_names:
#             if scen not in solutions:
#                 raise KeyError(
#                     f"Scenario '{scen}' missing from solutions dict. "
#                     f"Available: {list(solutions.keys())}"
#                 )
#             sol = solutions[scen]
#             ax.plot(
#                 sol.t, sol.y[s_idx],
#                 label=scen,
#                 color=colors.get(scen, "#000000"),
#                 linestyle=line_styles.get(scen, "-"),
#                 linewidth=2.8,
#             )
#
#         ax.set_xlabel("Time (h)",               fontsize=16)
#         ax.set_ylabel("Concentration (mg/L)",   fontsize=16)
#         ax.tick_params(axis="both", labelsize=12)
#         ax.grid(True, alpha=0.3)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.text(0.02, 0.95, tox_label,
#                 transform=ax.transAxes,
#                 fontsize=16, fontweight="bold", va="top", ha="left")
#
#         if legend_handles is None:
#             legend_handles, legend_labels = ax.get_legend_handles_labels()
#
#         leg = ax.get_legend()
#         if leg:
#             leg.remove()
#
#     if legend_handles:
#         fig.legend(
#             legend_handles, legend_labels,
#             loc="upper center",
#             ncol=3,
#             frameon=False,
#             fontsize=legend_fontsize,
#             handlelength=3.0,
#             handletextpad=0.8,
#             columnspacing=2.0,
#             bbox_to_anchor=legend_bbox,
#         )
#
#     fig.tight_layout(rect=[0, 0, 1, 0.90])
#     return fig
#
#
# def create_figure_bars_generic(
#     solutions:       Mapping[str, Any],
#     window:          Tuple[float, float],
#     scenario_names:  List[str],
#     colors:          Dict[str, str],
#     toxins:          List[str],
#     toxin_states:    List[str],
#     ae_states:       List[str],
#     lit_conc:        Dict[str, Dict[str, Any]],
#     lit_clear:       Dict[str, Dict[str, Any]],
#     legend_fontsize: int = 15,
#     legend_bbox:     Tuple[float, float] = (0.5, -0.07),
# ) -> plt.Figure:
#     """
#     Build the 1×2 bar figure (fig_bars).
#
#     Panel A: final plasma concentration [mg/L] vs literature
#     Panel B: renal clearance CLR=Ae/AUC [L/h] vs literature
#
#     Literature values are shown as error bars (mean ± range) at x_scenario_0
#     position so they do not overlap with the bars.
#     """
#     fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
#
#     x       = np.arange(len(toxins))
#     width   = 0.22
#     offsets = [-width, 0.0, +width]
#
#     # ── Panel A: final concentrations ─────────────────────────────────────────
#     finals: Dict[str, List[float]] = {s: [] for s in scenario_names}
#     for state in toxin_states:
#         s_idx = IDX.get(state)
#         for scen in scenario_names:
#             sol = solutions.get(scen)
#             finals[scen].append(
#                 float(sol.y[s_idx, -1])
#                 if (sol is not None and s_idx is not None)
#                 else np.nan
#             )
#
#     for i, scen in enumerate(scenario_names):
#         axL.bar(
#             x + offsets[i], finals[scen], width,
#             label=scen,
#             color=colors.get(scen, "#777777"),
#             alpha=0.85, edgecolor="black", linewidth=0.5,
#         )
#
#     lit_vals = [lit_conc[t]["value"]               for t in toxins]
#     lit_lo   = [lit_conc[t]["value"] - lit_conc[t]["range"][0] for t in toxins]
#     lit_hi   = [lit_conc[t]["range"][1] - lit_conc[t]["value"] for t in toxins]
#     axL.errorbar(
#         x, lit_vals, yerr=[lit_lo, lit_hi],
#         fmt="D", color="black", markerfacecolor="black",
#         markersize=6, capsize=5, capthick=2, elinewidth=2,
#         label="Literature Healthy", zorder=5,
#     )
#
#     axL.set_xlabel("Toxins",                              fontsize=16)
#     axL.set_ylabel("Final plasma concentration (mg/L)",   fontsize=16)
#     axL.set_xticks(x)
#     axL.set_xticklabels(toxins)
#     axL.tick_params(axis="both", labelsize=12)
#     axL.grid(True, alpha=0.3, axis="y")
#     axL.spines["top"].set_visible(False)
#     axL.spines["right"].set_visible(False)
#     axL.text(0.02, 0.95, "A", transform=axL.transAxes,
#              fontsize=16, fontweight="bold", va="top")
#
#     # ── Panel B: renal clearances ─────────────────────────────────────────────
#     clears: Dict[str, List[float]] = {s: [] for s in scenario_names}
#     for c_state, ae_state in zip(toxin_states, ae_states):
#         c_idx  = IDX.get(c_state)
#         ae_idx = IDX.get(ae_state)
#         for scen in scenario_names:
#             sol = solutions.get(scen)
#             clears[scen].append(
#                 clearance_ae_over_auc_window(sol, c_idx, ae_idx, window)
#                 if (sol is not None and c_idx is not None and ae_idx is not None)
#                 else np.nan
#             )
#
#     for i, scen in enumerate(scenario_names):
#         axR.bar(
#             x + offsets[i], clears[scen], width,
#             label=scen,
#             color=colors.get(scen, "#777777"),
#             alpha=0.85, edgecolor="black", linewidth=0.5,
#         )
#
#     litc_vals = [lit_clear[t]["value"]                 for t in toxins]
#     litc_lo   = [lit_clear[t]["value"] - lit_clear[t]["range"][0] for t in toxins]
#     litc_hi   = [lit_clear[t]["range"][1] - lit_clear[t]["value"] for t in toxins]
#     axR.errorbar(
#         x, litc_vals, yerr=[litc_lo, litc_hi],
#         fmt="D", color="black", markerfacecolor="black",
#         markersize=6, capsize=5, capthick=2, elinewidth=2,
#         label="Literature Healthy", zorder=5,
#     )
#
#     axR.set_xlabel("Toxins",          fontsize=16)
#     axR.set_ylabel("Clearance (L/h)", fontsize=16)
#     axR.set_xticks(x)
#     axR.set_xticklabels(toxins)
#     axR.tick_params(axis="both", labelsize=12)
#     axR.grid(True, alpha=0.3, axis="y")
#     axR.spines["top"].set_visible(False)
#     axR.spines["right"].set_visible(False)
#     axR.text(0.02, 0.95, "B", transform=axR.transAxes,
#              fontsize=16, fontweight="bold", va="top")
#
#     # ── Shared legend below both panels ───────────────────────────────────────
#     for ax in (axL, axR):
#         leg = ax.get_legend()
#         if leg:
#             leg.remove()
#
#     h1, l1 = axL.get_legend_handles_labels()
#     h2, l2 = axR.get_legend_handles_labels()
#     handles, labels = _dedupe_legend(h1 + h2, l1 + l2)
#
#     fig.legend(
#         handles, labels,
#         loc="lower center",
#         ncol=4,
#         frameon=False,
#         fontsize=legend_fontsize,
#         bbox_to_anchor=legend_bbox,
#         columnspacing=2.3,
#         handlelength=2.4,
#     )
#     fig.tight_layout(rect=[0, 0, 1, 0.93])
#     return fig
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 4 — Healthy-specific figure wrappers
# # ══════════════════════════════════════════════════════════════════════════════
#
# def create_figure_healthy_timecourses(solutions: Mapping[str, Any]) -> plt.Figure:
#     """Build fig_time for the healthy three-scenario comparison."""
#     scenario_names = [
#         "Healthy No Furosemide",
#         "Healthy With Furosemide (No Interaction)",
#         "Healthy With Furosemide",
#     ]
#     panels = [
#         ("IS",  "C_tox1_cen"),
#         ("HA",  "C_tox2_cen"),
#         ("pCS", "C_tox3_cen"),
#     ]
#     colors = {
#         scenario_names[0]: "#4d4d4d",   # mid-grey
#         scenario_names[1]: "#9a9a9a",   # light-grey
#         scenario_names[2]: "#1a1a1a",   # near-black
#     }
#     line_styles = {
#         scenario_names[0]: "-",
#         scenario_names[1]: ":",
#         scenario_names[2]: "--",
#     }
#     return create_figure_timecourses_generic(
#         solutions=solutions,
#         scenario_names=scenario_names,
#         panels=panels,
#         colors=colors,
#         line_styles=line_styles,
#         legend_fontsize=15,
#         legend_bbox=(0.5, 1.02),
#     )
#
#
# def create_figure_healthy_bars(
#     solutions: Mapping[str, Any],
#     window: Tuple[float, float] = (0.0, 12.0),
# ) -> plt.Figure:
#     """Build fig_bars for the healthy three-scenario comparison."""
#     scenario_names = [
#         "Healthy No Furosemide",
#         "Healthy With Furosemide (No Interaction)",
#         "Healthy With Furosemide",
#     ]
#     colors = {
#         scenario_names[0]: "#4d4d4d",
#         scenario_names[1]: "#9a9a9a",
#         scenario_names[2]: "#1a1a1a",
#     }
#     toxins       = ["IS",         "HA",          "pCS"]
#     toxin_states = ["C_tox1_cen", "C_tox2_cen",  "C_tox3_cen"]
#     ae_states    = ["Ae_t1",      "Ae_t2",        "Ae_t3"]
#
#     # ── Literature values (healthy subjects) ──────────────────────────────────
#     # Concentrations [mg/L] — total plasma, healthy reference
#     lit_conc = {
#         "IS":  {"value": 0.50, "range": (0.30, 0.70)},
#         "HA":  {"value": 3.00, "range": (1.00, 5.00)},
#         "pCS": {"value": 1.84, "range": (1.20, 2.50)},
#     }
#     # Renal clearances [L/h] — healthy reference
#     # IS:  Deguchi 2004 + healthy volunteer data
#     # HA:  high CLR due to OAT1-selective transport (no OAT3)
#     # pCS: lower CLR due to high Km at OAT1 (128 μM)
#     lit_clear = {
#         "IS":  {"value": 3.80,  "range": (3.44,  4.16)},
#         "HA":  {"value": 27.80, "range": (23.30, 32.30)},
#         "pCS": {"value": 1.60,  "range": (0.98,  2.22)},
#     }
#
#     return create_figure_bars_generic(
#         solutions=solutions,
#         window=window,
#         scenario_names=scenario_names,
#         colors=colors,
#         toxins=toxins,
#         toxin_states=toxin_states,
#         ae_states=ae_states,
#         lit_conc=lit_conc,
#         lit_clear=lit_clear,
#         legend_fontsize=15,
#         legend_bbox=(0.5, -0.07),
#     )
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 5 — Initial condition builders
# # ══════════════════════════════════════════════════════════════════════════════
#
# def _ic_from_solution_end(sol: Any) -> Dict[str, float]:
#     """
#     Extract the final state vector as an initial conditions dict.
#
#     Returns {state_name: value} for every state in IDX.
#     Used to start the 12 h scenarios from the 1500 h SS endpoint.
#     """
#     y_end = sol.y[:, -1]
#     return {name: float(y_end[i]) for name, i in IDX.items()}
#
#
# def _ic_from_solution_end_zero_drug(sol: Any) -> Dict[str, float]:
#     """
#     Extract the final state vector but zero all drug-related compartments.
#
#     WHY:
#     The drug scenarios (With Furosemide) start from the same toxin steady
#     state as the No Drug scenario, but with zero drug present in all
#     compartments. This isolates the drug effect from any pre-existing
#     drug concentration.
#
#     Zeroed states:
#     - C_ven             : central plasma drug concentration
#     - C_peri            : peripheral tissue drug concentration (ADDED)
#                           previously missing — with use_tissue=True,
#                           C_peri is non-zero at SS and must be reset
#     - C_{seg}_V/C/T     : drug in all kidney vascular/cellular/lumen segments
#     - C_bladder         : drug in bladder
#     - Ae_drg            : cumulative urinary drug excretion
#
#     Toxin concentrations (C_tox*) are kept at their SS values.
#     """
#     import constants as K
#
#     ic = _ic_from_solution_end(sol)
#
#     # Central plasma drug
#     for key in ("C_ven", "C_peri", "C_bladder", "Ae_drg"):
#         if key in ic:
#             ic[key] = 0.0
#
#     # Drug in all kidney tubular segments (vascular, cellular, lumen)
#     for seg in (K.PROX + K.NONPROX + K.CDS):
#         for suff in ("V", "C", "T"):
#             key = f"C_{seg}_{suff}"
#             if key in ic:
#                 ic[key] = 0.0
#
#     return ic
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 6 — Quantitative output printer
# # ══════════════════════════════════════════════════════════════════════════════
#
# def _print_pct_changes(
#     solutions: Dict[str, Any],
#     window: Tuple[float, float],
# ) -> None:
#     """
#     Print exact % changes in toxin concentration and clearance
#     between the three scenarios.
#
#     This is the quantitative output for your results section.
#     """
#     toxins       = ["IS",         "HA",          "pCS"]
#     toxin_states = ["C_tox1_cen", "C_tox2_cen",  "C_tox3_cen"]
#     ae_states    = ["Ae_t1",      "Ae_t2",        "Ae_t3"]
#
#     base_name   = "Healthy No Furosemide"
#     no_int_name = "Healthy With Furosemide (No Interaction)"
#     int_name    = "Healthy With Furosemide"
#
#     def _pct(new: float, base: float) -> float:
#         return np.nan if abs(base) < 1e-12 else 100.0 * (new - base) / base
#
#     print("\n" + "=" * 65)
#     print("  QUANTITATIVE RESULTS — Healthy 3-scenario comparison")
#     print("=" * 65)
#
#     for tox, c_state, ae_state in zip(toxins, toxin_states, ae_states):
#         c_idx  = IDX[c_state]
#         ae_idx = IDX[ae_state]
#
#         c_base = float(solutions[base_name].y[c_idx, -1])
#         c_noi  = float(solutions[no_int_name].y[c_idx, -1])
#         c_int  = float(solutions[int_name].y[c_idx, -1])
#
#         cl_base = clearance_ae_over_auc_window(solutions[base_name],   c_idx, ae_idx, window)
#         cl_noi  = clearance_ae_over_auc_window(solutions[no_int_name], c_idx, ae_idx, window)
#         cl_int  = clearance_ae_over_auc_window(solutions[int_name],    c_idx, ae_idx, window)
#
#         print(f"\n  {tox}:")
#         print(f"    Final C_ss [mg/L]:")
#         print(f"      No Drug       = {c_base:.5f}")
#         print(f"      Drug, No Int  = {c_noi:.5f}  ({_pct(c_noi, c_base):+.2f}%)")
#         print(f"      Drug + Inhib  = {c_int:.5f}  ({_pct(c_int, c_base):+.2f}%)")
#         print(f"    CL_R [L/h]  (Ae/AUC over {window[0]:.0f}–{window[1]:.0f} h):")
#         print(f"      No Drug       = {cl_base:.4f}")
#         print(f"      Drug, No Int  = {cl_noi:.4f}  ({_pct(cl_noi, cl_base):+.2f}%)")
#         print(f"      Drug + Inhib  = {cl_int:.4f}  ({_pct(cl_int, cl_base):+.2f}%)")
#
#     print("\n" + "=" * 65 + "\n")
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 7 — Main simulation driver
# # ══════════════════════════════════════════════════════════════════════════════
#
# def run_healthy_and_plot(
#     window:  Tuple[float, float] = (0.0, 12.0),
#     t_ss:    float = 1500.0,
#     t_sim:   float = 12.0,
#     dt_ss:   float = 0.5,
#     dt_sim:  float = 0.02,
# ) -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
#     """
#     Full simulation pipeline: SS warm-up → 3 scenarios → figures.
#
#     Parameters
#     ----------
#     window  : (t0, t1) [h] for clearance calculation in fig_bars
#     t_ss    : steady-state warm-up duration [h] (default 1500 h)
#     t_sim   : main simulation duration [h] (default 12 h)
#     dt_ss   : time step for SS warm-up [h] (coarse, 0.5 h is fine)
#     dt_sim  : time step for main runs [h] (fine, 0.02 h for smooth curves)
#
#     Returns
#     -------
#     (fig_time, fig_bars, solutions_dict)
#     """
#     try:
#         import plotting
#         if hasattr(plotting, "setup_professional_style"):
#             plotting.setup_professional_style()
#     except Exception:
#         pass
#
#     print("\n=== Healthy 3-scenario comparison ===")
#     print(f"    SS warm-up: {t_ss:.0f} h  |  Main runs: {t_sim:.0f} h")
#
#     # ── Step 1: Steady-state warm-up (no drug) ────────────────────────────────
#     print(f"\n▶ Steady-state warm-up ({t_ss:.0f} h, no drug) ...")
#     set_scenario("H_NO_DRUG")
#     reset_ode_debug_flags()
#     sol_ss = run_sim(t_end_h=t_ss, mode="healthy", dt=dt_ss)
#
#     # ── Step 2: Build initial conditions ─────────────────────────────────────
#     ic_no_drug = _ic_from_solution_end(sol_ss)
#     ic_drug    = _ic_from_solution_end_zero_drug(sol_ss)
#
#     # ── Step 3: Scenario 1 — No Furosemide ───────────────────────────────────
#     print("\n▶ Scenario 1: Healthy No Furosemide (12 h from SS) ...")
#     set_scenario("H_NO_DRUG")
#     reset_ode_debug_flags()
#     sol_no = run_sim(
#         t_end_h=t_sim,
#         mode="healthy",
#         dt=dt_sim,
#         custom_initial_conds=ic_no_drug,
#     )
#
#     # ── Step 4: Scenario 2 — With Drug, No Interaction ───────────────────────
#     print("\n▶ Scenario 2: Healthy With Furosemide (No Interaction) (12 h from SS) ...")
#     set_scenario("H_WITH_DRUG_NO_INT")
#     reset_ode_debug_flags()
#     sol_no_int = run_sim(
#         t_end_h=t_sim,
#         mode="healthy",
#         dt=dt_sim,
#         custom_initial_conds=ic_drug,
#     )
#
#     # ── Step 5: Scenario 3 — With Drug + Inhibition ───────────────────────────
#     print("\n▶ Scenario 3: Healthy With Furosemide (Interaction ON) (12 h from SS) ...")
#     set_scenario("H_WITH_DRUG")
#     reset_ode_debug_flags()
#     sol_drug = run_sim(
#         t_end_h=t_sim,
#         mode="healthy",
#         dt=dt_sim,
#         custom_initial_conds=ic_drug,
#     )
#
#     # ── Step 6: Assemble results ──────────────────────────────────────────────
#     solutions = {
#         "Healthy No Furosemide":                    sol_no,
#         "Healthy With Furosemide (No Interaction)": sol_no_int,
#         "Healthy With Furosemide":                  sol_drug,
#     }
#
#     # ── Step 7: Print quantitative results ───────────────────────────────────
#     _print_pct_changes(solutions, window)
#
#     # ── Step 8: Build figures ─────────────────────────────────────────────────
#     fig_time = create_figure_healthy_timecourses(solutions)
#     fig_bars = create_figure_healthy_bars(solutions, window=window)
#
#     return fig_time, fig_bars, solutions
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SECTION 8 — Script entry point
# # ══════════════════════════════════════════════════════════════════════════════
#
# if __name__ == "__main__":
#     fig_time, fig_bars, _solutions = run_healthy_and_plot(
#         window=(0.0, 12.0),
#         t_ss=1500.0,
#         t_sim=12.0,
#         dt_ss=0.5,
#         dt_sim=0.02,
#     )
#
#     outdir = _ensure_figures_dir()
#     base1 = os.path.join(outdir, "healthy_timecourses_3scenarios_greyscale")
#     base2 = os.path.join(outdir, "healthy_bars_3scenarios_greyscale")
#
#     for ext in ("png", "pdf"):
#         fig_time.savefig(f"{base1}.{ext}", dpi=300, bbox_inches="tight")
#         fig_bars.savefig(f"{base2}.{ext}", dpi=300, bbox_inches="tight")
#         print(f"  Saved {base1}.{ext}")
#         print(f"  Saved {base2}.{ext}")
#
#     plt.show()