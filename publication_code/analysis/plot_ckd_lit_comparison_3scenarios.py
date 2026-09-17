# plot_ckd_lit_comparison.py
# -*- coding: utf-8 -*-
"""
plot_ckd_lit_comparison.py
==========================
CKD-5 simulation and plotting: three scenarios compared against literature.

WHAT THIS SCRIPT DOES
----------------------
Runs three 12 h simulations from CKD-5 steady state and produces:

  fig_time  — 1×3 panel: plasma time-courses of IS, HA, pCS
  fig_bars  — 1×2 panel:
                Left  (Panel C): final CKD plasma concentrations vs literature
                Right (Panel D): CKD renal clearance vs literature

SCENARIOS
---------
  1) CKD No Furosemide
       drug["enabled"] = False, drug["inhibit"] = False
       Establishes the CKD baseline without any drug effect.

  2) CKD With Furosemide (No Interaction)
       drug["enabled"] = True,  drug["inhibit"] = False
       Drug is present and cleared renally, but does NOT competitively
       inhibit toxin OAT1 uptake. Separates pharmacokinetic from
       pharmacodynamic effects of furosemide.

  3) CKD With Furosemide
       drug["enabled"] = True,  drug["inhibit"] = True
       Full model: furosemide is present AND competitively inhibits
       toxin uptake at OAT1 (Km shift via Ki = 4.13 mg/L).

SIMULATION PROTOCOL
-------------------
  Pre-run  : 2100–11100 h at dt=0.5 h to reach CKD toxin steady state
             (no drug — establishes toxin accumulation baseline)
  Main run : 12 h at dt=0.02 h from steady-state initial conditions

INHIBITION CONTROL
------------------
  Inhibition is controlled ENTIRELY by C.drug["inhibit"] (bool).
  The ODE reads this flag at lines 834 and 863–866 of
  ode_system_twocompartment.py. There is NO separate constants flag
  that controls inhibition — K.INHIBIT_TOX_BY_DRUG does not exist
  in the current ODE and has been removed from this script.

UNIT CONVENTIONS (must match model output)
------------------------------------------
  Concentrations : mg/L
  Clearances     : L/h
  Time           : h

LITERATURE VALUES — UNITS AND SOURCES
--------------------------------------
  LIT_CONC_CKD  [mg/L]:
    IS  : 37.0 mg/L  (147 μmol/L × 251.24/1000)
          Range 24–50 mg/L. Representative CKD5 total plasma IS.
          NOTE: replace with your validated literature source.

    HA  : 109.0 mg/L
          Healthy <5 mg/L; CKD5 reported ~247±112 mg/L by some studies
          (PMC3739763, read in session). 109 mg/L is lower — verify
          against your target population before finalising.

    pCS : 23.0 mg/L  (122 μmol/L × 188.20/1000)
          Range 15–32 mg/L. Representative CKD5 total plasma pCS.
          NOTE: replace with your validated literature source.

  LIT_CLEAR_CKD [L/h]:
    IS  : 1.6 L/h  (26.7 mL/min)
    HA  : 6.0 L/h  (100 mL/min)
    pCS : 1.4 L/h  (23.3 mL/min)
    NOTE: these are placeholder values. Replace with published CKD5
    renal clearance data from your target literature.

CORRECTIONS FROM PREVIOUS VERSION
-----------------------------------
  1. reset_ode_debug_flags(): changed
         from: from ode_system import kidney_rhs
         to:   from ode_system_twocompartment import kidney_rhs
     The old import would either raise ImportError (if ode_system.py
     does not exist) or reset flags on the wrong module, leaving stale
     debug state in the actual ODE that runs.

  2. Removed K.INHIBIT_TOX_BY_DRUG = True/False from all configure
     functions. This constant is NOT read by ode_system_twocompartment.py.
     The ODE reads only C.drug["inhibit"]. Setting K.INHIBIT_TOX_BY_DRUG
     had zero effect on simulation results and was actively misleading.

  3. Added unit and source documentation to LIT_CONC_CKD and LIT_CLEAR_CKD.

  4. All other logic, simulation protocol, figure layout, and scenario
     structure are preserved exactly as in the original.
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

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — required for unattended/server runs
import matplotlib.pyplot as plt

import constants as K
import compounds as C
from state_vector import IDX
from simulation import run_sim

# "plotting" is an OPTIONAL internal styling helper module that is not part
# of this open-science package. Every call site already checks
# hasattr(plotting, ...) before using it. Make the import itself optional
# too, so the script runs standalone without that module.
try:
    import plotting
except ImportError:
    class _NoPlotting:
        """Stand-in so hasattr(plotting, 'x') checks resolve to False and
        every call site falls back to plain matplotlib behavior."""
        pass
    plotting = _NoPlotting()

# ── Version-safe trapezoidal integration ───────────────────────────────────
# NumPy >= 2.0 added np.trapezoid and deprecated np.trapz; later NumPy
# releases removed np.trapz entirely. Use hasattr() rather than a bare
# np.trapz fallback, which would raise AttributeError immediately.
# NOTE: the docstring for _clearance_auc_window() below claims
# "np.trapezoid is used for the AUC integral", but the code previously
# called the (now-removed) np.trapz directly — a documentation/
# implementation mismatch. Fixed here to match what the docstring always
# said it did.
def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def reset_ode_debug_flags() -> None:
    """
    Reset all per-run debug flags on the ODE function.

    This must be called before each simulation so that debug prints
    (infusion gate, peak window, P1 lumen, etc.) fire again for the
    new run rather than being suppressed by flags from the previous run.

    CORRECTION: imports from ode_system_twocompartment (not ode_system).
    The active ODE is always ode_system_twocompartment.kidney_rhs.
    """
    try:
        # CORRECTED: was 'from ode_system import kidney_rhs'
        # ode_system_twocompartment is the file that run_sim() uses.
        from ode_system_twocompartment import kidney_rhs

        kidney_rhs._printed_debug        = False
        kidney_rhs._apical_guarded       = False

        # Clear any per-timestep flags set during the previous run
        for attr in list(dir(kidney_rhs)):
            if (
                attr.startswith("_printed_cin_")
                or attr.startswith("_checked_")
                or attr.startswith("_warned_zero_")
                or attr.startswith("_p1_lumen_")
                or attr == "_flow_debug_done"
                or attr == "_dbg_last_bucket"
                or attr == "_printed_peak_debug"
            ):
                try:
                    delattr(kidney_rhs, attr)
                except AttributeError:
                    pass

        print("[DEBUG] ODE debug flags reset")

    except ImportError as e:
        print(f"[WARN] Could not reset ODE debug flags: {e}")


def setup_professional_style() -> None:
    """Apply project-wide matplotlib style if available."""
    if hasattr(plotting, "setup_professional_style"):
        plotting.setup_professional_style()
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _ensure_output_dir() -> str:
    """Create the shared top-level results/ folder if it does not exist."""
    outdir = os.path.join(_THIS_DIR, "..", "results")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _dedupe_legend(
    handles: list,
    labels: list,
) -> Tuple[list, list]:
    """Deduplicate legend entries while preserving insertion order."""
    seen: set = set()
    out_h: list = []
    out_l: list = []
    for h, l in zip(handles, labels):
        if l not in seen:
            out_h.append(h)
            out_l.append(l)
            seen.add(l)
    return out_h, out_l


def _clearance_auc_window(
    sol: Any,
    toxin_idx: Optional[int],
    Ae_idx: Optional[int],
    time_window: Tuple[float, float],
) -> float:
    """
    Compute renal clearance over a time window using CLR = Ae / AUC.

    Parameters
    ----------
    sol        : OdeResult from solve_ivp (returned by run_sim)
    toxin_idx  : row index in sol.y for the toxin's central concentration
    Ae_idx     : row index in sol.y for the cumulative excreted amount
    time_window: (t_start, t_end) [h]

    Returns
    -------
    float : CLR [L/h], or np.nan if inputs are invalid or AUC ≈ 0.

    Method
    ------
    CLR = ΔAe [mg] / AUC [mg/L·h] = L/h

    np.maximum.accumulate is applied to Ae to guard against the small
    numerical noise that can make Ae appear to decrease by a few ULP.
    np.trapezoid is used for the AUC integral (correct for non-uniform grids).
    """
    if sol is None or toxin_idx is None or Ae_idx is None:
        return np.nan

    t0, t1 = time_window
    t = np.asarray(sol.t, dtype=float)

    if t.size < 3:
        return np.nan
    if t1 <= t0 or t1 <= t[0] or t0 >= t[-1]:
        return np.nan

    t0_clip = max(t0, t[0])
    t1_clip = min(t1, t[-1])

    i0 = int(np.searchsorted(t, t0_clip, side="left"))
    i1 = int(np.searchsorted(t, t1_clip, side="right"))

    if i1 - i0 < 2:
        return np.nan

    t_w  = t[i0:i1]
    C_w  = np.clip(np.asarray(sol.y[toxin_idx][i0:i1], dtype=float), 0.0, np.inf)
    Ae_w = np.maximum.accumulate(np.asarray(sol.y[Ae_idx][i0:i1], dtype=float))

    dAe = max(0.0, float(Ae_w[-1] - Ae_w[0]))   # [mg]
    auc = float(_trapz(C_w, t_w))           # [mg/L·h]

    if auc <= 1e-9:
        return np.nan

    return dAe / auc   # [L/h]


def _legend_label(name: str) -> str:
    """Drop Healthy/CKD from legend text; row placement already shows condition."""
    s = str(name).strip()
    for prefix in ("Healthy ", "CKD-5 ", "CKD "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.lower().startswith("literature"):
        return "Literature"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Compound and state index maps
# ══════════════════════════════════════════════════════════════════════════════

TOXINS       = ["IS",          "HA",          "pCS"]
TOXIN_STATES = ["C_tox1_cen",  "C_tox2_cen",  "C_tox3_cen"]
AE_STATES    = ["Ae_t1",       "Ae_t2",        "Ae_t3"]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Literature reference values
#     Units: concentrations in mg/L, clearances in L/h
#     SOURCE: placeholder values — replace with validated citations
# ══════════════════════════════════════════════════════════════════════════════

LIT_CONC_CKD: Dict[str, Dict[str, Any]] = {
    # Total plasma concentrations in CKD5 [mg/L]
    # IS:  ~37 mg/L ≈ 147 μM (MW=251.24). CKD5 range varies widely.
    # HA:  ~109 mg/L. Note: healthy <5 mg/L (PMC3739763).
    #       CKD5 values up to 247±112 mg/L reported — verify your source.
    # pCS: ~23 mg/L ≈ 122 μM (MW=188.20).
    # ALL UNITS: mg/L (consistent with model output)
    "IS":  {"value": 37.0,  "range": (24.0, 50.0)},
    "HA":  {"value": 109.0, "range": (44.0, 175.0)},
    "pCS": {"value": 23.0,  "range": (15.0, 32.0)},
}

LIT_CLEAR_CKD: Dict[str, Dict[str, Any]] = {
    # Renal clearance in CKD5 [L/h]
    # IS:  1.6 L/h = 26.7 mL/min
    # HA:  6.0 L/h = 100 mL/min
    # pCS: 1.4 L/h = 23.3 mL/min
    # ALL UNITS: L/h (consistent with model output from _clearance_auc_window)
    "IS":  {"value": 1.6, "range": (1.4, 1.8)},
    "HA":  {"value": 6.0, "range": (4.7, 7.4)},
    "pCS": {"value": 1.4, "range": (1.1, 1.7)},
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Figure builders
# ══════════════════════════════════════════════════════════════════════════════

def create_figure_timecourses_generic(
    solutions:       Mapping[str, Any],
    idx_map:         Mapping[str, int],
    scenario_names:  List[str],
    panels:          List[Tuple[str, str]],
    colors:          Dict[str, str],
    line_styles:     Dict[str, str],
    legend_bbox:     Tuple[float, float] = (0.5, 1.02),
    legend_ncol:     int = 3,
    legend_fontsize: int = 15,
) -> plt.Figure:
    """
    Build the 1×3 time-course figure (fig_time).

    Each panel shows one toxin's plasma concentration over time for all
    three scenarios. A single shared legend is placed above all panels.

    Parameters
    ----------
    solutions      : {scenario_name: OdeResult}
    idx_map        : IDX dict mapping state name → row index in sol.y
    scenario_names : ordered list of scenario name strings
    panels         : [(toxin_label, state_name), ...] — one per panel
    colors         : {scenario_name: hex_color_string}
    line_styles    : {scenario_name: linestyle_string}
    legend_bbox    : bbox_to_anchor for fig.legend
    legend_ncol    : number of legend columns
    legend_fontsize: legend font size [pt]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 4), sharex=False, sharey=False
    )
    legend_handles = None
    legend_labels  = None

    for ax, (toxin_label, state_name) in zip(axes, panels):
        s_idx = idx_map.get(state_name)
        if s_idx is None:
            ax.set_axis_off()
            continue

        for name in scenario_names:
            sol = solutions.get(name)
            if sol is None:
                continue
            ax.plot(
                sol.t,
                sol.y[s_idx],
                label=name,
                color=colors.get(name, "#000000"),
                linestyle=line_styles.get(name, "-"),
                linewidth=2.5,
            )

        ax.set_xlabel("Time (h)", fontsize=16)
        ax.set_ylabel("Concentration (mg/L)", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.02, 0.95, toxin_label,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left",
        )

        # Collect legend handles from the first valid panel only
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Remove any per-axes legend that matplotlib auto-created
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # One shared legend centred above all panels
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=legend_ncol,
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
    idx_map:         Mapping[str, int],
    window:          Tuple[float, float],
    scenario_names:  List[str],
    colors:          Dict[str, str],
    toxins:          List[str],
    toxin_states:    List[str],
    ae_states:       List[str],
    lit_conc:        Dict[str, Dict[str, Any]],
    lit_clear:       Dict[str, Dict[str, Any]],
    clearance_fn,
    legend_bbox:     Tuple[float, float] = (0.5, -0.07),
    legend_ncol:     int = 4,
    legend_fontsize: int = 15,
    fig=None,
    axes=None,
    show_legend: bool = True,
    panel_letters: Tuple[str, str] = ("C", "D"),
    apply_tight_layout: bool = True,
) -> plt.Figure:
    """
    Build the 1×2 bar figure (fig_bars).

    Left panel (C)  : final plasma concentration vs literature (mg/L)
    Right panel (D) : renal clearance over window vs literature (L/h)

    Literature values are shown as error bars (mean ± range) plotted
    at the x-position of scenario_names[0] bars.

    Parameters
    ----------
    solutions      : {scenario_name: OdeResult}
    idx_map        : IDX dict mapping state name → row index in sol.y
    window         : (t_start, t_end) [h] for clearance calculation
    scenario_names : ordered list of 3 scenario name strings
    colors         : {scenario_name: hex_color_string}
    toxins         : ["IS", "HA", "pCS"]
    toxin_states   : ["C_tox1_cen", "C_tox2_cen", "C_tox3_cen"]
    ae_states      : ["Ae_t1", "Ae_t2", "Ae_t3"]
    lit_conc       : literature concentration dict [mg/L]
    lit_clear      : literature clearance dict [L/h]
    clearance_fn   : callable with signature
                     (sol, c_idx, ae_idx, window) → float [L/h]
    legend_bbox    : bbox_to_anchor for fig.legend
    legend_ncol    : number of legend columns
    legend_fontsize: legend font size [pt]

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(scenario_names) != 3:
        raise ValueError(
            f"create_figure_bars_generic expects exactly 3 scenarios, "
            f"got {len(scenario_names)}"
        )

    if axes is None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    else:
        axL, axR = axes
        if fig is None:
            fig = axL.figure

    n_tox = len(toxins)
    x     = np.arange(n_tox)
    width = 0.22

    # Bar x-positions for the three scenarios
    x_s0  = x - width   # scenario 0 (no drug)
    x_s1  = x           # scenario 1 (drug, no interaction)
    x_s2  = x + width   # scenario 2 (drug + interaction)

    sc0, sc1, sc2 = scenario_names

    # ── Panel C: final plasma concentrations ──────────────────────────────────
    finals: Dict[str, List[float]] = {name: [] for name in scenario_names}

    for state in toxin_states:
        s_idx = idx_map.get(state)
        for name in scenario_names:
            sol = solutions.get(name)
            if sol is not None and s_idx is not None:
                finals[name].append(float(sol.y[s_idx, -1]))
            else:
                finals[name].append(np.nan)

    axL.bar(x_s0, finals[sc0], width, label=_legend_label(sc0),
            color=colors[sc0], alpha=0.85, edgecolor="black", linewidth=0.5)
    axL.bar(x_s1, finals[sc1], width, label=_legend_label(sc1),
            color=colors[sc1], alpha=0.85, edgecolor="black", linewidth=0.5)
    axL.bar(x_s2, finals[sc2], width, label=_legend_label(sc2),
            color=colors[sc2], alpha=0.85, edgecolor="black", linewidth=0.5)

    # Literature error bars (plotted at x_s0 position)
    lit_vals = [lit_conc[t]["value"] for t in toxins]
    lit_lo   = [lit_conc[t]["value"] - lit_conc[t]["range"][0] for t in toxins]
    lit_hi   = [lit_conc[t]["range"][1] - lit_conc[t]["value"]  for t in toxins]

    axL.errorbar(
        x_s0, lit_vals,
        yerr=[lit_lo, lit_hi],
        fmt="D", color="black", markerfacecolor="black",
        markersize=6, capsize=5, capthick=2, elinewidth=2,
        label="Literature",
        zorder=5,
    )

    axL.set_xlabel("PBUT", fontsize=16)
    axL.set_ylabel("Plasma concentration (mg/L)", fontsize=16)
    axL.set_xticks(x)
    axL.set_xticklabels(toxins)
    axL.tick_params(axis="both", labelsize=12)
    axL.grid(True, alpha=0.25, axis="y")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.text(0.02, 0.95, panel_letters[0], transform=axL.transAxes,
             fontsize=16, fontweight="bold", va="top")

    # ── Panel D: renal clearances ─────────────────────────────────────────────
    clears: Dict[str, List[float]] = {name: [] for name in scenario_names}

    for c_state, ae_state in zip(toxin_states, ae_states):
        c_idx  = idx_map.get(c_state)
        ae_idx = idx_map.get(ae_state)
        for name in scenario_names:
            sol = solutions.get(name)
            clears[name].append(
                clearance_fn(sol, c_idx, ae_idx, window)
            )

    axR.bar(x_s0, clears[sc0], width, label=_legend_label(sc0),
            color=colors[sc0], alpha=0.85, edgecolor="black", linewidth=0.5)
    axR.bar(x_s1, clears[sc1], width, label=_legend_label(sc1),
            color=colors[sc1], alpha=0.85, edgecolor="black", linewidth=0.5)
    axR.bar(x_s2, clears[sc2], width, label=_legend_label(sc2),
            color=colors[sc2], alpha=0.85, edgecolor="black", linewidth=0.5)

    axR.set_xlabel("Toxins", fontsize=16)
    axR.set_ylabel("Clearance (L/h)", fontsize=16)
    axR.set_xticks(x)
    axR.set_xticklabels(toxins)
    axR.tick_params(axis="both", labelsize=12)
    axR.grid(True, alpha=0.25, axis="y")
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.text(0.02, 0.95, panel_letters[1], transform=axR.transAxes,
             fontsize=16, fontweight="bold", va="top")

    # ── Shared legend below both panels ───────────────────────────────────────
    for ax in (axL, axR):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if show_legend:
        h1, l1 = axL.get_legend_handles_labels()
        h2, l2 = axR.get_legend_handles_labels()
        handles, labels = _dedupe_legend(h1 + h2, l1 + l2)

        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fontsize,
            bbox_to_anchor=legend_bbox,
            columnspacing=2.3,
            handlelength=2.4,
        )

    if apply_tight_layout:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Scenario configuration helpers
# ══════════════════════════════════════════════════════════════════════════════

def _reset_compounds(mode: str = "ckd5") -> None:
    """
    Reinitialise all compound dicts for the given mode.

    Calls initialize_compounds() if available (preferred), otherwise
    falls back to load_compounds(). Raises if neither is present.
    """
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)
    elif hasattr(C, "load_compounds"):
        C.load_compounds(mode)
    else:
        raise RuntimeError(
            "compounds module has neither initialize_compounds() nor "
            "load_compounds(). Cannot reset compound parameters."
        )


def _configure_ckd_no_drug(mode: str = "ckd5") -> None:
    """
    Configure scenario 1: CKD, no furosemide.

    drug["enabled"] = False  — no infusion, no OAT1 transport of drug
    drug["inhibit"] = False  — no competitive inhibition of toxin OAT1

    Toxin production and secretion run under CKD Vmax scaling only.
    """
    _reset_compounds(mode)

    if C.drug is not None:
        C.drug["enabled"] = False
        C.drug["inhibit"] = False
        # Zero out all dosing keys to prevent any accidental infusion
        for key in ("IR", "t_start_h", "t_inf_h", "bolus_mg"):
            if key in C.drug:
                C.drug[key] = 0.0

    print("[CONFIG] Scenario 1 — CKD No Furosemide: "
          "drug.enabled=False, drug.inhibit=False")


def _configure_ckd_with_drug_no_interaction(mode: str = "ckd5") -> None:
    """
    Configure scenario 2: CKD with furosemide, NO OAT1 inhibition.

    drug["enabled"] = True   — drug is present, dosed, and renally cleared
    drug["inhibit"] = False  — OAT1 Km for toxins is NOT shifted
                               (Ki has no effect when inhibit=False)

    This isolates the pharmacokinetic presence of furosemide from its
    pharmacodynamic inhibitory effect on toxin secretion.

    NOTE: inhibition is controlled ONLY by C.drug["inhibit"].
    There is no K.INHIBIT_TOX_BY_DRUG flag in the ODE.
    """
    _reset_compounds(mode)

    if C.drug is None:
        raise RuntimeError(
            "C.drug is None after _reset_compounds(). "
            "Check initialize_compounds() in compounds.py."
        )

    C.drug["enabled"] = True
    C.drug["inhibit"] = False

    print("[CONFIG] Scenario 2 — CKD With Furosemide (No Interaction): "
          "drug.enabled=True, drug.inhibit=False")


def _configure_ckd_with_drug(mode: str = "ckd5") -> None:
    """
    Configure scenario 3: CKD with furosemide WITH OAT1 inhibition.

    drug["enabled"] = True   — drug is present, dosed, and renally cleared
    drug["inhibit"] = True   — OAT1 Km for each toxin is shifted by:
                               Km_eff = Km_base * (1 + Cu_drug / Ki)
                               where Ki = 4.13 mg/L (12.49 μM)
                               Source: Mihaila 2020, Toxins 12:391

    NOTE: inhibition is controlled ONLY by C.drug["inhibit"].
    There is no K.INHIBIT_TOX_BY_DRUG flag in the ODE.
    """
    _reset_compounds(mode)

    if C.drug is None:
        raise RuntimeError(
            "C.drug is None after _reset_compounds(). "
            "Check initialize_compounds() in compounds.py."
        )

    C.drug["enabled"] = True
    C.drug["inhibit"] = True

    print("[CONFIG] Scenario 3 — CKD With Furosemide: "
          "drug.enabled=True, drug.inhibit=True")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main driver
# ══════════════════════════════════════════════════════════════════════════════

def run_ckd_and_plot(
    window:    Tuple[float, float] = (0.0, 12.0),
    t_pre_h:   float = 2100.0,
    t_main_h:  float = 12.0,
    dt_pre:    float = 0.5,
    dt_main:   float = 0.02,
) -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
    """
    Run the full CKD simulation pipeline and return both figures.

    Execution order
    ---------------
    1. Pre-run (no drug, CKD mode) for t_pre_h hours → CKD steady state
    2. Extract steady-state concentrations as custom_ic for all three runs
    3. Run scenario 1 (No Furosemide) for t_main_h from custom_ic
    4. Run scenario 2 (Furosemide, No Interaction) for t_main_h from custom_ic
    5. Run scenario 3 (Furosemide + Interaction) for t_main_h from custom_ic
    6. Build fig_time (time-courses) and fig_bars (bar + literature)

    Parameters
    ----------
    window   : (t0, t1) [h] for clearance window in fig_bars
    t_pre_h  : pre-run duration [h] to reach CKD toxin steady state
    t_main_h : main simulation duration [h]
    dt_pre   : time step for pre-run [h] (coarse is fine)
    dt_main  : time step for main runs [h] (fine for smooth curves)

    Returns
    -------
    (fig_time, fig_bars, solutions_dict)
    """
    mode = "ckd5"

    setup_professional_style()

    # ── Pre-run: reach CKD toxin steady state ────────────────────────────────
    print(f"\n>>> Pre-run: CKD-5 baseline to steady state "
          f"({t_pre_h:.0f} h, dt={dt_pre} h) ...")
    _configure_ckd_no_drug(mode)
    reset_ode_debug_flags()

    sol_ss = run_sim(
        t_end_h=t_pre_h,
        mode=mode,
        dt=dt_pre,
    )

    # Extract steady-state endpoint as initial conditions for all main runs
    custom_ic = {name: float(sol_ss.y[idx, -1]) for name, idx in IDX.items()}

    # ── Scenario 1: CKD No Furosemide ────────────────────────────────────────
    print(f"\n>>> Scenario 1: CKD No Furosemide "
          f"({t_main_h} h from CKD steady state) ...")
    _configure_ckd_no_drug(mode)
    reset_ode_debug_flags()

    sol_no_drug = run_sim(
        t_end_h=t_main_h,
        mode=mode,
        dt=dt_main,
        custom_initial_conds=custom_ic,
    )

    # ── Scenario 2: CKD With Furosemide, No Interaction ──────────────────────
    print(f"\n>>> Scenario 2: CKD With Furosemide (No Interaction) "
          f"({t_main_h} h from CKD steady state) ...")
    _configure_ckd_with_drug_no_interaction(mode)
    reset_ode_debug_flags()

    sol_with_drug_no_int = run_sim(
        t_end_h=t_main_h,
        mode=mode,
        dt=dt_main,
        custom_initial_conds=custom_ic,
    )

    # ── Scenario 3: CKD With Furosemide + Interaction ────────────────────────
    print(f"\n>>> Scenario 3: CKD With Furosemide (With Interaction) "
          f"({t_main_h} h from CKD steady state) ...")
    _configure_ckd_with_drug(mode)
    reset_ode_debug_flags()

    sol_with_drug = run_sim(
        t_end_h=t_main_h,
        mode=mode,
        dt=dt_main,
        custom_initial_conds=custom_ic,
    )

    # ── Assemble results ──────────────────────────────────────────────────────
    scenario_names = [
        "CKD No Furosemide",
        "CKD With Furosemide (No Interaction)",
        "CKD With Furosemide",
    ]

    solutions_ckd: Dict[str, Any] = {
        "CKD No Furosemide":                     sol_no_drug,
        "CKD With Furosemide (No Interaction)":  sol_with_drug_no_int,
        "CKD With Furosemide":                   sol_with_drug,
    }

    # ── Visual style (greyscale) ──────────────────────────────────────────────
    colors: Dict[str, str] = {
        "CKD No Furosemide":                    "#4d4d4d",   # mid-grey
        "CKD With Furosemide (No Interaction)": "#9a9a9a",   # light-grey
        "CKD With Furosemide":                  "#1a1a1a",   # near-black
    }

    line_styles: Dict[str, str] = {
        "CKD No Furosemide":                    "-",
        "CKD With Furosemide (No Interaction)": ":",
        "CKD With Furosemide":                  "--",
    }

    panels = [
        ("IS",  "C_tox1_cen"),
        ("HA",  "C_tox2_cen"),
        ("pCS", "C_tox3_cen"),
    ]

    # ── Build figures ─────────────────────────────────────────────────────────
    fig_time = create_figure_timecourses_generic(
        solutions=solutions_ckd,
        idx_map=IDX,
        scenario_names=scenario_names,
        panels=panels,
        colors=colors,
        line_styles=line_styles,
        legend_bbox=(0.5, 1.02),
        legend_ncol=3,
        legend_fontsize=15,
    )

    fig_bars = create_figure_bars_generic(
        solutions=solutions_ckd,
        idx_map=IDX,
        window=window,
        scenario_names=scenario_names,
        colors=colors,
        toxins=TOXINS,
        toxin_states=TOXIN_STATES,
        ae_states=AE_STATES,
        lit_conc=LIT_CONC_CKD,
        lit_clear=LIT_CLEAR_CKD,
        clearance_fn=_clearance_auc_window,
        legend_bbox=(0.5, -0.07),
        legend_ncol=4,
        legend_fontsize=15,
    )

    return fig_time, fig_bars, solutions_ckd


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Script entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig_time, fig_bars, solutions_ckd = run_ckd_and_plot(
        window=(0.0, 2.0),
        t_pre_h=15000.0,
        t_main_h=2.0,
        dt_pre=0.5,
        dt_main=0.02,
    )

    outdir = _ensure_output_dir()
    base_time = os.path.join(outdir, "ckd_timecourses_3scenarios_greyscale")
    base_bars = os.path.join(outdir, "ckd_bars_3scenarios_greyscale")

    for ext in ("png", "pdf"):
        fig_time.savefig(f"{base_time}.{ext}", dpi=300, bbox_inches="tight")
        fig_bars.savefig(f"{base_bars}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved {base_time}.{ext}")
        print(f"Saved {base_bars}.{ext}")

    plt.close(fig_time)
    plt.close(fig_bars)