"""
Kidney PBPK Model — clearance flux decomposition driver [ScenarioManager-compatible]

Scenarios (each launched from steady-state):
  1) Healthy — No Furosemide
  2) Healthy — With Furosemide
  3) CKD-5  — No Furosemide
  4) CKD-5  — With Furosemide

Decomposes total renal clearance for furosemide and each uremic toxin
(IS, HA, pCS) into its OAT1-mediated active-transport and passive-diffusion
contributions, across all four scenarios above.

NOTE ON CKD SCALING VALUES IN THIS DOCSTRING (historical, see below):
  An earlier version of this file's docstring claimed the following
  per-compound CKD Vmax scales: IS baso=0.05/apical=0.05, HA baso=0.15/
  apical=0.15, drug baso=0.30/apical=0.40. As of the compound_parameters.py
  shipped in this package, the ACTUAL values are: drug baso=0.15/apical=0.15,
  IS baso=0.05/apical=0.15, HA baso=0.05/apical=0.15, pCS baso=0.05/
  apical=0.15 (see the "ROUND 5" notes in compound_parameters.py for the
  current calibration history). This script reads scales directly from
  compound_parameters.py at runtime, so the SIMULATION uses the correct
  current values regardless of this docstring — but if you are reading
  this file to understand the CKD scaling rationale, trust
  compound_parameters.py, not the numbers quoted here.
"""

from __future__ import annotations

import os
import sys
import datetime
from typing import Dict, Tuple, Optional, Sequence

import numpy as np

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from simulation import run_sim
from scenario_manager import set_scenario
from state_vector import IDX
import constants as K

# "plotting" is an OPTIONAL internal styling/saving helper module that is
# not part of this open-science package (it provided house-style figure
# formatting in the original lab environment). It is not required — every
# call site below already checks hasattr(plotting, ...) before using it,
# and falls back to plain matplotlib fig.savefig() if unavailable. We make
# the import itself optional here so the script runs standalone.
try:
    import plotting
except ImportError:
    class _NoPlotting:
        """Stand-in so hasattr(plotting, 'x') checks below resolve to False
        and every call site falls back to plain matplotlib behavior."""
        pass
    plotting = _NoPlotting()

import physiology as phys

import compounds as C
import compound_parameters as CP

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DURATION_SS_H   = 1500.0
DURATION_MAIN_H = 12.0
CKD_SCALE       = 0.10
TIME_WINDOW     = (6.0, 12.0)

# REMOVED: CP.set_ckd_vmax_scales(baso_scale=0.15, apical_scale=0.40)
# REMOVED: _ = CP.get_compound_parameters("ckd5")
# REASON:  These lines overrode the per-compound scales defined in
#          compound_parameters.py. All CKD Vmax scaling now comes
#          exclusively from compound_parameters.py _CKD_SCALES dict.

# If supported, set CKD physiology scaling once
if hasattr(K, "set_ckd_scale"):
    K.set_ckd_scale(CKD_SCALE)

BASELINE_IC = {
    "C_ven": 0.0,
    "C_tox1_cen": 0.5,     # IS healthy SS target
    "C_tox2_cen": 5.0,     # HA healthy SS target  (was 3.0 — corrected)
    "C_tox3_cen": 0.7,     # pCS healthy SS target (was 1.84 — corrected)
    "C_bladder": 0.0,
    "Ae_drg": 0.0,
    "Ae_t1": 0.0,
    "Ae_t2": 0.0,
    "Ae_t3": 0.0,
}

# ---------------------------------------------------------------------
# Greyscale palette + hatch patterns (COSMETIC ONLY)
# ---------------------------------------------------------------------
SCENARIO_COLORS = {
    "Healthy No Furosemide":   "#4D4D4D",
    "Healthy With Furosemide": "#9E9E9E",
    "CKD-5 No Furosemide":     "#7A7A7A",
    "CKD-5 With Furosemide":   "#BDBDBD",
}
SCENARIO_HATCHES = {
    "Healthy No Furosemide":   "",
    "Healthy With Furosemide": "//",
    "CKD-5 No Furosemide":     "xx",
    "CKD-5 With Furosemide":   "..",
}

DISPLAY_NAME = {
    "DRUG": "Furosemide",
    "IS": "IS",
    "HA": "HA",
    "pCS": "pCS",
}

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _timestamp_dir(base: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, outdir: str, name: str, exts=("png", "pdf"), dpi: int = 300) -> None:
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    if hasattr(plotting, "save_figure"):
        try:
            plotting.save_figure(fig, base, formats=list(exts), dpi=dpi)
            return
        except Exception as e:
            print(f"(plotting.save_figure failed for '{name}': {e})")
    for ext in exts:
        path = f"{base}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {path}")
        if name == "09_flux_decomposition_PBUTs_greyscale":
            pub_dir = os.path.join(_THIS_DIR, "..", "publication_code", "Figures")
            os.makedirs(pub_dir, exist_ok=True)
            pub = os.path.join(pub_dir, f"{name}.{ext}")
            fig.savefig(pub, dpi=dpi, bbox_inches="tight")
            print(f"  Saved {pub}")


def _steady_state_check(sol, label: str, rel_cv_thresh: float = 0.01) -> bool:
    print(f"\n{label} steady-state check (last 10% of points):")
    ok = True
    n = max(10, len(sol.t) // 10)
    for key, nice in [("C_tox1_cen", "IS"), ("C_tox2_cen", "HA"), ("C_tox3_cen", "pCS")]:
        if key not in IDX:
            print(f"  {nice}: (missing state '{key}')")
            ok = False
            continue
        series = sol.y[IDX[key]][-n:]
        mean = float(np.mean(series))
        sd = float(np.std(series))
        cv = (sd / mean) if mean > 0 else 0.0
        tag = "STEADY" if cv <= rel_cv_thresh else "NOT STEADY"
        print(f"  {nice}: CV={cv*100:.2f}% → {tag}")
        ok &= (cv <= rel_cv_thresh)
    return bool(ok)


def _ic_from_solution(sol) -> Dict[str, float]:
    return {name: float(sol.y[i, -1]) for name, i in IDX.items()}


def _ss_to_drug_start_ic(sol_ss) -> Dict[str, float]:
    """Start drug runs from toxin SS, zeroing only drug-related states."""
    eps = 1e-9
    y = sol_ss.y[:, -1].copy()
    if "C_ven" in IDX:
        y[IDX["C_ven"]] = eps
    for seg in getattr(K, "PROX", []) + getattr(K, "NONPROX", []) + getattr(K, "CDS", []):
        for suff in ("V", "C", "T"):
            key = f"C_{seg}_{suff}"
            if key in IDX:
                y[IDX[key]] = eps
    if "C_bladder" in IDX:
        y[IDX["C_bladder"]] = eps
    if "Ae_drg" in IDX:
        y[IDX["Ae_drg"]] = 0.0
    return {name: float(y[i]) for name, i in IDX.items()}


def _get_Q_GFR_Q_urine(mode: str) -> Tuple[float, float]:
    try:
        d = phys.make_phys(mode)
        Q_GFR   = float(d.get("Q_GFR",   0.0))
        Q_urine = float(d.get("Q_urine", 0.0))
        if Q_GFR > 0 and Q_urine > 0:
            return Q_GFR, Q_urine
    except Exception:
        pass
    flows   = K.get_tubular_flows_ml_min(mode)
    Q_GFR   = float(flows.get("P1",   0.0)) * K.ML_MIN_TO_L_H
    Q_urine = float(flows.get("CDC5", 0.0)) * K.ML_MIN_TO_L_H
    return Q_GFR, Q_urine


# ---------------------------------------------------------------------
# Flux computation
# ---------------------------------------------------------------------
def compute_fluxes_for_scenario_drug(
    sol,
    *,
    mode: str,
    time_window=(6.0, 12.0),
    include_drug: bool = True,
):
    EPS = getattr(K, "EPS", 1e-12)
    PROX = getattr(K, "PROX", ["P1", "P2", "P3"])
    USE_APICAL_MM = getattr(K, "USE_APICAL_MM", True)

    Q_GFR, Q_urine = _get_Q_GFR_Q_urine(mode)

    t0, t1 = map(float, time_window)
    if t0 > t1:
        t0, t1 = t1, t0

    mask = (sol.t >= t0) & (sol.t <= t1)
    if not np.any(mask):
        mask = (sol.t == sol.t[-1])
    y = np.mean(sol.y[:, mask], axis=1)

    def val(name: str) -> float:
        return float(y[IDX[name]]) if name in IDX else 0.0

    IS, HA, PCS, DRUG = C.tox1, C.tox2, C.tox3, C.drug
    TOX = {"IS": IS, "HA": HA, "pCS": PCS}

    def fu_over_BP(d: dict) -> float:
        fu = float(d.get("fu_p", 0.0))
        BP = float(d.get("BP", 1.0)) or 1.0
        return fu / BP

    Cven   = val("C_ven")
    Cu_drg = Cven * fu_over_BP(DRUG)

    Ki = {
        "IS":  float(IS.get("Ki_furo_to_IS_mgL",  1e12)),
        "HA":  float(HA.get("Ki_furo_to_HA_mgL",  1e12)),
        "pCS": float(PCS.get("Ki_furo_to_pCS_mgL", 1e12)),
    }
    inhib = {sp: (1.0 + (Cu_drg / max(Ki[sp], EPS))) for sp in ["IS", "HA", "pCS"]}

    flux = {
        "filtration":          {"IS": 0.0, "HA": 0.0, "pCS": 0.0},
        "basolateral_passive": {"IS": 0.0, "HA": 0.0, "pCS": 0.0},
        "basolateral_uptake":  {"IS": 0.0, "HA": 0.0, "pCS": 0.0},
        "apical_secretion":    {"IS": 0.0, "HA": 0.0, "pCS": 0.0},
        "inhibition_factors":  inhib,
        "window":              (t0, t1),
        "physiology":          {"Q_GFR": Q_GFR, "Q_urine": Q_urine},
        "sign_convention":     {"basolateral_passive": "+ = vascular→cell"},
    }

    Cpl = {"IS": val("C_tox1_cen"), "HA": val("C_tox2_cen"), "pCS": val("C_tox3_cen")}
    for sp in ["IS", "HA", "pCS"]:
        flux["filtration"][sp] = Q_GFR * fu_over_BP(TOX[sp]) * Cpl[sp]

    PASS = {
        "IS":  getattr(C, "TOX1_PASS", None),
        "HA":  getattr(C, "TOX2_PASS", None),
        "pCS": getattr(C, "TOX3_PASS", None),
    }
    TAG = {"IS": "tox1", "HA": "tox2", "pCS": "tox3"}

    W_raw = getattr(K, "W_BASO_PROX", {"P1": 0.40, "P2": 0.40, "P3": 0.20})
    norm  = sum(W_raw.get(s, 0.0) for s in PROX) or 1.0

    def mm_rate(Vmax, Km, S):
        return (float(Vmax) * float(S)) / (float(Km) + float(S) + EPS)

    for seg in PROX:
        w = W_raw.get(seg, 0.0) / norm
        for sp in ["IS", "HA", "pCS"]:
            tag = TAG[sp]
            fu_p_over_BP = fu_over_BP(TOX[sp])
            fu_cell = float(TOX[sp].get("fu_cell", 1.0))
            Cu_v = val(f"C_{tag}_{seg}_V") * fu_p_over_BP
            Cu_c = val(f"C_{tag}_{seg}_C") * fu_cell

            if PASS[sp] is not None:
                CLbs  = float(PASS[sp].get(f"{tag}_CL_PD_{seg}_basolateral_no_ui", 0.0))
                beta  = float(PASS[sp].get(f"{tag}_beta_{seg}", 1.0))
                gamma = float(PASS[sp].get(f"{tag}_gamma_plasma", 1.0))
                flux["basolateral_passive"][sp] += w * CLbs * (Cu_v * gamma - Cu_c * beta)

            Vmax = float(TOX[sp].get("Vmax_baso_uptake", 0.0))
            Km   = float(TOX[sp].get("Km_baso_uptake", 1e9))
            flux["basolateral_uptake"][sp] += w * mm_rate(Vmax, Km * inhib[sp], Cu_v)

            if USE_APICAL_MM:
                flux["apical_secretion"][sp] += w * mm_rate(
                    float(TOX[sp].get("Vmax_api_secr", 0.0)),
                    float(TOX[sp].get("Km_api_secr",   1e9)),
                    Cu_c)
            else:
                flux["apical_secretion"][sp] += w * float(TOX[sp].get("CL_sec_api", 0.0)) * Cu_c

    if include_drug:
        for k in ("filtration", "basolateral_passive", "basolateral_uptake", "apical_secretion"):
            flux[k]["DRUG"] = 0.0
        flux["filtration"]["DRUG"] = Q_GFR * fu_over_BP(DRUG) * Cven
        DRG_PASS  = getattr(C, "DRG_PASS", None)
        fu_cell_d = float(DRUG.get("fu_cell", 1.0))
        for seg in PROX:
            w = W_raw.get(seg, 0.0) / norm
            Cu_v = val(f"C_{seg}_V") * fu_over_BP(DRUG)
            Cu_c = val(f"C_{seg}_C") * fu_cell_d
            if DRG_PASS is not None:
                CLbs  = float(DRG_PASS.get(f"drg_CL_PD_{seg}_basolateral_no_ui", 0.0))
                beta  = float(DRG_PASS.get(f"drg_beta_{seg}", 1.0))
                gamma = float(DRG_PASS.get(f"drg_gamma_plasma", 1.0))
                flux["basolateral_passive"]["DRUG"] += w * CLbs * (Cu_v * gamma - Cu_c * beta)
            flux["basolateral_uptake"]["DRUG"] += w * mm_rate(
                float(DRUG.get("Vmax_baso_uptake", 0.0)),
                float(DRUG.get("Km_baso_uptake",   1e9)),
                Cu_v)
            if USE_APICAL_MM and "Vmax_api_secr" in DRUG:
                flux["apical_secretion"]["DRUG"] += w * mm_rate(
                    float(DRUG.get("Vmax_api_secr", 0.0)),
                    float(DRUG.get("Km_api_secr",   1e9)),
                    Cu_c)
            else:
                flux["apical_secretion"]["DRUG"] += w * float(DRUG.get("CL_sec_api", 0.0)) * Cu_c
        flux["inhibition_factors"]["DRUG"] = 1.0

    return flux


def compute_flux_contributions_percent(flux: Dict, species=("IS", "HA", "pCS", "DRUG")) -> Dict[str, Dict[str, float]]:
    processes = ("filtration", "basolateral_passive", "basolateral_uptake", "apical_secretion")
    out: Dict[str, Dict[str, float]] = {}
    for sp in species:
        vals  = {p: abs(float(flux[p].get(sp, 0.0))) for p in processes}
        total = sum(vals.values())
        out[sp] = ({p: float("nan") for p in processes} if total <= 0
                   else {p: 100.0 * vals[p] / total for p in processes})
    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_flux_panels(flux_by_scenario, *, species, title=None, outpath=None):
    import matplotlib.pyplot as plt
    if hasattr(plotting, "setup_professional_style"):
        try: plotting.setup_professional_style()
        except Exception: pass

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlesize": 16,
    })

    scenarios = list(flux_by_scenario.keys())
    # Process names live in the LaTeX caption; panels are lettered only.
    panels = [
        ("filtration",          "A"),
        ("basolateral_passive", "B"),
        ("basolateral_uptake",  "C"),
        ("apical_secretion",    "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    axes = axes.ravel()
    xtick_labels = [DISPLAY_NAME.get(sp, sp) for sp in species]
    x = np.arange(len(species), dtype=float)
    width = 0.82 / max(1, len(scenarios))
    left0 = x - (len(scenarios) - 1) * width / 2.0
    handles_for_legend = None

    for ax, (key, letter) in zip(axes, panels):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("")
        if key == "basolateral_passive":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylabel("Flux (mg/h)\n(symmetric log scale)")
        else:
            ax.set_ylabel("Flux (mg/h)")
        ax.text(
            0.02, 0.95, letter,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left",
        )
        handles = []
        for j, sc in enumerate(scenarios):
            vals  = [float(flux_by_scenario[sc][key].get(sp, 0.0)) for sp in species]
            rects = ax.bar(
                left0 + j * width, vals, width, label=sc,
                color=SCENARIO_COLORS.get(sc, "#999999"),
                hatch=SCENARIO_HATCHES.get(sc, ""),
                edgecolor="black", linewidth=0.4,
            )
            handles.append(rects[0])
        if handles_for_legend is None:
            handles_for_legend = handles

    if handles_for_legend is not None:
        fig.legend(handles_for_legend, scenarios,
                   loc="lower center", ncol=min(4, len(scenarios)),
                   frameon=False, fontsize=12,
                   bbox_to_anchor=(0.5, -0.02),
                   columnspacing=2.2, handlelength=2.6)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
# Steady-state and scenario runs
# ---------------------------------------------------------------------
def reach_steady_state():
    print("\n=== Steady-state (NO DRUG) ===")
    if hasattr(plotting, "setup_professional_style"):
        try: plotting.setup_professional_style()
        except Exception: pass
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds("healthy")

    print(f"▶ Healthy SS: {DURATION_SS_H:.0f} h (no drug)")
    set_scenario("H_NO_DRUG")
    sol_H_ss = run_sim(DURATION_SS_H, custom_initial_conds=BASELINE_IC,
                       mode="healthy", reset_compounds=False)

    print(f"▶ CKD-5 SS: {DURATION_SS_H:.0f} h (no drug)")
    set_scenario("CKD_NO_DRUG")
    sol_C_ss = run_sim(DURATION_SS_H, custom_initial_conds=BASELINE_IC,
                       mode="ckd5", reset_compounds=False)

    _steady_state_check(sol_H_ss, "Healthy", rel_cv_thresh=0.01)
    _steady_state_check(sol_C_ss, "CKD-5",  rel_cv_thresh=0.01)

    print("\nSS (final plasma, mg/L):")
    print(f"  Healthy → IS={sol_H_ss.y[IDX['C_tox1_cen']][-1]:.3f}  "
          f"HA={sol_H_ss.y[IDX['C_tox2_cen']][-1]:.3f}  "
          f"pCS={sol_H_ss.y[IDX['C_tox3_cen']][-1]:.3f}")
    print(f"  CKD-5  → IS={sol_C_ss.y[IDX['C_tox1_cen']][-1]:.3f}  "
          f"HA={sol_C_ss.y[IDX['C_tox2_cen']][-1]:.3f}  "
          f"pCS={sol_C_ss.y[IDX['C_tox3_cen']][-1]:.3f}")
    return sol_H_ss, sol_C_ss


def run_main_scenarios(sol_H_ss, sol_C_ss):
    scenarios = {
        "Healthy No Furosemide":  dict(scenario_name="H_NO_DRUG",      mode="healthy", ic=_ic_from_solution(sol_H_ss)),
        "Healthy With Furosemide":dict(scenario_name="H_WITH_DRUG",    mode="healthy", ic=_ss_to_drug_start_ic(sol_H_ss)),
        "CKD-5 No Furosemide":    dict(scenario_name="CKD_NO_DRUG",    mode="ckd5",    ic=_ic_from_solution(sol_C_ss)),
        "CKD-5 With Furosemide":  dict(scenario_name="CKD_WITH_DRUG",  mode="ckd5",    ic=_ss_to_drug_start_ic(sol_C_ss)),
    }
    solutions = {}
    meta      = {}
    for label, sc in scenarios.items():
        print(f"\n▶ Running {label} ({DURATION_MAIN_H:.1f} h from SS)")
        set_scenario(sc["scenario_name"])
        sol = run_sim(DURATION_MAIN_H, custom_initial_conds=sc["ic"],
                      mode=sc["mode"], reset_compounds=False)
        solutions[label] = sol
        meta[label]      = {"mode": sc["mode"], "scenario_name": sc["scenario_name"]}
    return solutions, meta


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> Dict[str, object]:
    K.VERBOSE = False
    if hasattr(plotting, "setup_professional_style"):
        try: plotting.setup_professional_style()
        except Exception: pass

    outdir = _timestamp_dir(os.path.join(_THIS_DIR, "..", "results", "flux_decomposition"))

    sol_H_ss, sol_C_ss = reach_steady_state()
    solutions, meta    = run_main_scenarios(sol_H_ss, sol_C_ss)

    flux_data_all = {}
    for name, sol in solutions.items():
        flux_data_all[name] = compute_fluxes_for_scenario_drug(
            sol, mode=meta[name]["mode"],
            time_window=TIME_WINDOW, include_drug=True)

    fig_tox = plot_flux_panels(flux_data_all, species=("IS", "HA", "pCS"),
        title=f"Renal flux decomposition (PBUTs) — mean over t in [{TIME_WINDOW[0]}, {TIME_WINDOW[1]}] h")
    _save(fig_tox, outdir, "09_flux_decomposition_PBUTs_greyscale")

    fig_drg = plot_flux_panels(flux_data_all, species=("DRUG",),
        title=f"Renal flux decomposition (Furosemide) — mean over t in [{TIME_WINDOW[0]}, {TIME_WINDOW[1]}] h")
    _save(fig_drg, outdir, "10_flux_decomposition_furosemide_greyscale")

    fig_all = plot_flux_panels(flux_data_all, species=("DRUG", "IS", "HA", "pCS"),
        title=f"Renal flux decomposition (all) — mean over t in [{TIME_WINDOW[0]}, {TIME_WINDOW[1]}] h")
    _save(fig_all, outdir, "11_flux_decomposition_all_greyscale")

    print("\n=== Flux contribution (%) — absolute magnitudes ===")
    for scen_name, flx in flux_data_all.items():
        print(f"\n{scen_name}")
        contrib = compute_flux_contributions_percent(flx, species=("IS", "HA", "pCS", "DRUG"))
        for sp in ("IS", "HA", "pCS", "DRUG"):
            c = contrib[sp]
            print(f"  {DISPLAY_NAME.get(sp, sp)}: "
                  f"Filt={c['filtration']:.1f}%, "
                  f"Pass={c['basolateral_passive']:.1f}%, "
                  f"Uptake={c['basolateral_uptake']:.1f}%, "
                  f"Apical={c['apical_secretion']:.1f}%")

    print("\nFigures saved to:", outdir)
    return {"solutions": solutions, "flux_data_all": flux_data_all,
            "healthy_ss": sol_H_ss, "ckd_ss": sol_C_ss, "outdir": outdir}


if __name__ == "__main__":
    main()