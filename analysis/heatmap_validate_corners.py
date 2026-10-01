# -*- coding: utf-8 -*-
"""10-point chronic ODE vs analytical decision test (Figs 7 and 8).

Five Ki × Vmax points (star + four corners of Fig 7) and five fu_drug ×
fu_toxin points (star + four corners of Fig 8). Flag |ODE − analytical|
> 2 percentage points.

Default is analytical-only (prints Css_furo). Chronic ODE:
    python heatmap_validate_corners.py --ode

Uses the same parameter setters as the grid:
    heatmap_ki_vmax_engine._force_ki, _set_vmax
and the Fig 5 driver:
    dose_response_tid_vs_qd._run_simulation (TID 80×3, 30 d)

Day-boundary: _run_simulation zeros C_ven once at the start of the 30-day
loop, then run_one_day_tid carries leftover. Check console RESET counts
on the five-panel figure, not this script.
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
for p in (_SRC_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from heatmap_analytical_formulas import (
    css_furo_theoretical, mm_pct, KI_DEFAULT, FU_P, BP, f_oat1_from_gfr,
)
from heatmap_ki_vmax_engine import (
    _force_ki, _set_vmax, _warmup_ic, _run_chronic_oral,
)

# Fig 7 axis limits from the IV/analytical maps (healthy-fraction Vmax).
KI_STAR, VMAX_STAR = 0.35, 0.05
KI_LO, KI_HI = 0.05, 5.0
VMAX_LO, VMAX_HI = 0.01, 0.30
KIVMAX_POINTS = [
    ("star", KI_STAR, VMAX_STAR),
    ("Ki_lo_V_lo", KI_LO, VMAX_LO),
    ("Ki_lo_V_hi", KI_LO, VMAX_HI),
    ("Ki_hi_V_lo", KI_HI, VMAX_LO),
    ("Ki_hi_V_hi", KI_HI, VMAX_HI),
]

# Fig 8: drug fu corners × toxin fu multiplier (Deltombe CKD-5 bases).
DRUG_FU_STAR, TOX_MULT_STAR = 0.05, 1.00
DRUG_FU_LO, DRUG_FU_HI = 0.01, 0.13
TOX_MULT_LO, TOX_MULT_HI = 0.30, 2.00
FU_POINTS = [
    ("star", DRUG_FU_STAR, TOX_MULT_STAR),
    ("fuD_lo_fuT_lo", DRUG_FU_LO, TOX_MULT_LO),
    ("fuD_lo_fuT_hi", DRUG_FU_LO, TOX_MULT_HI),
    ("fuD_hi_fuT_lo", DRUG_FU_HI, TOX_MULT_LO),
    ("fuD_hi_fuT_hi", DRUG_FU_HI, TOX_MULT_HI),
]


def _fallback_live_pk(vmax_scale: float) -> dict:
    """Order-of-magnitude CL when --ode is off (Table 1 CKD-5)."""
    # Secretion scales ~ with Vmax; GFR piece stays. Rough, labelled as such.
    s = vmax_scale / 0.05
    return {
        "CL_FURO": 1.70,
        "CL_R_IS": 0.20,
        "F_OAT1_IS": f_oat1_from_gfr(0.20, FU_P["IS"], BP["IS"]),
        "CL_R_HA": 0.49,
        "F_OAT1_HA": f_oat1_from_gfr(0.49, FU_P["HA"], BP["HA"]),
        "CL_R_pCS": 0.08,
        "F_OAT1_pCS": f_oat1_from_gfr(0.08, FU_P["pCS"], BP["pCS"]),
        "IS_base_blood": 28.16 * 0.70,
        "HA_base_blood": 142.86 * 0.84,
        "pCS_base_blood": 24.99 * 0.55,
        "note": "fallback Table 1; run --ode for live CL at this Vmax",
    }


def _print_row(label, toxin, ode_pct, anal_pct, css_furo, extra=""):
    if ode_pct is None:
        d_s = "   n/a"
        flag = ""
        ode_s = "  n/a"
    else:
        d = ode_pct - anal_pct
        d_s = f"{d:+6.2f}"
        flag = "  ** >2 pp **" if abs(d) > 2.0 else ""
        ode_s = f"{ode_pct:+7.2f}"
    print(f"  {label:16s} {toxin:4s}  ODE={ode_s}  anal={anal_pct:+7.2f}  "
          f"Δ={d_s}  Css_furo={css_furo:.3f} mg/L{extra}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ode", action="store_true",
                    help="run 10 chronic 30-day ODE points (slow)")
    args = ap.parse_args()

    print("=" * 72)
    print("Fig 7 / 8 corner check — analytical Css_furo = F·Dose_day/(CL·τ)")
    print("C_tox is total blood; %Δ matches plasma.")
    print("=" * 72)

    print("\n--- Ki × Vmax (Fig 7) ---")
    for label, ki, vmax in KIVMAX_POINTS:
        live = _fallback_live_pk(vmax)
        css_f = css_furo_theoretical(live["CL_FURO"])
        ode_pcts = None
        if args.ode:
            ic = _warmup_ic(vmax)
            ode_pcts = _run_chronic_oral(ic, ki, vmax)
            live["IS_base_blood"] = ic.get("C_tox1_cen")
            live["HA_base_blood"] = ic.get("C_tox2_cen")
            live["pCS_base_blood"] = ic.get("C_tox3_cen")
        for tox, cl_k, f_k, b_k in (
            ("IS", "CL_R_IS", "F_OAT1_IS", "IS_base_blood"),
            ("HA", "CL_R_HA", "F_OAT1_HA", "HA_base_blood"),
            ("pCS", "CL_R_pCS", "F_OAT1_pCS", "pCS_base_blood"),
        ):
            anal = mm_pct(
                live[b_k], live[cl_k], live[f_k], tox, css_f, ki=ki,
            )
            ode = None if ode_pcts is None else ode_pcts.get(tox)
            _print_row(label, tox, ode, anal["pct"], css_f,
                       extra=f"  Ki={ki} V={vmax}")

    print("\n--- fu_drug × fu_toxin (Fig 8) ---")
    print("  Analytical uses the same MM formula with Cu_furo ∝ fu_drug.")
    print("  ODE fu grid in heatmap_fu_sensitivity_ode.py is QD 240 mg/day;")
    print("  carry C_ven across days (do not zero at midnight).")
    for label, fu_d, mult in FU_POINTS:
        live = _fallback_live_pk(0.05)
        css_f = css_furo_theoretical(live["CL_FURO"])
        for tox, cl_k, f_k, b_k in (
            ("IS", "CL_R_IS", "F_OAT1_IS", "IS_base_blood"),
            ("pCS", "CL_R_pCS", "F_OAT1_pCS", "pCS_base_blood"),
        ):
            fu_t = FU_P[tox] * mult
            anal = mm_pct(
                live[b_k], live[cl_k], live[f_k], tox, css_f,
                ki=KI_DEFAULT, fu_p_furo=fu_d, fu_p_tox=fu_t,
            )
            _print_row(label, tox, None, anal["pct"], css_f,
                       extra=f"  fu_d={fu_d} fu_t={fu_t:.4f}")

    print("\nDecision: if every --ode point is within ~2 pp of analytical,")
    print("keep Fig 7/8 analytical maps and put this table in Supp S8.6.")
    print("Else run compute_heatmaps(scenario='chronic_oral') at 7×7.")
    if not args.ode:
        print("\nRe-run with --ode for the ten chronic ODE points.")


if __name__ == "__main__":
    main()
