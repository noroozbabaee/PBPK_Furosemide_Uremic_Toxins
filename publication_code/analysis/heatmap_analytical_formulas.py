# -*- coding: utf-8 -*-
"""Closed-form chronic SS used by manuscript Fig 7 / Fig 8.

The PDF builders heatmap_ki_vmax_analytical.py and
heatmap_fu_analytical_3panels.py are not in this repository. This module
is the same inhibition math as dose_response_tid_vs_qd.analytical_ss so
corner validation can be run without those missing figure scripts.

Css_furo (the “1.68 problem”)
    Css_furo = F_ORAL * dose_per_day / (CL_FURO * tau)
    Default: F=0.60, 240 mg/day, tau=24 h, CL_FURO from live ODE (~1.7 L/h)
    → Css_furo ≈ 0.60 * 240 / (1.70 * 24) ≈ 3.53 mg/L blood
    If someone used plasma CL or dropped F, Css_furo can land near 1.68.
    Print Css_furo at every validation point and do not silently substitute
    an ODE-averaged C_ven unless you call analytical_ss_ode_css.

%Δ toxin
    Cu_furo = Css_furo * fu_p,furo / BP_furo
    alpha   = 1 + Cu_furo / Ki
    s       = (Km + Cu_tox) / (alpha * Km + Cu_tox)
    CL_new  = CL_R * (1 - F_OAT1) + CL_R * F_OAT1 * s
    pct     = (CL_R / CL_new - 1) * 100
"""
from __future__ import annotations

from typing import Dict, Optional

F_ORAL = 0.60
TAU_H = 24.0
DOSE_PER_DAY = 240.0
BP_FURO = 0.82
FU_P_FURO = 0.05
KI_DEFAULT = 0.35

KM = {"IS": 3.84, "HA": 4.302, "pCS": 22.58}
FU_P = {"IS": 0.10, "HA": 0.52, "pCS": 0.05}  # HA CKD-5 Deltombe
BP = {"IS": 0.70, "HA": 0.84, "pCS": 0.55}

# Table 1 CKD-5 plasma SS used when live ODE baselines are unavailable.
# Convert to blood with * BP if you feed this as Css_tox blood.
CSS_PLASMA_TABLE1 = {"IS": 28.16, "HA": 142.86, "pCS": 24.99}


def css_furo_theoretical(cl_furo: float, dose_per_day: float = DOSE_PER_DAY,
                         f_oral: float = F_ORAL, tau: float = TAU_H) -> float:
    return f_oral * dose_per_day / (cl_furo * tau)


def mm_pct(
    css_tox_blood: float,
    cl_r: float,
    f_oat1: float,
    toxin: str,
    css_furo: float,
    ki: float = KI_DEFAULT,
    fu_p_furo: float = FU_P_FURO,
    fu_p_tox: Optional[float] = None,
) -> Dict[str, float]:
    cu_furo = css_furo * fu_p_furo / BP_FURO
    alpha = 1.0 + cu_furo / ki
    km = KM[toxin]
    fup = FU_P[toxin] if fu_p_tox is None else fu_p_tox
    bp = BP[toxin]
    cu_tox = css_tox_blood * fup / bp
    s = (km + cu_tox) / (alpha * km + cu_tox)
    cl_new = cl_r * (1.0 - f_oat1) + cl_r * f_oat1 * s
    pct = (cl_r / cl_new - 1.0) * 100.0 if cl_new > 0 else float("nan")
    return {
        "Css_furo": css_furo,
        "Cu_furo": cu_furo,
        "alpha": alpha,
        "s": s,
        "pct": pct,
        "CL_new": cl_new,
        "Cu_tox": cu_tox,
    }
