# -*- coding: utf-8 -*-
"""Closed-form chronic SS (Fig 7 / Fig 8).

Css_furo is **plasma**, because extract_baseline_pk defines
    CL_FURO = Ae / ∫(C_ven / BP) dt
so Css_plasma = F · Dose_day / (CL_FURO · τ).

Unbound driving conc. (same as the ODE):
    Cu_furo = Css_plasma · fu_p          # do NOT divide by BP again
    Cu_tox  = C_tox_blood · fu_p / BP    # C_tox*_cen is blood

F_OAT1 from filtration vs total plasma CL_R:
    CL_GFR  = Q_GFR · fu_p / BP
    F_OAT1  = 1 − CL_GFR / CL_R
CKD-5 Q_GFR ≈ 7.2 × 0.10 = 0.72 L/h (not 0.1 L/h).
"""
from __future__ import annotations

from typing import Dict, Optional

F_ORAL = 0.60
TAU_H = 24.0
DOSE_PER_DAY = 240.0
BP_FURO = 0.82
FU_P_FURO = 0.05
KI_DEFAULT = 0.35
# Healthy GFR 7.2 L/h × CKD tubular/GFR scale 0.10
Q_GFR_CKD5_DEFAULT = 7.2 * 0.10

KM = {"IS": 3.84, "HA": 4.302, "pCS": 22.58}
FU_P = {"IS": 0.10, "HA": 0.52, "pCS": 0.05}
BP = {"IS": 0.70, "HA": 0.84, "pCS": 0.55}
CSS_PLASMA_TABLE1 = {"IS": 28.16, "HA": 142.86, "pCS": 24.99}
CL_R_TABLE1 = {"IS": 0.20, "HA": 0.49, "pCS": 0.08}


def q_gfr_ckd5() -> float:
    try:
        import physiology as phys
        d = phys.make_phys("ckd5")
        q = float(d.get("Q_GFR", 0.0))
        if q > 0:
            return q
    except Exception:
        pass
    return Q_GFR_CKD5_DEFAULT


def f_oat1_from_gfr(cl_r: float, fu_p: float, bp: float,
                    q_gfr: Optional[float] = None) -> float:
    q = Q_GFR_CKD5_DEFAULT if q_gfr is None else q_gfr
    cl_gfr = q * fu_p / bp
    if cl_r <= 1e-12:
        return 0.0
    return float(max(0.0, min(0.99, 1.0 - cl_gfr / cl_r)))


def css_furo_theoretical(cl_furo_plasma: float,
                         dose_per_day: float = DOSE_PER_DAY,
                         f_oral: float = F_ORAL, tau: float = TAU_H) -> float:
    """Plasma Css (mg/L). CL must be plasma CL (Ae / AUC_plasma)."""
    return f_oral * dose_per_day / (cl_furo_plasma * tau)


def mm_pct(
    css_tox_blood: float,
    cl_r: float,
    f_oat1: float,
    toxin: str,
    css_furo_plasma: float,
    ki: float = KI_DEFAULT,
    fu_p_furo: float = FU_P_FURO,
    fu_p_tox: Optional[float] = None,
) -> Dict[str, float]:
    cu_furo = css_furo_plasma * fu_p_furo
    alpha = 1.0 + cu_furo / ki
    km = KM[toxin]
    fup = FU_P[toxin] if fu_p_tox is None else fu_p_tox
    bp = BP[toxin]
    cu_tox = css_tox_blood * fup / bp
    s = (km + cu_tox) / (alpha * km + cu_tox)
    cl_new = cl_r * (1.0 - f_oat1) + cl_r * f_oat1 * s
    pct = (cl_r / cl_new - 1.0) * 100.0 if cl_new > 0 else float("nan")
    return {
        "Css_furo": css_furo_plasma,
        "Cu_furo": cu_furo,
        "alpha": alpha,
        "s": s,
        "pct": pct,
        "CL_new": cl_new,
        "Cu_tox": cu_tox,
        "F_OAT1": f_oat1,
    }
