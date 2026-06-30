"""
helpers.py

Small utility functions used across the Kidney PBPK project.

Design rules:
- Avoid `from constants import X` for values that might be rebound.
- Keep imports light to reduce circular-import risks.
- Be defensive: handle None / non-finite values gracefully.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------

def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _f(x, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Unit conversions
# -----------------------------------------------------------------------------

def Km_uM_to_mg_per_L(km_uM: float, mw_g_per_mol: float) -> float:
    """µM → mg/L: km_uM * MW[g/mol] * 1e-3."""
    km = _f(km_uM)
    mw = _f(mw_g_per_mol)
    return km * mw * 1e-3


def Ki_uM_to_mgL(ki_uM: float, mw_g_per_mol: float) -> float:
    """µM → mg/L."""
    ki = _f(ki_uM)
    mw = _f(mw_g_per_mol)
    return ki * mw * 1e-3


def Vmax_pmolmin_per_million_to_mgph_per_million(vmax_pmol_min_per_million: float,
                                                 mw_g_per_mol: float) -> float:
    """
    pmol/min/1e6 cells → mg/h/1e6 cells
    Factor: 1 pmol = 1e-12 mol; min→h: *60; g→mg: *1000
    => pmol/min * MW[g/mol] * 60 * 1e-12 * 1000 = pmol/min * MW * 6e-8
    """
    vmax = _f(vmax_pmol_min_per_million)
    mw = _f(mw_g_per_mol)
    return vmax * mw * 6e-8


# -----------------------------------------------------------------------------
# Acid/base ionization
# -----------------------------------------------------------------------------

def frac_unionized(drug_type: str, pKa: float, pH: float) -> float:
    """
    Unionized fraction for a monoprotic acid/base.

    acid:  fu = 1 / (1 + 10^(pH - pKa))
    base:  fu = 1 / (1 + 10^(pKa - pH))
    other/unknown: returns 1.0
    """
    t = (drug_type or "").strip().lower()
    pka = _f(pKa, default=float("nan"))
    ph = _f(pH, default=float("nan"))

    if not (_isfinite(pka) and _isfinite(ph)):
        return 1.0

    # clamp exponent to avoid overflow in extreme values
    def _pow10(x: float) -> float:
        x = max(min(x, 50.0), -50.0)  # 10^50 is plenty large; avoids inf
        return 10.0 ** x

    if t == "acid":
        return 1.0 / (1.0 + _pow10(ph - pka))
    if t == "base":
        return 1.0 / (1.0 + _pow10(pka - ph))
    return 1.0


# -----------------------------------------------------------------------------
# Surface-area helpers
# -----------------------------------------------------------------------------

def derive_baso_from_apical(
    sa_apical_dm2: Dict[str, float],
    *,
    proximal_segments: Iterable[str] = ("P1", "P2", "P3"),
    prox_baso_over_apical: float = 1.0 / 30.0
) -> Dict[str, float]:
    """
    Basolateral area map (dm²) from apical:
      - proximal: baso = apical * prox_baso_over_apical
      - others:   baso = apical
    """
    prox = set(proximal_segments)
    ratio = _f(prox_baso_over_apical, default=1.0 / 30.0)
    out: Dict[str, float] = {}

    for seg, a in (sa_apical_dm2 or {}).items():
        ap = _f(a, default=0.0)
        out[seg] = ap * (ratio if seg in prox else 1.0)

    return out


def get_sa_baso_dm2() -> Dict[str, float]:
    """
    Compute SA_BASO_DM2 using the current constants.SA_APICAL_DM2.
    This avoids a stale module-level SA_BASO_DM2 if constants is changed.
    """
    import constants as K
    return derive_baso_from_apical(
        K.SA_APICAL_DM2,
        proximal_segments=getattr(K, "PROX", ("P1", "P2", "P3")),
        prox_baso_over_apical=getattr(K, "PROX_BASO_OVER_APICAL", 1.0 / 30.0),
    )


# -----------------------------------------------------------------------------
# Drug flags + flux helpers
# -----------------------------------------------------------------------------

def drug_on() -> bool:
    """True iff compounds.drug exists and is enabled."""
    try:
        import compounds as C
        return bool(getattr(C, "drug", None)) and bool(C.drug.get("enabled", False))
    except Exception:
        return False

# How much drug (mg/h) enters the tubular cell from the peritubular capillary via basolateral transport?
def drug_baso_uptake_flux_unb(Cv_unb: float, seg: Optional[str] = None) -> float:
    """
    Drug basolateral uptake flux (mg/h) using either linear clearance or MM.

    Note: `seg` is accepted for backward compatibility (not used here).
    Cv_unb: Unbound drug concentration in the vascular compartment, Units: mg/L
    """
    try:
        import compounds as C
        import constants as K
    except Exception:
        return 0.0

    drug = getattr(C, "drug", None)
    if not isinstance(drug, dict) or not drug:
        return 0.0

    Cv = _f(Cv_unb, default=0.0)
    if Cv <= 0.0:
        return 0.0

    eps = float(getattr(K, "EPS", 1e-12))

    if (drug.get("baso_model", "mm") or "").strip().lower() == "linear":
        return _f(drug.get("CL_scr_baso", 0.0)) * Cv

    vmax = _f(drug.get("Vmax_baso_uptake", 0.0))
    km = _f(drug.get("Km_baso_uptake", 0.0))
    return (vmax * Cv) / (km + Cv + eps)
