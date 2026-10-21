
"""
passive_diffusion.py

Passive diffusion calculations with Healthy/CKD pH switching.

Refactor goals
--------------
- No hidden dependence on "whatever constants.LUMEN_pH currently is".
- Update pH maps for the requested mode inside the function.
- Safer defaults and clearer naming.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from helpers import frac_unionized, derive_baso_from_apical
import constants as K


def build_passive_params(
    prefix: str,
    cpd: Mapping[str, float],
    *,
    mode: str = "healthy",
    sa_apical_dm2: Optional[Mapping[str, float]] = None,
    sa_baso_dm2: Optional[Mapping[str, float]] = None,
    proximal_segments: Sequence[str] = K.PROX,
    prox_baso_over_apical: float = K.PROX_BASO_OVER_APICAL,
    microvilli_factor_apical: float = K.MICROVILLI_FACTOR_APICAL,
) -> Dict[str, float]:
    """
    Build intrinsic (no-ionization) passive diffusion clearances per segment & membrane side.

    Notes
    -----
    - Uses compound intrinsic permeability P_app (cm/s) and surface areas (dm^2).
    - Unionized fractions depend on pH; we update constants' pH maps to the requested mode.

    Required keys in `cpd`
    ----------------------
    - "P_app": intrinsic permeability (cm/s)
    - "type": acid/base/neutral definition used by helpers.frac_unionized
    - "pKa": pKa value (float)

    Returns
    -------
    Dict[str, float] with keys like:
      {prefix}_alpha_{seg}, {prefix}_beta_{seg}, {prefix}_gamma_plasma,
      {prefix}_CL_PD_{seg}_apical_no_ui, {prefix}_CL_PD_{seg}_basolateral_no_ui,
      {prefix}_CL_PD_apical_TOTAL_no_ui, {prefix}_CL_PD_basolateral_TOTAL_no_ui,
      {prefix}_mode
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in {"healthy", "ckd5"}:
        raise ValueError("mode must be 'healthy' or 'ckd5'")

    # Ensure pH maps correspond to requested mode (safe no-op if already set)
    K.update_ph_maps(mode_key)

    if sa_apical_dm2 is None:
        sa_apical_dm2 = K.SA_APICAL_DM2
    if sa_baso_dm2 is None:
        sa_baso_dm2 = derive_baso_from_apical(
            sa_apical_dm2,
            proximal_segments=tuple(proximal_segments),
            prox_baso_over_apical=float(prox_baso_over_apical),
        )

    P_int = float(cpd["P_app"])  # cm/s

    lumen_pH = K.LUMEN_pH
    cell_pH = K.CELL_pH
    plasma_pH = K.PLASMA_pH

    out: Dict[str, float] = {}

    # Plasma unionized fraction (constant across segments)
    gamma = frac_unionized(cpd["type"], cpd["pKa"], plasma_pH)
    out[f"{prefix}_gamma_plasma"] = float(gamma)

    ap_tot = 0.0
    bs_tot = 0.0

    for seg in sa_apical_dm2.keys():
        if seg not in lumen_pH or seg not in cell_pH:
            raise KeyError(f"Segment {seg!r} missing from pH maps.")

        # Segment-specific unionized fractions
        alpha = frac_unionized(cpd["type"], cpd["pKa"], lumen_pH[seg])
        beta = frac_unionized(cpd["type"], cpd["pKa"], cell_pH[seg])

        out[f"{prefix}_alpha_{seg}"] = float(alpha)
        out[f"{prefix}_beta_{seg}"] = float(beta)

        # Convert areas and calculate clearances
        ap_cm2 = float(sa_apical_dm2[seg]) * K.DM2_TO_CM2 * float(microvilli_factor_apical)
        bs_cm2 = float(sa_baso_dm2[seg]) * K.DM2_TO_CM2

        # L/h = (cm/s)*(cm^2)*(3600 s/h)*(1 L/1000 cm^3)  -> encoded by CM_S_TO_L_H
        CL_ap_no = P_int * ap_cm2 * K.CM_S_TO_L_H
        CL_bs_no = P_int * bs_cm2 * K.CM_S_TO_L_H

        out[f"{prefix}_CL_PD_{seg}_apical_no_ui"] = float(CL_ap_no)
        out[f"{prefix}_CL_PD_{seg}_basolateral_no_ui"] = float(CL_bs_no)

        ap_tot += float(CL_ap_no)
        bs_tot += float(CL_bs_no)

    out[f"{prefix}_CL_PD_apical_TOTAL_no_ui"] = float(ap_tot)
    out[f"{prefix}_CL_PD_basolateral_TOTAL_no_ui"] = float(bs_tot)
    out[f"{prefix}_mode"] = mode_key  # stored as string for traceability

    return out
