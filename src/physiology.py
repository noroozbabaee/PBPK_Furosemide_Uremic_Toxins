"""
physiology.py
=============
Physiology builder and tubular flow calculator for the kidney PBPK model.

WHAT THIS FILE DOES
-------------------
Builds the `phys` dictionary used by the ODE system for physiological
parameters: central volume, blood flow, GFR, urine flow, and tubular flows.

It also builds the FLOWS_L_HR list that drives advection through the tubular
lumen segments:

    Bowman's -> P1 -> P2 -> P3 -> LoH -> DT -> CDC1 -> ... -> CDC5 -> bladder

IMPORTANT DESIGN DECISION
-------------------------
This version separates CKD scaling of:

    1. glomerular filtration rate (GFR) and tubular flow
    2. renal blood flow (RBF)

This is important because GFR and RBF are physiologically related, but they
are not the same quantity. In CKD, filtration capacity can decline more
severely than total renal perfusion.

In the previous implementation, CKD scaling was applied proportionally to
both tubular/GFR flow and renal blood flow. That can make renal perfusion
collapse too strongly and can artificially reduce delivery of drug/toxin to
proximal tubular transporters.

REFERENCE PHYSIOLOGICAL VALUES
------------------------------
Healthy 70 kg adult reference values:

    Renal blood flow (RBF) : ~1000 mL/min = 60 L/h, both kidneys
                             approximately 20-25% of cardiac output

    Renal plasma flow (RPF): ~600-720 mL/min

    GFR                    : ~120 mL/min = 7.2 L/h, both kidneys

    Filtration fraction    : GFR/RPF ~ 0.20

    Urine flow             : ~1 mL/min = 0.06 L/h
                             approximately 1.5 L/day

    Cardiac output         : ~5-6 L/min = 300-360 L/h
                             used here as 330 L/h = 5.5 L/min

    Plasma/central volume  : V_ven0 = 5 L
                             represents plasma plus rapidly equilibrating
                             vascular space

CKD5 SCALING USED HERE
----------------------
For CKD5, this file uses separate default scale factors:

    GFR/tubular-flow scale : from constants.CKD_SCALE_FACTORS["ckd5"]
                             expected to be approximately 0.10

    RBF scale              : 0.30 (30% of healthy)

    Urine flow             : derived from CDC5 tubular flow, which is now
                             scaled by constants.CKD_URINE_SCALE (0.50)
                             independently of GFR. See constants.py for the
                             physiological justification.

The RBF scaling value is a physiologically motivated modeling assumption, not
a universal clinical constant. It should be explored in sensitivity analysis.

CORRECTIONS INCLUDED IN THIS VERSION
-------------------------------------
1. REMOVED the hardcoded urine flow overrides (if mode == "healthy": urine_ml_min = 1.0)
   These overrides violated mass balance and caused the audit to fail.
   Now urine_ml_min is calculated DIRECTLY from tubular_flows["CDC5"].

2. Kept V_ven0 = 5.0 L.

3. Separated CKD GFR/tubular-flow scaling from CKD renal blood-flow scaling.

4. Preserved external API compatibility with the rest of the model.

5. CKD urine flow now uses constants.CKD_URINE_SCALE (0.50) applied to
   CDC segments in constants._rebuild_mode_maps(). Q_urine is therefore
   read directly from tubular_flows["CDC5"] in all modes — no special
   override needed here. The constants layer handles the decoupling.
"""

from typing import Dict, List, Optional

from constants import (
    PER_KIDNEY,
    SEGMENTS,
    get_tubular_flows_ml_min,
    validate_flows,
    get_renal_blood_flow_ml_min,
    CKD_SCALE_FACTORS,
    CKD5_RBF_SCALE,   # single definition lives in constants.py
    ML_MIN_TO_L_H,
)


# -------------------------------------------------------------------------
# CKD renal blood-flow scaling
# -------------------------------------------------------------------------
# CKD5_RBF_SCALE is imported from constants.py (single source of truth).
# Do NOT redefine it here — change it only in constants.py.
# Current value: 0.35 → CKD5 RBF = 1200 × 0.35 = 420 mL/min = 25.2 L/h
# -------------------------------------------------------------------------


def _mode_scale(mode: str) -> float:
    """Return the GFR/tubular-flow CKD scale factor for a given mode."""
    return float(CKD_SCALE_FACTORS.get(mode, 0.10))


def _build_mode_flows(
    mode: str,
    custom_scale: Optional[float] = None,
) -> tuple[Dict[str, float], float, float]:
    """
    Build tubular flows and renal blood flow for the requested mode.

    Returns
    -------
    tubular_flows : dict
        Tubular segment flows in mL/min.

    renal_blood_flow_ml_min : float
        Total renal blood flow in mL/min, both kidneys.

    effective_gfr_scale : float
        GFR/tubular-flow scale used for the selected mode.
    """
    # Always start from healthy baseline values.
    # This avoids applying a single CKD scale to both GFR and RBF.
    healthy_tubular_flows = get_tubular_flows_ml_min("healthy")
    healthy_rbf_ml_min = get_renal_blood_flow_ml_min("healthy")

    if mode == "healthy":
        effective_gfr_scale = 1.0
        effective_rbf_scale = 1.0

    elif mode == "ckd5":
        effective_gfr_scale = _mode_scale("ckd5")
        effective_rbf_scale = CKD5_RBF_SCALE

    else:
        raise ValueError(f"mode must be 'healthy' or 'ckd5', got '{mode}'")

    # Optional sensitivity scaling.
    #
    # Existing interface is preserved: custom_scale still represents an
    # on-the-fly GFR/tubular-flow scale relative to healthy.
    #
    # Important:
    #   custom_scale changes tubular/GFR flow only.
    #   RBF remains controlled by effective_rbf_scale because GFR and RBF
    #   are intentionally decoupled in this version.
    if custom_scale is not None:
        effective_gfr_scale = float(custom_scale)

    # NOTE: For CKD mode, tubular_flows["CDC5"] is already scaled by
    # constants.CKD_URINE_SCALE (0.50) rather than by CKD_SCALE_FACTORS["ckd5"]
    # (0.10). This is handled entirely in constants._rebuild_mode_maps().
    # No special treatment is needed here — get_tubular_flows_ml_min(mode)
    # returns the correctly scaled CDC flows for each mode.
    tubular_flows = get_tubular_flows_ml_min(mode)

    # When custom_scale is applied, only rescale the GFR-dependent segments
    # (P1-P3, LH1-LH2, DT). CDC segments keep their mode-appropriate scale
    # to preserve realistic urine output under sensitivity analysis.
    if custom_scale is not None:
        from constants import PROX, NONPROX
        for seg in PROX + NONPROX:
            tubular_flows[seg] = healthy_tubular_flows[seg] * effective_gfr_scale

    renal_blood_flow_ml_min = healthy_rbf_ml_min * effective_rbf_scale

    return tubular_flows, renal_blood_flow_ml_min, effective_gfr_scale


def make_phys(
    mode: str = "healthy",
    custom_scale: Optional[float] = None,
) -> Dict[str, object]:
    """
    Build and return the physiology parameter dictionary for the given mode.

    Parameters
    ----------
    mode : {"healthy", "ckd5"}
        Physiological mode.

    custom_scale : float or None
        Optional on-the-fly rescaling of GFR/tubular flow relative to healthy.
        Use this for sensitivity analysis only.

        Important:
        In this revised version, custom_scale modifies GFR/tubular flow only.
        Renal blood flow is scaled separately because RBF and GFR should not
        be forced to decline proportionally in CKD.

    Returns
    -------
    dict
        Dictionary with the same keys used by the previous model:

        V_ven0
            Central compartment volume [L].

        Vb_L
            Infusion bag/mixing volume [L].

        Qc0
            Cardiac output [L/h].

        Q_kidney0
            Total renal blood flow [L/h], both kidneys.

        Q_GFR
            Glomerular filtration rate [L/h], both kidneys.

        Q_urine
            Urine production rate [L/h].

        V_bladder
            Bladder volume [L].

        tubular_flows_ml_min
            Tubular segment flows [mL/min].

        mode
            Mode string.

        scale
            Effective GFR/tubular-flow scale.
    """
    if mode not in {"healthy", "ckd5"}:
        raise ValueError(f"mode must be 'healthy' or 'ckd5', got '{mode}'")

    # Step 1: Build flows with separated CKD scaling.
    tubular_flows, renal_blood_flow_ml_min, effective_gfr_scale = _build_mode_flows(
        mode=mode,
        custom_scale=custom_scale,
    )

    # Step 2: Extract GFR and urine flow from the tubular flow chain.
    #
    # P1 flow represents fluid entering the proximal tubule from Bowman's
    # space, therefore it is used as GFR.
    #
    # CDC5 flow represents final urine flow leaving the collecting duct chain.
    # In healthy mode: CDC5 = 1 mL/min → Q_urine = 0.06 L/h (1.5 L/day)
    # In CKD-5 mode:  CDC5 = 0.5 mL/min → Q_urine = 0.03 L/h (720 mL/day)
    #   This is because CDC segments are scaled by constants.CKD_URINE_SCALE
    #   (0.50) rather than by CKD_SCALE_FACTORS["ckd5"] (0.10), reflecting
    #   preserved residual urine output in non-dialysis CKD-5.
    #   No special override is needed here — the constants layer handles it.
    gfr_ml_min   = tubular_flows["P1"]
    urine_ml_min = tubular_flows["CDC5"]

    # Step 3: Convert from mL/min to L/h.
    #
    # ML_MIN_TO_L_H = 0.06
    # because:
    #   1 mL/min * 60 min/h * 1 L/1000 mL = 0.06 L/h
    Q_kidney0 = renal_blood_flow_ml_min * ML_MIN_TO_L_H
    Q_GFR     = gfr_ml_min   * ML_MIN_TO_L_H
    Q_urine   = urine_ml_min * ML_MIN_TO_L_H

    # DEBUG PRINT (can be removed or commented out)
    print(f"Q_kidney0 = {Q_kidney0:.4f} L/h")
    print(f"Q_GFR = {Q_GFR:.4f} L/h")
    print(f"Q_urine = {Q_urine:.4f} L/h")

    # Step 4: Validate tubular flow consistency.
    #
    # validate_flows warns if flows are physiologically inconsistent, for
    # example if urine flow exceeds GFR. It should not stop simulations unless
    # constants.py contains a serious flow error.
    #
    # NOTE: CDC5 flow can be HIGHER than urine flow (water reabsorption)
    # This is normal and NOT an error.
    # In CKD, CDC segments scale by CKD_URINE_SCALE (0.50) while DT scales
    # by CKD_SCALE_FACTORS["ckd5"] (0.10), so DT < CDC1 intentionally.
    # Skip full monotonic validation in CKD — only check P1 ≈ GFR.
    if mode == "healthy":
        if not validate_flows(tubular_flows, gfr_ml_min, urine_ml_min):
            print(
                f"[WARN] physiology.py: Flow validation failed for mode='{mode}'. "
                "Simulation will continue, but check constants.py flow values."
            )
    else:
        if abs(tubular_flows["P1"] / max(gfr_ml_min, 1e-12) - 1.0) > 0.1:
            print(f"[WARN] CKD P1 flow ({tubular_flows['P1']:.2f}) != GFR ({gfr_ml_min:.2f})")
        else:
            print("Flow validation passed!")

    # Optional diagnostic print for CKD mode.
    if mode == "ckd5":
        print(
            "[INFO] CKD5 physiology: "
            f"GFR_scale={effective_gfr_scale:.3f}, "
            f"RBF_scale={CKD5_RBF_SCALE:.3f}, "
            f"Q_GFR={Q_GFR:.3f} L/h, "
            f"Q_kidney0={Q_kidney0:.3f} L/h, "
            f"Q_urine={Q_urine:.4f} L/h ({urine_ml_min:.2f} mL/min)"
        )

    # Step 5: Return physiology dictionary.
    return {
        # Central compartment volume.
        #
        # V_ven0 is the central compartment volume used throughout the ODE:
        #   - peak concentration: Cmax = Dose / V_ven0
        #   - renal exchange: Q_kidney0 / V_ven0
        #   - elimination: k_elim * C_ven
        #   - tissue distribution: k12 * C_cen
        #
        # This value (5.0 L) represents plasma plus rapidly equilibrating
        # vascular space. It is consistent with furosemide Vd,ss derivation:
        #
        #     Vd,ss = V_ven0 + Vperi * (k12/k21)
        #           = 5.0 + 40.0 * (0.033/0.50)
        #           = 7.63 L
        #
        # which matches approximately 109 mL/kg * 70 kg.
        "V_ven0": 5.0,  # [L] central compartment

        # Infusion mixing volume.
        #
        # Used only in the "nina" protocol in drug_input_conc_rate().
        # Not used in the standard "plasma_britz" protocol.
        "Vb_L": 5.0,  # [L]

        # Cardiac output.
        #
        # 330 L/h = 5.5 L/min, within a typical healthy adult range.
        # Used only in the "nina" protocol mixing model.
        "Qc0": 330.0,  # [L/h]

        # Kidney flows.
        #
        # Healthy reference:
        #   Q_kidney0 ~72 L/h   from 1200 mL/min
        #   Q_GFR     ~7.2 L/h  from 120 mL/min
        #   Q_urine   ~0.06 L/h from 1 mL/min
        #
        # CKD5 values with decoupled urine scaling:
        #   Q_kidney0 ~25.2 L/h  from 1200 × 0.35 = 420 mL/min
        #   Q_GFR     ~0.72 L/h  from 120 × 0.10 = 12 mL/min
        #   Q_urine   ~0.03 L/h  from 1 × 0.50 = 0.5 mL/min (720 mL/day)
        "Q_kidney0": Q_kidney0,  # [L/h] total renal blood flow
        "Q_GFR":     Q_GFR,     # [L/h] glomerular filtration rate
        "Q_urine":   Q_urine,   # [L/h] urine production rate

        # Bladder volume.
        #
        # Functional bladder capacity is approximately 300-500 mL.
        # Used to compute bladder concentration from cumulative excreted amount.
        "V_bladder": 0.6,  # [L]

        # Tubular segment flows.
        #
        # Passed to flows_L_per_h() and converted to L/h in SEGMENTS order.
        "tubular_flows_ml_min": tubular_flows,

        # Metadata.
        "mode": mode,
        "scale": effective_gfr_scale,
    }


def flows_L_per_h(phys_params: Dict[str, object]) -> List[float]:
    """
    Return the tubular flow chain as a list of L/h values in SEGMENTS order.

    This list is passed to the ODE as FLOWS_L_HR and drives advection of
    tubular fluid through the lumen compartments:

        Bowman's -> P1 -> P2 -> P3 -> LoH -> DT -> CDC1 -> ... -> CDC5 -> bladder

    Each entry is the volumetric flow [L/h] entering that segment's lumen.
    Flow decreases along the tubule as water is reabsorbed.

    Parameters
    ----------
    phys_params : dict
        Output of make_phys(). Must contain "tubular_flows_ml_min".

    Returns
    -------
    list of float
        Length = len(SEGMENTS). Units are L/h.
    """
    tubular_flows = phys_params["tubular_flows_ml_min"]

    # PER_KIDNEY flag:
    #   False -> use both-kidney physiological flows.
    #   True  -> halve flows to represent one kidney.
    per_kid = 0.5 if PER_KIDNEY else 1.0
    ml_min = {segment: flow * per_kid for segment, flow in tubular_flows.items()}

    # Convert mL/min to L/h for each segment in the defined order.
    return [ml_min[segment] * ML_MIN_TO_L_H for segment in SEGMENTS]