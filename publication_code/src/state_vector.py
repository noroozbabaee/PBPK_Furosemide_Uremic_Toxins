"""
State vector definition and indexing
"""
import numpy as np
from constants import SEGMENTS

# State vector definition
STATE = [
    "C_ven",
    "C_peri",
    "C_tox1_cen",
    "C_tox2_cen",
    "C_tox3_cen",
    "C_bladder",
    "Ae_drg",
    "Ae_t1",
    "Ae_t2",
    "Ae_t3"
]

# Add all segment-specific states
for seg in SEGMENTS:
    STATE += [f"C_{seg}_V", f"C_{seg}_T", f"C_{seg}_C"]
    STATE += [f"C_tox1_{seg}_V", f"C_tox1_{seg}_T", f"C_tox1_{seg}_C"]
    STATE += [f"C_tox2_{seg}_V", f"C_tox2_{seg}_T", f"C_tox2_{seg}_C"]
    STATE += [f"C_tox3_{seg}_V", f"C_tox3_{seg}_T", f"C_tox3_{seg}_C"]

# Create index mapping
IDX = {name: i for i, name in enumerate(STATE)}


def initial_conditions(
        phys: dict,
        custom_initial_conds: dict | None = None,
        *,
        custom_initial_conditions: dict | None = None,
        compounds: dict | None = None,
        scenario: dict | None = None,
        prev_end: np.ndarray | None = None,
) -> np.ndarray:
    """Build the initial state vector y0."""

    n_states = len(IDX)
    y0 = np.zeros(n_states, dtype=float)

    # Choose which dict of custom initials to use
    init_dict = custom_initial_conds if custom_initial_conds is not None else custom_initial_conditions

    # Apply carry-over if provided
    if prev_end is not None:
        if len(prev_end) != n_states:
            raise ValueError(f"prev_end length {len(prev_end)} != number of states {n_states}")
        y0[:] = prev_end
        return y0

    # Apply custom initials if provided
    if init_dict:
        for name, val in init_dict.items():
            if name in IDX:
                y0[IDX[name]] = float(val)
            else:
                print(f"Warning: Unknown state '{name}' in custom initial conditions")

    # Set C_peri to 0 if not already set
    if "C_peri" in IDX and (init_dict is None or "C_peri" not in init_dict):
        y0[IDX["C_peri"]] = 0.0

    # Check if we have compounds
    try:
        import compounds as C
        drug = C.drug
        tox1 = C.tox1
        tox2 = C.tox2
        tox3 = C.tox3
    except (ImportError, AttributeError):
        # No compounds module
        drug = {}
        tox1 = tox2 = tox3 = {}

    # Set drug bolus if applicable
    if "C_ven" in IDX and (init_dict is None or "C_ven" not in init_dict):
        bolus = float(drug.get("bolus_mg", 0.0))
        drug_enabled = bool(drug.get("enabled", False))

        if drug_enabled and bolus > 0.0:
            V_ven0 = float(phys.get("V_ven0", 5.0))
            y0[IDX["C_ven"]] = bolus / V_ven0
            print(f"DEBUG: Drug bolus = {bolus} mg, C_ven = {y0[IDX['C_ven']]:.6f} mg/L")

    # Set toxin initial conditions
    for tox_name, tox_dict in [("C_tox1_cen", tox1), ("C_tox2_cen", tox2), ("C_tox3_cen", tox3)]:
        if tox_name in IDX and (init_dict is None or tox_name not in init_dict):
            y0[IDX[tox_name]] = float(tox_dict.get("C0", 0.0))

    # ── Tubular vascular compartment initialisation ───────────────────────────
    #
    # WHY THIS IS NEEDED
    # ------------------
    # The renal_exchange term in the central ODE is:
    #   (Qkid/Vcen) * (C_ren_exit - C_ven)
    # where C_ren_exit is the advection-chain exit from the last vascular
    # segment (CDC5_V). If all tubular vascular compartments start at zero
    # while C_ven and C_tox*_cen are non-zero (from C0 or a bolus), the
    # advection chain sees a massive concentration gradient on the first ODE
    # call, draining the central compartment at rate Qkid * C_cen / Vcen —
    # which for Qkid=72 L/h, Vcen=5 L gives a time constant of 5/72 = 0.07 h.
    # This is an unphysical transient that collapses CKD t½ from ~2.8h to 0.5h.
    #
    # THE FIX
    # -------
    # Initialise every tubular vascular compartment to the post-glomerular
    # peritubular concentration:
    #   C_seg_V = C_systemic * (1 - GFR/Qkid * fu_p/BP)
    # This is the mass-balance result at the glomerulus: the peritubular blood
    # entering P1 carries the systemic concentration minus what was filtered.
    # At this starting point, C_ren_exit ≈ C_ven and renal_exchange ≈ 0 at t=0.
    #
    # For the drug: C_ven is typically 0 at t=0 of a PK run (infusion not yet
    # started), so this has no effect on drug compartments. It matters for
    # toxins, whose C0 > 0 creates the problematic initial gradient.
    #
    # GUARD: only apply this when custom_initial_conds does NOT already
    # specify the vascular compartment values (e.g. warm-up carry-over),
    # and only when a C_systemic value is available (non-zero C0).
    # ─────────────────────────────────────────────────────────────────────────
    if init_dict is None:
        # Only auto-initialise when no custom ICs were supplied (fresh start)
        GFR   = float(phys.get("Q_GFR",      7.2))
        Qkid  = float(phys.get("Q_kidney0",  72.0))
        EPS   = 1e-12

        # Pairs: (systemic concentration value, fu_p, BP, vascular state prefix)
        _cmpd_vascular = []

        # Drug
        C_ven_val = float(y0[IDX.get("C_ven", 0)])
        if C_ven_val > 0.0:
            fu_p_d = float(drug.get("fu_p", 0.02))
            BP_d   = float(drug.get("BP",   0.82))
            C_in_d = C_ven_val * (1.0 - (GFR / max(Qkid, EPS)) * (fu_p_d / max(BP_d, EPS)))
            _cmpd_vascular.append(("C", None, C_in_d))

        # Toxins: always initialise since C0 > 0
        for prefix, tox_dict, cen_key in (
            ("C_tox1", tox1, "C_tox1_cen"),
            ("C_tox2", tox2, "C_tox2_cen"),
            ("C_tox3", tox3, "C_tox3_cen"),
        ):
            C_cen = float(y0[IDX.get(cen_key, 0)])
            if C_cen > 0.0:
                fu_p_t = float(tox_dict.get("fu_p", 0.04))
                BP_t   = float(tox_dict.get("BP",   0.70))
                C_in_t = C_cen * (1.0 - (GFR / max(Qkid, EPS)) * (fu_p_t / max(BP_t, EPS)))
                C_in_t = max(C_in_t, 0.0)   # clamp — cannot be negative
                _cmpd_vascular.append((prefix, None, C_in_t))

        # Set every tubular vascular compartment to C_in for that compound.
        # C_in is a conservative lower bound — the true peritubular concentration
        # varies along the nephron, but uniform initialisation eliminates the
        # zero-concentration shock without requiring segment-resolved steady state.
        for seg in SEGMENTS:
            # Drug vascular
            if len(_cmpd_vascular) > 0:
                C_in_d = _cmpd_vascular[0][2] if _cmpd_vascular[0][0] == "C" else 0.0
                vkey_d = f"C_{seg}_V"
                if vkey_d in IDX and C_in_d > 0.0:
                    y0[IDX[vkey_d]] = C_in_d

            # Toxin vascular compartments
            for prefix, _, C_in_t in _cmpd_vascular:
                if prefix.startswith("C_tox"):
                    vkey_t = f"{prefix}_{seg}_V"
                    if vkey_t in IDX and C_in_t > 0.0:
                        y0[IDX[vkey_t]] = C_in_t

        if len(_cmpd_vascular) > 0:
            print(
                f"[IC] Tubular vascular compartments initialised to post-glomerular "
                f"equilibrium (prevents renal_exchange transient at t=0)."
            )
            for prefix, _, C_in in _cmpd_vascular:
                print(f"  {prefix}: C_in = {C_in:.4e} mg/L  (all segments)")

    return y0


def get_initial_conditions_dict(mode: str = "healthy") -> dict:
    """Get the default initial conditions as a dictionary for the current mode."""
    from compounds import get_compound_parameters
    drug_params, tox1_params, tox2_params, tox3_params = get_compound_parameters(mode)

    return {
        "C_ven": 0.0,
        "C_peri": 0.0,
        "C_tox1_cen": tox1_params["C0"],
        "C_tox2_cen": tox2_params["C0"],
        "C_tox3_cen": tox3_params["C0"],
        "C_bladder": 0.0,
        "Ae_drg": 0.0,
        "Ae_t1": 0.0,
        "Ae_t2": 0.0,
        "Ae_t3": 0.0
        # All segment concentrations default to 0.0 (implicitly)
    }


# Optional: Print confirmation (remove after verification)
print(f"State vector initialized with {len(STATE)} states")
print(f"  C_ven index: {IDX['C_ven']}")
print(f"  C_peri index: {IDX['C_peri']}")
print(f"  C_tox1_cen index: {IDX['C_tox1_cen']}")