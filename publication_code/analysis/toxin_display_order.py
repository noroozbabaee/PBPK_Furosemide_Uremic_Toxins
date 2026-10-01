"""Manuscript PBUT order: IS, then HA, then pCS last.

ODE identity is unchanged: tox1=IS, tox2=HA, tox3=pCS.
Never swap this on a figure x-axis or panel row.

Canonical copy: analysis/toxin_display_order.py — keep both identical.
"""

TOXIN_DISPLAY_ORDER = ("IS", "HA", "pCS")

TOXIN_STATE_CEN = {
    "IS": "C_tox1_cen",
    "HA": "C_tox2_cen",
    "pCS": "C_tox3_cen",
}
TOXIN_AE = {
    "IS": "Ae_t1",
    "HA": "Ae_t2",
    "pCS": "Ae_t3",
}


def toxin_states_for_display():
    return [TOXIN_STATE_CEN[t] for t in TOXIN_DISPLAY_ORDER]


def toxin_ae_for_display():
    return [TOXIN_AE[t] for t in TOXIN_DISPLAY_ORDER]


def assert_toxin_display_order(species):
    tox_only = tuple(s for s in species if s in TOXIN_DISPLAY_ORDER)
    if not tox_only:
        return
    expected = tuple(t for t in TOXIN_DISPLAY_ORDER if t in tox_only)
    if tox_only != expected:
        raise ValueError(
            f"PBUT order must be IS, HA, pCS (pCS last). Got {species!r}."
        )
