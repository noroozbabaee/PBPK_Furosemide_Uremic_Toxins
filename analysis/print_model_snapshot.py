# -*- coding: utf-8 -*-
"""Print live compound + CKD scales (Windows CMD: one file, no python -c).

  cd /d ...\\PBPK_Furosemide_Uremic_Toxins_pythoncode
  pbpk_env\\Scripts\\python.exe analysis\\print_model_snapshot.py
"""
from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)

import constants as K  # noqa: E402
K.VERBOSE = False
import compound_parameters as CP  # noqa: E402


def _row(name, h, c, keys):
    print(f"\n--- {name} ---")
    print(f"  {'parameter':<22} {'healthy':>12} {'CKD-5':>12}")
    for k in keys:
        hv, cv = h.get(k), c.get(k)
        if hv is None and cv is None:
            continue
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:12.4g}"
            return f"{v!s:>12}"
        print(f"  {k:<22} {fmt(hv)} {fmt(cv)}")


def main() -> None:
    print("FILE", os.path.abspath(CP.__file__))
    print("Ki (all toxins)     ", CP.TOX1_BASE.get("Ki_furo_to_IS_mgL"))
    print("pCS pKa             ", CP.TOX3_BASE.get("pKa"))
    print("furosemide BP / fu_p CKD", CP.DRUG_BASE.get("BP"),
          CP.get_compound_parameters("ckd5")[0].get("fu_p"))
    sc = CP._CKD_SCALES
    print("CKD Vmax scales     drug baso/apical",
          sc["drug"]["baso"], sc["drug"]["apical"],
          " tox baso/apical", sc["tox"]["baso"], sc["tox"]["apical"])
    try:
        print("GFR_scale / RBF_scale",
              K.CKD_SCALE_FACTORS.get("GFR_scale"),
              K.CKD_SCALE_FACTORS.get("RBF_scale"))
    except Exception:
        pass

    dh, ISh, HAh, pCSh = CP.get_compound_parameters("healthy")
    dc, ISc, HAc, pCSc = CP.get_compound_parameters("ckd5")
    keys = (
        "fu_p", "BP", "pKa", "C0", "R0",
        "Km_baso_uptake", "Vmax_baso_uptake", "Vmax_api_secr",
        "Ki_furo_to_IS_mgL", "Ki_furo_to_HA_mgL", "Ki_furo_to_pCS_mgL",
        "k_elim_hinv",
    )
    _row("FUROSEMIDE", dh, dc, keys)
    _row("IS", ISh, ISc, keys)
    _row("HA", HAh, HAc, keys)
    _row("pCS", pCSh, pCSc, keys)
    print("\nC_ven in ODE = total blood. Plasma = blood / BP.")
    print("Unbound driving conc. Cu = C_ven * fu_p / BP.")


if __name__ == "__main__":
    main()
