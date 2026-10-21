# -*- coding: utf-8 -*-
"""Print t=0 vs stored pCS_base for CKD-5 chronic caches (Windows-safe)."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = (
    os.path.abspath(os.path.join(_PARENT, ".."))
    if os.path.basename(_PARENT) == "publication_code"
    else _PARENT
)
_RESULTS = os.path.join(_ROOT, "results")

BP_PCS = 0.55


def main() -> None:
    pattern = os.path.join(_RESULTS, "chronic_oral_cache_Ki0p35_dose*mg_x*pd_30days*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No caches in {_RESULTS}", file=sys.stderr)
        sys.exit(1)
    print(f"{'file':<62} {'t0_h':>8} {'pCS[0]':>8} {'pCS_base':>10} {'plasma0':>8}  note")
    print("-" * 110)
    for path in files:
        d = np.load(path, allow_pickle=False)
        t0 = float(np.asarray(d["t_all"]).flat[0])
        p0 = float(np.asarray(d["pCS_all"]).flat[0])
        base = (
            float(np.asarray(d["pCS_base"]).flat[0])
            if "pCS_base" in d.files
            else float("nan")
        )
        isb = (
            float(np.asarray(d["IS_base"]).flat[0])
            if "IS_base" in d.files
            else float("nan")
        )
        note = "CKD-5" if (8.0 <= isb <= 80.0) else "NOT CKD-5"
        if np.isfinite(base) and abs(p0 - base) > 0.5:
            note += "  t0!=base"
        print(
            f"{os.path.basename(path):<62} {t0:8.3f} {p0:8.3f} {base:10.3f} "
            f"{p0 / BP_PCS:8.2f}  {note}"
        )


if __name__ == "__main__":
    main()
