# -*- coding: utf-8 -*-
"""Audit local chronic NPZs vs the paper CKD-5 caches (Windows CMD).

PASS for the three-arm comparison:
  IS_base in 8–80 mg/L blood (paper ~19.7; NEVER ~0.54)
  filename contains ckd5
  40x3, 240x1, 80x3 present

  pbpk_env\\Scripts\\python.exe analysis\\inspect_chronic_npz.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
FOLDERS = [
    os.path.join(_ROOT, "results"),
    os.path.join(_ROOT, "publication_code", "results"),
]

PAPER = {
    "IS_base": (18.0, 22.0, 19.712),
    "pCS_ok": (10.0, 20.0),
}


def _scalar(z, key):
    if key not in getattr(z, "files", ()):
        return None
    return float(np.asarray(z[key]).reshape(-1)[0])


def main() -> None:
    files = []
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith(".npz"):
                files.append(os.path.join(folder, name))
    files.sort()
    if not files:
        print("No .npz under results\\ or publication_code\\results\\")
        print("Put chronic_oral_cache_Ki0p35_dose*mg_x*pd_30days_ckd5.npz in results\\")
        sys.exit(1)

    print(f"{'file':<72} {'IS_b':>8} {'pCS_b':>8} {'HA_b':>8} {'CL_furo':>8}  verdict")
    print("-" * 120)
    needed = {"40x3": False, "240x1": False, "80x3": False}
    needed_ha = {"40x3": False, "240x1": False, "80x3": False}
    for path in files:
        name = os.path.basename(path)
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as exc:
            print(f"{name:<72}  LOAD FAIL {exc}")
            continue
        files_keys = set(z.files)
        isb = _scalar(z, "IS_base")
        pcs = _scalar(z, "pCS_base")
        ha = _scalar(z, "HA_base")
        cl = _scalar(z, "CL_FURO")
        ha_all = "HA_all" in files_keys
        ha_n = int(np.asarray(z["HA_all"]).size) if ha_all else 0
        if ha is None and ha_all:
            ha = float(np.asarray(z["HA_all"]).reshape(-1)[0])
        notes = []
        if not ha_all:
            notes.append("NO HA_all")
        elif ha_n < 10:
            notes.append(f"HA_all n={ha_n} (too short)")
        else:
            notes.append(f"HA_all n={ha_n}")
        if "HA_base" not in files_keys:
            notes.append("no HA_base key")
        if isb is None:
            notes.append("no IS_base")
        elif isb <= 2.0:
            notes.append("HEALTHY/acute (~0.54) — REJECT")
        elif 8.0 <= isb <= 80.0:
            notes.append("CKD-5 IS")
            if 18.0 <= isb <= 22.0:
                notes.append("paper IS~19.7")
        else:
            notes.append("IS_base odd")
        if "ckd5" not in name.lower():
            notes.append("filename missing ckd5")
        nl = name.lower()
        ckd_ok = 8.0 <= (isb or 0) <= 80.0
        ha_ok = ckd_ok and ha_all and ha_n >= 10
        if "dose40mg" in nl and "x3" in nl:
            needed["40x3"] = needed["40x3"] or ckd_ok
            needed_ha["40x3"] = needed_ha["40x3"] or ha_ok
        if "dose240mg" in nl and "x1" in nl:
            needed["240x1"] = needed["240x1"] or ckd_ok
            needed_ha["240x1"] = needed_ha["240x1"] or ha_ok
        if "dose80mg" in nl and "x3" in nl:
            needed["80x3"] = needed["80x3"] or ckd_ok
            needed_ha["80x3"] = needed_ha["80x3"] or ha_ok
        def f(v):
            return f"{v:8.3f}" if v is not None else f"{'—':>8}"
        print(
            f"{name:<72} {f(isb)} {f(pcs)} {f(ha)} {f(cl)}  {'; '.join(notes)}"
        )

    print("-" * 120)
    print("Three-arm CKD-5 IS_base (must all be True):", needed)
    print("Three-arm with HA_all (required for Fig 5):", needed_ha)
    if all(needed_ha.values()):
        print("PASS: 40x3, 240x1, 80x3 CKD-5 caches with HA_all.")
    elif all(needed.values()):
        print("PARTIAL: three CKD-5 IS/pCS arms exist, but HA_all is missing.")
        print("Do not rename results_Ki0p35_*.npz — regenerate chronic_oral_cache_*")
        print("with: python analysis/regenerate_chronic_caches.py")
        sys.exit(3)
    else:
        print("FAIL: missing or healthy caches for",
              [k for k, v in needed.items() if not v])
        sys.exit(2)


if __name__ == "__main__":
    main()
