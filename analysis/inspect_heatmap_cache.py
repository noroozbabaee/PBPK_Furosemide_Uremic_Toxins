# -*- coding: utf-8 -*-
"""Print what a local heatmap pickle actually contains (Windows-safe).

  pbpk_env\\Scripts\\python.exe analysis\\inspect_heatmap_cache.py
  pbpk_env\\Scripts\\python.exe analysis\\inspect_heatmap_cache.py results\\heatmap_cache_correct.pkl
"""
from __future__ import annotations

import os
import sys
import pickle

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_RESULTS = os.path.join(_ROOT, "results")

CANDIDATES = [
    os.path.join(_RESULTS, "heatmap_cache_correct.pkl"),
    os.path.join(_RESULTS, "heatmap_ki_vmax_engine_cache.pkl"),
    os.path.join(_ROOT, "publication_code", "results", "heatmap_cache_correct.pkl"),
]


def _paths():
    if len(sys.argv) > 1:
        return [os.path.abspath(sys.argv[1])]
    found = [p for p in CANDIDATES if os.path.isfile(p)]
    extra = []
    if os.path.isdir(_RESULTS):
        for name in os.listdir(_RESULTS):
            if name.endswith(".pkl") and "heatmap" in name.lower():
                extra.append(os.path.join(_RESULTS, name))
    out = []
    for p in found + extra:
        if p not in out:
            out.append(p)
    return out


def _report(path: str) -> None:
    print("=" * 64)
    print(path)
    print("  size_MB =", round(os.path.getsize(path) / 1e6, 3))
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print("  python_type =", type(obj).__name__)
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("  n_entries =", len(keys))
        print("  first_keys:")
        for k in keys[:8]:
            print("   ", repr(k)[:120])
        sample = obj[keys[0]] if keys else None
        print("  first_value_type =", type(sample).__name__)
        if isinstance(sample, dict):
            print("  first_value_keys =", list(sample.keys())[:20])
            for tox in ("IS", "HA", "pCS", "pct_change"):
                if tox in sample:
                    print("   ", tox, "=", sample[tox])
        blob = " ".join(str(k) for k in keys[:50])
        print("  mentions_HA =", "HA" in blob or any(
            isinstance(obj[k], dict) and "HA" in obj[k] for k in keys[:20]
        ))
        print("  mentions_pCS =", "pCS" in blob or "pcs" in blob.lower())
        if "iv" in blob.lower() or "120" in blob:
            print("  note: keys look like IV / 120 mg cache")
    else:
        print("  repr =", repr(obj)[:400])
    print("  verdict: heatmap_cache_correct.pkl is the LEGACY IV 24 h")
    print("           Ki x Vmax ODE cache (heatmap_ki_vmax_sensitivity_ode.py).")
    print("           It is NOT manuscript Fig 7 (analytical) or Fig 8 (fu map).")
    print("           Do not use if it predates pCS pKa = -2.")


def main() -> None:
    paths = _paths()
    if not paths:
        print("No heatmap .pkl found. Pass a path:")
        print("  python analysis/inspect_heatmap_cache.py results\\heatmap_cache_correct.pkl")
        sys.exit(1)
    for p in paths:
        try:
            _report(p)
        except Exception as exc:
            print("FAILED", p, exc)


if __name__ == "__main__":
    main()
