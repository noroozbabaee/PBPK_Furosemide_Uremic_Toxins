# -*- coding: utf-8 -*-
"""Build Figure 3: Healthy (A/B) + CKD-5 (C/D) bars with one 4-entry legend.

Row labels carry Healthy vs CKD-5. Legend is only:

  No Furosemide
  With Furosemide (No Interaction)
  With Furosemide
  Literature

Requires the two three-scenario ODE runs (imported from analysis/).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
# Nested under publication_code/figures → still use the repo src/analysis/results
if os.path.basename(_PARENT) == "publication_code":
    _ROOT = os.path.abspath(os.path.join(_PARENT, ".."))
else:
    _ROOT = _PARENT
_ANALYSIS = os.path.join(_ROOT, "analysis")
_SRC = os.path.join(_ROOT, "src")
_RESULTS = os.path.join(_ROOT, "results")
_PUB_FIGURES = os.path.join(_ROOT, "publication_code", "Figures")
os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_PUB_FIGURES, exist_ok=True)
for _p in (_THIS_DIR, _SRC, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from combine_bar_figures_vector import main as _combined_main  # noqa: E402


def main() -> None:
    _combined_main()


if __name__ == "__main__":
    main()
