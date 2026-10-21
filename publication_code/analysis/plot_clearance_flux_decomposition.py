# -*- coding: utf-8 -*-
"""Thin wrapper — always runs the repo-root flux script + repo-root src/.

Do not edit physics here. Edit analysis/plot_clearance_flux_decomposition.py.
"""
from __future__ import annotations

import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TARGET = os.path.join(_ROOT, "analysis", "plot_clearance_flux_decomposition.py")
runpy.run_path(_TARGET, run_name="__main__")
