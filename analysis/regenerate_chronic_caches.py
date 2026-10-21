#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild paper chronic-oral caches with today's date and current compound_parameters.

Outputs in results/:
  chronic_oral_cache_Ki0p35_dose40mg_x3pd_30days_ckd5_YYYYMMDD.npz
  chronic_oral_cache_Ki0p35_dose80mg_x3pd_30days_ckd5_YYYYMMDD.npz
  chronic_oral_cache_Ki0p35_dose240mg_x1pd_30days_ckd5_YYYYMMDD.npz
  chronic_oral_cache_Ki0p35_dose240mg_x3pd_30days_ckd5_YYYYMMDD.npz
plus the matching chronic_oral_*.png/.pdf from the driver.

240 mg per administration:
  QD  = 240 mg × 1 (run with TID 80×3, SKIP_QD=False, or run_240mg_qd_and_tid.py)
  TID = 240 mg × 3 (SKIP_QD so the paired once-daily arm is not 720 mg)

Run order: 80 mg TID first so the paired QD is 240 mg once daily;
then 40 mg TID with SKIP_QD; then 240 mg TID with SKIP_QD.
"""
from __future__ import annotations

import importlib
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _configure_tid(gen, dose_per_admin: float, skip_qd: bool) -> None:
    gen.MODE = "ckd5"
    gen.DOSING_REGIMEN = "TID"
    gen.SKIP_QD = bool(skip_qd)
    gen.DOSE_PER_ADMIN = float(dose_per_admin)
    gen.N_DOSES_PER_DAY = 3
    gen.DOSE_INTERVAL_H = 8.0
    gen.DOSE_TIMES_H = [0.0, 8.0, 16.0]
    gen.DOSE_TOTAL_PER_DAY = gen.DOSE_PER_ADMIN * gen.N_DOSES_PER_DAY


def main() -> None:
    gen = importlib.import_module("dose_response_tid_vs_qd")
    print(f"RUN_DATE={gen.RUN_DATE}  Ki={gen.Ki}  OUTPUT={gen.OUTPUT_DIR}", flush=True)

    jobs = (
        (80.0, False),   # TID 80×3 + QD 240×1
        (40.0, True),    # TID 40×3 only
        (240.0, True),   # TID 240×3 only (not 720 mg QD)
    )
    for dose, skip_qd in jobs:
        print(
            f"\n########## TID {dose:.0f} mg x3  "
            f"(SKIP_QD={skip_qd}) ##########",
            flush=True,
        )
        importlib.reload(gen)
        _configure_tid(gen, dose, skip_qd)
        t0 = time.time()
        gen.main()
        print(f"  [block done in {(time.time() - t0) / 60:.1f} min]", flush=True)

    print("ALL CHRONIC CACHES REGENERATED.", flush=True)


if __name__ == "__main__":
    main()
