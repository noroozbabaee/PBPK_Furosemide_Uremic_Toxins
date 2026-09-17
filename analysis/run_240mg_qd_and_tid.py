#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""240 mg per administration, CKD-5, 30-day oral:

  QD  : 240 mg × 1 / day  (240 mg/day total)
  TID : 240 mg × 3 / day  (720 mg/day total)

SKIP_QD on the TID block so the paired once-daily arm is not 720 mg.

Writes dated caches/PNG/PDF under results/ and copies figures to
publication_code/Figures/.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PUB_FIG = os.path.join(_ROOT, "publication_code", "Figures")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _configure_qd_240(gen) -> None:
    gen.MODE = "ckd5"
    gen.DOSING_REGIMEN = "QD"
    gen.SKIP_QD = True
    gen.DOSE_PER_ADMIN = 240.0
    gen.N_DOSES_PER_DAY = 1
    gen.DOSE_INTERVAL_H = 24.0
    gen.DOSE_TIMES_H = [0.0]
    gen.DOSE_TOTAL_PER_DAY = 240.0


def _configure_tid_240(gen) -> None:
    gen.MODE = "ckd5"
    gen.DOSING_REGIMEN = "TID"
    gen.SKIP_QD = True
    gen.DOSE_PER_ADMIN = 240.0
    gen.N_DOSES_PER_DAY = 3
    gen.DOSE_INTERVAL_H = 8.0
    gen.DOSE_TIMES_H = [0.0, 8.0, 16.0]
    gen.DOSE_TOTAL_PER_DAY = 720.0


def _copy_figures(gen) -> None:
    os.makedirs(_PUB_FIG, exist_ok=True)
    tag = gen._run_tag(gen.DOSE_PER_ADMIN, gen.N_DOSES_PER_DAY, gen.Ki, gen.N_DAYS)
    for ext in ("png", "pdf"):
        src = os.path.join(gen.OUTPUT_DIR, f"chronic_oral_{tag}.{ext}")
        if os.path.isfile(src):
            dst = os.path.join(_PUB_FIG, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"  copied {dst}", flush=True)


def main() -> None:
    gen = importlib.import_module("dose_response_tid_vs_qd")
    print(f"RUN_DATE={gen.RUN_DATE}  Ki={gen.Ki}  OUTPUT={gen.OUTPUT_DIR}", flush=True)

    jobs = (
        ("QD 240 mg × 1", _configure_qd_240),
        ("TID 240 mg × 3", _configure_tid_240),
    )
    for label, configure in jobs:
        print(f"\n########## {label} ##########", flush=True)
        importlib.reload(gen)
        configure(gen)
        t0 = time.time()
        gen.main()
        _copy_figures(gen)
        print(f"  [block done in {(time.time() - t0) / 60:.1f} min]", flush=True)

    print("240 mg QD and 240 mg TID caches/figures complete.", flush=True)


if __name__ == "__main__":
    main()
