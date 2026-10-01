# Figure 7 / 8 vs what is actually in this repository
#
# Manuscript Fig 7: analytical Ki × Vmax, chronic oral, 3 toxins (IS, HA, pCS).
# Manuscript Fig 8: analytical fu × fu (filenames cited as
#   heatmap_ki_vmax_analytical.pdf and heatmap_fu_analytical_3panels.pdf).
#
# Those two PDF *builders* are NOT in this git tree. The inhibition math is
# reconstructed in analysis/heatmap_analytical_formulas.py (same as
# dose_response_tid_vs_qd.analytical_ss). Css_furo = F·Dose_day/(CL_FURO·τ).

## In the repo

| File | What it is | Dosing | Toxins | Fig 7? |
|---|---|---|---|---|
| `analysis/heatmap_ki_vmax_sensitivity_ode.py` | Legacy IV ODE | **IV 120 mg / 2 h**, 24 h | IS, pCS | **No** |
| `analysis/heatmap_ki_vmax_engine.py` | Collaborator engine: row-wise warm-up+no-drug, `TOX*_PASS` fingerprint, restore scales | `single_iv` or `chronic_oral` TID 80×3 | IS, HA, pCS | Only if `HEATMAP_SCENARIO=chronic_oral` |
| `figures/heatmap_ki_vmax_from_cache.py` | Supplementary IV 3-panel (Times, 6.27 in, no +30%) | IV | IS, pCS, HA | **No** |
| `analysis/heatmap_fu_sensitivity_ode.py` | Chronic **ODE** \(f_u\times f_u\) (QD 240; C_ven carried) | 30 d oral | IS, pCS | Fig 8 analogue, not Fig 7 |
| `analysis/heatmap_validate_corners.py` | 5 Ki×Vmax + 5 fu corners vs analytical | `--ode` = TID 80×3 | IS, HA, pCS | Decision test (~2 pp) |
| `analysis/heatmap_analytical_formulas.py` | Closed-form used by missing PDF builders | chronic SS | 3 toxins | Math only |

## Caption error (agree)

Vmax axis is a **fraction of healthy** basolateral OAT1 \(V_{\max}\) (it **replaces** the CKD-5 scale, default 0.05). It is **not** “relative to the CKD-5 value”.

## Blood vs %

`C_tox*_cen` is **total blood**. %Δ is the same in plasma. Single-dose maps: \(\Delta C\) at 24 h, not \(\Delta C_{ss}\).

## Plan

1. Day-boundary: `run_one_day_tid` carries C_ven; five-panel RESET counts must be 0.
2. `python analysis/heatmap_validate_corners.py --ode` (10 chronic runs).
3. If all |ODE−anal| ≲ 2 pp → keep analytical Figs 7–8, table in Supp S8.6.
4. Else `HEATMAP_SCENARIO=chronic_oral python -c "from heatmap_ki_vmax_engine import compute_heatmaps; compute_heatmaps()"` at 7×7.

## Send the collaborator

- `analysis/dose_response_tid_vs_qd.py`
- `analysis/heatmap_ki_vmax_engine.py` (`compute_heatmaps`, `_force_ki`, `_set_vmax`)
- `analysis/heatmap_ki_vmax_sensitivity_ode.py` (legacy IV)
- `analysis/heatmap_analytical_formulas.py` (Css_furo + MM)
- `analysis/heatmap_fu_sensitivity_ode.py` (chronic fu ODE; not the missing 3-panel analytical PDF)
- `analysis/heatmap_validate_corners.py`

Delete `results/heatmap_cache_correct.pkl` if it exists (pCS may pre-date pKa=−2).
