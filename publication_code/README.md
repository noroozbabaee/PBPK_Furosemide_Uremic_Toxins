# publication_code/

This folder is the paper-facing copy of the scripts that produce manuscript figures. The live sources remain in `src/`, `analysis/`, and `figures/` at the repository root; this tree is the same keepers, grouped for the paper workflow.

## Figure 3 — toxin bars, Healthy vs CKD-5

| Paper file | Script |
|---|---|
| `Figures/bars_healthy_ckd_combined.pdf` | `figures/make_bars_healthy_ckd_combined.py` |

The 2×2 layout is **A/B Healthy** (concentration, clearance) and **C/D CKD-5**. The legend has four entries only (No Furosemide, With Furosemide (No Interaction), With Furosemide, Literature). Row labels carry Healthy vs CKD-5.

```bash
PYTHONUTF8=1 python3 publication_code/figures/make_bars_healthy_ckd_combined.py
```

Output: `results/bars_healthy_ckd_combined.png` and `.pdf` (gitignored). Copy the PDF into the manuscript `Figures/` folder for `\includegraphics{Figures/bars_healthy_ckd_combined.pdf}`.

Healthy and CKD-5 bar/time-course drivers used by that combiner:

- `analysis/plot_healthy_lit_comparison_3scenarios.py`
- `analysis/plot_ckd_lit_comparison_3scenarios.py`

## Chronic oral five-panel (IS / pCS / HA / furosemide)

| Paper-style file | Script |
|---|---|
| `Figures/chronic_oral_five_panel.pdf` | `figures/make_chronic_oral_five_panel.py` |

TID 120, QD 240, TID 240, no-drug CKD-5. Requires 30-day ODE arms (HA is stored in the caches).

```bash
PYTHONUTF8=1 python3 figures/make_chronic_oral_five_panel.py
```

% change bars (IS / p-CS / HA; TID 120 vs QD 240 vs TID 240):

```bash
PYTHONUTF8=1 python3 figures/make_chronic_oral_pct_bars.py
```

## 240 mg per dose — once daily vs three times daily

| Outputs | Script |
|---|---|
| `chronic_oral_Ki0p35_dose240mg_x1pd_30days_ckd5_<date>.pdf` | `analysis/run_240mg_qd_and_tid.py` |
| `chronic_oral_Ki0p35_dose240mg_x3pd_30days_ckd5_<date>.pdf` | same |

QD = 240 mg × 1 (240 mg/day). TID = 240 mg × 3 (720 mg/day). CKD-5, 30 days, Ki = 0.35.

```bash
PYTHONUTF8=1 python3 analysis/run_240mg_qd_and_tid.py
```

## PowerPoint — furosemide final 24 h cycle (panel E)

Standalone greyscale slide of the five-panel **E** plot (TID 120 / QD 240 / TID 240).

```bash
PYTHONUTF8=1 python3 figures/make_ppt_furosemide_24h.py
```

Insert `publication_code/Figures/chronic_oral_furosemide_24h_cycle_ppt_legend.png` into PowerPoint (400 dpi). PDF/SVG are in the same folder.

## Dosing comparison (CKD-5 chronic only)

`analysis/compare_dosing_regimens_grey_HA_ADDED.py` loads **only** `chronic_oral_cache_*_ckd5_*.npz` for 40×3, 240×1, and 80×3. It refuses a healthy/acute 240 mg file (`IS_base` ~0.5 mg/L). Do not glob NPZs without `_ckd5_` in the name.


## Figure 4 — PBUT flux decomposition

| Paper file | Script |
|---|---|
| `Figures/09_flux_decomposition_PBUTs_greyscale.pdf` | `analysis/plot_clearance_flux_decomposition.py` |

Legend: Healthy / CKD-5 × No Furosemide / With Furosemide (not “Drug”).

```bash
PYTHONUTF8=1 python3 analysis/plot_clearance_flux_decomposition.py
```

The PDF is written to `publication_code/Figures/` (manuscript include path) and to a timestamped folder under `results/flux_decomposition/`.

