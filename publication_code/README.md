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

## Figure 4 — PBUT flux decomposition

| Paper file | Script |
|---|---|
| `Figures/09_flux_decomposition_PBUTs_greyscale.pdf` | `analysis/plot_clearance_flux_decomposition.py` |

Legend: Healthy / CKD-5 × No Furosemide / With Furosemide (not “Drug”).

```bash
PYTHONUTF8=1 python3 analysis/plot_clearance_flux_decomposition.py
```

The PDF is written to `publication_code/Figures/` (manuscript include path) and to a timestamped folder under `results/flux_decomposition/`.

