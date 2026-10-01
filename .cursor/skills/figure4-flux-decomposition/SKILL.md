---
name: figure4-flux-decomposition
description: >-
  Corrected manuscript Figure 4 (PBUT flux decomposition) for
  noroozbabaee/PBPK_Furosemide_Uremic_Toxins. Use when editing, running, or
  replacing Fig 4 / 09_flux_decomposition / plot_clearance_flux_decomposition,
  flux Vmax scoring, hatch greyscale 2×2, glomerular filtration titles, or
  deleting stale flux PDFs.
---

# Figure 4 — corrected flux decomposition

Read this **instead of** using `09_flux_decomposition_PBUTs_greyscale.pdf` or
any `results/flux_decomposition/20261001_*` folder.

Those old files scored **healthy** trajectories with leftover **CKD-5**
`Vmax_baso_uptake` / `Vmax_api_secr` because fluxes were computed after all
four ODEs. Healthy no-drug uptake:apical then looked like ~1:3 (CKD apical
scale 0.15 vs baso 0.05). That is **wrong**. Do not put those bars in the paper.

## Canonical files (keep)

| Role | Path |
|---|---|
| Physics + plot | `analysis/plot_clearance_flux_decomposition.py` |
| Numbers → tex | `analysis/flux_export.py` |
| Wrapper only | `publication_code/analysis/plot_clearance_flux_decomposition.py` (runpy to root) |

Edit physics only in the **repo-root** `analysis/` file. It inserts
**repo-root** `src/` (not `publication_code/src`).

Do **not** delete these scripts.

## Deleted / forbidden outputs

Removed from git (wrong numbers):

- `publication_code/Figures/09_flux_decomposition_PBUTs_greyscale.pdf`
- `publication_code/Figures/09_flux_decomposition_PBUTs_greyscale.png`

On the user’s Windows copy, delete if present:

`results\flux_decomposition\20261001_171848`

Never `git clean -fdx`.

Manuscript include after a **fresh** run:

`publication_code/Figures/Fig4.pdf`

Timestamped copies: `results/flux_decomposition/<YYYYMMDD_HHMMSS>/`
(`09_…` PBUTs, `10_…` furosemide, `11_…` all species, `flux_numbers.tex`).
Those `09_` names in a **new** timestamp folder are OK; the **old** committed
`09_` in `publication_code/Figures/` was not.

## What the corrected driver does

1. Healthy SS 1500 h, then CKD-5 SS 1500 h (`H_NO_DRUG` / `CKD_NO_DRUG`).
2. Four 12 h arms from that SS. `TIME_WINDOW = (0, 12)`.
3. **Immediately** after each `run_sim`, call `compute_fluxes_for_scenario_drug`
   while `C.tox1` still has that scenario’s Vmax. Print `IS Vmax_baso` and `fu_p`.
4. Plot 2×2 greyscale hatch, **12 × 7.6 in**, Times (Liberation Serif on Linux),
   16 pt bold titles:

| Panel | Title |
|---|---|
| A | Glomerular filtration |
| B | Basolateral passive diffusion |
| C | Basolateral OAT1 uptake |
| D | Apical secretion |

Species on Fig 4: **IS, HA, pCS** (not furosemide). Legend: Healthy / CKD-5 ×
No Furosemide / With Furosemide.

CKD Vmax comes only from `compound_parameters.py` `_CKD_SCALES` (drug/tox
baso **0.05**, apical **0.15**). Do not call `set_ckd_vmax_scales` in this
script.

`C_tox*_cen` is **total blood**. Filtration uses \(C_u = C_{\mathrm{blood}} f_{u,p}/BP\).

## Pass / fail on the log

Must see **two** different `IS Vmax_baso` values (healthy vs CKD), not one.

Healthy no-drug **uptake : apical ≈ 1**. If it is ~0.33, scoring is still
post-hoc CKD Vmax — throw the PDF away.

CKD-5 SS `IS` blood must be ~**19.7** mg/L (plasma ~28). If IS ~0.54, abort
(healthy IC on a CKD run).

## Windows (CMD, after `cd` into the repo)

Repo: `C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode`

```bat
cd /d C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode
git pull origin cursor/chronic-oral-caches-ki035-ee8a
pbpk_env\Scripts\python.exe analysis\plot_clearance_flux_decomposition.py
```

PowerShell: `Set-Location` the same path (no `cd /d`).

Figure 3 is a different 14×10 bar layout. Figure 5 is the 6.27 in five-panel.
Do not restyle Fig 4 to match those.

This figure is **not** the Ki×Vmax heatmap (that IV ODE map is not Fig 7).
For other paper figures see `.cursor/skills/pbpk-furosemide-paper/SKILL.md`.
