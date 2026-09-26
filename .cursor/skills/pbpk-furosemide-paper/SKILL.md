---
name: pbpk-furosemide-paper
description: >-
  PBPK furosemide–uremic toxin paper (noroozbabaee/PBPK_Furosemide_Uremic_Toxins).
  Use when editing this repo, Table 1/2, chronic oral CKD-5 caches, five-panel
  or %change figures, Figure 3 bars, Windows git pull, C_ven vs plasma, pCS pKa,
  dialysis clearance, dosing TID/QD, or reviewer comments (LN, TM, AC, KG, RM, Aurélie).
---

# PBPK furosemide × PBUT paper — session memory

Read this skill **before** changing ODE parameters, caches, or manuscript figures.

Windows working copy (user):

`C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode`

Python: `pbpk_env\Scripts\python.exe`

Git branch with this work: `cursor/chronic-oral-caches-ki035-ee8a` (PR #1 vs `main`).

Do **not** mix healthy/acute NPZ files with CKD-5 30-day chronic arms.

---

## Model facts (code, not captions)

| Item | Value |
|---|---|
| ODE states | **142** |
| Numeric model parameters (compounds + CKD diffs + tubular flows + pH + geometry + physiology scalars) | **184** |
| Central concentration `C_ven` | **total blood** (mg/L) |
| Unbound plasma driving transporters / GFR | \(C_u = C_{\mathrm{ven}} \cdot f_{u,p} / BP\) |
| Furosemide \(BP\) | **0.82** → plasma \(C_{\max} = C_{\mathrm{ven}} / 0.82\) |
| \(Q_c\) | **330 L/h blood** (cardiac output) |
| \(Q_{\mathrm{kidney}}\) | **72 L/h blood** (renal blood flow) |
| GFR (healthy scale in constants; CKD scaled) | plasma ultrafiltrate, not blood |
| pCS pKa | **−2.0** (sulfate; fully ionized; no passive tubular reabsorption) |
| Ki (furosemide at OAT1, all toxins) | **0.35 mg/L** (Costes-Albrespic 2025) |
| CKD-5 IS no-drug baseline (chronic caches) | ~**19–28 mg/L** (never ~0.5) |
| Healthy IS baseline | ~**0.54–0.77 mg/L** |

If a run script labels the furosemide peak as “plasma” but plots `C_ven`, the **manuscript** must say blood, or convert with \(BP\).

---

## Table 1 — PBUT steady plasma and native \(CL_R\) (no dialysis)

Agreed with Aurélie / AC: **omit hemodialysis-session clearances**. Clinical CKD-5 “clearance” during dialysis is artificial, not residual kidney \(CL_R\).

Fold change: concentration \(C_{\mathrm{CKD-5}} / C_{\mathrm{Healthy}}\); clearance \(CL_{R,\mathrm{Healthy}} / CL_{R,\mathrm{CKD-5}}\).

| | Healthy \(C\) (mg/L) | CKD-5 \(C\) (mg/L) | Fold \(C\) | Healthy \(CL_R\) (L/h) | CKD-5 \(CL_R\) (L/h) | Fold \(CL_R\) |
|---|---:|---:|---:|---:|---:|---:|
| IS | 0.77 | 28.16 | ~36.6 | 7.14 | 0.20 | ~35.7 |
| HA | 2.22 | 142.86 | ~64.3 | 31.50 | 0.49 | ~64.3 |
| pCS | 1.30 | 24.99 | ~19.2 | 1.61 | 0.08 | ~20.1 |

pCS pKa = −2 updates healthy/CKD-5 SS vs an older carboxylic pKa.

---

## Table 2 — furosemide PK

Peaks in ODE states are **blood**. Report plasma as \(C_{\mathrm{ven}}/BP\) when the paper says plasma. Do not invent a second ODE state.

---

## Chronic oral arms (CKD-5, 30 days, Ki 0.35)

| Label | Dose × /day | Total mg/day |
|---|---|---|
| TID 120 | 40 mg × 3 (0, 8, 16 h) | 120 |
| QD 240 | 240 mg × 1 | 240 |
| TID 240 | 80 mg × 3 | 240 |
| TID 240 mg/admin (optional runner) | 240 mg × 3 | 720 |
| No-drug | 0 | 0 |

Caches: `results/chronic_oral_cache_Ki0p35_dose{D}mg_x{N}pd_30days*.npz`  
Gitignores `results/*`. **Keep local NPZs**; they are not on GitHub.

**Sanity:** every compared arm must have `IS_base` in **8–80 mg/L** and agree within ~15%. `IS_base ≈ 0.54` is healthy — **abort**.

`compare_dosing_regimens_grey_HA_ADDED.py` searches `results/`, `publication_code/results/`, `figures/`, cwd; does not require `_ckd5_` in the **filename** if NPZ metadata (`MODE_CODE`, `IS_base`, dose clock) is CKD-5.

---

## Figures — which script

| Figure | Script (repo root or `publication_code/` twin) |
|---|---|
| Five-panel A–E (IS, HA, pCS 30 d; furo days 27–30; furo last 24 h) | `figures/make_chronic_oral_five_panel.py` |
| PPT panel E only | `figures/make_ppt_furosemide_24h.py` |
| %Δ bars A–C (IS, pCS, HA; Total/Bound/Unbound; Costes CI) | `figures/make_chronic_oral_pct_bars.py` |
| Time-course 2×2 check of three doses | `analysis/compare_dosing_regimens_grey_HA_ADDED.py` |
| 240 mg QD and 240×3 TID caches | `analysis/run_240mg_qd_and_tid.py` |
| Figure 3 Healthy/CKD-5 2×2 bars | `figures/combine_bar_figures_vector.py` or `figures/make_bars_healthy_ckd_combined.py` |
| Flux decomposition | `analysis/plot_clearance_flux_decomposition.py` |

Style: **Times New Roman** (Liberation Serif fallback on Linux), two-line y-labels, no top/right spines.  
PDF: `bbox_inches="tight"`, `pad_inches=0.04` via `_save_fig` — do not leave a letter-page of white.  
Do **not** put dialysis diamonds on CKD clearance panel D of Figure 3 (Aurélie).  
Figure 3 legend: four entries only (No Furosemide, With Furosemide (No Interaction), With Furosemide, Literature). Row labels: Healthy / CKD-5.

`create_figure_bars_generic` live kwargs: `axes=`, `show_legend=`, CKD needs `idx_map` and `clearance_fn`. **Never** `axes_pair`, `draw_legend`, `lit_label`.

HA %Δ bars: large **“No clinical HA data”**. Costes-Albrespic 2025: IS +31.9% (13.5–53.3), pCS +29.3% (6.1–57.8). At constant \(f_{u,p}\), Total/Bound/Unbound %Δ are identical — one label per group.

---

## Windows: update from GitHub without losing caches

`results\` and `pbpk_env\` are gitignored. Never `git clean -fdx`.

```bat
cd C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode
xcopy "%CD%" "C:\Users\p70091319\Desktop\PBPK_backup_before_github" /E /I /H /Y
git fetch origin
git stash push -u -m "windows local before github branch"
git checkout cursor/chronic-oral-caches-ki035-ee8a
git reset --hard origin/cursor/chronic-oral-caches-ki035-ee8a
```

Do **not** `git stash pop` (that restores broken scripts).

### Commands (Command Prompt)

```bat
pbpk_env\Scripts\python.exe publication_code\figures\make_chronic_oral_five_panel.py
pbpk_env\Scripts\python.exe publication_code\figures\make_chronic_oral_pct_bars.py
pbpk_env\Scripts\python.exe publication_code\figures\combine_bar_figures_vector.py
pbpk_env\Scripts\python.exe publication_code\analysis\compare_dosing_regimens_grey_HA_ADDED.py
```

---

## Reviewer / co-author map

- **LN** — \(Q_c\), \(Q_{\mathrm{others}}\) are **blood**; furosemide peak in scripts may be mis-labeled plasma → check `ode_system_twocompartment.py`.
- **TM** — schematic arrows: \(Q_c\), \(Q_{\mathrm{kidney}}\), Bowman inlet (GFR plasma).
- **AC / Aurélie** — no dialysis \(CL\) in Table 1; follow her table edits; “established” after “The PBPK model” is optional wording.
- **KG** — fold change as ratios above; do not use dialysis \(CL\) as native \(CL_R\).
- **RM** — keep phenotype of caches consistent.

Initiation protocol (no-drug SS vs with-drug 30 d vs IV PK extract):
`docs/scenario_initiation_ckd5_chronic_oral.md`. Do not mix those three.



