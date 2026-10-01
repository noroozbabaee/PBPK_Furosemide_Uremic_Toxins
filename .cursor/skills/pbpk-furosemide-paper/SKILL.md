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
| Five-panel A–E | `figures/make_chronic_oral_five_panel.py` |
| Ki×Vmax IV supplementary (not Fig 7) | `figures/heatmap_ki_vmax_from_cache.py` + `analysis/heatmap_ki_vmax_engine.py` |
| Fig 8 analytical $f_u\times f_u$ (IS, pCS, HA; no contours) | `figures/make_figure8_fu_heatmap.py` |
| Corner ODE vs analytical | `analysis/heatmap_validate_corners.py` |
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

**Always `cd` into the repo first.** `C:\Users\p70091319` is **not** a git repo. From the home folder:

- `git pull origin cursor/chronic-oral-caches-ki035-ee8a` → `fatal: not a git repository`
- `pbpk_env\Scripts\python.exe publication_code\analysis\...` → `The system cannot find the path specified`

```bat
cd /d C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode
git pull origin cursor/chronic-oral-caches-ki035-ee8a
```

**Figure 8** (analytical \(f_u\times f_u\), flux-decomposition style, no grid/contours):

```bat
cd /d C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode
pbpk_env\Scripts\python.exe figures\make_figure8_fu_heatmap.py
```

Outputs: `results\heatmap_fu_analytical_3panels.pdf` and `publication_code\Figures\heatmap_fu_analytical_3panels.pdf`. Do **not** run `heatmap_fu_sensitivity_ode.py` for Fig 8 (that is the ODE grid and takes hours).

If checkout is blocked by local edits:

```bat
cd /d C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode
xcopy "%CD%" "C:\Users\p70091319\Desktop\PBPK_backup_before_github" /E /I /H /Y
git fetch origin
git stash push -u -m "windows local before github branch"
git checkout cursor/chronic-oral-caches-ki035-ee8a
git reset --hard origin/cursor/chronic-oral-caches-ki035-ee8a
```

Do **not** `git stash pop` (that restores broken scripts).

### Commands (Command Prompt only — no PowerShell line-split, no `python -c` list comps)

CMD breaks a wrapped `-c "... [print ...]"` into `'[print' is not recognized`. Never paste multi-line Python into CMD. Use a `.py` file.

```bat
cd /d C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode
pbpk_env\Scripts\python.exe publication_code\analysis\compare_dosing_regimens_grey_HA_ADDED.py
pbpk_env\Scripts\python.exe publication_code\analysis\check_cache_t0_pcs.py
pbpk_env\Scripts\python.exe publication_code\figures\make_chronic_oral_five_panel.py
pbpk_env\Scripts\python.exe publication_code\figures\make_chronic_oral_pct_bars.py
pbpk_env\Scripts\python.exe publication_code\figures\combine_bar_figures_vector.py
```

Five-panel: if `dose40mg_x3pd`, `dose240mg_x1pd`, `dose80mg_x3pd` CKD-5 NPZs exist, it **skips ODE warm-up**. Confirm the log says `Existing 40×3, 240×1, 80×3 caches found` and all `IS_base=19.712`.

---

## Windows run log (verified on this machine)

Repo: `...\PBPK_Furosemide_Uremic_Toxins_pythoncode`  
Branch after pull: `cursor/chronic-oral-caches-ki035-ee8a`  
Interpreter: `pbpk_env\Scripts\python.exe`  
CKD-5 scales in live `compound_parameters.py`: drug/tox baso **0.05**, apical **0.15**; GFR_scale **0.100**, RBF_scale **0.35**.

### `compare_dosing_regimens_grey_HA_ADDED.py` (use these numbers in the paper)

All three arms: **CKD-5**, `IS_base = 19.712` mg/L **blood** (plasma 28.16). Not healthy (~0.54).

| Regimen | file | IS %Δ | HA %Δ | pCS %Δ | furo Cavg **blood** | furo Cmax **blood** |
|---|---|---:|---:|---:|---:|---:|
| TID 120 (40×3) | `..._dose40mg_x3pd_30days_ckd5.npz` | +15.8 | +0.6 | +17.3 | 2.168 | 3.601 |
| QD 240 (240×1) | `..._dose240mg_x1pd_30days_ckd5.npz` | +24.6 | +1.3 | +27.2 | 4.772 | 16.645 |
| TID 240 (80×3) | `..._dose80mg_x3pd_30days_ckd5.npz` | +28.1 | +1.2 | +32.9 | 4.358 | 7.173 |

Plasma furosemide = blood / **0.82**: Cavg **2.644 / 5.819 / 5.315**; Cmax **4.39 / 20.3 / 8.75**.

Last-24 h **blood** Cmax (DB / Dian): IS QD 25.020 ≤ TID 25.363; **HA QD 121.932 > TID 121.520**; pCS QD 23.056 ≤ TID 23.905. HA is the only toxin whose QD peak exceeds TID.

Outputs: `results\dosing_comparison_timecourse.{png,pdf}` and `publication_code\Figures\...` same names.

### `check_cache_t0_pcs.py` (local NPZs, do not mix for %Δ)

`t0 = 0` on all files. **pCS[0] is not identical on every NPZ:**

| file | pCS[0] blood | pCS_base | plasma0 (= blood/0.55) |
|---|---:|---:|---:|
| `dose120mg_x1pd` | 13.745 | 13.745 | 24.99 |
| `dose40mg_x3pd` | 13.745 | 13.745 | 24.99 |
| `dose240mg_x1pd` | 17.944 | 17.944 | 32.63 |
| `dose80mg_x3pd` | 17.944 | 17.944 | 32.63 |

The **published three-arm comparison** uses 40×3, 240×1, 80×3 with matching **IS_base = 19.712**. pCS %Δ is still vs each file’s own `pCS_base` (printed +17.3 / +27.2 / +32.9). Do not treat `dose120mg_x1pd` as a fourth paper arm unless re-simulated from the same warm-up SS.

### `make_chronic_oral_five_panel.py`

Log must include: skip warm-up; all four traces CKD-5 `IS_base ≈ 19.712`; first row **A. IS, B. pCS, C. HA**. Times New Roman, axis/legend 11 pt, panel letters 14 pt bold.
Caption (bars): `docs/figure_caption_chronic_oral_pct_bars.tex`. Do **not** swap B/C on the bar figure (A IS, B pCS, C HA).

---

## Reviewer / co-author map

- **LN** — \(Q_c\), \(Q_{\mathrm{others}}\) are **blood**; furosemide peak in scripts may be mis-labeled plasma → check `ode_system_twocompartment.py`.
- **TM** — schematic arrows: \(Q_c\), \(Q_{\mathrm{kidney}}\), Bowman inlet (GFR plasma).
- **AC / Aurélie** — no dialysis \(CL\) in Table 1; follow her table edits; “established” after “The PBPK model” is optional wording.
- **KG** — fold change as ratios above; do not use dialysis \(CL\) as native \(CL_R\).
- **RM** — keep phenotype of caches consistent; table notes under the table (`threeparttable`).
- **DB / Dian** — QD HA Cmax > TID HA Cmax; IS and pCS QD Cmax ≤ TID. Quote last-24 h peaks (plasma = blood/BP).
- **LN** — all table Cmax and \(\overline{C}_{\mathrm{furo}}\) in **plasma**; old TID Cmax 3.7 and 7.1 were blood.
- **Fig 7/8** — analytical chronic maps are **not** `heatmap_ki_vmax_from_cache.py` (IV ODE). See `docs/figure7_8_heatmap_map.md`. Vmax axis = fraction of **healthy** OAT1 \(V_{\max}\).

Initiation protocol (no-drug SS vs with-drug 30 d vs IV PK extract):
`docs/scenario_initiation_ckd5_chronic_oral.md`. Do not mix those three.



