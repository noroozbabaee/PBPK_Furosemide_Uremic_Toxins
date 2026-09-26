# Session log — chronic oral caches, tables, figures (25–26 Sep 2026)

This is the archive of decisions and results from the cloud-agent work on
branch `cursor/chronic-oral-caches-ki035-ee8a` (GitHub PR #1).  
The executable checklist for future agents is
`.cursor/skills/pbpk-furosemide-paper/SKILL.md`.

User Windows repo:

`C:\Users\p70091319\Desktop\Mywork\Project_PBPK_Model\PBPK_Furosemide_Uremic_Toxins_pythoncode`

---

## 1. Parameter count

One total for “everything the model uses” (compounds + healthy/CKD physiology):

- 92 healthy compound fields  
- 17 CKD diffs  
- 22 tubular flows  
- 13 pH  
- 8 geometry / weights  
- 10 physiology scalars  

**184 numeric parameters.** ODE state dimension is **142** (not the same as parameters).

---

## 2. Blood vs plasma; flows

In `src/ode_system_twocompartment.py`:

- `C_ven` and vascular tubular states are **total blood** concentrations (mg/L).
- Transporters and GFR see unbound **plasma** \(C_u = C_{\mathrm{ven}} \cdot f_{u,p}/BP\).
- Furosemide \(BP = 0.82\). If the paper quotes plasma \(C_{\max}\), divide blood peak by 0.82.

Flows:

- \(Q_c = 330\) L/h — **blood**, cardiac output  
- \(Q_{\mathrm{kidney}} = 72\) L/h — **blood**, renal blood flow  
- GFR — **plasma** ultrafiltrate (healthy 7.2 L/h before CKD scaling)  
- \(Q_{\mathrm{others}} = Q_c - Q_{\mathrm{kidney}}\) — blood  

Bowman inlet on the schematic is GFR (filtrate), not RBF.

---

## 3. Table 1 (PBUT) — values to keep

pCS pKa set to **−2** (sulfate conjugate; fully ionized; no passive reabsorption).  
Do **not** put dialysis-session clearances in Table 1 (Aurélie / AC).

| Toxin | Healthy \(C\) mg/L | CKD-5 \(C\) mg/L | Fold \(C\) | Healthy \(CL_R\) L/h | CKD-5 \(CL_R\) L/h |
|---|---:|---:|---:|---:|---:|
| IS | 0.77 | 28.16 | 36.5–36.6 | 7.14 | 0.20 |
| HA | 2.22 | 142.86 | 64.3 | 31.50 | 0.49 |
| pCS | 1.30 | 24.99 | 19.2 | 1.61 | 0.08 |

Fold for concentration: \(C_{\mathrm{CKD-5}}/C_{\mathrm{Healthy}}\).  
Fold for clearance: \(CL_{R,\mathrm{H}}/CL_{R,\mathrm{CKD-5}}\).  
KG’s dialysis-based CKD-5 \(CL\) is a different quantity (machine clearance).

---

## 4. Cache mixing bug (Windows)

`compare_dosing_regimens_grey_HA_ADDED.py` once plotted:

- QD 240 mg with `IS_base = 0.540` (healthy / acute)  
- TID with `IS_base = 19.712` (CKD-5 chronic)  
- `drg_avg ≈ 1.07` on the healthy file  

**Rule:** refuse the comparison. CKD-5 chronic `IS_base` must be ~8–80 mg/L and match across arms.

A second failure: `FileNotFoundError` for 40 mg × 3 because the glob required `_ckd5_` in the filename. Fix: search several folders; rank NPZs by metadata (`has_ha`, mtime); validate `MODE_CODE` and `IS_base` inside the file.

---

## 5. Figure 3 combiner breakage

User’s Windows `combine_bar_figures_vector.py` used kwargs that **do not exist**:

`axes_pair`, `draw_legend`, `lit_label`

Live APIs:

- `plot_healthy_lit_comparison_3scenarios.create_figure_bars_generic(..., axes=(ax, ax), show_legend=False)`  
- CKD twin needs `idx_map=IDX` and `clearance_fn=_ckd_mod._clearance_auc_window`  
- Import via module aliases (avoid `UnboundLocalError`)  
- No dialysis diamonds on panel D  
- One 4-entry bottom legend  

Copies: `figures/` and `publication_code/figures/` and `publication_code/analysis/`.

---

## 6. Chronic oral figures (final)

**Five-panel** — `figures/make_chronic_oral_five_panel.py`  
A IS, B p-CS, C HA (0–30 d); D furosemide days 27–30; E last 24 h.  
Traces: dashed TID 120, dotted QD 240, thick TID 240, grey no-drug.  
Times New Roman; two-line y-labels; wider panel gaps; PDF tight crop (`pad_inches=0.04`).

**% change bars** — `figures/make_chronic_oral_pct_bars.py`  
Same three regimens; Total/Bound/Unbound; Costes-Albrespic 2025 mean + 95% CI on IS and pCS; large “No clinical HA data” on panel C.

**PPT E** — `figures/make_ppt_furosemide_24h.py` (400 dpi).

**240 mg QD and 240×3** — `analysis/run_240mg_qd_and_tid.py` (`SKIP_QD=True` for TID-only).

Outputs (gitignored under `results/`; copies under `publication_code/Figures/` when the script writes them):

- `chronic_oral_five_panel.pdf` / `.png`  
- `chronic_oral_pct_bars.pdf` / `.png`  
- `dosing_comparison_timecourse.pdf` / `.png`  
- `bars_healthy_ckd_combined.pdf`  

NPZ caches stay only on the user’s disk unless they copy them.

---

## 7. Git / Windows pull

Remote had local modifications + untracked `publication_code/` files that blocked checkout.

Safe path: full folder backup → `git stash push -u` → checkout branch → `git reset --hard origin/cursor/chronic-oral-caches-ki035-ee8a`.  
Never `git clean -fdx` (deletes `results/` and `pbpk_env/`).  
Never `stash pop` if the goal is GitHub scripts.

---

## 8. Clinical %Δ (bars)

Costes-Albrespic M et al. Kidney Int Rep 2025;10:2165, >120 mg/day:

- IS +31.9% (95% CI 13.5–53.3)  
- pCS +29.3% (95% CI 6.1–57.8)  
- HA: no clinical panel  

Successful Windows bar run (26 Sep 2026): all three NPZs had **IS_base = 19.712 mg/L** (CKD-5, matched).

- `chronic_oral_cache_Ki0p35_dose40mg_x3pd_30days_ckd5.npz`  TID 120
- `chronic_oral_cache_Ki0p35_dose240mg_x1pd_30days_ckd5.npz` QD 240
- `chronic_oral_cache_Ki0p35_dose80mg_x3pd_30days_ckd5.npz`  TID 240

Share `results/chronic_oral_pct_bars.pdf` and `results/chronic_oral_pct_bars.txt`.


---

## 9. What not to do

- Do not combine healthy NPZ with CKD-5 chronic NPZ on one figure.  
- Do not treat dialysis \(CL\) as residual kidney \(CL_R\).  
- Do not call `C_ven` plasma without dividing by \(BP\).  
- Do not use pCS pKa of a carboxylic acid.  
- Do not pass `axes_pair` into the bar combiners.  
- Do not Print-to-PDF from Acrobat if you need a cropped figure (use the script PDF).
