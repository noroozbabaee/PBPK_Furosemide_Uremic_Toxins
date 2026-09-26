# Manuscript checklist — chronic oral results (archive)

**Phenotype:** CKD-5 chronic oral only.  
Healthy IS blood ≈ 0.54 mg/L. CKD-5 IS blood ≈ 19.71 mg/L = **plasma 28.16** (\(BP=0.70\)).

Windows caches (26 Sep 2026), all `IS_base = 19.712` mg/L blood:

- `chronic_oral_cache_Ki0p35_dose40mg_x3pd_30days_ckd5.npz` — TID 120  
- `chronic_oral_cache_Ki0p35_dose240mg_x1pd_30days_ckd5.npz` — QD 240  
- `chronic_oral_cache_Ki0p35_dose80mg_x3pd_30days_ckd5.npz` — TID 240 (\(3\times80\) mg)

Live PK from the same CKD-5 warm-up: \(CL_{R,\mathrm{IS}}=0.1953\) L/h, \(CL_{R,\mathrm{pCS}}=0.0840\) L/h.

Costes-Albrespic 2025 (clinical, unchanged): IS \(+31.9\%\), pCS \(+29.3\%\).

---

## CHANGE

1. **pCS no-drug plasma Css**  
   Old: 33 mg/L.  
   New: **24.99 mg/L** (pKa = −2). Never write 33 or 43.86 again.

2. **pCS % and with-drug Css for TID \(3\times80\) mg**  
   Old: +30.2% or +32.9%, 33 → 43.86 mg/L.  
   New (same NPZ as bars): about **+30.8%**, plasma **24.99 → 32.7 mg/L**.  
   32.9% only matched the *old* 33 → 43.86 pair. 30.2% matched neither.

3. **IS with-drug Css (optional one-decimal refresh)**  
   Old: +28.1%, 28.16 → 36.07 mg/L.  
   New: about **+28.8%**, plasma **28.16 → 36.3 mg/L**.  
   Keep 28.16 as the no-drug plasma baseline.

4. **HA absolutes (AC)**  
   Add: **+1.1%** (or +1.2%), plasma **142.86 → 144.4 mg/L**.

5. **Clearances in this paragraph**  
   Old: IS 0.15 L/h, pCS 0.06 L/h, “three-fold”.  
   New: IS **0.20** L/h (0.1953), pCS **0.08** L/h (0.0840), **2.3-fold**.

6. **pCS accumulation time**  
   Old: ~30 days; \(\tau \approx 116\) h still OK.  
   New: \(V=10\) L / \(CL_R=0.084\) → \(\tau \approx 119\) h; 4–5 \(\tau\) ≈ **20–25 days**, not 30.

7. **Wording**  
   Use AC: “above the no-drug CKD-5 baseline”.  
   Three toxins (IS, pCS, HA), not two.  
   LN: IV peak in scripts is blood `C_ven`; plasma = blood / \(BP\) (furosemide 0.82; IS 0.70; pCS 0.55; HA 0.84).

8. **Do not** put healthy IV % (+0.03–0.67%) in the same sentence as CKD-5 chronic % without labelling both phenotypes.  
   Do not use dialysis session \(CL\) as residual \(CL_R\).

---

## KEEP

1. Hypothesis: single-dose OAT1 block is too short vs toxin \(\tau\).  
2. Clinical targets: IS +31.9%, pCS +29.3%.  
3. TID \(3\times80\) mg/day (240 mg/day) as the Costes-Albrespic regimen.  
4. CKD-5 no-drug **plasma** Table 1: IS **28.16**, HA **142.86**, pCS **24.99** mg/L.  
5. IS no-drug 28.16 mg/L plasma (= 19.71 blood).  
6. Chronic IS rise ~**+28%** (28.1–28.8%) — in the clinical ballpark.  
7. HA barely moves (~**+1%**).  
8. IS (and HA) approach SS in **10–15 days**.  
9. Furosemide itself SS in **1–2 days**.  
10. No reciprocal PBUT→furosemide inhibition (already in the text).  
11. Single-dose IV remains **modest** vs clinical (healthy ≪ CKD-5 ≪ chronic oral). Do not replace those IV numbers with chronic oral numbers.

---

## Numbers to paste (CKD-5, TID 80×3, plasma)

| | No-drug | With TID 240 | %Δ |
|---|---:|---:|---:|
| IS | 28.16 | 36.3 | +28.8 |
| pCS | 24.99 | 32.7 | +30.8 |
| HA | 142.86 | 144.4 | +1.1 |

If `chronic_oral_pct_bars.txt` prints slightly different one-decimal %Δ, use **that** file; it is the same NPZs. %Δ applies to blood and plasma.

---

## Files to keep on the PC

- `results\chronic_oral_pct_bars.pdf` / `.png` / `.txt`  
- `results\chronic_oral_five_panel.pdf` / `.png`  
- the three `*_ckd5.npz` caches above  

## Section: dosing schedule vs daily dose (`sec:results-dosing-comparison`)

All three arms are **CKD-5** (`IS_base = 19.712` blood = plasma **28.16**). Healthy was not used.

### KEEP

- Three regimens: TID 120 (\(3\times40\)), QD 240 (\(1\times240\)), TID 240 (\(3\times80\)).
- Same no-drug CKD-5 SS (plasma): IS **28.16**, HA **142.86**, pCS **24.99**.
- HA remains ~1% (dose-sensitive but small).

### CHANGE (stale % vs current NPZ / bar figure)

| | Manuscript now | Current CKD-5 caches (last-day avg) |
|---|---|---|
| TID 120 IS | 15.8% | **~19.9%** |
| TID 120 HA | 0.6% | **~0.7%** |
| TID 120 pCS | 17.3% | **~20.3%** |
| TID 240 IS | 28.1% | **~28.8%** |
| TID 240 HA | 1.2% | **~1.1%** |
| TID 240 pCS | 30.2% | **~30.8%** |

15.8% and 17.3% are **not** the current 40×3 cache. 30.2% is the old pCS-% that did not match 33→43.86 (that pair is 32.9% and is obsolete anyway).

### Logic of the paragraph

The **title** is schedule independent of **total daily dose**.  
The **sentence you quote** is doubling TID 120 → TID 240, which **changes total daily dose**. That is a dose-response, not a schedule effect.

- Dose effect (same schedule): TID 120 vs TID 240. Current % do **not** double (19.9→28.8 is ×1.45, not ×2). Soften “approximately proportional”.
- Schedule effect (same 240 mg/day): **QD 240 vs TID 240** (and vs TID 120 only as extra). QD 240 is listed in the opening but missing from this sentence — add it if the figure shows it.

Ground truth after pull: `results/chronic_oral_pct_bars.txt` from the three `*_ckd5.npz` files.

