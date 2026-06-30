# Parameter Sources

Every kinetic parameter in this model traces to a literature value or an
explicit, documented calibration decision. This document consolidates
the most important ones in one place for quick reference. **The
authoritative source is always `src/compound_parameters.py`** — this
document summarizes it; if the two ever disagree, trust the code.

## OAT1 basolateral uptake kinetics (Km)

| Compound | Km (µM) | Km (mg/L) | Source |
|---|---|---|---|
| Furosemide | 38.90 | 12.87 | Hasannejad et al. 2004 |
| Indoxyl sulfate (IS) | 18.0 | 3.84 | Deguchi et al. 2004, *Kidney Int* 65:162 |
| Hippuric acid (HA) | 24.0 | 4.302 | Deguchi et al. 2004, *Kidney Int* 65:162 (note: HA is NOT transported by OAT3, per the same source) |
| p-Cresyl sulfate (pCS) | 120.0 | 22.58 | Watanabe et al. 2014, *Kidney Int* 86:840 |

µM→mg/L conversion uses each compound's molecular weight (defined in
`constants.py`: `MW_FUR`, `MW_IS`, `MW_HA`, `MW_pCS`).

## OAT1 competitive inhibition constant (Ki)

**Current model value: Ki = 0.35 mg/L** for furosemide's inhibition of
all three toxins' OAT1 uptake.

This was calibrated to reproduce the clinical signal in Costes-Albrespic
et al. 2025 (see README.md "Known limitations," item 1). The
literature-derived reference value, via Cheng-Prusoff correction from
Mihaila et al. 2020 (*Toxins* 12:391, fluorescein-displacement assay,
IC50 = 28.1 µM):

```
Ki = IC50 / (1 + [S]/Km,fluorescein) = 28.1 / (1 + 1.0/0.8) = 12.49 µM
Ki [mg/L] = 12.49 × 330.74 (furosemide MW) / 1000 = 4.13 mg/L
```

`src/compounds.py` documents a corrected, slightly different derivation
arriving at **3.77 mg/L** as its internal fallback/reference constant
(`_KI_FURO_MGL`). Both 4.13 and 3.77 mg/L are roughly an order of
magnitude above the 0.35 mg/L actually used in simulations — this gap
is real and unresolved; see README.md for full discussion. **Do not
treat 0.35 mg/L as a literature-derived value** — it is a fitted
parameter.

## Plasma protein binding (fu_p)

Healthy values (Deltombe & Eloot 2015, *Toxins* 7:3933, Table 4 —
unless noted):

| Compound | Healthy fu_p | CKD-5 fu_p | Direction of change in CKD |
|---|---|---|---|
| Furosemide | 0.02 | 0.05 | **increases** (CMPF/oleate displacement from albumin Site I; Takamura et al. 1997, *Clin Chem* 43:2274) |
| IS | 0.16 | 0.10 | decreases (more bound — 84%→90% protein-bound) |
| HA | 0.66 | 0.52 | decreases (34%→48% protein-bound) |
| pCS | 0.06 | 0.05 | decreases (94%→95% protein-bound) |

The directional asymmetry is intentional and documented in
`compound_parameters.py`'s "ROUND 2" revision notes: furosemide's
unbound fraction *increases* in CKD (competitive displacement by
uremic toxin-derived ligands like CMPF and oleate), while the toxins'
own unbound fractions *decrease* despite — or because of — their rising
absolute concentrations (individual binding-site affinity considerations
dominate over site saturation at these concentrations).

## Toxin generation rate (R0) and CKD-5 target concentrations

R0 values were back-calculated from each compound's actual ODE-simulated
renal clearance (CLR) at the target steady-state plasma concentration —
**not** from a simpler analytical clearance formula, because the
distributed tubular ODE produces systematically lower CLR than the
single-compartment analytical estimate (serial peritubular depletion
across P1→P2→P3, plus passive tubular reabsorption). See
`compound_parameters.py`'s "ROUND 3" notes for the full derivation
and why this distinction matters.

| Compound | Healthy target Css | CKD-5 target Css | Target source |
|---|---|---|---|
| IS | 0.5 mg/L | ~20–37 mg/L (model-dependent) | Vanholder et al. 2014 (healthy); various CKD-5 literature |
| HA | — | — | see `compound_parameters.py` |
| pCS | 0.7 mg/L | — | Sirich et al. 2013 (healthy) |

## Renal blood flow / GFR scaling in CKD-5

- GFR/tubular-flow scale: `CKD_SCALE_FACTORS["ckd5"]` ≈ 0.10 (i.e. CKD-5
  GFR ≈ 10% of healthy)
- Renal blood flow scale: `CKD5_RBF_SCALE` = 0.35 (RBF declines less
  than GFR — physiologically realistic, since filtration capacity is
  lost disproportionately to total renal perfusion)
- Urine flow: scaled independently via `CKD_URINE_SCALE` = 0.50,
  reflecting preserved residual urine output in non-dialysis CKD-5
  rather than a value proportional to the GFR decline (sources: Mair
  et al. 2021, *J Am Soc Nephrol* 32:1000; Hannedouche et al. 2025,
  *Kidney Int Reports*)

Full derivation and rationale: `src/constants.py` module docstring and
`src/physiology.py` module docstring.

## Furosemide oral/IV pharmacokinetic parameters

| Parameter | Value | Source |
|---|---|---|
| Oral bioavailability (F) | 0.60 | FDA label, NDA 016273 (2016) |
| Absorption rate constant (ka) | 1.5 h⁻¹ | Hammarlund et al. 1984, *Eur J Clin Pharmacol* 26:197 |
| Unbound fraction (fu_p, used in some scripts) | 0.05 | Takamura et al. 1997, *Clin Chem* 43:2274 |
| Blood:plasma ratio (BP) | 0.82 | Brater et al. 1980, *Clin Pharmacol Ther* 28:426 |

See `analysis/dose_response_tid_vs_qd.py`'s module docstring for the
complete, currently-maintained list of verified PK references used in
that script specifically.

## Clinical reference values (validation target)

> Costes-Albrespic M et al. *Kidney Int Rep.* 2025;10:2165–2177.

Patients prescribed >120 mg/day furosemide showed (vs. no furosemide):

| Toxin | Δ (clinical) | 95% CI |
|---|---|---|
| Indoxyl sulfate (IS) | +31.9% | [13.5%, 53.3%] |
| p-Cresyl sulfate (pCS) | +29.3% | [6.1%, 57.8%] |

This is the primary external validation target for
`analysis/dose_response_tid_vs_qd.py` and the figures it produces.
