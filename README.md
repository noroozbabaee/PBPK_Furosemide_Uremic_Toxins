# Kidney PBPK Model — Furosemide & Protein-Bound Uremic Toxins

A physiologically based pharmacokinetic (PBPK) model of the human kidney,
built to study how furosemide (a loop diuretic) competitively inhibits
OAT1/OAT3-mediated renal clearance of protein-bound uremic toxins (PBUTs)
in chronic kidney disease (CKD-5), and how this drug-toxin interaction
compares against the clinical signal reported in Costes-Albrespic et al.
2025 (*Kidney International Reports*).

This repository is shared under an open-science philosophy: every script
here was tested end-to-end (not just read) before being included, every
known bug or limitation is documented rather than hidden, and the goal is
that a new student or collaborator can clone this, run it, and understand
*why* every parameter has the value it does.

---

## What this model represents

The kidney is divided into four anatomical regions along the nephron:

- **Proximal tubule** (P1, P2, P3) — the main site of active drug/toxin
  secretion via OAT1 (basolateral uptake from blood) and MRP/MATE
  (apical secretion into urine), plus passive diffusion.
- **Loop of Henle** (LH1, LH2) — passive diffusion only.
- **Distal tubule** (DT) — passive diffusion only.
- **Collecting duct** (CDC1–CDC5) — passive diffusion only, drains to
  the bladder.

Each segment has three compartments: **vascular** (peritubular blood),
**cellular** (tubule epithelial cytoplasm), and **tubular** (luminal
fluid). The drug (furosemide) and three uremic toxins — indoxyl sulfate
(IS), hippuric acid (HA), and p-cresyl sulfate (pCS) — each have their
own central systemic compartment and move through this distributed
tubular system according to a 142-state ODE system, solved with SciPy's
Radau integrator.

Furosemide competes with the toxins for OAT1-mediated basolateral uptake.
This is modeled as competitive inhibition: furosemide raises the
effective Michaelis constant (Km) for toxin uptake, reducing toxin renal
clearance and causing plasma toxin accumulation — the central hypothesis
this codebase is built to investigate.

For the full biological and mathematical detail, read the docstring at
the top of `src/ode_system_twocompartment.py` first — it is the most
complete single explanation of the model in this repository.

---

## Repository structure

```
src/                          Core PBPK engine (11 files) — the model itself
analysis/                     Runnable simulation scripts (9 files)
analysis/sensitivity/         Parameter sensitivity sweeps (4 files)
figures/                      Pure visualization scripts (2 files) —
                               no ODE dependencies, only read saved results
docs/references/              Source literature PDFs
docs/example_outputs/         Example figure output from an earlier run,
                               kept for reference (not a literature source)
results/                      Run outputs land here (gitignored — empty
                               in the repository, populated when you run
                               scripts)
notebooks/                    (reserved for future student walkthroughs)
```

**Import convention.** Every script in `analysis/`, `analysis/sensitivity/`,
and `figures/` bootstraps its own `sys.path` to find `src/` regardless of
which directory you launch it from:

```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")        # or "..", "..", "src"
                                                          # for analysis/sensitivity/
sys.path.insert(0, _SRC_DIR)
```

This means you can run any script directly (`python analysis/some_script.py`)
from any working directory and it will find the model correctly, as long
as the folder structure itself stays intact.

**Output convention.** Every script writes its outputs (cached `.npz`/
`.pkl` data, CSVs, and figures as PNG+PDF) to the shared top-level
`results/` folder, not a local `figures/` subfolder next to the script.
This means all run outputs accumulate in one place regardless of which
script produced them.

---

## Getting started

```bash
pip install -r requirements.txt
```

Then run any script directly, e.g.:

```bash
python analysis/plot_healthy_lit_comparison_3scenarios.py
python analysis/dose_response_tid_vs_qd.py
```

Outputs (figures, CSVs, cached simulation data) appear in `results/`.

### A note on runtime

CKD-5 toxins have long equilibration time constants (IS tau ≈ 51h in
CKD-5), so most scripts run a **steady-state warm-up** before the actual
measurement window — typically 500h, sometimes much longer (one script
uses 15,000h by deliberate, documented choice — see its module
docstring). This means:

- Lightweight scripts (no warm-up, e.g. `analysis/sensitivity/dose_sensitivity.py`)
  run in well under a minute.
- Scripts with a single 500h warm-up typically run in 1–2 minutes.
- **Sensitivity-sweep and heatmap scripts** in `analysis/` and
  `analysis/sensitivity/` run the warm-up *once per grid point* — for
  example `analysis/heatmap_ki_vmax_sensitivity_ode.py` has a 15×12 grid
  × 2 toxins = 360 individual 500h-warm-up simulations. These can take
  **hours**, not minutes. They all checkpoint their results to a cache
  file (`.npz` or `.pkl`) every few points, so they can be safely
  interrupted and resumed, and a second run will reuse cached points
  rather than recomputing them.

If you only want to confirm a script works before committing to a full
run, every script's parameters (warm-up hours, grid size, number of days)
are plain module-level constants near the top of the file — temporarily
lowering them gives you a fast smoke test.

---

## Module map

### `src/` — core engine

| File | Role |
|---|---|
| `constants.py` | Geometry, physiology, CKD scaling factors, unit conversions — single source of truth for non-compound parameters |
| `helpers.py` | Unit conversions, acid/base ionization fractions, small utilities |
| `passive_diffusion.py` | Passive tubular diffusion clearances per segment and membrane side |
| `compound_parameters.py` | All compound PK parameters (Vmax, Km, Ki, fu_p, R0) with an extensive, dated revision history explaining every correction and its literature source |
| `compounds.py` | Runtime bridge: loads parameters from `compound_parameters.py`, builds passive-diffusion bundles, exposes `drug`/`tox1`/`tox2`/`tox3` dicts that the ODE reads directly |
| `parameter_audit.py` | Self-check system — compares live parameter values against expected values at simulation start and prints a pass/warn/error table |
| `state_vector.py` | Defines the 142-state ODE vector and its indexing (`IDX`), plus initial-condition construction |
| `physiology.py` | Builds the `phys` dict (GFR, renal blood flow, urine flow, tubular segment flows) with CKD-5 scaling decoupled between GFR and RBF |
| `ode_system_twocompartment.py` | The core ODE right-hand-side (`kidney_rhs`) — read this file's docstring first to understand the model's anatomy |
| `simulation.py` | `run_sim()` — the solver wrapper (SciPy Radau) that every other script calls to actually integrate the ODE |
| `scenario_manager.py` | Named-scenario convenience layer, e.g. `set_scenario("CKD_WITH_DRUG")` configures mode + drug/inhibition flags in one call |

### `analysis/` — runnable simulation scripts

| File | What it produces |
|---|---|
| `dose_response_tid_vs_qd.py` | Compares TID (three-times-daily) vs QD (once-daily) oral furosemide dosing at the same total daily dose; includes both a theoretical closed-form analytical prediction and an ODE-fed diagnostic analytical, alongside the full ODE simulation and the clinical reference |
| `furosemide_iv_pk_validation.py` | Validates IV furosemide PK (Cmax, t½, CL_R, AUC) in healthy and CKD-5 against published literature targets |
| `furosemide_iv_pk_debug.py` | Same IV PK run with verbose internal state snapshots for debugging |
| `plot_furosemide_pk_ckd_vs_healthy.py` | Healthy vs CKD-5 furosemide PK comparison; also used as a library by the debug script above (imports `_compute_pk_metrics` / `validate_against_literature`) |
| `plot_healthy_lit_comparison_3scenarios.py` | Three-scenario comparison (No Drug / Drug No-Interaction / Drug+Inhibition) in healthy mode against literature toxin concentrations and clearances |
| `plot_ckd_lit_comparison_3scenarios.py` | Same three-scenario comparison in CKD-5 mode |
| `plot_clearance_flux_decomposition.py` | Decomposes total renal clearance into OAT1-active vs passive-diffusion vs filtration contributions, across 4 scenarios (healthy/CKD-5 × no-drug/with-drug) |
| `heatmap_fu_sensitivity_ode.py` | 2D sensitivity grid: furosemide plasma protein binding (fu_p) × toxin fu_p → toxin accumulation, using full 30-day chronic-dosing ODE simulations at every grid point |
| `heatmap_ki_vmax_sensitivity_ode.py` | 2D sensitivity grid: OAT1 inhibition constant (Ki) × CKD-5 OAT1 Vmax scale → toxin accumulation; this is the single source of truth for this physics (see `figures/heatmap_ki_vmax_from_cache.py`) |

### `analysis/sensitivity/` — one-parameter-at-a-time sweeps

| File | Sweeps |
|---|---|
| `ki_sensitivity_per_toxin.py` | OAT1 inhibition constant (Ki), one figure per toxin (3 separate figures: IS, HA, pCS) |
| `fu_sensitivity.py` | Plasma protein binding (fu_p) for the drug and each toxin independently |
| `dose_sensitivity.py` | Furosemide dose level (0.1×–10× baseline), the lightest script — no steady-state warm-up needed |
| `r0_sensitivity.py` | Toxin generation rate (R0), in both healthy and CKD-5 modes, with the most extensively documented bug-fix history of any script in this repository (see its module docstring) |

### `figures/` — pure visualization (no ODE dependencies)

| File | Role |
|---|---|
| `make_three_way_comparison_figure.py` | Publication 4-panel figure: ODE vs theoretical-Analytical vs Clinical comparison, reads `.npz` results produced by `dose_response_tid_vs_qd.py` |
| `heatmap_ki_vmax_from_cache.py` | Publication-styled Ki×Vmax heatmap (horizontal colorbar layout); imports `compute_heatmap()` directly from `analysis/heatmap_ki_vmax_sensitivity_ode.py` rather than duplicating the simulation logic — both scripts share one cache file |

---

## Known limitations and open questions

This section exists because open science means documenting what is
*not* settled, not just what works. These are genuine, unresolved
items as of this repository's last update:

1. **Ki (OAT1 inhibition constant) is set to 0.35 mg/L** throughout the
   codebase, calibrated to reproduce the Costes-Albrespic 2025 clinical
   signal. The literature-derived reference value via the Cheng-Prusoff
   correction from Mihaila et al. 2020 (fluorescein IC50 assay) is
   **3.77 mg/L** — roughly 10x higher. `compound_parameters.py`'s
   internal self-check and `compounds.py`'s documented fallback constant
   both flag 0.35 as a deviation from that literature value. This is a
   **deliberate calibration choice**, not an oversight — but it means
   results should be interpreted as "tuned to match the clinical
   endpoint" rather than "predicted from first-principles in-vitro
   transporter kinetics." If you change Ki, do so deliberately and
   re-validate against the `analysis/plot_*_lit_comparison_3scenarios.py`
   scripts.

2. **`furosemide_iv_pk_validation.py`'s own literature-validation check**
   shows CKD-5 Cmax, t½, and AUC outside the cited literature target
   ranges for the IV dosing
   protocol it tests. This is a known, currently-unresolved calibration
   gap in the IV pharmacokinetics specifically — separate from the oral
   chronic-dosing results in `dose_response_tid_vs_qd.py`, which are
   the primary focus of this work and are better validated.

3. **`analysis/plot_ckd_lit_comparison_3scenarios.py`'s production
   default uses a 15,000-hour steady-state warm-up** — far longer than
   the ~500h convention used everywhere else in this codebase (which
   itself already represents >99.9% convergence for CKD-5 toxin
   kinetics). This was a deliberate choice by that script's author;
   it has not been reconciled with the shorter convention elsewhere.

4. **The `plotting` module referenced by some scripts as an optional
   house-style helper does not exist in this package.** Every script
   that references it (`analysis/plot_clearance_flux_decomposition.py`,
   `analysis/plot_ckd_lit_comparison_3scenarios.py`) has been fixed to
   treat it as optional and falls back to plain matplotlib styling —
   you do not need to supply this module for anything to work, but
   figures will use default (not custom house-style) formatting.

5. **Several scripts were originally written assuming a
   `scenario_manager.set_scenario()` signature that does not match this
   codebase's actual implementation** (`set_scenario(scenario_key: str)`,
   a single named preset, not the `mode=`/`drug_enabled=`/`inhibit=`
   keyword arguments some earlier script versions expected). All such
   calls have been fixed in every script included here — but if you
   bring in additional scripts from elsewhere, check this signature
   mismatch before trusting them.

---



## License

See `LICENSE`.
