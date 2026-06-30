# Model Overview

This document explains how the pieces of this codebase fit together, in
plain language, before you dive into the code itself. If you are new to
PBPK modeling or new to this specific codebase, read this first.

## The biological question

Furosemide is a commonly prescribed diuretic in chronic kidney disease
(CKD). It is cleared from the body mainly by active renal secretion
through organic anion transporters OAT1 and OAT3, which sit on the
blood-facing (basolateral) side of proximal tubule cells. These same
transporters also clear several uremic toxins that build up in CKD —
indoxyl sulfate (IS), hippuric acid (HA), and p-cresyl sulfate (pCS).

Because furosemide and these toxins compete for the same transporters,
giving furosemide to a CKD patient could reduce toxin clearance and
raise toxin plasma levels — a drug-disease interaction with potential
clinical relevance, since these toxins are implicated in cardiovascular
and other CKD-related morbidity. Costes-Albrespic et al. (2025) reported
clinical evidence for exactly this effect in a large CKD cohort. This
model exists to test whether a mechanistic PBPK simulation, built from
published transporter kinetics and physiology, reproduces that clinical
signal — and if not, to understand where and why it diverges.

## The model's architecture, layer by layer

### Layer 1 — physiological geometry (`constants.py`, `physiology.py`)

The kidney is represented as a sequence of tubular segments (proximal
tubule subsegments P1–P3, loop of Henle LH1–LH2, distal tubule DT,
collecting duct CDC1–CDC5), each with a defined surface area, blood
flow, and tubular fluid flow. These flows differ between "healthy" and
"CKD-5" modes — `physiology.py` builds the actual flow values for each
mode, with glomerular filtration rate (GFR) and renal blood flow (RBF)
scaled independently (CKD doesn't reduce RBF as much as it reduces
filtration capacity — this asymmetry matters for realistic drug
delivery to the transporters).

### Layer 2 — compound properties (`compound_parameters.py`, `compounds.py`)

Every compound (furosemide, IS, HA, pCS) has a set of kinetic parameters:
how strongly it binds plasma proteins (fu_p), how fast OAT1 transports
it (Vmax, Km), how it ionizes at different pH (pKa, type), how much of
it the body produces or how it's eliminated. `compound_parameters.py`
defines these values with literature citations for every single number,
and **keeps an explicit, dated revision history** — if you ever wonder
why a parameter has the value it does, that file's docstring almost
certainly explains it. `compounds.py` is the runtime layer that loads
these definitions and prepares them for the ODE.

### Layer 3 — the ODE itself (`ode_system_twocompartment.py`)

This is the heart of the model: a function `kidney_rhs(t, y)` that, given
the current state of all 142 variables, returns their rates of change.
For each tubular segment, it computes:

- passive diffusion between blood, cell, and lumen compartments
- active OAT1 uptake from blood into cell (Michaelis-Menten kinetics,
  with furosemide raising the *effective* Km for toxins when inhibition
  is active — this is where the drug-toxin interaction actually lives)
- active apical secretion from cell into lumen
- advection of luminal fluid downstream through the nephron

Read this file's own docstring (the first ~50 lines) for the most
complete plain-language explanation in the codebase.

### Layer 4 — running a simulation (`simulation.py`, `state_vector.py`)

`state_vector.py` defines what the 142 numbers in the ODE state vector
actually mean (`IDX["C_tox1_cen"]` etc.) and builds sensible initial
conditions. `simulation.py`'s `run_sim()` is the single entry point
every other script calls: it sets up the physiology and compounds for
the requested mode, builds the initial state, and calls
`scipy.integrate.solve_ivp` with the Radau method (good for the
stiff equations this system produces).

### Layer 5 — analysis and figures (`analysis/`, `figures/`)

Everything in `analysis/` calls `run_sim()` (sometimes many times, e.g.
for steady-state warm-ups followed by a measurement window, or for
parameter sweeps) and turns the results into numbers and figures.
`figures/` scripts are different in kind — they don't run any
simulation themselves, they only read `.npz`/`.pkl` files that
`analysis/` scripts already saved, and build polished publication
figures from that saved data.

## A typical simulation pattern

Most scripts in this repository follow the same basic recipe:

1. **Steady-state warm-up.** Run the model for a long time (500h is the
   common convention, since CKD-5 toxin time constants are tens of
   hours) with no drug, until toxin concentrations stop changing. This
   gives a physiologically realistic starting point — a CKD patient
   doesn't start at zero toxin concentration.
2. **Extract initial conditions** from the end of the warm-up.
3. **Run the actual scenario** (with or without drug, with or without
   inhibition) from those initial conditions, usually for a much
   shorter window (12h–30 days depending on the question).
4. **Compute metrics** — final concentration, area under the curve
   (AUC), realized clearance (Ae/AUC), percent change vs. the no-drug
   baseline.
5. **Compare to literature or clinical data**, often with explicit
   pass/warn flags if the simulated value falls outside a published
   range.

## Where to start reading code

If you want to understand the model from first principles, read in this
order:
1. This document
2. `src/ode_system_twocompartment.py` (just the docstring first)
3. `src/constants.py` (docstring + the CKD scaling section)
4. `src/compound_parameters.py` (skim the revision history — it tells
   the story of how this model was calibrated)
5. `analysis/plot_healthy_lit_comparison_3scenarios.py` (the cleanest,
   most literature-validated example script — run it and look at its
   output figures)

From there, `analysis/dose_response_tid_vs_qd.py` is the most directly
relevant script to the core scientific question this model was built to
answer.
