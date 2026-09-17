"""
compound_parameters.py
=======================
Compound parameter definitions for the kidney PBPK model.

PARAMETER REVISION HISTORY
---------------------------
ROUND 1 (code review):
  - Ki_furo_to_*_mgL:      0.35 mg/L   (calibrated to the Costes-Albrespic 2025
    clinical IS/pCS accumulation signal; used for all three toxins)
  - IS Km_baso_uptake_uM:  20    -> 18 uM        (Deguchi 2004; correct value)
  - CKD apical scales coupled to basal (apical <= basal; OAT1 is rate-limiting)
  - Removed bare print_parameter_changes() at module level

ROUND 2 (fu_p literature recalibration):
  Source: Deltombe/Eloot 2015, Toxins 7:3933, Table 4

  Healthy fu_p corrections (% protein-bound -> fu_p = 1 - %PB/100):
    IS  healthy: 0.04 -> 0.16   (84% bound -> fu_p=0.16; old implied 96%)
    HA  healthy: 0.50 -> 0.66   (34% bound -> fu_p=0.66; old implied 50%)
    pCS healthy: 0.04 -> 0.06   (94% bound -> fu_p=0.06; old implied 96%)

  CKD5 fu_p corrections (binding INCREASES in CKD -> fu_p DECREASES):
    IS  CKD5: 0.10 -> 0.10  (unchanged; 90% bound matches literature)
    HA  CKD5: 0.70 -> 0.52  (48% bound in CKD5; old 0.70 was wrong direction)
    pCS CKD5: 0.08 -> 0.05  (95% bound in CKD5; old 0.08 slightly too high)
    drug CKD5 fu_p: 0.20 -> 0.05  (modest CMPF/oleate displacement; Takamura 1997)

  Furosemide: fu_p INCREASES in CKD (displacement from Site I by oleate/CMPF).
  Toxins: fu_p DECREASES in CKD (MORE bound, because absolute concentration
          rises 10-70x; individual site affinity increases despite site saturation).

ROUND 3 (R0 recalibration against ODE CLR):
  R0 was originally calibrated using a single-compartment analytical formula:
    R0_analytical = CLR_analytical x C_target
    CLR_analytical = GFR*(fu_p/BP) + Vmax_baso*(fu_p/BP*C)/(Km + fu_p/BP*C)

  However the distributed tubular ODE produces a LOWER effective CLR than the
  analytical prediction due to two physical effects:
    1. Serial peritubular depletion: blood is extracted across P1->P2->P3 in
       series, so downstream segments see a lower driving concentration.
    2. Passive tubular reabsorption: concentrated lumen fluid in LH/DT/CDC
       returns some toxin to blood, raising C_ren_exit and reducing net CLR.

  These effects reduce ODE CLR to ~57-64% of analytical CLR. This is physically
  correct for a distributed tubular model. R0 must therefore be calibrated
  against the ACTUAL ODE CLR (read from simulation output), not the formula.

  Final R0 values (calibrated from ODE CLR x clinical C_target):
    IS  healthy: R0 = ODE_CLR(14.0 L/h) x C_target(0.5 mg/L)  = 7.00 mg/h
    HA  healthy: R0 = ODE_CLR(25.0 L/h) x C_target(5.0 mg/L)  = 125.0 mg/h
    pCS healthy: R0 = ODE_CLR(1.7 L/h)  x C_target(0.7 mg/L)  = 1.19 mg/h

    IS  CKD5: R0 = ODE_CLR(0.42 L/h) x C_target(37.0 mg/L) = 15.54 mg/h
    HA  CKD5: R0 = ODE_CLR(0.90 L/h) x C_target(110 mg/L)  = 99.00 mg/h
    pCS CKD5: R0 = ODE_CLR(0.15 L/h) x C_target(24.0 mg/L) = 3.60 mg/h  (unchanged)

ROUND 4 (Volume of distribution added):
  Vd_L added to all compounds. Relevant for transient PK (time to SS),
  NOT for steady-state concentrations (SS = R0/CLR regardless of Vd).
  Literature sources:
    Drug (furosemide): Vd_ss ~ 109 mL/kg x 70 kg = 7.6 L
                       Vd_central = 5 L (= Vcen), Vperi = 40 L
                       (two-compartment; Vd_ss = Vcen + Vperi*k12/k21 = 5+40*0.033/0.5 = 7.6 L)
    IS:   Vd ~ 0.20 L/kg x 70 kg = 14 L   (strongly albumin-bound; low Vd)
          Lim 2022; Sirich 2013 review
    HA:   Vd ~ 0.54 L/kg x 70 kg = 38 L   (moderately bound; higher Vd)
          Niwa 2010; estimated from t1/2 and CLR
    pCS:  Vd ~ 0.15 L/kg x 70 kg = 10 L   (highly bound; very low Vd)
          Meijers 2010; Simonsen 2012 review

KEY LITERATURE SOURCES
-----------------------
  fu_p:     Deltombe O et al. Toxins 2015;7(10):3933. Table 4.
            Viaene L et al. Biopharm Drug Dispos 2013;34(3):165.
            Takamura N et al. Clin Chem 1997;43(12):2274. (drug CKD fu_p)
  Km_baso:  Deguchi T et al. Kidney Int 2004;65(1):162. (IS 18 uM, HA 24 uM)
            Watanabe H et al. Kidney Int 2014;86:840. (pCS 120 uM)
  Ki furo:  0.35 mg/L, calibrated to Costes-Albrespic et al. Kidney Int Rep 2025.
  C_target: Sirich TL et al. JASN 2013;24(12):1887. (IS, pCS clinical SS)
            Deltombe 2015 Table 4. (HA, IS, pCS pre-HD concentrations)
            Vanholder R et al. JASN 2014;25:1897. (IS healthy/CKD review)
  Vd:       Lim YJ et al. Toxins 2022;14:19. (IS Vd)
            Niwa T. J Chromatogr B 2010;878:1421. (HA PK)
            Meijers BK et al. Kidney Int 2010;77:616. (pCS Vd)
            Brater DC. Annu Rev Pharmacol Toxicol 1983;23:45. (furosemide)
"""

from copy import deepcopy
from constants import MW_FUR, MW_IS, MW_HA, MW_pCS


def Km_uM_to_mg_per_L(km_uM: float, mw_g_per_mol: float) -> float:
    """Convert Km from micromolar to mg/L using molecular weight."""
    return float(km_uM * mw_g_per_mol * 1e-3)


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _scale_value(v, s, clamp_nonneg=True):
    out = _safe_float(v, 0.0) * float(s)
    if clamp_nonneg and out < 0.0:
        out = 0.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DRUG — FUROSEMIDE
# ══════════════════════════════════════════════════════════════════════════════
def _create_drug_base():
    return {
        "name":  "furosemide",
        "type":  "acid",
        "MW":    MW_FUR,            # 330.74 g/mol
        "pKa":   3.80,
        "P_app": 5.0e-7,            # apparent permeability [cm/s]

        # ── Protein binding ─────────────────────────────────────────────────
        "BP":      0.82,            # blood-to-plasma ratio (Brater 1983)
        "fu_p":    0.02,            # healthy: ~98% albumin-bound (Site II)
                                    # Andreasen 1981; furosemide label
        "fu_cell": 0.02,            # intracellular unbound fraction (~5x fu_p)

        # ── Volume of distribution ──────────────────────────────────────────
        # Two-compartment: Vd_ss = Vcen + Vperi*(k12/k21)
        #                        = 5.0  + 40.0*(0.033/0.50) = 7.6 L
        # ~109 mL/kg (Brater 1983; Vaska 1995)
        # Vperi and k12/k21 below determine peripheral distribution.
        "Vd_L":    7.6,             # apparent Vd_ss [L] (informational only)
        "Vperi_L": 40.0,            # peripheral compartment volume [L]
        "k12_hinv": 0.033,          # central -> peripheral rate [h-1]
        "k21_hinv": 0.50,           # peripheral -> central rate [h-1]
        "use_tissue": True,

        # ── Elimination ─────────────────────────────────────────────────────
        # t1/2_eff ~ 5.8 h; k_elim = ln(2)/5.8 ≈ 0.12 h-1
        # Source: Brater 1983 Annu Rev Pharmacol; Vaska 1995 Clin Pharmacokinet
        "k_elim_hinv": 0.12,

        # ── Dosing ──────────────────────────────────────────────────────────
        "bolus_mg":  0.0,
        "IR":        60.0,          # infusion rate [mg/h] = 120 mg over 2 h
        "t_inf_h":   2.0,           # infusion duration [h]
        "t_start_h": 0.05,          # infusion start [h]

        # ── OAT1 basolateral uptake ─────────────────────────────────────────
        # Km_OAT1 = 38.90 uM x 330.74/1000 = 12.87 mg/L (Hasannejad 2004)
        "Km_baso_uptake_uM":  38.90,
        "Km_baso_uptake":     12.87,
        "Vmax_baso_uptake":   4500.0,  # ROUND6: 6000->4500 to reduce apical bottleneck
        # OAT1 share of total basolateral Vmax (1.5/2.5); ODE uses this when use_oat3 is False
        "Vmax_OAT1":          2700.0,

        # ── OAT3 basolateral uptake ─────────────────────────────────────────
        # Vmax_OAT3 = Vmax_OAT1 / 1.5 (expression ratio; Scotcher 2016)
        # Km_OAT3 = 7.31 uM x 330.74/1000 = 2.42 mg/L (Hasannejad 2004)
        "Vmax_baso_OAT3": 0.0,  # zero OAT3
        "Km_baso_OAT3": 2.42,  # keep for reference
        "use_oat3": False,  # critical — tells ODE to use OAT1-only path

        # ── Apical secretion ────────────────────────────────────────────────
        "Km_api_secr":   20.0,
        "Vmax_api_secr": 5000.0,  # ROUND6: 2400->5000; unblocks apical bottleneck
        "CL_sec_api":    34.0,

        # ── Reabsorption (negligible for furosemide) ────────────────────────
        "CL_reabs_api":  0.0,
        "CL_reabs_baso": 0.0,

        # ── Intracellular ───────────────────────────────────────────────────
        "CL_int_cell": 1.3e-4,

        "enabled": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOX1 — INDOXYL SULFATE (IS)
# ══════════════════════════════════════════════════════════════════════════════
def _create_tox1_base():
    return {
        "name": "indoxyl_sulfate",
        "type": "acid",
        "MW":   MW_IS,              # 213.17 g/mol
        "pKa":  2.0,
        "P_app": 5e-7,

        # ── Protein binding ─────────────────────────────────────────────────
        # Deltombe 2015, Toxins 7:3933, Table 4
        # Healthy: 84% bound -> fu_p = 0.16
        # CKD5:    90% bound -> fu_p = 0.10 (more bound in CKD; fu_p decreases)
        "BP":      0.70,            # blood-to-plasma ratio
        "fu_p":    0.16,            # healthy unbound fraction
        "fu_cell": 0.20,

        # ── Volume of distribution ──────────────────────────────────────────
        # IS is strongly albumin-bound -> low Vd (~0.20 L/kg)
        # Lim YJ et al. Toxins 2022;14:19; Sirich 2013 review
        # 0.20 L/kg x 70 kg = 14 L
        "Vd_L": 14.0,

        # ── Production rate (R0) ────────────────────────────────────────────
        # Calibrated against ODE CLR (NOT analytical formula).
        # R0 held CONSTANT across healthy and CKD-5 (same production rate).
        # R0 = 5.5 mg/h: calibrated so healthy Css = 5.5/14.0 = 0.39 mg/L
        #   (within literature range 0.4-1.6 mg/L; Vanholder 2014)
        # NOTE: comment previously stated 7.00 — that was a stale draft value.
        "R0":  5.500,
        "C0":  0.5,                 # healthy SS target [mg/L] (Vanholder 2014)

        # ── OAT1 basolateral uptake ─────────────────────────────────────────
        # Km = 18 uM x 213.17/1000 = 3.84 mg/L (Deguchi 2004, Kidney Int 65:162)
        # Old value used wrong MW_IS=279 -> Km=5.02 mg/L (incorrect)
        "Km_baso_uptake_uM":  18.0,
        "Km_baso_uptake":      3.84,
        "Vmax_baso_uptake":  165.0,

        # ── Apical secretion ────────────────────────────────────────────────
        "Km_api_secr":   10.0,
        "Vmax_api_secr": 180.0,
        "CL_sec_api":    0.0,

        # ── Reabsorption ────────────────────────────────────────────────────
        "CL_reabs_api":  0.0,
        "CL_reabs_baso": 0.0,
        "CL_int_cell":   0.0,

        # ── Competitive inhibition by furosemide ────────────────────────────
        # Ki = 0.35 mg/L for all toxins (calibrated to Costes-Albrespic 2025).
        "Ki_furo_to_IS_mgL": 0.35,
        "Ki_IS_to_IS_mgL":   1e9,   # no self-inhibition (modelled via Km sat)
        "Ki_HA_to_IS_mgL":   1e9,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOX2 — HIPPURIC ACID (HA)
# ══════════════════════════════════════════════════════════════════════════════
def _create_tox2_base():
    return {
        "name": "hippuric_acid",
        "type": "acid",
        "MW":   MW_HA,              # 179.17 g/mol
        "pKa":  3.59,
        "P_app": 5e-7,

        # ── Protein binding ─────────────────────────────────────────────────
        # Deltombe 2015, Toxins 7:3933, Table 4
        # Healthy: 34% bound -> fu_p = 0.66
        # CKD5:    48% bound -> fu_p = 0.52 (more bound in CKD)
        "BP":      0.84,
        "fu_p":    0.66,
        "fu_cell": 0.70,

        # ── Volume of distribution ──────────────────────────────────────────
        # HA Vd — REVISED
        # Recomputed from: CLR = 25 L/h (ODE output) × t½ = 0.5 h (Niwa 2010) / ln2
        # = 18.0 L (0.26 L/kg for 70 kg adult)
        # Previous value 38 L (0.54 L/kg) was inconsistent with HA's physicochemical
        # profile (ionised acid, LogP 0.3, pKa 3.6, weak protein binding).
        # Source: Deguchi T et al. J Pharmacol Exp Ther 2005;314(2):932-938. PMID 15879000
        "Vd_L": 18.0,

        # ── Production rate (R0) ────────────────────────────────────────────
        # Calibrated against ODE CLR.
        # ODE CLR = 25.0 L/h (healthy); C_target = 5.0 mg/L
        # R0 = 25.0 x 5.0 = 125.0 mg/h
        # Previous value 195.35 used analytical CLR (39.1 L/h) -> overshot target
        "R0":  70.0,  # Round 4 recal: 125×(3.0/4.35); C_ss was 4.35→target 3.0
        "C0":  3.0,                 # healthy SS target [mg/L] (Deltombe 2015)

        # ── OAT1 basolateral uptake ─────────────────────────────────────────
        # Km = 24 uM x 179.17/1000 = 4.302 mg/L (Deguchi 2004, Kidney Int 65:162)
        # Note: OAT3 does NOT transport HA (Deguchi 2004)
        "Km_baso_uptake_uM":  24.0,
        "Km_baso_uptake":    4.302,
        "Vmax_baso_uptake":  350.0,

        # ── Apical secretion ────────────────────────────────────────────────
        "Km_api_secr":    10.0,
        "Vmax_api_secr": 1100.0,
        "CL_sec_api":    0.0,
        "CL_reabs_api":  0.0,
        "CL_int_cell":   0.0,

        # ── Competitive inhibition by furosemide ────────────────────────────
        "Ki_furo_to_HA_mgL": 0.35,
        "Ki_IS_to_HA_mgL":   1e9,
        "Ki_HA_to_HA_mgL":   1e9,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOX3 — p-CRESYL SULFATE (pCS)
# ══════════════════════════════════════════════════════════════════════════════
def _create_tox3_base():
    return {
        "name": "p_cresyl_sulfate",
        "type": "acid",
        "MW":   MW_pCS,             # 188.19 g/mol
        "pKa":  -2,
        "P_app": 5e-7,

        # ── Protein binding ─────────────────────────────────────────────────
        # Deltombe 2015, Toxins 7:3933, Table 4
        # Healthy: 94% bound -> fu_p = 0.06
        # CKD5:    95% bound -> fu_p = 0.05 (slightly more bound in CKD)
        "BP":      0.55,
        "fu_p":    0.06,
        "fu_cell": 0.20,

        # ── Volume of distribution ──────────────────────────────────────────
        # pCS is highly bound -> very low Vd (~0.15 L/kg)
        # Meijers BK et al. Kidney Int 2010;77:616
        # 0.15 L/kg x 70 kg = 10.5 L ~ 10 L
        "Vd_L": 10.0,

        # ── Production rate (R0) ────────────────────────────────────────────
        # Calibrated against ODE CLR.
        # ODE CLR = 1.7 L/h (healthy); C_target = 0.7 mg/L
        # R0 = 1.7 x 0.7 = 1.19 mg/h
        # Previous value 2.10 used analytical CLR (3.0 L/h) -> overshot target
        "R0":  2.1,    # Round 4 recal: 1.19×(0.7/0.80); C_ss was 0.80→target 0.7
        "C0":  0.7,                 # healthy SS target [mg/L] (Sirich 2013)

        # ── OAT1 basolateral uptake ─────────────────────────────────────────
        # Km = 120 uM x 188.19/1000 = 22.58 mg/L (Watanabe 2014, Kidney Int 86:840)
        "Km_baso_uptake_uM": 120.0,
        "Km_baso_uptake":    22.58,
        "Vmax_baso_uptake":  460.0,

        # ── Apical secretion ────────────────────────────────────────────────
        "Km_api_secr":   35.0,
        "Vmax_api_secr": 400.0,
        "CL_sec_api":    0.0,
        "CL_reabs_api":  0.0,
        "CL_reabs_baso": 0.0,
        "CL_int_cell":   0.0,

        # ── Competitive inhibition by furosemide ────────────────────────────
        "Ki_furo_to_pCS_mgL": 0.35,
        "Ki_IS_to_pCS_mgL":   1e9,
        "Ki_HA_to_pCS_mgL":   1e9,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL BASELINE DICTS
# ──────────────────────────────────────────────────────────────────────────────
DRUG_BASE = _create_drug_base()
TOX1_BASE = _create_tox1_base()
TOX2_BASE = _create_tox2_base()
TOX3_BASE = _create_tox3_base()


# ══════════════════════════════════════════════════════════════════════════════
# CKD-5 VMAX SCALING
# ══════════════════════════════════════════════════════════════════════════════
# Design principle: apical scale <= basal scale for toxins.
# OAT1 (basolateral) is rate-limiting for net tubular secretion.
# Per-compound basal scales reflect OAT1 expression reduction in CKD5:
#   IS  0.05: ~95% OAT1 reduction (Naud 2011 rat; Deguchi 2010 review)
#   HA  0.15: better preserved (secondary substrate; less CKD-sensitive)
#   pCS 0.05: same as IS (OAT1-mediated; similar CKD reduction to IS)
# Drug basal 0.30: less reduced than toxins (OAT3 also contributes)
# Drug apical 0.40: MRP2/MRP4 less affected by CKD than OAT1

_CKD_SCALES = {
    "tox": {
        "baso":   0.050,     # fallback for unlisted toxins
        "apical": 0.150,
    },
    "drug": {
        # OAT1 and OAT3 downregulation in CKD5 matched to IS/pCS (Naud 2011):
        # ~95% reduction. Using 0.05 here gives t½~1.5h in CKD5 vs 0.27h
        # with 0.30.  Source: Naud 2011 Nephrol Dial Transplant 26:2159.
        "baso":   0.05,   # CKD-5 OAT1 ~95% reduction (matched to IS/pCS)
        "apical": 0.15,
    },
    "per_compound": {
        "IS":  {"baso": 0.05, "apical": 0.150},
        "HA":  {"baso": 0.05, "apical": 0.150},
        "pCS": {"baso": 0.05, "apical": 0.150},
    },
}

_VMAX_KEYS = {
    "baso":   ["Vmax_baso_uptake", "Vmax_baso_uptake_pmol_min_per_million",
               "Vmax_baso_OAT3", "Vmax_OAT1"],
    "apical": ["Vmax_api_secr",    "Vmax_api_secr_pmol_min_per_million"],
}


# ══════════════════════════════════════════════════════════════════════════════
# CKD-5 PROTEIN BINDING + R0 ADJUSTMENTS
# ══════════════════════════════════════════════════════════════════════════════
# Source: Deltombe/Eloot 2015, Toxins 7:3933, Table 4
#
# All fu_p values DECREASE from healthy to CKD5 (protein binding increases
# in CKD for toxins, decreases for drug).
#
# R0 values back-calibrated: R0_ckd5 = ODE_CLR_ckd5 x C_target_ckd5
#   ODE CLR read from simulation output (not analytical formula).
#
# fu_cell_ckd5 set proportional to fu_p_ckd5, preserving fu_cell/fu_p ratio.

_CKD_FU_P = {
    # Drug: fu_p INCREASES in CKD (displacement by oleate/CMPF from Site I)
    # healthy 0.02 -> CKD5 0.05 (Takamura 1997, Clin Chem 43:2274)
    "drug": {
        "fu_p":       0.05,    # Takamura 1997: CMPF/oleate displacement
        "fu_cell":    0.05,
        # Hepatic CL is reduced ~80% in CKD5 due to uremic inhibition
        # of CYP2C9 and UGT1A9 (furosemide glucuronidation pathway).
        # healthy k_elim=0.25 h-1 → CKD5 k_elim=0.05 h-1
        # This together with reduced OAT1/OAT3 gives t½~1.5h in CKD5.
        # Sources: Andreasen 1983 Br J Clin Pharmacol 16:391;
        #          Nolin 2008 J Am Soc Nephrol 19:1433 (CYP inhibition in CKD)
        # k_elim left at healthy 0.12 h-1 (no extra CKD override)
    },

    # IS: 84% -> 90% bound; fu_p 0.16 -> 0.10
    # R0_ckd5 = ODE_CLR(0.42) x C_target(37) = 15.54 mg/h
    "IS":  {"fu_p": 0.10, "fu_cell": 0.1250, "R0": 5.5},  # ROUND7: ratio-consistent 0.20*(0.10/0.16)

    # HA: 34% -> 48% bound; fu_p 0.66 -> 0.52
    # R0_ckd5 = ODE_CLR(0.90) x C_target(110) = 99.0 mg/h
    "HA":  {"fu_p": 0.52, "fu_cell": 0.5515, "R0": 70},  # ROUND7: ratio-consistent 0.70*(0.52/0.66)

    # pCS: 94% -> 95% bound; fu_p 0.06 -> 0.05
    # R0_ckd5 = ODE_CLR(0.15) x C_target(24) = 3.60 mg/h
    "pCS": {"fu_p": 0.05, "fu_cell": 0.1667, "R0":  2.10},  # ROUND7: ratio-consistent 0.20*(0.05/0.06)
}


def _create_ckd_adjustments_dynamic(scales: dict) -> dict:
    baseline = {
        "drug": deepcopy(DRUG_BASE),
        "IS":   deepcopy(TOX1_BASE),
        "HA":   deepcopy(TOX2_BASE),
        "pCS":  deepcopy(TOX3_BASE),
    }
    per_compound_scale = scales.get("per_compound", {}) or {}
    out = {}

    for cmpd in ("drug", "IS", "HA", "pCS"):
        src           = baseline[cmpd]
        default_block = scales.get("drug" if cmpd == "drug" else "tox", {})
        sc_baso = float(per_compound_scale.get(cmpd, {}).get(
                        "baso", default_block.get("baso", 1.0)))
        sc_apic = float(per_compound_scale.get(cmpd, {}).get(
                        "apical", default_block.get("apical", 1.0)))
        adj = {}

        for k in _VMAX_KEYS["baso"]:
            if k in src:
                adj[k] = _scale_value(src[k], sc_baso)
        for k in _VMAX_KEYS["apical"]:
            if k in src:
                adj[k] = _scale_value(src[k], sc_apic)
        if "CL_sec_api" in src:
            adj["CL_sec_api"] = _safe_float(src["CL_sec_api"], 0.0)

        for k, v in _CKD_FU_P.get(cmpd, {}).items():
            adj[k] = float(v)

        out[cmpd] = adj

    return out


def set_ckd_vmax_scales(
    baso_scale=None, apical_scale=None, per_compound_scale=None,
    drug_baso_scale=None, drug_apical_scale=None,
):
    """Override CKD-5 Vmax scale factors and rebuild CKD_ADJUSTMENTS."""
    global _CKD_SCALES, CKD_ADJUSTMENTS
    if baso_scale        is not None: _CKD_SCALES["tox"]["baso"]   = float(baso_scale)
    if apical_scale      is not None: _CKD_SCALES["tox"]["apical"] = float(apical_scale)
    if drug_baso_scale   is not None: _CKD_SCALES["drug"]["baso"]  = float(drug_baso_scale)
    if drug_apical_scale is not None: _CKD_SCALES["drug"]["apical"]= float(drug_apical_scale)
    if per_compound_scale is not None: _CKD_SCALES["per_compound"] = per_compound_scale
    CKD_ADJUSTMENTS = _create_ckd_adjustments_dynamic(_CKD_SCALES)


CKD_ADJUSTMENTS = _create_ckd_adjustments_dynamic(_CKD_SCALES)


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def get_compound_parameters(mode: str = "healthy"):
    """
    Return (drug, tox1, tox2, tox3) parameter dicts for the given mode.

    Parameters
    ----------
    mode : "healthy" | "ckd5"

    Returns
    -------
    tuple of 4 dicts: (drug, IS, HA, pCS)
    """
    bases = [
        deepcopy(DRUG_BASE),
        deepcopy(TOX1_BASE),
        deepcopy(TOX2_BASE),
        deepcopy(TOX3_BASE),
    ]
    keys = ["drug", "IS", "HA", "pCS"]

    if mode.lower() in ("ckd5", "ckd-5", "ckd"):
        for base, key in zip(bases, keys):
            base.update(CKD_ADJUSTMENTS[key])
    elif mode.lower() != "healthy":
        raise ValueError(f"Unknown mode: '{mode}'. Must be 'healthy' or 'ckd5'.")

    # Derive Km_baso_uptake [mg/L] from uM where only the uM key is present
    for base in bases:
        if "Km_baso_uptake" not in base and "Km_baso_uptake_uM" in base and "MW" in base:
            base["Km_baso_uptake"] = Km_uM_to_mg_per_L(
                base["Km_baso_uptake_uM"], base["MW"]
            )

    return tuple(bases)


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_compound_summary(mode: str = "healthy"):
    """
    Print a full parameter summary for the given mode.
    Shows all key parameters with units and literature sources.
    """
    drug, IS, HA, pCS = get_compound_parameters(mode)
    W = 68
    label = mode.upper()

    print()
    print("=" * W)
    print(f"  COMPOUND PARAMETER SUMMARY  —  mode: {label}")
    print("=" * W)

    # ── DRUG ────────────────────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  FUROSEMIDE (drug)")
    print(f"  {'─'*W}")
    print(f"  {'MW':<28} {drug['MW']:.2f} g/mol")
    print(f"  {'BP (blood/plasma)':<28} {drug['BP']:.3f}")
    print(f"  {'fu_p (unbound fraction)':<28} {drug['fu_p']:.4f}   "
          f"({'%.0f' % ((1-drug['fu_p'])*100)}% bound)")
    print(f"  {'fu_cell':<28} {drug['fu_cell']:.4f}")
    print(f"  {'Vd_ss [L]':<28} {drug.get('Vd_L', 'n/a')}   "
          f"(central={5.0}L + peri={drug['Vperi_L']}L*{drug['k12_hinv']}/{drug['k21_hinv']})")
    print(f"  {'k_elim [h-1]':<28} {drug['k_elim_hinv']:.4f}  "
          f"(t1/2 = {0.693/drug['k_elim_hinv']:.1f} h)")
    print(f"  {'IR [mg/h]':<28} {drug['IR']:.1f}   "
          f"({drug['t_inf_h']:.0f}h infusion = {drug['IR']*drug['t_inf_h']:.0f} mg total)")
    print(f"  {'Km_OAT1 [mg/L]':<28} {drug['Km_baso_uptake']:.2f}  "
          f"({drug['Km_baso_uptake_uM']:.1f} uM)")
    print(f"  {'Vmax_OAT1 [mg/h]':<28} {drug['Vmax_baso_uptake']:.1f}")
    print(f"  {'Vmax_OAT3 [mg/h]':<28} {drug.get('Vmax_baso_OAT3', 'n/a')}")
    print(f"  {'Km_OAT3 [mg/L]':<28} {drug.get('Km_baso_OAT3', 'n/a')}")

    # ── TOXINS ──────────────────────────────────────────────────────────────
    tox_data = [
        ("INDOXYL SULFATE (IS)",   IS,  "Ki_furo_to_IS_mgL",
         "Sirich 2013: 0.5 mg/L healthy, 37 mg/L CKD5"),
        ("HIPPURIC ACID (HA)",     HA,  "Ki_furo_to_HA_mgL",
         "Deltombe 2015: 5 mg/L healthy, 110 mg/L CKD5"),
        ("p-CRESYL SULFATE (pCS)", pCS, "Ki_furo_to_pCS_mgL",
         "Sirich 2013: 0.7 mg/L healthy, 24 mg/L CKD5"),
    ]

    for tox_name, tox, ki_key, target_note in tox_data:
        print(f"\n  {'─'*W}")
        print(f"  {tox_name}")
        print(f"  {'─'*W}")
        print(f"  {'MW':<28} {tox['MW']:.2f} g/mol")
        print(f"  {'BP (blood/plasma)':<28} {tox['BP']:.3f}")
        print(f"  {'fu_p (unbound fraction)':<28} {tox['fu_p']:.4f}   "
              f"({'%.0f' % ((1-tox['fu_p'])*100)}% bound)")
        print(f"  {'fu_cell':<28} {tox['fu_cell']:.4f}")
        print(f"  {'Vd [L]':<28} {tox.get('Vd_L', 'n/a')}   "
              f"(~{tox.get('Vd_L',0)/70:.2f} L/kg, 70 kg)")
        print(f"  {'R0 [mg/h]':<28} {tox['R0']:.3f}   "
              f"(production rate; calibrated vs ODE CLR)")
        print(f"  {'C_target_SS [mg/L]':<28} {tox['C0']:.2f}   "
              f"({target_note})")
        print(f"  {'Km_OAT1 [mg/L]':<28} {tox['Km_baso_uptake']:.4f}  "
              f"({tox['Km_baso_uptake_uM']:.0f} uM)")
        print(f"  {'Vmax_OAT1 [mg/h]':<28} {tox['Vmax_baso_uptake']:.1f}")
        print(f"  {'Ki_furo [mg/L]':<28} {tox[ki_key]:.3f}   "
              f"(calibrated; Costes-Albrespic 2025)")

    print()
    print("=" * W)
    print(f"  R0 note: calibrated using R0 = ODE_CLR x C_target.")
    print(f"  ODE CLR < analytical CLR due to serial P1-P2-P3 depletion")
    print(f"  and passive reabsorption from downstream tubular segments.")
    print("=" * W)
    print()


def print_parameter_changes():
    """Print a table of all parameters that differ between healthy and CKD5."""
    drug_h, t1_h, t2_h, t3_h = get_compound_parameters("healthy")
    drug_c, t1_c, t2_c, t3_c = get_compound_parameters("ckd5")
    print()
    print("=" * 65)
    print("  PARAMETER CHANGES: CKD5 vs HEALTHY")
    print("=" * 65)
    for name, h, c in [
        ("Drug (furosemide)", drug_h, drug_c),
        ("IS",               t1_h,   t1_c),
        ("HA",               t2_h,   t2_c),
        ("pCS",              t3_h,   t3_c),
    ]:
        print(f"\n  {name}:")
        changed = False
        for k in sorted(h.keys()):
            if k in c and h.get(k) != c.get(k):
                changed = True
                hv, cv = h[k], c[k]
                if isinstance(hv, (int, float)) and hv not in (0, 1e9):
                    pct = (cv - hv) / hv * 100.0
                    print(f"    {k:<30} {hv:>10.4f}  ->  {cv:>10.4f}   ({pct:+.1f}%)")
                else:
                    print(f"    {k:<30} {hv!r:>10}  ->  {cv!r:>10}")
        if not changed:
            print("    (no changes from healthy baseline)")
    print()


def print_healthy_summary():
    """Concise one-screen summary for the healthy baseline (legacy helper)."""
    print_compound_summary("healthy")


def print_ckd5_summary():
    """Concise one-screen summary for the CKD5 mode."""
    print_compound_summary("ckd5")


# ──────────────────────────────────────────────────────────────────────────────
# QUICK SELF-CHECK (runs when file is imported successfully)
# ──────────────────────────────────────────────────────────────────────────────
def _self_check():
    """
    Verify key parameter values at import time AND print a full diagnostic
    so it is immediately visible which file is loaded and what CKD values
    are actually active at runtime.
    """
    import os

    # ── 1. PROVE WHICH FILE IS BEING LOADED ─────────────────────────────────
    this_file = os.path.abspath(__file__)
    print()
    print("=" * 70)
    print("  [COMPOUND_PARAMETERS DIAGNOSTIC]")
    print("=" * 70)
    print(f"  FILE LOADED : {this_file}")
    print(f"  FILE MTIME  : {os.path.getmtime(this_file):.0f}  "
          f"(use os.stat to verify this matches your edited file)")
    print()

    # ── 2. PRINT RAW _CKD_SCALES SO WE SEE WHAT IS ACTUALLY IN MEMORY ───────
    print("  _CKD_SCALES (raw dict in memory right now):")
    print(f"    drug  baso   = {_CKD_SCALES['drug']['baso']}")
    print(f"    drug  apical = {_CKD_SCALES['drug']['apical']}")
    print(f"    tox   baso   = {_CKD_SCALES['tox']['baso']}")
    print(f"    tox   apical = {_CKD_SCALES['tox']['apical']}")
    for cmpd, vals in _CKD_SCALES.get("per_compound", {}).items():
        print(f"    {cmpd:<5} baso={vals.get('baso','?')}  apical={vals.get('apical','?')}")
    print()

    # ── 3. PRINT THE RESULTING CKD5 VMAX VALUES ──────────────────────────────
    d_h, _, _, _  = get_compound_parameters("healthy")
    dc, ISc, HAc, pCSc = get_compound_parameters("ckd5")

    print("  DRUG Vmax healthy → CKD5 (what the ODE actually uses):")
    print(f"    Vmax_baso_uptake : {DRUG_BASE['Vmax_baso_uptake']:.1f} mg/h  →  "
          f"{dc['Vmax_baso_uptake']:.1f} mg/h  "
          f"(scale = {dc['Vmax_baso_uptake']/DRUG_BASE['Vmax_baso_uptake']:.3f})")
    print(f"    Vmax_api_secr    : {DRUG_BASE['Vmax_api_secr']:.1f} mg/h  →  "
          f"{dc['Vmax_api_secr']:.1f} mg/h  "
          f"(scale = {dc['Vmax_api_secr']/DRUG_BASE['Vmax_api_secr']:.3f})")
    print(f"    k_elim_hinv      : {DRUG_BASE['k_elim_hinv']:.3f} h-1  →  "
          f"{dc['k_elim_hinv']:.3f} h-1")
    print(f"    fu_p             : {DRUG_BASE['fu_p']:.3f}  →  {dc['fu_p']:.3f}")
    print()
    print("  TOXIN Vmax healthy → CKD5:")
    for name, base, ckd in [("IS", TOX1_BASE, ISc),
                              ("HA", TOX2_BASE, HAc),
                              ("pCS",TOX3_BASE, pCSc)]:
        sb = ckd["Vmax_baso_uptake"] / base["Vmax_baso_uptake"]
        sa = ckd["Vmax_api_secr"]    / base["Vmax_api_secr"]
        print(f"    {name:<4} baso: {base['Vmax_baso_uptake']:.0f} → "
              f"{ckd['Vmax_baso_uptake']:.1f} (×{sb:.3f})   "
              f"apical: {base['Vmax_api_secr']:.0f} → "
              f"{ckd['Vmax_api_secr']:.1f} (×{sa:.3f})")
    print()

    # ── 4. CRITICAL CHECK: drug baso scale must be 0.05 ─────────────────────
    drug_baso_scale_actual = (dc["Vmax_baso_uptake"] /
                               DRUG_BASE["Vmax_baso_uptake"])
    if abs(drug_baso_scale_actual - 0.05) < 0.005:
        print("  [✓] drug baso scale = 0.05  — paper CKD scale is ACTIVE")
    else:
        print(f"  [✗] drug baso scale = {drug_baso_scale_actual:.3f}  "
              f"— expected 0.05 (paper file).")
        print(f"      You may be loading a CACHED or OLD file.")
        print(f"      Try:  import importlib, compound_parameters")
        print(f"            importlib.reload(compound_parameters)")
        print(f"      Or restart your Python kernel / interpreter.")
    print()

    # ── 5. STANDARD PARAMETER CHECKS ────────────────────────────────────────
    d, IS, HA, pCS = get_compound_parameters("healthy")

    checks = [
        ("IS  fu_p healthy",   IS["fu_p"],                    0.16),
        ("HA  fu_p healthy",   HA["fu_p"],                    0.66),
        ("pCS fu_p healthy",   pCS["fu_p"],                   0.06),
        ("IS  fu_p CKD5",      ISc["fu_p"],                   0.10),
        ("HA  fu_p CKD5",      HAc["fu_p"],                   0.52),
        ("pCS fu_p CKD5",      pCSc["fu_p"],                  0.05),
        ("drug fu_p CKD5",     dc["fu_p"],                    0.05),
        ("drug k_elim CKD5",   dc["k_elim_hinv"],             0.12),
        ("drug Vmax_baso CKD5",dc["Vmax_baso_uptake"],        225.0),  # 4500*0.05
        ("drug Vmax_api  CKD5",dc["Vmax_api_secr"],           750.0),  # 5000*0.15
        ("IS  Ki_furo",        IS["Ki_furo_to_IS_mgL"],       0.35),
        ("HA  Ki_furo",        HA["Ki_furo_to_HA_mgL"],       0.35),
        ("pCS Ki_furo",        pCS["Ki_furo_to_pCS_mgL"],     0.35),
        ("IS  Km_OAT1",        IS["Km_baso_uptake"],          3.84),
        ("pCS pKa",            pCS["pKa"],                   -2.0),
        ("IS  R0",             IS["R0"],                      5.5),
        ("HA  R0",             HA["R0"],                     70.0),
        ("pCS R0",             pCS["R0"],                     2.1),
        ("drug k_elim healthy",d["k_elim_hinv"],              0.12),
    ]

    all_ok = True
    for label, actual, expected in checks:
        ok = abs(float(actual) - float(expected)) < 0.5   # tolerance for Vmax checks
        marker = "✓" if ok else "✗ FAIL"
        if not ok:
            all_ok = False
            print(f"  [{marker}] {label:<28} = {actual}  (expected {expected})")

    if all_ok:
        print("  [✓] All parameter checks passed")
    print("=" * 70)
    print()


_self_check()


__all__ = [
    "DRUG_BASE", "TOX1_BASE", "TOX2_BASE", "TOX3_BASE",
    "CKD_ADJUSTMENTS",
    "get_compound_parameters",
    "set_ckd_vmax_scales",
    "print_compound_summary",
    "print_parameter_changes",
    "print_healthy_summary",
    "print_ckd5_summary",
    "Km_uM_to_mg_per_L",
]
