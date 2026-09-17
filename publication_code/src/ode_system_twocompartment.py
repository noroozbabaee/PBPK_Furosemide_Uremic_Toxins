"""
ode_system_twocompartment.py
============================
ODE right-hand side for the kidney PBPK system.

WHAT THIS FILE DOES
-------------------
This file defines the mathematics that describe how drug and toxin
concentrations change over time inside the kidney and the body.
It is the "engine" of the simulation — the ODE solver calls kidney_rhs()
thousands of times per simulation to advance the solution forward in time.

BIOLOGICAL OVERVIEW
-------------------
The model represents one kidney divided into four anatomical regions:

  Proximal tubule (P1, P2, P3)  — 3 subsegments
      The main site of active drug/toxin secretion.
      OAT1 transporters on the basolateral (blood-facing) membrane
      move drugs and toxins FROM the peritubular blood INTO the
      tubule cell. MRP/MATE transporters on the apical (lumen-facing)
      membrane then move them INTO the tubular fluid.
      Passive diffusion also occurs in both directions.

  Loop of Henle (LH1, LH2)       — 2 subsegments (thin descending, thick ascending)
  Distal tubule  (DT)           — 1 segment
  Collecting duct (CDC1–CDC5)   — 5 subsegments
      These segments have NO active transport.
      Only passive diffusion occurs here (reabsorption back into blood).
      The final CDC5 drains into the bladder.

Each segment has THREE compartments:
  Vascular  (V)  — peritubular capillary blood flowing alongside the tubule
  Cellular  (C)  — epithelial cell cytoplasm (the tubule wall)
  Tubular   (T)  — the fluid inside the tubule lumen

SYSTEMIC COMPARTMENTS
---------------------
The drug also has:
  Central compartment (C_ven)    — plasma/blood pool (e.g. 5 L for furosemide)
  Peripheral compartment (C_peri)— slowly-equilibrating tissue (muscle, fat)

Toxins have their own central compartment (C_tox*_cen) driven by a constant
production rate R0 representing ongoing metabolism.

WHAT COMPOUNDS ARE IN THE MODEL
-------------------------------
  drug  = furosemide (the drug whose effect we study)
  tox1  = indoxyl sulfate (IS)   — uremic toxin
  tox2  = hippuric acid (HA)     — uremic toxin
  tox3  = p-cresyl sulfate (pCS) — uremic toxin

GLOBAL CONVENTIONS (used throughout this file)
----------------------------------------------
  Concentrations : mg/L   (total, bound + unbound)
  Flows          : L/h
  Clearances     : L/h
  Vmax           : mg/h   (maximum transport rate, per kidney)
  Km             : mg/L   (Michaelis constant; see note below on Ki)
  Ki             : mg/L   (competitive inhibition constant)
  Time           : h
  Volumes        : L

BOUND vs. UNBOUND CONCENTRATIONS
---------------------------------
All transporter equations use UNBOUND drug concentrations as the
driving force — only the unbound fraction can interact with transporters.

  Cu_vascular = C_vascular * (fu_p / BP)
      fu_p = unbound fraction in plasma  (0 to 1)
      BP   = blood-to-plasma ratio       (≥ 1)

  Cu_cellular = C_cellular * fu_cell

SIGN CONVENTIONS FOR FLUXES
-----------------------------
  Positive flux = enters the compartment
  Negative flux = leaves the compartment

  Basolateral PD flux (pd_b): positive = cell → vascular
      → Add +pd_b to vascular ODE
      → Add −pd_b to cellular ODE

  Apical PD flux (pd_a): positive = lumen → cell
      → Add +pd_a to cellular ODE
      → Add −pd_a to lumen ODE

  OAT1+OAT3 uptake: positive = vascular → cell (furosemide only)
      OAT3 is added alongside OAT1 for furosemide basolateral uptake.
      Km_OAT3 = 2.42 mg/L (7.31 µM, Hasannejad 2004).
      Vmax_OAT3 = Vmax_OAT1 / 1.5 (Scotcher 2016 expression ratio).
      Toxins use OAT1 only (see _baso_uptake_tox).

  Apical secretion: positive = cell → lumen
      → Subtract from cellular ODE (cell loses drug/toxin)
      → Add to lumen ODE (lumen gains drug/toxin)

STRUCTURE OF THIS FILE
-----------------------
  TissueDistribution class
      Handles the two-compartment (central + peripheral) distribution.

  drug_input_conc_rate()
      Computes the IV infusion rate of drug into the central compartment.

  _pd_baso()       — basolateral passive diffusion flux helper
  _pd_apic()       — apical passive diffusion flux helper
  _apical_secretion()  — active apical secretion (MRP/MATE transporters)
  _baso_uptake_tox()   — active basolateral OAT1 uptake for toxins

  _vascular_fluxes_prox()     — vascular ODE terms for proximal segments
  _vascular_fluxes_nonprox()  — vascular ODE terms for non-proximal segments
  _cellular_fluxes_prox()     — cellular ODE terms for proximal segments
  _cellular_fluxes_nonprox()  — cellular ODE terms for non-proximal segments
  _lumen_fluxes_prox()        — lumen ODE terms for proximal segments
  _lumen_fluxes_nonprox()     — lumen ODE terms for non-proximal segments

  kidney_rhs()     — MAIN ODE function (calls all helpers above)

EXECUTION ORDER IN kidney_rhs() (CORRECTED)
-------------------------------------------
  Steps 1–11   : Setup (compounds, volumes, flags, inhibition)
  Steps 15–17  : Vascular compartment loops (prox, then non-prox+CDS)
                 → This computes C_ren_exit_* (true renal venous outflow)
  Step 12      : Central compartment ODE (drug) — uses C_ren_exit_drug
  Step 13      : Debug prints for central/tissue
  Step 14      : Toxin central ODEs — uses C_ren_exit_t*
  Steps 18–19  : Cellular compartment loops
  Steps 20–21  : Lumen compartment loops
  Step 22      : Bladder/excretion update
  Step 23      : Return derivative vector

CRITICAL FIX (RENAL EXCHANGE):
  The vascular loops (Steps 15-17) MUST execute BEFORE the central ODE (Step 12)
  because the true renal venous outflow (C_ren_exit_drug) is needed to compute
  the correct renal exchange term. Using C_CDC5_V instead gives the wrong
  mass balance and underestimates renal clearance by ~50-80% in CKD.
"""

import math
import numpy as np

from constants import (
    EPS,
    PROX,
    NONPROX,
    CDS,
    V_SEG,
    USE_APICAL_MM,
)
import constants as K
from helpers import drug_baso_uptake_flux_unb
from state_vector import IDX
import compounds as C


# ══════════════════════════════════════════════════════════════════════════════
# OAT3 toggle wrapper for drug basolateral uptake
# ══════════════════════════════════════════════════════════════════════════════

_FIX6_WARNED = False


def _drug_baso_flux(drug: dict, Cu_drg_v: float, seg: str, EPS: float) -> float:
    if not hasattr(_drug_baso_flux, "_printed_path"):
        print(f"[_drug_baso_flux] use_oat3={drug.get('use_oat3', 'KEY MISSING')}, "
              f"Vmax_OAT1={drug.get('Vmax_OAT1','MISSING')}, "
              f"Vmax_baso_uptake={drug.get('Vmax_baso_uptake','MISSING')}")
        _drug_baso_flux._printed_path = True
    """
    Basolateral uptake flux for the DRUG [mg/h per unit weight].

    Delegates to drug_baso_uptake_flux_unb (OAT1 + OAT3 combined, from
    helpers.py) unless drug["use_oat3"] is explicitly False, in which case
    a pure OAT1 Michaelis-Menten flux is returned instead.
    """
    if drug.get("use_oat3", True):
        return drug_baso_uptake_flux_unb(Cu_drg_v, seg)

    Km_OAT1 = float(drug.get("Km_OAT1", drug.get("Km_baso_uptake", 1.0)))
    if "Vmax_OAT1" in drug:
        Vmax_OAT1 = float(drug["Vmax_OAT1"])
    else:
        Vmax_total = float(drug.get("Vmax_baso_uptake", 0.0))
        Vmax_OAT1 = Vmax_total * (1.5 / 2.5)
        global _FIX6_WARNED
        if not _FIX6_WARNED:
            print(f"\n[FIX #6 WARNING] drug['Vmax_OAT1'] not set.\n"
                  f"  Deriving from Vmax_baso_uptake={Vmax_total:.4e} × 0.60 "
                  f"= {Vmax_OAT1:.4e} mg/h.")
            _FIX6_WARNED = True
    return Vmax_OAT1 * Cu_drg_v / (Km_OAT1 + Cu_drg_v + EPS)


# ══════════════════════════════════════════════════════════════════════════════
# Module-level segment weights
# ══════════════════════════════════════════════════════════════════════════════

_RAW_BASO = {"P1": 0.40, "P2": 0.40, "P3": 0.20}
_RAW_APIC = {"P1": 0.40, "P2": 0.40, "P3": 0.20}
_s_b = sum(_RAW_BASO.values())
_s_a = sum(_RAW_APIC.values())

W_BASO = {k: _RAW_BASO[k] / _s_b for k in _RAW_BASO}
W_APIC = {k: _RAW_APIC[k] / _s_a for k in _RAW_APIC}


# ══════════════════════════════════════════════════════════════════════════════
# TissueDistribution — two-compartment systemic distribution for the drug
# ══════════════════════════════════════════════════════════════════════════════

class TissueDistribution:
    """
    Peripheral tissue compartment for the drug (two-compartment PK model).
    """

    def __init__(self, drug_dict: dict, Vcen: float):
        self.Vcen = float(Vcen)
        self.Vperi = float(drug_dict.get("Vperi_L", 0.0))
        self.k12 = float(drug_dict.get("k12_hinv", 0.0))
        self.k21 = float(drug_dict.get("k21_hinv", 0.0))
        self.use_tissue = bool(drug_dict.get("use_tissue", True))

        self.enabled = (
                self.use_tissue
                and self.Vcen > 0.0
                and self.Vperi > 0.0
                and self.k12 > 0.0
                and self.k21 > 0.0
        )

    def central_contribution(self, C_cen: float, C_peri: float) -> float:
        if not self.enabled:
            return 0.0
        to_peri = self.k12 * C_cen
        from_peri = self.k21 * (self.Vperi / max(self.Vcen, 1e-12)) * C_peri
        return -to_peri + from_peri

    def peripheral_rhs(self, C_cen: float, C_peri: float) -> float:
        if not self.enabled:
            return 0.0
        to_peri = self.k12 * (self.Vcen / max(self.Vperi, 1e-12)) * C_cen
        from_peri = self.k21 * C_peri
        return to_peri - from_peri

    def amount_total(self, C_cen: float, C_peri: float) -> float:
        return self.Vcen * C_cen + self.Vperi * C_peri

    def __repr__(self) -> str:
        if self.enabled:
            return (f"TissueDistribution(Vcen={self.Vcen:.2f} L, "
                    f"Vperi={self.Vperi:.2f} L, "
                    f"k12={self.k12:.4f} 1/h, k21={self.k21:.4f} 1/h)")
        return (f"TissueDistribution(DISABLED — use_tissue={self.use_tissue}, "
                f"Vcen={self.Vcen:.2f}, Vperi={self.Vperi:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# Drug infusion rate helper
# ══════════════════════════════════════════════════════════════════════════════

def drug_input_conc_rate(
        t_h: float,
        Vcen_L: float,
        Vb_L: float,
        Qc_Lph: float,
        d: dict,
        protocol: str,
) -> float:
    if not d.get("enabled", False):
        return 0.0

    IR = float(d.get("IR", 0.0))
    t0 = float(d.get("t_start_h", 0.0))
    T = float(d.get("t_inf_h", 0.0))

    if IR <= 0.0 or not (t0 <= t_h < t0 + T):
        return 0.0

    if protocol == "nina":
        mix = 1.0 - math.exp(-Qc_Lph * (t_h - t0) / max(Vb_L, 1e-12))
        return (IR / max(Vcen_L, 1e-12)) * mix

    return IR / max(Vcen_L, 1e-12)


def debug_drug_infusion(t_h: float, d: dict) -> bool:
    if 399.5 < t_h < 400.5:
        print(f"[DEBUG drug_infusion] t={t_h:.3f}h, enabled={d.get('enabled', False)}, "
              f"IR={d.get('IR', 0.0)}, t0={d.get('t_start_h', 0.0)}, "
              f"T={d.get('t_inf_h', 0.0)}, bolus={d.get('bolus_mg', 0.0)}")
        if not d.get("enabled", False):
            print("  -> NOT INFUSING: drug not enabled")
            return False
        IR = float(d.get("IR", 0.0))
        t0 = float(d.get("t_start_h", 0.0))
        T = float(d.get("t_inf_h", 0.0))
        if IR <= 0.0:
            print(f"  -> NOT INFUSING: IR={IR} <= 0")
            return False
        if not (t0 <= t_h < t0 + T):
            print(f"  -> NOT INFUSING: t={t_h} not in [{t0}, {t0 + T})")
            return False
        print(f"  -> SHOULD INFUSE: IR={IR} mg/h")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Low-level flux helpers — pure functions
# ══════════════════════════════════════════════════════════════════════════════

def _pd_baso(
        cmpd_pass: dict,
        prefix: str,
        seg: str,
        C_V: float,
        C_C: float,
        cmpd: dict,
        EPS: float,
) -> float:
    CLbs = cmpd_pass[f"{prefix}_CL_PD_{seg}_basolateral_no_ui"]
    beta = cmpd_pass[f"{prefix}_beta_{seg}"]
    gamma = cmpd_pass[f"{prefix}_gamma_plasma"]
    fu_BP = cmpd["fu_p"] / max(cmpd["BP"], EPS)
    return CLbs * (C_C * cmpd["fu_cell"] * beta - C_V * fu_BP * gamma)


def _pd_apic(
        cmpd_pass: dict,
        prefix: str,
        seg: str,
        C_C: float,
        C_T: float,
        cmpd: dict,
        EPS: float,
) -> float:
    CLap = cmpd_pass[f"{prefix}_CL_PD_{seg}_apical_no_ui"]
    beta = cmpd_pass[f"{prefix}_beta_{seg}"]
    alpha = cmpd_pass[f"{prefix}_alpha_{seg}"]
    return CLap * (C_T * alpha - C_C * cmpd["fu_cell"] * beta)


def _apical_secretion(
        cmpd: dict,
        seg: str,
        C_C: float,
        use_mm: bool,
        w_ap: float,
        EPS: float,
        enabled: bool = True,
) -> float:
    if not enabled:
        return 0.0
    Cu_cell = C_C * cmpd["fu_cell"]
    if use_mm:
        return (cmpd["Vmax_api_secr"] * w_ap * Cu_cell
                / (cmpd["Km_api_secr"] + Cu_cell + EPS))
    return cmpd["CL_sec_api"] * Cu_cell


def _baso_uptake_tox(
        cmpd: dict,
        Cu_v: float,
        Km_eff: float,
        w_baso: float,
        EPS: float,
) -> float:
    return w_baso * cmpd["Vmax_baso_uptake"] * Cu_v / (Km_eff + Cu_v + EPS)


# ══════════════════════════════════════════════════════════════════════════════
# Vascular compartment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _vascular_fluxes_prox(
        seg: str,
        dx: dict,
        S: dict,
        drug: dict, tox1: dict, tox2: dict, tox3: dict,
        DRG_PASS: dict, TOX1_PASS: dict, TOX2_PASS: dict, TOX3_PASS: dict,
        C_in_drug: float, C_in_t1: float, C_in_t2: float, C_in_t3: float,
        Qkid: float,
        Km_IS_eff: float, Km_HA_eff: float, Km_pCS_eff: float,
        Vv: float,
        EPS: float,
) -> tuple:
    w_baso = W_BASO[seg]

    Cu_drg_v = S[f"C_{seg}_V"] * (drug["fu_p"] / max(drug["BP"], EPS))
    Cu_IS_v = S[f"C_tox1_{seg}_V"] * (tox1["fu_p"] / max(tox1["BP"], EPS))
    Cu_HA_v = S[f"C_tox2_{seg}_V"] * (tox2["fu_p"] / max(tox2["BP"], EPS))
    Cu_pCS_v = S[f"C_tox3_{seg}_V"] * (tox3["fu_p"] / max(tox3["BP"], EPS))

    flow_d = Qkid * (C_in_drug - S[f"C_{seg}_V"])
    flow_1 = Qkid * (C_in_t1 - S[f"C_tox1_{seg}_V"])
    flow_2 = Qkid * (C_in_t2 - S[f"C_tox2_{seg}_V"])
    flow_3 = Qkid * (C_in_t3 - S[f"C_tox3_{seg}_V"])

    J_drg = w_baso * _drug_baso_flux(drug, Cu_drg_v, seg, EPS)
    J_IS = _baso_uptake_tox(tox1, Cu_IS_v, Km_IS_eff, w_baso, EPS)
    J_HA = _baso_uptake_tox(tox2, Cu_HA_v, Km_HA_eff, w_baso, EPS)
    J_pCS = _baso_uptake_tox(tox3, Cu_pCS_v, Km_pCS_eff, w_baso, EPS)

    Reabs_drg = drug["CL_reabs_baso"] * S[f"C_{seg}_C"] * drug["fu_cell"]

    pd_d_b = _pd_baso(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_V"], S[f"C_{seg}_C"], drug, EPS)
    pd_1_b = _pd_baso(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_V"], S[f"C_tox1_{seg}_C"], tox1, EPS)
    pd_2_b = _pd_baso(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_V"], S[f"C_tox2_{seg}_C"], tox2, EPS)
    pd_3_b = _pd_baso(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_V"], S[f"C_tox3_{seg}_C"], tox3, EPS)

    dx[f"C_{seg}_V"] += (flow_d + pd_d_b - J_drg + Reabs_drg) / max(Vv, EPS)
    dx[f"C_tox1_{seg}_V"] += (flow_1 + pd_1_b - J_IS) / max(Vv, EPS)
    dx[f"C_tox2_{seg}_V"] += (flow_2 + pd_2_b - J_HA) / max(Vv, EPS)
    dx[f"C_tox3_{seg}_V"] += (flow_3 + pd_3_b - J_pCS) / max(Vv, EPS)

    return (S[f"C_{seg}_V"], S[f"C_tox1_{seg}_V"],
            S[f"C_tox2_{seg}_V"], S[f"C_tox3_{seg}_V"])


def _vascular_fluxes_nonprox(
        seg: str,
        dx: dict,
        S: dict,
        drug: dict, tox1: dict, tox2: dict, tox3: dict,
        DRG_PASS: dict, TOX1_PASS: dict, TOX2_PASS: dict, TOX3_PASS: dict,
        C_in_drug: float, C_in_t1: float, C_in_t2: float, C_in_t3: float,
        Qkid: float,
        Vv: float,
        EPS: float,
) -> tuple:
    flow_d = Qkid * (C_in_drug - S[f"C_{seg}_V"])
    flow_1 = Qkid * (C_in_t1 - S[f"C_tox1_{seg}_V"])
    flow_2 = Qkid * (C_in_t2 - S[f"C_tox2_{seg}_V"])
    flow_3 = Qkid * (C_in_t3 - S[f"C_tox3_{seg}_V"])

    pd_d_b = _pd_baso(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_V"], S[f"C_{seg}_C"], drug, EPS)
    pd_1_b = _pd_baso(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_V"], S[f"C_tox1_{seg}_C"], tox1, EPS)
    pd_2_b = _pd_baso(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_V"], S[f"C_tox2_{seg}_C"], tox2, EPS)
    pd_3_b = _pd_baso(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_V"], S[f"C_tox3_{seg}_C"], tox3, EPS)

    dx[f"C_{seg}_V"] += (flow_d + pd_d_b) / max(Vv, EPS)
    dx[f"C_tox1_{seg}_V"] += (flow_1 + pd_1_b) / max(Vv, EPS)
    dx[f"C_tox2_{seg}_V"] += (flow_2 + pd_2_b) / max(Vv, EPS)
    dx[f"C_tox3_{seg}_V"] += (flow_3 + pd_3_b) / max(Vv, EPS)

    return (S[f"C_{seg}_V"], S[f"C_tox1_{seg}_V"],
            S[f"C_tox2_{seg}_V"], S[f"C_tox3_{seg}_V"])


# ══════════════════════════════════════════════════════════════════════════════
# Cellular compartment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cellular_fluxes_prox(
        seg: str,
        dx: dict,
        S: dict,
        drug: dict, tox1: dict, tox2: dict, tox3: dict,
        DRG_PASS: dict, TOX1_PASS: dict, TOX2_PASS: dict, TOX3_PASS: dict,
        Km_IS_eff: float, Km_HA_eff: float, Km_pCS_eff: float,
        APICAL_MM_DRUG: bool, APICAL_MM_T1: bool, APICAL_MM_T2: bool, APICAL_MM_T3: bool,
        drug_transport_enabled: bool,
        Vc: float,
        EPS: float,
):
    w_b = W_BASO[seg]
    w_ap = W_APIC[seg]

    pd_b_d = _pd_baso(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_V"], S[f"C_{seg}_C"], drug, EPS)
    pd_a_d = _pd_apic(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_C"], S[f"C_{seg}_T"], drug, EPS)
    Cu_drg_v = S[f"C_{seg}_V"] * (drug["fu_p"] / max(drug["BP"], EPS))
    J_d_baso = w_b * _drug_baso_flux(drug, Cu_drg_v, seg, EPS)

    _clrb = float(drug.get("CL_reabs_baso", 0.0))
    _use_active = bool(drug.get("use_active_baso_reabs", False))
    if _clrb > 0.0 and not _use_active:
        Reabs_d = 0.0
    else:
        Reabs_d = -_clrb * S[f"C_{seg}_C"] * drug["fu_cell"]

    Sec_d = -_apical_secretion(drug, seg, S[f"C_{seg}_C"],
                               APICAL_MM_DRUG, w_ap, EPS, drug_transport_enabled)
    Reab_api_d = drug["CL_reabs_api"] * S[f"C_{seg}_T"]
    Met_d = -drug["CL_int_cell"] * S[f"C_{seg}_C"] * drug["fu_cell"]

    dx[f"C_{seg}_C"] += (-pd_b_d + pd_a_d + J_d_baso + Reabs_d + Sec_d + Reab_api_d + Met_d) / max(Vc, EPS)

    # IS
    pd_b_1 = _pd_baso(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_V"], S[f"C_tox1_{seg}_C"], tox1, EPS)
    pd_a_1 = _pd_apic(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_C"], S[f"C_tox1_{seg}_T"], tox1, EPS)
    Cu_IS_v = S[f"C_tox1_{seg}_V"] * (tox1["fu_p"] / max(tox1["BP"], EPS))
    J_IS = _baso_uptake_tox(tox1, Cu_IS_v, Km_IS_eff, w_b, EPS)
    Sec_IS = -_apical_secretion(tox1, seg, S[f"C_tox1_{seg}_C"],
                                APICAL_MM_T1, w_ap, EPS)
    Met_IS = -tox1["CL_int_cell"] * S[f"C_tox1_{seg}_C"] * tox1["fu_cell"]

    dx[f"C_tox1_{seg}_C"] += (-pd_b_1 + pd_a_1 + J_IS + Sec_IS + Met_IS) / max(Vc, EPS)

    # HA
    pd_b_2 = _pd_baso(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_V"], S[f"C_tox2_{seg}_C"], tox2, EPS)
    pd_a_2 = _pd_apic(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_C"], S[f"C_tox2_{seg}_T"], tox2, EPS)
    Cu_HA_v = S[f"C_tox2_{seg}_V"] * (tox2["fu_p"] / max(tox2["BP"], EPS))
    J_HA = _baso_uptake_tox(tox2, Cu_HA_v, Km_HA_eff, w_b, EPS)
    Sec_HA = -_apical_secretion(tox2, seg, S[f"C_tox2_{seg}_C"],
                                APICAL_MM_T2, w_ap, EPS)
    Met_HA = -tox2["CL_int_cell"] * S[f"C_tox2_{seg}_C"] * tox2["fu_cell"]

    dx[f"C_tox2_{seg}_C"] += (-pd_b_2 + pd_a_2 + J_HA + Sec_HA + Met_HA) / max(Vc, EPS)

    # pCS
    pd_b_3 = _pd_baso(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_V"], S[f"C_tox3_{seg}_C"], tox3, EPS)
    pd_a_3 = _pd_apic(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_C"], S[f"C_tox3_{seg}_T"], tox3, EPS)
    Cu_pCS_v = S[f"C_tox3_{seg}_V"] * (tox3["fu_p"] / max(tox3["BP"], EPS))
    J_pCS = _baso_uptake_tox(tox3, Cu_pCS_v, Km_pCS_eff, w_b, EPS)
    Sec_pCS = -_apical_secretion(tox3, seg, S[f"C_tox3_{seg}_C"],
                                 APICAL_MM_T3, w_ap, EPS)
    Met_pCS = -tox3["CL_int_cell"] * S[f"C_tox3_{seg}_C"] * tox3["fu_cell"]

    dx[f"C_tox3_{seg}_C"] += (-pd_b_3 + pd_a_3 + J_pCS + Sec_pCS + Met_pCS) / max(Vc, EPS)


def _cellular_fluxes_nonprox(
        seg: str,
        dx: dict,
        S: dict,
        drug: dict, tox1: dict, tox2: dict, tox3: dict,
        DRG_PASS: dict, TOX1_PASS: dict, TOX2_PASS: dict, TOX3_PASS: dict,
        Vc: float,
        EPS: float,
):
    pd_b_d = _pd_baso(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_V"], S[f"C_{seg}_C"], drug, EPS)
    pd_a_d = _pd_apic(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_C"], S[f"C_{seg}_T"], drug, EPS)
    dx[f"C_{seg}_C"] += (-pd_b_d + pd_a_d) / max(Vc, EPS)

    pd_b_1 = _pd_baso(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_V"], S[f"C_tox1_{seg}_C"], tox1, EPS)
    pd_a_1 = _pd_apic(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_C"], S[f"C_tox1_{seg}_T"], tox1, EPS)
    dx[f"C_tox1_{seg}_C"] += (-pd_b_1 + pd_a_1) / max(Vc, EPS)

    pd_b_2 = _pd_baso(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_V"], S[f"C_tox2_{seg}_C"], tox2, EPS)
    pd_a_2 = _pd_apic(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_C"], S[f"C_tox2_{seg}_T"], tox2, EPS)
    dx[f"C_tox2_{seg}_C"] += (-pd_b_2 + pd_a_2) / max(Vc, EPS)

    pd_b_3 = _pd_baso(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_V"], S[f"C_tox3_{seg}_C"], tox3, EPS)
    pd_a_3 = _pd_apic(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_C"], S[f"C_tox3_{seg}_T"], tox3, EPS)
    dx[f"C_tox3_{seg}_C"] += (-pd_b_3 + pd_a_3) / max(Vc, EPS)


# ══════════════════════════════════════════════════════════════════════════════
# Lumen compartment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _lumen_fluxes_prox(
        seg: str,
        k: int,
        dx: dict,
        S: dict,
        drug: dict, tox1: dict, tox2: dict, tox3: dict,
        DRG_PASS: dict, TOX1_PASS: dict, TOX2_PASS: dict, TOX3_PASS: dict,
        up_d: float, up_1: float, up_2: float, up_3: float,
        FLOWS_L_HR: list,
        APICAL_MM_DRUG: bool, APICAL_MM_T1: bool, APICAL_MM_T2: bool, APICAL_MM_T3: bool,
        drug_transport_enabled: bool,
        Vt: float,
        EPS: float,
) -> tuple:
    w_ap = W_APIC[seg]
    Qin = float(FLOWS_L_HR[k])
    Qout = float(FLOWS_L_HR[k + 1])

    adv_d = Qin * up_d - Qout * S[f"C_{seg}_T"]
    pd_a_d = _pd_apic(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_C"], S[f"C_{seg}_T"], drug, EPS)
    Sec_d = _apical_secretion(drug, seg, S[f"C_{seg}_C"],
                              APICAL_MM_DRUG, w_ap, EPS, drug_transport_enabled)
    Reab_a_d = 0.0  # FIX #2: no proximal reabsorption
    dx[f"C_{seg}_T"] += (adv_d - pd_a_d + Sec_d + Reab_a_d) / max(Vt, EPS)

    adv_1 = Qin * up_1 - Qout * S[f"C_tox1_{seg}_T"]
    pd_a_1 = _pd_apic(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_C"], S[f"C_tox1_{seg}_T"], tox1, EPS)
    Sec_1 = _apical_secretion(tox1, seg, S[f"C_tox1_{seg}_C"],
                              APICAL_MM_T1, w_ap, EPS)
    dx[f"C_tox1_{seg}_T"] += (adv_1 - pd_a_1 + Sec_1) / max(Vt, EPS)

    adv_2 = Qin * up_2 - Qout * S[f"C_tox2_{seg}_T"]
    pd_a_2 = _pd_apic(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_C"], S[f"C_tox2_{seg}_T"], tox2, EPS)
    Sec_2 = _apical_secretion(tox2, seg, S[f"C_tox2_{seg}_C"],
                              APICAL_MM_T2, w_ap, EPS)
    dx[f"C_tox2_{seg}_T"] += (adv_2 - pd_a_2 + Sec_2) / max(Vt, EPS)

    adv_3 = Qin * up_3 - Qout * S[f"C_tox3_{seg}_T"]
    pd_a_3 = _pd_apic(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_C"], S[f"C_tox3_{seg}_T"], tox3, EPS)
    Sec_3 = _apical_secretion(tox3, seg, S[f"C_tox3_{seg}_C"],
                              APICAL_MM_T3, w_ap, EPS)
    dx[f"C_tox3_{seg}_T"] += (adv_3 - pd_a_3 + Sec_3) / max(Vt, EPS)

    return (S[f"C_{seg}_T"], S[f"C_tox1_{seg}_T"],
            S[f"C_tox2_{seg}_T"], S[f"C_tox3_{seg}_T"])


def _lumen_fluxes_nonprox(
        seg: str,
        j: int,
        dx: dict,
        S: dict,
        drug: dict, tox1: dict, tox2: dict, tox3: dict,
        DRG_PASS: dict, TOX1_PASS: dict, TOX2_PASS: dict, TOX3_PASS: dict,
        up_d: float, up_1: float, up_2: float, up_3: float,
        FLOWS_L_HR: list,
        Qurine: float,
        Vt: float,
        EPS: float,
) -> tuple:
    idx = len(PROX) + j
    Qin = float(FLOWS_L_HR[idx])
    Qout = float(FLOWS_L_HR[idx + 1]) if (idx + 1) < len(FLOWS_L_HR) else Qurine

    adv_d = Qin * up_d - Qout * S[f"C_{seg}_T"]
    pd_a_d = _pd_apic(DRG_PASS, "drg", seg,
                      S[f"C_{seg}_C"], S[f"C_{seg}_T"], drug, EPS)
    reab_end_d = (-drug["CL_reabs_api"] * S[f"C_{seg}_T"]
                  if (seg == "CDC5" and drug["CL_reabs_api"] > 0.0) else 0.0)
    dx[f"C_{seg}_T"] += (adv_d - pd_a_d + reab_end_d) / max(Vt, EPS)

    adv_1 = Qin * up_1 - Qout * S[f"C_tox1_{seg}_T"]
    pd_a_1 = _pd_apic(TOX1_PASS, "tox1", seg,
                      S[f"C_tox1_{seg}_C"], S[f"C_tox1_{seg}_T"], tox1, EPS)
    reab_end_1 = (-tox1["CL_reabs_api"] * S[f"C_tox1_{seg}_T"]
                  if (seg == "CDC5" and tox1["CL_reabs_api"] > 0.0) else 0.0)
    dx[f"C_tox1_{seg}_T"] += (adv_1 - pd_a_1 + reab_end_1) / max(Vt, EPS)

    adv_2 = Qin * up_2 - Qout * S[f"C_tox2_{seg}_T"]
    pd_a_2 = _pd_apic(TOX2_PASS, "tox2", seg,
                      S[f"C_tox2_{seg}_C"], S[f"C_tox2_{seg}_T"], tox2, EPS)
    reab_end_2 = (-tox2["CL_reabs_api"] * S[f"C_tox2_{seg}_T"]
                  if (seg == "CDC5" and tox2["CL_reabs_api"] > 0.0) else 0.0)
    dx[f"C_tox2_{seg}_T"] += (adv_2 - pd_a_2 + reab_end_2) / max(Vt, EPS)

    adv_3 = Qin * up_3 - Qout * S[f"C_tox3_{seg}_T"]
    pd_a_3 = _pd_apic(TOX3_PASS, "tox3", seg,
                      S[f"C_tox3_{seg}_C"], S[f"C_tox3_{seg}_T"], tox3, EPS)
    reab_end_3 = (-tox3["CL_reabs_api"] * S[f"C_tox3_{seg}_T"]
                  if (seg == "CDC5" and tox3["CL_reabs_api"] > 0.0) else 0.0)
    dx[f"C_tox3_{seg}_T"] += (adv_3 - pd_a_3 + reab_end_3) / max(Vt, EPS)

    return (S[f"C_{seg}_T"], S[f"C_tox1_{seg}_T"],
            S[f"C_tox2_{seg}_T"], S[f"C_tox3_{seg}_T"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ODE FUNCTION — CORRECTED EXECUTION ORDER
# ══════════════════════════════════════════════════════════════════════════════

def kidney_rhs(t: float, y, *, phys: dict, FLOWS_L_HR: list):
    """
    ODE right-hand side for the full kidney PBPK system.

    CORRECTED EXECUTION ORDER (vascular loops BEFORE central ODE):

    Steps 1–11   : Setup (compounds, volumes, flags, inhibition)
    Steps 15–17  : Vascular compartment loops (prox, then non-prox+CDS)
                  → Computes C_ren_exit_* (true renal venous outflow)
    Step 12      : Central compartment ODE (drug) — uses C_ren_exit_drug
    Step 13      : Debug prints for central/tissue
    Step 14      : Toxin central ODEs — uses C_ren_exit_t*
    Steps 18–19  : Cellular compartment loops
    Steps 20–21  : Lumen compartment loops
    Step 22      : Bladder/excretion update
    Step 23      : Return derivative vector

    This order ensures that renal exchange uses the TRUE venous outflow
    concentration from the serial vascular chain, not the local C_CDC5_V.
    """
    # ── Step 1: Resolve compound dicts ────────────────────────────────────────
    if C.drug is None or C.tox1 is None or C.tox2 is None or C.tox3 is None:
        raise ValueError(
            "Compounds not initialised. "
            "Call compounds.load_compounds(mode) before integrating."
        )
    drug, tox1, tox2, tox3 = C.drug, C.tox1, C.tox2, C.tox3

    try:
        DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS = (
            C.DRG_PASS, C.TOX1_PASS, C.TOX2_PASS, C.TOX3_PASS
        )
    except AttributeError:
        raise RuntimeError(
            "Passive transport bundles (DRG_PASS/TOX*_PASS) not found in "
            "`compounds`. Make sure load_compounds() sets these attributes."
        )

    # ── Step 2: One-time initialisation at t = 0 ─────────────────────────────
    if t == 0.0:
        kidney_rhs._alpha_max = {"P1": 0.0, "P2": 0.0, "P3": 0.0, "ALL": 0.0}
        kidney_rhs._apical_guarded = False
        kidney_rhs._printed_debug = False
        kidney_rhs._dbg_last_bucket = None
        kidney_rhs._tissue_dist = None
        kidney_rhs._prox_debug_fired = False
        kidney_rhs._flow_debug_done = False


    # ── Step 3: Build state dict S and zeroed derivative dict dx ─────────────
    S = {n: y[i] for n, i in IDX.items()}
    dx = {n: 0.0 for n in IDX.keys()}

    # ── Step 4: Toxin enable flags ────────────────────────────────────────────
    disable_toxins_global = bool(getattr(C, "DISABLE_TOXINS", False))
    tox1_enabled = bool(tox1.get("enabled", True)) and not disable_toxins_global
    tox2_enabled = bool(tox2.get("enabled", True)) and not disable_toxins_global
    tox3_enabled = bool(tox3.get("enabled", True)) and not disable_toxins_global

    if not tox1_enabled:
        for name in IDX:
            if name.startswith("C_tox1_"):
                S[name] = 0.0
    if not tox2_enabled:
        for name in IDX:
            if name.startswith("C_tox2_"):
                S[name] = 0.0
    if not tox3_enabled:
        for name in IDX:
            if name.startswith("C_tox3_"):
                S[name] = 0.0

    # ── Step 5: Volume accessor helpers ──────────────────────────────────────
    def V_vasc(seg):
        return float(phys.get("V_vasc_L", {}).get(seg, V_SEG[seg]))

    def V_cell(seg):
        return float(phys.get("V_cell_L", {}).get(seg, V_SEG[seg]))

    def V_tub(seg):
        return float(phys.get("V_tub_L", {}).get(seg, V_SEG[seg]))

    # ── Step 6: Extract physiology scalars ───────────────────────────────────
    Vcen = float(phys["V_ven0"])
    Qkid = float(phys["Q_kidney0"])
    GFR = float(phys["Q_GFR"])
    Qurine = float(phys["Q_urine"])
    Vb_L = float(phys["Vb_L"])
    Qc0 = float(phys["Qc0"])

    # ── Step 7: Validate flow chain ───────────────────────────────────────────
    n_seg = len(PROX) + len(NONPROX) + len(CDS)
    if len(FLOWS_L_HR) not in (n_seg + 1, n_seg):
        raise ValueError(
            f"FLOWS_L_HR must have length {n_seg + 1} or {n_seg}. "
            f"Got {len(FLOWS_L_HR)}."
        )

    # ── Step 8: Drug control flags ────────────────────────────────────────────
    if drug is None:
        drug_transport_enabled = False
        drug_inhibit = False
    else:
        IR = float(drug.get("IR", 0.0))
        bol = float(drug.get("bolus_mg", 0.0))
        base_enabled = bool(drug.get("enabled", False))
        drug_transport_enabled = base_enabled and (IR > 0.0 or bol > 0.0)
        drug_inhibit = bool(drug.get("inhibit", False))

    APICAL_MM_DRUG = bool(drug.get("use_apical_mm", USE_APICAL_MM))
    APICAL_MM_T1 = bool(tox1.get("use_apical_mm", USE_APICAL_MM))
    APICAL_MM_T2 = bool(tox2.get("use_apical_mm", USE_APICAL_MM))
    APICAL_MM_T3 = bool(tox3.get("use_apical_mm", USE_APICAL_MM))

    # ── Step 9: Validate apical secretion parameters ─────────────────────────
    if (abs(t) < 1e-12) and not getattr(kidney_rhs, "_apical_guarded", False):
        def _guard(name: str, tox: dict, use_mm: bool):
            if (not use_mm) and float(tox.get("CL_sec_api", 0.0)) <= 0.0:
                raise ValueError(
                    f"{name}: use_apical_mm=False but CL_sec_api <= 0. "
                    "Apical secretion would be completely OFF."
                )

        if drug_transport_enabled and (not APICAL_MM_DRUG) and float(drug.get("CL_sec_api", 0.0)) <= 0.0:
            raise ValueError("Drug: use_apical_mm=False but CL_sec_api <= 0 while drug is enabled.")
        _guard("IS", tox1, APICAL_MM_T1)
        _guard("HA", tox2, APICAL_MM_T2)
        _guard("pCS", tox3, APICAL_MM_T3)
        kidney_rhs._apical_guarded = True

    # ── Step 10: Competitive inhibition setup ─────────────────────────────────
    Ki_IS = float(tox1.get("Ki_furo_to_IS_mgL", np.inf))
    Ki_HA = float(tox2.get("Ki_furo_to_HA_mgL", np.inf))
    Ki_pCS = float(tox3.get("Ki_furo_to_pCS_mgL", np.inf))

    def km_eff(base_km: float, inhibitor_Cu: float, Ki: float) -> float:
        if (not drug_inhibit) or (not np.isfinite(Ki)) or (Ki <= 0.0):
            return base_km
        return base_km * (1.0 + inhibitor_Cu / Ki)

    def _upd_alpha(seg: str, Cu: float, Ki_mgL: float):
        if np.isfinite(Ki_mgL) and Ki_mgL > 0.0:
            a = 1.0 + (Cu / Ki_mgL)
            kidney_rhs._alpha_max[seg] = max(kidney_rhs._alpha_max[seg], float(a))
            kidney_rhs._alpha_max["ALL"] = max(kidney_rhs._alpha_max["ALL"], float(a))

    # ── Step 10b: Debug print at t = 0 ───────────────────────────────────────
    fu_p_drug = float(drug.get("fu_p", 1.0)) or 1.0
    BP_drug = float(drug.get("BP", 1.0)) or 1.0
    Cb0 = float(S.get("C_P1_V", S.get("C_ven", 0.0)))
    Cu0 = Cb0 * (fu_p_drug / BP_drug)

    if K.VERBOSE and (abs(t) < 1e-12) and not getattr(kidney_rhs, "_printed_debug", False):
        print(f"[drug] enabled={drug_transport_enabled}, inhibit={drug_inhibit}, "
              f"IR={drug.get('IR', 0)}, t0={drug.get('t_start_h', 0)}, "
              f"t_inf={drug.get('t_inf_h', 0)}, bolus={drug.get('bolus_mg', 0)}, "
              f"fu_p={fu_p_drug}, BP={BP_drug}, Vcen={Vcen:.3f} L")
        print(f"[apical modes] drug={APICAL_MM_DRUG}, IS={APICAL_MM_T1}, HA={APICAL_MM_T2}, pCS={APICAL_MM_T3}")
        if drug_inhibit:
            print(f"[inhibit] ON — Ki_IS={Ki_IS:.3f} mg/L, Ki_HA={Ki_HA:.3f} mg/L, "
                  f"Ki_pCS={Ki_pCS:.3f} mg/L, initial Cu_drug ≈ {Cu0:.4g} mg/L")
        else:
            print("[inhibit] OFF — furosemide does not shift toxin Km")
        kidney_rhs._printed_debug = True

    # ── Step 11: Drug infusion rate ───────────────────────────────────────────
    PROTOCOL = str(drug.get("protocol", "plasma_britz")).lower()
    K_ELIM = float(drug.get("k_elim_hinv", 0.0))

    Cin = drug_input_conc_rate(t, Vcen, Vb_L, Qc0, drug, PROTOCOL)

    if K.VERBOSE and Cin > 0.0 and not hasattr(kidney_rhs, f"_printed_cin_{int(t)}"):
        print(f"[ODE] t={t:.3f} h: Cin={Cin:.6e} mg/L/h, C_ven={S['C_ven']:.6e} mg/L")
        setattr(kidney_rhs, f"_printed_cin_{int(t)}", True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 15: Bowman's filtration and peritubular inlet concentrations
    # ══════════════════════════════════════════════════════════════════════════
    C_bow_drug = (drug["fu_p"] / max(drug["BP"], EPS)) * S["C_ven"]
    C_bow_t1 = (tox1["fu_p"] / max(tox1["BP"], EPS)) * S["C_tox1_cen"]
    C_bow_t2 = (tox2["fu_p"] / max(tox2["BP"], EPS)) * S["C_tox2_cen"]
    C_bow_t3 = (tox3["fu_p"] / max(tox3["BP"], EPS)) * S["C_tox3_cen"]

    C_in_drug = (1.0 - (GFR / max(Qkid, EPS)) * (drug["fu_p"] / max(drug["BP"], EPS))) * S["C_ven"]
    C_in_t1 = (1.0 - (GFR / max(Qkid, EPS)) * (tox1["fu_p"] / max(tox1["BP"], EPS))) * S["C_tox1_cen"]
    C_in_t2 = (1.0 - (GFR / max(Qkid, EPS)) * (tox2["fu_p"] / max(tox2["BP"], EPS))) * S["C_tox2_cen"]
    C_in_t3 = (1.0 - (GFR / max(Qkid, EPS)) * (tox3["fu_p"] / max(tox3["BP"], EPS))) * S["C_tox3_cen"]

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 16: Proximal vascular loop (P1 → P2 → P3)
    # ══════════════════════════════════════════════════════════════════════════
    for seg in PROX:
        Vv = V_vasc(seg)
        Cu_drg_v = S[f"C_{seg}_V"] * (drug["fu_p"] / max(drug["BP"], EPS))
        if drug_inhibit:
            _upd_alpha(seg, Cu_drg_v, Ki_IS)
            _upd_alpha(seg, Cu_drg_v, Ki_HA)
            _upd_alpha(seg, Cu_drg_v, Ki_pCS)
        Km_IS_eff = km_eff(tox1["Km_baso_uptake"], Cu_drg_v, Ki_IS)
        Km_HA_eff = km_eff(tox2["Km_baso_uptake"], Cu_drg_v, Ki_HA)
        Km_pCS_eff = km_eff(tox3["Km_baso_uptake"], Cu_drg_v, Ki_pCS)
        C_in_drug, C_in_t1, C_in_t2, C_in_t3 = _vascular_fluxes_prox(
            seg, dx, S,
            drug, tox1, tox2, tox3,
            DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS,
            C_in_drug, C_in_t1, C_in_t2, C_in_t3,
            Qkid, Km_IS_eff, Km_HA_eff, Km_pCS_eff,
            Vv, EPS,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 17: Non-proximal + CDS vascular loop
    # ══════════════════════════════════════════════════════════════════════════
    for seg in (NONPROX + CDS):
        C_in_drug, C_in_t1, C_in_t2, C_in_t3 = _vascular_fluxes_nonprox(
            seg, dx, S,
            drug, tox1, tox2, tox3,
            DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS,
            C_in_drug, C_in_t1, C_in_t2, C_in_t3,
            Qkid, V_vasc(seg), EPS,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CAPTURE TRUE RENAL VENOUS OUTFLOW (after all vascular segments)
    # This is the concentration exiting the last vascular segment (CDC5_V)
    # after flowing through all segments in series (P1→P2→P3→LH1→LH2→DT→CDC1-5)
    # ══════════════════════════════════════════════════════════════════════════
    C_ren_exit_drug = C_in_drug
    C_ren_exit_t1 = C_in_t1
    C_ren_exit_t2 = C_in_t2
    C_ren_exit_t3 = C_in_t3

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 12: Central compartment ODE (drug) - NOW USING CORRECT RENAL OUTFLOW
    # ══════════════════════════════════════════════════════════════════════════
    # Calculate renal exchange using TRUE venous outflow from advection chain
    renal_exchange = (Qkid / max(Vcen, 1e-12)) * (C_ren_exit_drug - S["C_ven"])
    elim = K_ELIM * S["C_ven"]

    # Rebuild TissueDistribution object only if parameters have changed
    _td = getattr(kidney_rhs, "_tissue_dist", None)
    rebuild_tissue = (
            _td is None
            or abs(_td.Vcen - Vcen) > 1e-12
            or abs(_td.Vperi - float(drug.get("Vperi_L", 0.0))) > 1e-12
            or abs(_td.k12 - float(drug.get("k12_hinv", 0.0))) > 1e-12
            or abs(_td.k21 - float(drug.get("k21_hinv", 0.0))) > 1e-12
            or _td.use_tissue != bool(drug.get("use_tissue", True))
    )
    if rebuild_tissue:
        kidney_rhs._tissue_dist = TissueDistribution(drug, Vcen)
        if K.VERBOSE:
            print(f"\n[INIT TISSUE] {kidney_rhs._tissue_dist}")

    tissue = kidney_rhs._tissue_dist
    C_peri = float(S.get("C_peri", 0.0))
    dist_contrib = tissue.central_contribution(S["C_ven"], C_peri)
    dCperi = tissue.peripheral_rhs(S["C_ven"], C_peri)
    dCven = Cin + renal_exchange - elim + dist_contrib

    # Apply non-negativity guard
    dx["C_ven"] = 0.0 if (S["C_ven"] <= 0.0 and dCven < 0.0) else dCven
    if tissue.enabled:
        dx["C_peri"] = 0.0 if (C_peri <= 0.0 and dCperi < 0.0) else dCperi
    else:
        dx["C_peri"] = 0.0

    # Warn if infusion is on but C_ven is still stuck at zero
    t0_dbg = float(drug.get("t_start_h", 0.0))
    if K.VERBOSE and (t > t0_dbg + 0.1) and Cin > 0.0 and S["C_ven"] == 0.0:
        if not hasattr(kidney_rhs, f"_warned_zero_{int(t)}"):
            print(f"[WARN] t={t:.3f} h: Cin={Cin:.4e} mg/L/h > 0 but C_ven is still 0.")
            setattr(kidney_rhs, f"_warned_zero_{int(t)}", True)

    # ── Step 13: Central/tissue debug prints ──────────────────────────────────
    if not hasattr(kidney_rhs, "_dbg_last_bucket"):
        kidney_rhs._dbg_last_bucket = None

    dbg_bucket = int(np.floor(t))
    if (K.VERBOSE and dbg_bucket != kidney_rhs._dbg_last_bucket and (dbg_bucket % 2 == 0)):
        print(f"\n[CENTRAL/TISSUE DEBUG]  t={t:.3f} h")
        print(f"  Vcen={Vcen:.4f} L, C_ven={S['C_ven']:.6e} mg/L, C_peri={C_peri:.6e} mg/L")
        print(f"  Cin={Cin:.6e}, renal_exchange={renal_exchange:.6e}, elim={elim:.6e}")
        print(f"  dist_contrib={dist_contrib:.6e}, dCven={dCven:.6e}, dCperi={dCperi:.6e}")
        print(f"  tissue_enabled={tissue.enabled}")
        if tissue.enabled:
            amt = tissue.amount_total(S["C_ven"], C_peri)
            print(f"  total drug amount (central+peripheral) = {amt:.6e} mg")
            print(f"  mass-balance (dist only) = {Vcen * dist_contrib + tissue.Vperi * dCperi:.6e} mg/h (should be ~0)")
        kidney_rhs._dbg_last_bucket = dbg_bucket

    if K.VERBOSE and 1.9 <= t <= 2.1 and not hasattr(kidney_rhs, "_printed_peak_debug"):
        print(f"\n[PEAK WINDOW DEBUG]  t={t:.6f}")
        print(f"  C_ven={S['C_ven']:.6e}, C_peri={C_peri:.6e}")
        print(f"  Cin={Cin:.6e}, renal_exchange={renal_exchange:.6e}")
        print(f"  dist_contrib={dist_contrib:.6e}, dCven={dCven:.6e}")
        setattr(kidney_rhs, "_printed_peak_debug", True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 14: Toxin central compartment ODEs - USING CORRECT RENAL OUTFLOW
    # ══════════════════════════════════════════════════════════════════════════
    for cmpd, cen_key, ren_exit, enabled in (
            (tox1, "C_tox1_cen", C_ren_exit_t1, tox1_enabled),
            (tox2, "C_tox2_cen", C_ren_exit_t2, tox2_enabled),
            (tox3, "C_tox3_cen", C_ren_exit_t3, tox3_enabled),
    ):
        if enabled:
            dx[cen_key] = (
                    (cmpd["R0"] / max(Vcen, EPS))
                    - (Qkid / max(Vcen, EPS)) * (S[cen_key] - ren_exit)
            )
        else:
            dx[cen_key] = 0.0

    if K.VERBOSE and abs(t) < 1e-12 and not getattr(kidney_rhs, "_fix5_val_printed", False):
        print("\n[FIX #5 VALIDATION] Toxin central ODEs — kidney exchange terms:")
        for label, cen_key, ren_exit in (
                ("IS ", "C_tox1_cen", C_ren_exit_t1),
                ("HA ", "C_tox2_cen", C_ren_exit_t2),
                ("pCS", "C_tox3_cen", C_ren_exit_t3),
        ):
            exch = (Qkid / max(Vcen, EPS)) * (S[cen_key] - ren_exit)
            print(f"  {label}: C_cen={S[cen_key]:.4e}  C_ren_exit={ren_exit:.4e}"
                  f"  exchange_term={exch:.4e} mg/L/h")
        kidney_rhs._fix5_val_printed = True

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 18: Proximal cellular loop
    # ══════════════════════════════════════════════════════════════════════════
    for seg in PROX:
        Cu_drg_v = S[f"C_{seg}_V"] * (drug["fu_p"] / max(drug["BP"], EPS))
        Km_IS_eff = km_eff(tox1["Km_baso_uptake"], Cu_drg_v, Ki_IS)
        Km_HA_eff = km_eff(tox2["Km_baso_uptake"], Cu_drg_v, Ki_HA)
        Km_pCS_eff = km_eff(tox3["Km_baso_uptake"], Cu_drg_v, Ki_pCS)

        if (K.VERBOSE and seg == "P1" and int(t) % 2 == 0
                and not hasattr(kidney_rhs, f"_p1_lumen_{int(t)}")):
            print(f"\n[P1 LUMEN DEBUG]  t={t}  C_vasc={S[f'C_{seg}_V']:.4e}  "
                  f"C_cell={S[f'C_{seg}_C']:.4e}  C_lumen={S[f'C_{seg}_T']:.4e}")
            setattr(kidney_rhs, f"_p1_lumen_{int(t)}", True)

        _cellular_fluxes_prox(
            seg, dx, S,
            drug, tox1, tox2, tox3,
            DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS,
            Km_IS_eff, Km_HA_eff, Km_pCS_eff,
            APICAL_MM_DRUG, APICAL_MM_T1, APICAL_MM_T2, APICAL_MM_T3,
            drug_transport_enabled, V_cell(seg), EPS,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 19: Non-proximal + CDS cellular loop
    # ══════════════════════════════════════════════════════════════════════════
    for seg in (NONPROX + CDS):
        _cellular_fluxes_nonprox(
            seg, dx, S,
            drug, tox1, tox2, tox3,
            DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS,
            V_cell(seg), EPS,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 20: Proximal lumen loop
    # ══════════════════════════════════════════════════════════════════════════
    up_d, up_1, up_2, up_3 = C_bow_drug, C_bow_t1, C_bow_t2, C_bow_t3

    for k, seg in enumerate(PROX):
        if (seg == "P1" and (1.9 <= t <= 2.1) and not getattr(kidney_rhs, "_prox_debug_fired", False)):
            print(f"\n[PROXIMAL DEBUG - P1]  t={t:.4f}  C_vasc={S[f'C_{seg}_V']:.4e}  "
                  f"C_cell={S[f'C_{seg}_C']:.4e}  C_lumen={S[f'C_{seg}_T']:.4e}")
            kidney_rhs._prox_debug_fired = True

        up_d, up_1, up_2, up_3 = _lumen_fluxes_prox(
            seg, k, dx, S,
            drug, tox1, tox2, tox3,
            DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS,
            up_d, up_1, up_2, up_3,
            FLOWS_L_HR,
            APICAL_MM_DRUG, APICAL_MM_T1, APICAL_MM_T2, APICAL_MM_T3,
            drug_transport_enabled, V_tub(seg), EPS,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 21: Non-proximal + CDS lumen loop (continued)
    # ══════════════════════════════════════════════════════════════════════════
    if K.VERBOSE and 1.99 < t < 2.01 and not hasattr(kidney_rhs, "_flow_debug_done"):
        print(f"\n[FLOW + LUMEN DEBUG]  t={t:.6f}")
        print(f"  Qurine={Qurine:.6f} L/h")
        for dbg_seg in ["P1", "P2", "P3", "DT", "CDC5"]:
            k_ = f"C_{dbg_seg}_T"
            print(f"  {k_:14s} = {S.get(k_, 0.0):.6f}")
        print(f"  C_CDC5_T = {S.get('C_CDC5_T', 0.0):.6f}")
        print(f"  ex_d = {Qurine * S.get('C_CDC5_T', 0.0):.6e}  Ae_drg = {S.get('Ae_drg', 0.0):.6e}")
        kidney_rhs._flow_debug_done = True

    for j, seg in enumerate(NONPROX + CDS):
        # ── FIX #8 VALIDATION: at CDC5, confirm Qout == Qurine ──────────────
        if seg == "CDC5" and not getattr(kidney_rhs, "_fix8_val_printed", False):
            _idx_cdc5 = len(PROX) + j
            if (_idx_cdc5 + 1) < len(FLOWS_L_HR):
                _Qout_cdc5 = float(FLOWS_L_HR[_idx_cdc5 + 1])
                _mismatch = abs(_Qout_cdc5 - Qurine) > 1e-9 * max(Qurine, 1e-12)
                print(f"\n[FIX #8 VALIDATION] CDC5 lumen exit flow check:")
                print(f"  FLOWS_L_HR[-1] (Qout CDC5) = {_Qout_cdc5:.6e} L/h")
                print(f"  phys['Q_urine']  (Qurine)   = {Qurine:.6e} L/h")
                if _mismatch:
                    print(
                        f"  ⚠ MISMATCH! Relative error = {abs(_Qout_cdc5 - Qurine) / max(Qurine, 1e-12) * 100:.2f}%.\n"
                        f"  Excretion in step 22 uses Qurine correctly,\n"
                        f"  but CDC5 lumen ODE uses the WRONG Qout.\n"
                        f"  Fix: ensure FLOWS_L_HR[-1] == Q_urine in caller.")
                else:
                    print(f"  ✓ FLOWS_L_HR[-1] == Qurine — consistent.")
            else:
                print(f"\n[FIX #8 VALIDATION] FLOWS_L_HR length = {len(FLOWS_L_HR)} "
                      f"(= n_seg, no terminal entry).\n"
                      f"  CDC5 Qout falls back to Qurine={Qurine:.6e} L/h — ✓ correct.")
            kidney_rhs._fix8_val_printed = True

        up_d, up_1, up_2, up_3 = _lumen_fluxes_nonprox(
            seg, j, dx, S,
            drug, tox1, tox2, tox3,
            DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS,
            up_d, up_1, up_2, up_3,
            FLOWS_L_HR, Qurine, V_tub(seg), EPS,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 22: Bladder and cumulative excretion (ONCE, after the lumen loop)
    # ══════════════════════════════════════════════════════════════════════════
    if K.VERBOSE and 1.99 < t < 2.01:
        print(f"  [EXCRETION CHECK] C_CDC5_T={S.get('C_CDC5_T', 0):.4e}  "
              f"Qurine={Qurine:.4e}  ex_d={Qurine * S.get('C_CDC5_T', 0):.4e}")

    ex_d = Qurine * S["C_CDC5_T"]
    ex_1 = Qurine * S["C_tox1_CDC5_T"]
    ex_2 = Qurine * S["C_tox2_CDC5_T"]
    ex_3 = Qurine * S["C_tox3_CDC5_T"]

    dx["Ae_drg"] += ex_d
    dx["Ae_t1"] += ex_1
    dx["Ae_t2"] += ex_2
    dx["Ae_t3"] += ex_3
    dx["C_bladder"] += ex_d / max(float(phys["V_bladder"]), EPS)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 23: Assemble and return derivative vector
    # ══════════════════════════════════════════════════════════════════════════
    return np.array([dx[n] for n in IDX.keys()], dtype=float)