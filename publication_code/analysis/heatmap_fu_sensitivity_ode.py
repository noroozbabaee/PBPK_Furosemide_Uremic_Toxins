# -*- coding: utf-8 -*-
"""
heatmap_fu_ode_real.py
=======================
2D heatmap using REAL ODE simulations (not analytical formula).

SCIENTIFIC QUESTION:
  How do furosemide fu_p AND toxin fu_p jointly determine the actual
  toxin plasma concentration at CKD-5 steady state under chronic
  oral furosemide 240 mg/day?

METHOD:
  For each (drug_fu_p, toxin_fu_p) grid point:
    1. Override fu_p values in compound dicts directly
    2. Run 500h CKD-5 warm-up (no drug) to reach toxin SS
    3. Run 30-day chronic oral furosemide simulation
    4. Read time-averaged IS and pCS from last cycle
    5. Store absolute Css and %Δ vs no-drug baseline

GRID:
  drug fu_p:  5 values  [0.02, 0.05, 0.07, 0.10, 0.13]
  toxin fu_p: 5 values  [0.5x, 0.75x, 1.0x, 1.5x, 2.0x] × Deltombe base
  Total: 5×5 = 25 grid points × 2 toxins = 50 simulations
  Each simulation: ~500h warmup + 30×24h dosing
  Estimated runtime: 30-90 minutes depending on your machine

  NOTE: a coarser 5×5 grid is used (not 60×60) because each point
  requires a full 30-day ODE run. Increase N_DRUG/N_TOX for finer
  resolution at the cost of runtime.

COLOUR AXIS:
  Panel A: IS plasma Css (mg/L) at periodic SS — absolute concentration
  Panel B: pCS plasma Css (mg/L) at periodic SS — absolute concentration
  Panel C: IS %Δ vs no-drug baseline
  Panel D: pCS %Δ vs no-drug baseline

OUTPUTS:
  ../results/heatmap_fu_ode_real.npz  — saved grid data (reload without re-run)
  ../results/heatmap_fu_ode_real.png
  ../results/heatmap_fu_ode_real.pdf

USAGE:
  Run from your PyCharm project root where compound_parameters.py lives.
  python heatmap_fu_ode_real.py

  To reload saved results without re-running:
    Set FORCE_RERUN = False  (default)
  To force a full re-run:
    Set FORCE_RERUN = True
"""

from __future__ import annotations
import os, sys, copy, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Version-safe trapezoidal integration ───────────────────────────────────
# NumPy >= 2.0 added np.trapezoid and deprecated np.trapz; some later
# NumPy releases removed np.trapz entirely. getattr(np, "trapezoid",
# np.trapz) would crash in that case because the default argument is
# evaluated eagerly. Use hasattr() instead.
def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

# ── Force fresh module imports ────────────────────────────────────────────────
for _mod in ["compound_parameters", "compounds", "simulation",
             "ode_system_twocompartment", "physiology", "constants"]:
    if _mod in sys.modules:
        del sys.modules[_mod]

import compound_parameters as CP
import compounds as C
from simulation import run_sim
from state_vector import IDX
import constants as K

# All outputs (cache .npz and figures) go to the shared top-level results/
# folder, not a local "figures/" subfolder.
OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
K.VERBOSE = False

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
FORCE_RERUN   = False       # set True to ignore saved cache
MODE          = "ckd5"
SS_WARMUP_H   = 500.0       # h — toxin SS warm-up (no drug)
N_DAYS        = 30          # days chronic dosing
TAU_H         = 24.0        # h — once-daily
DOSE_MG       = 240.0       # mg/day
F_ORAL        = 0.60        # FDA label NDA 016273
K_A           = 1.5         # h⁻¹ absorption rate constant
N_ORAL_STEPS  = 20          # piecewise segments
T_WINDOW_H    = 6.0         # h absorption window
DT_WITHIN     = 0.1         # h step during absorption
DT_BETWEEN    = 0.5         # h step post-absorption

# ── Grid definition ────────────────────────────────────────────────────────
# 9×7 coarse grid — each point is a full 30-day simulation
# Increase for finer resolution (runtime scales linearly)
#
# IMPORTANT: N_DRUG and N_TOX are derived from the grid arrays below, not
# hardcoded separately. The original version of this script hardcoded
# N_DRUG=9 and N_TOX=7 as SEPARATE constants from the array lengths, which
# only worked by coincidence (the arrays happened to have those lengths).
# If you ever edit DRUG_FU_GRID or TOXIN_FU_MULT without also updating a
# separately-hardcoded count, the grid arrays used to build IS_ss_grid /
# pCS_ss_grid (np.zeros((N_TOX, N_DRUG))) would silently mismatch. Deriving
# these from len(...) makes that mismatch impossible.

# Drug fu_p grid — spans healthy (0.02) to severe uremic displacement (0.13)
DRUG_FU_GRID = np.array([0.01,0.02,0.04, 0.05, 0.07,0.08, 0.10,0.11, 0.13])

# Toxin fu_p multiplier grid (× Deltombe 2015 CKD-5 base)
TOXIN_FU_MULT = np.array([0.3,0.50, 0.75, 1.00, 1.50,1.7, 2.00])

N_DRUG = len(DRUG_FU_GRID)
N_TOX  = len(TOXIN_FU_MULT)

# Deltombe 2015 CKD-5 base values
IS_FU_BASE  = 0.10
pCS_FU_BASE = 0.05

# ── Oral profile builder ──────────────────────────────────────────────────────
def _build_oral_profile():
    A0      = F_ORAL * DOSE_MG
    t_edges = np.linspace(0, T_WINDOW_H, N_ORAL_STEPS + 1)
    segs    = []
    for i in range(N_ORAL_STEPS):
        t0 = t_edges[i]; t1 = t_edges[i+1]; dt = t1 - t0
        A_abs = A0 * (np.exp(-K_A*t0) - np.exp(-K_A*t1))
        segs.append((t0, t1, A_abs / max(dt, 1e-9)))
    return segs

ORAL_SEGS = _build_oral_profile()


# ── Parameter override helpers ────────────────────────────────────────────────
def _restore_ki():
    """Prevent Ki from being reset by initialize_compounds()."""
    _, IS_h, HA_h, pCS_h = CP.get_compound_parameters("healthy")
    for tox, key, src in [
        (C.tox1, "Ki_furo_to_IS_mgL",  IS_h),
        (C.tox2, "Ki_furo_to_HA_mgL",  HA_h),
        (C.tox3, "Ki_furo_to_pCS_mgL", pCS_h),
    ]:
        if isinstance(tox, dict):
            tox[key] = float(src.get(key, 1e9))


def _set_fu(drug_fu: float, IS_fu: float, pCS_fu: float):
    """
    Override fu_p for drug and toxins in the live compound dicts.
    Also updates fu_cell proportionally (preserving fu_cell/fu_p ratio
    from the Deltombe 2015 base values).
    """
    # Drug fu_p — ratio applied to fu_cell as well
    drug_fu_base  = 0.05    # CKD-5 base
    drug_cell_base= 0.05
    C.drug["fu_p"]    = drug_fu
    C.drug["fu_cell"] = drug_cell_base * (drug_fu / drug_fu_base)

    # IS fu_p
    IS_cell_base = 0.125   # CKD-5 base fu_cell for IS
    C.tox1["fu_p"]    = IS_fu
    C.tox1["fu_cell"] = IS_cell_base * (IS_fu / IS_FU_BASE)

    # pCS fu_p
    pCS_cell_base = 0.1667  # CKD-5 base fu_cell for pCS
    C.tox3["fu_p"]    = pCS_fu
    C.tox3["fu_cell"] = pCS_cell_base * (pCS_fu / pCS_FU_BASE)


# ── Single warm-up run ────────────────────────────────────────────────────────
def _warmup(drug_fu: float, IS_fu: float, pCS_fu: float) -> dict:
    """Run 500h CKD-5 SS warm-up with no drug. Returns IC dict."""
    C.initialize_compounds(MODE)
    _restore_ki()
    _set_fu(drug_fu, IS_fu, pCS_fu)
    C.drug.update({"enabled": False, "IR": 0.0,
                   "t_start_h": 1e9, "t_inf_h": 0.0, "bolus_mg": 0.0})
    sol = run_sim(t_end_h=SS_WARMUP_H, mode=MODE, dt=1.0)
    ic  = {name: float(sol.y[i, -1]) for name, i in IDX.items()}
    for k in list(ic.keys()):
        if k.startswith("Ae_"): ic[k] = 0.0
    return ic


# ── One 24h dose cycle ────────────────────────────────────────────────────────
def _run_one_cycle(ic: dict, day: int,
                   drug_fu: float, IS_fu: float, pCS_fu: float) -> tuple:
    t_off = day * TAU_H
    t_all = []; y_all = []
    ic_cur = copy.deepcopy(ic)
    for k in ("C_ven", "C_peri", "Ae_drg", "C_bladder"):
        if k in ic_cur: ic_cur[k] = 0.0

    # Absorption segments
    for (t0, t1, IR) in ORAL_SEGS:
        dur = t1 - t0
        if dur < 1e-6: continue
        C.initialize_compounds(MODE); _restore_ki()
        _set_fu(drug_fu, IS_fu, pCS_fu)
        dp, *_ = CP.get_compound_parameters(MODE)
        for k, v in dp.items(): C.drug[k] = v
        C.drug.update({"enabled": True, "IR": IR, "t_inf_h": dur,
                       "t_start_h": 0.0, "bolus_mg": 0.0,
                       "inhibit": True})
        C.drug["fu_p"]    = drug_fu
        C.drug["fu_cell"] = 0.05 * (drug_fu / 0.05)
        for attr in ("tox1", "tox2", "tox3"):
            tox = getattr(C, attr, None)
            if isinstance(tox, dict): tox["enabled"] = True
        _set_fu(drug_fu, IS_fu, pCS_fu)
        sol = run_sim(t_end_h=dur, mode=MODE, dt=DT_WITHIN,
                      custom_initial_conds=copy.deepcopy(ic_cur))
        t_all.append(sol.t + t_off + t0)
        y_all.append(sol.y)
        ic_cur = {name: float(sol.y[i, -1]) for name, i in IDX.items()}

    # Post-absorption
    t_abs_end   = ORAL_SEGS[-1][1]
    t_remaining = TAU_H - t_abs_end
    if t_remaining > 0.1:
        C.initialize_compounds(MODE); _restore_ki()
        _set_fu(drug_fu, IS_fu, pCS_fu)
        dp, *_ = CP.get_compound_parameters(MODE)
        for k, v in dp.items(): C.drug[k] = v
        C.drug.update({"enabled": True, "IR": 0.0,
                       "t_inf_h": 0.0, "t_start_h": 1e9,
                       "bolus_mg": 0.0, "inhibit": True})
        C.drug["fu_p"] = drug_fu
        for attr in ("tox1", "tox2", "tox3"):
            tox = getattr(C, attr, None)
            if isinstance(tox, dict): tox["enabled"] = True
        _set_fu(drug_fu, IS_fu, pCS_fu)
        sol2 = run_sim(t_end_h=t_remaining, mode=MODE, dt=DT_BETWEEN,
                       custom_initial_conds=copy.deepcopy(ic_cur))
        t_all.append(sol2.t + t_off + t_abs_end)
        y_all.append(sol2.y)
        ic_cur = {name: float(sol2.y[i, -1]) for name, i in IDX.items()}

    return (np.concatenate(t_all),
            np.concatenate(y_all, axis=1),
            ic_cur)


# ── Run one full grid point ───────────────────────────────────────────────────
def _run_grid_point(drug_fu: float, IS_fu: float, pCS_fu: float) -> dict:
    """
    Run warm-up + 30-day chronic oral simulation for one fu combination.
    Returns dict with IS_base, pCS_base, IS_ss, pCS_ss, IS_pct, pCS_pct.
    """
    # Warm-up
    ic = _warmup(drug_fu, IS_fu, pCS_fu)
    IS_base  = float(ic.get("C_tox1_cen", 0))
    pCS_base = float(ic.get("C_tox3_cen", 0))

    # 30-day chronic dosing
    all_t=[]; all_IS=[]; all_pCS=[]
    ic_cur = copy.deepcopy(ic)
    for day in range(N_DAYS):
        t_c, y_c, ic_cur = _run_one_cycle(ic_cur, day, drug_fu, IS_fu, pCS_fu)
        all_t.append(t_c)
        if "C_tox1_cen" in IDX: all_IS.append(y_c[IDX["C_tox1_cen"]])
        if "C_tox3_cen" in IDX: all_pCS.append(y_c[IDX["C_tox3_cen"]])

    t_all   = np.concatenate(all_t)
    IS_all  = np.concatenate(all_IS)
    pCS_all = np.concatenate(all_pCS)

    # Time-average last cycle
    # _trapz is the version-safe module-level function defined above
    mask    = t_all >= (N_DAYS - 1) * TAU_H
    t_last  = t_all[mask]
    dt_last = t_last[-1] - t_last[0]
    IS_ss   = float(_trapz(IS_all[mask],  t_last) / dt_last)
    pCS_ss  = float(_trapz(pCS_all[mask], t_last) / dt_last)

    IS_pct  = (IS_ss  - IS_base)  / max(IS_base,  1e-9) * 100
    pCS_pct = (pCS_ss - pCS_base) / max(pCS_base, 1e-9) * 100

    return dict(IS_base=IS_base, pCS_base=pCS_base,
                IS_ss=IS_ss,     pCS_ss=pCS_ss,
                IS_pct=IS_pct,   pCS_pct=pCS_pct)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN GRID COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
cache_path = os.path.join(OUTPUT_DIR, "heatmap_fu_ode_real.npz")

if not FORCE_RERUN and os.path.exists(cache_path):
    print(f"[CACHE] Loading saved ODE grid from {cache_path}")
    cache = np.load(cache_path, allow_pickle=False)
    IS_ss_grid   = cache["IS_ss_grid"]
    pCS_ss_grid  = cache["pCS_ss_grid"]
    IS_pct_grid  = cache["IS_pct_grid"]
    pCS_pct_grid = cache["pCS_pct_grid"]
    IS_base_grid  = cache["IS_base_grid"]
    pCS_base_grid = cache["pCS_base_grid"]
    print(f"  IS  Css range: {IS_ss_grid.min():.2f} – {IS_ss_grid.max():.2f} mg/L")
    print(f"  pCS Css range: {pCS_ss_grid.min():.2f} – {pCS_ss_grid.max():.2f} mg/L")
else:
    print("=" * 65)
    print("RUNNING ODE HEATMAP — full 30-day simulations")
    print(f"Grid: {N_DRUG} × {N_TOX} = {N_DRUG*N_TOX} points")
    print(f"Each point: {SS_WARMUP_H:.0f}h warmup + {N_DAYS}×{TAU_H:.0f}h dosing")
    print("=" * 65)

    IS_ss_grid    = np.zeros((N_TOX, N_DRUG))
    pCS_ss_grid   = np.zeros((N_TOX, N_DRUG))
    IS_pct_grid   = np.zeros((N_TOX, N_DRUG))
    pCS_pct_grid  = np.zeros((N_TOX, N_DRUG))
    IS_base_grid  = np.zeros((N_TOX, N_DRUG))
    pCS_base_grid = np.zeros((N_TOX, N_DRUG))

    t_start = time.time()
    total   = N_TOX * N_DRUG

    for i, t_mult in enumerate(TOXIN_FU_MULT):
        for j, drug_fu in enumerate(DRUG_FU_GRID):
            IS_fu  = IS_FU_BASE  * t_mult
            pCS_fu = pCS_FU_BASE * t_mult
            n_done = i * N_DRUG + j + 1
            elapsed = time.time() - t_start
            eta     = (elapsed / n_done) * (total - n_done) if n_done > 1 else 0
            print(f"\n[{n_done:2d}/{total}] "
                  f"drug_fu={drug_fu:.3f}  IS_fu={IS_fu:.3f}  "
                  f"pCS_fu={pCS_fu:.3f}  "
                  f"ETA={eta/60:.1f}min")

            try:
                res = _run_grid_point(drug_fu, IS_fu, pCS_fu)
                IS_ss_grid[i, j]    = res["IS_ss"]
                pCS_ss_grid[i, j]   = res["pCS_ss"]
                IS_pct_grid[i, j]   = res["IS_pct"]
                pCS_pct_grid[i, j]  = res["pCS_pct"]
                IS_base_grid[i, j]  = res["IS_base"]
                pCS_base_grid[i, j] = res["pCS_base"]
                print(f"  IS  = {res['IS_ss']:.2f} mg/L  ({res['IS_pct']:+.1f}%)  "
                      f"base={res['IS_base']:.2f}")
                print(f"  pCS = {res['pCS_ss']:.2f} mg/L  ({res['pCS_pct']:+.1f}%)  "
                      f"base={res['pCS_base']:.2f}")
            except Exception as e:
                print(f"  [ERROR] {e} — filling with NaN")
                IS_ss_grid[i, j] = pCS_ss_grid[i, j]   = np.nan
                IS_pct_grid[i, j]= pCS_pct_grid[i, j]  = np.nan
                IS_base_grid[i, j]=pCS_base_grid[i, j] = np.nan

    # Save
    np.savez(cache_path,
             IS_ss_grid=IS_ss_grid,   pCS_ss_grid=pCS_ss_grid,
             IS_pct_grid=IS_pct_grid, pCS_pct_grid=pCS_pct_grid,
             IS_base_grid=IS_base_grid, pCS_base_grid=pCS_base_grid,
             DRUG_FU_GRID=DRUG_FU_GRID, TOXIN_FU_MULT=TOXIN_FU_MULT)
    print(f"\n[SAVED] {cache_path}")
    print(f"Total runtime: {(time.time()-t_start)/60:.1f} min")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE — 2×2 layout
# ═════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         12,
    "axes.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
})

fig = plt.figure(figsize=(14, 11), dpi=300)
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.38, wspace=0.32,
    left=0.09, right=0.96,
    top=0.91, bottom=0.10,
)

# X/Y tick labels
drug_labels = [f"{v:.2f}" for v in DRUG_FU_GRID]
tox_labels  = [f"{v:.2f}×" for v in TOXIN_FU_MULT]

PANELS = [
    # (title, grid, cmap, ylabel_extra, clinical_val, unit)
    ("A   IS plasma $C_{ss}$ — absolute",
     IS_ss_grid,   "Blues",   "mg/L", None,  "mg/L"),
    ("B   pCS plasma $C_{ss}$ — absolute",
     pCS_ss_grid,  "Purples", "mg/L", None,  "mg/L"),
    ("C   IS $\\Delta C_{ss}$ vs no-drug baseline",
     IS_pct_grid,  "YlOrRd",  "%",    31.9,  "%"),
    ("D   pCS $\\Delta C_{ss}$ vs no-drug baseline",
     pCS_pct_grid, "YlOrRd",  "%",    29.3,  "%"),
]

for idx, (title, grid, cmap, unit, clin_val, cbar_unit) in enumerate(PANELS):
    ax  = fig.add_subplot(gs[idx // 2, idx % 2])
    row, col = idx // 2, idx % 2

    vmin = float(np.nanmin(grid))
    vmax = float(np.nanmax(grid))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ── Heatmap ───────────────────────────────────────────────────────────
    im = ax.imshow(
        grid,
        cmap=cmap, norm=norm,
        aspect="auto", origin="lower",
        interpolation="nearest",
    )

    # ── Clinical contour (panels C and D) ─────────────────────────────────
    if clin_val is not None and vmin < clin_val < vmax:
        # imshow uses pixel coordinates — use contourf on the grid directly
        try:
            cs = ax.contour(
                np.arange(N_DRUG), np.arange(N_TOX), grid,
                levels=[clin_val],
                colors=["#1A1A8C"], linewidths=2.0, linestyles=["--"]
            )
            ax.clabel(cs, fmt=f"Clinical +{clin_val:.0f}%",
                      fontsize=9, colors="#1A1A8C",
                      inline=True, inline_spacing=4)
        except Exception:
            pass

    # ── Isoconcentration / iso-percent contours ────────────────────────────
    if unit == "%":
        iso_levels = [l for l in [5, 10, 15, 20, 25, 30]
                      if vmin < l < vmax]
    else:
        # absolute — pick sensible round numbers
        step = max(1, round((vmax - vmin) / 5))
        iso_levels = [l for l in range(
            int(np.ceil(vmin)), int(np.floor(vmax))+1, step)
            if vmin < l < vmax][:5]

    if iso_levels:
        try:
            cs2 = ax.contour(
                np.arange(N_DRUG), np.arange(N_TOX), grid,
                levels=iso_levels,
                colors=["#333333"], linewidths=0.7,
                linestyles=[":"], alpha=0.7
            )
            ax.clabel(cs2, fmt=f"%.1f{cbar_unit}",
                      fontsize=8, colors="#333333",
                      inline=True, inline_spacing=3)
        except Exception:
            pass

    # ── Current model star (drug_fu=0.05, toxin_mult=1.0) ─────────────────
    # drug_fu=0.05 is index 1 in DRUG_FU_GRID [0.02,0.05,0.07,0.10,0.13]
    # toxin_mult=1.0 is index 2 in TOXIN_FU_MULT [0.50,0.75,1.00,1.50,2.00]
    star_x = np.where(DRUG_FU_GRID == 0.05)[0]
    star_y = np.where(TOXIN_FU_MULT == 1.00)[0]
    if len(star_x) and len(star_y):
        ax.scatter(star_x[0], star_y[0],
                   s=300, marker="*", color="white",
                   edgecolors="black", linewidths=1.3,
                   zorder=10, label="Current model")

    # ── Colourbar ─────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(f"{cbar_unit}", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # ── Axis ticks and labels ─────────────────────────────────────────────
    ax.set_xticks(range(N_DRUG))
    ax.set_xticklabels(drug_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(N_TOX))
    ax.set_yticklabels(tox_labels, fontsize=9)

    ax.set_xlabel(r"Furosemide $f_{u,p}$", fontsize=12, labelpad=5)
    if col == 0:
        ax.set_ylabel(r"Toxin $f_{u,p}$ multiplier" "\n"
                      r"($\times$ Deltombe 2015 CKD-5 base)",
                      fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=6)

    if star_x.size and star_y.size:
        ax.legend(fontsize=9, frameon=True, loc="upper left",
                  framealpha=0.9, edgecolor="#cccccc")

# ── Shared suptitle ────────────────────────────────────────────────────────
fig.suptitle(
    "ODE-based sensitivity: furosemide $f_{u,p}$ × toxin $f_{u,p}$ "
    "→ uremic toxin plasma concentration\n"
    "Full 30-day PBPK simulation at each grid point  |  "
    "CKD-5  |  Furosemide 240 mg/day oral  |  "
    "$K_i = 0.35$ mg/L",
    fontsize=11, fontweight="bold", y=0.995,
)

# ── Footer ─────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    f"Grid: {N_DRUG}×{N_TOX} ODE simulations  |  "
    f"Each point: {SS_WARMUP_H:.0f}h warm-up + {N_DAYS}×{TAU_H:.0f}h dosing  |  "
    r"$\bigstar$ = current model (CKD-5 mild: $f_{u,p}^{furo}=0.05$, toxin mult=1.0)  |  "
    "Blue dashed = Costes-Albrespic 2025 clinical signal",
    ha="center", fontsize=8, color="#555555", style="italic",
)

# ── Save ───────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    p = os.path.join(OUTPUT_DIR, f"heatmap_fu_ode_real.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[saved] {p}")
plt.close(fig)
print("Done.")