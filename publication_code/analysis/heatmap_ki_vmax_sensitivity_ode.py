# -*- coding: utf-8 -*-
"""
heatmap_ki_vmax_fixed_v4.py
============================
Final clean version — all issues resolved, 15×12 grid, 24h simulation.

CHANGES vs v3:
  - Panel title corrected: "t=12h" → "t=24h post-dose"
  - Cache key now includes SIM_HOURS so old 12h cached results are
    automatically treated as stale and never reused for 24h runs
  - CLEAR_OLD_CACHE flag: if True, deletes cache before running
    (use once to wipe previous 12h results, then set back to False)
  - main() sets K.VERBOSE=False before any simulation
  - Cache key also includes N_KI and N_VMAX so changing grid resolution
    forces a full re-run rather than silently reusing stale results

PREVIOUSLY FIXED (v2/v3):
  - Ki=3.77 reference line removed
  - suppress_output removed from run_sim() calls
  - scenario_manager dependency removed
  - BP division removed from Css extraction (inflated IS×1.43, pCS×1.82)
  - Simulation extended from 12h to 24h
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm, Normalize
from typing import Dict, Any
import time
import warnings

warnings.filterwarnings("ignore")

# ── Path bootstrap ──────────────────────────────────────────────────────────
# This script lives in analysis/. The core PBPK engine lives in ../src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Output directory ──────────────────────────────────────────────────────────
# All outputs (cache .pkl and figures) go to the shared top-level results/
# folder, not a local "figures/" subfolder.
OUTDIR     = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(OUTDIR, exist_ok=True)
FIG_PNG    = os.path.join(OUTDIR, "heatmap_ki_vmax_IS_pCS.png")
FIG_PDF    = os.path.join(OUTDIR, "heatmap_ki_vmax_IS_pCS.pdf")
CACHE_FILE = os.path.join(OUTDIR, "heatmap_cache_correct.pkl")

# ── Cache control ─────────────────────────────────────────────────────────────
# Set CLEAR_OLD_CACHE = True if you ever need to force a full re-run and
# discard all cached grid points (e.g. after changing compound_parameters.py
# in a way that should invalidate every cached %Δ value, or after changing
# SIM_HOURS/N_KI/N_VMAX in a way the cache-key versioning doesn't catch).
# Defaults to False here since this is a fresh package with no legacy cache
# to migrate away from — leave it False for normal incremental use, where
# the cache key already encodes Ki, Vmax scale, base Ki, base R0, SIM_HOURS,
# N_KI, and N_VMAX, so any of those changing automatically invalidates only
# the affected entries without needing this flag.
CLEAR_OLD_CACHE = False

# ── Simulation parameters ─────────────────────────────────────────────────────
SIM_HOURS   = 24.0     # h — full drug PK cycle (furosemide t½~1.4h in CKD-5)
WARMUP_H    = 500.0    # h — toxin CKD-5 steady-state warm-up
DOSE_MG     = 120.0    # mg IV furosemide
INFUSION_H  = 2.0      # h — infusion duration → IR = DOSE_MG/INFUSION_H mg/h
DT_SIM      = 0.1      # h — ODE output step during 24h runs
DT_WARMUP   = 0.5      # h — ODE output step during warm-up

TOXIN_BP    = {"IS": 0.70, "pCS": 0.55}  # kept for reference, NOT used in Css extraction

# ── Grid ──────────────────────────────────────────────────────────────────────
N_KI   = 15
N_VMAX = 12

KI_MIN,   KI_MAX   = 0.05,  5.0
VMAX_MIN, VMAX_MAX = 0.01,  0.30

Ki_grid   = np.logspace(np.log10(KI_MIN), np.log10(KI_MAX), N_KI)
Vmax_grid = np.linspace(VMAX_MIN, VMAX_MAX, N_VMAX)


# ── Safe norm ─────────────────────────────────────────────────────────────────
def _safe_norm(pct_2d: np.ndarray):
    """Return (norm, cmap) that never raises regardless of data range."""
    valid = pct_2d[~np.isnan(pct_2d)]
    if len(valid) == 0:
        return Normalize(vmin=0, vmax=1), plt.cm.Greys

    vmin = float(np.nanmin(pct_2d))
    vmax = float(np.nanmax(pct_2d))
    vmax = min(vmax, 80.0)
    vmin = max(vmin, -10.0)
    if abs(vmax - vmin) < 0.5:
        vmax = vmin + 0.5

    if vmin >= 0:
        return Normalize(vmin=0.0, vmax=max(vmax, 1.0)), plt.cm.YlOrRd
    elif vmax <= 0:
        return Normalize(vmin=min(vmin, -1.0), vmax=0.0), plt.cm.YlGn_r
    else:
        return (TwoSlopeNorm(vmin=min(vmin, -0.1),
                             vcenter=0.0,
                             vmax=max(vmax, 0.1)),
                plt.cm.RdYlGn_r)


# ── Compound update helpers ───────────────────────────────────────────────────
def _force_ki(ki_value: float) -> None:
    """Override Ki in all live toxin dicts after initialize_compounds()."""
    import compounds as C
    for tox, key in [(C.tox1, "Ki_furo_to_IS_mgL"),
                     (C.tox2, "Ki_furo_to_HA_mgL"),
                     (C.tox3, "Ki_furo_to_pCS_mgL")]:
        if isinstance(tox, dict):
            tox[key] = float(ki_value)


def _set_vmax_scale(vmax_scale: float) -> None:
    """Override CKD-5 basolateral Vmax scale for IS and pCS."""
    try:
        import compound_parameters as CP
        CP._CKD_SCALES["per_compound"]["IS"]["baso"]  = vmax_scale
        CP._CKD_SCALES["per_compound"]["pCS"]["baso"] = vmax_scale
        CP.CKD_ADJUSTMENTS = CP._create_ckd_adjustments_dynamic(CP._CKD_SCALES)
    except Exception as e:
        print(f"    [warn] _set_vmax_scale: {e}")


# ── Single simulation ─────────────────────────────────────────────────────────
def run_ckd_simulation(ki_value: float, vmax_scale: float, toxin: str,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Run one Ki × Vmax grid point:
      1. 500h CKD-5 warm-up (no drug) → toxin steady state
      2. 24h no-drug run from SS → baseline Css
      3. 24h with-drug run from SS → inhibited Css
      4. %Δ = (Css_drug - Css_no) / Css_no × 100

    Css extracted from C_tox*_cen [mg/L plasma] — NO BP division.
    """
    result = {
        "ki": ki_value, "vmax_scale": vmax_scale, "toxin": toxin,
        "success": False,
        "Css_no_drug": np.nan, "Css_with_drug": np.nan,
        "pct_change": np.nan, "error": None,
    }
    try:
        import constants as K_mod
        K_mod.VERBOSE = False          # silence ODE output

        from simulation import run_sim
        from state_vector import IDX
        import compounds as C_mod

        _set_vmax_scale(vmax_scale)    # must come before initialize_compounds

        # ── Step 1: warm-up (no drug) ─────────────────────────────────────
        C_mod.initialize_compounds("ckd5")
        _force_ki(ki_value)
        C_mod.drug.update({
            "enabled":   False, "IR": 0.0,
            "t_start_h": 1e9,   "t_inf_h": 0.0,
            "bolus_mg":  0.0,   "inhibit": False,
        })
        sol_ss = run_sim(t_end_h=WARMUP_H, mode="ckd5", dt=DT_WARMUP)
        if sol_ss is None:
            result["error"] = "Warm-up failed"; return result

        # Extract end-of-warmup IC; zero drug compartments
        y_end   = sol_ss.y[:, -1]
        ic_base = {name: float(y_end[idx]) for name, idx in IDX.items()}
        ic_drug = {k: (0.0 if k in ("C_ven", "C_peri", "C_bladder", "Ae_drg")
                       else v)
                   for k, v in ic_base.items()}

        # ── Step 2: no-drug run (24h) ─────────────────────────────────────
        C_mod.initialize_compounds("ckd5")
        _force_ki(ki_value)
        C_mod.drug.update({
            "enabled":   False, "IR": 0.0,
            "t_start_h": 1e9,   "t_inf_h": 0.0,
            "bolus_mg":  0.0,   "inhibit": False,
        })
        sol_no = run_sim(t_end_h=SIM_HOURS, mode="ckd5", dt=DT_SIM,
                         custom_initial_conds=ic_base)
        if sol_no is None:
            result["error"] = "No-drug run failed"; return result

        # ── Step 3: with-drug run (120 mg IV over 2h, 24h total) ──────────
        C_mod.initialize_compounds("ckd5")
        _force_ki(ki_value)
        C_mod.drug.update({
            "enabled":   True,
            "IR":        DOSE_MG / INFUSION_H,   # 60 mg/h
            "t_inf_h":   INFUSION_H,              # 2h
            "t_start_h": 0.05,
            "bolus_mg":  0.0,
            "inhibit":   True,
        })
        for attr in ("tox1", "tox2", "tox3"):
            tox = getattr(C_mod, attr, None)
            if isinstance(tox, dict):
                tox["enabled"] = True
        sol_drug = run_sim(t_end_h=SIM_HOURS, mode="ckd5", dt=DT_SIM,
                           custom_initial_conds=ic_drug)
        if sol_drug is None:
            result["error"] = "Drug run failed"; return result

        # ── Step 4: extract Css at t=SIM_HOURS ────────────────────────────
        # C_tox*_cen is total PLASMA concentration [mg/L] — no BP division.
        key_idx  = "C_tox1_cen" if toxin == "IS" else "C_tox3_cen"
        Css_no   = float(sol_no.y[IDX[key_idx],   -1])
        Css_drug = float(sol_drug.y[IDX[key_idx], -1])

        pct = (Css_drug - Css_no) / Css_no * 100.0 if Css_no > 1e-9 else 0.0
        result.update({
            "Css_no_drug":   Css_no,
            "Css_with_drug": Css_drug,
            "pct_change":    pct,
            "success":       True,
        })
        if verbose:
            print(f"      Ki={ki_value:.3f}, scale={vmax_scale:.3f}:  "
                  f"no-drug={Css_no:.3f}  drug={Css_drug:.3f}  "
                  f"Δ={pct:+.1f}%")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"      ERROR: {e}")
    return result


# ── Heatmap computation with caching ─────────────────────────────────────────
def compute_heatmap(toxin: str, use_cache: bool = True) -> np.ndarray:
    """
    Compute %Δ grid for one toxin with pickle caching.

    Cache key encodes: toxin, ki, vmax_scale, base Ki from CP,
    base R0, SIM_HOURS, N_KI, N_VMAX — so any parameter change
    or grid size change invalidates stale entries automatically.
    """
    import compound_parameters as _CP

    cache: dict = {}
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"  Loaded {len(cache)} cached entries")
        except Exception:
            pass

    pct_2d    = np.full((N_VMAX, N_KI), np.nan)
    total     = N_VMAX * N_KI
    completed = 0
    cached_n  = 0
    t0        = time.time()

    print(f"\n  [{toxin}] {total} grid points  "
          f"({N_KI} Ki × {N_VMAX} Vmax)  |  "
          f"{SIM_HOURS:.0f}h simulation per point")

    _ki_base = float(_CP.TOX1_BASE.get("Ki_furo_to_IS_mgL", 0.35))
    _r0_base = float(_CP.TOX1_BASE.get("R0", 3.5))

    for i, scale in enumerate(Vmax_grid):
        for j, ki in enumerate(Ki_grid):
            # Cache key includes SIM_HOURS and grid dimensions
            key = (f"{toxin}_{ki:.6f}_{scale:.6f}"
                   f"_base{_ki_base:.4f}_r0{_r0_base:.3f}"
                   f"_t{SIM_HOURS:.0f}_nki{N_KI}_nv{N_VMAX}")

            if key in cache:
                pct_2d[i, j] = cache[key]
                cached_n += 1
                completed += 1
                continue

            elapsed   = time.time() - t0
            rate      = completed / max(elapsed, 1e-9)
            remaining = (total - completed) / max(rate, 1e-9)
            print(f"    [{completed+1:3d}/{total}] "
                  f"Ki={ki:.3f} mg/L  scale={scale:.3f}  "
                  f"ETA {remaining/60:.1f} min")

            res = run_ckd_simulation(ki, scale, toxin, verbose=True)

            if res["success"]:
                pct_2d[i, j] = res["pct_change"]
                cache[key]   = res["pct_change"]
            else:
                print(f"      FAILED: {res['error']}")

            completed += 1

            # Checkpoint save every 5 new points
            if completed % 5 == 0:
                try:
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(cache, f)
                except Exception:
                    pass

    # Final save
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass

    elapsed = time.time() - t0
    n_ok    = int(np.sum(~np.isnan(pct_2d)))
    print(f"\n  [{toxin}] Done in {elapsed/60:.1f} min  |  "
          f"from cache={cached_n}  new={total-cached_n}  "
          f"OK={n_ok}/{total}")
    return pct_2d


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(IS_pct: np.ndarray, pCS_pct: np.ndarray,
                current_Ki: float, current_scale: float) -> None:

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       11,
        "axes.linewidth":  0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.08, right=0.96, top=0.88,
                        bottom=0.14, wspace=0.32)

    for ax, pct_2d, toxin in zip(axes, [IS_pct, pCS_pct], ["IS", "pCS"]):

        if np.all(np.isnan(pct_2d)):
            ax.text(0.5, 0.5, f"No valid data for {toxin}",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            ax.set_title(f"{toxin} — NO DATA",
                         fontsize=13, fontweight="bold")
            continue

        norm, cmap = _safe_norm(pct_2d)

        im = ax.pcolormesh(Ki_grid, Vmax_grid, pct_2d,
                           cmap=cmap, norm=norm, shading="auto")

        # +30% clinical contour (Costes-Albrespic 2025)
        if np.nanmax(pct_2d) >= 30.0:
            try:
                cs = ax.contour(Ki_grid, Vmax_grid, pct_2d,
                                levels=[30.0],
                                colors=["white"],
                                linewidths=2.2, linestyles=["--"])
                ax.clabel(cs, fmt="+30% (clinical)",
                          fontsize=9, colors="white", inline=True)
            except Exception:
                pass

        # +10% contour
        if np.nanmax(pct_2d) >= 10.0:
            try:
                cs10 = ax.contour(Ki_grid, Vmax_grid, pct_2d,
                                  levels=[10.0],
                                  colors=["#FFFFAA"],
                                  linewidths=1.0, linestyles=[":"])
                ax.clabel(cs10, fmt="+10%",
                          fontsize=8, colors="#FFFFAA", inline=True)
            except Exception:
                pass

        # Current model operating point
        ax.scatter([current_Ki], [current_scale],
                   s=250, marker="*",
                   color="white", edgecolors="black",
                   linewidths=1.2, zorder=10,
                   label=(f"Current model\n"
                          f"$K_i$={current_Ki} mg/L, "
                          f"scale={current_scale}"))

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta C_{ss}$ vs no-drug CKD-5 (%)",
                       fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        ax.set_xscale("log")
        ax.set_xlabel(
            r"$K_i$ — furosemide OAT1 inhibition constant (mg/L)",
            fontsize=12)
        ax.set_ylabel(
            "CKD-5 basolateral OAT1 $V_{max}$ scale\n"
            "(fraction of healthy $V_{max}$)",
            fontsize=12)
        ax.set_title(
            f"{toxin}   |   ODE CKD-5  |  "
            f"120 mg IV  |  t=0–{SIM_HOURS:.0f}h",
            fontsize=13, fontweight="bold", loc="left")
        ax.legend(fontsize=9, loc="upper right",
                  framealpha=0.90, edgecolor="#cccccc")
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    fig.suptitle(
        r"Sensitivity heatmap: $K_i$ × CKD-5 OAT1 $V_{max}$ scale"
        r" $\rightarrow$ toxin plasma accumulation"
        "\n"
        f"Grid: {N_KI}×{N_VMAX}  |  "
        f"120 mg IV furosemide  |  "
        f"CKD-5  |  "
        "White dashed = +30% (Costes-Albrespic 2025)",
        fontsize=11, fontweight="bold", y=0.995,
    )

    fig.text(
        0.5, 0.01,
        f"Acute IV scenario (t=0–{SIM_HOURS:.0f}h); "
        "chronic oral SS (30-day) gives larger effects — "
        "see heatmap_fu_ode_real.py for chronic comparison.  "
        r"$\bigstar$ = current model ($K_i$=0.35 mg/L, scale=0.05).",
        ha="center", fontsize=8, color="#666", style="italic",
    )

    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_PDF,           bbox_inches="tight", facecolor="white")
    print(f"\n[saved] {FIG_PNG}")
    print(f"[saved] {FIG_PDF}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Silence all ODE output before anything runs
    import constants as K
    K.VERBOSE = False

    print("=" * 70)
    print(f"HEATMAP Ki × Vmax  —  v4  |  "
          f"{N_KI}×{N_VMAX} grid  |  {SIM_HOURS:.0f}h simulation")
    print("=" * 70)

    # ── Optional: clear old cache ─────────────────────────────────────────
    if CLEAR_OLD_CACHE and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"[CACHE] Deleted old cache: {CACHE_FILE}")
        print(f"        All {N_KI*N_VMAX} points will be re-computed.")
    elif os.path.exists(CACHE_FILE):
        print(f"[CACHE] Existing cache found — new grid keys will be computed,")
        print(f"        matching old keys reused if sim_hours/grid match.")

    # ── Current model point ───────────────────────────────────────────────
    try:
        import compounds as C
        current_Ki = (float(C.tox1.get("Ki_furo_to_IS_mgL", 0.35))
                      if isinstance(C.tox1, dict) else 0.35)
    except Exception:
        current_Ki = 0.35
    current_scale = 0.05

    print(f"\nCurrent model point:  Ki={current_Ki} mg/L, "
          f"Vmax scale={current_scale}")
    print(f"Grid:  {N_KI} Ki values  ×  {N_VMAX} Vmax scales  "
          f"=  {N_KI*N_VMAX} simulations per toxin")
    print(f"Simulation:  {WARMUP_H:.0f}h warm-up + {SIM_HOURS:.0f}h "
          f"(no-drug) + {SIM_HOURS:.0f}h (with-drug)  per point")

    # ── Run grids ─────────────────────────────────────────────────────────
    IS_pct  = compute_heatmap("IS",  use_cache=True)
    pCS_pct = compute_heatmap("pCS", use_cache=True)

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if np.any(~np.isnan(IS_pct)):
        print(f"IS  range: {np.nanmin(IS_pct):.1f}% "
              f"to {np.nanmax(IS_pct):.1f}%")
    if np.any(~np.isnan(pCS_pct)):
        print(f"pCS range: {np.nanmin(pCS_pct):.1f}% "
              f"to {np.nanmax(pCS_pct):.1f}%")

    i_scale = np.argmin(np.abs(Vmax_grid - current_scale))
    print(f"\n  At Vmax scale = {Vmax_grid[i_scale]:.3f}:")
    print(f"  {'Ki (mg/L)':>12}  {'IS %Δ':>10}  {'pCS %Δ':>10}  "
          f"{'IS≥30%':>8}  {'pCS≥30%':>8}")
    print("  " + "─" * 55)
    for Ki_ref in [5.0, 2.0, 1.0, 0.5, 0.35, 0.2, 0.1, 0.05]:
        j     = np.argmin(np.abs(Ki_grid - Ki_ref))
        is_v  = IS_pct[i_scale,  j]
        pcs_v = pCS_pct[i_scale, j]
        is_s  = f"{is_v:>+10.1f}" if not np.isnan(is_v)  else "       N/A"
        pcs_s = f"{pcs_v:>+10.1f}"if not np.isnan(pcs_v) else "       N/A"
        is_30  = "YES" if (not np.isnan(is_v)  and is_v  >= 30) else "NO"
        pcs_30 = "YES" if (not np.isnan(pcs_v) and pcs_v >= 30) else "NO"
        print(f"  {Ki_ref:>12.2f}  {is_s}  {pcs_s}  "
              f"{is_30:>8}  {pcs_30:>8}")

    # ── Figure ────────────────────────────────────────────────────────────
    make_figure(IS_pct, pCS_pct, current_Ki, current_scale)
    print("\nDone.")


if __name__ == "__main__":
    main()