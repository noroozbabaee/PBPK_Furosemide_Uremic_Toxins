# -*- coding: utf-8 -*-
"""
figure_paper_final_v5.py
=========================
Publication-quality 4-panel figure for the paper — PARAMETERIZED VERSION.

This is a drop-in replacement for figure_paper_final_v4.py. The LAYOUT,
PANELS, COLORS, FONT SIZES, and ANNOTATIONS are UNCHANGED from v4 — only
the data source is now configurable so it works with any run produced by
the updated Dose_response_3timesperday.py (which now saves two runs per
execution: Run A = TID, Run B = QD, both at the SAME total daily dose).

WHAT CHANGED vs v4
-------------------
  - Cache path is no longer hardcoded to "figures/chronic_oral_cache.npz".
    Instead it is built from RUN_TAG, matching the naming convention used
    by Dose_response_3timesperday.py:
        figures/chronic_oral_cache_{RUN_TAG}.npz
  - Css_avg, Ki, fu_p, BP, and ODE %Δ are no longer hardcoded — they are
    read from the matching results_{RUN_TAG}.npz file (saved alongside
    the cache by the updated dose-response script). This guarantees the
    figure always reflects the actual Ki/dose/Css used in that specific run.
  - Analytical and clinical reference values are also read from the
    results file (so they update automatically if CLIN_* or Ki change).
  - Output figure filename includes RUN_TAG so TID and QD figures never
    collide.

HOW TO USE
----------
  1. Run Dose_response_3timesperday.py once. It produces, per regimen:
       figures/chronic_oral_cache_{TAG}.npz   (raw ODE arrays)
       figures/results_{TAG}.npz              (derived metrics + run params)
     where TAG = Ki{Ki}_dose{DOSE}mg_x{N}pd_{DAYS}days

  2. Set RUN_TAG below to the TID tag for Figure A (3×80mg etc.), run this
     script → produces figure_paper_final_v5_{TAG}.png/.pdf

  3. Set RUN_TAG to the QD tag (1×240mg etc.), run this script again →
     produces a SECOND figure for the once-daily comparison, same layout.

  Nothing else needs to change — panels A/B/C/D are pixel-for-pixel
  identical in structure to v4, just driven by whichever run you select.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# ── Path setup ───────────────────────────────────────────────────────────────
# This script lives in figures/. It reads .npz cache/results files produced
# by analysis/dose_response_tid_vs_qd.py, which writes them to ../results/
# (shared top-level results folder), not a local "figures/" subfolder.
# This script's OWN output (the rendered comparison figure) also goes to
# ../results/, alongside the data it was built from.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — set this to the run you want to plot
# ═════════════════════════════════════════════════════════════════════════════
# Examples (must match a tag actually produced by Dose_response_3timesperday.py):
#   "Ki0p35_dose80mg_x3pd_30days"    → TID  3×80mg  (240 mg/day)
#   "Ki0p35_dose240mg_x1pd_30days"   → QD   1×240mg (240 mg/day, same total)
#
# Can also be passed as a command-line argument:
#   python figure_paper_final_v5.py Ki0p35_dose80mg_x3pd_30days
RUN_TAG = sys.argv[1] if len(sys.argv) > 1 else "Ki0p35_dose80mg_x3pd_30days"

cache_path   = os.path.join(RESULTS_DIR, f"chronic_oral_cache_{RUN_TAG}.npz")
results_path = os.path.join(RESULTS_DIR, f"results_{RUN_TAG}.npz")

# ── Load ODE timecourse from cache ─────────────────────────────────────────────
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"Cache not found: {cache_path}\n"
        f"Run analysis/dose_response_tid_vs_qd.py first with matching "
        f"parameters to generate it.\n"
        f"Available caches in {RESULTS_DIR}:\n  " +
        "\n  ".join(sorted(
            f for f in os.listdir(RESULTS_DIR) if f.startswith("chronic_oral_cache_")
        ) or [f"(none found in {RESULTS_DIR})"])
    )

cache    = np.load(cache_path, allow_pickle=False)
t_ts     = cache["t_all"]
IS_ts    = cache["IS_all"]
pCS_ts   = cache["pCS_all"]
drg_ts   = cache["drg_all"]
IS_base  = float(cache["IS_base"][0])
pCS_base = float(cache["pCS_base"][0])
BP_IS, BP_pCS = 0.70, 0.55

print(f"[CACHE] Loaded {len(t_ts)} time points from {cache_path}")
print(f"  IS_base  = {IS_base:.3f} mg/L blood  ({IS_base/BP_IS:.3f} plasma)")
print(f"  pCS_base = {pCS_base:.3f} mg/L blood  ({pCS_base/BP_pCS:.3f} plasma)")

# ── Load derived metrics + run parameters from results file ───────────────────
if not os.path.exists(results_path):
    available_results = sorted(
        f for f in os.listdir(RESULTS_DIR) if f.startswith("results_")
    )
    # Try to find a results file with the same Ki/dose/n_doses but a
    # different day count — the most common cause of this mismatch is an
    # old cache saved before the "_NNdays" suffix was added to the tag.
    base_tag_no_days = RUN_TAG.rsplit("_", 1)[0] if "days" in RUN_TAG else RUN_TAG
    close_matches = [f for f in available_results if base_tag_no_days in f]

    msg = (
        f"Results file not found: {results_path}\n\n"
        f"This usually means the cache file you loaded "
        f"({os.path.basename(cache_path)}) is STALE — it was produced by an "
        f"earlier version of analysis/dose_response_tid_vs_qd.py, before the "
        f"results_*.npz export and/or the '_NNdays' tag suffix existed.\n\n"
    )
    if close_matches:
        msg += (
            f"Did you mean one of these (same Ki/dose, different day count)?\n  "
            + "\n  ".join(close_matches) + "\n\n"
        )
    msg += (
        f"All available results_*.npz files in {RESULTS_DIR}:\n  "
        + "\n  ".join(available_results or [f"(none found in {RESULTS_DIR})"]) + "\n\n"
        f"Fix: delete the stale cache and results files for this tag, then "
        f"re-run analysis/dose_response_tid_vs_qd.py to regenerate both files "
        f"together with matching tags:\n"
        f"  rm {os.path.join(RESULTS_DIR, f'chronic_oral_cache_{RUN_TAG}.npz')} "
        f"{os.path.join(RESULTS_DIR, f'results_{RUN_TAG}.npz')}   (if either exists)\n"
        f"  python analysis/dose_response_tid_vs_qd.py"
    )
    raise FileNotFoundError(msg)

res = np.load(results_path, allow_pickle=False)

Ki        = float(res["Ki"][0])
fu_p      = 0.05  # furosemide fu_p (CKD5) — fixed compound property
BP        = 0.82  # furosemide blood:plasma ratio — fixed compound property
N_DAYS    = int(res["n_days"][0])
DOSE_PER_ADMIN  = float(res["dose_per_admin"][0])
N_DOSES_PER_DAY = int(res["n_doses"][0])

ODE_IS   = float(res["pct_IS"][0])
ODE_pCS  = float(res["pct_pCS"][0])
ANAL_IS  = float(res["anal_IS_pct"][0])
ANAL_pCS = float(res["anal_pCS_pct"][0])
Css_avg  = float(res["anal_Css_furo"][0])

# ODE-fed diagnostic analytical (NOT independent — see analytical_ss_ode_css
# docstring in Dose_response_3timesperday.py). Loaded for optional use /
# console reporting but NOT plotted as a 4th bar in Panel C by default,
# to keep the published 3-way comparison clean. Available here if you want
# to add a 4th diagnostic bar or report it in text.
try:
    ANAL_ODE_IS  = float(res["anal_ode_IS_pct"][0])
    ANAL_ODE_pCS = float(res["anal_ode_pCS_pct"][0])
    Css_avg_ode  = float(res["anal_ode_Css_furo"][0])
    _has_anal_ode = True
except KeyError:
    # Older results file saved before this diagnostic was added
    ANAL_ODE_IS = ANAL_ODE_pCS = Css_avg_ode = None
    _has_anal_ode = False

CLIN_IS     = float(res["clin_IS_pct"][0])
CLIN_pCS    = float(res["clin_pCS_pct"][0])
CLIN_IS_lo  = float(res["clin_IS_lo"][0])
CLIN_IS_hi  = float(res["clin_IS_hi"][0])
CLIN_pCS_lo = float(res["clin_pCS_lo"][0])
CLIN_pCS_hi = float(res["clin_pCS_hi"][0])

print(f"[RESULTS] Loaded from {results_path}")
print(f"  Regimen   : {N_DOSES_PER_DAY}×{DOSE_PER_ADMIN:.0f} mg "
      f"({DOSE_PER_ADMIN*N_DOSES_PER_DAY:.0f} mg/day total), {N_DAYS} days")
print(f"  Ki        = {Ki:.3f} mg/L")
print(f"  ODE  IS %Δ = {ODE_IS:+.1f}%   ODE  pCS %Δ = {ODE_pCS:+.1f}%")
print(f"  Anal IS %Δ (theoretical, independent) = {ANAL_IS:+.1f}%   "
      f"pCS %Δ = {ANAL_pCS:+.1f}%")
if _has_anal_ode:
    print(f"  Anal IS %Δ (ODE-fed, DIAGNOSTIC ONLY) = {ANAL_ODE_IS:+.1f}%   "
          f"pCS %Δ = {ANAL_ODE_pCS:+.1f}%")
    print(f"    (uses ODE's measured Css={Css_avg_ode:.3f} mg/L as input — "
          f"not an independent prediction, see analytical_ss_ode_css() docstring)")
print(f"  Clin IS %Δ = {CLIN_IS:+.1f}%   Clin pCS %Δ = {CLIN_pCS:+.1f}%")

TAU_H = 24.0

# ── Last-day cycle ─────────────────────────────────────────────────────────────
mask_last  = t_ts >= (N_DAYS - 1) * TAU_H
t_last     = t_ts[mask_last]
drg_last_blood = np.asarray(drg_ts[mask_last], dtype=float)
t_last_rel = t_last - t_last[0]   # 0–24h relative

def _trapz(y, x):
    """Version-safe trapezoidal integration (NumPy >=2.x removed np.trapz)."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

# Cu_furo = C_blood × fu_p / BP  (= C_plasma × fu_p)
alpha_last = 1.0 + drg_last_blood * fu_p / BP / Ki

IS_ts    = np.asarray(IS_ts, dtype=float) / BP_IS
pCS_ts   = np.asarray(pCS_ts, dtype=float) / BP_pCS
drg_ts   = np.asarray(drg_ts, dtype=float) / BP
IS_base  = float(IS_base) / BP_IS
pCS_base = float(pCS_base) / BP_pCS
IS_last  = IS_ts[mask_last]
pCS_last = pCS_ts[mask_last]
drg_last = drg_ts[mask_last]
dt_last  = t_last[-1] - t_last[0]
drg_avg  = float(_trapz(drg_last, t_last) / dt_last)

print(f"  Furo ODE avg (last cycle) = {drg_avg:.3f} mg/L plasma")

idx_cmax = int(np.argmax(drg_last))
Cmax_ode = float(drg_last[idx_cmax])
Tmax_ode = float(t_last_rel[idx_cmax])

active       = drg_last > 1.0
t_active_end = float(t_last_rel[active][-1]) if active.any() else 8.0

# ── Global rcParams ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         15,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
})

# ── Figure layout — 2×2 grid (UNCHANGED from v4) ────────────────────────────────
fig = plt.figure(figsize=(18, 14), dpi=300)
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1.1, 1.2],
    hspace=0.38,
    wspace=0.36,
    left=0.07, right=0.96, top=0.95, bottom=0.07
)
ax_A = fig.add_subplot(gs[0, 0])   # IS + pCS 30-day
ax_B = fig.add_subplot(gs[0, 1])   # Furosemide 30-day
ax_C = fig.add_subplot(gs[1, 0])   # Bar comparison
ax_D = fig.add_subplot(gs[1, 1])   # Mechanistic last-cycle

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — IS and pCS 30-day time course
# ══════════════════════════════════════════════════════════════════════════════
t_days = t_ts / TAU_H

ax_A.plot(t_days, IS_ts,  color="#1A5276", lw=0.9, label="IS (ODE)",  alpha=0.9)
ax_A.plot(t_days, pCS_ts, color="#7D3C98", lw=0.9, label="pCS (ODE)",
          alpha=0.9, ls="--")

ax_A.axhline(IS_base,  color="#1A5276", ls=":", lw=1.0, alpha=0.5)
ax_A.axhline(pCS_base, color="#7D3C98", ls=":", lw=1.0, alpha=0.5)

IS_ss  = IS_base  * (1 + ODE_IS /100)
pCS_ss = pCS_base * (1 + ODE_pCS/100)
ax_A.axhline(IS_ss,  color="#1A5276", ls="-.", lw=1.2, alpha=0.7)
ax_A.axhline(pCS_ss, color="#7D3C98", ls="-.", lw=1.2, alpha=0.7)

ax_A.text(30.3, IS_base,
          f"IS baseline\n{IS_base:.1f} mg/L",
          fontsize=14, color="#1A5276", va="center")
ax_A.text(30.3, IS_ss,
          f"IS SS\n{IS_ss:.1f} mg/L",
          fontsize=14, color="#1A5276", va="center")
ax_A.text(30.3, pCS_base,
          f"pCS baseline\n{pCS_base:.1f} mg/L",
          fontsize=14, color="#7D3C98", va="center")
ax_A.text(30.3, pCS_ss,
          f"pCS SS\n{pCS_ss:.1f} mg/L ",
          fontsize=14, color="#7D3C98", va="center")

ax_A.set_xlabel("Time (days)", fontsize=15)
ax_A.set_ylabel("Uremic toxin plasma concentration (mg/L)", fontsize=15)
ax_A.set_xlim(0, 30)
ax_A.set_title("A", fontweight="bold", loc="left", fontsize=17)
ax_A.legend(fontsize=13, frameon=False, loc="center right")
ax_A.tick_params(labelsize=13)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Furosemide 30-day time course
# ══════════════════════════════════════════════════════════════════════════════
ax_B.plot(t_days, drg_ts, color="#BA7517", lw=0.7, alpha=0.88,
          label="Furosemide plasma (ODE)")

# Analytical Css,avg reference
ax_B.axhline(Css_avg, color="#2E86C1", ls="--", lw=1.8,
             label=f"Analytical $C_{{ss,avg}}$ = {Css_avg:.2f} mg/L\n"
                  )

# ODE time-average last cycle
ax_B.axhline(drg_avg, color="#BA7517", ls="-.", lw=1.4, alpha=0.8,
             label=f"ODE avg (day {N_DAYS}) = {drg_avg:.3f} mg/L")

# Shade last cycle to highlight
ax_B.axvspan(N_DAYS - 1, N_DAYS, alpha=0.08, color="#BA7517", zorder=0)

ax_B.set_xlabel("Time (days)", fontsize=15)
ax_B.set_ylabel("Furosemide plasma concentration (mg/L)", fontsize=15)
ax_B.set_xlim(0, 30)
ax_B.set_ylim(bottom=0)
ax_B.set_title("B", fontweight="bold", loc="left", fontsize=17)
ax_B.legend(fontsize=14, frameon=False, loc="upper right")
ax_B.tick_params(labelsize=13)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Three-way bar comparison
# ══════════════════════════════════════════════════════════════════════════════
x = np.array([0.0, 1.0])
w = 0.22

COL_ODE  = "#2C5F8A"
COL_ANAL = "#E07B39"
COL_CLIN = "#C0392B"

ode_pcts  = [ODE_IS,  ODE_pCS]
anal_pcts = [ANAL_IS, ANAL_pCS]
clin_pcts = [CLIN_IS, CLIN_pCS]
clin_lo   = [CLIN_IS_lo,  CLIN_pCS_lo]
clin_hi   = [CLIN_IS_hi,  CLIN_pCS_hi]

for i, (ov, av, cv, lo, hi) in enumerate(
        zip(ode_pcts, anal_pcts, clin_pcts, clin_lo, clin_hi)):
    b1 = ax_C.bar(x[i]-w, ov, w*0.9, color=COL_ODE,  alpha=0.88,
                  label=f"ODE: {N_DAYS}d {N_DOSES_PER_DAY}x/day oral" if i==0 else "")
    b2 = ax_C.bar(x[i],   av, w*0.9, color=COL_ANAL, alpha=0.88,
                  label="Analytical (exact MM)" if i==0 else "")
    b3 = ax_C.bar(x[i]+w, cv, w*0.9, color=COL_CLIN, alpha=0.88,
                  label="Clinical (Costes-Albrespic 2025)" if i==0 else "")
    ax_C.errorbar(x[i]+w, cv, yerr=[[cv-lo],[hi-cv]],
                  fmt="none", color="black", capsize=5, lw=1.6, zorder=10)
    for bar, val in [(b1,ov),(b2,av),(b3,cv)]:
        ax_C.text(bar[0].get_x()+bar[0].get_width()/2,
                  val+0.6, f"{val:+.0f}%",
                  ha="center", va="bottom", fontsize=13, fontweight="bold")

for i, (cv, lo, hi) in enumerate(zip(clin_pcts, clin_lo, clin_hi)):
    ax_C.fill_between([x[i]+w-w*0.45, x[i]+w+w*0.45],
                      [lo, lo], [hi, hi],
                      color=COL_CLIN, alpha=0.08, zorder=0)

ax_C.axhline(0, color="0.4", lw=0.8, ls="--")
ax_C.set_xticks(x)
ax_C.set_xticklabels(["Indoxyl Sulphate (IS)",
                       "p-Cresyl Sulphate (pCS)"], fontsize=13)
ax_C.set_ylabel("$\\Delta C_{ss}$ vs no-drug baseline (%)", fontsize=14)
ax_C.set_ylim(0, 70)
ax_C.set_title("C  ",
               fontweight="bold", loc="left", fontsize=17)
ax_C.legend(fontsize=14, frameon=True, framealpha=0.95,
            edgecolor="lightgray", loc="upper left",
            handlelength=1.2, borderpad=0.6)
ax_C.tick_params(labelsize=13)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Mechanistic last-cycle panel
# ══════════════════════════════════════════════════════════════════════════════
ax_D2 = ax_D.twinx()

ax_D.fill_between(t_last_rel, drg_last, alpha=0.20, color="#BA7517")
ax_D.plot(t_last_rel, drg_last, color="#BA7517", lw=2.0,
          label=r"Furosemide plasma $(t)$ [ODE, day " + f"{N_DAYS}]")
ax_D.axhline(Css_avg, color="#2E86C1", ls="--", lw=1.6,
             label=f"Analytical $C_{{ss,avg}}$ = {Css_avg:.2f} mg/L")

# Active window
ax_D.axvspan(0, t_active_end, alpha=0.06, color="#BA7517", zorder=0)
ax_D.axvline(t_active_end, color="#BA7517", lw=0.8, ls=":", alpha=0.6)
ax_D.text(t_active_end/2, Cmax_ode * 0.6,
          f"Active\n{t_active_end:.0f}h / 24h",
          ha="center", fontsize=13, color="#8B4513")

# Alpha
ax_D2.plot(t_last_rel, alpha_last, color="#C0392B", lw=1.6, ls="-.",
           label=r"$\alpha = 1 + C_{u,furo}/K_i$", alpha=0.85)
ax_D2.axhline(1.0, color="#C0392B", lw=0.7, ls=":", alpha=0.5)
ax_D2.set_ylabel(r"OAT1 inhibition factor $\alpha$", fontsize=14,
                 color="#C0392B")
ax_D2.tick_params(labelsize=13, colors="#C0392B")
ax_D2.spines["right"].set_visible(True)
ax_D2.spines["right"].set_color("#C0392B")
ax_D2.spines["right"].set_linewidth(0.8)
ax_D2.set_ylim(0.9, max(5.5, float(alpha_last.max()) * 1.1))

# IS band
IS_c_val = IS_base * (1 + ODE_IS/100)
ax_D.axhspan(IS_c_val-0.3, IS_c_val+0.3, alpha=0.15, color="#1A5276")
ax_D.text(18, IS_c_val,
          f"IS ≈ {IS_c_val:.1f} mg/L ({ODE_IS:+.1f}%)\n"
          r"$\tau_{IS}=51\,\mathrm{h} \gg 6\,\mathrm{h}$",
          fontsize=13, color="#1A5276", va="center")

# Cmax annotation
ax_D.annotate(
    f"$C_{{max}}$={Cmax_ode:.0f} mg/L\n$T_{{max}}$={Tmax_ode*60:.0f} min",
    xy=(Tmax_ode, Cmax_ode),
    xytext=(Tmax_ode + 2, max(Cmax_ode - 2, 0.1)),
    fontsize=13, color="#8B4513",
    arrowprops=dict(arrowstyle="->", color="#8B4513", lw=1.0)
)

# t_active / tau_IS annotation
ax_D.annotate("",
    xy=(t_active_end, 2.5), xytext=(0, 2.5),
    arrowprops=dict(arrowstyle="<->", color="#BA7517", lw=1.2))
ax_D.text(t_active_end/2, 3.0,
          r"$t_{active}$/$\tau_{IS}$ = " +
          f"{t_active_end:.0f}/51 = {t_active_end/51:.2f}",
          ha="center", fontsize=14, color="#8B4513")

ax_D.set_xlabel("Time after dose (h)", fontsize=14)
ax_D.set_ylabel("Furosemide plasma concentration (mg/L)", fontsize=14)
ax_D.set_xlim(0, TAU_H)
ax_D.set_ylim(0, Cmax_ode * 1.2)
ax_D.set_title("D ",
               fontweight="bold", loc="left", fontsize=17)
ax_D.tick_params(labelsize=13)

h1, l1 = ax_D.get_legend_handles_labels()
h2, l2 = ax_D2.get_legend_handles_labels()
ax_D.legend(h1+h2, l1+l2, fontsize=14, frameon=False, loc="upper right")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    p = os.path.join(RESULTS_DIR, f"three_way_comparison_{RUN_TAG}.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
plt.close()
print("Done.")
