# How CKD-5 chronic oral runs are supposed to start

This is the protocol in `analysis/dose_response_tid_vs_qd.py`.  
If a figure or cache does not match this, it is not a valid “same baseline” comparison.

## Three different things (do not mix)

| Name | Drug? | What it is | What it is for |
|---|---|---|---|
| **A. Warm-up SS** | **No** | 500 h, `MODE=ckd5`, `drug.enabled=False` | The **only** toxin initial condition for chronic oral. Blood: IS ≈ 19.71, pCS ≈ 13.74, HA ≈ 120. Plasma: IS 28.16, pCS 24.99, HA 142.86. |
| **B. Live-PK 24 h** | **Yes, IV 120 mg / 2 h**, `inhibit=False` | Starts from A, then one IV dose to measure \(CL\) | **Not** chronic oral. **Not** the no-drug time course. Do not plot this as a figure arm. |
| **C. Chronic oral 30 d** | **Yes, oral** (or zero doses) | Starts from A; drug compartments zeroed; toxins **kept** at SS | With-drug (or true no-drug if 0 doses). %Δ vs A, last-day mean. |

Healthy (`IS_base ≈ 0.54` blood) is **none of the above**.

## How arm C is initiated (code)

```text
ic_ss = warmup()                    # A: no drug, CKD-5, 500 h
ic_cur = copy(ic_ss)
ic_cur[C_ven, C_peri, Ae_drg, C_bladder] = 0   # no leftover furosemide
# toxins C_tox1_cen, C_tox2_cen, C_tox3_cen UNCHANGED
for day in 1..30:
    run oral doses (or none); carry toxins AND residual drug to next day
```

So at **t = 0 of every 30-day NPZ**:

- IS, HA, pCS must equal warm-up SS (same numbers for TID 120, QD 240, TID 240, no-drug).
- Furosemide must be 0.

If `pCS_all[0]` is ~18 and `pCS_base` is ~13.74, that file **did not** start from A.

## What %Δ means

\[
\%\Delta = \frac{\text{last 24 h mean} - \text{warm-up SS}}{\text{warm-up SS}}
\]

Same in blood and plasma.  
Your Windows comparison (all `IS_base=19.712`):

- TID 120: IS +15.8%, HA +0.6%, pCS +17.3%  
- QD 240: IS +24.6%, HA +1.3%, pCS +27.2%  
- TID 240: IS +28.1%, HA +1.2%, pCS +32.9%

Those % use **A as baseline**, not the first point of a wrong curve.

## Figure 5 panel B starting at different heights

That is **not** “with drug vs without drug at t=0”. With-drug and without-drug must **share t=0**.  
They separate **after** the first doses. A grey line that starts at 14 and rises to 16 is a **with-drug** arm (TID 120), not no-drug. No-drug stays flat at SS.

## Agent errors to discount

- Treating 15.8% as “stale” when it is the last-day AUC on these NPZs.  
- Using layout-preview % (19.9 / 28.8) as if they were your caches.  
- Mixing healthy `IS_base≈0.54` with CKD-5 `≈19.71`.  
- Calling ODE `C_ven` / `C_tox*_cen` plasma without `/BP`.  
- Mixing furosemide plasma \(C_{\max}=20.3\) with blood \(C_{\max}=7.1\).  
- Treating step B (IV PK extract) as the no-drug baseline.
