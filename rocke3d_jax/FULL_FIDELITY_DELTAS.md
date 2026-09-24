# Full-fidelity port: delta ledger

One row per measured difference between **Track A** (representative driver,
tag/commit before this branch) and the **real Fortran** oracle, produced by
scripts in `fullfidelity/` so each row is reproducible. Track B (full-fidelity
modules) rows are added as phases land. Speed rows always carry the fidelity
level they were measured at.

Conventions: "Fortran change" = how much the real routine changes the field in
one step (the signal); "Track A error" = RMS difference to the real routine's
output given *identical inputs*. Error ≫ Fortran change means Track A's answer
is dominated by its own physics, not the real one.

## D1 — What the real P2SAoM40 runs (2026-09-24, `nm` on the binary)
| Item | Track A | Real Fortran |
|---|---|---|
| DRYCNV | ported, "faithful", 9.5× kernel | **not in the executable** |
| Free-atmosphere turbulence | layer-1 shortcut + DRYCNV | ATURB (`atm_diffus`), called inside SURFACE (SURFACE.f:1172) |
| Reproducibility of the real model here | n/a | 5-day re-run byte-identical to original restart |

## D2 — Track A vs real SURFACE(+ATURB), one step, identical inputs
Script: `fullfidelity/track_a_vs_fortran_surface.py` on dumps for itime
33312–33317 (1950-11-26 00:00–02:30), radiation disabled in Track A.
Raw: `fullfidelity/deltas_trackA_surface.json`.

| Field | Fortran change (RMS/step) | Track A change | **Track A error** (RMS vs Fortran) |
|---|---|---|---|
| T, layer 1 | 0.027 K | 0.356 K | **0.358 K (≈13× the real signal)** |
| T, layers 2–40 | 0.0053 K | 0.018 K | 0.019 K (≈3.5×) |
| Q, layer 1 | 8e-5 | 4.4e-4 | 4.4e-4 (≈5×) |
| U, layer 1 | 0.61 m/s | 1.22 | 0.91 m/s (≈1.5×) |

Values are stable across the 6 steps (T1 error 0.352–0.358 K).
**Reading:** Track A's one-step answer at layer 1 is dominated by its own
physics. This is a delta, not a verdict on the whole approach, and has known
causes to confirm before being quoted: Track A derives skin temperature from
layer-1 air temperature (real model carries prognostic ground/ocean/ice
temperatures), uses a simplified fixed-point Monin–Obukhov solve, no
sub-tiling, no real ATURB. Caveats: one restart date (November), six
consecutive steps, radiation off, restart-derived ground fields taken at
1950-11-26 (they evolve slowly). Controls run (itime 33312, layer-1 T RMS error vs real post-SURFACE state):

| Variant | T1 error |
|---|---|
| Track A as-is | 0.358 K |
| Track A with ocean skin temperature = real restart SST (`asst`) | 0.360 K (no help; U1 error worse, 0.91→1.23 m/s) |
| **Identity: apply no physics at all** | **0.027 K** |

So the skin-temperature simplification is *not* the cause, and Track A's
surface step is ~13× farther from real Fortran than doing nothing. The
remaining suspects are the flux formulas/units, the fixed-point solve and the
flux→tendency coupling; locating the dominant term is the first Phase-1
diagnostic (per-term comparison against dumped Fortran fluxes, which need a
further dump hook in SURFACE).

## D3 — Chaos noise floor of the real model (5 days, 240 steps)
Real Fortran twice from 1950-11-26, second time with T perturbed by ±1 ulp
(float64, ~6e-14) everywhere. After 5 days (restart at 1950-12-01):

| Field (layer 1 unless noted) | Pointwise RMS diff | Global-mean diff | Zonal-mean RMS diff | Field std |
|---|---|---|---|---|
| T | 0.048 K | 1.4e-4 K | 0.007 K | 2.74 K |
| U | 0.36 m/s | 0.011 m/s | 0.057 m/s | 6.5 m/s |
| Q | 1.8e-4 | 8.8e-6 | 3.4e-5 | 5.9e-3 |
| T layer 21 | 0.058 K | 3.9e-3 K | 0.012 K | 1.55 K |
Whole column (all layers): T rms 0.27 K (max 5.5 K), U/V rms 0.74 m/s (max 12.8),
p rms 0.33 hPa (max 2.2 hPa). Restart: only 36/257 variables stay bitwise identical.

**Use:** a rounding-level difference grows to ~1–3% of field variability
pointwise in 5 days, while global and zonal means stay within ~1e-3–1e-2 of
that. F2 acceptance must therefore be on statistics (global/zonal means,
spectra, RMSE relative to this floor), not pointwise equality; any port whose
5-day pointwise error is ≲ this floor is indistinguishable from Fortran
rounding. Each single-step (F1) comparison is deterministic (floor = 0), see D2.
Script/data: perturbed run recipe in `fullfidelity/PHASE0_LOG.md`; compare with
`compare_restarts.py`.

## D4 — Track B ATURB (A-grid: T, Q, TKE, PBL height) vs real Fortran, F0
`fullfidelity/aturb_ff.py` (float64 transcription of ATURB.f/PBL.f/TRIDIAG.f, not the old
placeholder). Inputs = the real model's own ATURB entry state and fluxes (dumps);
output compared with the real exit state. 4 calls (2 steps × NIsurf=2), all 3312 columns
(polar duplicates masked), on three dates (1950-11-26, 12-01, 01-01: different seasons). Tests: `fullfidelity/tests/test_aturb_ff.py`.

| Output | max abs error | Fortran signal (RMS change) | Bitwise-equal cells |
|---|---|---|---|
| T (K) | 7e-13 | 3.7e-3 K | 19–22 % |
| Q | 9e-18 | 2.3e-5 | 98 % |
| TKE e | 6e-15 | 7.9e-2 | 91 % |
| PBL height (m) | 9e-13 | — | 83–85 % |
| PBL top level (dclev) | 0 | — | 100 % |
Error is ~10 orders of magnitude below the signal and at float64 rounding level (~1e-15 relative).
Mutation checks (tests fail if wrong): deltx 0.608 → T err 6e-5 K, e err 3e-3; g=9.81 →
pblht err 1.3 m; b1=19.0 → e err 0.32. Velocity grid (`aturb_uv_ff.py`: regrid to B-grid, U/V diffusion, A-grid wind recompute incl. polar rotation):
U,V,UA,VA max error 1.4e-14 m/s vs Fortran change 0.06 m/s RMS (96 % of cells bitwise equal). 15 tests pass
(incl. 3 mutation checks). **Not yet exercised:** `kmmin` clamp never binds in these steps.
Byproduct: Track A used deltx=0.608, g=9.81, R=287 (real planet config: 0.60785…, 9.80665, 287.0487).

## Pending rows
- D5: speed at full fidelity (single-call and chained, CPU/GPU).
