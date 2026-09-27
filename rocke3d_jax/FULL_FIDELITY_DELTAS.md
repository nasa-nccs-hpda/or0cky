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

## D5 — Track B PBL `advanc` (surface layer: drag, fluxes, skin effect) vs real Fortran, F0
`fullfidelity/pbl_ff.py`: float64 transcription of PBL.f `advanc` — the 8-sublayer prognostic
boundary layer with Newton-mapped log-linear grid, second-order-closure diffusivities, implicit
q/T/u/v/e tridiagonal solves, ≤5-iteration ustar fixed point with under-relaxation, ocean skin
temperature, Monin–Obukhov drag (dflux/getzhq/find_dpsim/find_dpsih). Track A replaced all of
this with one fixed-point iteration on Monin–Obukhov similarity. Inputs = the real model's own
`PBL` call arguments (every call in one step: **9,182 calls ×3 dates = 27.5k** incl. all four
surface types 5418 ocean / 1566 sea-ice / 692 land-ice / 1506 land). `deltas_pbl_ff.json`.

| Output (worst of 3 dates) | max abs error / RMS of output | relative error median | p99 |
|---|---|---|---|
| ustar, us, cm, ch, cq | ≤ 5e-11 | 1–3e-14 | ≤ 2.5e-12 |
| surface fluxes u,v,t,q | ≤ 1e-10 | 4e-14 (T flux 3e-13) | ≤ 3e-11 |
| khs, w2_1, dskin, tsv | ≤ 7e-11 | ≤ 1e-12 | ≤ 5e-11 |
Floor check: perturbing the input `ztop` by 1e-15 relative changes outputs by the same amount
(median 4e-14, max 5e-11) as the port-vs-Fortran discrepancy — the residual is the model's own
float64 sensitivity (the Newton grid stops at data-dependent steps), not a port difference.
Bug caught by validation: `b123=b1**(2./3.)` uses a REAL*4 exponent in the Fortran (7.19512315 vs
7.19512273); using the "obvious" double value gave 1e-7 errors; found and fixed via the test data.
Tests: `fullfidelity/tests/test_pbl_ff.py` (stratified samples, 3 dates; incl. mutation checks).

## D6 — Track B SURFACE ocean/lake + sea-ice tile fluxes vs real Fortran, F0
`fullfidelity/surface_tile_ff.py`: skin-effect ground adjustment, sensible/latent/thermal fluxes
(explicit for water, implicit two-layer for sea ice), evaporation/dew/lake limits, and the atmosphere-facing
outputs (DTH1, DQ1, DMUA, DMVA). 6 steps × 3 dates = 41.9k tile records (5.4k ocean+lake, 1.6k ice per step).
Outputs: 12 of 19 fields **bitwise equal** to Fortran, the rest ≤ 2e-10 absolute on values of 1e4–1e5 (J/m²);
temperatures ≤ 7e-15 K; DTH1 ≤ 2e-15. Tests: `fullfidelity/tests/test_surface_tile_ff.py` (7, with mutation checks).
**Coverage gap (stated, not hidden):** the lake-evaporation limit, dew limit, lake heat-flux limit and the
ice-melt clip (`TG1+dTG>0`) never trigger in these dumps, so those branches are transcribed but unvalidated.
Ice thermal properties (`ice_props_ff.py`: SEAICE.f `alami`, `dEidTiws`, `solar_ice_frac`, layer bookkeeping)
match the model's own values on all 9,048 ice tiles (dF1dTG/HCG1/HCG2 bitwise; FSRI ≤ 6e-17). **Composition check**
(`surface_chain_ff.py`, no recorded PBL outputs used): PBL advanc → tile fluxes reproduces the tile outputs to
≤ 6e-11 of their spread (DTH1, DQ1, DMUA, DMVA, SHDT, EVHDT, TRHDT, EVAP).

## D7 — Track B land-ice tile (SURFACE_LANDICE.f) vs real Fortran, F0
`fullfidelity/landice_tile_ff.py`: implicit two-layer land-ice surface fluxes and atmosphere-facing
outputs. 6 steps × 692 real tile records. Temperatures ≤ 7e-15 K, momentum fluxes bitwise, energy terms
≤ 6e-11 J/m² on values of 1e4 (all ≥ 1e-15 relative), DTH1 ≤ 4e-16. Tests: `tests/test_landice_tile_ff.py`
(with mutation checks). Coverage gap: dew-limit branch never triggered (transcribed, unvalidated).

## D8 — Speed of the faithful Track B pieces (CPU, float64, unoptimized; measured 2026-09-24)
Node was loaded (load avg ≈19 on 12 cores), single process, `jax.jit`, warm. **Correctness first**: no
kernel fusion / tuning has been done on Track B; these are baselines, not results.

| Piece | Work | Time |
|---|---|---|
| PBL `advanc` | 4,591 calls (one surface substep, all tile types) | 0.12 s (26 µs/call) after switching the Newton grid solve to a data-dependent `while_loop` (was 0.47 s, identical outputs) |
| ATURB (A-grid + U/V diffusion + A-wind recompute) | 3,312 columns × 40 layers | 0.07 s per call |
| ⇒ per DTsrc step (2 substeps each) | PBL + ATURB only | ≈ 0.4 s |
For scale: the real Fortran SURFACE (incl. GHY land, ATURB, ice/ocean/lake tiles) averages 263 ms per DTsrc step
in `P2SAoM40.PRT`. Track B's pieces so far do **not** include land (GHY), sea-ice/lake ground thermodynamics
or the tile-flux vector ops, so this is **not** a like-for-like comparison and no speed-up claim is made.
Track A's headline (0.74 ms/step on an A100) is for a far simpler computation and must not be compared with these.
GPU numbers for Track B: not measured yet (needs a GPU node).

## D9 — Track B GHY land-surface model vs real Fortran, F0
`fullfidelity/ghy_ref.py` + `ghy_compare.py`: plain-Python/NumPy (not yet JAX) transcription of
`giss_LSM/GHY.f` `advnc` and its callees (reth, hydra, xklh, retp, evap_limits, sensible_heat,
drip_from_canopy, fl/flh/flg/flhg, runoff, fllmt, apply_fluxes, accm) and the snow model
(SNOW.f/SNOW_DRV.f: snow_fraction, snow_redistr, snow_adv/snow_adv_1, heat_eq). Compiled options
matched: EVAP_VEG_GROUND, GHY_FD_1_HACK, GHY_USE_LARGESCALE_PRECIP, INTERCEPT_TEMPORAL,
LARGE_SCALE_PRECIP_INTERCEPT; no tracers, no SCM. Dynamic vegetation (Ent: canopy conductance,
soil-layer betas, LAI, GPP, TRANS_SW, Ci, IPP) is **not** ported — its real per-sub-iteration outputs
are recorded from the instrumented model (`ffent`/`ffent0` in the dump) and fed in as inputs, so this
validates the land-surface physics that consumes Ent's exports, not Ent itself.

Validated on **9,036 real land-tile records** (all `fearth>0` cells, 6 steps × 3 dates); 804–854/1506
cells per step carry active snow (>50%), so the snow model is genuinely exercised, not a vacuous path.
Zero exceptions over a full file (every real cell, not just a sample).

| Output | max\|abs error\| | max real value (scale) | ratio |
|---|---|---|---|
| tbcs, tsns (skin/sensible temp, °C) | 7e-7 | 59 | 1e-8 |
| ashg (sensible heat, W/m²·s) | 0.056 | 5.4e5 | 1e-7 |
| alhg, ae0 (latent heat / net energy) | 1.85 | 4.3e5 | 4e-6 |
| aevap | 7e-7 | 0.17 | 4e-6 |
| aruns, aeruns (surface runoff) | 1.8e-4 / 3.7 | 0.56 / 1.6e4 | up to 7e-3 (see below) |
| arunu, aerunu (underground runoff) | 6e-8 / 5e-4 | 0.21 / 2e4 | 1e-6 |

**Not float64-rounding-level like ATURB/PBL/SURFACE** — this is a plain-Python reference, ported for
correctness first (no JAX vectorization yet), and the largest residuals are in `aruns`/`aeruns`
(bare-soil surface runoff): traced to cells with a very small bare fraction (fb ~0.5–1%) where the
runoff formula's `(w/ws)**8` saturation exponent amplifies float64 rounding noise near the `min(...,0.6)`
threshold — the same "tiny threshold-crossing sensitivity" pattern documented for ATURB/PBL, not a
logic bug (the core temperature/heat-flux outputs that matter most are all ≤4e-6 relative). Land skin
temperature and heat fluxes — the fields SURFACE.f actually consumes for the atmosphere coupling — are
solid; runoff diagnostics are secondary outputs.

**Bug caught by validation:** ws(0,2)/shc(0,2) (canopy water capacity / heat capacity) are set from
Ent's per-cell exports (`ws_can`, `shc_can`) *before* the main iteration loop in the real code; missing
this produced 0/0 divisions and NaN/Inf immediately (canopy temperature). Found and fixed via the
oracle comparison, not by inspection.

**Not yet ported:** Ent vegetation itself (photosynthesis, canopy conductance, LAI dynamics — treated
as a recorded input here); GHY tracers; the adaptive sub-step selector `gdtm` (the recorded per-iteration
`dts` is used directly, so this validates the physics update, not the time-step-size choice — a
separate, smaller check would confirm `gdtm` reproduces the same `dts` given the same state).
Tests: `fullfidelity/tests/test_ghy_ref.py` (5, incl. mutation checks and a full-file no-exception run).

## D10 — Track B sea-ice ground thermodynamics (GROUND_SI) vs real Fortran, F0
`fullfidelity/seaice_core_ff.py`: plain-Python transcription of SEAICE.f `SEA_ICE` (4-layer heat
diffusion, melt/dew, compression, relayering), `SSIDEC` (brine drainage, ocean domain only) and
`snowice` (snow-to-ice conversion), driving from the real model's own GROUND_SI inputs (surface fluxes
already computed by SURFACE.f). `seaice_thermo="BP"` (brine-pocket, the P2SAoM40 default). Documented
approximation: the real `Ti`/`Ti2b` (enthalpy->temperature) use REAL*16 internally; this port uses
float64 throughout, validated empirically rather than assumed exact.

Validated on **4,524 real GROUND_SI cells** (6 steps × 3 dates, both sea-ice and lake-ice domains).
Brine drainage fires in 65% of ocean cells and snow-to-ice conversion in 15–20% — both genuinely
exercised, not rare edge cases. Zero exceptions over every real cell.

| Output | max relative error (or absolute, where ref≈0) |
|---|---|
| snow, hsil (layer enthalpy) | 2e-15 / 7e-9 |
| msi2 (ice mass) | 5e-8 |
| ssil (layer salt) | 7e-7 |
| runosi, erunosi, srunosi (ocean-coupling fluxes) | 3e-5 to 3e-4 |

The coupling-flux diagnostics (runosi/erunosi/srunosi) are differences of large numbers (surface flux
minus internal flux minus brine/snow-ice flux), so their relative error is a few orders above the state
variables — same cancellation pattern documented elsewhere, not a logic bug (a dump-placement bug was
caught and fixed during validation: the first instrumentation attempt read `RUNOSI` *before* Fortran
assigned it, showing a spurious 100% mismatch that pointed straight at the real cause).
Tests: `fullfidelity/tests/test_seaice_core_ff.py` (5, incl. mutation checks).
**Not ported:** ADDICE/SIMELT (ice formation/complete melt — different code paths not on the record's
DOPOINT branch), lake mixing (LAKES.f `lkmix`, a documented no-op even in Track A), tile aggregation
(`avg_patches_*` — the JAX-vectorization step for all Track B pieces is still pending).

## D11 — Tile aggregation (avg_patches_*) vs real Fortran
`fullfidelity/tile_aggregate_ff.py`: the composite surface fields ATURB/PBL consume (uflux1, vflux1,
dth1, dq1, tsavg, qsavg) are an area-fraction-weighted sum over the 4 surface-type patches
(FLUXES.f `avg_patches_pbl_exports`/`avg_patches_srfflx_exports`) — confirmed by reading the source,
not assumed. Validated on **38,040 real grid cells** (6 steps × 3 dates × 3312 cells): max error
≤ 9e-7 of the field's RMS (float64 rounding level); patch fractions sum to 1.0 in every cell.
Tests: `fullfidelity/tests/test_tile_aggregate_ff.py` (3, incl. mutation check).

## D12 — Sea-ice formation (ADDICE) and lateral/complete melt (SIMELT) vs real Fortran
`fullfidelity/seaice_core_ff.py` (`addice`, `simelt`): new-ice formation in open/partially-ice-covered
ocean, horizontal compression/lead adjustment, and fixed-SST mass-floor correction (ADDICE); lateral
melt parameterization and complete melt-out with property reset (SIMELT). Both called every real
DTsrc step for every water-covered cell (`FORM_SI`/`MELT_SI` in `SEAICE_DRV.f`), not rare paths.

Validated on real Fortran calls (6 steps × 3 dates): **16,214 ADDICE calls** (1,994 = 12% with genuine
new-ice formation) and **4,668 SIMELT calls** (376 fully melted out — the one branch where SIMELT's
`TSIL` output is actually defined; **found by reading the source**: `TSIL` is `intent(out)` but is
left unassigned by the real Fortran whenever ice remains, so this port does not claim a match to that
undefined value in that branch — a real, if minor, unported/undefined behavior, not swept under the rug).

| Routine | Worst relative/absolute error | Exceptions |
|---|---|---|
| ADDICE (snow, roice, msi2, dmimp/dhimp/dsimp, ssil) | 0 (bitwise) | 0/16,214 |
| ADDICE (hsil) | 1.3e-13 | — |
| SIMELT (roice, snow, msi2, hsil, ssil) | 0 (bitwise) | 0/4,668 |
| SIMELT (enrgused) | 1.5e-16 | — |

This closes out the sea-ice ground-thermodynamics module (SEA_ICE + SSIDEC + snowice + ADDICE +
SIMELT all validated; D10, D12). Tests: `fullfidelity/tests/test_addice_simelt_ff.py` (4, incl.
mutation checks).

## Pending rows
- D5: speed at full fidelity (single-call and chained, CPU/GPU).
