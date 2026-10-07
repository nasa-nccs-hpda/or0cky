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
**Not ported at this point:** ADDICE/SIMELT, tile aggregation, and lake mixing — since closed out by
D11–D13 respectively. The JAX-vectorization step for all Track B pieces (still plain Python/NumPy) is
still pending.

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

## D13 — Lake mixing (LKSOURC/LKMIX) vs real Fortran
`fullfidelity/lakes_ff.py` (`lksourc_full`, `lkmix`): frazil-ice formation/mass exchange between the
two lake layers (`LKSOURC`) and static-stability mixing, implicit vertical heat diffusion, and
TKE-driven entrainment between them (`LKMIX`). Track A's `lakes_jax.py` `lkmix` is a documented
no-op placeholder; this ports the real two-layer physics from `LAKES.f`'s `GROUND_LK`, called every
real DTsrc step for every lake cell (`FLAKE>0`).

Validated on **3,644 real GROUND_LK calls** (6 steps × 3 dates; count varies by date since not every
cell has `FLAKE>0`), of which **648 had genuine frazil-ice formation** (`acefo` or `acefi` ≠ 0) — not
a rare edge case. Chained LKSOURC→LKMIX result:

| Output | max relative/absolute error | Exceptions |
|---|---|---|
| enrgfo, acefo, acefi, enrgfi | 0 (bitwise) | 0/3,644 |
| mlake, elake (post-LKSOURC) | 0 (bitwise) | — |
| mlake, elake (post-LKMIX) | 0 (bitwise) | — |

Bitwise-exact match on the first attempt across every real call. **Coverage gap, found by reading the
source, not assumed:** the real `GROUND_LK` always calls `LKMIX` with `TKE=0.` (the `U2rho`
entrainment term is commented out in this rundeck), so `LKMIX`'s TKE-driven entrainment branch is
transcribed here but never exercised by real Fortran calls — ported faithfully, unvalidated in
practice, same honesty standard as the SIMELT `TSIL`-undefined case in D12.
Tests: `fullfidelity/tests/test_lakes_ff.py` (4, incl. mutation checks).

## D14 — JAX-vectorized SEA_ICE/SSIDEC/snowice/SIMELT/ADDICE (`seaice_core_jax.py`) vs real Fortran
`fullfidelity/seaice_core_jax.py`: batched-array (jnp.where in place of Python if/else) port of
*all* of `seaice_core_ff.py` -- `sea_ice`, `ssidec`, `snowice`, `simelt`, `addice`, and their shared
`relayer`/`relayer_12`/`get_snow_ice_layer`/`set_snow_ice_layer`/`tice` helpers. Runs all real cells
from a file in one call instead of a Python loop (JIT-compilable: 4,524 GROUND_SI cells run in
4.6 ms cached on CPU after a 2.9 s one-time trace/compile, vs. the reference's per-cell Python loop;
16,214 ADDICE cells run in 13 ms cached after a 3.6 s compile).

`relayer_12` (needed by `sea_ice`/`snowice`/`addice`) has 9 mutually-exclusive leaf branches, each
hand-derived as a closed form in the pre-branch inputs and selected with nested `jnp.where` matching
the original's if/elif/else precedence. `addice` looks comparably long but is structurally easier:
its Python original is a *sequence* of if-blocks (new-ice formation, then an unconditionally-checked
downward lead-fraction rebalance, then an unconditionally-checked upward rebalance) rather than one
wide decision tree, so it was built as a chain of `state = where(cond, f(state), state)` merges that
reuse `relayer_12` exactly where the Fortran does -- avoiding the combinatorial branch-count risk
initially flagged for it in the first version of this module (see PHASE0_LOG.md).

**Coverage gap, found by inspection, not assumed:** 2 of ADDICE's 5 leaf paths -- new ice forming in
fully open ocean (`roice<=0 & acefo>0`) and the `qfixr` msi2-floor correction -- occur 0 times in the
available 3-date real record. Both are cross-checked against `seaice_core_ff` on synthetic inputs
instead (bitwise/near-bitwise match, see `test_addice_synthetic_branches_match_plain_python`), which
is validation against the plain-Python port, not against Fortran, and is documented as such.

Validated on the **same 4,524 real GROUND_SI cells, 4,668 real SIMELT cells, and 16,214 real ADDICE
cells** as D10/D12, three independent ways: directly against the real Fortran dumps, against
`seaice_core_ff` row-for-row (worst case 7e-15 relative for GROUND_SI, 1e-6 for ADDICE, i.e.
float64-rounding-level agreement with the already-validated plain Python), and for exact eager/
`jax.jit` equivalence. No NaN/Inf anywhere in any of the 25,406 real cells.

| Output | max rel err vs Fortran (batched) | vs D10/D12 (plain Python) |
|---|---|---|
| GROUND_SI snow | 2.2e-15 | 2e-15 |
| GROUND_SI hsil | 7.0e-8 | 7e-9 |
| GROUND_SI msi2 | 5.0e-8 | 5e-8 |
| GROUND_SI ssil | 6.8e-7 | 7e-7 |
| GROUND_SI runosi/erunosi/srunosi | 3.4e-5 / 6.5e-5 / 3.1e-4 | 3e-5 to 3e-4 |
| SIMELT (roice/snow/msi2/hsil/ssil) | 0 (bitwise) | 0 (bitwise) |
| SIMELT enrgused | 4.0e-16 | 1.5e-16 |
| ADDICE (roice/dmimp/dhimp/dsimp) | 0 (bitwise) | 0 (bitwise) |
| ADDICE (snow/msi2) | ~2-3e-16 | ~2-3e-16 |
| ADDICE hsil | 1.8e-13 | 1.3e-13 |
| ADDICE ssil | 1.7e-14 | — |

Essentially identical accuracy to the plain-Python reference at every field -- vectorizing did not
loosen any tolerance. `tsil` in SIMELT's `roice>0` branch is `NaN` (a sentinel, not a computed
value), matching the plain-Python `tsil=None` for the same documented-undefined-Fortran-output case.
Of ADDICE's 16,214 real calls, 1,994 (12%) have genuine new-ice formation (`acefo`/`acefi`≠0),
matching D12's count; 1,972 of those exercise `relayer_12` inside `addice` (branch 2b-ii). 215 real
cells exercise the downward lead-fraction rebalance (branch 3); the upward rebalance (branch 4) and
the two zero-coverage leaves are discussed above. Tests:
`fullfidelity/tests/test_seaice_jax_vectorized.py` (11, incl. mutation checks, jit-equivalence, a
cross-check against `seaice_core_ff`, and the ADDICE synthetic-branch checks).

**Speed** (CPU only, no GPU on this node; single `jax.jit`-compiled call vs. the plain-Python
per-cell loop it replaces): at the real 4,524-cell GROUND_SI record, 0.458s (Python) vs. 0.0100s
(JAX, cached) = **46x**; scaled to 90,480 cells (20x replication of the same real record, to check
this isn't a small-batch artifact) the per-cell JAX cost is unchanged (~2.0μs/cell at both scales)
and the measured speedup is **51x** — consistent, not a fixed-overhead illusion. One-time
trace/compile cost is ~2.9s for GROUND_SI and ~3.6s for ADDICE, amortized over every subsequent call.

## D15 — JAX-vectorized GHY land model (`ghy_jax.py`) vs real Fortran
`fullfidelity/ghy_jax.py`: full batched-array port of `ghy_ref.py`'s `GhyColumn.advnc` and every
method/helper it calls -- `reth`, `retp`, `hydra` (soil-hydraulics bisection), `xklh`, `evap_limits`,
`sensible_heat`, `drip_from_canopy`, `fl`/`flh`/`flg`/`flhg`, `runoff`, `fllmt`, `apply_fluxes`,
`accm`/`accm_zero`/`accm_final`, and the full `SNOW.f` model (`pass_water`, `snow_fraction`,
`snow_redistr`, `tridiag`, `heat_eq`, `snow_adv_1`, `snow_drv`, `snow`). This was scoped as the
largest remaining item in Track B (FULL_FIDELITY_PLAN.md's GHY section) precisely because it compounds
several axes of variability at once (per-cell active-soil-layer count, per-substep snow-layer count,
per-cell-per-timestep adaptive substep count); all three are handled the same way every other
fixed-size-plus-mask piece in this project is: fixed max array sizes (NGM=6 soil layers, TOTAL_NL=3
snow layers, 11 substeps -- covers 100% of the real 1-10 range measured for `ffnit`) with per-lane
boolean masks, never dynamic shapes.

**Individual functions have no intermediate real-Fortran ground truth to check against** (the
`ffg_*.bin` dump only has the state *after* the whole substep loop, not per-substep) -- they are
cross-checked against `ghy_ref.py` instead (already validated against Fortran, D9), on real cells'
recorded input state (`ghy_jax_compare.py`, `ghy_flux_chain_test.py`, `ghy_snow_test.py`); only the
assembled `advnc()` pipeline is checked against real Fortran (`ghy_advnc_test.py`), on all 9,036 real
land-cell substeps across all 6 real dump files.

| Output | max rel err vs Fortran | Note |
|---|---|---|
| tbcs, tsns | 3e-8 to 3e-5 | matches D9's plain-Python tolerance |
| ashg, arunu, aerunu | 4e-7 to 9e-5 | |
| alhg, aevap, ae0 | 1e-5 to 2e-2 | worst case is one date (nov26); still float64/branch-noise scale |
| w_out, ht_out, tp_out | 9e-5 to 8e-2 | worst case: 2/1506 cells, both a ~5e-6 (absolute) canopy-water
  difference inflated by a near-zero denominator, not a real state error |
| abetad | 3e-16 | bitwise-exact |
| aruns, aeruns | not a like-for-like relative-error field | see below |

`aruns`/`aeruns` (surface runoff accumulators) carry the SAME threshold-crossing sensitivity already
documented for the plain-Python reference's own validation against Fortran (D9: "residuals traced to
a threshold-crossing sensitivity in the bare-soil runoff formula, same pattern as ATURB/PBL branch
flips -- not a logic bug") -- a small fraction of cells flip whether runoff activates at all for a
given substep under tiny floating-point perturbation. Checked by magnitude instead: max absolute
value and count of runoff-active cells both agree to within 5%/20% across the full real record, ruling
out a systematic bug while accepting the same known chaos-sensitivity the plain-Python port already
has.

**Two real, confirmed bugs found and fixed** while assembling `advnc()` (beyond several found earlier
while building the pieces individually, see PHASE0_LOG.md): `evap_limits` needs the substep loop's
TOTAL `dt` (set once per `advnc()` call), not the per-substep `dts` -- invisible when only one substep
runs (they're equal), a clear ~5-60% error once a second substep is added; and `GhyColumn.snow()`'s
`for ibv in range(i_bare, i_vege+1)` skips `snow_drv` entirely for an inactive ibv, leaving its
fr_snow/nsn/dzsn/wsn/hsn/evap/snsh state untouched -- an unconditional call was silently overwriting
that state instead.

**Not vectorized:** none of GHY -- `ghy_jax.py` covers all of `ghy_ref.py`'s `advnc()`. GHY's own
scoping note also flagged the `heat_eq` used inside the snow model itself, which is now vectorized.

**Speed, and a real architecture bug found by measuring it (not assumed):** the first working version
of `advnc()` unrolled its per-cell substep loop with a Python `for i in range(11): ...` (11 = the
padded max substep count) -- correct (validated above), but measuring its speed, rather than just
declaring victory once it matched Fortran, surfaced a real problem: unrolling duplicates the ENTIRE
per-substep computation graph (hydra/xklh/evap_limits/.../snow's own nested `heat_eq` calls) 11 times,
which measured as **slower than plain Python in eager mode** (8.7s vs 0.63s for 300 cells -- the JAX
"vectorization" was a *regression*) and **impractically slow to `jax.jit`-compile** (XLA's own
slow-compile warning fired; still not finished after 300s). Root-caused to Python-level loop
unrolling, not vectorization itself, and fixed by rewriting the substep loop with `jax.lax.scan`
(compiles the loop body once, applies it via an XLA-level loop, rather than duplicating the graph) --
same per-lane masking design, same every-function call, only the control-flow primitive changed.
Re-validated bit-for-bit identical to the unrolled version's already-real-Fortran-checked output on
all 9,036 real cells across all 6 files (every number above generated by the `lax.scan` version).

| | plain-Python (`ghy_ref.py`) | JAX unrolled (broken) | JAX `lax.scan` |
|---|---|---|---|
| 1,506 real cells | 3.35s (2.22 ms/cell) | eager: slower than Python; jit: didn't finish compiling in 300s | jit compile: ~30s (one-time); cached: 0.17s (112μs/cell) |
| **speedup vs Python** | — | none (regression) | **~20x** |

This is the clearest instance in this project of "unrolling a *big* loop many times" being
qualitatively different from the small (2-6 iteration) bounded unrolls used everywhere else here
(hydra's 6-step bisection, tridiag's 3-element solve, relayer_12's branch selection, ...) -- those
stay cheap because the per-iteration body is small; GHY's substep body is dozens of functions deep,
so unrolling it 11x was the actual problem, not the masking design around it. Worth remembering for
any future large per-timestep loop in this codebase.

## D16 — JAX-vectorized lake mixing (`lakes_core_jax.py`) vs real Fortran
`fullfidelity/lakes_core_jax.py`: batched-array port of `lakes_ff.py`'s `LKSOURC`/`LKMIX` (D13). No
per-cell loop or dynamic layer count here -- lakes_ff.py is a small, fixed two-layer (upper/lower)
model, so this is a direct branch-by-branch `jnp.where` transcription, not a new technique. Matches
the plain-Python reference **bitwise (0.0 relative error)** on all 3,644 real lake cells across all 6
dump files (648 with genuine frazil-ice freezing), same as D13's own match to real Fortran. `lkmix`'s
`tke>0` branch (dead code in this rundeck -- `GROUND_LK` always calls `LKMIX` with `TKE=0.`, per D13)
is cross-checked against `lakes_ff.py` on 5,000 synthetic random inputs instead: bitwise match there
too. No NaN/Inf anywhere.

**Speed**: `jax.jit`-compiled, no `lax.scan` needed (lakes has no per-cell substep loop unlike GHY).
0.10μs/cell cached vs. the plain-Python reference's 9.95μs/cell -- **~104x speedup** on all 3,644
real cells. Tests: `fullfidelity/tests/test_lakes_jax.py` (5, incl. the tke>0 synthetic check,
jit-equivalence, and a mutation check).

This closes the last "still plain Python" gap flagged in STATUS.md's 8-stage comparison table --
every currently-validated piece of Track B physics (ATURB, PBL, SURFACE, SEAICE/ADDICE/SIMELT, LAKES,
tile aggregation, GHY) is now `jax.jit`-compilable with a measured real-data speedup.

## D17 — `ground_si` wrapper + a float32-precision bug across 3 core modules (also explains D15's pytest mystery)
`seaice_core_jax.ground_si(is_ocean, dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc,
fhoc, fsoc, wetsnow, tm, sm, ...)`: the one clearly-missing piece flagged in FULL_FIDELITY_PLAN.md's
chained-driver scoping -- a general-purpose, named-argument function chaining `sea_ice`→`ssidec`→
`snowice` with the ocean/other domain split done via `jnp.where(is_ocean, ...)` instead of a Python
branch, matching `seaice_core_ff.py`'s plain-Python `ground_si_ocean`/`ground_si_other` API. Adapted
from `seaice_jax_compare.py`'s `batched_ground_si` test helper (which unpacks a raw `(N,60)` dump
record array -- not reusable outside a test), now with a clean signature any future driver can call
directly. Validated against `batched_ground_si` (already checked against real Fortran, D10) at <1e-9
relative error on all 4,524 real ocean+lake cells (`test_ground_si_general_wrapper_matches_test_helper`
in `tests/test_seaice_jax_vectorized.py`).

**Real bug found while building the standalone validation script for this**: `seaice_core_jax.py`,
`ghy_jax.py`, and `lakes_core_jax.py` never enabled `jax_enable_x64` themselves -- they relied on
whatever script imported them having already called `jax.config.update("jax_enable_x64", True)`
(true of every existing test/compare script, by luck, which is why D9/D10/D12-D16 never caught this).
A validation script that imports `seaice_core_jax` on its own, with no such caller, silently ran in
JAX's float32 default and got real cell 1944 badly wrong: `ssidec` divides by `tsil`, which is ~0 at
the ice melt point; float32's less-precise near-zero `tsil` had a different sign/magnitude than the
float64 reference's, and that difference propagated through a brine-fraction division into `erunosi`
being off by ~5,415 units (`hflux` sign-flipped entirely, not just noisy) -- silently wrong, not a
crash. **Fixed** by moving `jax.config.update("jax_enable_x64", True)` inside each of the three core
modules themselves (immediately after `import jax`, before `import jax.numpy`), so correctness no
longer depends on import order. Re-validated at <1e-9 rel. error post-fix on all 4,524 cells (was up
to 5.4e3 abs. error pre-fix on the one affected cell); reran the full existing lakes (5/5) and
sea-ice (12/12, incl. the new test) suites -- all still pass, confirming the fix is a no-op for every
code path that was already being exercised correctly by luck of import order.

**Takeaway for the eventual chained driver**: since the driver's own top-level script is what
determines import order today, this bug would have resurfaced the moment the chained driver's entry
point imported these modules in a different order than the existing test scripts do -- fixing it at
the module level now removes that landmine before the chained driver is built, rather than after it
silently produces wrong numbers.

**Unexpected bonus: this also resolves the D15/PHASE0_LOG pytest-timing mystery.** D15 documented
`tests/test_ghy_jax.py` taking 3h42m-4hr under pytest twice, vs. under a minute for the identical
logic via a direct script, and left it as an open, unchased oddity after ruling out a real code
regression. Rerunning that exact same, unmodified test file after this fix (only change: `ghy_jax.py`
now sets `jax_enable_x64` itself instead of relying on the test file) took **6m34s** -- a ~36x drop,
with everything else identical (same 21 tests, same machine, same day). This is a clean natural
before/after comparison and strongly suggests the pytest-specific slowdown *was* this bug:
`jax_enable_x64` must be set before any array/JIT activity to behave correctly, and pytest's test
collection (which imports every file under `tests/`, not just the one being run) plausibly triggered
some JAX activity via a sibling test module before `test_ghy_jax.py`'s own late `config.update` call
took effect -- which would also explain why an isolated direct script (setting x64 first, importing
nothing else) never reproduced it. The exact JAX-internal mechanism wasn't chased further; the fix,
the isolated repro, and the 36x post-fix speedup are consistent enough that the practical outcome
(correct precision, fast tests) is what matters here.

## D18 — Tile outputs -> aggregation -> ATURB, as a real composition (`chain_aggregate_aturb.py`)
The SURFACE.f per-substep chain had one link no earlier row checked *as a composition*: D4 fed ATURB the
model's own recorded entry fluxes, D6 stopped at the ocean/ice tile outputs, and D11 aggregated recorded
per-tile fields. `fullfidelity/chain_aggregate_aturb.py` closes it. Ocean and sea-ice patch fields
(uflux1, vflux1, dth1, dq1, tsavg, qsavg) come from **our** PBL `advanc` + tile-flux chain (D5/D6, no recorded
PBL or tile outputs), are aggregated by **our** `tile_aggregate_ff.aggregate` together with the **recorded**
land-ice and land patches (land is dump-fed for the same reason as D9: Ent is not ported), converted to the
ATURB arrays as SURFACE.f:1091-1092 does (`tflux1=-dth1*MA(1)/dtsurf`, `qflux1=-dq1*MA(1)/dtsurf`, U/V/ts/qs
pass-through), and run through **our** `aturb_ff`/`aturb_uv_ff`. Result vs the real ATURB exit state:
3 dates x 2 NIsurf substeps, ~3,340-3,490 chained ocean+ice tiles per substep, all 3,170 cells.

| Field | max abs error (worst of 6 runs) | Fortran change (RMS) | recorded-flux baseline (D4-style) |
|---|---|---|---|
| T (K) | 7.4e-13 | 3.4e-3 to 4.4e-3 | 6.3e-13 |
| Q | 2.2e-14 | ~2e-5 | 7e-18 |
| TKE e | 2.2e-11 | ~9e-2 | 5e-15 |
| PBL height (m) | 2.2e-9 | — | 9e-13 |
| U, V (m/s) | 1.0e-11 | 6e-2 to 9e-2 | 1.4e-14 |
| UA, VA (m/s) | 6.1e-12 | ~5e-2 | 7e-15 |

Errors are 8+ orders below the signal; the chained-vs-baseline gap comes only from the ~1e-13 relative
differences in our tile fluxes (D6), amplified slightly by ATURB's PBL-top search (PBL height, TKE).
Verified exactly (0.0 error on every cell): the conversion identities against the recorded ATURB entry
arrays, and aggregation reproducing the recorded composite. Mutation checks: 0.1% error in `tflux1` moves T
by >1e3x the pass error; dropping the land patch from the aggregation is clearly visible.
Tests: `tests/test_chain_aggregate_aturb.py`. Small change to a validated file: `surface_chain_ff.run_chain`
gained an optional `return_pbl` flag (default unchanged) so callers can also get PBL's `tsv`/`qsrf`.
**Land-ice added the same day:** `chained_landice_patch` also computes the land-ice patch from our PBL (itype-3
records) + `landice_tile_ff` (D7), leaving **only land recorded**. 346-ish land-ice tiles per substep; errors
unchanged at roundoff (nov26 step 33312 ns=1: T 6.3e-13, U 2.5e-12, PBL height 4.7e-10), 6/6 test cases pass.
**Scope, stated plainly:** the land patch (GHY, needs Ent) is the only recorded piece; everything else in the
composite is our own code from real inputs.

## D19 — Two NIsurf substeps chained from real step-start state (`chain_two_substeps.py`, `substep_chain.py`)
First genuine **step-to-step** chain: substep 1 runs on the real step-start inputs; substep 2 runs on inputs built only
from substep 1's results computed by our own code, and the resulting ATURB exit state is compared with the real one.
New per-substep links, each transcribed from the Fortran and checked against the recorded substep-2 inputs:
`get_atm_layer1`-derived scalars (`utop, vtop, qtop, tkv=T1*PEK1, zs1, ztop`: **bit-exact**, 0.0 error), `get_dbl`
(`PBL_DRV.f`: `dbl, ug, vg` from our ATURB `pblht/dclev` and the composite `ustar/lmonin`; a bounded scan replaces the
Fortran early-exit loop; THBAR from `shared/Utilities.F90`), PBL profile carry-over (bitwise), `cm/ch/cq` carry-over,
ice/land-ice ground state carry-over from our tile outputs (`tg1/tg2/tr4` -> PBL `tg/tgv/qg_sat`), `e0/evapor`
accumulation. `loadbl` is a no-op for surface types that persist (it only initialises newly appearing types).

Result on 3 dates (all 3,170 cells; ~3,300-3,500 chained ocean+ice and ~350 land-ice tiles per substep), our final
substep-2 ATURB exit vs real: T <= 9.7e-13 K (signal 3.4e-3..3.6e-3), Q <= 1.1e-14 (~2e-5), TKE <= 7.6e-12 (~9e-2),
PBL height <= 1.5e-9 m, U/V <= 3.3e-12 m/s (~6e-2..7e-2), UA/VA <= 2.2e-12. Predicted substep-2 PBL input columns vs
recorded: <= 3e-12 (temperatures, `zs1`, `tkv`, `ug/vg`, `cm/ch/cq`), `dbl` <= 3.3e-9 m, profiles <= 1.8e-11.
Tests: `tests/test_chain_two_substeps.py` (7, incl. a mutation check that a 5% error in `dbl` is detected).
**Bug found on the way (mine, caught by the per-column diff):** the first run had land-ice `dbl/ug/vg` 66 m off; cause was
building the Coriolis parameter only from ocean/ice records so pure land-ice cells got 0 (`tmp=max(|f|,omega)` then
inflated `dbls`). The recorded land-ice `dbl` matched the formula exactly once `coriol` was taken from all records.
**Also found:** `SURFACE.f`'s `DO NS` loop ends at line 1178 -- `GROUND_SI`/`GROUND_LK` run once per DTsrc step after it (the
earlier plan text had them inside; corrected).
**Not chained, stated plainly:** the land patch (GHY, needs Ent) at both substeps and its PBL outputs in the composite
`ustar/lmonin`; the ocean tile's own state (unchanged inside the loop in the real model); GROUND_SI/GROUND_LK/FORM_SI after
the loop; and nothing beyond two substeps of one step (no step-to-step or radiation/dynamics coupling).

## D20 — Once-per-step GROUND_SI and GROUND_LK chained after the two substeps
After `DO NS` (SURFACE.f:1178) the model runs `GROUND_SI` (:1230) and `GROUND_LK` (:1232) once per DTsrc step on fluxes
accumulated over the substeps. Verified identities (0.0 error against the recorded inputs): GROUND_SI's `f0dt, f1dt, evap`
= sum over the two substeps of the ice tile's outputs, `srox0` = sum of `srheat*dtsurf`. Lake inputs likewise: `fodt`
(=E0), `evapo` (=EVAPOR), `srox(1)` from the open-water (itype 1) tile accumulators, `run0/fidt/srox(2)` from GROUND_SI's
`runosi/erunosi/solar_io` (a fully ice-covered lake cell has no open-water tile, so its accumulators are 0).
`chain_two_substeps.ground_si_stage` / `ground_lk_stage` replace those recorded inputs by OUR two-substep results
(everything else -- ocean fluxes `fmoc/fhoc/fsoc`, mixed-layer `tm/sm`, sea-ice and lake state, `roice`, `fsr2`, `hlake` --
stays recorded step-start state). Results (nov26; dec01/jan01 pass the same test bounds): accumulated inputs differ from
recorded by <= 5e-7 (f0dt, on values ~2e5, i.e. ~2e-12 relative); GROUND_SI outputs are **exactly as accurate as with
recorded inputs** (identical residuals to D10: hsil 9.8e-10, ssil 1.9e-7, msi2 1.4e-8 relative; coupling fluxes
runosi/erunosi/srunosi 8e-6/4e-5/7e-5 on a 1e-6 floor, the known loosest diagnostics); GROUND_LK outputs <= 2.5e-11
relative (enrgfo/acefo 2.4e-11, mlake/elake <= 2.5e-13) vs a bitwise-exact baseline with recorded inputs.
Tests (in `tests/test_chain_two_substeps.py`, 14 total): both stages on 3 dates, plus a mutation check that omitting
substep 2's flux from the accumulation is detected.
**Not chained:** ocean fluxes into GROUND_SI (`fmoc/fhoc/fsoc`, from the ocean model/precip), land runoff into lakes (GHY,
recorded), UNDERICE/RIVERF/FORM_SI (ADDICE, D14, exists separately) and anything between steps.

## D21 — Lake-side FORM_SI/ADDICE chained after GROUND_LK
Identities (0.0 error, all 636 lake cells on nov26): ADDICE's lake inputs `enrgfo/acefo/acefi/enrgfi` are exactly LKSOURC's
outputs, `roice` is the lake stage's `roice`, and the ice state `snow/hsil/ssil/msi2` is exactly GROUND_SI's final state
(cells with no ice record keep their unchanged step-start state). `chain_two_substeps.form_si_lake_stage` feeds ADDICE (D12/D14)
with OUR chained GROUND_SI state and OUR chained LKSOURC fluxes. Result (nov26; 3 dates pass `<1e-9`): input differences
<= 3e-7 absolute; outputs vs the real post-ADDICE state `snow` 8.6e-14, `roice` 2.5e-15, `hsil` 1.5e-14, `msi2` 4.7e-16
relative (recorded-input baseline <= 1.6e-16). `dmimp/dhimp/dsimp` are zero for lakes in these steps (not exercised).
Tests: +3 in `tests/test_chain_two_substeps.py` (17 total).
**Scope:** lake cells only -- ocean cells' ADDICE needs the ocean model's fluxes (`enrgfo`, `acefo` from the ocean, recorded);
SIMELT and RIVERF are not in this chain.
So the lake surface chain now runs end to end on our own results: PBL -> tile fluxes -> aggregation -> ATURB (x2) -> GROUND_SI
-> GROUND_LK -> ADDICE, with land runoff/state, ocean fluxes and between-step physics taken from the real dumps.

## D22 — Land tile (our PBL itype 4 -> our JAX GHY) in the surface chain, both substeps
`fullfidelity/land_chain.py` mirrors GHY_DRV.f `earth`: our PBL on the land records, its outputs become GHY's forcing
(`ts=tsv`, `qs=qsrf`, `rho=100*ps/(R*tsv)`, `ch`, `vs=ws`, `tprime/qprime`, `qm1=q1*ma1`; all verified bit-exact against the
recorded GHY inputs), `ghy_jax.advnc` runs, and the patch fields are formed as `earth` does (`uflux1=cm*ws*rho*us`,
`dth1=-(SHDT+dLWDT)/(sha*ma1)` with `SHDT=-ashg`, `dLWDT=dtsurf*(TRUP-stbo*(tbcs+tf)^4)`, `dq1=aevap/ma1`). Substep 2 gets its
GHY state from substep 1's outputs and its PBL ground columns from GHY (`tg=tsns+tf`, `tr4=(tbcs+tf)^4`, `qg_ij`, `evap_max`,
`fr_sat`). New/changed: `ghy_jax.advnc` now also returns `evap_max_ij`/`fr_sat_ij` (GHY_DRV.f:1292 calls `evap_limits(.false.)`
after advnc on the final state after a fresh `hydra`, with the last substep's Ent conductances) -- matches the recorded values
at roundoff (5e-17 abs on 1e-6 scale; `fr_sat` exact on 96% of cells, the rest sit on a threshold).
**REAL*4 literal found again:** `GHY_DRV.f`'s `0.001` in `qg_nsat + evap_max/(0.001*rcdhws)` is single precision
(0.0010000000474974513); using the double 0.001 gave a constant 4.75e-8 relative error in `qg_aver`, found by solving for the
implied `rcdhws` (constant ratio 1.0000000475) rather than assumed.
Single substep vs the recorded land patch (nov26): `uflux1/vflux1` 1.5e-13/1.9e-13 relative, `tsavg/qsavg` <= 2.6e-15/1.6e-13,
`dth1` median error 1e-14, 99th percentile 4e-9, max 1.9e-6 (1e-6 relative); `dq1` median 2.5e-19, 99th 2e-10, max 4.7e-7.
The maxima come from ONE cell (62,34) where GHY's runoff activates differently (`aruns` 3.0e-4 vs 3.9e-4) -- the documented D9
`aruns/aeruns` threshold-crossing sensitivity, not a new error source.
Two chained substeps WITH land, final substep-2 ATURB exit vs real (RMS over the grid / worst cell / signal RMS):
nov26 T 2.2e-7 / 6.1e-5 / 3.5e-3 K, TKE 5e-6 / 1.9e-3 / 8.7e-2, U 1.8e-6 / 4.0e-4 / 5.9e-2 m/s (the worst cell is the (62,34)
threshold cell propagating); dec01 T 5e-10 / 1.0e-7 / 3.6e-3, U 2.2e-9 / 4.6e-7 / 7.2e-2; jan01 T 7e-10 / 2.1e-7 / 3.4e-3,
U 9.8e-10 / 2.0e-7 / 6.3e-2. Without land chained (D19) every field was at 1e-12; the difference is GHY's own known accuracy.
**Recorded / inferred, stated plainly:** Ent's per-substep exports (canopy conductance, `betadl`, LAI, `dts`), the precipitation
and radiation forcing, and land's `TRUP_in_rad` -- a radiation output that is not dumped for land cells, so it is reconstructed
from substep 1's recorded land patch (`infer_trup`); the reconstruction is constant across substeps to 6e-14, which is the
consistency check available (it cannot be compared with the ocean/ice value: radiation gives each surface type its own).
Tests: `tests/test_land_chain.py` (12; bounds are set from measured accuracy, with the worst-cell bound explicitly allowing the
D9 threshold-flip cells).
**Not chained:** Ent itself (so no multi-step land), land runoff into the lake budget, ocean fluxes, between-step physics.

## D23 — Speed of the chained surface step vs the real Fortran (CPU, honest number)
The real run's own timer table (`ModelE_Support/huge_space/P2SAoM40/P2SAoM40.PRT`, 16,032 steps) gives, per DTsrc step, avg
`SURFACE()` 0.272 s (both NIsurf substeps: ocean/ice/land-ice/land tiles, PBL, GHY, aggregation, ATURB), `GROUND_SI()` 0.0021 s x2 calls,
`GROUND_LK()` 0.0003 s: about **0.28 s** for the scope our chain covers (one Fortran thread; hardware of that run not recorded here).
Track B, `time_chain_substep.py`, all 3,170 cells, warm (after one-time jit compile of ~220 s), one substep = ocean/ice PBL+tiles
0.13-0.16 s (3,492 tiles), land-ice 0.03 s (346), land PBL+GHY 0.19-0.22 s (753), aggregation+ATURB+wind update 0.06-0.11 s:
**0.45 s per substep, ~0.9 s per DTsrc step (+ GROUND_SI/LK, small)**; identical when pinned to one core
(`taskset -c 0`, single-threaded XLA): 0.46 s. So on CPU the JAX surface chain is **~3x SLOWER than the Fortran**, not faster.
The 20-100x figures in D14-D16 are vs the plain-Python references, never vs Fortran, and should not be read as a Fortran speedup.
Two measured wrapper mistakes fixed on the way (both from calling a `lax.scan`-based function un-jitted, which re-traces and
re-compiles every call): GHY 32 s -> 0.2 s and ATURB+UV 2.2 s -> 0.06 s per substep; results bit-identical afterwards.
Caveats: 'optimized' here is `jit` only; the Python glue between stages (per-record batch building for GHY, dict lookups) is still in
the timings; the Fortran number is from the original production run, not a same-node re-time; GPU behaviour is unmeasured (no GPU
node) and is where any speedup would have to come from.

## D24 — Ocean-cell ADDICE on the chained GROUND_SI state; SIMELT scoped OUT; glue profile
**Ocean ADDICE:** for ocean cells the ADDICE ice-state inputs equal GROUND_SI's final state exactly (0.0 error on all 482 nov26 cells
that had an ice record). `chain_two_substeps.form_si_ocean_stage` runs ADDICE (D12/D14) on OUR chained GROUND_SI state with the
ocean model's `enrgfo/acefo/acefi/enrgfi/salto/salti/flead` still recorded (the ocean model is not ported). The error is exactly
GROUND_SI's own (D10) residual passed through ADDICE (`hsil` 9.8e-10, `ssil` 1.9e-7, `msi2` 1.4e-8 relative; baseline with recorded
inputs 1e-13..1e-16); state-input difference ~1e-8 relative (accumulated-flux differences, 1.1 J/m^2 on 1e8). 3 dates pass.
(A first test version compared against the exact recorded-input baseline and failed -- wrong reference, since ADDICE cannot remove
upstream error; corrected to bound by the GROUND_SI residual and made relative rather than absolute.)
**SIMELT scoped out:** its recorded input state (`ffm_*`) matches neither GROUND_SI's input nor ADDICE's output (0-30% exact
matches), i.e. it runs after sea-ice dynamics/ocean stages (DYNSI/OCEANS) that change the ice state in between; that is beyond the
surface chain, so SIMELT stays validated in isolation (D12) and is not in the chain.
**Lake runoff not chained:** GROUND_LK first adds land runoff to MWL/GML/TLAKE/MLDLK; the dump records the lake state only AFTER
that addition, so the step-start lake state needed to chain it is not available without new instrumentation.
**Glue profile (warm substep, 0.45-0.5 s):** GHY stage ~0.09 s compute + ~0.1 s glue (per-record `GC.unpack` 0.035 s, batch
building, device transfers); PBL ~0.1 s compute + ~0.04 s unpack; ATURB+winds ~0.06 s all compute. Removing all glue would leave
~0.3 s per substep, still ~2x the Fortran on CPU, so glue is not the lever; not pursued.
Tests: +3 (`test_ocean_addice_on_chained_ground_si_state`), 20 in `tests/test_chain_two_substeps.py`.

## D25 — Land state carried over two consecutive steps (4 substeps): error growth
Identities first (bit-exact, 0.0): GHY prognostic state (`w, ht, nsn, dzsn, wsn, hsn, fr_snow`) and the PBL land profiles and
`cm/ch/cq` at step k+1 substep 1 equal step k substep 2's outputs. `land_chain.run_land_multistep` carries them (plus the ground
columns built from GHY's outputs) with OUR code over 4 substeps; the atmosphere-dependent inputs, Ent exports, precipitation/radiation
forcing and q1 stay recorded at every substep. Results (max / RMS over the ~750 land cells; scales: `tbcs` ~52 C, `w` ~0.8 m):
- dec01 `tbcs` RMS by substep: 1.1e-8, 1.4e-7, 5.9e-6, 1.4e-5 K; jan01: 1.2e-8, 2.2e-7, 1.3e-5, 2.9e-5; nov26: 2.3e-7, 9.6e-5, 1.2e-4,
  6.9e-5 (one known runoff-threshold cell drives the maxima). Soil water `w` RMS 5e-8 -> 1.9e-7 (max 3.8e-6 -> 1.5e-5), i.e. a slow
  ~linear drift of ~2e-5 relative after 4 substeps. `alhg` RMS ~1e-5..4e-5 relative.
- **Carry amplifies error, then saturates:** starting step 2 from RECORDED state gives `tbcs` RMS 1.1e-8 (dec01) vs 5.9e-6 when the
  state was carried from our own substeps -- a ~500x amplification of a ~1e-7 state error, growing ~10-40x per substep for the first
  substeps and then flattening near 1e-5 K. This is the behaviour of a sensitive nonlinear system (ground temperature <-> PBL
  stability <-> fluxes) and is consistent with, not proof of, the model's own measured chaos floor (D3: 1 ulp -> 0.27 K RMS in 5 days).
  It is not a bug signal: it appears from substep 2 onward with ~1e-7 input error, and the recorded-state errors (D9/D22) are 1e-8.
**Scope:** land only (its state is changed only by GHY, so it carries exactly). Ice/lake/ocean state CANNOT be carried this way: e.g.
next-step ice `tg1/tg2` and snow match the post-ADDICE state on only ~5% of tiles, because precipitation (CONDSE), sea-ice dynamics and
the ocean model modify it in between -- chaining those across steps needs those inputs dumped (new instrumentation) or ported.
Test: `tests/test_land_chain.py::test_land_state_carried_over_four_substeps_stays_small` (dec01).

## D26 — PRECIP_SI/PREC_SI (Stage 1 of the DYNSI/ocean port) vs real Fortran
First deliverable of the DYNSI/ocean-port commitment (Phase 5). New Fortran instrumentation
(`instrumentation/SEAICE_DRV_precsi.f.patch` + `ATM_DRV_precsi.f.patch`, applied after the existing
`SEAICE_DRV.f.patch`/`ATM_DRV.f.patch`) dumps `PRECIP_SI`'s per-cell `PREC_SI` call (sea-ice precipitation,
`SEAICE_DRV.f`/`SEAICE.f`); rebuilt the instrumented binary in a fresh scratch copy and reran all 3 real dates
(nov26/dec01/jan01, 6 steps each) to generate `ffw_<itime>.bin` -- a genuinely new oracle dump, not reused from
earlier work. (Found and fixed a stale doc bug on the way: `build_and_run.md`'s run command used `-l run.PRT`,
which is not a real flag -- `MODELE_DRV.f`'s parser only accepts `-r`/`-cold-restart`/`-i`/`--time`; corrected to
`-i I > run.PRT`.)

`fullfidelity/seaice_core_ff.py`'s new `prec_si`/`Fi` reuse `get_snow_ice_layer`/`relayer`/`relayer_12`/
`set_snow_ice_layer`/`tice`/`Mi`/`Em` verbatim (all already ported, D10/D14) -- only `Fi` (SEAICE.f, ~20 lines,
`seaice_thermo='BP'` case) was new. `seaice_core_jax.py`'s batched `prec_si` computes every branch (both
"has existing snow" vs "no snow", both "all layer-1 snow melts" vs "some remains", both compression-placement
sub-branches) for every lane and merges with `jnp.where`, the same pattern as `get_snow_ice_layer`'s existing
P/Q merge.

Validated on **13,572 real sea-ice cells** (6 steps x 3 dates, ~700-800/step, same cell population as
`ffi_*`/GROUND_SI since both are `si_ocn`-based): `snow`/`msi2`/`ssil`/`cmprs` 0.0 error; `hsil` 3.8e-6 abs on a
8.5e8 scale (~4.5e-15 relative); `tsil` 1.2e-8 abs (deg C); `run0`/`srun0`/`erun0` <= 5.2e-12 abs on an O(1) scale;
`wetsnow` flag matches on all 13,572 cells. Both plain-Python and JAX versions checked against real Fortran
directly (not just against each other). Non-vacuous: precip active on 91% of cells (nov26), melt/freeze runoff on
21 cells, wetsnow on 27, the "no existing snow" branch on 14; the snow-compression (`CMPRS>0`, `SNOMAX` exceeded)
and salt-in-runoff (`SRUN0!=0`, ice actually melting not just snow) branches never trigger in this 3-date record --
documented, not hidden (same pattern as D12/D14's ADDICE branches). Tests: `fullfidelity/tests/test_precsi_jax.py`
(6, incl. jit-equivalence and 2 mutation checks -- one forcing the compression branch to fire, confirming it is
real code, not dead weight).
**Scope, stated plainly:** `PRECIP_LK` (precip onto lake ice, `LAKES.f`, 139 lines, separate routine) is not yet
instrumented/ported -- next in Stage 1.

## D27 — PRECIP_LK (Stage 1 of the DYNSI/ocean port) vs real Fortran
Second Stage 1 deliverable. New Fortran instrumentation (`LAKES_precip_lk.f.patch` + `ATM_DRV_precip_lk.f.patch`,
applied after the D26 patches) dumps `PRECIP_LK`'s per-cell call (`SURFACE.f:322`, inside the `DO NS` loop, called
every substep, not once per step). Reran all 3 real dates with the newly rebuilt binary; **found and fixed a real
operational mistake along the way**: the second run silently executed zero steps because the FIRST run's own
checkpointing had overwritten both `fort.1.nc` and `fort.2.nc` in the scratch run directory with the post-6-step
state (GISS ModelE's double-buffered restart), so the "restart" file no longer held the original start time --
caught because the timer table showed 0 trips for every routine, not by assuming success. Fixed by re-copying the
untouched source restart files before every rerun. Confirmed the rebuild did not perturb `PRECIP_SI`'s own D26
output (`ffw_*.bin` byte-identical to the pre-rebuild dumps).

`fullfidelity/lakes_ff.py`'s new `precip_lk` is pure algebra (mass/energy bookkeeping for the lake reservoir plus
land-ice/sea-ice runoff and lake-ice melt, reusing `RHOW`/`SHW`/`TF` already defined) -- no new physics helpers
needed, unlike D26. `lakes_core_jax.py`'s batched version computes the `flake>0` and `flake<=0` branches for every
lane and merges with `jnp.where`.

Validated on **17,040 real lake/land-ice cells** (6 steps x 3 dates, ~950-1000/step): `mwl`/`gml`/`tlake`/`mldlk`/
`dlake`/`glake` **bitwise exact (0.0 error)** on all cells (both plain-Python and JAX); `gtemp`/`gtemp2`/`gtempr`
bitwise exact on the 10,932 `flake>0` cells where they are genuinely computed -- the `flake<=0` (land-ice-only)
pass-through case (6,108 cells) is correct by inspection (a bare return of the given value) but not independently
checked, since this dump does not separately capture the pre-call `gtemp`/`gtemp2`/`gtempr` (documented, not
hidden, in `precip_lk_compare.py`'s module docstring). Non-vacuous: 10,932 `flake>0` cells, 6,108 land-ice-only
cells, real lake-ice melt (`MELTI!=0`) on several hundred cells. Tests: `fullfidelity/tests/test_precip_lk_jax.py`
(6, incl. jit-equivalence and a mutation check).
**Scope, stated plainly:** `IRRIG_LK` (irrigation withdrawal, 128 lines, real for this rundeck -- `IRRIGATION_ON`
is defined) and `PRECIP_LI` (land-ice precip, 146 lines), both called immediately before `PRECIP_LK` in the same
`SURFACE.f` block, are not yet instrumented/ported.

## D28 — PRECIP_LI/PRECLI (Stage 1 of the DYNSI/ocean port) vs real Fortran
Third Stage 1 deliverable. New Fortran instrumentation (`LANDICE_DRV_precli.f.patch` + `ATM_DRV_precli.f.patch`,
applied after the D27 patches) dumps `PRECIP_LI`'s per-tile call (`SURFACE.f:~300`, inside the `DO NS` loop, just
before `PRECIP_LK`). Rebuilt and reran all 3 dates in one cycle (applying the fort.1.nc/fort.2.nc restart-refresh
fix learned in D27); confirmed the earlier `ffv_*`/`ffw_*` dumps were unperturbed (byte-identical) by the rebuild.

`fullfidelity/landice_precip_ff.py`'s `precli`/`precip_li` are new, self-contained (`PRECLI` "uses nothing, no
globals involved" per the real source's own comment) -- two-layer land-ice column physics: rain either heats/melts
the top layer (possibly melting through to the second layer, moving ice mass up) or snow accumulates (possibly
compacting into ice, moving mass down), the same structural shape as `PREC_SI` (D26) but simpler (no salt, no
brine-pocket thermodynamics). `landice_precip_jax.py`'s batched version merges both top-level branches (rain vs
snow) and their sub-branches with `jnp.where`.

Validated on **3,723 real land-ice tiles** (6 steps x 3 dates, ~200-250/step, `ihc=1` only per D22's trivial
height-class finding): `snow`/`tg1`/`tg2`/`run0`/`edifs`/`difs`/`erun2` **bitwise exact (0.0 error)**, both
plain-Python and JAX, first try. Non-vacuous: precip active on essentially every real cell; snow-compaction
(`DIFS!=0`) fires on 4 cells. **The entire "rain" branch (`ENRGP>=0`) is never exercised** by this 3-date record
(all real precip here is cold/snow) -- checked instead against the plain-Python reference on synthetic inputs
covering both rain sub-branches (partial top-layer melt, and melt-through-to-layer-2 with genuine ice-mass
transfer, confirmed >100/200 synthetic cells exercise the transfer), the same honest-scoping pattern as D14's
ADDICE synthetic-branch test. Tests: `fullfidelity/tests/test_precli_jax.py` (7, incl. the synthetic-rain test and
a mutation check).
**Scope, stated plainly:** `IRRIG_LK` (irrigation withdrawal, 128 lines, real for this rundeck) is called
immediately before `PRECIP_LK` in the same `SURFACE.f` block and remains not-yet-instrumented -- it depends on an
external prescribed irrigation-demand dataset (`irrig_water_pot`, from the `IRRIG` NetCDF input), which would be
recorded as an input (same pattern as Ent's forcing) rather than re-derived; deferred as its own item given the
added complexity (year-based cyclic/transient mode selection, groundwater-fallback logic).

## D29 — DYNSI/VPICEDYN/FORM/PLAST/RELAX (Stage 1 core, DYNSI/ocean port)
Fourth Stage 1 deliverable. This is the largest, most numerically dense routine ported in the project so far
(~1,300 lines: `DYNSI`'s own body, `VPICEDYN`, `FORM`, `PLAST`, `RELAX`) and the first delta where the initial
port was NOT bitwise/float64 exact on the first try -- three real, distinct bugs had to be found and fixed
before it was. Recorded here in full, including the debugging path, because each bug is a reusable lesson.
New Fortran instrumentation, all newly designed
(no prior per-cell dump pattern applied to a genuine 2D grid solve before): `ICEDYN_DRV_dynsi.f.patch` (adds 3
call sites inside `DYNSI` itself: `ffdump_geom()` once at entry, `ffdump_dynsi_in`/`ffdump_dynsi_out` bracketing
`CALL VPICEDYN`) + `ATM_DRV_dynsi.f.patch` (the 3 dump subroutines, units 970-972 -- 973+/980+ were rejected
after finding real conflicts, e.g. unit 981 is genuinely used by `PBL_DRV.f`'s `WRITET_PARALLEL`). Dumps are
**whole-grid snapshots** (not per-cell records, since this is a real 2D iterative solve): `ffz_geom.bin` once
(static geometry + FOCEAN mask), `ffy_<itime>_{in,out}.bin` once per `DYNSI` call (6/date).

**A real, embarrassing debugging detour** (documented because it cost real time and the fix is reusable
knowledge): the first several rebuild-and-rerun cycles produced zero dump files and a `DYNSI()` timer table
entry that never changed no matter what was put in `DYNSI`'s body -- including an unconditional `STOP` statement
as literally the first line, which still didn't fire. Root cause: the run script's `./P2SAoM40` is a **separate
file from `P2SAoM40.bin`** in the run directory (not a symlink) -- every rebuild had been faithfully copied to
`P2SAoM40.bin`, never to the actually-executed `P2SAoM40`, so a stale pre-D29 binary ran unchanged every time.
The `DYNSI()` timer's stable "6" trips across all those reruns was itself a red herring: GISS ModelE checkpoints
its cumulative CPU-timer table into the restart file, so the stale count just carried through from whatever
un-instrumented run originally produced that restart. Lesson for `build_and_run.md`: **always copy the rebuilt
binary to both `P2SAoM40.bin` and `P2SAoM40`** in the run directory, and don't trust a timer-table trip count as
proof of *this* run's execution when a restart file could be carrying it forward.

**Geometry (`icedyn_geom_ff.py`, `GEOMICDYN`/`ICDYN_MASKS`): validated bitwise exact**, all 15 fields (`DXT`,
`DXU`, `BYDX2`, `BYDXR`, `DYT`, `DYU`, `BYDY2`, `BYDYR`, `CST`, `CSU`, `TNGT`, `TNG`, `BYCSU`, `SINEN`, `BYDXDY`)
plus the `HEFFM`/`UVM` land/velocity masks, against `ffz_geom.bin`. This confirmed the ice-dyn grid geometry is
purely analytic (no dependence on any prognostic field) and that `RADIUS` -- a *runtime* planet parameter under
`USE_PLANET_RAD`, not the hardcoded 6371000 m default -- is inferred exactly from the dump's own `DXT` column
(`DXT = DLON*RADIUS`, `DLON` known exactly from `IMICDYN`) rather than assumed; it happens to equal Earth's
radius for this rundeck.

**`FORM`/`PLAST`/`RELAX`/`VPICEDYN` (`icedyn_dynsi_ff.py`): validated bitwise/float64-exact on all 18 real
records** (6 steps x 3 dates; `dynsi_compare.py`, `tests/test_dynsi_ff.py`). Final max relative error across
every field (`UICE`/`VICE`(:,:,1), `DMU`/`DMV`, `USI`/`VSI`) on every record is ~1e-8 to 4e-8 -- ordinary
float64 accumulation noise through a real iterative nonlinear solve (`KKI=2` every record on this Stage-1
window), not a remaining bug. Getting there required finding and fixing three separate real bugs, each
isolated by adding a new debug-only Fortran dump (`ffdump_form1`/`ffdump_relax1`/`ffdump_relax_coefs`,
NOT part of the permanent instrumentation set, discarded after use) and bisecting the pipeline stage by stage
(geometry -> `FORM`/`PLAST` -> `RELAX` stage 1 (UICE, I-direction) -> full `RELAX` -> full `VPICEDYN`):

1. **`osurf_tilt` assumed 0, actually defaults to 1** (`SEAICE.f`'s `INTEGER :: osurf_tilt = 1`, not
   overridden by this rundeck) -- `FORM`'s force-tilt term was using the wrong branch (geostrophic estimate
   instead of the explicit `AMASS*PGFUB/PGFVB` sea-surface-tilt term). Found by dumping `FORM`'s own output
   (`ETA`/`ZETA`/`PRESS`/`DWATN`/`FORCEX`/`FORCEY`) right after its first call in `VPICEDYN` and comparing
   independently of `RELAX`: `ETA`/`ZETA`/`PRESS`/`DWATN` were already exact, isolating the bug to `FORCEX`/
   `FORCEY` specifically. Fixed by passing `osurf_tilt=1`; `FORM`/`PLAST` then matched to machine epsilon.
2. **`BYDTS` passed as `DTS` (900s) instead of `1/DTS`** -- GISS naming convention (`BY<x>` = `1/<x>`) violated
   in the comparison harness, not in the ported module itself (`icedyn_dynsi_ff.py`'s `relax()`/`vpicedyn()`
   correctly use their `bydts` parameter as-is; the caller was constructing the wrong value). This inflated
   every `AMASS*BYDTS` term in `RELAX` by a factor of 900^2, dominating `BU`'s diagonal at the pole-adjacent
   row and corrupting the whole tridiagonal solve. Found by adding a whole-grid dump of `RELAX`'s own `AU`/
   `BU`/`CU`/`URT` coefficient arrays for the first (I-direction, cyclic) solve and comparing directly:
   `AU`/`CU` were exact, `BU`/`URT` were off by orders of magnitude specifically where `AMASS*BYDTS` terms
   dominate. Fixed by passing `bydts=1.0/dts`.
3. **A genuine transcription slip in `RELAX`'s own algebra**: the second-stage `VICE` J-direction `VRT` term
   used `(AA1+AA2)` where the real Fortran uses `(AA3+AA4)` (`ICEDYN.f:788`, distinct BYCSU-weighted `ETA`-only
   sums, not the plain `ETA+ZETA` sums `AA1`/`AA2` used in that block's own `AV`/`BV`/`CV` a few lines above --
   easy to conflate since both symbol pairs are in scope). This left `UICE` exact but `VICE` badly wrong.
   Found by comparing the full `RELAX` output (not just stage 1) against `ffr_<itime>.bin` after fix #2:
   `UICE` was now exact, `VICE` was not, pointing straight at the V-equations. Fixed the `vrt2` computation to
   compute and use its own `AA3`/`AA4`, matching the source exactly.

With all three fixed, the ONLY remaining discrepancy was `DMU`/`DMV` being consistently ~half their reference
value -- traced to `DTsrc` for this rundeck being **1800s, not 900s** (`decks/P2SAoM40.R:237`; `DYNSI` runs once
per full `DTsrc` step, unlike the 900s `NIsurf`-substep timestep this project has used everywhere else). A
second, more subtle correctness point also mattered here: `DMU`/`DMV` in the real code use `DWATN` as left by
`VPICEDYN`'s *last internal* `FORM` call (computed from the Euler-averaged `UICE`, one iteration stale), not a
fresh `DWATN` recomputed from the final converged velocity -- `vpicedyn()` now returns this `last_dwatn`
explicitly rather than requiring (or permitting) a fresh outside recomputation, which would silently give a
plausible-looking but wrong answer.

`TRIDIAG_cyclic`/`TRIDIAG_new`/`tridiag_thomas` (transcribed from `TRIDIAG_MOD`'s Sherman-Morrison-augmented
Thomas algorithm) were verified independently against dense linear-algebra residuals and were never the
problem -- all three bugs were in coefficient assembly or unit handling around the solves, not the solves
themselves, which is why isolating each one required dumping intermediate arrays rather than auditing the
solver. `icedyn_geom_ff.py` (`GEOMICDYN`/`ICDYN_MASKS`) remains bitwise exact as reported. Tests:
`tests/test_dynsi_ff.py` (24: geometry x3 dates, `VPICEDYN` x18 records, both tridiag solvers against dense
solves, a `RELAX` mutation check).
**Not yet ported:** the JAX/batched version (plain Python first, per this project's established order) --
`jax.lax.while_loop` for the `KKI` outer loop and batched-column tridiagonal solves are designed (see
`FULL_FIDELITY_PLAN.md`) but not yet implemented, now that there is a confirmed-correct plain-Python reference
to validate against. Also not yet ported: the earlier atm-stress/ocean-current regrid inside `DYNSI`'s own body
(`GAIRX`/`GAIRY`/`GWATX`/`GWATY`/`PGFUB`/`PGFVB`, `HEFF`/`AREA`/`AMASS`/`COR` derivation) -- these remain
recorded/real inputs rather than re-derived, since they depend on ocean-model fields (`OGEOZA`/`UOSURF`/
`VOSURF`) not yet ported.

## D30 — CALC_APRESS (Stage 1 of the DYNSI/ocean port)
Fifth Stage 1 deliverable, and a return to the project's usual first-try-exact pace after D29's three-bug
detour. `SEAICE_DRV.f`'s `CALC_APRESS` computes the total atmosphere+sea-ice pressure anomaly at the ocean
surface (`APRESS`) -- a single-formula, branch-free per-cell calculation: `100*(SRFP-1013.25) +
RSI*(SNOWI+ACE1I+MSI)*GRAV`, with a north/south-pole replication fixup. New instrumentation
(`SEAICE_DRV_apress.f.patch` + `ATM_DRV_apress.f.patch`, unit 977, `ffz_apress_<itime>.bin`, one record per
atm-grid cell per call, ~3,170/step) -- built by first reverting the throwaway D29 debug instrumentation
(`ffdump_form1`/`ffdump_relax1`/`ffdump_relax_stage1`/`ffdump_relax_coefs`, `ICEDYN.f`'s debug edits) back out
of the scratch tree and confirming a byte-for-byte match against the reconstructed clean post-D29 baseline
before diffing, so the new patch contains only the `CALC_APRESS` addition.

`GRAV` is, like `RADIUS` in D29, a *runtime* `USE_PLANET_RAD` planet parameter rather than the hardcoded
9.80665 m/s^2 default -- `apress_compare.py` infers it algebraically from a real ice-covered cell in the dump
(`GRAV = (APRESS - 100*(SRFP-1013.25)) / (RSI*(SNOWI+ACE1I+MSI))`) rather than assuming it; it happens to equal
Earth's standard gravity for this rundeck, same pattern as `RADIUS` happening to equal Earth's radius.

`fullfidelity/apress_ff.py`/`apress_jax.py`: bitwise/float64-exact on all 18 real records (6 steps x 3 dates,
3,170 real cells/record) on the first try -- no branches to get wrong, no ADI algebra to transcribe. Tests:
`tests/test_apress_jax.py` (39: real-record validation x18 for both plain-Python and JAX, a synthetic-input
cross-check, jit-vs-eager, a mutation check confirming all four inputs actually affect the output).

Full project regression suite (196 tests across all prior deltas) reconfirmed green after the D29 fixes and
before this delta, and after this delta's tests were added.

## D31 — seaice_to_atmgrid (Stage 1 of the DYNSI/ocean port)
Sixth Stage 1 deliverable. `SEAICE_DRV.f`'s `seaice_to_atmgrid` reconciles the ocean-grid sea-ice state
(`si_ocn`, just updated by `GROUND_SI`/`FORM_SI` on the ocean grid) onto the atm-grid copy (`si_atm`) that
`SURFACE`/`PBL`/radiation read from -- the state-copy half of D24's "`GROUND_SI`/`FORM_SI` runs twice" finding,
and genuinely needed to close the D25 ice-state-carry-over gap. Called from several sites each step
(`ATM_DRV.f`, `OCN_DRV.f`, `SURFACE.f`); new instrumentation (`SEAICE_DRV_s2ag.f.patch`+`ATM_DRV_s2ag.f.patch`,
unit 978, `ffz_s2ag_<itime>.bin`) dumps every call from every site into one pool, since it's the same pure
function regardless of caller (~12,680-19,020 records/step depending on date/step, i.e. multiple calls/step
each covering the full atm grid).

Ported the second loop (deriving `GTEMP`/`GTEMP2`/`GTEMPR`/`ZSNOWI`/`ZSI`/`FWSIM` from the just-copied state);
the first loop is a plain field copy with no arithmetic to port. Reused `Ti`/`Ti2b` directly from
`seaice_core_ff.py`/`seaice_core_jax.py` (already present from earlier GROUND_SI deltas) rather than
re-deriving them. **Not ported** (radiation-adjacent, out of scope per the standing SOCRATES-is-third-party
constraint): the third loop's conditional `RESET_SURF_FLUXES` call, which only redistributes `RAD_COM`'s
`FSF`/`TRSURF` diagnostic accumulators across a changed ice fraction and feeds directly into the radiation
scheme's own bookkeeping, not sea-ice prognostic state.

`fullfidelity/seaice_to_atmgrid_ff.py`/`seaice_to_atmgrid_jax.py`: validated on all 18 real records (6 steps x
3 dates, 12,680-19,020 records/record-set). `ZSNOWI`/`ZSI`/`FWSIM` (plain multiplies) are bitwise exact.
`GTEMP`/`GTEMP2`/`GTEMPR` needed an **absolute** tolerance (1e-6 degrees C) rather than relative -- the same
documented `Ti`/`Ti2b` REAL*16-internal/REAL*8-returned approximation already noted for `GROUND_SI`'s `TSIL`
field in D26 (a few jan01 records showed relative error up to 1.4e-6 purely from near-zero-temperature
denominators; absolute error at those same cells was ~1e-8 degrees, i.e. the approximation, not a new bug).
Tests: `tests/test_seaice_to_atmgrid_jax.py` (38: real-record validation x18 for both plain-Python (subsampled
for speed) and JAX, jit-vs-eager, a mutation check confirming both `MICE1`/`SNOWL` branches are exercised in
the real dump).

## D32 — UNDERICE / iceocean_fluxes / icelake_fluxes (Stage 1 of the DYNSI/ocean port)
Seventh Stage 1 deliverable. `SEAICE_DRV.f`'s `UNDERICE` computes basal mass/salt/heat fluxes between sea/lake
ice and the water below via `SEAICE.f`'s `iceocean_fluxes` (ocean domain: a 5-iteration Newton solve for the
interface temperature/salinity, `seaice_thermo='BP'`, `qsfix=.false.`, the defaults for this rundeck) or
`icelake_fluxes` (lakes domain: closed-form, no salinity) plus a shallow-lake flux-limiting wrapper. Reused
`tfrez`/`alami`/`dEidTi` directly from `seaice_core_ff.py`/`seaice_core_jax.py` (already ported for
`GROUND_SI`). New instrumentation (`SEAICE_DRV_underice.f.patch`+`ATM_DRV_underice.f.patch`, units 979/980,
`ffz_undocn_<itime>.bin`/`ffz_undlk_<itime>.bin`) dumps both domains separately since `UNDERICE` is called
from two sites with different `si_state`/`iceocn` pairs (`OCN_DRV.f` for ocean, `SURFACE.f` for lakes) and the
two domains use genuinely different physics, not just different data. `KOCEAN=1` for this rundeck
(`decks/P2SAoM40.R`), so the ocean domain's "fixed SST" fallback (`KOCEAN<1`) is dead code, not ported.

**One real bug, found and fixed via the same intermediate-value comparison discipline as D29**: `mflux`/`sflux`
matched exactly on the first try, but `hflux` did not (up to ~180% relative error on some cells). Root cause:
`Tb`/`lh` are set inside the 5-iteration Newton loop and the real Fortran's final `hflux` uses whatever `Tb`/
`lh` were last set *inside* the loop -- i.e. computed from the iteration's *pre-update* `Sb0`, one step behind
the loop's own final `Sb0` -- not a fresh `tfrez(Sb0)` recomputed from the final converged salinity (which is
what the port did initially). Fixed by simply not recomputing `Tb` after the loop and letting Python's
loop-scoped `Tb`/`lh` carry their last-set values through, matching the source's own (perhaps accidental, but
real) behavior.

`fullfidelity/underice_ff.py`/`underice_jax.py`: bitwise exact on all 18 real records (482 ocean-domain +
301 lake-domain cells on the first record alone) after the fix. The JAX port's 5-iteration Newton loop is a
plain Python `for` loop over batched arrays (fixed trip count, not data-dependent -- no `lax.scan` needed),
with the freezing/melting branch computed both ways and merged via `jnp.where` per iteration, the established
pattern. Honest scoping: the real 18-record window never has a lake shallower than the 0.4 m flux-limiting
threshold (checked and confirmed across all records, not assumed) -- that branch is cross-checked instead
against the plain-Python reference on synthetic shallow-lake inputs. Tests: `tests/test_underice_jax.py` (76:
real-record validation x2 domains x18 records for both plain-Python and JAX, jit-vs-eager, a
freezing-vs-melting mutation check, the synthetic shallow-lake branch check).

This closes the ice-dynamics side of Stage 1 except `IRRIG_LK`/`irrigate_extract` (explicitly deferred,
external-dataset dependency) and the ocean-grid `GROUND_SI`/`FORM_SI` plumbing (chaining the already-ported
physics onto `si_ocn`, not new physics to port).

## D33 — PRECIP_OC (first Stage 2 deliverable: the ocean numerical core)
The port's first delta inside the ~19,000-line ocean core proper (`OCNDYN.f`), rather than the ice-dynamics
periphery. Before porting anything, read `OCNDYN.f`'s full subroutine list (90 subroutines) and categorized its
real 6,062 lines: **~3,439 lines of genuine per-step dynamics/physics** (`OVtoM`/`OMtoV`/`OFLUX`/`OPFIL`/
`OADVM`/`OADVV`/`OPGF`/`OPGF0`/the `OADVT` advection family/`OBDRAG`/`OCOAST`/`OSTRES`/`ODIFF`/`OABFILx`/
`oabfily`/`GROUND_OC`/`OSOURC`/`PRECIP_OC`/`GLMELT`/`ADJUST_MEAN_SALT`), ~342 lines of `CHECKO`/
`CHECKO_serial` sanity-check diagnostics (this project's already-familiar `CHECKT` pattern), ~245 lines of
`CONSERV_O*` conservation diagnostics, and ~1,744 lines of one-time init/restart-I/O -- the same kind of
real-vs-dead-code correction D29 made for `DYNSI` (550 -> 1,304 lines), now applied to the much larger Stage 2
estimate. **A major de-risking finding**: `ORES_5x4.F90` confirms the ocean grid is IMO=72, JMO=46 -- the
**same resolution as the atmosphere** for this rundeck. This means the atm<->ocean regrid (`AG2OG_precip` and
its relatives, built on a general-purpose `HNTR8` area-weighted interpolation utility, not specific to any one
physics routine) can be treated as a recorded-input boundary exactly the way D29 treated `GAIRX`/`GWATX` --
Stage 2 routines can be validated against real dumps of their post-regrid inputs without needing to port the
interpolation utility itself.

`PRECIP_OC` (`OCNDYN.f:5008-5090`, 84 lines) applies precipitation (direct + through-ice runoff) to the
ocean's own top-layer prognostic state (`MO`/`G0M`/`S0M` -- mass, mass*enthalpy, mass*salinity -- the ocean's
OWN state, untouched by anything ported so far in this project). `TRACERS_OCEAN`/`TRACERS_WATER` both
undefined for this rundeck (confirmed throughout this project), so the tracer branch is dead code. New
instrumentation (`OCNDYN_precip_oc.f.patch`+`ATM_DRV_precip_oc.f.patch`, unit 973 -- reused from D29's
now-vacated debug-only units after confirming no active conflict) dumps `MO`/`G0M`/`S0M` before and after the
per-cell update, plus the real recorded inputs (`oPREC`/`oRSI`/`oRUNPSI`/`oEPREC`/`oERUNPSI`/`oSRUNPSI`,
post-`AG2OG_precip`) and `FOCEAN`/`DXYPO`.

`fullfidelity/precip_oc_ff.py`/`precip_oc_jax.py`: bitwise/float64-exact on all 18 real records (6 steps x 3
dates, ~1,810-1,893 real cells/record), first try -- no branches, no ADI algebra, a clean confirmation the
established methodology (record not-yet-ported dependencies as inputs, validate the arithmetic that's
actually being ported) scales past Stage 1 into the ocean core proper. Tests: `tests/test_precip_oc_jax.py`
(38: real-record validation x18 for both plain-Python and JAX, jit-vs-eager, a mutation check confirming all
eight inputs actually affect the output).

## D34 — OSOURC (Stage 2, called from GROUND_OC)
Second Stage 2 deliverable. `OCNDYN.f`'s `OSOURC` (123 lines, called from the not-yet-ported `GROUND_OC`)
applies surface mass/heat/salt fluxes (river+ice-melt runoff, evaporation, solar insolation) separately over
a cell's open-ocean and ice-covered fractions, checks each for below-freezing conditions (frazil-ice
formation via `GFREZS`/`TFREZS`/`Ei`/`FSSS`), recombines by ice fraction, and distributes the recombined flux
through the water column with an exponential two-band (Jerlov) solar-penetration profile. `TRACERS_OCEAN`
undefined for this rundeck, dead code not ported.

**A genuine dependency-tracing win**: `FSR`/`FSRZ`/`LSRPD` (the solar-penetration profile OSOURC needs) turned
out to be fully re-derivable rather than another recorded-input boundary. Traced `OCEAN_COM.f`'s `init_solar`:
`RFRAC`/`ZETA1`/`ZETA2`/`ZMAX_SOLAR` are hardcoded `PARAMETER`s (not runtime `USE_PLANET_RAD`-style values
like D29's `RADIUS`/`GRAV`), and `OLAYERS.F90`'s `L13` layering (`OCN_LAYERING L13`, this rundeck's build
flag) gives a fixed 13-entry `dZO` array -- so `icedyn_geom_ff.py`-style analytic reconstruction (not a real
Fortran dump) was possible and is exactly correct: `LSRPD=3` for this rundeck, confirmed by the fact that
every real-record test below passed bitwise exact using ONLY the derived values, no dump needed. `GFREZS`
turned out to be a hardcoded 41-point lookup table visible directly in `OCNFUNTAB.f`'s source (not file-based
like the `SHCGS`/`OFTAB`-table dependency `GROUND_OC` itself will need); `TFREZS` is closed-form and matches
`seaice_core_ff.py`'s existing `tfrez` exactly (same Fofonoff & Millard 1983 coefficients, different input
salinity units -- kg/kg vs PSU).

New instrumentation (`OCNDYN_osourc.f.patch`+`ATM_DRV_osourc.f.patch`, unit 974) dumps `OSOURC`'s full
argument list (inputs before the call, `MO`/`S0M`/`G0ML`/`GZML`/`DMOO`/`DEOO`/`DMOI`/`DEOI`/`DSOO`/`DSOI`
after) at its one real call site inside `GROUND_OC`, ~2,095 real (`FOCEAN>0`) cells/step.

`fullfidelity/osourc_ff.py`/`osourc_jax.py`: bitwise exact on all 18 real records, first try for the
plain-Python port. The JAX port caught one real bug before it ever reached real-data testing: an off-by-one
in the batched per-lane guard for the water-column insolation loop (`lsr >= l` where the real Fortran's `DO
L=2,LSR-1` requires the strict `lsr > l` -- an empty loop when `LSR<=2`). Found by reasoning through the
Fortran's `DO` loop semantics directly (start>end means zero iterations) before running the comparison, not
by a failing test -- but confirmed genuinely necessary immediately after: the real 18-record window does
contain shallow ocean columns with `LMIJ<LSRPD` (`LMIJ` ranges 2-13 in the first record alone), so this
wasn't a theoretical edge case. Tests: `tests/test_osourc_jax.py` (40: an `init_solar` derivation sanity
check, real-record validation x18 for both ports, jit-vs-eager, a mutation check confirming the
`LSR<LSRPD` branch and both freezing branches are genuinely exercised by real data).

## D35 — GROUND_OC's below-freezing layer sweep (Stage 2)
Third Stage 2 deliverable. The tail of `OCNDYN.f`'s `GROUND_OC` (after `OSOURC`, D34, has already updated
layer 1): a `DO L=2,LMM(I,J)` sweep checking each lower layer for below-freezing conditions (pressure-
corrected via `SHCGS`) and converting any below-freezing water to frazil-ice mass/salt/heat -- structurally
the same freezing check as `OSOURC`'s own open-ocean/under-ice branches, applied per-layer with an explicit
pressure correction to the freezing point (`GF0 = GFREZS(S0L) - SHCGS(GF00,S0L)*8.19d-8*P0L`, `TF0 =
TFREZS(S0L) - 7.53d-8*P0L`).

`SHCGS` is the one genuine external-data dependency in `GROUND_OC`'s own body (unlike D34's `FSR`/`FSRZ`/
`LSRPD`, not re-derivable from hardcoded PARAMETERs -- it's a specific-heat table read from the `OFTAB`
binary file at `init_OCEAN`). Rather than extracting/replicating the whole multi-dimensional lookup table,
this delta records the one place it's used -- `PCORR = SHCGS(GF00,S0L)*8.19d-8*P0L` -- as a recorded real
input, the established "record what's not yet ported" pattern. `P0L` (accumulated column pressure) is real
state, not a table lookup, but is recorded too rather than re-derived, since reconstructing it would mean
replaying the whole water column in the exact same accumulation order across two separate dump files --
fragile for no benefit, since `P0L` itself is trivial arithmetic, not physics insight worth re-deriving.

New instrumentation (`OCNDYN_ground_oc_sweep.f.patch`+`ATM_DRV_ground_oc_sweep.f.patch`, unit 975) dumps
every real `(i,j,l)` layer triple, ~22,224 records/step (all `FOCEAN>0` cells x their real layer count,
`L=2..LMM(I,J)`). **A design gap caught before wasting a validation cycle**: the first instrumentation
attempt recorded `PCORR` but not `P0L` itself -- missed that `P0L` is *also* used directly in the `TF0`
correction (`7.53d-8*P0L`, a different coefficient applied straight to the pressure, not folded into
`PCORR`), so the freezing branch couldn't have been validated without it. Caught by rereading the source
before writing the Python port (not by a failing test), fixed with one more instrumentation+rebuild cycle
before any port code was written.

`fullfidelity/ground_oc_sweep_ff.py`/`ground_oc_sweep_jax.py`: the non-freezing branch (all 22,224 real
records/date, 18 records total) is bitwise exact, first try after the `P0L` fix. **The freezing branch never
fires in this 18-record window** (checked across all records, not assumed) -- deep-ocean-layer freezing this
short into a 6-step run is physically rare -- so it's cross-checked against a from-scratch synthetic
below-freezing input instead, the same honest-scoping pattern as D28's rain branch and D32's shallow-lake
flux limiting. Tests: `tests/test_ground_oc_sweep_jax.py` (38: real-record validation x18 for both ports,
jit-vs-eager, the synthetic freezing-branch check, confirming the real window's freezing branch is genuinely
never triggered rather than silently skipped).

`GROUND_OC` itself (the whole subroutine, 238 lines: `OSOURC` call + this sweep + `OPRESS`/diagnostic
bookkeeping) is now fully understood and its two real-physics pieces (D34, D35) both validated -- what
remains for `GROUND_OC` as a unit is chaining the two together plus the `OIJ` diagnostic accumulation
(matches this project's `CHECKT`/`CONSERV_O*` pattern, likely skippable, not yet confirmed).

## D36: OSTRES2 -- momentum-stress application, and OCNDYN.f's legacy core found dead

**Major scope correction found while picking this delta.** OFLUX was the natural next candidate after
D33-D35's `GROUND_OC` pieces, but reading it showed it calls `OPFIL` (polar Fourier filtering with
land-basin masking, reading an external `AVR` reduction-matrix file, plus `OFFT`/`OFFTI`) -- a "new
architecture" item on D29's ADI-solve scale, not a quick delta. Grepping every call site of
`OFLUX`/`OADVM`/`OADVV`/`OPGF`/`OVtoM`/`OMtoV`/`OSTRES`/`OBDRAG`/`OPFIL` project-wide (not just within
`OCNDYN.f`) found **all of them are dead code**: `OCNDYN.f`'s entire leapfrog-dynamics driver is
literally named `OCEANS_old` and commented out in full (`OCNDYN.f:18-289`). The live driver is a
complete rewrite in `OCNDYN2.f` (`SUBROUTINE OCEANS`, called from `OCN_DRV.f:41`), with its own
renamed/restructured routines: `OFLUXV`, `ODHORZ`/`ODHORZ0`, `OPFIL2`, `OSTRES2`, `OBDRAG2`,
`OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2`/`OADVUZ`. Only `OCOAST` is shared unchanged between the two
files. `GROUND_OC` (hence D33-D35) is confirmed live, called from `OCNDYN2.f`'s real `OCEANS` at line
114 -- those three deltas are unaffected. This resolves the earlier ambiguous "~3,439 real-physics
lines, not fully differentiated" scoping for `OCNDYN.f`'s remainder: past `GROUND_OC`/`PRECIP_OC`/
`OSOURC`/`OCOAST`/`CONSERV_O*`/init routines, everything else in `OCNDYN.f` is legacy and should not be
ported -- the real remaining Stage 2 dynamical core is `OCNDYN2.f` (2,670 lines) instead. Also confirmed
via `decks/P2SAoM40.R`'s rundeck comment that `KOCEAN=1` means "ocn is prognostic" (a genuine 13-layer
dynamic ocean for this config, not slab/Q-flux as briefly hypothesized before rereading the comment).

Also scoped and **deferred** `GLMELT` (glacial meltwater, 72 lines): called only from `daily_OCEAN`'s
`end_of_day=.true.` branch (gated at `MODELE.f:802`), i.e. once per calendar day rather than once per
DTsrc step -- the existing 6-step/3-hour test windows are not confirmed to cross a day boundary, so
instrumenting it risked an unvalidatable zero-record delta. Left for a future delta with a
day-boundary-crossing test window.

Picked `OSTRES2` (`OCNDYN2.f:2478-2577`, ~100 lines) as the first delta from the corrected file: applies
atmospheric wind stress (`oDMUA`/`oDMVA`, momentum flux into open ocean) and ice-ocean stress
(`oDMUI`/`oDMVI`) to the ocean's layer-1 `UO`/`VO`/`UOD`/`VOD` velocities on the C/D grid. Both stress
sources are recorded real inputs (not yet ported upstream, same pattern as D29's `GAIRX`/`GWATX`). Live,
called unconditionally once per DTsrc step from `OCEANS` right after `GROUND_OC`; no external files, no
FFT -- deliberately smaller than the OFLUX/OPFIL combination. Confirmed via `OGEOM.f`'s `GEOMO` that the
needed static geometry (`DXYSO`/`DXYNO`/`DXYVO`/`COSIC`/`SINIC`) is fully analytic (same lat-lon formulas
as the atmosphere grid, `RADIUS=6371000.0` per D29's precedent) and that `LMU`/`LMV` (the depth masks
OSTRES2 gates on) are `MIN(LMM(i,j),LMM(i+1,j))`/`MIN(LMM(i,j),LMM(i,j+1))` (`OCNDYN.f:488,499`) --
recorded directly here rather than re-derived from `LMM`, since this delta's dump doesn't separately carry
the full `LMM` grid. Instrumented `OSTRES2` itself (not its caller) using variables already in its own
`USE` scope, dumping static geometry once (unconditional, unit 976) and a full-grid before/after record
per call (unit 981) -- avoided touching `OCEANS`'s call site or `OCNDYN.f`'s `init_OCEAN` at all. South
pole: no special case exists in `OSTRES2` at all (unlike `OVtoM`/`OMtoV`'s dead, commented-out `QSP`
branches) -- confirms this ocean grid's south pole sits inside Antarctic land, no polar-ocean singularity
there; only the Arctic-side north pole needs the `IVNP`/`COSIC`/`SINIC` rotation (`IVNP=IM/4=18` confirmed
from the dump).

**A restart-file double-buffering trap cost real time before any port code ran.** First rerun produced
zero dump files, including the unconditional geometry dump -- `OSTRES2` never executed at all. Root
cause: GISS ModelE's restart reader picks whichever of `fort.1.nc`/`fort.2.nc` has the *later* itime
(visible as `RESTART DISK READ, UNIT 2` vs `UNIT 1` in `run.PRT`), and checkpoints double-buffer across
both files -- D35's prior run in the same reused scratch directory had already advanced all three run
directories' checkpoints forward, so this rerun silently resumed from the *end* of the previous window.
Same root cause as D27's "second run silently did zero steps," but sharper: D27's fix note said only to
"re-copy fresh restart files" without stating that *both* files must be reset to the same pristine itime.
Fixed by restoring `run_nov26`'s pair from the untouched persistent master
(`ModelE_Support/huge_space/P2SAoM40/fort.1.nc`, confirmed itime=33312 via direct netCDF4 inspection --
this file has never been run in place) and by mirroring `run_dec01`/`run_jan01`'s still-pristine
`fort.1.nc` onto their stale `fort.2.nc`. Documented permanently in `build_and_run.md` so this doesn't
recur.

`fullfidelity/ostres2_ff.py`/`ostres2_jax.py`: static geometry validated bitwise-exact against the dump
(analytic derivation, not read from a file). Full-grid physics matches real Fortran to float64-op-order
noise (UO/UOD/VOD bitwise exact; VO ~1e-9 absolute, from the independently-recomputed-geometry order of
operations) on all 3 dates, first try. JAX port fully vectorized via `jnp.roll` for the C-grid wraparound-I
neighbor pattern (Fortran's `I=IMO; DO IP1=1,IMO ... I=IP1` idiom), matches both the real dump and the
plain-Python reference to float64 precision once `jax_enable_x64` was set (caught a float32-silent-
downcast before it could masquerade as a real port bug -- the initial jitless run showed ~1e-8 diffs
consistent with float32 rounding, not wrong physics). `fullfidelity/tests/test_ostres2_jax.py` (24 tests):
real-record validation for both ports x3 dates, geometry-derivation and `IVNP` checks, jit-vs-eager,
land/ocean mask mutation checks (masked-out cells must be byte-identical before/after), a non-vacuous
north-pole-exercised check, and a non-trivial-physics check (stress must actually move some real cell).

## D37: OCOAST -- coastal tracer-gradient damping, and a permanent restart-file fix

Picked `OCOAST` (`OCNDYN.f:4514-4570`, 57 lines) as the next `OCNDYN2.f`-live delta: damps the
horizontal X/Y gradient moments of ocean tracer fields (`GXMO`/`SXMO`/`GYMO`/`SYMO`) in coastal
grid boxes, multiplying layers `LMIN..LMM(I,J)` by `REDUCE = 1 - DTS/(86400*20)` where `LMIN` is
one more than the shallower of the cell's two horizontal neighbors' depth -- layers shallower
than `LMIN` (both neighbors at least as deep) are left untouched. Confirmed the one routine
genuinely shared unchanged between `OCNDYN.f` (definition) and `OCNDYN2.f` (the live `OCEANS`
driver, gated `if (OCoastal_drag==1)`, true for this rundeck's `OCoastal_drag=1`). No external
files; `DTS=DTSRC=1800.0` (`OCNDYN.f:405`) and `SECONDS_PER_DAY=86400.0` are both compile-time
constants, so `REDUCE` is fully analytic -- no dump needed for it.

**Hit D36's restart double-buffering issue again on rerun, and fixed it permanently this time.**
`dec01`'s restart pair had itself been consumed by D36's own prior rerun (D36 legitimately ran 6
real steps from `dec01`'s then-pristine `fort.1.nc`) -- unlike `nov26` (permanent untouched master
at `ModelE_Support/huge_space/P2SAoM40/fort.1.nc`), there was no longer any untouched `dec01`
restart anywhere. Also found, while checking: `jan01`'s exact restart already exists as a
permanent archive at `ModelE_Support/huge_space/P2SAoM40/1JAN1950.rsfP2SAoM40.nc` (itime=17520,
confirmed via direct netCDF4 inspection, never run in place). Regenerated `dec01`'s pristine
restart by copying the `nov26` master into a one-off bootstrap run directory, editing `I`'s
`YEARE/MONTHE/DATEE/HOURE` to run the full 5 days (240 steps, 1950-11-26 to 1950-12-01) instead
of the usual 6-step/3-hour window, and letting it run to completion (~13 min) -- while it ran,
`nov26`/`jan01` were rerun and validated immediately, since they didn't depend on it. **All three
dates' pristine restarts are now archived permanently** at
`ff_data/_pristine_restarts/fort1_{nov26,dec01,jan01}_itime{33312,33552,17520}.nc`, so this never
needs to be regenerated or hunted for again in any future session. `build_and_run.md` updated
with the exact restore procedure and dec01's regeneration recipe as a fallback.

`fullfidelity/ocoast_ff.py`/`ocoast_jax.py`: bitwise exact against real Fortran on all 3 dates,
both ports, first try. JAX port vectorizes the Fortran wraparound-I neighbor pattern via
`jnp.roll` (same technique as D36's `ostres2_jax.py`) and the variable-length
`L=LMIN..LMM(I,J)` inner loop via a broadcast layer-index comparison merged with `jnp.where`.
`tests/test_ocoast_jax.py` (17 tests): real-record validation x2 ports x3 dates, an analytic-
`REDUCE`-constant check, jit-vs-eager, and three mutation/non-vacuousness checks (some coastal
cell must exercise `LMIN>1`; changed cells must be scaled by exactly `REDUCE`, not some other
factor; cells with an empty `L`-range must be byte-identical before/after). Full regression: 505
passed (488 + this delta's 17), 0 failed, no cross-delta regressions.

## D38: OBDRAG2 -- implicit bottom-layer current drag

Picked `OBDRAG2` (`OCNDYN2.f:2592-2682`, 91 lines) as the next `OCNDYN2.f`-live delta: applies an
implicit bottom-drag deceleration only at the per-column deepest active layer -- `L=LMU(I,J)` for
the east-edge U/VOD velocity pair, `L=LMV(I,J)` for the north-edge V/UOD pair, everywhere else
untouched. The drag scales velocity by `(MO_l+MO_r)/(MO_l+MO_r+2*DTS*bdragfac)` where
`bdragfac=BDRAGX*sqrt(WSQ)` (`BDRAGX=1.0`, `WSQ` the local squared current speed plus a 1e-20
zero-speed floor) -- an unconditionally-stable implicit linear drag. Confirmed `OCN_GISS_TURB` is
not `#define`d for this build (checked the compiled `rundeck_opts.h`'s active flag list directly,
not assumed), so the tidal-enhancement branch (`taubx`/`tauby`/`rhobot`/`idrag`) never compiles
in -- only the simple `bdragfac=BDRAGX*sqrt(WSQ)` path is live. Called unconditionally from
`OCEANS` when `OBottom_drag==1` (true for this rundeck).

D37's permanent pristine-restart archive (`ff_data/_pristine_restarts/`) paid for itself
immediately here: restoring all three dates for this delta's rebuild was a single three-line
copy, no bootstrap run, no debugging -- a direct payoff of the fix built one delta earlier.

`fullfidelity/obdrag2_ff.py`/`obdrag2_jax.py`: float64-op-order exact against real Fortran on all
3 dates, both ports, first try, including the `Max(J1O,J1)..JNP` J-band (J=2..JM-1 for this
serial run) without needing an empirical correction. JAX port introduces a new technique beyond
D36/D37's uniform-J-range masking: since the affected layer varies per column, it gathers the one
relevant layer per cell via `jnp.take_along_axis`, computes the drag there, then scatters back
with a broadcast layer-index-equality mask merged via `jnp.where`. `tests/test_obdrag2_jax.py`
(17 tests): real-record validation x2 ports x3 dates, a compile-time-constants check, jit-vs-
eager, mask non-vacuousness, a per-layer mutation check (only the bottom layer may change,
checked layer-by-layer), and a physical-sanity check (drag never increases speed, and does fire
somewhere real). Full regression: 522 passed (505 + this delta's 17), 0 failed.

## D39: polar UOD/VOD relax block + polevel() -- pole-reconstruction physics

Picked the "relax UOD,VOD toward 4-pt avgs of UO,VO" block inside `OCEANS` itself
(`OCNDYN2.f:179-228`, ~55 lines) plus its `polevel()` helper (`OCNDYN2.f:1530-1564`, ~35 lines):
for each of the 13 ocean layers, relaxes the D-grid velocities (UOD at V-points, VOD at U-points)
toward a 4-point average of the C-grid velocities (UO, VO), `RELFAC=0.005` (~4-day damping time
constant). `polevel()` first reconstructs the North Pole row of UO/VO from the V-velocity ring at
J=JM-1 via a discrete wavenumber-1 projection -- the same idiom as D29's DYNSI pole handling.
Confirmed (again) no south-pole case exists, consistent with D36-D38.

Confirmed the Fortran's precomputed `nbyzu`/`nbyzv`/`i1yzu`/`i2yzu`/`i1yzv`/`i2yzv` segment lists
are an exact cached representation of the LMU/LMV point-masks (read their construction site
directly, `OCNDYN.f:505-518`) -- reused the masks already validated in D36-D38 rather than
reconstructing a new geometry primitive.

`fullfidelity/polerelax_ff.py`/`polerelax_jax.py`: float64-op-order exact against real Fortran on
all 3 dates, both ports, first try -- including both of UOD's distinct formulas (doubled term at
the pole-adjacent row vs. non-doubled in the interior) and confirming VOD's formula never needs
pole-row access at all. JAX port vectorizes `polevel`'s per-layer pole reconstruction across all
13 layers at once via a masked reduction. `tests/test_polerelax_jax.py` (21 tests): real-record
validation x2 ports x3 dates, `COSU`/`SINU` endpoint checks, jit-vs-eager, pole-row-reconstructed
non-vacuousness, a per-row UO/VO mutation check, mask non-vacuousness, and a physical-sanity
step-size bound (corrected once, from relative to absolute, after the relative version failed
near VOD~0 -- caught by the test itself, the tight real-Fortran check was never at risk).

## D40: ODHORZ0 -- pressure/equation-of-state prep, and a North Pole mask design gap

Picked `ODHORZ0` (`OCNDYN2.f:1718-1862`, 144 lines): prepares the pressure profile (`P`/`OPBOT`)
and seawater-equation-of-state quantities (`GUP`/`GDN`/`SUP`/`SDN`, `dZGdP`, `VBAR`, `DH3D`) for
the not-yet-ported horizontal pressure-gradient solve. Confirmed `USE_OPGFQ=0` for this rundeck --
the "Linear Upstream Scheme" branch is the only live one, halving the real scope (the "Quadratic
Upstream Scheme" alternative is dead code here). `VOLGSP` (seawater-EOS trilinear interpolation
over a 43x41x40-entry table, read from `OFTAB` at init -- same file as D35's `SHCGS`) is recorded
directly at its two per-cell outputs (`VUP`, `VDN`), the established pattern. Reused `polevel()`
(D39) unchanged.

**Caught a real North Pole masking bug via the dump comparison itself.** First validation attempt
failed `OPBOT`/`GUP`/`GDN`/`SUP`/`SDN` by large margins while `dZGdP`/`VBAR`/`DH3D`/`MO`/`UO`/`VO`
matched exactly. Tracing the worst mismatch to `(I=2, J=JM)` (real dump: `OPBOT=0` despite a
nonzero `LMM(2,JM)`) led back to `OCNDYN.f`'s `nbyzm` construction: the North Pole row is
hard-restricted to `I=1` only, independent of `LMM`'s value at other longitudes there -- a silent
exception the naive `LMM(I,J)>=L` mask (correct everywhere else, including D36-D39's `LMU`/`LMV`)
misses. Fixed with an explicit mask-override helper and pinned down with a dedicated regression
test. Also caught before the first validation run: a dropped `MMI=MO*DXYPO` divisor, fixed by
deriving `DXYPO(J)` analytically (reusing D36's geometry).

`fullfidelity/odhorz0_ff.py`/`odhorz0_jax.py`: bitwise/float64-tolerance exact against real
Fortran on all 3 dates, both ports, after the fixes above. JAX port vectorizes the pressure
integration via `jnp.cumsum` and the North Pole restriction via an explicit mask override.
`tests/test_odhorz0_jax.py` (17 tests): real-record validation x2 ports x3 dates, constants check,
jit-vs-eager, a dedicated North-Pole-one-cell-only regression test, pole-copy-fields-uniform
check, mask non-vacuousness.

## D42: ODHORZ -- the horizontal momentum + mass-continuity solve, first "new architecture" delta closed

Committed to `ODHORZ` (`OCNDYN2.f:1185-1560ish`) after confirming `OPFIL2`'s two per-layer outputs
(`USMOOTH`, `PGFX`) could be recorded directly (same pattern as D40's `VOLGSP`), decoupling this
delta from the `OPFIL2`/`AVR`-file dependency. Genuinely large and multi-physics: pressure/
geopotential/thickness accumulation, kinetic energy, pressure-gradient force, vorticity, Coriolis,
mass continuity -- called several times per DTsrc step (leapfrog/Euler-predictor pattern).
Comparable in scope to D29's ADI solve, the largest delta since then.

**Two real bugs caught before any validation run.** (1) Mid-writing: `OGEOZ`'s per-call init
needs `HOCEAN` (bathymetry), not yet recorded -- fixed with a small targeted instrumentation
addition (`ffz_odhorz_hocean.bin`) before writing further port code. (2) Re-reading the source
once more: `OPBOT` (2D, no layer index) accumulates mass-convergence contributions across all 13
layers within one call -- the first draft incorrectly reset it from `OPBOT0` every layer, which
would have silently discarded all but the last layer's contribution. Fixed before the first
comparison run.

`OMEGA` (planetary rotation rate, a genuine runtime parameter) used at Earth's standard sidereal
value and validated empirically -- same approach as D29's `RADIUS`/`GRAV`. Every `OGEOM.f`
geometry formula re-verified line-by-line against source before use.

`fullfidelity/odhorz_ff.py`: matches real Fortran to float64-tolerance on all 15 real call
records (5 invocations/date x 3 dates), first full validation run after the two fixes above.
`tests/test_odhorz_ff.py` (16 tests): real-record validation across all records/dates, multiple-
calls-per-window check, a dedicated `OPBOT`-accumulation regression pin, mask non-vacuousness,
bathymetry sanity check. JAX vectorization deliberately deferred as its own follow-up (this
project's GHY-lesson discipline: prove F0 correctness first on large multi-physics routines).

## D43: OFLUXV + OADVUZ -- long-timestep vertical mass redistribution

Picked `OFLUXV` (~90 lines) + `OADVUZ` (~35 lines): the vertical mass-redistribution step called
once per NOCEAN iteration right after the leapfrog `ODHORZ` loop. **Corrected an earlier scoping
assumption**: `OFLUXV` does not call `OPFIL2` at all -- the polar-filter setup module just happens
to sit next to it in the source file. No external dependencies: `DZO`/`ZE` (L13 layering) reused
from D34's `DZO_L13`; `OPRESS` recorded as a real input.

Traced the `nbyzm(j,2)`/`nbyzm(j,l+1)` layer-shifted masking before writing any code: single-layer
columns (`LMM==1`) are never touched by this routine at all. Real data has zero single-layer
columns across all 3 dates (checked explicitly) -- cross-checked against a synthetic two-column
case instead, same pattern as D28's rain branch.

**Two real bugs caught, both before/during the first validation runs.** (1) A `DXYPO`/`DTOLF`
bookkeeping error: `SMW`'s accumulation carries `DXYPO(J)/DTOLF`, later divided out again when
averaging onto U/V-points -- the `DXYPO(J)` term cancels but a residual `/DTOLF` remains; the
first draft dropped both factors, traced through the algebra by hand and fixed before running
anything. (2) The JAX port's dense vectorization produced NaN at genuinely-inactive columns (0/0
division) -- fixed by threading an explicit active-cell mask through `OADVUZ`'s `lax.scan` carry,
freezing R/CMUP/FMUP at inactive cells exactly as the Fortran's per-cell skip does.

`fullfidelity/ofluxv_ff.py`/`ofluxv_jax.py`: bitwise/float64-exact against real Fortran on all 3
dates, both ports, after the fixes above. `tests/test_ofluxv_jax.py` (18 tests): real-record
validation x2 ports x3 dates, jit-vs-eager, `ZE`/`DZO` derivation check, explicit no-real-single-
layer-columns confirmation plus synthetic cross-check, a dedicated NaN-regression pin, non-
vacuous mass-redistribution check.

## D44: ODHORZ's SMU/SMV accumulation -- completing ODHORZ for D45's tracer advection

While scoping D45 (`OADVTX2`/`OADVTY2`/`OADVTZ2`/`OADVT2`, tracer advection), found a real gap in
D42's `ODHORZ` port: the real Fortran's `OADVT2` dispatcher reads `SMU`/`SMV`/`SMW`
(`Use OCEAN_DYN, only : mb=>mmi,smu,smv,smw`) as its actual mass-flux inputs. `SMW` was already
ported in D43 (`OFLUXV`). `SMU`/`SMV` (`OCEAN_DYN`'s "integrated horizontal mass fluxes") turned
out to be accumulated **inside `ODHORZ` itself** (`OCNDYN2.f:1351,1443`) -- genuine real per-step
physics, not an external-file dependency, and D42's port never captured it since nothing D42
validated needed it.

Traced the accumulation by hand before writing any code: `SMU`/`SMV` are zeroed once per
`NOCEAN` sub-step (`NOCEAN=1` for this rundeck, confirmed in `OCEAN_COM.f`, so this reduces to
"once per `OCEANS` call" in practice) and accumulated as `smu[i,j,l] += mu[i,j]*xeven` /
`smv[i,j,l] += mv[i,j]*xeven`, where `xeven=1` only on the "even" leapfrog sub-step calls
(`qeven=.true.`) and 0 otherwise (2 of `ODHORZ`'s 5 calls per window contribute; 3 don't). The
`mu`/`mv` arrays reused for this accumulation are the exact same ones D42's port already computes
(and already validated, indirectly, via `MO`/`OPBOT`'s bitwise-exact match) for the mass-
continuity update -- so this delta only needed to *record* that accumulation, not re-derive the
underlying flux fields.

Extended `odhorz()` with optional `qeven`/`smu0`/`smv0` arguments (backward-compatible: omitting
them preserves D42's original 6-tuple return). New instrumentation: `ffdump_odhorz_smuv` (unit
993, per-call `SMU`/`SMV` before/after + `xeven`, same 5-calls-per-itime cadence as
`ffdump_odhorz`) and `ffdump_smfinal` (unit 994, the fully-integrated `SMU`/`SMV` right before
`OFLUXV` is called -- the real ground-truth input D45 will consume).

`fullfidelity/odhorz_smuv_compare.py`: two checks per date, both **bitwise-exact, first try, zero
mismatches** -- (1) each of the 5 calls' own `smu0->smu1`/`smv0->smv1` delta, replayed through the
extended `odhorz()`; (2) the full end-to-end chain (starting from `SMU=SMV=0`, using the port's
own output as each next call's input) against `ffz_smfinal`'s real recorded final values.
`tests/test_odhorz_smuv.py` (13 tests): per-call and chained-final validation across all 3 dates,
an explicit pin on the "2 of 5 calls contribute" `xeven` pattern, non-vacuous-accumulation check,
and a backward-compatibility check that the original D42 call signature still returns a 6-tuple.
JAX vectorization deliberately deferred, same as `ODHORZ` itself (D42's open item).

## D45: OADVT2/OADVTX2/OADVTY2/OADVTZ2 -- tracer advection of G0M and S0M

The largest and most intricate delta this session: the long-timestep advection of potential
enthalpy (`G0M`) and salt (`S0M`), via `OADVT2`'s Strang-splitting dispatcher (X half-step, Y,
Z, X half-step again) calling `OADVTX2`/`OADVTY2`/`OADVTZ2` (`OCNDYN2.f:1853-2423`, ~570 lines).
Confirmed `TRACERS_OCEAN` is not defined for this rundeck (the preprocessor block), so `OADVT2`
is called exactly twice per `OCEANS` invocation -- `G0M` (`QLIMIT=.FALSE.`) then `S0M`
(`QLIMIT=.TRUE.`) -- each re-deriving `MA` (mass) identically from the same `SMU`/`SMV`/`SMW`
flux fields (D43+D44), since mass evolution doesn't depend on which tracer rides along.

**Three real bugs found, each requiring a dedicated debugging cycle before landing on a
bitwise-exact result:**

1. **`OADVTX2`'s `mudt` array is genuinely stale-by-design.** `mudt` is declared once for the
   whole subroutine call (not reset per row/layer) and indices 1, 2, and `IM` are unconditionally
   refreshed from `MU` every pass regardless of activity, while every other index is only
   refreshed within that pass's own U-active segments -- everywhere else it deliberately retains
   whatever an earlier, unrelated (L,J) pass last left there. Also found (via
   `OCNDYN.f:1494`'s `get_i1i2`) that segments are explicitly **linear, not circular** ("Wraparound
   is disabled"), and that a single-cell M-segment not starting at I=1 is skipped entirely
   (`OCNDYN2.f:2066`). An initial port that reset `mudt` fresh each pass and assumed circular
   segments produced errors up to 1e14 in magnitude; rewriting `OADVTX2` as a precise,
   segment-based transliteration (via a Python `_get_i1i2` matching the real algorithm exactly)
   fixed it.
2. **A re-derived `MMI` did not match the real one.** `MA = MB` (`MB=>mmi`) at the top of every
   `OADVT2` call reads `OCEAN_DYN`'s `MMI`, which an initial attempt re-derived as
   `MO0*DXYPO(J)` from `ODHORZ0`'s already-validated `mo0` input -- this produced widespread
   (~16,600 cells) small-but-real mismatches, because `MMI` is a persistent module array that
   `ODHORZ0` only partially overwrites, not a dense recomputation. Fixed by dumping `MMI` directly
   (`ffdump_mmi`) instead of re-deriving it.
3. **`OADVTZ2`'s pole-row handling needs the same `nbyzm` restriction established in D40**:
   `nbyzm` restricts J=JM (the North Pole) to I=1 only. `OADVTZ2`'s `cmup`/`fmup`/... arrays are
   persistent per-(I,J) state carried across layers; processing every I at the pole pointwise
   (via `LMM(i,JM)`) let each I independently accumulate its own history, while the real Fortran
   leaves I=2..IM's state frozen at 0 forever (never touched) -- a divergence that grows with
   layer depth. Isolated via new debug-only instrumentation (dumping state after each of the 4
   sub-stages, S0M call only) that bisected the mismatch to exactly this routine, at exactly the
   pole row; removed before finalizing the delta's patches. Fixed with the same `m_active`-style
   override used since D40.

**A fourth, smaller fix**: `OADVTY2`'s pole-averaging (`mo(:,j,l)=sum(mo(:,j,l))/im`) used
`np.sum()` (a pairwise/blocked reduction) where ifort's `-fp-model strict` `SUM` intrinsic does
strict sequential left-to-right accumulation -- different rounding for the same 72 terms. Fixed
with a dedicated `_fortran_sum` helper.

New instrumentation: `ffdump_oadvt2_before`/`ffdump_oadvt2_after` (real per-step tracer-moment
and `SMW` state bracketing both `OADVT2` calls) and `ffdump_mmi` (the real mass field `OADVT2`
actually advects). `fullfidelity/oadvt2_ff.py`: **bitwise-exact, all 9 checked fields
(G0M/GXMO/GYMO/GZMO/S0M/SXMO/SYMO/SZMO/MA), all 3 dates**, after the four fixes above.
`tests/test_oadvt2_ff.py` (22 tests): real-record validation, an MA-identical-across-calls
cross-check, `get_i1i2` unit tests (linear-not-circular, single-cell-segment), a `SIGN`
semantics check, a `_fortran_sum` order-sensitivity regression pin, a dedicated pole-row
regression pin for the `OADVTZ2` mask bug, an `MMI`-is-recorded sanity check, and a
non-vacuous-advection check. JAX vectorization deliberately deferred (same "new architecture"
discipline as D42's `ODHORZ`) -- this routine's dynamic segment structure and pole-masking
subtlety make it a poor first candidate for batching.

## D46: mesoscale-mixing scoping -- correcting D41, sizing the next delta

With `OCNDYN2.f`'s entire real per-step dynamical core closed (D45), scoped the next item:
`OCNMESO_DRV.f`/`OCNTDMIX.f`/`OCNGM.f` (mesoscale/Gent-McWilliams mixing), previously surveyed
only at a glance in D41. That earlier scoping was **backwards**: it assumed `CONSTANT_MESO_
DIFFUSIVITY` selects a simplified path instead of the full Redi/GM slope calculation. Reading
`ocnmeso_drv` (`OCNMESO_DRV.f:131-432`) start-to-end shows the opposite -- `CONSTANT_MESO_
DIFFUSIVITY` only fixes the mesoscale-diffusivity *coefficient* (`meso_diffusivity_const=800`
m²/s, via the trivial `get_1d_mesodiff`); the full Redi/GM **skew-flux** machinery still runs.

Confirmed `use_tdmix=0` (module default, not overridden in the rundeck's parameter list) takes
`ocnmeso_drv`'s `else ! skew-GM` branch: `OCNGM.f`'s `GMKDIF` (density-gradient-derived isoneutral
slopes, itself calling `ISOSLOPE4` and `GET_PSI_DIAG`) then `GMFEXP` (applies the skew flux to
`G0M`/`S0M`, calling `computeFluxes`/`wrapAdjustFluxes`/`addFluxes`). `ocnmeso_drv`'s own
`densgrad` helper computes horizontal/vertical density gradients via `VOLGSP` (same seawater-EOS
table as D35/D40) and depends on a previously-unscoped routine found while tracing this,
`ocnstate_derived` (`OCNDYN2.f:1568-1706`, called twice per `OCEANS`, cell-centered thermodynamic
state `G3D`/`T3D`/`S3D`/`P3D`/`R3D`/`V3D`).

**`OCNTDMIX.f` (2,030 lines) confirmed entirely dead** for this build: `use_tdmix=0` means the
whole `if(use_tdmix==1)` block in `ocnmeso_drv` -- including every `OCNTDMIX.f` call -- never
executes. Do not port it. `simple_mesodiff`/`SIMPLE_MESODIFF` and the `ORIG_MESODIFF` path for
K3D remain confirmed dead (superseded by `USE_1D_MESODIFF`, implied by `CONSTANT_MESO_
DIFFUSIVITY`), consistent with D41.

Revised live-code estimate for the next delta: `ocnstate_derived` (138) + `densgrad` (144) +
`get_1d_mesodiff` (30) + `GMKDIF` (199) + `ISOSLOPE4` (136) + `GET_PSI_DIAG` (138) + `GMFEXP`
(170) + `computeFluxes`/`wrapAdjustFluxes`/`addFluxes` (~500) ≈ 1,455 lines -- comparable in
scale to D45, not yet ported. No port code written this delta; `FULL_FIDELITY_PLAN.md`'s
`OCNMESO_DRV.f`+`OCNTDMIX.f`+`OCNGM.f` row rewritten with the corrected scope.

## D47: ocnstate_derived + densgrad + get_1d_mesodiff -- the Gent-McWilliams prerequisites

First real port from D46's scoped mesoscale-mixing family: `ocnstate_derived` (`OCNDYN2.f`,
cell-centered intensive thermodynamic state), `densgrad`'s vertical-gradient portion
(`OCNMESO_DRV.f`, feeds `GMKDIF`/`GMFEXP`, not yet ported), and `get_1d_mesodiff` (the
constant-diffusivity-coefficient path). All three ported and **bitwise-exact on every checked
field, all 3 dates, first try** -- no bugs found.

**Corrected a mid-investigation assumption**: `ocnstate_derived` is called twice in the source
(`OCNDYN2.f:166` and a later unconditional call), but the first is gated by
`#ifdef TRACERS_OceanBiology`, not defined for this rundeck -- confirmed by the real dump
(1 record per itime, not 2) before writing any port code, not assumed.

`VOLGSP` (seawater-EOS lookup table, same `OFTAB` dependency as D35/D40) outputs recorded
directly (`VUP`/`VDN` for `ocnstate_derived`, plus `VUPU`/`VDNU` for `densgrad`'s vertical
gradient) -- the established "record what's not yet ported" pattern. `TEMGSP` (in-situ
temperature, `T3D`) similarly recorded but not consumed by anything in this delta's scope.
`RHOX`/`RHOY` (the horizontal density gradients, needing two more `VOLGSP` evaluations each at
inter-cell pressures) recorded as final outputs rather than further decomposed -- nothing
downstream in this delta needs the individual terms.

`densgrad` reads `OCEAN_DYN`'s module-level `DH` array, which is `ODHORZ0`'s already-validated
(D40) `DH3D` output, persisting unchanged from `ODHORZ0`'s single call earlier in the same
`OCEANS` invocation (`NOCEAN=1`, D44) -- reused directly via `ffz_odhorz0`'s existing dump rather
than re-instrumented.

**One cleanup, not a correctness bug**: `ocnstate_derived`'s first draft looped every I
pointwise via `LMM(i,j)`, including I>1 at the North Pole (J=JM) -- `nbyzm` restricts real
computation there to I=1 only (D40's established finding), so `VUP`/`VDN` are never written by
the real Fortran for I>1,J=JM; the pointwise loop computed a transient, harmless-but-noisy 1/0
there (caught by a `RuntimeWarning`, not a wrong final value -- the subsequent pole-copy step
already overwrote it correctly). Fixed with the same `m_active` mask used since D40, both for
cleanliness and to mirror the real control flow exactly rather than relying on a downstream
overwrite to paper over it.

`fullfidelity/ocnmeso_ff.py`/`ocnmeso_compare.py`: bitwise-exact (`G3D`/`S3D`/`P3D`/`VBAR`/`RHO`
from `ocnstate_derived`; `DZV`/`BYDZV`/`BYDH`/`RHOMZ`/`BYRHOZ` from `densgrad`), all 3 dates.
`tests/test_ocnmeso_ff.py` (15 tests): real-record validation, a rundeck-constant check for
`get_1d_mesodiff` (same precedent as D29's `RADIUS`/`GRAV`, no dump needed for a pure constant
broadcast), explicit pole-uniformity checks, non-vacuous-computation checks.

## D48: GMKDIF/ISOSLOPE4/GMFEXP scoping -- QCROSS confirmed dead, GET_PSI_DIAG confirmed diagnostic

Read `OCNGM.f`'s `GMKDIF`, `ISOSLOPE4`, `GMFEXP`, `computeFluxes`, `wrapAdjustFluxes`, and
`addFluxes` in full -- the remaining piece of D46/D47's mesoscale-mixing family. Two real scope
reductions found, neither assumed:

1. **`GET_PSI_DIAG`** (called from `GMKDIF`) is **purely diagnostic** -- it only writes `OIJL`
   (bolus-velocity diagnostics) and local scratch, never read back by anything downstream. Skip
   entirely, same precedent as D31's `RESET_SURF_FLUXES`.
2. **`QCROSS` is always false for this rundeck's actual call.** `ocnmeso_drv` calls
   `gmkdif(k3d,1d0)` with `RGMI_in` hardcoded to `1d0` in the source (not a rundeck parameter),
   and `QCROSS = .NOT.(RGMI.eq.1d0)` inside `GMKDIF` -- eliminating every `IF(QCROSS)` branch in
   both `GMKDIF` (roughly half its coefficient-setting logic: the `DXZ`/`CDXZ`/`BXZ`/`CXZ`/
   `EXZ`/`CEXZ`/`DYZ`/`BYZ`/`CDYZ`/`CYZ`/`CEYZ`/`EYZ` cross-term arrays) and `GMFEXP` (the
   `FXZ`/`FYZ` off-diagonal flux terms) as dead code.

`GIJL` updates throughout (`GMFEXP`/`wrapAdjustFluxes`/`addFluxes`) confirmed diagnostic-only,
same skip. Revised live-code estimate for the actual port: `GMKDIF` (~100, post-QCROSS) +
`ISOSLOPE4` (136, embarrassingly parallel per-cell) + `GMFEXP` (~120, post-QCROSS) +
`computeFluxes` (~90) + `wrapAdjustFluxes` (~110, salt/QLIMIT=true path, a global-sum-based
conservative flux limiter) + `addFluxes` (~145, enthalpy/QLIMIT=false path) ≈ 700 lines --
smaller than D46's first estimate once dead code is excluded. No port code written this delta.

## D49: ISOSLOPE4 -- the first Gent-McWilliams port, caught a real pole-row loop-bound bug

Ported `ISOSLOPE4` (`OCNGM.f:997-1130`, `QCROSS` branches excluded per D48): the isopycnal-slope-
derived diffusion coefficients (`AIX0-3`/`AIY0-3`/`ASX0-3`/`ASY0-3`/`S2X0-3`/`S2Y0-3`, 24 output
arrays), embarrassingly parallel per-cell with no sequential dependency -- unlike almost
everything else ported in Stage 2. Real inputs are exactly D47's `densgrad` outputs
(`RHOX`/`RHOY`/`RHOMZ`/`BYRHOZ`/`BYDH`/`DZV`) plus `K3D` (D47's `get_1d_mesodiff`, a
constant-800 broadcast) -- no new input instrumentation needed; `ffdump_isoslope4` records only
`ISOSLOPE4`'s own 24 outputs, reusing D47's existing `ffz_densgrad` dump as ground-truth input.

**One real bug, caught on the first validation run.** 8 of the 24 output arrays (`AIX0-3`/
`AIY0-3`) failed with suspiciously round differences (~1000, ~667) while the other 16
(`ASX0-3`/`ASY0-3`/`S2X0-3`/`S2Y0-3`) matched exactly. Traced to a wrong assumption about the
main loop's J range: unlike almost every other routine ported in Stage 2 (which restrict to
J=2..JM-1, excluding the North Pole row), `ISOSLOPE4`'s real loop genuinely includes J=JM. The
first draft assumed the usual pole-exclusion convention, silently leaving the pole row at its
zero-initialized value -- this matched by coincidence at cells where `RHOX`/`RHOY` also happened
to be zero there (zeroing the `AS`/`S2` products regardless), while missing the real nonzero
`AI` values that don't depend on `RHOX`/`RHOY` being nonzero. Fixed by extending the loop to
J=2..JM; bitwise-exact on all 24 fields, all 3 dates, after the fix.

`fullfidelity/gmredi_ff.py`/`gmredi_compare.py` (new module for the Gent-McWilliams family, first
piece). `tests/test_gmredi_ff.py` (9 tests): real-record validation, a dedicated regression pin
for the pole-row loop-bound bug, non-vacuous slope-limiting checks. `GMKDIF`'s remaining
coefficient-setting logic (post-`QCROSS`) and `GMFEXP`+its three flux helpers (~560 lines) remain
the next pieces of this family.

## D50: GMKDIF's remaining coefficients -- bitwise-exact first try

Ported `GMKDIF`'s remaining (post-`QCROSS`, D48) coefficient-setting logic
(`OCNGM.f:194-319`): `BXX`/`BYY`/`BZZ` and the `AZX`/`BZX`/`CZX`/`AEZX`/`EZX`/`CEZX`/`AZY`/`BZY`/
`CZY`/`AEZY`/`EZY`/`CEZY` Z-direction flux coefficients, gated by `L>KPL(I,J)` (GM excluded
within the mixed layer). Real inputs are exactly D49's already-validated `ISOSLOPE4` outputs plus
`KPL` (the mixed-layer-depth index, `OCEAN_COM.f`, set by `OCNKPP.f`'s `OCONV` -- not yet ported,
recorded directly, the established "record what's not yet ported" pattern). No new physics
instrumentation needed beyond `KPL` itself and `GMKDIF`'s own 15 output arrays.

Applied D49's finding directly rather than rediscovering it: the main loop's J range matches
`ISOSLOPE4`'s (J=2..JM, including the North Pole row). The real source's separate
"J=J_1STG+1" extension block (handling a domain-decomposition halo boundary) was identified as a
pure MPI-parallel artifact that never fires for this rundeck's serial execution (confirmed by
reasoning: J_STOP_STGR already equals JM in serial, making the block's own `if(J.lt.JM)` guard
always false) -- not ported, consistent with this project's precedent for skipping inactive
domain-decomposition-only code paths (verified correct by the bitwise-exact result, not just
assumed).

Also noted the real Fortran's intentional write-target offsets (`BXX` written at `(IM1,J,L)`, the
WEST neighbor of the loop's own `I`; `BYY` at `(I,J-1,L)`) and preserved them exactly rather than
"fixing" what looks like an off-by-one at first glance.

`fullfidelity/gmredi_ff.py` (extended)/`gmkdif_compare.py`: bitwise-exact on all 15 output
arrays, all 3 dates, first try -- no bugs found. `tests/test_gmkdif_ff.py` (12 tests): real-record
validation, a `KPL`-plausibility sanity check, a write-offset regression pin, and a
mixed-layer-exclusion non-vacuousness check. `GMFEXP`+its three flux helpers
(`computeFluxes`/`wrapAdjustFluxes`/`addFluxes`, ~465 lines) -- the actual flux application to
G0M/S0M -- remain the last piece of this family.

## D51: GMFEXP + computeFluxes + wrapAdjustFluxes + addFluxes -- Gent-McWilliams family closed

Ported the last piece of the mesoscale-mixing family: `GMFEXP` (`OCNGM.f:330-499`,
`QCROSS`-excluded per D48) and its three helpers `computeFluxes`/`wrapAdjustFluxes`/`addFluxes`
(496-995) -- the actual Gent-McWilliams skew-flux application to `G0M` (`QLIMIT=.FALSE.`) and
`S0M` (`QLIMIT=.TRUE.`), called twice per `OCEANS` invocation. **Bitwise-exact on all 4 checked
fields (`TRM`/`TXM`/`TYM`/`TZM`), both calls, all 3 dates, first try -- no bugs found**, the
largest delta this family and the first fully-correct-on-first-try delta of this size this
session.

Sidestepped a real ambiguity found while tracing the source: `MO` (the mass field `GMFEXP` reads)
could not be conclusively traced back to D45's `OADVT2`/`MO1` output from source alone (no
explicit `MO = MO1` sync was found between the two). Rather than guess, recorded `MO` directly as
a real input at the point `GMFEXP` actually reads it -- the established "when in doubt, record
it" discipline, and the bitwise-exact result confirms this was the right call regardless of how
`MO`/`MO1` actually relate upstream.

Two things confirmed from a close source read, not assumed: (1) `GMFEXP`'s own main loop never
touches the pole rows (its J range is 2..JM-1, unlike `ISOSLOPE4`/`GMKDIF`'s pole-inclusive
range) -- pole-row `TXM`/`TYM` pass through completely unchanged, pinned by a dedicated test.
(2) The real Fortran's own `TZM`-update code is commented out in full inside `computeFluxes` --
`TZM` is never actually updated by `GMFEXP` despite being an `INTENT(INOUT)` argument; the port
deliberately never touches it either, also pinned by a dedicated regression test.

`fullfidelity/gmredi_ff.py` (extended with `gmfexp` + three internal helpers matching
`computeFluxes`/`wrapAdjustFluxes`/`addFluxes`)/`gmfexp_compare.py`. `tests/test_gmfexp_ff.py`
(15 tests): both-calls real-record validation, a `QLIMIT` structural check, the `TZM`-untouched
and pole-`TXM`/`TYM`-untouched regression pins, non-vacuous-flux checks. `GIJL` diagnostic
accumulations not ported (same precedent as every OIJL/GIJL update throughout Stage 2).

**This closes the Gent-McWilliams mesoscale-mixing family opened in D46** (scoping) through D47
(prerequisites), D48 (scoping), D49 (`ISOSLOPE4`), D50 (`GMKDIF`'s remaining coefficients), and
now D51 (`GMFEXP`+helpers) -- `OCNMESO_DRV.f`+`OCNGM.f`'s entire real per-step live path (the
"skew-GM" branch) is now ported and validated; `OCNTDMIX.f` (2,030 lines) and `GET_PSI_DIAG`
confirmed dead/diagnostic, do not port.

## D52: OCNKPP.f scoping -- KPP vertical-mixing scheme sized and subroutine boundaries corrected

No port code this delta. Read all of `OCNKPP.f` (3,714 lines) end to end and corrected D41's
original (coarse) subroutine boundaries: `KPPMIX` is actually 225-836 (612 lines), not 225-1315
as D41 had it -- D41's range accidentally lumped in six separate subroutines nested after it
(`bldepth`, `wscale`, `ddmix`, `kmixinit`, `swfrac`, `z121`) plus `KVINIT`. Full corrected
subroutine list with line counts: `KPPMIX`(612), `bldepth`(225), `wscale`(64), `ddmix`(52),
`kmixinit`(80), `swfrac`(34), `z121`(23), `KVINIT`(46), `OCONV`(1,526), `get_kvtdiss`(145),
`STCONV`(378), `OVDIFF`(60), `OVDIFFS`(47), `REDUCE_FIG`(14), `alloc_kpp_com`(65),
`get_gradients0`(119).

**Confirmed entirely dead (four whole subroutines, 362 lines), each checked directly rather than
assumed:**
- `get_kvtdiss` (145 lines) -- its one call site is gated by `if(use_tdiss==1)`; `use_tdiss`
  defaults to 0 (`OCNKPP.f:23`) and `ocean_use_tdiss` does not appear anywhere in
  `decks/P2SAoM40.R` (grepped directly), so it's never overridden.
- `get_gradients0` (119 lines) -- its only 3 call sites in this file sit inside the `#ifdef
  OCN_GISS_SM` dead block (`OCN_GISS_SM` confirmed not `#define`d, same as D41's finding for this
  file); grepped the whole model tree and found no other caller anywhere.
- `wscale` (64 lines) and `swfrac` (34 lines) -- every call site to either, anywhere in
  `OCNKPP.f`, is commented out with an explicit `! inlined` note; their logic was hand-inlined
  directly into `KPPMIX`/`bldepth` instead of being called. (The live `wscale` calls found by a
  cross-file grep are a same-named but unrelated subroutine in `mxkprf.f`, the atmosphere's PBL
  mixing scheme -- a false-friend name collision, not a live caller of this file's `wscale`.)
- `ddmix` (52 lines) -- its one call site is `if (LDD) call ddmix(...)`, and
  `LOGICAL, PARAMETER :: LDD = .false.` (`OCNKPP.f`, active branch since `OCN_GISS_TURB` is not
  `#define`d) -- a compile-time constant, not a runtime check, so this call can never fire.

**Confirmed dead in part (embedded branches within otherwise-live subroutines):**
- Every `if(use_qus==1)` branch in `OCONV` (the `GXXML`/`GYYML`/`GXYML`/`SXXML`/`SYYML`/`SXYML`
  quadratic-moment diffusion block, ~30 lines) -- `INTEGER :: USE_QUS=0` is the declared default
  (`OCEAN_COM.f:52`) and `ocean_use_qus` does not appear in `decks/P2SAoM40.R`.
  `GXML`/`GYML`/`SXML`/`SYML` (the *linear* moments, ungated) remain live.
  Note: `use_qus` gates similar blocks throughout `OCNDYN.f`/`OCNDYN2.f`/`OCNMESO_DRV.f`/
  `OCN_TRACER.f` too -- this finding generalizes beyond just this file, flagged for whichever
  future delta touches those.
- Every `#ifdef TRACERS_OCEAN` block in `OCONV` (the `TRML`/`TXML`/`TYML`/`TXXML`/`TYYML`/`TXYML`
  tracer-diffusion and `FLT3D` blocks, ~50 lines combined) -- `TRACERS_OCEAN` confirmed not
  `#define`d for this rundeck since D45.
- The `#ifdef OCN_GISS_SM` `DTP4G3D`/`DTP4S3D` lookups inside `OCONV`'s moment-diffusion loop
  (~10 lines) -- same dead guard as `get_gradients0` above.

**Confirmed live, call graph traced precisely (not assumed from name/position):**
- `OCONV` (1,526 lines) -- the main-grid driver, called unconditionally from `OCNDYN2.f:153`
  ("Apply ocean vertical mixing"), once per `OCEANS` call. Calls `KPPMIX` once per column,
  `OVDIFF` for momentum (`UL`/`ULD`), and `OVDIFFS` extensively for `G0ML`/`S0ML` plus the linear
  moments `GXML`/`GYML`/`SXML`/`SYML` -- found via a case-sensitive grep miss in this delta's
  first pass (the real calls are written `Call OVDIFFS`, mixed-case, which an earlier
  `CALL OVDIFFS|call ovdiffs` grep silently skipped -- corrected by re-grepping case-insensitive).
- `KPPMIX` (612 lines) -- the boundary-layer diffusivity-profile computation itself, called once
  from `OCONV`. Internally calls only `z121` (unconditionally) and `ddmix` (dead, see above) --
  `bldepth`/`wscale`/`swfrac` are NOT called from within `KPPMIX` despite being positioned
  immediately after it in the file; their logic is inlined directly into `KPPMIX`'s own body.
- `bldepth` (225 lines) -- boundary-layer-depth computation, called separately from `OCONV`
  (`OCNKPP.f:2107`, before `KPPMIX`) and from `STCONV` (`OCNKPP.f:3257`) -- a sibling of
  `KPPMIX`, not nested inside it, confirming `HBL` is computed as a precursor step and fed into
  `KPPMIX` as an input.
- `z121` (23 lines) -- a small smoothing filter, called unconditionally from within `KPPMIX`.
- `KVINIT` (46 lines) -- "save surface variables before any fluxes are added," called every step
  from `OCNDYN.f`'s `PRECIP_OC` (gated by `ogrid%have_domain`, always true in serial), not yet
  read in detail.
- `kmixinit` (80 lines) -- called exactly once, at model startup from `init_OCEAN`
  (`OCNDYN.f:796`), not per-timestep. Same treatment as this project's other init-only routines
  (D29's `RADIUS`/`GRAV`, D50's `KPL` default): its output can be recorded/treated as a fixed
  known input rather than ported as per-step physics, once its output arrays are identified.
- `OVDIFF`(60)/`OVDIFFS`(47) -- generic tri-diagonal vertical-diffusion solvers, called
  extensively by both `OCONV` and `STCONV`.
- `REDUCE_FIG`(14) -- small moment-reduction helper, called 4x from `OCONV`
  (`GXMO`/`GYMO`/`SXMO`/`SYMO`).
- `alloc_kpp_com`(65) -- allocation-only boilerplate (called once from `init_OCEAN`), not real
  physics; same non-ported-infrastructure treatment as other `alloc_*` routines throughout this
  project.
- `STCONV`(378) -- the straits-specific analog of `OCONV`, confirmed live (`OCNDYN2.f:487`, gated
  by `IF(AM_I_ROOT())` and `IF(NO.EQ.NOCEAN)`, both always true in serial/`NOCEAN=1`) but reads
  `OSTRAITS.f`'s `STRAITS` module data (`must,mmst,g0mst,gzmst,...`) which has not been scoped at
  all yet -- a separate, currently-unscoped dependency, not included in this delta's live-scope
  count.

**Revised live-scope estimate** (main grid only, excluding `STCONV`'s 378 lines pending
`OSTRAITS.f` scoping): `OCONV`(1,526) + `KPPMIX`(612) + `bldepth`(225) + `KVINIT`(46) +
`z121`(23) + `OVDIFF`(60) + `OVDIFFS`(47) + `REDUCE_FIG`(14) = 2,553 lines, minus ~90 lines of
embedded dead branches found above (`use_qus`/`TRACERS_OCEAN`/`OCN_GISS_SM` inside `OCONV`) ≈
**~2,460 lines of genuine main-grid physics**, plus `kmixinit`'s 80-line one-time init (treated
as a fixed input once its output is identified, not per-step-ported) and `alloc_kpp_com`'s 65
lines of non-ported allocation boilerplate. This makes the KPP vertical-mixing scheme roughly
3.5x the size of the entire Gent-McWilliams family that D46-D51 just closed (~700 lines) --
confirmed as the single largest remaining physics item in Stage 2's ocean core.

No `FULL_FIDELITY_PLAN.md` row previously had this precision; updated with the corrected
subroutine boundaries and dead/live findings above. Next delta should start on the actual port,
most naturally `bldepth` first (self-contained, feeds `KPPMIX`), or continue scoping
`OSTRAITS.f`/`OSTRAITS_COM.f` (entirely unread) to unblock `STCONV`.

## D53: OCNKPP.f correction -- bldepth is dead, not live; KPPMIX inlines its own boundary-layer-depth logic

No port code. While reading `KPPMIX`'s body in full to prepare for the actual port, found that
D52's claim "`bldepth` is a live sibling of `KPPMIX`, called separately by `OCONV`/`STCONV`" was
**wrong** -- a real correction, caught before any port code was written against it.

Both of `bldepth`'s real call sites (`OCONV` at `OCNKPP.f:~2104`, `STCONV` at `~3254`) sit inside
an `#ifdef OCN_GISS_TURB` ... `#else` ... `#endif` block:
```
#ifndef OCN_GISS_TURB
      CALL KPPMIX(...)
#else
      call bldepth(...)
#endif
```
`OCN_GISS_TURB` is confirmed not `#define`d for this build (same finding as D41/D46), so the live
branch at **both** call sites is `KPPMIX` directly -- `bldepth` (225 lines) is entirely dead. D52
mis-scoped this because its call-graph grep (a plain `grep -n "call bldepth"`) found the two real
call sites and reasonably assumed a routine called unconditionally from two places was live,
without checking the surrounding `#ifdef` context the way D48's `QCROSS` finding or this same
delta's `use_qus`/`TRACERS_OCEAN` findings did. Re-reading `KPPMIX`'s body explains why: `KPPMIX`
contains a full commented-out `c       call bldepth (...)` at its own line 390, with the entire
boundary-layer-depth algorithm (the bulk-Richardson-number search loop, the `wmt`/`wst`
lookup-table velocity-scale interpolation, the `swfrac` shortwave-fraction inlining) copied
inline into `KPPMIX` itself immediately after -- `bldepth` as a separate callable subroutine was
superseded by this inlining and is now dead weight in the source. The same pattern repeats for
`blmix` (never a separate live subroutine at all in this file -- its logic was always inline
within `KPPMIX`, `OCNKPP.f:580-830`) and confirms `wscale`/`swfrac`'s D52 dead-finding from the
other direction: their logic lives inline in `KPPMIX`/`bldepth` rather than being called out to.

Read all 612 lines of `KPPMIX` in full to confirm this and to prepare for the next delta's actual
port. One more small input-level finding: `KPPMIX`'s `Coriol` argument (Coriolis parameter) is
never actually used in the live path -- it only appears inside a fully commented-out
`hekman`/`hmonob` block explicitly marked "NOT USED" in the source (`OCNKPP.f:550-561`) -- a dead
input, not affecting any output. `alphaDT`/`betaDS` are likewise live *arguments* but only
consumed inside `if (LDD) call ddmix(...)`, and `LDD` is the file's `.false.` compile-time
constant (D52's finding) -- so their actual values passed in from `OCONV`/`STCONV` don't affect
`KPPMIX`'s output either, though they still need to be threaded through as (unused) parameters
for a faithful port signature.

Corrected `FULL_FIDELITY_PLAN.md`'s `OCNKPP.f` row: `bldepth` moved from "live" to "dead" (D52's
scope estimate already excluded `STCONV`'s 378 lines but had *included* `bldepth`'s 225 as live
main-grid scope -- revised main-grid estimate is now **~2,235 lines**, not ~2,460). The real
live core reduces to essentially just `KPPMIX`(612, self-contained, inlines its own boundary-
layer-depth and mixing-coefficient logic) + `z121`(23, the only subroutine `KPPMIX` actually
calls) + `OCONV`(1,526) + `KVINIT`(46) + `OVDIFF`(60) + `OVDIFFS`(47) + `REDUCE_FIG`(14), minus
the already-found embedded-dead `use_qus`/`TRACERS_OCEAN` branches inside `OCONV`.

Next delta starts the actual port: `KPPMIX`+`z121`, the clean self-contained numerical core, fed
by `kmixinit`'s one-time (model-init, not per-step) `wmt`/`wst` lookup tables and `FZ500`
array -- both pure closed-form functions of fixed physical constants and the fixed vertical grid
`ZE`, ported directly rather than dumped, consistent with this project's precedent for
configuration-fixed setup data (D29's `RADIUS`/`GRAV`, D47's `MESO_DIFFUSIVITY_CONST`).

## D54: KPPMIX + z121 + kmixinit + init_solar ported -- first piece of the KPP scheme, and the
first delta in this project not validated bit-for-bit (with a fully diagnosed reason)

Ported `KPPMIX` (`OCNKPP.f:225-836`, confirmed the live routine for this build by D52/D53's
correction) and its one real dependency, `z121` (the 1-2-1 vertical smoothing filter). Also
ported `kmixinit` (`OCNKPP.f:1178-1256`) and `SW2OCEAN`'s `init_solar` (`OCEAN_COM.f:321-349`)
directly, not from a dump: both are pure closed-form functions of fixed physical constants and
the fixed vertical grid `ZE`, computed once at model startup -- the same precedent as D29's
`RADIUS`/`GRAV` and D47's `MESO_DIFFUSIVITY_CONST`.

`KPPMIX` is called from `OCONV`'s per-column loop inside a fixed-point iteration on `HBL` (up to
`ITER=4` times per column per `OCEANS` call, `OCNKPP.f:1978-2334` -- `G0ML`/`S0ML`/`UL`/`ULD` get
re-diffused between iterations using the previous iteration's coefficients). Rather than port
that outer iteration (a future OCONV delta's job), this delta records `KPPMIX`'s actual real
inputs and outputs at every real call (`ffdump_kppmix`, a new per-call streaming dump -- unlike
every other Stage-2 dump, one record per real Fortran call rather than one record per itime,
since a single itime makes this call up to ~4x per ocean column). 76,011 real calls captured
across the standard 3-dates-x-6-steps sweep.

**This is the first delta in the entire project not validated bit-for-bit, and the reason is
fully diagnosed, not a shrug:** `kmixinit`'s `wmt`/`wst` velocity-scale lookup tables use
`**(1./3.)` in two branches, and OCNKPP.f writes the exponent as `1./3.` with **no `d0` suffix on
either literal** -- meaning Fortran parses `1.` and `3.` as single-precision `REAL` and computes
the division in single precision *before* promoting the ~7-digit result to double and using it as
the exponent. This is different from, and less precise than, the double-precision `1.0d0/3.0d0`
a literal reading of "one third" would suggest, and it genuinely changes the answer once raised to
a REAL*8 base. Found and confirmed by writing a tiny standalone `ifort -fp-model strict` program
that reproduced the real dumped `cg` constant bit-for-bit only once the port's exponent was
changed from `1.0/3.0` (double) to `float64(float32(1.0)/float32(3.0))` (single-precision
division, then promoted) -- `kmixinit`'s `Vtc` constant had a second, independent instance of the
same bug class: `Vtc = concv * sqrt(0.2/concs/epsilon) / vonk**2 / Ricr` writes `0.2` with no
`d0` suffix, so it too is parsed as single precision before promotion; fixing both made `cg` and
`Vtc` match the real dumped values exactly (`_ONE_THIRD_SP` and the corrected literal handling in
`kppmix_ff.py`).

Even after both single-precision-literal fixes, the `wmt`/`wst` tables are **not quite**
bit-identical to a real one-time dump of the whole table (added temporarily via a DEBUG-ONLY
`ffdump_kmixinit_table` hook to settle this empirically rather than guess, then left in place
behind no `FFD_START` gate since `kmixinit` only ever runs once): of 429,944 cells, 62 (`wmt`) and
38 (`wst`) differ from the real values, every one by exactly 1 ULP (e.g. `...469729` vs
`...469728`). Traced one such cell by hand: both a `pow`-based and a mathematically-equivalent
`exp(y*log(x))`-based Python recomputation give the *same* 1-ULP-off answer as the mismatch,
confirming this is not a translation choice that can be fixed -- IEEE 754 requires `+`,`-`,`*`,
`/`,`sqrt` to be correctly rounded, but explicitly does **not** require `pow`/`exp`/`log` to be,
so two independently-correct libm implementations (glibc's, used by numpy/Python here, vs Intel's
libimf, used by ifort even under `-fp-model strict`) can legitimately disagree in the last bit for
the same bit-identical inputs. This affects ~0.023% of the table's cells.

That table-level 1-ULP noise then propagates through the bilinear interpolation and the HBL
bulk-Richardson search (which touches the table many times per call) to a per-call residual: **97
percentile error stays below 3e-10; the worst of all 76,011 real calls is ~5e-6** (in `GHAT`,
which divides by a velocity scale and so amplifies table noise the most); **`KBL` (the integer
boundary-layer-index output) never mismatches, once, across all 76,011 calls** -- the physically
and structurally meaningful part of the output is completely unaffected. Validated at this
suite's standard `atol=1e-6` tolerance (not a special-cased loosened one) in
`tests/test_kppmix_ff.py` (25 tests, including `test_akvs_and_akvg_are_identical`, a regression
pin for the real Fortran's `dift[ki]=difs[ki]` -- with LDD always false there's no distinct
tracer-diffusivity path, so `AKVS`/`AKVG` are structurally identical for this build, confirmed
against a real record not just asserted from source).

`fullfidelity/kppmix_ff.py` (new): `kmixinit`, `init_solar`, `z121`, `_wscale`/`_bfsfc_search`/
`_bfsfc_at_hbl` (KPPMIX's three repeated inline lookup patterns, factored into shared helpers
rather than duplicated four/two times as the real Fortran does), `kppmix`. `LDD`/`alphaDT`/
`betaDS`/`Coriol` (all real Fortran inputs, all structurally dead per D52/D53) omitted from the
port's signature entirely rather than threaded through unused. `fullfidelity/kppmix_compare.py`
(new): a per-call stream loader, the first of its kind in this project (every prior dump loader
assumes one record per itime).

Instrumentation: `OCNKPP_kppmix.f.patch` (one line added right after the real `CALL KPPMIX` in
`OCONV`, passing `I`,`J`,`ITER` plus everything already in scope -- no new upstream computation
needed), `ATM_DRV_kppmix.f.patch` (`ffdump_kppmix`, unit 1004, gated by `FFD_START`/`FFD_NSTEP`
same as every other dump; plus the temporary DEBUG-ONLY `ffdump_kmixinit_table`, unit 1005,
ungated, used only to diagnose the ULP question above -- kept in the patch file for
reproducibility but not load-bearing for the delta's own validation).

## D55: OVDIFFS + plain TRIDIAG ported -- bitwise-exact on every real call

Ported `OVDIFFS` (`OCNKPP.f:3470-3516`, the implicit tracer vertical-diffusion solver with
non-local transport) and the plain (non-`OPTIMIZED_TRIDIAG`) `TRIDIAG` it calls
(`solvers/TRIDIAG.f`; the rundeck does not define `OPTIMIZED_TRIDIAG`). Recorded at `OCONV`'s live
G0ML and S0ML call sites (per call, per column, per HBL iteration), 152,022 real calls across the
3-dates-x-6-steps sweep. **Bitwise-exact on all calls, both `U` and `FL`, first try.**

Two things checked against real data rather than assumed from the source:
- `DTP4G`/`DTP4S` (the `DTP4` argument) are structurally zero for this build: their only setter is
  inside `#ifdef OCN_GISS_SM` (dead). The source's `DTP4(LMIJ-1)` index in the bottom RHS
  (`OCNKPP.f:3511`, a quirk) therefore cannot affect this build's output. Pinned by
  `test_dtp4_is_zero_in_real_records`, not left as an untested claim.
- The momentum `OVDIFF` (60 lines, same structure) is not yet ported -- its own call sites in
  `OCONV`'s UL/ULD loop are the next step.

`fullfidelity/ovdiffs_ff.py` (new), `ovdiffs_compare.py` (new, per-call loader),
`tests/test_ovdiffs_ff.py` (new, 20 tests). Instrumentation: `OCNKPP_ovdiffs.f.patch`,
`ATM_DRV_ovdiffs.f.patch` (unit 1006, `ffz_ovdiffs_<itime>.bin`). **Both patch files were
reconstructed from the inserted text, not generated by `diff`**: the scratch `mE2` tree was
removed before regeneration; the patch headers say so.

## D56: momentum OVDIFF ported -- bitwise-exact on all 420,444 real calls

Ported the momentum `OVDIFF` (`OCNKPP.f:3410-3469`, velocity vertical diffusion) that `OCONV`
calls for the U-point (UL, tag 2: 281,808 calls) and D-grid (ULD, tag 3: 138,636 calls)
velocity components. Structurally different from D55's `OVDIFFS`: off-diagonals use `DTBYDZ(L)`,
the bottom RHS reads `DTP4(LMIJ)`, and there is no separate DT argument. Built on the same
plain `TRIDIAG` (`ovdiffs_ff.py`). **Bitwise-exact, first try, all calls, both components.**

Rebuilt the instrumented tree fresh from pristine source with only the two new dump hooks
(read-only, so they don't change the physics), since the earlier scratch tree was removed. Two
inserted fixed-form lines ran past column 72 on the first build attempt; rewrapped, rebuilt, rerun.

`fullfidelity/ovdiff_ff.py` (new), `tests/test_ovdiff_ff.py` (new, 19 tests, including a regression
pin that both U-point and D-grid calls are exercised). Instrumentation patches
`OCNKPP_ovdiff_momentum.f.patch` and `ATM_DRV_ovdiff_momentum.f.patch` are **reconstructed from the
inserted text, not generated by diff** (the scratch tree was removed; headers say so).

## D56: momentum OVDIFF ported -- bitwise-exact on all 420,444 real calls

Ported the momentum `OVDIFF` (`OCNKPP.f:3410-3469`), the velocity counterpart of D55's
`OVDIFFS`, from `OCONV`'s U-point (`UL`, tag 2, 281,808 calls) and D-grid (`ULD`, tag 3,
138,636 calls) momentum loops. Bitwise-exact on every call across the 3-dates x 6-steps sweep.

Structural differences from `OVDIFFS` were ported verbatim, not assumed equal: off-diagonal
coefficients use `DTBYDZ(L)` (not `L-1`), the bottom RHS reads `DTP4(LMIJ)` (not `LMIJ-1`), and
there is no separate `DT` argument. Built on the same plain `TRIDIAG` as D55.

`fullfidelity/ovdiff_ff.py` (new), `tests/test_ovdiff_ff.py` (new, 19 tests). Instrumentation:
`OCNKPP_ovdiff_momentum.f.patch` and `ATM_DRV_ovdiff_momentum.f.patch` (unit 1007). **Both patch
files were written from the inserted text, not generated by `diff`**: the scratch tree was rebuilt
from pristine source, and the headers say so. The dump build was a minimal one (pristine source
plus these hooks), not the full patch chain, since read-only hooks don't alter the recorded values.

## D57: REDUCE_FIG ported -- bitwise-exact on all 1,750,248 real calls

Ported `REDUCE_FIG` (`OCNKPP.f:3517-3530`), the significant-figure reduction applied to `OCONV`'s
live main-grid moments `GXMO`/`GYMO`/`SXMO`/`SYMO` (`OCNKPP.f:2843-2847`). The tracer-moment
calls at `OCNKPP.f:2860` sit inside the dead `TRACERS_OCEAN` block and are not ported.
Validated per call across the 3-dates x 6-steps sweep: bitwise-exact on every call, and
524,741 of those calls actually changed the value (the check is not vacuous).

Fortran `EXPONENT`, `SCALE` and `NINT` semantics are reproduced explicitly (`EXPONENT(0)=0`,
`NINT` rounds half away from zero). Tests (`tests/test_reduce_fig_ff.py`, 22 tests) cover both
guard branches -- the reduction branch and the unchanged high-exponent branch -- plus the
`EXPONENT` convention, with a small-value rounding case checked by hand.

`fullfidelity/reduce_fig_ff.py` (new), `tests/test_reduce_fig_ff.py` (new). Instrumentation:
`OCNKPP_reduce_fig.f.patch` and `ATM_DRV_reduce_fig.f.patch` (unit 1008). **Both patch files were
written from the inserted text, not generated by `diff`**; the headers say so.

## D58: KVINIT ported -- exact copy on all 198 real snapshot checks

Ported `KVINIT` (`OCNKPP.f:1315-1359`), the per-step pre-source surface snapshot that the KPP mixing
step reads. It is a pure copy with no arithmetic: `G0M1` (first LSRPD layers) and the first-layer
`S0M1`, `MO1`, `GXM1`, `GYM1`, `SXM1`, `SYM1`, `UO1`, `VO1`, `UOD1`, `VOD1` are copied from the
current fields. Validated by exact equality against 18 real per-step dumps (3 dates x 6 steps):
11 copies per step, 198 checks, zero mismatches. `KVINIT` runs inside `PRECIP_OC`, guarded by
`ogrid%have_domain`, which is true in serial execution.

`fullfidelity/kvinit_ff.py` (new), `tests/test_kvinit_ff.py` (new, 19 tests, including a
non-vacuity pin that the snapshot fields are nonzero). Instrumentation:
`OCNDYN_kvinit.f.patch` and `ATM_DRV_kvinit.f.patch` (unit 1009). **Both patch files were written
from the inserted text, not generated by `diff`**; the headers say so.

## D59: OCONV per-column setup block ported -- bitwise-exact on all 76,011 real KPPMIX inputs

Piece 1 of the `OCONV` driver port: the per-column setup that computes `KPPMIX`'s inputs
(`OCNKPP.f:~1928-2100`): the `ZSCALE`-rescaled `zgrid`/`hwide`/`byhwide`, the velocity shears
`Shsq`/`dVsq`, the density gradients `dbloc`/`Ritop`, and the surface forcing `Ustar`/`Bo`/`Bosol`.
Validated by exact equality against the real `KPPMIX` inputs recorded in D54, matched call by call
on `(i, j, iter, lmij)`: all 76,011 calls, every output, bitwise-exact.

Two real findings made this work, both caught by the data rather than assumed from the source:
- **`KMUV` depends on the row.** The shear sums run over `K=1..KMUV` with `KMUV=IM+2` at the North
  Pole (`OCNKPP.f:1714`) but `KMUV=4` on every other row (`OCNKPP.f:1807`). An early port used
  `IM+2` everywhere and mismatched on ~63k of 76k calls; an offline scan over the real record found
  the 4-point sum, and the source confirms it.
- **Operation order matters for `Bo`/`Bosol`.** Fortran's `- GRAV*BYRHO**2*(...)` multiplies
  `GRAV*BYRHO**2` first, then the parenthesized term; the port matches that grouping.

The seawater-EOS values (`VOLGSP`, `VOLGS`, `ALPHAGSP`, `BETAGSP`, `SHCGS`) are external-table
lookups (`OCNFUNTAB.f`), so they are recorded directly from the instrumented build rather than
ported, per this project's precedent (D35, D40, D47). `MLD`/`KMLD` are not outputs of `KPPMIX` and
are left to a later piece of `OCONV`.

**Ledger note:** an untracked `oconv_ff.py` (956 lines, a draft of the whole `OCONV` driver, with
an `ffz_oconv` instrumentation patch) sits in this working tree. Its provenance is unknown, and it
conflicts with the current tree (it hard-codes `LSRPD=1` where the real value is 3 and reuses unit
1006). It has not been adopted or validated here; the piece-by-piece port in `ocnsetup_ff.py` is
the validated path.

`fullfidelity/ocnsetup_ff.py` (new), `tests/test_ocnsetup_ff.py` (new, 20 tests). Instrumentation:
`OCNKPP_ocnsetup.f.patch` and `ATM_DRV_ocnsetup.f.patch` (unit 1010). **Both patch files were
written from the inserted text, not generated by `diff`.**

## D60: OCONV HBL-iteration scalar scaling ported -- k, GHATG and GHATS all bitwise-exact on every real call

Piece 2a of the `OCONV` driver port: the GHAT flux scaling (`OCNKPP.f:2250-2252`) and the density
rescaling of the diffusivities (`OCNKPP.f:2240-2270`, `KVTDISS=0`) that sit between `KPPMIX` and the
G/S `OVDIFFS` calls. Ported in `fullfidelity/hbl_glue_ff.py` (`scale_akv`, `ghat_scalar_fluxes`).

Validated against the real recorded pairs in `hbl_glue_compare.py`: every real `KPPMIX` call
(76,011 across the three dates) is followed by one G and one S `OVDIFFS` record. The `k` arrays
the Fortran passed in match the port's scaled `AKVG`/`AKVS` bitwise on every level. The G flux
(`GHATG`) matches bitwise. The S flux (`GHATS`) was first checked with `BYMML(1)` recovered from the
data (2.6e-13 relative, not bitwise). A follow-up hook records `BYMML(1)` directly (`ffz_bymml`,
unit 1011, one record per ITER at the S-OVDIFFS call). With that recorded value GHATS is bitwise-exact
on all 805,879 real levels across the three dates (`hbl_glue_bymml_compare.py`).

The production glue still takes `BYMML(1)` and `S0ML0(1)` as explicit inputs. Sourcing them from
the MML bookkeeping (`OCNKPP.f:1963-1983`) is piece 2b's job.

**Undetermined cases (recovery check only):** 40,559 calls have zero GHAT at the largest level, so
the recovery check cannot solve for BYMML there. The recorded S flux is zero in all of them, which
the formula also gives. The bitwise check with recorded BYMML covers them.

Instrumentation: `OCNKPP_bymml.f.patch` and `ATM_DRV_bymml.f.patch` (unit 1011). Unlike the D55-D59
patches, these were generated with `diff -u` against the pre-hook source, not reconstructed from
inserted text. The existing D54 `ffz_kppmix` and D55 `ffz_ovdiffs` dumps supply the rest.
`DXYPO(J)` comes from `odhorz_ff.geomo_dyn_arrays` (D36/D40).

## D61: OCONV mass bookkeeping + convergence test + post-loop flux save ported -- all bitwise-exact on real calls

Piece 2b, first part: the `MML(1)`/`BYMML(1)` mass that the GHATS flux and DELTAM terms use.
`OCNKPP.f:1813-1817` (non-pole) and `:1737-1743` (pole) set `MML = MO*DXYPO(J)` after the source
step (ITER>=2) and `MMLT = MO1*DXYPO(J)` before it (ITER=1), with `BYMML = 1/MML`. The `MML := MML0`
switch at ITER=2 is `OCNKPP.f:1981-1983`. Ported in `fullfidelity/oconv_mml_ff.py` as
`mass_bookkeeping(mo1_prev, mo_cur, dxypo_j, iter_)`.

Validated in `oconv_mml_compare.py` against the recorded `ffz_bymml` values (D60 follow-up dump):
- ITER>=2: 38,301 calls, MML and BYMML bitwise-exact. `MO(I,J,1)` is the setup record's exact `mo1`.
- ITER=1: 37,710 calls, bitwise-exact, but `MO1(I,J)` is not recorded. It is recovered from the
  recorded DELTAM as `MO - DELTAM*DTS` (DTS=1800). The check is therefore indirect, and is reported
  separately from the direct ITER>=2 check.

Only index 1 is live. `MML(L>1)` feeds the OCN_GISS_SM branch, which is dead for this build.
No post-loop code reads MML (grep over OCNKPP.f 2340-2700 finds none).

`S0ML0(1)` (= `S0M(I,J,1)`, `OCNKPP.f:1865`/`:1741`) remains an input: no dump records the S0M array.
Its value is recorded only as the S OVDIFFS `u0[1]`, which the GHATS check uses.

Instrumentation: `OCNKPP_post.f.patch` and `ATM_DRV_post.f.patch` (unit 1012, `ffz_post` record:
i, j, iter, lmij, S0M1(I,J), DM, FLG3D(0:13), FLS3D(0:13)), written after the iteration exits.
Both patches are `diff -u` output against the pre-hook source.

**Convergence and post-loop save (`oconv_save_ff.py`, validated by `oconv_save_compare.py`):**
- The decision to re-run the iteration (`OCNKPP.f:2337-2338`) reproduces the recorded call count on
  all 76,011 real KPPMIX calls (37,710 columns, every continue/stop decision consistent with the
  recorded HBL and KBL sequence).
- The post-loop save (`OCNKPP.f:2399-2405`) matches the `ffz_post` record bitwise on all 37,710
  columns: DM, FLG3D(0:LMIJ), FLS3D(0:LMIJ), including the `S0M1(I,J)` surface term.


## D62: OVDIFFS + TRIDIAG ported to batched JAX (speed-first, tolerance-checked)

Direction change from the user: JAX port and runtime speed take priority over bitwise verification.
Accuracy is checked to a stated tolerance instead (1e-9 relative, per column scale).

`fullfidelity/ovdiffs_jax.py`: batched OVDIFFS + TRIDIAG over N columns. Per-column active length
via masks (rows past lmij become identity), Thomas solve as two `lax.scan` sweeps, jitted.
`ovdiffs_jax_compare.py` runs it on the recorded D55 calls (ffz_ovdiffs, Nov-26 itime set):
50,564 calls. Worst relative error: u 3.1e-16, flux 6.3e-10 (flux is a difference of u values,
so it loses digits). Runtime about 0.2 s per 8,400 calls on CPU, including the first JIT.

Note: the earlier bitwise ports (D54-D61, numpy) stay as the reference; JAX ports are checked
against them, not against Fortran directly.

## D63: KPPMIX ported to batched JAX -- matches the numpy port to 2.7e-11 on all 76,011 real calls

`fullfidelity/kppmix_jax.py`: batched KPPMIX over N columns (one jit call per itime), restructured:
the `z121` smoothing is a `lax.scan` (sequential v[0] carry); the bulk-Richardson search is
vectorized over levels (`rib_ka` is the previous level's `rib_ku`, so there is no recurrence) with
argmax for the first crossing; `_wscale` computes both branches and merges with `where`. The
per-level boundary-layer loop and the kbl correction are masked vector ops.

Validation (`kppmix_jax_compare.py`, speed-first tolerance 1e-9): against the numpy port
(`kppmix_ff.kppmix`, D54) on every recorded call, 76,011 calls across the three dates:
- kbl: 0 mismatches.
- worst relative error: visc 2.5e-11, difs 2.7e-11, dift 2.7e-11, ghat 2.7e-11, hbl 6.6e-13.
- runtime: about 23 s for all 76,011 calls, including the first JIT compile.

Against the recorded Fortran values, the numpy port itself is off by up to 1.7e-7 (visc). That is the
existing D54 table residual (pow/exp rounding), not something the JAX port adds: the JAX port is
within 2.7e-11 of the numpy port. Accuracy against Fortran is therefore the D54 level, as the
speed-first policy accepts.

`kppmix_jax.py` loads the lookup tables (`kmixinit`, `init_solar`) from `kppmix_ff.py` unchanged.
Known limitation: `lsrpd` and the table shapes are fixed for this build (LMO=13, NNI=890, NNJ=480).

## D64: momentum OVDIFF and the OCONV setup block ported to batched JAX

Two more pieces of the JAX port, same policy as D62/D63 (speed first, tolerance-checked).

- `ovdiffs_jax.ovdiff_jax` (momentum OVDIFF, D56). The Thomas solver is factored out as `_thomas`,
  shared with `ovdiffs_jax`, so OVDIFFS results are unchanged (re-checked: worst 6e-10 on fl). The
  momentum coefficients (DTBYDZ off-diagonals, DTP4(LMIJ) bottom RHS) follow ovdiff_ff.py.
  `ovdiff_jax_compare.py`: 139,961 calls on the Nov-26 set; worst 5.9e-16 vs the numpy port and
  vs the Fortran record.
- `setup_jax.setup_jax` (the OCONV per-column setup, D59). Batched over columns with per-column
  lmij, kmuv (IM+2 at the pole, 4 elsewhere) and ZSCALE. The shear sums run over K under a mask.
  `setup_jax_compare.py`: all 25,282 calls on Nov-26 and Dec-01 (plus Jan-01 for the last check),
  every output within 3.5e-16 of the recorded KPPMIX inputs.

Not yet ported in JAX: the batched OCONV HBL loop itself (ITER glue, GHAT scaling, mass
bookkeeping, convergence, post-loop save). Each of its pieces now has a JAX version, but an
end-to-end check needs the full per-column state at loop entry (UL0, G0ML0, S0ML0, MO1, PO), which
no current dump records. That needs one more instrumentation pass.

## D65-D66: OCONV HBL-loop state dumps, straits confirmed active, batched JAX HBL loop validated

**Straits (confirmed active):** the instrumented reader prints `NMST = 12` in all 18 itime-steps of
the three dates (`OSTRAITS` present in the run directory). Straits are read and STCONV runs each
step. The port itself is not started (estimate 25-40 hours).

**Instrumentation:** `ffz_hblin` (ITER=1 entry state per column), `ffz_hblout` (exit state: UL, ULD,
G0ML, S0ML, HBL, KBL, iter), `ffz_momi` (momentum OVDIFF output with the column index I, ITER, K,
LMUV(K), UL and UL0 at the call), `ffz_ul0pt` (UL and UL0 at the setup call). Patches:
`OCNKPP_hblloop.f.patch`, `ATM_DRV_hblloop.f.patch`, `OSTRAITS_COM_nmst.f.patch` (all `diff -u`).
`ffdump_ovdiff` stores K in its `i` slot and does not record the column; `ffz_momi` is the
column-indexed replacement.

**Batched JAX HBL loop (`fullfidelity/ocnhbl_jax.py`):** ITER loop unrolled to 4 with per-column
masks, setup (`setup_jax`), KPPMIX (`kppmix_jax`), density rescale, GHAT, momentum OVDIFF and G/S
OVDIFFS (`ovdiff_jax`, `ovdiffs_jax`), convergence, D-grid pass, flux save. The seawater EOS per
ITER comes from the setup records (external table dependency, as in D59). Validated on all 18
itime-steps (`ocnhbl_jax_compare.py`, `ocnhbl_momentum_compare.py`):
- kbl: 0 mismatches on every column.
- worst relative error over all active levels: hbl 1.3e-7, ul 1e-8, g0ml 2e-10, s0ml 6e-13,
  flux saves (FLG3D/FLS3D) 1.5e-7.
- momentum records (ffz_momi, 7,776 calls per itime): worst 4.8e-9.

**Bugs found while validating (not in the Fortran):** a swapped return order in the entry-state
loader; the setup call reads the entry UL, not UL0 (UL0 is the momentum input); a fixed-form
line over 72 columns in the new dump routines (caught by the compiler).

## D67-D68: straits (STCONV) instrumented and ported to batched JAX -- exit state matches on all 18 real steps

**Scope (confirmed):** straits are active for P2SAoM40 (`NMST = 12`). `STCONV` (OCNKPP.f:3032-790)
runs once per model step from `OCNDYN2.f:487`. For each strait it builds two half-boxes (IQ = 1, 2),
runs the KPP diffusivities, momentum and G/S diffusion with an ITER loop (the same HBL convergence
rule as OCONV), applies the surface-free implicit G/S tendencies, and combines the half-boxes into
the strait's prognostic state (MUST, G0MST, GXMST, GZMST, S0MST, SXMST, SZMST).

**Instrumentation (D67):** `ffz_stin` (straits state at entry), `ffz_stout` (exit state), `ffz_stkpp`
(KPPMIX inputs and raw outputs per strait, half-box and ITER). Patches `OCNKPP_stconv.f.patch` and
`ATM_DRV_stconv.f.patch` (`diff -u`).

**Port (D68), `fullfidelity/stconv_jax.py`:** batched over the 24 half-boxes, ITER loop unrolled to 4
with masks, KPPMIX (`kppmix_jax`, D63), momentum OVDIFF (`ovdiff_jax`), G/S OVDIFFS (`ovdiffs_jax`),
REDUCE_FIG (vectorized), `EXPONENT` via `frexp`. Differences from OCONV that matter: no ZSCALE
(zgrid from ZE), plain shears (no RAVM), zero surface forcing, and the convergence test uses
`ZE(KBL)-ZE(KBL-1)`, not zgrid differences (a bug found and fixed in validation).
The EOS per ITER (BYRHO, DBLOC, DBSFC, RITOP) is passed in from the stkpp dump, as in D59/D66.

**Validation (`stconv_compare.py`), exit state vs real STCONV output:**
- Nov-26, Dec-01, Jan-01 (six steps each): worst relative error 3.8e-14 across all seven arrays.
- KPPMIX on the recorded straits inputs (numpy and JAX): exact (0.0).

**Bugs found in validation (not in the Fortran):** a wrong MMST index in BYDZ2 and the pressure
step; the convergence test using zgrid differences instead of ZE. Both fixed.

**Still open for straits:** `OSTRAITS.f` (1,219 lines, the namelist reader and the per-step straits
routines it provides) and the other straits calls on the step path (`STPGF`, `STADV`, `STBDRA` use the
straits state). Only STCONV is ported and validated so far.

## D69: straits bottom and side drag (STBDRA) ported to batched JAX -- exact on all 18 real steps

`STBDRA` (OSTRAITS.f:287-328) runs after STCONV each step (OCNDYN2.f:488): a bottom drag on the
lowest layer, a side drag on every layer, and a 20-day decay of the cross-strait gradients. Ported
in `fullfidelity/straits_jax.py` (`stbdra_jax`). Validated against the real post-drag state
(`ffz_stdrag`, instrumented in STBDRA: `OSTRAITS_stbdra.f.patch`, `ATM_DRV_stbdra.f.patch`): worst
relative error 0.0 on MUST, GXMST and SXMST across Nov-26, Dec-01 and Jan-01.

**Not yet validated:** `stadvt_jax` (the vectorized STADVT, OSTRAITS.f:171-285) is written in the same
module. Its validation needs the straits end-point arrays (MOE, G0ME, GXME, ...), which STADV reads and
updates and which are not dumped yet. `STPGF` (OSTRAITS.f:6-65) and the STADV loop (67-169) also
depend on those arrays. Next step for the straits port.

## D70-D73: the straits step ported to batched JAX and validated end to end (all three dates)

Scope: the straits (active, NMST = 12) are advanced once per step by four routines (OCNDYN2.f:483-488):
STPGF, STADV, STCONV, STBDRA. All four are now batched JAX ports, validated individually and chained.

- **STBDRA** (`straits_jax.stbdra_jax`, D69): exact (0.0) on all 18 steps.
- **STADV** (`straits_jax.stadv_jax` / `stadv_seq`, D70-D71): exact (0.0) on all 18 steps, with the
  shared-cell copy (KN2) applied in order; 4 of the 12 straits share end cells in this build.
  Namelist geometry (IST, JST, XST, YST) from `parse_straits_nml`.
- **STPGF** (`straits_jax.stpgf_jax`, D72): worst 4.3e-13 on all 18 steps. Uses the seawater EOS
  from the OFTAB table (`eos_jax.py`, D72): the table is the first record of OFTABLE_NEW (big-endian,
  80-byte title then VGSP). VOLGSP reproduces the recorded setup-record densities on 109,566 calls,
  all but 33 to 1e-9 (the 33 differ by about 5e-5, unexplained; noted).
- **STCONV** (`stconv_jax.stconv_jax`, D68): now computes its EOS from OFTAB (no recorded values):
  worst 4e-14 on all 18 steps.
- **Chain** (`straits_step_jax.straits_step`, `straits_step_compare.py`, D73): entry state and
  end-point arrays through STPGF->STADV->STCONV->STBDRA, compared with the recorded post-drag
  state and end-point arrays: worst 1.2e-12 on all 18 steps.

Instrumentation: `ffz_stin/stout` (D67), `ffz_stdrag` (D69), `ffz_me_in/out`, `ffz_stadv_in/out`,
`ffz_pgf_in/out` (D70-D71). Patches in `fullfidelity/instrumentation/` (`diff -u`).

Not done in the straits port: `init_STRAITS` (the one-time initial end-point state, computed from the
ocean grid at model start; the port takes the recorded initial state as input); the neighbour copy
is validated only for this build's shared end cells.

## D74-D75: OPFIL2 (polar zonal smoother) ported and validated on all three dates

OPFIL2 (OCNDYN2.f, `opfil2`) is live on the ocean step path at the polar rows, applied to the
west-east velocity (`usmooth`) and to the pressure-gradient term (`pgfx`). Two parts:
- **Application** (`fullfidelity/opfil2_ff.py`, validated): the FFT smoother on the FFT segments
  (OFFT/OFFTI, a real 72-point transform; the inverse is the exact inverse of the forward map) and
  the per-basin matrix filter with its running index into REDUCO, including wrap-around basins.
  Validated on 780 real calls per date (`opfil2_compare.py`): worst relative error 3.0e-15 (Nov-26),
  2.9e-15 (Dec-01), 2.3e-15 (Jan-01).
- **Setup** (`calc_opfil2_coeffs`, ~180 lines of basin geometry): NOT ported. Its outputs (SMOOTH,
  NMIN, REDUCO and the segment tables) are recorded once per run to `ffz_opcoef.bin` (D75) and
  read as inputs. This is a cut corner, the same precedent as the EOS.

Instrumentation: `ffz_opfil_in/out` (D74, per call: l, jmin, jmax, X(72,46)) and `ffz_opcoef.bin`
(D75). Patches `OCNDYN2_opfil2.f.patch`, `ATM_DRV_opfil2.f.patch` (`diff -u`). Two build errors
were fixed on the way: a line over 72 columns, and the output dump placed after a RETURN.

## D76: OCONV HBL loop computes its seawater EOS from the OFTAB table (no recorded densities)

`ocnhbl_jax.hbl_loop` now computes BYRHO, RHOM and RHO1 from G, S and PO with the table
(`eos_jax.volgsp`), instead of reading them from the setup dump. PO is still taken from the setup
record (its driver computation is not ported). Validation on all 18 real steps (`ocnhbl_jax_compare.py`):
KBL exact on every column; worst HBL 8.7e-8, worst flux save 1.6e-7, accuracy unchanged from D66.

## D77: OPFIL2 as batched linear operators (JAX)

OPFIL2 acts on each latitude row independently and linearly, so the application is one 72x72 operator
per (layer, row). `fullfidelity/opfil2_jax.py` builds them once from the validated scalar port
(D74-D75; 4.6 s) and applies them as a batched JAX product. On 260 recorded Nov-26 calls the batched
form agrees with the recorded output to 3.0e-15 (relative).

## D78: ODHORZ batched (numpy, grid vectorized)

`fullfidelity/odhorz_vec.py` replaces the (i, j) loops of the scalar ODHORZ port (`odhorz_ff.odhorz`)
with masked array operations. The layer loop stays sequential (PDN and OGEOZ are carried down the
layers); the polar row is handled by the same `polevel` call. Agrees with the scalar port to ~3e-17
and with the real Fortran outputs to ~2e-7 (relative) on all three dates
(`odhorz_vec_compare.py <date> <itime>`).

## D79: ODHORZ layer body in JAX

`fullfidelity/odhorz_jax.py` runs the D78 layer body under `jax.jit` (one compile for all 13 layers;
the layer loop stays a Python loop, the `lmm[1, JM] >= l` polar branches became masked updates).
Agrees with `odhorz_vec` to 1.5e-13 (dec01), 9.7e-14 (jan01), 1.9e-13 (nov26) and with the real
Fortran to the same ~2e-7 (`odhorz_jax_compare.py <date> <itime>`). The ~1e-13 gap to the numpy
version is not isolated; likely the summation order in the polar reduction. Tests:
`tests/test_odhorz_vec_jax.py` (6 tests).

**Correction (2026-10-06, found by the whole-ocean chain D118-D120):** `odhorz_ff.OMEGA` was the sidereal-day value 2*pi/86164.09054, 1.8e-6 relative too high; the real rotation period is 86400*365/366 s (`shared/Earth365DayOrbit.F90:101`, `Constants_mod.F90:282`). With the corrected value ODHORZ agrees with the real run to ~1e-17 (scalar and numpy) and ~1e-13 (JAX) instead of the ~2e-7 quoted in D78/D79 above; the 2e-7 residual was this constant, not an unexplained discrepancy.

## D80-D81: OADVT2 batched (numpy): OADVTY2, OADVTZ2, then OADVTX2

`fullfidelity/oadvt_vec.py` batches the three advection sweeps of the OADVT2 family, keeping each
sweep's one sequential dimension and vectorizing the rest.
- **OADVTY2 (D80):** sequential in j (46 steps), vectorized over (i, l). One bug found and fixed while
  porting: the scalar port updates RM first and the RY update then uses the *new* RM; the first batched
  version used the old RM (3.6e-6 / 4.3e-5 mismatch), now corrected. The pole average uses `cumsum`
  (sequential, as Fortran SUM under strict FP).
- **OADVTZ2 (D80):** sequential in l (13 steps), vectorized over (i, j); donor-layer RZ limiter applied in
  place as in the Fortran; layer l+1 is only read where cm < 0, which is unreachable at l = LMO.
- **OADVTX2 (D81):** sequential in i (71 steps), vectorized over all (l, j) passes at once. The
  Fortran's single MUDT array carries stale values between passes (see `oadvt2_ff.oadvtx2`), so MUDT,
  NCOURANT and the pass-skip logic are precomputed per pass in the scalar order (`_x_prepass`; they depend
  only on MU, the input MM and the masks), then the flux sweep runs on all passes together. The
  carried-over flux state across segments and the single-cell-segment skip are reproduced with masks.
- **Result:** every sweep is bitwise identical (0.0 difference) to the scalar port on all three dates for
  both G0M and S0M; the full `oadvt2_vec` matches the real Fortran dumps with 0.0 difference.
  Scalar -> batched time: X 0.21 s -> 0.07 s, Y 0.17 s -> 0.01 s, Z 0.26 s -> 0.01 s.
  Check: `oadvt_vec_compare.py <date> <itime>`; tests: `tests/test_oadvt_vec.py` (12 tests).
- Not done: a JAX (jit) version of the OADVT2 sweeps; the numpy X sweep still has a Python loop over i.

## D82: OCNMESO inputs batched (numpy)

`fullfidelity/ocnmeso_vec.py` replaces the (i, j) loops of `ocnstate_derived`, `densgrad_vertical` and
`get_1d_mesodiff` (D47) with masked array operations. `ocnstate_derived` keeps a loop over the 13 layers
(interface pressure accumulated in the scalar order, so rounding is unchanged); the other two have no
cross-layer recurrence. Bitwise identical (0.0 difference) to the scalar port on all three dates.
Scalar -> batched time: 0.06 s -> 0.005 s and 0.08 s -> 0.003 s. The inputs it consumes (VUP/VDN/VUPU/
VDNU from the EOS table) are still the recorded values, as in D47. Check: `ocnmeso_vec_compare.py
<date> <itime>`; tests: `tests/test_ocnmeso_vec.py` (3 tests).

## D83: Gent-McWilliams batched (numpy)

Batched `gmredi_ff.isoslope4`, `gmkdif` and `gmfexp` (with `_compute_fxx_fyy_fzz_fzx_fzy`, `_compute_fluxes`, `_wrap_adjust_fluxes`, `_add_fluxes`) into `gm_vec.py` (`isoslope4_vec`, `gmkdif_vec`, `gmfexp_vec`), with the (i, j, l) loops replaced by masked array operations on the same 1-based (IM+1, JM+1, LMO+1) arrays. All three functions are fully batched; nothing was left as a scalar call.

Rounding-order choices: every expression keeps the scalar operand order. The two off-by-one write targets (BXX at (IM1,J,L), BYY at (I,J-1,L)) are handled by a periodic west/east shift of the masked values. `_add_fluxes` (G0M path) applies the horizontal flux terms to TRM in the scalar's per-cell order (west-wrap term at cell IM first, then +FX, +FY, -FX(i+1), -FY(j+1), pole-box terms last) and keeps a loop over the 13 layers for the vertical redistribution. `_wrap_adjust_fluxes` (S0M limiter) uses `np.cumsum(...)[-1]` for the pole sums and for CONVPOS/CONVADJ in the scalar's (l, i) order per J then over J. The vertical convergence is rebuilt as (xy + fz[l]) - fz[l-1], the order the descending-l scalar loop produces.

Validation (`gm_vec_compare.py <date> <itime>`, `tests/test_gm_vec.py`, 9 tests passed): on nov26 33312, dec01 33552 and jan01 17520, the batched results are bitwise identical to the scalar port (max abs and max rel difference 0.0) for all 24 ISOSLOPE4 fields, all 15 GMKDIF fields, and TRM/TXM/TYM/TZM of both GMFEXP calls (qlimit False and True). Inputs are the real dumps, loaded as in gmredi_compare.py, gmkdif_compare.py and gmfexp_compare.py (each stage fed its own recorded inputs, not chained).

Timings (scalar -> batched, one host, single run): isoslope4 0.33 -> 0.022 s, gmkdif 0.19 -> 0.004 s, gmfexp 0.52 -> 0.014 s per call.

Not changed: QCROSS branches remain excluded as in the scalar port (D48).

## D84: OADVT2 sweeps under jax.jit

`fullfidelity/oadvt_jax.py` runs the D80-D81 sweeps under `jax.jit`: Y as a `lax.fori_loop` over j, Z as
13 unrolled layer steps, X as a `lax.fori_loop` over i inside a loop over courant sub-steps, with all
(l, j) passes as fixed-shape lanes. Agrees with `oadvt_vec` to 1e-16 to 3.5e-16 (relative, per sweep and
for the full call) and with the real Fortran dumps to the same 1e-16 level, on all three dates for
G0M and S0M (`oadvt_jax_compare.py <date> <itime>`; tests: `tests/test_oadvt_jax.py`, 6 tests).
- **Speed (CPU, warm, one tracer):** about 0.15 s, the same as the numpy version (0.15 s), so there is no
  CPU gain; the point is a jit-able, device-resident form for the chained step. The X MUDT/NCOURANT
  pre-pass is still numpy (0.04 s per call, two calls per OADVT2) and is now the largest single cost.
- Open: batch or jit the X pre-pass (its stale-MUDT state makes it sequential over passes); GPU timing
  (no GPU on this node).

## D85: Gent-McWilliams under jax.jit

`gm_jax.py` ports `gm_vec.py` (D83) to JAX: `isoslope4_jax`, `gmkdif_jax`, `gmfexp_jax`, each wrapped in `jax.jit` (x64 enabled). `gmfexp_jax` takes `qlimit` as a static argument (it selects the salt-limiter vs plain flux-add path); the other inputs (lmm/lmu/lmv, fields) are jnp arrays. Same operation order as gm_vec; in-place mutation replaced by `.at[...]`; the `if sumpos > 0` branch became `jnp.where`, and the `lmm[1,jp] > l` pole branches in the layer loop became `jnp.where` masks. The 13-layer redistribution loop is unrolled at trace time; pole and global sums use `jnp.cumsum`, which on CPU reproduced numpy's left-to-right cumsum exactly. Geometry (`geomo_dyn_arrays`) is numpy and enters as trace-time constants. Nothing left unconverted.

Validation against gm_vec on the real dumps (nov26 33312, dec01 33552, jan01 17520) via `gm_jax_compare.py <date> <itime>` and `tests/test_gm_jax.py` (9 tests, all pass, tolerance 1e-10 on every output array, both qlimit values):
- isoslope4 (24 arrays), gmkdif (15 arrays): max rel diff 0 (bitwise) on all three dates.
- gmfexp TRM, TZM: max rel diff 0 (bitwise) for both qlimit calls on all three dates.
- gmfexp TXM/TYM: max rel diff 1.3e-16 to 3.5e-16 (last-bit rounding in the division).

Warm CPU timings (mean of 5 calls after a compile call, jax 0.5.3, CPU; numpy gm_vec vs gm_jax): isoslope4 ~0.017 s vs ~0.0025 s; gmkdif ~0.0032 s vs ~0.0008 s; gmfexp ~0.012-0.014 s vs ~0.0046-0.0048 s (qlimit=False) and ~0.0025-0.0028 s (qlimit=True). Single machine, not a controlled benchmark; compile time excluded.

Files: `gm_jax.py`, `gm_jax_compare.py`, `tests/test_gm_jax.py`.

## D86: OCNMESO inputs under jax.jit

`fullfidelity/ocnmeso_jax.py` runs `ocnstate_derived`, `densgrad_vertical` and `get_1d_mesodiff` (D47,
batched in D82) under `jax.jit`; `ocnstate_derived` keeps its 13-layer loop (unrolled). Agrees with
`ocnmeso_vec` to 3.8e-16 relative (`ocnstate_derived`) and exactly (`densgrad_vertical`, k3d) on all
three dates (`ocnmeso_jax_compare.py <date> <itime>`; tests: `tests/test_ocnmeso_jax.py`, 3 tests).
Warm CPU time for both functions: about 6-7 ms (numpy) -> 3.5 ms (jax). The EOS-table outputs (VUP/VDN/
VUPU/VDNU) are still the recorded values, as in D47.

## D87: Sea-ice dynamics (VPICEDYN) batched (numpy)

`fullfidelity/icedyn_vec.py` batches the viscous-plastic ADI solve of `icedyn_dynsi_ff.py` (plast, form, relax, tridiag_thomas, tridiag_cyclic, vpicedyn). Nothing is left scalar.
- PLAST/FORM and every coefficient/RHS stencil in RELAX are shifted-slice array expressions with the scalar term order copied verbatim (including the stale-boundary quirks: after the J-direction solves only columns 2..NX1-1 are updated; the pole row/cyclic columns are filled before the stencils; `aa9` pole terms only on j=NYPOLE).
- Tridiagonal solves are batched across lines (`tridiag_thomas_batch`, `tridiag_cyclic_batch`, shape (n, nlines)); the Thomas / Sherman-Morrison recurrence stays a Python loop along the line. The cyclic `b[0]==1` doubling is applied per line with `np.where`. Pole-row means and the RMS accumulators use `np.cumsum` so the summation order equals the scalar left-to-right sum.
- The outer pseudo-timestep convergence loop is the same data-dependent Python `while`.
- Validation (`icedyn_vec_compare.py`, `tests/test_icedyn_vec.py`, 23 tests, all 18 real records = 3 dates x 6 steps): form, plast, relax, both tridiagonal solvers and the full vpicedyn are bit-identical to the scalar port (max scale-relative difference 0.0 on every record; same kki=2 on all). Vs the real Fortran dumps, same comparison/tolerance as `test_dynsi_ff.py`: worst max_rel uice1 2.4e-10 / 8.9e-10 / 5.1e-9 (nov26 / dec01 / jan01 first record; identical to the scalar port's values).
- Timing (first record per date): full vpicedyn scalar 1.4-1.5 s -> batched 0.025 s (~55x); relax 0.29 s -> 6 ms; form 60 ms -> 1 ms; plast 30 ms -> 0.4 ms.
- Limitations: only the validated OSURF_TILT=1 path was exercised with real data (the OSURF_TILT!=1 branch is ported but untested); every real record converged in kki=2, so the kki>2 / max_kki path of the outer loop is not exercised by real data.

## D88: Sea-ice dynamics under jax.jit

`fullfidelity/icedyn_jax.py`: icedyn_vec's stencils as jnp expressions, with `_form` and `_relax` each one `jax.jit` function (geometry passed as a pytree), tridiagonal solves via `lax.scan` along the line (Thomas forward/backward; cyclic Sherman-Morrison with per-line b0==1 doubling). The outer VPICEDYN convergence loop (and its RMS test, on the host) stays in Python. FORM supports OSURF_TILT=1 only (asserted).
- Validation (`icedyn_jax_compare.py`, `tests/test_icedyn_jax.py`, 21 tests; with icedyn_vec tests 44 passed in 36 s): vs icedyn_vec on real inputs: form <=1e-15, plast <=3.2e-16, tridiag 2.4e-16, relax <=2.2e-12, full vpicedyn <=8.2e-12 across all 18 records (target 1e-10). Vs the real Fortran: same max_rel as the numpy version to within roundoff (uice1 up to 1.5e-9 on jan01, tolerance 1e-6).
- Timing: jitted form ~6 ms, relax ~7 ms; the full vpicedyn is 0.043 s vs 0.025 s for numpy on this small (46x74) grid, so jit is not faster here (host conversions, per-call dispatch, and the lax.scan recurrences on CPU); its value is portability to GPU/batched ensembles, not CPU speed.

## D89: get_dq_cond / get_dq_evap (first cloud delta)

First atmosphere/cloud piece (scoping: `fullfidelity/scoping/ATM_CLOUDS_SCOPE.md`, D-C1). `fullfidelity/clouds_dq_ff.py` ports `get_dq_cond` and `get_dq_evap` (CLOUDS2.F90:7259-7336) plus the pure helpers they call, `QSAT` and `DQSATDT` (shared/Utilities.F90:33-67), vectorised over a flat record axis in numpy (fixed 3-iteration adjustment; the `QM>0` / `COND>0` guards and the final `max(0,min(dqsum,qm|cond))` clamp as masks). Constants (`bysha`, `mrat`, `rvap`, `tf`) are rebuilt from Constants_mod.F90 and equal the values the real model wrote, bit for bit. Owner/source: real ModelE (ROCKE-3D planet_2.0) source tree; validated against the instrumented real binary.

Single-precision-literal check (D54 hazard): the only un-suffixed literals in these routines/helpers/constants are `0.` and `1.`, exactly representable, so no correction was needed; the others carry `d0`. exp is the only transcendental.

Instrumentation (new, `diff -u` generated against pristine files, order-independent): `instrumentation/CLOUDS2_dq.f90.patch` (a `call ffdq_site(k)` before each of the 6 live call sites, CLOUDS2.F90:1364 MSTCNV updraft cond, 2067 MSTCNV downdraft evap, 2716 MSTCNV precip evap, 4002 LSCOND evap liquid, 4013 LSCOND evap ice, 4425 LSCOND cond; and a dump call at the end of each routine) and `instrumentation/ATM_DRV_clouds_dq.f.patch` (`ffdq_site`, `ffdump_dq`, unit 1050 verified unused). Only these two patches were needed in the build (fresh scratch copy, no earlier patch applied). Dumps: `ff_data/<date>/ffc_dq_<itime>.bin`, 13-double big-endian records (itime, site, kind, ncall, sm, qm, plk, mass, lhx, pl, cond, dqsum, f) plus `ffc_dq_consts.txt`. SAMPLED: every 10th call of each (itime, site) per step, 6 steps per date (nov26 33312, dec01 33552, jan01 17520). The 6 steps held about 595,000 / 590,000 / 556,000 real calls; 59,530 / 58,991 / 55,576 were recorded (all 6 sites present on every date; sites 4/5/6 are the rare LSCOND ones: 590/30/1,322 records on nov26, 571/33/1,680 dec01, 565/30/1,304 jan01). Not covered: the SCM.F90 call (not live), steps outside the 6-step windows, calls not on the stride.

Validation (`clouds_dq_compare.py`, `tests/test_clouds_dq_ff.py`, 20 tests incl. mutations and hand-derived cases, all pass): 174,097 sampled calls on 3 dates, 0 branch mismatches (every record's dq>0 vs dq==0 agrees). Bitwise-equal dqsum on 59,073/59,530 (nov26), 58,558/58,991 (dec01), 55,123/55,576 (jan01), about 99.2% overall; fcond/fevp equally. Remaining records differ at rounding level: worst absolute difference 4.6e-16 / 3.9e-16 / 7.0e-16; worst relative to the clamp scale (qm or cond) 2.9e-14 (site 3), worst relative to dqsum itself 1.6e-12 (a cancelling sum of three corrections amplifies a last-bit difference), worst fcond/fevp absolute 2.9e-14. Tolerance used in tests: 1e-12 relative to the clamp scale and 0 branch mismatches; stated honestly, this is rounding-level agreement, not bitwise. Cause not isolated: replacing numpy's exp by glibc `math.exp` changes nothing (same 59,073 exact on nov26), so the difference is consistent with the Intel-libm vs other-libm exp last-bit gap seen in D54 (or compiler operation-order differences), but I did not run a standalone ifort test to prove it.

Non-vacuity: 155,388 records have dqsum>0 (about 89%); evaporation records with COND<=0 (guard branch) and records at both clamps are present (asserted in tests). Mutation checks (niter 2 or 4, no clamp, bysha scaled 1%, evap guard removed) all make the comparison fail. Hand-derived (NOT real-Fortran) tests: guards, dry air, supersaturation with energy-consistent final state, full evaporation of small condensate, closed-form QSAT and the 130 K floor.

Limitations: sampled (10%), 6 steps per date only; only the argument ranges these 6 sites produce on this rundeck/these dates; sites 4-6 (LSCOND) have few samples, site 5 (ice evap) only about 30 per date; the OpenMP counters assume one thread (run with OMP_NUM_THREADS=1). No JAX version yet (elementwise, trivial to add). Files: `clouds_dq_ff.py`, `clouds_dq_compare.py`, `tests/test_clouds_dq_ff.py`, `instrumentation/CLOUDS2_dq.f90.patch`, `instrumentation/ATM_DRV_clouds_dq.f.patch`.

## D90: velocity filter chain FLTRUV/fltry2 (first dynamics delta)

First atmosphere-dynamics delta (scoping: `fullfidelity/scoping/ATM_DYNAMICS_SCOPE.md`, section 7). `fullfidelity/dyn_fltruv_ff.py` ports the end-of-`DYNAM` velocity filter chain of `ATMDYN.f` (lines 379-388): `FLTRUV` (1702-1812; 8th-order E-W Shapiro on U and V at J=2..JM, then the per-row/per-layer angular-momentum fix on U, ANG_UV=1), `fltry2` for U and then V (1627-1700; 8-pass N-S filter with the pole-crossing conditions), `CONSERV_AMB_EXT` x2 (2181-2211), the `GLOBALSUM` of the angular-momentum change (`GlobalSum_mod.F90` `globalSum_IJ`: zonal sum over I per J, then sum over J=1..JM) and `ADD_AM_AS_SOLIDBODY_ROTATION` (2146-2179). `DIAGCD` calls (diagnostics only) are not ported.
- Instrumentation (new patches, apply after `ATM_DRV.f.patch` and `MODELE.f.patch`; they need nothing else): `ATMDYN_fltruv.f.patch` (4 call sites in `DYNAM`) and `ATM_DRV_fltruv.f.patch` (adds `ffdump_fltruv_on/_geom/_in/_uv/_ny`, unit 1070; units 1070-1072 re-grepped over model/, shared/ and all earlier patch files: unused). Per `DYNAM` call (once per 30-min step; gated by `FFD_START/FFD_NSTEP`), big-endian f8 stream files in `ff_data/<date>/`: `ffd_fltruv_<itime>_in.bin` (itime, U, V (IM,JM,LM), MA (LM,IM,JM), MASUM (IM,JM) before FLTRUV), `_flt.bin` (itime, U, V after FLTRUV), `_ny.bin` (itime, DAMSUM, AM1 (IM,JM), U, V after both `fltry2`, before the solid-body fix), `_out.bin` (itime, U, V after the whole chain); plus `ffd_fltruv_geom.bin` once (JM, RADIUS, OMEGA, DXYN, DXYS, COSV). Fixed-form source, added lines <= 72 columns.
- Validation (`dyn_fltruv_compare.py`, `tests/test_dyn_fltruv_ff.py`, 48 tests): 18 real calls (3 dates x 6 steps: nov26 33312-33317, dec01 33552-33557, jan01 17520-17525), every stage and field, all 72x46x40 points including the pole rows J=2 and J=JM and row J=1 (untouched): U,V after FLTRUV, AM1, U,V after fltry2, DAMSUM, final U,V are **bitwise identical (max abs difference 0.0; 132,480/132,480 U points exact on every call)**. The recorded final U,V also equal the existing `ffd_<itime>_pre_condse` U,V bit for bit (independent cross-check, nov26 33312). Non-vacuous: the chain changes U by up to 1.7-2.4 m/s per call (max |U| scale ~tens of m/s); DAMSUM is nonzero on every call.
- Geometry: DXYN, DXYS, COSV are derived analytically from `GEOM_B.f` (half-polar-box 4x5 branch; the J=2 and J=JM COSV corrections included) and match the recorded dump arrays exactly (relative difference 0.0), and the whole chain was validated both with the recorded and with the analytic geometry. RADIUS and OMEGA are runtime planet parameters (`USE_PLANET_RAD`) and are read from the dump header, not derived.
- Exactness notes: all sums (the I loop in FLTRUV starting at I=IM, the L sum in `CONSERV_AMB_EXT`, the GLOBALSUM and `MASUM` row sums) are strictly sequential left-to-right (`seqsum`); this reproduced ifort bit for bit with no reassociation. No single-precision-literal issue arises here (`by4toN=1./(4.**8)` is exactly 2**-16). No `pow/exp` is involved except the analytic-geometry sin/cos, which matched exactly for this grid.
- Mutation tests (all detected): ANG_UV=0, NSHAP=6, dropped pole-crossing sign in `fltry2`; conservation test shows the angular-momentum fix reduces the row residual by >1e3.
- Limitations: the 18 calls are 6-step windows from three restarts only; serial single-rank domain (halo updates treated as no-ops, J_0STG=2, J_1STG=JM; the build ran with OMP_NUM_THREADS=1); constants DT=450, DT_XUfilter=DT_XVfilter=450, DT_YU/YVfilter unused (strength fixed 1d0 in the call), ANG_UV=1, DO_POLEFIX is not part of this chain (`isotropuv` is a separate, later delta). Not batched/JAX yet (the numpy form is already whole-field vectorised except the 72-step sequential I-order and 45-row loops in the AM routines; JAX version is a later step). Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session, 2026-10-05.

## D96: AFLUX + ADVECM + MAtoP (+ AVRX and the FFT72 radix FFT) (scoping items 3-4, dynamics)

`fullfidelity/dyn_aflux_ff.py` ports `AFLUX` (`ATMDYN.f` 483-745: B-grid mass fluxes MU,MV with the polar interpolation `POLWT`, polar MU*3 rule and `DO_POLEFIX` 2/3 scaling, uphill-flux topography adjustment, CONV, the MW column recursion), `ADVECM` (748-844) and `MAtoP` (`ATMDYN_COM.F90` 147-191, PEDN/PMID/PDSIG/PK/P). `AVRX` (`ATMDYN.f` 1327-1415) is needed by `AFLUX` (and later `PGF`), so the radix FFT of `FFT72.f` (`FFT`, `FFTI`, `DOCALC`, tables of `FFT0`) was transliterated statement by statement into `fullfidelity/dyn_fft72_ff.py` (batched over rows) instead of using `numpy.fft`. `fullfidelity/intel_libm_ff.py` is a small optional ctypes bridge to the Intel `libimf` `pow` (see exactness).
- Instrumentation (new patches, generated by `diff -u` against pristine files, local hunks, units 1081-1085 re-grepped over model/ and all patch files: unused; they apply to pristine files and also after `ATM_DRV.f.patch`, `ATM_DRV_clouds_dq`, `ATM_DRV_fltruv`, `ATMDYN_fltruv`, verified with `patch`): `ATMDYN_aflux_pgf_advecv.f.patch` (hooks at entry/exit of `AFLUX`, `ADVECM` (after `MAtoP`), live `PGF`, plus a local `ffspa0` copy of SPA taken just before `AVRX`), `MOMEN2ND_advecv.f.patch`, `ATM_DRV_dynB.f.patch` (helpers `ffdb_*`, anchored after `end subroutine finalize_atm`). The leapfrog pass index (1..5) is counted at `AFLUX` entry and reset when `itime` changes; all 5 passes of each step are recorded (6 steps x 3 dates). Files in `ff_data/<date>/`: `ffd_aflux_geom.bin` (once: constants incl. GRAV, RGAS, KAPA, KG2MB, MTOP, MFIXs, MFIX, MFRAC, DT, DLON, POLWT, ACOR, ACOR2, MIN/MAXCOLMASS, DO_POLEFIX, geometry DXYP, BYDXYP, DYP, BYDYP, DXP, DXV, DYV, DXYV, DXYN, DXYS, RAVPN, RAVPS, RAPVN, RAPVS, FCOR, COSV, IMAXJ, SINIV, COSIV, ZATMO, the AFLUX topography patch tables), `ffd_aflux_<itime>_p<k>_{in,out}.bin` (AFLUX: U,V,MA,MASUM,ME,MESUM,NS,DT,MRCH in; MU,MV,MW,CONV,SPA and SPA before AVRX out) and `ffd_aflux_advecm_<itime>_p<k>_{in,out}.bin` (ADVECM: MOLD,CONV,MW,DT1 in; MNEW,MSUM,PEDN,PMID,PDSIG,PK,P out). Layouts and loaders: `dyn_aflux_compare.py`. Fixed geometry is taken from the recorded one-time dump (not derived analytically; the analytic GEOM_B check was done only for DXYN/DXYS/COSV in D90). Pass k: 1 forward (MRCH=0, NS=4), 2 backward (MRCH=-1, NS=4), 3 even (MRCH=2, NS=4), 4 odd (MRCH=-2, NS=3), 5 even (MRCH=2, NS=2) - observed in the dumps. Dump volume ~1.8 GB per date for D96-D98 together (241 files).
- Validation (`dyn_aflux_compare.py`, `tests/test_dyn_aflux_ff.py`, 311 tests): 90 real calls (3 dates x 6 steps x 5 passes; nov26 33312-33317, dec01 33552-33557, jan01 17520-17525), all 72x46x40 points. AFLUX MU, MV (rows J=2..JM; row J=1 is never written and stays 0 in the real run), MW, CONV, SPA (after AVRX) and SPA before AVRX, and ADVECM MNEW, MSUM and MAtoP PEDN, PMID, PDSIG, P are **bitwise identical (max abs difference 0.0) on all 90 calls**. AFLUX is also exact with the real post-AVRX SPA substituted (isolating the FFT), and AVRX alone reproduces the recorded SPA bit for bit.
- Exactness: `PK = PMID**KAPA` with numpy pow differs by 1 ulp (max abs 8.9e-16) in 47 of 132,480 cells in the first call (about 0.04 %); a correctly rounded pow gives the same 1-ulp mismatches, because ifort calls the Intel libimf `pow`, which is not correctly rounded (the D54 lesson again). Calling the same libimf scalar `pow` through ctypes (`intel_libm_ff.pow_imf`, `matop(..., imf_pow=True)`; only on hosts with the Intel runtime, path overridable with `INTEL_LIBIMF_DIR`) reproduces all PK bit for bit. Default mode is numpy pow (tolerance 1e-15 absolute on PK); the libimf mode is tested and skipped where the library is absent. Other findings: (a) the `FFT0` tables must keep the Fortran sequential-overwrite semantics at N=KM/4 (C(18), S(0), S(36), C(54), S(72) are assigned several times, each reading the C(N) left by the previous statement); building them from a once-computed cosine gave 1e-14 differences in SPA, and a test pins this; (b) with that, numpy `cos`/`sin` for the FFT0 table and BYSN reproduced the Fortran (libimf) values exactly; (c) all sums (`Sum(U(:,2,L))`, `Sum(MV...)`, `Sum(DUMMYS)`, `Sum(CONV(I,J,:))`, ADVECM's L accumulation) are strictly sequential, matching ifort -O2 -fp-model strict with no FMA.
- Branches exercised (non-vacuity, counted over the 30 calls per date): topography adjustment (one global patch, mode 1) acted on 36,720 E-W and 36,600 N-S cells with a height step per date, moving flux up a level in 21k-23k E-W and 28k-30k N-S (cell, level) cases; `DO_POLEFIX` pole scaling; polar MU*3 rule; AVRX active on 26 rows (J=2..14 and 33..45, where DRAT<=1) and changing SPA there by up to ~6 m/s, skipped on the other rows. Not exercised, hence not validated: topography patches with mode 0 (the rundeck uses the default single patch, mode 1; the mode-0 code is ported but unchecked), the ADVECM column-mass exception paths (never triggered; ported only as a flag), MPI-halo branches (the run is serial), `MFIX/MFRAC` for other vertical grids (only the recorded L40 values).
- Mutation tests (all detected): no topography adjustment, topography mode 0, AVRX skipped, wrong NS, POLWT, reversed MFRAC (MW recursion), ADVECM without polar copy and with a changed DT1, FFT table without the overwrite semantics.
- Limitations: 6-step windows from three restarts only; serial domain; numpy only (the `AFLUX` topography adjustment and the MW recursion are vectorised over (I,J) with a loop over L, AVRX is batched over layers with a Python loop over the 26 active rows, so a JAX version is a straightforward later step); FFT is a literal transliteration (dict of vectors), not optimised. Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session, 2026-10-05.

## D97: PGF (pressure gradient force, live non-V2 version) (scoping item 5, dynamics)

`fullfidelity/dyn_pgf_ff.py` ports the live `PGF` (`ATMDYN.f` 1107-1324): per-column top-down integration of pressure and layer geopotential with the `**KAPA` powers (AdM = SPA scratch, GZ = PHI exported to the physics), polar columns copied from I=1, N-S derivative into DVT (two updates per cell in Fortran I order), smoothed E-W derivative PGFU through `AVRX` (rows 2..JM-1, from D96) into DUT, `DO_POLEFIX` ACOR/ACOR2 scaling of the polar velocity rows, and the final `UT,VT += DUT,DVT/VMASS`. `DIAG5D`/`DIAGCD` calls (diagnostics only) are not ported.
- Instrumentation: the same patch set as D96 (hooks at entry/exit of the live `PGF`, unit 1084, plus a local `ffpgfu0` copy of PGFU before `AVRX`). `ffd_pgf_<itime>_p<k>_in.bin`: DT1, MRCH, MAM, MAFTER, S0, SZ, UT, VT and the module accumulators DUT, DVT; `_out.bin`: UT, VT, DUT, DVT, GZ, PHI, SPA (AdM), PGFU before and after AVRX. U,V are not recorded (they do not enter the numerics of `PGF`; they only feed `DIAGCD`). Geometry/constants from `ffd_aflux_geom.bin`. Loaders: `dyn_pgf_compare.py`.
- Validation (`dyn_pgf_compare.py`, `tests/test_dyn_pgf_ff.py`, 145 tests): 90 real calls (3 dates x 6 steps x 5 passes), all points. With the Intel libimf `pow` (`imf_pow=True`) **every output - GZ, PHI, SPA(AdM), PGFU before/after AVRX, DUT, DVT, UT, VT - is bitwise identical on all 90 calls (0.0)**. With numpy pow (default) the differences come from `x**KAPA` and `.01d0**KAPA` (libimf pow is not reproduced by numpy, see D96): worst relative to the field scale over all 90 calls GZ 1.6e-14, AdM 3.7e-15, PGFU 1.4e-12, DUT 7.7e-13, DVT 4.8e-13, UT 1.7e-13, VT 1.7e-13 (tests allow 2e-12).
- Findings and non-vacuity: DUT and DVT are exactly zero at `PGF` entry in all 90 real calls (`ADVECV` zeroes them on exit), so the nonzero-entry path (the cell-IM update order of the DVT accumulation follows the Fortran I loop) has no real reference and is not validated; AVRX acts on PGFU rows J=2..14 and 33..45 only; the polar-fix scaling changes the pole rows (test); DUT/DVT reach ~1e11 in these units; UT/VT change on every sampled call.
- Mutation tests (all detected): AVRX skipped, wrong DT1, ACOR/ACOR2 set to 1, wrong KAPA reciprocal, altered ZATMO, DVT sign, polefix off.
- Limitations as D96 (3 windows, serial, numpy). Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session, 2026-10-05.

## D98: ADVECV (momentum advection and Coriolis) (scoping item 6, dynamics)

`fullfidelity/dyn_advecv_ff.py` ports `ADVECV` (`MOMEN2ND.f` 31-498): VMASS scaling of UT,VT, horizontal advection with the W-E, S-N, SW-NE and SE-NW mass-flux contributions, the `DO_POLEFIX` polar rows recomputed from x-y advection of rotated polar velocities (corner fluxes dropped, ACOR correction), vertical advection with SD=MW (`ASDU`, RAVPN/RAVPS), the Coriolis term with the SPA-derived metric term FD, the polar Coriolis override and the final division by VMASS. `DIAG5F/DIAG5D/DIAGCD` (diagnostics) are not ported.
- Instrumentation: same patch set as D96 (`MOMEN2ND_advecv.f.patch`, entry/exit hooks; unit 1085). `ffd_advecv_<itime>_p<k>_in.bin`: DT1, MRCH, U, V, MMEAN, MBEFOR, MAFTER, UT, VT and the module MU, MV, MW, SPA exactly as left by that pass's `AFLUX` (note `ADVECV`'s `PU=>MU`, `PV=>MV`, `SD=>MW` alias the `AFLUX` outputs, not the scaled `DYNAM` PU/PV/SD); `_out.bin`: UT, VT. Loaders: `dyn_advecv_compare.py`.
- Validation (`dyn_advecv_compare.py`, `tests/test_dyn_advecv_ff.py`, 116 tests): 90 real calls, **UT and VT bitwise identical (max abs difference 0.0, 129,600/129,600 UT points in rows J=2..JM exact on every call; row J=1 untouched)**. Operation order is the whole difficulty: DUT/DVT are accumulated sequentially, so each cell's updates are replayed in the exact Fortran order (IP1 role of iteration k, then I role of iteration k+1; cell IM reversed because the loop starts at I=IM), with the carried "previous J" fluxes of the J loop, vectorised over (J,L) and cells. The reversed order for all cells and the general order for cell IM are each detected by the tests (last-bit differences, below 1e-9 relative), so the exactness check is sensitive to it. The Coriolis cell-IM order is numerically irrelevant (two terms from zero commute) but kept.
- Non-vacuity: horizontal advection, the polar replacement (changes rows J=2 and JM by far more than rounding), vertical advection and Coriolis all contribute on every sampled call; UT changes by 1.6 m/s in the first checked call (max |UT| ~89 m/s). Not exercised and not ported: the MPI-halo branches (`J_0STG>2`, non-serial lower-boundary corner fluxes, `haveLatitude` false); POLWT is 1-4e-16 in this grid, so the polar interpolation is nearly the identity (the interpolated rows still enter the formulas and are replayed).
- Mutation tests (all detected): no polefix, DT1 perturbed by 1e-9, SPA zeroed, POLWT changed, ACOR=1, DUT accumulation order reversed, cell-IM order replaced.
- Limitations as D96; the numpy form is vectorised over (J,L) with an explicit per-cell update sequence (about 20 whole-field operations per component), which maps directly to a jitted shift-based JAX kernel (no data-dependent branches). Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session, 2026-10-05.

## D94: FFT72 `FFT`/`FFTI` + `AVRX` ported -- bitwise-exact radix port on all 3,744 x 3 recorded real rows (2026-10-05)

Ports `FFT`, `FFTI`, `DOCALC` (FFT72.f:59-250, 306-450), `FFT0` tables and `AVRX` (ATMDYN.f:1327-1415, including the one-time
`DRAT`/`NMIN`/`BYSN` tables) to numpy (`fullfidelity/dyn_avrx_ff.py`). Decision (scoping item 3, "radix port or real FFT"):
**the faithful radix port reaches bit-for-bit equality**, so it is the one to use; a numpy `rfft`/`irfft` version is kept
only to document the alternative and is NOT bitwise (max |diff| 1.6-1.9e-9 on rows of magnitude ~2e6, i.e. ~1e-15 relative; spectra differ by <=2.3e-10).
The ~200 straight-line statements of `DOCALC`/`FFTI` were machine-translated from the Fortran text (same operator order, no
re-association; real build is `ifort -O2 -fp-model strict -assume protect_parens`, no FMA) and the loops hand-written in the same order.

Instrumentation (new patches, generated by `diff -u` against pristine files, units 1071-1079 of which AVRX uses 1071-1073):
`ATMDYN_dynA.f.patch` (3 call-site tags before `CALL AVRX` at AFLUX 557 / ADVECV-labelled 1027 / PGF 1282, plus 3 per-row hooks
and 1 constants hook inside `AVRX`) and `ATM_DRV_dynA.f.patch` (the `ffd_avrx*` helpers). Records per ACTIVE row (rows with `DRAT<=1`:
J=2..14 and 33..45 = 26 rows/call): input row, FFT spectrum `AN,BN` (before truncation scaling), output row (after `FFTI`), call-site
id, itime, call counter. Sampling: every 17th AVRX call per (itime,site) (`FFD_AVRX_STRIDE`, coprime with the 40 layers so that all
layers occur); all calls are logged in `ffd_avrx_calls.bin`. Per date: 144 sampled calls of 2,400 (6 steps x (200 AFLUX + 200 PGF)),
3,744 rows, 6.6 MB. Call site 2 (`CALL AVRX` at ATMDYN.f:1027) is inside the DEAD `V2_PGF` variant of `PGF`: never executed (0 calls).
Dumps `ff_data/<date>/ffd_avrx.bin`, `ffd_avrx_calls.bin`, `ffd_avrx_consts.bin` (once: DRAT, NMIN, DXP, DYP, BYSN, C, S).

Result (`dyn_avrx_compare.py`, 3 dates x 3,744 rows x 72 = 269,568 values per date): FFT spectra `A,B` 0 diff (exact); AVRX output
0 diff (all 269,568 values bit-identical per date), with BOTH the recorded tables and the analytically derived ones
(`FFT0` cos tables via numpy `cos`, `DRAT/NMIN/BYSN` from analytic geometry `dyn_geom_ff.py`; numpy `cos`/`sin` happened to equal
ifort's on every table entry, DXP/DRAT/BYSN/C/S diff 0, NMIN identical). Non-vacuous: the truncation changes rows by up to
2.7e5-3.5e5 (row scale 1.7e6-2.0e6), every dumped row is modified. `FFTI(AN,BN)` of the recorded spectrum reproduces the input
row to 1e-9 (round-trip, not exact, as expected). 59 new tests (`tests/test_dyn_avrx_ff.py`): per (date,step,site) bitwise
(36), all-rows bitwise with dump/analytic tables, table checks, FFT definition (cos/sin/Nyquist/constant), batching independence,
`DRAT>1` rows untouched, 4 mutations of the port (RT3 1 ulp off, one `S` table entry 1 ulp off, `NMIN`+1, truncation disabled) each detected, and a check that the data distinguish the two 3-term summation orders in `DOCALC`. Not covered: the full 40-layer sweep of every call (only 1 in 17 calls recorded); `AVRX` init with
`xAVRX=byrt2` (order-4 moments, not this build).
Owner: project owner (G. Tamkin); drafted by a Claude Code session; sources: `FFT72.f`, `ATMDYN.f`, `GEOM_B.f` of modelE2_planet_2.0.

## D95: `isotropuv` + `shap1` + `SDRAG` ported -- bitwise-exact on all recorded real rows/columns (2026-10-05)

Ports `isotropuv`/`Hemisphere`/`at_pole`/`far_from_pole`/`shap1` (ATMDYN.f:1838-1955; `fullfidelity/dyn_isotropuv_ff.py`) and the
default (non-linear, `ANG_SDRAG=1`) branch of `SDRAG` (ATMDYN.f:1993-2141; `dyn_sdrag_ff.py`); geometry (`dyn_geom_ff.py`) is derived
analytically from GEOM_B.f and was cross-checked EXACT against the recorded dump (COSV, DXV, COSIV, SINIV, RAPVN, RAPVS, DXYV,
DXYN, DXYS), so the tests run with both analytic and dumped geometry; the pole-row FFT truncation reuses the D94 radix FFT.
Instrumentation: `ATMDYN_dynA.f.patch` (2 hooks per row inside `isotropuv` after the second `shap1` and after the back-conversion;
2 hooks around `CALL SDRAG` in `DYNAM`) + `ATM_DRV_dynA.f.patch` (`ffd_iso`, `ffd_sdrag`), units 1074-1079 (1071-1079 used overall, 1080 spare).
- `isotropuv` (30 calls/date = 5 per step): rows J=2,3,45,46 for every layer. Stat file `ffd_isotr_stat.bin` has EVERY (j,l) row
  (4,800/date: k, fac, n); full records `ffd_isotr.bin` for l=1,5,..,37 and for every row with n>1 (1,847 / 1,780 / 2,068 rows, 6.5 MB),
  including the x-y fields after `shap1` (before the pole FFT). Sub-iteration counts `n=int(fac)+1` that occur (all rows, 3 windows):
  n=1..12 ALL occur (nov26: 3950,150,48,67,130,90,80,41,62,57,111,14 rows for n=1..12; dec01 and jan01 similar, n=12 in 83/223 rows);
  n>1 occurs at J=46 (all n=1..12; `k` in the linear range and saturated at `khi=1e7`) and at J=2 in dec01/jan01 (n=2); J=3 and J=45 only n=1.
  All dumped rows (including every n>1 row): `k`, `fac`, `n`, post-`shap1` `ua,va`, and final `U,V` are BIT-IDENTICAL (nov26 265,968, dec01
  256,320, jan01 297,792 values), pole-row FFT truncation included. Statistics of all 4,800 rows/date: k/fac/n recomputed exactly.
- `SDRAG` (12 calls/date = 2 per step; rundeck: LS1=24, LSDRAG=LPSDRAG=37, X_SDRAG=(.002,.0002), C_SDRAG .0002, Wc_JDRAG=30, WMAX=200, ANG_SDRAG=1):
  columns are independent, so the dump is sampled by column (all J=2..JM, I=1,25,49: 135 columns/call, 1,620 columns/date, 5.8 MB, inputs
  U,V,T,PK,PEDN(L+1),4 MA values, outputs U,V; RGAS added to the constants). All 129,600 output values per date are BIT-IDENTICAL
  (both regimes: constant-C levels 24..36 and linear-in-wind levels 37..40; angular-momentum return below LS1). Non-vacuous: max |dU| 0.84-0.88 m/s (full grid), per-column up to 0.75 m/s.
  Full-grid statistics from the instrumentation: wl>wmaxj occurs in ZERO columns in all 36 calls (max wl 64-71 m/s vs wmaxj 200/150), so the
  **clamp branch (`X=1-(1-X)*wmaxj/wl`) is NOT validated against real Fortran**; it is ported and covered only by hand-derived tests labelled as such.
  Linear (`rtau`) branch and `UNRDRAG` are dead (not ported); `DIAGCD`/`AJL` diagnostic accumulations not ported; the `T` bound check (100-373 K) raises.
  Limitation: the `.15` literal in `COSV(J).LE..15` is REAL*4 (ported as float32(.15)); no velocity row has cosv within 1e-3 of .15 so the data cannot distinguish it.
- 113 + 55 new tests (`tests/test_dyn_isotropuv_ff.py`, `tests/test_dyn_sdrag_ff.py`): per-call bitwise (90 + 36), analytic geometry/tables, n coverage,
  `shap1` closed-form checks (constant row, Nyquist damping `(1-fac/n)^n`, conservation, `n=int(fac)+1`), pole-row harmonic truncation, and mutations
  (n without +1, khi, no pole FFT, hemisphere sign, dt, ang_sdrag off, LSDRAG shifted, X_SDRAG, Wc_JDRAG, RGAS 1 ulp) all detected.
Full-suite note: ran only the four dynamics test files (`test_dyn_fltruv_ff`, `_avrx_`, `_isotropuv_`, `_sdrag_`): 275 passed; the whole repo suite was not run here.

**Known duplication (noted at merge):** the FFT72 radix code exists twice, `dyn_avrx_ff.py` (D94, machine-translated `DOCALC`/`FFTI`) and `dyn_fft72_ff.py` (D96, transliterated for AFLUX); both are validated bitwise against the real model. Consolidate onto one before the JAX conversion.

## D99: AADVT driver (X/Y/Z sweeps, Courant nstep) + `adv1d` + `advection_1D_custom` + `limitq` ported, bitwise-exact on all 36 real calls (2026-10-05)

`AADVT` (QUS_DRV.f:70-196) with `AADVTX` (198-316), `AADVTY` (318-471), `AADVTZ` (474-573), and the 1-D kernels
`adv1d` (QUSDEF.f:43-221), `advection_1D_custom` (224-635, qlimit=.false. path) and `limitq` (639-758) as separate
importable functions (shared with moist convection). New files: `fullfidelity/dyn_aadvt_ff.py` (driver, sweeps,
Courant counters, scalar per-row reference loops), `dyn_adv1d_ff.py` (kernels), `dyn_aadvt_compare.py`,
`dyn_adv1d_compare.py`, `tests/test_dyn_aadvt_ff.py` (82 tests, ~40 s).
Validation: per call, 3 dates (nov26 33312, dec01 33552, jan01 17520) x 6 steps x 2 AADVT calls = 36 calls
(the two even leapfrog passes of DYNAM). Input (MMA, T, TMOM(9), MU, MV, MW, DT) and output (MMA, T, TMOM, FPEU,
FPEV) at the call boundary plus mass-unit checkpoints after X1, Y, Z and the per-row/column Courant nstep.
Result: **0.0 max difference on every array, every call (1,324,800 T/TMOM values per call all equal), all three
checkpoints, nstep arrays equal.** First try; no bug found in the port. Two details that mattered: `fracm**3` must be
`np.power(fracm, 3.)` (x*x*x differs from ifort in 516 of 38,448 synthetic cells), and `ierr/nerr` from adv1d's
qlimit loop are the LAST limitq call's status.
Reductions in Fortran order (polar `sum` via cumsum; fqu/fqv accumulated over l sequentially).
Coverage (measured): in the real windows EVERY row/column has nstep=1 (X: 126,720 row-sweeps, Z: 119,232 columns;
max Courant 0.28-0.39), qlimit is always .false. so `limitq` never runs, no cell reaches mass<=0. So the real
windows do NOT exercise multi-step masking: see D100.
Instrumentation: `ATMDYN_aadvt.f.patch`, `QUS_DRV_aadvt.f.patch`, `ATM_DRV_dynC.f.patch` (diff -u against pristine, local
hunks, verified to apply to pristine); units 1100 (in/out), 1101 (nstep), 1102 (checkpoints). 72 files, ~868 MB per
date in `ff_data/<date>/ffd_aadvt_*`. Owner: project owner; source: ModelE2_planet_2.0 (read-only).

## D100: nstep>1 and qlimit paths validated by stress dumps and a standalone ifort harness (2026-10-05)

Because D99's windows never exercise nstep>1, the helper (env `FFD_AADVT_STRESS=<f>`, off by default; a rerun with it
unset reproduced the first run byte for byte) scales MU and MW seen by one AADVT call by f. Real Fortran, f=4:
21 complete calls (7 per date; the model then stops in `aadvtx/aadvtz` courmax>1 at nstep=20, i.e. the model's own
error exit) with X nstep 1..4 (225 rows nstep 2, 7 nstep 3, 2 nstep 4) and Z nstep 1..3 (151 / 24 columns): **all 21
calls bitwise-exact (outputs, 3 checkpoints, nstep arrays)**. f=12 aborted in the first call (no complete call).
The batched implementation groups rows by their own nstep; the literal per-row Fortran loop reference gives identical
bits; running every row for max-nstep (the scoping-doc warning) is shown by a mutation test to change the result.
`adv1d` with qlimit=.true. and all `limitq` branches: validated against a STANDALONE ifort build of the real
QUSDEF.f adv1d+limitq (same flags) on 3,000 seeded synthetic lines (all 3 direction permutations; every limitq branch
hit): ierr/nerr match on all; 2293 of 2295 non-aborted lines bitwise, 2 differ by 1 ulp (1e-17) in one moment at
|fracm|>1 (unphysical) pow(x,3) cells. This is NOT model data; limitq in CLOUDS2 remains to be validated with real
dumps in the clouds work. Never exercised anywhere: y-direction limiter (`apply_limiter`, not ported, raises),
`prather_limits` (dead, not ported), the mass<=0 reset (hand/synthetic only). Data: `ff_data/qus1d_harness/{in,out}.bin`,
stress dumps `ff_data/<date>/ffd_aadvtS4_*`. Harness source: `instrumentation/qus1d_standalone_drv.f90`.

## D91: PRECIP_MP, ANVIL_OPTICAL_THICKNESS, MC_CLOUD_FRACTION, MC_PRECIP_PHASE (scoping D-C2)

Second cloud delta. `fullfidelity/clouds_helpers_ff.py` ports four stateless per-layer helpers of CLOUDS2.F90 used by MSTCNV: `PRECIP_MP` (5658-5680, the Marshall-Palmer precipitating-mass function, 8 call sites inside CONVECTIVE_MICROPHYSICS), `ANVIL_OPTICAL_THICKNESS` (5290-5328, live call 3114; the call at 3080 is inside `AIE_DIAG_FIX_MET`, not defined), `MC_CLOUD_FRACTION` (5332-5392, call 2675), `MC_PRECIP_PHASE` (5588-5654, call 2703). Elementwise numpy over a flat record axis, Fortran branches as masks. Owner/source: real ModelE (ROCKE-3D planet_2.0) source; validated against the instrumented real binary.

Single-precision-literal audit (D54 hazard): `1.E-20` in ANVIL_OPTICAL_THICKNESS is a REAL(4) literal (promoted value 9.99999968e-21), reproduced with a float32 round-trip (`f4`); it cannot be discriminated by the real data (it only matters when FCLD*RCLDE is about 1e-20). All other un-suffixed literals here (1., 3., 4., 6., .5, 1.5, 2., 5., 100., 450., 1000.) are exactly representable. `FLAM**4.` has a REAL(4) exponent; of the two candidate evaluations only `pow(x,4.)` reproduces the dumps (99.5% bitwise vs 90.2% for `(x*x)*(x*x)`).

Instrumentation (new, `diff -u` generated against pristine files; one pair for D91-D93): `instrumentation/CLOUDS2_helpers.f90.patch`, `instrumentation/ATM_DRV_clouds_helpers.f.patch` (`ffh_site`, `ffh_rec`; units 1051-1056 dumps, 1057 text; grepped across the tree and all patches). Verified to apply to pristine files and after the D89 patches; the built binary used only these two patches. Dumps `ff_data/<date>/ffc_{pmp,anv,mcf,cmp,mpp}_<itime>.bin` (big-endian f8; header itime, site, ncall; inputs then outputs; layouts in `clouds_helpers_compare.py`), `ffc_h_consts.txt`, `ffc_h_counts.txt`. SAMPLED, per (itime, site): the first 2 calls plus every Nth: PRECIP_MP N=20, ANVIL N=3, MC_CLOUD_FRACTION N=40, MC_PRECIP_PHASE N=40 (also CONVECTIVE_MICROPHYSICS N=20 and MASS_FLUX N=3, see D92/D93). Real call counts per step (nov26, 5 flushed steps): PRECIP_MP 80,601 sites 1-2 / 61,473 sites 3-4 / 20,506 sites 5-8 (about 16.1k/12.3k/4.1k per site per step), ANVIL 10,592, MC_CLOUD_FRACTION 197,041, MC_PRECIP_PHASE 197,041 (the last step of each window is not flushed to the counts file; ncall in its last record is a lower bound). Records per date (nov26/dec01/jan01): PRECIP_MP 22,102/22,062/20,460 (all 8 sites), ANVIL 4,273/4,112/3,660, MC_CLOUD_FRACTION 5,940/5,938/5,519, MC_PRECIP_PHASE 5,940/5,938/5,519. Dumps total about 7.4 MB per date for all six routines.

Validation (`clouds_helpers_compare.py`, `tests/test_clouds_helpers_ff.py`; the file has 29 tests, shared with D92): over the 3 dates, PRECIP_MP 64,624 records: 99.51% bitwise, worst relative 4.0e-16 (worst abs 4.3e-19); ANVIL 12,045: rcld and taumc 99.98% bitwise (3 records differ), worst rel 3.9e-16; MC_CLOUD_FRACTION 17,397: 100% bitwise; MC_PRECIP_PHASE 17,397: lhp, mcloud, heat1 and prcp 100% bitwise. The non-bitwise PRECIP_MP records are 1-2 ulp, consistent with the Intel-libm vs other-libm `exp` gap of D54 (a correctly rounded longdouble exp does not fix it either; cause not isolated further). Tolerance in tests: 1e-12 relative.

Branch coverage (counts on nov26/dec01/jan01). MC_CLOUD_FRACTION: deep x5 686/638/603, below-base virga 220/200/150, shallow 3,962/3,971/3,802, shallow detrainment (L=LMAX-1) 519/518/567, shallow-below-base zeroing 2,084/2,148/1,946, cap at 1.0 19/16/11. MC_PRECIP_PHASE: melt 261/291/286, phase-conversion heat 72/67/81, refreeze only 2/2/0 (two records per date at most), precip phase ice/liquid both about half. ANVIL: liquid 109/96/213, ice 4,164/4,016/3,447, RIMAX cap about 1,100-1,300 per date, TAUMC=100 cap 0/1/0. NEVER exercised by the real windows: MC_PRECIP_PHASE with `MC_REVP_ABV_CLDBASE=0` (the run uses 1, so the `L<=LMIN` arm is untested against real data), MC_PRECIP_PHASE `MCLOUD>AIRM` cap (0 records), `HEAT1` entry value nonzero (always 0 on entry). These are covered only by the scalar-transcription tests below.

Mutation checks on real data (all detected): ANVIL phase test swapped; ANVIL cap 100 -> 99 (dec01, one record); MC_CLOUD_FRACTION without the x5, virga or detrainment arm, and with DTsrc=3600; MC_PRECIP_PHASE with BYSHA +1%, the melt-test constant shifted, and `MC_REVP_ABV_CLDBASE=0` (differs from the dump, so the real run's use of the option 1 is pinned). Scalar line-by-line transcriptions of MC_CLOUD_FRACTION and MC_PRECIP_PHASE (both `MC_REVP_ABV_CLDBASE` values) on random inputs reach all branches and match the vectorised port exactly (NOT real-Fortran validation).

Limitations: sampled (about 0.5%-5% of calls), 6 steps per date; only argument ranges these dates produce; MC_PRECIP_PHASE and MC_CLOUD_FRACTION receive their inputs as dumped (the callers' state, CCM/WCU/PRCP/LHP profiles, is not ported yet). No JAX version. Files: `clouds_helpers_ff.py`, `clouds_helpers_compare.py`, `tests/test_clouds_helpers_ff.py`, `instrumentation/CLOUDS2_helpers.f90.patch`, `instrumentation/ATM_DRV_clouds_helpers.f.patch`.

## D92: CONVECTIVE_MICROPHYSICS (scoping D-C3)

`clouds_helpers_ff.py` also ports `CONVECTIVE_MICROPHYSICS` (CLOUDS2.F90:5396-5584; the `CLD_AER_CDNC` arms are not compiled for this rundeck). Cumulus updraft/fallspeed partition of convective condensate into precipitating (`CONDP`) and transported-or-precipitating (`CONDP1`) parts via PRECIP_MP: three phase branches (liquid `TP>=TF`, pure ice `TP<=TIG`, mixed in between with the graupel fraction FG), the critical-size formulas DCG/DCI, and, for liquid, the data-dependent `do ITER=1,ITMAX-1` DCW search (exit when `VT>=0 and VT>=WV`, or `VT>WMAX`, else run to completion) done with per-record masks. `CONDIP`/`CONDGP` are only assigned in the mixed branch, elsewhere the entry values pass through (their entry values are dumped and checked).

Single-precision-literal audit: DCG/DCI use REAL(4) literals `19.3`, `11.72`, `2.7`, `2.439` and the exponent `.4` (not exactly representable; `.4` as REAL(4) is 0.4000000059604645). They are applied through `f4()`. This is a real D54-type hazard, shown by a test: using them as double literals changes the real-dump comparison of CONDP by more than 1e-9 relative (detected on nov26), versus 5e-15 with the correct treatment. The liquid loop uses `.4d0` (double), `.267d0`, `5.15D3`, `1.0225D6`, `7.55D7` (all double).

Instrumentation: same patch pair as D91 (call-site 1 of 1, CLOUDS2.F90:1938-1946, plus entry values of CONDIP/CONDGP). Record: 21 inputs, CONDIP/CONDGP entry values, 4 outputs (27 doubles). Sampling N=20 per step (about 32k real calls per step: 32,419 / 31,969 / 32,666 / 32,658 on nov26 steps 1-4 of the counts file). Records: 9,788 / 9,715 / 9,016 (nov26/dec01/jan01). Note: an initial run had the record length declared one too short (condgp dropped); caught by the loader's record-size assertion, fixed and rerun before any validation.

Validation (same compare script and test file as D91; 9,788+9,715+9,016 = 28,519 records): CONDP 99.55% bitwise, worst relative 5.4e-15; CONDP1 99.58%, worst relative 3.9e-15; CONDIP 99.93%, 8.6e-16; CONDGP 99.95%, 7.1e-16; worst absolute difference 4.3e-19 (values are of order 1e-6 to 1e-3 in the precip-density units used). Differences are 1-2 ulp amplified by the DCW search or DC powers, same exp/pow libm category as D54/D91; no phase-branch mismatches (the branch is decided from TP/TF/TIG, which are inputs, and the outputs match for all records).

Branch coverage (nov26/dec01/jan01): water 4,816/4,797/4,761, ice 3,748/3,645/3,066, mixed 1,224/1,273/1,189 (FG interior 1,167/1,210/1,131; FG forced to 0 by TLMIN/TLMIN1<=TF 57/63/58); LFRZ=0 6,311/6,258/6,115 and LFRZ>0 3,477/3,457/2,901; PLAND<0.5 4,431/4,404/4,417 and >=0.5 385/393/344 in the liquid branch; WV clamped to 0 17/19/23; DCG capped at 1e-2 1,047/1,092/874, DCI capped 3,722/3,890/3,290. NEVER exercised by the real windows: the liquid DCW search always exits through the fall-speed test (`VT>=WV`) on its first pass; the `VT>WMAX` exit never occurs, the run-to-completion case occurs once (jan01, upper search only); the TIG floor (`TIG<TI-10`) never; FG=1 never (and cannot occur in the mixed branch, since TP<TF; the FG clip to [0,1] is dead code there). These branches are covered only by a scalar line-by-line transcription of the Fortran on 4,000 random inputs (all exits, all three phases, TIG floor), which matches the vectorised port to 1e-12 (NOT real-Fortran validation).

Mutation checks on real data (detected): naive double literals for 19.3/11.72/2.7/2.439/.4; BY6 +0.1%; RHOW +0.1%; TF 273.0; `FLAM**4.` as `(x*x)*(x*x)` (detected only as a drop in bitwise fraction, 99.6% to 90%, still within 1e-12). Limitations: sampled (about 3%), 6 steps per date; ITMAX=50, WMAX=50, FITMAX=0.02 on every record (the port vectorises over per-record ITMAX/WMAX and is tested with other values only against the transcription). Files: as D91.

## D93: MASS_FLUX (scoping D-C4)

`fullfidelity/clouds_massflux_ff.py` ports `MASS_FLUX` (CLOUDS2.F90:5683-5804) and `THBAR` (shared/Utilities.F90:5-31); `QSAT`/`DQSATDT` are imported from `clouds_dq_ff.py` (D89). The cloud-base closure iteration (`do ITER=1,8`, `FPLUME=.25`, `DFP` halved each pass; exit when `|DMSE1|<=1d-3`, otherwise FPLUME -= or += DFP) is replicated per record with an active mask: exit freezes FPLUME/FMP2/DQSUM/DMSE1/TNX/QNX at the exit-iteration values, a record that never exits still has FPLUME updated once more after the 8th pass while FMP2/DQSUM stay from pass 8 (as in the Fortran), and the nested evaporation blocks (`DQSUM>0` upper layer, then `DQSUM>0` lower layer) are masks. Module state the Fortran reads implicitly (AIRM, BYAM, SM, QM, PLK, PL at LMIN..LMIN+2) is passed as explicit inputs; the residual uses the module parameter `SLHE=LHE*BYSHA` while the plume terms use the argument `SLH`. TNX/QNX are module scalars side-effected by MASS_FLUX; the port returns NaN when no pass set them and the compare checks them only where set (6% of records leave the module value stale from an earlier call).

Single-precision-literal audit: `.25`, `0.5`, `.5`, `1.`, `8` are exact; `1.d-3` is double; `DELTX = bymrat-1.` is exact in the 1. operand. No correction needed.

Instrumentation: same patch pair as D91, call site CLOUDS2.F90:1056 (MSTCNV; the only call). Dump `ffc_mf_<itime>.bin`: 31 doubles (header + 10 arguments + 11 module-state values + FPLUME, FMP2, DQSUM, final ITER, final DMSE1, TNX, QNX). Sampling N=3 plus the first two calls per step; real calls about 3.3k per step (3,364 / 3,301 / 3,336 / 3,364 on nov26 steps 1-4); records 6,722 / 6,494 / 6,403 (nov26/dec01/jan01), total 19,619.

Validation (`clouds_massflux_compare.py`, `tests/test_clouds_massflux_ff.py`, 20 tests): FPLUME and FMP2 bitwise on all 19,619 records; the final iteration count (exit iteration 1..8 or 9 = no exit) equals the real one on all records, the final `DQSUM>0` branch agrees on all records, and the caller's `FPLUME<=.001` cycle test agrees on all records (0 branch flips); DQSUM 99.19% bitwise, worst abs 6.5e-18, worst rel 3.9e-14 (a cancelling difference; relative to the plume scale it is rounding level); DMSE1 99.53% bitwise, worst abs 8.7e-15 (it is the residual of a cancelling sum, relative difference meaningless near 0); TNX/QNX bitwise on all records where set. Tolerance in tests: FPLUME/FMP2 exact, DQSUM 1e-12 relative, DMSE1 1e-12 absolute.

Non-vacuity: exit iteration histogram is nonempty at every pass on every date (nov26: 1:16, 2:25, 3:44, 4:108, 5:219, 6:427, 7:748, 8:1,219, never-exits 3,916), so about 58% of calls use all 8 passes; final DQSUM>0 on 86% of records (evaporation blocks active), FPLUME<=.001 on 46/37/36 records (caller's `cycle`), FPLUME range 0.00098-0.499. Mutations on real data (detected): 7 or 9 passes, tolerance 1e-2 or 1e-4, nested lower-layer block removed, evaporation blocks removed, SLHE +1%, DELTX +0.1%. All records have LHX=LHE (the `ALT_MC_EXITS` option that can set LHX=LHS is not defined), so LHS inputs are untested. Limitations: sampled (about 1/3 of calls), 6 steps per date; THBAR is validated only inside MASS_FLUX (other callers in MSTCNV/LSCOND not yet). Files: `clouds_massflux_ff.py`, `clouds_massflux_compare.py`, `tests/test_clouds_massflux_ff.py`, patches as D91.

## D101: FILTER (sea-level-pressure filter) + SLP + SHAP1D + isotropslp + MAtoPMB (scoping item 11, dynamics)

`fullfidelity/dyn_filter_ff.py` ports the live path of `FILTER` (`ATMDYN.f` 1418-1624; live is MFILTR=1 with the SLP branch, 1463-1596): `SLP` (`shared/Utilities.F90` 98-127), `SHAP1D(8,X)` (`ATMDYN.f` 1958-1990), `isotropslp` (1814-1836, reusing `shap1` from `dyn_isotropuv_ff.py`), the per-row loop (PEDN(1)=X/Y, limits 0.9882/1.0118 of the old value, row-mass conservation by PDIF), the new MA from the filtered PEDN(1) (rows J=2..JM-1, layers with MFRAC>0), `MAtoPMB` (`ATM_UTILS.f` 241-284: MASUM, PEDN, PMID, PDSIG, PK, P; PEK, byMA and the ATMSRF copies are not ported), and the scaling of T (by PKOLD/PK), Q, QCL, QCI and QMOM (by MABEF/MA). Dead and not ported: the temperature-stratification filter (MFILTR>=2), the non-SLP pressure branch (pfilter_using_slp is true for planet_name='Earth'), V2_PSURF_FILTER, TRACERS_ON. The energy fix (`getTotalEnergy` before/after, `addEnergyAsDiffuseHeat`) is D102.
- Instrumentation (new patches, `diff -u` against pristine files, local hunks, units 1120-1125 re-grepped over model/ and all existing patch files: unused; they apply to pristine files and after the existing `ATM_DRV_dynA/B/C` and `ATMDYN_dynA/aflux_pgf_advecv/aadvt` patches, checked with `patch`): `ATMDYN_filter.f.patch` (hooks in FILTER: entry, after the SLP loop, after SHAP1D, after isotropslp, after the row loop, exit; plus a hook in CONSERV_KE before the regrid), `ATM_UTILS_filter.f.patch` (D102 hooks), `ATM_DRV_dynD.f.patch` (helpers `ffdd_*`, anchored after the unchanged `end subroutine daily_atm`; reads FFD_START/FFD_NSTEP itself). Dumps per date (big-endian float64 streams, 37 files, 342 MB): `ffd_filt_consts.bin` (83.7 kB: constants, geometry, MFIX/MFRAC, ZATMO, AXYP), per step `ffd_filt_<itime>_in.bin` (15.95 MB: PEDN(1), TSAVG, MA, PK, T, Q, QCL, QCI, QMOM at FILTER entry), `ffd_slp_<itime>.bin` (132 kB: X,Y after the SLP loop, X after SHAP1D, X after isotropslp, PEDN(1) after the row loop) and `ffd_filt_<itime>_out.bin` (18.07 MB: PEDN(1..LM+1), PMID, PK, MA, MASUM, T, Q, QCL, QCI, QMOM at exit). Layouts: `dyn_filter_compare.py` docstring.
- Validation (`dyn_filter_compare.py`, `tests/test_dyn_filter_ff.py`, 79 tests): 18 real calls (1 FILTER call per step, 6 steps, nov26 33312-33317, dec01 33552-33557, jan01 17520-17525), all 72x46 (x40 layers) points. With the Intel libimf `pow` (`imf_pow=True`; the same bridge as D96/D97) **every quantity is bitwise identical on all 18 calls (max abs difference 0.0)**: SLP output X and Y, SHAP1D, isotropslp, the row loop (stage by stage from recorded inputs), the full PEDN(1) chain, MAtoPMB (PEDN, PMID, PK, MASUM), the final PEDN, PMID, PK, MA, MASUM, T, Q, QCL, QCI and all nine QMOM moments (132,480/132,480 T points exact on every call). With numpy `**` (default) SHAP1D, isotropslp, the row loop and everything in D102 are still exactly 0.0; the only differences come from `(1.+BZBYT)**GBYRB` in SLP (1-2 of 3,312 cells differ in 6 of 18 calls, max 2.3e-13 mb, relative 2.2e-16) and `PMID**KAPA` in MAtoPMB (PK max 8.9e-16, about 0.04 % of cells as in D96) and their propagation: final PEDN <= 3.4e-13 mb, MA <= 1.1e-13 kg/m2, MASUM <= 5.5e-12, T <= 1.4e-14, Q <= 2.6e-18, QMOM <= 1.3e-18. The tests bound these (`test_numpy_pow_within_1ulp_scale`).
- Branches exercised in the 18 calls (counts of cells, 59,616 in all): SLP ZS=0 (ocean) 39,006; TAS<290.5 with TSL>290.5 (BETA recomputed) 1,188; TAS>290.5 with TSL>290.5 (warm TASn) 3,536; TAS<255 (cold TASn) 8,821; all 20,610 non-ocean cells take the `PS*(1+BZBYT)**GBYRB` branch. isotropslp touches rows J=2,3,JM-1,JM (COSP<0.15), each with n=1 sub-iteration. The filter changes PEDN(1) by up to 1.7-3.0 mb per call (about 1e-3 relative).
- Branches NEVER exercised in the windows (ported, tested only by hand-derived tests clearly labelled NOT validated against the real Fortran): the SLP `exp` branch (BETA<=1e-6, i.e. TAS within about 1e-6*ZS of 290.5 K), the +-1.18 % clip limits of the row loop (0 low and 0 high hits in 59,616 cells; the PDIF shift itself is real: up to 0.0127 mb), and `shap1` with n>1 inside isotropslp (needs fac>=1; fac<1 here). The hand tests check SLP against independent closed forms, the clip arithmetic cell by cell with row-mass conservation, and isotropslp against a scalar transcription of `shap1` for forced n>1.
- Mutation tests (all detected): SHAP1D order 6 instead of 8, wrong BYIM in PDIF, no-conservation, mutated MTOP in PE, wrong DXYV in the regrid, negated energy delta, changed PMTOP, reversed GLOBALSUM order, reversed PE layer sum.
- Limitations: three 6-step windows from three restarts only; serial domain (no MPI halo branches, `J_0S/J_1S`=2/JM-1 assumed); numpy only (the row loop is a Python scalar loop over 72x44 cells; a vectorised cumsum version, `slp_filter_pedn_fast`, is bitwise equal and tested); rows J=1 and JM of the SLP stage arrays are undefined memory in the Fortran and never used; the SLP and `**KAPA` pow bitwise results need the Intel runtime (libimf) and are skipped elsewhere; `exp` has no libimf bridge. Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session, 2026-10-05; not yet reviewed.

## D102: energy functions getTotalEnergy / CONSERV_KE / REGRID_BtoA_EXT / conserv_PE / GLOBALSUM / addEnergyAsDiffuseHeat (scoping item 11, dynamics)

Also in `fullfidelity/dyn_filter_ff.py` (same compare script and test file as D101): `getTotalEnergy` (`ATM_UTILS.f` 579-605) = GLOBALSUM_IJ of ((KE+PE)*AXYP)/AREAG, with `CONSERV_KE` (`ATMDYN.f` 2250-2281: B-grid sum over layers of ((MA(I,J-1)+MA(I+1,J-1))*DXYN(J-1)+(MA(I,J)+MA(I+1,J))*DXYS(J))*(U^2+V^2)*.25, rows J=2..JM), `regrid_btoa_ext` (2744-2779: south-pole row from row 2, interior with RAPVS/RAPVN, north-pole row from the B-grid row JM, then times byAXYP), `conserv_PE` (`DIAG.f` 1131-1165: ZATMO*(MASUM+MTOP)+SHA*Sum(T*PK*MA), poles replicated from I=1), `globalSum_IJ` (`GlobalSum_mod.F90` 166-248: sequential zonal sums, then sum over J), and `addEnergyAsDiffuseHeat` (`ATM_UTILS.f` 607-633: EDIFF=DELTA/((PSF-PMTOP)*SHA*MB2KG), T(:,:,L)-=EDIFF/PK(L) for all I,J,L).
- Instrumentation: the D101 patch set (`ATM_UTILS_filter.f.patch`, hook at the end of getTotalEnergy and around addEnergyAsDiffuseHeat; `ffdd_ke` in CONSERV_KE saves the B-grid field before the regrid), units 1124 (energy call records, `ffd_filt_<itime>_te1.bin` initial and `_te2.bin` final: MASUM, MA, PK, T, U, V inputs and KEB, KEIJ, PEIJ, TEIJ, TOTAL, 5.43 MB each) and 1125 (`ffd_filt_<itime>_eadd.bin`: DELTAENERGY, EDIFF, T before, PK, T after, 3.18 MB). 2 getTotalEnergy calls and 1 addEnergyAsDiffuseHeat call per step.
- Validation: the same 18 real steps (36 getTotalEnergy and 18 addEnergy calls). **KEB (B-grid, rows 2..JM), KEIJ (after regrid and byAXYP), PEIJ, TEIJ, TOTAL, EDIFF and the T field after the heat addition are bitwise identical (0.0) on all calls, with plain numpy (no libimf needed: these functions contain no pow/exp), and the recorded DELTAENERGY equals TE2-TE1 exactly.** The whole FILTER chain energy values E0, E1 equal the recorded totals exactly with libimf pow. Sequential sums matter: reversing the J order of the global sum or the layer order of the PE sum gives different bits (tested). The energy fix is small but real: EDIFF about -6e-5 to -7.7e-5 K per call (TE changes by well under 1e-3 relative).
- Not exercised: nothing in these functions branches on data except the pole conventions, which are exercised (pole rows of KE/PE are the polar-box formulas). MPI/halo variants and the SCM dummy versions (`ATMDYN_SCM_EXT.f`, `ATM_DUM.f`) are not live and not ported.
- Limitations as D101. Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session, 2026-10-05; not yet reviewed.

## D107: LSCOND particle size / optical thickness tail block (scoping D-C5) -- bitwise-exact with libimf on all 14,262 real column calls

Ported `CLOUDS2.F90:4927-5285` (from `SNdO = 59.68d0/(RWCLDOX**3)` to the end of LSCOND): the `OPTICAL_THICKNESS` loop
(`do L=1,LMCLD`: RCLD/RCLDE from WTEM, CSIZEL, CSIZELIP, TAUSSL/TAUSSLIP incl. the `use_vmp` precip arm, WMSUM) and the final
`do L=1,LMCLD` loop (CLDSV1, SVLHXL reset, boundary-layer / free-troposphere CLDSSL-TAUSSL rescaling with CKIJ and TAUMCL,
QLss/QIss). One column per call, explicit L loops, plain Python floats (`fullfidelity/clouds_lscond_size_ff.py`). The
CLD_AER_CDNC / AIE_DIAG_FIX_MET / COSP sub-blocks are not compiled in P2SAoM40 and are not ported; the non-VMP arm (dead,
`use_vmp=1` in the rundeck) is coded for `use_vmp=False` but unvalidated.
Validation: new instrumentation (`CLOUDS2_lscond.f90.patch` + `ATM_DRV_clouds_lscond.f.patch`, units 1170-1173) records the
block entry (23 LM arrays + LHP + 12 scalars) and exit (13 arrays + WMSUM) of every 4th LSCOND call of each step (a different
column subset each step): 4,754 columns per date x 3 dates = 14,262 columns (190,160 values per output field per date).
Result: with the platform libm, every output field agrees to max_rel <= 8.2e-14 (99.987-100.000% of values bitwise; 29-43 of
4,754 records per date have a last-bit difference); with Intel libimf `exp`/`pow` (optional ctypes bridge, `--imf`) ALL outputs are
bitwise equal on all 14,262 columns. Branch flips (SVLHXL reset, TAUSSL=0 / =100, TAUSSLIP=0, CLDSSL=0, QCLX=0): 0.
Single-precision-literal audit: none needed in this block (all non-integer literals carry d0). Branches exercised by the real data
(layer visits, 3 dates): liquid 195,660, ice 217,938, VMP ice precip under liquid cloud 35,807, TAUSSL cap 448, CSIZEL cap 741,
RWMAX cap 164, bmax>=.95 7,601, BL rescaling 37,499, free-troposphere rescaling 317,087, TAUMCL skip 58,060. Never exercised
by real data: negative TAUSS (`neg_tau`), the `stop_model('VMP: should not be here')` arm, the non-VMP arm -- covered only by
clearly labelled SYNTHETIC tests.
Files: `fullfidelity/clouds_lscond_size_ff.py`, `clouds_lscond_io.py`, `clouds_lscond_size_compare.py` (`--imf`),
`tests/test_clouds_lscond_size_ff.py` (33 passed, incl. 10 input mutations and 2 algorithm mutations, all detected).
Dumps: `ff_data/<date>/ffc_ls_tail_<itime>.bin` (6 files per date, ~9.5 MB each) + `ffc_ls_consts.txt`.
Owner: (project lead to assign). Review date: before the first radiation hand-off delta.

## D108: LSCOND main layer loop (scoping D-C6, part 1) -- bitwise-exact with libimf on all 5,706 real column calls

Ported the initialisation and `CLOUD_FORMATION: do L=LMCLD,1,-1` (CLOUDS2.F90:3211-4634 live code, use_vmp arm): RH00/RHF, VMP phase
(LHX/LHP from TMIN_WATER/TF), RH1 (incl. the Karcher-Lohmann ice formula), phase-change bookkeeping, autoconversion with the
VMP wtliq interpolation, evaporation of precip ER, condensation/evaporation QHEATL/QHEATI, cloud-water evaporation EC, PREBAR/PREICE/LHP
precipitation carry and HPHASE, QNEW and moment scaling, the RH>1 `get_dq_cond` pass, rain-out of cloud water, CLEARA/CLDSSL/
CLDSAVL and the HCNDSS/SSHR/DQLSC diagnostics, PRCPSS (`fullfidelity/clouds_lscond_ff.py::lscond_main`). Calls the D89
`get_dq_cond/get_dq_evap` by import. TRACERS_*, CLD_AER_CDNC, do_blU00 (=0) and debug blocks are not live and not ported.
Validation: the whole LSCOND call boundary (all inputs: 18 LM arrays + RNDSSL + PRECNVL + RA + 22 scalars; module state in/out: 24 LM
arrays + LHP + prebar1 + QMOM/SMOM (9x40) + UM/VM (4x40)) is recorded for every 10th LSCOND call of each step
(`ffc_ls_bnd_`; 317 columns/step, 1,902 per date, 5,706 total, 12.9 MB per step) plus a checkpoint after the main loop before CTEI
(`ffc_ls_mid_`, state + 9 local LM arrays + PREBAR/PREICE + PRCPSS/HCNDSS; 6.4 MB per step). N=10 and sizes are documented in
`clouds_lscond_io.py`. Result (entry -> mid checkpoint): with Intel libimf exp/pow: 100.0000% bitwise on all 4,741,686 compared
values per date (state, locals, PRCPSS, HCNDSS), 0 mismatches; with platform libm: 99.97-99.98% bitwise, max_rel <= 3.5e-9
except CLEARA/CLDSAVL/CLDSSL/RHF, where `CLEARA = DSQRT((1-RH)/(...))` amplifies a last-bit RH difference to at most 1.2e-7 absolute
(32/32/35 layer mismatches per date, every one coincident with a differing RH and within 4x the sqrt-amplified bound).
Single-precision literals reproduced with f4(): `-0.2` (cm0*10.**(-0.2*vdef)), `238.16`, `207.83`, `.999999`; `.001` in
WMUI=WMUIX*.001 (taken from the dump: 1.0000000475e-6 shows the REAL(4) rounding). Mutation checks show the f4(0.2) and f4(207.83)
corrections matter (77,920 / 10,257 mismatching values over 700 columns when replaced by the double literal); 238.16 and .999999 are NOT
observable on this data (no real value in the 4e-6 / 1e-6 wide gap) and rest on the source reading.
Branch counts over the real data (layer visits, 3 dates, 165,474 layers): ice-forced 87,125, water 78,349, form_clouds 27,906,
no_form 137,568, autoconversion cap CM 834, TEM cap 10,153, RH>1 condensation 4,339, rain-out liquid 878 / ice 4, phase change
LE->LS 29 / LS->LE 23, oldlat LS->LE 16,471, evaporation of precip ER (qcl 25,563, ice-precip 742, rh 1,601), no-form cloud
evaporation liquid 1,746 / ice 97, HPHASE melt 2,187 / freeze 25. NEVER exercised by real data: OLDLAT=LHE with LHX=LHS, ER>ERMAX clip
(and the unreachable ER<0 clip), QNEW<0 (and IERR=1), -- first, second and third are covered by SYNTHETIC tests (not validated
against Fortran); IERR=1 was not reachable even synthetically. The non-VMP arm raises NotImplementedError (dead in this rundeck).
Branch flips from last-bit differences (SVLHXL, LHP, QCLX/QCIX=0, CTEI-mixed layer, TAUSSL/CLDSSL=0, IERR): 0 in libm and libimf modes.

## D109: LSCOND CTEI + remainder (scoping D-C6, part 2) -- whole LSCOND column call bitwise-exact with libimf, 5,706 columns

Ported `CLOUD_TOP_ENTRAINMENT: do L=LMCLD-1,1,-1` (CLOUDS2.F90:4647-4924): SM/QM/WMXM set-up, the cycle conditions (cloud above,
clear below, CKR>CKM, CK<CKR, FPMAX, DSE>=DSEC), SIGK/EXPST/CKIJ, the bounded `do ITER=1,9` FPLUME iteration with `exit`,
post-mix SM/QM/WM updates, CTMIX of SM/QM and their moments (with the FSSL-ratio rescaling), U/V momentum mixing, temperature
correction for the phase difference between the layers, CLEARA/CLDSSL/CLDSAVL and HCNDSS/SSHR/DQLSC/DCTEI. `lscond()` chains
main loop + CTEI + the D107 tail (`lscond_main`, `lscond_ctei`, `lscond_tail`). Key finding: the integer power `**5` in the SIGK
formula is a libm `pow(x,5.)` call in the real build; x*x*x*x*x or ((x*x)*(x*x))*x differ in about 1 of 100 mixed layers (detected by a mutation test).
Validation: whole call (entry -> exit, 5,706 columns, 3,967,572 compared values per date): libimf mode 100.0000% bitwise, 0 branch
flips; platform libm: 99.96-99.97% bitwise, max_rel <= 2.8e-11 except the sqrt-amplified CLEARA family (<= 1.2e-7 abs). CTEI alone
(mid checkpoint -> tail-block entry, 948 calls per date present in both files): libimf 100% bitwise; libm 99.998-100%, max_rel <= 8e-12.
CTEI statistics (3 dates): 159,768 layer pairs visited, 2,533 mixed (2,414 iterations exited, 119 ran all 9 iterations, 3,678
iterations in total), 233 mixed pairs with FSSL(L)/FSSL(L+1)!=1, CKIJ set (L=1) 354, cycle reasons: cloud above 24,943, clear
layer 125,924, CK<CKR 6,013, DSE>=DSEC 355. NEVER exercised: CKR>CKM cycle, FPMAX<=0 cycle, FMASS clipped to the harmonic cap
(no synthetic coverage found either). UM/VM are validated for K=1..4 (all columns except the two pole columns have KMAX=4; the pole
column's other 68 K entries are not recorded).
Tests: `tests/test_clouds_lscond_ff.py` (45 tests: layout, main loop, whole call, CTEI, chain, branch coverage with explicit
never-exercised list, 10 constant mutations + COEEC/pow5/iteration-cap/f4 mutations, unit and SYNTHETIC tests); both new test files:
78 passed, 89 s.
Files: `fullfidelity/clouds_lscond_ff.py`, `clouds_lscond_io.py`, `clouds_lscond_compare.py` (`--imf`, `--quiet`, `--max N`),
`instrumentation/CLOUDS2_lscond.f90.patch`, `instrumentation/ATM_DRV_clouds_lscond.f.patch`.
Owner: (project lead to assign). Review date: before the first radiation hand-off delta.

## D114-D117: dynamics coupling glue (CALC_TROP/tropwmo, COMPUTE_WSAVE, calc_kea_3d, regrid_btoa_3d, DISSIP, CONSERV_SE + energy-fix block, PGRAD_PBL, recalc_agrid_uv, DAILY_ATMDYN) ported and validated bitwise

Status: all nine routines bitwise exact (max |diff| = 0, 0 mismatching cells) on 3 dates x 6 steps, including every call of recalc_agrid_uv (6-7 per step, call sites 1-5 identified) and one REAL end-of-day DAILY_ATMDYN call per date (the dump runs were extended to 50 steps to cross the day boundary). Same result with numpy pow and with the libimf bridge (tropwmo), and with recorded or analytic geometry (dyn_geom_ff + new rapj/idij/idjj/sinip/cosip/bydxp). 43 tests pass (graphcast-env python, ~16 s).
Files: fullfidelity/dyn_glue_ff.py, dyn_glue_io.py, dyn_glue_compare.py, tests/test_dyn_glue_ff.py; patches instrumentation/ATM_DRV_dynF.f.patch (helpers ffdg_*, units 1240-1251), ATMDYN_glue.f.patch, ATM_UTILS_glue.f.patch, ATMDYN_COM_glue.f90.patch, SURFACE_glue.f.patch, ATURB_glue.f.patch, CLOUDS2_DRV_glue.f90.patch, DIAG_glue.f.patch.
Reused by import from dyn_filter_ff (not duplicated): seqsum, conserv_ke (CONSERV_KE + regrid_btoa_ext), global_sum_ij, matopmb.
Real-data facts: tropopause level varies over >=5 levels; 57,060 columns checked; tropwmo branches exercised: pressure-limit exit, GISS failsafe, WMO candidate (61,897), zptf=0 (579), ldtdz false (57,910), jj cycle (3,987), jj valid exit (57,060), jj discard (4,837), ltropp overwritten after failsafe (22,264). NOT exercised in real data (hand tests only, labelled NOT VALIDATED): no pressure-limit exit, jj loop ended without exit, default ltropp=iplimt-1/ierr=1 (no "TROPWMO error" in any run), DAILY early returns and the ITIME==ITIMEI path. Dead/not hooked: ATM_DRV.f:729 and :1361 recalc_agrid_uv (cold start / relayer), STRATDYN regrid_btoa_3d callers, cubed-sphere PGRAD_PBL. regrid_btoa_ext/conserv_KE/conserv_PE were already covered by D101/D102.
Real DAILY_ATMDYN deltam: nov26 -3.27e-11, dec01 -3.82e-11, jan01 -4.18e-11 (MA change ~2e-12 per layer) - small but the real code path.
Limitations: only 6-step windows (DAILY: 1 call/date); aij diagnostics inside these routines are not ported; pole cells I>1 of PGRAD_PBL/recalc outputs are unset in Fortran and excluded.
Owner: Glenn Tamkin. Source: modelE2_planet_2.0 model/ATM_UTILS.f, ATMDYN.f, ATMDYN_COM.F90, DIAG.f, ATM_DRV.f. Review: next atmosphere session.

## D103: `AADVQ0` (moisture-advection flux-cycle preparation) ported, exact on all 18 real calls (2026-10-05)

`AADVQ0` (QUS3D.f:292-829), with `XSTEP` (835-893) and `ZSTEP` (896-928), ported in `fullfidelity/dyn_aadvq_ff.py`
(`aadvq0`, `xstep_rows`, `zstep`). It picks `ncyc` (<=10), per-level `ncycxy`, per-row `nstepx`, the z-extra columns
(`lminzij/lmaxzij/nstepz_extra/mw_extra`), rescales `MUs/MVs/MWs` (MV shifted one row, `pv_south` saved, `mw_extra` subtracted),
and builds the flow-out-both-sides tables. Validated per call (3 dates x 6 steps = 18 QDYNAM calls) against
`ffd_qdyn_<it>_q0.bin`: **all integer outputs equal, scaled MUs/MVs/MWs/pv_south max difference 0.0, checkflux tables identical.**
Two single-precision traps replicated: `byn = 1./ncyc` and `byNXY = 1./ncycxy(l)` are REAL*4 divisions promoted to double;
`mrat_limy = 0.20` is a REAL*4 literal (0.20000000298023224). Serial: both poles local, halo/pack code skipped (no state effect).
Coverage (measured): in the real windows ncyc=1 and ncycxy=1 on every level in all 18 calls, no z-extra column, X nstep reaches 2
(a few rows); so the cycle branches are validated by stress dumps (D104/D105) and a standalone ifort build of the real QUS3D.f (below).
Branch note: for j=1 the Fortran reads `mv(i,0,l)` out of bounds in the z-extra div1d; the port uses the memory neighbour (0 for l>1).
`ZSTEP` has no iteration cap in Fortran (a layer emptied to exactly 0 mass never converges; int32 wrap ends it); the port raises after 200000.
Instrumentation: `ATMDYN_qdynam.f.patch`, `QUS3D_qdynam.f.patch` (the live routines are in QUS3D.f, not QUS_DRV.f), `ATM_DRV_dynE.f.patch`
(`diff -u` vs pristine, local hunks, verified to apply to pristine); units 1140-1144, 1149 (grepped over tree and patches: unused).
Owner: project owner; source ModelE2_planet_2.0 (read-only).

## D104: moisture-advection sweeps `aadvqx/y/z`, `checkflux`, `aadvqz_column`, cycle loop `AADVQ` ported, exact (2026-10-05)

`AADVQ` (QUS3D.f:53-289, qlimit=.TRUE., tname 'q'): cycle loop `ncyc`, per level (ncycxy sub-cycles) y-checkflux, `aadvqy`, `aadvqx`,
z-checkflux, then the interface sweep `aadvqz` between layers L-1 and L carrying (`mwdn, fdn, fdn0, fmomdn`) sequentially in L (L=LM+1
closes the top), and the extra column advection `aadvqz_column` for z-extra columns. One expression tree (`face_flux`, `cell_update`) with
the direction permutation as data serves X/Y/Z; the sweeps are vectorised (every face flux uses the pre-update state of its two cells) and
tested against literal scalar transcriptions (`aadvqx_rowloop`, `aadvqy_loop`, `aadvqz_loop`, `aadvqz_column`). `fracm**3` = `np.power`.
Polar caps, `sbf/sbm/sfbm` and `scf/scm/sfcm` diagnostics accumulate left to right; `scf3d` accumulated per level.
Validation, per call, against 199 recorded stages (after y-checkflux, Y, X per level; after z-checkflux; the vertical carry after every
AADVQZ; full state after the L loop; AADVQ outputs RM/RMOM/MMA, the six zonal diagnostics, SCF3D): **max difference 0.0 on every array of
all 18 real calls (1,324,800 Q/QMOM values per call all equal).** Real coverage: limiter branches all hit (x: 1656 pos<0, 455 neg>0, 367 pos>rm,
30 neg<-rm over the windows; y and z likewise), y-checkflux fires 35 times, z-checkflux 21 times, x-checkflux never.
Stress (real Fortran, env `FFD_QDYN_STRESS=8` scales MUs/MVs/MWs of the call; SYNTHETIC input perturbation, model keeps running):
18 calls with ncyc 4 and 6, up to 3 extra z steps, 57 z-extra columns, x-checkflux fired 17 times: **all 18 bitwise exact incl. 800-1200
checkpoints per call** (`ffd_qdynS8_*`). Stress x16 reached ncyc 7-8 then ncyc>ncmax (model's own stop). ncycxy>1 was never reached by any
flux scaling of the real model state (u, v, w scaled separately or together up to x24: ncyc>ncmax first).
Standalone harness (`instrumentation/qus3d_standalone_{stubs,drv}.f90`; the REAL QUS3D.f compiled unmodified with serial stubs for the
domain-decomposition/ATM_COM/GEOM modules, same ifort flags): 7 SYNTHETIC cases (real state, hand-designed fluxes) -- zero flux, ncyc=2,
ncycxy=2 on one level, a uniform zonal flux giving nstepx=5, a combination, real fluxes x(3,3,4) with ncyc=3, and the real call -- all
bitwise exact against the port; the ncyc>ncmax case stops in both (stop_model / QusError). Hand-derived expectations asserted (ncyc==2,
ncycxy[l]==2 and 1 elsewhere). Data `ff_data/qus3d_harness`. NOT validated against any model run: ncycxy>1 (synthetic only),
`ncycxy>ncmax`, `too many steps in xstep` (unit test of ierr only), and a hand-built z-extra case (degenerate, see D103).

## D105: `QDYNAM` driver glue ported, exact on all 18 real calls (2026-10-05)

`QDYNAM` (ATMDYN.f:3039-3101) in `dyn_aadvq_ff.qdynam`: `MB = MAOLD*KG2MB*AXYP`, `AADVQ0`, concentration -> mass (`Q*MB`, `QMOM*MB`),
`AADVQ`, back conversion with the UPDATED `MMA` (`1/MMA` multiply). Recorded Q/QMOM at exit (`ffd_qdyn_<it>_fin.bin`): **exact equality on
all 1,324,800 values of every call, 3 dates x 6 steps.** `ain` (mass units at AADVQ entry) and MB (dumped) equal. The `AGC` diagnostic adds
are not ported (diagnostics only); `safv/sbfv` module arrays are not ported (unused outside AADVQ). Tests: `tests/test_dyn_aadvq_ff.py` (46 tests, ~70 s, all pass)
(mutations: checkflux off, X/Y spec swap, wrong carry, wrong KG2MB, REAL*8 byn -- all detected; the byn mutation only on the ncyc=6 stress call jan01 17520).
Dump size per date: 37 files, ~750 MB (6 x {in 16 MB, q0 5 MB, ain 11 MB, ck 52 MB (ncyc=1), out 13 MB, fin 11 MB}); no subsampling.
Stress S8: ~750-1100 MB per date (ncyc 4-6 gives 800-1200 checkpoints).

## D106: remaining QDYNAM pieces

The scoping doc lists nothing in QDYNAM beyond D103-D105 (`AADVQ0`, `AADVQ`, `aadvqx/y/z`, `aadvqz_column`, `XSTEP/ZSTEP`, `checkflux`).
Not ported, by decision: the `*2` unlimited variants (dead: qlimit=.TRUE.), upwind-halo pack tables (serial no-ops), `AGC` zonal diagnostic
adds (`jl_totntlh` etc.; diagnostics), `TrDYNAM` (tracers off). Remaining unexercised in model runs: ncycxy>1 and ncyc>=7 (synthetic/harness or stress only).

## D110-D113: `MSTCNV` (moist convection, CLOUDS2.F90:432-3207, 903 live lines) ported per column; validated on 19,374 real calls (3 dates)

**D110 (instrumentation).** New patches `instrumentation/CLOUDS2_mstcnv.f90.patch` + `ATM_DRV_clouds_mstcnv.f.patch` (diff -u vs pristine, local hunks; units 1200-1202, 1203-1239 spare; build in `<scratch>/mE_cloud4/mE2`). Call-boundary records (`ffc_mc_cols_<itime>.bin`, 1974 input + 2616 output doubles each: all arrays MSTCNV reads and every output it writes) for every 2nd convective call, every 25th non-convective call, and every call that has checkpoints; stage checkpoints (`ffc_mc_ck_<itime>.bin`, stages 1-9: DMSE test, MASS_FLUX result, plume set-up, end of ascent, downdraft, subsidence, before precip loop, end of cloud type, post-processing) for convective calls with mod(ncall,24)==0. About 66 % of the 3,165-3,169 calls per step (J=2..45; poles not called) are convective. KMAX is 4 on this grid so all K are dumped. Per date: 6,399-6,540 boundary records (6,111-6,265 convective), 479-526 checkpoint calls with 18,801-20,743 events, ~390 MB. Layouts: `clouds_mstcnv_io.py` (the Fortran constructors were generated from its tables).

**D111 (port, `clouds_mstcnv_ff.py`).** Statement-by-statement per-column function (1-based padded arrays, Fortran evaluation order, every cycle/exit), reusing get_dq_cond/evap, QSAT, THBAR, MASS_FLUX, PRECIP_MP, MC_CLOUD_FRACTION, MC_PRECIP_PHASE, CONVECTIVE_MICROPHYSICS, ANVIL_OPTICAL_THICKNESS, adv1d (with real limitq for Q). REAL(4) literals replicated: .001, .95 (x3), 0.7, .33, .050/.001 (U00L), 1.E-20. Stages validated in order against the checkpoints, then the boundary.

**D112 (validation, `clouds_mstcnv_compare.py`, `tests/test_clouds_mstcnv_ff.py`, 31 passed).**
- With exp/pow routed through the build's Intel libimf (`--imf`): bitwise-identical records nov26 6,528/6,540, dec01 6,389/6,399, jan01 6,429/6,435 (99.8 %); the 22 others differ by <= 4.9e-13 of the field scale (max_abs ~7.6e-11 on SMOM); 0 threshold flips; 0 decision-path mismatches (LMCMIN/LMCMAX/MCCONT/IERR/LERR); all 9 checkpoint stages follow the identical event sequence (59,000 events); stages 1-9 bitwise on every event for nov26 and dec01, <= 6e-15 (field scale) for jan01 (first residual: stage 1 DMSE 4e-15 abs, probably a vectorised SVML exp/pow variant in the real build; not verified).
- numpy/glibc mode: 5,282/6,540, 5,096/6,399, 5,227/6,435 records bitwise (~80 %); smooth fields <= 6.3e-11 of scale; decision paths and event sequences identical. Branch flips caused by 1-ulp noise: 48/39/44 records (<1 %) where a cancellation residue changes sign (SVWMXL 1e-22 vs 0, PRECNVL, CLDMCL>0) and 26/15/23 records (0.4 %) where CSIZEL/TAUMCL/CLDMCL deviate up to 7.8 % of field scale (ill-conditioning where WCU2 ~ 0 at plume top). All vanish with libimf. Stage localisation: the first inexact field in numpy mode is always COND (get_dq_cond, exp) in stage 4.
- Mutation checks (real-dump comparison must fail): PGRAD as double (um/vm 4.7e-10 vs 3e-15 baseline), .95 as double (3e-9), ksub forced 1 (1.5e-2, on the ksub=2 calls from the checkpoint dumps), MC_NEW_DDRFT_THETAV=0, MC_ENTR_MASS_LIM_PLUME=0, MC_FDDRT=1, entrainment constants, U00b, radiusl, MC_REVP_ABV_CLDBASE=0 all detected. Not detectable: the REAL(4) `.001` tests (differ only for values in a 5e-11 window) -- kept from source reading.
- Hand-derived tests: non-convective columns untouched; column integral of UM/VM conserved (real and port).

**D113 (not exercised in the 3 x 6-step windows).** limitq (positivity limiter, Q subsidence advection) never triggers (no limitq branch counted; tests show removing it changes nothing) so it is validated only by the D99/D100 standalone harness; IERR/LERR are always 0 (adv1d error returns, the negative-cloud-cover stop); the `SVLATL != VLAT` heat correction (svlat_phase_diff) never fired; plume reaching L=LM (out-of-range U_0(.,LM+1)); polar columns (not called); MC_NEW_DDRFT_THETAV=0, MC_ENTR_MASS_LIM_PLUME=0, MC_REVP_ABV_CLDBASE=0 arms (compile/runtime defaults not used); tracer, CLD_AER_CDNC, SCM, WEAKER_MC_LIMITS, COSP and lightning code (undefined in this build, not ported); the polar/KMAX=72 momentum case and ISC are untested; ksub=2 occurs in only ~1 % of calls (1 checkpoint block per date) so the 2-sub-step path rests on few examples; other seasons/columns than 3 dates x 6 steps. AIRXL/PRHEAT of non-convective calls are stale module state and excluded.

## D121: chained dynamics step plan (`scoping/ATM_DYNAMICS_CHAIN_PLAN.md`)

Read DYNAM (ATMDYN.f:186-390) and atm_phase1 (ATM_DRV.f:88-235) line by line and wrote the exact live call order, the argument bindings of every
call, which existing port implements each call, the data carried between the 5 leapfrog passes (1 forward 300 s, 1 backward 450 s, even 900 s,
odd 900 s, even 900 s; NS = 4,4,4,3,2) and every gap. Gaps found: the glue (MUs/MVs/MWs zero/accumulate/scale by DTLF, MASUM re-sum, UX/UT/VX/VT/TZ/MEVEN/MODD1/TT/TZT
copies and averaging, PU/PV/SD scaling, MMA=MEVEN*AXYP, NS control flow), a full-field SDRAG wrapper, ADVECM's MAtoP side effect on module PEDN/PMID/PDSIG/PK/P
(read by the same pass's SDRAG), and, discovered while validating D122, **DIAGA's polar Q fill** (DIAG.f:221-233, called from DYNAM when MODDA<2, NDAA=13,
ITIMEI=16032) which is the only state change of Q inside DYNAM. Diagnostics (DIAGA0, DIAGB, EPFLUX (empty), COMPUTE_MASS_FLUX_DIAGS, AIJ/AJL) are not ported; "no
state feedback" is an inference except for the DIAGA polar fill, which the comparison exposed.

## D122: `dyn_step.py` chained dynamics step, bitwise against real Fortran; new state dumps

New files: `fullfidelity/dyn_step.py`, `instrumentation/ATM_DRV_dynG.f.patch` (unit 1300; dumps `ffd_state_<itime>_s1..s4.bin` = step start, DYNAM exit, end of the
dynamics block after QDYNAM + energy fix, physics exports; 24 files per date, 6 steps x 3 dates, ~34 MB per full-state record; new files only, `\cp -n`).
The dumps on disk did not contain QCL at step start nor TMOM/QMOM/MUs/MVs/MWs at the end, so the patch was needed. Scratch build `mE_dynG`.

`dyn_step.dyn_step(state, ctx, itime)` runs a plan (list of Stage objects whose argument bindings are the actual arguments of the Fortran CALLs) on a workspace that
mirrors the Fortran variables. Pass sequence: forward, backward, even, odd, even; accumulation of MUs/MVs/MWs only in the even passes; AADVT twice; SDRAG twice; five
isotropuv; MAtoPMB; flux scaling by DTLF; FLTRUV + fltry2 + angular-momentum fix; then COMPUTE_WSAVE, QCL/QCI rescale, QDYNAM, energy fix, CALC_TROP, PGRAD_PBL,
calc_kea_3d. Starting only from the real step-start state:

* with the Intel libimf pow bridge (`imf_pow=True`): **bitwise identical, 0 unequal elements, for all 18 steps** (nov26 33312-17, dec01 33552-57, jan01 17520-25), every
  field of s2 (DYNAM exit), s3 (end of block: U V T Q QCL QCI MA PEDN PMID PK P MASUM TMOM QMOM MUs MVs MWs GZ), s4 (PTROPO LTROPO WSAVE KEA DPDX/DPDY(_0) PHI) and the
  existing pre_condse dump (first 2 steps per date); every input and output of the 264 (stage, field) per-call series (AFLUX, ADVECM(+MAtoP), ADVECV, PGF, AADVT of
  each pass) is bitwise equal, i.e. 0 of the 264 series has any unequal element. Stage-boundary replay (real inputs except the port output of the preceding stages): 115
  series, all bitwise.
* with numpy pow: first non-exact quantity is PK=PMID**KAPA in pass-1 ADVECM (1 ulp in 916 of 2.38e6 cells), then it propagates. Worst scale-relative difference over
  18 steps: u 3.5e-13, v 3.8e-13, T 1.1e-15, Q 8.8e-15, QCL 5.0e-16, MA 9.6e-15, PK 7.3e-16, TMOM 5.1e-14, QMOM 3.0e-14, MUs/MVs/MWs 3.8e-14/4.7e-14/9.8e-14, GZ 3.4e-14,
  WSAVE 5.6e-14, KEA 4.6e-14, PTROPO 9.3e-16, LTROPO exact, PGRAD_PBL dpdx/dpdy/dpdx0/dpdy0 2.7e-11/2.2e-12/1.7e-13/1.9e-14 (near-cancelling gradient terms; element-wise
  relative up to 7e-7 where the field is near zero). No tolerance was loosened to obtain these numbers.

Not done: the loop through the physics (CONDSE, RADIA, SURFACE, ocean, ATM_DIFFUS, DISSIP, FILTER) back to the next step's start; DIAGA/DIAGB bodies other than the polar Q
fill; MAtoPMB's ATMSRF exports; NIdyn != 4 / 8-pass restart path (generated, not exercised).

## D123: compare script, tests, timing

`fullfidelity/dyn_step_compare.py` (`python dyn_step_compare.py [--both-pow|--numpy-pow] [--boundary] [--timing] [--date d] [--steps n]`), `tests/test_dyn_step_ff.py`
(66 tests, 264 s: plan-structure tests without dumps; 18 bitwise chain tests; 3 boundary replays; 3 numpy-pow tolerance tests with per-field bounds 2e-12 (1e-10 for the
four PGRAD_PBL terms) set above the measured worst values; 27 drop-a-stage and 7 swap-two-stages mutation tests plus post-DYNAM reorderings, all of which must make the
chain differ from the real state or raise; the DIAGA stage test on dec01 33555; dump-based tests skip if the dumps are absent, libimf-dependent ones skip without the Intel
runtime). Mutation notes: swaps of independent neighbours (advecv<->pscale, qscale<->qdynam, pgf<->pscale) and dropping PU/PV/SD scaling in non-even passes are
legitimately benign and were replaced by dependent pairs after the first run flagged them. Warm CPU timing, one chained step on forest204 (single process, numpy 2.2.4,
12 cores on the node, OMP_NUM_THREADS=1): libimf mode cold 3.5 s, warm 3.34-3.40 s (pgf 0.79, aadvt 0.58, qdynam 0.54, advecm 0.41, aflux 0.34,
advecv 0.22); numpy-pow mode warm 2.35-2.41 s. A 48-step model day of dynamics is therefore ~2.7 min of this path (extrapolation, not measured).

## D124: CONDSE glue plan (scoping/CLOUDS_CONDSE_PLAN.md)
Read CLOUDS2_DRV.F90:3-2854 (cpp-live text). Plan lists the 32-step live call order with lines, inputs, outputs, carried and hidden state (FRAC_AREA_CNV not reset at 2068-2078; CSIZMC above LMCMAX; CLDSAV1 not in restart), gaps. Not live/unused: ISC=0 block (1288-1315, dump records isc=0), RIS/RI1/RI2 (do_blU00=0), diagnostics, ISCCP.

## D125: CONDSE boundary dumps (units 1330-1333)
Patches instrumentation/CLOUDS2_DRV_condse.f90.patch + ATM_DRV_clouds_condse.f.patch (diff -u vs pristine, local hunks). Per step an entry file (85.8 MB) and an exit file (77.9 MB), per date 6 steps (nov26 33312, dec01 33552, jan01 17520) + geom + consts = 1.2 GB per date, 3.4 GB total (ff_data/<date>/ffc_cse_*). Contents: atmosphere (U V T Q QCL QCI TMOM QMOM PK PMID PEDN PDSIG GZ MWS ...), PBL/surface inputs, cloud state and all 15 radiation hand-off arrays at entry and exit, RNDSS+seeds, UKM/VKM + pole arrays, precip/energy outputs. Old ffd_*_pre/post_condse exist only for 2 steps per date, so the new files carry the atmosphere too. Found/fixed during the build: TLS/QLS/TMC/QMC are (I,J,L); RNDSS halo shift.

## D126: CONDSE port (clouds_condse_ff.py, _io, _compare, tests/test_clouds_condse_ff.py)
Imports the MSTCNV and LSCOND ports; adds column set-up, post-MSTCNV block, LSCOND set-up, precip/energy bookkeeping, SNOAGE, hand-off, final merge, RANDU stream (bitwise vs RNDSS, all 18 steps), replicate/avg momentum transfer + recalc_agrid_uv, init_CLD derived constants (BYDTsrc, XMASS, BYBR; bybr bitwise only with libimf).
Validation: 3 dates x 6 steps x 3170 columns (57,060), 68 exit fields per step.
- libimf mode: setup and LSCOND inputs bitwise vs the MSTCNV/LSCOND boundary records (0 mismatches except downstream of the columns below); columns with any inexact cell 136 (0.24%, 1e-14..1e-23 relative, the known MSTCNV imf ulp residual); columns beyond the 1e-12 field-scale bound: 26 (nov26 24 incl. 17+6 from a carried CSIZSSIP chain, dec01 2, jan01 0): real threshold flips (e.g. dec01 33555 column (67,22) T off by 3.6e-5). Steps fully bitwise: jan01 17523 (68/68), nov26 33317 (67 exact + 1 bound).
- libm mode: 1000-2000 of 3170 columns inexact per step; 58-169 per step beyond the bound (2074 total, 3.6%); 32-44 of 68 fields FAIL per step (first failing cell printed by the compare script); no tolerance loosened.
- Module-state carry (CSIZELIP/TAUSSIP etc. from the previous column and step) is emulated; step-to-step chain in the compare script; first column of a window starts from zeros (step 1 of each window matches).
- Momentum chain from recorded tendencies: U,V,UALIJ,VALIJ bitwise (test). Tests: 23 passed (~2 min), incl. 5 mutations detected, labelled non-real tests for stop_model exits, carry, renormalisation.
- Timing (warm, 6 processes concurrent on 12 cores): about 40 ms/column imf, 35 ms/column libm, about 125 s (imf) / 110 s (libm) per step of 3170 columns.
Gaps: diagnostics/ISCCP, ISC=1 block, RIS/RI1/RI2, init_CLD parameter reads (only derived constants), stop_model exits unreached on real data, renormalisation mutation undetectable by construction (result is 0/1). No JAX version.

## D118: whole-ocean step chain, scoping (2026-10-06)

Read OCEANS (OCNDYN2.f:33-699) and every routine it calls on the live path; wrote `fullfidelity/scoping/OCEAN_CHAIN_PLAN.md`: the call order for this rundeck (NOCEAN=1, NDYNO=4, DTOLF=900, five ODHORZ calls, USE_QUS=0, OTIDE=0, straits active) with a line citation per call, the port that covers it, and its input status. Gaps found: (1) OCONV per-column preamble and the whole post-loop tail (UKM add-back, GZMO/SZMO flux update, GX/GY/SX/SY vertical diffusion, slope limits, KPL) had no port; hbl_loop took ALPHAGSP/BETAGSP/SHCGS from the setup dump and did not return AKVG/AKVS; (2) ODHORZ ports took the post-OPFIL2 USMOOTH/PGFX as recorded input and returned no SMU/SMV or OGEOZ; (3) ofluxv_jax returned no SMW; (4) horizontal densgrad (RHOX/RHOY) and the EOS inputs VUP/VDN/VUPU/VDNU were recorded; (5) SHCGS was recorded (D35); (6) ODIFF (OCNDYN.f:5092-5575, fires when mod(itime,6)==0, i.e. step 1 of each 6-step window) has no port; (7) gather/scatter of the straits end cells, PRECIP_OC/KVINIT/GROUND_OC glue (DXYPJ*FOCEAN accumulation along I, OPRESS) had no chained form; (8) AG2OG/IG2OG regrid fluxes, OPFIL2 coefficient setup and init_STRAITS remain recorded. Also found that the OPFIL2 coefficient dump (ffz_opcoef.bin) and the D65-D77 dumps (hblin/hblout/post/stin/stout/pgf/me/opfil) are no longer on disk under ff_data; the chain does not need them except the coefficients, which were re-dumped (D119).

## D119: chained live ocean step `ocean_step.py` (2026-10-06)

`fullfidelity/ocean_step.py` runs PRECIP_OC+KVINIT, GROUND_OC, OSTRES2, OCONV, OBDRAG2/OCOAST, polar relax, ODHORZ0, 5 x ODHORZ (with OPFIL2 computed, not recorded), OFLUXV, OADVT2 x2, straits (gather, STPGF, STADV, STCONV, STBDRA, scatter), second ODHORZ0, ocnstate_derived/densgrad (EOS from the OFTAB tables) and ocnmeso (isoslope4, gmkdif, gmfexp x2) on a state dict, calling the existing validated ports in JAX where a JAX version exists. New code: OCONV preamble + tail (numpy, vectorised over the 2.7k columns), GROUND_OC glue, SHCGS/ALPHAGSP/BETAGSP from OFTAB, horizontal densgrad, prefilter (USMOOTH/PGFX/OGEOZ with OPFIL2 operators), straits gather/scatter. Copies with minimal edits: `ocean_hbl.py` (hbl_loop with table EOS and AKVG/AKVS output), `ocean_odhorz.py` (returns MU/MV), `ocean_ofluxv.py` (returns SMW). Readers: `ocean_chain_io.py`. Recorded boundaries left: AG2OG/IG2OG fluxes (per step), ODIFF result (UO, VO, VONP on mod(itime,6)==0 steps), OPFIL2 coefficient file, init_STRAITS (first step state), OFTAB tables read from the model support files.
Instrumentation: `instrumentation/OCNDYN2_oceanchain.f.patch`, `OCNDYN_oceanchain.f.patch` (diff -u against pristine, units 1270-1272; hook `ffo_snap` writes 15 stage snapshots per step: tags 0..14, 12 steps per date, steps 7-12 only tags 0,1,12,13,14; `ffo_geom.bin`, `ffo_opcoef.bin`). Files `ffo_state_<itime>.bin` (12 per date, ~107 MB for full steps), `ffo_geom.bin`, `ffo_opcoef.bin` copied into ff_data/<date>/.
Findings: (a) OMEGA: the real Coriolis OMEGA is 2*pi/(86400*365/366) s (Earth365DayOrbit.F90:101, Constants_mod.F90:282), not 2*pi/86164.09054 as in odhorz_ff.py (rel. 1.8e-6 high). Found because the ODHORZ error vs the real run was linear in an OMEGA scale and vanished at 86163.934 s; with the correct value ODHORZ agrees with the real Fortran to 1e-13 (was 3e-7). The existing odhorz_ff/odhorz_vec/odhorz_jax still use the old value (not edited); ocean_odhorz.py uses the correct one. (b) SMW from ofluxv_jax is per area: the model's SMW carries DXYPO(J)/DTOLF, and is zero at and below each column's bottom layer. (c) the OGEOZ module array is updated by every ODHORZ call and feeds the next step's OCONV (ZSCALE); without it the free-running chain diverged at 1e-4.
Validation (replay from the real snapshot, per stage; relative = max|diff|/max|ref|): precip, ground, odiff exact; ostres vo 2e-9; oconv tracers/moments 1e-12..7e-10, velocities 3e-10; drag/polar 1e-16; dynamics (7 routines) tracers 1e-15, velocities 3e-13, smw 1e-11; straits 1e-18; post 2e-17; meso 6e-17.

## D120: compare scripts, tests, speed (2026-10-06)

`ocean_step_compare.py` (stage replay from real snapshots), `ocean_step_chain_compare.py` (per step chained from the real entry, and a 12-step free-running chain), `ocean_step_speed.py`, `tests/test_ocean_step.py` (12 tests: stage replay tolerances, full-step exit, non-vacuity, dropping each of 6 stages and swapping OSTRES/OCONV are detected, OMEGA pinned; skip if dumps missing; 65 s).
Results, 3 dates x 12 steps, each step run from the real entry state (chained through all stages): worst exit-state relative error g0m 7e-12, s0m 3e-13, gx/gy/gz 3e-11/6e-11/1e-10, sx/sy 7e-10/6e-10, sz 9e-11, mo 5e-15, uo/vo 2e-9, uod/vod 8e-10/1e-9, opress 0 (bitwise), ogeoz 4e-12, kpl exact, straits must 3e-13. Free-running 12 steps (port state fed to the next step; fluxes and ODIFF recorded): worst after step 12 about 1e-8 (sx/sy 6e-9, uo 3e-9, g0m 1.3e-11 nov26, 1.7e-11 dec01); no run-away. At steps with ODIFF (steps 1 and 7) UO/VO are the recorded ones, so their error there is 0 by construction.
Speed (CPU, warm, one thread-pool of this node, JAX 64-bit): 4.05 s per step total: oconv 1.3, dynamics 2.0 (OPFIL2 prefilter + 5 ODHORZ + OFLUXV + 2 OADVT2), straits 0.5, meso 0.2, rest <0.2. First call adds compile time (~30 s).
Not exercised / limits: columns with LMM==1 (none in these windows; the Fortran reads uninitialised arrays for them), freezing in the lower-layer GROUND_OC sweep (never fires; branch is the validated D35 one), ODIFF (recorded), SMW stale entries, multi-day behaviour, GLMELT (daily, not in OCEANS), OTIDE, the pole rows of OCONV velocity add-back (not used by the Fortran either).

## D130: where the CONDSE time goes (profile of clouds_condse_ff.py, one real step)

`clouds_condse_profile.py` (new) wraps the validated column ports with wall-clock timers and runs the whole step (3,170 columns, libm mode, step index 1 of each date, ff_data ffc_cse_* entry state) plus a cProfile of four latitude rows. Measured split of the column loop:

| date step 1 | column loop | MSTCNV | LSCOND (main+CTEI+tail) | radiation hand-off | other glue (set-up, LSCOND list building, stores, merge) |
|---|---|---|---|---|---|
| nov26 | 110.9 s (35.0 ms/col) | 98.0 s (88.3 %) | 3.25 s (2.9 %) | 0.81 s (0.7 %) | 8.87 s (8.0 %) |
| dec01 | 112.3 s (35.4 ms/col) | 99.3 s (88.4 %) | - | - | - |
| jan01 | 101.7 s (32.1 ms/col) | 89.2 s (87.7 %) | - | - | - |

About 2,010 of the 3,170 columns convect (nov26). So LSCOND is NOT the bottleneck (about 1 ms/column); MSTCNV is 88 %. Inside MSTCNV (cProfile, rows 20:24, with profiler overhead): about half of the time is `clouds_helpers_ff._dcw_search` (the 49-iteration fall-speed search, called twice per plume layer per column through `convective_microphysics`), the rest is the Python-level statement overhead of `_mstcnv` itself (3.2 s own time of 31 s), `precip_mp`, `dq._adjust`, `mass_flux`. Consequence: batching LSCOND alone saves at most 3 %, the MSTCNV batch is what removes the blocker (D132).

## D131: batched LSCOND (clouds_lscond_batch.py)

All columns of a step at once with numpy; the sequential layer loop `L=LMCLD..1` and the CTEI loop are Python loops over L with every statement a vector operation (Fortran `if` = np.where / mask); CTEI filters the columns through the `cycle` tests and compacts the mixing columns (a few per layer) before the bounded 9-iteration loop with a per-column active mask (an exited column keeps its last iteration). Same operation order and REAL(4) literals as clouds_lscond_ff; Python max/min semantics reproduced (pymax/pymin); exp/pow are evaluated element by element with the per-column port's scalar functions (libm or libimf) only on the columns that reach the statement (no transient overflow); get_dq_cond/evap are the same numpy helper in libm mode. The module-array carry (CSIZELIP, TAUSSLIP, CLDSAL, CLDSV1) is a forward fill along the column axis; DQLSC is the column-local increment (a diagnostic never read). The two pole columns (KMAX=72) stay per-column.
Validation (`clouds_lscond_batch_compare.py`, inputs = the LSCOND state of every column of the real step as built by the CONDSE chain): vs clouds_lscond_ff.lscond re-run per column in the same mode, 3,168 columns, 47 output fields (state arrays, moments, winds, W locals, scalars): **0 columns with any difference, 0 bit differences, worst difference 0** for libm on nov26, dec01 and jan01 (step 1) and for libimf on nov26. First attempt matched at once after one fix (the elif chain used QCIL where QCLL is needed for the liquid-to-ice switch). Speed (libm, 3,168 columns): per column 9.8 s (3.1 ms/col) -> batch 0.44 s (0.14 ms/col), 22x (dec01 19x, jan01 20x); libimf 9.5 s -> 0.89 s, 11x. Not exercised by real data (as in D107-D109): the dead arms listed there. Tests: tests/test_clouds_batch.py (bitwise vs per-column on two real rows, column independence, mutations).

## D132: batched MSTCNV (clouds_mstcnv_batch.py) and batched CONDSE (clouds_condse_batch.py)

MSTCNV: lock-step batch of the whole per-column port. The cloud-base loop over LMIN is a shared Python loop; the DMSE pre-test runs on all columns, MASS_FLUX (already vectorised) and everything after runs only on the compacted columns that pass (gather of persistent state, scatter back per LMIN); cloud types IC=1,2 and area partitions NPPL=1,2 use per-column run masks (cycle/break = mask update); the plume ascent loops over L with a per-column `alive` mask (each exit clears it; state written between two exits only for columns alive there); downdraft, subsidence and precipitation loops run over all layers with per-column layer-range masks (LDRAFT, LDMIN, LMAX become masks); the QUS vertical advection ADV1D is batched over columns of equal slice length (the qlimit loop of Q stays the scalar per-line loop of dyn_adv1d_ff, with the ierr==2 early return kept per line); convective_microphysics uses one early-exit fall-speed search for both calls. Two lock-step bugs found by the "column alone vs in batch" check and fixed: an unmasked update of non-selected columns in the advection set-up, and a downdraft mask that was permanently cleared before a column's own LDRAFT was reached.
Validation (`clouds_mstcnv_batch_compare.py`; reference = clouds_mstcnv_ff.mstcnv_column outputs of the real step): 3,168 non-polar columns (2,010 convecting nov26, 1,943 dec01, 1,966 jan01), 43 output fields incl. all scalars and moments: **0 differing columns, 0 bit differences, libm, all three dates**; libimf on the first 1,000 nov26 columns against a per-column libimf re-run (25 s): 0 differences. Time: per column ~98 s -> batch 4.3 s nov26 (4.0 dec01, 3.7 jan01), about 23x (1.3 ms/col). Not reproduced (diagnostic only, not read by any CONDSE exit field): DGDEEP DGSHLW DPHASE DPHADEEP DPHASHLW DTOTW DQCOND DGDQM DQMTOTAL DQMSHLW DQMDEEP DQCTOTAL DQCSHLW DQCDEEP, branch counters, checkpoints. Not batched: columns with KMAX=72 (the two poles) and the dead `mc_new_ddrft_thetav=0` arm (asserted off, the run has 1). Not exercised by real data: the limitq limiter / IERR>0 path (covered only by the synthetic adv_batch-vs-adv1d test).
CONDSE: `clouds_condse_batch.condse_step_batch` vectorises the column set-up, calls the batched MSTCNV, the convective post-processing, the batched LSCOND, bookkeeping/snow age (exp per column through the scalar function), stores, the radiation hand-off (vectorised branch table, validated against `_hand_off` on synthetic data) and the final merge; the two pole columns run through condse_column in the Fortran order (south pole, batch, north pole) so the LSCOND carry is continuous. Momentum back-transfer and recalc_agrid_uv are the per-column driver's own functions.
Results (`clouds_condse_batch_compare.py`, step index 1 after chaining step 0; wall time of the batched chain):
- vs the per-column chain (libm, fresh carry, before the momentum back-transfer): **0 of 62 exit fields differ (bitwise), nov26, dec01, jan01**.
- step wall time: per column ~111 s -> batched 4.9-5.0 s (libm), 7.6-8.3 s (libimf); about 22x libm.
- vs the REAL exit state, libm: nov26 11 fields exact / 18 within the 1e-12 field-scale bound / 39 FAIL, 1,249 columns inexact, 149 (4.7 %) beyond the bound; dec01 10/15/43, 1,143 and 122; jan01 11/18/39, 1,702 and 91 (the libm threshold flips of D126; statistical acceptance needed off libimf).
- vs the REAL exit state, libimf: **0 FAIL fields** on all three dates (nov26 34 exact + 34 bound, 4 inexact columns; dec01 50 + 18, 11; jan01 22 + 46, 37; no column beyond the bound in these steps). The libimf batch was compared with the real state, not with a per-column libimf CONDSE re-run (MSTCNV and LSCOND libimf batches were compared with per-column re-runs: exact).
Remaining per-column cost: the two pole columns (about 70 ms each); the rest of the step is 4.3 s MSTCNV + 0.4 s LSCOND + glue. Tests: tests/test_clouds_batch.py, 14 tests, 30 s (skip without dumps; mutations: pymin->pymax in LSCOND and MSTCNV, COESIG, CLDMIN, FDDET, TINY of the hand-off).
Files: clouds_condse_profile.py, clouds_lscond_batch.py, clouds_lscond_batch_compare.py, clouds_mstcnv_batch.py, clouds_mstcnv_batch_compare.py, clouds_condse_batch.py, clouds_condse_batch_compare.py, tests/test_clouds_batch.py. Next: JAX shapes of the same masks (the lock-step structure maps to jit with masks; the Python-level L/LMIN loops become lax.fori_loop), the pole columns, multi-step chaining with the dynamics step.

## D127: one-source-step plan (`scoping/ATM_STEP_PLAN.md`)
Live order read from MODELE.f:316-339, ATM_DRV.f (atm_phase1 3-294, atm_phase2 426-525), SURFACE.f. Findings: DRYCNV is not linked (D1); the `ATM_DIFFUS(2,LM-1)` call of atm_phase2 returns at ATURB.f:~116 (`lbase_min.eq.2`), verified on dumps (post_surface == pre_aturb == post_aturb, bitwise), so the F1 "DRYCNV" stage is `ATM_DIFFUS(1,1)` inside SURFACE (x2 substeps); NFILTR=MFILTR=1 (FILTER every step); NRAD=5 (radiation library only when MOD(Itime-ItimeI,5)==0: nov26/dec01 steps 0,5, jan01 step 2, confirmed on the new SRHR/TRHR dumps). SURFACE.f:1068-1089 TMOM/QMOM first-layer update had no port (implemented in atm_step.py). Recorded inputs: SOCRATES SRHR/TRHR/COSZ1, Ent exports/land forcing, CONDSE non-dynamic entry state, ice/ocean/lake surface state, PRECIP_* effects. Gaps listed in the plan (USAVG/VSAVG/TGVAVG/QGAVG composites, IRRIG_LK, diagnostics, ocean_driver recorded).

## D128: chained step `atm_step.py` + new step-boundary dumps
New instrumentation `instrumentation/ATM_DRV_atmstep.f.patch` (diff -u vs pristine, 5 hunks, unit 1360): `ffa_step_<it>_{a,r,d,e}.bin` (step start, end of phase 1 after RADIA incl. SRHR/TRHR/COSZ1, after DISSIP, end of phase 2), 24 files x 3 dates, plus the older surface-side dumps for steps 3-6 (ffa_c1/c2, ffp, ffs, ffl, ffg, fft, ffd pre/post), all with `\cp -n`. Checks (exact): site a == ffd_state s1 (18/18); e(k) == s1(k+1) (15/15). Rerun reproduces existing dumps except uninitialised-memory entries.
Chain: dyn (dyn_step) -> CONDSE (clouds_condse_ff) -> RADIA (recorded SRHR/TRHR/COSZ1 + RAD_DRV.f:5474 T update) -> SURFACE x2 (existing PBL/tile/land-ice/GHY/aggregation/ATURB+UV ports, rebuilt atmosphere-dependent PBL/tile columns, TMOM/QMOM first-layer glue) -> DISSIP -> FILTER; start/stop at any stage from the real boundary state.
Measured, libimf mode, 3 dates x 6 steps from the real step-start state (chained, field by field vs the real end state; categories A bitwise, B<=1e-12 of scale, C<=1e-6, D worse):
- dyn: bitwise (27/27 fields) all 18 steps; PEK, PDSIG, CONDSE entry state, and all atmosphere-dependent substep-1 PBL record columns (zs1,tkv,dbl,ug,vg,utop..,dpdx..,mdf,gusti,tdns,dtdt) rebuilt bitwise.
- radiation T-update glue: bitwise on 18/18 (replay from real post-CONDSE T).
- CONDSE: <=3.4e-14 scale-relative everywhere except dec01 33555 (one flipped column, PRECSS 1.7e-4, end-state QCL 1.2e-4 in that column).
- DISSIP + FILTER replay from the real post-SURFACE state: bitwise on 18/18 (numpy pow: <=1.4e-14 T).
- SURFACE with RECORDED land (GHY) patch: end state B for nearly all fields; exceptions W2GCM/EGCM/Q/PBLHT at 1.1e-12..6.5e-12 of scale in 0-4 columns per step (the ATURB-port rounding amplification noted in D19), LMONINPBL 1.7e-10. Step 0: nov26 W2GCM 4 columns 1.7e-12; dec01 all A/B; jan01 EGCM/W2GCM/Q one column (20,13), <=2.5e-12.
- SURFACE with the ported GHY: first failing stage = surface (land patch). nov26 step 0: T 6.2e-5 K (1.4e-7 of scale), U 4.0e-4 m/s (4.6e-6), Q 1.1e-6 (5.4e-5), EGCM 1.9e-3 (2.4e-4), all localised to land column (62,34) and neighbours (49 T columns), the D9/D22 runoff-threshold cell; dec01/jan01 step 0 worst 1.7e-7 (Q), 1e-9..1e-7 for the rest. Over all 18 steps worst gate field 1e-7..5e-4.
- numpy pow / platform libm: dyn <=3.8e-13 (DPDX 1.9e-11); CONDSE first failing stage: 96-145 of 3170 columns (3-4.6 %) beyond 1e-12 when chained, 1-25 columns (0.03-0.8 %) from the real entry state; PREC/QCL differences 1e-3..1e-1 of scale in those columns; end state D. Statistical acceptance needed off libimf.
- Free-running 6-step chain (own end state carried, RADIA's CLDSS/CLDMC masking RAD_DRV.f:2611-2615 ported and checked bitwise on the 3 radiation steps; surface state recorded): step 0 at rounding level, from step 1 CONDSE threshold flips in ~45 columns (inputs differing by only 1e-12) so after 6 steps T max 0.07-0.16 K (rms ~1e-3 K), U max 0.5-1.3 m/s, Q up to 6e-2 of scale: chaos, not a bug (consistent with D3); F2 needs ensemble statistics.
## D129: compare script, tests, timing, verdict
`atm_step_compare.py` (chained + isolated stages, land modes ghy/recorded, --condse-only, --free N, --timing), `tests/test_atm_step.py` (119 passed, 252 s with ATM_STEP_SLOW=1; mutation checks for radiation, dissip/filter, first-layer glue, PBL inputs; skip without dumps/libimf).
Warm CPU timing (nov26 33312, 2 cores, libimf): dyn 3.8 s, CONDSE 116-138 s (python, 40 ms/column), radiation glue 0.2 s, SURFACE 0.7 s (recorded land) / 1.1 s (GHY) warm (67 s first call incl. XLA compile), DISSIP 0.01 s, FILTER 0.5 s: ~6 s per step without CONDSE, ~125-145 s with.
F1 gate verdict: with the RECORDED land patch and libimf: MET with named exception columns (dec01 step 0 strictly A/B; nov26 W2GCM 4 columns 1.7e-12; jan01 one column <=2.5e-12). With the ported GHY: NOT MET on nov26 step 0 (U,V,Q,EGCM,W2GCM beyond 1e-6; first failing stage SURFACE/GHY land column (62,34)), PARTLY MET on dec01/jan01 step 0 (all gate fields <=1e-6 of scale, up to 1.7e-7). Without libimf: NOT MET (first failing stage CONDSE). Radiation (SOCRATES) and Ent are recorded throughout.

## D135: nov26 step 0 land cell (62,34): porting error in ghy_jax forcing (htprs/prs), not a threshold flip; fix candidate
Indexing: ffg records are 1-based (i,j) in columns 0,1; (62,34) = records 423 (substep 1) and 1176 (substep 2) of nov26 ffg_33312.bin (753 land cells per substep).
Reproduction (recorded GHY inputs of the cell, no PBL involved): `ghy_ref.GhyColumn` (numpy, via ghy_compare.run_cell) reproduces the real tbcs/tsns/ashg/alhg/aevap/aruns/arunu/aeruns/aerunu/ae0/abetad of this cell exactly (bitwise, both substeps); `ghy_jax.advnc` (via land_chain.run_ghy on the same records) does not: aruns 3.019e-4 vs real 3.856e-4, aevap 4.373e-3 vs 4.284e-3, alhg 10932 vs 10710, tbcs -6.2e-6, tsns -1.2e-5 (substep 1 over the batch: alhg 6.3e-4 of scale, aruns 3.7e-4).
Localisation (substep 1 of the cell, ghy_ref run stage by stage vs the ghy_jax stage functions on identical inputs): evap_limits, sensible_heat, snow(evap), fl all identical; first difference is `drip_from_canopy`: dripw[veg] jax 0.0 vs ref/real 6.7086e-9, dripw_scale[bare] 2.3629e-8 vs 1.0062e-8, which then changes snow flmlt[veg] (1.2552e-8 vs 1.9293e-8), flg f[0,veg], fc[veg], and the runoff rnf[veg] (2.0e-10 vs 3.1e-10) -> aruns. Not a threshold flip: no operation differs by an ulp; the inputs to the stage differ.
Root cause: `ghy_advnc_test.build_batch` (used by land_chain.run_ghy and thus atm_step land_mode 'ghy') passes `htprs = 0` and the raw `prs`. `GhyColumn.__init__` (ghy_ref.py:597-599, 648) conditions the precipitation first: pr=max(pr,0); prs=min(max(prs,0),pr); htprs = 0 if pr<=0 else htpr/pr*prs. With htprs=0 the jax port gets snowfs=0 instead of min(-htprs/FSN,prs); the cell has snowfall (htpr=-7.89, pr=4.115e-8, prs=2.363e-8, vegetated fr_snow 0.95, 3 snow layers), so ptmps, ptmp, dripw_scale and the canopy drip are wrong. (The ghy_ref line was matched against the real records, not re-read in the Fortran source in this session.)
Fix: new `fullfidelity/ghy_fix_candidate.py` (`condition_forcing`, `run_ghy_fixed` = land_chain.run_ghy with the conditioning, `patch_land_chain()`), `test_ghy_fix_candidate.py`, `ghy_fix_candidate_chain.py`. Existing files untouched; exact diff for land_chain.run_ghy after the forcing_over merge:
    pr = np.maximum(f["pr"], 0.0); prs = np.minimum(np.maximum(f["prs"], 0.0), pr); f["pr"], f["prs"] = pr, prs
    f["htprs"] = np.where(pr <= 0.0, 0.0, f["htpr"] / np.where(pr <= 0.0, 1.0, pr) * prs)
Result on the cell (recorded inputs, both substeps): tbcs, tsns, alhg, aevap, arunu, aeruns, aerunu bitwise equal to the real values; ashg 9e-13 abs, aruns 9.6e-18 abs (2.5e-14 rel), ae0 1.8e-11 abs (1 ulp class). Nov26 batch maxima (substep 1) drop alhg 6.3e-4 -> 5.0e-6 of scale, aevap 6.3e-4 -> 5.0e-6, tbcs 1.2e-7 -> 8.7e-9, tsns 2.4e-7 -> 2.3e-8, ashg 8.4e-7 -> 9.1e-8; dec01/jan01 unchanged (their forcing has no cells hitting this path).
NOT fully closed: aruns/aeruns/arunu/aerunu still differ from the real record in ~97 of 753 cells on nov26 (aruns max 1.1e-4 abs, 3.7e-4 of scale; e.g. cells (64,33), (60,32), (58,22)), and likewise on dec01/jan01 (aruns 2e-3 / 2.7e-4 of scale); `ghy_ref` also differs from the real values on those cells, so it is a second, still undiagnosed discrepancy common to both ports (not the (62,34) one), not yet analysed here.
Chained step (libimf, CONDSE run, land ghy, step 0, fix patched in via ghy_fix_candidate_chain.py): F1 gate now PARTLY MET on all three dates (was NOT MET on nov26): nov26 worst Q 9.96e-7 of scale (was 5.4e-5), EGCM 1.1e-7 (was 2.4e-4), U 3.8e-9, T 1.2e-9; worst cells now (43,31), (22,14) not (62,34); dec01 worst W2GCM 1.85e-7; jan01 worst Q 2.1e-7. The gate is still not MET (needs <= 1e-12 or named exception columns) because of the remaining aruns cells above.
Files: fullfidelity/ghy_fix_candidate.py, test_ghy_fix_candidate.py (pytest cell test + `python3` before/after table), ghy_fix_candidate_chain.py.

**Applied (2026-10-06, session driver):** the conditioning was applied to `land_chain.run_ghy` (existing file; `ghy_fix_candidate*.py` were not committed, superseded). 70 land tests pass; regression test `tests/test_land_chain_precip_conditioning.py` (2 tests, both substeps of cell (62,34) vs the real record at 1e-10 relative). I re-ran the nov26 step-0 F1 verdict with the ported GHY and libimf: PARTLY MET (13 fields C, 1 A, worst Q 9.96e-7 of scale), previously NOT MET. The second discrepancy (about 97 nov26 cells with runoff terms differing from the real record in both ghy_ref and ghy_jax) remains undiagnosed.

## D133: batched CONDSE wired into the chained atmosphere step (atm_step_fast.py)

New files (nothing existing edited): `fullfidelity/atm_step_fast.py`, `fullfidelity/atm_step_fast_compare.py`, `fullfidelity/tests/test_atm_step_fast.py`.
`atm_step_fast.stage_condse_fast` is `atm_step.stage_condse` with `clouds_condse_batch.condse_step_batch` (poles per-column, LSCOND module-array carry `ms` as in D132); `run_step` / `run_chain` bind it into `atm_step.stage_condse` for the duration of the call (`atm_step.run_step` resolves the stage at call time), so dyn, radia (recorded), surface, dissip, filter and all recorded-input rules are exactly atm_step's. `ensure_backend(ctx)` re-applies the libm/libimf switch (process-global in clouds_condse_ff). The batch has no column-subset mode (`cols=` raises).

Verification (step 0 of nov26, dec01, jan01, both modes, land_mode='recorded', from the real step-start state; `python3 atm_step_fast_compare.py --equiv [--imf]`): per-column chain (atm_step.run_step) vs fast chain.
- CONDSE-exit state (18 fields) and end-of-step state (30 fields): **0 differing cells in every field, all 3 dates, libm and libimf** (bitwise identical). Hence the F1 verdict is unchanged: libimf nov26 and jan01 `MET with named exception columns`, dec01 `MET`; libm `NOT MET` (the known libm pow dynamics mismatch, not a CONDSE issue), identical for both chains.
- Wall time per full step (dyn+CONDSE+radia+SURFACE(recorded land)+dissip+filter): per-column 139-196 s (libimf), 122-165 s (libm); fast **13.4-15.3 s (libimf)** [dyn ~4.0-4.8, CONDSE ~8.1-8.8, surface 0.6-1.0, filter 0.5], **8.7-9.6 s (libm)** [dyn ~3, CONDSE ~5-6]. Speed-up about 10-20x. (The per-column and fast runs of one date ran in one process, sequentially; other agents'/my other jobs ran concurrently on the node, so the timings are +/- noise.)
- Tests: `tests/test_atm_step_fast.py`, 4 passed in 226 s (patch restore, cols NotImplemented, libm CONDSE-stage bitwise vs per-column incl. carry fields + input-perturbation mutation, libimf fast end state vs real <= 1e-9 on prognostic fields + whole step < 120 s + a 1e-3 T-perturbation mutation that must worsen the end state >= 100x).
- Not done: GHY land mode was not exercised in these runs (recorded land; land is independent of CONDSE).
- A harmless RuntimeWarning (divide, `fw0/(fw0+fsi)` in clouds_condse_batch.hand_off_b) is emitted by numpy on masked entries that np.where discards; not changed.

## D134: multi-step fast chain (6 steps x 3 dates), step k+1 fed OUR end state of step k

`python3 atm_step_fast_compare.py --chain [--imf]` (atm_step_fast.run_chain; same hand-over as atm_step.run_free: atmosphere + ATURB/PBL hidden state + CONDSE cloud/precip carry + LSCOND `ms`). Recorded per step k: SOCRATES SRHR/TRHR/COSZ1 (with RADIA cloud masking carried), SURFACE tile/PBL/land-ice/GHY(recorded land)/Ent records, sea-ice/lake/ocean state, the non-atmosphere CONDSE entry fields. After each stage the state is compared with the real boundary dump of that stage of the SAME step (so for k>=1 the inputs of the stage already differ); the end state is compared with ffa_step_e = the real next-step start. Categories A bitwise / B <=1e-12 / C <=1e-6 / D worse of field scale. All 3 dates x 6 steps ran, libimf and libm. Per-step wall 9-16 s (some steps 24-31 s while 4-5 jobs shared the node; first step of each process includes jit warm-up, 26-67 s).

libimf (Intel libimf), end-of-step categories (30 fields) and gate-field growth:
| date step | stage categories at stage exit | columns >1e-12 (T,Q,QCL,QCI,TMOM,QMOM,U,V): dyn exit / CONDSE exit / new at CONDSE / end | end rel T / Q / U |
|---|---|---|---|
| nov26 0 | dyn A27; condse A14 B4; end B26 C2 A2 | 0/0/0/0 | 1.9e-15 / 1.6e-13 / 8.4e-15 |
| nov26 1 | dyn B24 C2 A1; condse D12 B5 A1; end D22 B8 | 0/176/176/597 | 1.2e-4 / 1.4e-2 / 3.1e-4 |
| nov26 2 | dyn D26; condse D18; end D30 | 3240/3240/0/3312 | 2.1e-4 / 2.4e-2 / 3.5e-3 |
| nov26 3,4,5 | all D | 3312 | T 3.1e-4..3.6e-4, Q 3.1e-2..6.4e-2, U 5.3e-3..5.6e-3 |
| dec01 0 | dyn A27; condse B10 A8; end B27 A2 C1 | 0/0/0/0 | 1.9e-15 / 5.8e-13 / 4.6e-14 |
| dec01 1 | dyn B24 C2; condse D12; end D22 C7 B1 | 0/209/209/3312 | 6.1e-5 / 5.2e-3 / 3.0e-4 |
| dec01 2..5 | all D | 3312 | T 9e-5..2.3e-4, Q 7e-3..3.8e-2, U 2.5e-3..1.1e-2 |
| jan01 0 | dyn A27; condse B12 A6; end B23 C5 A2 | 0/0/0/1 | 2.3e-15 / 1.1e-12 / 1.1e-13 |
| jan01 1 | dyn B23 C3; condse D12; end D19 C9 | 1/157/156/3312 | 8.4e-5 / 7.4e-3 / 3.0e-4 |
| jan01 2..5 | all D | 3312 | T 8.8e-5..2.1e-4, Q 1.7e-2..3.0e-2, U 3.9e-3..1.3e-2 |
(column count is the union over the 8 prognostic fields, of 3312 horizontal columns incl. pole rows' duplicates as counted by atm_step_compare.columns_over_bound.)

First failing stage (any field beyond the B bound, 1e-12 of scale): step 0, SURFACE (end state still within the D129 gate; first inexact stage is CONDSE, B level); steps >=1: `dyn` (because the stage inputs are already our drifted state; see below), then every stage.
libm: the dynamics is already C level at step 0 (numpy pow, known), the CONDSE of step 0 creates 131 / 143 / 96 new columns beyond 1e-12 (nov26/dec01/jan01) with worst relative error 2.3e-2 / 1.3e-2 / 1.0e-2 in those columns versus <= 1.0e-12 outside them; all steps >= 1 are D everywhere (3312 columns); end-state rel after step 5: T 2.0e-4..4.7e-4, Q 2.8e-2..6.5e-2, U 4.2e-3..7.9e-3 (nov26 shows one outside-column worst 1.3 in step 5 for a small-scale field; not analysed).

Interpretation (chaotic divergence), only what the runs show: in libimf mode step 0 is bitwise through dynamics and rounding-level (<= 2e-12 of scale) through the whole step with no CONDSE threshold flip beyond 1e-12 (0 columns). That rounding-level end state is the input of step 1: step 1's dynamics exit is still <=1e-12 of scale in 0-1 columns (smooth part, categories B/C only in a few fields), and at CONDSE of step 1 157-209 NEW columns jump to 1e-2 relative error (the largest relative error inside the new columns 1.4e-2..2.1e-2 versus <=1.0e-12 outside them): this is the first threshold-flip event; essentially all of the divergence that exceeds rounding level appears there, in one stage, localised in columns, not as smooth growth. Within the same step the differences spread through SURFACE/DISSIP/FILTER (597 columns at the end of nov26 step 1, 3312 at the end of dec01/jan01 step 1) and from step 2 on the dynamics exit already differs in 3240-3312 columns, so later growth is spreading plus slow amplification of an O(1e-2) local error (Q end rel 1.4e-2 -> 6.4e-2 in nov26 over steps 1-5; T 1.2e-4 -> 3.6e-4; U 3e-4 -> 5.6e-3), roughly an order of magnitude over the first flip event for U and little for Q, not exponential on this 6-step horizon. libm mode shows the same picture one step earlier (the rounding-level libm dynamics difference reaches the CONDSE flips already in step 0). Not shown by these runs: whether a flip-free chain would stay at rounding level (no such real data exists); which individual CONDSE branch flips (not instrumented here; D124-D126 attribute the isolated flips to threshold tests). The F1 gate (one step from the real state) is therefore met by the chain in libimf mode, but a free-running multi-step comparison against a single real trajectory cannot be bitwise beyond step 1; judge it statistically (F2), not pointwise.
Files with the raw output (scratchpad): ch_imf_nov26.json/log, ch_imf_rest.json/log, ch_libm.json/log, eq_*.json/log.

## D136: remaining land runoff discrepancy = irrigation dropped (porting error in ghy_ref and ghy_jax); fixed, all failing cells now match the real record

Root cause. GHY.f:2230-2234 zeroes irrig/htirrig and, for fv > 0, sets irrig(2) = irrig_in/fv, htirrig(2) = htirrig_in/fv (vegetated tile only); flg (GHY.f:1283, 1298) and flhg (1334, 1349) subtract them from f(1)/fh(1). `ghy_ref.GhyColumn.__init__` (ghy_ref.py:600-601) sets both to zero and ignores forcing['irrig'] (ghy_compare.unpack already computes irrig_tot/fv from ffg slot 148 (1-based), htirrig_tot is slot 149); `ghy_jax.flg/flhg/runoff` were written on that premise ("irrig is always 0 in this rundeck", ghy_jax.py:582, 614), which is false: the records carry irrigation on a fraction of cells (704 of 3012 cell-substeps in nov26 33312 + dec01 33552, values 5e-13 to 1.5e-10 m/s). Missing irrigation changes f(1) of the vegetated tile, hence satfrac*max(-f(1),0) and the infiltration terms of the runoff (aruns/aeruns), the layer-1 water and heat (w, ht, tp), and, through those, evap/alhg at the 1e-6 level. Not a threshold flip, not an ulp effect, not related to snow (most failing cells have fr_snow = 0), not the D135 precipitation conditioning.
Pattern before the fix (nov26 33312, ghy_ref, 191 failing cell-substeps in the listing of |aruns-real| > 1e-12 or aeruns > 1e-6; jax 97/95 cells beyond 1e-12 of scale in substeps 1/2): cells with irrig_tot != 0 (the "tiny bare fraction" impression of D9 was a coincidence of where irrigated land lies: bare fractions 0-5% in most of them, but also fb 0.2-0.96); w/ht/tp differ only in the first vegetated soil layer (e.g. (58,22): w(1,veg) -4.3e-7, tp -1.8e-5, ht -42; (41,15): aruns real 2.0e-9 vs port 0).
Fix (new files, nothing existing edited): `fullfidelity/ghy_ref_irrig.py` (GhyColumnIrrig, run_cell), `ghy_jax_irrig.py` (copy of ghy_jax.py with irrig/htirrig passed to flg/flhg; diff below), `land_chain_irrig.py` (`run_ghy_irrig` = land_chain.run_ghy with irrig = g[:,147]/fv, htirrig = g[:,148]/fv (0-based), using ghy_jax_irrig; `patch_land_chain()`), `ghy_irrig_compare.py`, `test_ghy_irrig.py` (2 tests, 27 s).
Exact diff to apply to existing code:
 - ghy_ref.py:600-601:  `self.irrig = np.array([0.0, forcing.get('irrig', 0.0) if forcing['fv'] > 0.0 else 0.0])` and same for htirrig (and ghy_compare.unpack must return forcing['htirrig'] = r[147]/fv; today it only sets 'irrig').
 - ghy_jax.py: flg(..., fm, irrig=None): subtract irrig[:,0] / irrig[:,1] from f0_bare / f0_vege; flhg(..., fm, htirrig=None): subtract htirrig[:,0]/[:,1] from fh0_bare / fh0_vege; advnc: build irrig2 = stack([0, forcing["irrig"]]), htirrig2 likewise and pass them to the two calls (the complete diff is `diff ghy_jax.py ghy_jax_irrig.py`, 45 lines).
 - land_chain.run_ghy after the D135 conditioning: `fv = np.asarray(f["fv"]); f["irrig"] = np.where(fv>0, g[:,147]/np.where(fv>0,fv,1.0), 0.0); f["htirrig"] = np.where(fv>0, g[:,148]/np.where(fv>0,fv,1.0), 0.0)`.
Result, jax (ghy_irrig_compare.py, first ffg file of each date, both substeps; cells beyond 1e-12 of field scale, before -> after): nov26 33312 aruns 97 -> 0 (max 1.1e-4 abs, 3.7e-4 of scale -> 2.9e-14 abs, 9.7e-14 of scale), aeruns 67 -> 0 (2.1e-4 -> 4e-15), alhg/aevap 66 -> 0 (5.0e-6 -> 1.3e-14), tbcs/tsns 34/40 -> 0 (8.7e-9 -> 7.6e-15), arunu 48 -> 0, aerunu 38 -> 0, ae0 70 -> 0 (3.1e-8 abs, 8.6e-14); substep 2 the same (aruns 95 -> 0); dec01 33552 aruns 90/86 -> 0 (2.1e-3 -> 5.2e-14 of scale), jan01 17520 aruns 78/77 -> 0 (2.7e-4 -> 4.3e-15). Every field of every cell in those six runs is now within 2e-13 of the field scale of the real record (a 1-ulp class), so previously matching cells still match. Not run with --all (the other 15 ffg files), only numpy was run on them.
Result, ghy_ref: aruns beyond 1e-12 cells 3074 -> 367 of 27108 cell-substeps (all 18 files), worst aruns 2.0e-4 -> 2.3e-7 abs; but ghy_ref also still carries a separate residual only on multi-substep (ffnit > 1) cells (all 176 remaining flagged cells in nov26 33312 + dec01 33552 have 2-3 substeps; alhg up to 0.27 W/m2 in e.g. (13,32), (60,23); aruns there 0): the jax port matches the real record on those, so it is ghy_ref-only and was not diagnosed (D9's "threshold" explanation of the ref's runoff residuals was wrong in part: the irrigation was the larger share).
Chained consequence: land_mode 'ghy' runs of atm_step should drop the remaining aruns-driven residual of D135; not re-run here (no chained step was run in this task).
Evidence the cause is irrigation and not something else: after the change the jax output agrees with the real record to 1e-13 of scale in all 11 outputs on all 3 x 2 batches (753 cells each), with no other code change.
Owner/limits: Track B GHY; validated against the instrumented real records only (irrig inputs are recorded, not computed by the port); dynamic vegetation (Ent) still a recorded input.

**Applied (2026-10-06, session driver):** the diffs were applied to `ghy_jax.py`, `land_chain.run_ghy` and `ghy_ref.py` (`ghy_compare.unpack` already returned both irrigation values); the agent's candidate copies were removed. Verified: 72 land/GHY tests and the new regression test `tests/test_land_chain_irrigation.py` (6 date-half cases to 1e-11 of field scale plus a non-vacuity check) pass; real GHY.f lines 2230-2234, 1283, 1298, 1334, 1349 and 1429 checked by hand. Note `ghy_ref` still has a separate residual on multi-substep (ffnit>1) cells (not in the jax port), undiagnosed.

## D137: ODIFF (ocean horizontal momentum diffusion) ported; ODIFF no longer a recorded input of the chained ocean step

Owner: full-fidelity port session (Claude, for G. Tamkin). 2026-10-06. Sources: OCNDYN.f:5092-5575 (ODIFF), OCNDYN.f:1236-1498
(init_odiff), OCNDYN2.f:1530-1564 (polevel), solvers/TRIDIAG.f (TRIDIAG; TRIDIAG_3D_DIST_new), OGEOM.f, OCEAN_COM.f:170 (FSLIP=0),
call site OCNDYN2.f:556-564 (every 6th step, DTDIFF=10800 s). All read-only.

Files (all new): `fullfidelity/ocean_odiff.py` (init_ODIFF operators + ODIFF), `odiff_compare.py`, `ocean_step_odiff.py`
(`stage_odiff_ported`, `ocean_step_odiff`; ocean_step.py unchanged), `ocean_step_odiff_compare.py`, `tests/test_ocean_odiff.py`.
No new instrumentation: the D118 dumps bracket ODIFF (tag 12 pre_odiff -> tag 13 post_odiff) on the first step of each window.

Result 1 (stage replay, tag 12 -> 13, firing step of each date, 3 dates): UO max rel 1.05e-16 / 1.85e-16 / 9.6e-17, VO 2.3e-16 /
1.1e-16 / 1.7e-16 (<= 1 ulp of the field maximum); VONP bitwise equal; about 2% of the 43,056 values differ in the last bit
(906/917, 1009/1021, 888/943), the rest are bitwise equal. Not claimed bitwise: residual is last-bit arithmetic (FMA/order) in the
Fortran build. Using the reciprocal form of TRIDIAG for the x sweep raises the mismatches to ~5,200/43,056, so the division form
(OPTIMIZED_TRIDIAG undefined) is the right one. The single-precision `SQRT(3.)` literal in the Munk length is reproduced
(float32 sqrt) and covered by a mutation test.
Result 2 (full ocean step, ported ODIFF, nothing recorded for ODIFF, entry = real tag 0, exit vs tag 14): max over 12 steps per date,
ODIFF steps / other steps: UO 4.5e-10 / 7.7e-10 (jan01), 9.8e-10 / 8.3e-10 (nov26), 4.3e-10 / 1.6e-9 (dec01); VO 5.8e-10 / 1.1e-9,
1.9e-9 / 2.1e-9, 5.2e-10 / 1.1e-9; VONP on the ODIFF steps 1.8e-9, 8.8e-9, 2.6e-9. Same level as the D118 numbers (UO/VO ~2e-9): no
degradation on ODIFF steps. Free-running 12-step chain: UO/VO <= 3.6e-9 (jan01), UO 1.9e-8 (nov26, step 33321, a non-ODIFF step),
<= 2.4e-9 (dec01).
New capability: the second ODIFF step of each window (itime 17526, 33318, ...) has no recorded tag 12/13 snapshot and could not be run
before; it now runs and is at the same error level (jan01 17526 UO 4.5e-10, VO 5.8e-10 from the real entry). It is validated only
through the exit snapshot (tag 14), not by a stage-boundary comparison.
Assumptions not independently verified: AKHFAC=1 (no rundeck override of the AKHFAC parameter was looked for beyond the default),
serial domain (global arrays), RADIUS=6371000, OMEGA as in ocean_odhorz.py. DH is taken from the live second ODHORZ0.

Still recorded after D137: AG2OG/IG2OG fluxes, OPFIL2 coefficient file (see D138), straits start state (init_STRAITS).

## D138: calc_opfil2_coeffs (OPFIL2 setup tables) ported; the recorded coefficient file is no longer needed

Source: OCNDYN2.f:887-1063 (module opfil2_coeffs), nbyzu/i1yzu/i2yzu from OCNDYN.f:505-518 (runs of LMU>=L). New files:
`fullfidelity/ocean_opfil2_coeffs.py` (`calc_opfil2_coeffs(lmu)` returns the same flat vector as ffo_opcoef.bin, so
`opfil2_ff.coef_from_dump` consumes it), `opfil2_coeffs_compare.py`, `ocean_step_opcoef_compare.py`; tests in tests/test_ocean_odiff.py.
Result (3 dates; the recorded files agree across dates): vector length 56,745 equal; all integer tables (nred=52,861, nfft=31, nfil=542,
nmn=44, nmin, jfft, n1/n2fft, jfil, i1/i2fil, indx_fil, n1/n2fil) and the assigned SMOOTH entries bitwise equal; REDUCO 665 of 52,861
values differ, max abs 1.1e-16 (libm sin / FMA level). 1,124 SMOOTH entries (n < the exit index of the Fortran do-loop) are never
assigned in the Fortran (uninitialised memory in the dump) and are never read by OPFIL2 (n >= NMIN only); they are excluded from the
comparison. Full step with BOTH ODIFF and the OPFIL2 coefficients computed (the file written from the port, unassigned SMOOTH
entries filled from the dump for the byte layout only): jan01 17520 uo 2.6e-10, vo 5.4e-10; 17521 uo 2.7e-10; 17526 uo 4.5e-10 -- identical
to the run with the recorded file at the printed precision. Assumes the serial decomposition (js0=2, js1=JM-1), which the recorded
nmn=44 confirms; geometry from the analytic OGEOM formulas with DLAT = 4*(pi/180).
Still recorded: AG2OG/IG2OG regrid fluxes, init_STRAITS start state.

## D139: JAX conversion of the dynamics step, stage 1: profile and the converted stages

**Profile** (dyn_step.py, numpy-pow mode, nov26 33312, warm, one 30-min step, 12-core CPU, no GPU; per stage kind, summed over the 5 leapfrog passes): total about 2.6 s. aadvt 0.607, qdynam 0.517, pgf 0.465, aflux 0.382, advecv 0.255, trop 0.081, advecm 0.059, filter_chain 0.058, iso 0.036, sdrag 0.029, matopmb 0.018, ke_init 0.010, ke_final 0.009, kea 0.005, se_init 0.005, pgrad 0.005, qscale 0.004, wsave 0.003, rest < 0.003 each. (Script: scratchpad prof.py; the same numbers come out of `dyn_step_compare.time_step`.)

**Converted to jitted JAX (new files in fullfidelity/, same operation order as the numpy ports, jax_enable_x64):**
- `dyn_jax_fft.py`: FFT72/FFTI (the dict-based dyn_fft72_ff transliteration traced with jnp arrays; elementwise ops only) and `avrx_field_jax` (AVRX over the 44 rows x 40 layers batch; the per-harmonic multiplier BYSN*DRAT is precomputed on the host with the same double product).
- `dyn_jax_advecv.py`: ADVECV whole (horizontal fluxes with the per-cell Fortran update order, polar-row fix, vertical advection vectorised over L because each level reads only its own old DUT, Coriolis, final division).
- `dyn_jax_pgf.py`: PGF whole: the top-down pressure/geopotential column recursion is a `lax.scan` over L (carry M,PU,PKU,PKPU,PKPPU), the bottom-up GZ integration a second `lax.scan`, then N-S/E-W derivatives, AVRX (through dyn_jax_fft), polar scaling, UT/VT update.  x**KAPA = `jnp.power` (numpy pow semantics; see below).
- `dyn_jax_filter.py`: FLTRUV (8-pass E-W Shapiro, angular-momentum fix as a `lax.scan` over I in the Fortran order I=IM,1..IM-1), fltry2 x2, CONSERV_AMB_EXT x2 (sequential L sum as `lax.scan`), ADD_AM_AS_SOLIDBODY_ROTATION, the GLOBALSUM sums (`seqsum_jax` = scan), calc_kea_3d (regrid_btoa_3d) and COMPUTE_WSAVE.
- `dyn_jax_pointwise.py`: SDRAG (all 45x72 columns at once, level loop LS1..LM unrolled; the T-range stop_model test is returned as a flag and raised on the host) and isotropuv (rows with COSV<0.15; shap1 with per-row sub-iteration counts via `lax.while_loop`; the two polar rows through the traced FFT72).
- `dyn_jax_env.py`: sets `XLA_FLAGS=--xla_cpu_max_isa=AVX` before jax is imported (see D140, FMA finding); `DYN_JAX_ALLOW_FMA=1` disables it.
- `dyn_step_jax.py`, `dyn_jax_compare.py`, `tests/test_dyn_jax.py` (D140/D141).

**Left in numpy, and why:** aadvt (+adv1d; 0.61 s, the largest stage: sequential moment recurrences with data-dependent branches, 391+320 lines of bitwise code, not converted in this task), qdynam/aadvq (0.52 s, 1075 lines), aflux/advecm/matop (0.44 s; the AVRX call inside aflux could reuse `avrx_field_jax`, not wired), calc_trop (0.08 s, per-column search loops), matopmb, conserv_se/ke, energy_fix, pgrad_pbl and the glue copies (< 0.04 s together).  Not ported: DISSIP and the SLP row loop (not part of the dynamics step).  Together the unconverted stages are about two thirds of the step time, so the whole-step speedup is bounded accordingly (D141).

**Not claimed:** bitwise agreement with the real Fortran or with the Intel libimf pow mode.  The JAX path is validated against the numpy chain in numpy-pow mode only; libimf pow cannot be called from JAX.

## D140: JAX dynamics stages validated against numpy and the real dumps; chained JAX step

**Method** (`dyn_jax_compare.py`, from fullfidelity/: `python dyn_jax_compare.py --steps 2 --endsteps 6`): (a) per stage: the numpy plan runs in boundary mode (`dyn_step_compare.Recorder`: before every dumped stage, inputs not produced by earlier ported stages are replaced by the real recorded `ffd_*` inputs); before each converted stage the workspace is copied, the JAX stage is run on the copy and its outputs are compared with the numpy stage's; for advecv/pgf also with the real per-call output dump.  2 steps per date x 3 dates (nov26 33312-33313, dec01 33552-33553, jan01 17520-17521), all 5 leapfrog passes.  (b) whole chained step `dyn_step_jax.dyn_step_jax` (converted stages + numpy for the rest) vs `dyn_step.dyn_step` (numpy-pow) and vs the real s3/s4 dumps, 6 steps x 3 dates.  Statistics: max|a-b|, relative to the field scale max|b|, elementwise relative, number of unequal elements.

**Finding 1 (important): XLA:CPU contracts a*b+c into FMA.**  Out of the box 23% of random jitted a*b+c results differ from numpy.  First run (default ISA, 3 dates x 1 step), JAX vs numpy per stage, scale-relative worst: advecv UT 5.0e-16, VT 4.3e-16; iso 1.7e-16; sdrag 8.3e-17; filter_chain U 1.7e-16 (V 0); kea 2.9e-16; wsave 2.0e-16; but pgf UT 3.8e-13, VT 3.4e-13, DUT 2.5e-12, DVT 1.8e-12, GZ/PHI 4.3e-14 (cancellation in the E-W pressure derivative amplifies 1e-16 differences in GZ), i.e. PGF DUT/DVT above the 1e-12 target.  End state of the step JAX vs numpy (default ISA): U 3.9e-13, V 5.8e-13 (worst over dates), MWS 1.6e-13, QMOM 7.9e-14, MVS 9.8e-14, others <= 1e-13.
**Fix:** `--xla_cpu_max_isa=AVX` (set by `dyn_jax_env.py` before jax import) removes the FMA instructions; jitted a*b+c is then identical to numpy.  Second finding: XLA's algebraic simplifier rewrites operations on closure constants ((x*c1)*c2 -> x*(c1*c2), x/const), which changed ADM by 2e-16 and isotropuv in a few hundred cells; the scalar constants of PGF and the iso constants are therefore passed as traced arguments instead of closure constants.

**Result with these two measures (this host, CPU, jax/jaxlib 0.5.3):**
- Per stage, JAX vs numpy, worst over 3 dates x 2 steps x 5 passes: **0.0 (bit-identical) for every field of every converted stage**: advecv UT,VT,U,V,UX,VX,DUT,DVT; pgf UT,VT,U,V,UX,VX,DUT,DVT,GZ,PHI,SPA(ADM); iso U,V,UX,VX,UT,VT; sdrag U,V; filter_chain U,V; wsave; kea (31 field series, 0 unequal elements).  (iso was 3.6e-17..1.1e-16 scale-relative, 225-308 unequal cells, before the traced-constants change.)
- JAX vs the real per-call dumps (advecv, pgf), equal to the numpy-vs-real numbers because the JAX output equals numpy's: advecv UT 2.9e-16, VT 2.1e-16; pgf UT 1.6e-13, VT 1.3e-13, DUT 5.1e-13, DVT 4.3e-13, GZ/PHI 1.3e-14, ADM 3.3e-15 (scale-relative; this is the numpy-pow vs libimf-pow-built Fortran difference, not a JAX effect).
- Whole chained step (6 steps x 3 dates), end state JAX vs numpy: **0.0 for all 22 compared fields (U,V,T,Q,QCL,QCI,MA,PEDN,PMID,PK,P,MASUM,TMOM,QMOM,MUS,MVS,MWS,GZ, WSAVE,KEA,PTROPO,PHI)**, i.e. JAX chain bitwise equal to the numpy-pow chain.  Hence JAX vs the real end state s3/s4 equals numpy-pow vs real; worst scale-relative over the 6 steps (nov26 / dec01): U 3.5e-13 / 2.5e-13, V 2.7e-13 / 2.4e-13, T 1.1e-15, Q 8.8e-15 / 4.8e-15, MA 6.4e-15, PK 7.3e-16, TMOM 3.2e-14 / 5.1e-14, QMOM 3.0e-14, MUS 3.1e-14, MVS 4.7e-14, MWS 9.8e-14, GZ 2.7e-14, WSAVE 5.6e-14, KEA 2.9e-14 (jan01 values in `dyn_jax_compare.py` output, same order of magnitude).  These bound the numpy-pow-vs-Fortran difference and are the numbers D123 reports for numpy-pow mode; they are not bitwise-with-Fortran (that needs libimf pow).
- Caveat on the bitwise statement: it holds on this CPU, jaxlib 0.5.3, with the AVX-only flag, for these 18 steps; other XLA versions/backends (and any GPU, where FMA is normal) may differ at the 1e-16..1e-13 level (see Finding 1 for the size of that effect).  Not claimed: bitwise with libimf/Fortran.

**Tests:** `tests/test_dyn_jax.py` (6 tests, 45 s warm cache, ~60 s wall): FMA-free probe; FFT/FFTI traced vs numpy (+mutation); every converted stage vs numpy on step 0 of nov26 in boundary mode (bound 1e-13, measured 0); advecv/pgf vs real dumps no worse than numpy (1.5x or 1e-14); direct advecv/pgf on dumped pass-3 inputs with mutations (perturbed DXV 1.001, KAPA x1.0001, both detected >1e-8/1e-6); end state vs numpy/real with bound 1e-12 and mutations (dropping SDRAG, swapping PGF before ADVECV, both detected by > 1e-7 / 1e-9).  Skipped if ff_data dumps are absent.

## D141: timing of the JAX dynamics step (CPU, no GPU) and structure

`python dyn_jax_compare.py --timing` (nov26 33312, numpy-pow, 12-core CPU, 3 warm calls each, per-stage min):

| stage (5 passes) | numpy warm | JAX warm | JAX cold (compile incl.) |
|---|---|---|---|
| advecv | 0.235 s | 0.039 s | 1.08 s |
| pgf | 0.490 s | 0.096 s | 6.45 s |
| iso | 0.043 s | 0.012 s | 11.33 s |
| sdrag | 0.034 s | 0.031 s | 1.67 s |
| filter_chain | 0.067 s | 0.026 s | 1.11 s |
| kea / wsave | 0.006 / 0.004 s | 0.001 / 0.001 s | 0.10 / 0.04 s |
| stages left in numpy (aadvt, qdynam, aflux, advecm, trop, ...) | 1.71 s | 1.77 s | |
| **whole step** | **2.6-2.8 s** (3 calls: 2.63, 2.61, 2.82) | **2.06-2.08 s** (2.06, 2.07, 2.08) | **23.5 s** cold (numpy cold 2.8 s) |

The converted stages sum to 0.88 s (numpy) vs 0.21 s (JAX; the figure includes jax<->numpy array conversion and the copy to writable arrays), a 4x speedup on those stages (advecv 6x, pgf 5x, iso 3.6x, filter_chain 2.6x, sdrag 1.1x); the whole step is 1.3x faster because about two thirds of the time (aadvt 0.6 s, qdynam 0.5 s, aflux 0.4 s, ...) stays numpy.  Compile cost: about 21 s once per process (iso 11 s and pgf 6 s dominate because the FFT72 straight-line code, thousands of ops per transform, is unrolled; the jit cache is per process, no persistent cache enabled).  With `DYN_JAX_ALLOW_FMA=1` (default ISA, FMA contraction allowed) the warm step was 2.0-2.2 s, i.e. no measurable speed gain from FMA here, so the bitwise-faithful mode costs nothing on this CPU.

**Jit structure:** not a single jit.  The step is the Python plan of `dyn_step.py` (leapfrog control flow, 5 passes) executing one jitted call per converted stage, with the workspace as numpy arrays between stages; the numpy stages (aflux, advecm, aadvt, qdynam, trop, ...) sit in between.  Making the whole step one jitted function needs all stages in JAX (aadvt/adv1d, aadvq, aflux/advecm/matop, trop first), the plan unrolled or expressed with `lax.fori_loop` over passes (the 5-pass structure and the NS control flow are static for NIdyn=4, so a Python-unrolled jit is possible; compile time would be large), and the data-dependent `stop_model` checks (ADVECM mass error, SDRAG T range) turned into returned flags.  Not attempted and not tested here.  On a GPU the per-stage dispatch and the host round trips between stages would dominate; this is the main reason to convert the remaining numpy stages next, aadvt first.  No GPU timing exists: all numbers are CPU.

## D142: AFLUX + ADVECM + MAtoP in JAX (dyn_jax_aflux.py)
`make_aflux(g, tab)` gives jitted `aflux` and `advecm` (MAtoP inside): same statement order as dyn_aflux_ff.py, Fortran SUM loops as
lax.scan carries, AVRX through the traced FFT72 (dyn_jax_fft), topography patch adjustment as an unrolled level loop with per-cell masks
on static patch blocks, MW recursion as a reverse scan, geometry as traced arguments, PK = jnp.power (numpy-pow semantics). Validated on all
real dumps (3 dates x 6 steps x 5 passes, AFLUX and ADVECM+MAtoP inputs): 0 unequal elements in every output field (bitwise vs the numpy
port; the numpy port is bitwise vs real only with libimf pow). Wind x6 stress input exercises the topography move branch (counted in the test).
Warm: aflux 0.40 s numpy -> 0.19 s JAX per step; advecm no gain (0.06 s, host transfers dominate). Compile ~12 s (aflux).

## D143: AADVT / adv1d in JAX (dyn_jax_aadvt.py)
adv1d / advection_1D_custom (qlimit=.false. only; the qlimit/limitq path stays numpy, it is dead in AADVT) with the cell axis as the
array axis (no transposes), upwind selection by where() on the (cyclic/clamped) neighbour, Courant counts as while_loop over trial 1..20 with
`active` mask, sub-stepping as fori_loop to the max nstep with a per-row mask (k < nstep); polar sums / fqu / fqv as scans; pow exponent traced.
Validated: 12 real calls (+ their stage checkpoints x1, y, z and per-row nstep) and all 21 aadvtS4 stress calls (nstep up to 4): 0 unequal in
rm, rmom, mm, fqu, fqv, checkpoints and nstep arrays, vs numpy and (S4 call) vs the real dump. Mutation: unmasked max-nstep run changes the result.
x6-scaled real input compared numpy vs JAX (both stop or agree bitwise). Warm 0.58 s -> 0.17 s. Compile ~2.3 s.

## D144: QDYNAM / AADVQ in JAX (dyn_jax_qdynam.py)
AADVQ0 cycle counts in JAX: NCYC trial while_loop (REAL*4 reciprocal), per-level NCYCXY search (while loops vmapped over levels), XSTEP/NSTEPX,
flux scalings, flow-out-both-sides masks (numpy lists -> boolean masks). AADVQ: nc cycle loop, L=1..LM+1 fori_loop with the vertical carry
(mwdn, fdn, fdn0, fmomdn) in the loop state, per-level NCYCXY inner loop, aadvqy, aadvqx with per-row masked NSTEPX sub-steps, checkflux, aadvqz.
Not in JAX: the extra-column branch (lminzij<LM): AADVQ0 z-extra partition and aadvqz_column advection stay numpy (fallback per call, main
sweeps still JAX); diagnostics sbf..scf3d not computed. Validated: 18 real calls and all 18 x8 stress calls (ncyc 4 and 6; 8 calls use the
numpy z-extra fallback): 0 unequal in q, qmom, MUs, MVs, MWs, ncyc, ncycxy, nstepx vs numpy AND vs the real dump `fin` files. Not reached: ncycxy>1,
ncyc>ncmax errors (error flags exist, untested). Warm 0.58 s -> 0.20 s (about 1.2 s on the stress calls with fallback). Compile ~4 s.

## D144b: chained step with D142-D144 stages (dyn_step_jax2.py, dyn_jax2_compare.py, tests/test_dyn_jax2.py)
`dyn_step_jax2.Kit` / `dyn_step_jax` extend D140's executor with aflux, advecm, aadvt, qdynam (dyn_step_jax.py untouched). Per-stage boundary mode (1 step/date): all 55
compared variables 0 unequal. 18-step end state (3 dates x 6): JAX vs numpy-pow chain 0 unequal in all 22 fields; JAX vs real equals numpy-pow vs real
exactly (u scale-rel 3.8e-13 worst, mus/mws 1e-13: the known libimf-pow gap). Whole step nov26 warm: numpy 2.70-2.79 s, JAX 1.01 s (2.7x;
D141 was 2.06 s); cold (compile) 40.5 s per process. Remaining numpy: trop 0.09 s, matopmb, se/ke/efix, pgrad, copies, z-extra branch of QDYNAM.
Tests: tests/test_dyn_jax2.py 8 tests, 110 s, all pass (exact equality, mutation checks: perturbed dyp, unmasked nstep, dropped aadvt/qdynam, w*1.001).
Caveats: needs --xla_cpu_max_isa=AVX (dyn_jax_env); CPU only, no GPU run; bitwise-with-libimf not claimed.

## D136b: F1-gate re-run with both land fixes (D135 precipitation conditioning, D136 irrigation)

`atm_step_compare.py --imf --dates nov26,dec01,jan01 --steps 0 --no-isolated` (libimf mode, step 0 from the real state, radiation and Ent exports recorded, categories A/B/C/D fixed in advance). **Ported GHY:** nov26 **MET** with named exception columns (11 fields B, 2 C, 1 A; worst W2GCM 5.1e-12 of scale); dec01 **PARTLY MET** (A 1, B 6, C 7; worst EGCM 9.9e-9); jan01 **PARTLY MET** (A 1, B 3, C 10; worst EGCM 7.6e-9). **Recorded land patch:** MET on all three dates (nov26 worst W2GCM 1.7e-12, dec01 worst EGCM 5.8e-13, jan01 worst EGCM 2.5e-12). Compared with D127-D129 (before the two land fixes): nov26 NOT MET -> MET; dec01/jan01 worst field 1.85e-7 / 2.1e-7 -> about 1e-8. Not re-run: steps 1-5 (CONDSE threshold flips dominate there regardless of the land model, D134), the libm mode, and the isolated-stage comparison. Log: scratchpad `f1_all_after_irrig.log`.

## D145: LSCOND batch in JAX (`clouds_lscond_jax.py`, `clouds_jax_env.py`)
All columns at once: main layer loop (L=LMCLD..1, downward PREBAR/PREICE/LHP carry) as `lax.scan`; CTEI layer pairs as `lax.scan`, the bounded `do ITER=1,9` with exit as `lax.fori_loop(0,9)` with a per-column active mask (all columns computed, selected by the alive/mix masks); the tail is whole-array (LMCLD,N) with a vector-add loop for WMSUM. Two modes: `libm` (exp/pow/get_dq through `jax.pure_callback` to the same scalar functions the numpy batch uses; not accelerator-resident) and `xla` (jnp.exp/jnp.power, on-device).
Result vs `clouds_lscond_batch` on the real step inputs captured from the chained CONDSE (nov26/dec01/jan01, steps 0 and 1, 3168 columns): **0 columns differ in any field, bit patterns included, in both modes** (6 of 6 runs each). Warm time 0.28 s (xla) / 1.15 s (libm callbacks) vs 0.47 s numpy; first call (compile) 6-8 s.
Findings that matter beyond this stage: (1) XLA:CPU's HLO algebraic simplifier changes IEEE results: x/35.0 -> x*(1/35.0) (20% of random results differ; x/10.0 33%), a/(b/c) -> a*c/b and (a/b)/c -> a/(b*c) (35%). With it, LSCOND differed in 43 of 44 compared fields (first seen as rh = ql/qsat). `--xla_disable_hlo_passes=algsimp` (set with `--xla_cpu_max_isa=AVX` by `clouds_jax_env.py`, import before jax) removes all of it (0 differences in the micro tests); this is the real cause behind the D139 "closure constants" workaround. (2) On this CPU build XLA's f64 exp and pow equal glibc libm (0 differences in 2e6 exp, 2e5 pow arguments), hence the xla mode is bitwise; this is a property of the CPU backend, a GPU will differ at 1e-16 and, by threshold sensitivity, flip branches in some columns (not testable here). Not tested: libimf mode in JAX (the libm callback mode routes through `sz.use_imf` in principle; not run). The "VMP: should not be here" stop is returned as flag `vmp_err` and raised on the host.

## D146: MSTCNV batch in JAX (`clouds_mstcnv_jax.py`)
Host: the LMIN loop (28 iterations), compaction of the columns passing the cloud-base test (padded to buckets 32..2048, gather/scatter in jit). JAX (jit): set-up, cloud-base test + MASS_FLUX (8-trip masked loop), the event block for the compacted columns (IC=1,2 as fori_loop, NPPL=1,2 as fori_loop, plume ascent as `lax.while_loop` with a per-column alive mask, downdraft/subsidence/precipitation as fori_loops with layer-range masks, microphysics size search as masked fixed-trip loop), and everything after the cloud-base loop. NUMPY: the QUS vertical advection ADV1D of the subsidence (incl. the scalar qlimit loop) runs as a host callback (`clouds_mstcnv_batch._adv_groups`), not converted. Poles and the `mc_new_ddrft_thetav=0` arm unsupported, as in the numpy batch. Diagnostics not reproduced, as in the numpy batch.
Result vs `clouds_mstcnv_batch` on the real MSTCNV inputs of nov26 step 1 (3168 columns, ~3283 cloud-base events): **0 of 43 output fields differ (value comparison)**; the full CONDSE comparison (D147) shows 0 differing exit fields on all six steps. Timing: compile 132-143 s per fresh process (22 jit instantiations over the buckets; `CLOUDS_JAX_CACHE=<dir>` enables the persistent cache), warm 4.9 s vs 4.3 s numpy, i.e. NOT faster on this CPU (compaction, host callbacks, masked full-width loops); the value is the accelerator-ready structure and the bitwise result. No GPU run was possible.

## D147: JAX CONDSE (`clouds_condse_jax.py`, `clouds_condse_jax_compare.py`, `tests/test_clouds_jax.py`)
`condse_step_jax` = `clouds_condse_batch.condse_step_batch` with `lscond_batch` and `mstcnv_batch` swapped (context manager) for the JAX kernels; everything else of the chain (column set-up, convective post-processing, bookkeeping, hand-off, the two pole columns, snow-age exp, recalc_agrid_uv) stays numpy.
Compared with the numpy batch on 3 dates x steps 0,1 (module carry chained per variant): **0 differing exit fields in all 6 steps (value and bit)**. Against the REAL exit state the JAX and numpy statuses are identical (libm mode: exact/bound/FAIL = 10/16/42, 11/18/39 (nov26), 10/16/42, 10/15/43 (dec01), 11/13/44, 11/18/39 (jan01), the known libm threshold-flip level of D126/D132; libimf is what matches the real model and is not available in the JAX path). Per step: numpy batch 4.8-5.6 s, JAX 5.6-6.0 s warm (xla mode LSCOND), first JAX step 163 s including compile.
Tests: `tests/test_clouds_jax.py`, 8 tests, 170 s in a fresh process (compile dominated): XLA flag/primitive checks, LSCOND libm and xla bitwise, MSTCNV bitwise, CONDSE bitwise + same status vs real, and mutation tests (1e-15 input perturbation; dtsrc; entrainment constant) that must change the result.

## D149: real-model dumps for a full model day (nov26 restart, 48 + 6 steps), one-ulp perturbed members (2026-10-06)

New directory `ff_data/nov26_day/` (1744 files, 37 GB; nothing existing was touched). Real instrumented ModelE (ifort 19.1.3, pristine tree copied to <scratch>/mE_day1/mE2, build 1.3 min, 0 errors) from `ff_data/_pristine_restarts/fort1_nov26_itime33312.nc` (itime 33312 = 1950-11-26 00:00, the first step of a model day; both fort.1/fort.2 restored), 54 steps (itime 33312-33365: the 48 steps of the day plus 6 so that the day boundary at 33360 lies inside the window), `FFD_START=33312 FFD_NSTEP=54`, OMP_NUM_THREADS=1. Wall 5 min 23 s for the full-dump control (6 jobs concurrent on 12 cores), rc=0.
**Which hooks widen.** All of them: every hook in the stack (D128 `ffa_step_{a,r,d,e}`, D125 `ffc_cse_in/out`, `ffp/ffs/ffl/ffg/fft`, `ffa_<it>_c1/c2`, D122 `ffd_state_s1..s4`, `ffd_<it>_{pre,post}_{condse,radia,surface,aturb}`) gates on the same `itime in [FFD_START, FFD_START+FFD_NSTEP)` test, so FFD_NSTEP=54 produced all 54 steps of every file type with no code change. Size 0.55 GB per step in total, 0.30 GB per step for the files `atm_step_fast.run_chain` actually reads (14.4 GB per 48-step day; D148's ~15 GB estimate holds for that set). NOT built (not needed for the atmosphere chain, so no dump exists for the day): the D118 ocean-chain hooks (FFO_*), the ffz_* ocean/ice dumps, the D101/D114 filter/glue dumps. The 54-step trajectory reproduces the existing 6-step dumps bit for bit (ffa_step a/r/e of steps 33312/33317, ffc_cse_in/out).
**New patch** `instrumentation/ATM_DRV_pert.f.patch` (unit 1410, re-grepped free; applies last): `ffpt_perturb` (+-1 ulp via `nearest()` in one cell of T or Q at the first step of the window, or in every cell: env `FFPT_PERT`) and `ffpt_dump` (U V T Q QCL QCI P at the end of atm_phase2, 6.4 MB per step, `ffpt_<tag>_<it>.bin`). With no env set both are no-ops; `ffpt_ctrl` equals `ffa_step_e` bit for bit on all 54 steps. Real members run from the same restart (3 min 24 s - 3 min 38 s each, concurrent): ctrl, p1 `T 36 23 10 +1`, p2 `T 20 12 20 -1`, p3 `Q 50 30 5 +1`, p4 `T 10 30 1 +1`, p5 every T cell +-1 ulp (checkerboard sign), 54 steps each.
**What fires at the day boundary** (MODELE.f:308-312 and 362-367, ATM_DRV.f `daily_atm`): `startNewDay` at the first step of each day (itime 33312 and 33360): diagnostics only (`daily_DIAG`, `reset_ADIAG`). `dailyUpdates` after the step ending at 00:00 (after step 33359): `daily_CAL`, `daily_OCEAN`, `daily_atm` = DIAG5A/DIAGCA diagnostics, `daily_atmdyn` (D117, ported), `daily_orbit` (solar position: COSZ1 is recorded), `daily_ch4ox` (RAD_DRV.f:1600-1698: **adds stratospheric/upper-level water vapour, Q(i,j,l) += xCH4*dH2O(j,l,month)*byMA, then copies the pole rows**: this IS atmosphere state and was not in the chain), `daily_RAD` (ozone/GHG/volcanic columns, and the SNOAGE aging RAD_DRV.f:1414-1425 which uses TDIURN), `daily_LAKE`, `daily_EARTH` (soil/Ent phenology), `daily_LI`, `UPDTYPE`. Measured on the dumps: between the end of step 33359 and the start of 33360 exactly MA, PEDN, PMID, PK, PDSIG, PEK, P (DAILY_ATMDYN: deltam -3.27e-11, MA change 2e-12) and Q (ch4ox: up to 2.7e-6 kg/kg, 49,399 of 132,480 cells, largest at the pole rows and in the upper layers) change; T, U, V, QCL, QCI, TMOM, QMOM and all hidden fields are unchanged. Our chain handles them as: DAILY_ATMDYN ported (bitwise on the real end-of-day state, 0 of 132,480 cells differ in MA/PEDN/PMID/PK/PDSIG/P/PEK, recorded constant MDRYA from the D117 dump); ch4ox as a RECORDED per-(j,l) added water mass DM = (Qa-Qe)*MAa (i-spread 4.7e-13 relative, so it depends on (j,l) only; reproduces the real Q to 4e-22 - structural check, the dH2O file is not read); SNOAGE taken from the next step's recorded CONDSE entry (our carried SNOAGE differed from it by up to 14 (numpy run) / 1.3 (JAX run) before replacement); everything else through the recorded tile/radiation inputs of the new day (ffg/ffp ... of 33360+ are the real post-daily records). Not ported: ch4ox's dH2O table, daily_RAD's SNOAGE aging, orbit, daily_EARTH/LAKE/LI.
Findings that matter: (1) the radiation steps in the window are 33312, 33317, ... 33362 (11; 10 in the first 48), (2) a one-ulp T perturbation is not local: after one step 132,407-132,480 of 132,480 T cells differ (p1 rms 4.5e-5 K, max 1.2e-2 K in a column ~25 cells away from the perturbed one; median |dT| 4e-14), i.e. a global ~1e-13 shift (cause inferred, not verified) plus CONDSE threshold flips, exactly the mechanism seen in our own chain (D134); the same ulp in Q stays at 2 cells / 2e-16 for two steps (p3). Owner: Glenn Tamkin. Sources: modelE2_planet_2.0 model/MODELE.f, ATM_DRV.f, RAD_DRV.f. Review: next session extending the window beyond one day.

## D150: one-model-day open-loop run of the chained atmosphere step (atm_day_open_loop.py), 54 steps (2026-10-06)

`fullfidelity/atm_day_open_loop.py` (new) drives `atm_step_fast`/`atm_step` step by step (own end state -> next start state; ATURB/PBL hidden state, CONDSE cloud/precip carry and LSCOND arrays carried) for 54 steps from the real nov26 start state. Recorded inputs (ATM_STEP_PLAN section 3.1): SRHR/TRHR of the last real radiation step held frozen over the next four steps (they equal each step's own recorded SRHR/TRHR exactly, checked on all 54), COSZ1 of every step, RADIA's CLDSS/CLDMC masking applied to OUR clouds on radiation steps, SURFACE tile/PBL/land-ice/land-patch(`recorded`)/aggregate records and Ent exports, ice/lake/ocean surface state, non-atmosphere CONDSE entry fields; at the day boundary DAILY_ATMDYN (ported) + ch4ox (recorded mass) + SNOAGE (recorded) as in D149. Compared after every step with the real end state (`ffa_step_<it>_e`) and the real CONDSE exit (`ffc_cse_out`).
**Paths run:** (a) numpy dynamics (dyn_step), libimf pow backend (`make_ctx(imf=True)`), batched numpy CONDSE (`clouds_condse_batch`), JAX-jitted surface chain as in D133: 14.2 s median per step (dyn 4.7, CONDSE 8.3, surface 0.65, filter 0.42, radia/dissip 0.03), step 0 67.6 s (JIT), **48 steps 821 s (13.7 min), 54 steps 951 s**. (b) JAX dynamics (`dyn_step_jax2`, XLA `--xla_cpu_max_isa=AVX`): 11.1 s median (dyn 1.25), step 0 100 s (compile), **48 steps 718 s (12.0 min), 54 steps 827 s**. About 8 of the 54 steps took ~27 s in both runs (node contention from my own concurrent jobs/other users, not diagnosed). The JAX path does NOT reproduce the numpy-imf step bit for bit in libimf mode (D144b: it equals the numpy-*pow* chain; PK differs from libimf at 1 ulp): after step 0 it differs from the real state by T rms 8.0e-5 K (a CONDSE threshold flip) where the imf-numpy run is at 7.2e-14 K. The JAX path is an equally valid rounding-level member of the same ensemble (below), not an equivalent of the numpy step.
**Divergence curve (numpy run; rms over valid cells of (ours - real), all layers):**

| step k | T (K) | Q (kg/kg) | U (m/s) | V (m/s) | P (mb) | mass-weighted global-mean dT (K) / dQ |
|---|---|---|---|---|---|---|
| 0 | 7.2e-14 | 1.5e-17 | 1.2e-14 | 1.3e-14 | 1.3e-13 | +8.9e-16 / -3e-20 |
| 1 | 2.0e-4 | 1.1e-6 | 1.7e-4 | 9.4e-5 | 1.8e-11 | -1.6e-6 / +4.5e-9 |
| 5 | 1.3e-3 | 9.6e-6 | 6.8e-3 | 6.0e-3 | 4.6e-3 | -8.0e-6 / +2.1e-8 |
| 11 | 5.7e-3 | 2.2e-5 | 3.0e-2 | 2.6e-2 | 4.0e-2 | +9.1e-5 / -1.9e-7 |
| 23 | 2.0e-2 | 4.3e-5 | 8.2e-2 | 7.6e-2 | 8.8e-2 | -1.5e-4 / +2.8e-7 |
| 35 | 4.2e-2 | 6.0e-5 | 1.3e-1 | 1.3e-1 | 1.2e-1 | -1.6e-4 / +2.5e-7 |
| 47 | 6.0e-2 | 7.3e-5 | 1.6e-1 | 1.7e-1 | 1.3e-1 | -2.4e-4 / +2.7e-7 |
| 53 | 7.3e-2 | 7.5e-5 | 1.8e-1 | 1.9e-1 | 1.3e-1 | -3.0e-4 / +1.8e-7 |

Step 0 is at rounding level (A/B category, as D134). The first inexactness appears at CONDSE of step 1 (new flipped columns; CLDSS rms 2.5e-3, max 0.72 at step 1) and then spreads: e-folding rate 0.076/step in T (0.041 Q, 0.059 U, 0.065 V) from step 3 on, smoothly through the day boundary (step 48 shows no discontinuity: T rms 6.0e-2 -> 6.2e-2). Cloud fields at the CONDSE exit at step 47: CLDSS rms 5.9e-2 (scale 1), CLDMC 2.2e-2, TAUSS 3.5 (scale 100), CSIZMC 11; QCL/QCI rms 1.6e-5 / 3.4e-5 kg/kg (see D151 for the floor). The JAX run ends at the same level (T 7.2e-2 K, U 1.8e-1, V 1.9e-1 at k=53). Global-mean drift of T over steps 10-53: -1.8e-4 K mean (numpy run, negative sign on 42 of 44 steps), -1.0e-4 K (JAX run, mixed signs); of Q +2e-7 kg/kg (numpy run). These are 0.1-0.3 mK and are discussed against the real members' own global-mean scatter in D151.
**What an open-loop run does and does not validate.** It validates, for one day on a realistic forcing: dynamics + CONDSE + SURFACE/ATURB + DISSIP + FILTER + the glue, the 5-step radiation heating cadence (frozen SRHR/TRHR, per-step COSZ1), the RADIA cloud-mask carry, the day-boundary handling of the atmosphere state, and that the divergence from the real run grows like the real model's own chaotic growth (D151), with no sign of a slow drift or a blow-up. It does NOT validate: radiation or any cloud/albedo-radiation feedback (SRHR/TRHR respond to the REAL run's clouds and temperatures, so they act like nudging toward the real trajectory and bias our divergence low, increasingly with time; harmless for <=1-2 days, invalid for F3), land/ocean/ice/Ent feedbacks (recorded land patch, tile records of the real run), pointwise equality (impossible after the first step, D3/D134), or anything beyond one day. Not exercised: `land_mode='ghy'` over the day, libm (non-imf) mode, the other two dates, a second day, any ocean-chain state for the day (not dumped), DAILY_ATMDYN on OUR (drifted) state against the real value (our deltam -3.8e-11 vs real -3.3e-11 is only reported).
Files: `fullfidelity/atm_day_open_loop.py`, `fullfidelity/atm_day_report.py`, `fullfidelity/tests/test_atm_day.py` (11 tests pass in 2.5 min; includes real-dump determinism checks, the day-boundary replay on the real end-of-day state, 6 steps of the replay incl. radiation freeze equality), outputs `ff_data/nov26_day/ours_imf_np/` and `ours_imf_jax/` (`step_<it>.npz`, `run.json`). Owner: Glenn Tamkin. Sources: D127-D134, D117, D144b. Review: when radiation is made closed-loop (D148 option B2).

## D151: noise floor of the real model vs the open-loop divergence (one model day, 5 real members) (2026-10-06)

Method (fixed before looking, D148 section 3.2 style): real members = the unperturbed control of the same binary and 5 perturbed members (D149), 54 steps each; floor N(k) of a metric = [min, max] over the 5 members of rms(member_k - control_k); ours = rms(ours_k - control_k) for the same metric. Class per step: within (ours <= max member), near (<= 2x max), beyond (> 2x), below (< 0.5 x min). Metrics: whole-column rms, layer 1/5/21 rms, zonal-mean rms, |mass-weighted global-mean difference|, for T Q U V QCL QCI P (`atm_day_report.py`; json, md and png in the scratchpad `d149_overlay_np_jax.md`, `d149_overlay.png`). Spread of members at step 47 (T rms): 6.9e-2 .. 7.3e-2 K (p1-p5), at step 23 2.1e-2 .. 2.5e-2; the Q-only member p3 starts 12 orders lower (2e-16 at step 0, T members 5e-5 - 1e-4) and is within a factor 3 of them by step 5 and the same order by step 11.
**Verdict (numpy run; 54 steps):**
- Whole-column rms: **T, U, V within the real noise floor at all 54 steps** (largest ratio ours/max-member 0.92 T, 0.90 U, 0.98 V; late steps ours is 0.85-0.9 of the lowest member, e.g. k=47 T 6.0e-2 vs [6.9e-2..7.3e-2]). **Q near at 15 steps** (max ratio 1.09, first exceeds the maximum member at step 2), P near at 2 (1.05, step 2), QCL near at 4 (1.09, from step 10), QCI near at 3 (1.11, from step 7). Nothing beyond 2x. **The variables that first exceed the real members' maximum are Q and P (step 2, by <= 9 %), then QCI (step 7) and QCL (step 10)**; U/V/T never do in the whole-column rms.
- Finer metrics are less clean (5 members only): zonal-mean rms near at T 8 steps (max ratio 1.18), Q 18 (1.16), P 12 (1.26), V 7 (1.06); |global-mean difference| near for T at 18 steps (max 1.76, step 43), Q 14 (1.67), V 13 (1.75), P 8 (1.96), and one 'beyond' for QCI (2.46, step 24); layer-wise rms: QCL at layer 5 beyond at 2 steps (max 5.3 at step 5, where the member floors of that layer are tiny), U/V layer 1 near at 1-2 steps (<= 1.2), T layer 21 below the floor at 38 steps (ours smaller than half the smallest member). In absolute terms the global-mean T difference of ours (-1e-4 to -3e-4 K) is the same size as the members' own (|mean over steps 10-53| of 1.6e-4, 9e-5, -7.6e-5, 2e-5, 1.6e-5 K, random signs); our numpy run has a persistent negative sign (42 of 44 steps) that the JAX run does not (mixed), so a small systematic cold/moist drift can be neither claimed nor excluded from one day.
- The two runs of our port (numpy vs JAX dynamics, differing at rounding level) diverge from each other exactly like real members (rms T 8.0e-5 -> 3.1e-4 (k=1) -> 1.6e-3 (k=5) -> 5.3e-3 (k=11) -> 2.0e-2 (k=23) -> 6.0e-2 (k=47) vs floors 4.3e-3..6.7e-3 at k=11 and 6.9e-2..7.3e-2 at k=47): our port behaves as one more member of the real model's chaotic ensemble. Early steps (k<=2) are uninformative: the floor range spans 12 orders (p3 vs p1/p2/p4/p5) and ours (7e-14 numpy, 8e-5 JAX) lies inside it. Growth rates (per step, from step 3): ours T 0.076 vs members 0.078-0.098 (p3 0.098), Q 0.041 vs 0.040-0.059, U 0.059 vs 0.057-0.087, V 0.065 vs 0.069-0.087.
Statement for F2 (one day, open loop): **our divergence from the real run is within the real model's own chaos level for T, U, V at every step, and at or just above its upper edge (<= 1.1x) for Q, P and the cloud condensates; it is never beyond 2x in the whole-column rms.** The acceptance band proposed in D148 ([0.5 N, 2 N]) is met for T, U, V, Q, P, QCL, QCI whole-column rms at k>=3 (QCL layer 5 and global-mean metrics excepted, above). Limits: 5 real members of which 4 are single-cell perturbations (p5 is the only everywhere-perturbed member, whose curve is inside the others), one start state, one day; the open-loop recorded radiation/surface nudge ours toward the control, which biases ours low (consistent with ours sitting at 0.85-0.9 of the lowest member late in the day) while the real members have their own radiation feedback; so 'within' here is necessary, not sufficient, evidence. A bug that produces a systematic error smaller than the chaotic divergence (< ~1e-2 K / 0.05 m/s in a day) would not be detected by this test; that needs the monthly-mean ensemble comparison (D148 S2) or stage-level checks. Not done: more members (D148 proposes 6-8), the 5-day horizon, the ghy land mode, the libm path.

## D152: radiation server, scratch build reproducing the instrumented P2SAoM40.bin (no model change)
Built the real model with `ATM_DRV_atmstep.f.patch` + the new `ATM_DRV_radsrv.f.patch` in its own scratch tree (1 min 40 s, 0 errors; original tree untouched) and ran 6 steps from the nov26 restart (itime 33312) with the hook active in dump mode: all 24 `ffa_step_*` dumps are byte-identical to `ff_data/nov26` (24 of 24), so the build reproduces the recorded trajectory and the hook is passive. Model timer: RADIA 10.9 s per radiation step (steps 0 and 5), 0.0008 s otherwise; whole 6-step run 51 s with dumps. Details: `d152_build_notes.md` (for build_and_run.md).

## D153: radiation server (file protocol) and the RADIA READ list
`instrumentation/ATM_DRV_radsrv.f.patch` (unit 1440): at `CALL RADIA` the model overwrites every RADIA input from a packet (52 records: T Q PK PMID PDSIG MA BYMA PEDN, 15 cloud hand-off arrays, TAUSS TAUMC CLDSS CLDMC, RQT KLIQ SNOAGE LTROPO, RSI ZSI SNOWI POND_MELT FLAG_DSWS FLAKE DLAKE FLICE FLAND FEARTH, GTEMPR1-4 TSAVG WSAVG SNOWLI ZSNOWI BARESW FRSNOW SNOWD), zeroes AIJ, calls the real RADIA (SOCRATES black box, unmodified), writes an output packet (T Q SRHR TRHR RQT KLIQ SNOAGE CLDSS CLDMC FSF TRSURF ALB FSRDIR SRVISSURF FSRDIF DIRVIS DIRNIR DIFNIR SRDN CFRAC COSZ1 + the exact per-call AIJ delta, which carries the TOA fluxes) and stops. `RADSRV_MODE=dump` records the live packets of every radiation step without changing the run. Python: `radiation_server.py` (`run_radiation(state, itime)`), `radiation_server_compare.py`, `tests/test_radiation_server.py`. The READ table with source lines and status (packet / restart / time / param / which dump holds it) is `scoping/RADIATION_SERVER_PLAN.md` section 2; the RADIA cloud block and post-processing, not read by D148, are summarised there (planet_rad build: the cloud block is only the taulim mask, the layer-reversed copy of the 15 CLOUDS_COM arrays and diagnostics). Cost: 27 s per call for step 0 (about 16 s start-up + 11 s RADIA), 42 s for step 5; file protocol only, no persistent loop.

## D154: bitwise oracle and input audit of the radiation server (nov26 steps 0 and 5)
Real step-0/step-5 state (33 of 52 fields from the existing ffc_cse_in/out, ffa_step_r dumps, equal to the live model bitwise; the other 19 from the new dump-mode packet) fed to the server: `T Q SRHR(0:40) TRHR(0:40) COSZ1` BITWISE equal (0.0 difference) to the recorded `ffa_step_<it>_r`; `RQT KLIQ SNOAGE CLDSS CLDMC FSF TRSURF ALB FSRDIR SRVISSURF FSRDIF DIRVIS DIRNIR DIFNIR SRDN CFRAC` bitwise equal to the live model outputs of the same trajectory; CLDSS/CLDMC/SNOAGE equal to the next step's ffc_cse_in; FSF(1)*COSZ1 equal to SRHEAT of all 5418 ocean tile records of ffs_33312. AIJ (TOA fluxes): exact delta vs rounded `AIJ_after-AIJ_before` of the live run differs in 194,607 of 5,497,920 entries, max abs 5.8e-11, max relative 4.6e-11 (consistent with rounding, bound not proved); TOA fluxes have no independent record. Audit (52 packet fields perturbed one at a time, 210 s, 10 concurrent runs): 45 move outputs at 1e-6; of the 7 that do not, TAUSS/TAUMC only affect the CLDSS/CLDMC mask, FLAKE/DLAKE move outputs when changed macroscopically, MA/TSAVG are not read by this build's RADIA (byMA is), SNOWI is read only in the KSIALB==1 branch (KSIALB of this rundeck not checked). Not auditable through the packet: PVT (Ent), BCdalbsn, RSDIST, gas/ozone/volcanic tables, random seeds (restart/time state; equal to the real model only at the restart date). Not tested: dec01/jan01, other steps. `tests/test_radiation_server.py`: 3 passed, 105 s.

## D155: one model day with free-running radiation (real RADIA via the radiation server, every 5th step), 54 steps (2026-10-06)

`fullfidelity/atm_day_free_rad.py` (new) is `atm_day_open_loop.py` (D150, numpy dynamics, libimf pow, batched CONDSE, `land recorded`) with one change: on the 11 radiation steps of the window (k = 0, 5, ..., 50; itime 33312..33362) the RADIA stage is the real, unmodified RADIA (SOCRATES a black box) called through `radiation_server`, with the 52-field packet assembled from OUR chained state: T and Q after OUR CONDSE of that step, PK PMID PDSIG PEDN MA (BYMA = 1/MA), LTROPO, the 15 cloud hand-off arrays and TAUSS/TAUMC/CLDSS/CLDMC of OUR CONDSE exit, SNOAGE/RQT/KLIQ carried from the previous server output (restart values from the live packet at step 0). The surface side stays REPLAYED from the real model: RSI ZSI SNOWI POND_MELT FLAG_DSWS FLAKE DLAKE FLICE FLAND FEARTH GTEMPR1-4 TSAVG WSAVG SNOWLI ZSNOWI BARESW FRSNOW SNOWD (20 fields) are the live values at that step. Server outputs replace the recorded ones exactly where D150 used them: SRHR/TRHR (frozen over the next four steps), COSZ1, masked CLDSS/CLDMC (carried to the next CONDSE entry), plus Q (negative-Q reset), SNOAGE, RQT, KLIQ (carried). T is advanced by the same `radia_apply` as before. The tile SRHEAT/TRSURF records feeding SURFACE are still the real ones: this is a free-running ATMOSPHERE over a replayed surface; radiation does not feed back on the surface here.
**New recording (dump mode, no new patch, build or hook):** the existing server binary (`mE_radsrv/mE2`, read-only) run once in `RADSRV_MODE=dump` over the 54-step window (I file `YEARE=1950,MONTHE=11,DATEE=27,HOURE=3`, as D149), 201 s wall, 1 thread. Recorded in `ff_data/nov26_day/rsv_n26_<it>_in.bin` / `_out.bin` for the 11 radiation steps (33.7 MB in + 55.5 MB out each, 1.2 GB total): the live 52-field input packet (this includes the 19 live-only fields that D154 had only for steps 0 and 5: RQT KLIQ LTROPO BYMA GTEMPR1-4 WSAVG ZSI SNOWI POND_MELT FLAG_DSWS DLAKE SNOWLI ZSNOWI BARESW FRSNOW SNOWD) and the live outputs (including `AIJD`, the rounded AIJ delta). Check that the dump run is the recorded trajectory: at all 11 radiation steps the 33 fields that exist in ffc_cse_in/out and ffa_step_r are bitwise equal to the live packet, BYMA = 1/MA bitwise, and the live outputs T Q SRHR TRHR COSZ1 equal `ffa_step_<it>_r` bitwise (`live_vs_dumps.txt`).
**Plumbing oracle at step 0 through the new assembler** (`python atm_day_free_rad.py check0`, 32 s): the assembler applied to the REAL step-0 state gives a packet equal to the live packet in 52 of 52 fields; the server returns T Q SRHR(0:40) TRHR(0:40) COSZ1 bitwise equal to the recorded `ffa_step_33312_r` and all other outputs (RQT KLIQ SNOAGE CLDSS CLDMC FSF TRSURF ALB FSRDIR SRVISSURF FSRDIF DIRVIS DIRNIR DIFNIR SRDN CFRAC) bitwise equal to the live outputs; the TOA AIJ columns equal the live rounded delta to 7e-12 (the D154 rounding). In the day run itself step 0 uses OUR chained state (T 7.2e-14 K rms from the real one, D150), so the server output there differs from the recorded one at rounding level, not bitwise; the day run is otherwise step-0-identical to D150 (rms T 7.21e-14, Q 1.49e-17, U 1.17e-14, V 1.31e-14, P 1.30e-13).
**Run:** `python atm_day_free_rad.py run --tag free_np`, 54 steps, **2250 s (37.5 min) total = 1056 s chain + 1194 s in 11 sequential server calls** (31, 47, 60, 72, 90, 105, 121, 137, 144, 178, 190 s: each call re-runs the real model from the restart up to its radiation step, so the cost grows with the step). Per call: `T` from `radia_apply` equals the server's own T update bitwise (max |diff| 0.0), COSZ1 equals the recorded one (0.0), and RADIA's taulim mask applied to OUR clouds equals the server's CLDSS/CLDMC (0.0), on all 11 calls. Outputs: `ff_data/nov26_day/ours_free_np/` (`step_<it>.npz`, `run.json`, `rad_in_<it>.npz` assembled packets, `rad_out_<it>.npz` server outputs with the TOA AIJ columns, `report.json/.md`).
**Two things found on the way.** (1) `radiation_server._prepare` ends the model window at 1950-11-26 03:00 (6 steps; enough for D154's steps 0 and 5): a server asked for a later radiation step runs to the end of the window without reaching the hook and returns no packet (the first attempt died at step 10, 33322). Fixed in the new file (`run_radiation_day`: same preparation, window to 1950-11-27 03:00); `radiation_server.py` itself is unchanged (suggest merging the window into it). (2) The AIJ column numbers: a static count of `DEFACC.f` gave 225/239/224/233 for SRNFP0/TRNFP0/SRINCP0/SRNFG (empty columns; this build has kaij = 1660, not the 750 of `radiation_server.KAIJ`, which is stale and unused); the real columns are +152 (the offset was not traced to its source) and were identified by value on the step-0 live output, not by the source: SRINCP0 = 376 (global mean 349.6 W m-2 = S0/RSDIST^2/4), SRNFP0 = 377 (237.1), TRNFP0 = 391 (-230.4; sign convention: net TOA = SRNFP0 + TRNFP0 = +6.7 W m-2 at this instant), SRNFG = 385 (global mean 165.56 vs 165.63 for SRHR(0)*COSZ1). The identification rests on those magnitudes, not on a named index table.
**Carried SNOAGE is stale relative to the real model** (as it was in D150): the real SNOAGE changes between radiation steps through surface-side snowfall resets that the replayed surface does not feed to our chain (our carry differs from the live value by up to 20-50 in 3.2e3 cells). Sensitivity (two server calls with the live SNOAGE put into our saved packets, k=15 and k=50): TOA net rms 0.21 / 0.07 W m-2 and column SRHR rms 7e-4 / 4e-4 (vs the 8.8 / 20.6 W m-2 and 0.52 / 1.27 of the state-driven differences), i.e. negligible for this day.
Files: `fullfidelity/atm_day_free_rad.py`, `fullfidelity/tests/test_atm_day_free_rad.py` (6 tests, 36 s; the day itself is not run in the test). Owner: Glenn Tamkin. Sources: D149-D154, RAD_DRV.f (SNOAGE, TRHR(0) definition), DEFACC.f. Review: when the surface is also made free (F3).

## D156: radiation-free-running day vs the real run: divergence and radiative-flux differences (2026-10-06)

Per step exactly as D150 (rms of ours - real over valid cells, all layers; mass-weighted global-mean differences; T Q U V QCL QCI P; `atm_day_report.py`). Free vs open loop (same numpy-imf path, so the only difference is the radiation):

| step k | T free / open (K) | Q free / open | U free / open (m/s) | V free / open (m/s) | P free / open (mb) |
|---|---|---|---|---|---|
| 0 | 7.2e-14 / 7.2e-14 | 1.5e-17 / 1.5e-17 | 1.2e-14 / 1.2e-14 | 1.3e-14 / 1.3e-14 | 1.3e-13 / 1.3e-13 |
| 5 | 1.41e-3 / 1.34e-3 | 1.01e-5 / 9.6e-6 | 7.3e-3 / 6.8e-3 | 6.0e-3 / 6.0e-3 | 4.5e-3 / 4.6e-3 |
| 11 | 5.8e-3 / 5.7e-3 | 2.24e-5 / 2.20e-5 | 3.07e-2 / 3.00e-2 | 2.41e-2 / 2.57e-2 | 3.3e-2 / 4.0e-2 |
| 23 | 2.18e-2 / 2.01e-2 | 4.49e-5 / 4.29e-5 | 8.5e-2 / 8.2e-2 | 8.4e-2 / 7.6e-2 | 8.7e-2 / 8.8e-2 |
| 35 | 4.24e-2 / 4.16e-2 | 5.87e-5 / 5.97e-5 | 1.41e-1 / 1.28e-1 | 1.46e-1 / 1.30e-1 | 1.13e-1 / 1.16e-1 |
| 47 | 6.71e-2 / 6.05e-2 | 6.63e-5 / 7.27e-5 | 1.85e-1 / 1.60e-1 | 1.94e-1 / 1.65e-1 | 1.40e-1 / 1.31e-1 |
| 53 | 7.95e-2 / 7.26e-2 | 7.10e-5 / 7.55e-5 | 2.05e-1 / 1.78e-1 | 2.21e-1 / 1.86e-1 | 1.31e-1 / 1.30e-1 |

The two runs are identical to rounding through step 1 (free minus open: T 4.6e-15 at k=1), separate at the first radiation call after chaos has acted (k=2: T 5.4e-5) and then differ from each other by as much as each differs from the real run (free minus open at k=53: T 7.9e-2, Q 7.7e-5, U 0.21, V 0.22, P 0.13): they are two chaotic realizations. Growth rate (e-folds/step from step 3): free T 0.0778, Q 0.0379, U 0.0614, V 0.0695 (open 0.0764, 0.0405, 0.0591, 0.0649; members 0.078-0.098, 0.040-0.059, 0.057-0.087, 0.069-0.087). Global-mean differences (steps 10-53, mass-weighted): dT free +7.6e-5 K mean (negative at 8 of 44 steps) vs open -1.8e-4 K (42 of 44); dQ free -2.2e-7 kg/kg (39 of 44 negative) vs open +2.7e-7 (3 of 44); the five real members' own means are -7.6e-5..+1.6e-4 K and -3.5e-7..+1.2e-7. Cloud fields at the CONDSE exit (rms, free / open): k=23 CLDSS 4.4e-2 / 4.0e-2, TAUSS 3.4 / 3.4, QLSS 2.5e-5 / 2.7e-5, QISS 5.3e-5 / 4.1e-5; k=47 CLDSS 6.9e-2 / 5.9e-2, CLDMC 2.3e-2 / 2.2e-2, QLSS 2.3e-5 / 1.6e-5, QISS 8.6e-5 / 4.1e-5, PREC 8.4e-2 / 7.6e-2.
**Radiative-flux differences at the 11 radiation steps** (free server output minus the real run; the real SRHR/TRHR from `ffa_step_r`, the other fields and the TOA columns from the dump-mode live outputs; global means area-weighted; `ours_free_np/report.md`). SRHR(0)*COSZ1 is the net SW at the surface, TRHR(0) the downward LW at the surface (RAD_DRV.f 4437-4551); there is no sign-checked "net surface" flux in the exports, so the two components are given.

| k | net SW sfc: gmean diff / rms diff (W m-2) | LW down sfc: gmean / rms | TOA net (SRNFP0+TRNFP0): real gmean, gmean diff / rms diff | column SRHR*COSZ1 rms diff (of field rms) | column TRHR rms diff (of field rms) | SRDN rms diff (of 600) |
|---|---|---|---|---|---|---|
| 0 | 0 / 0 | 0 / 0 | 6.69, -7e-15 / 3e-12 | 0 (3.96) | 8e-16 (7.74) | 0 |
| 5 | +0.079 / 3.8 | -0.014 / 0.74 | 6.86, +0.063 / 3.3 | 0.11 (3.99) | 0.36 (7.78) | 7.3 |
| 10 | -0.003 / 6.6 | -0.030 / 1.6 | 7.41, -0.016 / 5.9 | 0.38 (3.73) | 0.92 (7.75) | 12.1 |
| 20 | +0.131 / 14.5 | -0.118 / 2.6 | 4.50, +0.060 / 13.3 | 0.85 (3.77) | 1.67 (7.70) | 26.0 |
| 25 | -0.754 / 17.5 | +0.176 / 3.1 | 4.95, -0.697 / 15.4 | 1.08 (3.84) | 1.97 (7.66) | 31.2 |
| 30 | -0.194 / 20.0 | +0.020 / 3.7 | 7.73, -0.586 / 17.2 | 1.25 (4.05) | 2.16 (7.66) | 36.1 |
| 40 | +0.552 / 23.3 | -0.083 / 4.5 | 10.55, +0.410 / 20.7 | 1.13 (3.84) | 2.47 (7.69) | 41.8 |
| 50 | +0.224 / 25.0 | -0.141 / 5.3 | 6.46, +0.104 / 20.6 | 1.27 (3.92) | 2.59 (7.72) | 43.2 |

(All 11 steps are in `report.md`.) Global means: the largest |difference| over the day is 0.75 W m-2 (net SW at the surface, k=25), 0.18 (LW down), 0.70 (TOA net), with alternating signs and no trend; pointwise (rms) the TOA-net, surface-SW, SRDN and column heating-rate differences grow with the state divergence (TOA net 3.3 -> 20.6 W m-2, column SRHR rms 0.11 -> 1.27 and TRHR 0.36 -> 2.59 in the units of the exports, a third of the field rms at the end). Not available: a noise floor for the fluxes (the real members record U V T Q QCL QCI P only, no radiation).

## D157: noise-floor overlay of the free-radiation day, compared with the open loop (2026-10-06)

Same method and thresholds as D151 (members p1-p5 against the control, floor [min..max] per step, within <= max, near <= 2x max, below < 0.5 min, beyond > 2x max).
**Whole-column rms, 54 steps (free; open loop in brackets):** T within 54 (54), U 54 (54), V 54 (54); Q within 49, near 5 (39, 15); P within 52, near 2 (52, 2); QCL within 43, near 11 (50, 4); QCI within 44, near 10 (51, 3); nothing beyond 2x in any variable. Largest ratio ours/max-member (steps >= 3): T 0.96 (0.92), U 0.95 (0.90), V 0.96 (0.98), Q 1.13 (1.09), P 1.00 (1.05), QCL 1.44 at k=34 (1.09), QCI 1.95 at k=47 (1.11). Ratio ours/lowest member at k=53: T 1.00 (0.91), U 0.96 (0.83), V 0.97 (0.82), Q 0.98 (1.04); at k=47 T 0.97 (0.87), U 0.95 (0.82), V 0.97 (0.82). Zonal-mean rms: near at T 14, Q 9, V 9, P 16, QCL 10, QCI 8 steps (U 0), max ratio 1.18 T, 1.15 Q, 1.17 V, 1.50 P, 1.53 QCL, 1.98 QCI; |global-mean difference|: beyond at one step for QCL (2.50, k=34) and two each for QCI (max 2.44, k=25; first k=24) and P (max 2.47, k=4), near at 7 T, 6 Q, 12 V steps (max ratio 1.74 T, 1.71 V) (open loop: one beyond, QCI 2.46).
**Verdict.** (1) Free-radiation drift is within the real model's own chaos level for T, U, V at every one of 54 steps (whole-column rms), within for Q except 5 early steps (k = 2, 4-7, <= 13 % above the largest member, the same early steps as the open loop, which had 15), within for P except 2 steps; near (never beyond 2x) at 10-11 steps for the cloud condensates QCL/QCI, the one place where the free run is clearly further from the real run than the open loop (1.4-1.95x vs 1.1x the largest member). Those condensate excursions are single-cell events, not a smooth drift: at k=47 one cell (a real cloud-ice cell, 2.3e-2 kg/kg, absent in ours) carries 70 % of the squared QCI difference, at k=34 one cell 37 % (QCL) and 28 % (QCI); the open loop has the same kind of cells at lower weight, and the five members cannot say whether 1.4-1.95x is outside the true spread of this heavy-tailed statistic. (2) Compared with the open loop (D150/D151): the free curve is LARGER for T, U, V late in the day (k=53 T +9 %, U +15 %, V +19 %), smaller for Q (-6 %), the same for P; at late steps it sits at 0.95-1.0 of the lowest member instead of the open loop's 0.82-0.91, i.e. with radiation responding to our state the divergence rises into the member band, as the D150 caveat predicted for a nudged (recorded-radiation) run, and its growth rates match the members'. The open loop's persistent negative global-mean dT (42 of 44 steps, -1.8e-4 K) is absent in the free run (+7.6e-5 K, 8 of 44 negative), and dQ changes sign (open +2.7e-7, free -2.2e-7); both fall inside the spread of the five members' own means, so a systematic bias is neither shown nor excluded (inferred: the open loop's sign was a property of the frozen real radiation; one realization each). (3) Fluxes: the global-mean radiative differences are <= 0.75 W m-2 and show no trend (D156), pointwise differences grow with the cloud-field divergence.
**What this validates:** that the atmosphere (dynamics + CONDSE + the surface response of the replayed surface + real RADIA acting on OUR T, Q, clouds, with the 5-step cadence, the heating-rate freeze, the cloud-mask carry and the carried RQT/KLIQ/SNOAGE) behaves as one more member of the real model's chaotic ensemble over one day from one start state, with the radiative feedback of the atmosphere on itself closed and no drift or blow-up; and that the radiation-server packet assembled from our state is bitwise the real one when our state is the real one (step 0 oracle through the driver). **What it does not:** radiation feedback onto the surface (surface fluxes SRHEAT/TRSURF/albedo-dependent tile records are replayed, so ocean/ice/land/snow respond to the REAL radiation); any multi-day behaviour (one day; chaos growth e-folds in 13 steps, the metrics saturate at day's end), other start dates (dec01, jan01: the server restart only exists for nov26), a flux-level noise floor (members have no radiation records), pointwise correctness (impossible, D134), the significance of the condensate excursions (5 members, one start state, one day), and the cloud-aerosol/chemistry components of RADIA not in the packet (PVT, BCdalbsn, orbit, gas/ozone/volcanic tables, random seeds: restored from the restart, valid at nov26 steps 0-53 only because the model day is the one the restart belongs to). The cost: 37.5 min/day of which 19.9 min is the file-exchange server (re-running the real model to each radiation step); a persistent server would remove most of it. Owner: Glenn Tamkin. Sources: D149-D154, atm_day_report.py, ours_free_np/report.md. Review: when the surface is closed (F3) or a second start date is available.

## D158: stiff GHY cells (ffnit >= 12) of nov26_day: the ffg record holds only 11 sub-iterations, so both ports advanced those cells over less than dt; fixed (not a conditioning problem, not a JAX problem)

Owner: full-fidelity port session (Claude, for G. Tamkin). 2026-10-06. Sources read-only: giss_LSM/GHY.f advnc 2119-2720 (time loop 2389-2416), gdtm 3057-3145, evap_limits 652-1007; instrumented dumps ff_data/*/ffg_*.bin.

Root cause (two coupled facts, both shown by runs):
1. The ffg record has 450 slots; the Ent block `ffent(13, ff1)` starts at 1-based slot 300, so ghy_compare.unpack keeps only `n_avail = min(ffnit, 11)` iterations (ghy_compare.py, "writer overflow guard"). run_cell and ghy_advnc_test.build_batch then use dt = sum(recorded dts). For a cell with ffnit >= 12 that sum is < 900 s (e.g. 33321 (63,17): 11 recorded dts sum to 843.952 s, the real 12th iteration has dts = 56.048 s; 33329 (42,21): 751.95 s of 900). The ports advance only a fraction of the real time step, and the accumulated fluxes (ashg, alhg, aevap) and tbcs of the cell are wrong by the missing 6-27 % of the step. ffnit itself is exact; it is the number of records that is truncated.
2. The missing iterations could not simply be replayed because ghy_ref/ghy_jax never needed gdtm (recorded dts imposed, D9), and `ghy_ref.gdtm` as written was wrong for iterations >= 2: GHY.f:905 and 907 store the potential evaporation epb, epv in MODULE variables that gdtm reads in the next iteration (betas = evapb/epb, (evapvw*fw+evapvd*(1-fw))/epv, GHY.f:3102-3110), whereas ghy_ref.evap_limits keeps them local, so self.epb/self.epv stay at the pre-loop 1.0 and gdtm returns e.g. 218.2 s where the real run had 68.36 s (cell 33329 ij (42,21)). Harmless for ffnit <= 11 (recorded dts used), found only when the loop was reconstructed.

Evidence (all run in this session; scratchpad survey.py/recon.py/fin.py):
- Survey of ALL ffg files (nov26, dec01, jan01, nov26_day = 55 files), every cell with ffnit >= 8 run through ghy_compare.run_cell: 692 cells; ffnit 8/9/10/11: 412/181/67/17 cells, all 677 match the real record (ashg rel < 1e-6, tbcs < 1e-6 K); ffnit 12/13/14/15: 5/5/4/1 cells = 15, ALL 15 fail (ashg 1.8e-2 .. 3.8e-1 relative, tbcs up to 0.12 K), 14 of the 55 files, all in nov26_day. The older three dates have no cell with ffnit >= 12. The failure is therefore tied to ffnit >= 12 exactly (the record width), not to soil/snow/vegetation state or region: the 15 cells are at ij (64,19), (63,17), (42,21), (40,21), (41,21). Your four files list one cell each because the others fall under the loose test_ghy_jax tolerances or other substep; 11 more files (33319, 33320, 33330-33333, 33335, 33338, 33339 and the second substep of 33329/33336) contain such cells too.
- Mechanism check, cell 33321 (63,17) substep 1, ffnit 12: the sum of the 11 recorded dts is 843.9520; 900 - 843.9520 = 56.048 = the last recorded dts (a halved step), i.e. one more iteration of 56.048 s, which is exactly the 12th real iteration (ffnit = 12). Same arithmetic for all 15 cells.
- Fix check: ghy_ref_nit.advnc_full runs the real loop (hydra, xklh, gdtm; dts = dtr if dtm >= dtr else min(dtm, dtr/2)), with gdtm fed the stored epb/epv. With dts COMPUTED (not the recorded ones, nit not imposed): gdtm reproduces all 11 recorded dts of every one of the 15 cells (max |dts - recorded| 1.4e-13 s) and the loop ends after exactly ffnit iterations in 15/15 cells; ashg old -> new -> real: 33321 (63,17) 5.628e5 -> 5.89725e5 -> 5.89725e5; 33329 (42,21) 2.350e4 -> 2.86715e4 -> 2.86715e4; 33336 (41,21) 1.014e4 -> 1.6385e4 -> 1.6385e4; 33337 (41,21) -5.012e4 -> -6.02233e4 -> -6.02233e4. Over the 15 cells: max |ashg-real|/|real| 5.4e-7, max |tbcs-real| 9.8e-8 K, alhg/aevap rel 3.8e-6; 10 of 15 cells are bitwise equal in ashg and tbcs.
- Non-regression: all 592 cells with ffnit 8..11 of nov26_day, recomputed with gdtm-derived dts instead of the recorded ones: worst ashg rel / tbcs abs 5.7e-14, nit == ffnit in every one.
- JAX: ghy_advnc_test_nit.build_batch_nit (pads the Ent arrays, takes the extra dts from the numpy loop for ffnit > 11 cells only, dt_total = 900) with ghy_jax.advnc(..., max_substeps=width): files 33321, 33329, 33336, 33337, first 40 cells + every stiff cell each: stiff cells ashg < 1e-5, tbcs < 1e-5, alhg/aevap < 1e-4 relative, all other fields within the test_ghy_jax TOLERANCES; the old build_batch on the same cells still fails (ashg > 1e-2), so the test is not vacuous.

Residual (not removable with the existing dumps): the Ent exports of iterations > 11 (cnc, betadl, lai) are not recorded; the fix reuses those of iteration 11. This leaves alhg/aevap rel 3.8e-6 (33319, 33320, 33321), ashg rel 5e-7 (33329 substep 2) and abetad 4.4e-7 relative in the stiff cells (abetad is the mean of betad = sum(betadl) over iterations). To close it: a larger record (second ffg file or units 1490-1499, e.g. write ffent(1:13, 12:ffnit) of the cell), then a rebuild of mE_stiff from the instrumented tree. Not built: the cells match to <= 3.8e-6 without it, and ffnit >= 12 cells are 15 of 82,000+ cell records here (0.02 % of 55 files * 1506).

Files (all new; nothing existing edited): fullfidelity/ghy_ref_nit.py (advnc_full, run_cell_full), fullfidelity/ghy_advnc_test_nit.py (build_batch_nit), fullfidelity/tests/test_ghy_stiff_nit.py (20 tests, 1 s), fullfidelity/tests/test_ghy_jax_stiff_nit.py (5 tests, 71 s). All pass.

Exact diffs to apply to existing files (not applied):
 - ghy_ref.py, evap_limits: after the lines `epb = ...` and `epv = ...` (ghy_ref.py:888-891) add `self.epb = epb; self.epv = epv` (also store `self.epbs`, `self.epvs` if used elsewhere); then ghy_ref.advnc can compute dts itself (move the dtr loop of ghy_ref_nit.advnc_full into it) instead of taking recorded dts.
 - ghy_compare.run_cell: `dt = 900.0` (the model's NIsurf substep, equal to the sum of the recorded dts for every ffnit <= 11 cell) and call `ghy_ref_nit.advnc_full(col, ent_iters, dt, snowm, ffnit=refs['ffnit'], use_recorded_dts=False)` instead of `col.advnc(...)`.
 - ghy_advnc_test.build_batch: replace by ghy_advnc_test_nit.build_batch_nit (or merge it); every caller of ghy_jax.advnc (land_chain.run_ghy, ghy_advnc_test.run, tests/test_ghy_jax._build_run) must pass `max_substeps=ent_dts.shape[1]` (default 11 raises a scan length error on the padded arrays).
 - tests/test_ghy_jax.py: drop KNOWN_STIFF_CELL_FILES and the xfail once the two lines above are applied.
 - land_chain/atm_step land_mode 'ghy': a stiff cell in a chained run is not affected by the record limit (no ffg record; the live loop would be needed); the chained GHY currently takes dts from the record, so land_mode 'ghy' on nov26_day substeps with ffnit >= 12 should use build_batch_nit. A free-running land would need gdtm ported to JAX (not done; numpy gdtm is now validated against 11 x 15 recorded dts).
Limits: Ent still a recorded input; assumption flagged above (iteration > 11 Ent exports reuse iteration 11).

### D159-D161 (2026-10-06/07): persistent radiation server (real RADIA, SOCRATES untouched) - built and validated

**D159 built.** `instrumentation/ATM_DRV_radsrv_persist.f.patch` (diff -u vs the PRISTINE `ATM_DRV.f`; 2 hunks: `call radsrvp_hook` before `CALL RADIA`, routine `radsrvp_hook` appended; applies after `ATM_DRV_atmstep.f.patch` with offsets 1 and 131, 0 rejects; fixed-form lines <= 72 columns; units 1500-1503 only (1500-1529 reserved)). Env `RADSRVP` unset = the hook returns at once. Check: build with the hook, env unset, `FFD_START=33312 FFD_NSTEP=6`: the 24 `ffa_step_*.bin` are byte-identical to `ff_data/nov26` (cmp 24 of 24), rc 0.
Behaviour: the model starts from the restart, reaches the first radiation step (RADSRVP_ITIME) once, then the hook serves text requests over two FIFOs (`SERVE <itime> <seed|-> <in> <out>`, `PING`, `STOP`; EOF of the request pipe also ends the model). Each SERVE sets the model clock (copy of the entry clock stepped with `nextTick`, so no clock arithmetic of ours) and Itime, runs the daily updates of every day boundary crossed (`daily_orbit`, `daily_RAD`, `daily_earth`, per RADSRVP_DAILY, default ORE), overwrites the 52 RADIA inputs from the packet, zeroes AIJ, sets the random seed IX (entry value for `-`), calls the REAL `RADIA`, writes the 22-field output packet (AIJ exact). Bad requests (not a radiation step, itime before entry, earlier day than the server day, bad/short packet) return `ERR` and the server survives.
Python: `radiation_server_persist.py` (change: `SCRATCH` overridable with env `RADSRVP_SCRATCH`), `radiation_server_persist_oracle.py`, `tests/test_radiation_server_persist.py` (one change: the repeat-count assertion `n >= 3` was a miscount of the test's own call sequence, which has 2 repeats; now `n >= 2`; no numerical tolerance was changed).
Build (own scratch tree `<scratch>/mE_persist/mE2`, script `<scratch>/mE_persist/work/build.sh`: rsync of the pristine tree minus ModelE_Support, `patch ATM_DRV.f` with atmstep then radsrv_persist, `gmake RUN=P2SAoM40`; BUILD_RC=0, about 2 min). Run: `export RADSRVP_SCRATCH=<scratch>/mE_persist; cd fullfidelity; python -m pytest tests/test_radiation_server_persist.py`.

**D160 validation (all measured).**
- (a) nov26 steps 33312, 33322, 33332, 33362 and 33317 (5 steps, the last across the day boundary at 33360, daily updates O+R+E run by the server): 21 fields bitwise equal to the recorded live packet `ff_data/nov26_day/rsv_n26_<it>_out.bin`; AIJ within 1e-9 of the live rounded difference. Steps 33312 and 33317 equal the one-shot server on all 22 fields bitwise (AIJ exact), the persistent call being made after another packet (33322).
- (b) call order 33312, 33322, 33332, 33322, 33312, 33317, 33362 on one server: repeated packets bitwise identical on all 22 fields incl. AIJ. A request for an earlier day, and a non-radiation step, are refused with the server alive.
- Note on the seed: the model's RADIA seed at a step is a trajectory property; the client passes `ffc_cse_out SEEDS[1]` of the step (this is what the entry READY reply reports at the entry step: nov26 -588724193 = SEEDS[1]). With the seed `-` (entry seed) at step 33317 the 21 fields are still bitwise equal to live, but AIJ differs from the live delta by up to 1.0 (random-overlap diagnostics only); with the real seed AIJ matches to 1e-9.
- (c) cost: start-up 18-20 s (once per server); per call 10.6-11.0 s in the model (RADIA + packet IO), 10.8-11.1 s wall in Python; nov26 smoke 10.66/10.60/10.69/10.79 s. One-shot server for comparison: 27-190 s per call. Single thread, other jobs on the node.
- (d) oracles added with the real step-0 live packets (`radiation_server_persist_oracle.py dump|oracle <date>`; live dump packets for dec01 33552 and jan01 17522 already in `ff_data/<date>/rsv_n26_<it>_{in,out}.bin`): dec01 (seed 180935071) and jan01 (seed -1450606225, restart is two steps before its first radiation step) `all_bitwise: true`: T Q SRHR TRHR COSZ1 vs the recorded `ffa_step_<it>_r`, the other 16 outputs vs the live packet, CLDSS/CLDMC/SNOAGE vs the next step's CONDSE input; AIJ vs rounded live delta 4.5e-13 (jan01 shown). Per call 11.0 s (dec01), 10.8 s (jan01).
- (e) `tests/test_radiation_server_persist.py`: 8 passed in 299 s (0 skipped), including clean STOP (rc 0) and no orphan process after SIGKILL of Python or exit without stop.

**D161 limitations (precise).**
- Time can only move forward by whole days: any step of the entry day in any order, later days reached by running the daily updates; an earlier day than the server's current one is an error (restart the server). The day-boundary result was validated on exactly one boundary (nov26 33360, step 33362); not validated beyond one day or at a month/year boundary (the daily updates `daily_orbit`/`daily_RAD`/`daily_earth` run from the entry day's state: `daily_earth`'s Ent vegetation update uses the date only; anything the real daily updates change from the model's own evolved state, e.g. tdiurn-based diagnostics, is not reproduced and does not enter RADIA's packet inputs, but is untested for vegetation change over many days).
- The server's non-packet state is that of the entry step; the 52-field packet carries the inputs the one-shot audit (D154) found necessary. Order independence is shown on nov26 only (5 distinct steps, 2 repeats); dec01/jan01 were exercised at their single first radiation step.
- Needs the FIFO/PDEATHSIG machinery (Linux). Server RADSRVP_* env names are read into 256-character buffers (relative names).
- Not done here: the re-run of the free-radiation day through the persistent server (should equal D155-D157 bitwise) and the wiring of `atm_day_free_rad.py` to it; estimated saving per D155-D157's 11 calls: about 30 min of server time to about 2 min of server time plus one 20 s start-up (estimate from the measured per-call cost, not measured on the day run).

**Independent re-check by the parent session (2026-10-06 22:05, own script, own request order 33317, 33312, 33322, 33312):** 21 of 22 output fields bitwise equal to the recorded `rsv_n26_<it>_out.bin` at every call, the repeated 33312 call identical to the first; server time 10.6-10.7 s per call. AIJ vs the recorded AIJD: 5.8e-11 at the entry step 33312; 1.0 at 33322 when no seed is passed (the seed caveat above, confirmed).

**D158 diffs applied (2026-10-06 22:30).** `ghy_ref.evap_limits` stores `epb`/`epv`; `ghy_compare.run_cell` runs `ghy_ref_nit.advnc_full` over dt = 900 s with computed dts (lazy import; old path kept when there are no Ent records); `ghy_advnc_test.build_batch` now reconstructs ffnit >= 12 cells (old builder kept as `build_batch_recorded`); `land_chain` and the tests pass `max_substeps=ent_dts.shape[1]` (jit with `static_argnames`). `tests/test_ghy_jax.py` xfail removed: 72 passed. One explicit exception, not a global relaxation: for the four files with reconstructed stiff cells `abetad` is bounded at 5e-6 (measured 4.4e-7..2.2e-6; the Ent exports of iterations > 11 are unrecorded, see Residual above); every other file/field keeps its original tolerance. The four `test_old_path_fails_non_vacuous` checks now call the pre-D158 route directly, because `run_cell` itself became exact (difference 0.0 on those cells). Tests: ghy_ref 5, stiff_nit 20 passed; ghy_jax 72 passed; land_chain group passed in the earlier run (337 passed, 7 failed, all 7 explained and fixed above); full regression on the final tree still to be run.

# D163: minimal AIJ-style diagnostics for the F3 monthly comparison (scope, port, validation)

Date 2026-10-06. Code `fullfidelity/f3_diagnostics.py`, tests `fullfidelity/tests/test_f3_diagnostics.py` (31 passed, ~32 s, nothing skipped on this host).
Not committed. Sources: the real model tree (read-only) `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model`, the real monthly acc files, the
ff_data dumps. SOCRATES/RADIA is not ported or modified; the RADIA columns below are the real RADIA's own output as returned by the radiation server.

## 1. Scope: what F3 compares, how the real model accumulates it, what exists on disk

**What F3 compares** (RADIATION_AND_F2_PLAN.md section 3.2-3.4, FULL_FIDELITY_PLAN/GOAL: proposals, not an existing project decision): monthly means of about 25-30 `aij` fields plus
4 `aijl` fields and zonal means, port vs the real ensemble, per field global mean and 46-point zonal mean. The set is a *proposal* in the plan ("none verified yet"); this entry
reads the real code for each field. No set of F3 fields was fixed by the project beyond that proposal.

**How the real model accumulates** (all read in the source, not assumed):
- `AIJ` (here `aij_loc`, 1660 columns in this build, 46 x 72) and `AIJL` are *sums* over the accumulation period, written to `<MON><YEAR>.acc*.nc` and to the restart (`fort.1/2.nc`,
  float64 in the restart, float32 in the acc files). A monthly mean is formed at print time (`DIAG_PRT.f:3119-3131`, `ij_mapk`): numerator `aij*scale/(idacc(ia)+teeny)`; if
  `denom_aij(k)>0` the denominator is `aij(denom)/(idacc(ia(denom))+teeny)` and the map is the ratio. `scale_aij`, `ia_aij`, `denom_aij`, `sname_aij` are stored in the acc file.
- Each column is incremented at its own call site with its own sampling counter (`DIAG_COM.f:901-904`: `ia_src=1` every step, `ia_rad=2` radiation steps (every 5th), `ia_srf=3`
  surface samples, `ia_dga=4` DIAGA calls, ...). The sites that matter here:
  - `DIAGA` (`DIAG.f:98-856`), called from `DYNAM` (`ATMDYN.f:352-356`) in the even leap-frog pass when `MODDA<2`, `MODDA = Mod(NSTEP+4-NS+NDAA*NIdyn, NDAA*NIdyn+2)`; the rundeck has
    `NDAA=13` (DIAG_COM default 7), NIdyn=4, so once per 54 dynamics steps of 450 s = 13.5 source steps: **4 calls in the 54-step window** (real `idacc(ia_dga)` 88 -> 92). It does
    pressure-level interpolation of T, Q, Z, RH, U, V (`DIAG.f:392-470`), omega (`:472-496`), surface/sea-level pressure (`:276-296`), the column water `qatm` and the AIJL
    `TempL/SpHuL/z` (`:488-496`). It reads the *mid-dynamics* state plus the previous step's `atmsrf%TSAVG/QSAVG`.
  - `accum_ma_ia_src` (`DIAG.f:1408`, `ATM_DRV.f:504`): AIJL `airmass` += MA every step.
  - `RADIA` (`RAD_DRV.f:4750-4790` inside the radiation block that `RAD_DRV.f:2502` skips when `MODRD/=0`; `RAD_DRV.f:5479` adds `S0*COSZ1` to `IJ_SRINCP0` on *every* step with the last radiation step's `S0`).
  - `CONDSE` (`CLOUDS2_DRV.F90:1474`) `IJ_PREC += PRCP`, every step (also `IJ_PRECMC` :1137, `IJ_SNWF` :1135/1463 and ~25 cloud columns).
  - `SURFACE` (`SURFACE.f:386-387, 1745-1830`): `evap` every substep (`-dtsurf*qflux1`); `tsurf, qsurf, usurf, vsurf, wsurf, tauus/tauvs, trdn_surf, pblht, tgrnd(IJ_TG1)...` only on the
    substeps with `MODDSF=MOD(NIsurf*ITime+NS-1, NDASF*NIsurf+1)==0`, i.e. **one substep in three** (36 of 108 in the window; pattern by itime mod 3: substep 1, substep 2, none).
  - Others not ported here: `SOATM_DRV.f:2149` (sst), `SEAICE_DRV.f:1336` (sivol), GHY/LAKES/LANDICE columns, `conserv`/`consrv` (DIAGCA), ISCCP, AJ/AJL/AGC budgets.

**What real-model output exists on disk (looked, not assumed):**
| item | content | use |
|---|---|---|
| `ModelE_Support/prod_runs/P2SAoM40/{DEC1949, JAN..NOV1950}.accP2SAoM40.nc` | 12 monthly acc files of ONE unperturbed run (cold start 1 Dec 1949), full AIJ/AIJL/...; JAN1950: idacc[0]=1488 (itime 17520 -> 19008), NOV1950: 1440 | the only real **one-month** reference |
| `PARTIAL.accP2SAoM40.nc` | acc of a 1-step partial run (itime 16033, idacc 1) | not useful |
| `ff_data/_pristine_restarts/fort1_{nov26,dec01,jan01}_*.nc` | restarts; **nov26 and dec01 contain the running acc** (idacc[0]=1200 / 1440, aij float64; dec01 acc equals NOV1950.acc to float32 rounding); **jan01 has no acc block** (month start) | window differences |
| `ff_data/nov26_day/rsv_n26_*_out.bin`, field `AIJD` | the real RADIA's AIJ *increment* of 11 radiation calls (33312..33362) in all 1660 columns | RADIA-site validation only |
| `ff_data/nov26_day/ffa_*`, `ffd_*`, `ffc_*` ... | per-step real states (54 steps) | inputs for the accumulators |
| **no per-step AIJ and no 54-step acc of the window existed** | | produced here, below |

**The real one-day reference did not exist; it was produced here.** The real binary of the radiation-server build (`<scratch>/mE_persist/mE2/model/P2SAoM40.bin`, no RADSRV env, so the unmodified
RADIA/physics) was run from `fort1_nov26_itime33312.nc` for 54 steps (YEARE=1950,MONTHE=11,DATEE=27,HOURE=3 replacing the first YEARE line of `I`, as `radiation_server._prepare` does; 3 min; rc 0).
Its end-of-run `fort.2.nc` acc minus the acc stored in the restart is the real 54-step accumulation of all columns (idacc increments: ia_src 54, ia_rad 11, ia_srf 36, ia_dga 4, 483 of 1660 AIJ
columns non-zero). Saved as `ff_data/nov26_day/real_acc54_nov26.npz` (diff, before, after, idacc). That the same trajectory as the 54-step dumps was run is evidenced by every DIAGA/RADIA/CONDSE
field below agreeing to 1e-14 with accumulators fed by the *dumps* (a different trajectory could not).

**One real month does exist (single unperturbed member); what does NOT exist** (so it blocks or qualifies an F3 verdict): (i) the real perturbed ensemble (the plan's 8 members of January) and thus the
real monthly-mean noise floor sigma per field (only the 5-day D3 floor is measured); (ii) a demonstration that a re-run from `fort1_jan01_itime17520.nc` reproduces `JAN1950.acc` (bitwise reproducibility of a restarted
segment was shown for 5 days only); (iii) any port-side month. What would produce (i)/(ii): the real binary above started from the jan01 restart for 1488 steps (the 54 steps took 3 min, ~3.3 s/step, i.e.
~80 min per month per member by that measurement, ~10-12 core-hours for 8 members, 3 cores max here) with +-1 ulp perturbations (the existing `ffpt` perturbation patch) and the acc file taken from `fort.2.nc`/the monthly acc.

## 2. What was implemented (`f3_diagnostics.py`)

`F3Acc` re-implements the accumulation at the same call site and sampling rule as the Fortran, from the model state at that site (arrays in model layout; `to_nc_layout()` gives the acc-file layout):
`diaga` (pressure levels of T,Q,Z,RH,U,V,omega, p_freq counters, prsurf/prsurfq/slp/slpq, rh_layer1, qatm, AIJL TempL/SpHuL/z), `airmass`, `prec`, `radia`, `surface` (+ `surface_samples`, the 1-in-3 rule),
`field_from_aij`/`global_mean` (the monthly-mean rule above), and a window driver `run_window` that feeds the accumulators from the real per-step states. `diaga_state` gets the DIAGA-time workspace from the
existing bitwise chained dynamics (`dyn_step.dyn_step(..., itime=it, hook=...)`, stage `diaga`, started from the real step-start state); `PHI`, `MW`, `PK`, `PEDN`... are that workspace's values. Recorded inputs of the
window run: `atmsrf TSAVG/QSAVG` (ffa dumps), `COSZ1`, `TRHR(0)`, CONDSE exit `PREC`, end-of-step `MA`, RADIA's AIJ increment (radiation server dump packet).

## 3. Validation (measured, all vs the real model's own accumulation over the 54-step nov26 window)

Method: accumulators fed with real per-step states; compare `ported - real` per column over the whole (46 x 72) map; residual = max|diff| / max|real increment of that column| (a tolerance of 1e-12 of that scale
is in the test; nothing was loosened). Counters: ported idacc increments `{ia_src 54, ia_rad 11, ia_srf 36, ia_dga 4}` = real, exactly.

| field (AIJ column) | source in the real code | ported | validated against | residual (relative to column scale) |
|---|---|---|---|---|
| t_/q_/z_/u_/v_/rh_/omega_ x 20 pressure levels (cols 56-75 q, 77-96 rh, 162-181 t, 217-236 z, 246-265 u, 266-285 v, 294-313 omega; incl. the F3 set t_850, t_500, t_200, z_500, u_200) | DIAG.f:392-496 (DIAGA) | yes | real 54-step acc increment, 4 DIAGA calls | worst per family: t 8.2e-15, q 7.7e-15, rh 5.4e-15, z 8.3e-15, u 5.2e-15, v 9.7e-15, omega 4.0e-15 |
| p_freq_<level> (cols 13-32, count of calls with the level above ground) | DIAG.f:392-470 | yes | same | 0 (exact) |
| prsurf (151), prsurfq (152), slp (153), slpq (154) | DIAG.f:276-296 (`SLP` of Utilities, TS_SLP = atmsrf TSAVG; `SLP_FROM_T1` not defined in this build, as the match shows) | yes | same | 6.4e-15, 6.1e-15, 1.0e-14, 8.5e-15 |
| rh_layer1 (97), qatm (100) | DIAG.f:296, 493 | yes | same | 5.6e-15, 5.3e-14 |
| AIJL TempL (19), SpHuL (20), z (21) | DIAG.f:488-496 | yes | same (aijl diff) | 5.6e-15, 5.5e-15, 6.7e-15 |
| AIJL airmass (22) | DIAG.f:1408-1423 | yes | same (aijl diff) | 3.2e-14 |
| prec (315) | CLOUDS2_DRV.F90:1474 | yes | same | 4.4e-15 |
| incsw_toa (376) | RAD_DRV.f:5479 every step with persistent S0 (+ the radiation-step value inside the RADIA increment) | yes (S0 recovered from the server's SRINCP0 / COSZ1, a derived value) | same | 2.4e-14 |
| srnf_toa (377), trnf_toa (391), srnf_grnd (385) and the other 80 columns RADIA increments in the window (pcldt 49, pmccld, pcldl/m/h, LWPrad, IWPrad, FRMP, btemp_window, TOA/surface SW-LW budget, clear-sky and CRF terms, band fluxes 503-523, aerosol band-6 columns 406-429, ...: 83 columns total + 376) | RAD_DRV.f:4750-4790 in the radiation block | yes, **as the real RADIA output** (server AIJ increment, not computed here) | same | 0 (exact) to 1.2e-16 |
| evap (322) | SURFACE.f:1749 (every substep, `-dtsurf*qflux1`, dtsurf=900) | yes | same | 9.6e-15 |
| tsurf (182), qsurf (76), trdn_surf (395), tauus (291), tauvs (292) | SURFACE.f:1759-1810 (sampled 1 substep in 3: `surface_samples`) | yes (TSAVG/QSAVG/UFLUX1/VFLUX1 from the per-substep ATURB-exit dumps `ffa_<it>_c1/c2_out`, TRHR(0) from `ffa_step_r`) | same | 1.9e-14, 2.1e-14, 3.6e-14, 5.9e-15, 5.0e-15 |
| usurf (286), vsurf (287), wsurf (288), pblht (237), tgrnd (184; IJ_TG1 `SURFACE.f:1760`), gusti (289), RHsurf (98) | SURFACE.f:1759-1819 | **no** (need atmsrf USAVG/VSAVG/WSAVG/DBLAVG/GTEMPS after *each* substep; the dumps have USAVG/VSAVG only at step end, no WSAVG/DBLAVG/GTEMPS) | - | - |
| sst (204), sss, ssh, sivol (243), simass (242), ts_oice, ocean/ice/lake/land state columns (fractions 1-12, soil/snow/canopy/lake 36-48, 107-143, 187-203, ...) | SOATM_DRV.f:2149, SEAICE_DRV.f:1336, GHY/LAKES/LANDICE | **no** (surface/ocean/ice/land state is replayed, not computed, in every chained day) | - | - |
| prec_mc (321), snowfall (333), clwp (99), cldw/cldi (101/102), pscld/pdcld, mc cloud columns (53,54,149,150), cnv/scnv frequency | CLOUDS2_DRV.F90:937-965, 1135-1137, 1463, 1537 | **no** (inside CONDSE columns; dumps carry only the exit PREC/PRECSS, not PRCPMC separately) | - | - |
| sensht (356) and the other heat/water budget columns (dSE_Dyn 352, dKE_Dyn, runoff_soil 335, netht_*, ...) | SURFACE.f:1987, ATM_DRV conservation, GHY/LAKES | **no** | - | - |
| ISCCP, `aijk`, `aj`, `ajl`, `agc`, `consrv`, `adiurn`, `tdiurn`, `aijmm` (tsurf min/max) | DIAG*.f, DIAGCA | **no** | - | - |
| monthly-mean rule `field_from_aij` / `global_mean` | DIAG_PRT.f:3119-3131 | yes (rule transcribed) | sanity only: applied to JAN1950.acc it gives tsurf 11.87 C, slp 1011.0 hPa, prec 2.83 and evap 2.85 mm/day, t_500 -18.5 C, z_500 5583 m, u_200 17.1 m/s, srnf_toa 242.5 and trnf_toa -233.9 W/m2, pcldt 54.5 %. Not compared with a printed real map; the global-mean weighting (`ij_avg`) was not read line by line | - |

Totals: 257 AIJ columns accumulated (80 of them straight from the RADIA output, ~160 pressure-level/DIAGA, the rest as listed), all with non-zero real increments in the window and all equal to the real
accumulation at <= 5.3e-14 relative; plus 4 AIJL columns. 226 of the 483 real columns that changed in the window are not accumulated by this module (list above).

## 4. Limits and caveats (what this does and does not show)

- The accumulators are validated **given the real inputs at the call site** (DIAGA state from the bitwise real-state dynamics, recorded TSAVG/QSAVG, recorded PREC, recorded/server RADIA output). That validates the
  accumulation logic, sampling rules, interpolation and counters. It does **not** validate a chained *free-running* state: there, the inputs differ from the real ones by chaos and the monthly comparison is statistical (the
  plan's protocol); the accumulators are agnostic (they take any state dict), but the chained atmosphere (`atm_step.py:208` calls `dyn_step` with `itime`, so the `diaga` stage exists) was **not wired** to call them here, and
  the surface-site fields need per-substep composites that the chained SURFACE (`atm_step.stage_surface`, `r1/r2`) computes but does not export in the dump layout used here.
- Bitwise/near-bitwise DIAGA needs the Intel libimf `pow` for SLP (`dyn_filter_ff.slp`); the test skips without the bridge. The 1e-12 tolerance is far looser than the observed 5e-14; the ledger's residuals are the measured ones.
- One window, one start state (nov26, 54 steps = 27 h, includes the 00 UTC day boundary of 27 Nov), one real member. Seasonal/other-state coverage (dec01, jan01) was not checked: only nov26 has 54-step dumps and packets.
- The real reference was produced with a scratch copy of the model (radiation-server build, no env switches); its first `YEARE` line replaced as the existing helper does (that line also held `KDIAG=12*0,9`, which the replacement drops, as it does for the existing dump runs; whether KDIAG affects the accumulation was not tested separately, but the accumulated fields agree with the dump-fed accumulators to 1e-14).
- S0 on non-radiation steps is recovered from the server increment, not recomputed from `S0X*S00WM2*RATLS0/RSDIST`; for a free-running month RADIA's own S0 would have to be exported or reconstructed.
- The 83-column RADIA coverage is "columns non-zero in this window"; columns that only become non-zero elsewhere (other seasons) pass through the generic all-column loop but are unvalidated.

## 5. What blocks a one-month F3 comparison (status of this entry's part and the rest)

1. The port cannot produce a month: the surface/ocean/ice/land/Ent loop is replayed, not closed (~45 h estimated in the plan), 31 daily updates are unexercised, and the free-radiation server costs ~11 s per call (~298 calls/month, about 55 min).
2. Diagnostics still missing from the minimal set: the SURFACE-site fields (usurf/vsurf/wsurf, pblht, tgrnd, gusti), the surface-state fields (sst, sivol, runoff, ...), the CONDSE columns (prec_mc, snowfall, clwp), sensht and the energy-budget columns, the zonal `ajl/aj` and `consrv` families. Each needs a real-window check like the one above (the 54-step real acc diff makes that cheap now).
3. The chained model must call `F3Acc` at the sites (DIAGA via the `dyn_step` stage hook at ~110 calls/month, SURFACE per substep).
4. The real noise floor: the 1-month ensemble (and the jan01 re-run reproducibility check) is not produced; only the single JAN1950 member exists.
5. Only nov26 has been exercised for these diagnostics; no dec01/jan01 acc windows were validated (dec01 and nov26 restarts do carry acc, so a real 5-day diff nov26 -> dec01 exists for 240 steps, but the port has dumps for 54 steps only).

**Parent-session check (2026-10-06 22:50):** `tests/test_f3_diagnostics.py` re-run independently: 31 passed, 0 skipped, 30 s, tolerance 1e-12 relative to each column's own maximum increment (unchanged). The reference (`ff_data/nov26_day/real_acc54_nov26.npz`, 33 MB, outside git) is the real binary's own accumulated diagnostics, independent of our module; the test requires >= 250 non-zero columns to match. Not re-derived by the parent: the real-binary 54-step run that produced the reference, and the monthly-mean/global-mean formula (transcribed by the agent, `ij_avg` weighting not read line by line).

# D164: surface loop (ocean / sea ice / lake / land ice / land state computed, not replayed), 2026-10-06/07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when ADVSI or RIVERF is ported, or when the Ent decision is made.
Sources: pristine ModelE (read-only) MODELE.f, ATM_DRV.f, SURFACE.f, SEAICE_DRV.f, ICEDYN_DRV.f, LAKES.f, LANDICE_DRV.f/LANDICE.f, IRRIGMOD.f, OCN_DRV.f, OCNDYN.f, OCNDYN2.f, OCN_Interp.f, GHY.f, Ent/*.f; ledger D1-D161.
New code: `fullfidelity/surface_loop.py`, `tests/test_surface_loop.py` (4 passed, 62 s). No existing file was modified. Nothing committed.

## 1. Scope: every replayed/recorded surface quantity of the day loop

Real step order (MODELE.f:316-339, ATM_DRV.f:257, SURFACE.f, OCN_DRV.f): MELT_SI -> CONDSE -> RADIA -> PRECIP_SI, PRECIP_OC(+TOC2SST) -> SURFACE [IRRIG_LK, PRECIP_LI, PRECIP_LK, 2 substeps, GROUND_LI, UNDERICE/GROUND_SI/GROUND_LK/RIVERF/FORM_SI (lakes)] -> ocean_driver [DYNSI, UNDERICE, GROUND_SI, CALC_APRESS, OCEANS, FORM_SI, ADVSI].

| Quantity (replayed in the day loop) | Real routine | Validated port before D164 | Status after D164 |
|---|---|---|---|
| Ocean state (G0M, S0M, MO, UO, VO, moments, straits) | OCEANS | ocean_step (D119-D138) | carried from our computation |
| AG2OG fluxes: oprec, oeprec, orsi, orunpsi, oerunpsi, osrunpsi, oe0, oevapor, osolarw, odmua, odmva, omelti, oemelti, osmelti | AG2OG_precip/AG2OG_oceans (identity regrid, same grid) + tile accumulations + MELT_SI | none (only the cores) | computed; bitwise equal to ffo (0.0) at step 0 |
| orunosi, oerunosi, osrunosi, osolari (GROUND_SI) | GROUND_SI | seaice_core_jax | computed; 1.4e-6 / 2.5e-2 / 1.1e-7 / 3e-14 abs (scales 2.2 / 6.9e5 / 6.6e-3 / 2.2e4): the known loosest D17/D20 diagnostics |
| oapress | CALC_APRESS (srfp = PEDN(1), bitwise) | apress_jax | computed |
| odmui, odmvi, UI2rho (UNDERICE ustar) | DYNSI | VPICEDYN core only; input assembly and post-processing NOT ported | RECORDED (ffo tag 1, ffz_undocn) |
| oflowo, oeflowo (river outflow) | RIVERF (LAKES.f:1708-2216, 508 lines, original version; RVR_ELEV undefined; needs the river-direction file) | none | RECORDED (ffo tag 1) |
| init_STRAITS | init_STRAITS | straits step | state (MUST, G0MST ...) from the restart; MMST (static strait mass) RECORDED from ffo tag 0 (constant over steps) |
| OPFIL2 coefficients, ODIFF | | D137, D138 | computed (no recorded read) |
| Atmosphere SST export GTEMP, GTEMP2, SSS, MLHC | TOC2SST (OCNDYN.f:5577) | none | NEW (TEMGS from the OFTAB table), bitwise equal to the restart exports |
| Sea-ice state (RSI, SNOWI, MSI, HSI, SSI, flag_dsws, pond_melt) | MELT_SI, PRECIP_SI, GROUND_SI, FORM_SI, seaice_to_atmgrid | cores D10-D14, D26, D31, D32 | drivers wired; computed. ADVSI is NOT ported (ICEDYN_DRV.f:880-1636, about 756 lines with GOTO flux logic, RSIX/RSIY moments, EXPEL_COASTAL_ICEXS `connect` array): the one-step ice error is its signature |
| Sea-ice/ocean tile ground state (ffs/ffp ground columns, fft ftype) | SURFACE.f:430-700, seaice_to_atmgrid | tile fluxes D5-D6 | built from our state (`apply_state_to_records`); identity on the real step-0 records to rounding |
| Lake state (MWL, GML, TLAKE, MLDLK) | PRECIP_LK, GROUND_LK | D13/D16/D27 | computed (land runoff from our GHY in the coupled run) |
| IRRIG_LK withdrawal | IRRIG_LK + IRRIGMOD.irrigate_extract (460 lines, irrigation-demand data file) | none | RECONSTRUCTED from the recorded actual irrigation flux (ffg `irrig` x FEARTH): mass = min(irrig*rho*A*dt, available), equal to the Fortran in the full/partial/none branches; the demand file is not read; the recorded GHY forcing is the boundary. Bitwise match of all 614 lake tiles at step 0 |
| Land-ice state (SNOWLI, TLANDI) | PRECIP_LI, GROUND_LI (LNDICE) | PRECIP_LI D28; tile D7 | GROUND_LI/LNDICE ported here (NEW, 80 lines, small); bitwise at step 0 |
| Land GHY state (w, ht, snow layers) and runoff | GHY | ghy_jax, land_chain (D9, D22, D25, D135, D136, D158) | carried from our GHY across steps in the coupled run; GHY precipitation forcing = our CONDSE PREC/EPREC/PRECSS (identity pr=PREC/(dtsrc*rhow) checked, 0.0) |
| Ent exports (cnc, betadl, lai, TRANS_SW, Ci, GPP, IPP, dts, ws_can, shc_can, fv, height, albedo) | Ent (see 3) | none | RECORDED |
| Radiation (SRHR/TRHR/COSZ1, tile SRHEAT, TRHR0, ALB) | SOCRATES | never ported | RECORDED (frozen as in D150) |
| TRUP_in_rad, PBL profile columns, Ca/COSZ/vis_rad in ffg | | | RECORDED |
| GLMELT, daily_LAKE, daily_LI, daily_OCEAN, daily_SEAICE | day boundary | none | not exercised (window < day boundary) |

## 2. Implementation (surface_loop.py) and results (nov26, from the real restart `fort1_nov26_itime33312.nc`)

Wiring: surface state is loaded from the restart (checked: ocean equal to ffo tag 0 bitwise; TOC2SST exports equal to the restart's asst/sss/mlhc/ogeoza bitwise). `surface_pre` = MELT_SI, PRECIP_SI, AG2OG_precip, PRECIP_OC, TOC2SST, IRRIG_LK, PRECIP_LI, PRECIP_LK, seaice_to_atmgrid. `surface_post` = GROUND_LI, lake chain, ocean_driver (ocean_step without its PRECIP stage, ported ODIFF, computed OPFIL2). `apply_state_to_records` + `stage_surface_closed` (copy of atm_step.stage_surface with three stated differences) + `Loop`/`run_coupled` couple it to the atmosphere chain.

Measured, step 0 (from the restart, replay mode R1 = real tile outputs as flux input):
- MELT_SI reproduces the real CONDSE-entry RSI bitwise (0 of 3312 cells differ, both domains; 827 ffm records equal). PRECIP_SI vs ffw: exact except HSIL 1.2e-6 abs on 8.5e8. AG2OG_precip + PRECIP_OC vs ffo tag 1: 0.0 (G0M, S0M, MO).
- State columns of the real substep-1 tile records (ffs ocean 2095, lake 614, ice 783; ffl 346): ocean/lake/landice/ice snow, msi2, ssi, flag exactly 0.0; ice tg1/tg2 2.5e-14 (scale 46); tr4 <= 3.8e-6 on 5.6e9; tile sets identical.
- Fluxes into the ocean vs ffo tag 1: 0.0 for 14 of 18 fields (see table); ocean exit state vs tag 14: g0m 1.2e-12, s0m 2.5e-13, mo 2.2e-13, uo 1.1e-9, vo 6.1e-9, opress 2.4e-10, ogeoz 2.0e-10 (same level as D120 with recorded fluxes).
- Records built from our state versus the real records: identity to rounding (<= 6e-14 on 300 K).

Measured, free run 6 steps (R1, no ADVSI, RIVERF recorded; relative to field scale): step 1 ice snow 9.7e-4, msi2 9.5e-4, ptype 1.1e-3, tg 1.9e-3; ocean exit uo 2.3e-3, vo 5.4e-3 (g0m 1.4e-5); step 5 ice snow 4.8e-3, ocean uo 1.1e-2, vo 2.3e-2; tile-set mismatch 14 ocean tiles (all RSI near 1) from step 1; lake tg1 2e-3 -> 9.6e-3, lake MWL 1.4e-6 -> 7.2e-6. Land-ice error 0 for steps 1-4, 0.16 of scale at tg1 from step 4 (not investigated). The ice error appears at the first step and matches the size of an advective change (ice speed x 1800 s / grid) but is NOT shown to be caused by ADVSI by any isolation test other than the next item; the lake attribution to RIVERF is an inference (lakes are bitwise at step 0).
Isolation test (step 33313 with the ice state taken from the real ffm_33313 inputs = real post-ADVSI, ocean from ffo): ice columns at rounding level (tg 5e-14, msi2 1.4e-13), tile sets identical, ocean exit g0m 3.9e-12, uo 4.6e-9, vo 3.0e-8. So with the real ice entry the rest of the chain is accurate at the D120 level; only lakes remain off (tg1 0.063 K in one cell, MWL 1.4e-6).

Coupled run (atmosphere chain + closed surface, 6 steps, `run_coupled`, 3 cores, 22-45 s per step after a 155-195 s first step): atmosphere rms difference from the real run: T 7.4e-14 / 2.0e-4 / 5.2e-4 / 1.1e-3 / 1.3e-3 / 1.7e-3 K at steps 0-5 (open loop D150: 7.2e-14 ... 1.3e-3 at step 5). Against the 5 real one-ulp members (rms metric, steps 0-5): T within 4, near 2 (max ratio 1.44 at step 3); Q within 3, near 3 (1.07); U within 6 (0.72); V within 3, near 3 (1.30); QCL within 5, near 1; QCI within 6; P within 4, near 2 (1.45). Never beyond 2x. The open loop on the same steps: all within except Q near at 3 steps and P near at 1. The members' own spread at step 3 is 4.5e-5 .. 7.9e-4 K, so the coupled-vs-open-loop difference (1.1e-3 vs 6.8e-4) is not shown to come from the surface. Necessary, not sufficient evidence (5 members, one start, 6 steps). The surface state of the coupled run was not compared separately from the R1 numbers above. The carried land (GHY) state of the coupled run was not compared with the real record in this session (D25 validated the carry for 4 substeps).

Limit of the window: 6 steps, because the DYNSI boundary (ffy, ffz_undocn) exists for 6 steps and the RIVERF boundary (ffo tag 1) for 12; the 54-step day has neither (D149 did not build the ocean hooks). Longer runs need DYNSI ported and RIVERF recorded or ported.

## 3. Ent: size and the decision for the owner (not decided here)

What Ent exports to GHY (GHY.f:2338-2520, per sub-iteration and per call): cnc (canopy conductance), betadl(6), TRANS_SW, Ci, GPP, lai, IPP, dts; per call ws_can, shc_can, fv, canopy height, albedo(6). Ent also runs per sub-iteration (`ent_run` -> `ent_integrate` -> photosynth_cond, soil_bgc, summaries) with inputs Qf, pressure, CO2, ch, wind, vis_rad, direct_vis_rad, cosz, wet fraction, soil T/moisture/matric potential/ice fraction, and daily `update_vegetation_data` (prescribed LAI/height from data files; `do_phenology_activegrowth` default 0, `do_soilresp` 1; the rundeck does not override them). State: `ent_state` 1023 doubles per cell in the restart.
Size (my measurement, a name-matching call-graph over `model/Ent/*.f` + ENT_DRV.f; an over-approximation): per-iteration path about 4,260 code lines, of which 1,460 are in canopyradiation.f/canopygort.f, which the executable's symbol table (`nm`) does not contain, so about 2,800 lines (biophysics.f 594, FBBphotosynthesis.f 546, canopyspitters.f 501, patches.f 327, soilbgc.f 257, entcells.f 236, respauto_physio.f 145, cohorts/allometry smaller); daily prescribed update about 1,080 lines (ent_prescribed_drv.f, ENT_DRV.f, allometryfn.f, ent_prescr_veg.f) plus the LAI/height/vegetation data files. Whole Ent directory 26,500 lines (mostly unused variants). Excluded from the count: derived-type plumbing in ent_mod.f (4,122 lines) beyond the routines reached.
Assessment: a recorded-Ent boundary is reasonable and documentable for windows up to one day (exports from the real run at 1,500 values per cell-step; the D158 record already holds the iterations), but it cannot represent soil-moisture-dependent conductance feedback and a month-long run (F3) would need a month of recorded exports (not available), so F3 needs either a port of the per-iteration path (about 3 k lines, with validation against the ffg `ffent` records, which already exist per iteration) or an explicit surrogate. DECISION FOR THE OWNER: (a) keep Ent recorded for all runs up to one day, or (b) start the Ent port now. I recommend (a) for now and (b) before F3; this is a recommendation, not a decision taken.

## 4. Remaining recorded inputs and why

Radiation (never ported); Ent exports and the ffg land forcing columns (section 3); TRUP_in_rad, PBL profile columns, SRHEAT/TRHR0 tile columns; DYNSI result (odmui, odmvi, UI2rho/ustar): DYNSI input assembly (ICEDYN_DRV.f:328-877, about 550 lines, glue only; VPICEDYN already validated) is the smallest next piece; RIVERF outflow (508 lines + river-direction data); MMST; IRRIG demand (reconstructed from the recorded actual flux); ADVSI (largest gap: no dump brackets it except ffn_<it> (pre) -> ffm_<it+1> (post) for 33312 and 33313, so it could be validated for one or two steps only); tile templates (the real ffs/ffp/ffg/fft rows supply every static, atmospheric and radiative column and fix which tiles exist: tiles absent from the template are dropped, count reported per step, ptype 0 in the 6-step run).

## 5. Failures and cautions
- First pass of the lake tiles at step 0 differed (MWL 5e8 on 6e16) until IRRIG_LK was reconstructed; fixed, then exact.
- Free-running beyond step 0 is not accurate without ADVSI (1e-3 per step in the ice, growing); this is reported, not hidden.
- `ag2og_precip` weight-zero branch (RSI = 1) is unverified: no cell in the dumps has RSI = 1.
- Unmeasured: land-ice tg1 jump (0.16 of scale from step 4 in R1), the coupled-run surface state against the records, any date other than nov26.
- CPU: the first exploratory runs were not pinned; later runs used `taskset -c 0-2`.

**Parent-session check (2026-10-06 23:24):** `tests/test_surface_loop.py` re-run independently: 4 passed, 66 s (assertions: exact equality at step 0, ice tg < 1e-13, ocean exit g0m < 1e-10, uo/vo < 1e-7). No existing file was modified (`git status` shows only new files). NOT re-derived by the parent: the free-run and coupled-run comparison numbers (6 steps, nov26 only, 5 real members), the Ent call-graph size, and the attributions to ADVSI and RIVERF (the agent marks both as inference). **Open decision for the project owner (G. Tamkin): Ent vegetation exports - keep recorded for runs up to one day, or start the Ent port now; the agent recommends recorded now and a port before F3, because a month-long F3 run needs a month of Ent exports that does not exist.**

## D162 (2026-10-07): free-radiation day through the PERSISTENT radiation server; equals the one-shot day bitwise

Owner: Glenn Tamkin. Sources: D155-D157 (`atm_day_free_rad.py`), D159-D161 (`radiation_server_persist.py`). SOCRATES/RADIA untouched (the real RADIA runs inside the persistent server).

**Built.** `fullfidelity/atm_day_free_rad_persist.py` (new; `atm_day_free_rad.py` unchanged): `PersistentRadiation`, a callable plugged into the existing `run_day_free(server=...)` hook; it lazily starts one `PersistentServer("nov26")` (start-up once) and requests each radiation step in order with the real seed of that step (`ffc_cse_out_<it>.bin` SEEDS[1], nov26_day). `run_day_persist` also stores `persist` (start wall, seeds, per-call server/wall seconds) in run.json. `compare(tag, ref)` compares the two days bitwise (step_<it>.npz, rad_in, rad_out). Test: `fullfidelity/tests/test_atm_day_free_rad_persist.py` (2 tests; skipped when server binary/restart/dumps are absent; the day-equality test also skips without both saved days).

**Commands.**
```
cd fullfidelity
export RADSRVP_SCRATCH=<session scratchpad>/mE_persist OMP_NUM_THREADS=3   # binary mE2/model/P2SAoM40.bin
taskset -c 0-2 python atm_day_free_rad_persist.py run --tag free_persist     # 54 steps, 11 server calls, one server
python atm_day_free_rad_persist.py compare --tag free_persist --ref free_np   # vs the D155 one-shot day (ours_free_np)
python -m pytest tests/test_atm_day_free_rad_persist.py tests/test_atm_day_free_rad.py   # 8 passed, 97 s
```
Outputs: `ff_data/nov26_day/ours_free_persist/`.

**Results (measured).**
- Bitwise comparison against `ours_free_np`: state `step_<it>.npz` (T Q U V QCL QCI P; 54 steps, 378 arrays): 0 unequal; assembled radiation packets `rad_in` (11 steps, 572 arrays): 0 unequal; server outputs `rad_out` (11 steps, 242 arrays incl. TOA AIJ columns): 0 unequal. Per-step statistics in the run log equal D155 (e.g. step 0 rms T 7.21e-14; step 53 rms T 7.95e-02, U 2.05e-01, V 2.21e-01). Per call, `radia_apply` T equals the server's own T update (0.0 on all 11 calls).
- Radiation wall time: persistent 158.3 s in total for 11 calls (start-up 19.9 s inside the first call; per call 11.1-13.6 s server time, 11.7-14.6 s wall after the first; first call 31.7 s) vs one-shot 1193.9 s (32.6 ... 191.8 s per call, growing with the step). The one-shot total is 1194 s, not ~30 min: D155 recorded 1194 s of server time in a 2250 s day. Day total 1312 s (persistent) vs 2250 s (one-shot); the chain itself (non-radiation) took ~1153 s here vs ~1056 s there (shared node, 3 cores).
- Server time is a single-thread number on a shared node.

**Limitations.** One date (nov26), one boundary (33360); the first attempt of the run was killed by my own `pkill` during polling at step 15 and was simply restarted from scratch (no partial results used). The comparison only proves persistent == one-shot for this trajectory; both share the replayed surface and stale carried SNOAGE caveats of D155-D157 (unchanged). Per-call cost is dominated by RADIA itself (~11 s), not by file I/O.

**Parent-session check (2026-10-06 23:35):** own array-by-array comparison of `ff_data/nov26_day/ours_free_persist` vs `ours_free_np` (76 npz files, 1,192 arrays = 378 state + 572 rad_in + 242 rad_out): 0 not bitwise equal; the only key present in one side only is `_server_s` (timing metadata, excluded). Note: the .npz files differ at BYTE level (zip timestamps and that extra key), so a byte/hash comparison is the wrong test. `tests/test_atm_day_free_rad_persist.py` re-run: 2 passed, 62 s. Timings (radiation 158 s vs 1,194 s; day 1,312 s vs 2,250 s) are the agent's single-thread measurements on a shared node, not re-measured. The agent reports its first launch was killed by its own `pkill`; the ensemble processes of the other track were confirmed still running afterwards (7 `P2SAoM40`).

# D165: real JAN1950 one-month reference ensemble (8 members) and reproducibility of the stored JAN1950 acc

Date 2026-10-06/07. Owner: Glenn Tamkin. Not committed. Sources: real model binary of the D149-D151 build (`<scratch>/ac69365f.../mE_day1/mE2/model/P2SAoM40.bin`:
pristine physics + the D128/D125/... dump hooks + `ATM_DRV_pert.f.patch`; with `FFD_*` unset the hooks write nothing; D149 showed `ffpt_ctrl == ffa_step_e` bitwise,
and the unperturbed member below reproduces the stored real acc bitwise, which is the stronger evidence that the build is a physics no-op); restart
`ff_data/_pristine_restarts/fort1_jan01_itime17520.nc`; stored reference `ModelE_Support/prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc`. SOCRATES untouched; real source tree not written.
Review: when the F3 comparison is run.

## 1. Commands
Script `<scratch>/ens_jan/one.sh <name> "<pert>"` (copy of the D149 run recipe with the January window); launcher `runall.sh` (`xargs -P5`, 5 concurrent single-thread runs, 8 jobs).
Per run: run dir with `I P2SAoM40ln P2SAoM40uln runtime_opts` from `huge_space/P2SAoM40`, binary copied to `P2SAoM40.bin` and `P2SAoM40`; line 109 of `I` edited to
`YEARE=1950,MONTHE=2,DATEE=1,HOURE=0,` (only edit); jan01 restart copied to BOTH `fort.1.nc` and `fort.2.nc`; `sh P2SAoM40ln`;
`OMP_NUM_THREADS=1 FFPT_START=17520 FFPT_NSTEP=0 FFPT_TAG=<name> FFPT_PERT='<spec>' ./P2SAoM40 -i I` (FFPT_NSTEP=0: no dump files; FFPT_PERT unset for ctrl; FFD_* unset).
Outputs copied (`\cp`) to `ff_data/ens_jan1950/<member>/{fort.2.nc, JAN1950.accP2SAoM40.nc, run.time, pert.txt}` (2.3 GB, outside git).
Analysis: `fullfidelity/ens_jan1950_summary.py` (`repro`, `summary`); products `ff_data/ens_jan1950/{repro_ctrl.json, ens_jan1950_summary.json, ens_jan1950_summary.npz}`.

## 2. Members (all rc=0, 1488 steps, idacc[0]=1488, one core each, 5 concurrent; wall 82-87 min = 3.3-3.5 s/step)
The perturbation method is the D151 one (`ffpt_perturb`: +-1 ulp via `nearest()` of T or Q in the model state at atm_phase1 entry of the first step), same cells/signs as the 5 D151 members, applied at itime 17520:
| member | spec (FFPT_PERT) | wall |
|---|---|---|
| ctrl | none | 85m42s |
| p1 | T 36 23 10 +1 | 85m03s |
| p2 | T 20 12 20 -1 | 84m29s |
| p3 | Q 50 30 5 +1 | 83m13s |
| p4 | T 10 30 1 +1 | 86m46s |
| p5 | T whole field, +-1 ulp checkerboard (`T 0 0 0 1`) | 82m18s |
| p6 | T 60 40 15 -1 (new cell, same method; not a D151 member) | 85m07s |
| p7 | T 5 20 8 +1 (new cell, same method; not a D151 member) | 83m29s |
Month ends at itime 19008 (1 Feb 1950 00:00); the model writes `JAN1950.accP2SAoM40.nc` itself.

## 3. Reproducibility of the stored JAN1950 acc (unperturbed run, jan01 restart)
Result: **bitwise identical.** Every array of the model-written `ctrl/JAN1950.accP2SAoM40.nc` equals the stored file with 0 unequal elements (np.array_equal on all variables):
aij (5,497,920), aijl, aijk, aj, ajl, consrv, agc, areg, tdiurn, asjl, aisccp, adiurn, energy, ijhc, oij, oijl, icij, idacc (1488), and all name/scale/index metadata.
Only two scalars differ: `cputime` (wall-clock bookkeeping) and `itimee` (the stored run was configured to end at 1 Dec 1950, ours at 1 Feb 1950; run-end setting, not physics).
The jan01 restart carries no acc block, so the whole month is accumulated from zero in both; that is why the full-month acc is reproduced. The end-of-run `fort.2.nc` acc (float64, unrounded)
agrees with the stored float32 file to float32 rounding for aij/aijl/aijk/ajl/consrv etc. (0 unequal after casting to float32; max 3e-8 of scale before casting); agc/areg/oijl differ and `aj` has a different shape in fort.2 (6 vs 9 rows); not investigated (the model-written monthly acc file, which is what F3 uses, is bitwise).
Hence: the real model is deterministic over a month with this binary, thread count (OMP_NUM_THREADS=1; the thread count of the original production run is not known to me) and node; and the stored JAN1950 file is the unperturbed member.

## 4. Ensemble spread (noise floor), monthly means of AIJ columns (ij_mapk formula via `f3_diagnostics.field_from_aij`; 8 members including ctrl, ddof=1)
Definitions: gm = area-weighted global mean per member (area = axyp; cells where the ratio denominator is zero, i.e. below-ground pressure levels, excluded: t_850/q_850 use the 2962 defined cells);
gm sd = std over members of gm; grid sd rms = area-weighted rms over cells of the member std at each cell (pointwise noise floor); zonal sd rms = rms over the 46 latitudes of the member std of the zonal mean;
spatial std = spatial std of the ensemble-mean field (for scale). Units as in the acc/DIAG_PRT scaling (as stored, not re-converted: e.g. prsurf, slp in mb, prec/evap mm/day, temperatures degC, fluxes W/m2).
| field | gm mean | gm sd | gm min..max | grid sd rms | zonal sd rms | spatial std |
|---|---|---|---|---|---|---|
| prsurf | 984 | 2.63e-07 | 984 .. 984 | 1.21 | 0.98 | 64.8 |
| slp | 11.042 | 0.01 | 11.031 .. 11.065 | 1.25 | 1.01 | 8.15 |
| t_850 | 6.0601 | 0.0178 | 6.0292 .. 6.084 | 0.681 | 0.533 | 12.8 |
| t_500 | -18.549 | 0.0427 | -18.602 .. -18.48 | 0.556 | 0.303 | 12.1 |
| t_200 | -53.786 | 0.031 | -53.812 .. -53.724 | 0.439 | 0.276 | 4.38 |
| z_500 | 5583 | 0.54 | 5582.1 .. 5583.8 | 15.2 | 9.09 | 264 |
| u_200 | 17.085 | 0.0927 | 16.965 .. 17.211 | 1.79 | 0.712 | 13.9 |
| q_850 | 6.2401 | 0.0117 | 6.2197 .. 6.2557 | 0.265 | 0.0574 | 3.92 |
| omega_500 | -8.9433e-06 | 3.19e-05 | -5.9892e-05 .. 3.787e-05 | 0.014 | 0.00244 | 0.059 |
| qatm | 24.457 | 0.0457 | 24.373 .. 24.502 | 1.06 | 0.227 | 16.4 |
| prec | 2.8255 | 0.0105 | 2.809 .. 2.8438 | 0.885 | 0.124 | 3.53 |
| evap | 2.8457 | 0.0135 | 2.8259 .. 2.8606 | 0.251 | 0.0493 | 2.25 |
| tsurf | 11.859 | 0.0216 | 11.829 .. 11.886 | 0.719 | 0.701 | 16.9 |
| srnf_toa | 242.79 | 0.179 | 242.5 .. 243.14 | 7.14 | 1.4 | 120 |
| trnf_toa | -233.88 | 0.132 | -234.1 .. -233.66 | 4.81 | 1.25 | 33.8 |
| incsw_toa | 351.34 | 0 | 351.34 .. 351.34 | 4.72e-14 | 5.4e-14 | 156 |
| srnf_grnd | 171.47 | 0.2 | 171.18 .. 171.88 | 7.76 | 1.6 | 89.8 |
| trdn_surf | 332.16 | 0.137 | 331.98 .. 332.35 | 4.65 | 3.67 | 77.1 |
| tauus | -2.5803 | 0.98 | -3.8288 .. -1.211 | 23.1 | 8.58 | 115 |
| tauvs | -11.3 | 0.665 | -12.381 .. -10.145 | 20 | 5 | 83.1 |
| rh_layer1 | 78.298 | 0.0496 | 78.216 .. 78.381 | 1.72 | 0.715 | 15 |
All 177 computed fields (all pressure-level t,z,u,v,q,rh,omega,p_freq plus the above) are in `ens_jan1950_summary.json`; per-member monthly-mean maps in the npz (`monthly_mean[member, field, 46, 72]`, `stored`).
Pointwise members-vs-ctrl rms (examples): t_500 0.88-1.02 K, prec 1.00-1.24 mm/day, srnf_toa 8.9-10.4 W/m2, slp 2.0-3.1 mb. The ensemble-mean global means differ per member by e.g. 0.04 K (t_500), 0.18 W/m2 (srnf_toa).
Reading: a one-ulp perturbation produces a pointwise monthly-mean spread that is about 5 percent of the spatial std for t_500 and srnf_toa, about 25 percent for prec and tauus (not 'decorrelated': no correlation was computed), while global means are stable to a few 1e-3 of their spatial std.
`incsw_toa` has zero spread (prescribed solar input), a useful sanity check.

## 5. Limitations
- 8 members (7 perturbed + ctrl) give a noisy sd estimate (about +-25 percent on a std from n=8); p1-p5 repeat the D151 positions, p6-p7 are new cells; all perturbations are at the very first step, one realisation of a month, one season (January, from a spin-up state: the stored run started cold on 1 Dec 1949).
- The ensemble gives the floor for a monthly mean of the model that starts from the jan01 restart; it says nothing about what a port that differs by more than rounding (e.g. a different radiation/land path) would produce; the F3 acceptance thresholds remain proposals.
- Only AIJ-derived fields (the ones `f3_diagnostics.PORTED_NAMES` lists) are summarised; AIJL zonal profiles, aj/consrv are in the saved acc files but not summarised.
- The below-ground pressure-level fields are averaged over the defined cells only.
- Wall time/core-count: 5 concurrent single-thread runs on a 12-core node; run speed 3.3-3.5 s/step. The coordinator's later 4-core limit was not violated by this work (all runs had already finished).
- The D149 pitfall (a T ulp is not local after one step) applies: members are effectively independent weather realisations from day 1.

**Parent-session check (2026-10-07 05:20):** (1) reproducibility re-done with my own script on the stored files: ctrl `JAN1950.accP2SAoM40.nc` vs `prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc`: 143 variables, 14,078,759 values, only `itimee` (end-time setting, differs by construction) and `cputime` differ; every physical array is equal. (2) spread table recomputed from `ens_jan1950_summary.npz` (8 members, area weight ~cos(lat), my own pole weighting): t_500 global-mean sd 0.0427 K / grid rms 0.556 K, tsurf 0.0215 / 0.721, slp 0.0108 / 1.256, prec 0.0105 / 0.885, evap 0.0135 / 0.251, srnf_toa 0.179 / 7.13, trnf_toa 0.132 / 4.81: agrees with the table above to rounding. Not re-checked by the parent: the 8 model runs themselves (logs show rc=0, 85 min each), the perturbation specs, and `ens_jan1950_summary.py` has no test. Eight members give a noisy spread estimate (about +-25% on a std).

## Pending rows
- S0ML0(1) inside the OCONV iteration: the glue takes it as an input (S0M(I,J,1), not yet dumped).
  BYMML(1) is now supplied by `oconv_mml_ff.mass_bookkeeping` (D61).
- GPU speed numbers for the JAX-vectorized pieces (no GPU available on the node used for D14/D15/D16's
  CPU-only measurements).
- The actual `jax.lax.scan`-chained whole-model Track B step (wiring ATURB/PBL/SURFACE/SEAICE/LAKES/
  GHY together the way Track A's `run_steps_device` chains one full atmosphere step) -- every needed
  piece is now validated and jit-able (D17 closes the last missing one, `ground_si`), but the actual
  driver assembly, land-ice tile-flux call site, and PBL↔SURFACE data-flow tracing are still pending
  (see FULL_FIDELITY_PLAN.md's chained-driver section and STATUS.md's 8-stage table).
