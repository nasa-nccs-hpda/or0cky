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

## D17 — `ground_si` wrapper + a real float32-precision bug found and fixed across 3 core modules
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

## Pending rows
- GPU speed numbers for the JAX-vectorized pieces (no GPU available on the node used for D14/D15/D16's
  CPU-only measurements).
- The actual `jax.lax.scan`-chained whole-model Track B step (wiring ATURB/PBL/SURFACE/SEAICE/LAKES/
  GHY together the way Track A's `run_steps_device` chains one full atmosphere step) -- every needed
  piece is now validated and jit-able (D17 closes the last missing one, `ground_si`), but the actual
  driver assembly, land-ice tile-flux call site, and PBL↔SURFACE data-flow tracing are still pending
  (see FULL_FIDELITY_PLAN.md's chained-driver section and STATUS.md's 8-stage table).
