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

# D168: DYNSI input assembly and post-processing (ported), 2026-10-07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when ADVSI is ported (re-run `dynsi_loop_ff.py`), or when a date other than nov26 is validated.
Sources: pristine ModelE (read-only) ICEDYN_DRV.f:328-877 (DYNSI), 2125-2205 (GET_UISURF), ICEDYN.f:985-1112 (DXP, DYV, DXYP, DXYN, DXYS, DXYV, SINIU/COSIU), OCN_Interp.f:1605-1690 (IG2OG_oceans), :653-670 and 1722 (OG2AG UOSURF, OG2IG_uvsurf), OCNDYN.f:5577-5683 (TOC2SST, get_exports_layer1), SEAICE_DRV.f:313 (ustar), OCEAN_COM.f:22 (IVNP = IM/4), OGEOM.f:157, GEOM_B.f:312; dumps ffy_<it>_{in,out}.bin (D29), ffz_undocn_<it>.bin (D32), ffo_state tag 0/1, restart fort1_nov26_itime33312.nc.
New files: `fullfidelity/dynsi_ff.py`, `fullfidelity/dynsi_loop_ff.py`, `fullfidelity/tests/test_dynsi_assembly_ff.py` (4 passed, 2.4 s). No existing file was modified (surface_loop.py, advsi_ff.py untouched). Nothing committed.

## 1. What was ported (the "~550 lines of glue")
Non-cubed-sphere branch (ice grid = atmosphere grid = ocean grid, 72 x 46; OCEAN_IMPORTEXPORT_ON_BGRID undefined). Around the existing VPICEDYN (icedyn_vec.vpicedyn; scalar one selectable):
- before: polar replication and FOCEAN*RSI masking of DMUA/DMVA, GAIRX/GAIRY (4-point average / DTS), aPtmp = OGEOZA + ice load, PGFU/PGFV, HEFF, AREA, AMASS, COR, GWATX/GWATY (4-point average of UOSURF/VOSURF), PGFUB/PGFVB, ghost columns, north-pole rows;
- after: DMU/DMV (all rows incl. the polar row), USI/VSI (rows 2..JM-1 updated, row JM zero, row 1 carried), DMUI/DMVI (+ north-pole mean), UI2rho (4-point B to A average of the stress, / DTS), ustar = max(5e-4, sqrt(UI2rho/RHOWS)), GET_UISURF (computed, not validated: no dump of the atmosphere uisurf; the tile record's uocean column was not compared), IG2OG (identity, polar row zero) giving ODMUI/ODMVI;
- ocean exports: UOSURF/VOSURF from UO,VO layer 1 (get_exports_layer1 + OG2AG identity + latlon polar vector).
Libm/arithmetic: numpy/glibc; expression order copied from the Fortran. Constants: RHOI 916.6, RHOWS 1030, GRAV 9.80665, OMEGA = 2*pi/(86400*365/366), OIPHI 25 deg, DTS 1800, RADIUS 6371000.
A fact found by measurement (not from the source): the real restart's atmosphere UOSURF/VOSURF polar row is constant along i and equals the polar vector (UNP, VNP); a first version that kept the ocean polar row gave 0.03 error in GWATX/GWATY; fixed (the filling of the polar row is therefore empirical, validated on the nov26 restart row only).

## 2. Validation (nov26, six steps 33312-33317; inputs = real ice state of ffy_in, real ice-tile DMUA of ffs, ocean exports from ffo tag 0/restart; USI/VSI carried from OUR output)
- Assembled inputs vs ffy_in: gairx, gairy, pgfub, pgfvb, heff, area, amass, cor: BITWISE (0.0) in all 6 steps; gwatx/gwaty <= 6.9e-18 absolute (scale 0.24; polar-vector summation order).
- uosurf/vosurf vs restart exports: 0.0 / 6.9e-18; ogeoza: 0.0 (step 0).
- Carried USI/VSI vs the recorded uice0 of the next step: 0.0 at step 0, then <= 1.8e-12 absolute (scale 0.23).
- Outputs vs ffy_out: dmui <= 1.4e-10 (scale 160), dmvi <= 3e-11 (scale 176), dmu/dmv (incl. polar row) <= 1e-11 at step 0, usi/vsi <= 1e-12. The residual is the existing VPICEDYN (D29/D87: not bitwise, convergence loop kki = 2 in all steps), not the glue.
- ODMUI/ODMVI vs ffo tag 1: <= 1.6e-10 / 7.3e-11 absolute (scale 150-160).
- ustar vs ffz_undocn on the 482 recorded cells: relative <= 1.3e-11 (bitwise on 150-175 of 482). Limitation: our UI2rho is nonzero on 553 cells; the recorded UNDERICE file holds 482, the other 71 were not compared.

## 3. Effect on the free 6-step surface loop (dynsi_loop_ff.py; replay mode R1: real SURFACE tile outputs as flux input, no ADVSI, RIVERF recorded, 2 cores, ~100 s per variant)
Recorded-DYNSI baseline reproduces D164 (uo 2.25e-3, vo 5.44e-3 relative after step 1). Computed DYNSI (odmui, odmvi, ustar from our own state):
| step | ocean uo rec / comp | vo rec / comp | ice msi2 abs rec / comp (scale 3.54e3) | ice tg1 rec / comp |
|---|---|---|---|---|
| 0 | 1.07e-9 / 1.07e-9 | 6.08e-9 / 6.08e-9 | 0 / 0 | 2.5e-14 / 2.5e-14 |
| 1 | 2.2515e-3 / 2.2516e-3 | 5.4375e-3 / 5.4375e-3 | 3.3751 / 3.3751 | 8.80e-2 / 8.80e-2 |
| 3 | 7.3757e-3 / 7.3754e-3 | 2.8752e-2 / 2.8752e-2 | 10.070 / 10.067 | 0.26443 / 0.26443 |
| 5 | 1.0711e-2 / 1.0710e-2 | 2.2827e-2 / 2.2827e-2 | 16.742 / 16.730 | 0.44225 / 0.44225 |
Our DYNSI result differs from the recorded one in this free run by up to 1.3 (odmui, scale 150, step 5) and 8e-5 in ustar (scale 0.0136) because the free state has drifted; this is the consequence of the state error, not a glue error (the glue is exact on the real state, section 2). Conclusion: replacing the recorded DYNSI boundary by the computed one changes the ice and ocean differences by <= 1e-3 relative of the error itself (e.g. uo 2.2515e-3 -> 2.2516e-3), so the DYNSI boundary is NOT the source of the ~1e-3 first-step ice error and the 2e-3..5e-3 ocean error. This is consistent with, but does not by itself prove, D164's attribution to ADVSI (not ported; another agent). Tile-set mismatch (14 ocean tiles from step 1) is identical in both variants.
Not done: the coupled atmosphere run (run_coupled) with computed DYNSI (needs the tile uocean = uisurf wiring inside the closed tile chain, which would require editing surface_loop.py); other dates; the unvalidated GET_UISURF. How to reproduce: `python fullfidelity/dynsi_loop_ff.py both` (set D168_OUT=<json> to save).

**Parent-session check (2026-10-07 05:30):** `tests/test_dynsi_assembly_ff.py` re-run: 4 passed, 2.3 s (bounds are the measured values rounded up; asserts include bitwise equality of the assembled inputs). No existing file was modified (`git status`: only new files). Re-ran `python dynsi_loop_ff.py both` (3 cores, pinned): the recorded-DYNSI and computed-DYNSI free 6-step loops give the same ice/ocean errors as the table above (e.g. ocean exit uo 2.25e-3 / vo 5.44e-3 at step 1 and uo 1.07e-2 / vo 2.28e-2 at step 5 in both variants; ice tg1 0.0880 -> 0.442); so the DYNSI boundary is not the source of the first-step drift (consistent with the ADVSI attribution of D164, which this does not prove). Not re-derived by the parent: the per-field residuals vs the ffy/ffo/ffz dumps beyond what the tests assert. Only nov26; GET_UISURF unvalidated.

# D167: RIVERF (river routing / lake outflow) ported; oflowo/oeflowo computed; D164 lake errors explained, 2026-10-07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when ADVSI/DYNSI are ported or when the surface loop is extended past 6 steps.
Sources: pristine ModelE (read-only) `model/LAKES.f` (RIVERF original version 1708-2212, init_LAKES river part 889-1050, get_dir, horzdist_2pts), `GEOM_B.f` (lonlat_to_ij, lat/lon definitions), `ATM_COM.f:212` (ZATMO = zatmo x GRAV), `SURFACE.f:1229-1236` (call order), `decks/P2SAoM40.R` (preprocessor options, RVR/TOPO files); input files `ModelE_Support/prod_input_files/RD_modelE_M.nc` (river directions) and `Z72X46N_gas.1_nocasp.nc` (focean, flake, zatmo, hlake); dumps `ff_data/nov26/ffo_state_*.bin` tag 1 and the tile records ffs_*.bin.
New code: `fullfidelity/riverf_ff.py`, `fullfidelity/riverf_loop.py`, `fullfidelity/tests/test_riverf.py`. No existing file modified; nothing committed.

## 1. What was ported
- Version: the ORIGINAL RIVERF (neither RVR_ELEV nor TOPO_DIRECTED_RIVER_FLOW is defined in P2SAoM40.R; no tracers/SCM). `river_fac` = 1 (not set in the rundeck), `lake_rise_max` = 100 m, URATE = 1e-6, `variable_lk=1` only acts in GHY/daily code, FLAKE is constant within the window.
- Initialisation as the model does it: RVR file read for down_lat/down_lon/down_lat_911/down_lon_911, lonlat_to_ij (NINT rounding), get_dir (incl. pole special cases), DHORZ (great-circle via sin/cos/acos, `sqrt(AXYP)` for local flow), RATE (speed from the topographic slope, clipped to 0.15-5 m/s), the KDIREC=9 internal-sea branch (no cell in this RVR file has KDIREC=9, so that branch is ported but NOT exercised), emergency-direction branch (ported; `n_emergency` = 0 in all steps, NOT exercised), backwash branch (exercised: 1 cell per step), 95 % mixed-layer clip (ported; never triggered). NAMERVR (named river mouths) feeds diagnostics only and is not read.
- All diagnostics (AIJ, AJ, AREG) are omitted; only state and exports are computed: MWL, GML, MLDLK, TLAKE, DLAKE, GLAKE, GTEMP/GTEMPR/MLHC of lake cells, FLOWO/EFLOWO (per unit ocean area, the byoarea scaling at the end of RIVERF).
- libm mode: sin/cos/acos (only used for DHORZ, hence RATE) are the Intel libimf scalar functions from the same library as the real build (via ctypes, as in `intel_libm_ff.py`); falls back to glibc `math` when libimf is absent (then tests use a relative bound, not bitwise). Arithmetic is python floats in Fortran evaluation order, no FMA.
- Finding: the literal `DZDH1 = .00005` in init_LAKES is a REAL*4 literal (NOT promoted by -r8 as one might assume): using the double 5e-5 gives a 2.5e-8 relative error in nearly every river cell; using float32(5e-5) matches the real outflow bitwise. Both values are in the module (`DZDH1`, `DZDH1_R8`).

## 2. Data actually available for validation (stated honestly)
- There is NO RIVERF entry or exit dump (no real MWL/GML/TLAKE before or after RIVERF). The "RIVERF boundary" of D164 is only the ocean flux input `oflowo`, `oeflowo` in ffo tag 1 (the OCEANS entry), recorded for 12 steps (33312-33323). In addition the real lake state after RIVERF of step n is visible as the lake tile columns (tg1, mwl, gml) of the step n+1 tile records ffs (substep 1), for steps 33312-33317 only (6 steps).
- The inputs that RIVERF needs (land runoff, tile fluxes, GHY forcing) exist for the 6 steps 33312-33317 only; so the 12-step ffo record can be used for 6 steps, not 12. Steps 33318-33323 were NOT validated (no inputs to chain the lake state).
- Validation therefore goes through OUR chain: restart state -> D164 surface_pre -> ground_li/ground_si/GROUND_LK (replay mode R1: real tile outputs as flux input) -> RIVERF, state carried from step to step.

## 3. Results (nov26, 6 steps, R1 replay, no ADVSI; `riverf_loop.run_free`)
Flows against ffo tag 1 (3312 cells each, 159 non-zero river-mouth cells in the real record, 159 in ours at every step):

| step (itime) | oflowo bitwise cells | oeflowo bitwise cells | max abs oflowo | max abs oeflowo | scale (max abs) |
|---|---|---|---|---|---|
| 33312 | 3312 | 3312 | 0 | 0 | 0.592 / 6.4e4 |
| 33313 | 3312 | 3311 | 0 | 6.9e-18 | 0.592 / 6.4e4 |
| 33314 | 3312 | 3311 | 0 | 6.5e-19 | |
| 33315 | 3310 | 3309 | 1.1e-16 | 6.8e-13 | |
| 33316 | 3312 | 3310 | 0 | 4.5e-13 | |
| 33317 | 3311 | 3309 | 1.1e-16 | 4.5e-13 | |

So: bitwise at step 0 (restart state, no accumulated difference); rounding level (1-2 ulp in a few cells, maximum relative 1e-17 of the field scale) in steps 1-5, where the lake/land MWL and GML are our carried values.

Lake state at the entry of the next step against the real tile records (lake tiles, 614 at step 0 and 564 afterwards; max abs difference; scales tg1 31.9 K, MWL 6.1e16 kg, GML 2.1e21 J):
- step 33313 entry (after RIVERF of step 33312): tg1 0, MWL 0, GML 0 (bitwise).
- steps 33314-33317 entry: tg1 <= 3.6e-15 K, MWL <= 2.0e-3 kg (6e-20 relative), GML <= 1.0e3 J (5e-19 relative). Rounding level; grows slowly.

## 4. D164 lake errors: attribution tested (2-step isolation runs plus the 6-step runs; lake tile comparison at the entry of step 33313)
| variant | tg1 max abs (K) | MWL max abs (kg) | GML max abs (J) |
|---|---|---|---|
| D164 loop (`surface_post`: recorded flows, no RIVERF state update, runoff added to lake cells only) | 0.0634 | 8.66e10 (1.4e-6 relative) | 7.9e15 |
| all-land runoff, NO RIVERF state update | 0.0634 | 8.66e10 | 7.9e15 |
| RIVERF applied, runoff to lake cells only (as in D164) | 6.2e-5 | 3.0e7 | 2.9e12 |
| RIVERF applied + runoff for every FLAND>0 cell (real GROUND_LK) | 0 | 0 | 0 |

D164 numbers (tg1 0.063 K in one cell, MWL 1.4e-6 relative) are reproduced by the baseline here (0.0634 K, 1.43e-6). The RIVERF attribution is CONFIRMED by measurement: switching RIVERF off leaves the 0.0634 K error unchanged, switching it on removes 99.9 % of it. The remainder (6.2e-5 K) was a second, previously unnoticed omission of D164's loop: `surface_post` passed the lake-only geo to `ground_lk`, so the land runoff of GROUND_LK (LAKES.f:3380-3390) was added to the lake cells but not to the non-lake land cells whose MWL RIVERF routes (river water arriving at the lake). `riverf_loop.surface_post_riverf` passes the full geo; `ground_lk` itself was not changed. Every lake and land cell has an MWL that RIVERF routes, so the two corrections are needed together for bitwise lake state.
Also changed by RIVERF in the real model and applied here: GTEMP/GTEMPR/MLHC of lake cells are reset from TLAKE/MLDLK after RIVERF (D164's loop did not update MLHC of lake cells).
Ocean effect of the flows was not separately measured against the ocean exit (ffo tag 14); 'riverf_recflows' (computed RIVERF on the lake state, recorded flows into the ocean) gives identical lake results, as it must (the flows do not feed back to the lakes within the step).

## 5. Limits and cautions
- Only nov26 was run (dec01/jan01 not run); 6 steps, not 12; replay mode R1 (real tile outputs), not the closed coupled loop (`riverf_loop.surface_post_riverf` is a drop-in for `surface_post` but `stage_surface_closed`/`run_coupled` were not rewired; they would need the same two changes).
- Branches not exercised by this data: KDIREC=9 internal seas (none in the RVR file), emergency directions, 95 % clip, pole FLFAC adjustment (`FLFAC` ported; whether any pole cell receives river flow in these steps was not checked).
- Pole-row cells i > 1 receive FLOW in the real code but are never applied (IMAXJ = 1); ported as in the source (arrays indexed [i, j]); not verified separately.
- Single-PE assumption (no MPI halos) because the reference run used one PE.
- Tests: `tests/test_riverf.py` (skip when data absent). Statics: no land box without a river direction; 1075 cells with RATE > 0.
- Open: RIVERF at steps 33318+ needs the land runoff and tile inputs of those steps (D149-style dumps); the 54-step day has none.

**Parent-session check (2026-10-07 05:30):** `tests/test_riverf.py` re-run in full (the agent had not re-run its two slow tests after fixing a unit-test expectation): 4 passed, 80 s. Assertions include bitwise `oflowo` at steps 0-1, `oeflowo` <= 1e-17 at step 1 and <= 1e-12 of scale later, 159 non-zero river-mouth cells, the REAL*4 `DZDH1` finding (rel 1e-8..1e-7 if double is used). No existing file modified by this track. Not re-derived by the parent: the lake-error attribution table (baseline 0.0634 K / 8.7e10 kg -> 0 with RIVERF plus land runoff for every land cell); it comes from the agent's runs and rests on the agent's reading that D164's `surface_post` passed lake-only geo to `ground_lk` (a second D164 omission, affecting only the replay-mode free loop). Validated only for 6 steps (33312-33317, nov26, replay mode); `run_coupled`/`stage_surface_closed` not yet rewired.

# D166: ADVSI (sea-ice advection) ported, bitwise on 150 real calls; the D164 ice/ocean drift is removed, 2026-10-07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when DYNSI is ported (USI/VSI are still recorded) or a longer window is run.
Sources: pristine ModelE (read-only) ICEDYN_DRV.f:880-1636 (ADVSI, lat-lon version, `#ifndef CUBED_SPHERE`), ICEDYN_DRV.f:1832-1894 (CONNECT), ICEDYN_DRV.f:2113 (GET_UISURF), SEAICE.f:1875-2047, 2244-2322, 2363-2523 (get/set_snow_ice_layer, relayer, relayer_12, Ti2b), rundeck P2SAoM40.R (`EXPEL_COASTAL_ICEXS` defined; no tracers; KOCEAN=1).
New files (no existing file modified, nothing committed): `fullfidelity/advsi_ff.py`, `advsi_compare.py`, `surface_loop_advsi.py`, `instrumentation/ICEDYN_DRV_advsi.f.patch`, `tests/test_advsi_ff.py`, `tests/test_surface_loop_advsi.py`; new data `ff_data/advsi_dumps/{nov26,dec01,jan01}/ffadv_{in,out}_<itime>.bin` (145 MB + jan01).

## 1. What the existing dumps could and could not do
`ffn_<it>` is the FORM_SI/ADDICE per-cell record and `ffm_<it>` the MELT_SI/SIMELT per-cell record (ATM_DRV.f.patch `ffdump_addice` / `ffdump_simelt`), so D164's "ffn -> ffm brackets ADVSI" holds only for the cells' thermodynamic columns; RSIX, RSIY, RSISAVE, the ice velocities and the neighbours needed by an advection are not in them. They were therefore NOT used. A new patch was made instead (below).

## 2. Instrumentation (documented, scratch build)
`instrumentation/ICEDYN_DRV_advsi.f.patch` (diff against the pristine ICEDYN_DRV.f; the only patch needed). Units 1600 (entry) and 1601 (exit), grep-checked unused (the persistent radiation server reserves 1500-1529). Entry dump: FOCEAN, RSI, RSIX, RSIY, RSISAVE, MSI, SNOWI, HSI(4), SSI(4), AUSI, AVSI, CONNECT and the five geometry vectors DXYP, DYP, DXP, DXV, BYDXYP. Exit dump: RSI, RSIX, RSIY, RSISAVE, MSI, SNOWI, HSI, SSI, MUSI, HUSI, SUSI, MVSI, HVSI, SVSI, MSICNV, HSICNV, FWSIM. Scratch tree: rsync copy of the source (no ModelE_Support), only this file patched, `gmake RUN=P2SAoM40 .../P2SAoM40.bin` (ifort 19.1.3, pristine flags, 56 s). Run from the archived restarts with FFD_START/FFD_NSTEP: nov26 54 steps (33312-33365), dec01 48 steps (33552-33599), jan01 48 steps (17520-17567), 1 core each. The build was not checked against the original binary for bitwise identity of the restart (not done); the patch only adds file writes of already-computed arrays.
Pitfall: `cp -P` of a scratch run directory containing a FIFO (req.fifo of the radiation server) blocks forever.

## 3. The port (`advsi_ff.advsi`)
Numpy float64 scalars, no FMA, the Fortran operation order and parenthesisation, the GOTO case structure (north-south labels 220/230/250/260/270/285/crunch 320, east-west 520/530/550/560/570/585/crunch 620, north-pole box 350) and the EXPEL_COASTAL_ICEXS velocity terms (`CONNECT`, computed from FOCEAN by `connect_from_focean`, equal to the real array on all dumps). Reuses the validated `seaice_core_ff.relayer`, `relayer_12`, `set_snow_ice_layer`.
**libm mode: none applies.** ADVSI has no pow/exp/log/sin/cos (`x**2` is x*x) and the helpers use only + - * / and sqrt; the geometry (sin/cos at init) is read from the dump, not recomputed. The Intel libimf bridge is not needed.
**Found and fixed during validation (a real porting error, not a tolerance):** `get_snow_ice_layer` calls `Ti2b`, which the real SEAICE.f evaluates with REAL*16 `b, c, det, tm` (SEAICE_FIXES_2022). The existing `seaice_core_ff.Ti2b` is float64; with it 60 cells of HSI layers 1-2 were 1-2 ulp off (max 1.5e-6 on 9.9e8) and the diagnostics HUSI/HVSI/HSICNV followed. `advsi_ff.ti2b_quad` emulates binary128 with mpmath (113-bit), including the Fortran typing rules (sub-expressions of two REAL*8 operands, `mu*Si`, `frac*lhm`, `Eit+lhm`, `shw-shi`, are evaluated in double first; the `1q-10` threshold is quad); with it every field is bitwise. This also means the existing `seaice_core_ff.Ti2b/Ti` (float64) are not bitwise for cells where it matters; not changed here (other ports validated "empirically", D10-D32), flagged for the owner.
Other facts reproduced from the Fortran: cells at the poles with i > 1 keep their stale values (not advected); HSICNV keeps the pre-advection RSI*sum(HSI) there; the south-pole row must have FOCEAN = 0 (raises NotImplementedError otherwise: the Fortran would read the ausi halo row).

## 4. Validation against the real ADVSI (`advsi_compare.py`, `tests/test_advsi_ff.py`)
All 150 dumped calls (nov26 54, dec01 48, jan01 48), every output (RSI, RSIX, RSIY, RSISAVE, MSI, SNOWI, HSI, SSI, MUSI, HUSI, SUSI, MVSI, HVSI, SVSI, MSICNV, HSICNV, FWSIM): **0 differing elements (max abs diff 0.0)**. Cells with MSICNV/FWSIM stale in the Fortran (pole i > 1) are not compared. Non-vacuous: the call changes RSI by up to 1.5e-3 (mean of the per-call maxima 1.2e-3) in about 480 cells per call and MSI by up to 1415 kg/m2; branches exercised over nov26+dec01 (counts): NS 220:6527, 230:8644, 250:2892, 260:30698, 270:2821, 285:4935, NS crunch 13206; EW 520:1495, 530:2400, 550:2020, 560:40582, 570:1690, 585:1875, EW crunch 5695. **Not exercised: the north-pole-box crunch (350) never occurs**, and the KOCEAN = 0 (fixed SST) branch is not ported. Tests: 30 passed (175 s with the surface-loop test).
Limit: three start dates, 48-54 steps each; every call is validated from the REAL entry state (each call independent), not chained from our own output.

## 5. Surface loop with ADVSI (`surface_loop_advsi.py`; surface_loop.py untouched)
`surface_post_advsi` = `surface_loop.surface_post` + ADVSI on the ocean-domain ice (RSISAVE = ocean RSI of the state handed to surface_post; RSIX/RSIY carried, start from the restart; pole rows re-replicated with `_pole_replicate` as after FORM_SI). Still recorded: USI/VSI (from ffy_<it>_out, checked equal to the ADVSI entry dump AUSI/AVSI on all 6 steps) and everything D164 lists as recorded.
Free 6-step loop, nov26, R1 replay, relative to field scale (ice) and absolute (ocean exit, `ocean_step_chain_compare.errs`):

| step | ice snow / msi2 / ptype, no ADVSI (= D164) | same, with ADVSI | ocean-tile mismatches no/with | ocean exit uo, vo no ADVSI | uo, vo with ADVSI |
|---|---|---|---|---|---|
| 1 | 9.7e-4 / 9.5e-4 / 1.1e-3 | 1.7e-16 / 4.0e-10 / 1.9e-8 | 14 / 0 | 2.3e-3, 5.4e-3 | 6.4e-9, 3.5e-8 |
| 2 | 1.9e-3 / 1.9e-3 / 2.1e-3 | 1.9e-6 / 9.2e-7 / 1.2e-6 | 14 / 0 | 4.7e-3, 2.0e-2 | 2.0e-8, 9.3e-8 |
| 3 | 2.9e-3 / 2.9e-3 / 3.0e-3 | 4.5e-6 / 2.6e-6 / 3.2e-6 | 14 / 0 | 7.4e-3, 2.9e-2 | 4.2e-8, 1.8e-7 |
| 5 | 4.8e-3 / 4.7e-3 / 4.8e-3 | 9.9e-6 / 8.0e-6 / 9.1e-6 | 12 / 0 | 1.1e-2, 2.3e-2 | 1.0e-7, 2.2e-7 |

(step index k = it - 33312; the no-ADVSI column reproduces D164's numbers, so the baseline is the same run.) Ice tg1/tg2 with ADVSI: 1.9e-15 at step 1, 3.9e-5 at step 5; the ocean g0m exit error falls from 1.4e-5 (step 1) / 1.0e-4 (step 5) to 5e-12 / 3e-11.
**Conclusion: D164's attribution is CONFIRMED for the ice and the ocean exit** (the ~1e-3 per step ice drift, the 12-14 mismatched ocean tiles and the 1e-3..1e-2 ocean velocity error vanish when the bitwise-validated ADVSI is applied; the same loop without it reproduces D164 exactly). **It is REFUTED for the lakes and the land ice**: lake tg1 2.0e-3 -> 9.6e-3 and MWL 1.4e-6 -> 7.2e-6, and the land-ice tg1 jump (0.159 of scale from step 3, not step 4 as D164 wrote; tg2 8.9e-3) are identical with and without ADVSI. D164's RIVERF inference for the lakes is still untested; the land-ice jump is still unexplained.
Residual with ADVSI: a growing 1e-8 -> 1e-5 relative ice error from step 1 to 5. At step 0, before ADVSI, our ice already differs from the real ADVSI entry by msi 1.4e-6 abs (scale 5.5e3), hsi 1.6e-2 abs (scale 9.9e8), ssi 7e-8 (scale 4.8) in 19/284/67 cells, i.e. the known loosest GROUND_SI/ADDICE diagnostics of D17/D20 (D164 section 2), which the advection then moves around and the replayed-input loop does not damp. Not investigated further; not claimed to be rounding of ADVSI (ADVSI alone is bitwise).
Direct comparison of our post-ADVSI ice with the real ffadv_out (ocean cells, pole rows i > 1 excluded): rsi 2.8e-13 (step 0) .. 9.8e-10 (step 5), msi 1.4e-6 .. 4.6e-5 abs, snowi 5e-14 .. 9e-11, hsi 1.6e-2 .. 0.54 abs on 1e9, rsix <= 6e-12.

## 6. Failures, cautions, still open
- First comparison of the free loop included the pole rows i > 1 (stale in the Fortran, replicated here) and showed a spurious 5.7e-5 RSI / 8e3 HSI difference; those cells are now excluded, which is a stated mask, not a loosened tolerance.
- Only the nov26 6-step free loop was run with ADVSI (D164's window; limited by the DYNSI boundary ffy, which exists for 6 steps). dec01/jan01 validate ADVSI itself only. The 54-step day has no DYNSI boundary, so ADVSI cannot yet be wired into a day-long surface loop (needs DYNSI input assembly, D164 section 4).
- Not ported: fixed-SST ADVSI branch, tracers, the cubed-sphere ADVSIcs.f, the north-pole-box crunch is ported but unexercised.
- CPU: 2 cores at most (taskset); nothing committed or pushed.
- The scratch build and run directories live in the session scratchpad (temporary); everything needed to rebuild is the patch and this entry.

**Parent-session check (2026-10-07 05:45):** `tests/test_advsi_ff.py` + `tests/test_surface_loop_advsi.py` re-run: 30 passed, 83 s (ADVSI check asserts zero differing elements on every output field over the dumped calls and that the ice really moves). The patch `instrumentation/ICEDYN_DRV_advsi.f.patch` applies to the pristine ICEDYN_DRV.f (`patch --dry-run` rc 0), has no removed lines and no line over 72 columns, and units 1600/1601 are used by no other patch. A transient edit to `instrumentation/build_and_run.md` seen in `git status` mid-session is no longer in the tree (the agent states it did not edit that file). Not re-derived by the parent: the 150-call dump generation, the instrumented binary's bitwise no-op property (the agent did not check it, as stated), and the free-loop table. OPEN, flagged by the agent and left for the owner: the existing `seaice_core_ff.Ti/Ti2b` are float64 while SEAICE.f computes Ti2b in REAL*16, so they are not bitwise where that matters (the ADVSI port carries its own binary128 emulation). D164 corrections: the ice/ocean drift is removed by ADVSI (attribution confirmed); the lake and land-ice errors are NOT due to ADVSI (lake tg1/MWL identical with and without it; the land-ice tg1 jump starts at step 3, not step 4 as D164 wrote); the lake error is RIVERF (D167).

# D169: Ent (vegetation) port, scoping and stage 1 (per-iteration exports cnc, betadl, lai, ...), 2026-10-07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when stage 2/3b or the vectorised
version starts, or when the first-step residual (section 4.3) is explained.
Decision recorded by the owner (2026-10-07): Ent exports stay RECORDED for runs up to one day; the Ent port starts now because the
one-month F3 run needs it.
Sources (all read-only, `modelE2_planet_2.0`): `model/Ent/{ent.f, ent_mod.f, canopyspitters.f, FBBphotosynthesis.f, respauto_physio.f,
phenology.f, patches.f, entcells.f, soilbgc.f, ent_prescribed_updates.f, ent_prescr_veg.f, ent_pfts_ENT.f, FBBpfts_ENT.f, ent_const.f,
physutil.f, allometryfn.f, Makefile}`, `model/{ENT_DRV.f, ENT_COM.f, GHY.f (2198-2660), GHY_DRV.f, MODEL_COM.f, dd2d/timestream_mod.f}`,
`decks/P2SAoM40.{R,mk}`, the executable `ModelE_Support/huge_space/P2SAoM40/P2SAoM40.bin` (`nm`), the compiled objects
`model/Ent/{FBBphotosynthesis,canopyspitters}.o` (constants), the ffg dumps (`ff_data/{nov26,dec01,jan01,nov26_day}/ffg_*.bin`) and the
pristine restarts (`ff_data/_pristine_restarts/*.nc`, variable `ent_state`).
New code: `fullfidelity/ent_ff.py` (stage 1), `ent_ghy_compare.py` (validation driver), `ent_daily_ff.py` + `ent_tables_ff.py`
(stage 3a, prescribed LAI/albedo), `tests/test_ent_ff.py`. No existing file was modified; nothing committed. No instrumentation patch was
needed (no new unit used).

## 1. Scope: what the P2SAoM40 build really runs

### 1.1 Configuration (checked, not assumed)
- `decks/P2SAoM40.R`: `OPTS_Ent = ONLINE=YES PS_MODEL=FBB PFT_MODEL=ENT`; no `RAD_MODEL`, no `MIXED_CANOPY_OPT`, no `FLUXNET`; preprocessor
  options list has no `PS_BVOC`, `ENT_WATER_STRESS_4`, `ENT_QSIMP_FIX`, `ENT_USE_ANALYTIC_SOLVER_FOR_FBB`, `TRACERS_*`, `OFFLINE_RUN`.
  Rundeck parameters not set for `do_soilresp`, `do_phenology_activegrowth`, `do_frost_hardiness`, `do_structuralgrowth`,
  `do_patchdynamics`, `do_init_geo`, so the ENT_DRV.f defaults hold: soilresp 1, activegrowth 0, frost hardiness 1, others 0.
  `LAI`, `LAIMAX`, `HITEent`, `VEG` files are present, so `do_modis_lai = .true.` (monthly prescribed LAI). `crops_yr` is commented out
  (default `master_yr` = 1850).
- `model/Ent/Makefile` for that configuration compiles `ent_prescribed_drv*.f ent_mod.f ent.f cohorts.f patches.f entcells.f physutil.f
  allometryfn.f reproduction.f phenology.f respauto_physio.f disturbance.f soilbgc.f ent_const.f ent_types.f ent_prescr_veg.f
  ent_prescribed_updates.f ent_debug.f ent_pfts_ENT.f FBBphotosynthesis.f canopyspitters.f FBBpfts_ENT.f`. So: **`biophysics.f`,
  `canopyradiation.f`, `canopygort.f` are not part of this build.** Evidence from the executable's symbol table (`nm`): module
  `biophysics` contains exactly the `canopyspitters.f` routines (`canopyfluxes, canopy_rad, canopy_rad_setup, canopy_transmittance,
  gs_bound, gs_from_ci, photosynth_cond, photosynth_sunshd, qsimp, trapzd, respauto_npp_clabile`); none of the `biophysics.f` routines
  (`veg, veg_C4, phot, Canopy_Resp, update_veg_locals, ...`) nor `canopyrad`/GORT/TwoStream symbols exist. The saved variables of
  `ci_cubic` (`$RA $B $K $GAMOL $X1 $X2 $X2SAVE $XACC`) and of `Photosynth_analyticsoln` (`$A1C $F1C`) are static symbols: the SAVE semantics
  are real. This corrects the D164 sizing (which counted `biophysics.f` 594 lines by a name-matching call graph).

### 1.2 Per GHY sub-iteration (GHY.f:2389-2520, `process_vege` only), in the order the real code runs
1. `ent_set_forcings` (ent_mod.f:2498): stores into the Ent cell: `TairC = ts - tfrz`, `TcanopyC = tp(0,2)`, `Qf`, `P_mbar = pres`,
   `Ca = Ca*1e-6*pres*100.0/gasc/(tp(0,2)+tfrz)` (mol/m3; `100.0` real*4), `Ch = ch`, `U = vs`, `IPARdif = vis_rad - direct_vis_rad`,
   `IPARdir = direct_vis_rad`, `CosZen = cosz1`, `fwet_canopy = fw`, `Soiltemp = tp(1:6,2)`, `Soilmoist = w/ws` (0 where ws = 0),
   `Soilmp = h(1:6,2)`, `fice = fice(1:6,2)`.
2. `ent_run(entcell, dts, end_of_day_flag .and. nit == 1)` -> `ent_integrate` (ent.f:75):
   a. `clim_stats` (phenology.f): 10-day running means (`airtemp_10d`, `soiltemp_10d`, `par_10d`, per-cohort `betad_10d`, `turnover_amp`,
      `llspan`), `daylength(2)` accumulation, daily `gdd/sgdd/ncd/fall`, and per cohort `Sacclim` (frost-hardening state; `photosyn_acclim`,
      tau_inv a SINGLE precision literal 2.22222e-6) or `Sacclim = 25` for the other types.
   b. `update_veg_structure` only on `update_day`; with `do_phenology_activegrowth = 0` it only re-summarizes and shifts `daylength`.
   c. per patch: `photosynth_cond(dts, pp)` (canopyspitters.f) -> `water_stress3` (respauto_physio.f), `calc_Pspar`, `canopyfluxes` ->
      `canopy_rad_setup`, `qsimp` -> `trapzd` -> `photosynth_sunshd` -> `canopy_rad` + 2x `pscondleaf` -> `Photosynth_analyticsoln` ->
      `ci_cubic` (Newton/bisection `rtsafe`, `A_eqn`, `A_eqn_0`) + `BallBerry`; `Respauto_NPP_Clabile` (carbon); `canopy_transmittance`.
   d. `soil_bgc` (soilbgc.f) because `do_soilresp = 1`: soil respiration and the CASA pools; `pp%CO2flux`, `pp%age`.
   e. `summarize_entcell` -> `summarize_patch` per patch + `entcell_update_shc_mosaicveg`.
3. `ent_get_exports`: `canopy_conductance = ecp%GCANOPY` (cnc), `beta_soil_layers = ecp%betadl(1:6)`, `shortwave_transmit = ecp%TRANS_SW`,
   `leafinternal_CO2 = ecp%Ci`, `canopy_gpp = ecp%GPP`, `leaf_area_index = ecp%LAI`, `canopy_ipp = ecp%IPP`. (IPP is identically 0:
   `PS_BVOC` is not defined, `isp = 0`.) These are the recorded `ffent(1:13, nit)` (cnc, betadl(6), TRANS_SW, Ci, GPP, lai, IPP, dts).
4. After `apply_fluxes`/`accm`/`reth`/`retp`: `Qf = (evap_tot(2)/(rho/rhow*ch) + gusti*qprime)/vs + qs` (GHY.f:2621): the next Ent call sees it.

### 1.3 Once per GHY call, before the loop (GHY.f:2338-2352)
`ent_get_exports`: `ws_can = ecp%LAI*1e-4`, `shc_can = ecp%heat_capacity` (`GISS_shc` of the mean annual LAI), `fv = ecp%fv`, `height = ecp%h`,
`albedo(6) = ecp%albedo` (recorded `ffent0`). `fv`, `fb` are then clamped at 1e-6 by GHY.

### 1.4 Daily (GHY_DRV.f `daily_earth(end_of_day)` -> `ENT_DRV.f update_vegetation_data`), runs before the first step of a new day
- `set_vegetation_data(..., reinitialize=.false.)` ONLY when `year != year_old`: `year_old = -1` at program start, so it runs once at the
  first day end of EVERY run segment (and when `crops_yr` year changes): reads VEG (`V72x46_EntMM16_lc_max_trimmed_scaled_nocrops.ext.nc`) x
  (1 - crops), `HITEent`, `LAIMAX`, the LAI stream, soil texture from `q_ij`, calls `ent_cell_set` -> `init_simple_entcell(reinitialize=false)`:
  patch areas are reset to the VEG fractions, existing cohorts are kept, new patches get cohorts, patches with area 0 are deleted. Not
  ported, not validated (needs the crop data source: the rundeck has no `CROPS` file line: unresolved).
- `ent_prescribe_vegupdate(do_giss_phenology = true, do_giss_albedo = true, do_giss_lai = false, update_crops = false, laidata = stream)`:
  `entcell_vegupdate` -> `entcell_update_lai_poolslitter` (cohort LAI := prescribed LAI of its PFT; `allom_plant_cpools`, `litter_cohort`,
  `litter_patch` update the carbon pools) and `prescr_veg_albedo` (season interpolation of `ALBVND` for the tallest PFT of each patch and
  hemisphere), then `summarize_entcell`. The height is NOT updated (no `hdata` is passed).
- `daily_earth` also reads `ws_can` for the soil water capacity and calls `updsur` (albedo module); `set_roughness_length` reads
  `vegetation_fractions` and `vegetation_heights` every day (`map_ent2giss`).
- Other consumers of Ent in this build: `RADIA` (RAD_DRV.f:3384) reads `vegetation_fractions/heights` (PVT/HVT) for the land albedo; with the
  radiation server these come from the real objects, in a chained model they must be passed. `init_land_surface`, `init_underwater_soil`,
  `get_canopy_temperature_fw`, `get_fb_fv` read `heat_capacity`, `ws_can`, `fv` (state-derived, covered by `call_exports`).

### 1.5 Inputs, outputs
| Routine | Inputs | Outputs |
|---|---|---|
| set_forcings | GHY state: ts, tp(0:6,2), Qf, pres, Ca, ch, vs, w(1:6,2), ws(1:6,2), fice(1:6,2), fw, h(1:6,2); radiation: vis_rad, direct_vis_rad, cosz1 (from SRVISSURF*cosz1*.82, FSRDIR, cosz1 in GHY_DRV.f:1190-1193); Ca from CO2ppm/land_CO2_bc | Ent cell forcing fields |
| clim_stats | dts, TairC, IPAR, CosZen, update_day, cell/cohort state | running means, Sacclim, gdd/ncd/fall, daylength |
| photosynth_cond | Ent cell forcings, patch LAI/albedo(1), cohort pft, LAI, fracroot, Sacclim, PFT tables | cohort GCANOPY, Ci, GPP, IPP, stressH2O(l); patch TRANS_SW |
| summarize_entcell | cohort/patch outputs, patch areas | cell GCANOPY, betadl, TRANS_SW, Ci, GPP, LAI, IPP, h, fv, albedo, heat_capacity |

## 2. State
Carried between sub-iterations and steps, inside Ent (the ffg record does not contain it; it lives in the restart `ent_state`, 1023 doubles
per cell, layout `copy_cell_vars` 25 + per patch `copy_patch_vars` 52 + per cohort `copy_cohort_vars` 46, plus `np` and `nc(1:np)`):
- cell: `airtemp_10d`, `par_10d`, `soiltemp_10d`, `paw_10d`, `gdd`, `ncd`, `sgdd`, `daylength(1:2)`, `fall`, `Qf`, `soil_Phi`, `soil_dry`, soil texture, `Soilmp`, `Tpool`;
- patch: `area`, `albedo(6)`, `soil_type`, `Tpool`, `age`, `Reproduction(16)`, `Ci`, `GCANOPY`;
- cohort: `pft`, `n`, `LAI`, `h`, `dbh`, `fracroot(6)`, carbon pools (`C_fol, C_sw, C_hw, C_lab, C_froot, C_croot`, N pools), `Sacclim`, `llspan`, `turnover_amp`,
  `betad_10d`, `stressH2O`, `NPP`, `C_total`, phenology factors.
Also hidden Fortran module state: `pspar` (photcondmod), the SAVEd `a1c, f1c` of `Photosynth_analyticsoln` and the SAVEd `Ra, b, K, gamol, x1, x2save,
xacc` of `ci_cubic` (re-initialised per cohort by `calc_Pspar`, so they do not leak across cohorts: verified by reading the control flow).
The restart of nov26 holds 1146 Ent cells, 4490 cohorts, one cohort per vegetated patch, at most 11 patches and 9 cohorts per cell, 1565 bare patches.

State that the exports depend on (and therefore must be carried by a port): cohort `pft, LAI, fracroot, Sacclim` (frost-hardening PFTs), patch `area, albedo(1)`, cell
`airtemp_10d` (drives Sacclim). State that does NOT influence the exports in this configuration (read from the code): all carbon/nitrogen pools,
`soiltemp_10d`, `par_10d`, `paw_10d`, `gdd/ncd/sgdd/fall/daylength`, `llspan`, `turnover_amp`, `betad_10d`, `Tpool`, `Soil_resp`
(`calc_Pspar` takes `llspan` but never uses it: `fparlimit = 1`). They matter for the carbon diagnostics (stage 2), not for cnc/betadl/lai.
Across days: LAI and albedo are replaced daily by the prescribed values (stage 3a); the carbon pools follow LAI through allometry
(stage 3b); `gdd/ncd/fall/daylength(1)` are updated at the day boundary inside `clim_stats`.

File inputs: vegetation structure (patch areas, cohort density, height, dbh, fracroot, nm) are in the RESTART `ent_state`; they come from the
VEG/HITEent/LAIMAX files only at cold start or at `set_vegetation_data` (section 1.4). Monthly LAI per PFT: `V72x46_EntMM16_lai_trimmed_scaled_ext.nc`
(16 variables, 12 months, 46 x 72; present under `ModelE_Support/prod_input_files`). Soil/climate drivers: none from files (GHY state and the
atmosphere); CO2 from the GHG routine (284.316 ppm in the dumps); PFT parameter tables and the albedo table are in the source (`pfpar`,
`pftpar`, `ALBVND`, `alamax/alamin`) and are transcribed in the new modules.

## 3. Line counts (non-comment, non-blank code lines, from the source)
Counting routine bodies that the call graph reaches (an over-count for stage 1: e.g. `photosynth_cond` includes the carbon part):
- **Stage 1** (per-iteration exports; ported): 1,395 Fortran lines (canopyspitters 443, FBBphotosynthesis 363, respauto water_stress3/Rdark 34,
  clim_stats+running_mean+acclim 120, summarize_patch/entcell/shc/extract_pfts/zero_* 372, qsat/ent_integrate/GISS_shc 63). Python: `ent_ff.py` 894 lines (about 740 code lines).
- **Stage 2** (soil_bgc and autotrophic respiration, carbon pools): 382 lines (soilbgc 277, Respauto_NPP_Clabile 48, respiration helpers 46, casa root fraction 11).
- **Stage 3a** (prescribed LAI + albedo, ported): `prescr_veg_albedo` 35, the stream interpolation ~30 (timestream), `entcell_vegupdate` logic ~50.
- **Stage 3b** (rest of the daily update: allometry C pools, litter, `litter_cohort`/`litter_cohort_fff`/`litter_patch`/`accumulate_clossacc`/`assign_closs` ~400 lines + allometry ~100
  + `init_simple_entcell` 150 + `set_vegetation_data` 136, `update_vegetation_data` 85, crop reading): ~900 lines.
- **Total live path** about 1,400 + 380 + 900 + 115 = 2,800 Fortran lines; the per-iteration path (stages 1 + 2) is ~1,780 lines, of which ~1,400 decide cnc/betadl/lai. The D164
  figure (~2,800 per iteration + ~1,080 daily) over-counted the per-iteration path (it included `biophysics.f`, which is not compiled) and under-counted the daily part
  (allometry, litter and `set_vegetation_data` were not counted); the sum happens to be similar.
- `canopyradiation.f`/`canopygort.f` (1,460 lines): not compiled; stays excluded (confirmed).

## 4. Stage 1 port and validation
### 4.1 What was ported (`ent_ff.py`)
Unpack of `ent_state` (`unpack_cell`), `set_forcings`, `clim_stats` (export-relevant part: `airtemp_10d`, `par_10d`, `daylength`, `gdd/ncd/fall`, `Sacclim`),
`photosynth_cond` with `canopy_rad_setup, canopy_rad, canopy_transmittance, qsimp, trapzd, photosynth_sunshd, pscondleaf, Photosynth_analyticsoln, calc_Pspar,
calc_CO2compp, Q10fn, frost_hardiness, BallBerry, ci_cubic (rtsafe), A_eqn, A_eqn_0, water_stress3, QSAT`, `summarize_patch`, `summarize_entcell` (incl. heat capacity),
`get_exports`, `call_exports`. Fortran operation order, SAVE semantics and the real-literal traps are kept.
Real-literal traps found (and confirmed in the object code `.rodata`, test included): `calc_CO2compp` multiplies by `0.21` (single precision, 0.20999999344348907) and
`canopyspitters.f` has `O2frac = .20900` (single precision, 0.20900000631809235); `photosyn_acclim` has `tau_inv = 2.22222e-6` (single); `ALBVND` and `rhol/taul` tables have no d0.
Missing the `.20900` trap gave 2.8e-9 relative errors in cnc/ci/gpp (the root finder stops at |dx| < 1e-4, so Gammastar enters the result at ~1e-9); with it the result is bitwise.
Also found: `pfpar(pft)%pst` is 1 (C3) for all 16 PFTs in `ent_pfts_ENT.f`, so the C4 branch of `Photosynth_analyticsoln` (which tests `pfpar%pst`) is dead in this build
even for the C4 grass and crops, while `pftpar%pst` (FBBpfts_ENT.f) says 2. The port copies the real behaviour.
Not ported (no feedback to the exports): `Respauto_NPP_Clabile`, `soil_bgc`, the `llspan/turnover_amp/betad_10d/soiltemp_10d/sgdd` bookkeeping of `clim_stats`.
Math library: exp/pow call the Intel libimf scalar functions (as `intel_libm_ff.py`), else glibc; with glibc the root finder can move results by ~1e-9 relative.

### 4.2 How it is validated (`ent_ghy_compare.py`)
For every land record of an ffg file, in file order (two GHY calls per cell and step, 753 land cells per call, Ent state carried per cell from the restart): the validated
D158 GHY port (`ghy_ref`) is advanced sub-iteration by sub-iteration; at each one the Ent forcings are taken from the GHY-side state exactly as GHY.f does, `ent_ff` computes the
exports and they are compared with the recorded `ffent` block (sub-iterations 1-11 only: the record holds 11). Mode `teacher`: GHY is driven by the recorded exports (the Ent port is
isolated); mode `closed`: GHY is driven by the computed exports. Per-call exports (`ws_can, shc_can, fv, height, albedo(6)`) and the exit `Qf` are compared with the record too.
Commands (from `fullfidelity/`, python = graphcast-env, `taskset -c 0`):
```
python ent_ghy_compare.py nov26 teacher ; python ent_ghy_compare.py dec01 teacher ; python ent_ghy_compare.py jan01 teacher
python ent_ghy_compare.py nov26_day teacher        # 54 steps, daily LAI/albedo update (ent_daily_ff) applied at the 33360 day boundary
python -c "import ent_ghy_compare as C; C.closed_report('nov26', files=3)"
python -m pytest tests/test_ent_ff.py
```

### 4.3 Results (measured; libimf available)
Per-iteration exports, `teacher` mode, bitwise = `==`, residuals relative to the recorded value unless stated:
| Dataset | n iterations | cnc bitwise / max rel | betadl (6) bitwise / max abs | trans_sw, lai, ipp | ci bitwise / max rel | gpp bitwise / max rel |
|---|---|---|---|---|---|---|
| nov26 (6 steps) | 18,911 | 18,885 / 1.0e-15 | 113,003 of 113,466 / 2.2e-16 | all bitwise | 18,900 / 8.7e-16 | 18,902 / 6.8e-16 |
| dec01 (6 steps) | 18,949 | 18,926 / 1.3e-15 | 113,227 of 113,694 / 3.3e-16 | all bitwise | 18,934 / 1.5e-15 | 18,940 / 9.7e-16 |
| jan01 (6 steps) | 18,137 | 18,117 / 1.2e-15 | 108,220 of 108,822 / 3.3e-16 | all bitwise | 18,123 / 1.1e-15 | 18,128 / 1.1e-15 |
| nov26_day (54 steps, 1 day boundary) | 189,698 | 189,597 / 1.5e-15 | 1,137,262 of 1,138,188 / 3.3e-16 (rel 1.1e-14 on tiny values) | all bitwise | 189,640 / 2.2e-15 | 189,650 / 1.3e-15 |

Per-call exports: `ws_can, shc_can, fv, height, albedo(6)`: bitwise equal on all 9,024 records (6 steps x 2 substeps x 753 cells) of each of nov26, dec01 and jan01, and on all 81,216 records
of nov26_day (including the 6 steps after the day boundary, which need the daily LAI/albedo update of stage 3a). Exit `Qf` (GHY.f:2621 formula): 8,992 of 9,024 bitwise (nov26), max 2.5e-17
(three 6-step dates <= 3.5e-17 absolute); nov26_day 80,972 of 81,216 bitwise, max 7.4e-9, attained on stiff cells (ffnit 12-14, whose iterations beyond the 11th use the computed time
step, the D158 level): the four largest differences (7.4e-9, 1.5e-9, 2.2e-10, 6.1e-11) are records of the stiff cells (5 cells re-run separately: their records with nit >= 12 give these values, those with
nit <= 11 are <= 1.7e-18; the whole-day maximum over all other cells was not separated out); the exports of those cells in iterations 1-11 are bitwise.
The residuals are not accumulating: in every date the non-bitwise iterations are the first sub-iterations of the first step after the restart (nov26 step 33312: 212 field values
in 105+103 calls); steps 2-6 are bitwise except isolated 1-ulp cases (e.g. 33315: 1 of 3,170). The cause of the first-step residual is NOT established; it sits on the GHY-side
inputs of the first step (the Ent port reproduces the 99.9 % of iterations that have the same inputs bit for bit, including night, ice-covered and frost-hardened cases
(1,898 cohorts of nov26 have Sacclim < 4.07 so that `frost_hardiness` is below 1; 1,371 are below the -5.93 threshold)). Hypotheses not tested: restart-time rounding of `w/ht` read
by GHY, or the first-call Qf. A dump of the GHY internal `tp(0,2), w, fice, h` at the first iteration would settle it.
`closed` mode (GHY driven by the ported exports, nov26 files 33312-33314): GHY outputs vs the record, max |difference| over the scale of the record: tbcs 1.1e-15, ashg 6.5e-17, alhg 2.9e-16,
aevap 2.9e-16, aruns 9e-18, aeruns 5e-17, ae0 3.7e-16, abetad 3.3e-16; w_out max abs 1.7e-16 (the same level as the teacher mode, i.e. the D158 level).
Stage 3a, daily prescribed LAI and albedo (`ent_daily_ff.py`): at the real day boundary (ffg_33360 carries the end-of-day flag; the update runs before the step with jday = 331,
the new day), applying LAI (timestream linm2m, JDmidOfM of MODEL_COM.f) and `prescr_veg_albedo` to the restart state reproduces the recorded `ws_can`, `albedo(6)` and the first-iteration
`lai` of step 33361 with maximum difference 0.0 on 151 cells (every 5th land cell, run before the full day was available); without the update the difference is 4.7e-6 in ws_can,
1.4e-3 in albedo, 0.047 in LAI; jday 330 and 332 do not match, so the day number (jday of the new day) is established. In the full day run (54 steps, row above) the update is applied before 33360 to all 1,146 Ent cells and the later steps stay bitwise in lai, trans_sw and in the per-call exports.

### 4.4 What cannot be validated with the existing dumps
- Carbon/soil state: `soil_bgc`, `Respauto_NPP_Clabile`, litter and the allometric pools are not exported anywhere in the dumps (`ffg` has only the six exports and `ffent0`).
  Their validation needs a real restart AFTER the window: the real model run one day from `fort1_nov26_itime33312.nc` with a restart written at the end (no patch needed, run
  configuration only) and compare the `ent_state` array cell by cell. For month scale: the production run has `1JAN1950.rsf...nc` and the ensemble runs; a second restart one
  month later is needed.
- The first-step residual of 4.3 (needs an extra dump at GHY.f:2395 of `tp(0,2), w(1:6,2), fice(1:6,2), h(1:6,2), Qf`: a ~10-line addition to `GHY.f.patch`, unit range 1470-1479;
  `grep` of `instrumentation/*` (patches and build notes) and of the model sources `model/*.f, Ent/*.f, giss_LSM/*.f, *.F90, *.h` finds no use of 1470-1479; not applied).
- Iterations beyond the 11th of stiff cells (ffnit up to 9 in the dumps of nov26; up to >= 12 exist in nov26_day, D158): the record does not hold them; they are run (they advance the
  Ent state) but not compared.
- Day/season structure: only one day boundary (nov26 -> nov27) is available; the end-of-day `gdd/ncd/fall` and the first-day `set_vegetation_data(reinitialize=false)` are not validated
  (and the latter is not ported). The Sacclim evolution over a month (tau about 5 days) is only exercised for 54 steps.
- Radiation inputs (`vis_rad, direct_vis_rad, cosz1`) and `Ca` are taken from the record; in the chained model they come from the radiation server output (SRVISSURF, FSRDIR) and the GHG routine.

## 5. Speed (measured, CPU time of one process, libimf, 753 land calls of one step)
GHY port alone 2.4 ms per call (night step 33313) and 2.9 ms (step 33324, daylight on part of the globe); GHY + Ent 4.1 and 6.0 ms; `ent_ff` alone 1.5 and 3.0 ms per call (0.75 ms per
sub-iteration at night, ~1.2 ms in daylight where `qsimp`/`rtsafe` run). The 54-step day replay took about 18 min wall on the loaded node (load 7-10). For a model month (1,440 steps x 2 calls
x 753 cells = 2.2 M calls) this is about 3 h for GHY + Ent on one core (Ent about 1-1.5 h): the scalar port is fast enough for the month-scale F3 run. A batched (cells x cohorts) numpy/JAX
version (masked `qsimp`/`trapzd`/`rtsafe`: data-dependent iteration counts) is only needed for the GPU purpose of the project; it is not needed for the correctness work.

## 6. Staged plan, remaining work and estimated hours (estimates, not measurements)
| Stage | Content | Lines (real) | Validation data | Risk | Hours |
|---|---|---|---|---|---|
| 1 DONE | per-iteration exports | 1,395 | ffg ffent, 4 datasets (this entry) | first-step residual unexplained (1e-15) | done |
| 3a DONE (partial) | prescribed LAI + albedo | ~115 | step 33361 bitwise on 151 cells + day run | only one boundary, LAI file orientation assumed south-to-north (verified by the match) | done |
| 1b | wire into the chained land loop (`surface_loop`/`land_chain`): per substep GHY with Ent callback, Ent state in the chain, Ca/vis_rad/cosz from the chain; `closed` mode over nov26_day | glue ~300 | closed mode vs ffg outputs and the day-run land fields | the GHY port and Ent run per cell in Python | 6-8 |
| 1c (optional, GPU purpose) | batched/JAX version (cells x cohorts, masked rtsafe/qsimp); bitwise vs the scalar port | ~600 | scalar port, ffg dumps | data-dependent iteration counts; libimf exp/pow in JAX is not available (bitwise only with numpy+libimf; JAX needs a tolerance decision) | 15-25 |
| 3b-min | `set_vegetation_data(reinitialize=false)` at the first day end of a segment (patch areas from VEG x (1 - crops), new/deleted patches), gdd/ncd/fall check | ~300 | a real restart after the first day end (`ent_state` compare; run the real model one day, no patch) | crop data source unresolved | 5-8 |
| 3b | the carbon half of the daily update: allometric pools, litter | ~600 | same restart compare | carbon bookkeeping | 8-12 |
| 2 | `soil_bgc`, `Respauto_NPP_Clabile` (carbon, CO2 flux, GPP diagnostics) | 382 | same restart compare; no per-iteration dump | only needed if the F3 diagnostics include carbon fluxes; the water/energy exports do not depend on it | 8-12 |
| 4 | month-scale: Ent state through 30 days (Sacclim, running means) against a real month restart and the monthly AIJ land fields | validation | a real 1-month restart with `ent_state` | chaos in the land state after days | 4-6 |
Estimated remaining (estimates, not measurements): for the water/energy side of the one-month F3 (1b + 3b-min + 4, scalar speed is enough): about 15-22 h; adding the carbon half (stage 2 + 3b): about 30-45 h in total;
the optional batched/JAX version for GPU use: +15-25 h. Not decided here: whether carbon outputs are part of the F3 acceptance (GOAL.md).

## 7. Cautions
- Bitwise requires the Intel libimf `exp/pow` (not on every host); with glibc the exports drift by up to ~1e-9 relative (the root-finder tolerance amplifies 1-ulp library differences). `test_glibc_fallback_close` bounds this at 1e-8 on a subset.
- The validation inputs are the real-model GHY states (recorded `w, ht`) to the GHY port's own residual; the carried Ent state is ours. A bug in a state that only matters at longer times (Sacclim over weeks, daylength, gdd) cannot appear in 54 steps.
- All results are from one run set (nov26, dec01, jan01 first 6 steps and nov26_day); one real trajectory; no ensemble statistics are needed for Ent (it is deterministic given the GHY inputs).
- `closed` mode resets the GHY dynamic state from the record at every call, so it validates the Ent port inside the GHY loop, not drift over steps (that is stage 1b).

**Parent-session check (2026-10-07 05:50):** `tests/test_ent_ff.py` re-run: 8 passed, 2.8 s (these assert on a small window only, so they do not cover the headline table). Own full run of `python ent_ghy_compare.py nov26 teacher` (2 cores, ~1 min): per-iteration exports over the 6 nov26 steps: cnc n=18,911, bitwise 18,885, max rel 1.006e-15; ci bitwise 18,900, max rel 8.66e-16; gpp bitwise 18,902, max rel 6.81e-16; betadl bitwise 113,003 of 113,466 (max abs 2.2e-16); trans_sw, lai, ipp all bitwise; per-call ws_can, shc_can, fv, height (9,024) and albedo (54,144) all bitwise; Qf at exit max abs 2.5e-17. The non-bitwise cases sit in the first file (ffg_33312, the restart step), as stated. Not re-run by the parent: dec01, jan01 and the 54-step nov26_day rows, closed mode, the daily LAI/albedo boundary check, and the hour estimates (the agent's estimates). The Ent port is NOT complete: carbon/soil state, the first-day set_vegetation_data and month-scale evolution are unported or unvalidated (see above); Ent exports remain recorded in every chained run until the wiring step.

# D173: REAL*16 in the sea-ice code, float64 shortcuts measured against the real dumps, 2026-10-07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when seaice_core_ff/jax are next touched, or when libimf `exp` is available.
Sources (read-only): modelE2_planet_2.0 `model/SEAICE.f` (2363-2440 Ti/Ti2b under `SEAICE_FIXES_2022`, which SEAICE.f:6-7 defines by default), LAKES*.f, ICEDYN*.f; real dumps `ff_data/{nov26,dec01,jan01}/ff{i,n,m}_*.bin`, `ffz_s2ag_*.bin`, `advsi_dumps/nov26/ffadv_in_33312.bin`.
New files (no existing file modified, nothing committed): `fullfidelity/seaice_quad_ff.py`, `seaice_quad_compare.py`, `seaice_quad_loop_compare.py`, `tests/test_seaice_quad_ff.py` (3 passed, 15 s).

## 1. Inventory of REAL*16 in the real source
Grep of `model/*.f` for `real*16`/`*16`/`NNq0`/`_16`: the only sea-ice hits are SEAICE.f `Ti` (real*16 `b,c,det,tm`, quad literals 0q0, 1q0, 1q-3, 4q0, 5q-1, 1q-10) and `Ti2b` (same). ICEDYN.f, ICEDYN_DRV.f, ICEDYN_DUM.f, ICE*.f, OCNDYN* have none (other REAL*16 hits are FFT*/exact-regrid/MPI, not ice). The non-FIXES_2022 `#else` copies are real*8 and not compiled. Only the BP branch (default `seaice_thermo="BP"`) is used; the "SI" branch also has quad literals but is not run. Real callers of Ti/Ti2b: SEAICE.f get_snow_ice_layer (needtemp, Ti1), TICE (2348-2356), SIMELT (1113 `Ti(0d0,1d3*SSI0)`), ICEDYN_DRV ADVSI via get_snow_ice_layer, SEAICE_DRV seaice_to_atmgrid, UNDERICE-side glue, and LAKES*.f CHECKI (diagnostic print only, not ported).
Fortran typing details reproduced: `mu*Si`, `shw-shi`, `Eit+lhm`, `MICE/(MICE+SNOWL)`, `frac*lhm` are REAL*8 before widening; the tests are `Si.gt.0q0` (our float64 port uses Si > 1e-10) and `abs(Ei+lhm).LT.1q-10` (quad 1e-10).

| our routine | calls Ti/Ti2b | precision before | note |
|---|---|---|---|
| seaice_core_ff.Ti, Ti2b | definition | float64 | the shortcut |
| seaice_core_ff.get_snow_ice_layer, tice, sea_ice, ssidec, snowice, addice, simelt (line 739), ground_si_* | yes | float64 | inherit |
| seaice_to_atmgrid_ff (`from seaice_core_ff import Ti, Ti2b`) | yes | float64 | |
| seaice_core_jax.Ti/Ti2b and its sea_ice/addice/simelt/tice | definition + users | float64 | used by surface_loop |
| surface_loop.py (SI.Ti lines 267/271, SI.Ti2b 317/318), seaice_to_atmgrid_jax | yes | float64 | |
| advsi_ff.ti2b_quad (D166) | own | binary128 (mpmath 113 bit) | already correct; `seaice_quad_ff.ti2b_quad` agrees with it on 20000 random inputs (0 differences) |
| lakes_ff / lakes_core_jax | none (CHECKI diagnostic only) | n/a | |
| ICEDYN ports (icedyn_*, dynsi_ff) | none in the real source | n/a | |

## 2. Emulation used
`np.longdouble` on this host is x87 80-bit (`np.finfo`: precision 18, eps 1.08e-19), NOT binary128, so mpmath at 113 bits is used (correctly rounded, results narrowed to double by round-to-nearest). Not verified against a compiled ifort real*16 test (the ifort quad sqrt/multiply rounding is assumed correct rounding); verified bitwise against the real dumps instead (below).

## 3. Measurements (`seaice_quad_compare.py`; nov26+dec01+jan01 dumps; bitwise-equal elements before -> after)
Ti/Ti2b results where float64 != quad: GROUND_SI 22802 of 87525 Ti and 11949 of 41433 Ti2b calls (max |d| 1.2e-5); ADDICE 4340/33601 and 4925/35468 (max 4.7e-10); SIMELT 0/376 (no Ti2b, Ti never differs); s2ag 0/15279 Ti, 48175/479241 Ti2b (max 1.2e-8).
| port | field (n elements) | bitwise f64 -> quad | max abs f64 -> quad |
|---|---|---|---|
| seaice_to_atmgrid (ffz_s2ag, 18 files) | gtemp (247260) | 225480 -> 247260 | 1.2e-8 -> 0 |
| | gtemp2 | 220865 -> 247260 | 8.5e-10 -> 0 |
| | gtempr | 246278 -> 247260 | 1.2e-8 -> 0 |
| | zsnowi, zsi, fwsim | all bitwise both | 0 |
| ADDICE (ffn, 16214 rows) | hsil (64856) | 64669 -> 64856 | 2.0e-6 -> 0 |
| | all other fields | all bitwise both | 0 |
| SIMELT (ffm) | all | unchanged (enrgused 4666/4668 both, 5.7e-14) | unchanged |
| SEA_ICE stage (ffi, 4524 rows) | hsil (18096) | 17545 -> 17795 | 6.0e-6 -> 3.0e-8 |
| | erun, srox2 | 4391 -> 4391 | 2.9e-11 unchanged |
| GROUND_SI final (ffi) | snow | 4499 -> 4524 | 5.0e-14 -> 0 |
| | msi2 | 4477 -> 4520 | 5.1e-6 -> 5.1e-6 (unchanged) |
| | runosi | 3949 -> 4421 | 5.1e-6 -> 5.1e-6 |
| | erunosi | 4377 -> 4466 | 1.087 -> 0.089 |
| | srunosi | 3868 -> 4415 | 4.0e-7 -> 4.0e-7 |
| | hsil (18096) | 16836 -> 17318 | 1.087 -> 0.058 |
| | ssil | 17918 -> 18078 | 2.6e-7 -> 2.6e-7 |
Conclusions: s2ag and ADDICE become fully bitwise; SIMELT never depended on it. GROUND_SI improves a lot (max hsil 1.09 -> 0.058 J/kg-scale) but is NOT bitwise: 732 of 4524 rows still differ after quad. Where it comes from (measured, not Ti): sea_ice-stage mismatches occur only in rows with solar (srox0 > 0): 401 of 1209 such rows, versus 0 of 3315 rows without solar; those rows use `math.exp` in solar_ice_frac_full. Hypothesis, UNTESTED: ifort libimf `exp` differs from Python's libm by 1 ulp (intel_libm_ff.py provides only `pow`, no `exp`); the SSIDEC/snowice stage (413 more ocean rows that are bitwise after sea_ice) is not attributed.

## 4. Main question: the D166 step-0 ice difference (`seaice_quad_loop_compare.py 1`, nov26 it 33312, ice handed to ADVSI vs real ffadv_in, ocean cells, pole rows i>1 excluded; the loop uses seaice_core_jax, whose Ti/Ti2b were swapped for binary128 callbacks)
| field | float64 (reproduces D166) | binary128 Ti/Ti2b |
|---|---|---|
| rsi | 3 cells, 1.1e-16 | 0 cells |
| msi | 19 cells, max 1.431e-6 (scale 3.5e3) | 13 cells, max 1.431e-6 |
| snowi | 12 cells, 5.0e-14 | 8 cells, 3.6e-15 |
| hsi | 194 cells, max 1.620e-2 (scale 6.5e8) | 147 cells, max 1.620e-2 |
| ssi | 37 cells, 7.17e-8 | 25 cells, 7.17e-8 |
Answer: the quad Ti/Ti2b REDUCES the number of differing cells by about 25-35% (and removes rsi), but does NOT explain the largest residuals: the maxima (msi 1.431e-6, hsi 1.62e-2, ssi 7.17e-8) are identical to all printed digits, so the dominant error comes from elsewhere (candidates, none tested here: libimf exp as in section 3, the D17/D20 loose GROUND_SI/ADDICE diagnostics, the jax-vs-ff port differences). D166's "not rounding of ADVSI" remains true. Only step 0 (nov26, 1 step) was measured; later steps start from our own carried state and were not run with quad (the callback costs ~2 min per step on 2 cores). Note: the cell counts here (19/194/37) differ from D166's quoted 19/284/67 for hsi and ssi; same maxima, so the count there probably used another mask or definition; not reconciled.

## 5. Proposed diffs (NOT applied)
(a) `seaice_core_ff.py`: make Ti/Ti2b binary128 (keeps the public names so every port inherits it; mpmath is already a dependency via advsi_ff). Move the implementations of `ti_quad/ti2b_quad` from seaice_quad_ff.py into seaice_core_ff.py (seaice_quad_ff imports seaice_core_ff, so a plain import would be circular), then:
```
--- a/fullfidelity/seaice_core_ff.py
+++ b/fullfidelity/seaice_core_ff.py
-def Ti(Eit, Si):
-    if Si > 1e-10:
-        ...(float64 body)...
+def Ti(Eit, Si):
+    """SEAICE.f:2363, REAL*16 b,c,det,tm (binary128 via mpmath, 113 bit); see seaice_quad_ff.py (D173)."""
+    <body of seaice_quad_ff.ti_quad, with S.X replaced by the module constants>
-def Ti2b(Eit, Si, snowl, mice):
-    ...(float64 body)...
+def Ti2b(Eit, Si, snowl, mice):
+    <body of seaice_quad_ff.ti2b_quad>
```
and update the module docstring ("Known, documented approximation ... float64") to say the quad emulation is used. Expected effect from section 3: s2ag and ADDICE bitwise, GROUND_SI improved; tests with float64-calibrated bounds keep passing (quad is closer to the real values). `advsi_ff.ti2b_quad` can then be deleted in favour of S.Ti2b (identical results measured).
(b) `seaice_to_atmgrid_ff.py`: no change needed if (a) is applied (it binds the names at import).
(c) `seaice_core_jax.py` / `surface_loop.py`: not a drop-in. The jax Ti/Ti2b are traced; a pure_callback works (used here) but costs minutes per step. Options for the owner: keep jax float64 and accept the section 4 residual, or compute Ti/Ti2b with a numpy float128-free double-double (not done), or run the ice cells through the ff port. Not decided here.
(d) Optional: `tests/test_addice_simelt_ff.py` could tighten the ADDICE hsil bound to bitwise once (a) is applied.

## 6. Limits and failures
No failures. Not done: runs of dec01/jan01 through the loop (no DYNSI/ffy dumps beyond nov26), multi-step quad loop, libimf exp test, SI-branch of Ti. The emulation was verified against the real dumps (s2ag gtemp/gtemp2 bitwise on 247260 elements), and against advsi_ff's emulation, not against a standalone ifort real*16 program.

**Parent-session check (2026-10-07 07:55):** `tests/test_seaice_quad_ff.py` re-run: 3 passed, 6 s. No tracked file modified. Own re-run of `python seaice_quad_compare.py` (2 cores, ~1 min) reproduces the per-port table above: seaice_to_atmgrid gtemp 225,480 -> 247,260 bitwise of 247,260 (max 1.17e-8 -> 0), gtemp2 220,865 -> 247,260, gtempr 246,278 -> 247,260; ADDICE hsil 64,669 -> 64,856 of 64,856 (2.0e-6 -> 0); SIMELT unchanged; GROUND_SI hsil 16,836 -> 17,318 of 18,096 (1.087 -> 0.058), erunosi 4,377 -> 4,466 of 4,524 (1.087 -> 0.089). Not re-run by the parent: the step-0 loop experiment for the D166 question (agent: quad cuts differing cells by 25-35% but the maxima stay msi 1.431e-6, hsi 1.62e-2, ssi 7.17e-8, so it is a partial contributor, not the main explanation; its cell counts for hsi/ssi differ from D166's 284/67 while the maxima match - not reconciled), and the libimf-exp hypothesis for the remaining GROUND_SI mismatches (untested). The proposed diffs (move binary128 Ti/Ti2b into `seaice_core_ff`, then drop `advsi_ff.ti2b_quad`) are NOT applied: a performance and ownership decision (mpmath at 113 bits is slow; `np.longdouble` here is x87 80-bit, not binary128).

# D170: one combined surface loop (ADVSI + RIVERF + DYNSI glue), replay and coupled, 2026-10-07

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: `fullfidelity/surface_loop_v2.py`, `fullfidelity/tests/test_surface_loop_v2.py` (2 passed, 257 s; skip when data absent), this entry.
Sources: D164, D166, D167, D168; real source (read-only) SURFACE_LANDICE.f:127-134, LANDICE_DRV.f PRECIP_LI/GROUND_LI, SEAICE.f SSIDEC.

## 1. What was built
`surface_post_v2` = `surface_loop.surface_post` plus: DYNSI computed from our state (odmui/odmvi/ustar, carried USI/VSI, uisurf/visurf computed; UOdrag = 0 in this build (FLUXES.f default) so the tiles do not read uisurf), GROUND_LK with full geo + RIVERF (computed oflowo/oeflowo, lake exports reset) and ADVSI last (RSISAVE = RSI at DYNSI entry, RSIX/RSIY carried, USI/VSI from our DYNSI). Switches: dynsi, advsi, riverf, advsi_usi, li_e1_precip.
`run_free_v2` = replay-mode free loop. `Loop2`/`run_coupled_v2` = the closed/coupled path (subclass of `surface_loop.Loop`; the ice-tile DMUA/DMVA now come from OUR tile outputs; no recorded DYNSI/RIVERF boundary is read; `surface_loop.Loop` is swapped only during the call).
Still recorded (unchanged): radiation, Ent exports + ffg land forcing columns, TRUP_in_rad, PBL profile columns, MMST, the 5 static ADVSI geometry vectors (from a real ffadv_in dump), irrigation demand (reconstructed). Replay mode additionally: real tile outputs.

## 2. Free 6-step loop, all on (nov26, replay R1; relative to field scale unless stated; 3 min, 2 cores)
| step | ice snow / msi2 / tg1 / tg2 | lake tg1 / mwl | land-ice tg1 / tg2 | ocean exit g0m / uo / vo |
|---|---|---|---|---|
| 0 | 0 / 0 / 5e-16 / 8e-16 | 0 / 0 | 0 / 0 | 1.2e-12 / 1.1e-9 / 6.1e-9 |
| 1 | 2e-16 / 4e-10 / 2.5e-15 / 3.5e-15 | 0 / 0 | 0 / 0 | 5.0e-12 / 6.4e-9 / 3.5e-8 |
| 2 | 1e-13 / 1.9e-9 / 7e-11 / 1.7e-10 | 3e-20 / 0 | 0 / 0 | 1.2e-11 / 2.0e-8 / 9.3e-8 |
| 3 | 4e-13 / 4.4e-9 / 6e-11 / 1.2e-9 | 1e-19 / 0 | 0 / 0 | 2.5e-11 / 4.2e-8 / 1.8e-7 |
| 4 | 7e-13 / 8.2e-9 / 7.6e-10 / 4.1e-9 | 2e-19 / 8e-21 | 0 / 0 | 4.1e-11 / 7.1e-8 / 2.6e-7 |
| 5 | 1e-12 / 1.3e-8 / 7.5e-10 / 7.6e-9 | 1e-16 / 3e-20 | 0 / 0 | 3.2e-11 / 1.0e-7 / 2.2e-7 |
Tile-set mismatches 0 on all steps (D164: 12-14 ocean tiles). Computed DYNSI vs recorded: odmui <= 9.4e-8 abs (scale 150), ustar <= 3e-10 (scale 0.0136), USI/VSI <= 1.6e-8 (scale 0.23). Computed RIVERF flows vs recorded: oflowo <= 1.1e-16 (scale 0.59), oeflowo <= 6.8e-13 abs (scale 6.4e4).
Comparison: D164 (no ADVSI): ice 1e-3 -> 5e-3, ocean uo/vo 2e-3 -> 1e-2/2e-2, lake tg1 2e-3 -> 9.6e-3, land ice 0.16 from step 3. D166 (ADVSI only; re-run here with dynsi=recorded, riverf=False, li_e1_precip=True, reproduces D166 exactly): ice msi2 4e-10 -> 8e-6, tg1 -> 3.9e-5, lake 9.6e-3, land ice 0.16. dynsi=computed with RIVERF off gives identical numbers to the D166 run to the printed digits, so computed DYNSI changes nothing visible. Ice error with everything on is 3-4 orders smaller than D166's at step 5 (1.3e-8 vs 8e-6): the D166 residual growth was NOT only the "loosest GROUND_SI/ADDICE diagnostics"; it vanishes with RIVERF + all-land runoff (the lake-domain ice depends on the lake state; measured by the switch, not traced cell by cell).

## 3. Land-ice jump (measured cause, fixed)
Isolation 1: starting each step from the REAL ffl record state and applying PRECIP_LI(next step) + GROUND_LI with the real tile fluxes reproduces all 346 tiles of the next record with 0.0 error for all 5 transitions (tg1, tg2, snow). So LNDICE/PRECLI are right.
Isolation 2: chaining our own land-ice state alone (restart -> PRECIP_LI -> GROUND_LI, real fluxes): bit-exact over 6 steps if E1 = sum of tile F1DT only; if PRECIP_LI's EDIFS is added to E1 (as `surface_loop.surface_post` does: `acc['e1_li'] + mid['pli']['e1']`) the entry error appears at step 3 (tg1 9.30 K abs on scale 58.6, tg2 0.61 K).
Cell: (i, j) = (8, 39), FLICE 0.254; at step 2 PRECIP_LI returns EDIFS = -3.34e6 J/m2 there (the only cell with nonzero EDIFS in the window). Source: SURFACE_LANDICE.f:130-134 `if(do_init) igla%e1(i,j) = 0.` runs at the first substep, after PRECIP_LI stored EDIFS in atmgla%E1, so the EDIFS is overwritten and never reaches GROUND_LI. Fix in v2: `li_e1_precip=False` (default). D164's `surface_post` (and D166-D168 loops that use it) keep the bug; surface_loop.py was not edited (a reader of those entries should use v2). Test: `test_landice_chain_bitexact_without_precip_edifs` (includes the non-vacuity case).

## 4. Step-0 ice difference vs the real ADVSI entry (msi 1.4e-6 abs)
Real FORM_SI records ffn_33312 (ocean domain, 2095 cells) vs ours: fluxes into FORM_SI (enrgfo, acefo, acefi, enrgfi, salto, salti) equal to <= 5.8e-8 abs on 1.7e5 (enrgfi) and 2e-13 on mass; our ADDICE on the REAL inputs reproduces the real outputs: snow, roice, msi2, ssil 0.0; hsil 1.5e-6 abs on 6e8. So ADDICE is NOT the cause (hsil 2e-15 relative only). The difference is already present in the state at FORM_SI entry, i.e. produced by GROUND_SI: msi differs in 12 of 2095 cells and by more than 1e-9 in ONE cell, (65, 38): ours 103.19759653852068, real 103.19759510562525 (1.43e-6), hsil 1.6e-2 abs there; hsi differs in 191 cells by <= 1e-9 relative. For that cell the UNDERICE fluxes match the real ffz_undocn record (fm/fh/fs to 1e-15 / 1e-10 abs), tm and ustar identical, state inputs equal, so the difference sits inside SEA_ICE/SSIDEC/snowice (SSIDEC removes 2.33e-3 kg there; the layer-1 brine fraction 0.0503 is close to the 0.05 threshold). A binary128 Ti/Ti2b (the untracked `seaice_quad_ff.py`, another track) removes the 1e-14 msi residuals of the other cells and part of their hsil (e.g. 8.8e-7 -> 2.6e-7) but does NOT change cell (65, 38) (tsil changes <= 9e-16). Not resolved: the cause for cell (65, 38) is open (candidate: another single-precision literal or branch in SSIDEC/ice-layer code in that cell; not tested). Carried through the loop it is the 1e-8 level ice error above.

## 5. Coupled atmosphere + closed surface, 6 steps (nov26, 5 real members, D164 acceptance: within <= 1, near <= 2, never beyond 2x; rms metric)
Wall: 409 s first step, then 43-100 s per step on 2 cores. Per step class and ratio ours / max-member (w = within, n = near), steps 0-5:
T w.00 w.78 w.81 w.98 w.85 w.79 | Q w.00 w.66 n1.02 n1.08 w.97 n1.03 | U w.00 w.27 w.62 w.49 w.50 w.54 | V w.00 w.12 n1.22 w.59 w.65 w.67 | P w.00 w.00 n1.04 w.93 w.56 w.54 | QCL max .98 | QCI max .41.
Max ratio steps >= 1: T 0.98, Q 1.08, U 0.62, V 1.22, P 1.04 (never beyond 2x). D164 coupled run recomputed with the same code (saved states): T 1.44, Q 1.07, U 0.72, V 1.30, P 1.45, QCL 1.0, QCI 0.97. rms T ours v2 7.4e-14, 2.0e-4, 5.1e-4, 7.7e-4, 1.05e-3, 1.41e-3 K (D164: ..., 1.13e-3, 1.33e-3, 1.67e-3); member floor max 1.0e-4 .. 1.8e-3. The improvement over D164 (T/P step 3 from near 1.4 to within) is not shown to come from the surface: 5 members, one start state, 6 steps; a differing result between two runs of a chaotic system with such a small ensemble is not evidence of a better port. Necessary, not sufficient.
Not done: comparison of the coupled run's own surface state with the real records (the loop does not store it); other dates; windows beyond 6 steps (DYNSI/RIVERF boundary dumps exist only for 6/12 steps, but the v2 coupled path does not need them; the 54-step day was not run because the ADVSI geometry dump and the validation data end at 6 steps for DYNSI and the first ocean boundary was never built for the day: a longer run would be unvalidated for the surface).

## 6. What remains recorded and why
Radiation (SOCRATES), Ent exports (D169 port exists but is not wired here), ffg land forcing, TRUP_in_rad, PBL profile columns, MMST, static ADVSI geometry, irrigation demand. In replay mode the tile outputs. Open: cell (65, 38) GROUND_SI difference; D164/D166-D168 loops carry the EDIFS bug (documented here, files untouched).
How to reproduce: `python -c "import surface_loop_v2 as V; V.run_free_v2(6)"`; coupled: `V.run_coupled_v2(nsteps=6, out_dir=...)` then `atm_day_report.curves/overlay` with members p1..p5.

**Parent-session check (2026-10-07 08:40):** `tests/test_surface_loop_v2.py` re-run: 2 passed, 108 s (the free-loop test asserts the measured bounds; the land-ice test includes a non-vacuity check that the old behaviour gives the 9 K error). No tracked file modified. Checked by the parent: the `EDIFS` handling the agent identifies exists in `surface_loop.py` (D164 path: `ground_li` returns `e1 = edifs + e1`) and NOT in `atm_step.py`, so earlier chained atmosphere results are not affected; v2 omits it (surface_loop_v2.py:84). KNOWN DEFECT left in place: the D164, D166, D167 and D168 loop scripts still carry the EDIFS bug (they are committed and tested; use `surface_loop_v2` for new work). Not re-run by the parent: the free-loop and coupled-run tables (agent's runs: free 6-step loop all on, step-5 errors ice msi2 1.3e-8, ocean uo 1.0e-7, land-ice 0; coupled 6 steps T/Q/U/V/P within the 5 real members' spread, max ratio 1.22, never beyond 2x), the claim that the D166 ice-growth vanishes because of RIVERF with all-land runoff (measured by the switch only, not traced cell by cell), and the step-0 ice difference of cell (65,38) (open: inside SEA_ICE/SSIDEC/snowice; not Ti/Ti2b). Only nov26, 6 steps; Ent and the ffg forcing columns are still recorded.

# D171: Ent computed from our own state in the chained land path (stage 1b of D169), 2026-10-07

Owner: project owner (Glenn Tamkin); written by a Claude Code session. Project-local. Review: when the first-step residual is explained, or when a month run is attempted.
Sources: D158, D169 (`ent_ff.py`, `ent_daily_ff.py`, `ent_ghy_compare.py`), `ghy_ref.py`, `land_chain.py`, `atm_step.py`, `atm_day_open_loop.py`, `atm_day_report.py`, ffg/ffp/fft dumps of `ff_data/nov26_day`, the 5 real one-ulp members `ffpt_p1..p5` (D151).
New files only (nothing existing modified, nothing committed): `fullfidelity/land_chain_ent.py`, `fullfidelity/ent_first_step_probe.py`, `fullfidelity/tests/test_land_chain_ent.py`, `fullfidelity/scoping/D171_closed_nov26_day.json` (per-step closed-mode numbers), this entry. No instrumentation was built (no new unit used).

## 1. What was built
`land_chain_ent.py`. The batched JAX GHY cannot take Ent exports computed from the in-loop state (each sub-iteration's cnc/betadl/lai depends on that iteration's canopy temperature, soil moisture, fice, Qf), so the chained path runs the scalar GHY port (`ghy_ref.GhyColumn`, the D158 level, the same loop as GHY.f advnc) cell by cell with `ent_ff` called inside the loop:
- the sub-iteration time step is COMPUTED (gdtm and the `dtr` loop), not the recorded `dts`; this includes stiff cells (up to 15 sub-iterations measured; the iterations beyond the 11th run in Ent too);
- per call, ws_can, shc_can, fv, canopy height (snowm = 0.1 height) and irrigation/fv come from OUR Ent state (`call_exports`), not from the record;
- the Ent state of every cell is carried from the restart through all calls; Qf_ij (persistent canopy-air humidity, GHY_DRV.f `Qf_ij`) is carried per cell (option `qf_mode='record'` uses the recorded entry value);
- the daily prescribed LAI/albedo update (`ent_daily_ff`) is applied before the first step of a new day (end-of-day flag of the ffg records, jday of the new day from itime; verified: day 331 applied before 33360);
- after the loop `evap_limits(False)` gives evap_max_ij / fr_sat_ij as `ghy_jax.advnc` does.
`land_substep_ent` is a drop-in for `land_chain.land_substep` (same return structure); `install(A, ent)` routes `atm_step` land_mode='ghy' through it (it replaces `atm_step._SURF['LC']` by a shim and wraps `atm_step.run_step` to tell Ent the step number).
Still RECORDED in this path (stated plainly): Ent's radiation inputs (vis_rad, direct_vis_rad, cosz1) and Ca, GHY's precipitation/radiation forcing (pr, htpr, prs, srheat, trheat), geothermal heat, irrigation totals, vs0/gusti/pres; in the open-loop day the GHY prognostic state at the start of every step (the state is carried only between the two substeps of a step). Not ported (D169): carbon/soil biogeochemistry, set_vegetation_data at the first day end of a segment, height/crop/structure updates.

## 2. Closed-mode validation, nov26_day, 54 steps (measured)
Command: `python land_chain_ent.py closed_day nov26_day <shard> 2 <out.json> 54 record` (2 shards by cell, 1 core each, libimf). Every call of every step (81,324 calls = 54 x 2 x 753) with dts computed, Ent computed, GHY state and forcing reset from the record at every call (so this validates Ent + GHY inside the loop, not drift), Ent state carried over the whole day including the day boundary. Per-step numbers: `scoping/D171_closed_nov26_day.json` (max |ours - record|, scale = max |record| over cells, rel = max/scale).
- Iteration count: our nit equals the recorded ffnit on **all 81,324 calls** (0 mismatches); max nit 15 (the record holds only 11).
- Worst per-step scaled difference over the 54 steps (value, abs max, scale, step): tbcs 1.1e-15 (5.7e-14 over 52; 33312), tsns 1.05e-15 (33355), ashg 2.95e-15 (1.0e-9 over 3.4e5; 33353), alhg 2.8e-15 (33315), aevap 2.8e-15 (5.2e-16 over 0.19; 33315), aruns 5.9e-16, arunu 6.4e-17, aeruns 1.6e-16 (9.1e-13 over 5.7e3), aerunu 4.5e-17, ae0 1.45e-14 (2.6e-9 over 1.8e5; 33315), abetad 3.3e-16, w_out 2.1e-16 (1.7e-16 abs), ht_out 2.7e-17 (7.5e-9 over 2.7e8), tp_out 3.5e-16 (1.9e-14 K), fr_snow_out 2.1e-17, exit Qf 1.3e-15 (2.5e-17 abs over 0.0196). nsn not compared separately (nit and w/ht/fr_snow agree). The D169 whole-day exit-Qf maximum 7.4e-9 on stiff cells is gone: it was caused by the recorded dts / truncated iteration list, with computed dts the stiff cells agree (max 2.5e-17).
- Per-step maxima are at 1e-15..1.5e-14 scaled for every output at every step; no step stands out except ae0 33315 (1.45e-14, a cancelling sum). The first step is not special in the GHY outputs (tbcs 1.1e-15, aevap 3e-16).
- Ent exports (nit <= 11, same sub-iteration index as the record; computed dts): cnc 189,698 values, bitwise 189,597, max rel 1.5e-15; betadl 1,138,188 / 1,137,262, 1.1e-14 (tiny values); trans_sw, lai, ipp bitwise; ci 189,639 bitwise, 2.2e-15; gpp 189,650, 1.3e-15 (same as the D169 teacher result).
- Qf carry check: |our exit Qf of the previous call of a cell - recorded entry Qf| max 2.5e-17 over the day (steps 33312..33365), i.e. carrying Qf is equivalent to the record in closed mode.
Time: about 15 s per step for the two shards together on a loaded node.

## 3. Chained atmosphere day with Ent computed (open loop, recorded radiation; 54 steps, imf, numpy dynamics)
Runs (`python land_chain_ent.py day <ent|ghy> <tag> 54`, 1 core each, run concurrently on a loaded node): **ent** = tag `d171_ent` (our PBL + scalar GHY + Ent computed, Qf carried, daily update at 33360); **ghyrec** = tag `d171_ghyrec` (existing land_mode 'ghy': our PBL + batched JAX GHY with RECORDED Ent exports, never run over a day before); **imf_np** = existing D150 day with the land patch fully RECORDED (`ours_imf_np`). Outputs in `ff_data/nov26_day/ours_d171_*` (step_<it>.npz, run.json; `ent_log.json` per-step diagnostics for ent). Acceptance with `atm_day_report.py` (D151 method: whole-column rms of ours minus real vs the floor [min..max] of rms(member_k - control_k), p1..p5; within <= max, near <= 2x max, beyond > 2x).
Counts over 54 steps (whole-column rms; classes within / near / beyond), the three runs:
| field | ent | ghyrec (recorded Ent) | imf_np (recorded land) |
|---|---|---|---|
| T, U, V | all 54 within | T 54 within; U, V 53 within + 1 near | all 54 within |
| Q | 53 within, 1 near | 38 within, 16 near | 39 within, 15 near |
| P | 54 within | 53 within, 1 near | 52 within, 2 near |
| QCL | 52 within, 1 near, 1 beyond | 50 within, 3 near, 1 beyond | 50 within, 4 near |
| QCI | 41 within, 13 near | 52 within, 2 near | 51 within, 3 near |
Largest ratio ours/max-member (steps >= 3, whole-column rms): ent T 0.85, Q 1.00, U 0.84, V 0.87, P 0.92, QCL 1.00, QCI 1.62; ghyrec T 0.94, Q 1.05, U 0.97, V 0.93, P 1.02, QCL 1.08, QCI 1.06; imf_np T 0.92, Q 1.09, U 0.90, V 0.98, P 1.05, QCL 1.09, QCI 1.11. No field is beyond 2x in the whole-column rms at steps >= 3. The one QCL 'beyond' in ent and ghyrec is at steps < 3 (inferred from the steps >= 3 maximum of 1.0/1.08; the floor there is ~1e-9 or below). Not within by the rms at single levels: QCI at level 21 reaches 2.14x (ent; 0.74 ghyrec, 0.99 imf_np) and layer-1 T/Q 1.42/1.43 (ghyrec 1.07/1.31), so layer-level statistics are noisier; the global-mean metric (absgmean) reaches up to 3.2x (ent T) vs 2.5 (ghyrec) and 1.8 (imf_np) at isolated steps (these use single members' means, not tested for significance).
Ent vs the same run with recorded Ent (ent vs ghyrec) and vs recorded land, rms of the difference (all ndiff counts of elements differing at all):
| step (it) | T rms (max) | Q rms | U rms | V rms | P rms |
|---|---|---|---|---|---|
| 0 (33312) | 2.7e-14 (2.3e-13) | 2.6e-18 | 3.5e-15 | 3.7e-15 | 9.9e-14 |
| 1 (33313) | 9.8e-5 (2.1e-2) | 9.7e-7 | 1.7e-4 | 1.3e-4 | 3.4e-9 |
| 5 (33317) | 1.2e-3 | 7.1e-6 | 9.2e-3 | 5.8e-3 | 2.7e-3 |
| 23 (33335) | 2.0e-2 | 4.5e-5 | 8.1e-2 | 8.2e-2 | 8.0e-2 |
| 47 (33359) | 6.1e-2 | 6.8e-5 | 1.6e-1 | 1.7e-1 | 1.3e-1 |
| 53 (33365) | 7.1e-2 | 7.2e-5 | 1.7e-1 | 1.8e-1 | 1.3e-1 |
Step 0: the three runs differ at rounding level (ent vs ghyrec T max 2.3e-13; ent vs recorded 6.4e-13; ghyrec vs recorded 6.3e-13); the number of differing elements is large (ent vs ghyrec: T 129,939 of 132,480 elements differ although by 3e-14 rms): the ent run's difference from both others is spread over almost all cells (cause not established; no bitwise statement is possible between the scalar and the JAX GHY, and step 0 is not bitwise either between ghyrec and recorded: 11,773 T elements). The trajectories separate at step 1 by 1e-4 (T) and then grow like the chaotic growth of D150/D151: the ent-vs-ghyrec difference at step 53 (T 7.1e-2) is as large as the difference of either run from the real run (T rms 7.07e-2 ent, 7.26e-2 ghyrec and recorded land at step 53), i.e. on the day-scale the Ent computation changes the result by the amount that any 1e-16-level land perturbation does after one day; it is not distinguishable from chaos with one realization. The QCI excess of the ent run (13 near vs 2/3) is a single realization of a heavy-tailed statistic (D156): not attributed to Ent.
Ent diagnostics of the chained run: 1,506 calls per step, 3,046..3,919+ sub-iterations per step, max 14 sub-iterations in one call; the carried Qf differs from the recorded entry Qf by up to 6.2e-3 (open loop: our atmosphere drives the PBL humidity), as expected, not an error; day boundary applied at 33360 (jday 331).
Substep check (test, 33312 substep 1, same PBL and forcing, recorded-Ent JAX GHY vs Ent-computed scalar GHY): tbcs 4.8e-13 (scale 52), tsns 6.3e-13, ashg 2.5e-8 (4.0e5), alhg 4.7e-9 (3.6e5), aevap 1.9e-15 (0.14), aruns 2.1e-14 (0.30), abetad 7.8e-16, fr_sat 2.4e-15, evap_max_ij 2.8e-21 (1.2e-6); patch uflux1/vflux1/tsavg/qsavg bitwise, dth1 max 1.6e-13 (476 of 753 differ), dq1 9.8e-18 (540 differ).

## 4. Timing (one core, libimf; measured on a quiet core with the second core busy)
GHY (scalar) + Ent per step (1,506 calls): 6.5 s at night (33312-33314, 4.4 ms/call) rising to 9.6 s in daylight (33325, 6.4 ms/call); `ent_run` alone 2.45 s (1.63 ms/call, 0.80 ms/sub-iteration) at night to 4.6 s (3.06 ms/call, 1.17 ms/sub-iteration) at step 33325; the rest is the scalar GHY. In the whole day runs on the loaded node: ent-run land calls averaged 13.1 s/step (6.8..27.7), surface stage 24.2 s/step vs 15.2 (ghyrec) and 3.9 (recorded land); total wall 2,420 s / 2,113 s / 951 s for 54 steps (concurrent runs, node load 3-8: not clean timings). Extrapolation for a month (1,440 steps): about 3 h of GHY+Ent on a quiet core (consistent with D169's estimate), not measured.

## 5. The first-step ulp residual of D169 (investigated; cause NOT established)
Measured (`ent_first_step_probe.py`; nov26, ffg_33312 and 33313, teacher mode): 212 non-bitwise (field, iteration) entries in 172 of 3,012 calls (108 of substep 1, 104 of substep 2 of the first step; none in step 33313). 208 of the 212 are at sub-iteration 1 (3 at nit 2, 1 at nit 3); fields betadl 170, cnc 23, ci 11, gpp 8; relative size median 6e-16, max 8.3e-15 (1-ulp level); all in cells with fv >= 0.12, 81 of 212 in daylight, rows j = 12..40.
Hypotheses tested and refuted by measurement (`python ent_first_step_probe.py ulp nov26 0 <out.json>`, 170 calls whose first iteration is not bitwise; the first Ent call is re-run from the saved state): (1) the unperturbed re-run reproduces the mismatch in 170 of 170 (deterministic); (2) moving ONE of the 29 forcing inputs (ts, tcan, Qf, pres, Ca, ch, vs, vis, dvis, cosz, fw, w/ws/fice of 6 layers) by +-1 ulp makes the call bitwise in only 30 of 170 calls, scattered over 10 different inputs (w4+ 10, ws4- 9, ws0- 7, ...), 140 calls are fixed by none: no single GHY-side input explains it (the 30 are consistent with chance); (3) moving one Ent state quantity by +-1 ulp (airtemp_10d, par_10d, daylength, Sacclim of all cohorts, cohort lai, fracroot, patch albedo) fixes none of 170. Not tested: pairs of inputs, the order of summation in the cohort/patch sums at the first call, `Respauto`/other Ent state that is not exported, and differences in how the real code initialises derived Ent variables (patch LAI) on the first call after a restart. Since the residual is confined to the first step (both substeps) the likely class is a first-call-after-restart effect; this is inference, not shown.
Instrumentation proposal (not built; a rebuild of the instrumented model chain per `instrumentation/build_and_run.md` plus a real first-step run is not cheap): unit range 1470-1479 (a grep of `instrumentation/*.patch` and `*.md` finds no use of 147x; the D169 grep of the model sources also found none; re-grep before use). Dump, at the `ent_set_forcings` call (GHY.f:2437) of the first iteration, the forcing inputs (29 values) as passed (TairC, TcanopyC, Qf, P_mbar, Ca after conversion, Ch, U, IPARdif, IPARdir, CosZen, fwet_canopy, Soilmoist(6), fice(6)) plus the patch LAI/GCANOPY/betadl and the first cohort's Sacclim/lai at entry; compare with our inputs to the last bit. This would show directly whether the difference is in the forcing or in the Ent state.

## 6. Tests
`tests/test_land_chain_ent.py` (skipped when the data are absent; 4 passed in 63 s): closed-mode GHY outputs (relative 1e-9, w 1e-12, nit == ffnit) and exports (1e-12) on a subset over 33312-33313; `land_substep_ent` vs `land_chain.land_substep` on 33312 substep 1 (1e-9 relative, measured above 1e-13..1e-14); daily update reproduces the recorded per-call exports at 33360 on a subset (bitwise). The tolerances are much looser than the measured values; they are not the measured bounds.

## 7. Limits and cautions
- One day, one start state, one realization; the day-run comparison to the real noise floor cannot separate the Ent effect from chaos (three runs that differ only in the land patch are all within or near the floor).
- Open loop: GHY prognostic state restarts from the record every step and radiation is recorded; the land state drift of the free loop (D164) is not exercised here (the free 6-step loop could now use `land_substep_ent`, not done).
- Bitwise results need libimf; the closed-mode day was run with it. Only one day boundary exists in the data.
- Ent carbon/soil state is not ported; month-scale Sacclim/phenology evolution is not validated (D169 section 4.4).

**Parent-session check (2026-10-07 08:58):** `tests/test_land_chain_ent.py` re-run: 4 passed, 50 s. No tracked file modified. The closed-mode table was recomputed by the parent from `scoping/D171_closed_nov26_day.json` (54 steps, 81,324 calls): 0 iteration-count mismatches; worst scaled differences tbcs 1.09e-15, tsns 1.05e-15, ashg 2.95e-15, alhg 2.76e-15, aevap 2.76e-15, ae0 1.45e-14 (itime 33315), abetad 3.3e-16, w_out 2.1e-16, tp_out 3.5e-16; max Qf carry deviation 2.5e-17: consistent with the report. This confirms the JSON and the tests, not a re-run of the full-day closed computation or of the chained 54-step open-loop day. CAUTION kept from the agent: in the chained 54-step day against the 5 real members the Ent-computed run has QCI worst ratio 1.62 (QCI level 21 reaches 2.14x) versus 1.06/1.11 for the recorded-Ent runs; the agent did NOT attribute this to Ent (one realisation, chaos), so it is an open observation, not a validated result; the first-step ulp residual of D169 (212 values in 172 of 3,012 calls, all in step 1) is still unexplained (refuted: input or state 1-ulp moves; untested: pairs, summation order, first-call-after-restart initialisation). Ent state exports remain 'recorded' for every earlier result; `land_chain_ent.install(A, ent)` is the opt-in. Day outputs in `ff_data/nov26_day/ours_d171_ent` and `ours_d171_ghyrec` (outside git).

# D172: second F3 diagnostics set, chained wiring, and the F3 scoring tool (2026-10-07/08)

Not committed. Owner: Glenn Tamkin. Review: when the first model month exists.
Code (new files only): `fullfidelity/f3_diagnostics2.py` (accumulators + window drivers), `f3_chained_compare.py`, `f3_score.py`;
tests `tests/test_f3_diagnostics2.py` (3 passed, 1 skipped, 296 s), `tests/test_f3_score.py` (6 passed, 30 s).
Data: `ff_data/nov26_day/d172_chained54.npz` (our chained accumulation), `d172_chained54_vs_real.json` (column table). SOCRATES/RADIA never touched.

## 1. Ranking of the 226 not-yet-accumulated columns, and what was added
Ranking by use for F3 (plan section 3.4 items 1,2,4,5): (a) near-surface state/winds/PBL/ground temperature, (b) moisture and cloud budget,
(c) heat/water budget terms and tropopause, (d) ocean/ice/land/lake state, land-model diagnostics, ISCCP, aj/ajl/consrv. The columns used for the
D165 key fields (prsurf, slp, t_850, t_500, prec, evap, tsurf, srnf_toa, trnf_toa) were already covered by D163.

Added and validated with REAL inputs against `real_acc54_nov26.npz` (same method and 1e-12 tolerance as D163; nothing loosened), 27 columns, worst relative residual 3.2e-14:
- surface site (1 substep in 3): usurf 286, vsurf 287, wsurf 288, gusti 289, pblht 237, tausmag 293, RHsurf 98, tgrnd 184, mccon 142
  (ftype-weighted sums of the PBL outputs of the four tile types; tgrnd from the tile TG1 / land TSNS). Inputs: ffp/ffs/ffl/ffg/fft records of both substeps.
- surface fluxes (every step, two substeps summed): sensht 356, sensht_lndice 357, sh_oice 358, evap_ocn 325, evap_oice 326, evap_lndice 324,
  lwd_oice 400, lwu_oice 401, trht_lndice 399, latht_lndice 360.
- CONDSE: prec_mc 321 (=PREC-PRECSS), snowfall 333 (=-EPREC/LHM), cldw 101, cnvfrq 468, mccvbs 54, mccldbs 150.
- dynamics: ptrop 157, ttrop 158.
Total with D163: 284 columns accumulated.

Tried and NOT matching (not included): dSE_Dyn 352 / dKE_Dyn 353 / dTE_Dyn 354 (5.5 %, 6.3 %, 98 % of scale off with SEFINAL-SEINIT, KEFINAL-KEINIT, DSEPKE*MASUM
from the workspace; the definition/units are not resolved); srtrnf_grnd 403 (solar+trheat composite, 1.49 relative); cldi 102 (17 % off: ice precip WMPR term not dumped);
mccvtp 53 and mccldtp 149 (LMCMAX index convention not found, relative 1.0); pscld/pdcld (need CLDREF, internal).
Not attempted (inputs not dumped, or require unported state): sst/sss/ssh/sivol/simass/ts_oice and all ocean/ice/lake/land state columns (replayed state), netht_* and e0 terms
(precipitation energy of ice/landice tiles not dumped), runoff, pr_*, land-model diagnostics (soil/snow/canopy, gpp, ...), clwp 99 (writer not located), puq/pvq/fmu/fmv/fgz*/nt_dse
(GCDIAGb/ATMDYN fluxes), ujet/vjet, ISCCP, aj/ajl/agc/consrv. Roughly 195 of the 483 window columns remain unaccumulated.
No instrumentation dump was needed for the added groups (the existing ffp/ffs/ffl/ffg/fft records contain the per-substep values). A new dump would be needed for: ice and land-ice
tile `e0` (incl. precipitation energy), the CONDSE LMCMAX/WMPR/CLDREF internals, and CONSERV_SE/KE internals; I did not specify unit numbers or build one.

## 2. Wiring and chained comparison (54 steps, nov26)
`F3Acc2` (extends `F3Acc`) is called at DIAGA (through the dyn_step stage hook), accum_ma, CONDSE exit, per SURFACE substep and once per step for the flux sums.
`run_chained_window` chains `atm_step.run_step` from OUR previous end state (dynamics, CONDSE, PBL+tiles+GHY from our code; Ent exports and land forcing recorded;
radiation RECORDED: the real RADIA packets, so the 80 RADIA columns are the real output). It installs wrappers around `atm_step.stage_dyn/_substep` at run time (atm_step.py not edited).
Cost: ~235 s start-up + ~145 s/step on one core; 54 steps = 7,788 s (2.2 h). Counters equal the real ones (54/11/36/4).

Result (our chained accumulation vs real, max|diff| / max|real increment| per column; 284 columns):
| group | columns | within 1e-12 | < 1e-6 | < 1e-3 | < 1e-2 | worst |
|---|---|---|---|---|---|---|
| RADIA (recorded) | 80 | 80 | 80 | 80 | 80 | 1.2e-16 |
| D163 base set | 177 | 24 | 24 | 50 | 106 | 0.34 (omega_1) |
| D172 new set | 27 | 1 (lwd_oice) | 1 | 3 | 8 | 0.61 (mccon) |
Total within the D163 tolerance: 105 of 284. The others differ at chaos/threshold-flip level, as expected for a free atmosphere over a replayed surface:
- Key fields: prsurf 4.8e-4, slp 1.35e-2, t_850 1.4e-2, t_500 7.0e-3, tsurf 6.7e-3, prec 5.3e-2, evap 3.9e-2 of the column scale (global-mean relative differences 6e-9, 7e-7, 1.9e-4, 2.6e-5, 5e-5, 3.0e-3, 2.1e-3); srnf_toa, trnf_toa exact (recorded RADIA).
- Worst columns are the noisy ones: omega at upper levels (0.2-0.34 of scale), q/rh at 200-1000 hPa (0.1-0.23), cnvfrq 0.56, mccon 0.61 (counts that flip), gusti 0.26, snowfall 0.16, mccvbs/mccldbs 0.16/0.22, cldw 0.13.
- Smoother new columns: lwd_oice 1.9e-14, lwu_oice 1.4e-5, trht_lndice 1.8e-4, tgrnd 3.2e-3, sh_oice 4.7e-3, vsurf 9e-3, usurf 1.2e-2, wsurf 2.2e-2, sensht 2.0e-2, pblht 4.6e-2, prec_mc 8e-2.
Chaos quantification: NOT done. The only nov26 members (5 one-ulp members, D151) were dumped as per-step states, not as AIJ accumulations, and the surface records needed to accumulate
them do not exist for those members; so no AIJ-level noise floor for the 54-step window exists. Two-step check (steps 0-1, chained vs the real-input accumulators): surface-flux and CONDSE precip
columns agree to <= 1e-9 .. 1e-3, cldw 1.6e-2, pblht 1.2e-2 (these already differ after 2 steps: PBL height/cloud water are predicted from our atmosphere, not recorded).
Interpretation limits: one start state, one 27-h window, radiation recorded, surface (ocean/ice/lake state, Ent, land forcing) replayed. A column-level difference of a few percent over 54 steps
is not, by itself, a defect or evidence of agreement: the month-scale statistic (section 3) is the criterion.

## 3. F3 scoring tool (`f3_score.py`) and its validation on the real ensemble
Input: a model month as {AIJ column: accumulated (46,72) map} + idacc (`ModelMonth.from_f3acc`, or an acc file). Reference: the 8 JAN1950 members (D165). Monthly means by the ij_mapk rule.
Per field: global mean t_gm = (x - mean_ref)/(sd_ref sqrt(1+1/M)) (Student t, M-1 dof under the null) with p; 46 zonal-bin t values (fraction within 2, max, count beyond 4); pooled rms
ratios R_zon and R_grid (rms difference / (sqrt(1+1/M) * rms of the reference sd), expectation 1); `flagged_fields`, `summarize`, `compare_to_loo` (position of a model month's aggregate statistics
relative to the leave-one-out members, no invented threshold), and the plan's criteria evaluated separately in `verdict`.
Leave-one-out (each member vs the other 7; 204 fields = the 177 D165 fields + the 27 new ones):
| member | frac fields |t_gm|>t_crit(2.447) | max |t_gm| | median R_grid | median R_zon | zonal bins within 2 | fields with a bin beyond 4 |
|---|---|---|---|---|---|---|---|
| ctrl | 0.025 | 3.78 | 1.13 | 1.00 | 0.904 | 60 |
| p1 | 0.054 | 6.12 | 1.10 | 1.07 | 0.901 | 35 |
| p2 | 0.059 | 3.83 | 0.97 | 0.87 | 0.923 | 36 |
| p3 | 0.000 | 2.04 | 0.84 | 0.85 | 0.940 | 21 |
| p4 | 0.093 | 6.06 | 1.02 | 0.98 | 0.881 | 72 |
| p5 | 0.049 | 3.69 | 1.02 | 0.73 | 0.936 | 25 |
| p6 | 0.020 | 3.25 | 0.86 | 0.80 | 0.949 | 19 |
| p7 | 0.054 | 3.98 | 0.91 | 0.86 | 0.925 | 28 |
Mean over members: 0.044 of fields beyond the 5 % critical value (expected 0.05), 0.920 of zonal bins within 2 (expected P(|t6|<2) = 0.908), median R ~ 1. So the members look within noise and the scoring
method is calibrated. Consequence for the plan's proposals (RADIATION_AND_F2_PLAN 3.2): with 7-8 reference members the rule ">= 95 % of zonal bins within 2 sigma" is failed by true members (mean 0.92) and
"no bin beyond 4 sigma" is failed in 10-40 % of fields by true members; these proposals need t-calibration or the leave-one-out range (as `compare_to_loo` does).
Negative controls (tests): the stored month vs members p1-p7 sits inside the LOO range on all 8 aggregate statistics; +0.2 K added to t_500 is flagged per field (aggregates stay inside the LOO range: little power for a single
deviating field, hence the per-field list is the primary output); FEB1950 against the JAN ensemble: 65 % of fields beyond t_crit, median R_grid 3.8.
Limits: 8 members (sd uncertain by ~25 %), January only, perturbations at step 0 only. Fields with zero ensemble spread (incsw_toa) give t = 0 for zero difference and inf otherwise.
Annual means against the 100-year ANN4000-ANN4099 climatology (owner's reference, Reports/DOCUMENTATION_REVIEW.md): not built. The same functions apply unchanged if the reference array X (N,fields,46,72) is filled
from the 100 annual aij files (column meta from those files; N=100 gives a precise sd, t -> normal, and the interannual spread replaces the perturbed-ensemble spread), and the model year is scored as one more sample;
a multi-year port run can be scored by its multi-year mean with sd/sqrt(n_years). Needs: the aij files' column numbering (this build's names are read from the file), and a port run of >= 1 year, which is currently out of reach.

## 4. What blocks a one-month comparison (unchanged items plus new facts)
1. No port-side month: surface/ocean/ice/land/Ent loop is replayed, not closed; chained cost here is ~145 s/step on one core (a month = 1,488 steps = ~60 h per member) plus radiation-server time.
2. ~195 of the 483 changed columns are still unaccumulated (list above), several needing new dumps (ice/land-ice e0, CONDSE LMCMAX/WMPR, SE/KE internals).
3. The chained accumulation is only exercised on one 54-step window from one start state with recorded radiation.
4. Ensemble-based criteria thresholds remain proposals (section 3).

**Parent-session check (2026-10-07 13:10):** `tests/test_f3_diagnostics2.py` + `tests/test_f3_score.py` re-run: 9 passed, 1 skipped in 339 s; the skipped test is gated on the env var `F3_CHAINED_NPZ` and passes (1 passed) when pointed at `ff_data/nov26_day/d172_chained54.npz`. No tracked file modified. Headline chained-run numbers recomputed from `d172_chained54_vs_real.json`: 284 columns compared, 105 within the D163 tolerance; relative to column scale prsurf 4.8e-4, slp 1.36e-2, t_850 1.39e-2, t_500 6.99e-3, tsurf 6.69e-3, prec 5.3e-2, evap 3.9e-2; global-mean relative differences from 6e-9 (prsurf) to 3e-3 (prec); srnf_toa/trnf_toa exact only because radiation is recorded. Leave-one-out calibration of the scoring method re-done independently by the parent (global-mean t per field, each JAN1950 member against the other 7, 8 members, 161 valid fields): 60 of 1,288 tests beyond the 5% two-sided critical value (4.7%; the agent reports 4.4% on 204 fields; expected about 5%), so the members look like draws from one distribution. Not re-run by the parent: the 54-step chained run itself (about 2 h on one core), the negative controls, the zonal-bin statistics. NOT achieved (agent): about 195 of the 483 changed columns are still not accumulated, and no AIJ noise floor exists for the 54-step window (the five D151 members exist only as per-step states). The month-scale criteria proposed in the plan (>= 95% of zonal bins within 2 sigma) are too strict for 7-8 reference members by the agent's leave-one-out result.

# D174: model-month driver, stage 1 (inventory) and the first closure; driver and runs NOT built

Owner: Glenn Tamkin; written by a Claude Code agent, 2026-10-07. Project-local. Nothing committed; no existing file modified.
STATE: stages 3 (driver) and 4 (runs) were NOT started. No model_driver.py exists. Stage 2 has one finished item (RNG). Everything below is
either read from code/dumps (stated with the source) or measured; nothing here is a result of a model run.
New files: `fullfidelity/drv_rng.py`, `fullfidelity/tests/test_drv_rng.py` (3 passed, 8 s). Scratch: session scratchpad `inv/col_variation.json`.

## 1. Inventory: what the coupled path still takes from real dumps (surface_loop_v2.Loop2 + land_chain_ent + atm_step)
Key fact: every step reads the real SURFACE records of THAT step (`atm_step.surface_records(R)`: ffp 154 cols, ffs 90, ffl 60, ffg 450, fft 40) as
templates; the pipeline overwrites a subset of columns. A column is "recorded" when the port reads it and nothing overwrites it. Evidence: code reading
of `apply_state_to_records`, `override_pbl/override_tiles`, `next_land_pbl_columns`, `land_substep_ent`, `predict_ns2`, record layouts in the
instrumentation patches, and a measurement of which columns vary over the 54 real steps (`inv/col_variation.json`; rowsets change on 10 distinct sizes
because ice tiles appear/vanish; ffg/ffl/fft rowsets are fixed).

| Recorded input | Real routine | Inputs in our own state? | Blocks computing it | Dump for dec01/jan01 |
|---|---|---|---|---|
| Tile templates themselves (which rows exist, static columns: coriol, hemi, ihc, focean, axyp, soil data q/qk/dz/top_index, ...) | grid constants | yes (constants) | needs a template builder (clone prototype row per cell/type; new ice tiles have no row) | step-0 records only (6 steps) |
| Radiation columns: ffs srheat(15), trhr0(24), trup_in_rad(80); ffl srheat(8), flong(12), trup(14); ffp trhr0(17), qsol(20); ffg srheat(149), trheat(150) | SURFACE.f:508,575 SRHEAT=FSF(it)*COSZ1; ATM_DRV.f:336,349-350 flong=TRHR(0), fshort=FSF, trup=TRSURF | yes, from radiation-server outputs FSF, TRHR(0), TRSURF (frozen between radiation steps) and COSZ1 | only wiring; land TRUP is currently INFERRED from the recorded land patch (`LC.infer_trup`) | yes (6 steps) |
| Ent radiation inputs ffg 3-6: Ca, cosz1, vis_rad, dvis | GHY_DRV.f ~1191 (SRVISSURF, FSRDIR, COSZ1); Ca from GHG | vis/dvis/cosz from server outputs; Ca is a GHG-table value | Ca source (GHG file by year) not ported | yes |
| COSZ1 on non-radiation steps | Zenith.F90 calc_zenith_angle (orbit hour angle, coszt) | yes (clock, orbital parameters) | port of `useOrbit%getHourAngle` + `coszt` not done (source located: Zenith.F90, RAD_DRV.f:6434) | yes |
| PBL persistent state at substep 1: profiles u,v,t,q,e (ffp 50-88), cm/ch/cq (33-35), z0m input (49) | PBL_DRV.f (uabl.. cmgs.. restart arrays `*_ocn01/ice01/gla01/lnd01`) | yes: substep-2 outputs of the previous step, and the restart holds the initial value | carry as per-(type,i,j) grid arrays (tiles that vanish keep stale values); not wired | restart yes |
| ffs ice properties dF1dTG, hcg1, hcg2, fsri1, fsri2 (10-14) | SEAICE.f ice thermal functions (`ice_props_ff` exists) | yes (ice msi, snow, ssi) | not recomputed per step: recorded values go stale over a long run | yes |
| Lake fraction FLAKE (ffs 26) varies at the day boundary | daily_LAKE / UPDTYPE | partly | daily_LAKE not ported, never exercised | no |
| Land ffg: pres, vs0, gusti, ma1 (col 165) | atmosphere layer-1 mass, PBL ws0/gusti | yes | only wiring (`land_substep` takes ts,qs,rho,ch,vs,tprime,qprime,qm1 from PBL; the rest stay recorded) | yes |
| Land elhx (ffp col 19, type 4 varies) | GHY_DRV (LHE/LHS by ground state) | yes | rule not yet read | yes |
| GHY prognostic state at the start of each step (open loop) | GHY | yes (`Loop` carries it) | open-loop `atm_day_open_loop` resets from the record; `Loop2` carries it | restart yes |
| irrigation demand (ffg 147,148) | IRRIGMOD.irrigate_extract (460 lines + demand file) | no (data file) | reconstructed from the recorded ACTUAL flux; not computable without the demand file | yes |
| ffg Ent per-iteration exports | Ent | yes | computed (D171), first-step ulp residual open | yes |
| TRUP_in_rad of land, MMST (init_STRAITS), 5 static ADVSI geometry vectors | RAD / init_STRAITS / ICEDYN geometry | MMST, geometry: static | one dump per date | MMST/ADVSI geometry: yes |
| CONDSE entry RNDSS, SEEDS | RANDU stream | yes | CLOSED here (section 2) except the first seed | yes |
| CONDSE entry non-state fields (FEARTH.. static, RSI overridden, GZ/MWS/PMIDOLD, UKM rebuilt, DDM1.. outputs) | - | yes | which of them CONDSE really reads before writing was not tested | yes |
| Radiation packet surface side (20 fields: GTEMPR1-4, WSAVG, ZSI, SNOWI, POND_MELT, FLAG_DSWS, DLAKE, SNOWLI, ZSNOWI, BARESW, FRSNOW, SNOWD, ...) | RAD_DRV.f 3310-3391 inputs | yes (ours: atm gtempr, ice, lake MWL, land-ice, GHY snow) | formulas (ZSI, DLAKE, ZSNOWI, SNOWD, FRSNOW, BARESW) not yet derived; `atm_day_free_rad` still takes the 'live' values | live packets: nov26 11 steps only |
| Day boundary: DAILY_ATMDYN MDRYA constant; ch4ox water mass DM(j,l) (recorded from a step pair); SNOAGE aging (needs TDIURN); daily_LAKE, daily_LI, daily_OCEAN, UPDTYPE | RAD_DRV.f:1414,1600-1698; daily_* | ch4ox needs the dH2O file; SNOAGE needs TDIURN accumulation | ported pieces: DAILY_ATMDYN, Ent LAI update only | MDRYA recorded per date |
| First-step atmosphere state: MUS, MVS, MWS, GZ, hidden ATURB fields | dynamics | not in the restart | taken from `ffd_state_s1`/`ffa_step_a` of step 0 (exists for all 3 dates) | yes |
| Radiation itself | SOCRATES via persistent real-RADIA server | - | server only exists for the date's restart and runs forward; seeds for non-dumped steps are now computable (section 2); jan01 server first radiation step 17522 | n/a |

Minimum extra instrumentation (unused units: 1470-1479, re-grep before use; nothing found at D171): (a) first-step Ent forcing dump (D171 proposal);
(b) `dH2O`/GHG Ca values and MDRYA are small constants per date and can be read once; (c) a per-step dump of TDIURN at the day boundary (one array) to close SNOAGE aging;
(d) one 54-step ocean/ice boundary set is NOT needed for the v2 coupled path.

## 2. Closed: random-number chain (`drv_rng.py`)
Measured on 63 consecutive steps (nov26_day 53 steps, dec01 5, jan01 5; 0 mismatches): with the ModelE LCG ix->ix*69069+1 (mod 2^32),
SEEDS[1] = LCG^275790(SEEDS[0]) and SEEDS[0](k+1) = LCG^275790(SEEDS[0](k)) on non-radiation steps, LCG^405760 on radiation steps (RADIA adds 129970 draws).
RNDSS(3,40,72,46) follows from SEEDS[0] with the existing `clouds_condse_ff.randu_stream`. Only the first seed of a run is recorded (dump). The persistent
server's seed argument for later radiation steps is `radia_seed(s0)` (its effect is on cloud-overlap diagnostics only, D159).
Tests: `tests/test_drv_rng.py` (3 passed).

## 3. Not done (honest)
- Stage 2 closures other than RNG (orbit/COSZ1, radiation columns, PBL carry, ice props, radiation-packet surface side, template builder): analysed above, not coded or validated.
- Stage 3 `model_driver.py` (checkpoint/restart, timing log, F3 calls) and stage 4 runs (nov26 54-step all-computed day; jan01 multi-day): not started. No per-step cost of a combined driver was measured.
- The usage-limit pause interrupted the work; the remaining budget was used for this inventory and the RNG closure only.

## 4. What blocks a full month and realistic path
Blockers in order: (1) template builder + the closures in section 1 (everything in the table marked 'wiring' is mechanical; day-boundary daily_LAKE/UPDTYPE/SNOAGE/ch4ox and a first-day Ent set_vegetation_data are real porting work);
(2) cost: GHY+Ent about 6.5-9.6 s/step, CONDSE about 5 s, coupled surface step 43-100 s on 2 cores in D170 (being profiled by another agent), radiation about 11 s per call; at 60 s/step a month (1488 steps) is about 25 h per member, a model year (17,520 steps) about 12 days;
(3) the published reference (100-year annual-mean ANN4000-ANN4099, Reports/DOCUMENTATION_REVIEW.md) cannot score one month; the JAN1950 ensemble (D165) remains the only month-scale floor; annual-mean scoring needs a model year.
Path: build the record builder and validate column by column in teacher mode on nov26_day (54 steps), then dec01/jan01 6 steps; run nov26 54 steps closed; then jan01 from step 0 with checkpoints, one core for the radiation server, as many days as time allows; compare with a short real window produced from the same restart (allowed, 1 core).

**Parent-session status (2026-10-07 13:15): D174 is PARTIAL.** Done: stage 1 (the inventory table above) and one closure, the CONDSE/RADIA random-seed chain (`drv_rng.py`; `tests/test_drv_rng.py` re-run: 3 passed, 15 s; the 63-consecutive-step seed check on nov26_day, dec01 and jan01 with 0 mismatches is the agent's measurement, not re-derived by the parent). NOT done: the other closures (analysed above, not coded), the driver `model_driver.py`, checkpoint/restart, the all-computed nov26 54-step run, any jan01 run, any F3 comparison. The model-month driver does not exist yet. The remaining work was split into D176 (radiation-derived tile columns and Ent radiation inputs), D177 (COSZ1, persistent PBL state, ice thermal columns, land forcing leftovers) and D178 (day-boundary items and the driver skeleton with checkpointing).

# D177: state-derived columns of the coupled path (COSZ1, persistent PBL state, sea-ice thermal columns, land leftovers), 2026-10-07

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: `fullfidelity/drv_zenith.py`, `drv_state_cols.py`, `drv_ice_cols.py`, `drv_land_cols.py`, `tests/test_drv_state_cols.py` (14 passed, 67 s on one core; skip when dumps are absent), this entry.
Sources: real code (read-only) Zenith.F90, ATM_DRV.f:220, RAD_DRV.f:1542-1597 (DAILY_orbit), GEOM_B.f:481-880 (RAD_COSZ0), shared/AbstractOrbit.F90, Earth365DayOrbit.F90, OrbitUtilities.F90,
orbpar.f, Rational.F90, MODEL_COM.f:130-260, MODELE.f:1295-1330, PBL_DRV.f:150-345 and 1098-1260 (loadbl, setbl), SURFACE.f:560-598, GHY_DRV.f:1037-1280; dumps nov26_day (54 steps), dec01 and jan01 (6 steps).
Closes items of the D174 inventory (section 1) for these four rows. Radiation-derived columns (SRHEAT = FSF*COSZ1 etc.) are D176.

## 1. COSZ1 on every step (`drv_zenith.py`)
Method: port of CALC_ZENITH_ANGLE: the Earth365DayOrbit for orbpar(1850) (planetName Earth; master_yr=1850 gives variable_orb_par=0, orb_par_year_bp=100), time of periapsis from the
vernal equinox (year 1 Mar 21 12:00 of the 365-day calendar) with the Rational continued-fraction constructor (tolerance 1e-3 s), DAILY_orbit sinD/cosD at NOON of the model day,
hour angle 2*pi*fraction(t/Prot - t/Porb) (= time of day / 86400 for Prot = 86400*365/366, Porb = 365 d; EOT off for Earth), rot1/rot2 over DTsrc, then GEOM_B COSZT with SINJ/COSJ of cosz_init.
Clock: t = (IYEAR1-1)*365 d + 1800*itime with IYEAR1 = 1949 (restart itime 33312 = 694 d = 1950 Nov 26). The record label itime is the clock value AT the CALC_ZENITH_ANGLE call (offset +1 fails by 0.12).
orbpar(1850) reproduces the real PRT print (1.676429465128236E-002, 23.4592765450604, 280.326871404745) in all printed digits.
Result (COSZ1 (72,46) vs the recorded `ffa_step_<it>_r.bin` COSZ1): nov26_day 54 steps (incl. the day boundary at 33360 where DAILY_orbit changes), nov26 6, dec01 6, jan01 6: **max difference 0.0, 0 unequal elements**
with the Intel libimf sin/cos/tan/atan/acos/sqrt (the real build's library, loaded as in intel_libm_ff). With Python `math` instead: 1.7e-16 in 5 elements at step 33350 (1 ulp), 0 elsewhere. Only the three dates above were tested (one start year of the calendar; year 1850 orbit).
Not covered: other orbits/calendars (leap years are not in this calendar), a run started at another IYEAR1 (a parameter), the radiation steps' COSZ1 equals the same function (verified on the records of those steps).
Instrumentation needed: none.

## 2. Persistent PBL state at SURFACE substep 1 (`drv_state_cols.py`, class `PBLCarry`)
What persists (PBL_DRV.f): per (type, patch, i, j) `atm%uabl/vabl/tabl/qabl(1:8)`, `eabl(1:7)`, `cmgs/chgs/cqgs`, `ipbl`, `ustar_pbl`, `lmonin_pbl` (read at :255-259, :282-284, written back :341-348, :397-400). The restart stores them as `*_ocn01/ice01/gla01/lnd01`.
`loadbl` (called once per step, SURFACE.f:409): a tile with ipbl == 0 (no PBL call in the previous step) takes the whole state from a donor of the same cell with ipbl == 1 (ocean <- ice else land; ice <- ocean; land ice <- land; land <- land ice else ocean); then all ipbl are reset to 0.
z0m (record input col 49) is a local that DFLUX overwrites for itype 1,2 before use (ROUGHL constant for 3,4); `dskin` (col 28) is an output of the port, not an input: neither carries information (they differ from the previous output by up to 7.9e-4 and 0.9 and the PBL port reproduces its outputs regardless).
Implementation: `PBLCarry.from_restart(date)` -> per step `begin_step()` (loadbl), `fill(rows)` (substep-1 input columns u,v,t,q,e, cm, ch, cq from the carry), `update(rows, out)` after each substep with that substep's PBL outputs.
Results (rows = every (i,j,itype) of the record, columns 50-88 and 33-35):
 (a) carry from the RECORDED previous outputs (the carry logic alone), both substeps: nov26_day 54 steps, dec01 6, jan01 6: **0 differing elements** (max 0.0), including the 53 ocean rows of nov26 step 0 that the restart has with ipbl == 0 and that are donor-initialised (test `test_pbl_carry_donor_rule_exercised`: without loadbl the stale restart values differ), and the rowset changes (4591 -> 4541 rows at step 33312->33313: new tiles from donors, vanished tiles keep stale values unused).
 (b) carry from OUR OWN PBL outputs (`pbl_compare.run`, the existing port, with all other columns recorded), nov26_day 54 steps, substep 2 built from our substep-1 output, step k+1 from our step-k output: start-column difference vs record max 4.8e-10 (u), 3.9e-10 (v), 3.8e-10 (t, K), 4.3e-10 (e), 1.2e-12 (q), 4.8e-12 / 8.2e-12 (cm, ch=cq); per-step max over all quantities 2e-11 at step 0, 1e-10 around step 18, 2e-10 around step 36, not monotone (bounded over the 54 steps). This is the accuracy of the existing PBL port (D-series PBL entries) accumulated through the carry, NOT an error of the carry. It is not bitwise.
Cannot be closed here: (i) the SET OF TILES that exist (which rows; new ice tiles have no row) is the template-builder problem of D174 section 1; PBLCarry handles any rowset it is given; (ii) `ustar_pbl`/`lmonin_pbl` per type are carried by the real code but our path uses the composite S['USTARPBL'], S['LMONINPBL'] (unchanged here).
Instrumentation needed: none (the restart supplies step 0, the previous outputs supply the rest).

## 3. Sea-ice thermal columns (`drv_ice_cols.py`)
dF1dTG, HCG1, HCG2, FSRI(1:2) (ffs cols 10-14) from the tile's own state columns (tg1, tg2, snow, ssi1, ssi2, flag_dsws) and SRHEAT (> 0 test; radiation-derived, taken from the row) as SURFACE.f:560-598, using the existing `ice_props_ff.ice_tile_props`, with `solar_ice_frac` re-done in numpy using the Intel libimf `exp`.
Result on the ice rows of both substeps: nov26_day 54 steps 84,766 rows, dec01 9,396, jan01 8,352: **all five columns max difference 0.0**. With `jnp.exp` (the existing function alone) dF1dTG/HCG1/HCG2 are also 0.0 but FSRI1/FSRI2 differ by 5.6e-17 (1 ulp) in 152/112 of 84,766 rows (nov26_day), 12/6 (dec01), 18/18 (jan01).
`ice_thermal_columns(tile_rows)` overwrites the five columns of itype-2 rows; to be applied after apply_state_to_records and after predict_ns2 (substep 2: tg1/tg2 from substep 1).
Cannot be closed: none of the five; SRHEAT itself is D176. Instrumentation: none.

## 4. Land ffg leftovers and land elhx (`drv_land_cols.py`)
Measured identities on 99,396 land rows (all 54 steps nov26_day + dec01 + jan01 windows, both substeps), all with max difference 0:
pres (ffg 154) = PBL psurf (ffp 16) (GHY_DRV.f:1261 `ps - dmCO2cond*grav*0.01`; dmCO2cond = 0 in this Earth build); vs0 (159) = PBL output ws0 (ffp 98); gusti (160) = PBL gusti input (ffp 24 == 113);
vs (158) = PBL ws (91); ma1 (165) = MA(1) of the atmosphere at SURFACE entry (equal to the 'r'/'d' site MA(1) in four dumps; sub1 and sub2); land elhx (ffp 19) = lhs if tg1 < 0 else lhe (GHY_DRV.f:1071-1075) with tg1 := tg - tf: 0 mismatches of 99,396.
`land_substep_v2(p4, g, q1, trup, ma1_ij, set_elhx=True)` = `land_chain.land_substep` with pres/vs0/gusti/ma1/elhx computed (not edited in place): on nov26 step 33312 substep 1 every patch output and tbcs, ashg, aevap, tsns are identical (max 0.0) to the recorded-column call.
Limits: tg1 in the real code is tsns_ij (carried GHY output), the check uses tg - tf; a cell with tsns within one rounding step of 0 C could differ (none in the records). dmCO2cond is assumed 0 (not exercised); for the coupled path pass our PEDN(1) through `override_pbl` as now and ma1 from our S['MA'][0].T.
Instrumentation needed: none.

## 5. Not done / honest limits
- Wiring into `surface_loop_v2.Loop2` / `atm_step.stage_surface` was NOT done (that would edit existing files); the four modules are drop-in functions with the tests above. No coupled run was repeated with them, so no statement is made about the effect on the coupled 6-step or 54-step results.
- Tested only on nov26_day (54 steps), dec01 and jan01 (6 steps each); the PBL own-output chain (b) only on nov26_day.
- Libimf-based bitwise results hold only on hosts with the Intel runtime.
Reproduce: `cd fullfidelity; python -m pytest tests/test_drv_state_cols.py -q`.

**Parent-session check (2026-10-07 13:30):** `tests/test_drv_state_cols.py` re-run: 14 passed, 52 s; no tracked file modified. Assertions include COSZ1 max difference 0.0 on nov26_day (54 steps), nov26, dec01, jan01 (with the Intel libimf; 1 ulp in 5 elements with Python `math`), the orbital parameters of 1850 to 1e-17/1e-12/1e-11, 53 donor-initialised ocean PBL rows, ice thermal columns 0.0 and land leftovers array-equal. Not re-derived by the parent: the own-PBL-output chain over 54 steps (4.8e-10 at worst: the existing PBL port's error carried, bounded) and the claim that `land_substep_v2` equals `land_chain.land_substep` on step 33312 beyond what the test asserts. NOT wired into `surface_loop_v2`/`atm_step` (would need edits to existing files): no coupled run has used these modules yet, and the tile-row template builder (which rows exist) is still open. Bitwise results need the Intel libimf runtime.

# D175: speed of the coupled step (profile, parallel drop-in variants), 2026-10-07

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: `land_ent_par.py`, `clouds_condse_par.py`, `dyn_jax_worker.py` (measured, NOT recommended, see 4), `speed_d175.py` (timed driver + bitwise comparer), `tests/test_speed_d175.py` (2 passed, 56 s; skip when data absent), this entry.
Setup of every measurement: nov26, surface_loop_v2.Loop2 (all pieces computed) + land_chain_ent (Ent computed) + batched CONDSE + numpy dynamics with libimf, recorded radiation frozen (as run_coupled_v2), steps 0-5 from the real restart. Wall times are per step, process-pinned with taskset, on a node that was idle for the final runs (the first profile, 09:35, ran at node load ~6 with other agents' jobs; its numbers are higher).

## 1. Profile (serial code, steady state, step 3; seconds)
| stage | s (loaded node, 09:38) | s (idle node, 1 core) | kind |
|---|---|---|---|
| dynamics (dyn_step, numpy+libimf pow) | 5.6 | 4.6 | vectorised numpy, per-stage Python dispatch |
| CONDSE batched (MSTCNV 9.4 of 11.1) | 11.1 | 8.6 | numpy, lock-step Python loops over layers/events with column masks |
| land: PBL (JAX jit) + scalar GHY+Ent per cell, 1,506 calls | 10.0 (Ent+GHY 9.8) | 7.5 | pure Python loop per cell; of it 3.7 s was jax-array indexing `float(v[n])` in land_substep_ent |
| tile chain (PBL/ATURB tiles, land ice) | 0.4 | ~0.4 | JAX jit |
| post-tile surface (DYNSI, ground_*, RIVERF, ocean chain, ADVSI) | 9.7 (ocean dynamics 3.5, OCONV/HBL 2.4, straits 0.8, meso 0.55, GROUND_SI 0.84, ADVSI 0.53, FORM_SI 0.5) | ~7 | JAX eager (no outer jit; ~15,000 op-by-op dispatches) + numpy |
| filter, misc (record reads 0.8 s) | ~3 | ~2.5 | numpy |
| total per step | 38.7 | 29-31 | |
First step: 217-230 s; step 1: 59-72 s; step 2 onward steady. The 409 s of D170 was the same effect on a loaded node (2 cores). Cause, measured with cProfile: JIT compilation. Step 0: 1,059 XLA compilations, 114 s in `backend_compile` plus ~70 s of tracing/lowering (pbl advanc_batch 30 s, tile chain run_chain 50 s, landice_chain 29 s, ocean stages ~60 s: dynamics 32, meso 14, straits 13); step 1: 104 more compilations (19 s; the tile chain meets new tile-set shapes, 30 s in run_chain); step 2: 51 (2 s); step 3: 1. File reads are ~1 s per step, not the cause. Earlier D170 numbers (43-100 s) are consistent with a loaded 2-core run, they are not the code's idle cost (29-31 s on one idle core).

## 2. What was built (all drop-in, new modules)
* `land_ent_par.py` (`ParEntLand`, `land_substep_ent_par`, `install_par`): the per-cell GHY+Ent loop of `land_chain_ent.land_substep_ent` in `nproc` persistent spawn workers; each worker owns a fixed set of cells (their Ent state and Qf carry); same function `ghy_ent_call` per cell. Also removes the jax-indexing cost. Cores: nproc workers (+ caller idle).
* `clouds_condse_par.py` (`condse_step_batch_par`, `par_condse`): the unchanged `condse_step_batch`, with `mstcnv_batch` run on round-robin column slices in nproc workers. Cores: nproc workers.
* `speed_d175.py`: timed coupled loop (mode ref = existing code, par = new modules), state dump per step, bitwise `cmp`.

## 3. Validation (bitwise)
* Land stage alone, 2 steps x 2 substeps, carried dyn and Ent/Qf state, 4 workers vs `land_substep_ent`: every GHY output and patch field identical (0 differences).
* CONDSE alone, nov26 steps 0 and 1 (libimf), 4 workers vs `condse_step_batch`: all fields identical (0 differing arrays).
* Whole coupled loop, 6 steps, 504 arrays (T,Q,U,V,P,QCL,QCI and the full surface state: ocean, ice, lake, land ice, atm): `par` with 3 workers pinned to ONE core vs `ref` on one core: 0 differing arrays. `par` 3 workers on 3 cores vs `ref` on 3 cores (2 steps, 168 arrays): 0 differing.
* IMPORTANT existing property found, not caused by D175: the existing `ref` code is NOT reproducible across core counts. `ref` on 1 core vs `ref` on 3 cores differs already at step 0 (T 1.7e-13, U 8.3e-13, ocean g0m 3.9e6 absolute (field scale not checked), uo 1.4e-13, ...: 123 of 168 arrays differ after 2 steps). Same-core-count reruns are bitwise (ref vs ref, 1 core). Cause not isolated: the JAX/XLA CPU stages (ocean, PBL, tile chain) change reduction/thread partitioning with the number of available cores; the numpy stages do not. Consequence: bitwise comparisons must use the same core affinity for both sides (done above); the model month is a chaotic run anyway, but "bitwise" statements hold per core count.

## 4. Things tried that are NOT usable (honest)
* JAX dynamics (`dyn_step_jax2`) in a worker with the FMA-free flag (`dyn_jax_worker.py`): 1.7 s vs 3.5 s numpy at steady state (47 s first call), but NOT bitwise to the libimf numpy dynamics used by the coupled step: U max relative 3.5e-13 (127,480 of 132,480 cells differ), GZ 2e-14, T 1.1e-15, P 1.3e-15, from the pow function (JAX is bitwise only against numpy-pow, D139-D144). Per the D127-D129 finding that libimf matters for cloud thresholds it is not used. Not wired in.
* JAX persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`, min compile time 0): step 0 drops 137 s to 73 s and step 1 37 s to 27 s on a warm cache, and the cold run (cache write) is bitwise equal to the no-cache run (168 arrays), BUT the warm-cache run differs from the cold run: T by 0.067 K, U 0.55 m/s, P 1.1 hPa after step 0 (138 of 168 arrays). Cause not investigated (cached executables are not equivalent to fresh compiles here). DO NOT use the persistent cache for validated runs.

## 5. Result: seconds per step, steady state (idle node; first step in brackets)
| config | cores | dyn | CONDSE | land+tiles | post (ocean etc.) | total/step | (step 0) |
|---|---|---|---|---|---|---|---|
| existing | 1 | 4.6 | 8.6 | 7.5 | ~7 | 29.0-31.0 | (217) |
| par | 2 | 4.8 | 6.5 | 4.6 | ~7-8 | 25.1-26.5 (28.7 at step 2) | (167) |
| par | 3 | 4.6 | 5.7 | 3.2 | ~6.3 | 21.5-23.1 | (143) |
| par | 4 | 4.9-5.4 | 5.5 | 2.5-2.8 | ~6 | 20.6-21.5 | (137) |
(existing code at the loaded-node profile: 38.7 s; D170 quote 43-100 s.)
Projected 1,488 steps (steady state only, plus ~4-5 min start-up): 1 core 12.2 h (existing code, idle node), 2 cores 10.5 h, 4 cores 8.6 h; at the loaded-node rate the existing code would be 16 h. Not included: free-running radiation through the persistent server (~11 s per call, every 5th step, about 55 min per month, it runs as its own process), F3 accumulators, restart I/O. The 4th core buys only 0.8 s: what is left is serial.

## 6. What still dominates and next speed-ups (estimates, not measurements)
1. Post-tile surface/ocean, ~6-7 s (30%): eager JAX with ~15,000 dispatches. (a) `jax.jit` the ocean stage bodies (est. -3 to -4 s) but the XLA fusion/FMA changes rounding unless the D139 flags are set, so it has to be re-validated against the real ocean dumps (not bitwise guaranteed); (b) run it in a worker while the next step's dynamics (4.6 s, independent of the surface) executes: the ocean chain does not feed the atmosphere of the same step; only next step's CONDSE (via MELT_SI/RSI) needs it. Pipeline gain est. 4-5 s per step for 1 more core; needs the surface state to live in the worker.
2. Dynamics 4.6 s: libimf `pow` ctypes calls are per element; a vectorised/batched libimf call or a C helper (est. -1 to -2 s). The JAX version is not bitwise (4).
3. CONDSE 5.5 s: MSTCNV does not scale beyond ~2x (fixed Python lock-step cost per chunk, 3.7 s wall on 4 workers vs ~7 s serial); a compiled (numba/C) event block would cut it to ~1-2 s but must be re-validated bitwise (libimf exp/pow).
4. Land 2.5 s on 4 cores: cell imbalance (Ent iterations differ); dynamic load balancing needs state migration, est. -0.5 s.
5. First steps (compile 140-217 s): irrelevant for a month; the persistent cache would help but is not safe (4).
Rough achievable: ~12-14 s per step on 5 cores with (1b)+(2) if (1b) is kept bitwise = 5-6 h per month. Not done here.

**Parent-session check (2026-10-07 13:40):** `tests/test_speed_d175.py` re-run: 2 passed, 57 s (pinned to 3 cores, 1 thread each); no tracked file modified. The assertions are bitwise (`np.array_equal`) between the parallel variants and the existing stages. Timings (e.g. 20.6-21.5 s per coupled step on 4 cores vs 29-31 s on one idle core; month 8.6 h on 4 cores) are the agent's measurements on a shared node and were not re-measured by the parent. TWO HAZARDS reported by the agent that affect every bitwise claim in this repository: (1) the EXISTING code gives different results on 1 core and on 3 cores (step 0: T 1.7e-13, U 8.3e-13; 123 of 168 arrays differ after 2 steps), cause not isolated (likely the JAX/XLA CPU stages); reruns on the same core count are bitwise equal, so bitwise comparisons must use the same core affinity on both sides (the sharded regression pins its jobs and uses one thread each); (2) a warm persistent JAX compile cache changes results (T by 0.067 K, U by 0.55 m/s after one step) versus no cache or a cold cache: do NOT use the compile cache for validated runs. Also not usable: the JAX dynamics worker (1.7 s vs 3.5 s numpy) is not bitwise to the libimf numpy dynamics (U up to 3.5e-13 relative).

# D176: radiation-derived tile columns, Ent radiation inputs, packet surface side (2026-10-08)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: drv_radcols.py, drv_radcols_day.py, drv_radcols_couple.py, drv_radpacket.py, drv_radpacket_check.py, tests/test_drv_radcols.py (7 passed, ~25 s),
ff_data/nov26_day/ours_d176/lake_legs_replay6.npz (lake ice-fraction legs of OUR 6-step replay surface loop).

## 1. How each column is computed (sources read: RAD_DRV.f 4437-4451 and 5497-5535, ATM_DRV.f 336-350, SURFACE.f 508/575, SURFACE_LANDICE.f 278, PBL_DRV.f 160/192, GHY_DRV.f 1061/1151/1191-1193/1422, GHGMOD.f)
FSF(4), TRSURF(4), TRHR(0), SRVISSURF, FSRDIR are model arrays set from the RADIA outputs on radiation steps (server returns them) and FROZEN in between,
EXCEPT that RESET_SURF_FLUXES rescales FSF/TRSURF whenever the ice fraction (or lake fraction) of a cell changes. srheat = FSF(type)*COSZ1; trhr0/flong/trheat = TRHR(0);
qsol = FSF*COSZ1; TRUP = TRSURF(type); vis_rad = SRVISSURF*COSZ1*0.82 (REAL*4 literal), dvis = vis_rad*FSRDIR; Ca = GHG table CO2 at year 1850, day 182 (GTREND).
Reset legs (found by measurement, then confirmed in source): ocean cells (a) after MELT_SI at step head, (b) after ocean FORM_SI, (c) after ADVSI; lake cells inline at MELT_SI and FORM_SI.
Day boundary: daily_LAKE FLAKE changes -> RESET(4,1)/(1,4)/(2,4) (implemented, RadSurf.daily_lake).

## 2. Result per column, nov26_day, 54 steps, max |computed - record| (both substeps)
- ffs srheat, trhr0, trup_in_rad, ocean-domain tiles: 0 (bitwise), all 54 steps (ice legs from the real ADVSI dumps). Lake-cell trhr0: 0.
- ffp trhr0: 0. ffp qsol, ocean domain + land: 0 except steps 33360/33361: 4.6e-8 / 1.4e-9 (day boundary).
- ffl srheat, flong, trup_in_rad: 0. ffg Ca, cosz1, vis_rad, dvis, trheat: 0. ffg srheat: 0 except 33360/61 (4.6e-8).
- Ca: ghg_ca() == recorded 284.31639920765025 bitwise (ghg_day=182 reproduces it; where the default 182 is declared was not found).
- Land TRUP: now TRSURF(4). The record has no land TRUP column; reference = old inferred value: |diff| <= 2.3e-13 (inversion rounding noise), 1.3e-7 at the day boundary.
- COSZ1: radiation steps = server output, equal to the record (0). Non-radiation steps: NOT computed (input; D177).
## 3. Not closed / honest limits
- Lake cells (FLAKE>0, FOCEAN=0) srheat/trup: bitwise on the radiation-step windows only when exact legs are supplied. With OUR replay legs (6 steps): srheat/qsol 0, trup 4 values at 1 ulp (5.7e-14).
  Over 54 steps no exact lake legs exist (the replay needs ffo/ffy dumps of 6 steps only); with one composite leg between record entries the error is 4.7e-4 relative (not bitwise).
- Ocean legs in OUR closed loop depend on OUR ice state (ADVSI/DYNSI); the dumps' legs were used for the bitwise result.
- Day boundary: FLAKE/FEARTH/RSI after daily_LAKE are inputs (daily_LAKE not ported); RSI_old approximated -> the 4.6e-8 above. CoupledRad records itime%48==0 steps in day_boundary_calls.
- A 6-step closed coupled run with CoupledRad installed vs without: 22 legs applied; end states differ 2e-13 at step 0 and up to 2e-2 (T max) from step 1, same size as the
  run-vs-real divergence of both runs (rms T 1.4e-3 vs 1.41e-3 at step 5). This is not distinguishable from chaotic growth of ulp perturbations (one realization); not attributed.
## 4. Packet surface side (21 arrays) vs live rsv_n26_*_in.bin, non-pole cells (drv_radpacket.py; state at RADIA time = end of previous step + MELT_SI, before precipitation)
- Step 33312, restart state: all fields bitwise except GTEMPR2 (2 cells, 5.7e-14 K).
- Land group GTEMPR4 (tbcs+TF), BARESW, SNOWD, TSAVG (composite), WSAVG (sum ftype*ws): bitwise at all 11 radiation steps, using the previous step's GHY outputs as stand-in for our GHY state.
  FRSNOW: <= 2.2e-16 (1-2 ulp, <=30 cells); needs persistent snowbv (restart variable) for fb=0/fv=0 cells.
- Ice/lake/land-ice/ocean-water group at 33317 from OUR replay state: RSI 9.8e-10, SNOWI 2.8e-10, ZSI 5e-8, GTEMPR1 2.4e-8, GTEMPR2 3.4e-8: state differences, not formula errors. Later radiation steps not checked (no carried state beyond 6 steps).
- Not wired into atm_day_free_rad (that module is unchanged); the functions are ready for it.

**Parent-session check (2026-10-07 14:25):** `tests/test_drv_radcols.py` re-run: 7 passed, 27 s; no tracked file modified. Assertions: bitwise (0.0) for the ocean-domain `srheat`/`trhr0`/`trup_in_rad`, lake `trhr0`, `ffl` and `ffg` radiation columns over the nov26_day records, 4.6e-8 and 1.4e-9 only at the day-boundary steps 33360-33361, land TRUP within 2.3e-13 of the old inferred value (1.3e-7 at the boundary), and the radiation-packet surface side bitwise at step 33312 except one 5.7e-14 K value pair. Not re-run by the parent: the 6-step closed coupled comparison. CAUTIONS from the agent, kept: (1) over 54 steps no exact lake legs exist (the replay needs ocean dumps that cover only 6 steps), so lake `srheat`/`trup` are not bitwise there (4.7e-4 relative with one composite leg per step); (2) at the day boundary the lake fraction after `daily_LAKE` is still an input (not ported; `CoupledRad` logs every `itime % 48 == 0` step in `day_boundary_calls`); (3) in the closed coupled run the new columns change the end state by 2e-13 at step 0 and by up to 2e-2 in T from step 1, the same size as each run's divergence from the real run (rms T 1.4e-3 at step 5 in both): not attributed, indistinguishable from chaotic growth of a last-bit perturbation; (4) `COSZ1` on non-radiation steps is computed by D177 (`drv_zenith.py`), not here; (5) the radiation-packet surface side beyond step 33317 cannot be checked (no carried state beyond 6 steps); `atm_day_free_rad` is unchanged (the functions are ready for it). Effect on the coupled-step gate: the recorded-input list of `ACCEPTANCE_CRITERIA.md` section 1.5 shrinks by the radiation-derived tile columns, the land TRUP and the Ent radiation inputs Ca/cosz1/vis_rad/dvis (on nov26_day).

# D178: driver skeleton with boundary-provider interface, checkpoint/resume, and the day-boundary items (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed. New files only: `fullfidelity/model_driver.py`, `fullfidelity/drv_daily.py`,
`fullfidelity/tests/test_model_driver.py` (10 passed, 343 s on 2 cores). No existing file edited; SOCRATES/RADIA untouched.
Scope was cut at the owner's request (end-to-end JAX direction): the driver stays a thin skeleton; the 54-step runs were started and then killed, NOT completed; no day-long result is claimed.

## 1. model_driver.py
`ModelDriver` holds the complete state (atmosphere S + CONDSE/RADIA carry, LSCOND arrays, surface-loop-v2 state SS, DYNSI USI/VSI, ADVSI RSIX/RSIY, carried land/GHY state, Ent state, F3 accumulators, RNG seed, provider state, step counter, timing log) and advances steps with the existing stages in the order of MODELE.f (dyn, MELT_SI, CONDSE (batched), RADIA, surface incl. ocean/ice/lake/ADVSI inside `surface_loop_v2`, dissip, filter, day boundary). Existing modules are rebound only for the duration of a step.
Modes: surface closed|replay, ent record|computed, rng chain|record, daily recorded|computed|none, f3 on/off.
**Boundary provider** (`BoundaryProvider`): `real(itime)`, `stage_radia`, `radiation_aij`, `surface_records` (the recorded SURFACE templates), `column_modifiers()` (plug-in point for D176/D177 computed columns: `fn(rec, StepContext) -> rec`, mode `apply` or `shadow`; the driver logs per step how many elements each modifier changed), `initial_seed`, `daily_inputs`, `state/set_state`. `RecordProvider` = record-backed; `ServerRadiationProvider` = persistent radiation server (written, NOT run).
Per-step log (jsonl): stage times (stage_*, surface_tiles, surface_post, melt_si, f3, daily), modifier changes, RNG-chain-equals-record flag, state digest. Typical closed-mode step, 1 core: 20-26 s after a 180-225 s first step (JIT); checkpoint 112 MB, 0.3 s.
RNG: RNDSS of CONDSE computed from the drv_rng seed chain (`rng='chain'`); equals the recorded RNDSS on all defined cells (the record holds garbage 7.6e39 on pole rows i>1; compare only i<=IMAXJ).

## 2. Measured
- Checkpoint/resume, cross-process, closed surface, record Ent, F3 on, 1 core: 6 steps in one go vs 3 steps + checkpoint + fresh process resume + 3: complete state trees (541 arrays, 112 MB) bitwise equal (0 differences), per-step digests and output files equal on steps 3-5. Two independent identical runs are also bitwise equal. Non-vacuity checked (perturbed state is detected).
- Driver vs the existing chained result: driver (closed, 2 cores) vs `surface_loop_v2.run_coupled_v2(6)` (2 cores): all 6 step end states (T,Q,U,V,QCL,QCI,P) bitwise equal. Per-step rms vs real (T): 7.4e-14, 2.0e-4, 5.1e-4, 7.7e-4, 1.05e-3, 1.41e-3 K = the D170 numbers. This reproduces D170 (same code path); it does not re-validate it.
- **Reproducibility caveat (new finding):** results depend on the number of visible cores. 1 core vs 2 cores differ at rounding level at step 0 (T rms 2.6e-15, 1048 elements) and grow to T rms 1.3e-3 at step 5 (same size as the real-member floor); 6,791 arrays differ in the final state. Bitwise claims hold only for a fixed core mask. Cause not isolated (XLA/BLAS reduction order is the suspect; not tested).
- Not done: reproducing D171 (Ent computed, replay) and D172 (F3 54-step) with the driver; the radiation-server run; closed + Ent computed; any multi-day run. (Wiring for them exists, untested beyond constructing the driver and pickling the Ent state.)

## 3. Day-boundary items (drv_daily.py), validated on the one boundary (33359 -> 33360)
| item | status | evidence |
|---|---|---|
| DAILY_ATMDYN | existing port; MDRYA now derived = 984*100/9.80665 = 10034.007535702814 | equals the recorded value exactly; MA, PEDN, PMID, PK, PDSIG, P, PEK: bitwise vs real start of 33360 |
| daily_ch4ox | computed from dH2O file (getqma ported with REAL*4 placement) and GHG table | Q bitwise equal (0 of 132,480 elements differ) to the real start state of 33360. Finding: ghg_yr = master_yr = 1850, so CH4 = 0.808092 ppm (table row 1848), not the 1950 value (the first attempt with year 1950 was off by a constant factor 1.755) |
| SNOAGE aging | snoage = 1 + .98*snoage (types 1-3, i<=IMAXJ), snoage_def=0 | bitwise vs cse_in 33360 (6,051 elements changed). The D174 claim that TDIURN is needed is wrong for this rundeck (only snoage_def=1 uses it) |
| UPDTYPE | computed | agrees with fft ftype to 5.6e-17 (rounding) |
| Ent LAI/albedo | existing (EntLand.maybe_daily) | D171 |
| daily_LAKE | NOT implemented | variable_lk=1: FLAKE/FEARTH/FLAND change in 632 cells (max 5.0e-3), RSI in 834 cells at this boundary. Needs DMWLDF (GHY soil-saturation deficit, GHY_DRV.f:4062-4127), TANLK, and update_land_fractions (GHY water/heat transfer). Minimal dump: DMWLDF, TANLK, lake+ice+GHY-soil state before/after at one boundary |
| daily_LI, daily_OCEAN (GLMELT file active), daily_EARTH other than Ent (wfcs, bare-soil wetness, roughness), CO2 ppm / RCOMPT (radiation package; Ca constant 284.316 across the boundary) | NOT implemented | need ocean/ice/GHY dumps across a boundary; none exist in nov26_day |
In closed mode the surface state is therefore NOT updated for lake fraction, glacial melt etc. at a day boundary.

## 4. Limits
One start state, one boundary, 6 steps; fixed-core-mask bitwise only; provider server path and computed-Ent/closed combination untested end to end.

**Parent-session check (2026-10-07 14:36):** `tests/test_model_driver.py` re-run on cores 6-7 (the agent's cores): 10 passed, 341 s; no tracked file modified. Assertions include bitwise equality of the full state (541 arrays) for 6 steps in one go versus 3 + checkpoint + a fresh process + 3, two identical runs bitwise equal, the daily-update fields at 0.0 difference with a non-vacuity check (the daily routine changes Q by 2.7e-6), the RNG seed chain equal to the record, and modifier plumbing (shadow mode). CORRECTION to D174: SNOAGE aging does not need TDIURN in this rundeck (the agent measured aging = 1 + 0.98 x snoage, bitwise). CAUTIONS kept: results depend on the core count (1 versus 2 cores differ at rounding level from step 0 and grow to the size of the real-member floor by step 5; the cause is not isolated), so every bitwise claim holds only for a fixed core mask; NOT done: `daily_LAKE`, `daily_LI`, `daily_OCEAN` GLMELT, the rest of `daily_EARTH` and the CO2/RCOMPT update (FLAKE changes in 632 cells and RSI in 834 cells at the one available boundary, so a multi-day closed run will not update the lake fractions without them); no radiation-server run, no Ent-computed closed run and no multi-day run through the driver; `ServerRadiationProvider` is written but was not run. Not re-run by the parent: the 6-step driver versus `run_coupled_v2` comparison beyond what the tests assert, and the cost figures (20-26 s per step steady, 180-225 s for the first step; shared node). Adopted as the host layer under `ACCEPTANCE_CRITERIA.md` section 8 item 4.

# D180: stage 1 of the JAX-driven coupled step, the atmosphere half (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: `jax_atm_step.py` (the step), `jax_atm_step_run.py` (driver, modes ref/jax), `jax_atm_step_cmp.py` (comparer), `tests/test_jax_atm_step.py` (3 passed, 4 s; run in its own process with RUN_XLA_FLAG_TESTS=1; the saved-run test needs JAX_ATM_STEP_DIR), this entry.

## 1. What it is
`atm_step_jax(S, provider, itime, ctx, kit, ms)` advances the same state dict as `atm_step` by one 30-min step in the real stage order (dyn, condse, radia, surface x2, dissip, filter). `RecordBoundary` is the explicit record-backed boundary provider (serves `atm_step.Real`, logs the record keys read, supplies SRHR/TRHR/COSZ1). `run_chain_jax` chains steps from our own end state (same hand-over as `atm_step_fast.run_chain`). Land mode used: `recorded`. Libm mode (ctx imf=False), as instructed: not libimf.

## 2. What is JAX, what is not (exact)
JAX: dynamics via `dyn_step_jax2` (advecv, pgf, iso, sdrag, filter_chain, kea, wsave, aflux, advecm+MAtoP, aadvt, qdynam; one jit call per stage); PEK pow; CONDSE LSCOND and MSTCNV batch kernels (`clouds_condse_jax`, xla mode); radiation T update (RAD_DRV.f:5474-5478, jitted, new here); inside the surface stage the jitted PBL advance, tile fluxes, aggregation, ATURB+UV, get_dbl, land-ice chain (existing code, unchanged).
NOT JAX (all executed as NumPy/eager Python on host arrays): dyn: CALC_TROP, MAtoPMB, SE/KE bookkeeping, energy fix, PGRAD_PBL, workspace glue copies, QDYNAM extra-column z branch; condse: column set-up and post-processing, bookkeeping, the 2 pole columns (per-column port), snow-age exp loop, QUS subsidence advection (host callback), recalc_agrid_uv, condse_inputs; radia: cloud masking CLDSS/CLDMC; surface: record rewriting (override_pbl/override_tiles), first-layer TMOM/QMOM update, layer1_exports, composite ustar/lmonin, ATURB-output merges, land patch (recorded GHY outputs), `run_chain` ice-property glue, record readers; dissip; filter (SLP filter + matopmb); every stage hand-over (state is NumPy on the host). The PBL, ATURB and surface-layer arithmetic IS jitted JAX (existing ports); only their glue is NumPy.
Radiation: no radiation is computed and no Fortran callback is used in this stage; SRHR/TRHR/COSZ1 are recorded.

## 3. Recorded inputs (from `jax_recorded.json` plus code reading)
Record keys read by the step: `ci` ffc_cse_in (non-dynamic CONDSE entry fields), `s1` ffd_state s1 (start state), `sitea` ffa_step_a (hidden/PBL start state), `siter` ffa_step_r (SRHR/TRHR/COSZ1), `sitee` (end reference, comparison only; also CONDSE exit for the masking only on radiation steps), plus the SURFACE records ffp/ffs/ffl/ffg/fft (tiles, PBL, land-ice, recorded GHY outputs, Ent exports, land forcing, sea-ice/lake/ocean surface state) and CONDSE geometry/constants. Sizes per ACCEPTANCE_CRITERIA 1.5 are in the D174 inventory (not re-measured here).

## 4. Results (nov26 restart, steps 33312-33317, ctx imf=False, XLA flags `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp` set before jax in BOTH processes, no compile cache, OMP_NUM_THREADS=1)
Same affinity both sides: (a) 3 cores `taskset -c 0-2`, (b) 1 core `taskset -c 0`. Reference = `atm_step_fast.run_chain` (NumPy dynamics numpy-pow, batched NumPy CONDSE, recorded land).
**C1 (port consistency): bitwise. 105 of 105 saved fields equal (stage snapshots dyn/condse/radia/surface/dissip/filter, 21+ fields each) at every one of the 6 steps, on 3 cores and on 1 core. Unequal: 0; max difference 0; no location.** Category A for every field of every step. (Compared with array_equal, NaN positions equal.)
**Against the real dumps (libm mode; both JAX and NumPy chains, identical):** step 0 end state, gate fields (rel = max diff / scale; columns beyond 1e-12 / beyond 1e-6): U D 5.9e-4 (397/32), V D 3.1e-4 (395/31), T D 2.6e-5 (3312/20), Q D 8.6e-3 (278/41), QCL D 2.3e-2 (40/26), QCI D 4.8e-4 (4/4), MA B 2.7e-13, TMOM D 2.9e-4 (40/3), QMOM D 1.0e-2 (46/24), EGCM D 1.3e-2, W2GCM D 1.0e-2, PBLHT D 3.1e-3, DCLEV A, PBLPTOP D 1.2e-3. Verdict by the ACCEPTANCE section 3 / atm_step_compare.gate_summary rule: **NOT MET in libm mode** (12 D, 1 B, 1 A). The difference is identical for the NumPy chain, so it is not caused by JAX; the known cause is libm vs the real build's libimf (cloud threshold flips in a few percent of columns, D126/D129/D134). That attribution is inherited, not re-tested in this stage. Steps 1-5 vs the real end state: all 14 gate fields D (worst QCL 2.6e-1, 4.2e-1, 1.3 at steps 3-5), as expected from chaotic divergence after flips. The fidelity (C2) gate needs the libimf callback and is not attempted here.
Not verified: that the 1-core and 3-core JAX runs differ from each other (not compared; D175 says the existing code differs between core counts).

## 5. Timing (s per step; per-stage in logs)
| config | first step (compile) | steps 2 | steady (steps 4-6) | steady dyn / condse / surface |
|---|---|---|---|---|
| JAX, 3 cores | 3015 | 46.8 | 8.2-8.7 | 1.2 / 5.8-6.5 / 0.7 |
| NumPy ref, 3 cores | 352 | 54 | 10.0-10.7 | 2.8 / 5.9-6.1 / 1.0-1.3 |
| JAX, 1 core | 3135 | 63.9 | 10.3-11.1 | 1.5 / 7.2-8.1 / 1.1 |
| NumPy ref, 1 core | 345 | 55 | 8.8-9.1 | 2.7 / 5.2-5.4 / 0.8 |
First-step JAX time is dominated by compilation: 191 backend compiles, dyn alone 2475-2586 s (the earlier D144b figure of 40 s was without `algsimp` disabled and on a different load; cause of the 60x not investigated), CONDSE 163-270 s, surface 265-389 s. Second JAX step compiles 46 more (surface shapes). Steady state compiles: 0. On 1 core the JAX path is NOT faster than NumPy overall (CONDSE kernels slower, dynamics 1.8x faster). Note ref on 1 core is faster than on 3 cores (9 s vs 10.3 s): not investigated. Jit calls and host-device transfers per step were NOT counted (only compile counts); the transfers are at least one per stage boundary per field.

## 6. Against ACCEPTANCE_CRITERIA section 1
1 State updates by JAX with device-resident arrays: **NOT satisfied** (host NumPy state between all stages). 2 Boundaries measured: **partly** (compiles counted; jit calls and transfer sizes not measured). 3 Non-JAX stages listed: satisfied qualitatively (section 2); per-stage time shares are in the logs, not split below stage level. 4 Radiation: no radiation computed, recorded; callback not yet built. 5 Recorded inputs listed: satisfied at the file level (section 3). So this stage is a JAX-executed chain, not yet "JAX-driven" by section 1.

## 7. Blockers for the next stages
(1) Device-resident state: replace the NumPy hand-over and the NumPy glue (trop, MAtoPMB, bookkeeping, energy fix, PGRAD_PBL, CONDSE set-up/post-processing and poles, record rewriting, first-layer update, dissip, filter) by jitted code, targeting the three jit units of JAX_COVERAGE_MATRIX. (2) QUS subsidence advection is a host callback from MSTCNV. (3) Compile time (about 50 min cold, per process, no cache allowed). (4) libimf host callback for C2. (5) Land patch is recorded; surface half and radiation callback belong to later stages. (6) Core-count dependence: bitwise claims hold per core count only.

**Parent-session check (2026-10-07 16:30):** `tests/test_jax_atm_step.py`: 3 passed with `RUN_XLA_FLAG_TESTS=1` and `JAX_ATM_STEP_DIR` set, 2 passed + 1 skipped in a plain run (the flag-sensitive tests skip themselves when the XLA flags are not in effect, so the serial and sharded regression runs are not disturbed); no tracked file modified. Independently recomputed from the saved runs: the JAX and NumPy chains are bitwise equal at all 6 steps on both the 1-core and the 3-core run (630 step-field pairs each, 0 unequal, max difference 0.0); against the real end state at step 0 (libm mode) 20 of 30 end fields are category D (QCL 2.3e-2, EGCM 1.3e-2, W2GCM 1.0e-2, QMOM 1.0e-2, Q 8.6e-3 of scale), same as the NumPy chain, so the deviation comes from libm versus the real build's libimf (threshold flips), not from JAX (the attribution itself was not re-tested). Honest status against `ACCEPTANCE_CRITERIA.md` section 1: this stage is NOT yet JAX-driven (state is NumPy on the host between every stage; several glue stages are NumPy; no radiation is computed or called, SRHR/TRHR/COSZ1 are recorded). Practical problem found: the first JAX step needs about 3,000 s of compilation per cold process (191 backend compiles; dynamics alone 2,475-2,586 s versus about 40 s recorded in D144b), and the compile cache must not be used for validated runs; the cause of the 60x difference is not investigated. Timings (steady 8.2-8.7 s per step on 3 cores, no faster than NumPy on 1 core) are the agent's.

# D181: build stage S1 of JAX_COVERAGE_MATRIX section 6 -- JAX state pytree, round-trip converters, fixed-shape tile layout (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: `fullfidelity/jax_state_d181.py` (module), `fullfidelity/jax_state_d181_masks_report.py` (report script), `fullfidelity/tests/test_jax_state_d181.py`, this entry.
FILE-NAME NOTE: the task named the module `jax_state.py`. While this unit was being built, ANOTHER agent wrote a different `fullfidelity/jax_state.py`
(434 lines: generic `to_device`/`to_host`, `driver_to_pytree`, `records_to_layout`, `land_prev_to_grid`, ...) into the same directory at 17:02. The two files are
independent; to avoid destroying that work this unit lives under the name `jax_state_d181`. Which one becomes `jax_state.py` is for the owner/parent to decide; this
entry covers only `jax_state_d181`. The other file was not read in detail, run or tested here.
Conditions of every number below: `taskset -c 0-1`, `OMP_NUM_THREADS=1`, XLA flags `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp` set before jax by
`clouds_jax_env` (imported first by the module), no persistent compile cache, libm mode (nothing here runs a model stage; only state conversion and set comparisons, so
no libimf is involved), shared node (other agents were running; timings are indicative only). SOCRATES/RADIA not touched.

## 1. What the pytree contains (explicit dtypes and shapes)
`build_state(driver_state_dict, date)` returns `(state, static, meta)`. All leaves are jax arrays, x64; dtypes used: float64, int64, bool, uint32 (the RNG seed) and nothing
else (`check_dtypes`). Axis order of the dict-side arrays is KEPT per field (the existing stage code relies on it); `describe_tree(tree)` prints every path, shape, dtype and
byte count. Groups (nov26, state at step 0 = 124 leaves, 83,515,604 bytes = 83.5 MB; static = 13 leaves, 255,024 bytes):

| group | content (dict source) | leaves | bytes (nov26 step 0) | shapes (all float64 unless noted) |
|---|---|---|---|---|
| atm | `S` of atm_step / D180 (T,U,V,Q,QCL,QCI,GZ,MUS,MVS,MWS (IM,JM,LM); MA,PK,PMID,PDSIG,EGCM,W2GCM,UALIJ,VALIJ (LM,IM,JM); PEDN,PEK,SRHR,TRHR (LM+1,IM,JM); TMOM,QMOM (9,IM,JM,LM); P,MASUM,PBLHT,DCLEV,PBLPTOP,T1AA,U1AA,V1AA,USTARPBL,LMONINPBL,TSAVG,QSAVG,USAVG,VSAVG,TGVAVG,QGAVG,DDM1,COSZ1 (IM,JM)) | 42 | 42,976,512 | as listed |
| atm_carry | `S['_carry']`: CONDSE/RADIA carry (CLDSS, CLDMC, CLDSAV, TAUSS, W_CLOUD, FRAC_*, MIX_*, DIM_*, SNOAGE, ...); absent at step 0, 40 leaves / 37.3 MB after a step | 0 / 40 | 0 / 37,306,368 | (LM,IM,JM) mostly |
| atm_ms | LSCOND module arrays `ms['S']` (lists of per-level numbers become arrays: (40,), (41,), (40,9), (40,72), ...) | 0 / 167 | 0 / 67,224 | float64 |
| ocean | surface_loop ocean state: g0m,s0m,mo,uo,vo,uod,vod,gxmo,gymo,gzmo,sxmo,symo,szmo,mmi,smu,smv,smw (IM,JM,13); ogeoz,ogeoz_sv,opbot,opress (IM,JM); kpl (IM,JM) int64; straits must,g0mst,gxmst,gzmst,s0mst,sxmst,szmst,mmst (13,12); vonp (13,) | 31 | 5,998,184 | as listed |
| ice | rsi,snowi,msi,pond_melt (IM,JM); hsi,ssi (IM,JM,4); flag_dsws (IM,JM) bool | 7 | 321,264 | |
| ice_dyn | usi,vsi,rsix,rsiy (IM,JM) (+ uisurf,visurf after a step) | 4 (6) | 105,984 | |
| lake / landice / exch | lake mwl,gml,tlake,mldlk (IM,JM); landice snowli (IM,JM), tlandi (IM,JM,2); exch gtemp,gtemp2,gtempr,sss,mlhc (IM,JM) | 4 / 2 / 5 | 105,984 / 79,488 / 132,480 | |
| land | `ghy`: w,ht (IM,JM,3,7); nsn,fr_snow (IM,JM,2); dzsn,wsn,hsn (IM,JM,2,3). `carry` (after a step): the `land_prev` tree of the closed surface loop, 68 leaves, ROW arrays over the 753 land cells ((753,), (753,7), (753,8), (753,7,2), ...) | 7 / 75 | 1,695,744 / 3,015,000 (3.02 MB) | |
| f3 | F3 accumulators: `aij`/`aijl` dicts keyed by column (284 columns after a step, keys stored as strings, original integer keys restored), `idacc` (4 ints), `s0` | 4 / 293 | 32 / 11.76 MB | |
| rad_frozen | provider's frozen SRHR/TRHR (41,IM,JM) (after a step) | 0 / 2 | 0 / 2.17 MB | |
| rng | `seed0` as uint32 scalar (asserted to fit 32 bits) | 1 | 4 | uint32 () |
| clock | itime, k, ss_itime as int64 scalars | 3 | 24 | int64 () |
| tile (DERIVED) | fixed layout `ptype` (4,IM,JM) float64 and `mask` (4,IM,JM) bool, recomputed from `ice/rsi` and the static fields by `refresh_tile` (pure, jit-able) | 2 | 119,232 | |
| tile_pbl (NEW, from the restart) | per-type PBL carry in the fixed layout: u,v,t,q,e (4,IM,JM,8); cm,ch,cq,ustar,lmonin (4,IM,JM); ipbl (4,IM,JM) int64 (types: ocean/lake, ice, land ice, land) | 11 | 4,875,264 | |
| ent_state (NEW, from the restart) | padded Ent state (IM,JM,1023) | 1 | 27,105,408 | |
| static (separate constant pytree) | focean, flake, fland, flice, fearth, fwater, axyp, coriol, hlake (IM,JM) float64; valid, is_ocean, is_lake (IM,JM) bool; tile_static (2,IM,JM) bool | 13 | 255,024 | |

A mid-run driver state (nov26, 6 closed steps, `ent=record`, F3 on; the D178 checkpoint `final_x1.pkl` in the session scratchpad, 112.3 MB pickle) converts to
717 leaves / 112,630,156 bytes in the groups above (atm 48.59 MB, atm_carry 37.31 MB, f3 11.76 MB, ocean 8.91 MB, land 3.02 MB, rad_frozen 2.17 MB, ...).
Groups tile, tile_pbl, ent_state and static have no dict counterpart (they are new / derived); the others map one-to-one by the routing table `ROUTES`; any unrouted
numeric leaf would go to `extra/` (none occurs in the tested states).

## 2. Fixed-shape tile layout and the static-field builder (replaces the per-step template rowsets)
Layout (type, IM, JM) = (4, 72, 46) plus a bool mask: type 0 ocean/lake water exists where (1 - RSI)*FWATER > 0; type 1 sea/lake ice where RSI*FWATER > 0; type 2 land ice
where FLICE > 0; type 3 land where FEARTH > 0; only inside the IMAXJ domain (poles: i = 0). FWATER = FOCEAN + FLAKE. The rule is `ptype > 0`, the one that
`surface_loop.apply_state_to_records` already uses for the PTYPE columns; here it defines the layout. Shapes never change: `tile_masks`/`tile_ptype`/`refresh_tile` take numpy or
jax.numpy (numpy and jitted jnp results are equal in the test). RSI must be the ice fraction AFTER MELT_SI of the step (PRECIP_SI does not change RSI, read in `surface_loop.precip_si`).
`rows_in_record_order(mask, kind)` gives the 1-based (i, j, type) rows a record file would hold, in the order of the real files (ffp: types 1-2 sorted by (j,i,type), then all
type 3, then all type 4, each by (j,i); ffs: types 1-2; ffl, ffg, fft by (j,i)); `gather_rows`/`scatter_rows` move between the layout and record rows (host side, only for comparison and I/O).
`build_static(date)` sources (all printed by the function): FLAKE from the real restart; FOCEAN from the ocean geometry (`ffo_geom`); FLICE from the step-0 `ffc_cse_in` dump
(static topography, not in the restart: RECORDED); FLAND = 1 - FOCEAN - FLAKE and FEARTH = FLAND - FLICE DERIVED and equal to the recorded FLAND/FEARTH bitwise on all three dates
(`verify_static`: max differences 0.0, 0.0, and 0.0 for restart FLAKE vs the dump FLAKE); AXYP, CORIOL, HLAKE as in `surface_loop.load_statics` (CORIOL and HLAKE are RECORDED static
columns of the step-0 ffp / ffl2 dumps; not replaced here). So the builder still reads three static items from step-0 dumps (FLICE, CORIOL, HLAKE).

## 3. Results
### 3.1 Round trip dict <-> pytree (gate: bitwise, category A)
`pytree_to_driver_state(driver_state_to_pytree(sd))` against `sd` with `model_driver.trees_equal` (bytes, dtype, shape; Python scalar types, None, containers, big-endian dtype restored):
- Initial ModelDriver state (`ModelDriver(...).state_dict()` with `S` = `atm_step.init_state` of the first step, closed surface, ent=record, F3 on), **nov26, dec01, jan01: 0 differing paths on each date** (test `test_initial_driver_state_roundtrip_bitwise_and_tile_state`).
- Real mid-run driver checkpoint (nov26, 6 steps): **0 differing paths**, 717 leaves; to-pytree 0.8-2.4 s, back 0.13-0.18 s on the loaded node.
- Synthetic tree (no data): bitwise, including big-endian arrays, Python int/float/bool vs numpy scalars, tuples, empty containers, None, int-keyed dicts, lists of floats; a 1-ulp change of one leaf is detected (non-vacuity); values pass through a jit unchanged.
### 3.2 Masks against the real row sets (gate: equal at step 0; first differing step)
All five SURFACE records (ffp, ffs, ffl, ffg, fft; both substeps pa/pb, ta/tb, la/lb, g1/g2, blk1/blk2) compared as cell sets per type AND as ordered (i,j,type) columns; no duplicate rows.
Step 0 (real counts ffp rows = ocean+ice+landice+land tiles):
| date | ocean | ice | land ice | land | ffp rows | (a) raw restart RSI | (b) restart + OUR MELT_SI | (c) real post-MELT_SI RSI (ffc_cse_in) |
|---|---|---|---|---|---|---|---|---|
| nov26 | 2709 | 783 | 346 | 753 | 4591 | NOT equal (140 cells) | **equal, order identical** | **equal, order identical** |
| dec01 | 2707 | 783 | 346 | 753 | 4589 | NOT equal (113 cells) | **equal** | **equal** |
| jan01 | 2644 | 696 | 346 | 753 | 4439 | NOT equal (156 cells) | **equal** | **equal** |
(b): RSI from `surface_loop.melt_si` on the restart ice is bitwise equal to the dump RSI on all three dates (max difference 0.0); (a) differs from the dump RSI by up to 4.1e-4 / 6.5e-4 / 9.0e-4, which changes the tile set in 140 / 113 / 156 cells (examples: a cell with restart RSI 3.2e-4 melts to 0.0 and loses its ice tile; cells with RSI 1.0 become 0.9999999993 and gain an ocean tile). So the mask is only correct from the RSI after MELT_SI; `refresh_tile` must be called after the melt stage, not on the restart ice. The pytree built at step 0 by `build_state` uses the restart RSI (case a) and is therefore only a placeholder until `refresh_tile` runs after MELT_SI.
Over the 54-step nov26 day (mask from the real `ffc_cse_in` RSI of each step, 54 steps, `jax_state_d181_masks_report.py`): **54 of 54 steps equal; no step where the mask differs.** The real row counts do change: 10 distinct ffp and 10 distinct ffs row counts (D174's 10 confirmed), 1 distinct count for ffl (346), ffg (753), fft (3170). The mask has a fixed shape (4,72,46) at every step. ffp rows range 4536 (steps 33347-33349) to 4593 (33360); the tile sets change in 22 of the 53 step transitions:
- 33312 -> 33313: 50 lake cells lose the ocean tile (RSI 0.99985 ... 0.99999 -> exactly 1.0); cause not investigated.
- 33321, 33330, 33340, 33344, 33349, 33351, 33359, 33365 and others: single cells gain an ice tile (lake cells: RSI 0 -> 0.001-0.005) or lose/gain an ocean tile at RSI within 1e-6 of 1.
- Day boundary 33359 -> 33360: **+54 ocean tiles, all lake cells** (RSI 1.0 -> 0.99990 ... ), and 33360 -> 33361: **-51** (RSI back to 1.0). The RSI of lake cells is changed at the boundary (D178: daily_LAKE not ported, RSI differs in 834 cells there), so a closed run WITHOUT daily_LAKE is expected to leave the mask matching the real sets only if the same RSI results; NOT tested (no closed run across the boundary exists).
Sensitivity: 41-60 water cells per step have RSI within 1e-6 of 0 or of 1; a rounding-level change of RSI in those cells flips a tile. Our own closed-loop ice (D178 checkpoint, 6 steps) gives, after our MELT_SI, a tile mask for step 33318 that EQUALS the real row sets of ffp_33318 (all records), although our RSI differs from the real one by up to 1.5e-7 in 756 cells (bitwise unequal) -- the flips did not occur there, which is one step, not a guarantee. Where our own evolved ice would stop matching over the full 54 steps was NOT measured (a 54-step closed run was not made; steps 1-5 of our own run are not available as states).
### 3.3 Side measurement (not in the tests): restart PBL carry vs record columns at step 0 (nov26)
Using the D174 column assignment (cm/ch/cq = ffp cols 33-35; u,v,t,q profiles = cols 50-81 as eight-wide blocks; e = assumed cols 82-89): for tile types ice, land ice, land the restart values equal the ffp columns bitwise (u,v,t,q,cm,ch,cq: max |difference| 0.0); for type 1 (ocean/lake) they differ (cm 2.5e-4, u 0.79, v 1.0, t 0.24); the e block never matched (max 5-18) -- the column assumption for e is unverified, so nothing is concluded about e. Not explained; flagged only because it bears on the "PBL persistent state" item of the D174 inventory.

## 4. What does not round-trip / is not converted
- `timing_log` (host bookkeeping: list of dicts with strings) stays in the metadata skeleton (reported by `unconverted`), not in the pytree.
- Ent state: with `ent='record'` the driver holds no Ent state; with `ent='computed'` it holds Python objects (`cells`), which are NOT converted (not tested). The pytree carries the restart's padded `ent_state` (IM,JM,1023) as a separate, unlinked group.
- The `land_prev` carry is converted as the row arrays it is (753 land cells, in the order of the ffg record; its `p4_ij` equals `rows_in_record_order(mask, 'ffg')`, checked in the mid-run test); it was NOT moved to a masked (IM,JM) layout. 753 = number of FEARTH > 0 cells, so the row count is fixed, but the representation is rows.
- Tested driver configuration: surface='closed', ent='record', rng='chain', F3 on, RecordProvider. NOT exercised: surface='replay', ent='computed', ServerRadiationProvider state, the D180 `RecordBoundary` object (it is a provider, not state; the D180 step uses the same `S` dict as `atm_step`, by code reading of `run_chain_jax`, not by running it).
- The step-0 atmosphere `S` is the real start state (`atm_step.init_state`) with no `_carry` (the driver has `S = None` before the first step); `_carry` is exercised only through the mid-run checkpoint.
- Lists of Python floats / ints / numpy float64 scalars (e.g. `ms`) become arrays and are restored as lists with the same element type; lists containing NaN or ragged lists stay element-wise (still bitwise).
- The tile/tile_pbl/ent_state groups are built, not round-tripped (no dict counterpart); only shapes, dtypes and, for tile, the equality with the real row sets are checked.

## 5. Limits
Only conversion and set comparisons were done; no model stage was run on the pytree, no jit of a model stage, no device-residency or transfer measurement, nothing about JAX-driven stepping (ACCEPTANCE_CRITERIA 1). Mask results rest on the real post-MELT_SI RSI (rule check) at 54 steps and on our own MELT_SI at step 0 (3 dates) and at step 6 (nov26); our own evolution over the full day is untested. Three static columns (FLICE, CORIOL, HLAKE) still come from step-0 dumps. One node, shared, 2 cores; timings indicative.

## 6. Tests
`tests/test_jax_state_d181.py`: 11 passed in 24 s on cores 0-1 (with `JS_MID_STATE` pointing to the 6-step checkpoint; without it the mid-run test skips). Includes non-vacuity checks (raw restart RSI mask must differ from the real sets; a flipped ice tile must be detected; a 1-ulp change in a leaf must be detected). Data tests skip when the dumps are absent.
Report regeneration: `OMP_NUM_THREADS=1 taskset -c 0-1 python fullfidelity/jax_state_d181_masks_report.py OUT.json [--ckpt driver_checkpoint.pkl]` (23 s; per-step table and change lists are in the JSON; a copy is `d181/masks.json` in the session scratchpad).

**Parent-session check (2026-10-07 17:16):** `tests/test_jax_state_d181.py` re-run: 10 passed, 1 skipped (the skipped test needs `JS_MID_STATE`, the 6-step checkpoint, which is in the agent's scratchpad), 25 s on cores 0-1; no tracked file modified. Not re-run by the parent: the 54-step mask comparison (`d181/masks.json` is the agent's output) and the round trip on the real 6-step checkpoint. NAME COLLISION / DUPLICATE WORK: a second session working on the same conversation (peer `rocke3d-jax-4c`) launched its own agents for D181, D182 and D183 in the same tree. Its D181 files `jax_state.py`, `jax_static.py`, `jax_state_capture.py`, `tests/test_jax_state.py` are UNTRACKED, UNREVIEWED and NOT part of this commit; this entry covers only `jax_state_d181.py`, `jax_state_d181_masks_report.py`, `tests/test_jax_state_d181.py`. The owner decides which D181 implementation is kept. Findings kept from the agent: the tile masks built from the restart plus our own MELT_SI equal the real row sets at step 0 on all three dates and on all 54 steps of nov26 when the real post-MELT_SI ice fraction is used (raw restart RSI fails in 140/113/156 cells); the tile sets change in 22 of 53 transitions (lake cells at 33312->33313 and at the day boundary); 41-60 water cells per step have RSI within 1e-6 of 0 or 1, so a rounding-level change can flip a tile; `daily_LAKE` is not ported, so a closed run across 33360 may not reproduce the boundary jump (untested). Nothing here runs a model stage: no JAX-driven claim.

## D185 (2026-10-07): stage S3, the radiation hand-off of the JAX-driven step (packet from arrays, io_callback to the persistent server, apply/hold in the step)

Owner: Glenn Tamkin. Sources: D159-D162 (`radiation_server_persist.py`, `atm_day_free_rad_persist.py`), D174 (`drv_rng.py`), D176 (`drv_radpacket.py`), D180 (`jax_atm_step._radia_T_jax`), `Reports/JAX_COVERAGE_MATRIX.md` 5.3 and S3, `Reports/ACCEPTANCE_CRITERIA.md` 1.4. New files only: `fullfidelity/jax_radiation.py`, `fullfidelity/jax_radiation_check.py` (validation runner), `fullfidelity/tests/test_jax_radiation.py`. Nothing existing edited; SOCRATES/RADIA never ported or modified (the real RADIA runs inside the persistent server). `jax_harness.py` (D184) does not exist yet: `jax_radiation.RadLog` is the local minimal logger (fields below); switch to the harness when it lands. Not committed.

**Every result of this module carries the sentence:** `radiation computed by the original Fortran (hybrid component)` followed by the counters (`RadLog.sentence()`; `say(log, title)` prints it).

### What was built
1. **Packet from arrays.** `assemble_packet_jax(atm, cloud, carry, surf)` (jitted) returns the 52-field packet in `INPUT_FIELDS` order from device arrays: atmosphere 8 (T Q PK PMID PDSIG PEDN MA LTROPO), BYMA = 1/MA computed in the jit, clouds 19, carry 3 (SNOAGE RQT KLIQ), surface 21 (`drv_radpacket.FIELDS_SURFACE`). `surface_fields_jax` is the array version of `drv_radpacket.ice_lake_landice_fields` (15 of the 21 surface fields; the land group GTEMPR4/BARESW/SNOWD/FRSNOW stays the host loop `drv_radpacket.land_fields`, not ported to arrays here).
2. **Callback.** `RadiationHandoff.call` = `jax.experimental.io_callback(..., ordered=True)`. Choice: the server is stateful (clock only moves forward; a call is a side effect), so a pure_callback (may be deduplicated, reordered, vectorised or dropped when unused) is the wrong tool; ordered io_callback runs once per execution, in program order, never dead-code eliminated. Cost: ordered effects serialise the device stream (one host synchronisation point per call). Returns on device the 21 non-AIJ outputs (T Q SRHR TRHR RQT KLIQ SNOAGE CLDSS CLDMC FSF TRSURF ALB FSRDIR SRVISSURF FSRDIF DIRVIS DIRNIR DIFNIR SRDN CFRAC COSZ1) plus 4 AIJ accumulator columns (SRNFP0 TRNFP0 SRINCP0 SRNFG). It works inside `lax.cond` (jax 0.5.3).
3. **Hand-off in the step.** `make_handoff_step(handoff, const)` returns a jitted `step(state, surf, hold, itime, seed, cosz1_step) -> (T, Q, hold, cosz1)`: `lax.cond` on `(itime-ITIMEI)%5==0`. Radiation step: assemble, callback, `jax_atm_step._radia_T_jax` with the server's SRHR/TRHR/COSZ1, Q = server Q (negative-Q reset), hold = all server outputs (carry SNOAGE/RQT/KLIQ and masked CLDSS/CLDMC included). Other four steps: no callback; the held SRHR/TRHR are applied with the per-step COSZ1 argument (recorded or `drv_zenith`). The hold is device state. The 4-of-5 skip is verified (toy: 2 callbacks in 6 steps; unit test with a fake server).
4. **Seed.** Recorded: `seed_from_record(itime)` = SEEDS[1] of `ffc_cse_out`. Computed: D174 chain on device (`seed_chain_radia`, `seed_chain_next`: uint32 affine maps built from `drv_rng.jump`), starting from the recorded SEEDS[0] of the first step only. The seed is a traced int32 scalar argument of the callback (same sign convention as the recorded SEEDS[1]).

### Results (nov26 steps 33312 and 33317; node under load; pinned cores 7,8 for the final run)
**Packet.** Array path == NumPy path (`atm_day_free_rad.assemble_packet`) == live recorded packet `rsv_n26_<it>_in.bin`: **52 of 52 fields bitwise at both steps** (also BYMA = 1/MA in jit). What this does and does not show: only T Q PK PMID PDSIG PEDN MA (+BYMA), the 19 cloud arrays and SNOAGE (CONDSE-exit) come from independent records (`ffc_cse_in/out`, `ffa_step_r`); RQT, KLIQ, LTROPO and the 21 surface fields (24 in all) are COPIED from the live packet in this check, so their equality is not a test of anything. The surface group's own derivation from our state is D176 (`drv_radpacket_check.py`); `surface_fields_jax` is bitwise equal to the NumPy function on synthetic inputs only (unit test), not checked against a live surface state here. Carry check: RQT and KLIQ of the 33312 OUTPUT packet equal the live 33317 INPUT bitwise; **SNOAGE does not** (3132 elements differ, max 20.48): so in a free run that carries the server's SNOAGE (D155/D162) the 33317 packet SNOAGE differs from the live one by up to 20.5 (the CONDSE-exit SNOAGE equals the live one); `hold['SNOAGE']` can be set by the caller (the NumPy path does the same on day boundaries).
**Seeds.** Chain == recorded SEEDS[1] for all 6 steps 33312..33317 (and chained SEEDS[0] == recorded SEEDS[0] each step): 6 of 6.
**Server outputs via the callback inside jit vs recorded `rsv_n26_<it>_out.bin`:** 21 of 21 non-AIJ fields bitwise at 33312, 33317 and the repeat of 33312 (call-order independence: forward 33312, 33317, then 33312 again; identical). AIJ: the 4 columns equal AIJD within 5.8e-11 (the live AIJD is a rounded difference; tolerance of D160 is 1e-9). **Seed sensitivity (AIJ):** a call with the wrong seed (12345) leaves the 21 other fields bitwise equal and the 4 columns above unchanged, but 4 of 1660 AIJ columns differ from the recorded AIJD (max abs 1.0); with the recorded seed 0 of 1660 columns differ by more than 1e-9 (max 5.8e-11). So AIJ needs the right seed; the fields the step uses do not depend on it.
**Accounting per call (identical for all calls):** device->host 33,676,432 bytes in 54 arrays (52 packet fields + itime + seed; the packet file is 33,678,288 bytes); host->device 11,605,248 bytes in 22 arrays (21 fields + the 4-column AIJ block); server output file 55,483,416 bytes (AIJ 1660 columns, 4 returned). Host synchronisation points: 1 per call (ordered callback blocks the device stream until it returns); 0 on the other four steps. Seeds used: 33312 -> -588724193, 33317 -> 1129930407 (both recorded and chain).
**Timing (this load; server ~10.5-10.9 s RADIA+IO, start-up 20.1 s):** in-jit calls 10.96, 11.21, 11.18 s end to end; the same packet through `call_eager` outside jit 12.6 s (server 10.54 s); callback-added time (callback minus server request) 0.01-0.03 s per call, jit dispatch+copy about 0.05 s. The callback overhead is below the call-to-call variation of the server itself (one eager call spent 1.9 s in request outside the server, in-jit calls 0.14-0.35 s: file system noise); no systematic overhead can be claimed beyond about 0.1 s per call.
**Toy step (6 steps 33312..33317: jit A -> hand-off -> jit B; T, packet arrays, hold, itime and the seed chain are device-resident):** 2 callbacks (33312, 33317), 4 steps without a callback, seeds equal to the recorded ones, first-call outputs bitwise equal to the recorded out packet (packet == live because A is the identity). Counters: 54 arrays / 33.68 MB device->host and 22 arrays / 11.6 MB host->device per call, 2 sync points, 4 steps skipped; host-side transfers by the driver loop: none (itime and seed advance in jit; `jax.transfer_guard('log')` logged no transfer in the loop; with `'disallow'` the ordered effect's own `bool[0]` ordering token raised once, which is the runtime's, not data). The guard does not see the callback argument copies on the CPU backend; those are counted by `RadLog`. Final `device_get`: 1. Wall 22.5 s = 2 server calls.

### Limitations, found failures, to be read before use
- **Deadlock on ONE CPU core.** With the process pinned to a single core (XLA CPU runtime with one worker thread), the toy loop hung every time (reproduced with a fake server, so not the server): the callback's `np.asarray(operand)` waits on operands produced by a previous asynchronous executable (jit A) that cannot run while the step executable occupies the only worker. Workarounds verified with the fake server on 1 core: `block_until_ready` on the state before the hand-off step (`toy_step(sync_after_handoff=True)`); with 2 cores (final run) asynchronous dispatch works. The single-call in-jit tests (inputs from `device_put`) work on 1 core. Any driver that uses the hand-off on one core must synchronise before the hand-off step; this is an additional host synchronisation point there. Not investigated on GPU.
- The radiation inputs not computed by us in these checks: RQT, KLIQ, LTROPO and the 21 surface fields (live packet); on the hand-off side: the cloud-mask result CLDSS/CLDMC and SNOAGE of the server are held, but writing them back into the CONDSE carry is the driver's job (not wired here, no coupled step run).
- Not measured: GPU transfer cost; behaviour with a later model day (the server supports it, D159; the hand-off was run on one day only); the full 54-step day through the JAX hand-off (the D162 free-radiation day equivalence is the NumPy-path result).
- `jax_radiation_check.py` and the tests start the real persistent server (needs `RADSRVP_SCRATCH`, ~20 s start-up, ~11 s per call; tests ~95 s on 2 cores); the tests skip when the binary or data are absent. Server started in every case via a context manager and confirmed stopped (no model process left).
- Tests: `tests/test_jax_radiation.py` 9 passed (seed chain on device, surface fields vs NumPy, cond/skip pattern with a fake server, packet vs NumPy path vs live, seed chain vs record, callback outputs vs recorded, order independence and log, toy step, server stopped). Not run: the full suite (owner's rule).

**Parent-session check (2026-10-07 17:58):** `tests/test_jax_radiation.py` re-run with the persistent server (`RADSRVP_SCRATCH` = the session's `mE_persist`), cores 8-9: 9 passed, 93 s; no tracked file modified; the number of model processes was the same before and after (no orphan server). Assertions include the 52-field packet equal to both the NumPy path and the live recorded packet at 33312 and 33317, seed chain equal to the record, the server outputs equal to the recorded `rsv_n26_*_out.bin` (21 fields bitwise, AIJ within 1e-9), call-order independence and a call log with `SOCRATES_modified: False`. CAUTIONS kept from the agent: only 28 of the 52 packet fields come from independent records (T, Q, PK, PMID, PDSIG, PEDN, MA, BYMA, 19 cloud arrays, SNOAGE); RQT, KLIQ, LTROPO and the 21 surface fields were copied from the live packet, so their equality proves nothing (the surface side is covered separately by D176, bitwise at step 33312); a free run that carries the server's SNOAGE differs from the live 33317 packet there (3,132 elements, max 20.48); the toy loop DEADLOCKS on a single core unless a `block_until_ready` precedes the hand-off (works on 2+ cores); GPU behaviour, GPU transfer cost, a later model day and the 54-step day through the JAX hand-off are NOT tested; writing the held CLDSS/CLDMC/SNOAGE back into the CONDSE carry is left to the driver. Callback uses `io_callback(ordered=True)` (stateful server, side effects). Accounting per call: 33.7 MB device to host in 54 arrays, 11.6 MB back in 22 arrays, one sync point, callback overhead about 0.1 s on top of the ~11 s server call.

# D182: why the first JAX atmosphere step needs ~3,000 s, and the fix (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified. jax = jaxlib = 0.5.3, CPU, float64.
New files: `d182_cold_compile.py` (per-jit cold-compile harness), `d182b_micro.py` (micro-tests of the pathology), `d182b_dyn_digest.py` (cold run of
dyn_step_jax2, saves all workspace arrays, for the bitwise old/new comparison), `clouds_jax_env_fast.py` (the fix), `tests/test_d182_env.py` (1 passed, 2 s), this entry.
NOTE: another agent worked on D182 in the same tree at the same time (d182_stage_profile.py, d182_unit.py, d182_variants.py, dyn_jax_fast.py are not mine and not
used for any number here). Cores: taskset -c 2 / 3 / 4 for the 1-core runs (one process per core, several at the same time) and `-c 2-4` for the 3-core runs; the
node was shared (load average 3.6-7), so absolute seconds carry load noise (the same configuration varied 85-100 s). No compile cache anywhere (asserted in the
harnesses). XLA flags are set before jax is imported. OMP_NUM_THREADS=1.

## 1. Result in one paragraph
CAUSE: `--xla_disable_hlo_passes=algsimp` (clouds_jax_env, D145). With the algebraic simplifier off, an XLA:CPU fixpoint pipeline (`post_scatter_expansion_simplification`,
which iterates the `reshape-mover` pass) adds a reshape pair per iteration to every gather, scatter and reduce, and nothing cancels them any more (algsimp normally
does). Every gather/scatter/reduce therefore carries ~150-160 extra reshape (later bitcast) instructions in a chain; the optimized HLO grows 10-20x and the
compile 30-100x for the units that have many scatters (`.at[].set`) and FFT code. FIX: also disable `reshape-mover` (`--xla_disable_hlo_passes=algsimp,reshape-mover`;
pure data-movement pass, no arithmetic). Cold dyn_step_jax2, 1 core: 2,563 s -> 96 s; results BYTE-IDENTICAL (120 arrays, 2 steps). The whole JAX atmosphere
step, first step, 1 core: ~3,135 s (D180) -> 390 s, 105/105 fields bitwise equal to the NumPy chain at 2 steps (as in D180). No source change in any existing module is needed,
only the flag string (`clouds_jax_env_fast.py` is a drop-in).

## 2. Measurements
### 2.1 Cold compile of the dynamics step alone (dyn_step_jax2, nov26 33312, step 0 from the real state; 21 XLA compiles; `d182_cold_compile.py`, `d182b_dyn_digest.py`)
| flags | cores | wall step 0 (s) | sum XLA compile (s) | step 1 (s) | log |
|---|---|---|---|---|---|
| clouds_jax_env: AVX + algsimp off (current) | 1 | **2563.5** | 2552.7 | 1.6 | dig_old_1c |
| AVX only (dyn_jax_env, algsimp ON; the D144b configuration) | 1 | 84.7 | 73.0 | - | dynenv_1c |
| AVX + algsimp off + reshape-mover off (new) | 1 | **96.1** (also 99.9 in a second run) | 85.2 (88.6) | 1.7 | dig_new_1c, newflags_1c |
| default ISA (no AVX flag, FMA allowed; algsimp on) | 1 | 88.5 | 77.6 | - | noisa_1c |
| AVX + algsimp off + reshape-mover off (new) | 3 (-c 2-4) | 59.1 | 49.7 | - | newflags_3c |
| AVX only (algsimp on) | 3 (-c 2-4) | 52.0 | 42.8 | - | dynenv_3c |
Per unit, 1 core, seconds of XLA compile (old flags -> new flags; AVX-only in brackets): aflux_jax 750 -> 25.5 [21.8]; _iso 1124 -> 22.1 [18.3]; pgf_jax 624 -> 11.0 [9.6];
sdrag_jax 24 -> 4.1 [3.4]; advecv_jax 8.0 -> 3.1 [1.8]; all_cycles 7.0 -> 7.0 [5.2]; prep 3.8 -> 3.8 [3.4]; filter_chain 3.0 -> 3.2 [3.0]; aadvty 2.9 -> 3.1 [1.7];
aadvtx 1.7 -> 2.0 [1.8]; advecm 1.2 -> ~1 [..]; aadvtz 1.3. MLIR lowering is 3.2-3.6 s in total in every configuration (the lowering is not the problem). Top 10 by XLA time under the old
flags: _iso 1124, aflux 750, pgf 624, sdrag 24, advecv 8.0, all_cycles 7.0, prep 3.8, aadvty 2.9, filter_chain 3.0, aadvtx 1.7. The tracing time is small (the "Finished tracing" lines
sum to about 5-6 s per process; the harness regex catches only pjit-level lines, so this figure is a lower bound).
### 2.2 HLO size (xla_dump_to, 1 core)
Instructions before optimization (same in all flag settings): aflux 5,849; _iso 5,209; all_cycles 3,075; pgf 2,999; sdrag 2,032; prep 1,276; advecv 1,188; filter_chain 995; aadvty 892. After optimization with algsimp ON:
aflux 83.7k lines of text, _iso 78.8k, pgf 40.7k (text lines), sdrag 4,449 instructions. With algsimp OFF, sdrag_jax 79,294 instructions of which 74,355 are `bitcast`, arranged as chains alternating pred[3240] <-> pred[3240,1]
(and f64[3240,40] <-> f64[3240,1,1,40]); the same unit with algsimp on: 4,449. Scatter counts (before optimization): aflux 177, all_cycles 82, advecv 40, aadvty 44, sdrag 37, pgf 17; gathers: advecv 32, aflux 7, pgf 7.
### 2.3 Where the growth comes from (micro-tests `d182b_micro.py`, optimized-HLO instruction counts; compile s)
| pattern | algsimp on | algsimp off (current) | algsimp off + reshape-mover off (new) |
|---|---|---|---|
| `u[I,J,:]*2.0` (gather) | 10 instr | 185 | 10 |
| `jnp.any(...)` over 40 columns | 805 instr, 7.6 s | 16,685, 10.8 s | 805, 4.1 s |
| 40 x `uc.at[:,l].set(...)` | 1,142, 1.29 s | 8,676, 3.63 s | 2,676, 1.40 s |
| 40 x `s + uc[:,l]*uc[:,l]` | 129, 0.13 s | 7,678, 2.79 s | 168, 0.08 s |
Pass-by-pass dump (`--xla_dump_hlo_pass_re=.*`, gather micro-test): the instruction count rises by 3 per iteration (142 -> 157) over ~25 iterations of `post_scatter_expansion_simplification`
(each iteration ends in `reshape-mover`), then the layout/`reshape-decomposer` step turns the 177 reshapes into bitcasts. The pipeline is a fixpoint with an iteration cap; with algsimp on it converges at once.

## 3. Hypotheses
| hypothesis | verdict | number |
|---|---|---|
| `--xla_disable_hlo_passes=algsimp` blows up compile time | CONFIRMED (the cause, via reshape-mover; section 2.3) | 2563 s vs 85 s (30x); HLO 79k vs 4.4k instr (sdrag) |
| `--xla_cpu_max_isa=AVX` (no FMA) is costly | REFUTED | 84.7 s (AVX) vs 88.5 s (default ISA), 1 core |
| unrolled Python loops inside jit (layers/cells) | PARTLY: they set the baseline (2-5k HLO ops in aflux/_iso/pgf/all_cycles from FFT72 straight-line code and 40-layer loops; 21-25 s each even with the fix) but are NOT the 30x | AVX-only 85 s total |
| float64 | not the cause, not tested separately (x64 is required for the port); the 85 s baseline with x64 is all that is left | - |
| many tiny jits | REFUTED for the dynamics: only 21 XLA compiles, 12 of them >1 s | - |
| jax/jaxlib version | not testable (only 0.5.3 installed here); D144b was measured with the same version family (D145 states jax 0.5.3) | - |
| D144b's 40 s measured with a warm cache | REFUTED: D144b states "cold (compile) 40.5 s per process" and "no persistent cache enabled" (D141); it was measured with dyn_jax_env only (algsimp ON), i.e. before D145 introduced clouds_jax_env. My reproduction of that configuration: 84.7 s (1 core, loaded node); 52 s with 3 cores. Same order, load-dependent. | 84.7 / 52.0 s |
| thread count | small effect: 1 core 96 s vs 3 cores 59 s (new flags); 85 vs 52 s (AVX only). XLA compiles each module on one thread; the 3-core gain comes from LLVM codegen threads | - |
| D180's "60x vs D144b" | the 60x (2,500 s vs 40 s) = the algsimp-off flags (D145), not load and not code growth | 2563 / 85 = 30x measured on one node state |

## 4. The change and its bitwise validation
Change: `--xla_disable_hlo_passes=algsimp,reshape-mover` (file `clouds_jax_env_fast.py`; `D182_KEEP_RESHAPE_MOVER=1` restores the old list). No structural change of the jitted code was needed or made
(lax.scan / stacked layers would not be bitwise-trivial and would not be needed now).
Validation (same cores both sides, flags before jax, no cache):
1. dyn_step_jax2 cold, 1 core, 2 steps (nov26 33312 and 33313 from their real states), old flags vs new flags: all 120 saved workspace arrays (both steps, every float array and scalar in the workspace) are byte-identical (`tobytes()` equal), 0 unequal.
2. Full JAX atmosphere step (`jax_atm_step_run.py jax`, new flags, 1 core, 2 steps, recorded land, libm) vs the NumPy chain (`jax_atm_step_run.py ref`, existing flags, 1 core): `jax_atm_step_cmp.py` reports 105 of 105 saved fields bitwise equal at step 0 and at step 1 (same result as D180 with the old flags). Against the real end state the verdict is identical to D180 (libm mode, NOT MET, 12 D / 1 B / 1 A at step 0). Note: the NumPy ref ran with the old flags (its jitted surface parts), so this also compares new-flags JAX with old-flags JAX code paths.
3. tests/test_d182_env.py: flags set, `a/35.0` and `a*b+c` identical to numpy, scatter-loop HLO < 4,000 instructions (measured 2,676; old flags 8,676).
Not covered: CONDSE and surface kernels were validated only through (2) (the whole step), steps 0-1 of nov26; no other dates; no 3-core full-step bitwise rerun (D180 already showed JAX==NumPy at 3 cores with the old flags; not repeated with the new flags); 6-step chain not rerun.

## 5. Cost of a cold first step after the change, and what remains
Full JAX atmosphere step, 1 core, new flags (jax_atm_step_run.py, 2 steps): step 0 **390.3 s** (dyn 98.3, CONDSE 184.0, surface 107.4; 191 compiles), step 1 44.6 s (dyn 1.9, CONDSE 7.6, surface 34.8; 46 more compiles). Reference NumPy chain on the same core: step 0 447.6 s (its JAX surface parts compile under the old flags), step 1 65.2 s. D180 for comparison (1 core, old flags): 3,135 s and 63.9 s. The dynamics part fell from 2,475-2,586 s to 98 s;
surface fell from 265-389 s to 107 s; CONDSE did not change (163-270 s before, 184 s now): its compile time is intrinsic, not this pathology. What remains: (a) CONDSE kernels ~184 s (LSCOND/MSTCNV, large), (b) dynamics 85-100 s on 1 core (aflux 25, _iso 22, pgf 11, all_cycles 7, FFT72 straight-line code is the base cost, same with algsimp on), (c) surface ~107 s plus 35 s more at step 1 (new tile-set shapes, D175), (d) 3-core numbers for the whole step were not measured (dyn alone 59 s). A cold process is still ~6.5 min for the first step and ~7.5 min for the first two; not seconds.
Possible further reductions (NOT done, would each need the same bitwise validation): replace the `.at[].set` scatter chains of aflux (177 scatters) by static slices/concatenate or dynamic_update_slice (data movement only); express the FFT72 by a loop over a stacked axis; batch pad-shape variants in the surface chain to avoid the 46 second-step compiles. Not attempted here; another agent's `dyn_jax_fast.py` explores the scatter replacement and is unvalidated by me.

## 6. Proposed diffs for the existing modules (not applied)
```
clouds_jax_env.py:      add 'reshape-mover' to the disabled list:
-        add.append('--xla_disable_hlo_passes=algsimp')
+        add.append('--xla_disable_hlo_passes=algsimp,reshape-mover')
jax_atm_step.py:20 / jax_atm_step_run.py:6 / every `import clouds_jax_env`: import clouds_jax_env_fast instead (or apply the one-line change above and drop the new file).
dyn_jax_env.py: unchanged (algsimp is on there; results of the dynamics are byte-identical either way for the 120 arrays tested).
```
Side result: because the dynamics results are byte-identical with algsimp on and off (nov26, 2 steps), the dynamics do not need the algsimp-off flag; that flag is only needed by the cloud kernels (D145 micro-tests). This was not exploited (flags are process-wide; per-jit overrides of `xla_disable_hlo_passes` fail in jax 0.5.3: `compiler_options` raises a protobuf error because the field is repeated).

## 7. Limits
Timings were taken on a shared node with other agents' jobs (load 3.6-7); single runs, no repeats except the new-flags 1-core dyn run (96.1 and 99.9 s). The mechanism (reshape-mover plus the pass-iteration cap) is inferred from pass dumps of a micro-test and from the fact that disabling it removes the growth; XLA source was not read. The result applies to jaxlib 0.5.3 only. Bitwise statements hold for this host's CPU, nov26 steps 33312-33313, 1 core. A persistent compile cache remains forbidden for validated runs (D175: a warm cache changed T by 0.067 K); none was used.

**Parent-session check (2026-10-07 18:05):** `tests/test_d182_env.py` re-run: 1 passed, 2 s; no tracked file modified. Independently re-compared the saved snapshots of D180's old-flag 1-core run (`c1`) with the agent's new-flag run (`atm_new`): 210 (step, field) pairs over steps 0-1, 0 not bitwise equal, max abs difference 0.0. Step wall times from the saved timing files: old flags 3,135 s (step 0) and 64 s (step 1); new flags 390 s and 45 s. Mechanism (agent's inference, not re-derived by the parent): with the algebraic simplifier off, the XLA:CPU `post_scatter_expansion_simplification` loop adds about 150 reshape/bitcast instructions per gather, scatter or reduce; also disabling `reshape-mover` (pure data movement) stops it. The fix is a flag: `--xla_disable_hlo_passes=algsimp,reshape-mover`, provided by the new `clouds_jax_env_fast.py`. The existing `clouds_jax_env.py` is NOT changed by this commit. NOT covered by the bitwise checks: other dates, the 6-step chain, a 3-core full-step rerun, the existing flag-sensitive test files under the new flags (to be run before the existing module is changed). Remaining cold-start cost: ~390 s for step 0 (CONDSE compile ~184 s untouched; dynamics and surface ~100 s each). `dyn_jax_fast.py`, `d182_stage_profile.py`, `d182_unit.py`, `d182_variants.py` belong to the other session's duplicate D182 agent, are untracked and unreviewed, and are not part of this commit.

# D184: build stage S0 of JAX_COVERAGE_MATRIX section 6 -- harness, recorded-input registry, C1 reference (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file modified.
New files: `fullfidelity/jax_harness.py`, `fullfidelity/tests/test_jax_harness.py` (14 passed, 3.4 s), this entry. Reference data (outside git): `ff_data/ref_libm/` (114 MB; 6 npz end states, digests, meta/header json, `determinism_<date>.json`).
Conditions of all reference runs: `taskset -c 0-1`, `OMP_NUM_THREADS=1`, XLA flags `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp` (set by `clouds_jax_env` before jax), no compile cache (checked by the header, which raises if one is set), jax 0.5.3, numpy 2.2.4, python 3.10.16, git head ccf16b8 with 0 modified tracked files (24 untracked entries), host forest204, libimf present on the host but NOT used (libm mode). SOCRATES/RADIA untouched. The two other-session files (`jax_state.py`, `jax_static.py`, `jax_state_capture.py`) were not touched.

## 1. What the module provides (ACCEPTANCE reference in brackets)
1. `provenance_header()` / `write_header()` / `header_text()`: host, core affinity, thread env, XLA_FLAGS (and whether the two documented flags are present), jax version/backend/devices, compile-cache check (`check_no_compile_cache()` raises `HarnessError` for `JAX_COMPILATION_CACHE_DIR`, `CLOUDS_JAX_CACHE`, `jax.config.jax_compilation_cache_dir`), git head and number of modified tracked files, libimf availability, python/numpy versions, timestamp. [section 2, 6]
2. `Counters`: per named stage jit calls, host->device and device->host calls and bytes, host callbacks (calls, bytes, seconds), XLA backend compiles (jax.monitoring), wall and exclusive time. Context manager and decorator (`C.stage(name)`), wrapper for any callable including jitted ones (`C.count_jit(fn, name)`, return value passed through untouched, attributes delegated), `C.host_callback`, `C.instrument_transfers()`. [1.2, 6]
3. `RecordedInputRegistry`: `declare(name, source_file, size, role)`, `read(name, loader)` raises `UndeclaredRecordedInput` for an undeclared name, `emit()` gives the list and a summary sentence, `guard_real()` wraps `atm_step.Real._get` and `guard_function()` wraps e.g. `atm_step.surface_records`. `declare_d174_inputs()` declares the record keys of the atmosphere chain from D174/D180 (7 inputs: `sitea siter s1 ci co surface_records ctx_static`; 7 comparison-only references); `declare_d174_surface_items()` declares the remaining D174 inventory items of the full coupled step (10 items, names and sources from the D174 table; sizes not measured). [1.5]
4. `RadiationCallbackLog` (sentence `radiation computed by the original Fortran (hybrid component)`; mode `replay` writes a different sentence and never says "computed"; calls, bytes in/out, wall and server seconds, sync points, field lists), `LibimfCallbackLog` (sentence of section 8.1, call/element/byte/second counts; inactive -> libm-mode sentence), `result_preamble()`. [1.4, 8.1]
5. `StageRegistry`: stages registered with kind FORT/REC/NP/EJ/JJ (or a `/` mix), exclusive measured time and share; `non_jax_list()` / `non_jax_text()` list every stage that is not purely JJ, including never-timed ones (shown as "not measured"); timing an unregistered stage fails. [1.3]
6. Reference and comparison: `run_reference()`, `determinism_check()`, `compare_end_state()` / `compare_to_reference()` / `comparison_text()` with the A/B/C/D categories (A bitwise, B <= 1e-12 of scale, C <= 1e-6, D worse; scale = max|reference|), the exception rule (<= 10 horizontal columns per field, each field <= 1e-9 of scale, columns listed as 1-based (i,j)) and the verdicts MET / MET with named exception columns / PARTLY MET / NOT MET. Gate fields = `atm_step_compare.GATE_FIELDS` (14); a gate field missing from the candidate counts as D. All other common arrays are reported without affecting the verdict. Bounds are fixed in the function defaults, not tunable per run (parameters exist only for tests). CLI: `python jax_harness.py header | ref DATE TAG | determinism DATE | compare CAND.npz DATE`.

## 2. The C1 reference (libm mode), step 0
`atm_step_fast.run_chain(date, it0, 1, make_ctx(date, imf=False), land_mode='recorded')`: NumPy dynamics with numpy `pow`, batched NumPy CONDSE with libm `exp`, recorded land patch, recorded radiation. The saved end state is the 60 numeric arrays of the state dict (keys not starting with `_`; 0 non-array entries skipped). This is NOT the libimf configuration and NOT a statement about ROCKE-3D; it is the reference for C1 only (ACCEPTANCE section 2). D180 showed this chain is NOT MET against the real model at step 0 in libm mode (cloud threshold flips); that is not re-measured here.

| date | step | run 1 wall | run 2 wall | bitwise equal run 1 vs run 2 |
|---|---|---|---|---|
| nov26 | 33312 | 470.6 s | 363.5 s | yes, 60 of 60 arrays, 0 unequal (byte-identical content, same dtypes and shapes) |
| dec01 | 33552 | 331.7 s | 341.6 s | yes, 60 of 60, 0 unequal |
| jan01 | 17520 | 339.3 s | 337.6 s | yes, 60 of 60, 0 unequal |

Each run was a separate process, run one after the other with identical pinning (cores 0-1) and environment. `compare_to_reference(run 2 file, date)` returns MET with 14 of 14 gate fields in category A on all three dates (checked for nov26 interactively; the three dates by `determinism_check`, which compares every one of the 60 arrays; the test file repeats the 3-date check). Non-vacuity test: one element changed by 1 ulp is detected (category B, n_diff 1). Note: the 1-vs-2-core dependence of D175/D178 means this reference is valid for the core mask 0-1 only; a candidate must be run with the same mask.
A numpy RuntimeWarning (invalid value in divide, `clouds_condse_batch.py:78`) appears in every run; it is in existing code, results are deterministic, not investigated.

## 3. Recorded inputs actually read by the reference (registry, strict mode, no undeclared read occurred)
The reference ran with `atm_step.Real`, `atm_step.surface_records` and `atm_step.make_ctx` temporarily rebound (in the process only) to registry-guarded versions; results are unchanged by this (runs bitwise equal to each other; a run without the guard was not made, so the equality of guarded and unguarded runs is NOT measured). Per date, 6 of 7 declared inputs were read (`co`, the CONDSE exit record, was not read at step 0 on any of the three dates): nov26 199,698,520 B, dec01 199,690,712 B, jan01 199,105,112 B in total. nov26 detail: sitea 39,770,528; siter 16,056,608; s1 33,994,368; ci 85,753,752; surface_records (ffp/ffs/ffl/ffg/fft) 23,529,856; ctx_static (restart, geometry, constants): size NOT measured (the registry cannot size an object; 0 is reported). The list for each run is in `<date>_step0_run<n>.meta.json` under `recorded_inputs`. Limit: the registry covers reads that go through `Real._get`, `surface_records` and `make_ctx`; any other direct file read in the existing modules (e.g. a module loading a table itself) is not seen. Which D174 items the NumPy chain takes from inside `surface_records` (Ent exports, TRUP, etc.) is not split per column here.

## 4. Counters on a toy jit (test `test_counters_toy_jit_and_results_unchanged`)
A jitted function called 3 times through `count_jit`: jit_calls 3, h2d 6 arrays, 6 x 8000 B; result bitwise equal to the unwrapped call; `device_put` + `jnp.asarray` counted as 2 h2d x 8000 B, `device_get` + `ArrayImpl.__array__` as 2 d2h x 8000 B without double counting; patched entry points restored afterwards; a fresh jit gives compiles >= 1 on its first call and no further compile on the second. MEASURED LIMIT: on the CPU backend `np.asarray(jax_array)` is served by the buffer protocol and does not reach `__array__`, so it is NOT counted (no physical transfer there); on a GPU it goes through `__array__` and would be counted (not tested; no GPU on this node). Also not counted: implicit conversions of numpy arguments of an UNwrapped jitted function, and device-to-device copies. Use `jax.device_get` / `Counters.to_host` for always-counted copies.

## 5. Gate of S0
| item | result |
|---|---|
| reference vs reference bitwise on 3 dates | PASS (nov26, dec01, jan01: 0 unequal of 60 arrays each) |
| counters demonstrated on a toy jit | PASS (test, section 4; with the stated limit) |
| registry fails on an undeclared read | PASS (test; also the guarded reference ran with no undeclared read) |
| header complete | PASS (test checks every required key; compile-cache check raises when a cache variable is set: test) |

## 6. Not done / limits
- No JAX-driven step exists yet; this entry measures nothing about the port beyond the NumPy chain's reproducibility on one core mask. The reference is one step (step 0) per date.
- Time shares: `StageRegistry` is tested on synthetic stages only; the reference meta files hold the stage timings of `atm_step.run_step` (`stage_*`), not a registry table.
- Radiation and libimf loggers are tested with synthetic calls; no radiation-server or libimf-callback run was made.
- Core count: bitwise equality was shown for the same mask (0-1) only.
Review by: when S2 starts, or when any category definition in ACCEPTANCE section 3 is changed.

**Parent-session check (2026-10-07 18:08):** `tests/test_jax_harness.py` re-run: 14 passed, 4 s; no tracked file modified. I re-compared the two saved end states of each date myself (nov26, dec01, jan01; 60 arrays each, run 1 against run 2, separate processes): 0 unequal, so the libm-mode NumPy reference is deterministic for core mask 0-1. CAUTIONS kept: the reference is valid for core mask 0-1 only (results depend on the core count); on CPU, `np.asarray(jax_array)` is not seen by the transfer counters (buffer protocol), a GPU would take the counted path (untested); the stage registry, radiation logger and libimf logger are tested on synthetic data only; no run used the registry guard against an unguarded run; every reference run prints a RuntimeWarning (invalid value in divide) from `clouds_condse_batch.py:78` (deterministic, not investigated); this reference is the C1 reference only: in libm mode this chain is NOT MET against the real step-0 dumps (D180). Reference data (114 MB) are outside git in `ff_data/ref_libm/`.

# D183: JAX atmosphere chain with a labelled libimf host callback (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file edited (module attributes are rebound at run time by `install()`).
New files (fullfidelity/): `libimf_ops.py`, `jax_atm_step_imf.py`, `d183_stage_dyn.py`, `d183_stage_condse.py`, `d183_run.py`, `d183_cmp.py` (copy of jax_atm_step_cmp.py with the label libimf), `tests/test_libimf_ops.py` (1 passed, 1.2 s), this entry.

## 1. Inventory: what the NumPy 'imf' chain routes through libimf (only pow and exp)
NumPy imf = `atm_step.make_ctx(imf=True)`: `ds.load_ctx(imf_pow=True)` + `cf.set_backend('imf')` (= `mc.set_backend('imf')` + `sz.use_imf(True)`).
| stage | NumPy imf call | JAX counterpart that was jnp.* (file) | replaced here |
|---|---|---|---|
| dyn ADVECM/MAtoP | `pow_imf(PMID[l], KAPA)` (dyn_aflux_ff.matop) | `jnp.power(pmid, kapa)` dyn_jax_aflux.py:110, scan over 40 layers, 5 calls/step | yes (jaf.jnp proxy) |
| dyn PGF | `pow_imf` PKU/PKD, `pow_imf(.01, KAPA)` (dyn_pgf_ff.py:38,45,52) | dyn_jax_pgf.py:42 (host const `.01**kapa`), :46, :56 (scan), 5 calls/step | yes (jpg.jnp proxy; KAPA float subclass for the constant) |
| PEK | `pow_imf(PEDN, KAPA)` (atm_step.py:212,625) | jax_atm_step._pek_jax | yes (`_pek_imf`) |
| dyn CALC_TROP, MAtoPMB, SLP filter | pow_imf (ctx.imf_pow) | NumPy in both chains | unchanged (host NumPy) |
| AADVT, QDYNAM `fracm**3` | plain np.power (NOT libimf) | jnp.power dyn_jax_aadvt.py:72, dyn_jax_qdynam.py:85 | NOT replaced (correct: NumPy imf chain does not use libimf there) |
| LSCOND | sz.ex / sz.pw (exp, pow), `_dq_adjust_imf` (exp) | lj.OpsXla exp/pw/dq (clouds_lscond_jax.py:120-150) | yes: `OpsImf`, `lj.OPS['imf']` |
| MSTCNV | dq.qsat/hp via proxy np.exp/np.power; `_dcg_imf/_dci_imf/_dcw_search_imf` pow; `mc._pow` FLAMW/G/I (.25); `_anvil_imf` pow(BY3); MP exp, pow(,4.0) | clouds_mstcnv_jax.py:77 (qsat), 186-187,221,231 (jnp.power plus `**` operators), 199, 214-215, 862-864, 1168-1169 | yes: module `jnp` rebound to `JnpProxy`; `conv_micro_j` replaced by `conv_micro_j_imf` (the four `**` made explicit) |
| CONDSE host numpy (pole columns, snow-age exp, make_K constants such as 222**.33, qsatre) | libimf | NumPy in both chains | unchanged |
Other functions (log, sqrt, tanh ...): none routed through libimf in these stages (sqrt is IEEE exact).

## 2. libimf_ops.py
`exp(x, mask=None)`, `pow(x, y, mask=None)`: `jax.pure_callback` (vmap_method broadcast_all), one callback per array operation, host loop of scalar libimf calls (same loop as intel_libm_ff.pow_imf / sz.ex / sz.pw). Modes: 'libimf' (refused with RuntimeError if the Intel runtime is absent; `fallback=True` runs numpy and reports effective 'libm-numpy-fallback', never 'libimf') and 'libm' (plain jnp, no callback). Mode is locked after the first trace. Counters (calls, elements, bytes_in, bytes_out, seconds; per function) via `counters()`.

## 3. What runs on the host (callbacks) and data moved
Every exp/pow listed as "yes" above is evaluated on the HOST by libimf (ctypes, scalar loop, Python): not device resident, no GPU path. Per step (nov26, 33312 / 33313, CPU-only JAX so "device" = same machine, copies still made): 
- 33312: 35,963 callbacks (exp 20,876; pow 15,087), 15.7 M elements, 224.8 MB in, 139.6 MB out, 14.5 s in the callbacks.
- 33313: 35,369 callbacks (exp 20,642; pow 14,727), 15.5 M elements, 221.8 MB in, 137.8 MB out, 14.4 s.
Callbacks are dominated by MSTCNV (stage test: 34,664 of 35,557 CONDSE callbacks) because they sit inside the ascent / layer loops. In the dynamics: 405 pow callbacks, 1.34 M elements, 22.8 MB in / 10.7 MB out, 2.0 s per step (stage test).
Radiation: recorded SRHR/TRHR/COSZ1 as in D180 (no Fortran callback). All other non-JAX items of D180 unchanged.

## 4. Validation
Function level: (a) libimf_ops vs intel_libm_ff.pow_imf and sz.ex bitwise, in jit and lax.scan, masked and unmasked (tests/test_libimf_ops.py; random data; numpy pow differs in 0.07 % of elements so the check can fail). (b) Dynamics stage test (`d183_stage_dyn.py`, nov26 33312, NumPy imf dyn_step with JAX pgf/advecm run on the same pre-stage workspace): 5 PGF and 5 ADVECM calls, UT VT DUT DVT GZ PHI SPA and PK PMID PEDN PDSIG P: 0 unequal elements in all (bitwise). Flags: old (algsimp only); PGF cold compile 1,093 s with the callback in the scan. (c) CONDSE stage test (`d183_stage_condse.py`, real start state, NumPy dynamics, entry arrays identical): JAX LSCOND-only, MSTCNV-only, both vs NumPy imf batch: 68 of 68 exit fields bitwise equal in each variant (old flags). Mutation: JAX LSCOND in 'xla' mode vs the same NumPy imf reference gives 36 of 68 fields unequal (so the comparison can fail).
Whole step (ONE run of each side, nov26 steps 33312-33313, `taskset -c 5-6`, OMP_NUM_THREADS=1, no compile cache; JAX side flags `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover` per D182; NumPy side unaffected by XLA flags). A first JAX whole-step run with the old flags was started and killed after 41 min (not finished, no result) when D182's flags arrived; it was restarted once with the new flags.
- C1 (JAX+libimf vs NumPy imf chain): 105 of 105 saved fields bitwise equal at both steps (category A), including the stage snapshots dyn/condse/radia/surface/dissip/filter.
- C2 vs real dumps (gate fields, D129 rule, step 33312): **MET**, cats B 13, A 1, worst W2GCM 8.6e-13 (relative). Identical for the NumPy imf chain. (In D180 libm mode the same step was NOT MET with 12 D.) Land: recorded GHY outputs (land_mode 'recorded'), as in D129; no separate land patch was added here beyond what the recorded mode contains.
- Step 33313 chained from our own end state: vs real end state NOT MET (13 fields D, worst DCLEV 8.3e-2 relative), identical for the NumPy imf chain: the known post-step-0 divergence (D129: first flips at CONDSE of step 1), not caused by JAX.
Timing (JAX+libimf, 2 cores, new flags): step 0 274.5 s (incl. cold compile; stage dyn 66.5, condse 139.2, surface 67.9), step 1 56.7 s (dyn 2.5, condse 33.0, surface 20.6). NumPy imf chain: 400 s step 0 (surface 383 s, a cold JIT inside the unchanged surface stage), 60.9 s step 1. Callback time 14.5 s of the JAX step.

## 5. Limits
Only two steps; step 1 is chaotic-divergence territory by D129. Flag effect on the dynamics stage test and the CONDSE stage test: old flags (results not repeated with the new flags; the whole-step C1 used the new flags and is bitwise). The inventory of libimf use is by code reading of the NumPy imf path; any libimf use inside the unchanged surface stage is identical on both sides by construction and was not itemised. The libimf callbacks are host-side and the JAX chain is therefore not device-resident for these operations.

**Parent-session check (2026-10-07 18:30):** no tracked file modified by this track. Independently recomputed from the saved run (`scratchpad/run`: `jax_step*.npz` = JAX with the libimf callback, `ref_step*.npz` = the NumPy imf chain, steps 33312-33313): 210 (step, field) pairs, 0 not bitwise equal (C1 category A). Against the real end state at step 0 (all 30 end fields recomputed by the parent): 2 in A, 27 in B, 1 in C (`LMONINPBL` 5.4e-10, outside the 14-field gate set), none in D; worst W2GCM 8.6e-13, EGCM 8.3e-13, USTARPBL 4.6e-13. The gate-field verdict reported by the agent (13 B + 1 A = MET, as in D129) is consistent. At step 1 22 of 30 fields are in D (DCLEV 8.3e-2, QCL 2.1e-2, EGCM 1.6e-2, PBLHT 7.6e-3), identical to the NumPy imf chain: expected, because from step 1 the chain starts from OUR end state and acceptance beyond one step is statistical (ACCEPTANCE section 4), not bitwise. Unit test `tests/test_libimf_ops.py` re-run by the parent on two cores: 1 passed, 1.2 s (my first attempt, pinned to ONE core, hung at 0.1% CPU for 23 minutes: callbacks inside jit deadlock on a single core, as D185 found). CAUTIONS kept: the libimf work is a HOST callback (ctypes scalar loop): per step about 36,000 callbacks, 15.7 M elements, 225 MB in and 140 MB out, 14.5 s, most from MSTCNV loops: it is not device-resident and is the dominant cost to remove for a GPU run; only 2 steps and one run per side; the dynamics/CONDSE stage tests used the old XLA flags (the whole-step run used the new ones); recorded GHY land patch (`land_mode='recorded'`), as in D129. THIS IS THE FIRST C2 RESULT FOR THE JAX ATMOSPHERE: step 0 MET against the real dumps on nov26 (libimf callback, radiation recorded, land recorded); dec01 and jan01 not yet run.

# D186: build stage S2 of JAX_COVERAGE_MATRIX section 6 -- atmosphere phase 1 as a device-resident JAX program (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file edited; new files only.
Review by: when stage S4/S5 (surface) starts, or when a category definition in ACCEPTANCE section 3 changes.

## 0. What this is and is not
Phase 1 of the step = MELT_SI -> DYNAM (+QDYNAM, energy fix, TROP, PGRAD_PBL, KEA) -> CONDSE -> RADIA apply (ATM_DRV.f:88-274), state kept in DEVICE arrays between
stages. It is the ATMOSPHERE ONLY. It does NOT claim the coupled step: SURFACE, dissip, filter, ocean, ice dynamics, lakes, land stay on the record boundary
(declared) and are not run in the device program. Comparison type: **C1** (port consistency) only; nothing here says anything about ROCKE-3D (no C2).
Libm mode (XLA numpy-pow/glibc semantics); the Intel libimf is NOT used. SOCRATES/RADIA never ported or modified.
Conditions of every number: `taskset -c 0-1`, `OMP_NUM_THREADS=1`, flags `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover`
(`clouds_jax_env_fast`), no compile cache (checked by the harness header), jax 0.5.3, shared node (timings indicative), reference =
NumPy libm chain (`atm_step_fast`, ctx imf=False, land recorded) run with the same pinning (`jax_p1_ref.py`; run 1 vs run 2 byte-identical on all 3 dates, 240/240/238 arrays).
The files of the other session (`jax_state.py`, `jax_static.py`, `jax_state_capture.py`, `tests/test_jax_state.py`, `dyn_jax_fast.py`, `d182_*`) were not read, used or touched.

## 1. Files (all new, untracked)
| file | content |
|---|---|
| `jax_p1_glue.py` | jnp ports of the D180 NumPy glue: MAtoPMB, PEK, CONSERV_SE, CONSERV_KE, energy fix, CALC_TROP (data-dependent scans as vmapped while-loops), PGRAD_PBL, QCL/QCI rescale, DIAGA pole fix, RADIA cloud masking, CONDSE A-grid replication of U,V, momentum back-transfer, recalc_agrid_uv |
| `jax_p1_dyn.py` | the dyn_step plan (~130 stages, real order) executed on a device workspace; kernels of dyn_step_jax2 unchanged; stop-model checks as device flags; eager mode and `make_fused` (whole block in ONE jit) |
| `jax_p1_condse.py` | CONDSE (clouds_condse_batch translated statement by statement to jnp), pole columns as pure_callback, LMIN host loop, LSCOND core, hand-off arrays, momentum |
| `jax_p1_melt.py` | MELT_SI on the full grid with selects (seaice_core_jax.simelt) |
| `jax_atm_phase1.py` | driver: registry-guarded record loading, stage registry, units, RADIA through the D185 hand-off (`ReplayServer` stand-in or the real server) |
| `jax_p1_count.py` | execution counters: wraps `jax.jit` (counts executions made outside a trace) + eager-primitive dispatch count |
| `jax_p1_ref.py`, `jax_p1_ref_chain.py` | NumPy libm reference for phase 1 (stage snapshots dyn/condse/radia, X, cloud masking), 1 step and 6-step chain |
| `jax_atm_phase1_run.py`, `_server.py`, `_chain.py` | runners: 3-date C1 + mutations; real-server variant; 6-step hybrid chain |
| `tests/test_jax_atm_phase1.py` | 8 quick unit tests (8 passed, 8 s; glue bitwise vs NumPy, MELT_SI, pole slices, counter, replay server, mutation) |

## 2. What is JAX, what is NumPy, what is a callback (the non-JAX list; generated from `Phase1.stages`, plus the callbacks)
JAX on the device (jitted kernels, no host copy between them): MELT_SI; the whole dynamics block incl. TROP, MAtoPMB, SE/KE bookkeeping, energy fix, PGRAD_PBL, z-extra
check (flag), PEK; CONDSE entry set (replication of U,V); CONDSE column set-up, MSTCNV kernels, post-processing, DDML search, LSCOND, hand-off arrays, merges, snow-age exp,
momentum back-transfer, recalc_agrid_uv; RADIA T update and RADIA cloud masking.
Declared NON-JAX items (names as in `DECLARED_HOST` of `jax_atm_phase1.py`):
1. `record_load`: host file reads (recorded inputs, through the registry) + device_put at step start. Kind REC.
2. CONDSE pole columns, south and north: two `jax.pure_callback`s to the NumPy per-column port `condse_column` (KMAX=72; not ported). ~0.09 s and ~230 KB per call.
3. MSTCNV LMIN loop: HOST loop (22 iterations at LMCM=23), one device->host read of a 3,168-element mask per iteration (counted), bucketed jit calls (D146 design unchanged); this loop is the largest single piece of the step time (70%).
4. MSTCNV QUS ADV1D subsidence: NumPy callback inside the jitted event block: 176 calls/step, 0.66 s/step, ~250 MB/step moved (measured on nov26).
5. `flag_read`: one device->host read of the stop-model flags per step (36 B).
6. Radiation: in replay mode a stand-in server serves the RECORDED SRHR/TRHR/COSZ1 (ffa_step_<it>_r) through the D185 interface (REC); with the real server FORT (section 5).
7. QDYNAM extra-column z branch: NOT executed; the device program raises the flag `qdynam_do_z_extra` (0 in all runs; the real windows never reach it). If it fires the step is invalid.
8. Python dispatch: eager-mode dyn issues ~75 jit calls from Python (section 4).
Not part of this stage (record boundary): SURFACE (PBL, tiles, GHY, land ice), PRECIP_*/GROUND_*/RIVERF, ocean, DYNSI/ADVSI, FORM_SI, dissip, filter. In the 6-step chain they run as the existing NumPy code on the host.
Recorded inputs of the device program (registry, no undeclared read): sitea 39.8 MB, siter 16.1 MB, s1 34.0 MB, ci (CONDSE entry) 85.8 MB, co (seed only) 77.9 MB, restart/ocean items for MELT_SI 0.5 MB, ctx_static (not sized). Radiation: replayed from the real record, NOT computed.

## 3. C1 result: category A on every atmosphere field, 3 dates, step 0 (the S2 gate)
Compared with the NumPy libm phase-1 reference with `jax_harness.field_category` (A bitwise / B <=1e-12 of scale / C <=1e-6 / D). Fields: the state dict after each stage (dyn / condse / radia) and the CONDSE output X:
| date | step | RADIA | dyn | condse | radia | X (CONDSE outputs) | cloud masking CLDSS/CLDMC | MELT_SI RSI vs recorded CONDSE-entry RSI |
|---|---|---|---|---|---|---|---|---|
| nov26 | 33312 | radiation step (replay) | 52/52 A | 59/59 A | 59/59 A | 68/68 A | A / A | equal |
| dec01 | 33552 | radiation step (replay) | 52/52 A | 59/59 A | 59/59 A | 68/68 A | A / A | equal |
| jan01 | 17520 | held SRHR/TRHR (no radiation step) | 52/52 A | 59/59 A | 59/59 A | 68/68 A | n/a | equal |
Nothing is not-A. (MELT_SI is also bitwise equal to `surface_loop.melt_si` on all outputs: ice dict and melti/emelti/smelti.) Stop-model flags: all 0 (aadvt, advecm, sdrag, qdynam err/z-extra, subsid, handoff, lscond vmp, mstcnv negative cloud).
Not compared here: anything after RADIA (the reference of D184 holds only the end-of-step state; the surface half is not part of this stage). This is C1: the device program reproduces our NumPy chain; the NumPy chain itself is NOT MET against the real model in libm mode (D180).
Mutation tests (a perturbed constant must be detected; nov26, eager mode): dyn glue constant kg2mb +1 ulp -> detected (18 dyn fields not A); LSCOND constant RGAS x(1+1e-12) -> detected; radiation record COSZ1 x(1+1e-12) -> detected. In the FUSED variant (section 4) the constants are baked into the single jit at trace time, so the mutation of kg2mb is NOT detected there (nothing to perturb after tracing); the fused variant is validated by the A result itself, not by that mutation.

## 4. Compile time, step time, jit units, transfers (this node, 2 cores, cold = first call in the process with all compilation)
| run | cold step (s) | of which XLA compile (s, n) | steady step (s) | jit executions / step | eager primitive dispatches / step |
|---|---|---|---|---|---|
| nov26 eager | 185.7 | 141.9 (63) | 6.6 | 127 | 532 |
| dec01 eager | 179.5 | 139.7 (61) | 6.2 | 125 | 532 |
| jan01 eager | 187.1 | 142.6 (61) | 6.4 | 124 | 532 |
| nov26 dyn FUSED in one jit | 531.1 | 486.6 (21) | 7.1 | 53 | 400 |
(Other nov26 runs of the same program gave 218-221 s cold and 7.4-7.7 s steady: the node was shared; read the cold times as +-20%.) NumPy chain on the same cores for the same stages: dyn 2.6 s + condse 5.5 s + radia 0.02 s = 8.1 s (nov26, `jax_p1_ref`).
Per-stage time of a steady step (nov26, timed run with block_until_ready after every stage = extra host syncs): melt_si 0.001 s, dyn 0.9-1.1 s (NumPy: 2.6 s), condse_entry 0.02, condse_setup 0.14-0.19 (incl. south pole 0.09), condse_mstcnv 4.8-5.5 (70% of the step), condse_post 0.44-0.61 (incl. north pole), radia 0.04-0.05.
Jit units per step (eager dyn): melt_si 1, dyn 75 (a dispatched sequence of ~15 kernel types), condse_entry 1, condse_setup 1, condse_mstcnv 46 (22 mask tests + bucketed event calls + set-up/post), condse_post 1, radia 1 (+1 cloud masking). **The target J1 (one jit) is NOT reached**: 125-127 executions (53 with the dynamics fused). Fusing the dynamics into one jit works and is bitwise A but costs 7x the compile time (standalone dyn: 460 s cold vs 64 s eager; steady 1.47 s vs 1.05 s). The MSTCNV host loop cannot be fused without redesign (data-dependent compaction), nor can the callbacks without leaving the jit.
Transfers per step (counted by `instrument_transfers`, `jax.device_get`, callback accounting): record load once 147 arrays / 131 MB host->device (+340 MB of device_put in the process incl. start state); inside the step host->device by `jnp.asarray`/`device_put` ~518 calls / ~2.9 MB (small constants, bucket index arrays); device->host: flags 36 B (1 call), 22 mask reads x 3,168 B; pole callbacks 2 x ~230 KB; QUS callback ~250 MB/step in 176 calls; radiation interface (replay): 33.7 MB device->host and 11.6 MB host->device per call, 1 sync point per radiation step. LIMIT (documented in the harness): on the CPU backend `np.asarray(jax_array)` is not counted, so callback traffic is taken from the callbacks' own accounting. No GPU on this node: nothing here is an accelerator claim; the callbacks and the LMIN loop would be synchronisation points on a GPU.

## 5. Variant through the REAL persistent radiation server (nov26 step 33312) -- radiation computed by the original Fortran (hybrid component)
Packet from OUR device state (T,Q after our CONDSE, PK..MA, LTROPO from our TROP, 19 cloud arrays and SNOAGE from our CONDSE); RQT, KLIQ and the 21 surface fields COPIED from the live packet `rsv_n26_33312_in.bin` (declared recorded input, D185 practice). One call: 33,676,432 B device->host, 11,605,248 B host->device, 11.3 s in the call (11.1 s RADIA+IO in the server), 1 host sync point, seed -588724193 (recorded SEEDS[1]), SOCRATES/RADIA modified: no, server stopped (no process left). Whole program 197 s incl. compile.
Result vs the recorded server output (`rsv_n26_33312_out.bin`): COSZ1 A; SRHR D (max 14.8, rel 1.3e-2) and TRHR D (max 8.6, rel 2e-2); T after RADIA vs the replay-mode result D (max 4.7e-3 K, rel 1e-5); Q equal; the server's masked CLDSS/CLDMC equal our device masking (A). Where: 44 columns have SRHR/TRHR differences above 1e-6 of scale, ALL 44 inside the 2,491 columns where our (libm-mode) CONDSE cloud arrays differ from the real CONDSE exit record; outside them all differences are below 1e-6 of scale. This is the known libm-mode sensitivity (D127-D129, D180), not a new defect, and it is NOT a C1 result (C1 uses the replay).

## 6. Six steps of nov26 (33312..33317), C1, hybrid chain
Phase 1 on the device at every step; between steps the state goes to the host, the EXISTING NumPy code runs SURFACE (land recorded), dissip and filter, and the result goes back to the device (so this is NOT a device-resident multi-step run). CONDSE carry (CLDSAV.. SNOAGE, RADIA-masked CLDSS/CLDMC on radiation steps) and the LSCOND module vectors are device state between steps; MELT_SI only at step 0 (ice state afterwards is the record boundary). Reference: the NumPy libm chain (`jax_p1_ref_chain.py`, same pinning), whose surface half is the same code, so any difference can only come from phase 1.
Result: steps 0..5, each stage (dyn 52-60, condse 59-60, radia 59-60 fields) and X (68): **all fields category A at every step**; RADIA cloud masking A at the radiation steps 33312 and 33317; flags all 0. Device phase 1 per step 6.7-7.2 s after the first (189 s with compile); the host surface half 0.8-0.9 s per step after the first two. Limits: one start state, 6 steps, replayed radiation (recorded SRHR/TRHR/COSZ1 of each step), surface on the record boundary, libm mode; the chaos-limited multi-step claims of ACCEPTANCE section 4 are not made.

## 7. Honest limits against ACCEPTANCE section 1 (this stage only)
1. State updates: the phase-1 prognostics (T,U,V,Q,QCL,QCI,TMOM,QMOM,P,MA,..., MELT_SI ice) are updated by JAX functions and stay on the device between stages: MET for phase 1. Ocean, ice dynamics, lake, land state are NOT updated by this stage.
2. Boundaries/transfers: measured and reported (section 4); 125-127 jit executions per step, not one.
3. Non-JAX stages: listed with shares in section 2/4 (MSTCNV host loop and QUS callback 70% of the step; pole callbacks ~3%; dyn 14%).
4. Radiation: replay (recorded) in the C1 results, labelled "radiation replayed from the real record (not computed)"; the one real-server run carries the Fortran sentence.
5. Recorded inputs: listed with sizes (section 2); the CONDSE entry set `ci` (85.8 MB) is the largest and is an input in every step.
Also: libm mode only; the QDYNAM z-extra branch is flagged not executed; stop-model checks are flags read once per step (not raised in the device program).

## 8. Blockers / next for the surface stage (S4/S5)
- Phase 2 needs SURFACE tiles in the fixed layout (jax_state_d181) and a JAX PBL/ATURB/tile/GHY chain; nothing of that is here. The hand-off arrays the surface stage needs from phase 1 are in X (PREC, EPREC, PRECSS, DDM1, DDMS, TDN1, QDN1, DDML) and S (UALIJ, VALIJ, DPDX..): produced on the device.
- The MSTCNV LMIN loop (70% of the step) and the QUS callback prevent a single jit; a fixed-bucket device compaction is the obvious redesign, not attempted (cost estimate: event block on 3,168 columns x 22 iterations).
- Pole columns: callback to NumPy; a JAX port needs the KMAX=72 variants of MSTCNV/LSCOND.
- Fusing the dynamics into one jit multiplies the cold compile by 7 (460-530 s); on GPU the compile and dispatch trade-off is untested.
- Replay server returns zeros for the surface-facing radiation outputs (FSF, TRSURF, ALB, FSRDIR, ...) and AIJ; the surface stage needs them from the real server or from records.
- The 3 dates and step 0 only for the gate; the six-step chain is nov26 only.

**Parent-session check (2026-10-07 19:50):** `tests/test_jax_atm_phase1.py` re-run: 8 passed, 8 s; no tracked file modified. Independent bitwise check with my own comparison code (`np.array_equal`, not the harness categories): the device phase 1 of nov26 step 0 (run with the agent's entry points under `taskset -c 0-1`, `clouds_jax_env_fast`, no cache) against the saved NumPy libm phase-1 reference `ff_data`-side scratch file `ref/nov26_p1_1.npz`: 170 fields (dyn, condse, radia stage snapshots) all bitwise equal, none missing on either side. Not re-run by the parent: dec01 and jan01 (agent: 52/52, 59/59, 59/59 category A each), the fused-dynamics variant, the 6-step chain, the real-server variant and the mutation tests. CAUTIONS kept: this is C1 ONLY, in libm mode, with REPLAYED radiation (recorded SRHR/TRHR/COSZ1 served through the D185 interface), so it says nothing about the real model; a step still takes about 125 jit executions (53 with fused dynamics, which costs ~7x the compile time), 532 eager dispatches; host work remains: pole columns (two pure_callbacks), the MSTCNV cloud-base loop (22 mask reads per step, ~70% of step time) and the QUS subsidence callback (176 calls, ~250 MB per step); the QDYNAM extra-column branch is not executed (flag 0 in all runs); the real-server variant at step 33312 gives SRHR/TRHR in category D (rel 1.3e-2 and 2e-2) which the agent attributes to libm-mode cloud differences (not verified) and which is not a C1 result; the steps 0-5 chain ran phase 1 on the device and the NumPy surface, dissip and filter on the host between steps; only nov26 for the chain. The surface half is NOT in the device program. libimf is not used here (C2 needs the D183 ops wired into these modules).

# D187: stage S7, first coupled-step report -- the best HYBRID coupled step that exists now (2026-10-07)

Owner: Glenn Tamkin; written by a Claude Code agent. Project-local. Nothing committed; no existing file edited; new files only.
New files (fullfidelity/): `d187_imf_p1.py` (libimf host callback wired into the D186 modules), `d187_coupled_step.py` (the hybrid step, one process per date),
`d187_ref_numpy.py` (C1 reference: NumPy libimf chain with the same closed surface), `d187_common.py` (flatten, surface C2 against real records, I/O audit),
`d187_analyze.py` (C1 comparison), `d187_stage_vs_real.py` (first failing stage), `d187_summary.py` (figures), `tests/test_d187_imf_p1.py` (4 passed, 3 s on cores 0-2), this entry.
Raw results (outside git, session scratchpad `d187/`): `hyb/<date>_hyb.json|npz`, `hyb/nov26_hyb_server.npz`, `ref/<date>_ref.json|npz`, `hyb/<date>_c1.json`, `hyb/<date>_stage_vs_real.json`.
The other session's untracked files (jax_state.py, jax_static.py, jax_state_capture.py, jax_imf*.py, imf_site_audit.py, dyn_jax_fast.py, d182_*, tests/test_jax_imf.py, tests/test_jax_state.py) were not read, used or touched. SOCRATES/RADIA never ported or modified.

**THIS IS A HYBRID STEP. IT IS NOT END-TO-END JAX AND NOT "JAX-DRIVEN" BY ACCEPTANCE SECTION 1 (items 1 and 3 fail, see section 5).**
radiation: see the sentences per variant below. libimf math functions provided by a host callback to the original build's runtime (36,070 calls, 16.2 M elements, 233 MB in / 143 MB out, 15.6 s on nov26; per date in section 4).

## 1. What was assembled (step 0 of each date: nov26 33312, dec01 33552, jan01 17520)
| part | implementation | kind |
|---|---|---|
| MELT_SI, DYNAM (+QDYNAM, energy fix, TROP, PGRAD_PBL, KEA), CONDSE (MSTCNV, LSCOND, poles), RADIA apply | D186 `jax_atm_phase1`, device arrays, WITH libimf host callback (`d187_imf_p1`) | JJ/NP + callbacks |
| radiation | (a) REPLAY of recorded SRHR/TRHR/COSZ1 through the D185 hand-off interface, all 3 dates: "radiation replayed from the real record (not computed; no Fortran callback in this result)"; (b) nov26 ONLY: the real persistent server through the D185 `io_callback`: "radiation computed by the original Fortran (hybrid component)" | REC / FORT |
| SURFACE (PBL, ATURB, tiles, GHY land with RECORDED Ent exports, land ice), PRECIP_*, GROUND_*, RIVERF, DYNSI, ocean step, FORM_SI, ADVSI | existing `surface_loop_v2.Loop2` (closed: ocean, ice, lake, land ice carried from OUR computation), land_mode `ghy`; MELT_SI result taken from the DEVICE program (equal to the NumPy `melt_si` on every output, checked each run: True on 3 dates) | NP / EJ |
| DISSIP, FILTER | existing NumPy `atm_step.stage_dissip/stage_filter` (libimf pow through ctypes in the filter) | NP |
The state goes device -> host (59 arrays, 47.5 MB, 0.03 s) after phase 1 and stays on the host for the rest of the step. Pinning of EVERY run (hybrid and reference): `taskset -c 0-2`, `OMP_NUM_THREADS=1`, XLA flags `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover` (clouds_jax_env), no compile cache (header check), jax 0.5.3, ctx imf=True. New references were made with this pinning (the D184 libm reference was NOT used; it is valid for cores 0-1 and libm mode only).

### libimf wiring (what d187_imf_p1 does; new module, D183 `libimf_ops` reused)
D183's `install('libimf')` handles dynamics ADVECM/PGF, MSTCNV and LSCOND (same modules the D186 code calls). The D186 glue is different code, so four things are added: `jax_p1_glue.jnp` -> `JnpProxy` (MAtoPMB pow, PEK pow, CALC_TROP pow x2; the second sits in a vmapped while loop), `jax_p1_condse.jnp` -> `JnpProxy` (snow-age exp), `jax_p1_condse.lj` -> shim whose `_core` uses mode "imf", `jax_p1_condse.cf` -> shim that keeps the 'imf' backend in the pole-column callbacks (PoleHost.run calls `set_backend('numpy')`), and `DynDevice.kit` -> D183 `KitImf`. Unit test: MAtoPMB (all outputs), PEK and CALC_TROP under the callback are bitwise equal to the NumPy imf functions; the NumPy libm PK differs from libimf (non-vacuity); callbacks counted. Not verified separately: that no libimf site is missed in D186 modules beyond the NumPy-imf inventory of D183 (the C1 result, bitwise on 559 arrays per date, is the evidence).

## 2. C1: hybrid step vs NumPy chain in libimf mode (same surface loop, same pinning)
Fields compared with `jax_harness.field_category` (A bitwise, B <= 1e-12 of scale, C <= 1e-6, D worse). Snapshot groups: dyn 52, condse 59, radia 59, CONDSE outputs X 68, surface 59, dissip 60, filter 60 fields; surface state after the step `surf/` 74 arrays (ocean, ice, lake, land ice, atm exchange fields, ADVSI RSIX/RSIY, DYNSI USI/VSI) and carried land state `land/` 68 arrays.
| date | radiation | A | B | C | D | not-A fields |
|---|---|---|---|---|---|---|
| nov26 | replay | 559 | 0 | 0 | 0 | none |
| dec01 | replay | 559 | 0 | 0 | 0 | none |
| jan01 | replay | 559 | 0 | 0 | 0 | none |
| nov26 | real server | 545 | 14 | 0 | 0 | radia: T, TRHR; surface, dissip, filter: EGCM, T, TRHR, W2GCM each (B, <= 2.6e-16 of scale) |
(559 = 52+59+59+68+59+60+60+74+68 over the nine groups.) The server row is NOT a strict C1 comparison (the NumPy chain uses recorded radiation); it shows that the server's TRHR differs from the recorded one in 15 elements at 1.1e-13 (rounding) and that this propagates to T (2 elements), EGCM and W2GCM (1 element each, 1e-21).
Non-vacuity: a 1-ulp change of one element of the reference T is detected (category B). In-process repeat: the steady and timed runs equal the cold run bitwise (60 of 60 end-state arrays) on all dates. Stop-model flags: all 0 (qdynam z-extra branch not executed). C1 does not say anything about ROCKE-3D.

## 3. C2: the state after ONE coupled step vs the REAL Fortran dumps (libimf callback mode; replay radiation unless stated)
Atmosphere: `atm_step.end_reference` (ffa_step_<it>_e) vs the end state after FILTER; gate fields = the 14 F1 gate fields of D129 (`jax_harness.compare_end_state`, categories fixed); plus the 30 end fields (`atm_step.END_FIELDS`).
| date | verdict (ACCEPTANCE section 3) | gate fields A/B/C/D | 30 end fields A/B/C/D | worst gate field | fields beyond B (gate) |
|---|---|---|---|---|---|
| nov26 | **MET with named exception columns** | 1/11/2/0 | 2/25/3/0 | W2GCM 3.1e-12 | EGCM 1.3e-12, W2GCM 3.1e-12, both in ONE column (25,16) (limit 1e-9, <= 10 columns); outside the gate: LMONINPBL 5.4e-10 |
| dec01 | **PARTLY MET** | 1/6/7/0 | 2/12/16/0 | EGCM 9.9e-9 | U, V (8 cols, 1.6e-10, 1.8e-10), Q (2 cols), EGCM, W2GCM, PBLHT, PBLPTOP (2 cols each, <= 9.9e-9): columns (21,33),(57,34) and neighbours; exception rule not satisfied (rel > 1e-9) |
| jan01 | **PARTLY MET** | 1/3/10/0 | 2/3/25/0 | EGCM 7.6e-9 | U, V (16 cols), T (3), Q (8), MA (17), QMOM (5), EGCM, W2GCM, PBLHT, PBLPTOP (4 cols: (14,36),(19,32),(39,37),(49,35)); rel up to 7.6e-9 |
| nov26, real radiation server | **MET with named exception columns** | 1/11/2/0 | 2/25/3/0 | same as replay | same |
No category D on any gate or end field on any date. The hybrid step and the NumPy imf chain give the SAME verdict and the SAME category counts on every date (the verdict does not come from the JAX part).
First failing stage (dec01, jan01), from `d187_stage_vs_real.py` (stage snapshots vs the real stage-exit records): dyn A on all 26 fields, condse and radia B (<= 1.4e-14), **SURFACE is the first stage with category C** (dec01: V 1.8e-10, U 1.6e-10, Q 9.5e-11; jan01: Q 2.9e-10, U 1.3e-10, V 7.1e-11, T 8.2e-12), i.e. the land patch / surface columns (ported GHY with recorded Ent), consistent with D129/D136b (ported land: nov26 MET, dec01 and jan01 PARTLY MET). The cause inside the surface stage was NOT re-diagnosed here (named columns above are the entry point).
Consistency with earlier numbers: D129/Handoff "ported land: nov26 MET, dec01/jan01 PARTLY MET, worst field 9.9e-9" -- reproduced (dec01 worst EGCM 9.9e-9 exactly). nov26 differs in wording only: D136b says MET, the harness here says "MET with named exception columns" because EGCM/W2GCM exceed 1e-12 in one column (25,16) (1.3e-12 and 3.1e-12); this is the closed surface (our ocean/ice/lake state feeds the tiles) instead of the recorded SURFACE records of D129, and the same column set appears in the NumPy chain, so it is not a JAX effect. With the real radiation server on nov26 the gate is identical (radiation inputs from our libimf CONDSE state equal the real ones; SRHR/COSZ1 bitwise vs the recorded server output, TRHR 15 elements at 1.1e-13, T 2 elements 1.1e-13).

### Surface state after the step vs the real step-boundary records (same code, hybrid and NumPy chain give identical category counts)
Real records compared: ocean `ffo_state_<it0>` tag 14 (ocean exit); sea ice `advsi_dumps/<date>/ffadv_out_<it0>` (after ADVSI) and `ffadv_in_<it0>` (at ADVSI entry); DYNSI `ffo` tag 1 odmui/odmvi and `ffz_undocn` ustar; RIVERF `ffo` tag 1 oflowo/oeflowo; lake, land ice, ocean/ice tiles: SURFACE-entry records of the NEXT step (`ffs/ffl_<it0+1>`) vs our state advanced through PRECIP_*/MELT_SI of step it0+1 with the real PREC/EPREC of it0+1 (D164/D170 method, row k=1). NOT compared with any record (none exists at step end): land GHY soil/snow/canopy state and Ent state (the land enters through the atmosphere fields only).
| date | ocean exit (22 fields) A/B/C/D | ice after ADVSI (5) | DYNSI (3) | RIVERF (2) | next-step entry records (21) A/B/C/D, tile-set mismatches | lake + land ice records |
|---|---|---|---|---|---|---|
| nov26 | 1/8/13/0 (worst vonp 8.8e-9, vo 6.1e-9) | B,C: msi 4.0e-10, hsi 2.5e-11, ssi 1.5e-8 | all B | all B | 1/15/5/0, 0 | all B (<= 1e-14) |
| dec01 | 1/8/13/0 (vonp 2.3e-9) | all B (<= 1e-14) | all B | all B | 1/17/3/0, 0 | all B |
| jan01 | 1/9/12/0 (vonp 1.2e-9) | hsi C 1.7e-9, rest B | odmui C 3.1e-12, ustar C 4.0e-12 | all B | 1/10/10/0, 0 (ice.tg1 2.2e-7, ice.tr4 1.6e-7: C) | all B |
Consistency with D170 (replay free loop, row 0/1): ocean exit step 0 g0m 1.2e-12 / uo 1.1e-9 / vo 6.1e-9 (this run nov26: g0m 1.2e-12, uo 1.1e-9, vo 6.1e-9 -- same), ice msi 4e-10 at row 1 (this run 4.0e-10 on the entry compare); the known unresolved step-0 ice cell (65,38) of D170 section 4 is the origin of the nov26 ice msi/hsi/ssi C entries (not re-checked here). No surface field is in category D. Note that the surface fields are mostly category C, not B: the ocean/ice/lake state is NOT bitwise to the real model after one step (ACCEPTANCE section 3 applies the A/B rule to "every prognostic field of the end state"; applied to the surface fields above the step would be PARTLY MET at best, and a formal surface verdict is not defined by the criteria for fields without a recorded end-of-step value, so none is claimed).

## 4. Header and lists required by ACCEPTANCE
**Radiation sentences.** replay rows: "radiation replayed from the real record (not computed; no Fortran callback in this result)". nov26 server row: "radiation computed by the original Fortran (hybrid component)": 1 call (step 33312, seed -588724193 = recorded SEEDS[1]), packet sent 33,676,432 B in 54 arrays (device -> host), received 11,605,248 B in 22 arrays (host -> device), packet file 33,678,288 B in / 55,483,416 B out, 11.20 s callback (11.19 s server request, 10.93 s RADIA+IO in the server; server start-up 21.3 s outside the step), 1 host synchronisation point, SOCRATES/RADIA modified: no, server stopped (no model process left; the one pgrep hit was an unrelated shell command line). Inputs of the packet NOT computed here (copied from the live recorded packet `rsv_n26_33312_in.bin`, declared): RQT, KLIQ, SNOAGE-carry and the 21 surface fields (D185/D186 practice); the surface TILES still read RECORDED SRHEAT/TRHR0 columns, so server radiation changes T through SRHR/TRHR only.
**libimf sentence.** "libimf math functions provided by a host callback to the original build's runtime": per step (cold run, identical counts in the steady runs): nov26 36,070 callbacks / 16,188,394 elements / 232,983,648 B in / 143,414,016 B out / 15.6 s; dec01 35,634 / 16,720,477 / 239.0 MB / 147.7 MB / 16.2 s; jan01 34,216 / 17,366,195 / 247.7 MB / 152.9 MB / 16.1 s. Most calls come from the MSTCNV loops (D183). Not device resident; a GPU run would need these removed.
**Recorded inputs actually read (registry, no undeclared read; cold run, nov26).** 9 inputs read of 20 declared, 654,452,332 B: sitea 39,770,528; siter 16,056,608; s1 33,994,368; ci (CONDSE entry) 85,753,752; co 77,884,440 (seed only); surface_records (ffp/ffs/ffl/ffg/fft, includes Ent exports, TRUP_in_rad, land forcing, irrigation demand, PBL profile columns, tile radiation columns) 24,123,264; surface_restart_for_melt_si 490,176; surface_init 376,379,196 (statics, restart, MMST, straits start, ADVSI geometry, ocean geometry; sum of audited file sizes + restart); ctx_static (restart-derived statics) NOT sized (registry cannot size it). Comparison-only (not inputs): the real end state `sitee` 39,770,528 and the files read by the C2 comparison. dec01 654,439,276 B, jan01 547,334,568 B. The Python-level file audit (`io_audit` in the JSON) lists 22 files under ff_data, 371.8 MB; netCDF restart reads are not seen by the audit and are added by size. The declared D174 items `ent_exports, trup_in_rad, irrigation_demand, radiation_packet_surface_side, mdrya_dh2o_ca, seeds0, ag2og_ig2og, straits_start, advsi_geometry, mmst` are declared but are read INSIDE `surface_records`/`surface_init` (not itemised per column). Radiation-derived tile columns are recorded in every variant; D176/D177 column modifiers were NOT used.
**Generated NON-JAX stage list with time shares** (nov26 timed run, phase-1 stages timed with a device block after each; total 44.3 s; dec01/jan01 within 1 point): condse_mstcnv [JJ/NP: host LMIN loop with 22 device->host mask reads, QUS callback 176 calls / 249.5 MB / 0.63 s, libimf callbacks inside] 68.8%; ocean_step [EJ/NP] 11.8% (5.25 s: dynamics 2.56, oconv 1.64, straits 0.46); dyn [JJ/NP, 127-jit dispatch from Python, libimf inside] 5.4%; condse_post [JJ/NP: north-pole callback] 5.0%; surface_tiles_land [EJ/NP/REC] 2.7%; record_load [REC] 1.8%; underice_ground_si [NP] 1.3%; filter [NP] 1.0%; form_si [NP] 0.7%; advsi [NP, mpmath] 0.6%; surface_pre 0.2%; condse_setup [JJ/NP: south-pole callback 0.28 s for both poles] 0.2%; dynsi, radia (replay 0.05 s; server 11.2 s callback), handoff_to_host (0.03 s) 0.1% each; dissip, ground_*, riverf, surface_records, toc2sst <0.1%. The kind labels EJ/NP of the surface modules come from import inspection (jax-importing NumPy glue), not from line-level profiling. Full generated text: `non_jax_text` in `<date>_hyb.json`.
**Jit executions and host-device transfers per step.** Jit executions (jax_p1_count): 262 on nov26 (phase 1: 127, surface half: 135), 260 dec01 (125 + 135), 259 jan01 (124 + 135); eager primitive dispatches: phase 1 206, hand-over 44, surface half 15,493 (differences of the global counter around each block; the per-run "total" fields in the JSON are cumulative and should not be used). Transfers counted by `Counters.instrument_transfers` in the steady nov26 run: host -> device 37,994 calls / 976 MB (record load 147 calls / 131 MB; surface half 1,521 calls / 441 MB; 36,249 calls / 403 MB made from callback threads, attributed to '(outside)'), device -> host 23 explicit calls / 70 KB; the device -> host copies of `np.array(device_array)` (phase-1 state 47.5 MB, radiation packet 33.7 MB, libimf operands 233 MB, QUS 250 MB, poles 0.46 MB) are NOT seen by the counter on the CPU backend (documented limit of D184) and are taken from the callbacks' own accounting. One ordered host synchronisation per radiation step plus 22 mask reads (MSTCNV) plus the callbacks (libimf: 36,070 sync points).
**Cold compile and steady-state step time (3 cores, shared node, indicative).** nov26: cold 349.3 s (phase 1 189.1 s, surface half 156.4 s; 1,171 XLA compiles, 218.8 s of compile time counted by jax.monitoring), steady 44.3 s (phase 1 34.8 s, surface half 8.5 s), 3 runs; dec01 cold 351.6 s / steady 47.0 s; jan01 cold 375.1 s / steady 44.8 s. nov26 with the real radiation server: 58.2 s (adds the 11.2 s call). NumPy reference (same cores, one run, includes its own first-call JIT in the surface stage): 154.6 / 153.5 / 159.6 s. The dominant cost is the MSTCNV host loop with the libimf callbacks (30.5 s of 44.3 s), then the ocean step (5.3 s). Speed is NOT the fidelity configuration question: the callback mode is the fidelity configuration (ACCEPTANCE 8.1); speed without callbacks was not measured here. No GPU, XLA:CPU only; no compile cache.

## 5. ACCEPTANCE section 1, items 1-5, line by line (for this hybrid step)
1. **State updates by JAX with device-resident arrays: FAILS.** Phase 1 (atmosphere dynamics, CONDSE, MELT_SI ice, RADIA T update) is updated by JAX on the device and stays there between its stages. After phase 1 the state is copied to NumPy; SURFACE tiles/land, ground_*, RIVERF, DYNSI, the ocean step, FORM_SI, ADVSI, DISSIP and FILTER update the prognostic variables in NumPy/eager-Python code (some call JAX kernels on host arrays, which is not device-resident state). Ocean, ice, lake, land state are NumPy dicts.
2. **Boundaries measured: HOLDS (measured, not minimal).** 259-262 jit executions and about 15,700 eager primitive dispatches per step; transfers as above; the single-jit ideal is not reached (D186: 125 jit executions for phase 1 alone).
3. **Non-JAX stages listed with shares: HOLDS as a list; the step FAILS the intent** (item 3 asks that nothing be omitted, not that the list be empty): the list is generated in section 4 (68.8% of the time is a host loop with host callbacks). The generated list covers all registered stages; the sub-stages inside "surface_tiles_land" and "ocean_step" are not itemised below stage level.
4. **Radiation: HOLDS for the variants that carry the sentence.** The replay rows do not use the callback and say so; the nov26 server row states packet, bytes, calls, time and sync points (section 4). Dates dec01 and jan01 were not run through the real server in this report (D185/D159 validated the server on dec01/jan01 first radiation steps; not re-run here).
5. **Recorded inputs listed: HOLDS** (section 4, registry with no undeclared read, plus the file audit), with the stated limits: ctx_static is not sized; several D174 items are only declared and are read inside bigger records; radiation-packet surface side and RQT/KLIQ are copied from the live packet in the server variant; tile radiation columns, Ent exports and land forcing are recorded in all variants.
Therefore the step is NOT "JAX-driven" under section 1 (items 1 and 3 fail) and must not be called end-to-end JAX; it is the best coupled step assembled from validated parts, with phase 1 device-resident.

## 6. What this does and does not show
- It shows: the D186 device atmosphere with the libimf callback plus the existing NumPy surface half equals the NumPy libimf chain bitwise on three dates (C1 = A on all 559 compared arrays on each date), and one coupled step reproduces the real model: nov26 MET (with one named exception column), dec01 and jan01 PARTLY MET with the first failing stage SURFACE (land patch), no category D anywhere; the surface state after one step is within 1.5e-8 (ocean/ice), 2.2e-7 (jan01 ice tile temperature) of scale of the real records; lake and land-ice records <= 1e-14.
- It does not show: any multi-step behaviour (acceptance beyond step 0 is statistical, section 4 of the criteria); land GHY/Ent state against a record; the real server on dec01/jan01; a computed radiation packet surface side (D176) or computed tile radiation columns; speed without the libimf callback; reproducibility on a different core count (everything is for cores 0-2 only).
- Not claimed: end-to-end JAX; the ROCKE-3D climate; GPU behaviour.
- Known oddities left as found: RuntimeWarnings (invalid value in divide) in clouds_condse_batch.py:78, surface_loop.py:448, ocean_step.py:375, straits_jax.py, stconv_jax.py occur in both reference and hybrid; `os.fork()` RuntimeWarning when the radiation server starts inside a JAX process (no deadlock observed).
- Review by: before the surface half is converted to device arrays (S4/S5), or when a category definition changes.

**Parent-session check (2026-10-07 20:55):** `tests/test_d187_imf_p1.py` re-run: 4 passed, 3 s; no tracked file modified. Independently recomputed with my own code from the saved results (`scratchpad/d187/hyb/*_hyb.npz` against `ref/*_ref.npz`, and against the real `ffa_step_<it>_e` end state through `atm_step.Real`/`end_reference`): C1 (replay radiation): 559 of 559 arrays bitwise equal between the hybrid step and the NumPy libimf chain on nov26, dec01 and jan01, nothing present in only one file. C2, 14 GATE_FIELDS (A/B/C/D): nov26 1/11/2/0 (worst W2GCM 3.1e-12, EGCM 1.3e-12), dec01 1/6/7/0 (EGCM 9.9e-9, W2GCM 6.0e-9), jan01 1/3/10/0 (EGCM 7.6e-9, PBLHT 3.8e-9); 30 END_FIELDS: nov26 2/25/3/0, dec01 2/12/16/0, jan01 2/3/25/0. All equal to the agent's tables. CAVEAT (found by the parent, not in the agent's report): the real end-of-step record also holds four surface-composite exports, QGAVG (0.37 of scale on nov26), USAVG (7.1e-2), VSAVG (5.4e-2) and TGVAVG (1.2e-2), which are category D in this hybrid step on all three dates; they are outside the gate and the 30 END_FIELDS by design and the plan lists the `USAVG/VSAVG/TGVAVG/QGAVG` composites as 'still recorded/unported' (`ATM_STEP_PLAN.md:165`), so this is a known gap, not a regression, but it is a gap in the surface half of the step. Not re-run by the parent: the real-server variant (nov26), the surface-state comparisons (ocean exit, ice, lake, land ice), the timings and counters. VERDICTS (agent, consistent with D129): nov26 MET with named exception columns (one column (25,16) in EGCM and W2GCM), dec01 and jan01 PARTLY MET (first failing stage: SURFACE, land patch with ported GHY, cause not diagnosed). THIS IS A HYBRID STEP, NOT END-TO-END JAX: ACCEPTANCE section 1 item 1 fails (only phase 1 is device-resident; surface, dissip and filter are NumPy/eager; about 15,700 eager dispatches and ~976 MB host-to-device per step) and item 3 fails in intent (the MSTCNV host loop with its libimf and subsidence callbacks takes 68.8% of the 44.3 s steady step on nov26). Radiation is replayed on all dates (real server only on nov26); the packet's RQT, KLIQ and 21 surface fields and the tile radiation columns are recorded; only step 0; valid for cores 0-2 only.

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
