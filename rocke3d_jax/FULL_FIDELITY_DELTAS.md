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
