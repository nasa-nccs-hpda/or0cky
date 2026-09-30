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

## Pending rows
- GPU speed numbers for the JAX-vectorized pieces (no GPU available on the node used for D14/D15/D16's
  CPU-only measurements).
- The actual `jax.lax.scan`-chained whole-model Track B step (wiring ATURB/PBL/SURFACE/SEAICE/LAKES/
  GHY together the way Track A's `run_steps_device` chains one full atmosphere step) -- every needed
  piece is now validated and jit-able (D17 closes the last missing one, `ground_si`), but the actual
  driver assembly, land-ice tile-flux call site, and PBL↔SURFACE data-flow tracing are still pending
  (see FULL_FIDELITY_PLAN.md's chained-driver section and STATUS.md's 8-stage table).
