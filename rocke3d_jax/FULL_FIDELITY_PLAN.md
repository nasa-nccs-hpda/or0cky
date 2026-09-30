# Full-Fidelity Port: Implementation Plan

Branch: `full-fidelity-port` (from `main` @ 62f87a1 + Round 2 working tree).
Written 2026-09-24. Supersedes the "closed, not pursuing" decision in
STATUS.md (2026-09-23): the project goal changed — HQ wants a port whose
answers can be defended against the real Fortran, not a representative
workload. Everything in STATUS.md stays as the **baseline** ("representative
driver"); this branch adds a second, separately-labelled track and records the
**deltas** between them.

## Progress (updated 2026-09-24, end of first autonomous session)

| Item | Status | Evidence |
|---|---|---|
| Phase 0: build/instrument real ModelE, reproducibility, noise floor | **done** | `fullfidelity/PHASE0_LOG.md`, ledger D1–D3 |
| Phase 1.1 ATURB (A-grid + U/V) | **done, F0 rounding-level** | D4 |
| Phase 1.2 PBL `advanc` (surface layer, all 4 tile types) | **done, F0** | D5 |
| Phase 1.3 SURFACE ocean/lake + sea-ice tile fluxes, ice properties | **done, F0** (limiter branches unvalidated) | D6 |
| Phase 1.3b land-ice tile | **done, F0** (dew limit unvalidated) | D7 |
| Phase 1.4 GHY / land (EARTH + `advnc` + snow) | **done, F0** (Ent vegetation exports taken as recorded input, not ported) | D9 |
| Phase 1.5a SEAICE ground thermodynamics (SEA_ICE/SSIDEC/snowice/ADDICE/SIMELT) | **done, F0** | D10, D12 |
| Phase 1.5b Tile aggregation (`avg_patches_*`) | **done, F0** | D11 |
| Phase 1.5c Lake mixing (LAKES.f `LKSOURC`/`LKMIX`) | **done, F0 bitwise** (TKE-entrainment branch dead code in this rundeck, cross-checked synthetically) | D13 |
| JAX-vectorization: lake mixing (`lakes_core_jax.py`) | **done** 2026-09-28, bitwise match, ~104x CPU speedup | D16 |
| Phase 2 Radiation (SOCRATES) | scoped 2026-09-27, proof-of-concept only (one kernel via subprocess); paused — user redirected effort to remaining small items first | plan §Phase 2 |
| Phases 3–5 (clouds, dynamics, ocean) | not started | — |
| JAX-vectorization: SEA_ICE/SSIDEC/snowice/SIMELT/ADDICE (`seaice_core_jax.py`) | **done**, same accuracy as plain Python, jit-compilable (46-51x CPU speedup) | D14 |
| JAX-vectorization: GHY (`ghy_ref.py`) | **done** 2026-09-27/28, matches real Fortran to plain-Python's own D9 tolerance, ~20x CPU speedup (lax.scan, after an unrolled-loop regression was measured and fixed) | D15 |

**Scoping learned so far.** (a) The pieces ported so far are stateless column/tile functions and validated at
1e-11–1e-16 relative; the method (instrumented real model → per-call records → JAX port → tests with
mutation checks) works and took roughly an hour of effort per ~1k Fortran lines. (b) The land model is a
different animal: `giss_LSM/GHY.f` (4.6k lines) keeps its state in module-level variables (soil water/heat
in 6 layers × bare/vegetated, 5 snow layers, canopy), is driven by Ent vegetation (26.5k lines of library; only
the canopy conductance/photosynthesis path is needed), and needs the multi-layer land state from the restart.
Expect it to be the longest single item in Phase 1. (c) Moist convection + large-scale condensation
(`CLOUDS2.F90`: MSTCNV 2.7k, LSCOND 2.0k lines, heavily branching) is next in size. (d) Track B's current
pieces are unoptimized (CPU float64 ≈ 0.4 s per DTsrc step for PBL+ATURB alone, ledger D8): performance work
comes only after each rung's correctness.

## 1. Definition of "full fidelity" (decide this first, in writing)

Fidelity is a ladder, not a switch. Each rung is a claim we can test:

| Rung | Claim | Test |
|---|---|---|
| F0 | Module-level: each ported routine matches Fortran on captured inputs | Per-routine input/output dumps, tolerance table |
| F1 | Column-physics step: SURFACE→GROUND→PBL/ATURB→RADIA→CLOUDS→DRYCNV chain matches Fortran for 1 DTsrc from the real restart | Field-by-field diff of the state after one step |
| F2 | Multi-step (e.g. 1 day = 48 steps) with dynamics: statistically indistinguishable from Fortran | Global means, zonal means, RMSE vs. Fortran output; chaos-aware (ensemble spread, not bitwise) |
| F3 | Coupled (ocean + sea ice) run: monthly `acc` diagnostics match the real P2SAoM40 `*.accP2SAoM40.nc` | Compare to the 12 monthly acc files already on disk |

**Proposed target: F2 for the atmosphere, with F1 as the hard gate.** F3
(ocean GCM) is scoped but recommended as a stretch/decision point (section 6).
Bitwise equality is *not* a goal: float32/float64 and operation order differ,
and the atmosphere is chaotic (Round 2 already showed marginal-cell branch
flips). Success criterion for each rung must be stated as tolerances agreed
before coding, not fitted afterwards.

## 2. Scope: what "full" means for P2SAoM40 (measured, from the rundeck)

Line counts of the real source in `modelE2_planet_2.0/model/` (Fortran only):

| Area | Files | Lines | Current JAX state |
|---|---|---|---|
| Radiation | RADIATION.f, RAD_DRV.f, RAD_UTILS.f, RAD_COM.f | ~19,500 | graybody stand-in |
| Radiation library (SOCRATES) | `model/socrates` + `ModelE_Support/socrates` | ~90,000 (F90, external lib) | none |
| Moist convection + large-scale cloud | CLOUDS2_DRV.F90, CLOUDS2.F90 (+COM) | ≥3,800 (CLOUDS2.F90 not yet counted) | none (out of scope so far) |
| Land surface / snow | GHY_DRV.f (+giss_LSM), VEG/ENT | 4,939+ | placeholder |
| PBL / turbulence | PBL.f, PBL_DRV.f, ATURB.f | 6,762 | PBL faithful kernels; ATURB partial; driver uses shortcut |
| Sea ice | SEAICE.f, SEAICE_DRV.f | 5,394 | placeholder |
| Lakes | LAKES.f | 4,076 | placeholder |
| Land ice | LANDICE*.f | 2,073 | none |
| Surface/fluxes | SURFACE.f, FLUXES.f | 5,123 | partial (single-type dispatch) |
| Dry convection | DRYCNV.f | 551 | faithful |
| Atmospheric dynamics | ATMDYN.f, MOMEN2ND.f, QUS_DRV/QUS3D, STRATDYN | ~8,500 | none |
| Ice dynamics | ICEDYN.f, ICEDYN_DRV.f | 3,682 | none |
| Ocean GCM | OCNDYN/OCNDYN2/OCNKPP/… | ≥12,500 | none |

Excluding the SOCRATES library and the ocean, ≈ 55–60k lines of Fortran are in
play; with them, well over 150k. This is a multi-month effort for one
person; the plan below is ordered so that each phase produces a
*measurable fidelity delta* and can be the stopping point.

> **2026-09-24 finding (see `fullfidelity/PHASE0_LOG.md`):** the real
> P2SAoM40 executable contains no DRYCNV — free-atmosphere mixing is ATURB
> (`atm_diffus`). Track A's DRYCNV work benchmarks a routine the real
> configuration does not run. Phase 1 must therefore make **ATURB** the first
> port target. Also confirmed: the real model is bitwise reproducible here
> (5-day re-run == original restart, byte-identical).

## 3. Method (what makes this different from the previous port)

The earlier ports were validated against self-written "Fortran-like" test
programs, and GHY's validation turned out to be rigged. The new method
removes that failure mode:

1. **Real-Fortran oracle only.** Every routine is validated against dumps
   from the *actual* ModelE executable, never a hand-written stand-in.
   Phase 0 builds an instrumented ModelE (write inputs/outputs of each
   routine to NetCDF at chosen steps) from the existing build tree
   (`P2SAoM40.mk`, `.o` files already present).
2. **Validation cannot be vacuous.** Every test asserts that outputs are
   non-trivial (not all zeros/constants), compares *all* output fields, and
   is mutation-checked (perturb the port; the test must fail).
3. **Port structure.** Layer-first (L,J,I) arrays (= Fortran memory order);
   per-column physics as pure functions vmapped over the grid; loops with
   dependencies as `lax.scan`; branches as `jnp.where` with the dead-branch
   NaN caveat handled explicitly. Device-resident chained stepping from day
   one (Round 2 lesson), so performance work is not a rewrite later.
4. **Faithfulness first, speed second.** No fused/optimized variant until the
   faithful version passes its rung. Keep the faithful version as the
   reference for later optimization (same as this project's Phase 1/Round 2).
5. **Delta capture is built in** (section 5), not a report written at the end.

## 4. Phases

Each phase ends with: tests green, a delta record appended, STATUS-style
write-up, and a go/no-go note. Effort figures are rough single-engineer
estimates and the least reliable part of this plan.

### Phase 0 — Oracle & harness (2–3 weeks) — HARD GATE
- Confirm the ModelE build is reproducible here (compiler, NetCDF, MPI
  stubs); run the 194-step reference or a short (e.g. 48-step) segment from
  `1JAN1950.rsfP2SAoM40.nc` and confirm it reproduces `P2SAoM40.PRT`.
- Add a small dump facility (Fortran `call dump_state(name, arrays)`)
  around: SURFACE, GHY, PBL/ATURB, RADIA, CONDSE, DRYCNV, and the full
  post-step state. Store as NetCDF under a versioned oracle directory
  (not committed if large; commit a manifest with checksums).
- Build the comparison harness: field-by-field metrics (max abs, RMS, rel,
  correlation, location of max), tolerance table per field, machine-readable
  output (JSON) so deltas can be diffed run-to-run.
- Establish the **noise floor**: run Fortran twice with a 1-ulp perturbation
  to measure intrinsic divergence; port tolerances are set relative to it
  (this reuses the Round 2 method).
- **Risk:** if the model cannot be rebuilt/instrumented here, everything
  downstream validates only against restart/acc files (much weaker). The
  fallback and its consequences must be written down, not discovered late.

### Phase 1 — Land surface & PBL fidelity (5–7 weeks) — closes the known placeholders
1. **ATURB** real closure constants, `find_pbl_top`, tridiagonal, wired into
   the driver replacing the layer-1 shortcut (1.4k lines). Validate F0 on
   dumps.
2. **PBL.f/PBL_DRV.f** Newton solve replacing the fixed-point iteration
   (a documented "Simplified" row today).
3. **SURFACE/FLUXES** full area-weighted sub-tiling over ocean / sea ice /
   land ice / land fractions (removes "single dominant itype").
4. **GHY_DRV** (4.9k lines, multi-layer soil + snow, needs its own
   orchestration and the multi-layer restart state already located in
   `1JAN1950.rsfP2SAoM40.nc`). Includes evaporation, runoff, soil thermal
   properties, snow melt — all placeholders today.
5. **SEAICE(+DRV)** thermodynamics (5.4k), **LAKES** (4k), **LANDICE** (2k).
- Gate F0 per routine, then F1 for the surface/PBL sub-chain.
- Expected delta: correct polar/land fluxes; the current sensible-heat
  pattern correlation vs. period-mean (−0.014, one-step vs. mean — not a
  like-for-like check) should be re-measured properly against the oracle.

### Chained whole-model Track B driver — SCOPED 2026-09-28, in progress

Prompted by the user asking directly whether Track A and Track B are both "end-to-end" the same way:
they aren't yet, on the *chaining* axis (Track A's `p2saom40_driver.py` fuses one real per-column
workflow into a single device-resident `jax.lax.scan`; Track B has validated, fast, `jax.jit`-able
modules but they were never wired into one driver). Read the real per-cell orchestration
(`SURFACE.f`'s `SURFACE` subroutine and `GHY_DRV.f`'s `earth`) to scope this properly rather than
guess, and the finding is good news: **almost every piece already exists and is already validated**
-- this is an *integration* task over Track B's existing D4-D16 modules, not a new porting effort.

Real per-cell, per-DTsrc-step call order, **now traced line-precisely against the persistent source**
(`/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model/SURFACE.f`, not a rebuilt scratch
copy — that tree already exists and doesn't need rebuilding for read-only tracing). This corrects the
step order from the first scoping pass above: **land ice runs BEFORE `EARTH`, not after** (the
opposite of what the first pass guessed without checking).
1. `SURFACE.f:472 DO ITYPE=ITYPE_MIN,ITYPE_OCEANICE` (comment: "no earth or landice type") —
   surface-layer solve + tile fluxes per (I,J) for ocean (ITYPE=1) and sea ice (ITYPE=2); **have**:
   `pbl_ff.py` (D5).
2. `SURFACE.f:883/893 CALL SURFACE_LANDICE(NS==1,MODDD,DTSURF,atmglas(ipatch),ipatch)` — land ice,
   looped over glacial-ice `ipatch` (NOT part of the ITYPE loop above, and NOT indexed by (I,J) tiles
   the same way — `atmglas` patches are their own array; `#ifdef GLINT2` selects a height-point
   variant `atmglas_hp`/`ihp` instead, but **`GLINT2` is not defined for P2SAoM40**
   (`grep GLINT2 decks/P2SAoM40.R` — no match), so the plain `atmglas`/`ipatch` branch is the real one
   to port); **have**: `landice_tile_ff.py` (D7), already ported/validated at the per-tile-flux level
   — the ipatch/patch-array plumbing above it (as opposed to grid (I,J)) is the still-untraced part.
3. `SURFACE.f:902 CALL EARTH(NS,MODDSF,MODDD)` (land): its own `CALL PBL(...,itype=land,...)` then,
   after `NIsurf` Ent-driven substeps, `CALL ADVNC(...)` — the real GHY call; **have**: `pbl_ff.py`,
   `ghy_jax.py` (D15).
4. `SURFACE.f:1055-1057 call avg_patches_pbl_exports/avg_patches_srfflx_exports/
   avg_patches_srfstate_exports` — tile-flux aggregation across all tiles (ocean/ice/land-ice/land),
   confirmed to run once, after all four tile types above, not per-tile; **have**:
   `tile_aggregate_ff.py` (D11).
5. `SURFACE.f:1172 CALL ATM_DIFFUS(1,1,dtsurf)` — the real ATURB call (free-atmosphere turbulent
   mixing, using the aggregated tile fluxes); **have**: `aturb_ff.py`, `aturb_uv_ff.py` (D4).
6. `SURFACE.f:1230 CALL GROUND_SI(si_atm,icelak,atmice,atmocn)` — sea-ice AND lake-ice ground
   thermodynamics; **DONE 2026-09-28**: added `seaice_core_jax.ground_si(is_ocean, ...)`, a single
   named-argument wrapper unifying the ocean/other domain split (`sea_ice`→`ssidec`→`snowice`,
   D10/D12/D14) via `jnp.where(is_ocean,...)` instead of a Python branch, matching
   `seaice_core_ff.py`'s `ground_si_ocean`/`ground_si_other` split. Validated against
   `seaice_jax_compare.batched_ground_si` (already checked against real Fortran) at <1e-9 relative
   error across all 4,524 real cells (`test_ground_si_general_wrapper_matches_test_helper`).
7. `SURFACE.f:1232 CALL GROUND_LK` — lake mixing; **have**: `lakes_core_jax.py` (D16), already a
   complete `lksourc_full`+`lkmix` chain.

All 7 steps live inside the **same single Fortran subroutine** (`SURFACE.f:21 SUBROUTINE SURFACE`),
called once per `DTsrc` step from `MODELE.f:332`. **Correction (2026-09-28, later):** they are NOT all inside the
`SURFACE.f:385 DO NS=1,NIsurf` loop. That loop (`END DO` at `SURFACE.f:1178`) wraps steps 1-5 only (ocean/ice
tiles, land ice, EARTH, aggregation + first-layer update, ATM_DIFFUS), executed `NIsurf`=2 times per step.
`GROUND_SI` (:1230) and `GROUND_LK` (:1232) run **once per DTsrc step after the loop**, on fluxes accumulated over
the substeps (followed by `RIVERF`, `FORM_SI`/ADDICE). Each NS iteration also starts with `loadbl` (PBL state
load), `recalc_agrid_uv`, `atm_exports_phasesrf` and `get_dbl`, which are further links not yet ported. Verified
empirically: PBL profiles (`upbl/vpbl/tpbl/qpbl/epbl`) carry over bitwise from one substep's outputs to the next.

**Bug found and fixed while adding step 6**: `seaice_core_jax.py`, `ghy_jax.py`, and
`lakes_core_jax.py` never called `jax.config.update("jax_enable_x64", True)` themselves — they
silently ran in JAX's float32 default unless whatever script imported them happened to enable x64
first (every existing test/compare script did, by luck of import order, which is why D9/D10/D12-D16
never caught this). Writing a standalone validation script that imported `seaice_core_jax` directly
(no enabling caller) exposed it immediately: cell row 1944's `ssidec` divides by a `tsil` value that
should be ~0 at the melt point: float32 (`ghy`/`seaice`) computed a differently-signed/-scaled
near-zero tsil than the float64 plain-Python reference and real Fortran ground truth, propagating into
a brine-fraction division that put `erunosi` off by ~5,415 units (`hflux` sign-flipped entirely) — not
noise, a wrong answer that would have silently shipped in any script that imported the core module
without a caller-side x64 enable. Fixed by moving `jax.config.update("jax_enable_x64", True)` inside
each of the three core modules (right after `import jax`, before `import jax.numpy`), so they are
correct regardless of caller/import order. Re-validated at <1e-9 rel. error post-fix (was up to
5.4e3 abs. error pre-fix on the affected cell); full existing test suites (lakes: 5/5, sea-ice:
12/12 incl. the new test) still pass.

Land-ice's call site is now located precisely (step 2 above) — it is **not** a fourth `ITYPE` inside
the ocean/ice loop as first guessed, it's a separate loop over `atmglas(ipatch)` glacial-ice patches
that runs strictly *before* `EARTH`.

**`ipatch`↔`(I,J)` mapping — RESOLVED, trivial for P2SAoM40**: `atmglas` is allocated as
`atmglas(1-min(nhc_local,2)/2 : nhc_local)` (`FLUXES.f:1782`), where `nhc_local` comes from
`call sync_param("NHC", nhc_local)` (only reached because `GLINT2` is undefined here) with a Fortran
default of `NHC_LOCAL = 1` if the rundeck doesn't override it — and P2SAoM40's rundeck
(`decks/P2SAoM40.R`) has no `NHC` line, so `nhc_local` stays `1`, making `atmglas` a **single-element
array** (`atmglas(1:1)`). So for this configuration `do ipatch=1,ubound(atmglas,1)` is `do ipatch=1,1`
— there is no real patch-routing problem at all, just one land-ice "patch" that is itself a full
(I,J)-grid-sized tile (`SURFACE_LANDICE.f:181-182`'s own `DO J.../DO I...` loop), gated per-cell by
`if (igla%ftype(i,j) <= 0) cycle` (`SURFACE_LANDICE.f:188`) exactly like the ocean/ice `ITYPE` loop's
own `ptype`/`ftype` gating — and `landice_tile_ff.py` (D7) already takes `ptype` as a named input, so
no new plumbing is needed here at all. (`NHC>1`, multiple height classes, is a real code path for
other rundecks but not this one — not a scoping gap for P2SAoM40 specifically.)

**Ocean/ice PBL↔SURFACE data flow — already validated, not new work**: traced the real exchange
(`SURFACE.f:614-635` fills `pbl_args%TG/TR4/TGV/dtsurf/qg_sat/uocean/vocean/ocean/snow/sss_loc` from
tile state, `SURFACE.f:641/643 CALL PBL(I,J,1,ITYPE,PTYPE,pbl_args,atmocn/atmice)`, then
`SURFACE.f:649-658` reads back `pbl_args%us/vs/ws/ws0/gusti/qsrf/cm/ch/cq/TSV`) -- and it turns out D6
already validates exactly this composition end-to-end (`surface_chain_ff.py`: "PBL advanc → tile
fluxes reproduces the tile outputs to ≤ 6e-11 of their spread", explicitly **not** using recorded PBL
outputs). So for ocean/ice this item is already closed, just not previously cross-referenced from the
chained-driver section.

**Land's PBL↔advnc data flow — the real remaining structural question, already honestly scoped by
D9, not a new finding**: `earth()`'s real `CALL ADVNC(entcells(i,j), Ca, ...)` (`GHY_DRV.f:~1227`)
passes the live `entcells(i,j)` Ent object directly into `advnc`, which calls into Ent *internally*
during its own execution (not as a separate call in `earth()` before/after `advnc`) -- confirming,
not contradicting, D9's stated scope: dynamic vegetation (Ent: canopy conductance, LAI, GPP, soil
betas, Ci, IPP) is genuinely not ported, and `ghy_jax.advnc`'s `ent_dts`/`ent_cnc`/`ent_betadl`/
`ent_lai` inputs are real per-substep Ent exports recorded from the instrumented dump, not computed.
This is fine for the single-step, dump-fed validation D9/D15 already did, but it is the real blocker
for a genuinely self-contained, **multi-step** chained land branch: without a ported Ent (a
substantial separate effort, not scoped here), a chained driver's land branch can only run one step
at a time using recorded Ent exports, not evolve its own vegetation state step-to-step the way a true
`run_steps_device`-style loop would need to. Ocean/ice/land-ice/lakes have no such dependency and are
fully closeable now; land is the one branch where "chained" and "full fidelity" pull against each
other, which is worth surfacing plainly rather than glossing over -- it's a concrete answer to why
Track A's simpler, chainable design and Track B's faithful-but-harder-to-chain design are both
reasonable choices for different questions, not one clearly subsuming the other.

What's genuinely new work, not already-validated pieces waiting to be wired together: (a) assembling
the ocean/ice/land-ice/lakes branches (all now fully scoped and closeable, per above) into one
`jax.lax.scan`-chained driver over the NS=1..NIsurf body (steps 1-5) followed by the once-per-step GROUND_SI/GROUND_LK, the way
`p2saom40_driver.py`'s `run_steps_device` does for Track A -- deferring land/GHY to single-step,
dump-fed mode (matching D9/D15's existing validation) rather than blocking the whole driver on
porting Ent. Given how much is already built, validated, and now precisely traced (land-ice's call
site/patch question and the ocean/ice PBL data flow are both fully closed above), this looks
considerably more tractable than the original "biggest remaining item" framing suggested -- the
scoping is essentially complete; what's left is implementation, not further tracing. Not yet
attempted: SEAICE/LAKES/GHY's own prognostic state (ice thickness, lake temperature, soil moisture)
feeding back into the NEXT step's tile fractions/properties -- the genuine "whole model" coupling
loop, distinct from one step's tile-flux computation.

**Progress 2026-09-28 (D18):** steps 1, 4, 5 are now checked as a composition for the ocean/ice share (`chain_aggregate_aturb.py`: our PBL+tile -> aggregation -> ATURB vs the real exit state, roundoff). Next: run our `landice_tile_ff` inside that composite, then the `lax.scan` assembly (ocean/ice/land-ice/lakes; land stays dump-fed).

**Progress 2026-09-28 (D19):** the NS=1 -> NS=2 chain (steps 1-5 over two substeps) is validated with land recorded. Remaining before a `lax.scan` driver: GROUND_SI/GROUND_LK/FORM_SI after the loop, a step-to-step chain (needs the other physics that runs between steps: radiation, dynamics, clouds), and land/Ent.

**Progress 2026-09-28 (D20):** GROUND_SI and GROUND_LK are chained after the two substeps. Remaining for a whole DTsrc step: ocean fluxes into GROUND_SI, FORM_SI/ADDICE+SIMELT in the chain (D12/D14 exist), land/Ent, then a lax.scan wrapper and the between-step physics (radiation, dynamics, clouds).

**Progress 2026-09-28 (D22):** land (our PBL + JAX GHY) is in the two-substep chain; all four surface tiles are now our own numbers. Remaining recorded: Ent exports, precip/radiation forcing, land TRUP, ocean fluxes. Next: `jit`/`lax.scan` wrapper + warm timing, land runoff into the lake budget, ocean-cell ADDICE/SIMELT.

### JAX-vectorization of GHY (`ghy_ref.py`) — DONE 2026-09-27 (see D15)

**Update:** completed the same day it was scoped below. The scoping held up well against
implementation: fixed-size-plus-mask worked for all three variability axes as predicted, the
bisection's `exact` branch was handled as a normal ~7%-of-calls outcome (not approximated), and
`snow_redistr`'s while-loop was replaced with a closed-form conservative overlap-matrix remap
(verified equivalent to 2e-9 over 20,000 random trials) rather than forced into `lax.while_loop`.
One prediction turned out to be a **false alarm, corrected by the actual implementation**: the
scoping note below concluded that naive `dts=0`/`dts≈0` substep padding doesn't work (crashes or
perturbs accumulators) and that a per-lane "run vs. keep prior state" mask was needed instead --
true, and that mask design was used, but building it revealed the `dts` pitfall doesn't actually
matter once the mask exists: every division in this port is either explicitly guarded or, if not,
produces inf/nan in a padding lane that is *entirely discarded* by the mask select (never blended
numerically into the kept state), so padding lanes can use any `dts` value including 0 with no ill
effect. The original scoping section is kept below for the record of how the estimate was built.

**A problem the scoping did NOT anticipate, found the next day by measuring speed instead of
stopping at correctness:** the substep loop's first working form was a Python-level `for i in
range(11): ...` unroll, which duplicates the whole per-substep computation graph 11 times -- this
measured as *slower than plain Python* in eager mode and too slow for `jax.jit` to finish compiling
in 300s. Fixed by switching to `jax.lax.scan`; see D15 for the full measurement and fix. Lesson for
next time: this project's usual "small bounded loop -> Python unroll" pattern only holds when the
loop body is small (2-6 trivial iterations, as in hydra's bisection or tridiag); an 11-iteration loop
whose body is itself dozens of functions deep needs `lax.scan` from the start.

### JAX-vectorization of GHY (`ghy_ref.py`) — SCOPED 2026-09-27, not started

The rest of Track B's correctness-validated Python (ATURB, PBL, SURFACE, SEAICE/SSIDEC/snowice/
ADDICE/SIMELT, LAKES, tile aggregation) is now JAX-vectorized (batched arrays + `jnp.where`,
`jax.jit`-compilable, D14). GHY is the one holdout, and it is not comparable in difficulty — the
plan flagged this back at Phase 1 scoping ("expect it to be the longest single item in Phase 1") and
a full read of `ghy_ref.py` (1,253 lines) confirms why, concretely rather than by guess:

- **Two axes of genuinely variable per-cell array size**, not just branchiness:
  1. `n`, the number of active soil layers (1..`NGM`=6), set once per cell in `__init__` from the
     static soil-depth config (`dz`) — it does not change over a cell's own timesteps, so across a
     *batch* of cells it is a fixed-at-trace-time-per-lane integer. Fixed-size-array-plus-mask (the
     same technique used for `LMI`=4 in `seaice_core_jax.py`) covers this cleanly: pad to `NGM`+1
     slots, mask `k > n`.
  2. `nsn`, the number of active snow layers, which changes **within** a cell's own substep loop
     (0, 1, or 3 — `snow_redistr` only ever produces 1 or 3, never 2; 0 means no snow layer exists
     yet). Also coverable with a fixed 3-slot array plus a per-substep mask, but the mask is now
     substep-dependent, not just cell-dependent.
- **A genuinely variable per-cell, per-real-timestep iteration count**: `advnc`'s `for it in
  ent_iters` loop runs once per Ent-recorded adaptive substep, and that count varies cell-to-cell.
  Measured directly from the real `ffg_*.bin` record (9,036 real land cells, 3 dates): `ffnit` ranges
  **1 to 10**, with 38%/38%/14% of cells at 1/2/3 substeps respectively and a long thin tail out to
  10 (0 cells ever exceed the 11-slot recording buffer). **Checked empirically, and the obvious fix
  does NOT work**: padding every cell's substep list to a common max with `dts=0` no-op rows crashes
  outright (`snow_adv_1` divides raw water/heat amounts by `self.dts` -- `ZeroDivisionError` on real
  data, confirmed by running it); padding with a tiny nonzero `dts` (1e-6) instead avoids the crash
  but is **not** a no-op either -- `w` and per-substep temperatures stay put (diff ~1e-12) but the
  step's accumulator scalars (`tbcs`, `aruns`, etc.) move by **up to 1.8** on real cells, because
  several formulas (`drip_from_canopy`'s `dr`/`dr_scale`, among others) divide a *not*-proportionally-
  small quantity by `dts`, so as `dts`→0 those terms blow up rather than vanish. **Conclusion:** the
  padding must gate the whole substep body with a per-lane mask (`jnp.where` selecting "run this
  substep's full state update" vs. "keep the pre-substep state, this lane is done"), not attempt to
  make a real substep degenerate by shrinking `dts` -- the same "select between two branch-local
  candidate states" pattern used throughout `seaice_core_jax.py`, just applied at the substep-loop
  level instead of inside one function.
- **Two data-dependent `while`-loops** (`SNOW.f`'s `snow_redistr`, and `fllmt`'s negative-runoff
  redistribution) whose trip counts are bounded by small constants (`TOTAL_NL`=3 and `n`≤6
  respectively) — the same "bounded unroll with `jnp.where`-gated advance" technique used for PBL's
  Newton solve applies, just two more instances of it.
- **A small tridiagonal solve** (`heat_eq`, size 1–3) for the snow heat equation — fixed-size with
  masked rows, not a new technique.
- **A fixed-iteration (6-step) binary search** into the 65-entry soil hydraulic tables (`hydra`'s
  bisection into `THM`/`HLM`/`XKLM`/`DLM`, called twice per substep), with an `exact=True` branch
  (a bisection step landing on an exact table hit) that changes the loop body itself, not just the
  answer, on that iteration. **Checked empirically, not assumed rare:** instrumented `hydra` and ran
  it over all 9,036 real land cells (272,124 individual bisection calls) — `exact=True` fires
  **19,605 times (7.2%)**. Not a corner case; the vectorized bisection must handle it as a normal
  outcome (a per-iteration `jnp.where` branch, same technique as everywhere else here), not something
  approximated away.
- **~15 more `GhyColumn` methods** not yet touched here (`sensible_heat`, `drip_from_canopy`, `flg`,
  `flhg`, `fl`, `flh`, `runoff`, `fllmt`, `apply_fluxes`, `gdtm`, `snow`, `accm`/`accm_zero`/
  `accm_final`) each with 2-way bare/vegetated (`ibv`) masking and several smaller branches of their
  own — comparable in density to what `seaice_core_ff.py` had, just roughly 1.5–2x the line count.

**Net assessment:** every individual technique needed here already has a precedent elsewhere in
Track B (fixed-size masking, bounded-unroll while-loops, small tridiagonal solves, fixed-iteration
bisection) — this is not a new *kind* of problem, it is the same toolkit applied across more axes of
variability at once (per-cell layer count, per-substep snow-layer count, per-cell-per-step substep
count), which compounds the surface area for a transcription mistake without a comparable increase in
real-data volume to catch one (9,036 real land-tile records total, vs. tens of thousands for the
sea-ice pieces). Estimated effort: at least as large as all of `seaice_core_jax.py` (the sea-ice
vectorization, ~750 hand-derived lines across several sessions), likely 1.5–2x that. **Recommended
approach if/when this is picked up:** vectorize and validate one increment at a time against the
real dumps, in this order: (1) the per-lane substep-count mask (now known to need whole-body gating,
not a `dts` trick — see above; cheap once designed correctly, unblocks everything else), (2)
`hydra`/`xklh` (self-contained, reuses the bisection technique already proven in this project, now
known to need its `exact` branch handled as a normal ~7% outcome, not an edge case), (3) the non-snow
flux/runoff chain (`sensible_heat` through `apply_fluxes`, `fr_snow=0` cells only — per the
ATURB/PBL/SEAICE ledger this covers a majority of real cells and is the highest per-line-of-effort
payoff), (4) the snow model last (`SNOW.f`'s functions, the most novel and riskiest piece). Not
started; kept here as a scoping record, verified against real data on its two riskiest assumptions
rather than a partial, unvalidated implementation.

### Phase 2 — Radiation (6–10 weeks + a decision) — biggest single item, SCOPED 2026-09-27

Concrete finding from reading the real call chain (not guessed): P2SAoM40's CPP flags are
`USE_PLANET_RAD` + `GISS_RAD_OFF`, meaning the classic GISS graybody/correlated-k radiation is
compiled OUT and **SOCRATES is the sole active radiation scheme** — confirming the "S" in P2SAoM40
really is exercised, not a fallback.

- **Entry point:** `RCOMPX` (`RADIATION.f`), the single-column radiation subroutine, called from
  `RADIA` in `RAD_DRV.f` (the per-DTsrc-step driver, gated by `NRAD=5`). `RCOMPX` does not take its
  inputs as arguments — it reads/writes the `RADPAR` module's ~100+ scalars/arrays (temperature
  profile, water vapor, ozone, well-mixed gases, cloud optical properties, spectral surface albedo,
  orbital/zenith angle, aerosols). `RADIA` populates all of this per column in **~1,500+ lines** of
  setup code before each `RCOMPX` call (up to 5 calls/column for tracer diagnostics).
- Inside `RCOMPX`, when `USE_PLANET_RAD`+`GISS_RAD_OFF` are set, it calls `planet_rad.F90`'s
  `init_planet_rad` → `set_planet_alb_param` → **`run_planet_rad`** (the actual SOCRATES driver:
  `set_control_lw/sw`, `set_dimen`, `set_atm`, `set_cld`, `set_aer`, `set_bound_lw/sw`, then
  `radiance_calc` — the real SOCRATES library call, `libsocrates.a`, precompiled, modern F90 with
  derived types, no `BIND(C)` interfaces) → `get_planet_radout` → `deallocate_planet_rad`.
- **Implication for 2a (call SOCRATES from Python, the user's chosen approach, 2026-09-25):** the
  natural interception point is `RCOMPX` at the `RADIA` call site, not `run_planet_rad` directly —
  `RCOMPX`'s module-state inputs are what a real per-column call needs, and are themselves populated
  by ~1,500 lines of real, non-trivial GCM logic (this is not a small shim). Plan:
  1. Dump-hook `RADIA`'s `CALL RCOMPX` (all of `RADPAR`'s inputs before, outputs after) — same
     instrumented-model method used for every other module this project, at real, gated NRAD=5 steps.
  2. Write a `BIND(C)`/`ISO_C_BINDING` Fortran shim that: sets the same `RADPAR` module variables
     from flat C arrays, calls `RCOMPX`, and copies the outputs back out — compiled into a `.so`
     (the existing `libsocrates.a` is static; linking it into a shared object the shim exports is
     required for `ctypes`/`cffi` to load it).
  3. Python wrapper (`ctypes`) calling that `.so`, validated against the Phase 2.1 dump.
  4. Only then decide 2c (porting SOCRATES's own kernels to JAX) — 2a should be the whole of Phase 2's
     first deliverable, since it already gives a *real* SOCRATES radiation answer.
- **Started 2026-09-27 (proof-of-concept, not full radiation):** confirmed empirically that
`libsocrates.a`'s object files are not `-fPIC` (`ifort -shared -Wl,--whole-archive libsocrates.a`
fails with `relocation ... can not be used when making a shared object` on ~40 object files) --
`ctypes`/`cffi` cannot load a `.so` built from the existing archive without a full SOCRATES source
recompile with `-fPIC` (source found at `ModelE_Support/socrates/src/`, ~150 files, separate work).
Built and ran a small statically-linked Fortran driver (`fullfidelity/socrates_poc/gauss_angle_driver.f90`,
compiled with the same `ifort` 19.1.3 toolchain the real model uses) around one real SOCRATES kernel,
`gauss_angle` (the Gaussian-quadrature IR-flux solver called from `monochromatic_ir_radiance`), and
called it from Python (`socrates_py.py`) over a subprocess/stdin-stdout, getting back real,
physically-sane flux values from the actual compiled library -- no reimplementation. This settles the
calling *architecture* for Phase 2a (subprocess, not ctypes) but wraps one small kernel, not
`RCOMPX`/`run_planet_rad`. Next concrete steps: (1) dump-hook `RADIA`'s `CALL RCOMPX` for a real
oracle, (2) write the much larger driver program that sets the ~100 `RADPAR`/`planet_rad` inputs and
calls `run_planet_rad` (or `RCOMPX` directly) the same stdin/stdout way, (3) validate against (1). This is honestly the largest remaining unit of
  work in the project — larger than everything ported so far (Phase 0 + all of Phase 1) combined,
  by line count of real Fortran touched (~1,500+ lines of setup alone, before SOCRATES's own 90k-line
  library). It deserves a dedicated session using the same dump-hook methodology, not a rushed
  shortcut.
- **Continued 2026-09-28: read `RCOMPX`'s actual body (`RADIATION.f:1696-1942`) and the rundeck's
  RADPAR overrides, replacing the "~1,500 lines, ~100 scalars" estimate above with a concrete,
  traced call graph** for this exact build (`USE_PLANET_RAD`+`GISS_RAD_OFF`) -- a real narrowing,
  not a guess:
  - With `GISS_RAD_OFF` defined, the classic-GISS blocks inside `RCOMPX` itself (`TAUGAS`, `GETCLD`,
    `THERML`, `SOLARM`) are `#ifndef`'d OUT — they never run at all, confirming SOCRATES is not just
    "the active scheme" but the *only* code path inside `RCOMPX`, alongside a few small
    unconditional helpers.
  - **Aerosols/dust are effectively off, not just unused:** `RADPAR`'s defaults are `MADAER=0,
    MADDST=0` (never overridden in `ModelE_Support/huge_space/P2SAoM40/I`) and `NTRACE=0`, so the
    `IF(MADAER.ne.0.OR.NTRACE>0)` guard is always false — `getaer`/`getdst` are **never called**;
    `SRAEXT`/`SRASCT`/`SRAGCB`/`TRAALK` and the dust equivalents are just zeroed. Only `MADVOL=2`
    (the rundeck's one override) is active, so the only real aerosol-column work is
    `get_volc_column` + `getvol` (volcanic).
  - `GETEPS` (cloud heterogeneity, called unconditionally) turned out to be trivial for this
    rundeck: `KCLDEP=4` (default) makes it a **static per-(ILON,JLAT) climatology lookup**
    (`EPLOW`/`EPMID`/`EPHIG` arrays selected by pressure level), not a live computation — read once,
    index per column.
  - The real per-column work inside `RCOMPX`, in call order, is now known to be exactly: gas
    absorber amounts (`seth2o`, `getgas`, `fpxscalegas` — real atmospheric composition), volcanic
    aerosol column (`get_volc_column`+`getvol`), surface albedo/emissivity (`GETSUR` — genuinely
    substantial, takes ~25 named inputs covering every surface type's temperature/snow/ice state),
    cloud heterogeneity (`GETEPS`, now known-trivial), then `set_planet_alb_param` +
    **`run_planet_rad`** (the real SOCRATES call, explicit args `ULGAS, CLDEPS, PRNB, PRNX` plus
    whatever it reads from `RADPAR`/`planet_rad` module state internally — not yet traced), then
    `get_planet_radout`.
  - **`GETSUR` (`ALBEDO.f:281-866`, ~585 lines) read next, confirmed genuinely substantial, not
    reducible the way aerosols/`GETEPS` were:** computes per-surface-type (ocean/sea-ice/land-ice/
    land/lake) visible+near-IR+thermal-band albedo and emissivity, zenith-angle- and wind-speed-
    dependent ocean albedo, snow-age-dependent snow albedo, from ~25 named inputs (`POCEAN, POICE,
    PEARTH, PLICE, PLAKE`, per-surface-type temperatures `TGO/TGOI/TGE/TGLI`, snow/ice amounts
    `SNOWOI/SNOWD/SNOWLI/ZOICE/FMP/ZSNWOI`, vegetation fractions `PVT(12)`, wind `WMAG`, `COSZ`). A
    real if-branchy port, not a table lookup like `GETEPS` turned out to be -- but its input surface
    overlaps substantially with state this project's SURFACE/SEAICE/GHY modules already track
    (surface-type fractions, temperatures, snow/ice mass), which may make it more tractable than a
    from-scratch 585-line read suggests once ported; not yet attempted.
  - **Not yet done:** tracing `run_planet_rad`'s own full input surface (planet_rad.F90, not read
    this session), `getgas`/`getvol`/`GETSUR`'s own bodies in full detail, and everything from the
    original plan (dump-hook, BIND(C) shim, Python driver, validation). The estimate above (6-10
    weeks) is not revised by this narrowing — `GETSUR` and `run_planet_rad` are still real, and
    SOCRATES's own 90k-line library is unaffected — but the "setup" work is now known to be smaller
    and more concrete than "~1,500 lines, ~100 scalars" implied, which should make the next session's
    dump-hook step faster to scope correctly.

RADIA is 65.6% of Fortran runtime, and the graybody stand-in is the most
visible fidelity gap. Options, to be decided at Phase 0 from measurements:
- **2a. Call SOCRATES from Python** (ctypes/f2py wrapper of `libsocrates.a`)
  as the "real" radiation, keeping everything else JAX. Fast to reach F1;
  radiation is then *not* JAX-accelerated (honest: reported as such), and
  breaks the "pure Python" claim for that component.
- **2b. Port RAD_DRV/RADIATION/RAD_UTILS (19.5k) to JAX and use SOCRATES
  gas/aerosol tables as data.** Feasible only if the P2SAoM40 config uses the
  parts of SOCRATES reachable through ModelE's interface; needs code-path
  analysis first (most of the 90k lines are unused options).
- **2c. Port the needed SOCRATES two-stream/correlated-k kernels to JAX.**
  Largest effort, best end state (GPU-resident radiation); do only after 2a
  proves the interface and gives an oracle for 2c.
- Recommendation: **2a first** (weeks, gives F1 immediately and an oracle),
  **then 2c for the code paths actually exercised** if HQ needs an
  all-Python/GPU claim. NRAD=5 gating and the bimodal cost profile must be
  preserved.

### Phase 3 — Clouds / moist convection (4–6 weeks)
CLOUDS2/CLOUDS2_DRV (moist convection, large-scale condensation, cloud
fraction feeding radiation). Required for F1 to mean anything — radiation and
surface fluxes depend on it. Port with `lax.scan` over levels and per-column
plume logic; watch branch-heavy code (`where` evaluates all branches).

### Phase 4 — Atmospheric dynamics (8–12 weeks) — needed for F2
ATMDYN (advection, pressure gradient, coriolis, filters), MOMEN2ND, QUS
(second-order-moment advection of T/Q/tracers), STRATDYN. Global stencils on
the lat-lon grid with polar treatment and FFT filtering (FFT72). This is the
first phase with genuine communication/stencil structure; decide sharding
strategy (single-GPU is enough at 72×46×40) only after profiling.
- Gate F2: 1-day run vs. Fortran, chaos-aware metrics.

### Phase 5 — Ice dynamics and ocean — SCOPED 2026-09-28, COMMITTED (was "stretch, decision point")

**Status change:** the 2026-09-24 estimate below (kept for history) recommended NOT committing to this and
deciding later. User decision 2026-09-28, after D25 exposed that ice/lake/ocean state cannot be carried across
DTsrc steps the way land's can (real precipitation/moist-convection, sea-ice dynamics, and the full ocean model
all act on that state between steps): commit to the full faithful port, not a reduced ocean. This is now the
largest single item in the project.

**Why this is being done at all:** D25 (two-substep land chain) exposed that ice/lake/ocean state cannot be
carried across DTsrc steps the way land's can, because real precipitation (moist convection), sea-ice dynamics
and the ocean model all act on that state between one step's `SURFACE`/`GROUND_SI`/`GROUND_LK` and the next
step's. User decision (2026-09-28): port these too, at the same rigor as everything else — not a reduced or
approximate treatment. This is now the largest single item in the project, larger than everything done to date.

### Real per-DTsrc-step call order (traced from `MODELE.f:315-340`, corrects the earlier "SURFACE ends the
per-cell physics" framing)
```
call atm_phase1
CALL PRECIP_SI(si_ocn, iceocn, atmice)   ! precip onto sea ice -- ocean-grid state (si_ocn), NOT si_atm
CALL PRECIP_OC(atmocn, iceocn)           ! precip onto open ocean
CALL SURFACE                             ! everything chained in D18-D25 (atm-grid si_atm GROUND_SI/GROUND_LK inside)
call ocean_driver                        ! DYNSI, UNDERICE, GROUND_SI(ocean-grid!), CALC_APRESS, OCEANS, FORM_SI, ADVSI
call atm_phase2
```
`ocean_driver` (`OCN_DRV.f:3-60`, real body is the `#else` branch of `#ifdef CUBED_SPHERE` — **not cubed-sphere
here**, confirmed: rundeck uses `Atm72x46`/lat-lon, not `CUBED_SPHERE`) runs, in order: `seaice_to_atmgrid`,
`DYNSI(atmice,iceocn,si_ocn)` (called once, not twice — the twice-call in the source is the dead
`#ifdef CUBED_SPHERE` branch), `UNDERICE`, `GROUND_SI(si_ocn,...)`, `CALC_APRESS`, `OCEANS`, `FORM_SI`,
`seaice_to_atmgrid` (again), `ADVSI`, `SI_diags`.

**Real, new finding: `GROUND_SI`/`FORM_SI`(=ADDICE) are called TWICE per step, on two grids.** Once inside
`SURFACE` on the atmosphere-grid ice state (`si_atm` — this is everything D17/D20/D21/D24 already chained), and
once inside `ocean_driver` on the ocean-grid ice state (`si_ocn`). Grids are the SAME resolution here (both
72x46, confirmed: no `CUBED_SPHERE`, `Atm72x46` for atm, ocean also 4x5deg per `OCEAN_hycom`-adjacent naming /
`o: dynamic 4x5 horizontal resolution with 13 layer ocean` in the rundeck header) but are logically separate
state arrays reconciled by `seaice_to_atmgrid`, so this is real new plumbing to trace (not just "call the same
function again"), though the underlying subroutine bodies (`sea_ice`/`ssidec`/`snowice`/`addice`) are already
ported and validated (D10/D12/D14/D17) and should apply directly to the ocean-grid call once its own inputs are
traced and dumped.

### Dead code confirmed and excluded (checked against P2SAoM40.R, not assumed)
- `#ifdef CUBED_SPHERE` branches throughout `ICEDYN_DRV.f`/`OCN_DRV.f` (cs2ll regrid, `ICE2CSint`, height-point
  handling) — rundeck is lat-lon (`Atm72x46`), confirmed no `CUBED_SPHERE` define.
- Ocean tides: `OTIDEV`/`OTIDEW` calls in `OCNDYN2.f` are gated `If (OTIDE > 0)`; rundeck sets `OTIDE = 0`
  (`decks/P2SAoM40.R:175`) — never executes. `OTIDELL.f` (tide-generating-potential source) doesn't even exist
  in this source tree despite being named in the rundeck's component list — a further sign it's inert here.
- Ocean tracers (age/vent/gasx/CFC/watermass) and all `TRMO`/`TXMO`... moment arrays: gated `#ifdef
  TRACERS_OCEAN`, not defined for P2SAoM40 (no `TRACERS_OCEAN` in the rundeck). `CARBON`/`NITR` diagnostic calls
  are gated the same way.
- Ocean biogeochemistry (`obio_model`, `#ifdef TRACERS_OceanBiology`): not defined for P2SAoM40 — dead code.
  This also means `OCNGISS_TURB`/`OCNGISS_SM` (linked components, but need to confirm whether their real bodies
  are called outside the obio-only path — **not yet checked, do before starting KPP/mixing work**).
So the real prognostic ocean state for this rundeck is only **mass, heat (G0M), and salt (S0M)** — 3 fields, not
the full N-tracer machinery the source supports. `use_qus` (a runtime flag, need to confirm its value for this
rundeck) selects `OADVT3` (a fancier flux form) vs `OADVT2` (the simpler moment-advection path) for the same
heat/salt advection — check before assuming which one runs.

### Real line counts (subroutine bodies, not whole files — whole-file counts overcount unrelated subroutines,
learned the hard way earlier in this project)
| Piece | File | Lines | Note |
|---|---|---|---|
| `PRECIP_SI` | `SEAICE_DRV.f` | 140 | driver, per-cell loop |
| `PREC_SI` | `SEAICE.f` | 246 | the real physics; reuses `get_snow_ice_layer`/`relayer`/`relayer_12`/`set_snow_ice_layer`/`tice`/`Mi`/`Em` (**already ported**, D10/D14); needs one new function `Fi` (~20 lines, read in full, trivial) |
| `DYNSI` (own body) | `ICEDYN_DRV.f:328-877` | 550 | regrid (A-grid<->B-grid ice-dyn grid), air/water stress setup, pressure-gradient force, calls `VPICEDYN` then post-processes stress/velocity back to A-grid |
| `VPICEDYN` | `ICEDYN.f:1233-1353` | 121 | outer pseudo-timestep loop (up to 20 iters, RMS-velocity convergence check via `GLOBALSUM` over the whole ice-dyn grid) calling `FORM` then `RELAX` twice per iteration (predictor + modified-Euler corrector) |
| `FORM` | `ICEDYN.f:154-293` | 140 | **not yet read in detail** -- forms the nonlinear viscous-plastic rheology terms (strain rates -> `ETA`/`ZETA` viscosities) from the current velocity guess |
| `RELAX` | `ICEDYN.f:390-882` | 493 | **read enough to de-risk, not line-by-line yet**: uses `TRIDIAG_new`/`TRIDIAG_cyclic` (line-by-line ADI-style implicit sweep, U-direction then V-direction) for the actual linear solve -- NOT a generic sparse/iterative solver. This is the same class of operation already handled in this project (GHY's `heat_eq` tridiagonal solve, D9/D15), just larger (per-row/column across the whole ice-dyn grid) and applied inside the outer pseudo-timestep loop |
| **DYNSI core solve subtotal** | | **1,304** | corrects the earlier 550-line estimate, which only counted `DYNSI`'s own body, not its callees |

**Grid resolution finding (de-risks the regrid layer):** `IMICDYN`/`JMICDYN` are set by a compile-time
`#ifdef CUBED_SPHERE` gate in `ICEDYN.f:1385-1399` -- `IMICDYN=2*IM, JMICDYN=2*IM` under cubed-sphere,
**`IMICDYN=IM, JMICDYN=JM` otherwise**. P2SAoM40 is confirmed not cubed-sphere (`Atm72x46`), so the
ice-dynamics grid is the SAME resolution as the atmosphere (72x46), just B-grid-staggered (not a genuine
spatial refinement) -- the A-grid<->B-grid regrid (`band_pack`/`pack_a2i`, since `#ifndef CUBED_SPHERE`) is
a same-resolution staggering operation, not cross-resolution interpolation.

**Implementation design implication:** the outer pseudo-timestep loop is a data-dependent iteration count
(convergence-based, up to 20) requiring `jax.lax.while_loop` with a whole-grid RMS reduction each
iteration -- a new control-flow pattern for this project (everything so far has used bounded `lax.scan`/
Python unrolls). The inner linear solve (`RELAX`) being tridiagonal, not generic-sparse, is the key
de-risking finding: it fits the same "small bounded solve, vectorized across many independent instances"
pattern already proven in GHY (D9/D15), just applied per grid row/column instead of per soil layer.

**RELAX's full structure, now read in full (not just skimmed) -- this is the last major unknown, now resolved:**
It is a textbook 2-step ADI (Alternating Direction Implicit) method, run once for U then once for V (4 tridiagonal
solves total per `RELAX` call):
1. **U, I-direction (cyclic)**: builds `AU/BU/CU` (rheology + drag + mass/dt coefficients, per grid point) and
   `URT` (a RHS built from `FXY`, itself built from **`UICEC`/`VICEC`** -- the frozen copy taken at the *start* of
   the current pseudo-timestep, not updated mid-sweep). Solves `TRIDIAG_cyclic` along I, independently for every
   J row -- **a batched cyclic tridiagonal solve, rows are mutually independent** (matches `aturb_ff.py`'s
   existing tridiag usage and GHY's `heat_eq`, D9/D15, just applied along a grid direction instead of soil depth).
2. **U, J-direction**: builds new `AV/BV/CV`/`VRT` -- crucially, `VRT`'s I-neighbor terms now use the
   **just-updated** `UICE(I+1/-1,J,1)` from step 1 (a real, deliberate sequential dependency on step 1's result),
   while J-neighbor terms stay implicit (going into the solve). Solves `TRIDIAG_new` along J (domain-decomposed,
   `grid_ICDYN`-aware) -- **independent for every I column**.
3. **V, I-direction**: same shape as step 1 but for V, using **frozen `UICEC`** again (a new freeze point, not
   step 1/2's updated U).
4. **V, J-direction**: same shape as step 2 but for V, using step 3's just-updated V for I-neighbor terms.

So within `RELAX`, the four solves are strictly sequential (each depends on the previous one's output), but
**each individual solve is a batch of independent 1D tridiagonal solves** or its transpose -- ideal for
`jax.vmap`/batched Thomas-algorithm solves, the same pattern already proven correct and fast elsewhere in this
project. `PLAST` (94 lines, called from `FORM`) is pure fixed-stencil per-grid-point arithmetic (finite-difference
strain rates -> nonlinear viscosity via the elliptical yield curve, clamped to `[ZMIN,ZMAX]`) -- as tractable as
GHY's flux calculations, no new technique needed.

**DONE, see D29.** `DYNSI`'s own body's geometry setup (`GEOMICDYN`/`ICDYN_MASKS`) and the full numerical core
(`FORM`, `PLAST`, `RELAX`'s 4 ADI sub-solves, `VPICEDYN`'s outer convergence loop) are ported to plain Python
(`fullfidelity/icedyn_geom_ff.py`, `fullfidelity/icedyn_dynsi_ff.py`) and validated bitwise/float64-exact on all
18 real records (6 steps x 3 dates). Three real bugs found and fixed along the way (`osurf_tilt` default,
`BYDTS` unit convention, an `AA1+AA2`-vs-`AA3+AA4` transcription slip in `RELAX`'s `VICE` equations) -- see D29
in `FULL_FIDELITY_DELTAS.md` for the full bisection story. Remaining for this item: the JAX/batched version
(`jax.lax.while_loop` for the outer convergence loop, batched column tridiagonal solves -- both still just
designed, not implemented) now that a confirmed-correct plain-Python reference exists to validate against; and
the earlier atm-stress/ocean-current regrid inside `DYNSI`'s own body (`GAIRX`/`GAIRY`/`GWATX`/`GWATY`/`PGFUB`/
`PGFVB`, `HEFF`/`AREA`/`AMASS`/`COR` derivation), still taken as recorded/real inputs since it depends on
ocean-model fields (`OGEOZA`/`UOSURF`/`VOSURF`) not yet ported.
| `UNDERICE` | `SEAICE_DRV.f:187-375` | 188 | **DONE, see D32** — `iceocean_fluxes`/`icelake_fluxes` (the real physics) ported/validated exact; `Tm`/`Sm`/`mlsh`/`Ustar` (ocean-model fields) recorded as real inputs, same pattern as D29 |
| `CALC_APRESS` | `SEAICE_DRV.f:10-43` | 33 | **DONE, see D30** — trivial, bitwise exact, first try |
| `seaice_to_atmgrid` | `SEAICE_DRV.f:1716-1819` | 103 | **DONE, see D31** — state copy + GTEMP/GTEMP2/GTEMPR/ZSNOWI/ZSI/FWSIM derivation ported/validated; the 3rd loop's `RESET_SURF_FLUXES` call is radiation-adjacent (touches `RAD_COM`'s FSF/TRSURF only) and deliberately not ported |
| `FORM_SI` (=ocean-grid ADDICE) | `SEAICE_DRV.f` | 198 | driver around the **already-ported** `addice`/`simelt` (D12/D14); likely small new work, mostly plumbing |
| **Ice-dynamics subtotal** | | **~1,460** | tractable, comparable to work already done |
| `OCEANS` (driver) | `OCNDYN2.f:33-699` | 666 | the **live** driver (see correction below); calls into `OCNDYN2.f`'s own routines plus `GROUND_OC`/`PRECIP_OC`/`OSOURC`/`OCOAST` in `OCNDYN.f` |
| `OCNDYN.f` | | 6,062 total; **legacy dynamical core confirmed dead, D36** | **Corrected 2026-09-29 (D36)**: `OCNDYN.f`'s own leapfrog-dynamics driver (`OCEANS_old`, `OCNDYN.f:18-289`) is entirely commented out, and grepping every call site of its `OVtoM`/`OMtoV`/`OFLUX`/`OPFIL`/`OADVM`/`OADVV`/`OPGF`/`OPGF0`/`OBDRAG`/`OSTRES` project-wide found **none are called from anywhere live** — all superseded by `OCNDYN2.f`'s rewrite (`OFLUXV`, `ODHORZ`/`ODHORZ0` [D40], `OPFIL2`, `OSTRES2`, `OBDRAG2`, the `OADVT2` family). Do not port these from `OCNDYN.f`. What's still live and real in `OCNDYN.f`: `PRECIP_OC` (84 lines, **DONE, see D33**), `OSOURC` (123 lines, **DONE, see D34**), `GROUND_OC`'s below-freezing sweep (**DONE, see D35**), `OCOAST` (57 lines, shared unchanged with `OCNDYN2.f`, **DONE, see D37**, bitwise exact first try), plus `CONSERV_O*`/`CHECKO*` diagnostics (likely skippable) and one-time init/restart-I/O (`init_OCEAN`, `daily_OCEAN`, etc., not per-step work). `GLMELT` (72 lines) scoped and **deferred** (D36): fires only once per calendar day via `daily_OCEAN`'s `end_of_day` gate, not confirmed to occur within the existing 6-step/3-hour test windows. `OSOURC`'s `FSR`/`FSRZ`/`LSRPD` dependency is fully analytic (D34); `GROUND_OC`'s `SHCGS` is the one genuine external-file dependency (`OFTAB`, recorded at its use point, D35). Confirmed via `ORES_5x4.F90`: ocean grid IMO=72,JMO=46, **same resolution as the atmosphere** — `AG2OG_precip`/`OG2AG_*` (atm<->ocean regrid) can be treated as a recorded-input boundary. |
| `OCNDYN2.f` | | 2,670 total | **The real live Stage-2 dynamical core (D36 correction)**. `SUBROUTINE OCEANS` (33-699, table row above) is the actual driver. `OSTRES2` (2478-2577) is **DONE, see D36**. `OBDRAG2` (2592-2682, 91 lines) is **DONE, see D38** — confirmed `OCN_GISS_TURB` not `#define`d, tidal-enhancement branch dead. The polar UOD/VOD relax block + `polevel()` (179-228, 1530-1564, ~90 lines) is **DONE, see D39**. `ODHORZ0` (1718-1862, 144 lines, pressure/EOS prep) is **DONE, see D40** — confirmed `USE_OPGFQ=0`; `VOLGSP` recorded at its outputs; caught a North Pole masking design gap (`nbyzm` restricts J=JM to I=1 only) via the dump comparison. `ODHORZ` (1185-1717, 532 lines, the actual horizontal-momentum + mass-continuity solve `ODHORZ0` preps for) is **DONE (plain-Python only), see D42, extended D44** — `OPFIL2`'s two per-layer outputs recorded directly, decoupling from the `OPFIL2`/`AVR`-file dependency; two real bugs caught before any validation run (missing `HOCEAN` recording; `OPBOT`'s cross-layer accumulation reset incorrectly); float64-tolerance exact on all 15 real call records; `OMEGA` validated empirically at Earth-standard; JAX vectorization deliberately deferred as its own follow-up (GHY-lesson discipline). **D44 extension**: `ODHORZ` also accumulates `SMU`/`SMV` (`OCEAN_DYN`'s integrated horizontal mass fluxes, `:1351,1443`) — a real gap found while scoping the tracer-advection family below (D42 never captured it since nothing D42 validated needed it); `NOCEAN=1` for this rundeck confirmed via `OCEAN_COM.f`, so the accumulation reduces to "once per `OCEANS` call", 2 of 5 `ODHORZ` calls per window contributing (`qeven=.true.`); bitwise-exact, first try, both the per-call replay and the full chained accumulation against a new ground-truth `ffz_smfinal` dump. `OFLUXV`+`OADVUZ` (711-800, 2471-2521, ~125 lines, long-timestep vertical mass redistribution) is **DONE, see D43** — confirmed, contrary to an earlier scoping note, that `OFLUXV` does NOT call `OPFIL2` at all (the polar-filter setup module just sits next to it in the source); two real bugs caught (a `DXYPO`/`DTOLF` bookkeeping error; a JAX dense-vectorization NaN at inactive columns, fixed via an explicit active-cell mask through `OADVUZ`'s `lax.scan` carry); bitwise/float64-exact, both ports, all 3 dates. `OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2` (tracer advection family, ~570 lines) is **DONE, see D45** — confirmed `OADVT2` is a thin Strang-splitting dispatcher (X half-step, Y, Z, X half-step again), called twice per `OCEANS` invocation for `G0M`/`S0M` since `TRACERS_OCEAN` is not defined for this rundeck, each call re-deriving `MA` identically from the same `SMU`/`SMV`/`SMW` inputs. **Three real bugs found and fixed**: (1) `OADVTX2`'s `mudt` array is a single persistent array with non-obvious staleness semantics, and its segments (via `OCNDYN.f:1494`'s `get_i1i2`) are linear not circular, with a single-cell-segment skip not caught on first read; (2) `MMI` (the mass `OADVT2` actually advects) had to be dumped directly rather than re-derived from `MO0*DXYPO(J)`, since it's a persistent module array `ODHORZ0` only partially overwrites; (3) `OADVTZ2` needed the same `nbyzm`-restricts-J=JM-to-I=1 pole mask established in D40. A fourth fix (`np.sum()`'s pairwise reduction vs ifort's sequential `SUM`) closed a small pole-row rounding mismatch. Bitwise-exact on all 9 checked fields, all 3 dates, after the fixes. JAX deferred (dynamic segment structure + pole masking make it a poor first batching candidate). **Scoping conclusion (post-D45): `OCNDYN2.f`'s entire real per-step dynamical core is now ported** — `OFLUXV`/`ODHORZ`/`ODHORZ0`/`OADVT2` family all done. |
| `OCNQUS.f` (advection) | | 1,846 total; **confirmed entirely dead code for this rundeck, 2026-09-29** | **Resolved the earlier "may overlap, needs disambiguating" flag.** `OCNQUS.f` holds `OADVT3`/`OADVTX3`/`OADVTY3`/`OADVTZ3`/`OADVTX4`/`OADVTY4`/`OADVTZ4` — the "Quadratic Upstream Scheme" (QUS, full 2nd-order moments including cross terms XY/YZ/ZX), selected only when `USE_QUS==1` (`OCNDYN2.f:365`: `if(use_qus==1) CALL OADVT3(...) else CALL OADVT2(...)`). `USE_QUS` defaults to 0 (`OCEAN_COM.f`) and is **not overridden in `decks/P2SAoM40.R`** — so `OCNDYN2.f`'s own simpler `OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2` family (linear moments only, already scoped above) is the live one for this rundeck; all of `OCNQUS.f` can be skipped. Do not port anything from this file. |
| `OCNKPP.f` (vertical mixing) | | 3,714 total | **Scoped 2026-09-29 (D41 exploration), not yet ported.** Real subroutines: `KPPMIX` (225-1315, the actual K-Profile-Parameterization boundary-layer mixing scheme — Large/McWilliams/Doney-style, genuinely intricate), `OCONV` (1361-3032, the main per-step driver that calls `KPPMIX` per column — confirmed live, called from `OCEANS`), `STCONV` (3032-3410, straits convection), `OVDIFF`/`OVDIFFS` (3410-3517, vertical diffusion solvers). `OCN_GISS_TURB`/`OCN_GISS_SM`-guarded branches within this file are dead (confirmed not `#define`d, same finding as D38's `OBDRAG2`). The live core (`KPPMIX`+`OCONV`, ~2,800 lines) is a "new architecture"-scale item, likely the single largest remaining piece — **not started**. |
| `OCNMESO_DRV.f`+`OCNTDMIX.f`+`OCNGM.f` (mesoscale/GM-Redi mixing) | | 1,217+2,030+1,270 = 4,517 total | **Corrected and substantially rescoped, D46 (2026-09-30).** D41's exploration had it backwards: `CONSTANT_MESO_DIFFUSIVITY` does **not** mean a simplified path — it only fixes the mesoscale-diffusivity *coefficient* (`meso_diffusivity_const=800` m²/s, set via `get_1d_mesodiff`, `OCNMESO_DRV.f:1187-1216`, trivial). The full Redi/Gent-McWilliams **skew-flux** machinery still runs: confirmed `use_tdmix=0` (default, not overridden in the rundeck) takes `ocnmeso_drv`'s (131-432) `else ! skew-GM` branch (393-421), calling `OCNGM.f`'s `GMKDIF` (125-323, density-gradient-derived isoneutral slopes, itself calling `ISOSLOPE4` at `OCNGM.f:997-1132` and `GET_PSI_DIAG` at `OCNGM.f:1133-1270`) then `GMFEXP` (325-494, applies the skew flux to `G0M`/`S0M`, itself calling `computeFluxes`/`wrapAdjustFluxes`/`addFluxes`, `OCNGM.f:496-996`). `ocnmeso_drv`'s own `densgrad` helper (434-577) computes horizontal/vertical density gradients via `VOLGSP` (the same seawater-EOS table as D35/D40, real-input-recording pattern applies) and depends on a previously-unscoped routine, `ocnstate_derived` (`OCNDYN2.f:1568-1706`, called twice per `OCEANS` — cell-centered `G3D`/`T3D`/`S3D`/`P3D`/`R3D`/`V3D` thermodynamic state). `OCNTDMIX.f` (2,030 lines) is confirmed **entirely dead** for this build (`use_tdmix=0`, the whole `if(use_tdmix==1)` block at `ocnmeso_drv:235-391` — including every `OCNTDMIX.f` call — never executes); do not port it. `simple_mesodiff`/`SIMPLE_MESODIFF` and `ORIG_MESODIFF`'s `make_k3d_cellcenter`-for-K3D path are confirmed dead (superseded by `USE_1D_MESODIFF`, implied by `CONSTANT_MESO_DIFFUSIVITY`). **`ocnstate_derived` + `densgrad`'s vertical-gradient portion + `get_1d_mesodiff` are DONE, see D47** — bitwise-exact on `G3D`/`S3D`/`P3D`/`VBAR`/`RHO`/`DZV`/`BYDZV`/`BYDH`/`RHOMZ`/`BYRHOZ`, all 3 dates, first try (no bugs; one cleanup, a North-Pole `m_active` mask reused from D40 to avoid a cosmetic divide-by-zero that a downstream pole-copy was already correctly overwriting). D47 also corrected D46's note that `ocnstate_derived` is called twice — the first call site is gated by `#ifdef TRACERS_OceanBiology`, not defined for this rundeck, confirmed via the real dump (1 record/itime, not 2) rather than assumed. `densgrad` reuses D40's already-validated `DH3D` (`ODHORZ0`'s output) directly rather than re-instrumenting it. `RHOX`/`RHOY` (densgrad's horizontal gradients) recorded as final outputs, not further decomposed. **`GMKDIF`/`ISOSLOPE4`/`GMFEXP` fully read (D48 scoping), two more real scope reductions found.** `GET_PSI_DIAG` (`OCNGM.f:1133-1270`, called from `GMKDIF`) confirmed **purely diagnostic** — writes only `OIJL`/local scratch, never read back by `GMFEXP` or anything else; skip entirely (same "skip pure diagnostics" precedent as D31's `RESET_SURF_FLUXES`). **`QCROSS` is always false for this rundeck's actual call**: `ocnmeso_drv` calls `gmkdif(k3d,1d0)` with `RGMI_in` hardcoded to `1d0`, and `QCROSS = .NOT.(RGMI.eq.1d0)` — eliminating every `IF(QCROSS)` branch in both `GMKDIF` (the `DXZ`/`CDXZ`/`BXZ`/`CXZ`/`EXZ`/`CEXZ`/`DYZ`/`BYZ`/`CDYZ`/`CYZ`/`CEYZ`/`EYZ` cross-term coefficient arrays, roughly half of its coefficient-setting logic) and `GMFEXP` (the `FXZ`/`FYZ` off-diagonal flux terms) as dead code for this build. `GIJL` updates throughout (`GMFEXP`/`wrapAdjustFluxes`/`addFluxes`) are diagnostic-only, same skip. **`ISOSLOPE4` is DONE, see D49** — bitwise-exact on all 24 output arrays (`AIX0-3`/`AIY0-3`/`ASX0-3`/`ASY0-3`/`S2X0-3`/`S2Y0-3`), all 3 dates, after fixing a real pole-row loop-bound bug (the main loop genuinely includes J=JM, unlike almost every other routine in Stage 2 which excludes it — caught via suspiciously round max-diffs on 8 of the 24 fields, the other 16 masked by `RHOX`/`RHOY` happening to be zero at the pole). Real inputs are exactly D47's `densgrad` outputs + `get_1d_mesodiff`'s constant K3D, no new input instrumentation needed. **Still to port: `GMKDIF`'s remaining (post-`QCROSS`) coefficient-setting logic (~100) + `GMFEXP` (~120, post-`QCROSS`) + `computeFluxes` (~90) + `wrapAdjustFluxes` (~110, the QLIMIT=true/salt path, includes a global-sum-based conservative limiter) + `addFluxes` (~145, the QLIMIT=false/enthalpy path) ≈ 565 lines** — the actual flux application, still a genuinely intricate anisotropic scheme. Not yet ported. `OCNGISS_TURB.f`/`OCNGISS_SM.f` (904+348=1,252) confirmed dead (their `#ifdef` guards not defined for this build, same as `OCNKPP.f`'s finding) — do not port these two files. |
| `OSTRAITS.f`+`OSTRAITS_COM.f` (straits) | | 1,036+183 = 1,219 | parameterized narrow channels (real geography, `OSTRAITS=OSTRAITS_72x46.nml` in rundeck) — **not yet read** |
| **Ocean-core subtotal (excl. `OCNGISS_TURB.f`/`OCNGISS_SM.f`, confirmed dead D41)** | | **~15,778** (~17,624 minus `OCNQUS.f`'s confirmed-dead 1,846) | before excluding any further dead code within `OCNKPP.f`'s/`OCNMESO_DRV.f`'s own preprocessor branches (partially traced, not fully quantified) or `OSTRAITS.f` (not yet checked) |
| **Grand total, everything in this item** | | **~17,150+** (revised down from ~19,000+ after `OCNDYN.f`'s D36 dead-code correction and `OCNQUS.f`'s D41 dead-code finding) | still comparable to or larger than the entire rest of this project (D1-D25) |

### Plan
1. **Stage 1 — ice dynamics** (this is what actually closes the D25 gap for ice/lake state carry-over):
   `PRECIP_SI`/`PREC_SI` -- **DONE, see D26**. `PRECIP_LK` -- **DONE, see D27**. `PRECIP_LI`/`PRECLI` --
   **DONE, see D28** (3,723 real cells). `DYNSI`/`VPICEDYN`/`FORM`/`PLAST`/`RELAX` (the 1,304-line numerical
   core) -- **DONE, see D29**, bitwise/float64-exact on all 18 real records. `IRRIG_LK`/`irrigate_extract`
   (373 lines, external dataset dependency) deferred as its own item; next is `UNDERICE`, `CALC_APRESS`,
   `seaice_to_atmgrid`, ocean-grid `GROUND_SI`/`FORM_SI` plumbing. New Fortran instrumentation needed (no
   existing dump covers these) — batch all remaining Stage 1 dump hooks into one patch set and one
   rebuild+rerun, matching how the original oracle build batched multiple subroutines' hooks together.
2. **Stage 2 — ocean core**: read `OCNDYN.f`, `OCNQUS.f`, `OCNKPP.f`, `OCNMESO_DRV.f`/`OCNTDMIX.f`/`OCNGM.f`,
   `OSTRAITS.f` in full before estimating further (the ~19k figure above is file-level, not yet
   subroutine-level or dead-code-excluded the way Stage 1's figures are) — do not commit to a firmer schedule
   until that reading is done, the same discipline used for radiation's `GETSUR`/`RCOMPX` scoping.
3. Validate every piece against real Fortran with the same dump-hook-and-validate method used throughout
   (F0: bit-for-bit/float64-rounding match on real dumps; JAX-vectorize only after F0 is proven, per the GHY
   lesson — unrolling a big per-cell loop, or here a big per-timestep grid solve, naively can be a *regression*,
   not a speedup, so measure before declaring victory).
4. User has explicitly chosen the full faithful port (not a reduced/mixed-layer-only ocean) — do not silently
   substitute a simplified ocean; if a genuine simplification looks warranted after Stage 2 reading, surface it
   as a decision, the same way SOCRATES-as-black-box and this ocean scope itself were surfaced.



## 5. Capturing deltas (the requirement that findings "complement" existing ones)

- **Tracks, not overwrites.** Existing results are labelled *Track A:
  representative driver* (tag = current state). New results are *Track B:
  full-fidelity*. STATUS.md gets a new "Full-fidelity port" section; Track A
  text is not edited except to link.
- **Delta ledger** (`FULL_FIDELITY_DELTAS.md` + machine-readable
  `deltas/*.json`), one row per phase/module, recording for Track A vs. Track
  B vs. Fortran oracle: accuracy metrics (max/RMS/correlation per field),
  step time (CPU, GPU-single-call, GPU-chained), lines of code ported, and
  the reason for any change. Generated by the Phase 0 harness so it is
  reproducible, not hand-typed.
- **Performance expectations to record honestly:** faithful physics is more
  expensive than the stand-ins; the 356× / 5.8× figures are for the simplified
  chain and **will drop**. The delta ledger shows by how much, and the headline
  for HQ is speed *at full fidelity*, measured the same way (single-call and
  chained, CPU and GPU).
- **Deck:** extend `status_slides/` with a Track A vs. Track B slide per
  phase; reuse the verification-slide pattern (reproducibility, independent
  plausibility check, scope caveats).
- Git: one branch, small commits per module with the tests; tag at each rung
  (`ff-F0-…`, `ff-F1`, …).

## 6. Risks and decisions

| Risk | Consequence | Mitigation |
|---|---|---|
| Cannot instrument/rebuild Fortran | Only restart/acc validation → weak fidelity claim | Resolve in Phase 0 before any porting; state fallback explicitly |
| SOCRATES dependency (90k lines) | Radiation dominates effort | 2a → 2c staging; measure exercised code paths |
| Chaos makes multi-step comparison ambiguous | Cannot prove "match" beyond F1 | Noise-floor + ensemble metrics agreed up front |
| Scale (>55k lines excluding radiation lib/ocean) | Schedule | Phase gates, each independently valuable; stop at F1 if needed |
| Performance regression vs. Track A | HQ sees headline speedup vanish | Set expectation now; report fidelity-vs-speed ledger |
| Placeholder/rigged validation recurring | False confidence | Rules in Method §3.1–3.2; mutation-check every test |
| JAX limitations (NaN in dead `where` branches, long unrolled chains) | Bugs / slowness | Use `scan`, safe-branch pattern, follow Round 2 findings |

**Decisions needed from HQ/user (not blocking Phase 0):**
1. Target rung: F1 hard gate + F2 (recommended), or F3.
2. Radiation: is calling SOCRATES from Python (2a) acceptable, or must radiation be pure JAX (2c)?
3. Is ocean GCM in scope (Phase 5)?

## 7. Immediate next steps (Phase 0, first two weeks)
1. Verify ModelE build/run reproducibility from `P2SAoM40.mk` and the
   restart; record the run recipe in this directory.
2. Write the dump-instrumentation patch and the oracle manifest format.
3. Implement the comparison harness + noise-floor measurement on the
   *existing* Track A driver so the first delta row (Track A vs. real Fortran
   at F1) exists before any new physics is ported.
4. Commit the Round 2 work on this branch (currently uncommitted in the
   working tree) so Track A is fully captured at the branch point.

## 8. Confidence notes
Line counts are from `wc -l` of the ModelE tree on 2026-09-24 (CLOUDS2.F90,
the SOCRATES source count and OCN files not all measured; ocean is a lower
bound). Effort estimates are judgment, not measured. The claim that F1/F2
are achievable without bitwise agreement is a methodological position to
confirm with HQ, not a result.
