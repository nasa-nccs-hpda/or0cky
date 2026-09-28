# Phase 0 log — oracle & harness (full-fidelity-port)

## 2026-09-24: build/run environment found (gate item 1)

**The real ModelE can be built and run from this node.** Recipe recovered from
the earlier session logs (`decks/E1oM20_Test_session_2026-09-17.md` and the
P2SAoM40 build transcript):

- Toolchain: Intel `ifort` 19.1.3 (`intel/2020Update4`) + `netcdf4/4.9.3s`,
  flags `-O2 -ftz -convert big_endian -assume protect_parens -fp-model strict`,
  NETCDFHOME `/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s` (from `~/.modelErc`),
  built with `gmake setup RUN=P2SAoM40` in `modelE2_planet_2.0/decks`.
- Load with **Lmod directly** (`source /usr/share/lmod/lmod/init/bash`); the
  legacy `/usr/share/Modules/init/bash` (used by generated `.qsub` scripts)
  cannot see these modulefiles. The intel module also does not put `ifort` on
  PATH in a non-login shell — `env_modele.sh` prepends the compiler dir.
  Use: `source fullfidelity/env_modele.sh`.
- Verified here: `ifort --version` = 19.1.3.304; F77-style NetCDF
  (`include 'netcdf.inc'`) compiles and opens the real `fort.1.nc`.
- **Gotcha for dump code:** `use netcdf` (F90 module) fails with ifort — the
  installed `netcdf.mod` was built by gfortran. The model itself avoids this
  (uses `netcdf.inc`). Instrumentation must use `netcdf.inc` or plain
  Fortran stream files (`access='stream'`), which NumPy reads directly.
  Stream files are the simpler choice.

## Reproducibility oracle available

Original run left two checkpoints: `fort.1.nc` = 1950-11-26 00:00 and
`fort.2.nc` = 1950-12-01 00:00 (`P2SAoM40.PRT` shows both writes). So a
5-day (240-step) re-run from `fort.1.nc` with the existing binary can be
compared to `fort.2.nc`: this tests that we can reproduce Fortran results
here, and gives a 5-day reference trajectory endpoint for later F2 tests.
Scratch run dir: session scratchpad `run_repro/` (not committed; 187 MB
restarts). Result below.

## Next
- Confirm reproduction (bitwise or within noise floor) of `fort.2.nc`.
- Rebuild with dump hooks in a *copy* of the model tree (never edit the
  original in place), first around SURFACE/PBL/DRYCNV.
- Comparison harness + noise-floor measurement.

## 2026-09-24: reproducibility result (gate item 1 — PASSED)

Re-ran 1950-11-26 → 1950-12-01 (240 DTsrc steps, ~16 min wall, 1 thread) from
`fort.1.nc` with the original `P2SAoM40.bin` (`ifort` build, this node).
The resulting restart is **byte-identical (`cmp`) to the original run's
`fort.2.nc`** — all 257 variables bitwise equal (`compare_restarts.py`).
Sanity: the start and end restarts differ (26 identical of 257 between
11-26 and 12-01), and the output has a new inode/mtime, so this is a genuine
re-computation, not a copy. `cputime` is 0 in restarts, so no timing noise.

Consequences:
- The real model is **bitwise deterministic** here; Fortran-vs-Fortran noise
  floor is exactly 0 for identical inputs. Any nonzero port difference is a
  port difference. (Chaos floor for *perturbed* runs still needs measuring —
  next item.)
- A 5-day / 240-step, full-model, real-Fortran reference trajectory exists
  with known-good endpoints: the F2 test target.
- A ~16 min cycle for re-running Fortran (with instrumentation) is cheap.

## 2026-09-24: KEY FINDING — DRYCNV is not part of real P2SAoM40 physics

`nm P2SAoM40.bin` shows **no `drycnv` symbol**; the executable contains
`atm_diffus_`, `e_gcm_`, `find_pbl_top_` (ATURB.f). The rundeck's object list
has ATURB and no DRYCNV, and ATM_DRV.f's comment says the DRYCNV slot is a
dummy call when ATURB is used (`CALL ATM_DIFFUS (2,LM-1,dtsrc)`).
The PBL similarity functions (`socpbl_mp_find_dpsim_`) *are* used (surface layer).

Implication for HQ's skepticism: Track A's "DRYCNV: Faithful, 9.5x" kernel
validates a routine the real configuration never executes. The
free-atmosphere turbulence that must be ported for fidelity is **ATURB**
(1,405 lines; currently only partial), not DRYCNV. Priority of ATURB in
Phase 1 rises accordingly (it becomes the vertical-mixing core, not a
"wire it in" item). Track A's DRYCNV timing remains valid as a stand-alone
kernel benchmark but must not be cited as part of the P2SAoM40 workload.

## Instrumentation build (in progress)
A working copy of the model tree (236 MB, everything but ModelE_Support) builds
in a scratch dir with `gmake -C decks RUN=P2SAoM40 <copy>/model/P2SAoM40.bin`
(paths are relative to the tree root, `.modelErc` supplies NETCDFHOME).
Dump hooks will be added in that copy only. ATM_DRV.f `atm_phase1` and
MODELE.f main loop give the natural hook points (before/after CONDSE, RADIA,
SURFACE, ATM_DIFFUS).

## 2026-09-24: instrumented ModelE built and validated (gate item 2 — PASSED)
Dump hooks (`instrumentation/*.patch`, recipe in `instrumentation/build_and_run.md`)
built in a scratch copy in ~1 min. Instrumented vs original binary, 6 steps:
restart 256/257 bitwise identical (only wall-clock `cputime` differs) — the
hooks do not perturb results. Dumps read with `ffdump_reader.py`.
Findings from the dumps (step 33312): dynamics runs *before* CONDSE (pre_condse
state != restart state); SURFACE changes T by ≤0.28 K, Q ≤2.9e-3, U/V ≤4 m/s
across the column (that is real ATURB acting on all layers, not just layer 1);
the phase-2 ATM_DIFFUS slot changes nothing (dummy, as documented).

## 2026-09-24: chaos noise floor measured (gate item 3 — DONE)
Perturbed-T (±1 ulp) 5-day run vs unperturbed: see FULL_FIDELITY_DELTAS.md D3.
Recipe: copy `fort.1.nc`, add `nextafter` ±1ulp noise to `t` with netCDF4 in
r+ mode, run as for the reproducibility run. ~16 min.
**Phase 0 gate: PASSED** — oracle reproducible, hooks non-perturbing,
noise floor known, first Track A delta (D2) measured.

## 2026-09-24: Phase 1 item 1 DONE — ATURB fully ported (T, Q, TKE, PBL, U/V) at rounding level
See FULL_FIDELITY_DELTAS.md D4. Test data: ff_data/{nov26,dec01,jan01} (untracked, 1 GB;
regenerate with instrumentation/build_and_run.md). `ffa_geom.txt`/`ffa_consts.txt` are
written by the instrumented model on first ATURB call.

## 2026-09-24: Phase 1 item 2 DONE — PBL `advanc` ported (D5)
27.5k real PBL calls (4 surface types, 3 dates): worst max-abs/rms 5e-11; residual equals the model's own
float64 sensitivity. Test data ff_data/*/ffp_*.bin. Next: SURFACE tile-flux logic (ocean, sea-ice
explicit/implicit fluxes), then GHY (land), SEAICE/LAKES thermodynamics, land ice.


## 2026-09-27: Phase 1 item 4 DONE — GHY land-surface model ported (D9)
9,036 real land-tile records (6 steps x 3 dates), 804-854/1506 with active snow per step, zero exceptions
on a full-file run. Plain-Python reference (not yet JAX); largest residuals (aruns/aeruns, up to 7e-3 of
field scale) traced to a threshold-crossing sensitivity in the bare-soil runoff formula, same pattern as
ATURB/PBL branch flips -- not a logic bug. Core outputs (tbcs, tsns, ashg, alhg, ae0, aevap) all <=4e-6
relative. Bug caught: ws(0,2)/shc(0,2) canopy capacities must come from Ent's per-cell exports
(ws_can/shc_can), set once before the iteration loop -- missing this gave immediate NaN/Inf.
Ent vegetation itself (canopy conductance/LAI/photosynthesis) is NOT ported; its real per-substep
outputs are read from the dump and used as inputs. Test data: ff_data/*/ffg_*.bin + ffg_thm.txt.
Next: SEAICE/LAKES ground thermodynamics and tile aggregation (`avg_patches_*`), then JAX-vectorize
the ported pieces, then Phase 2 (radiation via SOCRATES from Python, per user decision 2026-09-25).


## 2026-09-27: Phase 1 item 5a DONE — sea-ice ground thermodynamics ported (D10)
4,524 real GROUND_SI cells (6 steps x 3 dates, sea-ice + lake-ice), zero exceptions. SEA_ICE+SSIDEC+
snowice all validated; relative errors 1e-15-3e-4 (coupling-flux diagnostics are the loosest, same
cancellation pattern as elsewhere). One dump-placement bug caught and fixed (read RUNOSI before Fortran
assigned it -- looked like total failure, was purely an instrumentation ordering mistake).
Remaining for Phase 1: ADDICE/SIMELT (ice formation/melt-out), lake mixing (documented no-op in Track A
too), then JAX-vectorize all Track B pieces so far (currently plain Python/NumPy reference code, correct
but not fast), then Phase 2 (radiation via SOCRATES from Python, per user decision 2026-09-25).


## 2026-09-27: Phase 1 item 5b DONE — tile aggregation ported (D11)
38,040 real grid cells (6 steps x 3 dates x 3312 cells), max error <=9e-7 of field RMS (float64 rounding).
Simple area-fraction-weighted sum over the 4 surface-type patches (FLUXES.f avg_patches_*), confirmed
by reading the source. Test data ff_data/*/fft_*.bin.

## 2026-09-27: Phase 1 item 5a completed — ADDICE + SIMELT ported (D12)
16,214 real ADDICE calls (1,994 = 12% genuine new-ice formation) and 4,668 real SIMELT calls (376 fully
melted out), zero exceptions. Bitwise or 1e-13-level match. Found by reading the source: SIMELT's TSIL
output is intent(out) but left UNASSIGNED by the real Fortran whenever ice remains -- the port returns
tsil=None in that branch rather than fabricating a value that would falsely appear "matched". This
closes out sea-ice ground thermodynamics (SEA_ICE+SSIDEC+snowice+ADDICE+SIMELT, D10+D12).
Then scoped Phase 2 (radiation): confirmed libsocrates.a is not -fPIC (blocks a straightforward
ctypes/.so approach); built a small statically-linked subprocess driver around one real SOCRATES kernel
(gauss_angle) as a proof-of-concept for the calling architecture. Not a SOCRATES rewrite -- calls the
real, unmodified library. User then redirected: "move to the smaller remaining items (ADDICE/SIMELT,
lake mixing, JAX-vectorization) instead" -- pausing further radiation work.

## 2026-09-27: Phase 1 item 5c DONE — lake mixing ported (D13)
3,644 real GROUND_LK calls (6 steps x 3 dates), 648 with genuine frazil-ice formation, zero exceptions,
**bitwise-exact match on the first attempt** for both LKSOURC and chained LKMIX. Track A's lakes_jax.py
lkmix was a documented no-op; this ports the real two-layer physics (static-stability mixing, implicit
heat diffusion, TKE entrainment). Coverage gap found by reading the source: GROUND_LK always calls
LKMIX with TKE=0. (the U2rho entrainment term is commented out in this rundeck), so LKMIX's TKE-driven
entrainment branch is transcribed but never exercised by real calls -- documented as unvalidated in
practice, not silently assumed correct. Test data ff_data/*/ffl2_*.bin.
This completes all items in "the smaller remaining items" except JAX-vectorization of the Track B
reference ports (ghy_ref.py, seaice_core_ff.py — still plain Python/NumPy), which is next.

## 2026-09-27: JAX-vectorization of the sea-ice hot path DONE (D14)
seaice_core_jax.py: batched-array (jnp.where, no Python loop over cells) port of sea_ice/ssidec/
snowice/simelt and their relayer/relayer_12/get_snow_ice_layer/set_snow_ice_layer/tice helpers.
relayer_12 alone required hand-deriving 9 mutually-exclusive leaf branches as closed forms in the
pre-branch inputs (its Python original is a 4-way nested if/elif/else). Validated on the same 4,524
GROUND_SI + 4,668 SIMELT real cells as D10/D12: matches Fortran at the same tolerances as the plain-
Python reference (worst case unchanged to 2 significant figures), matches seaice_core_ff row-for-row
to 7e-15 relative, jax.jit-compiles (4.6ms cached for 4,524 cells vs 2.9s one-time compile), zero
NaN/Inf. Scope decision, not an oversight: ADDICE is NOT vectorized here -- it chains 4 sequential
decision blocks with ~15 leaf branches total, several rebalancing corrections only lightly exercised
by the 3-date real record; rushing its jnp.where conversion would add more untested branch surface
than the lake-mixing/ADDICE-Python effort has real data to validate against. Full test suite (62
tests across fullfidelity/) still green after this change. Remaining JAX-vectorization work: ADDICE,
GHY (ghy_ref.py, a much larger stateful multi-layer column solver).

## 2026-09-27: measured real speedup, then vectorized ADDICE anyway (D14 cont'd)
Measured actual speedup (had only shown jit-compiles + correctness before): 46x at the real 4,524-
cell record (0.458s Python vs 0.0100s JAX cached), 51x at 20x scale (90,480 cells, to rule out a
small-batch artifact) -- per-cell JAX cost flat at ~2us/cell at both scales.

Then re-examined ADDICE (deferred above as "too branchy to vectorize safely") and found the earlier
assessment overstated the risk: ADDICE's Python original is a SEQUENCE of if-blocks (new-ice
formation, then an unconditionally-checked downward lead-fraction rebalance, then an unconditionally-
checked upward rebalance), not one wide decision tree like relayer_12 -- so it doesn't carry
relayer_12's combinatorial branch-count risk even though it's comparably long. Built it as a chain of
state=where(cond,f(state),state) merges reusing the already-vectorized relayer_12, validated on all
16,214 real ADDICE calls: bitwise/near-bitwise match to Fortran and to seaice_core_ff (same tolerances
as D12), zero NaN/Inf, jax.jit-compiles (13ms cached for 16,214 cells vs 3.6s compile). Coverage gap
found and handled honestly: 2 of ADDICE's 5 leaf paths (new ice in fully open ocean; the qfixr msi2-
floor correction) occur 0 times in the real 3-date record -- validated against seaice_core_ff on
synthetic inputs instead (bitwise/near-bitwise match), documented as such rather than silently
assumed correct. seaice_core_jax.py's module docstring updated to reflect ADDICE now being in scope.
Full test suite green. Remaining JAX-vectorization work: GHY only (ghy_ref.py).

## 2026-09-27: scoped (not implemented) JAX-vectorization of GHY
Read all 1,253 lines of ghy_ref.py and wrote a detailed scoping section into FULL_FIDELITY_PLAN.md
(after Phase 1, before Phase 2) rather than rushing a partial implementation -- matches how Phase 2
radiation was handled when it turned out to be similarly large. Confirmed every individual technique
needed has a precedent already used in this project (fixed-size-plus-mask for variable layer counts,
bounded-unroll while-loops, small tridiagonal solves, fixed-iteration bisection) but GHY needs several
of them AT ONCE (per-cell active-soil-layer count n, per-substep snow-layer count nsn in {0,1,3}, and
a per-cell-per-timestep substep count from Ent's adaptive stepping), which compounds risk relative to
the sea-ice work without a comparable real-data volume to catch a mistake (9,036 real land-tile
records vs tens of thousands for sea-ice).

Verified two specific risk points empirically rather than assuming, and both corrected the initial
guess: (1) hydra()'s bisection "exact table hit" branch, expected to be a rare edge case, actually
fires 19,605/272,124 times (7.2%) on real data -- not rare, must be a normal branch in any vectorized
version. (2) The obvious substep-count-padding scheme (pad every cell's Ent-iteration list to a common
max with dts=0 "no-op" rows) does not work: dts=0 crashes with ZeroDivisionError (snow_adv_1 divides
by self.dts), and padding with a tiny dts=1e-6 instead avoids the crash but is not a no-op either --
key accumulator scalars (tbcs, aruns, ...) move by up to 1.8 because several formulas divide a
not-proportionally-small quantity by dts. Correct fix: gate the whole substep body with a per-lane
mask (run vs keep-prior-state), not attempt to make dts degenerate. Documented as a corrected finding,
not a hedge, in the plan. GHY JAX-vectorization is scoped and NOT started -- next up if resumed:
the per-lane substep mask, then hydra/xklh, then the non-snow flux chain, then the snow model last.

## 2026-09-27: GHY JAX-vectorization DONE (D15) -- completed the scoping plan same-day
Followed the scoping order exactly: static setup + reth/retp/hydra/xklh first (cross-checked against
ghy_ref on 400 real cells; two real bugs found -- jnp.sum reduces in a different order than Python's
sum() for the small texture-table dot products, confirmed empirically (29% mismatch over 200k trials)
and fixed with an explicit left-to-right accumulation helper; and a genuine transcription bug in
hydra's bisection setup, thr1 wrongly initialized to the clamped theta instead of thets). Then the
non-snow flux chain (evap_limits through apply_fluxes): found evap_limits' "if evapvw<0: fw=1,fd=0"
mutation persists into every later substep method, and retp never computed tsn1 (silently breaking
sensible_heat for cold snow layers), and flhg was designed to return fh0 as a disconnected value
instead of merging it into flh's fh array (a ~123,000-unit error in one real cell's heat content made
this one obvious). Then the snow model (pass_water/snow_fraction/snow_redistr/tridiag/heat_eq/
snow_adv_1/snow_drv/snow) -- the scoping's predicted riskiest piece, validated on all 9,036 real
land-cell substeps with zero NaN-pattern mismatches once a genuinely NaN-aware test comparison was
used (naive max()-based comparison was silently swallowing NaN diffs, discovered while chasing what
turned out to be a real degenerate 0/0 case already present in the plain-Python reference itself, not
introduced by this port). snow_redistr's while-loop was reformulated as a closed-form overlap-matrix
remap rather than forced into lax.while_loop, verified equivalent (2e-9 over 20k random trials).

Assembling advnc() (the substep-masked driving loop) surfaced two more real bugs: evap_limits needs
the TOTAL dt (constant across the whole advnc() call), not the per-substep dts -- invisible with a
single substep (they're equal), a clear error once 2+ substeps run; and GhyColumn.snow()'s per-ibv
gating (skipped entirely for an inactive ibv) was missing from my first version.

Final result: all of GHY (ghy_ref.py's advnc() and everything it calls) is now JAX-vectorized,
matching real Fortran across all 9,036 real land-cell substeps at the SAME tolerance the plain-Python
reference achieves against Fortran (D9), including sharing its one known chaos-sensitive field
(aruns/aeruns runoff-activation threshold flips). The scoping's "dts=0 padding is unsafe" finding
turned out to be moot in practice: the per-lane whole-substep-select mask design discards a padding
lane's entire candidate state regardless of whether it's finite or NaN/inf, so the dts value used for
padding never matters. This closes out JAX-vectorization for the whole of Track B's currently-
validated physics (ATURB, PBL, SURFACE, SEAICE/ADDICE/SIMELT, LAKES, tile aggregation, and now GHY).

## 2026-09-28: measured GHY speed, found and fixed a real regression (D15 cont'd)
Wrote a pytest suite (tests/test_ghy_jax.py) and ran it -- all tests passed, but the run took 3h42m
(vs the same logic run via a direct script finishing in under a minute for the same cell counts).
Diagnosed with a quick repeated-call timing check outside pytest: no progressive slowdown across 6
consecutive advnc() calls in one process (~8s each), so this is specific to pytest's invocation
somehow, not a real property of the code -- left as an open, honestly-documented oddity rather than
spending more time chasing it, since correctness is proven either way (direct-script re-verification
matches the pytest run's PASSED results).

**RESOLVED, later the same day** (see the 2026-09-28 x64 bug-fix entry below, D17): once
`jax.config.update("jax_enable_x64", True)` was moved from the test file into `ghy_jax.py` itself
(fixing the unrelated silent-float32-fallback bug found while building `ground_si`), a full rerun of
this exact same `tests/test_ghy_jax.py` suite (all 21 tests, unchanged) took **6m34s** -- a ~36x drop
from the previously-observed 3h42m-4hr, with no other change to the test file or its logic. This is a
clean natural A/B (same suite, same machine, only the x64-enable's location changed) and strongly
suggests the pytest-specific slowdown *was* this bug all along: `jax_enable_x64` is documented
upstream as needing to be set before any array/JIT activity, and pytest's test collection (which
imports every file under `tests/` up front, not just the one being run) was likely triggering some
JAX array or trace activity via a sibling test module before `test_ghy_jax.py`'s own late
`config.update` call took effect -- plausibly explaining why a fresh, isolated direct-script call
(which sets x64 first, before touching `jax`, and imports nothing else) never reproduced it. Not
chasing the exact JAX-internal mechanism further since the fix, the isolated repro, and the 36x
speedup after fixing it are all consistent and the practical outcome (correct dtype, fast pytest) is
what matters -- but the earlier "left as an open, unexplained oddity" framing above is superseded by
this.

Then measured actual speed (the point of "vectorizing" in the first place, not yet checked): plain-
Python 0.63s for 300 cells vs JAX eager 8.7s -- the JAX version was SLOWER, a regression, and
jax.jit-compiling it didn't finish in 300s (XLA's own slow-compile warning fired). Root cause: the
substep loop was a Python `for i in range(11): ...` unroll, which duplicates the entire per-substep
computation graph (hydra/xklh/evap_limits/.../snow's own nested heat_eq calls -- dozens of functions)
11 times. Every OTHER bounded loop in this project (hydra's 6-step bisection, tridiag, relayer_12,
snow_adv_1's mass-densification loop) stays a small Python unroll because the per-iteration body is
small (a handful of arithmetic ops); GHY's substep body is not small, so unrolling it 11x was
qualitatively different and the actual problem.

Fixed by rewriting the substep loop with jax.lax.scan (compiles the substep body once, applies it via
an XLA-level loop) -- same per-lane masking design (`i < n_substeps`), same every function call, only
the control-flow primitive changed. Re-validated: bit-for-bit identical results to the already real-
Fortran-checked unrolled version on all 9,036 real cells across all 6 files (every D15 error number
unchanged). Speed: jit compile ~30s (one-time), cached run 112us/cell vs plain-Python's 2.2ms/cell --
**~20x speedup**, confirmed at both 300-cell and full-file (1,506-cell) scale. This is now the
template lesson for this codebase: a big per-timestep substep/step loop needs lax.scan, not a Python
unroll, even though small bounded loops (2-6 iterations) are fine unrolled.

Also wrote the 8-stage Real-Fortran/Track-A/Track-B comparison summary into STATUS.md, per the user's
explicit request for that structure. Writing it out honestly surfaced that lake mixing (lakes_ff.py)
was the last piece of Track B's currently-validated physics still plain Python -- closed that gap
immediately: lakes_core_jax.py (LKSOURC/LKMIX, no per-cell loop so no lax.scan needed, straightforward
jnp.where transcription) matches the plain-Python reference BITWISE on all 3,644 real lake cells, plus
a synthetic check of lkmix's dead-code tke>0 branch (also bitwise match, D16). ~104x CPU speedup. This
now means every currently-validated Track B module (ATURB, PBL, SURFACE, SEAICE/ADDICE/SIMELT, LAKES,
tile aggregation, GHY) is jax.jit-compilable with a measured real-data speedup -- the honest remaining
gaps for the 8-stage comparison are (1) no single chained whole-model Track B step yet (each module is
fast on its own but not wired together the way Track A's run_steps_device chains one atmosphere step),
and (2) no GPU access on any node used this session (a hardware blocker, not a scoping choice).

## 2026-09-28: resumed Phase 2 radiation scoping (all smaller items now closed out)
With ADDICE/SIMELT, lake mixing, and JAX-vectorization (sea-ice, GHY, lakes) all done, picked radiation
back up -- the item explicitly paused earlier ("move to the smaller remaining items instead"). Read
RCOMPX's actual body (RADIATION.f:1696-1942, not read in the earlier 2026-09-27 scoping pass) and the
rundeck's RADPAR overrides, which narrowed the earlier vague "~1,500 lines, ~100 scalars" estimate to
a concrete traced call graph: with GISS_RAD_OFF defined, TAUGAS/GETCLD/THERML/SOLARM (the classic-GISS
blocks) never run at all inside RCOMPX itself, not just "unused" -- SOCRATES is the only real code
path. MADAER=0 and MADDST=0 are RADPAR's defaults and are never overridden in the P2SAoM40 rundeck, so
getaer/getdst (aerosol/dust optical properties) are NEVER CALLED -- only MADVOL=2 (volcanic) is active.
GETEPS turned out to be a trivial static per-column climatology lookup (KCLDEP=4 default), not a live
computation. The real remaining per-column work is now known precisely: seth2o/getgas/fpxscalegas (gas
amounts), get_volc_column+getvol (volcanic aerosol only), GETSUR (surface albedo, genuinely
substantial, ~25 named inputs), GETEPS (now known trivial), then set_planet_alb_param+run_planet_rad
(the real SOCRATES call) and get_planet_radout. Not yet traced: run_planet_rad's own full input
surface (planet_rad.F90), or getgas/getvol/GETSUR's bodies in detail. Documented in
FULL_FIDELITY_PLAN.md's Phase 2 section rather than rushing into the dump-hook/shim implementation --
this narrows the estimate's uncertainty without changing its size (still a multi-week undertaking:
GETSUR and run_planet_rad are real, and SOCRATES's own 90k-line library is unaffected by any of this).

## 2026-09-28: chained whole-model Track B driver -- scoping, ground_si, and a real x64 bug
User asked directly whether Track A and Track B are both "end-to-end" the way Scott's P2SAoM40
workflow suggests -- answer: not yet on the chaining axis (Track A fuses one real per-column workflow
into a single lax.scan; Track B has validated fast modules but no driver wiring them together).
Traced the real per-cell orchestration (SURFACE.f's SURFACE subroutine, GHY_DRV.f's earth) to scope
the integration properly: almost everything needed already exists and is validated (D4-D16) -- this
is an integration task, not new porting. The one clearly-missing piece was a `ground_si` wrapper
chaining sea_ice->ssidec->snowice with ocean/lake domain gating; the exact logic already existed as a
test-only helper (seaice_jax_compare.batched_ground_si, unpacks a raw dump row) so promoted it into a
proper named-argument function in seaice_core_jax.py (D17).

Building a standalone script to validate the new function (deliberately NOT reusing the test suite's
own x64-enabling import, to check the function works for an arbitrary caller) immediately surfaced a
real bug: seaice_core_jax.py, ghy_jax.py, and lakes_core_jax.py never call
jax.config.update("jax_enable_x64", True) themselves -- every existing test/compare script happened to
enable x64 before importing them, so D9/D10/D12-D16 never caught this. Running seaice_core_jax's
ground_si in JAX's float32 default got one real cell (row 1944, an ocean cell near the ice melt point)
badly wrong: ssidec divides by tsil, which is ~0 there, and float32's less-precise near-zero tsil had
a different sign than float64's, propagating through a brine-fraction division into erunosi being off
by ~5,415 units -- not noise, a flipped-sign wrong answer, on a dump file that all 4,524 cells across
were otherwise passing. Fixed by moving the x64 enable inside all three core modules themselves
(before jax.numpy is imported), so correctness no longer depends on caller import order. Re-validated
at <1e-9 rel. error post-fix across all 4,524 real ocean+lake cells; reran lakes (5/5) and sea-ice
(12/12, incl. a new test cross-checking the promoted ground_si against the test helper) -- all still
pass, confirming the fix changes nothing for code paths that were already accidentally correct.
Documented in FULL_FIDELITY_DELTAS.md (D17) and FULL_FIDELITY_PLAN.md's chained-driver section. Per
the "verify dramatic results" habit, didn't stop at "the number moved" -- traced the actual mechanism
(a near-singularity division sensitive to float precision, the same class of issue as GHY's
aruns/aeruns threshold-crossing sensitivity, D9) before calling it understood.

Remaining before the actual jax.lax.scan-chained driver can be assembled: precisely locating land-ice's
tile-flux call site and ITYPE plumbing, and tracing the real per-cell data flow between PBL and each
SURFACE tile call (what PBL_ARGS/tile-fraction arrays route between them).

## 2026-09-28 (continued): land-ice call site traced -- corrects the earlier step order
Found the permanent (non-scratch, not session-scoped) model source tree still exists at
`/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0` -- no need to rebuild a scratch copy
just to read source for tracing (rebuilding is only needed to regenerate dump files). Grepped
SURFACE.f directly for the actual call sites instead of re-deriving from memory, and it **corrected**
the first scoping pass's guessed order: land ice (`CALL SURFACE_LANDICE`, SURFACE.f:883/893) runs
BEFORE `CALL EARTH` (SURFACE.f:902), not after as guessed on 2026-09-28 (originally). Also found land
ice is not a 4th `ITYPE` in the ocean/ice loop at all -- it's a separate loop over `atmglas(ipatch)`
glacial-ice patches, a different indexing scheme entirely (patches, not (I,J) grid tiles); `#ifdef
GLINT2` selects a height-point variant but GLINT2 is not defined for P2SAoM40's rundeck (checked
`decks/P2SAoM40.R`, no match), confirming the plain patch-loop branch is the real one. Located the
tile-aggregation call precisely too: `avg_patches_pbl_exports`/`avg_patches_srfflx_exports`/
`avg_patches_srfstate_exports` at SURFACE.f:1055-1057, confirmed to run once after all four tile types
(ocean, ice, land-ice, land), not per-tile. And confirmed all 7 steps (ocean/ice tiles through
GROUND_LK) live inside ONE Fortran subroutine (`SURFACE.f:21 SUBROUTINE SURFACE`), itself called once
per DTsrc step from `MODELE.f:332`, with its own internal `DO NS=1,NIsurf` loop (SURFACE.f:385)
wrapping all 7 steps -- so the real per-cell chain to reproduce is one DO-NS iteration's body, and
that NIsurf sub-stepping already matches Track A's own NIsurf loop in `p2saom40_driver.py`'s
`_step_dev`, so the two tracks' granularity lines up without extra reconciliation work. Updated
FULL_FIDELITY_PLAN.md's chained-driver section with the corrected order and exact line numbers.
Remaining before assembly: the `ipatch`<->`(I,J)` mapping for land ice, and the PBL_ARGS/tile-fraction
data flow between PBL and each SURFACE tile call.

## 2026-09-28 (late): D18 composition link built; shared checkout switched to main
Found the shared checkout on `main` (reflog: `checkout: moving from full-fidelity-port to main`, not done by me);
all my commits were intact and pushed, so continued in a separate git worktree (`dev/ff_worktree`) rather than
switch the checkout back under whoever moved it. Traced the SURFACE.f code between tile aggregation and ATURB
("UPDATE FIRST LAYER QUANTITIES"): a pure algebraic map (`tflux1=-dth1*MA1/dtsurf`, `qflux1=-dq1*MA1/dtsurf`),
verified with 0.0 error against the recorded ATURB entry arrays. Built `chain_aggregate_aturb.py`: our PBL+ocean/ice
tile chain -> our aggregation (recorded land-ice/land) -> our ATURB vs the real exit state; roundoff-level agreement on
3 dates x 2 substeps (D18). Not yet: land-ice tile from our own code in the composite, land/Ent, the lax.scan driver.

Correction to the 2026-09-28 tracing entry above: `END DO ! end of surface time step` is at SURFACE.f:1178, right after
ATM_DIFFUS, so the NS loop wraps only ocean/ice tiles, land ice, EARTH, aggregation and ATURB. GROUND_SI/GROUND_LK
(and RIVERF, FORM_SI) run once per DTsrc step after the loop. Found while checking what a substep-to-substep chain needs.

## 2026-09-28 (night): D19 -- two substeps chained
Checked what substep 2's PBL/tile inputs depend on: PBL profiles and cm/ch/cq carry over bitwise; layer-1 scalars and
`get_dbl` inputs come from the ATURB exit state; ice/land-ice ground state carries over from the tile outputs; water-tile
`z0m` is provably irrelevant (output unchanged when perturbed). Built `substep_chain.py` (layer-1 exports, get_dbl, THBAR)
and `chain_two_substeps.py`; per-column diff caught a Coriolis bug of mine (land-ice dbl 66 m off) that a plain
end-result check would have blamed on physics. Final substep-2 exit state matches the real one at roundoff on 3 dates.

## 2026-09-28 (night): D20 -- GROUND_SI / GROUND_LK after the loop
Accumulation identities verified with 0.0 error (ice-tile f0dt/f1dt/evap summed over substeps = GROUND_SI inputs; srox0 =
sum srheat*dtsurf; lake fodt/evapo/srox from open-water tile accumulators). Chained both stages on our two-substep results;
outputs match the recorded-input baseline (lakes 2.5e-11 relative). One test-harness detail: fully ice-covered lake cells have
no open-water tile (accumulators default to 0).

## 2026-09-28 (night): D21 -- lake ADDICE chained
ADDICE lake inputs are exactly LKSOURC outputs + GROUND_SI final state (0.0 error). Chained on our results: outputs <= 8.6e-14
relative. The lake-side surface chain is now complete on our own numbers up to FORM_SI.

## 2026-09-28 (night): D22 -- land in the chain
Added land (PBL itype 4 -> ghy_jax) to the two-substep chain. Identities first: GHY forcing == PBL outputs bit-exact; land patch
formulas exact given GHY outputs. Needed: (1) advnc's post-loop evap_limits(.false.) outputs (evap_max_ij, fr_sat_ij) -- added
to ghy_jax, matches at roundoff; (2) the REAL*4 `0.001` literal in the qg blend (found via constant ratio 1.0000000475);
(3) TRUP_in_rad for land, not dumped -> reconstructed from the recorded patch (constant across substeps). Bulk agreement excellent;
the max is one runoff-threshold cell (D9 sensitivity). First test run had four failing tests -- all my own bounds set tighter than
GHY's measured accuracy (dq1 99th pct 3e-7 relative etc.); reset from measurements, not to force a pass on a wrong result.
