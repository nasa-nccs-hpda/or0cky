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

## 2026-09-28 (night): D23 -- timing the chain, and what it says
First warm timing of a full substep: GHY stage was 32 s and ATURB+UV 2.2 s until both were wrapped in jax.jit (un-jitted scans
re-trace every call); after that 0.45 s per substep, identical on one core. The Fortran's own PRT timer table gives ~0.28 s per step
for the same scope, so Track B on CPU is ~3x slower than Fortran (not faster). Recorded plainly in D23/STATUS; the plain-Python
speedups quoted earlier must not be read as Fortran speedups.

## 2026-09-28 (night): D24 -- ocean ADDICE, SIMELT scoping, glue profile
Ocean-cell ADDICE chained on our GROUND_SI state (error = upstream residual). SIMELT's input state matches neither GROUND_SI input nor
ADDICE output (it runs after sea-ice dynamics/ocean), so it is out of the surface chain. Lake runoff addition cannot be chained
without step-start lake state (not dumped). cProfile of a warm substep: glue is ~1/3 of the time, compute floor ~0.3 s -> still ~2x
Fortran; not the lever. Test-writing note: two of my first bounds here were wrong references (exact baseline vs upstream residual;
absolute vs relative) -- fixed by reasoning about what the stage can and cannot correct, not by loosening until green.

## 2026-09-28 (night): D25 -- land carried over two steps
Checked what carries exactly across a step boundary: GHY state and PBL land profiles do (0.0); ice state does not (5% match --
precip/dynamics/ocean act in between). Built run_land_multistep (4 substeps). Errors: state drifts slowly (2e-5 relative), flux errors
amplify ~500x from recorded-state level then saturate at ~1e-5 K RMS. Also reported the fresh-start baseline so the amplification is
attributed to carry-over, not to step-2 data.

## 2026-09-28 (night): DYNSI + prognostic ocean scoped -- committed to full port
User decision after D25: port DYNSI and the prognostic ocean too, not a reduced/mixed-layer approximation.
Traced the real per-step order (PRECIP_SI/PRECIP_OC before SURFACE; DYNSI/UNDERICE/ocean-grid GROUND_SI/CALC_APRESS/
OCEANS/FORM_SI/ADVSI in ocean_driver, right AFTER SURFACE in the same step, not "between steps" as first assumed).
New finding: GROUND_SI/FORM_SI(ADDICE) run TWICE per step on two grids (atm-grid si_atm inside SURFACE, already
chained D17-D24; ocean-grid si_ocn inside ocean_driver, not yet). Confirmed real, not cubed-sphere (Atm72x46);
confirmed dead code: OTIDE=0 (tides never run), TRACERS_OCEAN and TRACERS_OceanBiology not defined (no ocean
tracers, no obio) -- real prognostic ocean state here is just mass/heat/salt, not the full N-tracer machinery.
Real subroutine-level line counts: ice-dynamics side (PRECIP_SI+PREC_SI+DYNSI+UNDERICE+CALC_APRESS+
seaice_to_atmgrid+FORM_SI) ~1,460 lines, tractable, comparable to work already done; ocean numerical core
(OCNDYN.f+OCNQUS+OCNKPP+OCNMESO_DRV/OCNTDMIX/OCNGM+OSTRAITS) ~19,000 lines at the file level (not yet read in
detail or dead-code-excluded) -- bigger than the entire project to date. Written up in FULL_FIDELITY_PLAN.md
Phase 5. PREC_SI already checked to need only one new helper (Fi, ~20 lines) beyond what D10/D14 already ported.
Plan: Stage 1 (ice dynamics, closes the actual D25 gap) first, Stage 2 (ocean numerics) scoped in detail only
after Stage 1, mirroring how radiation's GETSUR/RCOMPX were read before estimating further.

## 2026-09-28 (night): D26 -- PRECIP_SI ported, first Stage 1 deliverable
Built new Fortran instrumentation for PRECIP_SI, rebuilt in a fresh scratch copy, reran all 3 real dates (found
and fixed a stale doc bug along the way: `-l run.PRT` isn't a real CLI flag, corrected to `-i I`). Plain-Python
and JAX ports both match real Fortran at float64 rounding level on all 13,572 real cells, first try after fixing
one NameError (forgot to unpack mice2/mice3). Only one genuinely new helper needed (Fi, ~20 lines); everything
else reused already-ported SEAICE.f functions. Reran the full sea-ice (33) and GHY test suites to confirm no
regression. New dump files (ffw_*.bin) copied into the shared ff_data/ directory alongside the existing dumps.

## 2026-09-28 (night): D27 -- PRECIP_LK ported; caught a real operational mistake
Added PRECIP_LK instrumentation, rebuilt, reran all 3 dates -- second run silently did ZERO steps because the
first run's own checkpointing overwrote both fort.1.nc and fort.2.nc in the run directory with the post-run state
(GISS ModelE double-buffers restart writes across both files). Caught this from the timer table showing 0 trips
for every routine, not by trusting "Terminated normally". Fixed by re-copying fresh restart files from the
untouched source before every rerun -- worth remembering for any future multi-run cycle in a shared scratch
directory. Confirmed the rebuild didn't perturb PRECIP_SI's own D26 dumps (byte-identical). precip_lk itself was
pure bookkeeping (bitwise-exact both ports, no new physics helpers) -- the real cost of this increment was the
run-directory mistake, not the port.

## 2026-09-28 (night): D28 -- PRECIP_LI ported; third Stage 1 item in one rebuild cycle
Added PRECIP_LI instrumentation on top of the D27 build (batched into the same scratch tree rather than a fresh
copy), applied the fort.1.nc/fort.2.nc restart-refresh fix immediately this time. Both ports match real Fortran
bitwise-exact on all 3,723 real land-ice tiles, first try. Real record is entirely cold precip -- rain branch
checked via synthetic inputs (had to fix my own synthetic-input construction once: the first attempt at forcing
the melt-through-to-layer-2 branch used too small an ENRGP and silently exercised 0/200 cells instead of >100,
caught by the test's own non-vacuous assertion rather than a passing-but-empty check). Read IRRIG_LK/
irrigate_extract (373 combined lines) and scoped but deferred it: it depends on an external prescribed irrigation
dataset, adding a "record the external forcing, port the arithmetic" pattern not yet needed elsewhere in Stage 1.

## 2026-09-28 (late night): D29 -- DYNSI/VPICEDYN/FORM/PLAST/RELAX, closed exact after 3 real bugs
Traced the ice-dyn B-grid geometry setup (`GEOMICDYN`/`ICDYN_MASKS`) fully: purely analytic, no dependence on any
prognostic field, `RADIUS` a runtime planet parameter (`USE_PLANET_RAD`) inferred exactly from the dump rather
than assumed -- validated bitwise exact against a new `ffz_geom.bin` dump. Read `DYNSI`'s own body, `FORM`,
`PLAST`, and `VPICEDYN` completely (no remaining unknowns): `VPICEDYN`'s outer KKI loop runs FORM+RELAX twice per
iteration (predictor, then modified-Euler-averaged corrector), converges via an RMS-velocity check against the
previous iterate, capped at 20 iterations.

Lost real time (would have been much longer without the user telling me directly to stop stalling and keep
going) to a debugging dead-end: new dump files never appeared no matter what was changed in `DYNSI`'s body,
including an unconditional `STOP` as its literal first statement -- which still didn't fire, even though the
compiled object and linked binary both provably contained the new code (checked via `nm`/`strings`/md5sum).
Root cause, found by comparing `ls -la` on the two files: the run script executes a file named `P2SAoM40`,
which is NOT a symlink to `P2SAoM40.bin` -- every rebuild-and-copy cycle in this session had been updating only
`P2SAoM40.bin`, leaving the actually-executed `P2SAoM40` untouched since a much earlier build. The stable "6"
trip count shown for `DYNSI()` in every rerun's timer table (which looked like evidence DYNSI was running) was
itself stale: GISS ModelE checkpoints its cumulative timer table into the restart file, so the count just
carried forward from whatever original run produced that restart, independent of whether the current run calls
the routine at all. Fixed `build_and_run.md` with both lessons. Once fixed, all 3 dates produced real `ffy_*`/
`ffz_*` dumps (6 steps each, ~380KB/214KB per in/out pair).

Ported `FORM`/`PLAST`/`RELAX`/`VPICEDYN` to plain Python (`icedyn_dynsi_ff.py`), using a 1-padded array
convention (Fortran index I,J maps directly to python index [I,J]) specifically because `RELAX` is ~150 lines of
extremely dense tridiagonal-coefficient algebra where off-by-one translation risk is high. `TRIDIAG_cyclic`/
`TRIDIAG_new` transcribed directly from `TRIDIAG_MOD` (Sherman-Morrison-augmented Thomas algorithm for the
cyclic I-direction solves; plain Thomas for the J-direction solves). First validation attempt against
`ffy_33312_{in,out}.bin` ran to completion (2 KKI iterations) but was NOT exact -- ~1% mean / ~7% max relative
error, pervasive across ~1800/2150 real cells. Rather than accept "close enough," bisected it with three
more rounds of debug-only Fortran dumps (`ffdump_form1`/`ffdump_relax1`/`ffdump_relax_coefs`, discarded after
use, not part of the permanent instrumentation): (1) `FORM`'s own output (`ETA`/`ZETA`/`PRESS`/`DWATN`) was
already exact but `FORCEX`/`FORCEY` wasn't -- traced to assuming `osurf_tilt=0` when `SEAICE.f` actually
defaults it to 1 for this rundeck; (2) with `FORM` fixed, `RELAX`'s own `AU`/`BU`/`CU`/`URT` coefficient dump
showed `AU`/`CU` exact but `BU`/`URT` off by orders of magnitude -- traced to passing `bydts=900` (the raw
timestep) instead of `1/900` (GISS's actual `BYDTS` convention) into the comparison harness, not a bug in the
ported module itself; (3) with that fixed, `UICE` was now exact but `VICE` was badly wrong -- a genuine
transcription slip in `RELAX`'s own source: the `VICE` J-direction `VRT` term should use `(AA3+AA4)`
(BYCSU-weighted ETA-only sums) but was written using `(AA1+AA2)` (the ETA+ZETA sums used in that block's own
AV/BV/CV a few lines above -- easy to conflate, same names in scope for two different quantities). Fixed all
three; final remaining mismatch was `DMU`/`DMV` consistently half their reference value, traced to this
rundeck's `DTsrc=1800s` (not the 900s used everywhere else in this project -- `DYNSI` runs once per full
`DTsrc`, not per `NIsurf` substep), confirmed in `decks/P2SAoM40.R:237`. Also had `vpicedyn()` return the
`DWATN` left by its OWN last internal `FORM` call (Euler-averaged velocity, one iteration stale) rather than
letting a caller recompute a fresh, plausible-but-wrong one from the final converged velocity.

Result: bitwise/float64-exact (~1e-8 to 4e-8 max relative error, ordinary float64 noise through a real
nonlinear iterative solve) on all 18 real records (6 steps x 3 dates). `TRIDIAG_cyclic`/`TRIDIAG_new` were
verified independently against dense-matrix residuals early on and were never the problem -- all three bugs
were in coefficient assembly or unit handling, which is exactly why bisecting via intermediate dumps (not
auditing the solver) was the right approach. `tests/test_dynsi_ff.py` (24 tests) added. Ran the full project
regression suite (196 tests, ~19 min) after these fixes: all green, nothing else broke. JAX/batched port of
DYNSI is the clear next step, now that a confirmed-correct plain-Python reference exists to validate it
against -- deferred for now in favor of finishing the smaller remaining Stage 1 items first.

## 2026-09-29: D30 -- CALC_APRESS, first-try exact
Before adding new instrumentation, reverted D29's throwaway debug dumps out of the scratch tree
(`ffdump_form1`/`ffdump_relax1`/`ffdump_relax_stage1`/`ffdump_relax_coefs`, `ICEDYN.f`'s debug edits) and
confirmed the result matched a reconstructed clean post-D29 baseline byte-for-byte before diffing, so the new
patch (`SEAICE_DRV_apress.f.patch`+`ATM_DRV_apress.f.patch`, unit 977) contains only the `CALC_APRESS` addition.
`CALC_APRESS` itself is trivial (one branch-free formula plus a pole-replication fixup) -- ported and validated
bitwise/float64-exact on the first try, all 18 real records, 3,170 cells/record. `GRAV` is another
`USE_PLANET_RAD` runtime parameter (like D29's `RADIUS`); inferred algebraically from a real ice-covered dump
cell rather than assumed, confirmed to equal Earth's standard 9.80665 m/s^2 for this rundeck. 39 tests added.

## 2026-09-29: D31 -- seaice_to_atmgrid, reused Ti/Ti2b from earlier deltas
Ported `seaice_to_atmgrid`'s state-derivation loop (`GTEMP`/`GTEMP2`/`GTEMPR`/`ZSNOWI`/`ZSI`/`FWSIM`), the
piece that reconciles ocean-grid sea-ice state back onto the atm-grid copy after `GROUND_SI`/`FORM_SI` --
directly closes part of the D24 "runs twice" gap. New instrumentation dumps every call from every call site
(`ATM_DRV.f`/`OCN_DRV.f`/`SURFACE.f` all call the same pure function each step) into one pool rather than
distinguishing sites, since there's nothing site-specific to the function itself. Reused `Ti`/`Ti2b` straight
from `seaice_core_ff.py`/`seaice_core_jax.py` (already ported for `GROUND_SI`) -- no new thermodynamics to
derive. Deliberately did NOT port the third loop's `RESET_SURF_FLUXES` call: read it first, confirmed it only
touches `RAD_COM`'s `FSF`/`TRSURF` diagnostic accumulators for the next radiation call, not sea-ice state --
squarely radiation-adjacent, matching the user's standing "SOCRATES/radiation is third-party, don't touch it"
constraint. Hit the same near-zero-temperature relative-error artifact as D26's `TSIL` (documented `Ti`/`Ti2b`
REAL*16-vs-REAL*8 approximation) on a few jan01 records; switched to absolute tolerance for the three
temperature fields, matching the established pattern rather than treating it as a new bug. 38 tests added.

## 2026-09-29: D32 -- UNDERICE/iceocean_fluxes/icelake_fluxes, one real bug found via intermediate dumps
Read `iceocean_fluxes` (5-iteration Newton solve for ice-ocean interface T/S, ~200 lines of dense branch
logic) and `icelake_fluxes` (closed-form, no salinity) fully. Confirmed `KOCEAN=1` for this rundeck so the
ocean domain's fixed-SST fallback branch is dead code. Reused `tfrez`/`alami`/`dEidTi` straight from earlier
`GROUND_SI` work -- no new thermodynamics primitives needed. New instrumentation dumps ocean-domain and
lake-domain calls separately (units 979/980) since `UNDERICE` is called from two different sites
(`OCN_DRV.f`/`SURFACE.f`) with genuinely different physics per domain, not just different data.

First validation attempt: `mflux`/`sflux` exact immediately, `hflux` badly wrong on some cells (up to ~180%
relative error). Found the bug fast this time (one intermediate-dump-and-compare cycle, not D29's three):
the real Fortran's post-loop `hflux` formula uses `Tb`/`lh` exactly as they were last set *inside* the 5-
iteration loop (from the iteration's pre-update `Sb0`), not a fresh recompute from the loop's final converged
`Sb0` -- the port had (wrongly) added a fresh `tfrez(Sb0)` call after the loop, one iteration ahead of what
the source actually does. Removed it; Python's loop-scoped variables naturally carry the correct last-set
values once the erroneous recompute was gone. Bitwise exact on all 18 records after the fix, first-recheck.

Checked whether the real 18-record window ever exercises UNDERICE's shallow-lake (<0.4 m) flux-limiting
branch before writing a test that assumes it does -- it doesn't, in any of the 18 records -- so that branch
is validated against the plain-Python reference on synthetic shallow-lake inputs instead, matching D28's
established honest-scoping pattern rather than silently dropping the check. 76 tests added. This closes the
ice-dynamics side of Stage 1 except the explicitly-deferred `IRRIG_LK` and the ocean-grid `GROUND_SI`/
`FORM_SI` chaining (plumbing, not new physics).

## 2026-09-29: D33 -- first Stage 2 delta, PRECIP_OC, plus OCNDYN.f's real scope
Started Stage 2 (the ocean numerical core) by reading `OCNDYN.f` in full rather than porting anything first --
same "read before estimating" discipline as D29's `DYNSI` correction. Found `OCNDYN.f`'s real 6,062 lines
split into ~3,439 lines of genuine per-step physics, ~587 lines of diagnostics (`CHECKO`/`CONSERV_O*`,
matching the existing `CHECKT` pattern), and ~1,744 lines of one-time init/restart-I/O -- a large, concrete
de-risking of the Stage 2 estimate. Also found (`ORES_5x4.F90`) that the ocean grid is the SAME resolution as
the atmosphere (IMO=72,JMO=46) for this rundeck -- meaning the atm<->ocean regrid layer (a general HNTR8
utility, not physics-specific) can be treated as a recorded-input boundary exactly like D29's DYNSI regrids,
rather than something that needs porting before any ocean-core routine can be validated.

Picked `PRECIP_OC` as the first real Stage 2 port: small (84 lines), in a familiar "PRECIP_*" family (already
ported SI/LK/LI versions for ice/lake/land-ice), and touches the ocean's own MO/G0M/S0M prognostic state for
the first time in this project. New instrumentation dumps before/after MO/G0M/S0M plus the real recorded
oPREC/oRSI/oRUNPSI/oEPREC/oERUNPSI/oSRUNPSI inputs. Bitwise exact on all 18 real records, first try -- no
branches, confirms the established dump-hook-and-validate methodology scales cleanly past Stage 1 into the
ocean core itself. 38 tests added.

## 2026-09-29: D34 -- OSOURC, FSR/FSRZ/LSRPD derived analytically instead of dumped
Read `OSOURC` (called from `GROUND_OC`, 123 lines) fully: applies runoff/evap/solar fluxes to a cell's
open-ocean and ice-covered fractions separately, checks each for frazil-ice formation, recombines, and
spreads insolation down the water column. Traced its solar-penetration-profile dependency (`FSR`/`FSRZ`/
`LSRPD`) back to `OCEAN_COM.f`'s `init_solar` and found it's fully analytic -- hardcoded `RFRAC`/`ZETA1`/
`ZETA2`/`ZMAX_SOLAR` PARAMETERs plus `OLAYERS.F90`'s fixed `L13` layer-thickness array (this rundeck's build
flag) -- so it was re-derived in Python rather than dumped, and its correctness is proven implicitly by every
downstream real-record test matching exactly. `GFREZS` is a hardcoded 41-point table visible directly in
`OCNFUNTAB.f`; `TFREZS` is closed-form and matches the project's existing `tfrez` exactly. Neither needed the
one genuine external-data dependency in this area (`GROUND_OC`'s own `SHCGS`, which reads the `OFTAB` binary
file at init -- deferred to when `GROUND_OC` itself is ported).

New instrumentation dumps `OSOURC`'s full argument list at its one real call site. Plain-Python port bitwise
exact on all 18 records, first try. The JAX port caught a real off-by-one before real-data testing even ran:
reasoning through the Fortran's `DO L=2,LSR-1` loop semantics directly showed the batched per-lane guard
needed strict `lsr > l`, not `lsr >= l` (Fortran's loop is empty when the upper bound is below the lower
bound, i.e. when `LSR<=2`). Confirmed genuinely necessary, not just theoretical, once checked against real
data: `LMIJ` (hence `LSR`) ranges 2-13 across real ocean columns in this window. 40 tests added.

## 2026-09-29: D35 -- GROUND_OC's below-freezing sweep, a caught design gap before wasting a cycle
Read GROUND_OC's tail (the L=2..LMM(I,J) below-freezing layer sweep, after OSOURC has updated layer 1) and
designed instrumentation recording SHCGS's one use point (PCORR) as a real input, same pattern as D34's
FSR/FSRZ. Before writing the Python port, reread the source once more and caught that P0L itself is ALSO
used directly in the TF0 correction (a different coefficient, 7.53d-8 vs PCORR's 8.19d-8, applied straight
to P0L rather than folded into PCORR) -- the first instrumentation pass hadn't dumped P0L, which would have
made the freezing branch impossible to validate. Fixed with one more instrumentation+rebuild+rerun cycle
before any port code was written, rather than discovering it via a failing test later.

Non-freezing branch bitwise exact on all 18 records (22,224 real (i,j,l) triples/date) after the fix. The
freezing branch never fires in this window at all -- checked explicitly across every record rather than
assumed -- physically plausible since deep-layer freezing this early into a short run is rare; cross-checked
against a synthetic below-freezing input instead. 38 tests added. GROUND_OC's two real-physics pieces
(OSOURC, D34; this sweep, D35) are both now validated; what's left for GROUND_OC as a whole is wiring them
together plus its OIJ diagnostic bookkeeping (likely skippable, matches the CHECKT/CONSERV_O* pattern).

## 2026-09-29: D36 scoping -- OCNDYN.f's legacy dynamical core is dead code; live core is OCNDYN2.f
Investigated GLMELT (72 lines, glacial meltwater) as a D36 candidate: found it's called only from
`daily_OCEAN`'s `end_of_day=.true.` branch (`OCNDYN.f:1548`, gated at `MODELE.f:802`), i.e. once per
calendar day, not once per DTsrc step -- the existing 6-step/3-hour test windows are not confirmed to
cross a day boundary, so it may dump zero records. Deferred rather than risk an unvalidatable delta.

Pivoted to OFLUX (mass-flux computation) and found it depends on OPFIL, which reads an external `AVR`
reduction-matrix file and does land-basin-aware Fourier filtering (OFFT/OFFTI) -- a "new architecture"
item on the scale of D29's ADI solve, not a quick delta. While scoping it, grepped every call site of
OFLUX/OADVM/OADVV/OPGF/OVtoM/OMtoV/OSTRES/OBDRAG/OPFIL project-wide and found **all of them are dead
code**: `OCNDYN.f`'s entire driver subroutine is literally named `OCEANS_old` and commented out in full
(`OCNDYN.f:18-289`), superseded by a live rewrite in `OCNDYN2.f` (`SUBROUTINE OCEANS`, called from
`OCN_DRV.f:41`) with its own renamed/rewritten routines: `OFLUXV`, `ODHORZ`/`ODHORZ0`, `OPFIL2`,
`OSTRES2`, `OBDRAG2`, `OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2`/`OADVUZ`. `OCOAST` is the one routine shared
unchanged between the two (defined once in `OCNDYN.f`, called live from `OCNDYN2.f`). `GROUND_OC` (hence
D33-D35) is also called live from `OCNDYN2.f`'s `OCEANS` (line 114), confirming those three deltas are
sound. This resolves the earlier "6,062-line, not yet read" ambiguity for `OCNDYN.f`'s remaining
subroutines: everything in `OCNDYN.f` past `GROUND_OC`/`PRECIP_OC`/`OSOURC`/`OCOAST` and a handful of
`CONSERV_O*`/init routines is legacy and should not be ported. The real remaining Stage 2 dynamical core
lives entirely in `OCNDYN2.f` (2,670 lines) instead.

Confirmed the P2SAoM40 rundeck (`decks/P2SAoM40.R`) sets `KOCEAN=1` meaning "ocn is prognostic" (a real
13-layer dynamic ocean, not slab/Q-flux as briefly hypothesized before rereading the rundeck comment) --
so this dynamical core is genuinely live physics for this config, just relocated to a different file than
initially assumed.

## 2026-09-29: D36 -- OSTRES2 (momentum-stress application), and a restart double-buffering trap
Picked OSTRES2 (~100 lines, `OCNDYN2.f`) as the first `OCNDYN2.f` delta: applies wind stress (`oDMUA`/
`oDMVA`, atmosphere-ocean momentum flux) and ice-ocean stress (`oDMUI`/`oDMVI`) to the ocean's layer-1
`UO`/`VO`/`UOD`/`VOD` velocities. Live, called unconditionally once per DTsrc step from `OCEANS` right
after `GROUND_OC`, no external files, no FFT -- a deliberately smaller item after OFLUX/OPFIL's "new
architecture" scale ruled that combination out for now. Confirmed via `OGEOM.f`'s `GEOMO` that the needed
static ocean-grid geometry (`DXYSO`/`DXYNO`/`DXYVO`/`COSIC`/`SINIC`) is fully analytic (same lat-lon
formulas as the atmosphere grid, matching D34's precedent), and that `LMU`/`LMV` (the depth masks OSTRES2
guards on) are `MIN(LMM(i,j),LMM(i+1,j))` / `MIN(LMM(i,j),LMM(i,j+1))` respectively (`OCNDYN.f:488,499`) --
simple derived quantities, not separately file-sourced. Instrumented `OSTRES2` itself (not the driver) to
dump its own static geometry once (unconditional, unit 976) plus a full-grid before/after record per call
(unit 981) using its own already-in-scope USE'd variables -- avoided touching `OCEANS`'s call site or
`OCNDYN.f`'s `init_OCEAN` entirely. Patches: `OCNDYN2_ostres2.f.patch`, `ATM_DRV_ostres2.f.patch`. Rebuilt
clean.

First rerun produced **zero** dump files, including the unconditional geometry dump -- meaning OSTRES2
never executed at all, not even once. Root cause was operational, not physics: GISS ModelE's restart
reader picks whichever of `fort.1.nc`/`fort.2.nc` has the *later* itime (`RESTART DISK READ, UNIT 2` in
the log, not UNIT 1), so after D35's prior run had already advanced all three run directories' checkpoints
forward, this rerun silently started from the *end* of the previous window instead of its beginning --
same root cause as D27's "second run silently did zero steps" bug, but sharper: D27's fix note said
"re-copy fresh restart files" without stating that *both* fort.1.nc and fort.2.nc must be reset to the
same pristine itime, since the reader takes the max of the two, not fort.1.nc specifically. `run_nov26`
had both files advanced (restored from the untouched persistent master at
`ModelE_Support/huge_space/P2SAoM40/fort.1.nc`, confirmed itime=33312 by inspecting it directly with
netCDF4 -- this file has never been run in-place, only copied from); `run_dec01`/`run_jan01` still had a
pristine `fort.1.nc` but a stale `fort.2.nc`, fixed by copying `fort.1.nc` over `fort.2.nc` in each. Rerun
in progress.

## 2026-09-29: D37 -- OCOAST ported and validated; a permanent fix for the restart-file trap
Picked `OCOAST` (`OCNDYN.f:4514-4570`, 57 lines) as the next `OCNDYN2.f`-live delta: damps the
X/Y horizontal gradient moments of ocean tracer fields (`GXMO`/`SXMO`/`GYMO`/`SYMO`) in coastal
grid boxes, by a single `REDUCE = 1 - DTS/(86400*20)` factor applied to layers `LMIN..LMM(I,J)`
where `LMIN` is one more than the shallower of the two horizontal neighbors' depth. Confirmed the
one routine genuinely shared unchanged between `OCNDYN.f` (definition) and `OCNDYN2.f` (the live
caller, gated `if (OCoastal_drag==1)`, true for this rundeck). No external files; `DTS=DTSRC=
1800.0` (`OCNDYN.f:405`) and `SECONDS_PER_DAY=86400.0` are both compile-time constants, so
`REDUCE` is fully analytic, no dump needed for it.

**Hit the D36 restart double-buffering issue again, and this time fixed it permanently rather
than patching around it per-run.** Rerunning after the rebuild found `dec01`'s restart pair had
itself been advanced by D36's own prior rerun (D36 legitimately ran 6 real steps from `dec01`'s
then-pristine `fort.1.nc`, consuming it) -- there was no longer any untouched pristine `dec01`
restart anywhere, unlike `nov26` (which has a permanent untouched master at
`ModelE_Support/huge_space/P2SAoM40/fort.1.nc`) and `jan01` (found this delta: an exact-match
permanent archive at `ModelE_Support/huge_space/P2SAoM40/1JAN1950.rsfP2SAoM40.nc`, itime=17520,
never run in place). Regenerated `dec01`'s pristine restart properly: built a one-off bootstrap
run directory, copied the `nov26` master in, edited `I`'s `YEARE/MONTHE/DATEE/HOURE` to run the
full 5 days (240 steps) from 1950-11-26 to 1950-12-01 instead of the usual 6-step/3-hour window,
and let it run to completion (~13 minutes). **Archived all three dates' pristine restarts
permanently** at `ff_data/_pristine_restarts/fort1_{nov26,dec01,jan01}_itime{...}.nc` so this
never has to be regenerated or hunted for again, in this or any future session -- this is the
fix build_and_run.md's D36 note should have prescribed from the start rather than only documenting
the symptom. `build_and_run.md` updated accordingly.

While `dec01`'s bootstrap ran in the background, ran `nov26`/`jan01` immediately (they didn't
depend on it) and validated the port against both -- bitwise exact, first try, before `dec01`
was even available. `fullfidelity/ocoast_ff.py`/`ocoast_jax.py`: the JAX port vectorizes the
Fortran wraparound-I neighbor pattern via `jnp.roll` (same technique as D36's `ostres2_jax.py`)
and the variable-length `L=LMIN..LMM(I,J)` inner loop via a broadcast layer-index comparison
merged with `jnp.where`. Both bitwise exact on all 3 dates once `dec01`'s dump landed too.
`tests/test_ocoast_jax.py` (17 tests): real-record validation x2 ports x3 dates, an analytic-
`REDUCE`-constant check, jit-vs-eager, and three mutation/non-vacuousness checks (some coastal
cell must exercise `LMIN>1`; changed cells must be scaled by exactly `REDUCE`; cells with an
empty `L`-range must be byte-identical before/after). Full regression: 505 passed (488 + this
delta's 17), 0 failed.

## 2026-09-29: D38 -- OBDRAG2 ported and validated, permanent restart fix paid off immediately
Picked `OBDRAG2` (`OCNDYN2.f:2592-2682`, 91 lines) as the next delta: implicit bottom-layer drag
applied only at the per-column deepest active layer (`L=LMU(I,J)` for the east-edge U/VOD pair,
`L=LMV(I,J)` for the north-edge V/UOD pair), scaling velocity by
`(MO_l+MO_r)/(MO_l+MO_r+2*DTS*BDRAGX*sqrt(WSQ))`. Confirmed `OCN_GISS_TURB` is not `#define`d for
this build (checked the compiled `rundeck_opts.h` flag list directly), so the tidal-enhancement
branch (`taubx`/`tauby`/`rhobot`/`idrag`) never compiles in -- only the simple
`bdragfac=BDRAGX*sqrt(WSQ)` path is live, `BDRAGX=1.0` a plain constant.

D37's permanent pristine-restart archive (`ff_data/_pristine_restarts/`) paid for itself
immediately: restoring all three dates for this delta's rebuild was a single three-line copy
from the archive, no bootstrap run needed, no debugging. (The turn that produced this delta was
also resumed mid-flight after a tool interruption during the restart-copy step -- verified the
actual on-disk state directly before redoing it, rather than assuming the interrupted command's
outcome, per the harness's own guidance on interrupted tool calls.)

`fullfidelity/obdrag2_ff.py`/`obdrag2_jax.py`: float64-op-order exact (1e-17 to 1e-18 absolute,
well within tolerance) against real Fortran on all 3 dates, both ports, first try -- including
getting the Fortran `Max(J1O,J1)..JNP` J-band right (J=2..JM-1 for this serial run) without
needing an empirical correction cycle. JAX port vectorizes the "operate only at the per-column
bottom layer" selection via `jnp.take_along_axis` to gather the one relevant layer per cell,
computes the drag there, then scatters back with a broadcast layer-index-equality mask merged
via `jnp.where` (a new technique beyond D36/D37's uniform-J-range masking, needed here because
the affected layer varies per column). `tests/test_obdrag2_jax.py` (17 tests): real-record
validation x2 ports x3 dates, a compile-time-constants check, jit-vs-eager, mask non-vacuousness,
a per-layer mutation check (only the bottom layer may change, checked layer-by-layer not just
globally), and a physical-sanity check (drag never increases speed, and does fire somewhere
real). Full regression: 522 passed (505 + this delta's 17), 0 failed.

## 2026-09-29: D39 -- polar UOD/VOD relax block + polevel() ported and validated
Picked the "relax UOD,VOD toward 4-pt avgs of UO,VO" block inside `OCEANS` itself (`OCNDYN2.f:
179-228`, ~55 lines) plus its `polevel()` helper (`OCNDYN2.f:1530-1564`, ~35 lines) as D39: for
each of the 13 ocean layers, relaxes the D-grid velocities (UOD at V-points, VOD at U-points --
the cross-registration convention established since D36) toward a 4-point average of the C-grid
velocities (UO, VO), RELFAC=0.005 (~4-day damping time constant). Before relaxing, `polevel()`
reconstructs the North Pole row of UO/VO from the ring of V-velocities at J=JM-1 via a discrete
wavenumber-1 (Fourier mode 1) projection, the same trigonometric idiom as D29's DYNSI pole
handling and OVtoM/OMtoV's dead south-pole code. Confirmed no south-pole case exists here either
(consistent with D36-D38: this ocean grid's south pole sits inside Antarctic land).

Confirmed `nbyzu`/`nbyzv`/`i1yzu`/`i2yzu`/`i1yzv`/`i2yzv` (the Fortran's precomputed per-(J,L)
contiguous-I-segment lists) are an exact cached representation of "cells where LMU(I,J)>=L" (or
LMV) by reading their construction site directly (`OCNDYN.f:505-518`: `qexist(:) = (l <=
lmu(:,j))` fed through a run-length-encoding helper `get_i1i2`) -- so this delta uses the LMU/LMV
point-masks already validated in D36-D38 rather than reconstructing the segment-list structure,
avoiding a whole new geometry primitive.

Instrumented inside `OCEANS` directly (captured UO/VO/UOD/VOD pre-copies right after
`relfac=.005d0`, dumped the full before/after record right after the L-loop closes, before the
subsequent leapfrog-init reinitialization block) -- same pattern as D36's OSTRES2. Patches:
`OCNDYN2_polerelax.f.patch`, `ATM_DRV_polerelax.f.patch` (units 986/987). Rebuilt clean, reran
all 3 dates with the now-permanent pristine-restart archive (a plain three-line copy, no
debugging), confirmed no regression against every earlier delta's dumps.

`fullfidelity/polerelax_ff.py`/`polerelax_jax.py`: float64-op-order exact against real Fortran on
all 3 dates, both ports, first try -- including getting both of UOD's two distinct formulas
right (the pole-adjacent row's doubled `2.*uo(i,j+1,l)` term vs. the interior rows' non-doubled
`uo(i-1,j+1,l)+uo(i,j+1,l)` pair) and confirming VOD's formula never needs pole-row access at all
(its J,J-1 pair stays within rows 1..JM-1 for the full J=2..JM-1 range). JAX port loops over the
13 layers implicitly via broadcast (not a Python loop), vectorizing `polevel`'s per-layer pole
reconstruction across all layers at once via a masked reduction, and the UOD/VOD 4-point averages
via `jnp.roll`. A first draft of the test suite's own physical-sanity check (bounding the
relaxation step size relative to the field's own magnitude) was too strict near VOD~0 and had to
be corrected to an absolute bound -- caught by the test itself failing, not by inspection; the
tight `test_ff_matches_real_fortran` check was never at risk, only the looser sanity check.
`tests/test_polerelax_jax.py` (21 tests): real-record validation x2 ports x3 dates, `COSU`/`SINU`
endpoint checks, jit-vs-eager, a pole-row-reconstructed-nonvacuously check, a per-row mutation
check (UO/VO must be byte-identical off the pole row), mask non-vacuousness, and the corrected
physical-sanity bound. Full regression pending a final rerun before commit.

## 2026-09-29: D40 -- ODHORZ0 ported and validated; a North Pole mask design gap caught by the dump
Picked `ODHORZ0` (`OCNDYN2.f:1718-1862`, 144 lines) as D40: prepares the pressure profile
(`P`/`OPBOT`) and seawater-equation-of-state quantities (`GUP`/`GDN`/`SUP`/`SDN`, `dZGdP`, `VBAR`,
`DH3D`) that the (not yet ported) horizontal pressure-gradient solve will need. Confirmed
`USE_OPGFQ=0` for this rundeck (checked `OCEAN_COM.f`'s default, not overridden in
`decks/P2SAoM40.R`) -- the "Linear Upstream Scheme" branch is the only live one; the "Quadratic
Upstream Scheme" alternative is dead code here, halving the real scope. `VOLGSP` (the seawater
equation-of-state trilinear interpolation over a 43x41x40-entry table, `OCNFUNTAB.f`'s `OCFUNC`
module, read from `OFTAB` at init -- the same file D35's `SHCGS` depends on) is recorded directly
at its two per-cell call outputs (`VUP`, `VDN`), the established pattern, rather than replicating
the table. Reused `polevel()` (ported in D39) unchanged -- called once more per layer here.

**Caught a real bug via the design-gap-before-declaring-done reflex, but this time via the dump
comparison itself rather than a rereading of the source.** First validation attempt showed
`OPBOT`/`GUP`/`GDN`/`SUP`/`SDN` failing by large margins (~1.4e7 for `OPBOT`) while
`dZGdP`/`VBAR`/`DH3D`/`MO`/`UO`/`VO` matched exactly -- isolating the bug to the routine's own
pressure/EOS arithmetic, not the recorded-`VUP`/`VDN` or `polevel` pieces. Tracing the single
worst-mismatched cell found it was `(I=2, J=JM)`: the real dump has `OPBOT=0` there even though
`LMM(2,JM)` is a valid nonzero depth. Rereading `OCNDYN.f`'s `nbyzm` construction (already read in
D39, but not re-checked closely enough for this specific case) showed the North Pole row is
**hard-restricted to I=1 only** (`i1yzm(1,jm,l)=i2yzm(1,jm,l)=1`, set unconditionally, independent
of `LMM`'s value at other longitudes there) -- a real, silent exception the naive
`LMM(I,J)>=L` mask (correct everywhere else, including for D36-D39's `LMU`/`LMV`) does not
capture. Fixed with an explicit `m_active(i,j,l)` helper overriding the mask at `J=JM`, then
re-validated. A dedicated regression test (`test_north_pole_only_cell_one_is_active`) pins this
down so it can't silently regress in a later delta that reuses the pattern. Also caught, while
fixing the above: the original draft had dropped the `MMI=MO*DXYPO` divisor from
`GUP`/`GDN`/`SUP`/`SDN` entirely -- fixed by deriving `DXYPO(J)` analytically (reusing D36's
`ostres2_ff.geomo_arrays()`, `DXYPO=DXYS+DXYN`) before the first validation run, so this one never
reached the dump-comparison stage as a live failure.

`fullfidelity/odhorz0_ff.py`/`odhorz0_jax.py`: bitwise/float64-tolerance exact against real
Fortran on all 3 dates, both ports, after the two fixes above (the JAX port's `opbot`/`gup`/`gdn`
show ~1e-8/1e-11 absolute noise from `cumsum`-vs-sequential-sum floating-point reordering,
~1e-15 relative -- expected and harmless). JAX port vectorizes the top-down pressure integration
via `jnp.cumsum` and the North Pole one-cell-only restriction via an explicit mask override
mirroring the plain-Python `m_active` helper. `tests/test_odhorz0_jax.py` (17 tests): real-record
validation x2 ports x3 dates, compile-time-constants check, jit-vs-eager, a dedicated North-Pole-
one-cell-only regression test, a pole-copy-fields-uniform check, and mask non-vacuousness. Full
regression pending a final rerun before commit.

## 2026-09-29: D42 -- ODHORZ ported and validated (plain-Python); first "new architecture"-scale delta closed
Committed to porting `ODHORZ` (`OCNDYN2.f:1185-1560ish`, the horizontal momentum + mass-continuity
solve `ODHORZ0`, D40, prepares pressure/EOS inputs for) after confirming via full reading that
`OPFIL2`'s two per-layer outputs (`USMOOTH`, `PGFX`) could be recorded directly, decoupling this
delta from the `OPFIL2`/`AVR`-file "new architecture" dependency -- the same pattern as D40's
`VOLGSP`. This is genuinely large and multi-physics (pressure/geopotential/thickness
accumulation, kinetic energy, pressure-gradient force, vorticity, Coriolis, mass continuity,
called several times per DTsrc step in a leapfrog/Euler-predictor pattern with a separate
"H"/history state and current INOUT state) -- comparable in scope to D29's ADI solve, the
largest single delta since then.

**A design gap caught mid-writing, before any validation run**: the first draft needed `OGEOZ`'s
per-call initialization (`-HOCEAN*GRAV`), and `HOCEAN` (bathymetry) had not been recorded --
caught while writing the port itself (not by a failing test), fixed with one more
instrumentation+rebuild+rerun cycle (a small, targeted addition: one static geometry dump,
`ffz_odhorz_hocean.bin`) before writing any further port code.

**A second real bug caught before the first validation run, by re-reading the Fortran once
more**: `OPBOT` (a 2D array, no layer index) accumulates mass-convergence contributions across
ALL 13 layers within a single `ODHORZ` call (`opbot(i,j) = opbot(i,j) + convij*grav`, using the
running value) -- the first draft incorrectly reset it from `OPBOT0` every layer, which would
have silently kept only the last layer's contribution. Fixed before running the comparison.

`OMEGA` (planetary rotation rate) is a genuine runtime parameter
(`omega = 2*pi/rotationPeriod`, not a hardcoded constant) -- used at Earth's standard sidereal
value and validated empirically against real Fortran output, the same approach as D29's
`RADIUS`/`GRAV` (both independently confirmed Earth-standard for this rundeck via exact matches).

Re-verified every `OGEOM.f` geometry formula (`SINVO`/`SINPO`/`DXPO`/`DYPO`/`DXVO`/`DYVO`) line by
line against the source before use, rather than trusting the earlier D36-era derivations by
analogy -- caught no errors this time, but the discipline paid for itself given how much rode on
getting the Coriolis terms (`corofj = 2*omega*sinpo(j)` / `sinvo(j)`) right.

`fullfidelity/odhorz_ff.py`: matches real Fortran to float64-tolerance (max ~7e-8 absolute,
consistent with floating-point operation-order noise across the full multi-term momentum
equation) on **all 15 real call records** (5 `ODHORZ` invocations per date x 3 dates) after the
two fixes above -- first full validation run, no further bugs found. `tests/test_odhorz_ff.py`
(16 tests): real-record validation across all records and dates, a multiple-calls-per-window
sanity check, a dedicated regression pin for the `OPBOT` cross-layer-accumulation bug, mask
non-vacuousness, and a bathymetry sanity check. **JAX vectorization deliberately deferred** as
its own follow-up, per this project's established discipline (GHY's lesson: prove F0 correctness
first; a routine this large and multi-physics deserves dedicated care when batching it, not a
rushed pass appended to an already-long delta). Full regression pending a final rerun before
commit.

## 2026-09-30: D43 -- OFLUXV + OADVUZ ported and validated
Picked `OFLUXV` (`OCNDYN2.f:711-800`, ~90 lines) + `OADVUZ` (2471-2521, ~35 lines) as D43: the
"long-timestep vertical redistribution of mass" step, called once per NOCEAN iteration right
after the leapfrog `ODHORZ` loop closes. Reading it fully corrected an earlier scoping
assumption: `OFLUXV` does **not** call `OPFIL2` at all (the earlier "OFLUXV, calls OPFIL2" note
was wrong -- `OPFIL2`'s filter-coefficient setup module just happens to sit immediately after
`OFLUXV` in the source file, unrelated). No external dependencies at all: `DZO`/`ZE` (L13 fixed
layering) already analytically available from D34's `DZO_L13`; `OPRESS` recorded as a real input.

Traced the exact `nbyzm(j,2)`/`nbyzm(j,l+1)` layer-shifted masking carefully before writing any
port code: layer 1's rescale and the bottom-layer update are BOTH gated on "column reaches layer
2" -- meaning single-layer columns (`LMM==1`) are never touched by this routine at all, a genuine
edge case verified by reading the source, not assumed. Real data across all 3 test dates has zero
single-layer columns (checked explicitly), so this branch is cross-checked against a synthetic
two-column case instead -- the same honest-scoping pattern as D28's rain branch.

**A real `DXYPO`/`DTOLF` bookkeeping bug caught before the first validation run**: `SMW`'s
accumulation carries a `DXYPO(J)/DTOLF` factor in the Fortran, later divided out again (`bydxypo`)
when averaging onto U/V-points -- at U-points the `DXYPO(J)` cancels completely since both
neighbor terms share the same J, but a residual `/DTOLF` remains; at V-points each term's own
`DXYPO(J)`/`DXYPO(J+1)` cancels individually, again leaving `/DTOLF` in both terms. The first
draft dropped both factors as if they cancelled completely -- traced through the algebra by hand
before running anything, caught the missing `/DTOLF`, fixed.

`fullfidelity/ofluxv_ff.py`: bitwise/float64-exact against real Fortran on all 3 dates, first
full run after the fix. `fullfidelity/ofluxv_jax.py`: vectorized over (I,J) via `jax.lax.scan`
sequential in L for `OADVUZ`'s Courant-style recurrence -- **a second real bug caught here**: a
first, fully-dense version produced NaN at genuinely-inactive (I,J) columns (0/0 division, since
naive vectorization computes every cell unconditionally where the Fortran's nbyz-gated loop skips
them entirely, leaving state frozen) -- fixed by threading an explicit active-cell mask through
the scan's carry (R/CMUP/FMUP all frozen, not recomputed, at inactive cells), matching the
Fortran's per-cell skip semantics exactly. `tests/test_ofluxv_jax.py` (18 tests): real-record
validation x2 ports x3 dates, jit-vs-eager, a `ZE`/`DZO` derivation check, an explicit "no real
single-layer columns" confirmation plus the synthetic cross-check, a dedicated NaN-regression
pin for the OADVUZ masking bug, and a non-vacuous mass-redistribution check. Full regression
pending a final rerun before commit.

## 2026-09-30: D44 -- ODHORZ's SMU/SMV accumulation (completing ODHORZ for D45)
Started scoping D45 (the `OADVTX2`/`OADVTY2`/`OADVTZ2`/`OADVT2` tracer-advection family, next per
D41's own scoping notes) by re-reading `OADVT2`'s dispatcher (`OCNDYN2.f:1853-1914`) and its
callers in `OCEANS` (lines 355-369, `CALL OADVT2(MO1,G0M,...,DTDUM,.FALSE.,...)` /
`CALL OADVT2(MO1,S0M,...,DTDUM,.TRUE.,...)`). It reads `SMU`/`SMV`/`SMW` from `OCEAN_DYN` as real
mass-flux inputs (`USE OCEAN_DYN, only : mb=>mmi,smu,smv,smw`). `SMW` is D43's already-ported
`OFLUXV` output. `SMU`/`SMV` turned out not to be set anywhere I'd already ported -- grepped the
whole tree for `mmi\s*=`/`smu\s*=`/`smv\s*=` assignments outside the dead `#ifdef TRACERS_OCEAN`
block and found exactly one live site: inside `ODHORZ` itself
(`OCNDYN2.f:1351` `SMU(I,J,L) = SMU(I,J,L) + MU(I,J)*xeven`, `:1443` the `SMV` equivalent). D42's
port never captured this since MO/OPBOT's validation never exercised it.

Traced the whole accumulation scheme by hand from the `OCEANS` driver before writing any code,
since it initially looked far more complex than it turned out to be: `Do NO=1,NOCEAN` --
`NOCEAN=1` for this rundeck (`OCEAN_COM.f:112`, not overridden in the rundeck), confirmed by
balance-tracing the Do/enddo nesting all the way from `Do NO=1,NOCEAN` (line 249) through
`Call OFLUXV` (line 350) and finding they're still at the same nesting depth -- i.e. `OFLUXV` and
`OADVT2` fire *inside* the NOCEAN loop, once per NOCEAN sub-step, which for `NOCEAN=1` is simply
once per `OCEANS` call. `SMU`/`SMV` are zeroed once at the top of that loop
(`OCNDYN2.f:259-260`), then each of the 5 `ODHORZ` calls per window contributes
`mu[i,j]*xeven`/`mv[i,j]*xeven`, where `xeven=1` only for the 2 "even" leapfrog-substep calls
(`qeven=.true.`, inside the `do n=1,neven` loop) and 0 for the other 3 (the 2 initial odd-state-
init calls before the loop, plus the 1 odd leapfrog-corrector call) -- confirmed by dumping
`xeven` directly rather than assuming the call sequence.

Crucially, the `mu`/`mv` arrays used for this accumulation are the *exact same* arrays D42's port
already computes for the mass-continuity (`MO`/`OPBOT`) update, already implicitly validated by
D42's own bitwise-exact match -- so this delta only needed to record the accumulation, not
re-derive the flux fields themselves.

New instrumentation (additive, doesn't touch D42/D43's already-validated dump formats):
`ffdump_odhorz_smuv` (unit 993, per-call SMU/SMV before/after + xeven, same 5-records-per-itime
cadence as `ffdump_odhorz`) and `ffdump_smfinal` (unit 994, the fully-integrated SMU/SMV right
before `OFLUXV` is called -- the real ground-truth D45 will consume). Extended `odhorz()` with
optional `qeven`/`smu0`/`smv0` args (backward-compatible: the original D42 call signature still
returns a 6-tuple untouched).

`fullfidelity/odhorz_smuv_compare.py`: **bitwise-exact, first try, zero mismatches** on both the
per-call replay and the full 5-call chained accumulation against `ffz_smfinal`'s real recorded
final values, all 3 dates -- confirming the hand-traced accumulation scheme (including the
`xeven` pattern and the `NOCEAN=1` simplification) was exactly right. `tests/test_odhorz_smuv.py`
(13 tests). JAX deliberately deferred, same as `ODHORZ` itself. Full regression pending a final
rerun before commit.

## 2026-09-30: D45 -- OADVT2/OADVTX2/OADVTY2/OADVTZ2 (tracer advection), three real bugs, all found
by isolating real mismatches rather than by inspection
Picked up the tracer-advection family scoped since D41: `OADVT2`'s dispatcher
(`OCNDYN2.f:1853-1914`) plus `OADVTX2`/`OADVTY2`/`OADVTZ2` (1916-2423). Confirmed `TRACERS_OCEAN`
is not defined for this rundeck, so `OADVT2` fires exactly twice per `OCEANS` call (`G0M`
QLIMIT=.FALSE., `S0M` QLIMIT=.TRUE.), each re-deriving `MA` identically from the same real
`SMU`/`SMV`/`SMW` (D43/D44). Added `ffdump_oadvt2_before`/`_after` (real tracer-moment + SMW
state bracketing both calls).

First full port attempt produced errors up to 1e14 in magnitude. Root cause: `OADVTX2`'s `mudt`
array is a SINGLE array declared once for the whole subroutine call (not reset per row/layer) --
indices 1, 2, IM get unconditionally refreshed from `MU` every pass regardless of activity, while
every other index is only refreshed within that pass's own U-active segments, otherwise
deliberately retaining whatever an earlier, unrelated (L,J) pass left there. Reading
`OCNDYN.f:1494`'s `get_i1i2` (the segment-builder) revealed two more subtleties missed on the
first read: segments are explicitly linear, not circular ("Wraparound is disabled" -- longitude
periodicity at the dateline is handled by OADVTX2's own explicit I=1/I=IM special-casing, not by
the segment structure), and a single-cell M-segment not starting at I=1 is skipped entirely
(`i1yzm>1 .and. i1yzm==i2yzm`, line 2066). Rewrote `OADVTX2` as a precise segment-based
transliteration (a Python `_get_i1i2` matching the real algorithm line-for-line) -- fixed most of
the error but left a widespread (~16,600 cells) small-but-real (~1e-5 relative) mismatch.

Isolated the SECOND bug by checking global/per-layer mass conservation (near-perfect, ruling out
a gross flux error) then bisecting which of X1/Y/Z/X2 introduced the divergence via a
debug-only instrumentation addition (`ffdump_oadvt2_stage`, dumping MA/RM/RX/RY/RZ after each
of OADVT2's 4 sub-stages, S0M call only) -- found the error was ALREADY present before ANY
advection ran, in the seed `MA=MMI` itself. An earlier attempt had re-derived `MMI` as
`MO0*DXYPO(J)` using `ODHORZ0`'s own already-validated `mo0` input; this doesn't match the real
`MMI`, because `MMI` is a persistent `OCEAN_DYN` module array that `ODHORZ0` only partially
overwrites (not a dense recomputation covering every cell). Fixed by dumping `MMI` directly
(`ffdump_mmi`) instead of re-deriving it -- this alone took all 8 tracer-moment fields
(G0M/GXMO/GYMO/GZMO/S0M/SXMO/SYMO/SZMO) to bitwise-exact, leaving only `MA` itself with a small
residual mismatch.

Used the same stage-debug instrumentation to isolate the THIRD bug to exactly `OADVTZ2`, at
exactly the pole row (J=JM), growing with layer depth -- the classic signature of
persistent-per-cell state diverging over a sequential layer scan. Recognized this as the same
`nbyzm` North-Pole restriction found in D40 (J=JM restricted to I=1 only): `OADVTZ2`'s
`cmup`/`fmup`/... arrays are carried across layers per-(I,J), so processing every I at the pole
pointwise (via `LMM(i,JM)`) let each I accumulate its OWN history, while the real Fortran leaves
I=2..IM's state frozen at 0 forever. Applied the same `m_active`-style override used since D40 --
fixed the pole mismatch completely.

A fourth, smaller fix surfaced along the way: `OADVTY2`'s pole-averaging
(`mo(:,j,l)=sum(mo(:,j,l))/im`) used `np.sum()`, whose pairwise/blocked reduction rounds
differently from ifort's `-fp-model strict` sequential `SUM` intrinsic for the same 72 terms --
replaced with a dedicated `_fortran_sum` left-to-right accumulator.

Removed the debug-only `ffdump_oadvt2_stage` instrumentation before finalizing the delta's
patches (it served its bisection purpose but was never meant to be permanent), did a final clean
rebuild + 3-date rerun to regenerate dumps without it, and re-validated: still bitwise-exact.
`fullfidelity/oadvt2_ff.py`/`oadvt2_compare.py`: all 9 checked fields, all 3 dates, 0.000e+00
max-abs-diff. `tests/test_oadvt2_ff.py` (22 tests). JAX deferred (same discipline as D42's
`ODHORZ` -- this routine's dynamic segment structure and pole-masking subtlety make it a poor
first batching candidate). Full regression: 629 passed, 0 failed. Committed.

## 2026-09-30: D46 -- mesoscale-mixing scoping, correcting D41's backwards assumption
With `OCNDYN2.f`'s real per-step core fully closed (D45), moved to the next item:
`OCNMESO_DRV.f`/`OCNTDMIX.f`/`OCNGM.f` (mesoscale/Gent-McWilliams mixing). D41 had only glanced
at this family and concluded `CONSTANT_MESO_DIFFUSIVITY` meant a simplified, non-GM path was
live. Reading `ocnmeso_drv` (`OCNMESO_DRV.f:131-432`) properly this time showed the OPPOSITE:
`CONSTANT_MESO_DIFFUSIVITY` implies `#define USE_1D_MESODIFF` (a chain inside `OCNMESO_DRV.f`'s
own header, easy to miss without reading start-to-end), which only fixes the diffusivity
*coefficient* fed into the full Redi/GM skew-flux scheme -- it does not replace that scheme with
something simpler. Traced `use_tdmix` (module default 0, confirmed not overridden by grepping the
rundeck's parameter list for `ocean_use_tdmix`) to find `ocnmeso_drv` actually takes its
`else ! skew-GM` branch: `GMKDIF` (`OCNGM.f:125-323`) then `GMFEXP` (`OCNGM.f:325-494`), both
real, substantial routines I hadn't previously read at all.

Followed the dependency chain one level further: `ocnmeso_drv`'s `densgrad` helper needs
`G3D`/`S3D`/`P3D`/`V3D` module state that turned out to be set by a routine I'd never scoped,
`ocnstate_derived` (`OCNDYN2.f:1568-1706`) -- found by grepping for where these arrays get
assigned, since `densgrad`'s own `USE` statement just consumes them without hinting at the
source. Confirmed it's called twice per `OCEANS` invocation (lines 166 and 557, the second right
before `ocnmeso_drv` itself at line 577).

**Good news found along the way**: `OCNTDMIX.f` (2,030 lines, the single largest file in this
family) is entirely dead code for this build -- `use_tdmix=0` means the whole `if(use_tdmix==1)`
block in `ocnmeso_drv` (including every `OCNTDMIX.f` call) never executes. That's over 2,000
lines removed from the real scope in one grep-and-confirm.

No port code written -- this was a scoping delta, matching D41's precedent (read fully, size
accurately, don't rush a port before understanding the real call graph). Revised estimate for
the actual port: ~1,455 real live lines across `ocnstate_derived`+`densgrad`+`get_1d_mesodiff`
(OCNMESO_DRV.f/OCNDYN2.f side) + `GMKDIF`+`ISOSLOPE4`+`GET_PSI_DIAG`+`GMFEXP`+its three flux
helpers (OCNGM.f side) -- comparable in scale to D45. `FULL_FIDELITY_PLAN.md`'s row for this
family rewritten with the corrected scope and the OCNTDMIX.f dead-code finding.

## 2026-09-30: D47 -- ocnstate_derived + densgrad + get_1d_mesodiff, bitwise-exact first try
Ported the first piece of D46's scoped mesoscale-mixing family: `ocnstate_derived`, `densgrad`'s
vertical-gradient portion, and `get_1d_mesodiff`. Before writing any port code, re-checked
D46's note that `ocnstate_derived` is called twice in the source -- ran the instrumented build
and the real dump came back with exactly 1 record per itime, not 2, prompting a closer look:
the first call site (`OCNDYN2.f:166`) is gated by `#ifdef TRACERS_OceanBiology`, not defined for
this rundeck. Caught by checking the actual dump rather than trusting the earlier source read,
consistent with this project's standing discipline of verifying against real output.

Reused D40's already-validated `ODHORZ0` output (`DH3D`, `OCEAN_DYN`'s module-level `DH` array)
as `densgrad`'s real input rather than re-instrumenting it -- confirmed `ODHORZ0` runs once per
`OCEANS` call (`NOCEAN=1`, D44) and `densgrad` reads the same, unchanged `DH` later in the same
call. Recorded `VOLGSP`'s outputs (`VUP`/`VDN`/`VUPU`/`VDNU`) directly per the established
"record what's not yet ported" pattern (same `OFTAB` dependency as D35/D40); `TEMGSP`'s `T3D`
similarly recorded though nothing in this delta's scope consumes it; `RHOX`/`RHOY` (needing two
more `VOLGSP` evaluations each) recorded as final outputs rather than further decomposed.

First validation run: bitwise-exact on every field, all 3 dates, but with a `RuntimeWarning`
(divide by zero) from `ocnstate_derived`'s `rho` computation. Traced it to the North Pole again:
the first draft looped every I pointwise via `LMM(i,j)`, but `nbyzm` restricts real computation
at J=JM to I=1 only (D40's finding, reused throughout D42-D47 now) -- for I>1 at the pole, the
real Fortran's `VUP`/`VDN` are simply never written (stay 0), and the SAME dump reflects that,
so the pointwise loop computed a transient 1/0 that the subsequent pole-copy step then silently
overwrote with the correct value. Not a correctness bug (results were already right), but fixed
with the same `m_active` mask used since D40 anyway -- cleaner, and matches the real control flow
instead of relying on a downstream overwrite to paper over a divide-by-zero.

`fullfidelity/ocnmeso_ff.py`/`ocnmeso_compare.py`: bitwise-exact on `G3D`/`S3D`/`P3D`/`VBAR`/`RHO`
and `DZV`/`BYDZV`/`BYDH`/`RHOMZ`/`BYRHOZ`, all 3 dates, after the pole-mask cleanup.
`tests/test_ocnmeso_ff.py` (15 tests). `GMKDIF`/`ISOSLOPE4`/`GET_PSI_DIAG`/`GMFEXP` (the actual
Gent-McWilliams skew-flux application, ~1,143 lines in `OCNGM.f`) remain the next piece of this
family, not yet started.

## 2026-09-30: D48 -- GMKDIF/ISOSLOPE4/GMFEXP scoping, two more real dead-code findings
Read the rest of the mesoscale-mixing family in full: `GMKDIF`, `ISOSLOPE4`, `GET_PSI_DIAG`,
`GMFEXP`, and its three helpers (`computeFluxes`/`wrapAdjustFluxes`/`addFluxes`). Two real findings,
neither assumed going in:

`GET_PSI_DIAG` (called from inside `GMKDIF`) turned out to be purely diagnostic -- its own header
comment even says so ("Calculate bolus velocity diagnostics"), and its only writes are to `OIJL`
and subroutine-local scratch arrays, never read back by anything else. Confirmed by reading its
full USE list (`use odiag, only : oijl=>oijl_loc,ijl_mfub,ijl_mfvb`) before deciding to skip it,
not assumed from the name alone.

The bigger find: `GMKDIF` has an internal `QCROSS` flag (`QCROSS = .NOT.(RGMI.eq.1d0)`) that gates
roughly half of its own coefficient-setting logic (the cross-term arrays coupling isoneutral and
thickness diffusion) and a matching half of `GMFEXP`'s flux terms. Checked the actual call site in
`ocnmeso_drv` (`call gmkdif(k3d,1d0)`) -- `RGMI_in` is hardcoded to `1d0` in the source itself, not
a rundeck-tunable parameter, so `QCROSS` is always false for every configuration that reaches this
call. That's a genuine, provable dead-code finding (not a guess about typical parameter values),
cutting the remaining real scope roughly in half.

No port code written -- this was a scoping delta, same precedent as D41/D46 (read fully before
committing to a port). Revised estimate for the actual port: ~700 real live lines (`GMKDIF` ~100,
`ISOSLOPE4` 136, `GMFEXP` ~120, `computeFluxes` ~90, `wrapAdjustFluxes` ~110, `addFluxes` ~145) --
smaller than D46's first estimate now that `GET_PSI_DIAG` and the `QCROSS` branches are excluded.
`ISOSLOPE4` in particular looks like a good port candidate next: every input (`RHOX`/`RHOY`/
`RHOMZ`/`BYRHOZ`/`BYDH`/`DZV` from D47, `K3D` from D47's `get_1d_mesodiff`) is already validated,
and the computation is embarrassingly parallel per-cell with no sequential dependency across I/J/L
-- unlike almost everything else ported so far in Stage 2.

## 2026-09-30: D49 -- ISOSLOPE4 ported, a pole-row loop-bound bug caught by round-number diffs
Ported `ISOSLOPE4` as planned -- its real inputs are exactly D47's already-validated `densgrad`
outputs plus `get_1d_mesodiff`'s constant-800 `K3D`, so the only new instrumentation needed was a
dump of `ISOSLOPE4`'s own 24 output arrays (reusing `ffz_densgrad` as ground-truth input, no new
input dump).

First validation run: 16 of 24 fields (`ASX0-3`/`ASY0-3`/`S2X0-3`/`S2Y0-3`) bitwise-exact, but the
other 8 (`AIX0-3`/`AIY0-3`) failed with suspiciously round max-diffs (~1000, ~667 -- not noisy,
data-dependent numbers). Investigated the single worst cell directly: manually recomputing the
formula by hand at that exact (I,J,L) gave the CORRECT (real) answer, while the actual port
function returned 0. That meant the cell was never being touched by the loop at all. The cell was
at J=JM (the North Pole row) -- and every other routine ported so far in Stage 2 restricts its
main loop to J=2..JM-1, excluding the pole (handled separately via an explicit copy step). Assumed
the same convention here without checking, which was wrong: `ISOSLOPE4`'s real loop genuinely
includes J=JM. The reason 16 of 24 fields still matched despite the bug: at the specific cells
checked, `RHOX`/`RHOY` (from D47) happened to be zero at the pole, so the `AS`/`S2` products
(`AIxST*SIx`) come out zero either way -- masking the bug in those fields while the `AI` arrays
themselves (which don't multiply by `RHOX`/`RHOY`) exposed it directly.

Fixed by extending the loop's upper bound from JM-1 to JM; bitwise-exact on all 24 fields, all 3
dates, immediately after. `fullfidelity/gmredi_ff.py`/`gmredi_compare.py` (new module for the
Gent-McWilliams family). `tests/test_gmredi_ff.py` (9 tests), including a dedicated regression pin
for the pole-row bug specifically (checking AIX0 at J=JM is both nonzero in real data and matched
by the port). `GMKDIF`'s remaining (post-QCROSS) coefficient logic and `GMFEXP`+helpers are next.

## 2026-09-30: D50 -- GMKDIF's remaining coefficients, bitwise-exact first try
Ported the rest of `GMKDIF` (post-`QCROSS`, D48): `BXX`/`BYY`/`BZZ` and the twelve Z-direction
flux coefficients (`AZX` family), gated by `L>KPL(I,J)`. Found one new real dependency while
reading it closely: `KPL` (mixed-layer-depth index) is set by `OCNKPP.f`'s `OCONV`, which isn't
ported yet -- recorded it directly as a real input, same discipline as every other not-yet-ported
dependency this project has handled (OPRESS, VUP/VDN, etc.).

Applied D49's J-range finding (main loop includes J=JM) directly this time instead of assuming
the old pole-exclusion convention again -- paid off immediately, first validation run came back
bitwise-exact on all 15 fields, all 3 dates, no debugging needed. Reasoned through (rather than
guessed) why the real source's separate "J=J_1STG+1" extension block never fires for this
rundeck's serial execution: it exists to handle a domain-decomposition halo boundary in a
multi-process run, and in serial J_STOP_STGR already equals JM, making the block's own guard
always false. The bitwise-exact result confirmed this reasoning was right rather than just
assumed.

`fullfidelity/gmredi_ff.py` (extended)/`gmkdif_compare.py`. `tests/test_gmkdif_ff.py` (12 tests).
Only `GMFEXP`+its three flux helpers (`computeFluxes`/`wrapAdjustFluxes`/`addFluxes`, ~465 lines
-- the actual flux application to G0M/S0M) remain to close out the Gent-McWilliams family.

## 2026-09-30: D51 -- GMFEXP+helpers ported, bitwise-exact first try, Gent-McWilliams family closed
Ported the last piece: `GMFEXP` plus `computeFluxes`/`wrapAdjustFluxes`/`addFluxes`. Re-derived
all four routines by hand from the full source read during D48's scoping (re-verified the exact
formulas, loop bounds, and pole-handling structure before writing any code, rather than working
from memory of the earlier read).

One real ambiguity surfaced while doing this: `GMFEXP` reads a module-level `MO` array, but D45's
`OADVT2` only explicitly updates `MO1` (a separate dummy argument bound to `MA`). Searched the
whole `OCEANS` driver body between the `OADVT2` calls and `ocnstate_derived`/`GMFEXP` for an
explicit `MO = MO1` sync and found none -- inconclusive from source alone whether `MO` reflects
the post-advection mass or an earlier snapshot. Rather than spend more time chasing this through
the leapfrog even/odd state alternation (which has tripped this project up before, e.g. D42's
restart double-buffering issue), just recorded `MO` directly at the point `GMFEXP` reads it --
the same "when in doubt, record it" discipline used for other ambiguous upstream quantities all
session. The bitwise-exact result confirms this was the right call.

Confirmed two things from the close re-read that mattered for the port's correctness, not
assumed: `GMFEXP`'s own main loop excludes the pole rows entirely (unlike `ISOSLOPE4`/`GMKDIF`'s
pole-inclusive J range established in D49/D50) -- so `TXM`/`TYM` at the poles should pass through
completely unchanged, which became a dedicated regression test. And the real Fortran's own
`TZM`-update code inside `computeFluxes` is commented out in full (several lines of dead code
still sitting in the source) -- `TZM` is an `INTENT(INOUT)` argument that's never actually
written to, confirmed by checking the real dump's `TZM` before/after are bit-identical before
writing the port, then pinned as its own regression test too.

First validation run came back bitwise-exact on all 4 fields (`TRM`/`TXM`/`TYM`/`TZM`), both
calls (`G0M`/`S0M`), all 3 dates -- no debugging needed, the largest delta of this family closed
cleanly on the first try. `fullfidelity/gmredi_ff.py` (extended)/`gmfexp_compare.py`.
`tests/test_gmfexp_ff.py` (15 tests). This closes the Gent-McWilliams mesoscale-mixing family
opened in D46: `OCNMESO_DRV.f`+`OCNGM.f`'s entire real per-step live path is now ported and
validated.

## D52: OCNKPP.f scoped end to end -- KPP vertical mixing sized as the next big item

With Gent-McWilliams closed, `OCNKPP.f` (3,714 lines, the K-Profile-Parameterization vertical
mixing scheme) is now the largest unscoped file in Stage 2's ocean core. D41 had scoped it only
at a glance back on 2026-09-29, lumping several nested subroutines into one rough `KPPMIX`
range. This delta read the whole file line by line and corrected that.

The corrected subroutine list turned up more dead code than expected. Four subroutines are
entirely dead: `get_kvtdiss` (145 lines, gated by `use_tdiss` which defaults to 0 and is never
set in the rundeck -- checked the rundeck directly, same discipline as D48's `QCROSS` finding),
`get_gradients0` (119 lines, its only call sites sit inside the already-known-dead
`OCN_GISS_SM` block, and a model-tree-wide grep found no other caller anywhere), and `wscale`
(64 lines) and `swfrac` (34 lines), whose every call site in this file is commented out with an
explicit "! inlined" note -- their logic was hand-inlined into `KPPMIX`/`bldepth` rather than
called. A cross-file grep for a live `wscale` turned up a same-named subroutine in `mxkprf.f`,
which turned out to be the *atmosphere's* PBL mixing scheme -- an unrelated false-friend name
collision, not a caller of this file's `wscale`. `ddmix` (52 lines) is also dead, gated by a
compile-time `LOGICAL, PARAMETER :: LDD = .false.` rather than a runtime flag.

One real process error this delta, caught and corrected within the same pass: an early grep for
`OVDIFFS` call sites (`CALL OVDIFFS|call ovdiffs`) came back with hits only inside `STCONV`,
suggesting `OCONV` (the main-grid driver) used some other mechanism entirely for tracer vertical
diffusion. Re-grepping case-insensitively found the actual answer: the real source writes these
calls as `Call OVDIFFS` (capital C, otherwise lowercase) -- a case variant the first grep's
literal alternation didn't cover. `OCONV` does call `OVDIFFS` extensively (for `G0ML`/`S0ML` and
the linear moments `GXML`/`GYML`/`SXML`/`SYML`); the tracer-moment-diffusion calls right after
those (`GXXML`/`GYYML`/etc.) turned out to be dead anyway, gated by `use_qus` defaulting to 0
and never set in the rundeck -- a new dead-flag finding of its own, and one that generalizes:
`use_qus` gates similar blocks in `OCNDYN.f`/`OCNDYN2.f`/`OCNMESO_DRV.f`/`OCN_TRACER.f` too,
flagged for whenever those are next touched.

Traced the live call graph precisely rather than assuming from subroutine position in the file:
`bldepth` is a *sibling* of `KPPMIX`, not nested inside it -- both `OCONV` and `STCONV` call it
separately, before their own `KPPMIX`/`KPPMIX`-equivalent call, confirming boundary-layer depth
is computed as a precursor step. `kmixinit` is called exactly once at model startup
(`init_OCEAN`), not per-timestep, so it gets the same treatment as other init-only routines
already in this project (D29's `RADIUS`/`GRAV`, D50's `KPL` default) -- its output can be
recorded as a fixed known input once identified, rather than ported as per-step physics.
`alloc_kpp_com` is allocation-only boilerplate, also not real physics to port. `STCONV` is
confirmed live (called from `OCNDYN2.f:487`, both its gating conditions always true in this
serial/`NOCEAN=1` build) but depends entirely on `OSTRAITS.f`'s `STRAITS` module data, which has
not been scoped at all -- left out of this delta's live-scope count as a separate follow-on.

Net result: a (revised-again in D53, see below) main-grid live-scope estimate -- about 3x the
size of the entire Gent-McWilliams family just closed in D46-D51. This is confirmed as the
single largest remaining physics item in Stage 2's ocean core. No port code this delta;
`FULL_FIDELITY_PLAN.md`'s `OCNKPP.f` row rewritten with the corrected boundaries and findings
above.

## D53: correction -- bldepth is dead, not live; KPPMIX inlines its own boundary-layer logic

Started reading `KPPMIX`'s full body to prepare for the actual port (the natural next step after
D52), and immediately hit something that contradicted D52's own finding: D52 had called
`bldepth` a live sibling of `KPPMIX`, called separately by `OCONV` and `STCONV`. Re-checking both
call sites found them sitting inside `#ifndef OCN_GISS_TURB -> CALL KPPMIX(...) #else ->
call bldepth(...) #endif` -- and `OCN_GISS_TURB` is confirmed undefined for this build (the same
finding D41/D46 already established for this exact file). So `KPPMIX` is the live branch at
*both* call sites; `bldepth` never actually runs. D52's grep (`call bldepth`) found the two real
call sites and reasonably read "called from two places, unconditionally" as "live" without
checking the surrounding `#ifdef` -- the same kind of miss this project has caught before (D48's
`QCROSS`, and this same delta's own `use_qus`/`TRACERS_OCEAN` findings) checked the guard and
this one, on a faster first pass, didn't.

Reading `KPPMIX`'s own body explained why immediately: right where the commented-out
`c call bldepth (...)` sits (`OCNKPP.f:390`), the *entire* boundary-layer-depth algorithm is
inlined directly afterward -- the bulk-Richardson-number search loop with its `goto`, the
`wmt`/`wst` lookup-table velocity-scale interpolation, the `swfrac` shortwave-fraction formula
inlined a second time. Read the rest of `KPPMIX` straight through after that and found the same
pattern for `blmix` -- never a separately-called live subroutine in this file at all, its whole
boundary-layer-mixing-coefficient computation (`OCNKPP.f:580-830`) is inline in `KPPMIX` too.
This is consistent with (and explains) D52's other finding that `wscale`/`swfrac` have no live
caller anywhere -- their logic was hand-copied into `KPPMIX`/`bldepth` rather than called out to,
and since `bldepth` itself turns out to be dead, that inlined logic only actually executes once,
inside `KPPMIX`.

One more small finding from the full read: `KPPMIX`'s `Coriol` (Coriolis parameter) argument is
threaded through the signature but never used in the live path -- its only reference is inside a
block the source itself has commented out with an explicit "NOT USED" label
(`OCNKPP.f:550-561`, the `hekman`/`hmonob` depth-limit check). `alphaDT`/`betaDS` are live
arguments too, but only read inside `if (LDD) call ddmix(...)`, and `LDD` is the compile-time
`.false.` constant D52 already found -- so like `Coriol`, they need to be threaded through a
faithful port's signature but don't affect its output.

Corrected `FULL_FIDELITY_PLAN.md`'s `OCNKPP.f` row: revised main-grid live-scope estimate is now
~2,235 lines (`OCONV`+`KPPMIX`+`z121`+`KVINIT`+`OVDIFF`+`OVDIFFS`+`REDUCE_FIG`, `bldepth`'s 225
lines removed), still roughly 3x the Gent-McWilliams family's size. No port code this delta
either -- but the KPPMIX read that surfaced this correction also means the next delta can start
the actual port immediately: `KPPMIX`+`z121`, fed by a direct (not dumped) port of `kmixinit`'s
`wmt`/`wst` lookup tables and `FZ500` array, since `kmixinit` is pure closed-form math over fixed
physical constants and the fixed vertical grid `ZE` -- the same "record/derive fixed setup data
directly" precedent as D29's `RADIUS`/`GRAV` and D47's `MESO_DIFFUSIVITY_CONST`.

## D54: KPPMIX + z121 + kmixinit + init_solar ported -- and a genuine, diagnosed departure from
bit-for-bit validation

Ported the first real piece of the KPP scheme: `KPPMIX` itself (confirmed the live routine by
D52/D53, not `bldepth`), its one dependency `z121`, and `kmixinit`/`init_solar` (the one-time
setup that builds `KPPMIX`'s velocity-scale lookup tables and shortwave-fraction profile) --
ported directly rather than dumped, since both are pure functions of fixed constants and the
fixed vertical grid.

`KPPMIX` sits inside `OCONV`'s fixed-point HBL iteration (up to 4 calls per column per `OCEANS`
call). Rather than port that whole iterative structure in one delta, I instrumented the real
`CALL KPPMIX` call site directly and dumped every real call's inputs and outputs as its own
record -- a new dump shape for this project (one record per real Fortran call, not one per
itime), since a single itime now produces thousands of records instead of one. This keeps the
same "port one routine, record its real neighbors" discipline this whole project has used, just
applied to a routine called a variable number of times per step instead of once.

First validation pass came back close but not exact -- errors around 1e-7 to 1e-8, small enough
to suggest precision rather than a logic bug, but real. Chased it down methodically rather than
accepting it: wrote a tiny standalone `ifort -fp-model strict` program to reproduce `kmixinit`'s
`cg` constant bit-for-bit, and it only matched once I used a *single-precision* `1./3.` exponent
promoted to double, not the "obviously correct" double-precision `1.0d0/3.0d0` translation I'd
started with. The real source writes `(concs*vonk*epsilon)**(1./3.)` -- neither `1.` nor `3.` has
a `d0` suffix, so Fortran parses them as single-precision `REAL`, divides in single precision
(~7 significant digits), and only then promotes that already-rounded result to double before
using it as the exponent. This is a genuinely different number from `1.0d0/3.0d0`, and it matters
once raised to a real*8 base. Found the identical bug pattern a second time in the same
subroutine's `Vtc` constant (`sqrt(0.2/concs/epsilon)`, `0.2` with no `d0` suffix) once I knew to
look for it. Fixing both made `cg` and `Vtc` match the real dumped constants exactly.

Even with both fixes, the `wmt`/`wst` lookup tables (892x482 entries each) aren't *quite*
bit-identical to a real one-time dump of the whole table, which I added a temporary DEBUG-ONLY
hook to capture rather than guess at: 62 and 38 cells respectively (out of 429,944) differ from
the real values, every single one by exactly 1 ULP. Traced one by hand -- recomputing it via
`pow` and, separately, via the mathematically equivalent `exp(y*log(x))` gives the *same* 1-ULP-
off answer both ways, which settles it: this isn't a translation choice I can fix, because IEEE
754 requires `+`,`-`,`*`,`/`,`sqrt` to round correctly but explicitly does not require `pow`/
`exp`/`log` to -- two independently correct implementations (glibc's, under numpy/Python here,
and Intel's libimf, under ifort even with `-fp-model strict`) are allowed to disagree in the last
bit for bit-identical inputs, and apparently do, for about 0.02% of this specific table's cells.

That's a small but real crack in this project's "always bitwise-exact" record, and I'd rather
document it precisely than paper over it. Quantified the actual consequence across all 76,011
real `KPPMIX` calls in the standard 3-dates-x-6-steps sweep: the table-level 1-ULP noise
propagates through the bilinear interpolation and the HBL bulk-Richardson search to a max
absolute residual of about 5e-6 (in `GHAT`, which divides by a small velocity scale and so
amplifies the noise the most), with a *median* residual around 2e-12 and 99th-percentile around
2.5e-10 -- and `KBL`, the integer boundary-layer-index output that actually drives downstream
branching, never mismatched once. Validated in `tests/test_kppmix_ff.py` at this suite's
ordinary `atol=1e-6` tolerance (not a special loosened one -- the existing convention already
comfortably accommodates a residual this small), including a regression pin
(`test_akvs_and_akvg_are_identical`) for the fact that `AKVS`/`AKVG` come out structurally
identical for this build (no double-diffusion means no distinguishing tracer diffusivity from
heat diffusivity), confirmed against a real record rather than just asserted from reading the
source.

`fullfidelity/kppmix_ff.py`/`kppmix_compare.py` (new, 25 tests). `LDD`/`alphaDT`/`betaDS`/
`Coriol` (all real Fortran inputs to `KPPMIX`, all confirmed structurally dead by D52/D53) left
out of the port's own signature entirely, rather than threaded through unused. Remaining in this
file: `OCONV`'s own ~1,526-line per-column driver (including the fixed-point iteration this delta
deliberately didn't port), `KVINIT`, `OVDIFF`/`OVDIFFS`, `REDUCE_FIG`, and `STCONV` (still
blocked on unscoped `OSTRAITS.f`).
