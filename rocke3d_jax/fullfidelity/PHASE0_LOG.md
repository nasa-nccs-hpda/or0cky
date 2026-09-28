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
