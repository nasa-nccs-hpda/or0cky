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
