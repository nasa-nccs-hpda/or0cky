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
