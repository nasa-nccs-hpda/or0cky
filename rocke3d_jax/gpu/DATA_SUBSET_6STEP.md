# Minimal data for a 6-step nov26 run on a GPU node (measured, not guessed)

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07. Review by: when the step code or the run window changes.

## Method
Each run below was executed once from the real nov26 start state on this node and traced with `strace -f -e trace=openat,open`; every successful read-only open of a file under the project data tree was recorded (the lists are in `subset_*.txt`, size in bytes then path relative to `ff_data/` or `modelE2_planet_2.0/`; `subset_*.paths` are the bare paths for `rsync --files-from`). Python libraries and our source files are excluded.

## Two alternative sets (they are NOT nested: the two code paths read the same steps from different directories)

| Set | Run traced | Files | Size | Source directories |
|---|---|---|---|---|
| **atm6**: atmosphere-only step, 6 steps | `fullfidelity/jax_atm_step_run.py ref` (NumPy reference mode of D180's runner; its `jax` mode reads the same files: checked for step 0 only, 17 of the 58 files, none outside the set) | 58 | about 1.0 GB | `ff_data/nov26/` (the 6-step window) and the per-step records |
| **coupled6**: closed coupled driver, 6 steps | `model_driver.ModelDriver(surface="closed", ent="record", f3=True).run(6)` | 70 | about 1.26 GB | `ff_data/nov26_day/` (the 54-step day, first 6 steps), `ff_data/nov26/` (static and glue files), the restart, `ff_data/advsi_dumps/`, 3 input tables from `modelE2_planet_2.0/ModelE_Support/prod_input_files/` |

The union of both is about 2.2 GB. The coupled set additionally needs the restart `ff_data/_pristine_restarts/fort1_nov26_itime33312.nc` (179 MB), the ocean state `ff_data/nov26/ffo_state*.bin` (87 MB), the radiation outputs `rsv_n26_*_out.bin` (106 MB) and the six per-step CONDSE input files (about 490 MB).

## Copying to Discover
From a node that sees this storage (paths on Discover are the owner's to choose; keep the `ff_data/` structure and set `FF_DATA` or edit the `FF` constant that the modules read):
```
cd /panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai
rsync -av --files-from=<repo>/gpu/subset_atm6.paths . <destination>/        # atmosphere-only, about 1.0 GB
rsync -av --files-from=<repo>/gpu/subset_coupled6.paths . <destination>/     # coupled, about 1.26 GB
```
(The `.paths` entries beginning with `modelE2_planet_2.0/` live under `.../dev/`, not under `ilab-agentic-ai/`; copy those three small files separately.)

## Not covered by these lists
- **The Intel runtime for bitwise results.** The traced runs read `libimf.so` and `libintlc.so.5` from the 2020 Update 4 compiler runtime (`.../intel/.../compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin/`). A Discover container without them runs in libm mode only (rounding level); the headline fidelity comparison needs the libimf host callback (`ACCEPTANCE_CRITERIA.md` section 8).
- **Output and scratch files**, and anything read by a step beyond the sixth.
- **The radiation server run** (a Fortran executable plus its restart): these lists use the recorded radiation outputs (`rsv_n26_*_out.bin`), not a live server.
- **Only nov26.** dec01 and jan01 need their own traces.
- The `jax` mode of the atmosphere runner was traced only through the start of step 0 (the trace crawled under `strace` during JIT compilation); later steps are assumed to read the same files as the NumPy reference mode and were not confirmed.
