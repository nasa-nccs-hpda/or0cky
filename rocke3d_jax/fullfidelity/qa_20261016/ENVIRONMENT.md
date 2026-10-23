# Running the QA tests without the author's conda environment

Owner: G. Tamkin. Written by a Claude Code session on 2026-10-16. Project-local. Status: DRAFT. Only the author's session has run these; no second person has. Review by: when a second developer first reruns them.

## 1. What needs what

| Test | Needs | Where it was tried |
|---|---|---|
| Package scoring (`scoring_tool/d207_score.py`), `qa_20261016/spread_rmse.py`, `localize_qc.py`, `follow_columns.py`, `review_package_plots.py` | Python 3 with numpy (plots also matplotlib); reads only the review package | Clean environment, site anaconda base python (below): all four scripts that were rerun finished with exit code 0; the localization and spread JSON files are identical to the first runs. The original runs used the author's `graphcast-env`. |
| `clouds_condse_compare.py` (CONDSE against the real Fortran) | numpy only; reads `ff_data/`. `--imf` also needs the Intel libimf runtime (default search directory in `intel_libm_ff.py`, override `INTEL_LIBIMF_DIR`) | Clean environment, step 0: no libimf 151 flipped columns and with `--imf` 6, both with the same flipped-column lists as the first runs. |
| CPU model runs (`cpu_day_run_imf.py`, `d212_day.py`, `d193_day.py`; scoring with `d207_score.py` when the CPU XLA flags are exported) | JAX 0.5.3 (jaxlib 0.5.3), Python 3.10, NumPy 2.2.4, Intel libimf for libimf mode. The CPU XLA flag `--xla_cpu_max_isa=AVX` (set by `clouds_jax_env`) is NOT recognised by the site python's JAX 0.4.30: XLA aborts at start (`Unknown flags in XLA_FLAGS`). Do not use the site python for these. | Author's `graphcast-env` on forest204, CPU backend, 2026-10-21: a full 54-step day ran (about 16 s per step). The site python was tried the same day and aborted. |
| V100 jobs (`gpu/v100_step6_graphcast.sbatch`) | JAX 0.5.3 with the CUDA 12 plugin, Python 3.10, access to `gpulogin1` and a `compute` V100 node | Author's `graphcast-env` only. |

## 2. The scoring and analysis scripts: use the site python, an empty environment

This section covers only the scripts in the first two rows of section 1 (numpy only). The CPU model runs (third row) need JAX 0.5.3, e.g. the shared environment of section 7 or `graphcast-env`; the site python cannot run them (corrected 2026-10-21; this file earlier implied the site python was enough for all CPU work).

The site anaconda base python on panfs is shared and needs no personal environment:

    PY=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/python3
    # reported by the author's clean run: Python 3.9.12, numpy 1.21.5, matplotlib 3.5.1
    export REVIEW_PKG=<unpacked review package directory>
    cd <repo>/projects/imvi/rocke3d_jax/fullfidelity
    E="env -i HOME=$HOME/qa_home MPLCONFIGDIR=$HOME/qa_home PATH=/usr/bin:/bin OMP_NUM_THREADS=1"   # no PYTHONPATH, no activated env
    mkdir -p $HOME/qa_home $HOME/qa_out
    $E $PY qa_20261016/follow_columns.py
    $E $PY qa_20261016/localize_qc.py $HOME/qa_out
    $E $PY qa_20261016/spread_rmse.py $HOME/qa_out
    $E $PY review_package_plots.py $HOME/qa_out          # about 1.5 minutes
    $E $PY clouds_condse_compare.py --dates nov26 --steps 0,1,2,3 --json $HOME/qa_out/cse_libm.json

Notes:
* The package location comes from `REVIEW_PKG`; if it is unset the scripts use the author's package path, which is group-readable (`ilab`). The external reviewer's account is not in `ilab`, so give them an unpacked copy.
* `clouds_condse_compare.py` exits with status 1 whenever it reports failing fields. That is the expected result both without `--imf` (151 flipped columns at step 0) and with it (6), so do not treat exit 1 as a crash; read the last lines (`TOTAL steps ... failing fields ...`). It needs about 105 seconds per step on one core; `--imf` about 125 seconds.
* `ff_data/` was a tracked symlink to the real data directory in the archive branch and is not carried over; set `FF_DATA` to the data directory; the CONDSE compare reads it, the package scripts do not.
* In the author's shell `python3` resolved to the same site python but with numpy 2.0.2 and matplotlib 3.9.4 visible (extra packages from the author's environment); the clean run above is the one that shows the minimum needed. Results printed identically.

## 3. The V100 jobs

They need JAX with CUDA, which the site python does not have. The author's `graphcast-env` is described by `gpu/environment-graphcast-env.yml` (jax/jaxlib 0.5.3, `jax-cuda12-plugin` and `jax-cuda12-pjrt` 0.5.3, the `nvidia-*-cu12` wheels, numpy 2.2.4, scipy 1.15.2, Python 3.10.16, netcdf4, xarray). To build your own, untested:

    CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
    $CONDA env create -n graphcast-env -f gpu/environment-graphcast-env.yml      # needs network access to the conda and pip indexes

The recipe calls `conda run -n graphcast-env`, which resolves to the submitting user's own environments directory, so an environment with that name works without editing the sbatch. Before submitting, check:
* `gpu/v100_step6_graphcast.sbatch` takes `PROJ` from `ROCKE_PROJ` or the submit directory (submit from the `rocke3d_jax` directory); `FF_DATA` and `MODELE_PROD_INPUT` are `$PROJ/ff_data` and `$PROJ/modelE2_planet_2.0/ModelE_Support/prod_input_files`, so provide those (links are fine).
* Submit from forest204 with `ssh gpulogin1 "cd <project dir> && sbatch gpu/v100_step6_graphcast.sbatch"` (partition `compute`, feature `v100`; no SLURM client on forest204). The job needs 2 or more CPU cores (host callbacks inside jit) and no JAX compile cache (the step runner asserts both).
* A bitwise rerun of the author's numbers is expected only on the same code (`061486d`) and a V100; two runs on different nodes were bitwise identical at steps 0 and 5, but a different GPU model was not tried.

## 4. Known gaps

* The env export has not been used to rebuild an environment, and the CUDA wheels depend on the node driver (580.x on the V100 nodes the author used).
* The CPU model runs were tried only in the author's `graphcast-env`, not in the shared environment of section 7 or a freshly built one.
* The clean-environment check covered the scripts listed in section 1 (CONDSE compare at step 0 only), not the V100 jobs.
* The author's absolute paths were removed from the kept scripts on 2026-10-16 (housekeeping, not yet re-tested on a second account); the remaining absolute paths are NCCS site paths (anaconda, Intel libimf, netcdf).

## 5. Environment variables the kept code reads (no author defaults any more)

| Variable | Needed by | If unset |
|---|---|---|
| `FF_DATA` | everything reading recorded Fortran data (`ff_data/`) | falls back to the relative path `ff_data` (cwd), so tests skip or fail on missing files |
| `MODELE_PROD_INPUT`, `MODELE_SRC` | ModelE input files / source for the tests and drivers | relative fallback `modelE2_planet_2.0/...`, same effect |
| `ROCKE_MSTCNV_PY` | `ROCKE_MSTCNV_HOST=1` (numba worker), `run_all_tests_flags_on.sh` | **no default**: error. Must be a python with numba (the author used numba 0.60 with NumPy 2.0.2; `graphcast-env` cannot import numba) |
| `ROCKE_SOCRATES_DIR` | `--rad native` (`products_day0.npz`, `static0.npz`, `spec/spec_{lw,sw}.npz`) | exits with a message |
| `ROCKE_PROJ`, `FORTRAN_RUNS`, `FORTRAN_HUGE_SPACE` | `gpu/*.sh` | error with a message |
| `PY` | `run_all_tests*.sh` | `python` on `PATH` |
| `INTEL_LIBIMF_DIR` | libimf mode | NCCS site path (unchanged) |

## 6. Hidden dependency found by the import gate (2026-10-16)

`advsi_ff.py` imports `mpmath` at module level (REAL*16 reference), and `jax_coupled`, `gpu_step_run`, `d193_day`, `d212_day`, `gpu_day_run`, `cpu_day_run_imf` and others import it through that. The author's `graphcast-env` did not contain it: it came from `~/.local/lib/python3.10/site-packages` (mpmath 1.3.0), so the env export did not reproduce the working setup. `gpu/environment-graphcast-env.yml` now lists `mpmath==1.3.0` (edit untested; rebuild still untested).

## 7. Shared environment (2026-10-18)

A copy of the working JAX environment is at `/explore/nobackup/projects/ilab/conda/envs/rocke3d-jax` (group `ilab`, read-only): Python 3.10.16, jax 0.5.3, NumPy 2.2.4, mpmath 1.3.0. It was made with `conda-pack` from the author's `graphcast-env` (5.3 GB tar), unpacked, relocated with `conda-unpack`, and `mpmath==1.3.0` installed. It passed the 6-step V100 gate bitwise (job 37943470). `conda create --clone` and `conda env create -f gpu/environment-graphcast-env.yml` both stalled on HTTP 429 from `repo.anaconda.com`, so the yml rebuild remains untested. The numba Python for `ROCKE_MSTCNV_PY` is `/panfs/ccds02/app/modules/jupyter/ilab/dev/pytorch-kernel-copy_20260408/bin/python`. Use `PYTHONNOUSERSITE=1`: with the user site on, a personal `~/.local` supplied `mpmath` to the author's environment and hid the missing dependency.
