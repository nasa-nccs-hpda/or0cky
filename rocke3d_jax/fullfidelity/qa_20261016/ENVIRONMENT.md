# Running the QA tests without the author's conda environment

Owner: G. Tamkin. Written by a Claude Code session on 2026-10-16. Project-local. Status: DRAFT. Only the author's session has run these; no second person has. Review by: when a second developer first reruns them.

## 1. What needs what

| Test | Needs | Where it was tried |
|---|---|---|
| Package scoring (`scoring_tool/d207_score.py`), `qa_20261016/spread_rmse.py`, `localize_qc.py`, `follow_columns.py`, `review_package_plots.py` | Python 3 with numpy (plots also matplotlib); reads only the review package | Clean environment, site anaconda base python (below): all four scripts that were rerun finished with exit code 0; the localization and spread JSON files are identical to the first runs. The original runs used the author's `graphcast-env`. |
| `clouds_condse_compare.py` (CONDSE against the real Fortran) | numpy only; reads `ff_data/`. `--imf` also needs the Intel libimf runtime (default search directory in `intel_libm_ff.py`, override `INTEL_LIBIMF_DIR`) | Clean environment, step 0: no libimf 151 flipped columns and with `--imf` 6, both with the same flipped-column lists as the first runs. |
| V100 jobs (`gpu/v100_step6_graphcast.sbatch`) | JAX 0.5.3 with the CUDA 12 plugin, Python 3.10, access to `gpulogin1` and a `compute` V100 node | Author's `graphcast-env` only. |

## 2. The CPU tests: use the site python, an empty environment

The site anaconda base python on panfs is shared and needs no personal environment:

    PY=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/python3
    # reported by the author's clean run: Python 3.9.12, numpy 1.21.5, matplotlib 3.5.1
    export REVIEW_PKG=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/rocke3d_review_package_20261016   # or your unpacked copy of the tar
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
* `ff_data/` is the repository's tracked symlink to the real data directory; the CONDSE compare reads it, the package scripts do not.
* In the author's shell `python3` resolved to the same site python but with numpy 2.0.2 and matplotlib 3.9.4 visible (extra packages from the author's environment); the clean run above is the one that shows the minimum needed. Results printed identically.

## 3. The V100 jobs

They need JAX with CUDA, which the site python does not have. The author's `graphcast-env` is described by `gpu/environment-graphcast-env.yml` (jax/jaxlib 0.5.3, `jax-cuda12-plugin` and `jax-cuda12-pjrt` 0.5.3, the `nvidia-*-cu12` wheels, numpy 2.2.4, scipy 1.15.2, Python 3.10.16, netcdf4, xarray). To build your own, untested:

    CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
    $CONDA env create -n graphcast-env -f gpu/environment-graphcast-env.yml      # needs network access to the conda and pip indexes

The recipe calls `conda run -n graphcast-env`, which resolves to the submitting user's own environments directory, so an environment with that name works without editing the sbatch. Before submitting, check:
* `PROJ`, `FF_DATA` and `MODELE_PROD_INPUT` in `gpu/v100_step6_graphcast.sbatch` point at the author's checkout paths; change them to yours (and the `--output` path).
* Submit from forest204 with `ssh gpulogin1 "cd <project dir> && sbatch gpu/v100_step6_graphcast.sbatch"` (partition `compute`, feature `v100`; no SLURM client on forest204). The job needs 2 or more CPU cores (host callbacks inside jit) and no JAX compile cache (the step runner asserts both).
* A bitwise rerun of the author's numbers is expected only on the same code (`061486d`) and a V100; two runs on different nodes were bitwise identical at steps 0 and 5, but a different GPU model was not tried.

## 4. Known gaps

* The env export has not been used to rebuild an environment, and the CUDA wheels depend on the node driver (580.x on the V100 nodes the author used).
* The clean-environment check covered the scripts listed in section 1 (CONDSE compare at step 0 only), not the V100 jobs.
* `gpu/v100_step6_graphcast.sbatch` and `fullfidelity/qa_20261016/rebuild_review_package_tar.sh` still contain the author's absolute paths.
