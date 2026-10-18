# Data, environments and paths a new developer needs (2026-10-18)

Owner: G. Tamkin. Written by a Claude Code session from the paths used in the runs of 2026-10-16 to 2026-10-18. Project-local. Status: DRAFT. The code in this branch contains no author paths; this file is the one place that records where the external pieces are today. Sizes and locations are as measured on 2026-10-18.

## 1. What is outside the repository

| Variable the code reads | What it is | Size | Location on 2026-10-18 (owner's directories, group `ilab`, mode rwxr-x---) |
|---|---|---|---|
| `FF_DATA` | Recorded Fortran runs (the "ff_data" tree): restarts, per-step dumps, the five one-ulp members, reference outputs | 158 GB | `/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data` |
| `MODELE_PROD_INPUT` | Complete ModelE `ModelE_Support/prod_input_files` (must contain `S72x460098M.ext.nc`; the partial copy in the old checkout is not enough) | 0.9 GB | `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files` |
| `MODELE_SRC` | The ModelE checkout (source for a few tests; parent of the line above) | 12 GB | `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0` |
| `FORTRAN_HUGE_SPACE` | Prebuilt production, hook-free Fortran binary `P2SAoM40.bin` with its run files (`I`, `P2SAoM40ln`, `P2SAoM40uln`, `runtime_opts`) | 1.4 GB | `.../modelE2_planet_2.0/ModelE_Support/huge_space/P2SAoM40` |
| `ROCKE_SOCRATES_DIR` | SOCRATES products for `--rad native`: `products_day0.npz`, `static0.npz`, `spec/spec_{lw,sw}.npz` | 0.5 GB | `/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/socrates_nov26` |
| `FORTRAN_RUNS` | Writable scratch for the Fortran timing runs | a few GB per pair | any writable directory (the runs used `gpu_runs/fortran`, which is git-ignored) |
| `ROCKE_MSTCNV_PY` | Python with numba, for the fast configuration only | | `/panfs/ccds02/app/modules/jupyter/ilab/dev/pytorch-kernel-copy_20260408/bin/python` (Python 3.12.11, numba 0.60.0, NumPy 2.0.2; owner `iluser`, group `ilab`, readable by `ilab`). The JAX environment cannot import numba |
| `CONDA_BIN`, `ROCKE_ENV` | conda executable and the JAX environment (jax 0.5.3, NumPy 2.2.4, Python 3.10.16, mpmath 1.3.0) | | conda from the NCCS module: `/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda`; environment (absolute path, group `ilab`, read-only): `/explore/nobackup/projects/ilab/conda/envs/rocke3d-jax`. It is a `conda-pack` copy of the author's `graphcast-env` plus `mpmath`, made 2026-10-18 |

The `ff_data` and `modelE2_planet_2.0` links in a checkout are git-ignored. Access today is by membership of the Unix group `ilab`; I did not check who is in it.

## 2. What each variant reads from `ff_data` (the minimum was not measured)

* The scored day and the Fortran reference: `nov26_day2` (73 GB; 108 steps of dumps and the five members `ffpt_p1..p5`).
* The Fortran timing runs: `_pristine_restarts/fort1_nov26_itime33312.nc` (0.5 GB) plus the binary above.
* Tests: `nov26`, `dec01`, `jan01` (14-15 GB each), `ens_jan1950` (2.3 GB), `advsi_dumps`, `qus1d_harness`, `qus3d_harness`, `ref_libm`. A test whose data is absent skips; 40 skip even with everything present (23 declare why, 17 were not traced).
* The smallest subset that still reproduces the scored day and the timing pairs has not been determined; copying `nov26_day2` and `_pristine_restarts` is the likely minimum (73.5 GB) but this was not tried.

## 3. Cluster facts the runner assumes

* NCCS ADAPT: partition `compute`, `--constraint=v100`, Tesla V100 nodes (gpu005 to gpu013 seen); `forest204` has no SLURM client, so submit with `SBATCH_HOST=gpulogin1` (ssh). `/discover` is not reachable from there.
* Timing pairs need exclusive nodes (the runner asks for them); fidelity-only runs do not.
* The compute nodes are Intel Xeon Gold 6248 (AVX-512). NumPy's AVX-512 `pow` differs in the last bit from the non-AVX-512 host the tests were written on; the runner turns it off for the tests only (`NPY_DISABLE_CPU_FEATURES`). Whether it moves the JAX fidelity ratios is untested.

## 4. Environment block for `gpu/run_variants.sh`

```
export ROCKE_PROJ=<this checkout>/rocke3d_jax            # optional; default is the checkout of the script
export FF_DATA=<ff_data>
export MODELE_PROD_INPUT=<modelE2_planet_2.0>/ModelE_Support/prod_input_files
export FORTRAN_HUGE_SPACE=<modelE2_planet_2.0>/ModelE_Support/huge_space/P2SAoM40
export ROCKE_SOCRATES_DIR=<socrates_nov26>
export FORTRAN_RUNS=<writable directory>
export ROCKE_MSTCNV_PY=/panfs/ccds02/app/modules/jupyter/ilab/dev/pytorch-kernel-copy_20260408/bin/python
export CONDA_BIN=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
export ROCKE_ENV=/explore/nobackup/projects/ilab/conda/envs/rocke3d-jax     # a name also works; an absolute path is passed to conda run -p
export PYTHONNOUSERSITE=1                                # recommended: a personal ~/.local must not change results
export SBATCH_HOST=gpulogin1                             # when the local host has no sbatch
bash gpu/run_variants.sh <run-name>                      # DRY_RUN=1 writes the batch files only; ONLY="fid_fast tests_off" submits a subset
```

The script checks the key files before submitting and stops with a message if one is missing. Results: `gpu_runs/<run-name>/SUMMARY.md`. A complete run takes about 40 minutes of wall time on three GPU nodes plus CPU slots. The 108-step timing pairs include a cold compile of 15 to 18 minutes in step 0.

## 5. Not verified

* That `gpu/environment-graphcast-env.yml` rebuilds a working environment: two attempts from the file stalled on HTTP 429 (rate limit) from `repo.anaconda.com` and never got to a test. The shared environment was made by `conda-pack` instead, not from the file.
* The shared environment `rocke3d-jax` passed the 6-step V100 gate (job 37943470, 2026-10-18, run as the author with the user site off): step 0 (561 fields) and step 5 (572 fields) bitwise identical to job 37943287. It has NOT been used by another person, and the full runner (`gpu/run_variants.sh`) has not been run with it.
* That anyone other than the author can read these directories and run the script end to end. The data trees were made group-readable on 2026-10-18 (0 unreadable files found afterwards).
* The numba Python is shared and readable by `ilab`; it was not rebuilt or tested here beyond the author's earlier fast-configuration runs.
