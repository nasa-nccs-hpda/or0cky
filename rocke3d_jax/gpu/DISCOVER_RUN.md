# Running the JAX port on Discover: step by step

Written 2026-10-07 from the owner's paths. Nothing here has been run on the cluster yet (this node cannot reach it). Do the steps in order; each one is cheap and tells you whether the next is worth trying.

Paths (the owner's):
- Project directory on Discover: `P=/discover/nobackup/gtamkin/dev/ilab-agentic-ai/projects/imvi/rocke3d_jax`
- Container: `/discover/nobackup/projects/QEFM/qefm-core/../containers/qefm-core-sandbox` (has torch; whether it has JAX with CUDA is what step 3 tells you)
- Data goes under `$P` (see step 2), the job script reads it from `$P/ff_data` and `$P/modelE2_planet_2.0/ModelE_Support/prod_input_files` through the environment variables `FF_DATA` and `MODELE_PROD_INPUT`.

## 1. Get the code
```
cd $P && git pull origin full-fidelity-port       # needs the push of the commits below to be done (see the end)
```

## 2. Move the data (atmosphere-only set, 1.0 GB, 58 files)
A ready tarball is on the node where the work is done:
`/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/rocke3d_atm6_data.tar` (md5 in the file next to it, `.md5`). Get it to Discover by whatever route you use (scp/rsync/Globus; I cannot do the transfer from here), then:
```
cd $P && tar -xf /path/to/rocke3d_atm6_data.tar          # creates $P/ff_data/nov26/...
md5sum -c /path/to/rocke3d_atm6_data.tar.md5              # run next to the tar file
```
The coupled-run set (1.26 GB, 70 files) and the exact file lists are in `gpu/DATA_SUBSET_6STEP.md`; I can build a tarball for it on request. Not needed for steps 3 to 5.

## 3. Does the container have JAX with CUDA? (about 1 minute, no data needed)
```
cd $P && MODE=smoke sbatch gpu/run_gpu_job.sbatch
```
Read `gpu_runs/<jobid>/stdout.txt`. Look at: `GPU backend active` (PASS/WARN), `float64 arrays`, `jit+scan 6 steps`, `pure_callback inside jit`, `jax starts with --xla_cpu_max_isa=AVX ...` (WARN or FAIL here means the CPU flags of our bitwise validation must be dropped for the GPU build), `Intel libimf runtime` (WARN expected: libm mode only, unless you copy the 2020 Update 4 runtime and set `INTEL_LIBIMF_DIR`). If `GPU backend active` is WARN, the container has no CUDA JAX: stop here and tell me; I will write a container recipe.

## 4. GPU speed and device math (about 2 minutes)
```
cd $P && MODE=probe sbatch gpu/run_gpu_job.sbatch
```
Reports float64 matmul and scan timings per device and how many ulps the device `exp/sin/log/pow` differ from NumPy.

## 5. Your existing benchmark (the manual command, now one line)
```
cd $P && MODE=bench sbatch gpu/run_gpu_job.sbatch
```
Same command as your manual session; stdout and stderr are kept in separate files.

## 6. The JAX atmosphere step on the GPU (needs step 2; long: first call compiles)
```
cd $P && MODE=atm sbatch --time=3:00:00 gpu/run_gpu_job.sbatch
```
6 steps of nov26 from the recorded boundary data, per-step timings in `gpu_runs/<jobid>/atm/jax_timing.json`. On CPU the first step took about 3,000 s (cold compile); on the GPU it may differ. This is the atmosphere half only and is NOT yet "JAX-driven" in the sense of `Reports/ACCEPTANCE_CRITERIA.md`; the number to read is the steady-state step time.

## 7. Folding the results back
```
python gpu/collect_gpu_results.py gpu_runs/<jobid>
```
Paste the printed table to me, or commit `gpu_runs/<jobid>/results.json`. Every result must say: libm mode (no libimf), the GPU model, the git commit, and that radiation is recorded in this stage (no Fortran callback yet).

## What can go wrong (known)
- Results depend on the CPU core count and on any warm JAX compile cache (D175). The job script unsets the cache variables; do not set them.
- Bitwise equality with the Fortran needs libimf (not in the container). On the GPU expect category C or D against the real model until the libimf host callback exists.
- The XLA flags in `clouds_jax_env.py` are CPU flags; if step 3 says the GPU build rejects them, the atmosphere runner needs a GPU-specific flags path (a small change I can make).

## Result log

**2026-10-07 17:11, job 58776063, node warpa009, `MODE=smoke`** (repository commit `c7c42c8` cloned to `/discover/nobackup/gtamkin/dev/or0cky_port/rocke3d_jax`): **27 pass, 2 warn, 0 fail**, wall 52 s. GPU: NVIDIA A100-SXM4-40GB (driver 535.104.12), JAX 0.6.1 / jaxlib 0.6.1 with the CUDA backend ACTIVE, Python 3.11.11, numpy 1.26.4, 8 of 48 CPUs usable by the job, 29.5 GiB device memory limit. float64 works; jit+scan on the model-sized state compiled in 0.6 s; a host callback inside jit costs 1.6 ms for a 1 MiB round trip; the CPU-only XLA flags (`--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp`) do not break the GPU build; netCDF4, scipy and mpmath import; all nine project modules import (including `jax_atm_step` and `model_driver`). Warnings (expected): `pytest` is not installed in the container; the Intel libimf runtime is not present, so GPU results are libm-mode unless the libimf host callback is used. NOT yet measured: probe (`MODE=probe`), any real model step on the GPU, the data tarball.

**2026-10-07 23:22, job 58778982, node warpa004, `MODE=probe`** (repository commit `c7c42c8`): exit 0, wall 4 s. NVIDIA A100-SXM4-40GB, JAX 0.6.1, float64 on.
- float64 matmul 2048 x 2048: **1.2 ms** per product (this CPU node: about 820 ms on one core).
- `jit` + `lax.scan` of 100,000 trivial sequential steps: **575 ms** on the A100 (the same loop took 41 ms on one CPU core): about 5.7 microseconds per step, i.e. kernel-launch overhead. Long chains of small sequential operations are SLOWER on the GPU than on a CPU core; only wide, parallel work wins.
- Device float64 math functions against NumPy on 2,000,000 values (max difference is 1 ulp in every case): `exp` 6.14% of values not bitwise equal, `sin` 11.96%, `log` 0.22%, `pow(x, 0.25)` 9.07%.
Consequences: (1) bitwise equality (category A) with the CPU, with libm or with libimf, is NOT achievable on the GPU, because its `exp`/`pow` differ by 1 ulp in 6-12% of values; a GPU result can only be judged at rounding-level categories (B/C) for one step and statistically (noise floor) beyond that, unless every transcendental call goes through a host callback to libimf (which removes most of the speed-up). (2) The step's many small sequential operations (about 125 jit executions, ~15,000 eager operations in the hybrid step) are launch-bound on the GPU: the speed-up must come from fusing them into few large kernels (the J1/J2/J3 plan), not from moving the present structure to the GPU unchanged.

