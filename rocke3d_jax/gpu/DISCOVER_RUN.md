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
