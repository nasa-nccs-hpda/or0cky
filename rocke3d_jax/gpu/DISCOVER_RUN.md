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

## Running the assembled coupled step on the GPU (D198, 2026-10-08; NOT yet run on a GPU, the work node has none)

What runs: `fullfidelity/gpu_step_run.py`, the assembled device-resident coupled step of D191/D195 (`jax_coupled.Coupled`) for 1 to 6 steps of nov26 (it 33312..33317), on whatever backend JAX finds. Labels it prints and saves: **LIBM mode** (exp/pow from the CUDA device math, no Intel libimf, no libimf callback), **radiation REPLAYED** from the recorded record (no Fortran server), **recorded inputs** (Ent exports, land forcing, tile radiation columns, CONDSE entry set, SURFACE templates), **GPU result NOT BITWISE with the CPU/Fortran result**, hybrid step (host callbacks for QUS subsidence, the two CONDSE pole columns, the OADVT2 pre-pass, a host template build). The CPU-only XLA flags are not applied on the GPU. Details and validation numbers: `fullfidelity/scoping/D198_GPU_ASSEMBLED_STEP_ENTRY.md`.

1. **Data (the old coupled tarball is NOT enough for this runner):** `gpu_data_stage/rocke3d_coupled6_data.tar` holds the `nov26_day/` records used by the closed driver; this runner reads the per-step records from `ff_data/nov26/` (48 of its 66 files are not in the old tarball). Use the new one, on the work node next to the old ones:
   `/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/rocke3d_coupled6b_data.tar` (1.58 GB, 66 files, md5 `f607c3a5be059ad0e3b6a8f7b0076ae0` in the `.md5` next to it; file list `gpu/subset_coupled6b.paths` / `.txt`, from an strace of a complete 6-step run of the runner). Transfer it to Discover by your usual route, then in `$P`: `tar -xf rocke3d_coupled6b_data.tar` (creates `$P/ff_data/...` and `$P/modelE2_planet_2.0/ModelE_Support/prod_input_files/...`) and check `md5sum -c rocke3d_coupled6b_data.tar.md5` next to the tar file. It covers 1 to 6 steps of nov26 and nothing else (no dec01/jan01).
2. **Container packages:** the runner needs numpy, jax with CUDA, netCDF4, scipy, mpmath: all imported fine in the smoke test of job 58776063. It does NOT need pytest, hydra/omegaconf, git, gcc, the Intel runtime, the radiation server or the NumPy reference (checked on the work node with pytest/hydra/omegaconf made unimportable and `INTEL_LIBIMF_DIR` pointing nowhere). `OMP_NUM_THREADS=1` is set by the command; at least 2 CPU cores are required (host callbacks inside jit), the job has 8.
3. **Command:** `gpu/step_command.txt` is filled: `cd $PROJ/fullfidelity && OMP_NUM_THREADS=1 python -u gpu_step_run.py nov26 $OUT/step --steps 6 --save first-last`. For the shortest practice run edit `--steps 6` to `--steps 1` (then step 0 is also repeated once for a steady-state time). Other options (`python gpu_step_run.py -h`): `--ref <npz>` in-script comparison with a saved end state (`{k}` in the file name = step number), `--save all|first-last|last|none`, `--timed` (device block after every stage, per-stage times, slower), `--io-audit`.
4. **Submit:** `cd $P && MODE=step sbatch --time=3:00:00 gpu/run_gpu_job.sbatch`. Optional second run: `GPU_KEEP_ALGSIMP=1 MODE=step sbatch ...` has no effect now (the runner sets no algsimp flag on the GPU); ignore it for this mode.
5. **Expect (warning on time):** the first step compiles about 114 XLA programs: **on the work node's CPU (2 cores) the cold first step took 520 to 580 s, of which 410 to 464 s was compilation, and later steps took 10 to 14 s each** (99 jit executions + about 490 eager dispatches + 176 QUS host callbacks moving about 250 MB per step). The GPU compile time and step time are unknown: the probe shows kernel-launch-bound behaviour for long chains of small operations, so a GPU step may well be slower than 10 s. The requested 3 hours are ample for 6 steps; if the first step has not finished after about 30 minutes, tell me before cancelling. Output appears in `stdout.txt` as each step finishes, and `step/result.json` is rewritten after every step.
6. **What to paste back:** (a) the whole `gpu_runs/<jobid>/stdout.txt` (it is short: header JSON, 5 LABEL lines, one line per step, SUMMARY), (b) `gpu_runs/<jobid>/step/result.json`, (c) `metadata.txt`, and the last 30 lines of `stderr.txt` if the exit code is not 0. If the network allows, also copy `step/nov26_libm_step0.npz` (376 MB) back to the work node and run `python fullfidelity/gpu_step_run.py --compare gpu_step0.npz <reference.npz> <outdir>` (numpy only, no jax): categories A/B/C/D and max relative differences. References on the work node in `gpu_data_stage/` (md5 in `ref_npz.md5`): `ref_cpu_libm_nov26_step0.npz` (the same runner on the CPU, libm, step 0), `ref_cpu_libimf_nov26_step0.npz` (the Fortran-faithful libimf CPU result of D191, step 0), `ref_cpu_libm_nov26_step5.npz` (libm CPU, step 5 = it 33317 of a 6-step run). You can also give the reference to the job (`--ref`) if you copy it to Discover: `--ref $P/ref_cpu_libimf_nov26_step0.npz` compares step 0 inside the job.
7. **How to read it:** a GPU result is expected to be category C or D against the CPU libimf result in fields with threshold behaviour (clouds, precipitation, PBL), as the libm CPU run already is (D198 section 3: 126 A, 170 B, 47 C, 215 D of 558 arrays), and different from the libm CPU result as well (the same step with another XLA version already moves 198 of 558 arrays to category D). This is rounding-level chaos at thresholds, not an error by itself; the dynamics group (T, U, V, P of the DYNAM stage) is the clean indicator (CPU libm vs libimf: all B/C, max 1.7e-11). Acceptance beyond one step is statistical (ACCEPTANCE section 4); a single GPU run proves only that the step executes, how long it takes and where it lands at rounding level.
