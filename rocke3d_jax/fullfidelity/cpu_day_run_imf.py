"""CPU model day in LIBIMF mode, in the configuration of the review package (written 2026-10-16; nothing existing is edited).

Why: the review package JAX run (GPU, native SOCRATES radiation) made zero libimf calls, and the stand-alone CONDSE comparison shows the cloud scheme matching the
real Fortran in column A with libimf and flipping without it (Reports/QA_HANDOFF_20261016.md section 3 item 8, section 5 item 1).  This runs the same day on the CPU,
where the project's Fortran-faithful libimf path (host callbacks into the Intel libimf) is the default of d193_day / jax_coupled, to see whether the early QCL/QCI
excess disappears.

What it does: forces the CPU backend, imports the CPU XLA flags (clouds_jax_env) before jax, sets d193_day.DAYDIR (d212_day reads it at call time; its own default is
'nov26_day' while the package run used 'nov26_day2'; override with CPU_DAYDIR) and then runs d212_day exactly as `python d212_day.py ...` would, with the same arguments.

Usage (from fullfidelity/; at least 2 cores because host callbacks run inside jit; the earlier CPU run used 4; scratch directories for TMPDIR and D189_SHIM_DIR):
    taskset -c <4 cores> env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> \\
      ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1 \\
      python -u cpu_day_run_imf.py OUTDIR --rad native --surf-fields computed --seed computed --cosz computed --nsteps 54 --timed 0
(the flag list is that of the package README section 2; check it there, it is not repeated authoritatively here).  Writes OUTDIR/ours_d193/step_<it>.npz and OUTDIR/d193_run.json
as d212_day does; score with d207_score.py (CPU XLA flags exported, see Reports/SOCRATES_TRIAL_RUN_RECIPE.md).
CPU_DAY_DRYRUN=1 stops after the checks below and prints them (no model is built).

Not verified: that nov26_day and nov26_day2 give identical day-1 records; the time of --rad native on the CPU (an earlier CPU libimf run with server radiation took about 17 s per step)."""
import os
import sys

# CPU only, decided before jax is imported (a GPU node would otherwise pick the GPU backend and libimf callbacks from there are untested).
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
assert os.environ['JAX_PLATFORMS'].split(',')[0] == 'cpu', 'this wrapper is for the CPU backend: JAX_PLATFORMS=%r' % os.environ['JAX_PLATFORMS']

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import clouds_jax_env  # noqa: E402,F401  (CPU XLA flags BEFORE jax)
import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
assert jax.default_backend() == 'cpu', 'JAX backend is %r, expected cpu' % jax.default_backend()
assert not os.environ.get('JAX_COMPILATION_CACHE_DIR') and not os.environ.get('CLOUDS_JAX_CACHE'), 'no JAX compile cache for validated runs (D175)'
assert len(os.sched_getaffinity(0)) >= 2, 'host callbacks inside jit need >= 2 CPU cores (use taskset -c with at least 2 cores)'

import intel_libm_ff  # noqa: E402

assert intel_libm_ff.available(), 'Intel libimf not found (set INTEL_LIBIMF_DIR to the directory holding libimf.so); this wrapper exists to run in libimf mode'

import d193_day as _D193  # noqa: E402

_D193.DAYDIR = os.environ.get('CPU_DAYDIR', 'nov26_day2')

print(f'cpu_day_run_imf: backend={jax.default_backend()} devices={[str(d) for d in jax.devices()]} cores={sorted(os.sched_getaffinity(0))} '
      f'XLA_FLAGS={os.environ.get("XLA_FLAGS", "")!r} libimf_available=True DAYDIR={_D193.DAYDIR} argv={sys.argv[1:]}', flush=True)

if __name__ == '__main__':
    if os.environ.get('CPU_DAY_DRYRUN') == '1':
        print('CPU_DAY_DRYRUN=1: checks passed, not running the day', flush=True)
        sys.exit(0)
    import runpy
    sys.argv = ['d212_day.py'] + sys.argv[1:]
    runpy.run_module('d212_day', run_name='__main__')
