"""GPU model day (item 1, 2026-10-11): the D193 day loop (54 or 108 steps, nov26_day records, radiation REPLAYED) on a GPU in LIBM mode.
    python -u gpu_day_run.py OUTDIR [--nsteps 54] [--timed 0]
Writes OUTDIR/ours_d193/step_<it>.npz (T U V Q P QCL QCI) for `multiday_score.py --ours OUTDIR/ours_d193`, and OUTDIR/d193_run.json (per-step walls, flags, masks).
What differs from d193_day.py (nothing existing is edited): gpu_step_run is imported first (libm wiring, jax before any CPU XLA flag, so they are no-ops);
the XLA-flag provenance requirement is dropped; the day-boundary hook (DAILY_ATMDYN + ch4ox, host NumPy) uses the libm context (A.make_ctx imf=False), because
libimf is not available on GPU nodes.  GPU result: NOT bitwise with the CPU / Fortran; judged by the scorer.  Radiation replayed; recorded inputs as D193."""
import os
import sys

_A = list(sys.argv)
sys.argv = [_A[0], 'nov26', os.environ.get('GPU_IMPORT_SCRATCH', '/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/_import_scratch')]
import gpu_step_run as G   # noqa: E402,F401  (backend, libm wiring)
sys.argv = _A

import atm_step as A       # noqa: E402
import jax_harness as H    # noqa: E402

_prov = H.provenance_header
H.provenance_header = lambda *a, **k: _prov(*a, **{**k, 'require_xla_flags': False})
_mk = A.make_ctx
A.make_ctx = lambda date, imf=False, *a, **k: _mk(date, False, *a, **k)

import d193_day as _D193   # noqa: E402
_D193.DAYDIR = os.environ.get('GPU_DAYDIR', _D193.DAYDIR)   # nov26_day2 for the 108-step (two-day) record set; DayReal and ch4ox_increment read it at call time (as d207_day.py)

if __name__ == '__main__':
    drv = os.environ.get('GPU_DAY_DRIVER', 'd193_day')      # d193_day: radiation replayed; d212_day: pass its own flags (--rad server --surf-fields computed ...)
    if drv == 'd193_day':
        import d193_day as D   # noqa: E402
        D.main()
    else:
        import runpy
        runpy.run_module(drv, run_name='__main__')
