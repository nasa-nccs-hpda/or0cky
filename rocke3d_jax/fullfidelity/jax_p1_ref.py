"""D186: C1 reference for the atmosphere PHASE 1 (stages dyn, condse, radia) of the NumPy chained step in libm mode.

Same code path as the D184 reference (atm_step_fast with the batched CONDSE, ctx imf=False, recorded radiation), stopped after the 'radia' stage so that the
phase-1 fields (which the D184 end-of-step reference no longer holds: surface, dissip and filter come after) can be compared stage by stage.
Run: taskset -c 0-1 env OMP_NUM_THREADS=1 python jax_p1_ref.py DATE OUTDIR [TAG]
Saves OUTDIR/<date>_p1_<tag>.npz with keys  dyn/<F>, condse/<F>, radia/<F>  (stage snapshots of the state dict), X/<F> (CONDSE output dict), and
rad/CLDSS, rad/CLDMC (RADIA cloud masking on radiation steps), plus <date>_p1_<tag>.json (timing, flags).
SOCRATES/RADIA not involved (recorded SRHR/TRHR/COSZ1)."""
import clouds_jax_env  # noqa: F401  (XLA flags before any jax import; the NumPy chain itself does not use jax in these stages)
import json
import os
import sys
import time

import numpy as np

import atm_step as A
import atm_step_fast as F


def run(date, outdir, tag='1', itime=None):
    os.makedirs(outdir, exist_ok=True)
    it0 = itime if itime is not None else dict(A.DATES)[date]
    ctx = A.make_ctx(date, imf=False)
    F.ensure_backend(ctx)
    R = A.Real(date, it0, A.FF)
    tm = {}
    ms = {}
    t0 = time.perf_counter()
    with F.fast_condse():
        S, sn = A.run_step(date, it0, ctx, R=R, ms=ms, land_mode='recorded', stop='radia', timing=tm)
    wall = time.perf_counter() - t0
    d = {}
    for st in ('dyn', 'condse', 'radia'):
        for k, v in sn[st].items():
            if isinstance(v, np.ndarray) and v.dtype.kind in 'fiu':
                d[f'{st}/{k}'] = np.asarray(v)
    X = S['_condse_X']
    for k, v in X.items():
        if isinstance(v, np.ndarray):
            d[f'X/{k}'] = np.asarray(v)
    if '_cloud_rad' in S:
        d['rad/CLDSS'], d['rad/CLDMC'] = (np.asarray(a) for a in S['_cloud_rad'])
    base = os.path.join(outdir, f'{date}_p1_{tag}')
    np.savez(base + '.npz', **d)
    json.dump(dict(date=date, itime=it0, wall=wall, timing={k: v for k, v in tm.items() if k.startswith('stage_')},
                   is_radiation_step=bool(A.is_radiation_step(it0)), n_arrays=len(d), taskset=os.sched_getaffinity(0).__repr__(),
                   XLA_FLAGS=os.environ.get('XLA_FLAGS'), OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')),
              open(base + '.json', 'w'), indent=1, default=str)
    return base


if __name__ == '__main__':
    date, outdir = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else '1'
    print(run(date, outdir, tag))
