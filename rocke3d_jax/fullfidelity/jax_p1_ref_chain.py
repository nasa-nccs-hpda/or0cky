"""D186: NumPy libm-mode CHAIN reference for phase 1 over several steps (atm_step_fast.run_chain, land recorded, ctx imf=False): per step the stage snapshots
dyn / condse / radia, the CONDSE output X and the RADIA cloud masking, saved to OUTDIR/<date>_chain_step<k>.npz (keys as jax_p1_ref.py).
Run: taskset -c 0-1 env OMP_NUM_THREADS=1 python jax_p1_ref_chain.py DATE OUTDIR NSTEPS"""
import clouds_jax_env  # noqa: F401  (same flags as the D184 reference)
import json
import os
import sys
import time

import numpy as np

import atm_step as A
import atm_step_fast as F


def main(date, outdir, n):
    os.makedirs(outdir, exist_ok=True)
    it0 = dict(A.DATES)[date]
    ctx = A.make_ctx(date, imf=False)

    def cb(k, it, R, sn, S, tm):
        d = {}
        for st in ('dyn', 'condse', 'radia'):
            for key, v in sn[st].items():
                if isinstance(v, np.ndarray) and v.dtype.kind in 'fiu':
                    d[f'{st}/{key}'] = np.asarray(v)
        for key, v in S['_condse_X'].items():
            if isinstance(v, np.ndarray):
                d[f'X/{key}'] = np.asarray(v)
        if '_cloud_rad' in S:
            d['rad/CLDSS'], d['rad/CLDMC'] = (np.asarray(a) for a in S['_cloud_rad'])
        np.savez(os.path.join(outdir, f'{date}_chain_step{k}.npz'), **d)
        print('saved step', k, it, flush=True)
    t0 = time.perf_counter()
    out = F.run_chain(date, it0, n, ctx, land_mode='recorded', on_step=cb)
    json.dump(dict(date=date, n=n, wall=time.perf_counter() - t0, steps=out, taskset=repr(os.sched_getaffinity(0))), open(os.path.join(outdir, f'{date}_chain.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
