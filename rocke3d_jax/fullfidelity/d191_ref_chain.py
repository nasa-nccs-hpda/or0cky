"""D191 C1 reference for the chained steps: the NumPy libimf chain with the closed surface (surface_loop_v2.Loop2), steps 0..N-1 of one date, each step from OUR
previous end state (atmosphere, hidden state, CONDSE carry, LSCOND module arrays, ocean, ice, lake, land ice, land GHY state, DYNSI/ADVSI carry).  Per step the same
snapshot groups as d187_ref_numpy.py (dyn, condse, radia, surface, dissip, filter, X, surf, land) are saved to OUTDIR/<date>_chain_ref_step<k>.npz.  MELT_SI is the
NumPy surface_loop.melt_si (Loop.pre_cse; our RSI replaces the recorded CONDSE-entry RSI); radiation is the recorded SRHR/TRHR/COSZ1 of each step (as D186/D187).
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 python d191_ref_chain.py DATE OUTDIR NSTEPS"""
import clouds_jax_env  # noqa: F401
import json
import os
import sys
import time

import numpy as np

import atm_step as A
import atm_step_fast as F
import surface_loop as L
import surface_loop_v2 as V2
import jax_harness as H
import d187_common as K


def main(date, outdir, n):
    os.makedirs(outdir, exist_ok=True)
    assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
    it0 = dict(A.DATES)[date]
    hdr = H.provenance_header(extra=dict(task='D191 NumPy libimf chain reference (closed surface), steps 0..%d' % (n - 1), date=date), require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    ctx = A.make_ctx(date, imf=True)
    F.ensure_backend(ctx)
    st = L.load_statics(date)
    st['ctx'] = L.make_ocean_ctx(date)
    SS = L.init_surface_state(date, st=st)
    loop = V2.Loop2(date, date, L.FF, st, SS, flags=None, it0=it0)
    orig_surface = A.stage_surface
    A.stage_surface = loop.stage_surface
    S, ms, out = None, {}, []
    try:
        for k in range(n):
            it = it0 + k
            R = A.Real(date, it)
            loop.pre_cse(R)
            tm = {}
            t0 = time.perf_counter()
            with F.fast_condse():
                S, sn = A.run_step(date, it, ctx, R=R, S=S, ms=ms, land_mode='ghy', timing=tm)
            wall = time.perf_counter() - t0
            arrs = {}
            for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter'):
                for kk, v in sn[stg].items():
                    if isinstance(v, np.ndarray) and v.dtype.kind in 'fiub':
                        arrs[f'{stg}/{kk}'] = np.asarray(v)
            for kk, v in S['_condse_X'].items():
                if isinstance(v, np.ndarray):
                    arrs[f'X/{kk}'] = np.asarray(v)
            arrs.update(K.surface_snapshot(loop))
            np.savez(os.path.join(outdir, f'{date}_chain_ref_step{k}.npz'), **arrs)
            X = S.get('_condse_X')
            carry = {key: np.array(X[key], copy=True) for key in A.CARRY_KEYS if X is not None and key in X}
            if '_cloud_rad' in S:
                carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
            S = {key: v for key, v in S.items() if not key.startswith('_')}
            if carry:
                S['_carry'] = carry
            out.append(dict(k=k, itime=it, wall=wall, stage_seconds={a: b for a, b in tm.items() if a.startswith('stage_')}))
            print('ref step', k, it, 'wall %.1f s' % wall, flush=True)
            json.dump(dict(date=date, n=n, header=hdr, steps=out, taskset=sorted(os.sched_getaffinity(0))), open(os.path.join(outdir, f'{date}_chain_ref.json'), 'w'),
                      indent=1, default=float)
    finally:
        A.stage_surface = orig_surface
    print('done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
