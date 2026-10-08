"""D201 driver: d200_day.py (d193_day.main, nit_strict=False) with a SWITCH for the D199 qg_aver/elhx change (commit 93976f0) and a per-step capture of the
land (aux_land) arrays.  New file only; no tracked file edited.
  --fix 1 (default) = HEAD behaviour (elhx in the land result: qg_sat of the substep that just ran)
  --fix 0           = pre-D199 behaviour for the jitted path: the land result carries no `elhx` (next_land_pbl_columns uses qsat(tg, rec[:,19]) = next row's elhx)
  --cap DIR         = write DIR/land_<it>.npz (flattened aux_land + host ej/ei) every step
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d201_day.py OUT --fix 0|1 --cap DIR --nsteps N --c1dir DUMMY --nit-strict 0"""
import clouds_jax_env  # noqa: F401
import functools
import json
import os
import sys

import numpy as np

import jax_coupled as C
import jax_surface as JS
import d193_day as D
import d187_common as K

fix = 1
if '--fix' in sys.argv:
    k = sys.argv.index('--fix'); fix = int(sys.argv[k + 1]); del sys.argv[k:k + 2]
cap = None
if '--cap' in sys.argv:
    k = sys.argv.index('--cap'); cap = sys.argv[k + 1]; del sys.argv[k:k + 2]
strict = 0
if '--nit-strict' in sys.argv:
    k = sys.argv.index('--nit-strict'); strict = int(sys.argv[k + 1]); del sys.argv[k:k + 2]

if not fix:
    _orig_land = JS._land

    @functools.wraps(_orig_land)
    def _land_old(*a, **kw):
        r = _orig_land(*a, **kw)
        r = dict(r)
        r.pop('elhx')            # pre-D199: next_land_pbl_columns falls back to qsat(tg, rec[:, 19])
        return r
    JS._land = _land_old

_STEP = [0]


def _allarr(arrays):
    if cap:
        os.makedirs(cap, exist_ok=True)
        d = K.flatten(C.tree_np(arrays['aux_land']), 'land/')
        h = arrays['host']
        d = {k: np.asarray(v) for k, v in d.items()}
        d['host/ej'] = np.asarray(h['ej']); d['host/ei'] = np.asarray(h['ei'])
        d['filter/T'] = np.asarray(arrays['filter']['T'])[41, 20, :]
        np.savez(os.path.join(cap, f'land_{_STEP[0]}.npz'), **d)
    _STEP[0] += 1
    return {}


D.allarr_of = _allarr

if __name__ == '__main__':
    C.Coupled = functools.partial(C.Coupled, nit_strict=bool(strict))
    try:
        D.main()
    finally:
        import jax_surface as JS2
        json.dump(dict(fix=fix, strict=strict, rebuild_log=JS2.NIT_LOG, mismatch=JS2.NIT_MISMATCH), open(os.path.join(sys.argv[1], 'd201_nit.json'), 'w'), indent=1)
