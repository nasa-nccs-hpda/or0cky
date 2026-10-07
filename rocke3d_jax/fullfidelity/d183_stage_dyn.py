"""D183 stage-level check of the dynamics libimf callbacks: NumPy dyn_step (imf_pow=True) with a hook that runs the JAX 'pgf' and 'advecm' stage
(libimf_ops, installed by jax_atm_step_imf.install) on the same pre-stage workspace and compares the stage outputs bitwise with the NumPy post-stage
workspace.  Usage (fresh process, flags first): taskset -c 5-6 env OMP_NUM_THREADS=1 python d183_stage_dyn.py [date=nov26] [itime=33312]"""
import clouds_jax_env  # noqa: F401
import sys
import time
import numpy as np
import dyn_step as ds
import dyn_step_jax2 as dj2
import jax_atm_step_imf as I
import libimf_ops as L

date = sys.argv[1] if len(sys.argv) > 1 else 'nov26'
it = int(sys.argv[2]) if len(sys.argv) > 2 else 33312
I.install('libimf')
ctx = ds.load_ctx(date, imf_pow=True)
kit = I.KitImf(ctx)
VARS = {'pgf': ('UT', 'VT', 'DUT', 'DVT', 'GZ', 'PHI', 'SPA'), 'advecm': ('PK', 'PMID', 'PEDN', 'PDSIG', 'P', 'MNEW', 'MSUM')}
res = {}


def hook(when, st, w, c):
    if st.kind not in VARS:
        return
    if when == 'pre':
        hook.w2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
        t0 = time.perf_counter()
        dj2.exec_stage_jax(st, hook.w2, c, kit, dj2.JAX_KINDS)
        hook.dt = time.perf_counter() - t0
    else:
        for v in VARS[st.kind]:
            for key in (v, v.lower()):
                if key in w and key in hook.w2 and isinstance(w[key], np.ndarray):
                    a, b = np.asarray(hook.w2[key]), np.asarray(w[key])
                    if a.shape == b.shape:
                        r = res.setdefault((st.kind, key), [0, 0, 0.0])
                        ne = a != b
                        r[0] += 1; r[1] += int(ne.sum()); r[2] = max(r[2], float(np.abs(a - b).max()))
                        break
        print(st.kind, 'jax stage s', round(hook.dt, 1), flush=True)


s1 = ds.load_state(ds.state_path(date, it, 1))
ds.dyn_step(s1, ctx, itime=it, hook=hook)
print('compared (stage kind, field): [stage calls, unequal elements, max abs diff]')
for k, v in sorted(res.items()):
    print(k, v)
print(L.counters())
