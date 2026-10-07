"""D183 stage-level check of the CONDSE libimf callbacks.  Real start state of step `itime`, NumPy dynamics (imf), then CONDSE twice from the same
entry arrays: NumPy batch (cf backend 'imf', clouds_condse_batch) vs JAX batch kernels with libimf_ops (LSCOND mode 'imf', MSTCNV rebound).
Variants: ls (JAX LSCOND only), mc (JAX MSTCNV only), both.  Reports per exit field bitwise equality.
Usage: taskset -c 5-6 env OMP_NUM_THREADS=1 python d183_stage_condse.py [date=nov26] [itime=33312] [variants=ls,mc,both]"""
import clouds_jax_env  # noqa: F401
import copy
import sys
import time
import numpy as np
import atm_step as A
import clouds_condse_batch as cb
import clouds_condse_jax as ccj
import jax_atm_step_imf as I
import libimf_ops as L

date = sys.argv[1] if len(sys.argv) > 1 else 'nov26'
it = int(sys.argv[2]) if len(sys.argv) > 2 else 33312
variants = (sys.argv[3] if len(sys.argv) > 3 else 'ls,mc,both').split(',')
I.install('libimf')
ctx = A.make_ctx(date, imf=True)
R = A.Real(date, it, A.FF)
S = A.init_state(R); A._native(S)
A.stage_dyn(S, R, ctx)
inp = A.condse_inputs(S, R)
t0 = time.perf_counter()
Xn, cn = cb.condse_step_batch(copy.deepcopy(inp), ctx.cfg, ms={})
print('numpy imf batch condse', round(time.perf_counter() - t0, 1), 's', flush=True)
for v in variants:
    L.reset_counters()
    t0 = time.perf_counter()
    Xj, cj = ccj.condse_step_jax(copy.deepcopy(inp), ctx.cfg, ms={}, ls_mode='imf', use_ls=v in ('ls', 'both'), use_mc=v in ('mc', 'both'))
    dt = time.perf_counter() - t0
    bad = []
    nf = 0
    for k in sorted(Xn):
        if k not in Xj or not isinstance(Xn[k], np.ndarray):
            continue
        a, b = np.asarray(Xj[k], float), np.asarray(Xn[k], float)
        if a.shape != b.shape:
            continue
        nf += 1
        ne = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if ne.any():
            bad.append((k, int(ne.sum()), float(np.abs(np.where(ne, a - b, 0)).max())))
    print(f'variant {v}: {nf} exit fields compared, {nf - len(bad)} bitwise equal, {len(bad)} unequal; wall {dt:.1f} s (incl. compile)')
    for b_ in bad[:30]:
        print('   unequal', b_)
    print('   counters', L.counters()['total'], {k: L.counters()[k]['calls'] for k in ('exp', 'pow')}, flush=True)
