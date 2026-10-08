"""D189 CONDSE-level check on the real step-0 inputs of a date: the full CONDSE (column set-up, LSCOND, MSTCNV, post-processing, poles) with
  (N) NumPy batch, libimf backend          (the reference of D183: 68/68)
  (O) existing JAX path: clouds_mstcnv_jax host loop + D183 libimf_ops callbacks, LSCOND mode 'imf'
  (D) NEW: clouds_mstcnv_dev (device loop, fused callbacks), LSCOND mode 'imf' (unchanged)
compares every exit field array bitwise (D vs O, D vs N, O vs N) and reports wall time (cold, steady), libimf_ops counters (LSCOND, and MSTCNV for O),
fused counters + QUS for D.  Also a mutation: one MSTCNV constant of D perturbed -> the comparison must fail.
Usage: taskset -c 0-2 env OMP_NUM_THREADS=1 python d189_condse_check.py nov26"""
import clouds_jax_env  # noqa: F401
import copy
import json
import sys
import time

import numpy as np

import d189_common as C

date = sys.argv[1]
it = C.DATES[date]
import atm_step as A  # noqa: E402
import clouds_condse_batch as cb  # noqa: E402
import clouds_condse_jax as ccj  # noqa: E402
import jax_atm_step_imf as I  # noqa: E402
import libimf_ops as L  # noqa: E402
import libimf_fused as FX  # noqa: E402
import clouds_mstcnv_dev as md  # noqa: E402

I.install('libimf')
FX.set_mode('libimf')
print('fused', FX.status(), flush=True)
ctx, inp = C.step0_inputs(date, it)
out = dict(date=date, itime=it)


def timed(f):
    t0 = time.perf_counter()
    r = f()
    return r, time.perf_counter() - t0


(Xn, cn), tn = timed(lambda: cb.condse_step_batch(copy.deepcopy(inp), ctx.cfg, ms={}))
print(f'N numpy imf batch: {tn:.1f} s', flush=True)
out['numpy_s'] = tn

orig_mc = ccj.mj.mstcnv_jax
cap = {}


def wrap(name, fn):
    def w(R, c, *a, **k):
        o = fn(R, c, *a, **k)
        cap[name] = o
        return o
    return w


ccj.mj.mstcnv_jax = wrap('old', orig_mc)
# ---- O: existing
L.reset_counters()
(Xo, co), t_o_cold = timed(lambda: ccj.condse_step_jax(copy.deepcopy(inp), ctx.cfg, ms={}, ls_mode='imf'))
L.reset_counters()
(Xo, co), t_o = timed(lambda: ccj.condse_step_jax(copy.deepcopy(inp), ctx.cfg, ms={}, ls_mode='imf'))
co_cnt = L.counters()
print(f'O existing JAX path: cold {t_o_cold:.1f} s steady {t_o:.1f} s; libimf_ops total {co_cnt["total"]}', flush=True)
out.update(old_cold_s=t_o_cold, old_steady_s=t_o, old_libimf_ops=co_cnt)

# ---- D: new
ccj.mj.mstcnv_jax = wrap('new', md.mstcnv_dev)
try:
    L.reset_counters(); FX.reset_counters(); md.reset_qus()
    (Xd, cd), t_d_cold = timed(lambda: ccj.condse_step_jax(copy.deepcopy(inp), ctx.cfg, ms={}, ls_mode='imf'))
    L.reset_counters(); FX.reset_counters(); md.reset_qus()
    (Xd, cd), t_d = timed(lambda: ccj.condse_step_jax(copy.deepcopy(inp), ctx.cfg, ms={}, ls_mode='imf'))
    cd_l, cd_f, cd_q = L.counters(), FX.counters(), dict(md.QUS)
    print(f'D device MSTCNV: cold {t_d_cold:.1f} s steady {t_d:.1f} s; LSCOND libimf_ops {cd_l["total"]}; fused {cd_f["total"]}; QUS {cd_q}', flush=True)
    out.update(new_cold_s=t_d_cold, new_steady_s=t_d, new_libimf_ops_lscond=cd_l, new_fused=cd_f, new_qus=cd_q)
finally:
    ccj.mj.mstcnv_jax = orig_mc

nf, neq, bad = C.compare_dicts(cap['new'], cap['old'])
print(f'MSTCNV STAGE (new vs old, all outputs): {nf} arrays compared, {neq} bitwise equal, {len(bad)} unequal')
for x in bad[:20]:
    print('   UNEQUAL', x)
out['mstcnv_stage'] = dict(compared=nf, equal=neq, unequal=bad)

for nm, a, b in (('D vs O', Xd, Xo), ('D vs N', Xd, Xn), ('O vs N', Xo, Xn)):
    nf, neq, bad = C.compare_dicts(a, b)
    print(f'{nm}: {nf} exit-field arrays compared, {neq} bitwise equal, {len(bad)} unequal')
    for x in bad[:20]:
        print('   UNEQUAL', x)
    out[nm] = dict(compared=nf, equal=neq, unequal=bad)

# ---- mutation: perturb one MSTCNV tune constant of the device path (one relative 1e-9 on mc_fddrt), compare with O
orig_dev = md.mstcnv_dev


def mutated(R, c, *a, **k):
    c2 = dict(c)
    c2['mc_fddrt'] = c['mc_fddrt'] * (1 + 1e-9)
    return orig_dev(R, c2, *a, **k)


ccj.mj.mstcnv_jax = mutated
try:
    Xm, _ = ccj.condse_step_jax(copy.deepcopy(inp), ctx.cfg, ms={}, ls_mode='imf')
finally:
    ccj.mj.mstcnv_jax = orig_mc
nf, neq, bad = C.compare_dicts(Xm, Xo)
print(f'MUTATION (mc_fddrt * (1+1e-9)) vs O: {nf} compared, {neq} equal, {len(bad)} unequal -> detected = {len(bad) > 0}')
out['mutation'] = dict(compared=nf, equal=neq, unequal=len(bad), detected=len(bad) > 0)
with open(C.SCR + f'/condse_check_{date}.json', 'w') as f:
    json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
