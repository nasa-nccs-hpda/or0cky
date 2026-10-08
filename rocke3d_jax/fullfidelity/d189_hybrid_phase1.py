"""D189 step-0 timing in the D186/D187 hybrid: phase 1 (MELT_SI, DYNAM, CONDSE, RADIA apply) of step 0 of a date, run twice in one process:
  OLD  jax_p1_condse.CondseDevice.mstcnv  (host cloud-base loop, D183 libimf_ops callbacks, as in D187)
  NEW  the same step with CondseDevice.mstcnv replaced (on the instance only; no file edited) by clouds_mstcnv_dev's device loop with fused libimf callbacks
Everything else is identical (D187 libimf wiring of dynamics, LSCOND, snow age, poles; replayed radiation).  For each variant: run 1 cold, run 2 steady, run 3 steady
with a device block after every stage (stage shares).  Reports phase-1 seconds, condse_mstcnv seconds, jit executions, libimf_ops (D183) and fused callbacks (counts,
bytes, seconds), QUS callbacks, and compares the whole phase-1 result (stage snapshots dyn / condse / radia, X) OLD vs NEW bitwise.
Usage: taskset -c 0-2 env OMP_NUM_THREADS=1 python d189_hybrid_phase1.py DATE"""
import clouds_jax_env  # noqa: F401
import json
import sys
import time
import types

import numpy as np

import d187_imf_p1 as W
INST = W.install()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_atm_phase1 as P1  # noqa: E402
import jax_p1_count as CNT  # noqa: E402
import libimf_ops as LI  # noqa: E402
import libimf_fused as FX  # noqa: E402
import clouds_mstcnv_jax as mj  # noqa: E402
import clouds_mstcnv_dev as md  # noqa: E402
import atm_step as A  # noqa: E402
import d189_common as C  # noqa: E402

date = sys.argv[1]
it0 = dict(A.DATES)[date]
FX.set_mode('libimf')
F = np.float64


def mstcnv_new(self, R, stats=None):
    """CondseDevice.mstcnv with the cloud-base loop on the device (clouds_mstcnv_dev); the 'negative cloud' flag stays on the device (returned, as before)."""
    c = self.cfg['tune']
    lmcm = int(self.cfg['lmcm'])
    K = mj.make_K(c)
    for n_ in ("xmass", "bydtsrc", "dtsrc", "bybr"):
        K[n_] = np.float64(self.cfg[n_])
    K['fmpscale'] = np.float64(min(1.0, F(self.cfg['dtsrc']) / (F(1.0) * mj.mc.SECONDS_PER_HOUR)))
    Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
    S, Sc, I_, aux = self._mc_setup(Kj, R, lmcm)
    S, Sc, err, nev = md._events_device(Kj, S, Sc, I_, aux, lmcm, md.BUCKETS)
    if stats is not None:
        stats['events_per_lmin_device'] = nev
        stats['lmin_host_syncs'] = 0
    o = md._post_dev(Kj, S, Sc, I_, aux)
    return o, err


ph = W.make_phase1(date, rad='replay')
orig_mstcnv = ph.cs.mstcnv
res = {}
snap = {}


def one(variant, label, timed):
    rec = ph.load_records(it0, first=True)
    dev = ph.to_device(rec, first=True)
    jax.block_until_ready(dev)
    hold = ph.new_hold(dev['rad'])
    LI.reset_counters(); FX.reset_counters(); md.reset_qus()
    ph.qus.update(calls=0, seconds=0.0, bytes=0)
    a = CNT.snapshot()
    t0 = time.perf_counter()
    r = ph.step(it0, dev, dev['S'], {}, P1.CS.ms_zero(), hold, dev['ice'], timed=timed, rec=rec)
    jax.block_until_ready((r['S'], r['X'], r['flags']))
    fl = P1.D.flags_to_host(r['flags'])
    dt = time.perf_counter() - t0
    b = CNT.snapshot()
    li, fx = LI.counters(), FX.counters()
    out = dict(variant=variant, run=label, phase1_s=dt, jit_executions=b['jit_calls'] - a['jit_calls'],
               libimf_ops=li['total'], fused=fx['total'], fused_tags={k: v['calls'] for k, v in fx['tags'].items()},
               qus_old=dict(ph.qus), qus_new=dict(md.QUS), stage_seconds=r['info']['stage_seconds'], flags={k: int(v) for k, v in fl.items()},
               stage_jit=r['info']['stage_jit_calls'])
    print(json.dumps({k: v for k, v in out.items() if k not in ('stage_jit', 'fused_tags')}, default=float), flush=True)
    res[f'{variant}_{label}'] = out
    return r


def host_arrays(r):
    d = {}
    for grp in ('dyn', 'condse', 'radia'):
        for k, v in r['snaps'][grp].items():
            if hasattr(v, 'shape'):
                d[f'{grp}.{k}'] = np.asarray(v)
    for k, v in r['X'].items():
        if hasattr(v, 'shape'):
            d[f'X.{k}'] = np.asarray(v)
    return d


import os  # noqa: E402
VARIANTS = tuple(os.environ.get('D189_VARIANTS', 'old,new').split(','))
for variant in VARIANTS:
    ph.cs.mstcnv = orig_mstcnv if variant == 'old' else types.MethodType(mstcnv_new, ph.cs)
    if variant == 'fast':
        # libimf_ops' host loops (still used by dynamics, LSCOND, glue, snow age) switched to the C-shim loop of libimf_fused: same scalar libimf functions
        LI.host_exp, LI.host_pow = FX.host_exp, FX.host_pow
    for label, timed in (('cold', False), ('steady', False), ('timed', True)):
        r = one(variant, label, timed)
        if label == 'steady':
            snap[variant] = host_arrays(r)
ref_v = 'old' if 'old' in snap else VARIANTS[0]
for v_ in VARIANTS:
    if v_ != ref_v:
        nf, neq, bad = C.compare_dicts(snap[v_], snap[ref_v])
        print(f'PHASE-1 RESULT {v_} vs {ref_v}: {nf} arrays compared, {neq} bitwise equal, {len(bad)} unequal')
nf, neq, bad = C.compare_dicts(snap[VARIANTS[-1]], snap[ref_v])
print(f'PHASE-1 RESULT new vs old: {nf} arrays compared, {neq} bitwise equal, {len(bad)} unequal')
for x in bad[:30]:
    print('   UNEQUAL', x)
res['compare_new_vs_old'] = dict(compared=nf, equal=neq, unequal=bad)
with open(C.SCR + f'/hybrid_phase1_{date}' + os.environ.get('D189_TAG', '') + '.json', 'w') as f:
    json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
