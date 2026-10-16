"""D183 part 3: stage-level comparison of the JAX stages (plain = XLA pow/exp, and imf = libimf host callbacks of jax_imf.py) with the
libimf-mode NumPy chain, on identical inputs, real nov26 step 0.  NOT a C2 claim: only stage agreement with the libimf-mode NumPy reference
(which itself is validated against the real dumps in D127-D129).

  python jax_imf_stage_compare.py dyn     OUT.json   # AFLUX/ADVECM(+MAtoP) and PGF stages, 5 passes, inputs = the NumPy-imf chain workspace
  python jax_imf_stage_compare.py pek     OUT.json   # PEK = PEDN**KAPA on the end-of-dynamics PEDN
  python jax_imf_stage_compare.py lscond  OUT.json   # LSCOND batch kernel
  python jax_imf_stage_compare.py mstcnv  OUT.json   # MSTCNV batch kernel
  python jax_imf_stage_compare.py condse  OUT.json   # whole CONDSE (LSCOND + MSTCNV in JAX, rest NumPy) vs NumPy-imf and vs the real exit state
Run pinned (taskset -c 5-7, OMP_NUM_THREADS=1), own process each (XLA flags before jax; jit caches trace once per process).
Per variant the output lists, per written field: n elements, n unequal, max |diff|, max ulp; and the callback METER (calls, MiB, host s).
"""
import clouds_jax_env  # noqa: F401
import copy
import json
import sys
import time

import numpy as np

import jax_imf as JI
from imf_site_audit import ulp_diff


def fstat(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind != 'f':
        return dict(n=int(a.size), n_diff=int((a != b).sum()), max_abs=float(np.abs(a.astype(float) - b.astype(float)).max()) if a.size else 0.0, max_ulp=0)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    nn = np.isnan(a) & np.isnan(b)
    d = np.where(nn, 0.0, np.abs(a - b))
    u = ulp_diff(a, b)
    ne = (a != b) & ~nn
    return dict(n=int(a.size), n_diff=int(ne.sum()), max_abs=float(np.nanmax(d)) if a.size else 0.0, max_ulp=int(u.max()) if a.size else 0,
                bitwise=bool(not ne.any()))


def merge(agg, key, st):
    if key not in agg:
        agg[key] = dict(st, calls=1)
        return
    o = agg[key]
    o['n'] += st['n']; o['n_diff'] += st['n_diff']; o['calls'] += 1
    o['max_abs'] = max(o['max_abs'], st['max_abs']); o['max_ulp'] = max(o['max_ulp'], st['max_ulp'])


# ------------------------------------------------------------------------------------------------ dynamics
def run_dyn(date='nov26'):
    import dyn_step as ds
    import dyn_step_jax2 as dj2
    it0 = dict(ds.DATES)[date]
    ctx = ds.load_ctx(date, imf_pow=True)                        # reference chain: libimf pow everywhere (the real build's)
    kinds = ('aflux', 'advecm', 'pgf')
    kit_plain = dj2.Kit(ctx)
    with JI.patch_dyn():
        kit_imf = dj2.Kit(ctx)
    res = dict(plain={}, imf={}, kinds=kinds, timing={})
    snap = {}

    def hook(when, st, w, c):
        if st.kind not in kinds:
            return
        if when == 'pre':
            w1 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            t0 = time.perf_counter(); dj2.exec_stage_jax(st, w1, c, kit_plain, kinds); tp = time.perf_counter() - t0
            w2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            JI.METER.reset()
            t0 = time.perf_counter()
            with JI.patch_dyn():
                dj2.exec_stage_jax(st, w2, c, kit_imf, kinds)
            ti = time.perf_counter() - t0
            snap.update(w1=w1, w2=w2, pre={k: v for k, v in w.items() if isinstance(v, np.ndarray)}, meter=JI.METER.snapshot(), tp=tp, ti=ti)
        else:
            tm = res['timing'].setdefault(st.kind, dict(plain=[], imf=[], meter=[]))
            tm['plain'].append(snap['tp']); tm['imf'].append(snap['ti']); tm['meter'].append(snap['meter'])
            for tag, key in (('plain', 'w1'), ('imf', 'w2')):
                for k, v in w.items():
                    if isinstance(v, np.ndarray) and k in snap[key] and isinstance(snap[key][k], np.ndarray):
                        a, b = np.asarray(snap[key][k]), np.asarray(v)
                        # only the variables this stage wrote (changed against the pre-stage value, in either chain)
                        pre = snap['pre'].get(k)
                        changed = pre is None or not np.array_equal(pre, b, equal_nan=True) or not np.array_equal(pre, a, equal_nan=True)
                        if changed and a.shape == b.shape:
                            merge(res[tag], f"{st.kind}.{k}", fstat(a, b))

    s1 = ds.load_state(ds.state_path(date, it0, 1))
    t0 = time.perf_counter()
    ds.dyn_step(s1, ctx, itime=it0, hook=hook)
    res['wall_total_incl_compile'] = time.perf_counter() - t0
    return res


def run_pek(date='nov26'):
    import numpy as _np
    import dyn_step as ds
    import jax.numpy as jnp
    import jax_atm_step as J
    import intel_libm_ff as il
    it0 = dict(ds.DATES)[date]
    ctx = ds.load_ctx(date, imf_pow=True)
    s1 = ds.load_state(ds.state_path(date, it0, 1))
    w = ds.dyn_step(s1, ctx, itime=it0)
    pedn = _np.asarray(w['PEDN'], float)
    kapa = float(ctx.gg['kapa'])
    ref = il.pow_imf(pedn, kapa)
    plain = _np.array(J._pek_jax(jnp.asarray(pedn), kapa))
    JI.METER.reset()
    t0 = time.perf_counter()
    imf = _np.array(JI.pek(pedn, kapa))
    imf2 = _np.array(JI.pek(pedn, kapa))                    # second call = no compile
    t1 = time.perf_counter() - t0
    return dict(plain=fstat(plain, ref), imf=fstat(imf, ref), meter=JI.METER.snapshot(), seconds_two_calls_incl_compile=t1)


# ------------------------------------------------------------------------------------------------ CONDSE
def _capture(date='nov26'):
    import clouds_condse_batch as cb
    import clouds_condse_ff as cf
    import clouds_condse_io as cio
    it0 = dict(cio.DATES)[date]
    cf.set_backend('imf')
    cfg = cf.make_cfg(date)
    inp, ref = cio.load_step(date, it0)
    store = {}
    o_ls, o_mc = cb.lb.lscond_batch, cb.mb.mstcnv_batch

    def ls(S, P):
        store['ls_in'] = copy.deepcopy((S, P))
        S2, W = o_ls(S, P)
        store['ls_out'] = copy.deepcopy((S2, W))
        return S2, W

    def mcw(R, tune):
        store['mc_in'] = copy.deepcopy((R, tune))
        o = o_mc(R, tune)
        store['mc_out'] = copy.deepcopy(o)
        return o
    cb.lb.lscond_batch, cb.mb.mstcnv_batch = ls, mcw
    try:
        t0 = time.perf_counter()
        X, cnt = cb.condse_step_batch(inp, cfg, ms={})
        store['numpy_seconds'] = time.perf_counter() - t0
    finally:
        cb.lb.lscond_batch, cb.mb.mstcnv_batch = o_ls, o_mc
    store.update(inp=inp, cfg=cfg, X=X, ref=ref)
    return store


def _cmp_dict(out, want, skip=()):
    r = {}
    for k, v in want.items():
        if k in out and k not in skip:
            try:
                r[k] = fstat(np.asarray(out[k]), np.asarray(v))
            except Exception as e:      # non-array entries
                r[k] = dict(error=str(e))
    return r


def _summ(d):
    arrs = {k: v for k, v in d.items() if 'n_diff' in v}
    return dict(fields=len(arrs), fields_unequal=sorted(k for k, v in arrs.items() if v['n_diff']),
                total_elements_unequal=int(sum(v['n_diff'] for v in arrs.values())), max_ulp=int(max([v['max_ulp'] for v in arrs.values()] or [0])))


def run_lscond(date='nov26'):
    import clouds_lscond_jax as lj
    c = _capture(date)
    S, P = c['ls_in']
    So, Wo = c['ls_out']
    res = dict(numpy_imf_seconds=c['numpy_seconds'])
    for tag, mode in (('plain_xla', 'xla'),):
        t0 = time.perf_counter()
        S2, W2 = lj.lscond_jax(copy.deepcopy(S), P, mode=mode)
        res[tag] = dict(S=_cmp_dict(S2, So), W=_cmp_dict(W2, Wo), seconds_incl_compile=time.perf_counter() - t0)
        res[tag]['summary'] = _summ({**res[tag]['S'], **{'W.' + k: v for k, v in res[tag]['W'].items()}})
    JI.METER.reset()
    with JI.patch_lscond_meter():
        t0 = time.perf_counter()
        S3, W3 = lj.lscond_jax(copy.deepcopy(S), P, mode='libm')            # callbacks -> sz.ex/pw -> libimf (backend 'imf' is set by _capture)
        t1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        lj.lscond_jax(copy.deepcopy(S), P, mode='libm')
        t2 = time.perf_counter() - t0
    res['imf_callback'] = dict(S=_cmp_dict(S3, So), W=_cmp_dict(W3, Wo), seconds_first_incl_compile=t1, seconds_second=t2, meter=JI.METER.snapshot())
    res['imf_callback']['summary'] = _summ({**res['imf_callback']['S'], **{'W.' + k: v for k, v in res['imf_callback']['W'].items()}})
    return res


def run_mstcnv(date='nov26'):
    import jax
    import clouds_mstcnv_jax as mj
    c = _capture(date)
    R, tune = c['mc_in']
    Mo = c['mc_out']
    res = dict(numpy_imf_seconds=c['numpy_seconds'])
    JI.METER.reset()
    with JI.patch_adv_meter():
        t0 = time.perf_counter()
        o = mj.mstcnv_jax(copy.deepcopy(R), tune)
        res['plain_xla'] = dict(F=_cmp_dict(o, Mo), seconds_incl_compile=time.perf_counter() - t0, meter=JI.METER.snapshot())
    res['plain_xla']['summary'] = _summ(res['plain_xla']['F'])
    assert not any(k.startswith(('pow@', 'exp@')) for k in res['plain_xla']['meter'])
    jax.clear_caches()
    JI.METER.reset()
    with JI.patch_mstcnv(), JI.patch_adv_meter():
        t0 = time.perf_counter()
        o = mj.mstcnv_jax(copy.deepcopy(R), tune)
        t1 = time.perf_counter() - t0
        m1 = JI.METER.snapshot()
        JI.METER.reset()
        t0 = time.perf_counter()
        mj.mstcnv_jax(copy.deepcopy(R), tune)
        t2 = time.perf_counter() - t0
        m2 = JI.METER.snapshot()
    res['imf_callback'] = dict(F=_cmp_dict(o, Mo), seconds_first_incl_compile=t1, seconds_second=t2, meter_first=m1, meter_second=m2)
    res['imf_callback']['summary'] = _summ(res['imf_callback']['F'])
    assert any(k.startswith(('pow@', 'exp@')) for k in m1), 'patch not effective: no libimf callback was called'
    return res


def run_condse(date='nov26'):
    import jax
    import clouds_condse_compare as cc
    import clouds_condse_jax as cj
    c = _capture(date)
    X = c['X']
    res = dict(numpy_imf_seconds=c['numpy_seconds'])

    def vs_real(Xj):
        out = {'exact': 0, 'bound': 0, 'FAIL': 0}
        for n in cc.ALL_FIELDS:
            if n in Xj and n in c['ref'] and n not in ('UALIJ', 'VALIJ'):
                out[cc.stat(Xj[n], c['ref'][n], None)['status']] += 1
        return out
    res['numpy_imf_vs_real_exit'] = vs_real(X)
    JI.METER.reset()
    with JI.condse_imf():
        t0 = time.perf_counter()
        Xj, _ = cj.condse_step_jax(c['inp'], c['cfg'], ms={}, ls_mode='libm')
        t1 = time.perf_counter() - t0
        m1 = JI.METER.snapshot()
        JI.METER.reset()
        t0 = time.perf_counter()
        Xj2, _ = cj.condse_step_jax(c['inp'], c['cfg'], ms={}, ls_mode='libm')
        t2 = time.perf_counter() - t0
        m2 = JI.METER.snapshot()
    res['imf_callback'] = dict(fields=_cmp_dict(Xj, X), seconds_first_incl_compile=t1, seconds_second=t2, meter_first=m1, meter_second=m2,
                               vs_real_exit=vs_real(Xj))
    res['imf_callback']['summary'] = _summ(res['imf_callback']['fields'])
    return res


def main(argv):
    mode, out = argv[0], argv[1]
    if not JI.available():
        print('libimf not available'); return 2
    r = dict(dyn=run_dyn, pek=run_pek, lscond=run_lscond, mstcnv=run_mstcnv, condse=run_condse)[mode]()
    json.dump(r, open(out, 'w'), indent=1, default=float)
    print('written', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
