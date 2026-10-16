"""D183 part 1: call-site audit of the Intel libimf scalar calls (pow, exp) of the libimf-mode NumPy chain on the real nov26 step-0 data.

Every libimf `pow` / `exp` call made by the libimf-mode NumPy dynamics (dyn_step.dyn_step with imf_pow=True) and by the libimf-mode
batched CONDSE (clouds_condse_batch.condse_step_batch with clouds_condse_ff.set_backend('imf')) is intercepted by wrapping the ctypes
function pointers (nothing in the repository is modified; the wrappers are installed at run time only).  Per call site (first stack
frame outside the thin bridge helpers) the arguments and the libimf result are kept, and afterwards compared with
  (a) glibc scalar libm (math.pow / math.exp, = what a Python float `**` gives),
  (b) numpy's vectorised np.power / np.exp,
  (c) XLA:CPU jnp.power / jnp.exp (what the JAX stages evaluate),
element by element, in ulps.

Usage (own process, pinned):  taskset -c 5-7 env OMP_NUM_THREADS=1 python imf_site_audit.py OUT.json [dyn|condse|both] [date]
Not a test of the model: an inventory of where libimf differs from the libm that the JAX stages use.
"""
import clouds_jax_env  # noqa: F401  (flags before jax; the audit evaluates jnp.exp/jnp.power under the same flags as the JAX stages)
import array
import json
import math
import os
import sys
import time

import numpy as np

import intel_libm_ff as il

_SKIP_FILES = {'intel_libm_ff.py', 'imf_site_audit.py'}
_SKIP = {('clouds_lscond_size_ff.py', 'ex'), ('clouds_lscond_size_ff.py', 'pw'), ('clouds_lscond_batch.py', 'exm'),
         ('clouds_lscond_batch.py', 'pwm'), ('clouds_mstcnv_ff.py', 'exp'), ('clouds_mstcnv_ff.py', 'power'),
         ('dyn_filter_ff.py', '_pow_arr'), ('dyn_glue_ff.py', '_scalar_pow'), ('dyn_pgf_ff.py', '_pw')}


class Rec:
    def __init__(self, fn):
        self.fn = fn
        self.x = array.array('d')
        self.y = array.array('d')
        self.r = array.array('d')
        self.callers = {}


SITES = {}


def _site():
    f = sys._getframe(2)
    while f is not None:
        base = os.path.basename(f.f_code.co_filename)
        nm = f.f_code.co_name
        if base in _SKIP_FILES or (base, nm) in _SKIP or nm in ('<listcomp>', '<genexpr>'):
            f = f.f_back
            continue
        break
    if f is None:
        return 'unknown', ''
    key = f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}"
    g = f.f_back
    while g is not None and (os.path.basename(g.f_code.co_filename) in _SKIP_FILES or g.f_code.co_name in ('<listcomp>', '<genexpr>')):
        g = g.f_back
    caller = f"{os.path.basename(g.f_code.co_filename)}:{g.f_code.co_name}" if g is not None else ''
    return key, caller


def _wrap(fn, impl):
    name = fn

    if name == 'pow':
        def w(x, y):
            r = impl(x, y)
            k, c = _site()
            s = SITES.setdefault((k, 'pow'), Rec('pow'))
            s.x.append(x); s.y.append(y); s.r.append(r)
            s.callers[c] = s.callers.get(c, 0) + 1
            return r
    else:
        def w(x):
            r = impl(x)
            k, c = _site()
            s = SITES.setdefault((k, 'exp'), Rec('exp'))
            s.x.append(x); s.r.append(r)
            s.callers[c] = s.callers.get(c, 0) + 1
            return r
    return w


def install():
    """Wrap every libimf pow/exp function pointer the repository uses.  Returns a list of what was wrapped."""
    done = []
    f = il._load()
    il._pow = _wrap('pow', f)
    done.append('intel_libm_ff._pow')
    import clouds_lscond_size_ff as sz
    lib = sz._load_imf()
    if lib:
        lib.pow = _wrap('pow', lib.pow)
        lib.exp = _wrap('exp', lib.exp)
        done.append('clouds_lscond_size_ff lib')
    import clouds_mstcnv_ff as mc
    lib2 = mc._imf()
    if lib2:
        lib2.pow = _wrap('pow', lib2.pow)
        lib2.exp = _wrap('exp', lib2.exp)
        done.append('clouds_mstcnv_ff lib')
    return done


def ulp_diff(a, b):
    """signed-magnitude-correct ulp distance of finite doubles (NaN/inf equal -> 0, otherwise a large marker)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    ia = np.where(ia < 0, np.int64(-2 ** 63) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2 ** 63) - ib, ib)
    d = np.abs(ia - ib)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return np.where(same, 0, d)


def _xla(fn, x, y):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    n = len(x)
    C = 1 << 16
    out = np.empty(n)
    if fn == 'pow':
        f = jax.jit(lambda a, b: jnp.power(a, b))
    else:
        f = jax.jit(lambda a, b: jnp.exp(a))
    for s in range(0, n, C):
        xa = np.ones(C); ya = np.ones(C)
        m = min(C, n - s)
        xa[:m] = x[s:s + m]
        ya[:m] = y[s:s + m] if y is not None else 1.0
        out[s:s + m] = np.asarray(f(jnp.asarray(xa), jnp.asarray(ya)))[:m]
    return out


def analyse(rec, fn):
    x = np.frombuffer(rec.x, float)
    r = np.frombuffer(rec.r, float)
    y = np.frombuffer(rec.y, float) if fn == 'pow' else None
    n = len(x)
    # distinct argument pairs (the numpy chain evaluates masked / padded elements repeatedly)
    key = np.stack([x, y], 1) if fn == 'pow' else x[:, None]
    uq, idx = np.unique(key, axis=0, return_index=True)
    xu, ru = x[idx], r[idx]
    yu = y[idx] if fn == 'pow' else None
    with np.errstate(all='ignore'):
        if fn == 'pow':
            g = np.empty(len(xu))
            for i, (a, b) in enumerate(zip(xu.tolist(), yu.tolist())):
                try:
                    g[i] = math.pow(a, b)
                except (ValueError, OverflowError, ZeroDivisionError):
                    g[i] = np.nan
            npv = np.power(xu, yu)
        else:
            g = np.empty(len(xu))
            for i, a in enumerate(xu.tolist()):
                try:
                    g[i] = math.exp(a)
                except OverflowError:
                    g[i] = np.inf
            npv = np.exp(xu)
    xl = _xla(fn, xu, yu)
    out = dict(fn=fn, calls=n, distinct_args=int(len(xu)), callers=rec.callers,
               x_min=float(np.nanmin(x)) if n else None, x_max=float(np.nanmax(x)) if n else None)
    if fn == 'pow':
        out['exponents'] = sorted({float(v) for v in np.unique(y)[:8]})
        out['n_exponents'] = int(len(np.unique(y)))
    for tag, v in (('glibc', g), ('numpy', npv), ('xla', xl)):
        d = ulp_diff(ru, v)
        out[tag] = dict(n_diff=int((d > 0).sum()), max_ulp=int(d.max()) if len(d) else 0, ulp1=int((d == 1).sum()), ulp_gt1=int((d > 1).sum()))
        # weight to calls: fraction of all calls whose argument pair differs
        bad = d > 0
        if bad.any():
            kb = {tuple(k) for k in uq[bad].tolist()}
            keyc = [tuple(k) for k in key.tolist()]
            out[tag]['n_calls_diff'] = int(sum(1 for k in keyc if k in kb))
        else:
            out[tag]['n_calls_diff'] = 0
    return out


def run_dyn(date='nov26'):
    import dyn_step as ds
    it0 = dict(ds.DATES)[date]
    ctx = ds.load_ctx(date, imf_pow=True)
    s1 = ds.load_state(ds.state_path(date, it0, 1))
    t0 = time.perf_counter()
    ds.dyn_step(s1, ctx, itime=it0)
    return time.perf_counter() - t0


def run_condse(date='nov26'):
    import atm_step as A
    import clouds_condse_batch as cb
    import clouds_condse_ff as cf
    import clouds_condse_io as cio
    it0 = dict(cio.DATES)[date]
    cf.set_backend('imf')
    cfg = cf.make_cfg(date)
    inp, ref = cio.load_step(date, it0)
    t0 = time.perf_counter()
    cb.condse_step_batch(inp, cfg, ms={})
    return time.perf_counter() - t0


def main(argv):
    out = argv[0]
    which = argv[1] if len(argv) > 1 else 'both'
    date = argv[2] if len(argv) > 2 else 'nov26'
    if not il.available():
        print('libimf not available'); return 2
    wrapped = install()
    res = dict(date=date, wrapped=wrapped, python=sys.version.split()[0], numpy=np.__version__)
    for stage, runner in (('dyn', run_dyn), ('condse', run_condse)):
        if which not in (stage, 'both'):
            continue
        SITES.clear()
        t = runner(date)
        sites = {}
        for (k, fn), rec in sorted(SITES.items()):
            print(stage, k, fn, len(rec.x), flush=True)
            sites[f"{k}|{fn}"] = analyse(rec, fn)
        res[stage] = dict(seconds_traced=t, sites=sites, total_calls=int(sum(len(r.x) for r in SITES.values())))
        # keep the raw arguments of each site (compressed) for the stage tests
        np.savez_compressed(out.replace('.json', f'_{stage}_args.npz'),
                            **{f"{k}|{fn}|{nm}": np.frombuffer(getattr(rec, nm), float) for (k, fn), rec in SITES.items()
                               for nm in (('x', 'y', 'r') if fn == 'pow' else ('x', 'r'))})
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print('written', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
