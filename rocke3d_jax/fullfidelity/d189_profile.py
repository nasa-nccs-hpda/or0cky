"""D189 part 1: PROFILE of the existing libimf MSTCNV (clouds_mstcnv_jax with D183 install): per-site callback table.

Runs the existing implementation (D183 `install('libimf')`) on the real step-0 CONDSE inputs of a date with the MSTCNV only (NumPy LSCOND), after
rebinding libimf_ops.pow / exp to site-tagged wrappers (same callbacks, same values; the wrapper only records the calling source line at trace time and, when
the callback runs on the host, calls, masked elements, bytes and seconds per site).  Also counts the host syncs of the cloud-base loop (np.asarray(mask),
bool(err)) and the QUS callbacks, saves the captured MSTCNV arguments and the baseline outputs to the scratchpad for the later bitwise comparison.
Usage: taskset -c 0-2 env OMP_NUM_THREADS=1 python d189_profile.py nov26"""
import clouds_jax_env  # noqa: F401
import collections
import functools
import inspect
import sys
import threading
import time

import numpy as np

import d189_common as C

SITES = collections.OrderedDict()
_lk = threading.Lock()


def _site():
    st = inspect.stack(0)
    out = []
    for f in st[2:]:
        fn = f.filename.split('/')[-1]
        if fn in ('libimf_ops.py', 'd189_profile.py'):
            continue
        out.append(f'{fn}:{f.function}:{f.lineno}')
        if len(out) == 2:
            break
    return ' <- '.join(out)


def install_tagging():
    import jax
    import jax.numpy as jnp
    import libimf_ops as L

    def mk(kind, orig_cb):
        def wrapper(*args, mask=None):
            site = kind + ' ' + _site()
            SITES.setdefault(site, dict(calls=0, elements=0, bytes_in=0, seconds=0.0))
            x = jnp.asarray(args[0], jnp.float64)
            if kind == 'exp':
                shp = x.shape
                y = None
            else:
                y = jnp.asarray(args[1], jnp.float64)
                shp = jnp.broadcast_shapes(x.shape, y.shape)
            m = jnp.ones(shp, bool) if mask is None else jnp.broadcast_to(jnp.asarray(mask, bool), shp)

            def cb(*ops, _s=site):
                t0 = time.perf_counter()
                r = orig_cb(*ops)
                dt = time.perf_counter() - t0
                with _lk:
                    d = SITES[_s]
                    d['calls'] += 1
                    d['elements'] += int(np.count_nonzero(ops[-1]))
                    d['bytes_in'] += sum(np.asarray(o).nbytes for o in ops)
                    d['seconds'] += dt
                return r
            L._state['traced'] = True
            if kind == 'exp':
                return jax.pure_callback(cb, jax.ShapeDtypeStruct(shp, jnp.float64), jnp.broadcast_to(x, shp), m, vmap_method='broadcast_all')
            return jax.pure_callback(cb, jax.ShapeDtypeStruct(shp, jnp.float64), jnp.broadcast_to(x, shp), jnp.broadcast_to(y, shp), m, vmap_method='broadcast_all')
        return wrapper

    L.exp = lambda x, mask=None, _w=mk('exp', L._cb_exp): _w(x, mask=mask)
    L.pow = lambda x, y, mask=None, _w=mk('pow', L._cb_pow): _w(x, y, mask=mask)


def main(date):
    import jax
    import jax_atm_step_imf as I
    import libimf_ops as L
    import clouds_mstcnv_jax as mj
    import clouds_condse_batch as cb
    import clouds_condse_jax as ccj
    it = C.DATES[date]
    I.install('libimf')
    install_tagging()
    ctx, inp = C.step0_inputs(date, it)
    syncs = collections.Counter()
    orig_asarray = np.asarray

    qus = dict(calls=0, bytes=0, seconds=0.0)
    orig_host_adv = mj._host_adv

    def host_adv(*a, **k):
        t0 = time.perf_counter()
        r = orig_host_adv(*a, **k)
        qus['calls'] += 1
        qus['bytes'] += sum(np.asarray(x).nbytes for x in a if hasattr(x, 'shape')) + sum(x.nbytes for x in r)
        qus['seconds'] += time.perf_counter() - t0
        return r
    mj._host_adv = host_adv
    # host syncs of the cloud-base loop: count np.asarray(mask) and bool(err) by wrapping the module-level jitted callees
    n_events = {}
    orig_base, orig_event = mj._base, mj._event_call
    cnt = collections.Counter()

    def base(*a):
        cnt['_base jit executions'] += 1
        cnt['host syncs (np.asarray(mask))'] += 1
        return orig_base(*a)

    def event(*a):
        cnt['_event_call jit executions'] += 1
        cnt['host syncs (bool(err))'] += 1
        return orig_event(*a)
    mj._base, mj._event_call = base, event
    rec = C.Recorder(mj)
    L.reset_counters()
    stats = {}
    with rec:
        t0 = time.perf_counter()
        Xj, cj = ccj.condse_step_jax(inp, ctx.cfg, ms={}, ls_mode='imf', use_ls=False, use_mc=True)
        wall = time.perf_counter() - t0
    print(f'{date}: condse (MSTCNV JAX libimf, NumPy LSCOND) cold wall {wall:.1f} s')
    # steady: second run of the stand-alone MSTCNV on the captured arguments (compile caches warm)
    R, c = rec.calls[0]
    C.dump(dict(R=R, c=c), f'cap_{date}.pkl')
    L.reset_counters()
    for d in SITES.values():
        d.update(calls=0, elements=0, bytes_in=0, seconds=0.0)
    qus.update(calls=0, bytes=0, seconds=0.0)
    cnt.clear()
    st = {}
    t0 = time.perf_counter()
    o = mj.mstcnv_jax(R, c, stats=st)
    steady = time.perf_counter() - t0
    tot = L.counters()
    print(f'steady stand-alone MSTCNV {steady:.2f} s; events {st.get("events")}; buckets {sorted(set(st.get("buckets", [])))}')
    print('libimf total', {k: tot['total'][k] for k in tot['total']}, 'exp calls', tot['exp']['calls'], 'pow calls', tot['pow']['calls'])
    print('QUS', qus)
    print('jit/syncs', dict(cnt))
    rows = sorted(SITES.items(), key=lambda kv: -kv[1]['seconds'])
    print(f'{"site":110s} {"calls":>7s} {"elements":>10s} {"MB in":>8s} {"s":>7s}')
    for s, d in rows[:45]:
        print(f'{s:110s} {d["calls"]:7d} {d["elements"]:10d} {d["bytes_in"]/1e6:8.1f} {d["seconds"]:7.2f}')
    print('n sites', len(rows), 'sum calls', sum(d['calls'] for d in SITES.values()))
    C.dump(dict(R=R, c=c, out=o, X=Xj, inp=None, steady=steady, libimf=tot, qus=dict(qus), cnt=dict(cnt), stats=st, wall_cold=wall,
                sites=dict(SITES)), f'base_{date}.pkl')


if __name__ == '__main__':
    main(sys.argv[1])
