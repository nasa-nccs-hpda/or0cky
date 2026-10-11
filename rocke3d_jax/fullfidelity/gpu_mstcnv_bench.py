"""Stage benchmark of the cloud-base stage (condse_mstcnv: D189 device MSTCNV) on its own, from captured real inputs (D214 optimisation harness).

  capture (CPU, once):   python -u gpu_mstcnv_bench.py capture OUT.pkl [DATE] [STEP]     runs the assembled step up to STEP (default 1) and saves the inputs
                                                                                          (R, cfg) of the first mstcnv call of that step
  bench (any device):    python -u gpu_mstcnv_bench.py bench IN.pkl [REPEATS]               compiles and times setup / events / post separately and together

The bench replicates jax_coupled.mstcnv_new piece by piece (same functions, same order), blocks after each piece and times it; the first run includes
compilation (a few minutes on the A100 instead of the 11 min of the whole step); the repeats are steady times.  It also prints the event counts per LMIN.
Diagnostic only: libm mode on a GPU (no libimf), radiation not involved.
"""
import pickle
import sys
import time

import numpy as np

if sys.argv[1:2] != ['capture']:
    import clouds_jax_env  # noqa: F401  (the exactness XLA flags must be set BEFORE jax is imported; the CPU results of earlier bench runs were compiled without them)

mode = sys.argv[1]


def _widths_bench(path, repeats=3):
    """Time ONE cloud-base event block (md._event_body at a fixed LMIN) at several widths nb: is the cost per sequential step (flat in nb) or per element?"""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    import clouds_mstcnv_jax as mj
    import clouds_mstcnv_dev as md
    from types import SimpleNamespace
    d = pickle.load(open(path, 'rb'))
    cfg = d['cfg']
    R = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) and np.asarray(a).dtype.kind in 'fiub' else a, d['R'])
    lmcm = int(cfg['lmcm'])
    K = mj.make_K(cfg['tune'])
    for n_ in ('xmass', 'bydtsrc', 'dtsrc', 'bybr'):
        K[n_] = np.float64(cfg[n_])
    K['fmpscale'] = np.float64(min(1.0, np.float64(cfg['dtsrc']) / (np.float64(1.0) * mj.mc.SECONDS_PER_HOUR)))
    Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
    S, Sc, I_, aux = jax.jit(lambda K, R, lmcm: mj._setup(SimpleNamespace(**K), R, lmcm), static_argnums=2)(Kj, R, lmcm)
    lmin = 3
    mask, fmp0, fmp2 = jax.jit(lambda K, S, I: md._base_dev(K, lmin, S, I))(Kj, S, I_)
    N = int(mask.shape[0])
    idx_all = np.flatnonzero(np.asarray(mask))
    print('backend', jax.default_backend(), 'LMIN', lmin, 'event columns available at this LMIN:', idx_all.size, 'of N =', N, flush=True)
    for nb in (32, 128, 512, 1024):
        cnt = min(nb, idx_all.size)
        idxg = np.zeros(nb, np.int64)
        idxg[:cnt] = idx_all[:cnt]
        idxs = np.full(nb, N, np.int64)
        idxs[:cnt] = idx_all[:cnt]
        valid = np.arange(nb) < cnt
        f = jax.jit(lambda K, S, Sc, I, aux, fmp0, fmp2, idxg, idxs, valid: md._event_body(K, lmin, fmp0, fmp2, idxg, idxs, valid, S, Sc, I, aux))
        args = (Kj, S, Sc, I_, aux, fmp0, fmp2, jnp.asarray(idxg), jnp.asarray(idxs), jnp.asarray(valid))
        ts = []
        for r in range(repeats + 1):
            t0 = time.perf_counter()
            out = f(*args)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        print(f'nb {nb:5d} (valid columns {cnt:4d}): first (compile) {ts[0]:8.2f} s   steady {np.mean(ts[1:]):7.3f} s  ({", ".join(f"{x:.3f}" for x in ts[1:])})', flush=True)
        if md.TRACE:
            md.TRACE_LOG.clear()
            t_start = time.perf_counter()
            out = f(*args)
            t_ret = time.perf_counter()
            jax.block_until_ready(out)
            t_end = time.perf_counter()
            log = list(md.TRACE_LOG)
            print(f'   call returned (dispatch) after {t_ret - t_start:.4f} s; result ready after {t_end - t_start:.4f} s; first marker {log[0][1] - t_start:.4f} s after the call start; last marker {t_end - log[-1][1]:.4f} s before the result was ready')
            prev = None
            print('   phase timeline of one steady call (seconds since the previous marker; ic = 1 and 2 passes are both in the log):')
            for tag, t in log:
                if prev is not None:
                    print(f'      {tag:44s} {t - prev:8.4f}')
                prev = t


def _phases_bench(path, nb=128, repeats=3):
    """Ablation timing of the event block: the block truncated after phase k (md.STOP), cumulative steady seconds and the delta per phase."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    import clouds_mstcnv_jax as mj
    import clouds_mstcnv_dev as md
    from types import SimpleNamespace
    d = pickle.load(open(path, 'rb'))
    cfg = d['cfg']
    R = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) and np.asarray(a).dtype.kind in 'fiub' else a, d['R'])
    lmcm = int(cfg['lmcm'])
    K = mj.make_K(cfg['tune'])
    for n_ in ('xmass', 'bydtsrc', 'dtsrc', 'bybr'):
        K[n_] = np.float64(cfg[n_])
    K['fmpscale'] = np.float64(min(1.0, np.float64(cfg['dtsrc']) / (np.float64(1.0) * mj.mc.SECONDS_PER_HOUR)))
    Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
    S, Sc, I_, aux = jax.jit(lambda K, R, lmcm: mj._setup(SimpleNamespace(**K), R, lmcm), static_argnums=2)(Kj, R, lmcm)
    lmin = 3
    mask, fmp0, fmp2 = jax.jit(lambda K, S, I: md._base_dev(K, lmin, S, I))(Kj, S, I_)
    N = int(mask.shape[0])
    idx_all = np.flatnonzero(np.asarray(mask))
    cnt = min(nb, idx_all.size)
    idxg = np.zeros(nb, np.int64)
    idxg[:cnt] = idx_all[:cnt]
    idxs = np.full(nb, N, np.int64)
    idxs[:cnt] = idx_all[:cnt]
    valid = np.arange(nb) < cnt
    print('backend', jax.default_backend(), 'LMIN', lmin, 'nb', nb, 'valid', cnt, flush=True)
    prev = 0.0
    names = {0: 'preamble + call (stop before the plume loop)', 1: '+ plume ascent (NPPL loop)', 2: '+ after-plume bookkeeping', 3: '+ downdraft', 4: '+ subsidence (loops + QUS callbacks)', 5: '+ diagnostics', 6: '+ precipitation', 99: 'whole block incl. the scatter back'}
    for stop in (0, 1, 2, 3, 4, 5, 6, 99):
        md.STOP = stop
        f = jax.jit(lambda K, S, Sc, I, aux, fmp0, fmp2, idxg, idxs, valid: md._event_body(K, lmin, fmp0, fmp2, idxg, idxs, valid, S, Sc, I, aux))
        args = (Kj, S, Sc, I_, aux, fmp0, fmp2, jnp.asarray(idxg), jnp.asarray(idxs), jnp.asarray(valid))
        ts = []
        for r in range(repeats + 1):
            t0 = time.perf_counter()
            out = f(*args)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        cur = float(np.mean(ts[1:]))
        print(f'stop {stop:2d} {names[stop]:52s} cumulative {cur:7.3f} s   delta {cur - prev:7.3f} s   (compile+first {ts[0]:.1f} s)', flush=True)
        prev = cur
    md.STOP = 99


if mode == 'capture':
    import gpu_step_run as G
    import jax
    out_path = sys.argv[2]
    date = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].isdigit() else 'nov26'
    step = int(sys.argv[-1]) if len(sys.argv) > 3 and sys.argv[-1].isdigit() else 1
    C = G.C
    cp = C.Coupled(date)
    cs = cp.ph.cs
    orig = cs.mstcnv
    cur = {'k': 0, 'done': False}

    def wrapped(R, stats=None):
        if cur['k'] == step and not cur['done']:
            cur['done'] = True
            cfg = {k: (v if not hasattr(v, 'item') else v.item()) for k, v in cs.cfg.items() if not callable(v)}
            with open(out_path, 'wb') as f:
                pickle.dump(dict(R=jax.device_get(R), cfg=cfg, date=date, step=step), f, protocol=4)
            print('captured', out_path, flush=True)
        return orig(R, stats)

    cs.mstcnv = wrapped
    sr = G.build_registry()
    state, rec, dev = cp.initial_state()
    for k in range(step + 1):
        cur['k'] = k
        if k > 0:
            rec = cp.ph.load_records(cp.it0 + k, first=False)
            dev = cp.ph.to_device(rec, first=False)
        out, state, arrays, sr = G.run_step(cp, k, state, rec, dev, sr, False)
        print(f'[step {k}] wall {out["step_seconds"]:.2f} s', flush=True)
        if cur['done']:
            break
elif mode == 'phases':
    _phases_bench(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 128)
elif mode == 'widths':
    _widths_bench(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
else:
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    import clouds_mstcnv_jax as mj
    import clouds_mstcnv_dev as md
    from types import SimpleNamespace
    d = pickle.load(open(sys.argv[2], 'rb'))
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    cfg = d['cfg']
    R = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) and np.asarray(a).dtype.kind in 'fiub' else a, d['R'])
    lmcm = int(cfg['lmcm'])
    K = mj.make_K(cfg['tune'])
    for n_ in ('xmass', 'bydtsrc', 'dtsrc', 'bybr'):
        K[n_] = np.float64(cfg[n_])
    F = np.float64
    K['fmpscale'] = np.float64(min(1.0, F(cfg['dtsrc']) / (F(1.0) * mj.mc.SECONDS_PER_HOUR)))
    Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
    setup = jax.jit(lambda K, R, lmcm: mj._setup(SimpleNamespace(**K), R, lmcm), static_argnums=2)
    print('backend', jax.default_backend(), jax.devices(), 'lmcm', lmcm, flush=True)
    rows = []
    for r in range(rep + 1):
        t = {}
        t0 = time.perf_counter()
        S, Sc, I_, aux = setup(Kj, R, lmcm)
        jax.block_until_ready((S, Sc))
        t['setup'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        S2, Sc2, err, nev = md.events(Kj, S, Sc, I_, aux, lmcm, md.BUCKETS)
        jax.block_until_ready((S2, Sc2))
        t['events'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        o = md._post_dev(Kj, S2, Sc2, I_, aux)
        jax.block_until_ready(o)
        t['post'] = time.perf_counter() - t0
        t['total'] = t['setup'] + t['events'] + t['post']
        rows.append(t)
        print(f'run {r} ({"compile" if r == 0 else "steady"}): setup {t["setup"]:.3f}  events {t["events"]:.3f}  post {t["post"]:.3f}  total {t["total"]:.3f} s', flush=True)
    import hashlib
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves((S2, Sc2, o)):
        h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    print('result digest (sha256 of all output leaves; equal digests = bitwise equal results):', h.hexdigest()[:24])
    import os as _os
    if _os.environ.get('BENCH_DUMP'):
        leaves = jax.tree_util.tree_leaves_with_path((S2, Sc2, o))
        np.savez(_os.environ['BENCH_DUMP'], **{jax.tree_util.keystr(p): np.asarray(v) for p, v in leaves})
        print('dumped', len(leaves), 'leaves to', _os.environ['BENCH_DUMP'])
    nev = np.asarray(nev)
    print('rounds (round-based loop only; 0 for the LMIN loop):', int(nev[0]))
    print('events per LMIN:', [int(x) for x in nev[1:lmcm]], ' total', int(nev.sum()), ' error flag', bool(err))
    ss = rows[1:]
    print('steady mean: ' + '  '.join(f'{k} {np.mean([x[k] for x in ss]):.3f}' for k in ('setup', 'events', 'post', 'total')))
