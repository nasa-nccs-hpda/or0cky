"""Kernel-level profile of ONE cloud-base event block (md._event_body at LMIN=3) on the GPU: jax.profiler trace of a steady call, aggregated by hlo_op / kernel name.
    python -u gpu_mstcnv_trace.py IN.pkl [NB=128] [LMIN=3]      (speed/step-time, 2026-10-12; diagnostic only; reuses the setup of gpu_mstcnv_bench._widths_bench)"""
import glob, gzip, json, os, pickle, sys, time
import gpu_step_run  # noqa: F401  (GPU preamble first)
import numpy as np
from collections import defaultdict
def main(path, NB=128, lmin=3):
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
    mask, fmp0, fmp2 = jax.jit(lambda K, S, I: md._base_dev(K, lmin, S, I))(Kj, S, I_)
    N = int(mask.shape[0])
    idx_all = np.flatnonzero(np.asarray(mask))
    print('backend', jax.default_backend(), 'LMIN', lmin, 'event columns available at this LMIN:', idx_all.size, 'of N =', N, flush=True)
    cnt = min(NB, idx_all.size)
    nb = NB
    idxg = np.zeros(nb, np.int64); idxg[:cnt] = idx_all[:cnt]
    idxs = np.full(nb, N, np.int64); idxs[:cnt] = idx_all[:cnt]
    valid = np.arange(nb) < cnt
    f = jax.jit(lambda K, S, Sc, I, aux, fmp0, fmp2, idxg, idxs, valid: md._event_body(K, lmin, fmp0, fmp2, idxg, idxs, valid, S, Sc, I, aux))
    args = (Kj, S, Sc, I_, aux, fmp0, fmp2, jnp.asarray(idxg), jnp.asarray(idxs), jnp.asarray(valid))
    t0 = time.perf_counter(); jax.block_until_ready(f(*args)); print('compile+first', round(time.perf_counter() - t0, 1), flush=True)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t0)
    print('steady calls', [round(x, 4) for x in ts], 'valid cols', cnt, flush=True)
    d = os.environ.get('TRACE_DIR', 'trace_mstcnv')
    jax.profiler.start_trace(d)
    jax.block_until_ready(f(*args))
    jax.profiler.stop_trace()
    ev = json.load(gzip.open(glob.glob(d + '/**/*.trace.json.gz', recursive=True)[-1]))['traceEvents']
    pn = {e['pid']: e['args']['name'] for e in ev if e.get('ph') == 'M' and e['name'] == 'process_name'}
    g = [e for e in ev if e.get('ph') == 'X' and 'GPU' in pn.get(e['pid'], '')]
    print('GPU kernel events in one call:', len(g), ' summed device time', round(sum(e['dur'] for e in g) / 1e6, 4), 's')
    ts_ = sorted(g, key=lambda e: e['ts'])
    if ts_:
        print('span', round((ts_[-1]['ts'] + ts_[-1]['dur'] - ts_[0]['ts']) / 1e6, 4), 's')
    agg = defaultdict(lambda: [0, 0.0])
    for e in g:
        key = e['name'].rstrip('0123456789_')
        a = agg[key]; a[0] += 1; a[1] += e['dur'] / 1e6
    print('by kernel family (count, seconds):')
    for n, (c, s) in sorted(agg.items(), key=lambda x: -x[1][0])[:25]:
        print(f'   x{c:<7d} {s:8.4f} s  {n[:80]}')
    ds = [e['dur'] for e in g]
    print('kernel duration us: median', np.median(ds), 'p90', np.percentile(ds, 90), 'max', max(ds))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 128, int(sys.argv[3]) if len(sys.argv) > 3 else 3)
