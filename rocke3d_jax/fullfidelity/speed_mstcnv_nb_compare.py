"""Speed track: compare clouds_mstcnv_nb (numba) with the NumPy port per column and time both.
    python speed_mstcnv_nb_compare.py IN.pkl [STRIDE=40] [NCOMPARE=all sampled]   (run with a numba-capable python)
Prints, per output field, the number of columns bitwise equal and the max abs / max rel difference."""
import pickle, sys, time
import numpy as np
import clouds_mstcnv_ff as mc
import clouds_mstcnv_nb as nb
d = pickle.load(open(sys.argv[1], 'rb')); R, cfg = d['R'], d['cfg']
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 40
tune = cfg['tune']
N = R['pl'].shape[1]
idx = list(range(0, N, stride))
tv = np.array([float(tune[k]) for k in nb.TUNE_NAMES])
nks = R['ra'].shape[0]
def col(j):
    r = {k: np.asarray(R[k][..., j]) for k in R}
    r.update(lmcm=cfg['lmcm'], xmass=cfg['xmass'], bydtsrc=cfg['bydtsrc'], dtsrc=cfg['dtsrc'], bybr=cfg['bybr'], dcl=int(R['dcl'][j]))
    return r
def run_nb(r):
    o_sc = np.zeros(15); o_l = np.zeros((37, 40)); o_l1 = np.zeros((2, 41)); o_m = np.zeros((2, 9, 40)); o_k = np.zeros((2, nks, 40))
    nb.mstcnv_col(float(r['pearth']), float(r['pland']), int(r['dcl']), int(r['lmcm']), float(r['xmass']), float(r['bydtsrc']),
                  float(r['dtsrc']), float(r['bybr']), *(np.ascontiguousarray(r[k], dtype=np.float64) for k in
                  ('pl ple plk airm byam etal tl tvl sm qm qcll qcil sdl wturb gzl smom qmom ra um vm u0 v0'.split())),
                  tv, nks, o_sc, o_l, o_l1, o_m, o_k)
    return o_sc, o_l, o_l1, o_m, o_k
t0 = time.perf_counter(); run_nb(col(idx[0])); print(f'numba first call (compile/cache load): {time.perf_counter()-t0:.1f} s')
stats = {}
tn = tr = 0.0
for j in idx:
    r = col(j)
    t = time.perf_counter(); ref = mc.mstcnv_column(r, tune); tr += time.perf_counter() - t
    t = time.perf_counter(); o_sc, o_l, o_l1, o_m, o_k = run_nb(r); tn += time.perf_counter() - t
    cmp = {}
    for i, n in enumerate(nb.OUT_SC[:14]): cmp[n] = (np.float64(ref[n]), o_sc[i])
    for i, n in enumerate(nb.OUT_L): cmp[n] = (ref[n], o_l[i])
    cmp['lhp'] = (ref['lhp'], o_l1[0]); cmp['precnvl'] = (ref['precnvl'], o_l1[1])
    cmp['smom'] = (ref['smom'], o_m[0]); cmp['qmom'] = (ref['qmom'], o_m[1])
    cmp['um'] = (ref['um'], o_k[0]); cmp['vm'] = (ref['vm'], o_k[1])
    for n, (a, b) in cmp.items():
        a = np.asarray(a, float); b = np.asarray(b, float)
        eq = bool(np.array_equal(a, b, equal_nan=True))
        ad = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        den = np.maximum(np.abs(a), 1e-300)
        rd = float(np.nanmax(np.abs(a - b) / den)) if a.size else 0.0
        s = stats.setdefault(n, [0, 0, 0.0, 0.0]); s[0] += eq; s[1] += 1; s[2] = max(s[2], ad); s[3] = max(s[3], rd)
print(f'columns {len(idx)}: numpy port {tr/len(idx)*1e3:.2f} ms/col, numba {tn/len(idx)*1e3:.3f} ms/col (single thread, first call excluded)')
bad = {n: s for n, s in stats.items() if s[0] < s[1]}
print(f'fields bitwise equal in all columns: {len(stats)-len(bad)} of {len(stats)}')
for n, s in sorted(bad.items(), key=lambda kv: -kv[1][3])[:25]:
    print(f'  {n:10s} equal {s[0]}/{s[1]}  max abs {s[2]:.3e}  max rel {s[3]:.3e}')
