"""Speed track: end-to-end test of mstcnv_host (worker process + shared file) against the NumPy port.
    python speed_mstcnv_host_test.py IN.pkl [NCOMPARE=64] [REPEATS=5]     (run with the JAX environment's python: no numba needed here)"""
import pickle, sys, time
import numpy as np
import mstcnv_host as H
d = pickle.load(open(sys.argv[1], 'rb')); R, cfg = d['R'], d['cfg']
ncmp = int(sys.argv[2]) if len(sys.argv) > 2 else 64
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N = R['pl'].shape[1]
Rh = dict(R)
ra = np.asarray(R['ra'])
Rh['ra'] = ra
t = time.perf_counter(); h = H.MstcnvHost(N, nks=Rh['ra'].shape[0]); print(f'worker start {time.perf_counter()-t:.1f} s')
t = time.perf_counter(); v = h.run(Rh, cfg); print(f'first run {time.perf_counter()-t:.1f} s (includes numba compile or cache load)')
ts = []
for _ in range(reps):
    t = time.perf_counter(); v = h.run(Rh, cfg); ts.append(time.perf_counter() - t)
print('steady run seconds (copy in + compute + handshake):', ' '.join(f'{x:.3f}' for x in ts))
# check against the NumPy port on a sample of columns
import clouds_mstcnv_ff as mc
tune = cfg['tune']
bad = {}
for j in range(0, N, max(1, N // ncmp)):
    r = {k: np.asarray(R[k][..., j]) for k in R}
    r.update(lmcm=cfg['lmcm'], xmass=cfg['xmass'], bydtsrc=cfg['bydtsrc'], dtsrc=cfg['dtsrc'], bybr=cfg['bybr'], dcl=int(R['dcl'][j]))
    ref = mc.mstcnv_column(r, tune)
    for i, n in enumerate(H.OUT_L):
        if not np.array_equal(ref[n], v['o_l'][i, j]): bad[n] = bad.get(n, 0) + 1
    for i, n in enumerate(H.OUT_SC[:14]):
        if not np.array_equal(np.float64(ref[n]), v['o_sc'][i, j]): bad[n] = bad.get(n, 0) + 1
    for n, k in (('lhp', 0), ('precnvl', 1)):
        if not np.array_equal(ref[n], v['o_l1'][k, j]): bad[n] = bad.get(n, 0) + 1
    for n, k in (('smom', 0), ('qmom', 1)):
        if not np.array_equal(ref[n], v['o_m'][k, j]): bad[n] = bad.get(n, 0) + 1
    for n, k in (('um', 0), ('vm', 1)):
        if not np.array_equal(ref[n], v['o_k'][k, j]): bad[n] = bad.get(n, 0) + 1
print('fields with mismatches (count of sampled columns):', bad or 'none')
h.close()
