"""Spike (speed/step-time): time of the per-column scalar MSTCNV port (clouds_mstcnv_ff.mstcnv_column, interpreted NumPy-scalar Python) on columns of the captured nov26 step-1 inputs.
    python spike_mstcnv_percol.py IN.pkl [STRIDE=40]      CPU only; diagnostic: the interpreted time bounds what a compiled port must beat, it is not a result for a compiled one."""
import pickle, sys, time
import numpy as np
import clouds_mstcnv_ff as mc
d = pickle.load(open(sys.argv[1], 'rb'))
R, cfg = d['R'], d['cfg']
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 40
tune = cfg['tune']
N = R['pl'].shape[1]
def col(j):
    r = {k: np.asarray(R[k][..., j]) for k in R}
    r.update(lmcm=cfg['lmcm'], xmass=cfg['xmass'], bydtsrc=cfg['bydtsrc'], dtsrc=cfg['dtsrc'], bybr=cfg['bybr'], dcl=int(R['dcl'][j]))
    return r
ts, conv = [], 0
for j in range(0, N, stride):
    r = col(j)
    t = time.perf_counter(); o = mc.mstcnv_column(r, tune); ts.append(time.perf_counter() - t)
    conv += int(o['lmcmax'] > 0)
ts = np.array(ts)
print(f'columns timed {ts.size} of {N} (stride {stride}); convective {conv}; per column: median {np.median(ts)*1e3:.1f} ms mean {ts.mean()*1e3:.1f} ms max {ts.max()*1e3:.1f} ms')
print(f'extrapolated to all {N} columns at the mean: {ts.mean()*N:.1f} s (single core, interpreted)')
