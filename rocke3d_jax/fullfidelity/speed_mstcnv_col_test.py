"""column_mstcnv vs clouds_mstcnv_ff.mstcnv_column on single columns (graphcast-env python)."""
import pickle, sys
import numpy as np
import clouds_mstcnv_ff as mc
import mstcnv_host as H
d = pickle.load(open(sys.argv[1], 'rb')); R, cfg = d['R'], d['cfg']; tune = cfg['tune']
N = R['pl'].shape[1]; bad = {}; n = 0
for j in range(0, N, 97):
    r = {k: np.asarray(R[k][..., j]) for k in R}
    r.update(lmcm=cfg['lmcm'], xmass=cfg['xmass'], bydtsrc=cfg['bydtsrc'], dtsrc=cfg['dtsrc'], bybr=cfg['bybr'], dcl=int(R['dcl'][j]))
    a = mc.mstcnv_column(r, tune); b = H.column_mstcnv(r, tune); n += 1
    for k in a:
        if not np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float)): bad[k] = bad.get(k, 0) + 1
print(f'{n} columns; mismatching keys:', bad or 'none')
