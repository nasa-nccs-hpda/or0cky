"""D221: which columns of the five per-step SURFACE record files (ffp/ffs/ffl/ffg/fft, two substeps each) vary over the steps of a record directory.
Rows are keyed by their first columns (ffp/ffs: i, j, type [, substep flag]); fixed-rowset files are compared row by row.  Usage: python surface_records_audit.py [DAYDIR]"""
import sys
import json
import numpy as np
import atm_step as A

FF = A.FF
DAY = sys.argv[1] if len(sys.argv) > 1 else 'nov26_day2'
M = A._surf_mods()
FILES = (('ffp', 'PC', 3), ('ffs', 'ST', 3), ('ffl', 'LI', 2), ('ffg', 'GC', 2), ('fft', 'TA', 2))
its = sorted(int(f.split('_')[-1].split('.')[0]) for f in __import__('os').listdir(f'{FF}/{DAY}') if f.startswith('ffp_'))
out = {}
for nm, mod, nk in FILES:
    base = None
    ncols = None
    varies = None
    n_rowsets = set()
    for it in its:
        a = np.asarray(M[mod].load(f'{FF}/{DAY}/{nm}_{it}.bin'))
        h = len(a) // 2
        n_rowsets.add(h)
        ncols = a.shape[1]
        if varies is None:
            varies = np.zeros(ncols, bool)
        for half in (0, 1):
            blk = a[half * h:(half + 1) * h]
            key = [tuple(r) for r in blk[:, :nk].astype(int)]
            d = base.setdefault(half, {}) if base is not None else None
            if base is None:
                base = {}
                d = base.setdefault(half, {})
            for r, k in zip(blk, key):
                if k in d:
                    varies |= ~((d[k] == r) | (np.isnan(d[k]) & np.isnan(r)))
                else:
                    d[k] = r.copy()
    out[nm] = dict(ncols=int(ncols), n_steps=len(its), distinct_rowset_sizes=len(n_rowsets), constant_cols=int((~varies).sum()), varying_cols=[int(c) for c in np.nonzero(varies)[0]])
    print(nm, out[nm]['ncols'], 'cols;', out[nm]['constant_cols'], 'constant;', len(out[nm]['varying_cols']), 'vary;', out[nm]['distinct_rowset_sizes'], 'rowset sizes', flush=True)
json.dump(out, open('surface_records_audit.json', 'w'))
