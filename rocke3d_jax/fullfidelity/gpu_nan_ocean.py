"""Which ocean and strait cells go NaN in step 1, and what did the step-0 end state (= the step-1 input) hold there? (numpy only)

usage: python gpu_nan_ocean.py STEP0.npz STEP1.npz

The strait arrays are (layer 1..13, strait 1..12); the 3-D ocean arrays are (i, j, layer). Prints the NaN slots of the strait arrays and the step-0
values there (mass mmst, flux must), the number and place of NaN ocean columns in mo / mo1 / opress, and the step-0 values of mo in those columns
(zero, negative, tiny?). A diagnostic to find the first bad quantity; JAX_DEBUG_NANS stops at a benign masked 0/0 in stadvt_jax and cannot.
"""
import sys

import numpy as np


def main(f0, f1):
    a, b = np.load(f0), np.load(f1)
    print(f'step0 {f0}\nstep1 {f1}')
    print('\n-- strait arrays (layer, strait), NaN count in step 1 / total')
    for k in ('must', 'mmst', 'g0mst', 'gxmst', 'gzmst', 's0mst', 'sxmst', 'szmst'):
        kk = 'surf/ocean/' + k
        if kk in b.files:
            bad = ~np.isfinite(b[kk])
            print(f'  {k:6s} {int(bad.sum()):4d} / {bad.size}   NaN in step 0: {int((~np.isfinite(a[kk])).sum())}')
    kk = 'surf/ocean/must'
    if kk in b.files:
        bad = ~np.isfinite(b[kk])
        bs = sorted(set(int(s) for _, s in np.argwhere(bad)))
        print('  straits (0-based) with NaN in must:', bs)
        for s in bs[:6]:
            ls = [int(l) for l in np.argwhere(bad[:, s])[:, 0]]
            print(f'   strait {s}: NaN layers {ls}; step-0 mmst {np.array2string(a["surf/ocean/mmst"][:, s], precision=3)}')
            print(f'             step-0 must {np.array2string(a["surf/ocean/must"][:, s], precision=3)}')
    print('\n-- 3-D / 2-D ocean arrays, NaN count in step 1')
    for k in ('mo', 'gxmo', 'sxmo', 'mo1', 'opress'):
        kk = 'surf/ocean/' + k
        if kk in b.files:
            print(f'  {k:6s} {int((~np.isfinite(b[kk])).sum()):6d} / {b[kk].size}   (step 0: {int((~np.isfinite(a[kk])).sum())})')
    if 'surf/ocean/mo1' in b.files:
        bad = ~np.isfinite(b['surf/ocean/mo1'])
        ij = np.argwhere(bad)
        print(f'\n  mo1 NaN columns: {len(ij)}; first (i,j) 0-based: {[tuple(int(x) for x in p) for p in ij[:8]]}')
        mo0 = a['surf/ocean/mo']
        for p in ij[:5]:
            i, j = int(p[0]), int(p[1])
            print(f'   col ({i},{j}) step-0 mo[:5] {np.array2string(mo0[i, j, :5], precision=4)} mo1 {a["surf/ocean/mo1"][i, j]:.6g}')
        m = mo0[np.isfinite(mo0)]
        print(f'  step-0 mo: min {m.min():.4g} (over all cells incl. land zeros), count of mo<0 {(mo0 < 0).sum()}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
