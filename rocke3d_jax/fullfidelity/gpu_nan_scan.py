"""Where does the NaN live? (numpy only, for a saved step npz of gpu_step_run.py)

usage: python gpu_nan_scan.py STEP.npz [MAXLIST]

Prints, per group (the part of the key before '/'), how many float arrays contain non-finite values, then the arrays with the FEWEST
non-finite values first (the sparsest are the closest to the source), with the fraction non-finite, the first non-finite index, and the
number of distinct horizontal columns (first two axes) touched. A diagnostic, not a verdict.
"""
import sys

import numpy as np


def main(fn, maxlist=40):
    z = np.load(fn)
    groups, rows = {}, []
    for k in z.files:
        a = z[k]
        if a.dtype.kind != 'f':
            continue
        bad = ~np.isfinite(a)
        g = k.split('/')[0]
        groups.setdefault(g, [0, 0])
        groups[g][1] += 1
        n = int(bad.sum())
        if n:
            groups[g][0] += 1
            first = tuple(int(x) for x in np.argwhere(bad)[0])
            cols = int(bad.reshape(a.shape[0], a.shape[1], -1).any(axis=2).sum()) if a.ndim >= 2 else n
            rows.append((n / a.size, k, n, a.size, first, cols))
    print(f'file {fn}')
    print(f"{'group':<12}{'arrays with non-finite / float arrays':>40}")
    for g, (b, t) in groups.items():
        print(f'{g:<12}{b:>20} / {t}')
    print(f'\narrays with non-finite values, sparsest first (showing {maxlist} of {len(rows)}):')
    print(f"{'fraction':>10} {'count':>8}/{'size':<8} {'first index':<18}{'columns':>8}  key")
    for fr, k, n, s, first, cols in sorted(rows)[:maxlist]:
        print(f'{fr:>10.3e} {n:>8}/{s:<8} {str(first):<18}{cols:>8}  {k}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
