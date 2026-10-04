"""Validate ovdiffs_jax.ovdiff_jax (momentum OVDIFF) against the recorded real calls (D64).

Records: ffz_ovdiff_<itime>.bin (D56), one per momentum OVDIFF call, with the Fortran u output.
Runs the batched JAX solver on each itime's calls and reports the worst relative error against
the recorded u, and against the numpy port ovdiff_ff.py (exact reference). Speed-first tolerance 1e-9.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovdiffs_compare import load_ovdiffs_records  # noqa: E402
from ovdiffs_jax import ovdiff_jax  # noqa: E402
from ovdiff_ff import ovdiff  # noqa: E402

TOL = 1e-9


def run(path):
    recs = load_ovdiffs_records(path)
    stk = lambda key: np.stack([r[key] for r in recs])  # noqa: E731
    lmij = np.array([r['lmij'] for r in recs], dtype=np.int64)
    t0 = time.time()
    u = np.asarray(ovdiff_jax(stk('k'), stk('ghat'), stk('dtp4'), stk('dtbydz'), stk('bydz2'),
                              lmij, stk('u0')))
    el = time.time() - t0
    worst_rec = worst_np = 0.0
    for n, r in enumerate(recs):
        m = r['lmij']
        ref_np = ovdiff(r['k'], r['ghat'], r['dtp4'], r['dtbydz'], r['bydz2'], m, r['u0'])
        sc = max(np.max(np.abs(r['u_real'][1:m + 1])), 1e-300)
        worst_rec = max(worst_rec, float(np.max(np.abs(u[n, 1:m + 1] - r['u_real'][1:m + 1])) / sc))
        worst_np = max(worst_np, float(np.max(np.abs(u[n, 1:m + 1] - ref_np[1:m + 1])) / sc))
    return len(recs), worst_rec, worst_np, el


if __name__ == '__main__':
    tot = 0; wr = wn = 0.0; tt = 0.0
    for f in sorted(glob.glob(f'{sys.argv[1]}/ffz_ovdiff_*.bin')):
        n, a, b, el = run(f)
        tot += n; wr = max(wr, a); wn = max(wn, b); tt += el
        print(os.path.basename(f), 'calls', n, 'rec', f'{a:.1e}', 'numpy', f'{b:.1e}', 'sec', round(el, 2))
    print('TOTAL calls', tot, 'worst vs recorded', f'{wr:.2e}', 'worst vs numpy', f'{wn:.2e}',
          'sec', round(tt, 2), 'PASS' if wn <= TOL else 'FAIL', 'tol', TOL)
