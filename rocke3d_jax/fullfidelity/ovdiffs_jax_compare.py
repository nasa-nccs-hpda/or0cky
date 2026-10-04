"""Validate ovdiffs_jax.py against the recorded real OVDIFFS calls (D62).

Records come from ffz_ovdiffs_<itime>.bin (D55): every call's inputs and the Fortran outputs u and
fl. The batched JAX port is run on all calls of one itime at once. Accuracy is reported as a
maximum relative error against a stated tolerance (speed-first port; not bitwise).
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovdiffs_compare import load_ovdiffs_records  # noqa: E402
from ovdiffs_jax import ovdiffs_jax  # noqa: E402

TOL = 1e-9   # stated tolerance for the speed-first port (relative to column scale)


def stack(recs):
    keys = ['k', 'ghat', 'dtp4', 'dtbydz', 'bydz2', 'u0']
    arrs = {kk: np.stack([r[kk] for r in recs]) for kk in keys}
    lmij = np.array([r['lmij'] for r in recs])
    dt = np.array([r['dt'] for r in recs])
    return arrs, lmij, dt


def run(path):
    recs = load_ovdiffs_records(path)
    arrs, lmij, dt = stack(recs)
    assert np.all(dt == dt[0]), 'dt varies within one itime'
    u_ref = np.stack([r['u_real'] for r in recs])
    fl_ref = np.stack([r['fl_real'] for r in recs])
    t0 = time.time()
    u, fl = ovdiffs_jax(arrs['k'], arrs['ghat'], arrs['dtp4'], arrs['dtbydz'], arrs['bydz2'],
                        float(dt[0]), lmij, arrs['u0'])
    u = np.asarray(u); fl = np.asarray(fl)
    elapsed = time.time() - t0
    # compare only the active rows (1..lmij for u, 1..lmij-1 for fl)
    umax = fl_max = 0.0
    for n in range(len(recs)):
        m = lmij[n]
        du = np.abs(u[n, 1:m + 1] - u_ref[n, 1:m + 1])
        scale = np.maximum(np.abs(u_ref[n, 1:m + 1]).max(), 1e-300)
        umax = max(umax, float(du.max() / scale))
        if m > 1:
            df = np.abs(fl[n, 1:m] - fl_ref[n, 1:m])
            fscale = np.maximum(np.abs(fl_ref[n, 1:m]).max(), 1e-300)
            fl_max = max(fl_max, float(df.max() / fscale))
    return len(recs), umax, fl_max, elapsed


if __name__ == '__main__':
    root = sys.argv[1]
    worst_u = worst_f = 0.0
    total = 0
    for f in sorted(glob.glob(f'{root}/ffz_ovdiffs_*.bin')):
        n, um, fm, el = run(f)
        total += n
        worst_u = max(worst_u, um); worst_f = max(worst_f, fm)
        print(os.path.basename(f), 'calls', n, 'max rel u', um, 'max rel fl', fm, 'sec', round(el, 3))
    print('TOTAL calls', total, 'worst rel u', worst_u, 'worst rel fl', worst_f,
          'tolerance', TOL, 'PASS' if max(worst_u, worst_f) <= TOL else 'FAIL')
