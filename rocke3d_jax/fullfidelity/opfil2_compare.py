"""Validate opfil2_ff.opfil2 against the real OPFIL2 calls (D74/D75).

ffz_opfil_in_<itime>.bin / ffz_opfil_out_<itime>.bin: one record per call: l, jmin, jmax, then
X(72,46) (the Fortran X, column-major). ffz_opcoef.bin: setup coefficients (written once per run).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opfil2_ff import opfil2, coef_from_dump, IM, JM  # noqa: E402

RS = 3 + IM * JM
TOL = 1e-9


def load_calls(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64).reshape(-1, RS)
    out = []
    for r in raw:
        l, jmin, jmax = int(round(r[0])), int(round(r[1])), int(round(r[2]))
        x = r[3:].reshape((IM, JM), order='F')
        out.append((l, jmin, jmax, x))
    return out


def main(d):
    v = np.fromfile(f'{d}/ffz_opcoef.bin', dtype='>f8').astype(np.float64)
    c = coef_from_dump(v)
    worst = 0.0
    n = 0
    for fi in sorted(glob.glob(f'{d}/ffz_opfil_in_*.bin')):
        it = fi.split('_')[-1][:-4]
        ins = load_calls(fi)
        outs = load_calls(f'{d}/ffz_opfil_out_{it}.bin')
        assert len(ins) == len(outs)
        for (l, jmin, jmax, x), (l2, _, _, xo) in zip(ins, outs):
            got = opfil2(x, l, jmin, jmax, c)
            band = slice(jmin - 1, jmax)
            ref = xo[:, band]
            sc = max(np.max(np.abs(ref)), 1e-300)
            worst = max(worst, float(np.max(np.abs(got[:, band] - ref)) / sc))
            n += 1
    print('OPFIL2 calls', n, 'worst rel error', f'{worst:.1e}', 'PASS' if worst <= TOL else 'FAIL', 'tol', TOL)


if __name__ == '__main__':
    main(sys.argv[1])
