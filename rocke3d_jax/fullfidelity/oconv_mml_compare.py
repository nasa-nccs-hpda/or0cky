"""Validate oconv_mml_ff.mass_bookkeeping against real OCONV dumps (D61).

ITER>=2 (MO(I,J,1)*DXYPO(J)) is checked bitwise: MO(I,J,1) is the setup record's mo1 field, exact.
ITER=1 (MO1(I,J)*DXYPO(J)) needs MO1, which no dump records. It is recovered from the recorded
DELTAM = (MO-MO1)*BYDTS with BYDTS = 1/DTS, DTS = 1800 s (OCNKPP.f:1845, OVDIFFS dt). That gives
an indirect check, so the ITER=1 result is reported separately.

Dump layout: ffz_bymml (ffdump_bymml, 9 doubles) and ffz_setup (ffdump_setup, 1218 doubles;
mo1 at index 45, DELTAM at index 1160), both per-call streams from one run directory.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbl_glue_bymml_compare import load_bymml  # noqa: E402
from oconv_mml_ff import mass_bookkeeping  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402

RS_SETUP = 1218
MO1_INDEX_MO = 45      # mo1 = MO(I,J,1), setup record
DELTAM_INDEX = 1160    # DELTAM, setup record
DTS = 1800.0


def check_dir(run):
    dxypo = geomo_dyn_arrays()[-1]
    c = dict(it2=0, it2_mml_bad=0, it2_by_bad=0, it1=0, it1_mml_bad=0, it1_by_bad=0)
    for bf in sorted(glob.glob(f'{run}/ffz_bymml_*.bin')):
        itime = bf.split('_')[-1][:-4]
        B = load_bymml(bf)
        S = np.fromfile(f'{run}/ffz_setup_{itime}.bin', dtype='>f8').reshape(-1, RS_SETUP)
        assert len(S) == len(B)
        for b, row in zip(B, S):
            dx = dxypo[b['j']]
            mo_cur = row[MO1_INDEX_MO]
            if b['iter'] >= 2:
                mml, by = mass_bookkeeping(None, mo_cur, dx, b['iter'])
                c['it2'] += 1
                c['it2_mml_bad'] += int(mml != b['mml'])
                c['it2_by_bad'] += int(by != b['bymml'])
            else:
                mo1_rec = mo_cur - row[DELTAM_INDEX] * DTS
                mml, by = mass_bookkeeping(mo1_rec, mo_cur, dx, 1)
                c['it1'] += 1
                c['it1_mml_bad'] += int(mml != b['mml'])
                c['it1_by_bad'] += int(by != b['bymml'])
    return c


if __name__ == '__main__':
    tot = None
    for run in sys.argv[1:]:
        c = check_dir(run)
        print(run, c)
        tot = c if tot is None else {k: tot[k] + c[k] for k in c}
    print('TOTAL', tot)
