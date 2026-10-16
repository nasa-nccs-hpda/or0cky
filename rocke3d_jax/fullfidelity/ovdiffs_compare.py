"""Validate ovdiffs_ff.py against real OVDIFFS per-call dumps (D55).

`ffz_ovdiffs_<itime>.bin`: one 109-double record per OCONV G0ML/S0ML OVDIFFS call (tag 0/1):
tag, i, j, lmij, dt, then u0/k/ghat/dtp4/dtbydz/bydz2/u/fl (13 each).
"""
import numpy as np

LMO = 13
RECLEN = 109


def _pad(flat):
    a = np.zeros(LMO + 1)
    a[1:] = flat
    return a


def load_ovdiffs_records(path):
    raw = np.fromfile(path, dtype='>f8')
    assert raw.size % RECLEN == 0, (raw.size, RECLEN)
    recs = []
    for r in range(raw.size // RECLEN):
        row = raw[r * RECLEN:(r + 1) * RECLEN]
        rec = dict(tag=int(round(row[0])), i=int(round(row[1])),
                   j=int(round(row[2])), lmij=int(round(row[3])), dt=row[4])
        off = 5
        for name in ["u0", "k", "ghat", "dtp4", "dtbydz", "bydz2"]:
            rec[name] = _pad(row[off:off + LMO]); off += LMO
        rec["u_real"] = _pad(row[off:off + LMO]); off += LMO
        rec["fl_real"] = _pad(row[off:off + LMO]); off += LMO
        assert off == RECLEN
        recs.append(rec)
    return recs
