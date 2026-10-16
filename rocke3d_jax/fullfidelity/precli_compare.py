"""Compare landice_precip_ff.precip_li against real-Fortran PRECIP_LI dumps (ffx_<itime>.bin, 40 doubles,
Stage 1 of the DYNSI/ocean port, D28).

Layout (0-based Python cols): 0:i 1:j 2:ihc 3:ftype 4:prcp 5:enrgp 6:snow(in) 7:tg1(in) 8:tg2(in)
-- after PRECIP_LI -- 9:snow 10:tg1 11:tg2 12:run0(=RUNO) 13:edifs(=E1) 14:difs(=IMPLM) 15:erun2(=IMPLH)
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landice_precip_ff as L

NREC = 40


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    return L.precip_li(r[3], r[4], r[5], r[6], r[7], r[8])


def compare_row(r):
    out = run_row(r)
    ref = dict(snow=r[9], tg1=r[10], tg2=r[11], runo=r[12], e1=r[13], implm=r[14], implh=r[15])
    return {k: dict(got=out[k], ref=v, abs=abs(out[k] - v)) for k, v in ref.items()}


if __name__ == "__main__":
    rec = load(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    idx = np.linspace(0, len(rec) - 1, min(n, len(rec))).astype(int)
    for i in idx:
        rows = compare_row(rec[i])
        print(f"row {i}:")
        for k, v in rows.items():
            print(f"  {k:6s} abs={v['abs']:.3e} got={v['got']:.6e} ref={v['ref']:.6e}")
