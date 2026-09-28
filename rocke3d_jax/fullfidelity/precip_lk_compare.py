"""Compare lakes_ff.precip_lk / lakes_core_jax.precip_lk against real-Fortran PRECIP_LK dumps
(ffv_<itime>.bin, 40 doubles, Stage 1 of the DYNSI/ocean port, D27).

Layout (0-based Python cols): 0:i 1:j 2:flake 3:flice 4:rsi 5:prcp 6:enrgp 7:runpsi 8:runo_li 9:melti
10:emelti 11:axyp 12:mwl(in) 13:gml(in) 14:tlake(in) 15:mldlk(in)
-- after PRECIP_LK -- 16:mwl 17:gml 18:tlake 19:mldlk 20:dlake 21:glake 22:gtemp 23:gtemp2 24:gtempr

Note: gtemp/gtemp2/gtempr are only genuinely computed (and so only meaningfully validated) for flake>0
cells -- for flake<=0 (land-ice-only) cells the real Fortran leaves them untouched, and this dump does not
separately capture their pre-call value, so the port's pass-through for that case is correct by inspection
(a bare return of the given value) rather than independently checked here.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lakes_ff as L

NREC = 40


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    return L.precip_lk(r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
                       r[15], 0.0, 0.0, 0.0)


def compare_row(r):
    out = run_row(r)
    ref = dict(mwl=r[16], gml=r[17], tlake=r[18], mldlk=r[19], dlake=r[20], glake=r[21], gtemp=r[22],
               gtemp2=r[23], gtempr=r[24])
    rows = {}
    for k, v in ref.items():
        if k in ("gtemp", "gtemp2", "gtempr") and r[2] <= 0:
            continue
        rows[k] = dict(got=out[k], ref=v, abs=abs(out[k] - v))
    return rows


if __name__ == "__main__":
    rec = load(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    idx = np.linspace(0, len(rec) - 1, min(n, len(rec))).astype(int)
    for i in idx:
        rows = compare_row(rec[i])
        print(f"row {i} flake={rec[i,2]:.3f}:")
        for k, v in rows.items():
            print(f"  {k:8s} abs={v['abs']:.3e} got={v['got']:.6e} ref={v['ref']:.6e}")
