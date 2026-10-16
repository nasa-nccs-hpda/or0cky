"""Compare seaice_core_ff.prec_si / seaice_core_jax.prec_si against real-Fortran PRECIP_SI dumps
(ffw_<itime>.bin, 40 doubles, Stage 1 of the DYNSI/ocean port, D26).

Layout (0-based Python cols): 0:i 1:j 2:dtsrc 3:snow(in) 4:msi2(in) 5:9=hsil(in) 9:13=ssil(in) 13:prcp 14:enrgp
-- after PREC_SI -- 15:snow 16:msi2 17:21=hsil 21:25=ssil 25:29=tsil 29:run0 30:srun0 31:erun0 32:wetsnow 33:cmprs
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S

NREC = 40


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    return S.prec_si(r[3], r[4], list(r[5:9]), list(r[9:13]), r[13], r[14])


def compare_row(r):
    out = run_row(r)
    ref = dict(snow=r[15], msi2=r[16], hsil=r[17:21], ssil=r[21:25], tsil=r[25:29], run0=r[29], srun0=r[30],
               erun0=r[31], wetsnow=r[32] > 0.5, cmprs=r[33])
    rows = {}
    for k in ("snow", "msi2", "run0", "srun0", "erun0", "cmprs"):
        rows[k] = dict(got=out[k], ref=ref[k], abs=abs(out[k] - ref[k]))
    rows["hsil"] = dict(abs=float(np.max(np.abs(np.array(out["hsil"]) - ref["hsil"]))))
    rows["ssil"] = dict(abs=float(np.max(np.abs(np.array(out["ssil"]) - ref["ssil"]))))
    rows["tsil"] = dict(abs=float(np.max(np.abs(np.array(out["tsil"]) - ref["tsil"]))))
    rows["wetsnow_match"] = bool(out["wetsnow"] == ref["wetsnow"])
    return rows


if __name__ == "__main__":
    rec = load(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    idx = np.linspace(0, len(rec) - 1, min(n, len(rec))).astype(int)
    for i in idx:
        rows = compare_row(rec[i])
        print(f"row {i}:")
        for k, v in rows.items():
            if k == "wetsnow_match":
                print(f"  {k}: {v}")
            else:
                extra = f" got={v['got']:.6e} ref={v['ref']:.6e}" if "got" in v else ""
                print(f"  {k:8s} abs={v['abs']:.3e}{extra}")
