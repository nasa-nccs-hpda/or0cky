"""Compare seaice_core_ff.simelt against real-Fortran MELT_SI/SIMELT dumps (ffm_<itime>.bin, 30 doubles).

Layout (1-based Fortran offsets): 1:i 2:j 3:dt 4:roice 5:snow 6:msi2 7:10=hsil 11:14=ssil 15:pocean
16:tm 17:tfo 18:enrgmax
-- after SIMELT -- 19:roice 20:snow 21:msi2 22:25=hsil 26:29=ssil 30:enrgused
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S

NREC = 30


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    dt, roice, snow, msi2 = r[2], r[3], r[4], r[5]
    hsil, ssil = list(r[6:10]), list(r[10:14])
    pocean, tm, tfo, enrgmax = r[14], r[15], r[16], r[17]
    return S.simelt(dt, roice, snow, msi2, hsil, ssil, pocean, tm, tfo, enrgmax)


def compare_row(r):
    out = run_row(r)
    ref = dict(roice=r[18], snow=r[19], msi2=r[20], hsil=r[21:25], ssil=r[25:29], enrgused=r[29])
    rows = {}
    for k in ("roice", "snow", "msi2", "enrgused"):
        rows[k] = dict(got=out[k], ref=ref[k], abs=abs(out[k] - ref[k]))
    rows["hsil"] = dict(abs=float(np.max(np.abs(np.array(out["hsil"]) - ref["hsil"]))),
                        ref=float(np.max(np.abs(ref["hsil"]))))
    rows["ssil"] = dict(abs=float(np.max(np.abs(np.array(out["ssil"]) - ref["ssil"]))),
                        ref=float(np.max(np.abs(ref["ssil"]))))
    rows["melted_out"] = bool(out["roice"] == 0.0)
    return rows
