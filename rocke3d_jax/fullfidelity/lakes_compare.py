"""Compare lakes_ff (LKSOURC/LKMIX) against real-Fortran GROUND_LK dumps (ffl2_<itime>.bin, 40 doubles).

Layout (1-based Fortran offsets): 1:i 2:j 3:roice 4:5=mlake(pre) 6:7=elake(pre) 8:run0 9:fodt 10:fidt
11:12=srox 13:fsr2 14:evapo 15:hlake 16:dtsrc
-- after LKSOURC -- 17:18=mlake 19:20=elake 21:enrgfo 22:acefo 23:acefi 24:enrgfi
-- after LKMIX -- 25:26=mlake 27:28=elake 29:tke
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lakes_ff as L

NREC = 40


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    roice = r[2]
    mlake0, elake0 = [r[3], r[4]], [r[5], r[6]]
    run0, fodt, fidt = r[7], r[8], r[9]
    srox = [r[10], r[11]]
    fsr2, evapo, hlake, dtsrc = r[12], r[13], r[14], r[15]
    src = L.lksourc_full(roice, mlake0, elake0, run0, fodt, fidt, srox, fsr2, evapo)
    tke = 0.0   # real model passes TKE=0. always (commented-out U2rho term)
    mix = L.lkmix(src["mlake"], src["elake"], hlake, tke, roice, dtsrc)
    return src, mix


def compare_row(r):
    src, mix = run_row(r)
    ref_src = dict(mlake=r[16:18], elake=r[18:20], enrgfo=r[20], acefo=r[21], acefi=r[22], enrgfi=r[23])
    ref_mix = dict(mlake=r[24:26], elake=r[26:28])
    rows = {}
    for k in ("enrgfo", "acefo", "acefi", "enrgfi"):
        rows[k] = dict(got=src[k], ref=ref_src[k], abs=abs(src[k] - ref_src[k]))
    rows["mlake_src"] = dict(abs=float(np.max(np.abs(np.array(src["mlake"]) - ref_src["mlake"]))),
                             ref=float(np.max(np.abs(ref_src["mlake"]))))
    rows["elake_src"] = dict(abs=float(np.max(np.abs(np.array(src["elake"]) - ref_src["elake"]))),
                             ref=float(np.max(np.abs(ref_src["elake"]))))
    rows["mlake_mix"] = dict(abs=float(np.max(np.abs(np.array(mix["mlake"]) - ref_mix["mlake"]))),
                             ref=float(np.max(np.abs(ref_mix["mlake"]))))
    rows["elake_mix"] = dict(abs=float(np.max(np.abs(np.array(mix["elake"]) - ref_mix["elake"]))),
                             ref=float(np.max(np.abs(ref_mix["elake"]))))
    return rows
