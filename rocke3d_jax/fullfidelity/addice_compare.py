"""Compare seaice_core_ff.addice against real-Fortran FORM_SI/ADDICE dumps (ffn_<itime>.bin, 40 doubles).

Layout (1-based Fortran offsets): 1:i 2:j 3:domain(1=OCEAN) 4:snow 5:roice 6:9=hsil 10:13=ssil 14:msi2
15:enrgfo 16:acefi 17:enrgfi 18:acefo 19:salto 20:salti 21:flead 22:qfixr
-- after ADDICE -- 23:snow 24:roice 25:28=hsil 29:32=ssil 33:msi2 34:dmimp 35:dhimp 36:dsimp
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S

NREC = 40


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    snow, roice = r[3], r[4]
    hsil, ssil = list(r[5:9]), list(r[9:13])
    msi2 = r[13]
    enrgfo, acefi, enrgfi, acefo, salto, salti = r[14], r[15], r[16], r[17], r[18], r[19]
    flead = r[20]
    qfixr = r[21] > 0.5
    return S.addice(snow, roice, hsil, ssil, msi2, enrgfo, acefo, acefi, enrgfi, salto, salti, flead, qfixr)


def compare_row(r):
    out = run_row(r)
    ref = dict(snow=r[22], roice=r[23], hsil=r[24:28], ssil=r[28:32], msi2=r[32],
               dmimp=r[33], dhimp=r[34], dsimp=r[35])
    rows = {}
    for k in ("snow", "roice", "msi2", "dmimp", "dhimp", "dsimp"):
        rows[k] = dict(got=out[k], ref=ref[k], abs=abs(out[k] - ref[k]))
    rows["hsil"] = dict(abs=float(np.max(np.abs(np.array(out["hsil"]) - ref["hsil"]))),
                        ref=float(np.max(np.abs(ref["hsil"]))))
    rows["ssil"] = dict(abs=float(np.max(np.abs(np.array(out["ssil"]) - ref["ssil"]))),
                        ref=float(np.max(np.abs(ref["ssil"]))))
    return rows
