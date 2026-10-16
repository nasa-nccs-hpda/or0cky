"""Compare seaice_core_ff.sea_ice against real-Fortran GROUND_SI records (ffi_<itime>.bin, 60 doubles).

Record layout (1-based Fortran offsets, see instrumentation/SEAICE_DRV.f.patch):
1:i 2:j 3:domain(1=OCEAN,0=other) 4:dtsrc 5:snow 6:roice 7:10=hsil(1:4) 11:14=ssil(1:4) 15:msi2
16:f0dt 17:f1dt 18:evap 19:srox0 20:fmoc 21:fhoc 22:fsoc 23:wetsnow(in) 24:tm 25:sm
-- after SEA_ICE --
26:snow 27:30=hsil 31:34=ssil 35:msi2 36:run 37:erun 38:srun 39:wetsnow(out) 40:melt12 41:cmprs 42:srox2
-- after SSIDEC / final --
43:snow 44:47=hsil 48:51=ssil 52:msi2 53:runosi 54:erunosi 55:srunosi 56:mflux 57:hflux 58:sflux
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S

NREC = 60


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def run_row(r):
    dtsrc, snow, roice = r[3], r[4], r[5]
    hsil, ssil = list(r[6:10]), list(r[10:14])
    msi2 = r[14]
    f0dt, f1dt, evap, srox0 = r[15], r[16], r[17], r[18]
    fmoc, fhoc, fsoc = r[19], r[20], r[21]
    wetsnow = r[22] > 0.5
    out = S.sea_ice(dtsrc, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)
    return out


def compare_row(r):
    out = run_row(r)
    ref = dict(snow=r[25], hsil=r[26:30], ssil=r[30:34], msi2=r[34], run=r[35], erun=r[36], srun=r[37],
               wetsnow=r[38] > 0.5, melt12=r[39], cmprs=r[40], srox2=r[41])
    rows = {}
    for k in ("snow", "msi2", "run", "erun", "srun", "melt12", "cmprs", "srox2"):
        rows[k] = dict(got=out[k], ref=ref[k], abs=abs(out[k] - ref[k]))
    rows["hsil"] = dict(abs=float(np.max(np.abs(np.array(out["hsil"]) - ref["hsil"]))))
    rows["ssil"] = dict(abs=float(np.max(np.abs(np.array(out["ssil"]) - ref["ssil"]))))
    rows["wetsnow_match"] = bool(out["wetsnow"] == ref["wetsnow"])
    return rows


def run_full(r):
    """Full GROUND_SI (SEA_ICE + SSIDEC/snowice per domain), matching cols 43-58 (final state)."""
    dtsrc, snow, roice = r[3], r[4], r[5]
    hsil, ssil = list(r[6:10]), list(r[10:14])
    msi2 = r[14]
    f0dt, f1dt, evap, srox0 = r[15], r[16], r[17], r[18]
    fmoc, fhoc, fsoc = r[19], r[20], r[21]
    wetsnow = r[22] > 0.5
    tm, sm = r[23], r[24]
    if r[2] > 0.5:
        return S.ground_si_ocean(dtsrc, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc,
                                 wetsnow, tm, sm)
    return S.ground_si_other(dtsrc, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)


def compare_full(r):
    out = run_full(r)
    ref = dict(snow=r[42], hsil=r[43:47], ssil=r[47:51], msi2=r[51], runosi=r[52], erunosi=r[53],
               srunosi=r[54])
    rows = {}
    for k in ("snow", "msi2", "runosi", "erunosi", "srunosi"):
        rows[k] = dict(got=out[k], ref=ref[k], abs=abs(out[k] - ref[k]))
    rows["hsil"] = dict(abs=float(np.max(np.abs(np.array(out["hsil"]) - ref["hsil"]))),
                        ref=float(np.max(np.abs(ref["hsil"]))))
    rows["ssil"] = dict(abs=float(np.max(np.abs(np.array(out["ssil"]) - ref["ssil"]))),
                        ref=float(np.max(np.abs(ref["ssil"]))))
    return rows


if __name__ == "__main__":
    rec = load(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    idx = np.linspace(0, len(rec) - 1, min(n, len(rec))).astype(int)
    for i in idx:
        rows = compare_row(rec[i])
        print(f"row {i} domain={'OCEAN' if rec[i,2]>0.5 else 'other'}:")
        for k, v in rows.items():
            if k == "wetsnow_match":
                print(f"  {k}: {v}")
            else:
                extra = f" got={v['got']:.6e} ref={v['ref']:.6e}" if "got" in v else ""
                print(f"  {k:8s} abs={v['abs']:.3e}{extra}")
