"""Compare seaice_to_atmgrid_ff.seaice_to_atmgrid_cell against the real Fortran ffz_s2ag_<itime>.bin
dump (D31). Records come from several call sites per step (ATM_DRV.f/OCN_DRV.f/SURFACE.f), all the
same pure function -- validated as one pool, no need to distinguish which site produced a record.
"""
import sys
import numpy as np

from seaice_to_atmgrid_ff import seaice_to_atmgrid_cell


def read_s2ag(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 20)
    return dict(i=raw[:, 0], j=raw[:, 1], ncall=raw[:, 2], rsi=raw[:, 3], snowi=raw[:, 4],
                msi=raw[:, 5], hsi1=raw[:, 6], hsi2=raw[:, 7], ssi1=raw[:, 8], ssi2=raw[:, 9],
                ssi3=raw[:, 10], ssi4=raw[:, 11], gtemp=raw[:, 12], gtemp2=raw[:, 13],
                gtempr=raw[:, 14], zsnowi=raw[:, 15], zsi=raw[:, 16], fwsim=raw[:, 17])


def main(path):
    d = read_s2ag(path)
    n = len(d["i"])
    print(f"{n} records")

    # gtemp/gtemp2/gtempr: absolute tolerance (degrees C) -- the real Fortran computes Ti/Ti2b in
    # REAL*16 internally (SEAICE_FIXES_2022), only the final REAL*8 result is returned, so tiny
    # near-zero temperatures can show large *relative* error from ordinary quad-vs-double rounding
    # (same documented approximation as GROUND_SI's TSIL field, D26). zsnowi/zsi/fwsim are plain
    # multiplies with no such precision gap, so relative tolerance is fine there.
    worst_abs = {"gtemp": 0.0, "gtemp2": 0.0, "gtempr": 0.0}
    worst_rel = {"zsnowi": 0.0, "zsi": 0.0, "fwsim": 0.0}

    for idx in range(n):
        out = seaice_to_atmgrid_cell(d["rsi"][idx], d["snowi"][idx], d["msi"][idx], d["hsi1"][idx],
                                      d["hsi2"][idx], d["ssi1"][idx], d["ssi2"][idx], d["ssi3"][idx],
                                      d["ssi4"][idx])
        for k in worst_abs:
            err = abs(out[k] - d[k][idx])
            if err > worst_abs[k]:
                worst_abs[k] = err
        for k in worst_rel:
            ref = d[k][idx]
            rel = abs(out[k] - ref) / max(abs(ref), 1e-6)
            if rel > worst_rel[k]:
                worst_rel[k] = rel

    ok = True
    for k, v in worst_abs.items():
        status = "OK" if v < 1e-6 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{k:8s} max_abs_err={v:.3e} (deg C) {status}")
    for k, v in worst_rel.items():
        status = "OK" if v < 1e-8 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{k:8s} max_relerr={v:.3e} {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffz_s2ag_33312.bin"
    sys.exit(main(path))
