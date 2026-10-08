"""Validate ocoast_ff.py's OCOAST port against real Fortran dumps (D37).

Usage: python3 ocoast_compare.py
"""
import sys
import numpy as np
from ocoast_ff import ocoast, IM, JM, LMO

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")


def load_geom(path):
    raw = np.fromfile(path, dtype='>f8')
    im_r, jm_r = raw[0:2].astype(int)
    assert (im_r, jm_r) == (IM, JM)
    lmm_flat = raw[2:2 + IM * JM]
    assert 2 + IM * JM == raw.size
    lmm = np.zeros((IM + 1, JM + 1), dtype=int)
    lmm[1:, 1:] = lmm_flat.reshape(JM, IM).T.astype(int)
    return lmm


def load_record(path):
    raw = np.fromfile(path, dtype='>f8')
    n = IM * JM * LMO
    assert raw.size == 1 + 8 * n, raw.size
    itime = raw[0]
    off = 1
    names = ["gxmo0", "sxmo0", "gymo0", "symo0", "gxmo1", "sxmo1", "gymo1", "symo1"]
    out = {"itime": itime}
    for name in names:
        flat = raw[off:off + n]; off += n
        a = np.zeros((IM + 1, JM + 1, LMO + 1))
        # Fortran (IM,JM,LMO) column-major -> reshape (LMO,JM,IM) then transpose to (IM,JM,LMO)
        a[1:, 1:, 1:] = flat.reshape(LMO, JM, IM).transpose(2, 1, 0)
        out[name] = a
    return out


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_ocoast_geom.bin")
    rec = load_record(f"{ff}/ffz_ocoast_{itime}.bin")

    gxmo, sxmo, gymo, symo = ocoast(lmm, rec["gxmo0"], rec["sxmo0"], rec["gymo0"], rec["symo0"])

    results = {}
    for name, computed, expected in [
        ("GXMO", gxmo, rec["gxmo1"]), ("SXMO", sxmo, rec["sxmo1"]),
        ("GYMO", gymo, rec["gymo1"]), ("SYMO", symo, rec["symo1"]),
    ]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-9, rtol=1e-9))
    return results


if __name__ == "__main__":
    dates = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
    all_ok = True
    for date, itime in dates:
        results = run_one(date, itime)
        print(f"== {date} (itime={itime}) ==")
        for name, (maxdiff, ok) in results.items():
            status = "OK" if ok else "FAIL"
            print(f"  {name}: max_abs_diff={maxdiff:.3e}  [{status}]")
            all_ok = all_ok and ok
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)
