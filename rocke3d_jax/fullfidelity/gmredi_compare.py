"""Validate gmredi_ff.py's isoslope4 port against real Fortran dumps (D49)."""
import sys
import numpy as np
from gmredi_ff import isoslope4, IM, JM, LMO
from ocnmeso_compare import load_densgrad, FF_DEFAULT
from odhorz_compare import load_lmm

ISOSLOPE4_FIELDS = ["aix0", "aix1", "aix2", "aix3", "aiy0", "aiy1", "aiy2", "aiy3",
                     "asx0", "asx1", "asx2", "asx3", "asy0", "asy1", "asy2", "asy3",
                     "s2x0", "s2x1", "s2x2", "s2x3", "s2y0", "s2y1", "s2y2", "s2y3"]


def _to_ijl_standard(flat):
    a = np.zeros((IM + 1, JM + 1, LMO + 1))
    a[1:, 1:, 1:] = flat.reshape(LMO, JM, IM).transpose(2, 1, 0)
    return a


def load_isoslope4(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    expected = 1 + 24 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    rec = {"itime": itime}
    for name in ISOSLOPE4_FIELDS:
        rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
    assert off == raw.size
    return rec


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    expected = load_isoslope4(f"{ff}/ffz_isoslope4_{itime}.bin")

    k3d = np.where(np.arange(LMO + 1)[None, None, :] <= lmm[:, :, None], 800.0, 0.0)
    k3d[:, :, 0] = 0.0

    computed = isoslope4(lmm, dg["rhox"], dg["rhoy"], dg["rhomz"], dg["byrhoz"],
                          dg["bydh"], dg["dzv"], k3d)

    results = {}
    for name in ISOSLOPE4_FIELDS:
        diff = np.abs(computed[name] - expected[name])
        results[name.upper()] = (diff.max(),
                                  np.allclose(computed[name], expected[name], atol=1e-6, rtol=1e-6))
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
