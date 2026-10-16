"""Validate gmredi_ff.py's gmkdif port against real Fortran dumps (D50)."""
import sys
import numpy as np
from gmredi_ff import gmkdif, IM, JM, LMO
from gmredi_compare import load_isoslope4, ISOSLOPE4_FIELDS
from ocnmeso_compare import FF_DEFAULT
from odhorz_compare import load_lmm

GMKDIF_FIELDS = ["bxx", "byy", "bzz", "azx", "bzx", "czx", "aezx", "ezx", "cezx",
                  "azy", "bzy", "czy", "aezy", "ezy", "cezy"]


def _to_ijl_standard(flat):
    a = np.zeros((IM + 1, JM + 1, LMO + 1))
    a[1:, 1:, 1:] = flat.reshape(LMO, JM, IM).transpose(2, 1, 0)
    return a


def _to_ij(flat):
    a = np.zeros((IM + 1, JM + 1))
    a[1:, 1:] = flat.reshape(JM, IM).T
    return a


def load_gmkdif(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    n2 = IM * JM
    expected = 1 + n2 + 15 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    rec = {"itime": itime}
    kpl_d = _to_ij(raw[off:off + n2]); off += n2
    rec["kpl"] = np.round(kpl_d).astype(int)
    for name in GMKDIF_FIELDS:
        rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
    assert off == raw.size
    return rec


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    isos = load_isoslope4(f"{ff}/ffz_isoslope4_{itime}.bin")
    expected = load_gmkdif(f"{ff}/ffz_gmkdif_{itime}.bin")

    computed = gmkdif(lmm, expected["kpl"],
                       isos["aix0"], isos["aix1"], isos["aix2"], isos["aix3"],
                       isos["aiy0"], isos["aiy1"], isos["aiy2"], isos["aiy3"],
                       isos["asx0"], isos["asx1"], isos["asx2"], isos["asx3"],
                       isos["asy0"], isos["asy1"], isos["asy2"], isos["asy3"],
                       isos["s2x0"], isos["s2x1"], isos["s2x2"], isos["s2x3"],
                       isos["s2y0"], isos["s2y1"], isos["s2y2"], isos["s2y3"])

    results = {}
    for name in GMKDIF_FIELDS:
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
