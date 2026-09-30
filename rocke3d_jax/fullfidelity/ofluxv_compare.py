"""Validate ofluxv_ff.py's OFLUXV port against real Fortran dumps (D43)."""
import sys
import numpy as np
from ofluxv_ff import ofluxv, IM, JM, LMO
from odhorz0_compare import load_geom as load_lmm, load_lmv, FF_DEFAULT
from odhorz_compare import load_lmu

FF_DEFAULT = FF_DEFAULT  # re-exported


def load_record(path):
    raw = np.fromfile(path, dtype='>f8')
    n2 = IM * JM
    n3 = IM * JM * LMO
    expected = 2 + 2 * n2 + 6 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    dtolf = raw[off]; off += 1

    def next2d():
        nonlocal off
        a = np.zeros((IM + 1, JM + 1))
        a[1:, 1:] = raw[off:off + n2].reshape(JM, IM).T
        off += n2
        return a

    def next3d():
        nonlocal off
        a = np.zeros((IM + 1, JM + 1, LMO + 1))
        a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0)
        off += n3
        return a

    rec = {"itime": itime, "dtolf": dtolf}
    rec["opbot0"] = next2d()
    rec["opress0"] = next2d()
    for name in ["mo0", "uo0", "vo0", "mo1", "uo1", "vo1"]:
        rec[name] = next3d()
    assert off == raw.size
    return rec


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    rec = load_record(f"{ff}/ffz_ofluxv_{itime}.bin")

    mo, uo, vo = ofluxv(lmm, lmu, lmv, rec["dtolf"], rec["opbot0"], rec["opress0"],
                        rec["mo0"], rec["uo0"], rec["vo0"])

    results = {}
    for name, computed, expected in [("MO", mo, rec["mo1"]), ("UO", uo, rec["uo1"]),
                                      ("VO", vo, rec["vo1"])]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6))
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
