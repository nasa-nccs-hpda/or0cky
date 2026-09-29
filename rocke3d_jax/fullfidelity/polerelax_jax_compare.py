"""Validate polerelax_jax.py's batched polar relax port against real Fortran dumps (D39), and
cross-check it against the plain-Python reference (polerelax_ff.py)."""
import sys
import numpy as np
from polerelax_ff import polerelax, IM, JM, LMO
from polerelax_jax import polerelax_jax
from polerelax_compare import load_geom, load_record, FF_DEFAULT


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmu, lmv = load_geom(f"{ff}/ffz_polerelax_geom.bin")
    rec = load_record(f"{ff}/ffz_polerelax_{itime}.bin")

    uo_ref, vo_ref, uod_ref, vod_ref = polerelax(
        lmu, lmv, rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"])

    args0 = [to0_2d(lmu).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_3d(rec["uo0"]), to0_3d(rec["vo0"]), to0_3d(rec["uod0"]), to0_3d(rec["vod0"])]
    uo_j, vo_j, uod_j, vod_j = polerelax_jax(*args0)

    results = {}
    for name, jaxval, expected1 in [
        ("UO", uo_j, rec["uo1"]), ("VO", vo_j, rec["vo1"]),
        ("UOD", uod_j, rec["uod1"]), ("VOD", vod_j, rec["vod1"]),
    ]:
        expected0 = to0_3d(expected1)
        diff = np.abs(np.asarray(jaxval) - expected0)
        ok = np.allclose(np.asarray(jaxval), expected0, atol=1e-9, rtol=1e-9)
        results[name] = (diff.max(), ok)

    xcheck = {}
    for name, jaxval, refval in [
        ("UO", uo_j, uo_ref), ("VO", vo_j, vo_ref),
        ("UOD", uod_j, uod_ref), ("VOD", vod_j, vod_ref),
    ]:
        ref0 = to0_3d(refval)
        diff = np.abs(np.asarray(jaxval) - ref0)
        xcheck[name] = diff.max()

    return results, xcheck


if __name__ == "__main__":
    dates = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
    all_ok = True
    for date, itime in dates:
        results, xcheck = run_one(date, itime)
        print(f"== {date} (itime={itime}) ==")
        for name, (maxdiff, ok) in results.items():
            status = "OK" if ok else "FAIL"
            print(f"  {name} vs Fortran: max_abs_diff={maxdiff:.3e}  [{status}]")
            all_ok = all_ok and ok
        for name, maxdiff in xcheck.items():
            print(f"  {name} jax vs plain-python: max_abs_diff={maxdiff:.3e}")
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)
