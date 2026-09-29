"""Validate ocoast_jax.py's batched OCOAST port against real Fortran dumps (D37), and
cross-check it against the plain-Python reference (ocoast_ff.py)."""
import sys
import numpy as np
from ocoast_ff import ocoast, IM, JM, LMO
from ocoast_jax import ocoast_jax
from ocoast_compare import load_geom, load_record, FF_DEFAULT


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_ocoast_geom.bin")
    rec = load_record(f"{ff}/ffz_ocoast_{itime}.bin")

    gxmo_ref, sxmo_ref, gymo_ref, symo_ref = ocoast(
        lmm, rec["gxmo0"], rec["sxmo0"], rec["gymo0"], rec["symo0"])

    lmm0 = to0_2d(lmm).astype(np.float64)
    args0 = [lmm0, to0_3d(rec["gxmo0"]), to0_3d(rec["sxmo0"]),
             to0_3d(rec["gymo0"]), to0_3d(rec["symo0"])]
    gxmo_j, sxmo_j, gymo_j, symo_j = ocoast_jax(*args0)

    results = {}
    for name, jaxval, expected1 in [
        ("GXMO", gxmo_j, rec["gxmo1"]), ("SXMO", sxmo_j, rec["sxmo1"]),
        ("GYMO", gymo_j, rec["gymo1"]), ("SYMO", symo_j, rec["symo1"]),
    ]:
        expected0 = to0_3d(expected1)
        diff = np.abs(np.asarray(jaxval) - expected0)
        ok = np.allclose(np.asarray(jaxval), expected0, atol=1e-9, rtol=1e-9)
        results[name] = (diff.max(), ok)

    xcheck = {}
    for name, jaxval, refval in [
        ("GXMO", gxmo_j, gxmo_ref), ("SXMO", sxmo_j, sxmo_ref),
        ("GYMO", gymo_j, gymo_ref), ("SYMO", symo_j, symo_ref),
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
