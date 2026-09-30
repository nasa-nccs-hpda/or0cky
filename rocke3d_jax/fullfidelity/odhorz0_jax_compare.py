"""Validate odhorz0_jax.py's batched ODHORZ0 port against real Fortran dumps (D40), and
cross-check it against the plain-Python reference (odhorz0_ff.py)."""
import sys
import numpy as np
from odhorz0_ff import odhorz0, IM, JM, LMO
from odhorz0_jax import odhorz0_jax
from odhorz0_compare import load_geom, load_lmv, load_record, FF_DEFAULT


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    rec = load_record(f"{ff}/ffz_odhorz0_{itime}.bin")

    ref = odhorz0(lmm, lmv, rec["opress"], rec["g0m"], rec["gzm"], rec["s0m"], rec["szm"],
                  rec["mo0"], rec["uo0"], rec["vo0"], rec["vup"], rec["vdn"])

    args0 = [to0_2d(lmm).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_2d(rec["opress"]), to0_3d(rec["g0m"]), to0_3d(rec["gzm"]),
             to0_3d(rec["s0m"]), to0_3d(rec["szm"]), to0_3d(rec["mo0"]),
             to0_3d(rec["uo0"]), to0_3d(rec["vo0"]), to0_3d(rec["vup"]), to0_3d(rec["vdn"])]
    out = odhorz0_jax(*args0)

    results = {}
    dump_key = {"opbot": "opbot", "gup": "gup", "gdn": "gdn", "sup": "sup", "sdn": "sdn",
                "dzgdp": "dzgdp", "vbar": "vbar", "dh3d": "dh3d", "mo": "mo1", "uo": "uo1", "vo": "vo1"}
    for name, dkey in dump_key.items():
        jaxval = np.asarray(out[name])
        expected0 = to0_2d(rec[dkey]) if rec[dkey].ndim == 2 else to0_3d(rec[dkey])
        diff = np.abs(jaxval - expected0)
        ok = np.allclose(jaxval, expected0, atol=1e-6, rtol=1e-6)
        results[name] = (diff.max(), ok)

    xcheck = {}
    for name in dump_key:
        ref0 = to0_2d(ref[name]) if ref[name].ndim == 2 else to0_3d(ref[name])
        diff = np.abs(np.asarray(out[name]) - ref0)
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
