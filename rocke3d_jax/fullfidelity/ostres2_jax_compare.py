"""Validate ostres2_jax.py's batched OSTRES2 port against real Fortran dumps (D36), and
cross-check it against the plain-Python reference (ostres2_ff.py)."""
import sys
import numpy as np
import jax.numpy as jnp
from ostres2_ff import ostres2, IM, JM
from ostres2_jax import ostres2_jax
from ostres2_compare import load_geom, load_record, FF_DEFAULT


def to0(a1indexed):
    """Convert a 1-indexed (IM+1,JM+1) array to 0-indexed (IM,JM)."""
    return np.asarray(a1indexed[1:, 1:], dtype=np.float64)


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    geom = load_geom(f"{ff}/ffz_ostres2_geom.bin")
    rec = load_record(f"{ff}/ffz_ostres2_{itime}.bin")

    # Reference (plain Python, 1-indexed)
    uo_ref, vo_ref, uod_ref, vod_ref = ostres2(
        geom["lmu"], geom["lmv"], rec["dmua"], rec["dmva"], rec["dmui"], rec["dmvi"],
        rec["mo1"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], ivnp=geom["ivnp"])

    # JAX (0-indexed)
    args0 = [jnp.asarray(to0(geom["lmu"]).astype(np.float64)),
             jnp.asarray(to0(geom["lmv"]).astype(np.float64)),
             jnp.asarray(to0(rec["dmua"])), jnp.asarray(to0(rec["dmva"])),
             jnp.asarray(to0(rec["dmui"])), jnp.asarray(to0(rec["dmvi"])),
             jnp.asarray(to0(rec["mo1"])), jnp.asarray(to0(rec["uo0"])),
             jnp.asarray(to0(rec["vo0"])), jnp.asarray(to0(rec["uod0"])),
             jnp.asarray(to0(rec["vod0"]))]
    ivnp0 = geom["ivnp"] - 1
    uo_j, vo_j, uod_j, vod_j = ostres2_jax(*args0, ivnp0)

    results = {}
    for name, jaxval, expected1idx in [
        ("UO", uo_j, rec["uo1"]), ("VO", vo_j, rec["vo1"]),
        ("UOD", uod_j, rec["uod1"]), ("VOD", vod_j, rec["vod1"]),
    ]:
        expected0 = to0(expected1idx)
        diff = np.abs(np.asarray(jaxval) - expected0)
        ok = np.allclose(np.asarray(jaxval), expected0, atol=1e-9, rtol=1e-9)
        results[name] = (diff.max(), ok)

    # cross-check jax vs plain-python reference too
    xcheck = {}
    for name, jaxval, refval in [
        ("UO", uo_j, uo_ref), ("VO", vo_j, vo_ref),
        ("UOD", uod_j, uod_ref), ("VOD", vod_j, vod_ref),
    ]:
        ref0 = to0(refval)
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
