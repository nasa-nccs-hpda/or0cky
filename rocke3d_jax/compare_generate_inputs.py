"""
Generate the ONE shared input dataset used by every leg of the P2SAoM40
Fortran-vs-JAX comparison (Fortran/CPU, JAX/CPU, JAX/GPU).

Grid shapes are P2SAoM40's real dimensions, confirmed from this project's
own P2SAoM40 restart file (im=72, jm=46, lm=40 atmosphere layers) -- see
mantle/benchmark_all_p2saom40.py (historical, moved 2026-09-22) for how those
were originally sourced.

Writes, into ./compare_data/:
  - drycnv_{T,Q,PK,PDSIG}.dat   -- raw float64, Fortran (column-major) order
  - pbl_{z,z0m,z0h,z0q,lmonin,ustar,tstar,qstar,tg,qg}.dat -- same
  - *.npy versions of the same arrays, for the JAX/Python side

Using one fixed seed for both the .dat (Fortran) and .npy (JAX) files means
every leg of the comparison consumes byte-for-byte identical input, so any
difference in outputs is attributable to the implementation, not the inputs.

Usage:
    python compare_generate_inputs.py
"""

import os
import numpy as np

IM, JM, LM = 72, 46, 40  # P2SAoM40 native grid
SEED = 12345
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_data")


def save_both(name, arr):
    """Save as .npy (native C order, for JAX/NumPy) and as raw .dat in
    Fortran (column-major) element order, for the Fortran side to read
    with a plain `read` into an identically-shaped array."""
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    # NOTE: ndarray.tofile() always writes in C order regardless of the
    # array's memory layout -- np.asfortranarray(arr).tofile(...) does NOT
    # give Fortran-order bytes despite appearances. tobytes(order='F') is
    # the one that actually respects the order argument.
    with open(os.path.join(OUT_DIR, f"{name}.dat"), "wb") as f:
        f.write(arr.tobytes(order="F"))


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # DRYCNV inputs: (im, jm, lm), same ranges as benchmark_all*.py
    T = rng.uniform(200.0, 300.0, size=(IM, JM, LM))
    Q = rng.uniform(0.0, 0.02, size=(IM, JM, LM))
    PK = rng.uniform(0.5, 1.0, size=(IM, JM, LM))
    PDSIG = rng.uniform(0.1, 0.2, size=(IM, JM, LM))
    for name, arr in [("drycnv_T", T), ("drycnv_Q", Q), ("drycnv_PK", PK), ("drycnv_PDSIG", PDSIG)]:
        save_both(name, arr)

    # PBL inputs: (im*jm,), same ranges as benchmark_all*.py
    n = IM * JM
    z = rng.uniform(1.0, 100.0, size=n)
    z0m = rng.uniform(0.01, 0.1, size=n)
    z0h = rng.uniform(0.01, 0.1, size=n)
    z0q = rng.uniform(0.01, 0.1, size=n)
    lmonin = rng.uniform(-100.0, 100.0, size=n)
    ustar = rng.uniform(0.1, 1.0, size=n)
    tstar = rng.uniform(0.1, 1.0, size=n)
    qstar = rng.uniform(0.01, 0.1, size=n)
    tg = rng.uniform(280.0, 320.0, size=n)
    qg = rng.uniform(0.01, 0.05, size=n)
    for name, arr in [
        ("pbl_z", z), ("pbl_z0m", z0m), ("pbl_z0h", z0h), ("pbl_z0q", z0q),
        ("pbl_lmonin", lmonin), ("pbl_ustar", ustar), ("pbl_tstar", tstar),
        ("pbl_qstar", qstar), ("pbl_tg", tg), ("pbl_qg", qg),
    ]:
        save_both(name, arr)

    print(f"Wrote shared inputs to {OUT_DIR}")
    print(f"  DRYCNV shape: ({IM}, {JM}, {LM})")
    print(f"  PBL shape:    ({n},)")
    print(f"  Seed: {SEED}")


if __name__ == "__main__":
    main()
