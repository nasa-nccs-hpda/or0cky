"""
P2SAoM40 JAX leg of the Fortran-vs-JAX comparison.

Reads the SAME shared input arrays compare_fortran.f90 reads (written by
compare_generate_inputs.py), runs the JAX DRYCNV/PBL implementations,
times 100 calls (matching the Fortran side's iteration count), and writes
outputs + timing tagged by whichever JAX device is actually active
(cpu/gpu) -- so the same script is used for both the CPU leg (run here)
and the GPU leg (run via compare_submit_gpu.sbatch on a SLURM GPU node).

Usage:
    python compare_jax.py               # uses whatever JAX picks by default
    JAX_PLATFORMS=cpu python compare_jax.py
    JAX_PLATFORMS=cuda python compare_jax.py
"""

import json
import os
import sys
import time

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drycnv import dry_convection_mixing_jit, dry_convection_mixing_lf_jit
from pbl import simil_jit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "compare_data")
ITER = 100


def load(name):
    return np.load(os.path.join(DATA_DIR, f"{name}.npy"))


def main():
    device = jax.devices()[0].platform  # 'cpu' or 'gpu'
    print(f"JAX active device: {device}  ({jax.devices()})")

    # --- DRYCNV ---
    T, Q, PK, PDSIG = load("drycnv_T"), load("drycnv_Q"), load("drycnv_PK"), load("drycnv_PDSIG")
    T_j, Q_j, PK_j, PDSIG_j = (jax.numpy.asarray(x) for x in (T, Q, PK, PDSIG))

    # (a) level-last API (original layout): includes the in-call transposes.
    T_out, Q_out = dry_convection_mixing_jit(T_j, Q_j, PK_j, PDSIG_j)  # warm-up / compile
    jax.block_until_ready((T_out, Q_out))
    start = time.time()
    for _ in range(ITER):
        T_out, Q_out = dry_convection_mixing_jit(T_j, Q_j, PK_j, PDSIG_j)
    jax.block_until_ready((T_out, Q_out))
    drycnv_levellast_time = (time.time() - start) / ITER

    # (b) layer-first (L, J, I): C-order layer-first is byte-for-byte Fortran's (I, J, L)
    # memory layout. The one-time layout change happens here, outside the timed loop,
    # exactly as a port that keeps ModelE's own array layout would never pay for it.
    T_lf, Q_lf, PK_lf, PDSIG_lf = (jax.numpy.moveaxis(x, -1, 0) for x in (T_j, Q_j, PK_j, PDSIG_j))
    T_lf, Q_lf, PK_lf, PDSIG_lf = (jax.numpy.asarray(np.ascontiguousarray(np.asarray(x))) for x in (T_lf, Q_lf, PK_lf, PDSIG_lf))
    T_out_lf, Q_out_lf = dry_convection_mixing_lf_jit(T_lf, Q_lf, PK_lf, PDSIG_lf)  # warm-up / compile
    jax.block_until_ready((T_out_lf, Q_out_lf))
    start = time.time()
    for _ in range(ITER):
        T_out_lf, Q_out_lf = dry_convection_mixing_lf_jit(T_lf, Q_lf, PK_lf, PDSIG_lf)
    jax.block_until_ready((T_out_lf, Q_out_lf))
    drycnv_time = (time.time() - start) / ITER  # headline number (layer-first, device-resident)

    # outputs for the accuracy comparison: from the layer-first path, in the original layout
    np.save(os.path.join(DATA_DIR, f"drycnv_jax_{device}_T_out.npy"), np.moveaxis(np.asarray(T_out_lf), 0, -1))
    np.save(os.path.join(DATA_DIR, f"drycnv_jax_{device}_Q_out.npy"), np.moveaxis(np.asarray(Q_out_lf), 0, -1))

    # --- PBL ---
    names = ["z", "z0m", "z0h", "z0q", "lmonin", "ustar", "tstar", "qstar", "tg", "qg"]
    pbl_np = [load(f"pbl_{n}") for n in names]
    pbl_j = [jax.numpy.asarray(x) for x in pbl_np]

    outs = simil_jit(*pbl_j)  # warm-up / compile
    jax.block_until_ready(outs)

    start = time.time()
    for _ in range(ITER):
        outs = simil_jit(*pbl_j)
    jax.block_until_ready(outs)
    pbl_time = (time.time() - start) / ITER

    for out_name, arr in zip(["u", "t", "q", "dpsim", "dpsih", "dpsiq"], outs):
        np.save(os.path.join(DATA_DIR, f"pbl_jax_{device}_{out_name}_out.npy"), np.asarray(arr))

    timing = {
        "device": device,
        "jax_devices": [str(d) for d in jax.devices()],
        "drycnv_mean_seconds": drycnv_time,
        "drycnv_levellast_api_mean_seconds": drycnv_levellast_time,
        "pbl_mean_seconds": pbl_time,
        "iterations": ITER,
    }
    timing_path = os.path.join(DATA_DIR, f"timing_jax_{device}.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"DRYCNV mean time ({device}, layer-first): {drycnv_time:.6e} s/call")
    print(f"DRYCNV mean time ({device}, level-last API incl. transposes): {drycnv_levellast_time:.6e} s/call")
    print(f"PBL    mean time ({device}): {pbl_time:.6e} s/call")
    print(f"Wrote {timing_path}")


if __name__ == "__main__":
    main()
