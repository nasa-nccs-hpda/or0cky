"""
P2SAoM40: JAX vs. Real Compiled Fortran (CPU), Apples-to-Apples
==================================================================

Unlike benchmark_all.py / benchmark_all_cpu.py (which benchmark JAX against
a NumPy re-implementation as a stand-in for "Fortran-like" behavior), this
script times the JAX modules against *actual compiled Fortran* -- the same
algorithms (DRYCNV dry convection, PBL similarity), same P2SAoM40 grid
shapes, same iteration count (100), same CPU, same ifort toolchain used to
build this project's real ModelE2/P2SAoM40 GCM.

Fortran side: timing_drycnv_p2saom40.f90 / timing_pbl_p2saom40.f90 in this
directory. Each internally loops 100 calls with system_clock and prints
"MEAN_SECONDS <value>" -- the mean wall time per call. Grid dims (im=72,
jm=46, lm=40) are hardcoded there to match this project's real P2SAoM40
restart file dimensions (see benchmark_all_p2saom40.py for how those were
confirmed).

This is a *performance* comparison, not a numerical cross-check -- the
Fortran and JAX sides use independent random draws over the same value
ranges, not identical input arrays. For existing numerical
validation of the underlying math (find_dpsim/find_dpsih against
gfortran-compiled references), see compare_fortran_jax.py and FINDINGS.md.

Usage:
    python benchmark_p2saom40_vs_fortran.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # This node has no GPU (nvidia-smi absent)

import subprocess
import sys
import time

import jax
import numpy as np

sys.path.insert(0, '.')
from drycnv import dry_convection_mixing_jit
from pbl import simil_jit

HERE = os.path.dirname(os.path.abspath(__file__))
IM, JM, LM = 72, 46, 40  # P2SAoM40 native grid -- see benchmark_all_p2saom40.py
ITERATIONS = 100         # must match the loop count baked into the .f90 drivers


def ensure_fortran_binary(name):
    """Compile <name>.f90 with ifort -O2 if the binary doesn't exist yet."""
    src = os.path.join(HERE, f"{name}.f90")
    exe = os.path.join(HERE, name)
    if os.path.exists(exe):
        return exe
    print(f"Compiling {name}.f90 with ifort -O2 ...")
    result = subprocess.run(
        ["ifort", "-O2", "-o", exe, src],
        cwd=HERE, capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(exe):
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"Failed to compile {name}.f90 -- is `module load intel/2020Update4` "
            f"active in this shell?"
        )
    return exe


def run_fortran_timing(exe_path):
    """Run a compiled timing driver and parse its MEAN_SECONDS line."""
    result = subprocess.run([exe_path], cwd=HERE, capture_output=True, text=True, timeout=120)
    for line in result.stdout.splitlines():
        if line.strip().startswith("MEAN_SECONDS"):
            return float(line.split()[1])
    raise RuntimeError(f"Could not parse MEAN_SECONDS from {exe_path} output:\n{result.stdout}")


def time_jax_drycnv():
    key = jax.random.PRNGKey(42)
    T = jax.random.uniform(key, (IM, JM, LM), minval=200.0, maxval=300.0)
    Q = jax.random.uniform(key, (IM, JM, LM), minval=0.0, maxval=0.02)
    PK = jax.random.uniform(key, (IM, JM, LM), minval=0.5, maxval=1.0)
    PDSIG = jax.random.uniform(key, (IM, JM, LM), minval=0.1, maxval=0.2)

    _ = dry_convection_mixing_jit(T, Q, PK, PDSIG)  # warm-up / compile

    start = time.time()
    for _ in range(ITERATIONS):
        _ = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    return (time.time() - start) / ITERATIONS


def time_jax_pbl():
    n = IM * JM
    key = jax.random.PRNGKey(42)
    z = jax.random.uniform(key, (n,), minval=1.0, maxval=100.0)
    z0m = jax.random.uniform(key, (n,), minval=0.01, maxval=0.1)
    z0h = jax.random.uniform(key, (n,), minval=0.01, maxval=0.1)
    z0q = jax.random.uniform(key, (n,), minval=0.01, maxval=0.1)
    lmonin = jax.random.uniform(key, (n,), minval=-100.0, maxval=100.0)
    ustar = jax.random.uniform(key, (n,), minval=0.1, maxval=1.0)
    tstar = jax.random.uniform(key, (n,), minval=0.1, maxval=1.0)
    qstar = jax.random.uniform(key, (n,), minval=0.01, maxval=0.1)
    tg = jax.random.uniform(key, (n,), minval=280.0, maxval=320.0)
    qg = jax.random.uniform(key, (n,), minval=0.01, maxval=0.05)

    _ = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)  # warm-up

    start = time.time()
    for _ in range(ITERATIONS):
        _ = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)
    return (time.time() - start) / ITERATIONS


def report(name, jax_time, fortran_time):
    print(f"\n{'=' * 60}")
    print(f"{name}  (P2SAoM40 grid: im={IM}, jm={JM}, lm={LM})")
    print(f"{'=' * 60}")
    print(f"JAX (CPU) mean time:     {jax_time:.6e} s/call")
    print(f"Fortran (ifort -O2) time: {fortran_time:.6e} s/call")
    ratio = fortran_time / jax_time
    if ratio >= 1.0:
        print(f"JAX is {ratio:.2f}x faster than compiled Fortran")
    else:
        print(f"Compiled Fortran is {1.0 / ratio:.2f}x faster than JAX")


def main():
    print("=" * 60)
    print("P2SAoM40: JAX vs. Real Compiled Fortran (CPU), apples-to-apples")
    print("=" * 60)
    print(f"Grid: im={IM}, jm={JM}, lm={LM}  |  iterations={ITERATIONS}")

    drycnv_exe = ensure_fortran_binary("timing_drycnv_p2saom40")
    pbl_exe = ensure_fortran_binary("timing_pbl_p2saom40")

    jax_drycnv_time = time_jax_drycnv()
    fortran_drycnv_time = run_fortran_timing(drycnv_exe)
    report("DRYCNV (Dry Convection)", jax_drycnv_time, fortran_drycnv_time)

    jax_pbl_time = time_jax_pbl()
    fortran_pbl_time = run_fortran_timing(pbl_exe)
    report("PBL (Planetary Boundary Layer)", jax_pbl_time, fortran_pbl_time)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
