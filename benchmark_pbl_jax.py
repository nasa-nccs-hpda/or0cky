"""
Benchmark Script for JAX PBL (Planetary Boundary Layer) Subroutines
======================================================================

This script benchmarks the JAX implementation of ROCKE-3D's PBL subroutines
against a NumPy (Fortran-like) implementation.

Usage:
    python benchmark_pbl_jax.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import time
import numpy as np
import jax
import jax.numpy as jnp
from pbl_jax import simil_jit


def simil_numpy(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg):
    """
    NumPy implementation of simil (Fortran-like, CPU baseline).
    """
    kappa = 0.4
    gamamu = 16.0
    by3 = 1.0 / 3.0
    zetm = -1.0
    
    # Compute zet for momentum, heat, moisture
    zet = z / lmonin
    zet0_m = z0m / lmonin
    zet0_h = z0h / lmonin
    zet0_q = z0q / lmonin
    
    # Compute dpsim (momentum) - vectorized
    x = np.where(1.0 - gamamu * zet >= 0, (1.0 - gamamu * zet) ** 0.25, 0.0)
    x0_m = np.where(1.0 - gamamu * zet0_m >= 0, (1.0 - gamamu * zet0_m) ** 0.25, 0.0)
    
    # Unstable: zet > zetm
    mask_unstable_gt = (zet < 0) & (zet > zetm)
    mask_unstable_le = (zet < 0) & (zet <= zetm)
    
    dpsim = np.zeros_like(zet)
    dpsim = np.where(
        mask_unstable_gt,
        np.log(((1 + x) * (1 + x) * (1 + x * x)) / 
               ((1 + x0_m) * (1 + x0_m) * (1 + x0_m * x0_m))) - 
        2 * (np.arctan(x) - np.arctan(x0_m)),
        np.where(
            mask_unstable_le,
            np.log(((1 + (1 - gamamu * zetm) ** 0.25) * 
                    (1 + (1 - gamamu * zetm) ** 0.25) * 
                    (1 + (1 - gamamu * zetm) ** 0.5)) / 
                   ((1 + x0_m) * (1 + x0_m) * (1 + x0_m * x0_m))) - 
            2 * (np.arctan((1 - gamamu * zetm) ** 0.25) - np.arctan(x0_m)) + 
            np.log(zet / zetm) - 
            1.140125 * ((-zet) ** by3 - (-zetm) ** by3),
            # Stable: zet >= 0
            np.where(
                zet <= 1.0,
                -4.7 * (zet - zet0_m),
                -4.7 * (1.0 - zet0_m) + 
                1.0 * (5.0 - 4.7) * np.log(zet / 1.0) - 
                5.0 * (zet - 1.0)
            )
        )
    )
    
    # Compute dm, cm
    dm = np.maximum(np.log(z / z0m) - dpsim, 1e-3)
    cm = (kappa ** 2) / (dm ** 2)
    
    # Compute dpsih (heat) - vectorized
    dpsih = np.log(z / z0h) - gamamu * (zet - zet0_h)
    dh = np.maximum(np.log(z / z0h) - dpsih, 1e-3)
    ch = (kappa ** 2) / (dm * dh)
    
    # Compute dpsiq (moisture)
    dpsiq = dpsih  # Same as dpsih for simplicity
    
    # Compute u, t, q
    u = (ustar / kappa) * (np.log(z / z0m) - dpsim)
    t = tg + (tstar / kappa) * (np.log(z / z0h) - dpsih)
    q = qg + (qstar / kappa) * (np.log(z / z0q) - dpsiq)
    
    return u, t, q, dpsim, dpsih, dpsiq


def benchmark():
    """Benchmark JAX vs. NumPy for PBL subroutines."""
    print("=" * 60)
    print("Benchmark: JAX vs. NumPy for PBL Subroutines")
    print("=" * 60)

    # Define grid sizes to test
    grid_sizes = [1000, 10000, 100000, 1000000]
    iterations = 100

    for size in grid_sizes:
        print(f"\nGrid Size: {size}")
        print("-" * 60)

        # Generate random inputs
        key = jax.random.PRNGKey(42)
        z = jax.random.uniform(key, (size,), minval=1.0, maxval=100.0)
        z0m = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        z0h = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        z0q = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        lmonin = jax.random.uniform(key, (size,), minval=-100.0, maxval=100.0)
        ustar = jax.random.uniform(key, (size,), minval=0.1, maxval=1.0)
        tstar = jax.random.uniform(key, (size,), minval=0.1, maxval=1.0)
        qstar = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        tg = jax.random.uniform(key, (size,), minval=280.0, maxval=320.0)
        qg = jax.random.uniform(key, (size,), minval=0.01, maxval=0.05)

        # Convert to NumPy arrays
        z_np = np.array(z)
        z0m_np = np.array(z0m)
        z0h_np = np.array(z0h)
        z0q_np = np.array(z0q)
        lmonin_np = np.array(lmonin)
        ustar_np = np.array(ustar)
        tstar_np = np.array(tstar)
        qstar_np = np.array(qstar)
        tg_np = np.array(tg)
        qg_np = np.array(qg)

        # Warm-up runs
        _ = simil_numpy(z_np, z0m_np, z0h_np, z0q_np, lmonin_np, ustar_np, tstar_np, qstar_np, tg_np, qg_np)
        _ = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)

        # Benchmark NumPy (CPU)
        start_time = time.time()
        for _ in range(iterations):
            u_np, t_np, q_np, dpsim_np, dpsih_np, dpsiq_np = simil_numpy(
                z_np, z0m_np, z0h_np, z0q_np, lmonin_np, ustar_np, tstar_np, qstar_np, tg_np, qg_np
            )
        numpy_time = (time.time() - start_time) / iterations

        # Benchmark JAX (CPU)
        start_time = time.time()
        for _ in range(iterations):
            u_jax, t_jax, q_jax, dpsim_jax, dpsih_jax, dpsiq_jax = simil_jit(
                z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
            )
        jax_time = (time.time() - start_time) / iterations

        # Compute speedup
        speedup = numpy_time / jax_time

        print(f"NumPy (CPU) time:  {numpy_time:.6f} seconds")
        print(f"JAX (CPU) time:    {jax_time:.6f} seconds")
        print(f"Speedup (JAX/NumPy): {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
