"""
Comprehensive Benchmark for ROCKE-3D JAX Implementation
========================================================

This script benchmarks all JAX-implemented ROCKE-3D modules against
NumPy (Fortran-like) implementations and compares performance.

Usage:
    python benchmark_all.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import sys
sys.path.insert(0, '.')

import time
import numpy as np
import jax
import jax.numpy as jnp
from drycnv import dry_convection_mixing_jit
from pbl import simil_jit


def dry_convection_numpy(T, Q, PK, PDSIG, deltx=0.608):
    """NumPy implementation of dry convection (Fortran-like, CPU baseline)."""
    T_out = np.copy(T)
    Q_out = np.copy(Q)
    TV = T_out * (1 + Q_out * deltx)

    for L in range(T.shape[-1] - 1):
        unstable = TV[..., L] > TV[..., L + 1]
        PKMS = PK[..., L] * PDSIG[..., L] + PK[..., L + 1] * PDSIG[..., L + 1]
        TVMS = TV[..., L] * PK[..., L] * PDSIG[..., L] + TV[..., L + 1] * PK[..., L + 1] * PDSIG[..., L + 1]
        QMS = Q_out[..., L] * PDSIG[..., L] + Q_out[..., L + 1] * PDSIG[..., L + 1]
        RDP = 1.0 / (PDSIG[..., L] + PDSIG[..., L + 1])
        THM = TVMS / (PKMS * (1 + QMS * RDP * deltx))
        QM = QMS * RDP

        T_out[..., L] = np.where(unstable, THM, T_out[..., L])
        T_out[..., L + 1] = np.where(unstable, THM, T_out[..., L + 1])
        Q_out[..., L] = np.where(unstable, QM, Q_out[..., L])
        Q_out[..., L + 1] = np.where(unstable, QM, Q_out[..., L + 1])

        TV = T_out * (1 + Q_out * deltx)

    return T_out, Q_out


def simil_numpy(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg):
    """NumPy implementation of simil (Fortran-like, CPU baseline)."""
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
    dpsiq = dpsih
    
    # Compute u, t, q
    u = (ustar / kappa) * (np.log(z / z0m) - dpsim)
    t = tg + (tstar / kappa) * (np.log(z / z0h) - dpsih)
    q = qg + (qstar / kappa) * (np.log(z / z0q) - dpsiq)
    
    return u, t, q, dpsim, dpsih, dpsiq


def benchmark_module(name, jax_func, numpy_func, input_generator, grid_sizes, iterations=100):
    """Benchmark a single module."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")
    
    for size in grid_sizes:
        print(f"\nGrid Size: {size}")
        print("-" * 60)
        
        # Generate inputs
        inputs = input_generator(size)
        
        # Convert to NumPy arrays (if needed)
        inputs_np = tuple(np.array(x) for x in inputs)
        
        # Warm-up runs
        _ = numpy_func(*inputs_np)
        _ = jax_func(*inputs)
        
        # Benchmark NumPy (CPU)
        start_time = time.time()
        for _ in range(iterations):
            _ = numpy_func(*inputs_np)
        numpy_time = (time.time() - start_time) / iterations
        
        # Benchmark JAX (CPU)
        start_time = time.time()
        for _ in range(iterations):
            _ = jax_func(*inputs)
        jax_time = (time.time() - start_time) / iterations
        
        # Compute speedup
        speedup = numpy_time / jax_time
        
        print(f"NumPy (CPU) time:  {numpy_time:.6f} seconds")
        print(f"JAX (CPU) time:    {jax_time:.6f} seconds")
        print(f"Speedup (JAX/NumPy): {speedup:.2f}x")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("ROCKE-3D JAX Benchmark Suite")
    print("=" * 60)
    
    # Define grid sizes to test
    grid_sizes = [1000, 10000, 100000]
    iterations = 100
    
    # Benchmark DRYCNV
    def drycnv_input_generator(size):
        key = jax.random.PRNGKey(42)
        I, J = int(np.sqrt(size)), int(np.sqrt(size))
        L = 20
        T = jax.random.uniform(key, (I, J, L), minval=200.0, maxval=300.0)
        Q = jax.random.uniform(key, (I, J, L), minval=0.0, maxval=0.02)
        PK = jax.random.uniform(key, (I, J, L), minval=0.5, maxval=1.0)
        PDSIG = jax.random.uniform(key, (I, J, L), minval=0.1, maxval=0.2)
        return T, Q, PK, PDSIG
    
    benchmark_module(
        "DRYCNV (Dry Convection)",
        dry_convection_mixing_jit,
        dry_convection_numpy,
        drycnv_input_generator,
        grid_sizes,
        iterations
    )
    
    # Benchmark PBL
    def pbl_input_generator(size):
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
        return z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
    
    benchmark_module(
        "PBL (Planetary Boundary Layer)",
        simil_jit,
        simil_numpy,
        pbl_input_generator,
        grid_sizes,
        iterations
    )
    
    print("\n" + "=" * 60)
    print("All benchmarks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
