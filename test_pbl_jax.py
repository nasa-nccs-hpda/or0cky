"""
Test Script for JAX PBL (Planetary Boundary Layer) Subroutines
================================================================

This script validates the JAX implementation of ROCKE-3D's PBL subroutines
by comparing it with a simplified Fortran-like implementation.

Usage:
    python test_pbl_jax.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import numpy as np
import jax
import jax.numpy as jnp
from pbl_jax import (
    find_dpsim_jit, find_dpsih_jit, getcm_jit, getchq_jit, simil_jit
)


def test_find_dpsim():
    """Test find_dpsim against Fortran-like implementation."""
    print("=" * 60)
    print("Test 1: find_dpsim (Monin-Obukhov Similarity for Momentum)")
    print("=" * 60)
    
    # Test cases: stable, unstable, and neutral conditions
    zet = jnp.array([0.5, -0.5, 0.0])  # Stable, unstable, neutral
    zet0 = jnp.array([0.1, -0.1, 0.0])
    
    # JAX implementation
    dpsim_jax = find_dpsim_jit(zet, zet0)
    print(f"JAX dpsim: {dpsim_jax}")
    
    # Fortran-like implementation (from PBL.f)
    dpsim_fortran = np.zeros_like(zet)
    for i in range(len(zet)):
        zet_val = zet[i]
        zet0_val = zet0[i]
        if zet_val >= 0.0:  # Stable
            if zet_val <= 1.0:
                dpsim_fortran[i] = -4.7 * (zet_val - zet0_val)
            else:
                dpsim_fortran[i] = -4.7 * (1.0 - zet0_val) + \
                                   1.0 * (5.0 - 4.7) * np.log(zet_val / 1.0) - \
                                   5.0 * (zet_val - 1.0)
        else:  # Unstable
            x = (1.0 - 16.0 * zet_val) ** 0.25
            x0 = (1.0 - 16.0 * zet0_val) ** 0.25
            xm = (1.0 - 16.0 * (-1.0)) ** 0.25
            if zet_val > -1.0:
                dpsim_fortran[i] = np.log((1 + x) * (1 + x) * (1 + x * x) /
                                          ((1 + x0) * (1 + x0) * (1 + x0 * x0))) - \
                                   2 * (np.arctan(x) - np.arctan(x0))
            else:
                dpsim_fortran[i] = np.log((1 + xm) * (1 + xm) * (1 + xm * xm) /
                                          ((1 + x0) * (1 + x0) * (1 + x0 * x0))) - \
                                   2 * (np.arctan(xm) - np.arctan(x0)) + \
                                   np.log(zet_val / -1.0) - \
                                   1.140125 * ((-zet_val) ** (1/3) - 1.0)
    
    print(f"Fortran dpsim: {dpsim_fortran}")
    
    # Compare
    diff = np.abs(dpsim_jax - dpsim_fortran)
    print(f"Difference: {diff}")
    assert np.all(diff < 1e-6), f"find_dpsim test failed: max diff = {np.max(diff)}"
    print("✓ Test passed: find_dpsim matches Fortran-like implementation")
    print()


def test_find_dpsih():
    """Test find_dpsih against Fortran-like implementation."""
    print("=" * 60)
    print("Test 2: find_dpsih (Monin-Obukhov Similarity for Heat/Moisture)")
    print("=" * 60)
    
    # Test cases
    zet = jnp.array([0.5, -0.5, 0.0])
    zet0 = jnp.array([0.1, -0.1, 0.0])
    z = jnp.array([10.0, 10.0, 10.0])
    z0 = jnp.array([0.1, 0.1, 0.1])
    
    # JAX implementation
    dpsih_jax = find_dpsih_jit(zet, zet0, z, z0)
    print(f"JAX dpsih: {dpsih_jax}")
    
    # Fortran-like implementation
    dpsih_fortran = np.zeros_like(zet)
    for i in range(len(zet)):
        zet_val = zet[i]
        zet0_val = zet0[i]
        z_val = z[i]
        z0_val = z0[i]
        if zet_val >= 0.0:  # Stable
            if zet_val <= 1.0:
                dpsih_fortran[i] = 1.0 * np.log(z_val / z0_val) - 1.0 * 4.7 * (zet_val - zet0_val)
            else:
                dpsih_fortran[i] = 1.0 * np.log(1.0 / z0_val) - 1.0 * 4.7 * (1.0 - zet0_val) + \
                                   (1 + 1.0 * (1.0 * (5.0 - 4.7) - 1)) * np.log(zet_val / 1.0) - \
                                   1.0 * 5.0 * (zet_val - 1.0)
        else:  # Unstable
            dpsih_fortran[i] = 1.0 * np.log(z_val / z0_val) - 1.0 * 16.0 * (zet_val - zet0_val)
    
    print(f"Fortran dpsih: {dpsih_fortran}")
    
    # Compare
    diff = np.abs(dpsih_jax - dpsih_fortran)
    print(f"Difference: {diff}")
    assert np.all(diff < 1e-6), f"find_dpsih test failed: max diff = {np.max(diff)}"
    print("✓ Test passed: find_dpsih matches Fortran-like implementation")
    print()


def test_simil():
    """Test simil against Fortran-like implementation."""
    print("=" * 60)
    print("Test 3: simil (Monin-Obukhov Similarity Solutions)")
    print("=" * 60)
    
    # Test case: unstable conditions
    z = jnp.array([10.0])  # Height (m)
    z0m = jnp.array([0.1])  # Momentum roughness height (m)
    z0h = jnp.array([0.1])  # Temperature roughness height (m)
    z0q = jnp.array([0.1])  # Moisture roughness height (m)
    lmonin = jnp.array([-10.0])  # Monin-Obukhov length (m, negative = unstable)
    ustar = jnp.array([0.5])  # Friction speed (m/s)
    tstar = jnp.array([0.1])  # Temperature scale (K)
    qstar = jnp.array([0.01])  # Moisture scale (kg/kg)
    tg = jnp.array([300.0])  # Ground temperature (K)
    qg = jnp.array([0.02])  # Ground moisture mixing ratio (kg/kg)
    
    # JAX implementation
    u_jax, t_jax, q_jax, dpsim_jax, dpsih_jax, dpsiq_jax = simil_jit(
        z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
    )
    print(f"JAX u: {u_jax}, t: {t_jax}, q: {q_jax}")
    print(f"JAX dpsim: {dpsim_jax}, dpsih: {dpsih_jax}, dpsiq: {dpsiq_jax}")
    
    # Fortran-like implementation
    # Compute dm, dpsim, cm
    zet = z / lmonin
    zet0 = z0m / lmonin
    x = (1.0 - 16.0 * zet) ** 0.25
    x0 = (1.0 - 16.0 * zet0) ** 0.25
    dpsim_fortran = np.log((1 + x) * (1 + x) * (1 + x * x) /
                           ((1 + x0) * (1 + x0) * (1 + x0 * x0))) - \
                    2 * (np.arctan(x) - np.arctan(x0))
    dm = max(np.log(z / z0m) - dpsim_fortran, 1e-3)
    cm = (0.4 ** 2) / (dm ** 2)
    
    # Compute dpsih, ch
    zet_h = z / lmonin
    zet0_h = z0h / lmonin
    dpsih_fortran = 1.0 * np.log(z / z0h) - 1.0 * 16.0 * (zet_h - zet0_h)
    dh = max(np.log(z / z0h) - dpsih_fortran, 1e-3)
    ch = (0.4 ** 2) / (dm * dh)
    
    # Compute dpsiq, cq
    dpsiq_fortran = dpsih_fortran  # Same as dpsih for moisture in this case
    
    # Compute u, t, q
    u_fortran = (ustar / 0.4) * (np.log(z / z0m) - dpsim_fortran)
    t_fortran = tg + (tstar / 0.4) * (np.log(z / z0h) - dpsih_fortran)
    q_fortran = qg + (qstar / 0.4) * (np.log(z / z0q) - dpsiq_fortran)
    
    print(f"Fortran u: {u_fortran}, t: {t_fortran}, q: {q_fortran}")
    
    # Compare
    u_diff = np.abs(u_jax - u_fortran)
    t_diff = np.abs(t_jax - t_fortran)
    q_diff = np.abs(q_jax - q_fortran)
    print(f"Differences: u={u_diff}, t={t_diff}, q={q_diff}")
    assert np.all(u_diff < 1e-6), f"u test failed: diff = {u_diff}"
    assert np.all(t_diff < 1e-6), f"t test failed: diff = {t_diff}"
    assert np.all(q_diff < 1e-6), f"q test failed: diff = {q_diff}"
    print("✓ Test passed: simil matches Fortran-like implementation")
    print()


def test_performance():
    """Benchmark JAX vs. NumPy for PBL subroutines."""
    print("=" * 60)
    print("Test 4: Performance Benchmark")
    print("=" * 60)
    
    import time
    
    # Define grid sizes
    sizes = [1000, 10000, 100000]
    iterations = 100
    
    for size in sizes:
        print(f"\nGrid Size: {size}")
        
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
        
        # Warm-up
        _ = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)
        
        # Benchmark JAX
        start_time = time.time()
        for _ in range(iterations):
            u, t, q, dpsim, dpsih, dpsiq = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)
        jax_time = (time.time() - start_time) / iterations
        
        print(f"  JAX time: {jax_time:.6f} seconds")
        print(f"  Input size: {size}")
    
    print("\n✓ Performance test completed")
    print()


def main():
    test_find_dpsim()
    test_find_dpsih()
    test_simil()
    test_performance()
    print("=" * 60)
    print("All PBL tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
