"""
Unit Tests for PBL (Planetary Boundary Layer)
==============================================

This module tests the JAX implementation of ROCKE-3D's PBL subroutines
against Fortran-like implementations.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import numpy as np
import jax
import jax.numpy as jnp
from rocke3d_jax.pbl import (
    find_dpsim_jit, find_dpsih_jit, getcm_jit, getchq_jit, simil_jit
)


def test_find_dpsim():
    """Test find_dpsim against Fortran-like implementation."""
    print("Testing find_dpsim...")
    
    # Test cases: stable, unstable, and neutral conditions
    zet = jnp.array([0.5, -0.5, 0.0])
    zet0 = jnp.array([0.1, -0.1, 0.0])
    
    # JAX implementation
    dpsim_jax = find_dpsim_jit(zet, zet0)
    
    # Fortran-like implementation
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
    
    # Compare
    diff = np.abs(dpsim_jax - dpsim_fortran)
    assert np.all(diff < 1e-6), f"find_dpsim test failed: max diff = {np.max(diff)}"
    print("✓ find_dpsim test passed")


def test_find_dpsih():
    """Test find_dpsih against Fortran-like implementation."""
    print("Testing find_dpsih...")
    
    # Test cases
    zet = jnp.array([0.5, -0.5, 0.0])
    zet0 = jnp.array([0.1, -0.1, 0.0])
    z = jnp.array([10.0, 10.0, 10.0])
    z0 = jnp.array([0.1, 0.1, 0.1])
    
    # JAX implementation
    dpsih_jax = find_dpsih_jit(zet, zet0, z, z0)
    
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
    
    # Compare
    diff = np.abs(dpsih_jax - dpsih_fortran)
    assert np.all(diff < 1e-6), f"find_dpsih test failed: max diff = {np.max(diff)}"
    print("✓ find_dpsih test passed")


def test_simil():
    """Test simil against Fortran-like implementation."""
    print("Testing simil...")
    
    # Test case: unstable conditions
    z = jnp.array([10.0])
    z0m = jnp.array([0.1])
    z0h = jnp.array([0.1])
    z0q = jnp.array([0.1])
    lmonin = jnp.array([-10.0])
    ustar = jnp.array([0.5])
    tstar = jnp.array([0.1])
    qstar = jnp.array([0.01])
    tg = jnp.array([300.0])
    qg = jnp.array([0.02])
    
    # JAX implementation
    u_jax, t_jax, q_jax, dpsim_jax, dpsih_jax, dpsiq_jax = simil_jit(
        z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
    )
    
    # Fortran-like implementation
    kappa = 0.4
    gamamu = 16.0
    by3 = 1.0 / 3.0
    zetm = -1.0
    
    zet = z / lmonin
    zet0_m = z0m / lmonin
    x = (1.0 - gamamu * zet) ** 0.25
    x0_m = (1.0 - gamamu * zet0_m) ** 0.25
    dpsim_fortran = np.log(((1 + x) * (1 + x) * (1 + x * x)) / 
                           ((1 + x0_m) * (1 + x0_m) * (1 + x0_m * x0_m))) - \
                    2 * (np.arctan(x) - np.arctan(x0_m))
    dm = max(np.log(z / z0m) - dpsim_fortran, 1e-3)
    
    zet_h = z / lmonin
    zet0_h = z0h / lmonin
    dpsih_fortran = 1.0 * np.log(z / z0h) - 1.0 * gamamu * (zet_h - zet0_h)
    dh = max(np.log(z / z0h) - dpsih_fortran, 1e-3)
    
    dpsiq_fortran = dpsih_fortran
    
    u_fortran = (ustar / kappa) * (np.log(z / z0m) - dpsim_fortran)
    t_fortran = tg + (tstar / kappa) * (np.log(z / z0h) - dpsih_fortran)
    q_fortran = qg + (qstar / kappa) * (np.log(z / z0q) - dpsiq_fortran)
    
    # Compare
    u_diff = np.abs(u_jax - u_fortran)
    t_diff = np.abs(t_jax - t_fortran)
    q_diff = np.abs(q_jax - q_fortran)
    assert np.all(u_diff < 1e-6), f"u test failed: diff = {u_diff}"
    assert np.all(t_diff < 1e-6), f"t test failed: diff = {t_diff}"
    assert np.all(q_diff < 1e-6), f"q test failed: diff = {q_diff}"
    print("✓ simil test passed")


def test_performance():
    """Test performance of PBL subroutines."""
    print("Testing performance...")
    
    import time
    
    # Define grid sizes
    sizes = [1000, 10000, 100000]
    iterations = 100
    
    for size in sizes:
        print(f"  Grid Size: {size}")
        
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
        
        print(f"    JAX time: {jax_time:.6f} seconds")
    
    print("✓ Performance test completed")


if __name__ == "__main__":
    test_find_dpsim()
    test_find_dpsih()
    test_simil()
    test_performance()
    print("\nAll PBL tests passed!")
