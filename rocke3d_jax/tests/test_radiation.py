"""
Unit Tests for Radiation Module
===============================

This module tests the JAX implementation of ROCKE-3D's radiative transfer
subroutines against Fortran-like implementations.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import numpy as np
import jax
import jax.numpy as jnp
from rocke3d_jax.radiation import (
    blackbody_radiation_jit,
    solar_radiation_jit,
    net_radiation_jit,
)


def test_blackbody_radiation():
    """Test blackbody radiation against Stefan-Boltzmann law."""
    print("Testing blackbody radiation...")
    
    # Test inputs
    T = jnp.array([[288.0, 300.0]])  # Surface temperature (K)
    emissivity = 0.95
    
    # JAX implementation
    lw_up_jax = blackbody_radiation_jit(T, emissivity)
    
    # Fortran-like implementation
    sigma_sb = 5.670374419e-8
    lw_up_fortran = emissivity * sigma_sb * T ** 4
    
    # Compare
    diff = np.abs(lw_up_jax - lw_up_fortran)
    assert np.all(diff < 1e-12), f"Blackbody radiation test failed: max diff = {np.max(diff)}"
    print(f"  JAX output: {lw_up_jax}")
    print(f"  Fortran output: {lw_up_fortran}")
    print("✓ Blackbody radiation test passed")


def test_solar_radiation():
    """Test solar radiation against simplified model."""
    print("Testing solar radiation...")
    
    # Test inputs
    cosz = jnp.array([[1.0, 0.5]])  # Cosine of solar zenith angle
    albedo = jnp.array([[0.2, 0.3]])  # Surface albedo
    solar_constant = 1361.0
    
    # JAX implementation
    sw_absorbed_jax = solar_radiation_jit(cosz, albedo, solar_constant)
    
    # Fortran-like implementation
    sw_down = solar_constant * np.maximum(cosz, 0.0)
    sw_absorbed_fortran = (1.0 - albedo) * sw_down
    
    # Compare
    diff = np.abs(sw_absorbed_jax - sw_absorbed_fortran)
    assert np.all(diff < 1e-12), f"Solar radiation test failed: max diff = {np.max(diff)}"
    print(f"  JAX output: {sw_absorbed_jax}")
    print(f"  Fortran output: {sw_absorbed_fortran}")
    print("✓ Solar radiation test passed")


def test_net_radiation():
    """Test net radiation against simplified model."""
    print("Testing net radiation...")
    
    # Test inputs
    T = jnp.array([[288.0, 300.0]])  # Surface temperature (K)
    cosz = jnp.array([[1.0, 0.5]])   # Cosine of solar zenith angle
    albedo = jnp.array([[0.2, 0.3]])  # Surface albedo
    emissivity = 0.95
    solar_constant = 1361.0
    
    # JAX implementation
    net_rad_jax = net_radiation_jit(T, cosz, albedo, emissivity, solar_constant)
    
    # Fortran-like implementation
    sigma_sb = 5.670374419e-8
    lw_up = emissivity * sigma_sb * T ** 4
    lw_down = emissivity * sigma_sb * T ** 4
    lw_net = lw_down - lw_up
    sw_down = solar_constant * np.maximum(cosz, 0.0)
    sw_absorbed = (1.0 - albedo) * sw_down
    net_rad_fortran = sw_absorbed + lw_net
    
    # Compare
    diff = np.abs(net_rad_jax - net_rad_fortran)
    assert np.all(diff < 1e-12), f"Net radiation test failed: max diff = {np.max(diff)}"
    print(f"  JAX output: {net_rad_jax}")
    print(f"  Fortran output: {net_rad_fortran}")
    print("✓ Net radiation test passed")


def test_performance():
    """Test performance of radiation subroutines."""
    print("Testing performance...")
    
    import time
    
    # Define grid sizes
    sizes = [1000, 10000, 100000]
    iterations = 100
    
    for size in sizes:
        print(f"  Grid Size: {size}")
        
        # Generate random inputs
        key = jax.random.PRNGKey(42)
        T = jax.random.uniform(key, (size,), minval=280.0, maxval=320.0)
        cosz = jax.random.uniform(key, (size,), minval=0.0, maxval=1.0)
        albedo = jax.random.uniform(key, (size,), minval=0.0, maxval=0.5)
        
        # Warm-up
        _ = net_radiation_jit(T, cosz, albedo)
        
        # Benchmark JAX
        start_time = time.time()
        for _ in range(iterations):
            net_rad = net_radiation_jit(T, cosz, albedo)
        jax_time = (time.time() - start_time) / iterations
        
        print(f"    JAX time: {jax_time:.6f} seconds")
    
    print("✓ Performance test completed")


if __name__ == "__main__":
    test_blackbody_radiation()
    test_solar_radiation()
    test_net_radiation()
    test_performance()
    print("\nAll radiation tests passed!")
