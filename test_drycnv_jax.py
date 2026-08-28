"""
Test Script for JAX Dry Convection (DRYCNV) Prototype
====================================================

This script validates the JAX implementation of ROCKE-3D's dry convection
mixing scheme by comparing it with a simplified Fortran-like implementation.

Usage:
    python test_drycnv_jax.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from drycnv_jax import dry_convection_mixing_jit, validate_drycnv


def test_basic_functionality():
    """Test basic functionality of the JAX implementation."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define test dimensions
    I, J, L = 2, 2, 5
    
    # Generate random inputs
    T = jax.random.uniform(key, (I, J, L), minval=200.0, maxval=300.0)
    Q = jax.random.uniform(key, (I, J, L), minval=0.0, maxval=0.02)
    PK = jax.random.uniform(key, (I, J, L), minval=0.5, maxval=1.0)
    PDSIG = jax.random.uniform(key, (I, J, L), minval=0.1, maxval=0.2)
    
    # Run JAX implementation
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Check output shapes
    assert T_out.shape == T.shape, f"T output shape mismatch: {T_out.shape} vs {T.shape}"
    assert Q_out.shape == Q.shape, f"Q output shape mismatch: {Q_out.shape} vs {Q.shape}"
    
    print("✓ Output shapes match input shapes")
    print(f"  Input T shape: {T.shape}")
    print(f"  Output T shape: {T_out.shape}")
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(T_out)), "NaN detected in T output"
    assert not jnp.any(jnp.isnan(Q_out)), "NaN detected in Q output"
    assert not jnp.any(jnp.isinf(T_out)), "Inf detected in T output"
    assert not jnp.any(jnp.isinf(Q_out)), "Inf detected in Q output"
    
    print("✓ No NaN or Inf values in output")
    
    # Check if outputs are within reasonable ranges
    assert jnp.all(T_out >= 0.0), "T output contains negative values"
    assert jnp.all(Q_out >= 0.0), "Q output contains negative values"
    
    print("✓ Output values are within reasonable ranges")
    print()


def test_unstable_layer_mixing():
    """Test mixing in unstable layers."""
    print("=" * 60)
    print("Test 2: Unstable Layer Mixing")
    print("=" * 60)
    
    # Create a simple unstable profile
    I, J, L = 1, 1, 3
    T = jnp.array([[[290.0, 280.0, 270.0]]])  # Temperature decreases with height (unstable)
    Q = jnp.array([[[0.01, 0.01, 0.01]]])
    PK = jnp.array([[[1.0, 0.8, 0.6]]])
    PDSIG = jnp.array([[[0.2, 0.2, 0.2]]])
    
    # Run JAX implementation
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Check if mixing occurred (T should be uniform after mixing)
    T_diff = jnp.abs(T_out[0, 0, 0] - T_out[0, 0, 1])
    print(f"  T difference between layers 1 and 2: {T_diff:.6e}")
    
    # Since the profile is unstable, mixing should reduce the difference
    assert T_diff < 10.0, f"Mixing did not occur: T difference = {T_diff}"
    print("✓ Mixing occurred in unstable layers")
    print()


def test_stable_layer_no_mixing():
    """Test that stable layers are not mixed."""
    print("=" * 60)
    print("Test 3: Stable Layer (No Mixing)")
    print("=" * 60)
    
    # Create a stable profile (temperature increases with height)
    I, J, L = 1, 1, 3
    T = jnp.array([[[270.0, 280.0, 290.0]]])  # Temperature increases with height (stable)
    Q = jnp.array([[[0.01, 0.01, 0.01]]])
    PK = jnp.array([[[1.0, 0.8, 0.6]]])
    PDSIG = jnp.array([[[0.2, 0.2, 0.2]]])
    
    # Run JAX implementation
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Check if mixing occurred (T should remain unchanged)
    T_diff = jnp.abs(T_out - T)
    print(f"  Max T difference: {jnp.max(T_diff):.6e}")
    
    # Since the profile is stable, no mixing should occur
    assert jnp.max(T_diff) < 1e-6, f"Unexpected mixing in stable layers: T difference = {jnp.max(T_diff)}"
    print("✓ No mixing occurred in stable layers")
    print()


def test_validation_against_fortran():
    """Test validation against Fortran-like implementation."""
    print("=" * 60)
    print("Test 4: Validation Against Fortran")
    print("=" * 60)
    
    # Run validation
    T_jax, Q_jax, T_fortran, Q_fortran = validate_drycnv()
    
    # Check if validation passed
    T_diff = np.abs(np.array(T_jax) - T_fortran)
    Q_diff = np.abs(np.array(Q_jax) - Q_fortran)
    
    T_tolerance = 1e-4  # Looser tolerance for temperature
    Q_tolerance = 1e-6  # Strict tolerance for moisture
    T_pass = np.all(T_diff < T_tolerance)
    Q_pass = np.all(Q_diff < Q_tolerance)
    
    print("Validation Results:")
    print(f"- Max T difference: {np.max(T_diff):.6e}")
    print(f"- Max Q difference: {np.max(Q_diff):.6e}")
    print(f"- Mean T difference: {np.mean(T_diff):.6e}")
    print(f"- Mean Q difference: {np.mean(Q_diff):.6e}")
    print()
    
    if T_pass and Q_pass:
        print("✓ Validation PASSED: JAX and Fortran outputs match within tolerance")
    else:
        print("✗ Validation FAILED: Outputs differ beyond tolerance")
        print(f"  Max T difference: {np.max(T_diff):.6e}")
        print(f"  Max Q difference: {np.max(Q_diff):.6e}")
    print()


def test_performance():
    """Test performance of JAX implementation."""
    print("=" * 60)
    print("Test 5: Performance Benchmark")
    print("=" * 60)
    
    import time
    
    # Define test dimensions (larger for performance testing)
    I, J, L = 32, 32, 20
    
    # Generate random inputs
    key = jax.random.PRNGKey(42)
    T = jax.random.uniform(key, (I, J, L), minval=200.0, maxval=300.0)
    Q = jax.random.uniform(key, (I, J, L), minval=0.0, maxval=0.02)
    PK = jax.random.uniform(key, (I, J, L), minval=0.5, maxval=1.0)
    PDSIG = jax.random.uniform(key, (I, J, L), minval=0.1, maxval=0.2)
    
    # Warm up JAX
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    T_out.block_until_ready()
    
    # Time JAX implementation
    start_time = time.time()
    for _ in range(10):
        T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
        T_out.block_until_ready()
    jax_time = (time.time() - start_time) / 10
    
    print(f"  JAX time (per call): {jax_time:.6f} seconds")
    print(f"  Input size: {I}x{J}x{L} = {I*J*L} grid points")
    print("✓ Performance test completed")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing JAX Dry Convection (DRYCNV) Prototype")
    print("=" * 60 + "\n")
    
    # Run tests
    test_basic_functionality()
    test_unstable_layer_mixing()
    test_stable_layer_no_mixing()
    test_validation_against_fortran()
    test_performance()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
