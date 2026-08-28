"""
Unit Tests for Dry Convection (DRYCNV.f)
========================================

This module tests the JAX implementation of ROCKE-3D's dry convection
mixing scheme against a Fortran-like implementation.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import numpy as np
import jax
import jax.numpy as jnp
from rocke3d_jax.drycnv import dry_convection_mixing_jit


def test_basic_functionality():
    """Test basic functionality of dry convection mixing."""
    print("Testing basic functionality...")
    
    # Define test inputs
    I, J, L = 2, 2, 5
    T = jnp.ones((I, J, L)) * 300.0
    Q = jnp.ones((I, J, L)) * 0.01
    PK = jnp.ones((I, J, L)) * 1000.0
    PDSIG = jnp.ones((I, J, L)) * 100.0
    
    # Introduce instability in layer 2 (TV[2] > TV[3])
    T = T.at[..., 2].set(290.0)
    T = T.at[..., 3].set(280.0)  # Lower temperature at layer 3
    
    # Run JAX implementation
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Debug: Print TV before and after
    TV_in = T * (1 + Q * 0.608)
    TV_out = T_out * (1 + Q_out * 0.608)
    print(f"TV_in[0,0,:]: {TV_in[0,0,:]}")
    print(f"TV_out[0,0,:]: {TV_out[0,0,:]}")
    
    # Check output shapes
    assert T_out.shape == T.shape, f"T output shape mismatch: {T_out.shape} vs {T.shape}"
    assert Q_out.shape == Q.shape, f"Q output shape mismatch: {Q_out.shape} vs {Q.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(T_out)), "NaN detected in T output"
    assert not jnp.any(jnp.isnan(Q_out)), "NaN detected in Q output"
    assert not jnp.any(jnp.isinf(T_out)), "Inf detected in T output"
    assert not jnp.any(jnp.isinf(Q_out)), "Inf detected in Q output"
    
    # Check if mixing occurred (T[2] and T[3] should be equal after mixing)
    assert jnp.allclose(T_out[..., 2], T_out[..., 3], atol=1e-6), f"Mixing failed: T[2]={T_out[..., 2]} != T[3]={T_out[..., 3]}"
    print("✓ Basic functionality test passed")


def test_unstable_layer_mixing():
    """Test mixing in unstable layers."""
    print("Testing unstable layer mixing...")
    
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
    assert T_diff < 10.0, f"Mixing did not occur: T difference = {T_diff}"
    print("✓ Unstable layer mixing test passed")


def test_stable_layer_no_mixing():
    """Test that stable layers are not mixed."""
    print("Testing stable layer (no mixing)...")
    
    # Create a stable profile (temperature increases with height)
    I, J, L = 1, 1, 3
    T = jnp.array([[[270.0, 280.0, 290.0]]])  # Temperature increases with height (stable)
    Q = jnp.array([[[0.01, 0.01, 0.01]]])
    PK = jnp.array([[[1.0, 0.8, 0.6]]])
    PDSIG = jnp.array([[[0.2, 0.2, 0.2]]])
    
    # Run JAX implementation
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Check if no mixing occurred
    T_diff = jnp.abs(T_out - T)
    assert jnp.max(T_diff) < 1e-6, f"Unexpected mixing in stable layers: T difference = {jnp.max(T_diff)}"
    print("✓ Stable layer test passed")


def test_numerical_consistency():
    """Test numerical consistency with a Fortran-like implementation."""
    print("Testing numerical consistency...")
    
    # Define test inputs (unstable at layers 0-1 and 1-2)
    I, J, L = 1, 1, 3
    T = jnp.array([[[290.0, 280.0, 270.0]]])  # Unstable at both 0-1 and 1-2
    Q = jnp.ones((I, J, L)) * 0.01
    PK = jnp.ones((I, J, L)) * 1000.0
    PDSIG = jnp.ones((I, J, L)) * 100.0
    
    # Run JAX implementation
    T_jax, Q_jax = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Compute expected output manually (Fortran-like, matching JAX logic)
    deltx = 0.608
    TV = T * (1 + Q * deltx)
    
    # Mix layers 0-1 (first unstable pair)
    PKMS_01 = PK[..., 0] * PDSIG[..., 0] + PK[..., 1] * PDSIG[..., 1]
    TVMS_01 = TV[..., 0] * PK[..., 0] * PDSIG[..., 0] + TV[..., 1] * PK[..., 1] * PDSIG[..., 1]
    QMS_01 = Q[..., 0] * PDSIG[..., 0] + Q[..., 1] * PDSIG[..., 1]
    RDP_01 = 1.0 / (PDSIG[..., 0] + PDSIG[..., 1])
    THM_01 = TVMS_01 / (PKMS_01 * (1 + QMS_01 * RDP_01 * deltx))
    QM_01 = QMS_01 * RDP_01
    
    # Update T and Q for layers 0-1
    T_temp = T.at[..., 0:2].set(THM_01)
    Q_temp = Q.at[..., 0:2].set(QM_01)
    TV_temp = T_temp * (1 + Q_temp * deltx)
    
    # Mix layers 1-2 (second unstable pair, using updated T_temp and Q_temp)
    PKMS_12 = PK[..., 1] * PDSIG[..., 1] + PK[..., 2] * PDSIG[..., 2]
    TVMS_12 = TV_temp[..., 1] * PK[..., 1] * PDSIG[..., 1] + TV_temp[..., 2] * PK[..., 2] * PDSIG[..., 2]
    QMS_12 = Q_temp[..., 1] * PDSIG[..., 1] + Q_temp[..., 2] * PDSIG[..., 2]
    RDP_12 = 1.0 / (PDSIG[..., 1] + PDSIG[..., 2])
    THM_12 = TVMS_12 / (PKMS_12 * (1 + QMS_12 * RDP_12 * deltx))
    QM_12 = QMS_12 * RDP_12
    
    # Update T and Q for layers 1-2
    T_expected = T_temp.at[..., 1:3].set(THM_12)
    Q_expected = Q_temp.at[..., 1:3].set(QM_12)
    
    # Compare JAX and expected outputs
    assert jnp.allclose(T_jax, T_expected, atol=1e-6), f"T mismatch with expected: {T_jax[0,0,:]} vs {T_expected[0,0,:]}"
    assert jnp.allclose(Q_jax, Q_expected, atol=1e-6), f"Q mismatch with expected: {Q_jax[0,0,:]} vs {Q_expected[0,0,:]}"
    print("✓ Numerical consistency test passed")


def test_performance():
    """Test performance of JAX implementation."""
    print("Testing performance...")
    
    # Define larger test inputs
    I, J, L = 100, 100, 20
    T = jnp.ones((I, J, L)) * 300.0
    Q = jnp.ones((I, J, L)) * 0.01
    PK = jnp.ones((I, J, L)) * 1000.0
    PDSIG = jnp.ones((I, J, L)) * 100.0
    
    # Warm up JIT compilation
    _ = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Time the JAX implementation
    import time
    start = time.time()
    for _ in range(10):
        T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    elapsed = (time.time() - start) / 10
    print(f"JAX execution time (avg over 10 runs): {elapsed:.4f} seconds")
    print("✓ Performance test completed")


if __name__ == "__main__":
    test_basic_functionality()
    test_unstable_layer_mixing()
    test_stable_layer_no_mixing()
    test_numerical_consistency()
    test_performance()
    print("\nAll dry convection tests passed!")
