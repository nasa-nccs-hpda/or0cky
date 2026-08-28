"""
Unit Tests for LAKES_COM JAX Implementation
=============================================

This script tests the JAX implementation of the lake common variables
from ROCKE-3D's LAKES_COM.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_lakes_com_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from lakes_com_jax import (
    NRVRMX, NRVR, IRVRMTH, JRVRMTH, NAMERVR, RVROUT,
    compute_lake_enthalpy, compute_lake_depth,
    compute_lake_glake, compute_mixed_layer_depth,
    initialize_lakes_com,
)
from constant_jax import SHW, RHOW


# Test constants
np.random.seed(42)

# Grid size
IM_TEST = 36
JM_TEST = 24


def test_constants():
    """Test lake common constants."""
    print("Testing constants...")
    
    # Check NRVRMX
    assert NRVRMX == 42, "NRVRMX should be 42"
    
    # Check NRVR
    assert NRVR == 0, "NRVR should be 0"
    
    # Check IRVRMTH and JRVRMTH
    assert IRVRMTH.shape == (NRVRMX,), "IRVRMTH shape is incorrect"
    assert JRVRMTH.shape == (NRVRMX,), "JRVRMTH shape is incorrect"
    
    # Check NAMERVR
    assert len(NAMERVR) == NRVRMX, "NAMERVR length is incorrect"
    
    # Check RVROUT
    assert RVROUT.shape == (NRVRMX,), "RVROUT shape is incorrect"
    
    print("✅ Constants test passed!")


def test_compute_lake_enthalpy():
    """Test compute_lake_enthalpy function."""
    print("Testing compute_lake_enthalpy...")
    
    # Create test tlake and mwl arrays
    tlake = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 30.0 - 10.0
    mwl = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0 + 10.0
    
    # Compute lake enthalpy
    gml = compute_lake_enthalpy(tlake, mwl)
    
    # Check shape
    assert gml.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for gml, got {gml.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(gml)), "gml contains NaN"
    assert not jnp.any(jnp.isinf(gml)), "gml contains Inf"
    
    # Check values (gml = tlake * mwl * shw)
    expected_gml = tlake * mwl * SHW
    assert jnp.allclose(gml, expected_gml, rtol=1e-5), "gml values are incorrect"
    
    print("✅ compute_lake_enthalpy test passed!")


def test_compute_lake_depth():
    """Test compute_lake_depth function."""
    print("Testing compute_lake_depth...")
    
    # Create test mwl and flake arrays
    mwl = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0 + 10.0
    flake = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute lake depth
    dlake = compute_lake_depth(mwl, flake)
    
    # Check shape
    assert dlake.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for dlake, got {dlake.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(dlake)), "dlake contains NaN"
    assert not jnp.any(jnp.isinf(dlake)), "dlake contains Inf"
    
    # Check values (dlake = mwl / (rhow * flake))
    expected_dlake = jnp.where(
        flake > 0.0,
        mwl / (RHOW * flake),
        0.0
    )
    assert jnp.allclose(dlake, expected_dlake, rtol=1e-5), "dlake values are incorrect"
    
    print("✅ compute_lake_depth test passed!")


def test_compute_lake_glake():
    """Test compute_lake_glake function."""
    print("Testing compute_lake_glake...")
    
    # Create test gml and flake arrays
    gml = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1e8
    flake = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute GLAKE
    glake = compute_lake_glake(gml, flake)
    
    # Check shape
    assert glake.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for glake, got {glake.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(glake)), "glake contains NaN"
    assert not jnp.any(jnp.isinf(glake)), "glake contains Inf"
    
    # Check values (glake = gml / flake)
    expected_glake = jnp.where(
        flake > 0.0,
        gml / flake,
        0.0
    )
    assert jnp.allclose(glake, expected_glake, rtol=1e-5), "glake values are incorrect"
    
    print("✅ compute_lake_glake test passed!")


def test_compute_mixed_layer_depth():
    """Test compute_mixed_layer_depth function."""
    print("Testing compute_mixed_layer_depth...")
    
    # Create test mwl and flake arrays
    mwl = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0 + 10.0
    flake = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute mixed layer depth
    mldlk = compute_mixed_layer_depth(mwl, flake)
    
    # Check shape
    assert mldlk.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for mldlk, got {mldlk.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(mldlk)), "mldlk contains NaN"
    assert not jnp.any(jnp.isinf(mldlk)), "mldlk contains Inf"
    
    # Check values (mldlk = mwl / (rhow * flake))
    expected_mldlk = jnp.where(
        flake > 0.0,
        mwl / (RHOW * flake),
        0.0
    )
    assert jnp.allclose(mldlk, expected_mldlk, rtol=1e-5), "mldlk values are incorrect"
    
    print("✅ compute_mixed_layer_depth test passed!")


def test_initialize_lakes_com():
    """Test initialize_lakes_com function."""
    print("Testing initialize_lakes_com...")
    
    # Initialize LAKES_COM arrays
    lakes_com = initialize_lakes_com(IM_TEST, JM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "MWL", "GML", "TLAKE",
        "MDLK", "FLAKE", "TANLK",
        "SVFLAKE", "T2LBOT", "EKT",
        "DLAKE", "GLAKE", "DLAKE0",
    ]
    for key in expected_keys:
        assert key in lakes_com, f"{key} is missing from lakes_com"
    
    # Check shapes
    assert lakes_com["MWL"].shape == (IM_TEST, JM_TEST), f"MWL shape is incorrect"
    assert lakes_com["GML"].shape == (IM_TEST, JM_TEST), f"GML shape is incorrect"
    assert lakes_com["TLAKE"].shape == (IM_TEST, JM_TEST), f"TLAKE shape is incorrect"
    assert lakes_com["MDLK"].shape == (IM_TEST, JM_TEST), f"MDLK shape is incorrect"
    assert lakes_com["FLAKE"].shape == (IM_TEST, JM_TEST), f"FLAKE shape is incorrect"
    assert lakes_com["DLAKE"].shape == (IM_TEST, JM_TEST), f"DLAKE shape is incorrect"
    
    # Check all arrays are initialized to zero
    for key in expected_keys:
        assert jnp.all(lakes_com[key] == 0.0), f"{key} should be initialized to zero"
    
    print("✅ initialize_lakes_com test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing LAKES_COM JAX Implementation")
    print("=" * 70)
    
    test_constants()
    test_compute_lake_enthalpy()
    test_compute_lake_depth()
    test_compute_lake_glake()
    test_compute_mixed_layer_depth()
    test_initialize_lakes_com()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
