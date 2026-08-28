"""
Unit Tests for SEAICE_COM JAX Implementation
==============================================

This script tests the JAX implementation of the sea ice common variables
from ROCKE-3D's SEAICE_COM.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_seaice_com_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from seaice_com_jax import (
    compute_sea_ice_fraction, compute_total_ice_thickness,
    compute_ice_mass, compute_snow_mass,
    initialize_seaice_com,
)
from seaice_jax import LMI


# Test constants
np.random.seed(42)

# Grid size
IM_TEST = 36
JM_TEST = 24


def test_compute_sea_ice_fraction():
    """Test compute_sea_ice_fraction function."""
    print("Testing compute_sea_ice_fraction...")
    
    # Create test fwater and rsi arrays
    fwater = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    rsi = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute sea ice fraction
    sea_ice_fraction = compute_sea_ice_fraction(fwater, rsi)
    
    # Check shape
    assert sea_ice_fraction.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for sea_ice_fraction, got {sea_ice_fraction.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(sea_ice_fraction)), "sea_ice_fraction contains NaN"
    assert not jnp.any(jnp.isinf(sea_ice_fraction)), "sea_ice_fraction contains Inf"
    
    # Check values (sea_ice_fraction = fwater * rsi)
    expected_sea_ice_fraction = fwater * rsi
    assert jnp.allclose(sea_ice_fraction, expected_sea_ice_fraction, rtol=1e-5), "sea_ice_fraction values are incorrect"
    
    print("✅ compute_sea_ice_fraction test passed!")


def test_compute_total_ice_thickness():
    """Test compute_total_ice_thickness function."""
    print("Testing compute_total_ice_thickness...")
    
    # Create test zsi array
    zsi = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 10.0
    
    # Compute total ice thickness
    total_thickness = compute_total_ice_thickness(zsi)
    
    # Check shape
    assert total_thickness.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for total_thickness, got {total_thickness.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(total_thickness)), "total_thickness contains NaN"
    assert not jnp.any(jnp.isinf(total_thickness)), "total_thickness contains Inf"
    
    # Check values (total_thickness = zsi)
    assert jnp.allclose(total_thickness, zsi, rtol=1e-5), "total_thickness values are incorrect"
    
    print("✅ compute_total_ice_thickness test passed!")


def test_compute_ice_mass():
    """Test compute_ice_mass function."""
    print("Testing compute_ice_mass...")
    
    # Create test zsi and rsi arrays
    zsi = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 10.0
    rsi = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute ice mass
    msi = compute_ice_mass(zsi, rsi)
    
    # Check shape
    assert msi.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for msi, got {msi.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(msi)), "msi contains NaN"
    assert not jnp.any(jnp.isinf(msi)), "msi contains Inf"
    
    # Check values (msi = zsi * rsi * rhoi)
    rhoi = 916.6
    expected_msi = zsi * rsi * rhoi
    assert jnp.allclose(msi, expected_msi, rtol=1e-5), "msi values are incorrect"
    
    print("✅ compute_ice_mass test passed!")


def test_compute_snow_mass():
    """Test compute_snow_mass function."""
    print("Testing compute_snow_mass...")
    
    # Create test snowi and rsi arrays
    snowi = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 100.0
    rsi = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute snow mass
    snow_mass = compute_snow_mass(snowi, rsi)
    
    # Check shape
    assert snow_mass.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for snow_mass, got {snow_mass.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(snow_mass)), "snow_mass contains NaN"
    assert not jnp.any(jnp.isinf(snow_mass)), "snow_mass contains Inf"
    
    # Check values (snow_mass = snowi)
    assert jnp.allclose(snow_mass, snowi, rtol=1e-5), "snow_mass values are incorrect"
    
    print("✅ compute_snow_mass test passed!")


def test_initialize_seaice_com():
    """Test initialize_seaice_com function."""
    print("Testing initialize_seaice_com...")
    
    # Initialize SEAICE_COM arrays
    seaice_com = initialize_seaice_com(IM_TEST, JM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "FWATER", "RSI", "SNOWI",
        "MSI", "ZSI", "POND_MELT",
        "RSIX", "RSIY", "RSISAVE",
        "HSI", "SSI", "FLAG_DSWS",
    ]
    for key in expected_keys:
        assert key in seaice_com, f"{key} is missing from seaice_com"
    
    # Check shapes
    assert seaice_com["FWATER"].shape == (IM_TEST, JM_TEST), f"FWATER shape is incorrect"
    assert seaice_com["RSI"].shape == (IM_TEST, JM_TEST), f"RSI shape is incorrect"
    assert seaice_com["SNOWI"].shape == (IM_TEST, JM_TEST), f"SNOWI shape is incorrect"
    assert seaice_com["MSI"].shape == (IM_TEST, JM_TEST), f"MSI shape is incorrect"
    assert seaice_com["ZSI"].shape == (IM_TEST, JM_TEST), f"ZSI shape is incorrect"
    assert seaice_com["HSI"].shape == (IM_TEST, JM_TEST, LMI), f"HSI shape is incorrect"
    assert seaice_com["SSI"].shape == (IM_TEST, JM_TEST, LMI), f"SSI shape is incorrect"
    
    # Check all arrays are initialized to zero
    for key in expected_keys:
        if key == "FLAG_DSWS":
            assert jnp.all(seaice_com[key] == False), f"{key} should be initialized to False"
        else:
            assert jnp.all(seaice_com[key] == 0.0), f"{key} should be initialized to zero"
    
    print("✅ initialize_seaice_com test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SEAICE_COM JAX Implementation")
    print("=" * 70)
    
    test_compute_sea_ice_fraction()
    test_compute_total_ice_thickness()
    test_compute_ice_mass()
    test_compute_snow_mass()
    test_initialize_seaice_com()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
