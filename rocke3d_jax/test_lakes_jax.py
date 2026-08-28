"""
Unit Tests for LAKES JAX Implementation
========================================

This script tests the JAX implementation of the lake model
from ROCKE-3D's LAKES.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_lakes_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from lakes_jax import (
    MINMLD, HLAKE_MIN, TMAXRHO, KVLAKE, TFL,
    AC1LMIN, AC2LMIN, FLEADLK, BYZETA,
    RIVER_FAC, INIT_FLAKE, VARIABLE_LK,
    LAKE_RISE_MAX, LAKE_ICE_MAX, POWER_LAW_LAKES, C_LAKE,
    compute_lake_temperature, compute_lake_heat_content,
    compute_mixed_layer_depth, compute_lake_ice_mass,
    initialize_lakes,
)
from constant_jax import SHW, RHOW


# Test constants
np.random.seed(42)

# Grid size
IM_TEST = 36
JM_TEST = 24


def test_constants():
    """Test lake constants."""
    print("Testing constants...")
    
    # Check MINMLD
    assert jnp.abs(MINMLD - 1.0) < 1e-5, "MINMLD is incorrect"
    
    # Check HLAKE_MIN
    assert jnp.abs(HLAKE_MIN - 1.0) < 1e-5, "HLAKE_MIN is incorrect"
    
    # Check TMAXRHO
    assert jnp.abs(TMAXRHO - 4.0) < 1e-5, "TMAXRHO is incorrect"
    
    # Check KVLAKE
    assert jnp.abs(KVLAKE - 1e-5) < 1e-10, "KVLAKE is incorrect"
    
    # Check TFL
    assert jnp.abs(TFL - 0.0) < 1e-5, "TFL is incorrect"
    
    # Check AC1LMIN
    assert jnp.abs(AC1LMIN - 0.1) < 1e-5, "AC1LMIN is incorrect"
    
    # Check AC2LMIN
    assert jnp.abs(AC2LMIN - 0.1) < 1e-5, "AC2LMIN is incorrect"
    
    # Check FLEADLK
    assert jnp.abs(FLEADLK - 0.0) < 1e-5, "FLEADLK is incorrect"
    
    # Check BYZETA
    assert jnp.abs(BYZETA - 1.0 / 0.35) < 1e-5, "BYZETA is incorrect"
    
    # Check RIVER_FAC
    assert jnp.abs(RIVER_FAC - 1.0) < 1e-5, "RIVER_FAC is incorrect"
    
    # Check INIT_FLAKE
    assert INIT_FLAKE == 1, "INIT_FLAKE should be 1"
    
    # Check VARIABLE_LK
    assert VARIABLE_LK == 0, "VARIABLE_LK should be 0"
    
    # Check LAKE_RISE_MAX
    assert jnp.abs(LAKE_RISE_MAX - 100.0) < 1e-5, "LAKE_RISE_MAX is incorrect"
    
    # Check LAKE_ICE_MAX
    assert jnp.abs(LAKE_ICE_MAX - 5.0) < 1e-5, "LAKE_ICE_MAX is incorrect"
    
    # Check POWER_LAW_LAKES
    assert POWER_LAW_LAKES == 0, "POWER_LAW_LAKES should be 0"
    
    # Check C_LAKE
    assert jnp.abs(C_LAKE - 0.235) < 1e-5, "C_LAKE is incorrect"
    
    print("✅ Constants test passed!")


def test_compute_lake_temperature():
    """Test compute_lake_temperature function."""
    print("Testing compute_lake_temperature...")
    
    # Create test gml and mwl arrays
    gml = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1e8
    mwl = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0 + 10.0
    
    # Compute lake temperature
    tlake = compute_lake_temperature(gml, mwl)
    
    # Check shape
    assert tlake.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for tlake, got {tlake.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(tlake)), "tlake contains NaN"
    assert not jnp.any(jnp.isinf(tlake)), "tlake contains Inf"
    
    # Check values (tlake = gml / (mwl * shw))
    expected_tlake = gml / (mwl * SHW)
    assert jnp.allclose(tlake, expected_tlake, rtol=1e-5), "tlake values are incorrect"
    
    print("✅ compute_lake_temperature test passed!")


def test_compute_lake_heat_content():
    """Test compute_lake_heat_content function."""
    print("Testing compute_lake_heat_content...")
    
    # Create test tlake and mwl arrays
    tlake = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 30.0 - 10.0
    mwl = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0 + 10.0
    
    # Compute lake heat content
    gml = compute_lake_heat_content(tlake, mwl)
    
    # Check shape
    assert gml.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for gml, got {gml.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(gml)), "gml contains NaN"
    assert not jnp.any(jnp.isinf(gml)), "gml contains Inf"
    
    # Check values (gml = tlake * mwl * shw)
    expected_gml = tlake * mwl * SHW
    assert jnp.allclose(gml, expected_gml, rtol=1e-5), "gml values are incorrect"
    
    print("✅ compute_lake_heat_content test passed!")


def test_compute_mixed_layer_depth():
    """Test compute_mixed_layer_depth function."""
    print("Testing compute_mixed_layer_depth...")
    
    # Create test mwl and flake arrays
    mwl = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0 + 10.0
    flake = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    
    # Compute mixed layer depth
    hlake = compute_mixed_layer_depth(mwl, flake)
    
    # Check shape
    assert hlake.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for hlake, got {hlake.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(hlake)), "hlake contains NaN"
    assert not jnp.any(jnp.isinf(hlake)), "hlake contains Inf"
    
    # Check values (hlake = max(mwl / (rhow * flake), minmld))
    expected_hlake = jnp.where(
        flake > 0.0,
        jnp.maximum(mwl / (RHOW * flake), MINMLD),
        0.0
    )
    assert jnp.allclose(hlake, expected_hlake, rtol=1e-5), "hlake values are incorrect"
    
    print("✅ compute_mixed_layer_depth test passed!")


def test_compute_lake_ice_mass():
    """Test compute_lake_ice_mass function."""
    print("Testing compute_lake_ice_mass...")
    
    # Create test icelak array
    icelak = np.random.rand(IM_TEST, JM_TEST, 2).astype(np.float32) * 100.0
    
    # Compute total lake ice mass
    total_ice = compute_lake_ice_mass(icelak)
    
    # Check shape
    assert total_ice.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for total_ice, got {total_ice.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(total_ice)), "total_ice contains NaN"
    assert not jnp.any(jnp.isinf(total_ice)), "total_ice contains Inf"
    
    # Check values (total_ice = sum(icelak, axis=2))
    expected_total_ice = jnp.sum(icelak, axis=2)
    assert jnp.allclose(total_ice, expected_total_ice, rtol=1e-5), "total_ice values are incorrect"
    
    print("✅ compute_lake_ice_mass test passed!")


def test_initialize_lakes():
    """Test initialize_lakes function."""
    print("Testing initialize_lakes...")
    
    # Initialize LAKES arrays
    lakes = initialize_lakes(IM_TEST, JM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "FLAKE", "MWL", "GML",
        "TLAKE", "ICELAK", "SNOWLAK", "TSLAKE",
    ]
    for key in expected_keys:
        assert key in lakes, f"{key} is missing from lakes"
    
    # Check shapes
    assert lakes["FLAKE"].shape == (IM_TEST, JM_TEST), f"FLAKE shape is incorrect"
    assert lakes["MWL"].shape == (IM_TEST, JM_TEST), f"MWL shape is incorrect"
    assert lakes["GML"].shape == (IM_TEST, JM_TEST), f"GML shape is incorrect"
    assert lakes["TLAKE"].shape == (IM_TEST, JM_TEST), f"TLAKE shape is incorrect"
    assert lakes["ICELAK"].shape == (IM_TEST, JM_TEST, 2), f"ICELAK shape is incorrect"
    assert lakes["SNOWLAK"].shape == (IM_TEST, JM_TEST), f"SNOWLAK shape is incorrect"
    
    # Check all arrays are initialized to zero
    for key in expected_keys:
        assert jnp.all(lakes[key] == 0.0), f"{key} should be initialized to zero"
    
    print("✅ initialize_lakes test passed!")


def test_lkmix():
    """Test lake mixing."""
    print("Testing lkmix...")
    
    # Inputs
    im, jm = 2, 2
    mwl = jnp.ones((im, jm)) * 1000.0  # Lake water mass [kg/m^2]
    gml = jnp.ones((im, jm)) * 1e7  # Lake heat content [J/m^2]
    flake = jnp.ones((im, jm)) * 0.5  # Lake fraction [1]
    tlake = jnp.ones((im, jm)) * 10.0  # Lake temperature [C]
    
    # Call lkmix
    from lakes_jax import lkmix
    mwl_new, gml_new, tlake_new = lkmix(mwl, gml, flake, tlake)
    
    # Check that outputs are unchanged (placeholder implementation)
    assert jnp.allclose(mwl_new, mwl), f"MWL should be unchanged: {mwl_new} vs {mwl}"
    assert jnp.allclose(gml_new, gml), f"GML should be unchanged: {gml_new} vs {gml}"
    assert jnp.allclose(tlake_new, tlake), f"TLAKE should be unchanged: {tlake_new} vs {tlake}"
    
    print("✅ lkmix passed!")


def test_precip_lk():
    """Test precipitation onto lakes."""
    print("Testing precip_lk...")
    
    # Inputs
    im, jm = 2, 2
    mwl = jnp.ones((im, jm)) * 1000.0  # Lake water mass [kg/m^2]
    gml = jnp.ones((im, jm)) * 1e7  # Lake heat content [J/m^2]
    flake = jnp.ones((im, jm)) * 0.5  # Lake fraction [1]
    tlake = jnp.ones((im, jm)) * 10.0  # Lake temperature [C]
    prcp = jnp.ones((im, jm)) * 5.0  # Precipitation [kg/m^2]
    enrgp = jnp.ones((im, jm)) * 1e5  # Energy of precipitation [J/m^2]
    
    # Call precip_lk
    from lakes_jax import precip_lk
    mwl_new, gml_new, tlake_new = precip_lk(mwl, gml, flake, tlake, prcp, enrgp)
    
    # Check that lake water mass increased
    assert jnp.all(mwl_new >= mwl), f"MWL should increase: {mwl_new} vs {mwl}"
    
    # Check that lake heat content increased
    assert jnp.all(gml_new >= gml), f"GML should increase: {gml_new} vs {gml}"
    
    print("✅ precip_lk passed!")


def test_ground_lk():
    """Test ground energy balance for lakes."""
    print("Testing ground_lk...")
    
    # Inputs
    im, jm = 2, 2
    mwl = jnp.ones((im, jm)) * 1000.0  # Lake water mass [kg/m^2]
    gml = jnp.ones((im, jm)) * 1e7  # Lake heat content [J/m^2]
    flake = jnp.ones((im, jm)) * 0.5  # Lake fraction [1]
    tlake = jnp.ones((im, jm)) * 10.0  # Lake temperature [C]
    fsf = jnp.ones((im, jm)) * 200.0  # Downwelling shortwave radiation [W/m^2]
    flong = jnp.ones((im, jm)) * 300.0  # Downwelling longwave radiation [W/m^2]
    albedo = jnp.ones((im, jm)) * 0.1  # Surface albedo [1]
    
    # Call ground_lk
    from lakes_jax import ground_lk
    mwl_new, gml_new, tlake_new = ground_lk(mwl, gml, flake, tlake, fsf, flong, albedo)
    
    # Check that lake heat content changed (due to radiation)
    assert jnp.all(gml_new != gml), f"GML should change: {gml_new} vs {gml}"
    
    # Check that lake temperature changed
    assert jnp.all(tlake_new != tlake), f"TLAKE should change: {tlake_new} vs {tlake}"
    
    print("✅ ground_lk passed!")


def test_minmld():
    """Test minimum mixed layer depth."""
    print("Testing minmld...")
    
    # Inputs
    im, jm = 2, 2
    mwl = jnp.ones((im, jm)) * 1000.0  # Lake water mass [kg/m^2]
    flake = jnp.ones((im, jm)) * 0.5  # Lake fraction [1]
    
    # Call minmld
    from lakes_jax import minmld
    min_mld = minmld(mwl, flake)
    
    # Check that min_mld is positive
    assert jnp.all(min_mld > 0.0), f"min_mld should be positive: {min_mld}"
    
    # Check that min_mld is >= MINMLD
    assert jnp.all(min_mld >= MINMLD), f"min_mld should be >= MINMLD: {min_mld} vs {MINMLD}"
    
    print("✅ minmld passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing LAKES JAX Implementation")
    print("=" * 70)
    
    test_constants()
    test_compute_lake_temperature()
    test_compute_lake_heat_content()
    test_compute_mixed_layer_depth()
    test_compute_lake_ice_mass()
    test_initialize_lakes()
    test_lkmix()
    test_precip_lk()
    test_ground_lk()
    test_minmld()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
