"""
Unit Tests for RAD_COM JAX Implementation
=========================================

This script tests the JAX implementation of the radiation common variables
from ROCKE-3D's RAD_COM.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_rad_com_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from rad_com_jax import (
    NRAD, OMEGT_DEF, OBLIQ_DEF, ECCN_DEF,
    OMEGT, OBLIQ, ECCN,
    VARIABLE_ORB_PAR, ORB_PAR_YEAR_BP, ORB_PAR,
    IM, JM, LM, LM_REQ, LM_TOTAL,
    compute_solar_forcing, compute_solar_fluxes, compute_thermal_fluxes,
    initialize_rad_com,
)
from constant_jax import SOLAR_CONSTANT, STBO


# Test constants
np.random.seed(42)

# Grid size
IM_TEST = 36
JM_TEST = 24
LM_TEST = 10


def test_constants():
    """Test radiation constants."""
    print("Testing constants...")
    
    # Check NRAD
    assert NRAD == 5, "NRAD should be 5"
    
    # Check orbital parameters
    assert jnp.abs(OMEGT_DEF - 282.9) < 1e-5, "OMEGT_DEF is incorrect"
    assert jnp.abs(OBLIQ_DEF - 23.44) < 1e-5, "OBLIQ_DEF is incorrect"
    assert jnp.abs(ECCN_DEF - 0.0167) < 1e-5, "ECCN_DEF is incorrect"
    
    # Check actual orbital parameters
    assert jnp.abs(OMEGT - OMEGT_DEF) < 1e-5, "OMEGT should match OMEGT_DEF"
    assert jnp.abs(OBLIQ - OBLIQ_DEF) < 1e-5, "OBLIQ should match OBLIQ_DEF"
    assert jnp.abs(ECCN - ECCN_DEF) < 1e-5, "ECCN should match ECCN_DEF"
    
    # Check orbital parameter control
    assert VARIABLE_ORB_PAR == -2, "VARIABLE_ORB_PAR should be -2"
    assert ORB_PAR_YEAR_BP == 0, "ORB_PAR_YEAR_BP should be 0"
    assert jnp.allclose(ORB_PAR, jnp.array([ECCN_DEF, OBLIQ_DEF, OMEGT_DEF])), "ORB_PAR is incorrect"
    
    # Check default model resolution
    assert IM == 72, "IM should be 72"
    assert JM == 46, "JM should be 46"
    assert LM == 40, "LM should be 40"
    assert LM_REQ == 3, "LM_REQ should be 3"
    assert LM_TOTAL == LM + LM_REQ, "LM_TOTAL should be LM + LM_REQ"
    
    print("✅ Constants test passed!")


def test_compute_solar_forcing():
    """Test compute_solar_forcing function."""
    print("Testing compute_solar_forcing...")
    
    # Create test inputs
    cosz = np.random.rand(IM_TEST, JM_TEST).astype(np.float32)
    itype = np.random.randint(1, 5, (IM_TEST, JM_TEST)).astype(np.int32)
    
    # Compute solar forcing
    fsf = compute_solar_forcing(SOLAR_CONSTANT, cosz, itype)
    
    # Check shape
    assert fsf.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for fsf, got {fsf.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(fsf)), "fsf contains NaN"
    assert not jnp.any(jnp.isinf(fsf)), "fsf contains Inf"
    
    # Check values (fsf = solar_constant * cosz)
    expected_fsf = SOLAR_CONSTANT * cosz
    assert jnp.allclose(fsf, expected_fsf, rtol=1e-5), "fsf values are incorrect"
    
    print("✅ compute_solar_forcing test passed!")


def test_compute_solar_fluxes():
    """Test compute_solar_fluxes function."""
    print("Testing compute_solar_fluxes...")
    
    # Create test inputs
    fsf = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 1000.0
    albedo = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 0.5 + 0.1
    
    # Compute solar fluxes
    srdn, srvis, fsrdir = compute_solar_fluxes(fsf, albedo)
    
    # Check shapes
    assert srdn.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for srdn, got {srdn.shape}"
    assert srvis.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for srvis, got {srvis.shape}"
    assert fsrdir.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for fsrdir, got {fsrdir.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(srdn)), "srdn contains NaN"
    assert not jnp.any(jnp.isnan(srvis)), "srvis contains NaN"
    assert not jnp.any(jnp.isnan(fsrdir)), "fsrdir contains NaN"
    
    # Check values (srdn = fsf, srvis = 0.5 * fsf, fsrdir = 1.0)
    assert jnp.allclose(srdn, fsf, rtol=1e-5), "srdn values are incorrect"
    assert jnp.allclose(srvis, 0.5 * fsf, rtol=1e-5), "srvis values are incorrect"
    assert jnp.allclose(fsrdir, 1.0, rtol=1e-5), "fsrdir values are incorrect"
    
    print("✅ compute_solar_fluxes test passed!")


def test_compute_thermal_fluxes():
    """Test compute_thermal_fluxes function."""
    print("Testing compute_thermal_fluxes...")
    
    # Create test inputs
    temperature = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 100.0 + 200.0
    emissivity = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 0.2 + 0.8
    
    # Compute thermal fluxes
    trhr = compute_thermal_fluxes(temperature, emissivity)
    
    # Check shape
    assert trhr.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for trhr, got {trhr.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(trhr)), "trhr contains NaN"
    assert not jnp.any(jnp.isinf(trhr)), "trhr contains Inf"
    
    # Check values (trhr = emissivity * STBO * temperature^4)
    expected_trhr = emissivity * STBO * temperature**4
    assert jnp.allclose(trhr, expected_trhr, rtol=1e-5), "trhr values are incorrect"
    
    print("✅ compute_thermal_fluxes test passed!")


def test_initialize_rad_com():
    """Test initialize_rad_com function."""
    print("Testing initialize_rad_com...")
    
    # Initialize RAD_COM arrays
    rad_com = initialize_rad_com(IM_TEST, JM_TEST, LM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "RQT", "TCHG", "SRHR", "TRHR",
        "TRSURF", "FSF", "FSRDIR",
        "DIRVIS", "SRVISSURF", "SRDN",
        "FSRDIF", "DIRNIR", "DIFNIR",
        "SRNFLB_SAVE", "TRNFLB_SAVE",
        "TAUSUMW", "TAUSUMI",
    ]
    for key in expected_keys:
        assert key in rad_com, f"{key} is missing from rad_com"
    
    # Check shapes
    lm_total = LM_TEST + LM_REQ
    assert rad_com["RQT"].shape == (IM_TEST, JM_TEST, LM_REQ), f"RQT shape is incorrect"
    assert rad_com["TCHG"].shape == (IM_TEST, JM_TEST, lm_total), f"TCHG shape is incorrect"
    assert rad_com["SRHR"].shape == (IM_TEST, JM_TEST, lm_total + 1), f"SRHR shape is incorrect"
    assert rad_com["TRHR"].shape == (IM_TEST, JM_TEST, lm_total + 1), f"TRHR shape is incorrect"
    assert rad_com["FSF"].shape == (IM_TEST, JM_TEST, 4), f"FSF shape is incorrect"
    assert rad_com["FSRDIR"].shape == (IM_TEST, JM_TEST), f"FSRDIR shape is incorrect"
    assert rad_com["TAUSUMW"].shape == (IM_TEST, JM_TEST), f"TAUSUMW shape is incorrect"
    
    # Check all arrays are initialized to zero
    for key in expected_keys:
        assert jnp.all(rad_com[key] == 0.0), f"{key} should be initialized to zero"
    
    print("✅ initialize_rad_com test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing RAD_COM JAX Implementation")
    print("=" * 70)
    
    test_constants()
    test_compute_solar_forcing()
    test_compute_solar_fluxes()
    test_compute_thermal_fluxes()
    test_initialize_rad_com()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
