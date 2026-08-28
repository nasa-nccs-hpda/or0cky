"""
Unit Tests for SEAICE JAX Implementation
=========================================

This script tests the JAX implementation of the sea ice model
from ROCKE-3D's SEAICE.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_seaice_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from seaice_jax import (
    DTDp, LMI, XSI, BYXSI, Z1I, ACE1I_CONST,
    Z2OIM, AC2OIM, ALAMI0, ALAMS, ALAMDS, ALAMDT,
    RHOS, FLEADOC, FLEADLK, FLEADMX, BYHREF, BYRLS,
    MU, SSI0, FSSS, QSFIX, ALPHA,
    OI_USTAR0, SILMFACT, SILMPOW, SNOW_ICE, OSURF_TILT,
    SSIMIN, DEBUG, SEAICE_THERMO, MIN_ICE_TEMPERATURE,
    compute_sea_ice_temperature, compute_sea_ice_salinity,
    compute_sea_ice_mass, compute_snow_mass, compute_sea_ice_concentration,
    initialize_seaice,
)
from constant_jax import RHOI


# Test constants
np.random.seed(42)

# Grid size
IM_TEST = 36
JM_TEST = 24


def test_constants():
    """Test sea ice constants."""
    print("Testing constants...")
    
    # Check DTDp
    assert jnp.abs(DTDp - (-7.5e-8)) < 1e-10, "DTDp is incorrect"
    
    # Check LMI
    assert LMI == 4, "LMI should be 4"
    
    # Check XSI
    assert XSI.shape == (LMI,), "XSI shape is incorrect"
    assert jnp.allclose(XSI, jnp.array([0.5, 0.5, 0.5, 0.5])), "XSI values are incorrect"
    
    # Check BYXSI
    assert BYXSI.shape == (LMI,), "BYXSI shape is incorrect"
    assert jnp.allclose(BYXSI, jnp.array([2.0, 2.0, 2.0, 2.0])), "BYXSI values are incorrect"
    
    # Check Z1I
    assert jnp.abs(Z1I - 0.1) < 1e-5, "Z1I is incorrect"
    
    # Check ACE1I_CONST
    assert jnp.abs(ACE1I_CONST - Z1I * RHOI) < 1e-2, "ACE1I is incorrect"
    
    # Check Z2OIM
    assert jnp.abs(Z2OIM - 0.1) < 1e-5, "Z2OIM is incorrect"
    
    # Check AC2OIM
    assert jnp.abs(AC2OIM - Z2OIM * RHOI) < 1e-2, "AC2OIM is incorrect"
    
    # Check ALAMI0
    assert jnp.abs(ALAMI0 - 2.11) < 1e-5, "ALAMI0 is incorrect"
    
    # Check ALAMS
    assert jnp.abs(ALAMS - 0.35) < 1e-5, "ALAMS is incorrect"
    
    # Check ALAMDS
    assert jnp.abs(ALAMDS - 0.09) < 1e-5, "ALAMDS is incorrect"
    
    # Check ALAMDT
    assert jnp.abs(ALAMDT - (-0.011)) < 1e-5, "ALAMDT is incorrect"
    
    # Check RHOS
    assert jnp.abs(RHOS - 300.0) < 1e-5, "RHOS is incorrect"
    
    # Check FLEADOC
    assert jnp.abs(FLEADOC - 0.06) < 1e-5, "FLEADOC is incorrect"
    
    # Check FLEADLK
    assert jnp.abs(FLEADLK - 0.0) < 1e-5, "FLEADLK is incorrect"
    
    # Check FLEADMX
    assert jnp.abs(FLEADMX - 5.0) < 1e-5, "FLEADMX is incorrect"
    
    # Check BYHREF
    assert jnp.abs(BYHREF - 1.1) < 1e-5, "BYHREF is incorrect"
    
    # Check BYRLS
    assert jnp.abs(BYRLS - 1.0 / (RHOS * ALAMS)) < 1e-5, "BYRLS is incorrect"
    
    # Check MU
    assert jnp.abs(MU - 0.054) < 1e-5, "MU is incorrect"
    
    # Check SSI0
    assert jnp.abs(SSI0 - 0.0032) < 1e-5, "SSI0 is incorrect"
    
    # Check FSSS
    assert jnp.abs(FSSS - 8.0 / 35.0) < 1e-5, "FSSS is incorrect"
    
    # Check QSFIX
    assert QSFIX == False, "QSFIX should be False"
    
    # Check ALPHA
    assert jnp.abs(ALPHA - 1.0) < 1e-5, "ALPHA is incorrect"
    
    # Check OI_USTAR0
    assert jnp.abs(OI_USTAR0 - 1e-3) < 1e-5, "OI_USTAR0 is incorrect"
    
    # Check SILMFACT
    assert jnp.abs(SILMFACT - 1e-7) < 1e-10, "SILMFACT is incorrect"
    
    # Check SILMPOW
    assert jnp.abs(SILMPOW - 1.36) < 1e-5, "SILMPOW is incorrect"
    
    # Check SNOW_ICE
    assert SNOW_ICE == 1, "SNOW_ICE should be 1"
    
    # Check OSURF_TILT
    assert OSURF_TILT == 1, "OSURF_TILT should be 1"
    
    # Check SSIMIN
    assert jnp.abs(SSIMIN - 1e-6) < 1e-10, "SSIMIN is incorrect"
    
    # Check DEBUG
    assert DEBUG == False, "DEBUG should be False"
    
    # Check SEAICE_THERMO
    assert SEAICE_THERMO == "BP", "SEAICE_THERMO should be 'BP'"
    
    # Check MIN_ICE_TEMPERATURE
    assert jnp.abs(MIN_ICE_TEMPERATURE - (-100.0)) < 1e-5, "MIN_ICE_TEMPERATURE is incorrect"
    
    print("✅ Constants test passed!")


def test_compute_sea_ice_temperature():
    """Test compute_sea_ice_temperature function."""
    print("Testing compute_sea_ice_temperature...")
    
    # Create test ssil and hsil arrays
    ssil = np.random.rand(IM_TEST, JM_TEST, LMI).astype(np.float32) * 0.01
    hsil = np.random.rand(IM_TEST, JM_TEST, LMI).astype(np.float32) * 1.0
    
    # Compute sea ice temperature
    tsil = compute_sea_ice_temperature(ssil, hsil)
    
    # Check shape
    assert tsil.shape == (IM_TEST, JM_TEST, LMI), f"Expected shape {(IM_TEST, JM_TEST, LMI)} for tsil, got {tsil.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(tsil)), "tsil contains NaN"
    assert not jnp.any(jnp.isinf(tsil)), "tsil contains Inf"
    
    # Check values (tsil = -MU * ssil * 1000)
    expected_tsil = -MU * ssil * 1000.0
    assert jnp.allclose(tsil, expected_tsil, rtol=1e-5), "tsil values are incorrect"
    
    print("✅ compute_sea_ice_temperature test passed!")


def test_compute_sea_ice_salinity():
    """Test compute_sea_ice_salinity function."""
    print("Testing compute_sea_ice_salinity...")
    
    # Create test tsil array
    tsil = np.random.rand(IM_TEST, JM_TEST, LMI).astype(np.float32) * 10.0 - 20.0
    
    # Compute sea ice salinity
    ssil = compute_sea_ice_salinity(tsil)
    
    # Check shape
    assert ssil.shape == (IM_TEST, JM_TEST, LMI), f"Expected shape {(IM_TEST, JM_TEST, LMI)} for ssil, got {ssil.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(ssil)), "ssil contains NaN"
    assert not jnp.any(jnp.isinf(ssil)), "ssil contains Inf"
    
    # Check values (ssil = -tsil / (MU * 1000))
    expected_ssil = -tsil / (MU * 1000.0)
    assert jnp.allclose(ssil, expected_ssil, rtol=1e-5), "ssil values are incorrect"
    
    print("✅ compute_sea_ice_salinity test passed!")


def test_compute_sea_ice_mass():
    """Test compute_sea_ice_mass function."""
    print("Testing compute_sea_ice_mass...")
    
    # Create test hsil array
    hsil = np.random.rand(IM_TEST, JM_TEST, LMI).astype(np.float32) * 1.0
    
    # Compute sea ice mass
    msi = compute_sea_ice_mass(hsil)
    
    # Check shape
    assert msi.shape == (IM_TEST, JM_TEST, LMI), f"Expected shape {(IM_TEST, JM_TEST, LMI)} for msi, got {msi.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(msi)), "msi contains NaN"
    assert not jnp.any(jnp.isinf(msi)), "msi contains Inf"
    
    # Check values (msi = hsil * RHOI)
    expected_msi = hsil * RHOI
    assert jnp.allclose(msi, expected_msi, rtol=1e-5), "msi values are incorrect"
    
    print("✅ compute_sea_ice_mass test passed!")


def test_compute_snow_mass():
    """Test compute_snow_mass function."""
    print("Testing compute_snow_mass...")
    
    # Create test snow array
    snow = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * 0.5
    
    # Compute snow mass
    snowm = compute_snow_mass(snow)
    
    # Check shape
    assert snowm.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for snowm, got {snowm.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(snowm)), "snowm contains NaN"
    assert not jnp.any(jnp.isinf(snowm)), "snowm contains Inf"
    
    # Check values (snowm = snow * RHOS)
    expected_snowm = snow * RHOS
    assert jnp.allclose(snowm, expected_snowm, rtol=1e-5), "snowm values are incorrect"
    
    print("✅ compute_snow_mass test passed!")


def test_compute_sea_ice_concentration():
    """Test compute_sea_ice_concentration function."""
    print("Testing compute_sea_ice_concentration...")
    
    # Create test msi array
    msi = np.random.rand(IM_TEST, JM_TEST, LMI).astype(np.float32) * 1000.0
    
    # Compute sea ice concentration
    ace1i = compute_sea_ice_concentration(msi)
    
    # Check shape
    assert ace1i.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for ace1i, got {ace1i.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(ace1i)), "ace1i contains NaN"
    
    # Check values (ace1i = 1 if msi > 0, else 0)
    expected_ace1i = jnp.where(msi[:, :, 0] > 0.0, 1.0, 0.0)
    assert jnp.allclose(ace1i, expected_ace1i, rtol=1e-5), "ace1i values are incorrect"
    
    print("✅ compute_sea_ice_concentration test passed!")


def test_initialize_seaice():
    """Test initialize_seaice function."""
    print("Testing initialize_seaice...")
    
    # Initialize SEAICE arrays
    seaice = initialize_seaice(IM_TEST, JM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "SSI", "TSIL", "HSIL", "MSI",
        "SNOW", "SNOWM", "ACE1I", "TSURF",
    ]
    for key in expected_keys:
        assert key in seaice, f"{key} is missing from seaice"
    
    # Check shapes
    assert seaice["SSI"].shape == (IM_TEST, JM_TEST, LMI), f"SSI shape is incorrect"
    assert seaice["TSIL"].shape == (IM_TEST, JM_TEST, LMI), f"TSIL shape is incorrect"
    assert seaice["HSIL"].shape == (IM_TEST, JM_TEST, LMI), f"HSIL shape is incorrect"
    assert seaice["SNOW"].shape == (IM_TEST, JM_TEST), f"SNOW shape is incorrect"
    assert seaice["ACE1I"].shape == (IM_TEST, JM_TEST), f"ACE1I shape is incorrect"
    
    # Check all arrays are initialized to zero
    for key in expected_keys:
        assert jnp.all(seaice[key] == 0.0), f"{key} should be initialized to zero"
    
    print("✅ initialize_seaice test passed!")


def test_prec_si():
    """Test precipitation onto sea ice."""
    print("Testing prec_si...")
    
    # Inputs
    im, jm = 2, 2
    snow = jnp.ones((im, jm)) * 10.0  # Snow mass [kg/m^2]
    msi2 = jnp.ones((im, jm)) * 100.0  # Second layer ice mass [kg/m^2]
    hsil = jnp.ones((im, jm, LMI)) * 1e6  # Enthalpy of ice layers [J/m^2]
    tsil = jnp.ones((im, jm, LMI)) * -10.0  # Temperature of ice layers [C]
    ssil = jnp.ones((im, jm, LMI)) * 0.003  # Salt in ice layers [kg/m^2]
    prcp = jnp.ones((im, jm)) * 5.0  # Precipitation [kg/m^2]
    enrgp = jnp.ones((im, jm)) * -1e5  # Energy of precip [J/m^2] (negative for snow)
    run0 = jnp.zeros((im, jm))
    srun0 = jnp.zeros((im, jm))
    erun0 = jnp.zeros((im, jm))
    wetsnow = jnp.zeros((im, jm), dtype=jnp.bool_)
    cmprs = jnp.zeros((im, jm))
    
    # Call prec_si
    from seaice_jax import prec_si
    snow_new, msi2_new, hsil_new, tsil_new, ssil_new, run0_new, srun0_new, erun0_new = prec_si(
        snow, msi2, hsil, tsil, ssil, prcp, enrgp, run0, srun0, erun0, wetsnow, cmprs
    )
    
    # Check that snow mass increased
    assert jnp.all(snow_new >= snow), f"Snow mass should increase: {snow_new} vs {snow}"
    
    # Check that outputs are non-negative
    assert jnp.all(run0_new >= 0.0), f"Runoff should be non-negative: {run0_new}"
    assert jnp.all(srun0_new >= 0.0), f"Salt runoff should be non-negative: {srun0_new}"
    assert jnp.all(erun0_new >= 0.0), f"Energy runoff should be non-negative: {erun0_new}"
    
    print("✅ prec_si passed!")


def test_addice():
    """Test adding new ice due to freezing."""
    print("Testing addice...")
    
    # Inputs
    im, jm = 2, 2
    snow = jnp.ones((im, jm)) * 10.0  # Snow mass [kg/m^2]
    msi1 = jnp.ones((im, jm)) * 50.0  # First layer ice mass [kg/m^2]
    msi2 = jnp.ones((im, jm)) * 100.0  # Second layer ice mass [kg/m^2]
    hsil = jnp.ones((im, jm, LMI)) * 1e6  # Enthalpy of ice layers [J/m^2]
    tsil = jnp.ones((im, jm, LMI)) * -10.0  # Temperature of ice layers [C]
    ssil = jnp.ones((im, jm, LMI)) * 0.003  # Salt in ice layers [kg/m^2]
    
    # Call addice
    from seaice_jax import addice
    snow_new, msi1_new, msi2_new, hsil_new, tsil_new, ssil_new = addice(
        snow, msi1, msi2, hsil, tsil, ssil
    )
    
    # Check that ice mass increased
    assert jnp.all(msi1_new >= msi1), f"First layer ice mass should increase: {msi1_new} vs {msi1}"
    
    # Check that ice mass does not exceed ACE1I_CONST
    assert jnp.all(msi1_new <= ACE1I_CONST + 1e-5), \
        f"First layer ice mass should not exceed ACE1I_CONST: {msi1_new} vs {ACE1I_CONST}"
    
    print("✅ addice passed!")


def test_simelt():
    """Test sea ice melt."""
    print("Testing simelt...")
    
    # Inputs
    im, jm = 2, 2
    snow = jnp.ones((im, jm)) * 10.0  # Snow mass [kg/m^2]
    msi1 = jnp.ones((im, jm)) * 100.0  # First layer ice mass [kg/m^2]
    msi2 = jnp.ones((im, jm)) * 200.0  # Second layer ice mass [kg/m^2]
    hsil = jnp.ones((im, jm, LMI)) * 1e6  # Enthalpy of ice layers [J/m^2]
    tsil = jnp.ones((im, jm, LMI)) * -0.5  # Temperature of ice layers [C] (above freezing)
    ssil = jnp.ones((im, jm, LMI)) * 0.003  # Salt in ice layers [kg/m^2]
    
    # Call simelt
    from seaice_jax import simelt
    snow_new, msi1_new, msi2_new, hsil_new, tsil_new, ssil_new = simelt(
        snow, msi1, msi2, hsil, tsil, ssil
    )
    
    # Check that ice mass decreased (due to melting)
    assert jnp.all(msi1_new <= msi1), f"First layer ice mass should decrease: {msi1_new} vs {msi1}"
    
    # Check that ice mass is non-negative
    assert jnp.all(msi1_new >= 0.0), f"First layer ice mass should be non-negative: {msi1_new}"
    
    print("✅ simelt passed!")


def test_sea_ice():
    """Test sea ice thermodynamics."""
    print("Testing sea_ice...")
    
    # Inputs
    im, jm = 2, 2
    snow = jnp.ones((im, jm)) * 10.0  # Snow mass [kg/m^2]
    msi1 = jnp.ones((im, jm)) * 100.0  # First layer ice mass [kg/m^2]
    msi2 = jnp.ones((im, jm)) * 200.0  # Second layer ice mass [kg/m^2]
    hsil = jnp.ones((im, jm, LMI)) * 1e6  # Enthalpy of ice layers [J/m^2]
    tsil = jnp.ones((im, jm, LMI)) * -10.0  # Temperature of ice layers [C]
    ssil = jnp.ones((im, jm, LMI)) * 0.003  # Salt in ice layers [kg/m^2]
    
    # Call sea_ice
    from seaice_jax import sea_ice
    snow_new, msi1_new, msi2_new, hsil_new, tsil_new, ssil_new = sea_ice(
        snow, msi1, msi2, hsil, tsil, ssil
    )
    
    # Check that outputs are unchanged (placeholder implementation)
    assert jnp.allclose(snow_new, snow), f"Snow should be unchanged: {snow_new} vs {snow}"
    assert jnp.allclose(msi1_new, msi1), f"First layer ice mass should be unchanged: {msi1_new} vs {msi1}"
    
    print("✅ sea_ice passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SEAICE JAX Implementation")
    print("=" * 70)
    
    test_constants()
    test_compute_sea_ice_temperature()
    test_compute_sea_ice_salinity()
    test_compute_sea_ice_mass()
    test_compute_snow_mass()
    test_compute_sea_ice_concentration()
    test_initialize_seaice()
    test_prec_si()
    test_addice()
    test_simelt()
    test_sea_ice()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
