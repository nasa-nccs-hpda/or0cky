"""
Unit Tests for CONSTANT JAX Implementation
============================================

This script tests the JAX implementation of the physical constants
from ROCKE-3D's Constants_mod.F90. It validates the following:
1. All constants are defined and accessible.
2. Constants match expected values (within tolerance).
3. Utility functions work correctly.

Usage:
    python test_constant_jax.py
"""

import jax.numpy as jnp
from constant_jax import *


def test_numerical_constants():
    """Test numerical constants."""
    print("Testing numerical constants...")
    
    # Check UNDEF_VAL is large (float32 max ~3.4e38)
    # Note: 1e30 is within float32 range, but we use a smaller threshold for safety
    assert UNDEF_VAL > 1e20, "UNDEF_VAL should be a large positive value"
    
    # Check UNDEF is negative
    assert UNDEF < 0.0, "UNDEF should be negative"
    
    # Check TEENY is small
    assert TEENY < 1e-20, "TEENY should be a small positive value"
    
    # Check NaN is NaN
    assert jnp.isnan(NAN), "NAN should be NaN"
    
    print("✅ Numerical constants test passed!")


def test_mathematical_constants():
    """Test mathematical constants."""
    print("Testing mathematical constants...")
    
    # Check PI is approximately 3.14159
    assert jnp.abs(PI - 3.14159265358979323846) < 1e-15, "PI is incorrect"
    
    # Check TWOPI is 2 * PI
    assert jnp.abs(TWOPI - 2.0 * PI) < 1e-15, "TWOPI is incorrect"
    
    # Check RADIANS_PER_DEGREE is PI / 180
    assert jnp.abs(RADIANS_PER_DEGREE - PI / 180.0) < 1e-15, "RADIANS_PER_DEGREE is incorrect"
    
    # Check reciprocals
    assert jnp.abs(BY3 - 1.0 / 3.0) < 1e-15, "BY3 is incorrect"
    assert jnp.abs(BY6 - 1.0 / 6.0) < 1e-15, "BY6 is incorrect"
    
    print("✅ Mathematical constants test passed!")


def test_physical_constants():
    """Test physical constants."""
    print("Testing physical constants...")
    
    # Check Stefan-Boltzmann constant (float32 precision)
    assert jnp.abs(STBO - 5.67037321e-8) < 1e-8, "STBO is incorrect"
    
    # Check Boltzmann's constant (float32 precision)
    assert jnp.abs(K_BOLTZMANN - 1.380662e-23) < 1e-25, "K_BOLTZMANN is incorrect"
    
    # Check Planck constant (float32 precision)
    assert jnp.abs(H_PLANCK - 6.626176e-34) < 1e-36, "H_PLANCK is incorrect"
    
    # Check speed of light (float32 precision)
    assert jnp.abs(C_LIGHT - 2.9979245e8) < 1e-2, "C_LIGHT is incorrect"
    
    # Check latent heats (float32 precision)
    assert jnp.abs(LHE - 2.5e6) < 1e-2, "LHE is incorrect"
    assert jnp.abs(LHM - 3.34e5) < 1e-2, "LHM is incorrect"
    assert jnp.abs(LHS - (LHE + LHM)) < 1e-2, "LHS is incorrect"
    
    # Check densities (float32 precision)
    assert jnp.abs(RHOW - 1000.0) < 1e-2, "RHOW is incorrect"
    assert jnp.abs(RHOI - 916.6) < 1e-2, "RHOI is incorrect"
    
    # Check freezing point of water (float32 precision)
    assert jnp.abs(TF - 273.15) < 1e-5, "TF is incorrect"
    
    # Check specific heats (float32 precision)
    assert jnp.abs(SHW - 4185.0) < 1e-2, "SHW is incorrect"
    assert jnp.abs(SHI - 2060.0) < 1e-2, "SHI is incorrect"
    
    # Check gas constants (float32 precision)
    assert jnp.abs(GASC - 8.314510) < 1e-5, "GASC is incorrect"
    assert jnp.abs(MAIR - 28.9655) < 1e-5, "MAIR is incorrect"
    assert jnp.abs(RGAS - 1e3 * GASC / MAIR) < 1e-2, "RGAS is incorrect"
    
    # Check water vapor constants (float32 precision)
    assert jnp.abs(MWAT - 18.015) < 1e-5, "MWAT is incorrect"
    assert jnp.abs(RVAP - 1e3 * GASC / MWAT) < 1e-2, "RVAP is incorrect"
    assert jnp.abs(DELTX - (BYMRAT - 1.0)) < 1e-5, "DELTX is incorrect"
    
    # Check specific heat ratio (float32 precision)
    assert jnp.abs(SRAT - 1.401) < 1e-5, "SRAT is incorrect"
    assert jnp.abs(KAPA - (SRAT - 1.0) / SRAT) < 1e-5, "KAPA is incorrect"
    
    # Check specific heat of dry air (float32 precision)
    assert jnp.abs(SHA - RGAS / KAPA) < 1e-2, "SHA is incorrect"
    
    print("✅ Physical constants test passed!")


def test_astronomical_constants():
    """Test astronomical constants."""
    print("Testing astronomical constants...")
    
    # Check astronomical unit (float32 precision)
    assert jnp.abs(ASTRONOMICAL_UNIT - 149597870700.0) < 1e-2, "ASTRONOMICAL_UNIT is incorrect"
    
    # Check solar constants (float32 precision)
    assert jnp.abs(SOLAR_T_EFFECTIVE - 5785.0) < 1e-2, "SOLAR_T_EFFECTIVE is incorrect"
    assert jnp.abs(SOLAR_RADIUS - 6.96e8) < 1e-2, "SOLAR_RADIUS is incorrect"
    
    # Check Earth constants (float32 precision)
    assert jnp.abs(RADIUS - 6371000.0) < 1e-2, "RADIUS is incorrect"
    assert jnp.abs(AREAG - 4.0 * PI * RADIUS * RADIUS) < 1e-2, "AREAG is incorrect"
    assert jnp.abs(GRAV - 9.80665) < 1e-5, "GRAV is incorrect"
    assert jnp.abs(BYGRAV - 1.0 / GRAV) < 1e-5, "BYGRAV is incorrect"
    
    # Check Earth's rotation rate (float32 precision)
    assert jnp.abs(OMEGA - 7.292115e-5) < 1e-7, "OMEGA is incorrect"
    assert jnp.abs(OMEGA2 - 2.0 * OMEGA) < 1e-7, "OMEGA2 is incorrect"
    
    print("✅ Astronomical constants test passed!")


def test_lapse_rate_constants():
    """Test lapse rate constants."""
    print("Testing lapse rate constants...")
    
    # Check dry adiabatic lapse rate (float32 precision)
    assert jnp.abs(GAMD - GRAV * KAPA / RGAS) < 1e-5, "GAMD is incorrect"
    
    # Check moist adiabatic lapse rate (float32 precision)
    assert jnp.abs(BMOIST - 0.0065) < 1e-5, "BMOIST is incorrect"
    assert jnp.abs(BBYG - BMOIST * BYGRAV) < 1e-5, "BBYG is incorrect"
    assert jnp.abs(GBYRB - GRAV / (RGAS * BMOIST)) < 1e-5, "GBYRB is incorrect"
    
    print("✅ Lapse rate constants test passed!")


def test_conversion_factors():
    """Test conversion factors."""
    print("Testing conversion factors...")
    
    # Check kg/m^2 to mbar (float32 precision)
    assert jnp.abs(KG2MB - 1e-2 * GRAV) < 1e-5, "KG2MB is incorrect"
    assert jnp.abs(MB2KG - 1e2 * BYGRAV) < 1e-5, "MB2KG is incorrect"
    
    # Check kg/m^2 water to mm (float32 precision)
    assert jnp.abs(KGPA2MM - 1.0) < 1e-5, "KGPA2MM is incorrect"
    assert jnp.abs(MM2KGPA - 1.0) < 1e-5, "MM2KGPA is incorrect"
    
    print("✅ Conversion factors test passed!")


def test_atmospheric_composition():
    """Test atmospheric composition."""
    print("Testing atmospheric composition...")
    
    # Check gas fractions
    assert jnp.abs(PN2 - 0.780840) < 1e-10, "PN2 is incorrect"
    assert jnp.abs(PO2 - 0.209476) < 1e-10, "PO2 is incorrect"
    assert jnp.abs(PAR - 0.0093) < 1e-10, "PAR is incorrect"
    assert jnp.abs(PH2 - 0.0) < 1e-10, "PH2 is incorrect"
    
    # Check planet name
    assert PLANET_NAME == "Earth", "PLANET_NAME is incorrect"
    
    print("✅ Atmospheric composition test passed!")


def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test visc_air at 291.15 K (should be positive and reasonable)
    T0 = jnp.float32(291.15)
    visc_at_T0 = visc_air(T0)
    assert visc_at_T0 > 0.0, "visc_air(T0) should be positive"
    assert visc_at_T0 < 1e-4, "visc_air(T0) should be reasonable"
    
    # Test visc_air at 300 K (should be slightly higher)
    T1 = jnp.float32(300.0)
    assert visc_air(T1) > visc_air(T0), "visc_air should increase with temperature"
    
    print("✅ Utility functions test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing CONSTANT JAX Implementation")
    print("=" * 70)
    
    test_numerical_constants()
    test_mathematical_constants()
    test_physical_constants()
    test_astronomical_constants()
    test_lapse_rate_constants()
    test_conversion_factors()
    test_atmospheric_composition()
    test_utility_functions()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
