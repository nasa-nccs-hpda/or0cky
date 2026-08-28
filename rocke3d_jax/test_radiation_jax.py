"""
Unit Tests for RADIATION JAX Implementation
=============================================

This script tests the JAX implementation of the radiative transfer calculations
from ROCKE-3D's RADIATION.f. It validates the following:
1. Basic functionality (shapes, NaN/Inf checks).
2. Numerical consistency with expected behavior.
3. Edge cases (e.g., zero temperature, zero fluxes).

Usage:
    python test_radiation_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from radiation_jax import (
    planck_function_jit,
    stefan_boltzmann_jit,
    compute_solar_flux_jit,
    compute_lw_flux_jit,
    compute_radiative_fluxes_jit,
    compute_atmospheric_absorption_jit,
    compute_net_radiation_jit,
)


# Test constants
np.random.seed(42)

# Grid size
I = 2
J = 2

# Temperature (K)
temperature = np.random.rand(I, J).astype(np.float32) * 100.0 + 200.0

# Wavelength (m) for Planck function
wavelength = np.random.rand(I, J).astype(np.float32) * 1e-5 + 1e-7

# Solar zenith angle (cosine)
cosz = np.random.rand(I, J).astype(np.float32)

# Albedo
albedo = np.random.rand(I, J).astype(np.float32) * 0.5 + 0.1

# Emissivity
emissivity = np.random.rand(I, J).astype(np.float32) * 0.2 + 0.8

# Downwelling longwave radiation (W/m^2)
flong = np.random.rand(I, J).astype(np.float32) * 500.0

# Pressure (Pa)
pressure = np.random.rand(I, J).astype(np.float32) * 50000.0 + 50000.0

# Water vapor mixing ratio (kg/kg)
water_vapor = np.random.rand(I, J).astype(np.float32) * 0.02

# CO2 concentration (ppm)
co2 = np.array([[400.0, 420.0], [380.0, 410.0]], dtype=np.float32)


def test_planck_function():
    """Test Planck function calculation."""
    print("Testing Planck function...")
    
    B = planck_function_jit(wavelength, temperature)
    
    # Check shapes
    assert B.shape == (I, J), f"Expected shape {(I, J)} for B, got {B.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(B)), "B contains NaN"
    assert not jnp.any(jnp.isinf(B)), "B contains Inf"
    
    # Check ranges (Planck function should be positive)
    assert jnp.all(B > 0.0), "B contains non-positive values"
    
    print("✅ Planck function test passed!")


def test_stefan_boltzmann():
    """Test Stefan-Boltzmann law."""
    print("Testing Stefan-Boltzmann law...")
    
    lw_flux = stefan_boltzmann_jit(temperature, emissivity)
    
    # Check shapes
    assert lw_flux.shape == (I, J), f"Expected shape {(I, J)} for lw_flux, got {lw_flux.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(lw_flux)), "lw_flux contains NaN"
    assert not jnp.any(jnp.isinf(lw_flux)), "lw_flux contains Inf"
    
    # Check ranges (LW flux should be positive)
    assert jnp.all(lw_flux > 0.0), "lw_flux contains non-positive values"
    
    print("✅ Stefan-Boltzmann test passed!")


def test_solar_flux():
    """Test solar flux calculation."""
    print("Testing solar flux...")
    
    fsf, srdflb, srnflb = compute_solar_flux_jit(1365.0, cosz, albedo)
    
    # Check shapes
    assert fsf.shape == (I, J), f"Expected shape {(I, J)} for fsf, got {fsf.shape}"
    assert srdflb.shape == (I, J), f"Expected shape {(I, J)} for srdflb, got {srdflb.shape}"
    assert srnflb.shape == (I, J), f"Expected shape {(I, J)} for srnflb, got {srnflb.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(fsf)), "fsf contains NaN"
    assert not jnp.any(jnp.isnan(srdflb)), "srdflb contains NaN"
    assert not jnp.any(jnp.isnan(srnflb)), "srnflb contains NaN"
    
    # Check ranges (fluxes should be positive)
    assert jnp.all(fsf >= 0.0), "fsf contains negative values"
    assert jnp.all(srdflb >= 0.0), "srdflb contains negative values"
    
    print("✅ Solar flux test passed!")


def test_lw_flux():
    """Test longwave flux calculation."""
    print("Testing longwave flux...")
    
    trdflb, truflb, trnflb = compute_lw_flux_jit(temperature, emissivity, flong)
    
    # Check shapes
    assert trdflb.shape == (I, J), f"Expected shape {(I, J)} for trdflb, got {trdflb.shape}"
    assert truflb.shape == (I, J), f"Expected shape {(I, J)} for truflb, got {truflb.shape}"
    assert trnflb.shape == (I, J), f"Expected shape {(I, J)} for trnflb, got {trnflb.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(trdflb)), "trdflb contains NaN"
    assert not jnp.any(jnp.isnan(truflb)), "truflb contains NaN"
    assert not jnp.any(jnp.isnan(trnflb)), "trnflb contains NaN"
    
    # Check ranges (upwelling LW should be positive)
    assert jnp.all(truflb > 0.0), "truflb contains non-positive values"
    
    print("✅ Longwave flux test passed!")


def test_radiative_fluxes():
    """Test combined radiative flux calculation."""
    print("Testing combined radiative fluxes...")
    
    fsf, srdflb, srnflb, trdflb, truflb, trnflb, net_sw, net_lw = compute_radiative_fluxes_jit(
        temperature, 1365.0, cosz, albedo, emissivity, flong
    )
    
    # Check shapes
    assert fsf.shape == (I, J), f"Expected shape {(I, J)} for fsf, got {fsf.shape}"
    assert srdflb.shape == (I, J), f"Expected shape {(I, J)} for srdflb, got {srdflb.shape}"
    assert srnflb.shape == (I, J), f"Expected shape {(I, J)} for srnflb, got {srnflb.shape}"
    assert trdflb.shape == (I, J), f"Expected shape {(I, J)} for trdflb, got {trdflb.shape}"
    assert truflb.shape == (I, J), f"Expected shape {(I, J)} for truflb, got {truflb.shape}"
    assert trnflb.shape == (I, J), f"Expected shape {(I, J)} for trnflb, got {trnflb.shape}"
    assert net_sw.shape == (I, J), f"Expected shape {(I, J)} for net_sw, got {net_sw.shape}"
    assert net_lw.shape == (I, J), f"Expected shape {(I, J)} for net_lw, got {net_lw.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(fsf)), "fsf contains NaN"
    assert not jnp.any(jnp.isnan(srdflb)), "srdflb contains NaN"
    assert not jnp.any(jnp.isnan(srnflb)), "srnflb contains NaN"
    assert not jnp.any(jnp.isnan(trdflb)), "trdflb contains NaN"
    assert not jnp.any(jnp.isnan(truflb)), "truflb contains NaN"
    assert not jnp.any(jnp.isnan(trnflb)), "trnflb contains NaN"
    assert not jnp.any(jnp.isnan(net_sw)), "net_sw contains NaN"
    assert not jnp.any(jnp.isnan(net_lw)), "net_lw contains NaN"
    
    print("✅ Combined radiative fluxes test passed!")


def test_atmospheric_absorption():
    """Test atmospheric absorption calculation."""
    print("Testing atmospheric absorption...")
    
    lw_absorbed, lw_transmitted = compute_atmospheric_absorption_jit(
        temperature, pressure, water_vapor, co2
    )
    
    # Check shapes
    assert lw_absorbed.shape == (I, J), f"Expected shape {(I, J)} for lw_absorbed, got {lw_absorbed.shape}"
    assert lw_transmitted.shape == (I, J), f"Expected shape {(I, J)} for lw_transmitted, got {lw_transmitted.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(lw_absorbed)), "lw_absorbed contains NaN"
    assert not jnp.any(jnp.isnan(lw_transmitted)), "lw_transmitted contains NaN"
    
    # Check ranges (absorbed and transmitted should be positive)
    assert jnp.all(lw_absorbed >= 0.0), "lw_absorbed contains negative values"
    assert jnp.all(lw_transmitted >= 0.0), "lw_transmitted contains negative values"
    
    print("✅ Atmospheric absorption test passed!")


def test_net_radiation():
    """Test net radiation calculation."""
    print("Testing net radiation...")
    
    net_radiation = compute_net_radiation_jit(
        temperature, 1365.0, cosz, albedo, emissivity, flong
    )
    
    # Check shapes
    assert net_radiation.shape == (I, J), f"Expected shape {(I, J)} for net_radiation, got {net_radiation.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(net_radiation)), "net_radiation contains NaN"
    assert not jnp.any(jnp.isinf(net_radiation)), "net_radiation contains Inf"
    
    print("✅ Net radiation test passed!")


def test_edge_cases():
    """Test edge cases (e.g., zero temperature, zero fluxes)."""
    print("Testing edge cases...")
    
    # Zero temperature
    temperature_zero = jnp.zeros((I, J))
    lw_flux = stefan_boltzmann_jit(temperature_zero, emissivity)
    assert jnp.all(lw_flux == 0.0), "LW flux should be zero for zero temperature"
    
    # Zero cosine of zenith angle (sun at horizon)
    cosz_zero = jnp.zeros((I, J))
    fsf, srdflb, srnflb = compute_solar_flux_jit(1365.0, cosz_zero, albedo)
    assert jnp.all(fsf == 0.0), "Solar flux should be zero for cosz=0"
    
    # Zero albedo (black surface)
    albedo_zero = jnp.zeros((I, J))
    fsf, srdflb, srnflb = compute_solar_flux_jit(1365.0, cosz, albedo_zero)
    assert jnp.all(srdflb == fsf), "Absorbed SW should equal downwelling SW for albedo=0"
    
    print("✅ Edge cases test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing RADIATION JAX Implementation")
    print("=" * 70)
    
    test_planck_function()
    test_stefan_boltzmann()
    test_solar_flux()
    test_lw_flux()
    test_radiative_fluxes()
    test_atmospheric_absorption()
    test_net_radiation()
    test_edge_cases()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
