"""
Unit Tests for FLUXES JAX Implementation
==========================================

This script tests the JAX implementation of the surface flux calculations
from ROCKE-3D's FLUXES.f. It validates the following:
1. Basic functionality (shapes, NaN/Inf checks).
2. Numerical consistency with expected behavior.
3. Edge cases (e.g., zero wind, zero fluxes).

Usage:
    python test_fluxes_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from fluxes_jax import (
    compute_momentum_flux,
    compute_heat_flux,
    compute_moisture_flux,
    compute_surface_fluxes_jit,
    compute_radiative_flux,
    compute_energy_balance,
)


# Test constants
np.random.seed(42)

# Test data
I = 2  # Grid size (i)
J = 2  # Grid size (j)

# Surface wind components (m/s)
us = np.random.rand(I, J).astype(np.float32) * 10.0 - 5.0
vs = np.random.rand(I, J).astype(np.float32) * 10.0 - 5.0

# Surface temperature (K)
tsv = np.random.rand(I, J).astype(np.float32) * 50.0 + 273.0

# Surface specific humidity (kg/kg)
qsrf = np.random.rand(I, J).astype(np.float32) * 0.02

# Air density (kg/m^3)
rho = np.random.rand(I, J).astype(np.float32) * 2.0 + 0.5

# Drag coefficients
cdm = np.random.rand(I, J).astype(np.float32) * 0.1
cdh = np.random.rand(I, J).astype(np.float32) * 0.1
cq = np.random.rand(I, J).astype(np.float32) * 0.1

# Wind components at first layer (m/s)
u1 = np.random.rand(I, J).astype(np.float32) * 10.0 - 5.0
v1 = np.random.rand(I, J).astype(np.float32) * 10.0 - 5.0

# Temperature at first layer (K)
t1 = np.random.rand(I, J).astype(np.float32) * 50.0 + 273.0

# Specific humidity at first layer (kg/kg)
q1 = np.random.rand(I, J).astype(np.float32) * 0.02

# Radiation data
fsf = np.random.rand(I, J).astype(np.float32) * 1000.0  # Shortwave (W/m^2)
flong = np.random.rand(I, J).astype(np.float32) * 500.0  # Longwave (W/m^2)
albedo = np.random.rand(I, J).astype(np.float32) * 0.5 + 0.1  # Albedo


def test_momentum_flux():
    """Test momentum flux calculation."""
    print("Testing momentum flux...")
    
    uflux, vflux = compute_momentum_flux(us, vs, rho, cdm, u1, v1)
    
    # Check shapes
    assert uflux.shape == (I, J), f"Expected shape {(I, J)} for uflux, got {uflux.shape}"
    assert vflux.shape == (I, J), f"Expected shape {(I, J)} for vflux, got {vflux.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(uflux)), "uflux contains NaN"
    assert not jnp.any(jnp.isnan(vflux)), "vflux contains NaN"
    assert not jnp.any(jnp.isinf(uflux)), "uflux contains Inf"
    assert not jnp.any(jnp.isinf(vflux)), "vflux contains Inf"
    
    print("✅ Momentum flux test passed!")


def test_heat_flux():
    """Test sensible heat flux calculation."""
    print("Testing heat flux...")
    
    ws = jnp.sqrt(us**2 + vs**2)
    tflux = compute_heat_flux(tsv, t1, rho, cdh, ws)
    
    # Check shapes
    assert tflux.shape == (I, J), f"Expected shape {(I, J)} for tflux, got {tflux.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(tflux)), "tflux contains NaN"
    assert not jnp.any(jnp.isinf(tflux)), "tflux contains Inf"
    
    print("✅ Heat flux test passed!")


def test_moisture_flux():
    """Test latent heat flux calculation."""
    print("Testing moisture flux...")
    
    ws = jnp.sqrt(us**2 + vs**2)
    qflux = compute_moisture_flux(qsrf, q1, rho, cq, ws)
    
    # Check shapes
    assert qflux.shape == (I, J), f"Expected shape {(I, J)} for qflux, got {qflux.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(qflux)), "qflux contains NaN"
    assert not jnp.any(jnp.isinf(qflux)), "qflux contains Inf"
    
    print("✅ Moisture flux test passed!")


def test_surface_fluxes():
    """Test combined surface flux calculation."""
    print("Testing surface fluxes...")
    
    uflux, vflux, tflux, qflux = compute_surface_fluxes_jit(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1
    )
    
    # Check shapes
    assert uflux.shape == (I, J), f"Expected shape {(I, J)} for uflux, got {uflux.shape}"
    assert vflux.shape == (I, J), f"Expected shape {(I, J)} for vflux, got {vflux.shape}"
    assert tflux.shape == (I, J), f"Expected shape {(I, J)} for tflux, got {tflux.shape}"
    assert qflux.shape == (I, J), f"Expected shape {(I, J)} for qflux, got {qflux.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(uflux)), "uflux contains NaN"
    assert not jnp.any(jnp.isnan(vflux)), "vflux contains NaN"
    assert not jnp.any(jnp.isnan(tflux)), "tflux contains NaN"
    assert not jnp.any(jnp.isnan(qflux)), "qflux contains NaN"
    
    print("✅ Surface fluxes test passed!")


def test_radiative_flux():
    """Test radiative flux calculation."""
    print("Testing radiative flux...")
    
    solar, lw_net = compute_radiative_flux(tsv, fsf, flong, albedo)
    
    # Check shapes
    assert solar.shape == (I, J), f"Expected shape {(I, J)} for solar, got {solar.shape}"
    assert lw_net.shape == (I, J), f"Expected shape {(I, J)} for lw_net, got {lw_net.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(solar)), "solar contains NaN"
    assert not jnp.any(jnp.isnan(lw_net)), "lw_net contains NaN"
    
    # Check ranges
    assert jnp.all(solar >= 0.0), "solar contains negative values"
    
    print("✅ Radiative flux test passed!")


def test_energy_balance():
    """Test energy balance calculation."""
    print("Testing energy balance...")
    
    solar, lw_net = compute_radiative_flux(tsv, fsf, flong, albedo)
    _, _, tflux, qflux = compute_surface_fluxes_jit(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1
    )
    
    net_energy = compute_energy_balance(solar, lw_net, tflux, qflux)
    
    # Check shapes
    assert net_energy.shape == (I, J), f"Expected shape {(I, J)} for net_energy, got {net_energy.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(net_energy)), "net_energy contains NaN"
    assert not jnp.any(jnp.isinf(net_energy)), "net_energy contains Inf"
    
    print("✅ Energy balance test passed!")


def test_edge_cases():
    """Test edge cases (e.g., zero wind, zero fluxes)."""
    print("Testing edge cases...")
    
    # Zero wind
    us_zero = jnp.zeros((I, J))
    vs_zero = jnp.zeros((I, J))
    uflux, vflux = compute_momentum_flux(us_zero, vs_zero, rho, cdm, u1, v1)
    assert jnp.all(uflux == 0.0), "uflux should be zero for zero wind"
    assert jnp.all(vflux == 0.0), "vflux should be zero for zero wind"
    
    # Zero temperature difference
    tsv_zero_diff = t1.copy()
    tflux = compute_heat_flux(tsv_zero_diff, t1, rho, cdh, jnp.sqrt(us**2 + vs**2))
    assert jnp.all(tflux == 0.0), "tflux should be zero for zero temperature difference"
    
    # Zero moisture difference
    qsrf_zero_diff = q1.copy()
    ws = jnp.sqrt(us**2 + vs**2)
    qflux = compute_moisture_flux(qsrf_zero_diff, q1, rho, cq, ws)
    assert jnp.all(qflux == 0.0), "qflux should be zero for zero moisture difference"
    
    print("✅ Edge cases test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FLUXES JAX Implementation")
    print("=" * 70)
    
    test_momentum_flux()
    test_heat_flux()
    test_moisture_flux()
    test_surface_fluxes()
    test_radiative_flux()
    test_energy_balance()
    test_edge_cases()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
