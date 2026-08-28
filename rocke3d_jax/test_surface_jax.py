"""
Unit Tests for SURFACE JAX Implementation
============================================

This script tests the JAX implementation of the surface flux calculations
from ROCKE-3D's SURFACE.f. It validates the following:
1. Basic functionality (shapes, NaN/Inf checks).
2. Numerical consistency with expected behavior.
3. Edge cases (e.g., zero wind, zero fluxes).
4. Different surface types (ocean, sea ice, land).

Usage:
    python test_surface_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from surface_jax import (
    compute_surface_properties,
    compute_ocean_fluxes,
    compute_land_fluxes,
    compute_seaice_fluxes,
    compute_surface_fluxes_jit,
)


# Test constants
np.random.seed(42)

# Grid size
I = 2
J = 2

# Surface type (1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land)
itype = np.array([[1, 2], [3, 4]], dtype=np.int32)

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

# Ocean velocities (m/s)
uocean = np.random.rand(I, J).astype(np.float32) * 2.0 - 1.0
vocean = np.random.rand(I, J).astype(np.float32) * 2.0 - 1.0

# Surface pressure (Pa)
ps = np.random.rand(I, J).astype(np.float32) * 20000.0 + 90000.0

# Surface pressure scaling factor
psk = np.random.rand(I, J).astype(np.float32) * 1.0 + 0.5


def test_surface_properties():
    """Test surface properties calculation."""
    print("Testing surface properties...")
    
    thv1, rho_calc, qsat = compute_surface_properties(t1, q1, ps, psk)
    
    # Check shapes
    assert thv1.shape == (I, J), f"Expected shape {(I, J)} for thv1, got {thv1.shape}"
    assert rho_calc.shape == (I, J), f"Expected shape {(I, J)} for rho, got {rho_calc.shape}"
    assert qsat.shape == (I, J), f"Expected shape {(I, J)} for qsat, got {qsat.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(thv1)), "thv1 contains NaN"
    assert not jnp.any(jnp.isnan(rho_calc)), "rho contains NaN"
    assert not jnp.any(jnp.isnan(qsat)), "qsat contains NaN"
    
    print("✅ Surface properties test passed!")


def test_ocean_fluxes():
    """Test ocean surface flux calculation."""
    print("Testing ocean fluxes...")
    
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_ocean_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    
    # Check shapes
    assert uflux.shape == (I, J), f"Expected shape {(I, J)} for uflux, got {uflux.shape}"
    assert vflux.shape == (I, J), f"Expected shape {(I, J)} for vflux, got {vflux.shape}"
    assert tflux.shape == (I, J), f"Expected shape {(I, J)} for tflux, got {tflux.shape}"
    assert qflux.shape == (I, J), f"Expected shape {(I, J)} for qflux, got {qflux.shape}"
    assert solar.shape == (I, J), f"Expected shape {(I, J)} for solar, got {solar.shape}"
    assert lw_net.shape == (I, J), f"Expected shape {(I, J)} for lw_net, got {lw_net.shape}"
    assert net_energy.shape == (I, J), f"Expected shape {(I, J)} for net_energy, got {net_energy.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(uflux)), "uflux contains NaN"
    assert not jnp.any(jnp.isnan(vflux)), "vflux contains NaN"
    assert not jnp.any(jnp.isnan(tflux)), "tflux contains NaN"
    assert not jnp.any(jnp.isnan(qflux)), "qflux contains NaN"
    
    print("✅ Ocean fluxes test passed!")


def test_land_fluxes():
    """Test land surface flux calculation."""
    print("Testing land fluxes...")
    
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_land_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo
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
    
    print("✅ Land fluxes test passed!")


def test_seaice_fluxes():
    """Test sea ice surface flux calculation."""
    print("Testing sea ice fluxes...")
    
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_seaice_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
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
    
    print("✅ Sea ice fluxes test passed!")


def test_surface_fluxes():
    """Test combined surface flux calculation for all types."""
    print("Testing combined surface fluxes...")
    
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_surface_fluxes_jit(
        itype, us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    
    # Check shapes
    assert uflux.shape == (I, J), f"Expected shape {(I, J)} for uflux, got {uflux.shape}"
    assert vflux.shape == (I, J), f"Expected shape {(I, J)} for vflux, got {vflux.shape}"
    assert tflux.shape == (I, J), f"Expected shape {(I, J)} for tflux, got {tflux.shape}"
    assert qflux.shape == (I, J), f"Expected shape {(I, J)} for qflux, got {qflux.shape}"
    assert solar.shape == (I, J), f"Expected shape {(I, J)} for solar, got {solar.shape}"
    assert lw_net.shape == (I, J), f"Expected shape {(I, J)} for lw_net, got {lw_net.shape}"
    assert net_energy.shape == (I, J), f"Expected shape {(I, J)} for net_energy, got {net_energy.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(uflux)), "uflux contains NaN"
    assert not jnp.any(jnp.isnan(vflux)), "vflux contains NaN"
    assert not jnp.any(jnp.isnan(tflux)), "tflux contains NaN"
    assert not jnp.any(jnp.isnan(qflux)), "qflux contains NaN"
    assert not jnp.any(jnp.isnan(solar)), "solar contains NaN"
    assert not jnp.any(jnp.isnan(lw_net)), "lw_net contains NaN"
    assert not jnp.any(jnp.isnan(net_energy)), "net_energy contains NaN"
    
    print("✅ Combined surface fluxes test passed!")


def test_edge_cases():
    """Test edge cases (e.g., zero wind, zero fluxes)."""
    print("Testing edge cases...")
    
    # Zero wind
    us_zero = jnp.zeros((I, J))
    vs_zero = jnp.zeros((I, J))
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_ocean_fluxes(
        us_zero, vs_zero, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    assert jnp.all(uflux == 0.0), "uflux should be zero for zero wind"
    assert jnp.all(vflux == 0.0), "vflux should be zero for zero wind"
    
    # Zero temperature difference
    tsv_zero_diff = t1.copy()
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_ocean_fluxes(
        us, vs, tsv_zero_diff, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    assert jnp.all(tflux == 0.0), "tflux should be zero for zero temperature difference"
    
    # Zero moisture difference
    qsrf_zero_diff = q1.copy()
    uflux, vflux, tflux, qflux, solar, lw_net, net_energy = compute_ocean_fluxes(
        us, vs, tsv, qsrf_zero_diff, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    assert jnp.all(qflux == 0.0), "qflux should be zero for zero moisture difference"
    
    print("✅ Edge cases test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SURFACE JAX Implementation")
    print("=" * 70)
    
    test_surface_properties()
    test_ocean_fluxes()
    test_land_fluxes()
    test_seaice_fluxes()
    test_surface_fluxes()
    test_edge_cases()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
