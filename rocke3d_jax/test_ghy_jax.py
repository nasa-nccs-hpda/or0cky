"""
Unit Tests for GHY JAX Implementation
======================================

This script tests the JAX implementation of the GHY (Global Land Model) module
from ROCKE-3D. It validates the correctness and performance of the ported subroutines.

Usage:
    python test_ghy_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from ghy_jax import (
    compute_sensible_heat_jit,
    compute_evap_limits_jit,
    compute_runoff_jit,
    get_soil_properties_jit,
    compute_snow_melt_jit,
    compute_soil_moisture_table,
    SATURATED_MOISTURE,
)


def test_compute_sensible_heat():
    """Test sensible heat flux calculation."""
    print("Testing compute_sensible_heat_jit...")
    
    # Inputs
    tg = jnp.array([300.0, 290.0, 280.0])  # Ground temperature (K)
    t1 = jnp.array([295.0, 285.0, 275.0])  # Air temperature (K)
    rho = jnp.array([1.2, 1.1, 1.0])        # Air density (kg/m^3)
    ch = jnp.array([0.001, 0.0015, 0.002])  # Stanton number
    ws = jnp.array([5.0, 10.0, 15.0])       # Wind speed (m/s)
    
    # Compute sensible heat
    sensible_heat = compute_sensible_heat_jit(tg, t1, rho, ch, ws)
    
    # Expected: rho * cp * ch * ws * (tg - t1)
    cp = 1004.6
    expected = rho * cp * ch * ws * (tg - t1)
    
    # Check results
    assert jnp.allclose(sensible_heat, expected, rtol=1e-5), \
        f"Sensible heat mismatch: {sensible_heat} vs {expected}"
    
    print("✅ compute_sensible_heat_jit passed!")


def test_compute_evap_limits():
    """Test evaporation limits calculation."""
    print("Testing compute_evap_limits_jit...")
    
    # Inputs
    wc = jnp.array([0.3, 0.4, 0.2])        # Current soil moisture (m^3/m^3)
    wsat = jnp.array([0.5, 0.6, 0.4])       # Saturated soil moisture (m^3/m^3)
    wfc = jnp.array([0.4, 0.5, 0.3])        # Field capacity (m^3/m^3)
    wpwp = jnp.array([0.1, 0.15, 0.1])      # Permanent wilting point (m^3/m^3)
    
    # Compute evaporation limits
    evap_pot, evap_act, evap_lim = compute_evap_limits_jit(wc, wsat, wfc, wpwp)
    
    # Check that actual evaporation is <= potential evaporation
    assert jnp.all(evap_act <= evap_pot + 1e-10), \
        f"Actual evaporation exceeds potential: {evap_act} vs {evap_pot}"
    
    # Check that evaporation is zero when wc <= wpwp
    assert jnp.all(jnp.where(wc <= wpwp, evap_act == 0.0, True)), \
        f"Evaporation should be zero when wc <= wpwp"
    
    print("✅ compute_evap_limits_jit passed!")


def test_compute_runoff():
    """Test runoff calculation."""
    print("Testing compute_runoff_jit...")
    
    # Inputs
    precip = jnp.array([0.01, 0.02, 0.005])  # Precipitation (m/s)
    wc = jnp.array([0.3, 0.4, 0.2])          # Current soil moisture (m^3/m^3)
    wsat = jnp.array([0.5, 0.6, 0.4])         # Saturated soil moisture (m^3/m^3)
    ksat = jnp.array([1e-4, 1e-5, 1e-6])     # Saturated hydraulic conductivity (m/s)
    slope = jnp.array([0.1, 0.05, 0.01])     # Surface slope
    
    # Compute runoff
    runoff, infiltration = compute_runoff_jit(precip, wc, wsat, ksat, slope)
    
    # Check that runoff + infiltration <= precip
    assert jnp.all(runoff + infiltration <= precip + 1e-10), \
        f"Runoff + infiltration exceeds precipitation: {runoff + infiltration} vs {precip}"
    
    # Check that runoff is non-negative
    assert jnp.all(runoff >= 0.0), \
        f"Runoff should be non-negative: {runoff}"
    
    print("✅ compute_runoff_jit passed!")


def test_get_soil_properties():
    """Test soil properties calculation."""
    print("Testing get_soil_properties_jit...")
    
    # Inputs
    q_in = jnp.array([0.2, 0.3, 0.4])       # Soil moisture (m^3/m^3)
    dz_in = jnp.array([0.1, 0.2, 0.3])       # Soil layer thickness (m)
    texture = jnp.array([0, 1, 2])          # Soil texture (0: sand, 1: silt, 2: clay)
    
    # Compute soil properties
    lambda_soil, c_soil, k_soil, d_soil = get_soil_properties_jit(q_in, dz_in, texture)
    
    # Check that thermal conductivity is positive
    assert jnp.all(lambda_soil > 0.0), \
        f"Thermal conductivity should be positive: {lambda_soil}"
    
    # Check that heat capacity is positive
    assert jnp.all(c_soil > 0.0), \
        f"Heat capacity should be positive: {c_soil}"
    
    # Check that hydraulic conductivity is positive
    assert jnp.all(k_soil > 0.0), \
        f"Hydraulic conductivity should be positive: {k_soil}"
    
    print("✅ get_soil_properties_jit passed!")


def test_compute_snow_melt():
    """Test snow melt calculation."""
    print("Testing compute_snow_melt_jit...")
    
    # Inputs
    ts = jnp.array([273.15, 270.0, 268.0])  # Surface temperature (K)
    wsn = jnp.array([0.1, 0.2, 0.05])       # Snow water equivalent (m)
    fsf = jnp.array([200.0, 150.0, 100.0])   # Downwelling shortwave radiation (W/m^2)
    flong = jnp.array([300.0, 280.0, 260.0]) # Downwelling longwave radiation (W/m^2)
    albedo = jnp.array([0.8, 0.7, 0.6])      # Surface albedo
    
    # Compute snow melt
    snow_melt, ts_new, wsn_new = compute_snow_melt_jit(ts, wsn, fsf, flong, albedo)
    
    # Check that snow melt is non-negative
    assert jnp.all(snow_melt >= 0.0), \
        f"Snow melt should be non-negative: {snow_melt}"
    
    # Check that updated snow water equivalent is non-negative
    assert jnp.all(wsn_new >= 0.0), \
        f"Snow water equivalent should be non-negative: {wsn_new}"
    
    # Check that snow melt does not exceed available snow
    assert jnp.all(snow_melt * 3600.0 <= wsn + 1e-10), \
        f"Snow melt exceeds available snow: {snow_melt * 3600.0} vs {wsn}"
    
    print("✅ compute_snow_melt_jit passed!")


def test_compute_soil_moisture_table():
    """Test soil moisture table computation."""
    print("Testing compute_soil_moisture_table...")
    
    # Compute tables (note: this function is not JIT-compiled)
    thm, xklm, dlm = compute_soil_moisture_table(hmin=-100.0, delh1=-0.01)
    
    # Check shapes (nth_total = 64, so 65 rows)
    assert thm.shape[0] == 65, f"thm should have 65 rows: {thm.shape}"
    assert thm.shape[1] == 4, f"thm should have 4 columns: {thm.shape}"
    assert xklm.shape == thm.shape, f"xklm shape mismatch: {xklm.shape}"
    assert dlm.shape == thm.shape, f"dlm shape mismatch: {dlm.shape}"
    
    # Check that theta is between 0 and saturated moisture
    assert jnp.all(thm >= 0.0), f"Theta should be non-negative: {thm}"
    assert jnp.all(thm <= SATURATED_MOISTURE.max()), \
        f"Theta should not exceed saturated moisture: {thm}"
    
    # Check that conductivity and diffusivity are positive
    assert jnp.all(xklm > 0.0), f"Conductivity should be positive: {xklm}"
    assert jnp.all(dlm > 0.0), f"Diffusivity should be positive: {dlm}"
    
    print("✅ compute_soil_moisture_table passed!")


def test_performance():
    """Test performance of JIT-compiled functions."""
    print("\nTesting performance...")
    
    # Large inputs for performance testing
    size = 10000
    tg = jnp.ones(size) * 300.0
    t1 = jnp.ones(size) * 295.0
    rho = jnp.ones(size) * 1.2
    ch = jnp.ones(size) * 0.001
    ws = jnp.ones(size) * 5.0
    
    # Warm up JIT
    _ = compute_sensible_heat_jit(tg, t1, rho, ch, ws).block_until_ready()
    
    # Time the function
    import time
    start = time.time()
    for _ in range(100):
        sensible_heat = compute_sensible_heat_jit(tg, t1, rho, ch, ws).block_until_ready()
    end = time.time()
    
    elapsed = (end - start) / 100
    print(f"  compute_sensible_heat_jit (size={size}): {elapsed:.6f} s")
    
    # Check for reasonable performance (should be < 0.1s for size=10000)
    assert elapsed < 0.1, f"Performance too slow: {elapsed:.6f} s"
    
    print("✅ Performance tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("GHY JAX Unit Tests")
    print("=" * 60)
    
    test_compute_sensible_heat()
    test_compute_evap_limits()
    test_compute_runoff()
    test_get_soil_properties()
    test_compute_snow_melt()
    test_compute_soil_moisture_table()
    test_performance()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
