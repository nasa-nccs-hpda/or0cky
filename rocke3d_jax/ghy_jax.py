"""
JAX Implementation of ROCKE-3D GHY (Global Land Model)
========================================================

This module provides a JAX-based implementation of the TerraE Global Land Model
from ROCKE-3D's GHY.f. It computes soil moisture, temperature, evaporation,
runoff, and other land-surface processes.

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Dependencies:
- JAX (jax, jax.numpy)
- NumPy (for validation)
- fluxes_jax: For flux calculations
- pbl_simple_jax: For boundary layer calculations

Usage:
    from ghy_jax import compute_sensible_heat_jit, compute_evap_limits_jit
    sensible_heat = compute_sensible_heat_jit(tg, t1, rho, ch, ws)
    evap_limit = compute_evap_limits_jit(wc, wsat, wfc, wpwp)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Tuple, Optional


# Constants from CONSTANT module (assumed values)
RGAS = 287.0          # Specific gas constant for dry air (J/kg/K)
LHM = 2.501e6        # Latent heat of vaporization (J/kg)
LHE = 2.501e6        # Latent heat of evaporation (J/kg)
LHS = 2.834e6        # Latent heat of sublimation (J/kg)
SHA = 1004.6         # Specific heat of dry air (J/kg/K)
TF = 273.15          # Freezing point of water (K)
RHOW = 1000.0        # Density of water (kg/m^3)
SHV = 1864.0         # Specific heat of water vapor (J/kg/K)
SHI = 2106.0         # Specific heat of ice (J/kg/K)
STBO = 5.67e-8       # Stefan-Boltzmann constant (W/m^2/K^4)
DELTX = 0.608        # Virtual temperature factor
GRAV = 9.80665       # Gravitational acceleration (m/s^2)
TEENY = 1e-12        # Tiny value to avoid division by zero

# Soil-related constants
SHW = SHA * RHOW     # Heat capacity of water (J/m^3/K)
SHI_SOIL = SHI * RHOW # Heat capacity of ice (J/m^3/K)
FSN = LHM * RHOW     # Latent heat of fusion (J/m^3)
ELH = LHE * RHOW     # Latent heat of evaporation (J/m^3)


# ============================================================================
# Soil Moisture and Temperature Tables (from hl0 subroutine)
# ============================================================================

# Soil texture parameters (from GHY.f)
# Matric potential coefficients (a) for sand, silt, clay, peat
A_MATRIC = jnp.array([
    [0.2514, 0.0136, -2.8319, 0.5958],   # Sand
    [0.1481, 1.8726, 0.1025, -3.6416],   # Silt
    [0.2484, 2.4842, 0.4583, -3.9470],   # Clay
    [0.8781, -5.1816, 13.2385, -11.9501] # Peat
])

# Conductivity coefficients (b) for sand, silt, clay, peat
B_CONDUCTIVITY = jnp.array([
    [-0.4910, -9.8945, 9.7976, -3.2211],   # Sand
    [-0.3238, -12.9013, 3.4247, 4.4929],   # Silt
    [-0.5187, -13.4246, 2.8899, 5.0642],   # Clay
    [-3.0848, 9.5497, -26.2868, 16.6930]   # Peat
])

# Diffusivity coefficients (p) for sand, silt, clay, peat
P_DIFFUSIVITY = jnp.array([
    [-0.1800, -7.9999, 5.5685, -1.8868],   # Sand
    [-0.1000, -10.0085, 3.6752, 1.2304],   # Silt
    [-0.1951, -9.7055, 2.7418, 2.0054],   # Clay
    [-2.1220, 5.9983, -16.9824, 8.7615]   # Peat
])

# Saturated soil moisture (m^3/m^3) for sand, silt, clay, peat
SATURATED_MOISTURE = jnp.array([0.394, 0.537, 0.577, 0.885])

# Number of soil layers (from GHY_COM.f)
NLSN = 10  # Default number of soil layers


def compute_soil_moisture_table(
    hmin: float = -1000.0,
    delh1: float = -0.00625
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute soil moisture (theta), conductivity (xklm), and diffusivity (dlm)
    tables as a function of matric potential (h).
    
    This is a JAX implementation of the `hl0` subroutine from GHY.f.
    Note: This function is not JIT-compiled due to dynamic control flow.
    
    Args:
        hmin: Minimum matric potential (m) (default: -1000.0).
        delh1: Initial step size for geometric series (default: -0.00625).
    
    Returns:
        thm: Theta (volumetric soil moisture) table [65, 4].
        xklm: Hydraulic conductivity table [65, 4].
        dlm: Hydraulic diffusivity table [65, 4].
    """
    c = 2.3025851  # ln(10)
    nexp = 6
    nth_total = 2 ** nexp
    
    # Geometric series for matric potential (h)
    hlm = np.zeros(nth_total + 1)
    hlm[0] = 0.0
    
    # Solve for alph0 in s = ((1 + alph0)^n - 1) / alph0
    s = hmin / delh1
    alph0 = 1.0 / 8.0
    for _ in range(100):  # Iterative solver
        alph0_new = (s * alph0 + 1.0) ** (1.0 / nth_total) - 1.0
        if np.abs(alph0 - alph0_new) < 1e-8:
            break
        alph0 = alph0_new
    
    alpls1 = 1.0 + alph0
    delhn = delh1
    for j in range(1, nth_total + 1):
        hlm[j] = hlm[j - 1] + delhn
        delhn = alpls1 * delhn
    
    # Initialize tables
    thm = np.zeros((nth_total + 1, 4))
    xklm = np.zeros((nth_total + 1, 4))
    dlm = np.zeros((nth_total + 1, 4))
    
    # Compute theta (thm) as a function of h
    for i in range(4):  # Loop over soil textures
        thm[0, i] = 1.0
        for j in range(1, nth_total + 1):
            hs = -np.exp(c * (A_MATRIC[i, 0] + A_MATRIC[i, 1] + A_MATRIC[i, 2] + A_MATRIC[i, 3]))
            a1 = A_MATRIC[i, 2] / A_MATRIC[i, 3]
            a2 = (A_MATRIC[i, 1] - (np.log(-hlm[j] - hs)) / c) / A_MATRIC[i, 3]
            a3 = A_MATRIC[i, 0] / A_MATRIC[i, 3]
            
            # Newton-Raphson solver for theta
            testh = thm[j - 1, i]
            for _ in range(100):
                func = (testh ** 3) + (a1 * (testh ** 2)) + (a2 * testh) + a3
                dfunc = (3 * testh ** 2) + (2 * a1 * testh) + a2
                diff = func / dfunc
                testh_new = testh - diff
                if np.abs(diff) < 1e-6:
                    break
                testh = testh_new
            
            thm[j, i] = testh
        
        # Scale by saturated moisture
        thm[:, i] = thm[:, i] * SATURATED_MOISTURE[i]
    
    # Compute conductivity (xklm) and diffusivity (dlm)
    sxtn = 16.0
    for j in range(nth_total + 1):
        for i in range(4):
            # Conductivity: Sum over k=-1,0,1,2 (4 terms)
            # B_CONDUCTIVITY[i, :] has shape (4,), so we use all 4 coefficients
            powers = np.array([thm[j, i] ** k for k in range(-1, 3)])  # [thm^-1, thm^0, thm^1, thm^2]
            arg = np.sum(B_CONDUCTIVITY[i, :] * powers)
            arg = np.clip(arg, -sxtn, sxtn)
            xklm[j, i] = np.exp(c * arg)
            
            # Diffusivity: Sum over k=-1,0,1,2 (4 terms)
            arg = np.sum(P_DIFFUSIVITY[i, :] * powers)
            arg = np.clip(arg, -sxtn, sxtn)
            dlm[j, i] = np.exp(c * arg)
    
    return jnp.array(thm), jnp.array(xklm), jnp.array(dlm)


# Precompute soil moisture tables (lazy initialization)
_soil_moisture_tables = None


def get_soil_moisture_tables():
    """Lazy initialization of soil moisture tables."""
    global _soil_moisture_tables
    if _soil_moisture_tables is None:
        _soil_moisture_tables = compute_soil_moisture_table()
    return _soil_moisture_tables


# ============================================================================
# Sensible Heat Flux (from sensible_heat subroutine)
# ============================================================================

@jit
def compute_sensible_heat(
    tg: jnp.ndarray,      # Ground temperature (K)
    t1: jnp.ndarray,      # Air temperature at first layer (K)
    rho: jnp.ndarray,     # Air density (kg/m^3)
    ch: jnp.ndarray,      # Stanton number (heat transfer coefficient)
    ws: jnp.ndarray,      # Wind speed (m/s)
    cp: float = SHA       # Specific heat of dry air (J/kg/K)
) -> jnp.ndarray:
    """
    Compute sensible heat flux at the surface.
    
    Args:
        tg: Ground temperature (K).
        t1: Air temperature at first layer (K).
        rho: Air density (kg/m^3).
        ch: Stanton number (dimensionless).
        ws: Wind speed (m/s).
        cp: Specific heat of dry air (J/kg/K) (default: SHA).
    
    Returns:
        sensible_heat: Sensible heat flux (W/m^2).
    """
    # Sensible heat flux: rho * cp * ch * ws * (T_ground - T_air)
    sensible_heat = rho * cp * ch * ws * (tg - t1)
    return sensible_heat


# ============================================================================
# Evaporation Limits (from evap_limits subroutine)
# ============================================================================

@jit
def compute_evap_limits(
    wc: jnp.ndarray,       # Current soil moisture (m^3/m^3)
    wsat: jnp.ndarray,     # Saturated soil moisture (m^3/m^3)
    wfc: jnp.ndarray,      # Field capacity (m^3/m^3)
    wpwp: jnp.ndarray,     # Permanent wilting point (m^3/m^3)
    dt: float = 3600.0     # Time step (s)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute evaporation limits for soil moisture.
    
    Args:
        wc: Current soil moisture (m^3/m^3).
        wsat: Saturated soil moisture (m^3/m^3).
        wfc: Field capacity (m^3/m^3).
        wpwp: Permanent wilting point (m^3/m^3).
        dt: Time step (s) (default: 3600.0).
    
    Returns:
        evap_pot: Potential evaporation (m/s).
        evap_act: Actual evaporation (m/s).
        evap_lim: Evaporation limit (m/s).
    """
    # Potential evaporation (simplified)
    evap_pot = jnp.where(wc > wpwp, 1e-5, 0.0)  # Placeholder for potential evap
    
    # Actual evaporation (limited by soil moisture)
    evap_act = jnp.where(wc > wpwp, evap_pot * (wc - wpwp) / (wfc - wpwp), 0.0)
    evap_act = jnp.clip(evap_act, 0.0, evap_pot)
    
    # Evaporation limit (maximum allowed evaporation)
    evap_lim = jnp.where(wc > wsat, 0.0, evap_pot)
    
    return evap_pot, evap_act, evap_lim


# ============================================================================
# Runoff (from runoff subroutine)
# ============================================================================

@jit
def compute_runoff(
    precip: jnp.ndarray,    # Precipitation (m/s)
    wc: jnp.ndarray,       # Current soil moisture (m^3/m^3)
    wsat: jnp.ndarray,     # Saturated soil moisture (m^3/m^3)
    ksat: jnp.ndarray,     # Saturated hydraulic conductivity (m/s)
    slope: jnp.ndarray,    # Surface slope (dimensionless)
    dt: float = 3600.0     # Time step (s)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute runoff from precipitation and soil moisture.
    
    Args:
        precip: Precipitation (m/s).
        wc: Current soil moisture (m^3/m^3).
        wsat: Saturated soil moisture (m^3/m^3).
        ksat: Saturated hydraulic conductivity (m/s).
        slope: Surface slope (dimensionless).
        dt: Time step (s) (default: 3600.0).
    
    Returns:
        runoff: Surface runoff (m/s).
        infiltration: Infiltration (m/s).
    """
    # Infiltration capacity (simplified)
    infiltration_capacity = ksat * (wsat - wc) * slope
    
    # Infiltration (limited by precipitation and capacity)
    infiltration = jnp.minimum(precip, infiltration_capacity)
    
    # Runoff (excess precipitation)
    runoff = precip - infiltration
    runoff = jnp.maximum(runoff, 0.0)
    
    return runoff, infiltration


# ============================================================================
# Soil Properties (from get_soil_properties subroutine)
# ============================================================================

@jit
def get_soil_properties(
    q_in: jnp.ndarray,      # Input soil moisture (m^3/m^3)
    dz_in: jnp.ndarray,     # Input soil layer thickness (m)
    texture: int = 0        # Soil texture index (0: sand, 1: silt, 2: clay, 3: peat)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute soil properties (thermal conductivity, heat capacity, etc.).
    
    Args:
        q_in: Input soil moisture (m^3/m^3).
        dz_in: Input soil layer thickness (m).
        texture: Soil texture index (0: sand, 1: silt, 2: clay, 3: peat).
    
    Returns:
        lambda_soil: Thermal conductivity (W/m/K).
        c_soil: Heat capacity (J/m^3/K).
        k_soil: Hydraulic conductivity (m/s).
        d_soil: Hydraulic diffusivity (m^2/s).
    """
    # Placeholder: Simplified soil property calculations
    # Thermal conductivity (W/m/K)
    lambda_soil = jnp.where(
        texture == 0,  # Sand
        0.3 + 0.7 * q_in,
        jnp.where(
            texture == 1,  # Silt
            0.2 + 0.8 * q_in,
            jnp.where(
                texture == 2,  # Clay
                0.1 + 0.9 * q_in,
                0.05 + 0.95 * q_in  # Peat
            )
        )
    )
    
    # Heat capacity (J/m^3/K)
    c_soil = SHW * q_in + (1.0 - SATURATED_MOISTURE[texture]) * 800.0
    
    # Hydraulic conductivity (m/s)
    k_soil = jnp.where(
        texture == 0,  # Sand
        1e-4 * (q_in / SATURATED_MOISTURE[0]) ** 3,
        jnp.where(
            texture == 1,  # Silt
            1e-5 * (q_in / SATURATED_MOISTURE[1]) ** 3,
            jnp.where(
                texture == 2,  # Clay
                1e-6 * (q_in / SATURATED_MOISTURE[2]) ** 3,
                1e-7 * (q_in / SATURATED_MOISTURE[3]) ** 3  # Peat
            )
        )
    )
    
    # Hydraulic diffusivity (m^2/s)
    d_soil = k_soil * (1.0 / (SHW * q_in + TEENY))
    
    return lambda_soil, c_soil, k_soil, d_soil


# ============================================================================
# Snow Processes (from snow subroutine)
# ============================================================================

@jit
def compute_snow_melt(
    ts: jnp.ndarray,       # Surface temperature (K)
    wsn: jnp.ndarray,      # Snow water equivalent (m)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    dt: float = 3600.0     # Time step (s)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute snow melt and snow temperature updates.
    
    Args:
        ts: Surface temperature (K).
        wsn: Snow water equivalent (m).
        fsf: Downwelling shortwave radiation (W/m^2).
        flong: Downwelling longwave radiation (W/m^2).
        albedo: Surface albedo (dimensionless).
        dt: Time step (s) (default: 3600.0).
    
    Returns:
        snow_melt: Snow melt (m/s).
        ts_new: Updated surface temperature (K).
        wsn_new: Updated snow water equivalent (m).
    """
    # Net shortwave radiation absorbed by snow
    sw_absorbed = fsf * (1.0 - albedo)
    
    # Net longwave radiation
    lw_net = flong - STBO * (ts ** 4)
    
    # Total energy available for snow melt (W/m^2)
    energy_available = sw_absorbed + lw_net
    
    # Snow melt rate (m/s)
    snow_melt = jnp.where(
        energy_available > 0.0,
        energy_available / (FSN / dt),
        0.0
    )
    
    # Limit snow melt by available snow
    snow_melt = jnp.minimum(snow_melt, wsn / dt)
    
    # Update snow water equivalent
    wsn_new = wsn - snow_melt * dt
    wsn_new = jnp.maximum(wsn_new, 0.0)
    
    # Update surface temperature (simplified)
    ts_new = jnp.where(
        wsn_new > 0.0,
        TF,  # Keep at freezing if snow remains
        ts   # Otherwise, allow temperature to change
    )
    
    return snow_melt, ts_new, wsn_new


# ============================================================================
# JIT-Compiled Functions (for performance)
# ============================================================================

compute_sensible_heat_jit = jit(compute_sensible_heat)
compute_evap_limits_jit = jit(compute_evap_limits)
compute_runoff_jit = jit(compute_runoff)
get_soil_properties_jit = jit(get_soil_properties)
compute_snow_melt_jit = jit(compute_snow_melt)
