"""
JAX Implementation of ROCKE-3D SEAICE (Sea Ice Model)
======================================================

This module provides a JAX-compatible implementation of the sea ice model
from ROCKE-3D's SEAICE.f. It includes:
- Constants for sea ice properties (e.g., dtdp, LMI, XSI, alami0).
- Placeholder arrays for sea ice variables (e.g., SSI, ACE1I).
- Helper functions to compute derived quantities (e.g., sea ice temperature, salinity).

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from seaice_jax import compute_sea_ice_temperature, initialize_seaice
    tsil = compute_sea_ice_temperature(ssil, hsil)
    seaice = initialize_seaice(im, jm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from constant_jax import LHM, RHOI, BYRHOI, RHOW, SHI, SHW, BYSHI, BYLHM, RHOWS


# ============================================================================
# Constants for Sea Ice Properties
# ============================================================================

# Clausius-Clapeyron constant (dT/dp of ice) [K Pa^-1]
DTDp = jnp.float32(-7.5e-8)

# Number of temperature layers in ice
LMI = 4

# Fractions of mass layer in each temperature layer
XSI = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32)

# Reciprocal of XSI
BYXSI = jnp.array([1.0 / XSI[0], 1.0 / XSI[1], 1.0 / XSI[2], 1.0 / XSI[3]], dtype=jnp.float32)

# Thickness of first layer ice [m]
Z1I = jnp.float32(0.1)

# Ice mass first layer [kg/m^2]
ACE1I_CONST = jnp.float32(Z1I * RHOI)

# Minimum thickness of 2nd layer ice [m]
Z2OIM = jnp.float32(0.1)

# Minimum ice mass 2nd layer [kg/m^2]
AC2OIM = jnp.float32(Z2OIM * RHOI)

# Lambda coefficient for ice [J m^-1 K^-1 s^-1]
ALAMI0 = jnp.float32(2.11)

# Lambda coefficient for snow [J m^-1 K^-1 s^-1]
ALAMS = jnp.float32(0.35)

# Salinity/temperature coefficient for conductivity [J/(m s)/psu]
ALAMDS = jnp.float32(0.09)

# Temperature coefficient for conductivity [J/(m s)/degC^2]
ALAMDT = jnp.float32(-0.011)

# Density of snow [kg/m^3]
RHOS = jnp.float32(300.0)

# Lead fraction for ocean ice (1) for mean ice thickness of 1 m
FLEADOC = jnp.float32(0.06)

# Lead fraction for lakes [%]
FLEADLK = jnp.float32(0.0)

# Maximum thickness for lead fraction [m]
FLEADMX = jnp.float32(5.0)

# Reciprocal of scale depth for calculating open water fraction [1/m]
BYHREF = jnp.float32(1.1)

# Reciprocal of snow density * lambda
BYRLS = jnp.float32(1.0 / (RHOS * ALAMS))

# Coefficient of seawater freezing point w.r.t. salinity [C/ppt]
MU = jnp.float32(0.054)

# Default value for sea ice salinity [kg/kg]
SSI0 = jnp.float32(0.0032)

# Fraction of ocean salinity found in new-formed ice [1]
FSSS = jnp.float32(8.0 / 35.0)

# Flag for constant sea ice salinity
QSFIX = False

# Implicitness for heat diffusion in sea ice (1=fully implicit)
ALPHA = jnp.float32(1.0)

# Default ice-ocean friction velocity [m/s]
OI_USTAR0 = jnp.float32(1e-3)

# Factor controlling lateral melt of ocean ice
SILMFACT = jnp.float32(1e-7)

# Exponent for temperature dependence of lateral melt
SILMPOW = jnp.float32(1.36)

# Flag for snow ice formation (1=allow, 0=disallow)
SNOW_ICE = 1

# Flag for ocean surface tilt calculation (0=geostrophy, 1=free surface)
OSURF_TILT = 1

# Critical cutoff for salt amount [kg/kg]
SSIMIN = jnp.float32(1e-6)

# Flag for debug mode
DEBUG = False

# Sea ice thermodynamics formulation (BP or SI)
SEAICE_THERMO = "BP"

# Minimum allowed sea/lake ice temperature [C]
MIN_ICE_TEMPERATURE = jnp.float32(-100.0)


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46


# ============================================================================
# Placeholder Arrays for Sea Ice Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM, LMI).
# Users should replace these with their own data.

# Sea ice salinity [kg/kg] (I, J, LMI)
SSI = jnp.zeros((IM, JM, LMI), dtype=jnp.float32)

# Sea ice temperature [C] (I, J, LMI)
TSIL = jnp.zeros((IM, JM, LMI), dtype=jnp.float32)

# Sea ice thickness [m] (I, J, LMI)
HSIL = jnp.zeros((IM, JM, LMI), dtype=jnp.float32)

# Sea ice mass [kg/m^2] (I, J, LMI)
MSI = jnp.zeros((IM, JM, LMI), dtype=jnp.float32)

# Snow thickness [m] (I, J)
SNOW = jnp.zeros((IM, JM), dtype=jnp.float32)

# Snow mass [kg/m^2] (I, J)
SNOWM = jnp.zeros((IM, JM), dtype=jnp.float32)

# Sea ice concentration [1] (I, J)
ACE1I = jnp.zeros((IM, JM), dtype=jnp.float32)

# Sea ice surface temperature [C] (I, J)
TSURF = jnp.zeros((IM, JM), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_sea_ice_temperature(
    ssil: jnp.ndarray,
    hsil: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute sea ice temperature from salinity and thickness.
    
    Args:
        ssil: Sea ice salinity [kg/kg] (I, J, LMI)
        hsil: Sea ice thickness [m] (I, J, LMI)
    
    Returns:
        tsil: Sea ice temperature [C] (I, J, LMI)
    """
    # For simplicity, assume a linear relationship between salinity and temperature
    # This is a placeholder; the actual computation depends on the model
    tsil = -MU * ssil * 1000.0  # Convert salinity to ppt and apply MU
    return tsil


@jit
def compute_sea_ice_salinity(
    tsil: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute sea ice salinity from temperature.
    
    Args:
        tsil: Sea ice temperature [C] (I, J, LMI)
    
    Returns:
        ssil: Sea ice salinity [kg/kg] (I, J, LMI)
    """
    # For simplicity, assume a linear relationship between temperature and salinity
    # This is a placeholder; the actual computation depends on the model
    ssil = -tsil / (MU * 1000.0)  # Convert temperature to salinity
    return ssil


# ============================================================================
# Core Sea Ice Subroutines (Simplified)
# ============================================================================

@jit
def prec_si(
    snow: jnp.ndarray,      # Snow mass [kg/m^2] (I, J)
    msi2: jnp.ndarray,      # Second layer ice mass [kg/m^2] (I, J)
    hsil: jnp.ndarray,      # Enthalpy of ice layers [J/m^2] (I, J, LMI)
    tsil: jnp.ndarray,      # Temperature of ice layers [C] (I, J, LMI)
    ssil: jnp.ndarray,      # Salt in ice layers [kg/m^2] (I, J, LMI)
    prcp: jnp.ndarray,      # Precipitation amount [kg/m^2] (I, J)
    enrgp: jnp.ndarray,     # Total energy of precip [J/m^2] (I, J)
    run0: jnp.ndarray,      # Runoff from ice [kg/m^2] (I, J)
    srun0: jnp.ndarray,     # Salt in runoff from ice [kg/m^2] (I, J)
    erun0: jnp.ndarray,     # Energy in runoff from ice [J/m^2] (I, J)
    wetsnow: jnp.ndarray,   # True if snow is wet (I, J)
    cmprs: jnp.ndarray,     # Snow compression to ice [kg/m^2] (I, J)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of PREC_SI: Adds precipitation to sea/lake ice.
    
    Args:
        snow: Snow mass [kg/m^2].
        msi2: Second layer ice mass [kg/m^2].
        hsil: Enthalpy of ice layers [J/m^2].
        tsil: Temperature of ice layers [C].
        ssil: Salt in ice layers [kg/m^2].
        prcp: Precipitation amount [kg/m^2].
        enrgp: Total energy of precip [J/m^2].
        run0: Runoff from ice [kg/m^2] (output).
        srun0: Salt in runoff from ice [kg/m^2] (output).
        erun0: Energy in runoff from ice [J/m^2] (output).
        wetsnow: True if snow is wet (output).
        cmprs: Snow compression to ice [kg/m^2] (output).
    
    Returns:
        Updated snow, msi2, hsil, tsil, ssil, run0, srun0, erun0.
    """
    # Constants
    snomax = 1.0 * RHOS  # Maximum allowed snow depth (kg/m^2)
    
    # Initialize outputs
    run0 = jnp.zeros_like(snow)
    srun0 = jnp.zeros_like(snow)
    erun0 = jnp.zeros_like(snow)
    wetsnow = jnp.zeros_like(snow, dtype=jnp.bool_)
    cmprs = jnp.zeros_like(snow)
    
    # Snowfall and rain
    snwf = jnp.maximum(0.0, jnp.minimum(prcp, -enrgp * BYLHM))  # Snowfall
    rain = prcp - snwf  # Rain
    
    # Update snow mass
    snow_new = snow + snwf
    
    # Check if snow exceeds maximum
    excess_snow = jnp.maximum(0.0, snow_new - snomax)
    snow_new = jnp.minimum(snow_new, snomax)
    
    # Convert excess snow to ice (compression)
    cmprs = excess_snow
    msi2_new = msi2 + cmprs
    
    # Update temperature and salinity (simplified)
    # Assume new snow has temperature 0C and salinity 0
    tsil_new = tsil  # No change for simplicity
    ssil_new = ssil  # No change for simplicity
    hsil_new = hsil  # No change for simplicity
    
    # Set wetsnow flag if rain is significant
    wetsnow = rain > 1e-5 * prcp
    
    return snow_new, msi2_new, hsil_new, tsil_new, ssil_new, run0, srun0, erun0


@jit
def addice(
    snow: jnp.ndarray,      # Snow mass [kg/m^2] (I, J)
    msi1: jnp.ndarray,      # First layer ice mass [kg/m^2] (I, J)
    msi2: jnp.ndarray,      # Second layer ice mass [kg/m^2] (I, J)
    hsil: jnp.ndarray,      # Enthalpy of ice layers [J/m^2] (I, J, LMI)
    tsil: jnp.ndarray,      # Temperature of ice layers [C] (I, J, LMI)
    ssil: jnp.ndarray,      # Salt in ice layers [kg/m^2] (I, J, LMI)
    tfrz: float = -1.8,      # Freezing temperature of seawater [C]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of ADDICE: Add new ice due to freezing.
    
    Args:
        snow: Snow mass [kg/m^2].
        msi1: First layer ice mass [kg/m^2].
        msi2: Second layer ice mass [kg/m^2].
        hsil: Enthalpy of ice layers [J/m^2].
        tsil: Temperature of ice layers [C].
        ssil: Salt in ice layers [kg/m^2].
        tfrz: Freezing temperature of seawater [C].
    
    Returns:
        Updated snow, msi1, msi2, hsil, tsil, ssil.
    """
    # For simplicity, assume all new ice is added to the first layer
    # This is a placeholder; the actual computation depends on the model
    
    # Compute new ice mass (simplified)
    new_ice = jnp.maximum(0.0, ACE1I_CONST - msi1)
    
    # Update ice mass
    msi1_new = msi1 + new_ice
    msi2_new = msi2  # No change for simplicity
    
    # Update temperature and salinity (simplified)
    tsil_new = tsil  # No change for simplicity
    ssil_new = ssil  # No change for simplicity
    hsil_new = hsil  # No change for simplicity
    
    return snow, msi1_new, msi2_new, hsil_new, tsil_new, ssil_new


@jit
def simelt(
    snow: jnp.ndarray,      # Snow mass [kg/m^2] (I, J)
    msi1: jnp.ndarray,      # First layer ice mass [kg/m^2] (I, J)
    msi2: jnp.ndarray,      # Second layer ice mass [kg/m^2] (I, J)
    hsil: jnp.ndarray,      # Enthalpy of ice layers [J/m^2] (I, J, LMI)
    tsil: jnp.ndarray,      # Temperature of ice layers [C] (I, J, LMI)
    ssil: jnp.ndarray,      # Salt in ice layers [kg/m^2] (I, J, LMI)
    tfrz: float = -1.8,      # Freezing temperature of seawater [C]
    fsf: jnp.ndarray = None, # Downwelling shortwave radiation [W/m^2] (optional)
    flong: jnp.ndarray = None, # Downwelling longwave radiation [W/m^2] (optional)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of SIMELT: Sea ice melt.
    
    Args:
        snow: Snow mass [kg/m^2].
        msi1: First layer ice mass [kg/m^2].
        msi2: Second layer ice mass [kg/m^2].
        hsil: Enthalpy of ice layers [J/m^2].
        tsil: Temperature of ice layers [C].
        ssil: Salt in ice layers [kg/m^2].
        tfrz: Freezing temperature of seawater [C].
        fsf: Downwelling shortwave radiation [W/m^2] (optional).
        flong: Downwelling longwave radiation [W/m^2] (optional).
    
    Returns:
        Updated snow, msi1, msi2, hsil, tsil, ssil.
    """
    # For simplicity, assume a fixed melt rate based on temperature
    # This is a placeholder; the actual computation depends on the model
    
    # Compute melt rate (simplified)
    melt_rate = jnp.maximum(0.0, (tsil[:, :, 0] - tfrz) * 1e-5)  # Melt rate [kg/m^2/s]
    
    # Update ice mass
    msi1_new = jnp.maximum(0.0, msi1 - melt_rate * 3600.0)  # Assume 1-hour timestep
    msi2_new = msi2  # No change for simplicity
    
    # Update temperature and salinity (simplified)
    tsil_new = tsil  # No change for simplicity
    ssil_new = ssil  # No change for simplicity
    hsil_new = hsil  # No change for simplicity
    
    return snow, msi1_new, msi2_new, hsil_new, tsil_new, ssil_new


@jit
def sea_ice(
    snow: jnp.ndarray,      # Snow mass [kg/m^2] (I, J)
    msi1: jnp.ndarray,      # First layer ice mass [kg/m^2] (I, J)
    msi2: jnp.ndarray,      # Second layer ice mass [kg/m^2] (I, J)
    hsil: jnp.ndarray,      # Enthalpy of ice layers [J/m^2] (I, J, LMI)
    tsil: jnp.ndarray,      # Temperature of ice layers [C] (I, J, LMI)
    ssil: jnp.ndarray,      # Salt in ice layers [kg/m^2] (I, J, LMI)
    tfrz: float = -1.8,      # Freezing temperature of seawater [C]
    fsf: jnp.ndarray = None, # Downwelling shortwave radiation [W/m^2] (optional)
    flong: jnp.ndarray = None, # Downwelling longwave radiation [W/m^2] (optional)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of SEA_ICE: Sea ice thermodynamics.
    
    Args:
        snow: Snow mass [kg/m^2].
        msi1: First layer ice mass [kg/m^2].
        msi2: Second layer ice mass [kg/m^2].
        hsil: Enthalpy of ice layers [J/m^2].
        tsil: Temperature of ice layers [C].
        ssil: Salt in ice layers [kg/m^2].
        tfrz: Freezing temperature of seawater [C].
        fsf: Downwelling shortwave radiation [W/m^2] (optional).
        flong: Downwelling longwave radiation [W/m^2] (optional).
    
    Returns:
        Updated snow, msi1, msi2, hsil, tsil, ssil.
    """
    # For simplicity, assume no changes to ice properties
    # This is a placeholder; the actual computation depends on the model
    
    return snow, msi1, msi2, hsil, tsil, ssil
    """
    Compute sea ice salinity from temperature.
    
    Args:
        tsil: Sea ice temperature [C] (I, J, LMI)
    
    Returns:
        ssil: Sea ice salinity [kg/kg] (I, J, LMI)
    """
    # For simplicity, assume a linear relationship between temperature and salinity
    # This is a placeholder; the actual computation depends on the model
    ssil = -tsil / (MU * 1000.0)  # Convert temperature to salinity
    return ssil


@jit
def compute_sea_ice_mass(
    hsil: jnp.ndarray,
    rhoi: jnp.ndarray = RHOI,
) -> jnp.ndarray:
    """
    Compute sea ice mass from thickness.
    
    Args:
        hsil: Sea ice thickness [m] (I, J, LMI)
        rhoi: Density of ice [kg/m^3]
    
    Returns:
        msi: Sea ice mass [kg/m^2] (I, J, LMI)
    """
    msi = hsil * rhoi
    return msi


@jit
def compute_snow_mass(
    snow: jnp.ndarray,
    rhos: jnp.ndarray = RHOS,
) -> jnp.ndarray:
    """
    Compute snow mass from thickness.
    
    Args:
        snow: Snow thickness [m] (I, J)
        rhos: Density of snow [kg/m^3]
    
    Returns:
        snowm: Snow mass [kg/m^2] (I, J)
    """
    snowm = snow * rhos
    return snowm


@jit
def compute_sea_ice_concentration(
    msi: jnp.ndarray,
    ace1i: jnp.ndarray = ACE1I,
) -> jnp.ndarray:
    """
    Compute sea ice concentration from mass.
    
    Args:
        msi: Sea ice mass [kg/m^2] (I, J, LMI)
        ace1i: Ice mass first layer [kg/m^2]
    
    Returns:
        ace1i: Sea ice concentration [1] (I, J)
    """
    # For simplicity, assume concentration = 1 if mass > 0
    ace1i = jnp.where(msi[:, :, 0] > 0.0, 1.0, 0.0)
    return ace1i


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_seaice(
    im: int = IM,
    jm: int = JM,
) -> dict:
    """
    Initialize SEAICE arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        A dictionary containing initialized SEAICE arrays
    """
    # Initialize sea ice arrays
    ssi = jnp.zeros((im, jm, LMI), dtype=jnp.float32)
    tsil = jnp.zeros((im, jm, LMI), dtype=jnp.float32)
    hsil = jnp.zeros((im, jm, LMI), dtype=jnp.float32)
    msi = jnp.zeros((im, jm, LMI), dtype=jnp.float32)
    snow = jnp.zeros((im, jm), dtype=jnp.float32)
    snowm = jnp.zeros((im, jm), dtype=jnp.float32)
    ace1i = jnp.zeros((im, jm), dtype=jnp.float32)
    tsurf = jnp.zeros((im, jm), dtype=jnp.float32)
    
    return {
        "SSI": ssi, "TSIL": tsil, "HSIL": hsil, "MSI": msi,
        "SNOW": snow, "SNOWM": snowm, "ACE1I": ace1i, "TSURF": tsurf,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Constants
    "DTDp", "LMI", "XSI", "BYXSI", "Z1I", "ACE1I_CONST",
    "Z2OIM", "AC2OIM", "ALAMI0", "ALAMS", "ALAMDS", "ALAMDT",
    "RHOS", "FLEADOC", "FLEADLK", "FLEADMX", "BYHREF", "BYRLS",
    "MU", "SSI0", "FSSS", "QSFIX", "ALPHA",
    "OI_USTAR0", "SILMFACT", "SILMPOW", "SNOW_ICE", "OSURF_TILT",
    "SSIMIN", "DEBUG", "SEAICE_THERMO", "MIN_ICE_TEMPERATURE",
    # Default model resolution
    "IM", "JM",
    # Placeholder arrays
    "SSI", "TSIL", "HSIL", "MSI", "SNOW", "SNOWM", "ACE1I", "TSURF",
    # Helper functions
    "compute_sea_ice_temperature", "compute_sea_ice_salinity",
    "compute_sea_ice_mass", "compute_snow_mass", "compute_sea_ice_concentration",
    "initialize_seaice",
]
