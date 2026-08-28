"""
JAX Implementation of ROCKE-3D CONSTANT Module
==============================================

This module provides a JAX-compatible implementation of the physical constants
defined in ROCKE-3D's Constants_mod.F90. It includes all the constants required
for atmospheric modeling, radiative transfer, and surface processes.

Key Features:
- All constants are defined as JAX-compatible values.
- Constants are organized by category (numerical, physical, astronomical, etc.).
- Includes Earth-specific and general physical constants.

Usage:
    from constant_jax import RGAS, GRAV, STBO, LHE, TF, etc.
    # Use constants directly in JAX functions
"""

import jax.numpy as jnp
from jax import jit


# ============================================================================
# Numerical Constants
# ============================================================================

# Missing value
UNDEF_VAL = jnp.float32(1e30)   # huge(1.d0) (float32 max ~3.4e38)
IUNDEF_VAL = 2**31 - 1           # huge(1) for int32
UNDEF = jnp.float32(-1e30)       # Missing value
TEENY = jnp.float32(1e-30)       # Small positive value to avoid division by zero

# NaN (Not a Number)
INT_NAN = -1                     # i.e., = Z'FFFFFFFFFFFFFFFF'
NAN = jnp.float32(jnp.nan)       # NaN


# ============================================================================
# Mathematical Constants
# ============================================================================

# Pi and related constants
PI = jnp.float32(3.14159265358979323846)
TWOPI = jnp.float32(2.0 * PI)
RADIANS_PER_DEGREE = jnp.float32(PI / 180.0)
RADIAN = jnp.float32(1.0)

# Useful reciprocals and roots
ZERO = jnp.float32(0.0)
ONE = jnp.float32(1.0)
RT2 = jnp.float32(0.7071067811865476)    # sqrt(0.5)
BYRT2 = jnp.float32(1.4142135623730951)   # 1/sqrt(0.5) = sqrt(2)
RT3 = jnp.float32(0.5773502691896257)    # sqrt(1/3)
BYRT3 = jnp.float32(1.7320508075688772)   # 1/sqrt(1/3) = sqrt(3)
RT12 = jnp.float32(0.28867513459481287)   # sqrt(1/12)
BYRT12 = jnp.float32(3.4641016151377544)   # 1/sqrt(1/12) = sqrt(12)
BY3 = jnp.float32(0.3333333333333333)     # 1/3
BY6 = jnp.float32(0.16666666666666666)    # 1/6
BY9 = jnp.float32(0.1111111111111111)     # 1/9
BY12 = jnp.float32(0.08333333333333333)   # 1/12


# ============================================================================
# Physical Constants
# ============================================================================

# Stefan-Boltzmann constant (W/m^2 K^4)
STBO = jnp.float32(5.67037321e-8)  # CODATA recommended in 2010

# Boltzmann's constant (J K^-1)
K_BOLTZMANN = jnp.float32(1.380662e-23)

# Planck constant (J s)
H_PLANCK = jnp.float32(6.626176e-34)

# Speed of light in a vacuum (m s^-1)
C_LIGHT = jnp.float32(2.9979245e8)

# Latent heats (J kg^-1)
LHE = jnp.float32(2.5e6)          # Latent heat of evaporation at 0 C
LHM = jnp.float32(3.34e5)         # Latent heat of melt at 0 C
BYLHM = jnp.float32(1.0 / LHM)    # 1/LHM
LHS = jnp.float32(LHE + LHM)      # Latent heat of sublimation at 0 C

# Densities (kg m^-3)
RHOW = jnp.float32(1000.0)        # Density of pure water
RHOWS = jnp.float32(1030.0)       # Density of average sea water
BYRHOWS = jnp.float32(1.0 / RHOWS)  # 1/RHOWS
RHOI = jnp.float32(916.6)         # Density of pure ice
BYRHOI = jnp.float32(1.0 / RHOI)  # 1/RHOI

# Freezing point of water (K)
TF = jnp.float32(273.15)
BYTF = jnp.float32(1.0 / TF)      # 1/TF

# Specific heats (J kg^-1 K^-1)
SHW = jnp.float32(4185.0)         # Heat capacity of water (at 20 C)
BYSHW = jnp.float32(1.0 / SHW)    # 1/SHW
SHI = jnp.float32(2060.0)         # Heat capacity of pure ice (at 0 C)
BYSHI = jnp.float32(1.0 / SHI)    # 1/SHI
SHV = jnp.float32(0.0)            # Specific heat of water vapor (currently 0 for energy conservation)

# Gas constants
GASC = jnp.float32(8.314510)       # Gas constant (J K^-1 mol^-1)
BYGASC = jnp.float32(1.0 / GASC) # 1/GASC
MAIR = jnp.float32(28.9655)       # Molar mass of dry air (g/mol)
RGAS = jnp.float32(1e3 * GASC / MAIR)  # Gas constant for dry air (J K^-1 kg^-1) = 287.05...

# Water vapor constants
MWAT = jnp.float32(18.015)        # Molar mass of water vapor (g/mol)
RVAP = jnp.float32(1e3 * GASC / MWAT)  # Gas constant for water vapor (J K^-1 kg^-1) = 461.5...
MRAT = jnp.float32(MWAT / MAIR)  # Mass ratio of air to water vapor (0.62197)
BYMRAT = jnp.float32(1.0 / MRAT)  # 1/MRAT (1.6078)
DELTX = jnp.float32(BYMRAT - 1.0)  # Coefficient of humidity in virtual temperature definition (0.6078)

# Specific heat ratio
SRAT = jnp.float32(1.401)          # Ratio of specific heats (cp/cv)
KAPA = jnp.float32((SRAT - 1.0) / SRAT)  # Ideal gas law exponent for dry air (0.2862)
BYKAPA = jnp.float32(1.0 / KAPA)
BYKAPAP1 = jnp.float32(1.0 / (KAPA + 1.0))
BYKAPAP2 = jnp.float32(1.0 / (KAPA + 2.0))

# Specific heat of dry air (J kg^-1 K^-1)
SHA = jnp.float32(RGAS / KAPA)
BYSHA = jnp.float32(1.0 / SHA)

# Viscosity constants
VISC_AIR0 = jnp.float32(1.7e-5)          # Dynamic viscosity of air (kg m^-1 s^-1)
VISC_AIR_KIN0 = jnp.float32(1.46e-5)     # Kinematic viscosity of air (m^2 s^-1)
VISC_WTR_KIN = jnp.float32(1.05e-6)      # Kinematic viscosity of water (m^2 s^-1)

# Avogadro's constant
AVOG = jnp.float32(6.02214129e23)         # Avogadro's constant (molecules/mol)
BYAVOG = jnp.float32(1.0 / AVOG)          # 1/Avogadro's constant
LOSCHMIDT_CONSTANT = jnp.float32(2.6867805e19)  # Loschmidt constant (cm^-3 at STP)


# ============================================================================
# Astronomical Constants
# ============================================================================

# Astronomical unit (m)
ASTRONOMICAL_UNIT = jnp.float32(149597870700.0)

# Solar constants
SOLAR_CONSTANT = jnp.float32(1365.0)     # Solar constant (W/m^2)
SOLAR_T_EFFECTIVE = jnp.float32(5785.0)  # Effective solar temperature (K)
SOLAR_RADIUS = jnp.float32(6.96e8)       # Radius of the Sun (m)

# Earth constants
RADIUS = jnp.float32(6371000.0)          # Radius of the Earth (m)
AREAG = jnp.float32(4.0 * PI * RADIUS * RADIUS)  # Surface area of the Earth (m^2)

# Gravitational acceleration (m s^-2)
GRAV = jnp.float32(9.80665)
BYGRAV = jnp.float32(1.0 / GRAV)

# Earth's rotation rate (s^-1)
OMEGA = jnp.float32(7.292115e-5)         # Earth's rotation rate (rad/s)
OMEGA2 = jnp.float32(2.0 * OMEGA)        # 2 * Earth's rotation rate

# Days per year (solar days per orbital period)
DAYS_PER_YEAR = jnp.float32(365.25)      # Approximate (can be updated for exoplanets)


# ============================================================================
# Lapse Rate Constants
# ============================================================================

# Dry adiabatic lapse rate (K m^-1)
GAMD = jnp.float32(GRAV * KAPA / RGAS)

# Moist adiabatic lapse rate (K m^-1)
BMOIST = jnp.float32(0.0065)
BBYG = jnp.float32(BMOIST * BYGRAV)        # Moist adiabatic lapse rate divided by grav (K s^2 m^-2)
GBYRB = jnp.float32(GRAV / (RGAS * BMOIST))  # Grav divided by rgas and bmoist (kg m^2 s^-1 J^-1)


# ============================================================================
# Conversion Factors
# ============================================================================

# kg/m^2 to milli-bars (mbar m^2 kg^-1)
KG2MB = jnp.float32(1e-2 * GRAV)
MB2KG = jnp.float32(1e2 * BYGRAV)

# kg/m^2 water to mm (mm m^2 kg^-1)
KGPA2MM = jnp.float32(1.0)
MM2KGPA = jnp.float32(1.0)


# ============================================================================
# Atmospheric Composition (Earth Defaults)
# ============================================================================

# Fraction of gases in dry air (0-1)
PN2 = jnp.float32(0.780840)   # Nitrogen
PO2 = jnp.float32(0.209476)   # Oxygen
PAR = jnp.float32(0.0093)     # Argon
PH2 = jnp.float32(0.0)        # Hydrogen

# Planet name
PLANET_NAME = "Earth"


# ============================================================================
# Utility Functions
# ============================================================================

@jit
def visc_air(T):
    """
    Compute the dynamic viscosity of air as a function of temperature (K).
    Uses the Sutherland formula.
    
    Args:
        T: Temperature (K)
    
    Returns:
        Dynamic viscosity of air (kg/m s)
    """
    n0 = jnp.float32(1.827e-5)
    T0 = jnp.float32(291.15)
    C = jnp.float32(120.0)
    return n0 * jnp.sqrt((T / T0)**3) * (T0 + C) / (T + C)


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Numerical constants
    "UNDEF_VAL", "IUNDEF_VAL", "UNDEF", "TEENY", "INT_NAN", "NAN",
    # Mathematical constants
    "PI", "TWOPI", "RADIANS_PER_DEGREE", "RADIAN",
    "ZERO", "ONE", "RT2", "BYRT2", "RT3", "BYRT3", "RT12", "BYRT12",
    "BY3", "BY6", "BY9", "BY12",
    # Physical constants
    "STBO", "K_BOLTZMANN", "H_PLANCK", "C_LIGHT",
    "LHE", "LHM", "BYLHM", "LHS",
    "RHOW", "RHOWS", "BYRHOWS", "RHOI", "BYRHOI",
    "TF", "BYTF",
    "SHW", "BYSHW", "SHI", "BYSHI", "SHV",
    "GASC", "BYGASC", "MAIR", "RGAS",
    "MWAT", "RVAP", "MRAT", "BYMRAT", "DELTX",
    "SRAT", "KAPA", "BYKAPA", "BYKAPAP1", "BYKAPAP2",
    "SHA", "BYSHA",
    "VISC_AIR0", "VISC_AIR_KIN0", "VISC_WTR_KIN",
    "AVOG", "BYAVOG", "LOSCHMIDT_CONSTANT",
    # Astronomical constants
    "ASTRONOMICAL_UNIT", "SOLAR_CONSTANT", "SOLAR_T_EFFECTIVE", "SOLAR_RADIUS",
    "RADIUS", "AREAG", "GRAV", "BYGRAV",
    "OMEGA", "OMEGA2", "DAYS_PER_YEAR",
    # Lapse rate constants
    "GAMD", "BMOIST", "BBYG", "GBYRB",
    # Conversion factors
    "KG2MB", "MB2KG", "KGPA2MM", "MM2KGPA",
    # Atmospheric composition
    "PN2", "PO2", "PAR", "PH2", "PLANET_NAME",
    # Utility functions
    "visc_air",
]
