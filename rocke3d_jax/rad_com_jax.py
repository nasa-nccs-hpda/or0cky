"""
JAX Implementation of ROCKE-3D RAD_COM (Radiation Common Variables)
====================================================================

This module provides a JAX-compatible implementation of the radiation common
variables from ROCKE-3D's RAD_COM.f. It includes:
- Constants for orbital parameters (e.g., OMEGT_def, OBLIQ_def, ECCN_def).
- Placeholder arrays for radiation variables (e.g., SRHR, TRHR, FSF).
- Helper functions to compute derived quantities (e.g., solar forcing).

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from rad_com_jax import compute_solar_forcing, initialize_rad_com
    fsf = compute_solar_forcing(solar_constant, cosz, itype)
    rad_com = initialize_rad_com(im, jm, lm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from constant_jax import STBO, SOLAR_CONSTANT


# ============================================================================
# Orbital Parameters (Default for Earth)
# ============================================================================

# Number of radiation time steps per physics time step
NRAD = 5

# Default orbital parameters
OMEGT_DEF = jnp.float32(282.9)    # Precession angle (degrees from vernal equinox)
OBLIQ_DEF = jnp.float32(23.44)    # Obliquity angle (degrees)
ECCN_DEF = jnp.float32(0.0167)    # Eccentricity

# Actual orbital parameters (can be updated)
OMEGT = jnp.float32(OMEGT_DEF)
OBLIQ = jnp.float32(OBLIQ_DEF)
ECCN = jnp.float32(ECCN_DEF)

# Orbital parameter control
VARIABLE_ORB_PAR = -2  # -2: Use defaults
ORB_PAR_YEAR_BP = 0    # Offset from model year or 1950
ORB_PAR = jnp.array([ECCN_DEF, OBLIQ_DEF, OMEGT_DEF], dtype=jnp.float32)


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46
LM = 40
LM_REQ = 3
LM_TOTAL = LM + LM_REQ


# ============================================================================
# Placeholder Arrays for Radiation Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM, LM_TOTAL).
# Users should replace these with their own data.

# Radiative equilibrium temperatures above model top
RQT = jnp.zeros((IM, JM, LM_REQ), dtype=jnp.float32)

# Total temperature change in adjusted forcing runs
TCHG = jnp.zeros((IM, JM, LM_TOTAL), dtype=jnp.float32)

# Solar and thermal radiative heating rates
SRHR = jnp.zeros((IM, JM, LM_TOTAL + 1), dtype=jnp.float32)  # Solar (W/m^2)
TRHR = jnp.zeros((IM, JM, LM_TOTAL + 1), dtype=jnp.float32)  # Thermal (W/m^2)

# Upward thermal radiation at the surface from rad step
TRSURF = jnp.zeros((IM, JM, LM_TOTAL + 1), dtype=jnp.float32)

# Solar forcing over each surface type
FSF = jnp.zeros((IM, JM, 4), dtype=jnp.float32)  # 4 surface types

# Solar incident at surface, direct fraction
FSRDIR = jnp.zeros((IM, JM), dtype=jnp.float32)

# Direct beam solar incident at surface
DIRVIS = jnp.zeros((IM, JM), dtype=jnp.float32)

# Incident solar direct+diffuse visible at surface
SRVISSURF = jnp.zeros((IM, JM), dtype=jnp.float32)

# Total incident solar at surface
SRDN = jnp.zeros((IM, JM), dtype=jnp.float32)

# Diffuse visible incident solar at surface
FSRDIF = jnp.zeros((IM, JM), dtype=jnp.float32)

# Direct and diffuse NIR incident solar at surface
DIRNIR = jnp.zeros((IM, JM), dtype=jnp.float32)
DIFNIR = jnp.zeros((IM, JM), dtype=jnp.float32)

# Net solar and thermal radiation (saved for diagnostics)
SRNFLB_SAVE = jnp.zeros((IM, JM, LM_TOTAL + 1), dtype=jnp.float32)
TRNFLB_SAVE = jnp.zeros((IM, JM, LM_TOTAL + 1), dtype=jnp.float32)

# Column-sum water and ice cloud optical depths (for diagnostics)
TAUSUMW = jnp.zeros((IM, JM), dtype=jnp.float32)
TAUSUMI = jnp.zeros((IM, JM), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_solar_forcing(
    solar_constant: jnp.ndarray = SOLAR_CONSTANT,
    cosz: jnp.ndarray = 1.0,
    itype: jnp.ndarray = 1,
) -> jnp.ndarray:
    """
    Compute solar forcing for a given surface type.
    
    Args:
        solar_constant: Solar constant (W/m^2)
        cosz: Cosine of solar zenith angle (dimensionless)
        itype: Surface type (1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land)
    
    Returns:
        fsf: Solar forcing (W/m^2)
    """
    # Solar forcing: FSF = solar_constant * cosz
    fsf = solar_constant * cosz
    return fsf


@jit
def compute_solar_fluxes(
    fsf: jnp.ndarray,
    albedo: jnp.ndarray = 0.3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute solar fluxes (direct, diffuse, total).
    
    Args:
        fsf: Solar forcing (W/m^2) (I, J)
        albedo: Surface albedo (dimensionless) (I, J)
    
    Returns:
        srdn: Total incident solar at surface (W/m^2) (I, J)
        srvis: Incident solar direct+diffuse visible at surface (W/m^2) (I, J)
        fsrdir: Solar incident at surface, direct fraction (dimensionless) (I, J)
    """
    # For simplicity, assume all solar flux is direct
    srdn = fsf
    srvis = fsf * 0.5  # Assume 50% is visible
    fsrdir = jnp.ones_like(fsf)  # Assume all is direct
    return srdn, srvis, fsrdir


@jit
def compute_thermal_fluxes(
    temperature: jnp.ndarray,
    emissivity: jnp.ndarray = 0.98,
) -> jnp.ndarray:
    """
    Compute thermal fluxes (longwave radiation).
    
    Args:
        temperature: Temperature (K)
        emissivity: Surface emissivity (dimensionless)
    
    Returns:
        trhr: Thermal radiative heating rate (W/m^2)
    """
    # Thermal flux: TRHR = emissivity * STBO * temperature^4
    trhr = emissivity * STBO * temperature**4
    return trhr


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_rad_com(
    im: int = IM,
    jm: int = JM,
    lm: int = LM,
) -> dict:
    """
    Initialize RAD_COM arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
        lm: Number of vertical layers
    
    Returns:
        A dictionary containing initialized RAD_COM arrays
    """
    lm_total = lm + LM_REQ
    
    # Initialize radiation arrays
    rqt = jnp.zeros((im, jm, LM_REQ), dtype=jnp.float32)
    tchg = jnp.zeros((im, jm, lm_total), dtype=jnp.float32)
    srhr = jnp.zeros((im, jm, lm_total + 1), dtype=jnp.float32)
    trhr = jnp.zeros((im, jm, lm_total + 1), dtype=jnp.float32)
    trsurf = jnp.zeros((im, jm, lm_total + 1), dtype=jnp.float32)
    fsf = jnp.zeros((im, jm, 4), dtype=jnp.float32)
    fsrdir = jnp.zeros((im, jm), dtype=jnp.float32)
    dirvis = jnp.zeros((im, jm), dtype=jnp.float32)
    srvisurf = jnp.zeros((im, jm), dtype=jnp.float32)
    srdn = jnp.zeros((im, jm), dtype=jnp.float32)
    fsrdif = jnp.zeros((im, jm), dtype=jnp.float32)
    dirnir = jnp.zeros((im, jm), dtype=jnp.float32)
    difnir = jnp.zeros((im, jm), dtype=jnp.float32)
    srnflb_save = jnp.zeros((im, jm, lm_total + 1), dtype=jnp.float32)
    trnflb_save = jnp.zeros((im, jm, lm_total + 1), dtype=jnp.float32)
    tausumw = jnp.zeros((im, jm), dtype=jnp.float32)
    tausumi = jnp.zeros((im, jm), dtype=jnp.float32)
    
    return {
        "RQT": rqt, "TCHG": tchg, "SRHR": srhr, "TRHR": trhr,
        "TRSURF": trsurf, "FSF": fsf, "FSRDIR": fsrdir,
        "DIRVIS": dirvis, "SRVISSURF": srvisurf, "SRDN": srdn,
        "FSRDIF": fsrdif, "DIRNIR": dirnir, "DIFNIR": difnir,
        "SRNFLB_SAVE": srnflb_save, "TRNFLB_SAVE": trnflb_save,
        "TAUSUMW": tausumw, "TAUSUMI": tausumi,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Orbital parameters
    "NRAD", "OMEGT_DEF", "OBLIQ_DEF", "ECCN_DEF",
    "OMEGT", "OBLIQ", "ECCN",
    "VARIABLE_ORB_PAR", "ORB_PAR_YEAR_BP", "ORB_PAR",
    # Default model resolution
    "IM", "JM", "LM", "LM_REQ", "LM_TOTAL",
    # Placeholder arrays
    "RQT", "TCHG", "SRHR", "TRHR", "TRSURF",
    "FSF", "FSRDIR", "DIRVIS", "SRVISSURF", "SRDN",
    "FSRDIF", "DIRNIR", "DIFNIR",
    "SRNFLB_SAVE", "TRNFLB_SAVE", "TAUSUMW", "TAUSUMI",
    # Helper functions
    "compute_solar_forcing", "compute_solar_fluxes", "compute_thermal_fluxes",
    "initialize_rad_com",
]
