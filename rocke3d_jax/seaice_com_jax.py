"""
JAX Implementation of ROCKE-3D SEAICE_COM (Sea Ice Common Variables)
============================================================================

This module provides a JAX-compatible implementation of the sea ice common
variables from ROCKE-3D's SEAICE_COM.f. It includes:
- Constants for sea ice common variables (e.g., LMI).
- Placeholder arrays for sea ice common variables (e.g., FWATER, RSI, SNOWI).
- Helper functions to compute derived quantities.

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from seaice_com_jax import compute_sea_ice_fraction, initialize_seaice_com
    rsi = compute_sea_ice_fraction(fwater, si)
    seaice_com = initialize_seaice_com(im, jm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from seaice_jax import LMI


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46


# ============================================================================
# Placeholder Arrays for Sea Ice Common Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM, LMI).
# Users should replace these with their own data.

# Water fraction of grid box [1] (I, J)
FWATER = jnp.zeros((IM, JM), dtype=jnp.float32)

# Sea ice concentration [1] (I, J)
RSI = jnp.zeros((IM, JM), dtype=jnp.float32)

# Snow amount on sea ice [kg/m^2] (I, J)
SNOWI = jnp.zeros((IM, JM), dtype=jnp.float32)

# Mass of ice second layer [kg/m^2] (I, J)
# Note: MSI includes the mass of salt in sea ice
MSI = jnp.zeros((IM, JM), dtype=jnp.float32)

# Total ice thickness [m] (I, J)
ZSI = jnp.zeros((IM, JM), dtype=jnp.float32)

# Melt pond mass [kg/m^2] (I, J)
POND_MELT = jnp.zeros((IM, JM), dtype=jnp.float32)

# First-order moments of ice concentration (for advection) (I, J)
RSIX = jnp.zeros((IM, JM), dtype=jnp.float32)
RSIY = jnp.zeros((IM, JM), dtype=jnp.float32)

# Saved value of sea ice concentration before DYNSI [1] (I, J)
RSISAVE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Enthalpy of each ice layer [J/m^2] (I, J, LMI)
HSI = jnp.zeros((IM, JM, LMI), dtype=jnp.float32)

# Sea ice salt content [kg/m^2] (I, J, LMI)
SSI = jnp.zeros((IM, JM, LMI), dtype=jnp.float32)

# Flag for wet snow on ice [1] (I, J)
FLAG_DSWS = jnp.zeros((IM, JM), dtype=jnp.bool_)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_sea_ice_fraction(
    fwater: jnp.ndarray,
    rsi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute sea ice fraction of the grid box.
    
    Args:
        fwater: Water fraction of grid box [1] (I, J)
        rsi: Sea ice concentration [1] (I, J)
    
    Returns:
        sea_ice_fraction: Sea ice fraction of grid box [1] (I, J)
    """
    sea_ice_fraction = fwater * rsi
    return sea_ice_fraction


@jit
def compute_total_ice_thickness(
    zsi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute total ice thickness (placeholder for now).
    
    Args:
        zsi: Total ice thickness [m] (I, J)
    
    Returns:
        total_thickness: Total ice thickness [m] (I, J)
    """
    return zsi


@jit
def compute_ice_mass(
    zsi: jnp.ndarray,
    rsi: jnp.ndarray,
    rhoi: jnp.ndarray = 916.6,
) -> jnp.ndarray:
    """
    Compute ice mass from thickness and concentration.
    
    Args:
        zsi: Total ice thickness [m] (I, J)
        rsi: Sea ice concentration [1] (I, J)
        rhoi: Density of ice [kg/m^3]
    
    Returns:
        msi: Ice mass [kg/m^2] (I, J)
    """
    msi = zsi * rsi * rhoi
    return msi


@jit
def compute_snow_mass(
    snowi: jnp.ndarray,
    rsi: jnp.ndarray,
    rhos: jnp.ndarray = 300.0,
) -> jnp.ndarray:
    """
    Compute snow mass from thickness and concentration.
    
    Args:
        snowi: Snow amount on sea ice [kg/m^2] (I, J)
        rsi: Sea ice concentration [1] (I, J)
        rhos: Density of snow [kg/m^3]
    
    Returns:
        snow_mass: Snow mass [kg/m^2] (I, J)
    """
    # For simplicity, assume snowi is already in kg/m^2
    snow_mass = snowi
    return snow_mass


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_seaice_com(
    im: int = IM,
    jm: int = JM,
) -> dict:
    """
    Initialize SEAICE_COM arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        A dictionary containing initialized SEAICE_COM arrays
    """
    # Initialize sea ice common arrays
    fwater = jnp.zeros((im, jm), dtype=jnp.float32)
    rsi = jnp.zeros((im, jm), dtype=jnp.float32)
    snowi = jnp.zeros((im, jm), dtype=jnp.float32)
    msi = jnp.zeros((im, jm), dtype=jnp.float32)
    zsi = jnp.zeros((im, jm), dtype=jnp.float32)
    pond_melt = jnp.zeros((im, jm), dtype=jnp.float32)
    rsix = jnp.zeros((im, jm), dtype=jnp.float32)
    rsiy = jnp.zeros((im, jm), dtype=jnp.float32)
    rsisave = jnp.zeros((im, jm), dtype=jnp.float32)
    hsi = jnp.zeros((im, jm, LMI), dtype=jnp.float32)
    ssi = jnp.zeros((im, jm, LMI), dtype=jnp.float32)
    flag_dsws = jnp.zeros((im, jm), dtype=jnp.bool_)
    
    return {
        "FWATER": fwater, "RSI": rsi, "SNOWI": snowi,
        "MSI": msi, "ZSI": zsi, "POND_MELT": pond_melt,
        "RSIX": rsix, "RSIY": rsiy, "RSISAVE": rsisave,
        "HSI": hsi, "SSI": ssi, "FLAG_DSWS": flag_dsws,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Default model resolution
    "IM", "JM",
    # Placeholder arrays
    "FWATER", "RSI", "SNOWI", "MSI", "ZSI",
    "POND_MELT", "RSIX", "RSIY", "RSISAVE",
    "HSI", "SSI", "FLAG_DSWS",
    # Helper functions
    "compute_sea_ice_fraction", "compute_total_ice_thickness",
    "compute_ice_mass", "compute_snow_mass",
    "initialize_seaice_com",
]
