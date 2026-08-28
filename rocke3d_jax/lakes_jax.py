"""
JAX Implementation of ROCKE-3D LAKES (Lake Model)
==================================================

This module provides a JAX-compatible implementation of the lake model
from ROCKE-3D's LAKES.f. It includes:
- Constants for lake properties (e.g., MINMLD, HLAKE_MIN, TMAXRHO).
- Placeholder arrays for lake variables (e.g., FLAKE, MWL, GML).
- Helper functions to compute derived quantities (e.g., lake temperature, mixed layer depth).

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from lakes_jax import compute_lake_temperature, initialize_lakes
    tlake = compute_lake_temperature(gml, mwl)
    lakes = initialize_lakes(im, jm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from constant_jax import GRAV, BYGRAV, SHW, RHOW, LHM, SHI, TEENY, UNDEF, STBO


# ============================================================================
# Constants for Lake Properties
# ============================================================================

# Minimum mixed layer depth in lake [m]
MINMLD = jnp.float32(1.0)

# Minimum sill depth for lake [m]
HLAKE_MIN = jnp.float32(1.0)

# Temperature of maximum density (pure water) [C]
TMAXRHO = jnp.float32(4.0)

# Lake diffusion constant at mixed layer depth [m^2/s]
KVLAKE = jnp.float32(1e-5)

# Freezing temperature for lakes [C]
TFL = jnp.float32(0.0)

# Minimum ice thickness for lake ice [kg/m^2]
AC1LMIN = jnp.float32(0.1)
AC2LMIN = jnp.float32(0.1)

# Lead fraction for lakes
FLEADLK = jnp.float32(0.0)

# Reciprocal of solar radiation extinction depth for lake [1/m]
BYZETA = jnp.float32(1.0 / 0.35)

# River flow factor (default = 1)
RIVER_FAC = jnp.float32(1.0)

# Lake initialization flag (0=no change, 1=reset, 2=remove excess water)
INIT_FLAKE = 1

# Variable lakes flag (0=off, 1=on)
VARIABLE_LK = 0

# Maximum lake rise over sill level before spillover [m]
LAKE_RISE_MAX = jnp.float32(100.0)

# Maximum lake ice thickness [m] (water equivalent)
LAKE_ICE_MAX = jnp.float32(5.0)

# Power law lakes flag (0=off, 1=on)
POWER_LAW_LAKES = 0

# Lake volume constant for power law lakes
C_LAKE = jnp.float32(0.235)


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46


# ============================================================================
# Placeholder Arrays for Lake Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM).
# Users should replace these with their own data.

# Lake fraction [1] (I, J)
FLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Lake water mass [kg/m^2] (I, J)
MWL = jnp.zeros((IM, JM), dtype=jnp.float32)

# Lake heat content [J/m^2] (I, J)
GML = jnp.zeros((IM, JM), dtype=jnp.float32)

# Lake temperature [C] (I, J)
TLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Lake ice mass [kg/m^2] (I, J, 2)
ICELAK = jnp.zeros((IM, JM, 2), dtype=jnp.float32)

# Lake snow mass [kg/m^2] (I, J)
SNOWLAK = jnp.zeros((IM, JM), dtype=jnp.float32)

# Lake surface temperature [C] (I, J)
TSLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_lake_temperature(
    gml: jnp.ndarray,
    mwl: jnp.ndarray,
    shw: jnp.ndarray = SHW,
    tmaxrho: jnp.ndarray = TMAXRHO,
) -> jnp.ndarray:
    """
    Compute lake temperature from heat content and water mass.
    
    Args:
        gml: Lake heat content [J/m^2] (I, J)
        mwl: Lake water mass [kg/m^2] (I, J)
        shw: Specific heat of water [J/kg/K]
        tmaxrho: Temperature of maximum density [C]
    
    Returns:
        tlake: Lake temperature [C] (I, J)
    """
    # Lake temperature: T = GML / (MWL * SHW)
    tlake = jnp.where(
        mwl > 0.0,
        gml / (mwl * shw),
        tmaxrho  # Default to TMAXRHO if MWL = 0
    )
    return tlake


@jit
def compute_lake_heat_content(
    tlake: jnp.ndarray,
    mwl: jnp.ndarray,
    shw: jnp.ndarray = SHW,
) -> jnp.ndarray:
    """
    Compute lake heat content from temperature and water mass.
    
    Args:
        tlake: Lake temperature [C] (I, J)
        mwl: Lake water mass [kg/m^2] (I, J)
        shw: Specific heat of water [J/kg/K]
    
    Returns:
        gml: Lake heat content [J/m^2] (I, J)
    """
    gml = tlake * mwl * shw
    return gml


@jit
def compute_mixed_layer_depth(
    mwl: jnp.ndarray,
    flake: jnp.ndarray,
    minmld: jnp.ndarray = MINMLD,
) -> jnp.ndarray:
    """
    Compute mixed layer depth for lakes.
    
    Args:
        mwl: Lake water mass [kg/m^2] (I, J)
        flake: Lake fraction [1] (I, J)
        minmld: Minimum mixed layer depth [m]
    
    Returns:
        hlake: Mixed layer depth [m] (I, J)
    """
    # Mixed layer depth: H = MWL / (RHOW * FLAKE)
    hlake = jnp.where(
        flake > 0.0,
        jnp.maximum(mwl / (RHOW * flake), minmld),
        0.0
    )
    return hlake


@jit
def compute_lake_ice_mass(
    icelak: jnp.ndarray,
    rhow: jnp.ndarray = RHOW,
) -> jnp.ndarray:
    """
    Compute total lake ice mass from ice layers.
    
    Args:
        icelak: Lake ice mass [kg/m^2] (I, J, 2)
        rhow: Density of water [kg/m^3]
    
    Returns:
        total_ice_mass: Total lake ice mass [kg/m^2] (I, J)
    """
    total_ice_mass = jnp.sum(icelak, axis=-1)
    return total_ice_mass


# ============================================================================
# Core Lake Subroutines (Simplified)
# ============================================================================

@jit
def lkmix(
    mwl: jnp.ndarray,      # Lake water mass [kg/m^2] (I, J)
    gml: jnp.ndarray,      # Lake heat content [J/m^2] (I, J)
    flake: jnp.ndarray,    # Lake fraction [1] (I, J)
    tlake: jnp.ndarray,    # Lake temperature [C] (I, J)
    kvlake: float = KVLAKE,  # Lake diffusion constant [m^2/s]
    dt: float = 3600.0,    # Time step [s]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of LKMIX: Lake mixing (turbulent diffusion).
    
    Args:
        mwl: Lake water mass [kg/m^2].
        gml: Lake heat content [J/m^2].
        flake: Lake fraction [1].
        tlake: Lake temperature [C].
        kvlake: Lake diffusion constant [m^2/s].
        dt: Time step [s].
    
    Returns:
        Updated mwl, gml, tlake.
    """
    # For simplicity, assume no mixing (placeholder)
    # In a full implementation, this would compute turbulent diffusion
    # and update the lake temperature and heat content.
    
    return mwl, gml, tlake


@jit
def precip_lk(
    mwl: jnp.ndarray,      # Lake water mass [kg/m^2] (I, J)
    gml: jnp.ndarray,      # Lake heat content [J/m^2] (I, J)
    flake: jnp.ndarray,    # Lake fraction [1] (I, J)
    tlake: jnp.ndarray,    # Lake temperature [C] (I, J)
    prcp: jnp.ndarray,      # Precipitation [kg/m^2] (I, J)
    enrgp: jnp.ndarray,    # Energy of precipitation [J/m^2] (I, J)
    tfl: float = TFL,      # Freezing temperature for lakes [C]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of PRECIP_LK: Precipitation onto lakes.
    
    Args:
        mwl: Lake water mass [kg/m^2].
        gml: Lake heat content [J/m^2].
        flake: Lake fraction [1].
        tlake: Lake temperature [C].
        prcp: Precipitation [kg/m^2].
        enrgp: Energy of precipitation [J/m^2].
        tfl: Freezing temperature for lakes [C].
    
    Returns:
        Updated mwl, gml, tlake.
    """
    # Add precipitation to lake water mass
    mwl_new = mwl + prcp * flake
    
    # Add precipitation energy to lake heat content
    gml_new = gml + enrgp * flake
    
    # Update lake temperature (simplified)
    tlake_new = jnp.where(
        mwl_new > 0.0,
        gml_new / (mwl_new * SHW),
        tlake  # Keep old temperature if MWL = 0
    )
    
    return mwl_new, gml_new, tlake_new


@jit
def ground_lk(
    mwl: jnp.ndarray,      # Lake water mass [kg/m^2] (I, J)
    gml: jnp.ndarray,      # Lake heat content [J/m^2] (I, J)
    flake: jnp.ndarray,    # Lake fraction [1] (I, J)
    tlake: jnp.ndarray,    # Lake temperature [C] (I, J)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation [W/m^2] (I, J)
    flong: jnp.ndarray,    # Downwelling longwave radiation [W/m^2] (I, J)
    albedo: jnp.ndarray,   # Surface albedo [1] (I, J)
    dt: float = 3600.0,    # Time step [s]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simplified version of GROUND_LK: Ground energy balance for lakes.
    
    Args:
        mwl: Lake water mass [kg/m^2].
        gml: Lake heat content [J/m^2].
        flake: Lake fraction [1].
        tlake: Lake temperature [C].
        fsf: Downwelling shortwave radiation [W/m^2].
        flong: Downwelling longwave radiation [W/m^2].
        albedo: Surface albedo [1].
        dt: Time step [s].
    
    Returns:
        Updated mwl, gml, tlake.
    """
    # Net shortwave radiation absorbed by lake
    sw_absorbed = fsf * (1.0 - albedo) * flake
    
    # Net longwave radiation (simplified)
    lw_net = flong * flake - STBO * (tlake + 273.15) ** 4 * flake
    
    # Total energy flux into lake [W/m^2]
    energy_flux = sw_absorbed + lw_net
    
    # Update lake heat content [J/m^2]
    gml_new = gml + energy_flux * dt
    
    # Update lake temperature [C]
    tlake_new = jnp.where(
        mwl > 0.0,
        gml_new / (mwl * SHW),
        tlake  # Keep old temperature if MWL = 0
    )
    
    # Lake water mass is unchanged (no evaporation/condensation for simplicity)
    mwl_new = mwl
    
    return mwl_new, gml_new, tlake_new


@jit
def minmld(
    mwl: jnp.ndarray,      # Lake water mass [kg/m^2] (I, J)
    flake: jnp.ndarray,    # Lake fraction [1] (I, J)
) -> jnp.ndarray:
    """
    Compute minimum mixed layer depth for lakes.
    
    Args:
        mwl: Lake water mass [kg/m^2].
        flake: Lake fraction [1].
    
    Returns:
        min_mld: Minimum mixed layer depth [m] (I, J)
    """
    min_mld = jnp.where(
        flake > 0.0,
        jnp.maximum(mwl / (RHOW * flake), MINMLD),
        0.0
    )
    return min_mld


@jit
def compute_lake_ice_mass(
    icelak: jnp.ndarray,
    rhow: jnp.ndarray = RHOW,
) -> jnp.ndarray:
    """
    Compute total lake ice mass from ice layers.
    
    Args:
        icelak: Lake ice mass [kg/m^2] (I, J, 2)
        rhow: Density of water [kg/m^3]
    
    Returns:
        total_ice_mass: Total lake ice mass [kg/m^2] (I, J)
    """
    total_ice_mass = jnp.sum(icelak, axis=-1)
    return total_ice_mass


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_lakes(
    im: int = IM,
    jm: int = JM,
) -> dict:
    """
    Initialize LAKES arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        A dictionary containing initialized LAKES arrays
    """
    # Initialize lake arrays
    flake = jnp.zeros((im, jm), dtype=jnp.float32)
    mwl = jnp.zeros((im, jm), dtype=jnp.float32)
    gml = jnp.zeros((im, jm), dtype=jnp.float32)
    tlake = jnp.zeros((im, jm), dtype=jnp.float32)
    icelak = jnp.zeros((im, jm, 2), dtype=jnp.float32)
    snowlak = jnp.zeros((im, jm), dtype=jnp.float32)
    tslake = jnp.zeros((im, jm), dtype=jnp.float32)
    
    return {
        "FLAKE": flake, "MWL": mwl, "GML": gml,
        "TLAKE": tlake, "ICELAK": icelak, "SNOWLAK": snowlak,
        "TSLAKE": tslake,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Constants
    "MINMLD", "HLAKE_MIN", "TMAXRHO", "KVLAKE", "TFL",
    "AC1LMIN", "AC2LMIN", "FLEADLK", "BYZETA",
    "RIVER_FAC", "INIT_FLAKE", "VARIABLE_LK",
    "LAKE_RISE_MAX", "LAKE_ICE_MAX", "POWER_LAW_LAKES", "C_LAKE",
    # Default model resolution
    "IM", "JM",
    # Placeholder arrays
    "FLAKE", "MWL", "GML", "TLAKE", "ICELAK", "SNOWLAK", "TSLAKE",
    # Helper functions
    "compute_lake_temperature", "compute_lake_heat_content",
    "compute_mixed_layer_depth", "compute_lake_ice_mass",
    "initialize_lakes",
]
