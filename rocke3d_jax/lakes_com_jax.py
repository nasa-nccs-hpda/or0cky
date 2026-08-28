"""
JAX Implementation of ROCKE-3D LAKES_COM (Lake Common Variables)
====================================================================

This module provides a JAX-compatible implementation of the lake common
variables from ROCKE-3D's LAKES_COM.f. It includes:
- Constants for lake common variables (e.g., NRVRMX).
- Placeholder arrays for lake common variables (e.g., MWL, GML, TLAKE).
- Helper functions to compute derived quantities.

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from lakes_com_jax import compute_lake_enthalpy, initialize_lakes_com
    glake = compute_lake_enthalpy(tlake, mwl)
    lakes_com = initialize_lakes_com(im, jm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from constant_jax import SHW, RHOW


# ============================================================================
# Constants for Lake Common Variables
# ============================================================================

# Maximum number of named rivers
NRVRMX = 42

# Actual number of named rivers (can be updated)
NRVR = 0

# Indexes for named river mouths
IRVRMTH = jnp.zeros(NRVRMX, dtype=jnp.int32)
JRVRMTH = jnp.zeros(NRVRMX, dtype=jnp.int32)

# Named rivers
NAMERVR = ["" for _ in range(NRVRMX)]

# Discharges from named rivers
RVROUT = jnp.zeros(NRVRMX, dtype=jnp.float32)


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46


# ============================================================================
# Placeholder Arrays for Lake Common Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM).
# Users should replace these with their own data.

# Lake water mass [kg/m^2] (I, J)
MWL = jnp.zeros((IM, JM), dtype=jnp.float32)

# Total enthalpy of lake [J/m^2] (I, J)
GML = jnp.zeros((IM, JM), dtype=jnp.float32)

# Lake temperature [C] (I, J)
TLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Mixed layer depth in lake [m] (I, J)
MDLK = jnp.zeros((IM, JM), dtype=jnp.float32)

# Variable lake fraction [1] (I, J)
FLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Tan(alpha) = slope for conical lake [1] (I, J)
TANLK = jnp.zeros((IM, JM), dtype=jnp.float32)

# Previous lake fraction [1] (I, J)
SVFLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Bottom temperature of lake layer 2 [C] (I, J)
T2LBOT = jnp.zeros((IM, JM), dtype=jnp.float32)

# Turbulent Kinetic Energy released by SURFACE [J/m^2] (I, J)
EKT = jnp.zeros((IM, JM), dtype=jnp.float32)

# Mean depth of lake [m] (I, J)
DLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Like GML but per unit area of lake [J/m^2] (I, J)
GLAKE = jnp.zeros((IM, JM), dtype=jnp.float32)

# Sill depth = lake depth if lake is at sill level [m] (I, J)
DLAKE0 = jnp.zeros((IM, JM), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_lake_enthalpy(
    tlake: jnp.ndarray,
    mwl: jnp.ndarray,
    shw: jnp.ndarray = SHW,
) -> jnp.ndarray:
    """
    Compute lake enthalpy from temperature and water mass.
    
    Args:
        tlake: Lake temperature [C] (I, J)
        mwl: Lake water mass [kg/m^2] (I, J)
        shw: Specific heat of water [J/kg/K]
    
    Returns:
        gml: Lake enthalpy [J/m^2] (I, J)
    """
    gml = tlake * mwl * shw
    return gml


@jit
def compute_lake_depth(
    mwl: jnp.ndarray,
    flake: jnp.ndarray,
    rhow: jnp.ndarray = RHOW,
) -> jnp.ndarray:
    """
    Compute lake depth from water mass and lake fraction.
    
    Args:
        mwl: Lake water mass [kg/m^2] (I, J)
        flake: Lake fraction [1] (I, J)
        rhow: Density of water [kg/m^3]
    
    Returns:
        dlake: Lake depth [m] (I, J)
    """
    dlake = jnp.where(
        flake > 0.0,
        mwl / (rhow * flake),
        0.0
    )
    return dlake


@jit
def compute_lake_glake(
    gml: jnp.ndarray,
    flake: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute GLAKE (enthalpy per unit area of lake).
    
    Args:
        gml: Lake enthalpy [J/m^2] (I, J)
        flake: Lake fraction [1] (I, J)
    
    Returns:
        glake: Enthalpy per unit area of lake [J/m^2] (I, J)
    """
    glake = jnp.where(
        flake > 0.0,
        gml / flake,
        0.0
    )
    return glake


@jit
def compute_mixed_layer_depth(
    mwl: jnp.ndarray,
    flake: jnp.ndarray,
    rhow: jnp.ndarray = RHOW,
) -> jnp.ndarray:
    """
    Compute mixed layer depth for lakes.
    
    Args:
        mwl: Lake water mass [kg/m^2] (I, J)
        flake: Lake fraction [1] (I, J)
        rhow: Density of water [kg/m^3]
    
    Returns:
        mldlk: Mixed layer depth [m] (I, J)
    """
    mldlk = jnp.where(
        flake > 0.0,
        mwl / (rhow * flake),
        0.0
    )
    return mldlk


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_lakes_com(
    im: int = IM,
    jm: int = JM,
) -> dict:
    """
    Initialize LAKES_COM arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        A dictionary containing initialized LAKES_COM arrays
    """
    # Initialize lake common arrays
    mwl = jnp.zeros((im, jm), dtype=jnp.float32)
    gml = jnp.zeros((im, jm), dtype=jnp.float32)
    tlake = jnp.zeros((im, jm), dtype=jnp.float32)
    mldlk = jnp.zeros((im, jm), dtype=jnp.float32)
    flake = jnp.zeros((im, jm), dtype=jnp.float32)
    tanlk = jnp.zeros((im, jm), dtype=jnp.float32)
    svflake = jnp.zeros((im, jm), dtype=jnp.float32)
    t2lbot = jnp.zeros((im, jm), dtype=jnp.float32)
    ekt = jnp.zeros((im, jm), dtype=jnp.float32)
    dlake = jnp.zeros((im, jm), dtype=jnp.float32)
    glake = jnp.zeros((im, jm), dtype=jnp.float32)
    dlake0 = jnp.zeros((im, jm), dtype=jnp.float32)
    
    return {
        "MWL": mwl, "GML": gml, "TLAKE": tlake,
        "MDLK": mldlk, "FLAKE": flake, "TANLK": tanlk,
        "SVFLAKE": svflake, "T2LBOT": t2lbot, "EKT": ekt,
        "DLAKE": dlake, "GLAKE": glake, "DLAKE0": dlake0,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Constants
    "NRVRMX", "NRVR", "IRVRMTH", "JRVRMTH", "NAMERVR", "RVROUT",
    # Default model resolution
    "IM", "JM",
    # Placeholder arrays
    "MWL", "GML", "TLAKE", "MDLK", "FLAKE",
    "TANLK", "SVFLAKE", "T2LBOT", "EKT",
    "DLAKE", "GLAKE", "DLAKE0",
    # Helper functions
    "compute_lake_enthalpy", "compute_lake_depth",
    "compute_lake_glake", "compute_mixed_layer_depth",
    "initialize_lakes_com",
]
