"""
JAX Implementation of ROCKE-3D SOMTQ_COM (Second-Order Moments for Temperature and Moisture)
==================================================================================================

This module provides a JAX-compatible implementation of the second-order moments
for temperature and moisture from ROCKE-3D's SOMTQ_COM.f. It includes:
- Constants for moment indices (e.g., NMOM, MX, MY, MZ).
- Placeholder arrays for temperature and moisture moments (TMOM, QMOM).
- Helper functions to compute derived quantities.

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from somtq_com_jax import compute_tmom_qmom, initialize_somtq_com
    tmom, qmom = compute_tmom_qmom(t, q, mz)
    somtq_com = initialize_somtq_com(nmom, im, jm, lm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple


# ============================================================================
# Constants for Moment Indices (from QUSDEF)
# ============================================================================

# Number of moments
NMOM = 9

# Moment indices
MX = 1   # x-component
MY = 2   # y-component
MZ = 3   # z-component
MXX = 4  # xx-component
MYY = 5  # yy-component
MZZ = 6  # zz-component
MXY = 7  # xy-component
MYZ = 8  # yz-component
MZX = 9  # zx-component

# Moments with a vertical component
ZMOM = jnp.array([MZ, MZZ, MYZ, MZX], dtype=jnp.int32)

# Moments with no vertical component
XYMOM = jnp.array([MX, MY, MXX, MXY, MYY], dtype=jnp.int32)

# Moments with a horizontal component
IHMOM = jnp.array([MX, MY, MXX, MYY, MXY, MYZ, MZX], dtype=jnp.int32)

# Moments with no horizontal component
ZOMOM = jnp.array([MZ, MZZ], dtype=jnp.int32)

# Direction switches
XDIR = jnp.array([MX, MY, MZ, MXX, MYY, MZZ, MXY, MYZ, MZX], dtype=jnp.int32)
YDIR = jnp.array([MY, MX, MZ, MYY, MXX, MZZ, MXY, MZX, MYZ], dtype=jnp.int32)
ZDIR = jnp.array([MZ, MY, MX, MZZ, MYY, MXX, MYZ, MXY, MZX], dtype=jnp.int32)

# Prather limits flag (default = 0)
PRATHER_LIMITS = 0

# Flux flags
FLUX_NEGATIVE = -1
FLUX_NONNEGATIVE = 1


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46
LM = 40


# ============================================================================
# Placeholder Arrays for Temperature and Moisture Moments
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (NMOM, IM, JM, LM).
# Users should replace these with their own data.

# Temperature moments [K/m] (NMOM, I, J, L)
TMOM = jnp.zeros((NMOM, IM, JM, LM), dtype=jnp.float32)

# Moisture moments [kg/kg/m] (NMOM, I, J, L)
QMOM = jnp.zeros((NMOM, IM, JM, LM), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_tmom_qmom(
    t: jnp.ndarray,
    q: jnp.ndarray,
    mz: int = MZ,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute temperature and moisture moments (TMOM, QMOM).
    
    Args:
        t: Temperature [K] (I, J, L)
        q: Specific humidity [kg/kg] (I, J, L)
        mz: Index for vertical moment (default = MZ)
    
    Returns:
        tmom: Temperature moments [K/m] (NMOM, I, J, L)
        qmom: Moisture moments [kg/kg/m] (NMOM, I, J, L)
    """
    # For simplicity, assume all moments are zero except MZ
    # This is a placeholder; the actual computation depends on the model
    tmom = jnp.zeros((NMOM, t.shape[0], t.shape[1], t.shape[2]), dtype=jnp.float32)
    qmom = jnp.zeros((NMOM, q.shape[0], q.shape[1], q.shape[2]), dtype=jnp.float32)
    
    # Set MZ moment to temperature and moisture tendencies
    tmom = tmom.at[mz - 1, :, :, :].set(t)
    qmom = qmom.at[mz - 1, :, :, :].set(q)
    
    return tmom, qmom


@jit
def compute_vertical_moment(
    tmom: jnp.ndarray,
    qmom: jnp.ndarray,
    mz: int = MZ,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract vertical moments from TMOM and QMOM.
    
    Args:
        tmom: Temperature moments [K/m] (NMOM, I, J, L)
        qmom: Moisture moments [kg/kg/m] (NMOM, I, J, L)
        mz: Index for vertical moment (default = MZ)
    
    Returns:
        t_mz: Vertical temperature moment [K/m] (I, J, L)
        q_mz: Vertical moisture moment [kg/kg/m] (I, J, L)
    """
    t_mz = tmom[mz - 1, :, :, :]
    q_mz = qmom[mz - 1, :, :, :]
    return t_mz, q_mz


@jit
def compute_horizontal_moments(
    tmom: jnp.ndarray,
    qmom: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extract horizontal moments from TMOM and QMOM.
    
    Args:
        tmom: Temperature moments [K/m] (NMOM, I, J, L)
        qmom: Moisture moments [kg/kg/m] (NMOM, I, J, L)
    
    Returns:
        t_mx: X-component temperature moment [K/m] (I, J, L)
        t_my: Y-component temperature moment [K/m] (I, J, L)
        q_mx: X-component moisture moment [kg/kg/m] (I, J, L)
        q_my: Y-component moisture moment [kg/kg/m] (I, J, L)
    """
    t_mx = tmom[MX - 1, :, :, :]
    t_my = tmom[MY - 1, :, :, :]
    q_mx = qmom[MX - 1, :, :, :]
    q_my = qmom[MY - 1, :, :, :]
    return t_mx, t_my, q_mx, q_my


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_somtq_com(
    nmom: int = NMOM,
    im: int = IM,
    jm: int = JM,
    lm: int = LM,
) -> dict:
    """
    Initialize SOMTQ_COM arrays with the specified grid size.
    
    Args:
        nmom: Number of moments
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
        lm: Number of vertical layers
    
    Returns:
        A dictionary containing initialized SOMTQ_COM arrays
    """
    # Initialize temperature and moisture moments
    tmom = jnp.zeros((nmom, im, jm, lm), dtype=jnp.float32)
    qmom = jnp.zeros((nmom, im, jm, lm), dtype=jnp.float32)
    
    return {
        "TMOM": tmom,
        "QMOM": qmom,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Constants
    "NMOM", "MX", "MY", "MZ", "MXX", "MYY", "MZZ", "MXY", "MYZ", "MZX",
    "ZMOM", "XYMOM", "IHMOM", "ZOMOM",
    "XDIR", "YDIR", "ZDIR",
    "PRATHER_LIMITS", "FLUX_NEGATIVE", "FLUX_NONNEGATIVE",
    # Default model resolution
    "IM", "JM", "LM",
    # Placeholder arrays
    "TMOM", "QMOM",
    # Helper functions
    "compute_tmom_qmom", "compute_vertical_moment", "compute_horizontal_moments",
    "initialize_somtq_com",
]
