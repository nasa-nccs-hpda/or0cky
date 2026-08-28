"""
JAX Implementation of ROCKE-3D PBL_COM (Planetary Boundary Layer Common Variables)
=======================================================================================

This module provides a JAX-compatible implementation of the planetary boundary
layer common variables from ROCKE-3D's PBL_COM.f. It includes:
- Constants for PBL parameters (e.g., egcm_init_max).
- Placeholder arrays for PBL variables (e.g., roughl, dclev, egcm).
- Helper functions to compute derived quantities.

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from pbl_com_jax import initialize_pbl_com
    pbl_com = initialize_pbl_com(im, jm, lm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of grid points
IM = 72
JM = 46
LM = 40

# Number of PBL layers (from SOCPBL module)
NPBL = 8


# ============================================================================
# Constants
# ============================================================================

# Maximum initial value of egcm
EGCM_INIT_MAX = jnp.float32(0.5)


# ============================================================================
# Placeholder Arrays for PBL Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM, LM).
# Users should replace these with their own data.

# ROUGHL: log10(zgs / roughness length), prescribed with zgs=30 m
ROUGHL = jnp.zeros((IM, JM), dtype=jnp.float32)

# DCLEV: Layer to which dry convection mixes (1)
DCLEV = jnp.zeros((IM, JM), dtype=jnp.float32)

# PBLHT: Boundary layer height (m)
PBLHT = jnp.zeros((IM, JM), dtype=jnp.float32)

# PBLPTOP: Pressure at the top of the PBL (mb)
PBLPTOP = jnp.zeros((IM, JM), dtype=jnp.float32)

# UGEO, VGEO: Components of geostrophic wind at the top of the BL
UGEO = jnp.zeros((IM, JM), dtype=jnp.float32)
VGEO = jnp.zeros((IM, JM), dtype=jnp.float32)

# BLDDEP: Boundary layer depth (m)
BLDDEP = jnp.zeros((IM, JM), dtype=jnp.float32)

# T1_AFTER_ATURB, U1_AFTER_ATURB, V1_AFTER_ATURB: First-layer temp/winds after ATURB
T1_AFTER_ATURB = jnp.zeros((IM, JM), dtype=jnp.float32)
U1_AFTER_ATURB = jnp.zeros((IM, JM), dtype=jnp.float32)
V1_AFTER_ATURB = jnp.zeros((IM, JM), dtype=jnp.float32)

# EGCM: 3D turbulent kinetic energy in the whole atmosphere
EGCM = jnp.zeros((IM, JM, LM), dtype=jnp.float32)

# W2GCM: Vertical component of EGCM
W2GCM = jnp.zeros((IM, JM, LM), dtype=jnp.float32)

# T2GCM: 3D turbulent temperature variance in the whole atmosphere
T2GCM = jnp.zeros((IM, JM, LM), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_pbl_height(
    bldep: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute PBL height from boundary layer depth.
    
    Args:
        bldep: Boundary layer depth (m) (I, J)
    
    Returns:
        pblht: Boundary layer height (m) (I, J)
    """
    return bldep


@jit
def compute_geostrophic_wind(
    u: jnp.ndarray,
    v: jnp.ndarray,
    pblht: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute geostrophic wind components at the top of the PBL.
    
    Args:
        u: Zonal wind (m/s) (I, J, L)
        v: Meridional wind (m/s) (I, J, L)
        pblht: Boundary layer height (m) (I, J)
    
    Returns:
        ugeo: Zonal geostrophic wind (m/s) (I, J)
        vgeo: Meridional geostrophic wind (m/s) (I, J)
    """
    # For simplicity, assume geostrophic wind = wind at the top of the PBL
    # This is a placeholder; the actual computation depends on the model
    ugeo = u[:, :, 0]  # Wind at the first layer
    vgeo = v[:, :, 0]
    return ugeo, vgeo


@jit
def compute_tke(
    egcm: jnp.ndarray,
    w2gcm: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute turbulent kinetic energy (TKE) from EGCM and W2GCM.
    
    Args:
        egcm: 3D turbulent kinetic energy (m^2/s^2) (I, J, L)
        w2gcm: Vertical component of TKE (m^2/s^2) (I, J, L)
    
    Returns:
        tke: Total turbulent kinetic energy (m^2/s^2) (I, J, L)
    """
    # Total TKE = EGCM + W2GCM
    tke = egcm + w2gcm
    return tke


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_pbl_com(
    im: int = IM,
    jm: int = JM,
    lm: int = LM,
) -> dict:
    """
    Initialize PBL_COM arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
        lm: Number of vertical layers
    
    Returns:
        A dictionary containing initialized PBL_COM arrays
    """
    # Initialize PBL arrays
    roughl = jnp.zeros((im, jm), dtype=jnp.float32)
    dclev = jnp.zeros((im, jm), dtype=jnp.float32)
    pblht = jnp.zeros((im, jm), dtype=jnp.float32)
    pblptop = jnp.zeros((im, jm), dtype=jnp.float32)
    ugeo = jnp.zeros((im, jm), dtype=jnp.float32)
    vgeo = jnp.zeros((im, jm), dtype=jnp.float32)
    bldep = jnp.zeros((im, jm), dtype=jnp.float32)
    t1_after_aturb = jnp.zeros((im, jm), dtype=jnp.float32)
    u1_after_aturb = jnp.zeros((im, jm), dtype=jnp.float32)
    v1_after_aturb = jnp.zeros((im, jm), dtype=jnp.float32)
    egcm = jnp.zeros((im, jm, lm), dtype=jnp.float32)
    w2gcm = jnp.zeros((im, jm, lm), dtype=jnp.float32)
    t2gcm = jnp.zeros((im, jm, lm), dtype=jnp.float32)
    
    return {
        "ROUGHL": roughl, "DCLEV": dclev, "PBLHT": pblht,
        "PBLPTOP": pblptop, "UGEO": ugeo, "VGEO": vgeo,
        "BLDDEP": bldep,
        "T1_AFTER_ATURB": t1_after_aturb,
        "U1_AFTER_ATURB": u1_after_aturb,
        "V1_AFTER_ATURB": v1_after_aturb,
        "EGCM": egcm, "W2GCM": w2gcm, "T2GCM": t2gcm,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Default model resolution
    "IM", "JM", "LM", "NPBL",
    # Constants
    "EGCM_INIT_MAX",
    # Placeholder arrays
    "ROUGHL", "DCLEV", "PBLHT", "PBLPTOP",
    "UGEO", "VGEO", "BLDDEP",
    "T1_AFTER_ATURB", "U1_AFTER_ATURB", "V1_AFTER_ATURB",
    "EGCM", "W2GCM", "T2GCM",
    # Helper functions
    "compute_pbl_height", "compute_geostrophic_wind", "compute_tke",
    "initialize_pbl_com",
]
