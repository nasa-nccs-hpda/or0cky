"""
JAX Implementation of ROCKE-3D ATM_COM (Atmospheric Common Variables)
========================================================================

This module provides a JAX-compatible implementation of the atmospheric common
variables from ROCKE-3D's ATM_COM.f. It includes:
- Constants for model resolution (e.g., LM_REQ, REQ_FAC).
- Placeholder arrays for atmospheric variables (e.g., MA, U, V, T, Q).
- Helper functions to compute derived quantities (e.g., byMA, PK, PEK).

Key Features:
- All arrays are JAX-compatible (no dynamic allocation).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from atm_com_jax import compute_pk_pek, compute_byMA
    pk, pek = compute_pk_pek(pmid, kapa)
    byMA = compute_byMA(MA)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from constant_jax import KAPA


# ============================================================================
# Model Resolution Constants
# ============================================================================

# Number of extra radiative equilibrium layers
LM_REQ = 3

# Factors for REQ layer pressures
REQ_FAC = jnp.array([0.5, 0.2], dtype=jnp.float32)  # Edge
REQ_FAC_M = jnp.array([0.75, 0.35, 0.1], dtype=jnp.float32)  # Mid-points
REQ_FAC_D = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32)  # Delta


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of vertical layers (LM)
LM = 40

# Total number of layers (LM + LM_REQ)
LM_TOTAL = LM + LM_REQ


# ============================================================================
# Placeholder Arrays for Atmospheric Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (LM_TOTAL).
# Users should replace these with their own data.

# Initialize placeholder arrays with zeros
# Shape: (LM_TOTAL,) for 1D arrays, (I, J, LM_TOTAL) for 3D arrays
# For simplicity, we assume a default grid size (I=1, J=1)
I_DEFAULT = 1
J_DEFAULT = 1

# Main atmospheric prognostic variables
MA = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # Air mass (kg/m^2)
MAOLD = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # MA before dynamics
U = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # East-west velocity (m/s)
V = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # North-south velocity (m/s)
T = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # Potential temperature (K)
Q = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # Specific humidity (kg/kg)
QCL = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # Cloud liquid water (kg/kg)
QCI = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)  # Cloud ice water (kg/kg)

# Surface pressure (hecto-Pascals)
P = jnp.zeros((I_DEFAULT, J_DEFAULT), dtype=jnp.float32)

# Surface elevation (m)
ZATMO = jnp.zeros((I_DEFAULT, J_DEFAULT), dtype=jnp.float32)


# ============================================================================
# Derived Quantities
# ============================================================================

# Pressure at mid-point of box (mb)
PMID = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)

# Pressure at lower edge of box (mb)
PEDN = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL + 1), dtype=jnp.float32)

# PK = PMID**KAPA
PK = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)

# PEK = PEDN**KAPA
PEK = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL + 1), dtype=jnp.float32)

# byMA = 1/MA (m^2/kg)
BYMA = jnp.zeros((I_DEFAULT, J_DEFAULT, LM_TOTAL), dtype=jnp.float32)


# ============================================================================
# Helper Functions
# ============================================================================

@jit
def compute_byMA(MA: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the reciprocal of air mass (byMA = 1/MA).
    
    Args:
        MA: Air mass (kg/m^2) (I, J, L)
    
    Returns:
        byMA: Reciprocal of air mass (m^2/kg) (I, J, L)
    """
    return 1.0 / MA


@jit
def compute_pk_pek(
    pmid: jnp.ndarray,
    pedn: jnp.ndarray,
    kapa: jnp.ndarray = KAPA,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute PK = PMID**KAPA and PEK = PEDN**KAPA.
    
    Args:
        pmid: Pressure at mid-point of box (mb) (I, J, L)
        pedn: Pressure at lower edge of box (mb) (I, J, L+1)
        kapa: Ideal gas law exponent (dimensionless)
    
    Returns:
        pk: PMID**KAPA (I, J, L)
        pek: PEDN**KAPA (I, J, L+1)
    """
    pk = pmid ** kapa
    pek = pedn ** kapa
    return pk, pek


@jit
def compute_pmid_pedn(
    p: jnp.ndarray,
    ps: jnp.ndarray,
    dsig: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute PMID (mid-point pressure) and PEDN (edge pressure).
    
    Args:
        p: Surface pressure (mb) (I, J)
        ps: Surface pressure (mb) (I, J)
        dsig: Pressure thickness (mb) (I, J, L)
    
    Returns:
        pmid: Pressure at mid-point of box (mb) (I, J, L)
        pedn: Pressure at lower edge of box (mb) (I, J, L+1)
    """
    # Compute PMID (mid-point pressure)
    # PMID(L) = P - 0.5 * DSIG(L)
    # For simplicity, assume P is the surface pressure and DSIG is the layer thickness
    # This is a placeholder; the actual computation depends on the model setup
    pmid = p[:, :, jnp.newaxis] - 0.5 * dsig
    
    # Compute PEDN (edge pressure)
    # PEDN(1) = P (surface pressure)
    # PEDN(L+1) = PEDN(L) - DSIG(L)
    pedn = jnp.zeros((p.shape[0], p.shape[1], dsig.shape[2] + 1), dtype=jnp.float32)
    pedn = pedn.at[:, :, 0].set(p)
    for l in range(dsig.shape[2]):
        pedn = pedn.at[:, :, l + 1].set(pedn[:, :, l] - dsig[:, :, l])
    
    return pmid, pedn


@jit
def compute_pdsig(
    p: jnp.ndarray,
    dsig: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute PDSIG = P * DSIG (surface pressure * pressure thickness).
    
    Args:
        p: Surface pressure (mb) (I, J)
        dsig: Pressure thickness (mb) (I, J, L)
    
    Returns:
        pdsig: P * DSIG (mb^2) (I, J, L)
    """
    return p[:, :, jnp.newaxis] * dsig


def initialize_atm_com(
    i: int = I_DEFAULT,
    j: int = J_DEFAULT,
    lm: int = LM,
) -> dict:
    """
    Initialize ATM_COM arrays with the specified grid size.
    
    Args:
        i: Number of grid points in x-direction
        j: Number of grid points in y-direction
        lm: Number of vertical layers
    
    Returns:
        A dictionary containing initialized ATM_COM arrays
    """
    lm_total = lm + LM_REQ
    
    # Initialize main atmospheric variables
    ma = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    maold = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    u = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    v = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    t = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    q = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    qcl = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    qci = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    
    # Initialize surface variables
    p = jnp.zeros((i, j), dtype=jnp.float32)
    zatmo = jnp.zeros((i, j), dtype=jnp.float32)
    
    # Initialize derived quantities
    pmid = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    pedn = jnp.zeros((i, j, lm_total + 1), dtype=jnp.float32)
    pk = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    pek = jnp.zeros((i, j, lm_total + 1), dtype=jnp.float32)
    byma = jnp.zeros((i, j, lm_total), dtype=jnp.float32)
    
    return {
        "MA": ma, "MAOLD": maold, "U": u, "V": v, "T": t, "Q": q,
        "QCL": qcl, "QCI": qci, "P": p, "ZATMO": zatmo,
        "PMID": pmid, "PEDN": pedn, "PK": pk, "PEK": pek, "BYMA": byma,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Model resolution constants
    "LM_REQ", "REQ_FAC", "REQ_FAC_M", "REQ_FAC_D", "LM", "LM_TOTAL",
    # Placeholder arrays
    "MA", "MAOLD", "U", "V", "T", "Q", "QCL", "QCI", "P", "ZATMO",
    "PMID", "PEDN", "PK", "PEK", "BYMA",
    # Helper functions
    "compute_byMA", "compute_pk_pek", "compute_pmid_pedn", "compute_pdsig",
    "initialize_atm_com",
]
