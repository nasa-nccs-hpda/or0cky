"""
JAX Implementation of ROCKE-3D Dry Convection (DRYCNV.f)
========================================================

This module provides a JAX-based implementation of the dry convection mixing
scheme from ROCKE-3D's DRYCNV.f.

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Usage:
    from rocke3d_jax.drycnv import dry_convection_mixing_jit
    T_out, Q_out = dry_convection_mixing_jit(T, Q, PK, PDSIG)
"""

import jax
import jax.numpy as jnp
from jax import jit


from jax import lax


@jit
def dry_convection_mixing(
    T: jnp.ndarray,
    Q: jnp.ndarray,
    PK: jnp.ndarray,
    PDSIG: jnp.ndarray,
    deltx: float = 0.608,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimized JAX implementation of dry convection mixing.
    Uses jax.lax.scan with lax.dynamic_slice for dynamic layer counts.
    
    Args:
        T: Temperature (I, J, L)
        Q: Moisture (I, J, L)
        PK: Pressure (I, J, L)
        PDSIG: Layer thickness (I, J, L)
        deltx: Virtual temperature factor (default: 0.608)
    
    Returns:
        Updated T, Q after mixing unstable layers
    """
    # Initialize outputs
    T_out = jnp.copy(T)
    Q_out = jnp.copy(Q)
    TV = T_out * (1 + Q_out * deltx)

    # Precompute RDP for all layers
    RDP = 1.0 / (PDSIG[..., :-1] + PDSIG[..., 1:])

    # Define the loop body for lax.scan
    def loop_body(carry, L):
        T_curr, Q_curr, TV_curr = carry
        unstable = TV_curr[..., L] > TV_curr[..., L + 1]
        
        # Use lax.dynamic_slice to avoid dynamic slicing issues
        T_slice = lax.dynamic_slice_in_dim(T_curr, L, 2, axis=-1)
        Q_slice = lax.dynamic_slice_in_dim(Q_curr, L, 2, axis=-1)
        PK_slice = lax.dynamic_slice_in_dim(PK, L, 2, axis=-1)
        PDSIG_slice = lax.dynamic_slice_in_dim(PDSIG, L, 2, axis=-1)
        
        # Vectorized mixing for layers L and L+1
        PKMS = PK_slice[..., 0] * PDSIG_slice[..., 0] + PK_slice[..., 1] * PDSIG_slice[..., 1]
        TVMS = TV_curr[..., L] * PK_slice[..., 0] * PDSIG_slice[..., 0] + TV_curr[..., L + 1] * PK_slice[..., 1] * PDSIG_slice[..., 1]
        QMS = Q_slice[..., 0] * PDSIG_slice[..., 0] + Q_slice[..., 1] * PDSIG_slice[..., 1]
        
        # Ensure RDP[L] is broadcast correctly
        RDP_L = RDP[..., L]
        THM = TVMS / (PKMS * (1 + QMS * RDP_L * deltx))
        QM = QMS * RDP_L
        
        # Apply mixing (broadcast unstable to match T_slice shape)
        T_new = jnp.where(unstable[..., jnp.newaxis], THM[..., jnp.newaxis], T_slice)
        Q_new = jnp.where(unstable[..., jnp.newaxis], QM[..., jnp.newaxis], Q_slice)
        
        # Apply mixing using dynamic_update_slice
        T_curr = lax.dynamic_update_slice(T_curr, T_new, (0, 0, L))
        Q_curr = lax.dynamic_update_slice(Q_curr, Q_new, (0, 0, L))
        TV_curr = T_curr * (1 + Q_curr * deltx)
        
        return (T_curr, Q_curr, TV_curr), None

    # Use lax.scan to iterate over layers
    (T_out, Q_out, _), _ = lax.scan(
        loop_body,
        (T_out, Q_out, TV),
        jnp.arange(T.shape[-1] - 1),
    )

    return T_out, Q_out


# JIT-compiled version for performance
dry_convection_mixing_jit = dry_convection_mixing
