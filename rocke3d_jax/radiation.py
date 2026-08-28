"""
JAX Implementation of ROCKE-3D Radiative Transfer
==================================================

This module provides JAX-based implementations of radiative transfer subroutines.
Currently includes:
- blackbody_radiation: Stefan-Boltzmann law for thermal emission.
- solar_radiation: Simple solar radiation model.

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Usage:
    from rocke3d_jax.radiation import blackbody_radiation_jit
    lw_up = blackbody_radiation_jit(T, emissivity=0.95)
"""

import jax
import jax.numpy as jnp
from jax import jit


# Constants
sigma_sb = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
solar_constant = 1361.0    # Solar constant (W/m²)


@jit
def blackbody_radiation(
    T: jnp.ndarray,
    emissivity: float = 0.95,
) -> jnp.ndarray:
    """
    JAX implementation of blackbody radiation (Stefan-Boltzmann law).
    Computes upward longwave radiation at the surface.
    
    Args:
        T: Surface temperature (K) (I, J) or (I, J, L).
        emissivity: Surface emissivity (default: 0.95).
    
    Returns:
        Upward longwave radiation (W/m²).
    """
    return emissivity * sigma_sb * T ** 4


@jit
def solar_radiation(
    cosz: jnp.ndarray,
    albedo: jnp.ndarray,
    solar_constant: float = 1361.0,
) -> jnp.ndarray:
    """
    JAX implementation of solar radiation (simplified).
    Computes absorbed solar radiation at the surface.
    
    Args:
        cosz: Cosine of solar zenith angle (I, J).
        albedo: Surface albedo (I, J).
        solar_constant: Solar constant (W/m², default: 1361.0).
    
    Returns:
        Absorbed solar radiation (W/m²).
    """
    # Downward solar radiation (account for zenith angle)
    sw_down = solar_constant * jnp.maximum(cosz, 0.0)
    # Absorbed solar radiation (1 - albedo) * sw_down
    sw_absorbed = (1.0 - albedo) * sw_down
    return sw_absorbed


@jit
def net_radiation(
    T: jnp.ndarray,
    cosz: jnp.ndarray,
    albedo: jnp.ndarray,
    emissivity: float = 0.95,
    solar_constant: float = 1361.0,
) -> jnp.ndarray:
    """
    JAX implementation of net radiation (solar + longwave).
    Computes net radiation at the surface.
    
    Args:
        T: Surface temperature (K) (I, J).
        cosz: Cosine of solar zenith angle (I, J).
        albedo: Surface albedo (I, J).
        emissivity: Surface emissivity (default: 0.95).
        solar_constant: Solar constant (W/m², default: 1361.0).
    
    Returns:
        Net radiation (W/m²).
    """
    # Upward longwave radiation
    lw_up = blackbody_radiation(T, emissivity)
    # Downward longwave radiation (assume atmospheric emission = surface emission)
    lw_down = emissivity * sigma_sb * T ** 4
    # Net longwave radiation
    lw_net = lw_down - lw_up
    # Absorbed solar radiation
    sw_absorbed = solar_radiation(cosz, albedo, solar_constant)
    # Net radiation
    net_rad = sw_absorbed + lw_net
    return net_rad


# JIT-compiled versions for performance
blackbody_radiation_jit = jit(blackbody_radiation)
solar_radiation_jit = jit(solar_radiation)
net_radiation_jit = jit(net_radiation)
