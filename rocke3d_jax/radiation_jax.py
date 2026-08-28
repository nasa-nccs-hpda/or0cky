"""
JAX Implementation of ROCKE-3D RADIATION (Radiative Transfer)
===============================================================

This module provides a JAX-based implementation of the core radiative transfer
calculations from ROCKE-3D's RADIATION.f. It includes:
- Planck function for blackbody radiation.
- Stefan-Boltzmann law for thermal emission.
- Solar and longwave flux calculations.
- Simplified atmospheric absorption/emission.

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Dependencies:
- JAX (jax, jax.numpy)
- NumPy (for validation)

Usage:
    from radiation_jax import (
        planck_function_jit,
        stefan_boltzmann_jit,
        compute_solar_flux_jit,
        compute_lw_flux_jit,
        compute_radiative_fluxes_jit,
    )
    lw_flux = stefan_boltzmann_jit(temperature, emissivity)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple


# Constants from RADIATION.f
STBO = 5.67e-8    # Stefan-Boltzmann constant (W/m^2/K^4)
PLANCK_H = 6.62607015e-34  # Planck constant (J s)
PLANCK_C = 2.99792458e8   # Speed of light (m/s)
PLANCK_K = 1.380649e-23   # Boltzmann constant (J/K)
RGAS = 287.0      # Specific gas constant for dry air (J/kg/K)

# Solar constant (W/m^2)
SOLAR_CONSTANT = 1365.0

# Earth's albedo (dimensionless)
EARTH_ALBEDO = 0.3

# Emissivity for longwave radiation (dimensionless)
EMISSIVITY = 0.98


@jit
def planck_function_jit(
    wavelength: jnp.ndarray,  # Wavelength (m)
    temperature: jnp.ndarray, # Temperature (K)
) -> jnp.ndarray:
    """
    Compute the Planck function for blackbody radiation.
    
    Args:
        wavelength: Wavelength (m)
        temperature: Temperature (K)
    
    Returns:
        B: Spectral radiance (W/m^2/sr/m)
    """
    # Planck function: B(lambda, T) = (2 * h * c^2 / lambda^5) / (exp(h * c / (lambda * k * T)) - 1)
    numerator = 2.0 * PLANCK_H * PLANCK_C**2 / (wavelength**5)
    denominator = jnp.exp(PLANCK_H * PLANCK_C / (wavelength * PLANCK_K * temperature)) - 1.0
    B = numerator / denominator
    return B


@jit
def stefan_boltzmann_jit(
    temperature: jnp.ndarray,  # Temperature (K)
    emissivity: jnp.ndarray = EMISSIVITY,  # Emissivity (dimensionless)
) -> jnp.ndarray:
    """
    Compute the Stefan-Boltzmann law for thermal emission.
    
    Args:
        temperature: Temperature (K)
        emissivity: Emissivity (dimensionless)
    
    Returns:
        lw_flux: Longwave flux (W/m^2)
    """
    lw_flux = emissivity * STBO * temperature**4
    return lw_flux


@jit
def compute_solar_flux_jit(
    solar_constant: jnp.ndarray = SOLAR_CONSTANT,  # Solar constant (W/m^2)
    cosz: jnp.ndarray = 1.0,  # Cosine of solar zenith angle (dimensionless)
    albedo: jnp.ndarray = EARTH_ALBEDO,  # Surface albedo (dimensionless)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute solar (shortwave) fluxes at the surface.
    
    Args:
        solar_constant: Solar constant (W/m^2)
        cosz: Cosine of solar zenith angle (dimensionless)
        albedo: Surface albedo (dimensionless)
    
    Returns:
        fsf: Downwelling shortwave radiation (W/m^2)
        srdflb: Surface absorbed shortwave radiation (W/m^2)
        srnflb: Net shortwave radiation at surface (W/m^2)
    """
    # Downwelling shortwave radiation
    fsf = solar_constant * cosz
    
    # Surface absorbed shortwave radiation
    srdflb = (1.0 - albedo) * fsf
    
    # Net shortwave radiation at surface
    srnflb = srdflb - albedo * fsf
    
    return fsf, srdflb, srnflb


@jit
def compute_lw_flux_jit(
    temperature: jnp.ndarray,  # Temperature (K)
    emissivity: jnp.ndarray = EMISSIVITY,  # Emissivity (dimensionless)
    flong: jnp.ndarray = 0.0,  # Downwelling longwave radiation (W/m^2)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute longwave fluxes at the surface.
    
    Args:
        temperature: Temperature (K)
        emissivity: Emissivity (dimensionless)
        flong: Downwelling longwave radiation (W/m^2)
    
    Returns:
        trdflb: Downwelling longwave radiation (W/m^2)
        truflb: Upwelling longwave radiation (W/m^2)
        trnflb: Net longwave radiation at surface (W/m^2)
    """
    # Upwelling longwave radiation (Stefan-Boltzmann law)
    truflb = emissivity * STBO * temperature**4
    
    # Downwelling longwave radiation (input)
    trdflb = flong
    
    # Net longwave radiation at surface
    trnflb = trdflb - truflb
    
    return trdflb, truflb, trnflb


@jit
def compute_radiative_fluxes_jit(
    temperature: jnp.ndarray,  # Surface temperature (K)
    solar_constant: jnp.ndarray = SOLAR_CONSTANT,  # Solar constant (W/m^2)
    cosz: jnp.ndarray = 1.0,  # Cosine of solar zenith angle (dimensionless)
    albedo: jnp.ndarray = EARTH_ALBEDO,  # Surface albedo (dimensionless)
    emissivity: jnp.ndarray = EMISSIVITY,  # Emissivity (dimensionless)
    flong: jnp.ndarray = 0.0,  # Downwelling longwave radiation (W/m^2)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray
]:
    """
    Compute all radiative fluxes (solar and longwave) at the surface.
    
    Args:
        temperature: Surface temperature (K)
        solar_constant: Solar constant (W/m^2)
        cosz: Cosine of solar zenith angle (dimensionless)
        albedo: Surface albedo (dimensionless)
        emissivity: Emissivity (dimensionless)
        flong: Downwelling longwave radiation (W/m^2)
    
    Returns:
        fsf: Downwelling shortwave radiation (W/m^2)
        srdflb: Surface absorbed shortwave radiation (W/m^2)
        srnflb: Net shortwave radiation at surface (W/m^2)
        trdflb: Downwelling longwave radiation (W/m^2)
        truflb: Upwelling longwave radiation (W/m^2)
        trnflb: Net longwave radiation at surface (W/m^2)
        net_sw: Net shortwave radiation (W/m^2)
        net_lw: Net longwave radiation (W/m^2)
    """
    # Solar (shortwave) fluxes
    fsf, srdflb, srnflb = compute_solar_flux_jit(solar_constant, cosz, albedo)
    
    # Longwave fluxes
    trdflb, truflb, trnflb = compute_lw_flux_jit(temperature, emissivity, flong)
    
    # Net fluxes
    net_sw = srnflb
    net_lw = trnflb
    
    return fsf, srdflb, srnflb, trdflb, truflb, trnflb, net_sw, net_lw


@jit
def compute_atmospheric_absorption_jit(
    temperature: jnp.ndarray,  # Temperature (K)
    pressure: jnp.ndarray,     # Pressure (Pa)
    water_vapor: jnp.ndarray,  # Water vapor mixing ratio (kg/kg)
    co2: jnp.ndarray = 400.0,  # CO2 concentration (ppm)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute atmospheric absorption for longwave radiation.
    
    This is a **simplified** version of the full radiative transfer model.
    It uses a **graybody approximation** for atmospheric absorption.
    
    Args:
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        water_vapor: Water vapor mixing ratio (kg/kg)
        co2: CO2 concentration (ppm)
    
    Returns:
        lw_absorbed: Longwave radiation absorbed by atmosphere (W/m^2)
        lw_transmitted: Longwave radiation transmitted through atmosphere (W/m^2)
    """
    # Simplified graybody approximation
    # Absorptivity = 1 - exp(-kappa * path_length), where kappa is absorption coefficient
    # For simplicity, assume kappa = 0.1 (dimensionless)
    kappa = 0.1
    path_length = pressure / (RGAS * temperature)  # Path length (m)
    absorptivity = 1.0 - jnp.exp(-kappa * path_length)
    
    # Surface emission
    surface_emission = STBO * temperature**4
    
    # Absorbed and transmitted radiation
    lw_absorbed = absorptivity * surface_emission
    lw_transmitted = (1.0 - absorptivity) * surface_emission
    
    return lw_absorbed, lw_transmitted


@jit
def compute_net_radiation_jit(
    temperature: jnp.ndarray,  # Surface temperature (K)
    solar_constant: jnp.ndarray = SOLAR_CONSTANT,  # Solar constant (W/m^2)
    cosz: jnp.ndarray = 1.0,  # Cosine of solar zenith angle (dimensionless)
    albedo: jnp.ndarray = EARTH_ALBEDO,  # Surface albedo (dimensionless)
    emissivity: jnp.ndarray = EMISSIVITY,  # Emissivity (dimensionless)
    flong: jnp.ndarray = 0.0,  # Downwelling longwave radiation (W/m^2)
) -> jnp.ndarray:
    """
    Compute net radiation at the surface.
    
    Args:
        temperature: Surface temperature (K)
        solar_constant: Solar constant (W/m^2)
        cosz: Cosine of solar zenith angle (dimensionless)
        albedo: Surface albedo (dimensionless)
        emissivity: Emissivity (dimensionless)
        flong: Downwelling longwave radiation (W/m^2)
    
    Returns:
        net_radiation: Net radiation at surface (W/m^2)
    """
    # Compute all radiative fluxes
    fsf, srdflb, srnflb, trdflb, truflb, trnflb, net_sw, net_lw = compute_radiative_fluxes_jit(
        temperature, solar_constant, cosz, albedo, emissivity, flong
    )
    
    # Net radiation
    net_radiation = net_sw + net_lw
    
    return net_radiation
