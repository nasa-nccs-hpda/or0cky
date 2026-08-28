"""
JAX Implementation of ROCKE-3D SURFACE (Surface Fluxes)
==========================================================

This module provides a JAX-based implementation of the surface flux calculations
from ROCKE-3D's SURFACE.f. It computes surface fluxes (sensible heat, evaporation,
radiation, momentum drag) for different surface types (ocean, sea ice, land, etc.).

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Dependencies:
- JAX (jax, jax.numpy)
- NumPy (for validation)
- fluxes_jax: For flux calculations
- pbl_simple_jax: For boundary layer calculations

Usage:
    from surface_jax import compute_surface_fluxes_jit
    uflux, vflux, tflux, qflux, solar, lw_net = compute_surface_fluxes_jit(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo
    )
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Optional
from fluxes_jax import (
    compute_momentum_flux,
    compute_heat_flux,
    compute_moisture_flux,
    compute_radiative_flux,
)


# Constants from CONSTANT module (assumed values)
RGAS = 287.0      # Specific gas constant for dry air (J/kg/K)
LHE = 2.501e6     # Latent heat of evaporation (J/kg)
LHS = 2.834e6     # Latent heat of sublimation (J/kg)
SHA = 1004.6      # Specific heat of dry air (J/kg/K)
TF = 273.15       # Freezing point of water (K)
RHOW = 1000.0     # Density of water (kg/m^3)
DELTX = 0.608     # Virtual temperature factor
GRAV = 9.80665    # Gravitational acceleration (m/s^2)
STBO = 5.67e-8    # Stefan-Boltzmann constant (W/m^2/K^4)


@jit
def compute_surface_properties(
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
    ps: jnp.ndarray,      # Surface pressure (Pa)
    psk: Optional[jnp.ndarray] = None,  # Surface pressure scaling factor (Pa^kappa)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute surface properties (virtual temperature, density, etc.).
    
    Args:
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
        ps: Surface pressure (Pa)
        psk: Surface pressure scaling factor (Pa^kappa) (optional)
    
    Returns:
        thv1: Virtual temperature at first layer (K)
        rho: Air density at surface (kg/m^3)
        qsat: Saturation specific humidity (kg/kg)
    """
    # Virtual temperature
    thv1 = t1 * (1.0 + q1 * DELTX)
    
    # Air density (ideal gas law: rho = p / (R * T))
    rho = ps / (RGAS * thv1)
    
    # Saturation specific humidity (simplified Clausius-Clapeyron)
    # qsat = 0.622 * es / (ps - es), where es = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
    es = 611.2 * jnp.exp(17.67 * (t1 - TF) / (t1 - 29.65))
    qsat = 0.622 * es / (ps - es)
    
    return thv1, rho, qsat


# JIT-compiled version for Fortran comparison
compute_surface_properties_jit = jit(compute_surface_properties)


@jit
def compute_ocean_fluxes(
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    cdh: jnp.ndarray,     # Stanton number (heat transfer coefficient)
    cq: jnp.ndarray,      # Dalton number (moisture transfer coefficient)
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    uocean: jnp.ndarray,   # Ocean x-velocity (m/s)
    vocean: jnp.ndarray,   # Ocean y-velocity (m/s)
    lh: jnp.ndarray = LHE, # Latent heat of evaporation (J/kg)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute surface fluxes for ocean surface type.
    
    Args:
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        tsv: Surface virtual temperature (K)
        qsrf: Surface specific humidity (kg/kg)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        cdh: Stanton number (heat transfer coefficient)
        cq: Dalton number (moisture transfer coefficient)
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
        fsf: Downwelling shortwave radiation (W/m^2)
        flong: Downwelling longwave radiation (W/m^2)
        albedo: Surface albedo (dimensionless)
        uocean: Ocean x-velocity (m/s)
        vocean: Ocean y-velocity (m/s)
        lh: Latent heat of evaporation (J/kg)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
        net_energy: Net energy flux (W/m^2)
    """
    # Compute wind speed
    ws = jnp.sqrt(us**2 + vs**2)
    
    # Compute momentum fluxes (adjust for ocean currents)
    uflux, vflux = compute_momentum_flux(us, vs, rho, cdm, u1 - uocean, v1 - vocean)
    
    # Compute sensible heat flux
    tflux = compute_heat_flux(tsv, t1, rho, cdh, ws)
    
    # Compute latent heat flux
    qflux = compute_moisture_flux(qsrf, q1, rho, cq, ws, lh)
    
    # Compute radiative fluxes
    solar, lw_net = compute_radiative_flux(tsv, fsf, flong, albedo)
    
    # Compute net energy flux
    net_energy = solar + lw_net - tflux - qflux
    
    return uflux, vflux, tflux, qflux, solar, lw_net, net_energy


@jit
def compute_land_fluxes(
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    cdh: jnp.ndarray,     # Stanton number (heat transfer coefficient)
    cq: jnp.ndarray,      # Dalton number (moisture transfer coefficient)
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    lh: jnp.ndarray = LHE, # Latent heat of evaporation (J/kg)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute surface fluxes for land surface type.
    
    Args:
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        tsv: Surface virtual temperature (K)
        qsrf: Surface specific humidity (kg/kg)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        cdh: Stanton number (heat transfer coefficient)
        cq: Dalton number (moisture transfer coefficient)
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
        fsf: Downwelling shortwave radiation (W/m^2)
        flong: Downwelling longwave radiation (W/m^2)
        albedo: Surface albedo (dimensionless)
        lh: Latent heat of evaporation (J/kg)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
        net_energy: Net energy flux (W/m^2)
    """
    # Compute wind speed
    ws = jnp.sqrt(us**2 + vs**2)
    
    # Compute momentum fluxes
    uflux, vflux = compute_momentum_flux(us, vs, rho, cdm, u1, v1)
    
    # Compute sensible heat flux
    tflux = compute_heat_flux(tsv, t1, rho, cdh, ws)
    
    # Compute latent heat flux
    qflux = compute_moisture_flux(qsrf, q1, rho, cq, ws, lh)
    
    # Compute radiative fluxes
    solar, lw_net = compute_radiative_flux(tsv, fsf, flong, albedo)
    
    # Compute net energy flux
    net_energy = solar + lw_net - tflux - qflux
    
    return uflux, vflux, tflux, qflux, solar, lw_net, net_energy


@jit
def compute_seaice_fluxes(
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    cdh: jnp.ndarray,     # Stanton number (heat transfer coefficient)
    cq: jnp.ndarray,      # Dalton number (moisture transfer coefficient)
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    uocean: jnp.ndarray,   # Ocean x-velocity (m/s)
    vocean: jnp.ndarray,   # Ocean y-velocity (m/s)
    lh: jnp.ndarray = LHS, # Latent heat of sublimation (J/kg)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute surface fluxes for sea ice surface type.
    
    Args:
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        tsv: Surface virtual temperature (K)
        qsrf: Surface specific humidity (kg/kg)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        cdh: Stanton number (heat transfer coefficient)
        cq: Dalton number (moisture transfer coefficient)
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
        fsf: Downwelling shortwave radiation (W/m^2)
        flong: Downwelling longwave radiation (W/m^2)
        albedo: Surface albedo (dimensionless)
        uocean: Ocean x-velocity (m/s)
        vocean: Ocean y-velocity (m/s)
        lh: Latent heat of sublimation (J/kg)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
        net_energy: Net energy flux (W/m^2)
    """
    # Compute wind speed
    ws = jnp.sqrt(us**2 + vs**2)
    
    # Compute momentum fluxes (adjust for ocean currents)
    uflux, vflux = compute_momentum_flux(us, vs, rho, cdm, u1 - uocean, v1 - vocean)
    
    # Compute sensible heat flux
    tflux = compute_heat_flux(tsv, t1, rho, cdh, ws)
    
    # Compute latent heat flux (sublimation)
    qflux = compute_moisture_flux(qsrf, q1, rho, cq, ws, lh)
    
    # Compute radiative fluxes
    solar, lw_net = compute_radiative_flux(tsv, fsf, flong, albedo)
    
    # Compute net energy flux
    net_energy = solar + lw_net - tflux - qflux
    
    return uflux, vflux, tflux, qflux, solar, lw_net, net_energy


@jit
def compute_landice_fluxes(
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    cdh: jnp.ndarray,     # Stanton number (heat transfer coefficient)
    cq: jnp.ndarray,      # Dalton number (moisture transfer coefficient)
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    lh: jnp.ndarray = LHS, # Latent heat of sublimation (J/kg)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute surface fluxes for land ice surface type.
    
    Args:
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        tsv: Surface virtual temperature (K)
        qsrf: Surface specific humidity (kg/kg)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        cdh: Stanton number (heat transfer coefficient)
        cq: Dalton number (moisture transfer coefficient)
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
        fsf: Downwelling shortwave radiation (W/m^2)
        flong: Downwelling longwave radiation (W/m^2)
        albedo: Surface albedo (dimensionless)
        lh: Latent heat of sublimation (J/kg)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
        net_energy: Net energy flux (W/m^2)
    """
    # Compute wind speed
    ws = jnp.sqrt(us**2 + vs**2)
    
    # Compute momentum fluxes
    uflux, vflux = compute_momentum_flux(us, vs, rho, cdm, u1, v1)
    
    # Compute sensible heat flux
    tflux = compute_heat_flux(tsv, t1, rho, cdh, ws)
    
    # Compute latent heat flux (sublimation)
    qflux = compute_moisture_flux(qsrf, q1, rho, cq, ws, lh)
    
    # Compute radiative fluxes
    solar, lw_net = compute_radiative_flux(tsv, fsf, flong, albedo)
    
    # Compute net energy flux
    net_energy = solar + lw_net - tflux - qflux
    
    return uflux, vflux, tflux, qflux, solar, lw_net, net_energy


@jit
def compute_surface_fluxes_jit(
    itype: jnp.ndarray,   # Surface type (1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land)
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    cdh: jnp.ndarray,     # Stanton number (heat transfer coefficient)
    cq: jnp.ndarray,      # Dalton number (moisture transfer coefficient)
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    uocean: jnp.ndarray,   # Ocean x-velocity (m/s)
    vocean: jnp.ndarray,   # Ocean y-velocity (m/s)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute surface fluxes for all surface types.
    
    Args:
        itype: Surface type (1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land)
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        tsv: Surface virtual temperature (K)
        qsrf: Surface specific humidity (kg/kg)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        cdh: Stanton number (heat transfer coefficient)
        cq: Dalton number (moisture transfer coefficient)
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
        fsf: Downwelling shortwave radiation (W/m^2)
        flong: Downwelling longwave radiation (W/m^2)
        albedo: Surface albedo (dimensionless)
        uocean: Ocean x-velocity (m/s)
        vocean: Ocean y-velocity (m/s)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
        net_energy: Net energy flux (W/m^2)
    """
    # Initialize outputs
    uflux = jnp.zeros_like(us)
    vflux = jnp.zeros_like(vs)
    tflux = jnp.zeros_like(tsv)
    qflux = jnp.zeros_like(qsrf)
    solar = jnp.zeros_like(fsf)
    lw_net = jnp.zeros_like(flong)
    net_energy = jnp.zeros_like(fsf)
    
    # Ocean (itype=1)
    mask_ocean = (itype == 1)
    uflux_ocean, vflux_ocean, tflux_ocean, qflux_ocean, solar_ocean, lw_net_ocean, net_energy_ocean = compute_ocean_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    uflux = jnp.where(mask_ocean, uflux_ocean, uflux)
    vflux = jnp.where(mask_ocean, vflux_ocean, vflux)
    tflux = jnp.where(mask_ocean, tflux_ocean, tflux)
    qflux = jnp.where(mask_ocean, qflux_ocean, qflux)
    solar = jnp.where(mask_ocean, solar_ocean, solar)
    lw_net = jnp.where(mask_ocean, lw_net_ocean, lw_net)
    net_energy = jnp.where(mask_ocean, net_energy_ocean, net_energy)
    
    # Ocean Ice (itype=2)
    mask_ice = (itype == 2)
    uflux_ice, vflux_ice, tflux_ice, qflux_ice, solar_ice, lw_net_ice, net_energy_ice = compute_seaice_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo, uocean, vocean
    )
    uflux = jnp.where(mask_ice, uflux_ice, uflux)
    vflux = jnp.where(mask_ice, vflux_ice, vflux)
    tflux = jnp.where(mask_ice, tflux_ice, tflux)
    qflux = jnp.where(mask_ice, qflux_ice, qflux)
    solar = jnp.where(mask_ice, solar_ice, solar)
    lw_net = jnp.where(mask_ice, lw_net_ice, lw_net)
    net_energy = jnp.where(mask_ice, net_energy_ice, net_energy)
    
    # Land Ice (itype=3)
    mask_landice = (itype == 3)
    uflux_landice, vflux_landice, tflux_landice, qflux_landice, solar_landice, lw_net_landice, net_energy_landice = compute_landice_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo
    )
    uflux = jnp.where(mask_landice, uflux_landice, uflux)
    vflux = jnp.where(mask_landice, vflux_landice, vflux)
    tflux = jnp.where(mask_landice, tflux_landice, tflux)
    qflux = jnp.where(mask_landice, qflux_landice, qflux)
    solar = jnp.where(mask_landice, solar_landice, solar)
    lw_net = jnp.where(mask_landice, lw_net_landice, lw_net)
    net_energy = jnp.where(mask_landice, net_energy_landice, net_energy)
    
    # Land (itype=4)
    mask_land = (itype == 4)
    uflux_land, vflux_land, tflux_land, qflux_land, solar_land, lw_net_land, net_energy_land = compute_land_fluxes(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1, fsf, flong, albedo
    )
    uflux = jnp.where(mask_land, uflux_land, uflux)
    vflux = jnp.where(mask_land, vflux_land, vflux)
    tflux = jnp.where(mask_land, tflux_land, tflux)
    qflux = jnp.where(mask_land, qflux_land, qflux)
    solar = jnp.where(mask_land, solar_land, solar)
    lw_net = jnp.where(mask_land, lw_net_land, lw_net)
    net_energy = jnp.where(mask_land, net_energy_land, net_energy)
    
    return uflux, vflux, tflux, qflux, solar, lw_net, net_energy
