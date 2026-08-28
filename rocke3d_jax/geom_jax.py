"""
JAX Implementation of ROCKE-3D GEOM (Geometry Variables)
========================================================

This module provides a JAX-compatible implementation of the geometry variables
from ROCKE-3D's GEOM.f. It includes:
- Constants for grid spacing (e.g., DLON, DLAT).
- Placeholder arrays for geometry variables (e.g., LAT, LON, DXYP).
- Helper functions to compute derived quantities (e.g., SINLAT2D, COSLAT2D).

Key Features:
- All arrays are JAX-compatible (no dynamic allocation in JIT functions).
- Helper functions are JIT-compiled for performance.
- Designed for use in JAX-based atmospheric models.

Usage:
    from geom_jax import compute_sinlat2d_coslat2d, compute_rapj
    sinlat2d, coslat2d = compute_sinlat2d_coslat2d(lat2d)
    rapj = compute_rapj(im, jm)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from constant_jax import TWOPI, RADIUS, AREAG


# ============================================================================
# Default Model Resolution (can be overridden)
# ============================================================================

# Default number of longitudinal and latitudinal grid boxes
IM = 72
JM = 46

# Half the number of longitudinal boxes
IMH = IM // 2

# Real values related to number of longitudinal grid boxes
FIM = jnp.float32(IM)
BYIM = jnp.float32(1.0 / FIM)

# Grid spacing in longitude (degrees and radians)
DLON = jnp.float32(TWOPI * BYIM)  # Radians
DLON_DG = jnp.float32(360.0 / IM)  # Degrees

# Equatorial value of J
FJEQ = jnp.float32(0.5 * (1 + JM))

# Index of southernmost latitude (currently 2)
J1U = 2

# Lowest, highest lat index for SH, NH for A, B grid
JRANGE_HEMI = jnp.array([
    [1, JM // 2],
    [1 + JM // 2, JM],
    [J1U, J1U - 1 + JM // 2],
    [J1U - 1 + JM // 2, JM + J1U - 2],
], dtype=jnp.int32).reshape((2, 2, 2))


# ============================================================================
# Placeholder Arrays for Geometry Variables
# ============================================================================

# Note: In JAX, we cannot dynamically allocate arrays like in Fortran.
# Instead, we define placeholder arrays with a fixed size (IM, JM).
# Users should replace these with their own data.

# Latitude of mid-point of primary grid box (radians)
LAT = jnp.zeros(JM, dtype=jnp.float32)

# Latitude of southern edge of primary grid box (radians)
LATV = jnp.zeros(JM, dtype=jnp.float32)

# Latitude of mid points of primary and secondary grid boxes (degrees)
LAT_DG = jnp.zeros((JM, 2), dtype=jnp.float32)

# Longitude of mid points of primary grid box (radians)
LON = jnp.zeros(IM, dtype=jnp.float32)

# Longitude of east edge of primary grid box (radians)
LONV = jnp.zeros(IM, dtype=jnp.float32)

# Longitude of mid points of primary and secondary grid boxes (degrees)
LON_DG = jnp.zeros((IM, 2), dtype=jnp.float32)

# Area of grid box (m^2)
DXYP = jnp.zeros(JM, dtype=jnp.float32)

# Inverse area of grid box (m^-2)
BYDXYP = jnp.zeros(JM, dtype=jnp.float32)

# 2D area of grid box (m^2)
aDXYP = jnp.zeros((IM, JM), dtype=jnp.float32)

# Sine and cosine of latitude (2D)
SINLAT2D = jnp.zeros((IM, JM), dtype=jnp.float32)
COSLAT2D = jnp.zeros((IM, JM), dtype=jnp.float32)

# Distance between points on primary grid (m)
DXP = jnp.zeros(JM, dtype=jnp.float32)
DYP = jnp.zeros(JM, dtype=jnp.float32)

# Inverse distance between points on primary grid (m^-1)
BYDXP = jnp.zeros(JM, dtype=jnp.float32)
BYDYP = jnp.zeros(JM, dtype=jnp.float32)

# Sine and cosine of latitude at primary grid points
SINP = jnp.zeros(JM, dtype=jnp.float32)
COSP = jnp.zeros(JM, dtype=jnp.float32)

# Sine and cosine of latitude at secondary grid points
SINLATV = jnp.zeros(JM, dtype=jnp.float32)
COSLATV = jnp.zeros(JM, dtype=jnp.float32)

# Area scalings for primary and secondary grid
RAPVS = jnp.zeros(JM, dtype=jnp.float32)
RAPVN = jnp.zeros(JM, dtype=jnp.float32)
RAVPS = jnp.zeros(JM, dtype=jnp.float32)
RAVPN = jnp.zeros(JM, dtype=jnp.float32)

# Longitudinal sin, cos for wind, pressure grid
SINIV = jnp.zeros(IM, dtype=jnp.float32)
COSIV = jnp.zeros(IM, dtype=jnp.float32)
SINIP = jnp.zeros(IM, dtype=jnp.float32)
COSIP = jnp.zeros(IM, dtype=jnp.float32)
SINU = jnp.zeros(IM, dtype=jnp.float32)
COSU = jnp.zeros(IM, dtype=jnp.float32)

# Scaling for A grid U/V to B grid points (function of lat. j)
RAPJ = jnp.zeros((IM, JM), dtype=jnp.float32)

# Scaling for B grid -> A grid conversion (1/4, 1/im at poles)
RAVJ = jnp.zeros((IM, JM), dtype=jnp.float32)

# J index of adjacent U/V points for A grid (function of lat. j)
IDJJ = jnp.zeros((IM, JM), dtype=jnp.int32)

# I index of adjacent U/V points for A grid (function of lat/lon)
IDIJ = jnp.zeros((IM, JM, 4), dtype=jnp.int32)

# Varying number of adjacent velocity points
KMAXJ = jnp.zeros(JM, dtype=jnp.int32)

# Varying number of used longitudes
IMAXJ = jnp.zeros(JM, dtype=jnp.int32)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_lat_lon(
    im: int = IM,
    jm: int = JM,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute latitude and longitude arrays for the primary grid.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        lat: Latitude of mid-point of primary grid box (radians) (JM)
        lon: Longitude of mid-point of primary grid box (radians) (IM)
        lat_dg: Latitude of mid-point of primary grid box (degrees) (JM)
        lon_dg: Longitude of mid-point of primary grid box (degrees) (IM)
    """
    # Compute longitude (evenly spaced)
    lon = jnp.linspace(0.0, TWOPI - DLON, im, dtype=jnp.float32)
    lon_dg = jnp.linspace(0.0, 360.0 - DLON_DG, im, dtype=jnp.float32)
    
    # Compute latitude (Gaussian grid or evenly spaced)
    # For simplicity, use evenly spaced latitudes
    lat = jnp.linspace(-jnp.pi / 2 + DLON, jnp.pi / 2 - DLON, jm, dtype=jnp.float32)
    lat_dg = jnp.linspace(-90.0 + DLON_DG, 90.0 - DLON_DG, jm, dtype=jnp.float32)
    
    return lat, lon, lat_dg, lon_dg


@jit
def compute_sinlat2d_coslat2d(
    lat2d: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute sine and cosine of latitude for a 2D array.
    
    Args:
        lat2d: Latitude (radians) (I, J)
    
    Returns:
        sinlat2d: Sine of latitude (I, J)
        coslat2d: Cosine of latitude (I, J)
    """
    sinlat2d = jnp.sin(lat2d)
    coslat2d = jnp.cos(lat2d)
    return sinlat2d, coslat2d


def compute_rapj(
    im: int = IM,
    jm: int = JM,
) -> jnp.ndarray:
    """
    Compute RAPJ (scaling for B grid -> A grid conversion).
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        rapj: Scaling for B grid -> A grid conversion (I, J)
    """
    # For simplicity, assume RAPJ = 1.0 / IM for all points
    rapj = jnp.ones((im, jm), dtype=jnp.float32) / im
    return rapj


def compute_dxyp(
    lat: jnp.ndarray,
    radius: jnp.ndarray = RADIUS,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute DXYP (area of grid box) and BYDXYP (inverse area).
    
    Args:
        lat: Latitude (radians) (J)
        radius: Radius of the Earth (m)
    
    Returns:
        dxyp: Area of grid box (m^2) (J)
        bydxyp: Inverse area of grid box (m^-2) (J)
    """
    # Area of grid box: DXYP = 2 * pi * radius^2 * sin(lat) * dlat / JM
    # For simplicity, assume evenly spaced latitudes
    jm = len(lat)
    dlat = jnp.pi / jm  # Latitude spacing (radians)
    # Use absolute value of sin(lat) to ensure positive area
    dxyp = 2 * jnp.pi * radius**2 * jnp.abs(jnp.sin(lat)) * dlat / jm
    bydxyp = 1.0 / dxyp
    return dxyp, bydxyp


def compute_axyp(
    lat2d: jnp.ndarray,
    lon2d: jnp.ndarray,
    radius: jnp.ndarray = RADIUS,
) -> jnp.ndarray:
    """
    Compute AXYP (2D area of grid box).
    
    Args:
        lat2d: Latitude (radians) (I, J)
        lon2d: Longitude (radians) (I, J)
        radius: Radius of the Earth (m)
    
    Returns:
        axyp: 2D area of grid box (m^2) (I, J)
    """
    # Area of grid box: AXYP = radius^2 * |sin(lat)| * dlat * dlon
    jm = lat2d.shape[1]
    im = lat2d.shape[0]
    dlat = jnp.pi / jm  # Latitude spacing (radians)
    dlon = TWOPI / im   # Longitude spacing (radians)
    axyp = radius**2 * jnp.abs(jnp.sin(lat2d)) * dlat * dlon
    return axyp


def compute_kmaxj(
    jm: int = JM,
) -> jnp.ndarray:
    """
    Compute KMAXJ (varying number of adjacent velocity points).
    
    Args:
        jm: Number of latitudinal grid boxes
    
    Returns:
        kmaxj: Varying number of adjacent velocity points (J)
    """
    # For simplicity, assume KMAXJ = 4 for all points
    kmaxj = jnp.ones(jm, dtype=jnp.int32) * 4
    return kmaxj


def compute_imaxj(
    jm: int = JM,
    im: int = IM,
) -> jnp.ndarray:
    """
    Compute IMAXJ (varying number of used longitudes).
    
    Args:
        jm: Number of latitudinal grid boxes
        im: Number of longitudinal grid boxes
    
    Returns:
        imaxj: Varying number of used longitudes (J)
    """
    # For simplicity, assume IMAXJ = IM for all points
    imaxj = jnp.ones(jm, dtype=jnp.int32) * im
    return imaxj


# ============================================================================
# Initialization Function
# ============================================================================

def initialize_geom(
    im: int = IM,
    jm: int = JM,
) -> dict:
    """
    Initialize GEOM arrays with the specified grid size.
    
    Args:
        im: Number of longitudinal grid boxes
        jm: Number of latitudinal grid boxes
    
    Returns:
        A dictionary containing initialized GEOM arrays
    """
    # Compute latitude and longitude
    lat, lon, lat_dg, lon_dg = compute_lat_lon(im, jm)
    
    # Compute 2D latitude and longitude
    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing='ij')
    
    # Compute sine and cosine of latitude
    sinlat2d, coslat2d = compute_sinlat2d_coslat2d(lat2d)
    
    # Compute area arrays
    dxyp, bydxyp = compute_dxyp(lat)
    axyp = compute_axyp(lat2d, lon2d)
    
    # Compute scaling arrays
    rapj = compute_rapj(im, jm)
    ravj = jnp.ones((im, jm), dtype=jnp.float32) / 4.0
    
    # Compute KMAXJ and IMAXJ
    kmaxj = compute_kmaxj(jm)
    imaxj = compute_imaxj(jm, im)
    
    # Initialize IDJJ and IDIJ (placeholder values)
    idjj = jnp.zeros((im, jm), dtype=jnp.int32)
    idij = jnp.zeros((im, jm, 4), dtype=jnp.int32)
    
    return {
        "LAT": lat, "LON": lon, "LAT_DG": lat_dg, "LON_DG": lon_dg,
        "LAT2D": lat2d, "LON2D": lon2d,
        "SINLAT2D": sinlat2d, "COSLAT2D": coslat2d,
        "DXYP": dxyp, "BYDXYP": bydxyp, "aDXYP": axyp,
        "RAPJ": rapj, "RAVJ": ravj,
        "KMAXJ": kmaxj, "IMAXJ": imaxj,
        "IDJJ": idjj, "IDIJ": idij,
    }


# ============================================================================
# Summary
# ============================================================================

__all__ = [
    # Constants
    "IM", "JM", "IMH", "FIM", "BYIM", "DLON", "DLON_DG",
    "FJEQ", "J1U", "JRANGE_HEMI",
    # Placeholder arrays
    "LAT", "LATV", "LAT_DG", "LON", "LONV", "LON_DG",
    "DXYP", "BYDXYP", "aDXYP", "SINLAT2D", "COSLAT2D",
    "DXP", "DYP", "BYDXP", "BYDYP",
    "SINP", "COSP", "SINLATV", "COSLATV",
    "RAPVS", "RAPVN", "RAVPS", "RAVPN",
    "SINIV", "COSIV", "SINIP", "COSIP", "SINU", "COSU",
    "RAPJ", "RAVJ", "IDJJ", "IDIJ", "KMAXJ", "IMAXJ",
    # Helper functions
    "compute_lat_lon", "compute_sinlat2d_coslat2d", "compute_rapj",
    "compute_dxyp", "compute_axyp", "compute_kmaxj", "compute_imaxj",
    "initialize_geom",
]
