"""
Unit Tests for GEOM JAX Implementation
=======================================

This script tests the JAX implementation of the geometry variables
from ROCKE-3D's GEOM.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_geom_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from geom_jax import (
    IM, JM, IMH, FIM, BYIM, DLON, DLON_DG,
    FJEQ, J1U, JRANGE_HEMI,
    compute_lat_lon, compute_sinlat2d_coslat2d, compute_rapj,
    compute_dxyp, compute_axyp, compute_kmaxj, compute_imaxj,
    initialize_geom,
)
from constant_jax import TWOPI, RADIUS


# Test constants
np.random.seed(42)

# Grid size
IM_TEST = 36
JM_TEST = 24


def test_constants():
    """Test geometry constants."""
    print("Testing constants...")
    
    # Check IM and JM
    assert IM == 72, "IM should be 72"
    assert JM == 46, "JM should be 46"
    
    # Check IMH
    assert IMH == IM // 2, "IMH should be IM // 2"
    
    # Check FIM and BYIM
    assert FIM == IM, "FIM should be IM"
    assert jnp.abs(BYIM - 1.0 / IM) < 1e-5, "BYIM should be 1.0 / IM"
    
    # Check DLON and DLON_DG
    assert jnp.abs(DLON - TWOPI / IM) < 1e-5, "DLON should be TWOPI / IM"
    assert jnp.abs(DLON_DG - 360.0 / IM) < 1e-5, "DLON_DG should be 360.0 / IM"
    
    # Check FJEQ
    assert jnp.abs(FJEQ - 0.5 * (1 + JM)) < 1e-5, "FJEQ should be 0.5 * (1 + JM)"
    
    # Check J1U
    assert J1U == 2, "J1U should be 2"
    
    # Check JRANGE_HEMI
    assert JRANGE_HEMI.shape == (2, 2, 2), "JRANGE_HEMI shape is incorrect"
    
    print("✅ Constants test passed!")


def test_compute_lat_lon():
    """Test compute_lat_lon function."""
    print("Testing compute_lat_lon...")
    
    # Compute latitude and longitude
    lat, lon, lat_dg, lon_dg = compute_lat_lon(IM_TEST, JM_TEST)
    
    # Check shapes
    assert lat.shape == (JM_TEST,), f"Expected shape {(JM_TEST,)} for lat, got {lat.shape}"
    assert lon.shape == (IM_TEST,), f"Expected shape {(IM_TEST,)} for lon, got {lon.shape}"
    assert lat_dg.shape == (JM_TEST,), f"Expected shape {(JM_TEST,)} for lat_dg, got {lat_dg.shape}"
    assert lon_dg.shape == (IM_TEST,), f"Expected shape {(IM_TEST,)} for lon_dg, got {lon_dg.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(lat)), "lat contains NaN"
    assert not jnp.any(jnp.isnan(lon)), "lon contains NaN"
    
    # Check latitude range
    assert jnp.all(lat >= -jnp.pi / 2), "lat should be >= -pi/2"
    assert jnp.all(lat <= jnp.pi / 2), "lat should be <= pi/2"
    
    # Check longitude range
    assert jnp.all(lon >= 0.0), "lon should be >= 0"
    assert jnp.all(lon <= TWOPI), "lon should be <= 2*pi"
    
    print("✅ compute_lat_lon test passed!")


def test_compute_sinlat2d_coslat2d():
    """Test compute_sinlat2d_coslat2d function."""
    print("Testing compute_sinlat2d_coslat2d...")
    
    # Create test lat2d array
    lat2d = np.random.rand(IM_TEST, JM_TEST).astype(np.float32) * jnp.pi - jnp.pi / 2
    
    # Compute sinlat2d and coslat2d
    sinlat2d, coslat2d = compute_sinlat2d_coslat2d(lat2d)
    
    # Check shapes
    assert sinlat2d.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for sinlat2d, got {sinlat2d.shape}"
    assert coslat2d.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for coslat2d, got {coslat2d.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(sinlat2d)), "sinlat2d contains NaN"
    assert not jnp.any(jnp.isnan(coslat2d)), "coslat2d contains NaN"
    
    # Check values (sin^2 + cos^2 = 1)
    assert jnp.allclose(sinlat2d**2 + coslat2d**2, 1.0, rtol=1e-5), "sin^2 + cos^2 should equal 1"
    
    print("✅ compute_sinlat2d_coslat2d test passed!")


def test_compute_rapj():
    """Test compute_rapj function."""
    print("Testing compute_rapj...")
    
    # Compute rapj
    rapj = compute_rapj(IM_TEST, JM_TEST)
    
    # Check shape
    assert rapj.shape == (IM_TEST, JM_TEST), f"Expected shape {(IM_TEST, JM_TEST)} for rapj, got {rapj.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(rapj)), "rapj contains NaN"
    assert not jnp.any(jnp.isinf(rapj)), "rapj contains Inf"
    
    # Check values (rapj = 1.0 / IM)
    expected_rapj = jnp.ones((IM_TEST, JM_TEST), dtype=jnp.float32) / IM_TEST
    assert jnp.allclose(rapj, expected_rapj, rtol=1e-5), "rapj values are incorrect"
    
    print("✅ compute_rapj test passed!")


def test_compute_dxyp():
    """Test compute_dxyp function."""
    print("Testing compute_dxyp...")
    
    # Create test lat array
    lat = np.linspace(-jnp.pi / 2, jnp.pi / 2, JM_TEST, dtype=np.float32)
    
    # Compute dxyp and bydxyp
    dxyp, bydxyp = compute_dxyp(lat)
    
    # Check shapes
    assert dxyp.shape == (JM_TEST,), f"Expected shape {(JM_TEST,)} for dxyp, got {dxyp.shape}"
    assert bydxyp.shape == (JM_TEST,), f"Expected shape {(JM_TEST,)} for bydxyp, got {bydxyp.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(dxyp)), "dxyp contains NaN"
    assert not jnp.any(jnp.isnan(bydxyp)), "bydxyp contains NaN"
    
    # Check values (dxyp > 0, bydxyp = 1 / dxyp)
    assert jnp.all(dxyp > 0.0), "dxyp should be positive"
    assert jnp.allclose(bydxyp, 1.0 / dxyp, rtol=1e-5), "bydxyp should be 1 / dxyp"
    
    print("✅ compute_dxyp test passed!")


def test_compute_axyp():
    """Test compute_axyp function."""
    print("Testing compute_axyp...")
    
    # Create test lat2d and lon2d arrays
    lat = np.linspace(-jnp.pi / 2, jnp.pi / 2, JM_TEST, dtype=np.float32)
    lon = np.linspace(0.0, TWOPI, IM_TEST, dtype=np.float32)
    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing='ij')
    
    # Compute axyp
    axyp = compute_axyp(lat2d, lon2d)
    
    # Check shape
    assert axyp.shape == (JM_TEST, IM_TEST), f"Expected shape {(JM_TEST, IM_TEST)} for axyp, got {axyp.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(axyp)), "axyp contains NaN"
    assert not jnp.any(jnp.isinf(axyp)), "axyp contains Inf"
    
    # Check values (axyp > 0)
    assert jnp.all(axyp > 0.0), "axyp should be positive"
    
    print("✅ compute_axyp test passed!")


def test_compute_kmaxj():
    """Test compute_kmaxj function."""
    print("Testing compute_kmaxj...")
    
    # Compute kmaxj
    kmaxj = compute_kmaxj(JM_TEST)
    
    # Check shape
    assert kmaxj.shape == (JM_TEST,), f"Expected shape {(JM_TEST,)} for kmaxj, got {kmaxj.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(kmaxj)), "kmaxj contains NaN"
    
    # Check values (kmaxj = 4 for all points)
    expected_kmaxj = jnp.ones(JM_TEST, dtype=jnp.int32) * 4
    assert jnp.allclose(kmaxj, expected_kmaxj), "kmaxj values are incorrect"
    
    print("✅ compute_kmaxj test passed!")


def test_compute_imaxj():
    """Test compute_imaxj function."""
    print("Testing compute_imaxj...")
    
    # Compute imaxj
    imaxj = compute_imaxj(JM_TEST, IM_TEST)
    
    # Check shape
    assert imaxj.shape == (JM_TEST,), f"Expected shape {(JM_TEST,)} for imaxj, got {imaxj.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(imaxj)), "imaxj contains NaN"
    
    # Check values (imaxj = IM for all points)
    expected_imaxj = jnp.ones(JM_TEST, dtype=jnp.int32) * IM_TEST
    assert jnp.allclose(imaxj, expected_imaxj), "imaxj values are incorrect"
    
    print("✅ compute_imaxj test passed!")


def test_initialize_geom():
    """Test initialize_geom function."""
    print("Testing initialize_geom...")
    
    # Initialize GEOM arrays
    geom = initialize_geom(IM_TEST, JM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "LAT", "LON", "LAT_DG", "LON_DG",
        "LAT2D", "LON2D",
        "SINLAT2D", "COSLAT2D",
        "DXYP", "BYDXYP", "aDXYP",
        "RAPJ", "RAVJ",
        "KMAXJ", "IMAXJ",
        "IDJJ", "IDIJ",
    ]
    for key in expected_keys:
        assert key in geom, f"{key} is missing from geom"
    
    # Check shapes
    assert geom["LAT"].shape == (JM_TEST,), f"LAT shape is incorrect"
    assert geom["LON"].shape == (IM_TEST,), f"LON shape is incorrect"
    assert geom["LAT2D"].shape == (JM_TEST, IM_TEST), f"LAT2D shape is incorrect"
    assert geom["SINLAT2D"].shape == (JM_TEST, IM_TEST), f"SINLAT2D shape is incorrect"
    assert geom["DXYP"].shape == (JM_TEST,), f"DXYP shape is incorrect"
    assert geom["aDXYP"].shape == (JM_TEST, IM_TEST), f"aDXYP shape is incorrect"
    assert geom["RAPJ"].shape == (IM_TEST, JM_TEST), f"RAPJ shape is incorrect"
    assert geom["KMAXJ"].shape == (JM_TEST,), f"KMAXJ shape is incorrect"
    assert geom["IMAXJ"].shape == (JM_TEST,), f"IMAXJ shape is incorrect"
    assert geom["IDJJ"].shape == (IM_TEST, JM_TEST), f"IDJJ shape is incorrect"
    assert geom["IDIJ"].shape == (IM_TEST, JM_TEST, 4), f"IDIJ shape is incorrect"
    
    print("✅ initialize_geom test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing GEOM JAX Implementation")
    print("=" * 70)
    
    test_constants()
    test_compute_lat_lon()
    test_compute_sinlat2d_coslat2d()
    test_compute_rapj()
    test_compute_dxyp()
    test_compute_axyp()
    test_compute_kmaxj()
    test_compute_imaxj()
    test_initialize_geom()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
