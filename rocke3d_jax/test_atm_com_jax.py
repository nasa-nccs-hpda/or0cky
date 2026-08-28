"""
Unit Tests for ATM_COM JAX Implementation
==========================================

This script tests the JAX implementation of the atmospheric common variables
from ROCKE-3D's ATM_COM.f. It validates the following:
1. Constants are correctly defined.
2. Helper functions work correctly.
3. Array initialization works as expected.

Usage:
    python test_atm_com_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from atm_com_jax import (
    LM_REQ, REQ_FAC, REQ_FAC_M, REQ_FAC_D, LM, LM_TOTAL,
    compute_byMA, compute_pk_pek, compute_pmid_pedn, compute_pdsig,
    initialize_atm_com,
)


# Test constants
np.random.seed(42)

# Grid size
I = 2
J = 2
LM_TEST = 10


def test_constants():
    """Test model resolution constants."""
    print("Testing constants...")
    
    # Check LM_REQ
    assert LM_REQ == 3, "LM_REQ should be 3"
    
    # Check REQ_FAC
    assert REQ_FAC.shape == (LM_REQ - 1,), "REQ_FAC shape is incorrect"
    assert jnp.allclose(REQ_FAC, jnp.array([0.5, 0.2])), "REQ_FAC values are incorrect"
    
    # Check REQ_FAC_M
    assert REQ_FAC_M.shape == (LM_REQ,), "REQ_FAC_M shape is incorrect"
    assert jnp.allclose(REQ_FAC_M, jnp.array([0.75, 0.35, 0.1])), "REQ_FAC_M values are incorrect"
    
    # Check REQ_FAC_D
    assert REQ_FAC_D.shape == (LM_REQ,), "REQ_FAC_D shape is incorrect"
    assert jnp.allclose(REQ_FAC_D, jnp.array([0.5, 0.3, 0.2])), "REQ_FAC_D values are incorrect"
    
    # Check LM and LM_TOTAL
    assert LM == 40, "LM should be 40"
    assert LM_TOTAL == LM + LM_REQ, "LM_TOTAL should be LM + LM_REQ"
    
    print("✅ Constants test passed!")


def test_compute_byMA():
    """Test compute_byMA function."""
    print("Testing compute_byMA...")
    
    # Create test MA array
    MA = np.random.rand(I, J, LM_TEST).astype(np.float32) * 10000.0 + 1000.0
    
    # Compute byMA
    byMA = compute_byMA(MA)
    
    # Check shape
    assert byMA.shape == (I, J, LM_TEST), f"Expected shape {(I, J, LM_TEST)} for byMA, got {byMA.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(byMA)), "byMA contains NaN"
    assert not jnp.any(jnp.isinf(byMA)), "byMA contains Inf"
    
    # Check values (byMA = 1/MA)
    expected_byMA = 1.0 / MA
    assert jnp.allclose(byMA, expected_byMA, rtol=1e-5), "byMA values are incorrect"
    
    print("✅ compute_byMA test passed!")


def test_compute_pk_pek():
    """Test compute_pk_pek function."""
    print("Testing compute_pk_pek...")
    
    # Create test PMID and PEDN arrays
    PMID = np.random.rand(I, J, LM_TEST).astype(np.float32) * 1000.0 + 100.0
    PEDN = np.random.rand(I, J, LM_TEST + 1).astype(np.float32) * 1000.0 + 100.0
    
    # Compute PK and PEK
    PK, PEK = compute_pk_pek(PMID, PEDN)
    
    # Check shapes
    assert PK.shape == (I, J, LM_TEST), f"Expected shape {(I, J, LM_TEST)} for PK, got {PK.shape}"
    assert PEK.shape == (I, J, LM_TEST + 1), f"Expected shape {(I, J, LM_TEST + 1)} for PEK, got {PEK.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(PK)), "PK contains NaN"
    assert not jnp.any(jnp.isnan(PEK)), "PEK contains NaN"
    assert not jnp.any(jnp.isinf(PK)), "PK contains Inf"
    assert not jnp.any(jnp.isinf(PEK)), "PEK contains Inf"
    
    # Check values (PK = PMID**KAPA, PEK = PEDN**KAPA)
    from constant_jax import KAPA
    expected_PK = PMID ** KAPA
    expected_PEK = PEDN ** KAPA
    assert jnp.allclose(PK, expected_PK, rtol=1e-5), "PK values are incorrect"
    assert jnp.allclose(PEK, expected_PEK, rtol=1e-5), "PEK values are incorrect"
    
    print("✅ compute_pk_pek test passed!")


def test_compute_pmid_pedn():
    """Test compute_pmid_pedn function."""
    print("Testing compute_pmid_pedn...")
    
    # Create test P and DSIG arrays
    P = np.random.rand(I, J).astype(np.float32) * 1000.0 + 1000.0
    DSIG = np.random.rand(I, J, LM_TEST).astype(np.float32) * 100.0 + 50.0
    
    # Compute PMID and PEDN
    PMID, PEDN = compute_pmid_pedn(P, P, DSIG)
    
    # Check shapes
    assert PMID.shape == (I, J, LM_TEST), f"Expected shape {(I, J, LM_TEST)} for PMID, got {PMID.shape}"
    assert PEDN.shape == (I, J, LM_TEST + 1), f"Expected shape {(I, J, LM_TEST + 1)} for PEDN, got {PEDN.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(PMID)), "PMID contains NaN"
    assert not jnp.any(jnp.isnan(PEDN)), "PEDN contains NaN"
    
    # Check PEDN(1) = P (surface pressure)
    assert jnp.allclose(PEDN[:, :, 0], P, rtol=1e-5), "PEDN(1) should equal P"
    
    print("✅ compute_pmid_pedn test passed!")


def test_compute_pdsig():
    """Test compute_pdsig function."""
    print("Testing compute_pdsig...")
    
    # Create test P and DSIG arrays
    P = np.random.rand(I, J).astype(np.float32) * 1000.0 + 1000.0
    DSIG = np.random.rand(I, J, LM_TEST).astype(np.float32) * 100.0 + 50.0
    
    # Compute PDSIG
    PDSIG = compute_pdsig(P, DSIG)
    
    # Check shape
    assert PDSIG.shape == (I, J, LM_TEST), f"Expected shape {(I, J, LM_TEST)} for PDSIG, got {PDSIG.shape}"
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(PDSIG)), "PDSIG contains NaN"
    assert not jnp.any(jnp.isinf(PDSIG)), "PDSIG contains Inf"
    
    # Check values (PDSIG = P * DSIG)
    expected_PDSIG = P[:, :, jnp.newaxis] * DSIG
    assert jnp.allclose(PDSIG, expected_PDSIG, rtol=1e-5), "PDSIG values are incorrect"
    
    print("✅ compute_pdsig test passed!")


def test_initialize_atm_com():
    """Test initialize_atm_com function."""
    print("Testing initialize_atm_com...")
    
    # Initialize ATM_COM arrays
    atm_com = initialize_atm_com(I, J, LM_TEST)
    
    # Check all arrays are present
    expected_keys = [
        "MA", "MAOLD", "U", "V", "T", "Q",
        "QCL", "QCI", "P", "ZATMO",
        "PMID", "PEDN", "PK", "PEK", "BYMA",
    ]
    for key in expected_keys:
        assert key in atm_com, f"{key} is missing from atm_com"
    
    # Check shapes
    assert atm_com["MA"].shape == (I, J, LM_TEST + LM_REQ), f"MA shape is incorrect"
    assert atm_com["P"].shape == (I, J), f"P shape is incorrect"
    assert atm_com["PMID"].shape == (I, J, LM_TEST + LM_REQ), f"PMID shape is incorrect"
    assert atm_com["PEDN"].shape == (I, J, LM_TEST + LM_REQ + 1), f"PEDN shape is incorrect"
    
    # Check all arrays are initialized to zero
    for key in expected_keys:
        assert jnp.all(atm_com[key] == 0.0), f"{key} should be initialized to zero"
    
    print("✅ initialize_atm_com test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ATM_COM JAX Implementation")
    print("=" * 70)
    
    test_constants()
    test_compute_byMA()
    test_compute_pk_pek()
    test_compute_pmid_pedn()
    test_compute_pdsig()
    test_initialize_atm_com()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
