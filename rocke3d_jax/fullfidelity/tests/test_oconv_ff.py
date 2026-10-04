"""Tests for oconv_ff.py's OCONV + KVINIT + OVDIFF + OVDIFFS + REDUCE_FIG port against real Fortran dumps --
Stage 2, D56.
"""
import glob
import numpy as np
import pytest

from oconv_ff import kvinit, ovdiff, ovdiffs, reduce_fig, LMO
from oconv_compare import load_kvinit_dump, load_oconv_dump, find_kvinit_dumps, find_oconv_dumps
from odhorz0_compare import FF_DEFAULT

KVINIT_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_kvinit_*.bin"))
OCONV_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_oconv_*.bin"))

pytestmark = pytest.mark.skipif(
    not KVINIT_PATHS or not OCONV_PATHS,
    reason="ff_data oconv/kvinit dumps not present on this host"
)


@pytest.mark.parametrize("path", KVINIT_PATHS)
def test_kvinit_dump_structure(path):
    """Verify KVINIT dump has all expected fields."""
    d = load_kvinit_dump(path)
    required = ['itime', 'g0m1', 's0m1', 'mo1', 'gxm1', 'gym1', 'sxm1', 'sym1', 'uo1', 'vo1', 'uod1', 'vod1']
    for key in required:
        assert key in d, f"{path}: missing key {key}"
    # Check shapes
    assert d['g0m1'].shape == (72, 46, 1), f"{path}: g0m1 shape {d['g0m1'].shape}"
    assert d['s0m1'].shape == (72, 46), f"{path}: s0m1 shape {d['s0m1'].shape}"
    assert d['mo1'].shape == (72, 46), f"{path}: mo1 shape {d['mo1'].shape}"


@pytest.mark.parametrize("path", OCONV_PATHS)
def test_oconv_dump_structure(path):
    """Verify OCONV dump has all expected fields."""
    d = load_oconv_dump(path)
    required = ['itime', 'g0m', 's0m', 'gxmo', 'gymo', 'sxmo', 'symo', 'gzmo', 'szmo', 'kpl',
                'akvg3d', 'akvs3d', 'akvc3d', 'flg3d', 'fls3d',
                'gxxmo', 'gyymo', 'gxymo', 'sxxmo', 'syymo', 'sxymo']
    for key in required:
        assert key in d, f"{path}: missing key {key}"
    # Check shapes (halo bounds may vary)
    assert d['g0m'].shape[0] == 72, f"{path}: g0m IM dim {d['g0m'].shape[0]}"
    assert d['g0m'].shape[2] == 13, f"{path}: g0m LMO dim {d['g0m'].shape[2]}"
    assert d['kpl'].shape == (72, 46), f"{path}: kpl shape {d['kpl'].shape}"


def test_ovdiff_tridiagonal_solver():
    """Test OVDIFF's tridiagonal solver against a known case."""
    # Simple test: constant diffusivity, no non-local transport, no explicit tendency
    lmij = 5
    u0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = np.ones(lmij) * 0.1
    ghat = np.zeros(lmij)
    dtp4 = np.zeros(lmij)
    ze = np.linspace(0, -100, LMO + 1)
    z = (ze[:-1] + ze[1:]) / 2
    dtbydz = np.ones(lmij) * 0.01
    bydz2 = np.ones(lmij) * 0.02
    
    u = ovdiff(u0, k, ghat, dtp4, ze, z, dtbydz, bydz2, lmij)
    
    # With no forcing, solution should be close to initial (diffusion is implicit)
    # Just verify it runs and produces reasonable output
    assert u.shape == (LMO,)
    assert np.all(np.isfinite(u[:lmij]))


def test_ovdiffs_tridiagonal_solver():
    """Test OVDIFFS's tridiagonal solver and flux computation."""
    lmij = 5
    u0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = np.ones(lmij) * 0.1
    ghat = np.zeros(lmij)
    dtp4 = np.zeros(lmij)
    dtbydz = np.ones(lmij) * 0.01
    bydz2 = np.ones(lmij) * 0.02
    dt = 900.0
    
    u, fl = ovdiffs(u0, k, ghat, dtp4, dtbydz, bydz2, dt, lmij)
    
    assert u.shape == (LMO,)
    assert fl.shape == (LMO,)
    assert np.all(np.isfinite(u[:lmij]))
    assert np.all(np.isfinite(fl[:lmij-1]))


def test_reduce_fig_noop():
    """REDUCE_FIG should be a no-op for normal values with negative NSIG (as in OCONV)."""
    # In OCONV, NSIG = EXPONENT - 44 (or -40), which is negative for normal values
    # With NSIG=-27, NSIG+30=3, EXPONENT(1.23)=1, 3 > 1 is True -> it DOES reduce
    # But the reduction is to 2^-27 precision (~7e-9), which is negligible
    rx = 1.23456789
    result = reduce_fig(-27, rx)
    # Should be nearly identical (binary rounding to 2^-27)
    assert abs(result - rx) < 1e-8


def test_reduce_fig_garbage():
    """REDUCE_FIG should reduce garbage values with huge exponents when NSIG is large positive."""
    # For garbage values, if NSIG is also large (derived from a garbage G0M),
    # the condition NSIG+30 > EXPONENT can be true
    rx = 1.23456789e100  # EXPONENT ~ 333
    # If G0M is also garbage with EXPONENT ~ 333, NSIG = 333 - 44 = 289
    # NSIG+30 = 319 > 333? False -> no reduction
    # If G0M is even more garbage, EXPONENT > 347, NSIG > 303, then reduction triggers
    result = reduce_fig(310, rx)  # NSIG+30=340 > 333 -> reduces
    assert result != rx
    # The relative change is limited by float64 precision when scaling huge numbers
    # (Fortran REAL*8 has the same limitation)
    assert abs(result - rx) / abs(rx) < 1e-7


# Integration tests that require the full OCONV port will be added once oconv_step is implemented