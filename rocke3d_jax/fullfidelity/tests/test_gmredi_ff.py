"""Tests for gmredi_ff.py's isoslope4 port against real Fortran dumps -- Stage 2 of the
DYNSI/ocean port, D49 (the first piece of the Gent-McWilliams mesoscale-mixing scheme itself,
scoped D46/D48). Plain-Python only."""
import glob

import numpy as np
import pytest

from gmredi_ff import isoslope4, IM, JM, LMO
from gmredi_compare import load_isoslope4, ISOSLOPE4_FIELDS, FF_DEFAULT
from ocnmeso_compare import load_densgrad
from odhorz_compare import load_lmm

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
ISOSLOPE4_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_isoslope4_*.bin"))

pytestmark = pytest.mark.skipif(not ISOSLOPE4_PATHS, reason="ff_data isoslope4 dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    expected = load_isoslope4(f"{ff}/ffz_isoslope4_{itime}.bin")
    k3d = np.where(np.arange(LMO + 1)[None, None, :] <= lmm[:, :, None], 800.0, 0.0)
    k3d[:, :, 0] = 0.0
    return lmm, dg, expected, k3d


@pytest.mark.parametrize("date,itime", DATES)
def test_isoslope4_matches_real_fortran(date, itime):
    lmm, dg, expected, k3d = _load(date, itime)
    computed = isoslope4(lmm, dg["rhox"], dg["rhoy"], dg["rhomz"], dg["byrhoz"],
                          dg["bydh"], dg["dzv"], k3d)
    for name in ISOSLOPE4_FIELDS:
        assert np.allclose(computed[name], expected[name], atol=1e-6, rtol=1e-6), \
            f"{date} {name.upper()}"


@pytest.mark.parametrize("date,itime", DATES)
def test_isoslope4_pole_row_is_exercised(date, itime):
    """Regression pin for the real bug this delta found: ISOSLOPE4's main loop, unlike almost
    every other routine ported in Stage 2, is NOT restricted to J=2..JM-1 -- it genuinely
    computes at J=JM (I=1) too. A first draft assumed the usual pole-exclusion convention and
    silently left the pole row at its zero-initialized value, matching by coincidence at cells
    where the real ASX/S2X output also happened to be zero (RHOX/RHOY=0 there) while missing a
    real nonzero AIX0/AIY0/etc value."""
    lmm, dg, expected, k3d = _load(date, itime)
    computed = isoslope4(lmm, dg["rhox"], dg["rhoy"], dg["rhomz"], dg["byrhoz"],
                          dg["bydh"], dg["dzv"], k3d)
    assert np.any(expected["aix0"][:, JM, :] != 0.0), f"{date}: real AIX0 is all-zero at the pole"
    assert np.allclose(computed["aix0"][:, JM, :], expected["aix0"][:, JM, :],
                        atol=1e-6, rtol=1e-6), f"{date}: AIX0 mismatch at the pole row"


@pytest.mark.parametrize("date,itime", DATES)
def test_isoslope4_nonvacuous(date, itime):
    """The slope-limiting branch (ARIV>0 and slope^2 > limit) and the ordinary (unlimited)
    branch must both actually fire somewhere in real data."""
    lmm, dg, expected, k3d = _load(date, itime)
    assert np.any(expected["asx0"] != 0.0), f"{date}: ASX0 never nonzero"
    assert np.any(expected["s2y0"] != 0.0), f"{date}: S2Y0 never nonzero"
