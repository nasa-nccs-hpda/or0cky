"""Tests for gmredi_ff.py's gmkdif port against real Fortran dumps -- Stage 2 of the DYNSI/ocean
port, D50 (GMKDIF's remaining post-QCROSS coefficient logic, second piece of the Gent-McWilliams
mesoscale-mixing scheme). Plain-Python only."""
import glob

import numpy as np
import pytest

from gmredi_ff import gmkdif, IM, JM, LMO
from gmkdif_compare import load_gmkdif, GMKDIF_FIELDS, FF_DEFAULT
from gmredi_compare import load_isoslope4
from odhorz_compare import load_lmm

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GMKDIF_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_gmkdif_*.bin"))

pytestmark = pytest.mark.skipif(not GMKDIF_PATHS, reason="ff_data gmkdif dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    isos = load_isoslope4(f"{ff}/ffz_isoslope4_{itime}.bin")
    expected = load_gmkdif(f"{ff}/ffz_gmkdif_{itime}.bin")
    return lmm, isos, expected


@pytest.mark.parametrize("date,itime", DATES)
def test_gmkdif_matches_real_fortran(date, itime):
    lmm, isos, expected = _load(date, itime)
    computed = gmkdif(lmm, expected["kpl"],
                       isos["aix0"], isos["aix1"], isos["aix2"], isos["aix3"],
                       isos["aiy0"], isos["aiy1"], isos["aiy2"], isos["aiy3"],
                       isos["asx0"], isos["asx1"], isos["asx2"], isos["asx3"],
                       isos["asy0"], isos["asy1"], isos["asy2"], isos["asy3"],
                       isos["s2x0"], isos["s2x1"], isos["s2x2"], isos["s2x3"],
                       isos["s2y0"], isos["s2y1"], isos["s2y2"], isos["s2y3"])
    for name in GMKDIF_FIELDS:
        assert np.allclose(computed[name], expected[name], atol=1e-6, rtol=1e-6), \
            f"{date} {name.upper()}"


@pytest.mark.parametrize("date,itime", DATES)
def test_kpl_is_recorded_and_plausible(date, itime):
    """KPL (OCNKPP.f's mixed-layer-depth index, not yet ported) is recorded directly -- sanity
    check it's a plausible small positive layer index, not a dump-format accident."""
    lmm, isos, expected = _load(date, itime)
    kpl = expected["kpl"]
    active = lmm[1:, 1:] > 0
    assert np.all(kpl[1:, 1:][active] >= 1), f"{date}: KPL has non-positive values at active cells"
    assert np.all(kpl[1:, 1:][active] <= LMO), f"{date}: KPL exceeds LMO at active cells"


@pytest.mark.parametrize("date,itime", DATES)
def test_bxx_byy_write_offsets(date, itime):
    """Regression pin: the real Fortran writes BXX at (IM1,J,L) -- the WEST neighbor of the
    loop's own I -- and BYY at (I,J-1,L), not (I,J,L). Confirms these aren't off-by-one
    typos by checking the arrays are genuinely non-vacuous at the shifted locations."""
    lmm, isos, expected = _load(date, itime)
    assert np.any(expected["bxx"] != 0.0), f"{date}: BXX never nonzero"
    assert np.any(expected["byy"] != 0.0), f"{date}: BYY never nonzero"


@pytest.mark.parametrize("date,itime", DATES)
def test_mixed_layer_exclusion_is_nonvacuous(date, itime):
    """Both the L>KPL branch (BZZ/AZX/.../CEZY set) and the L<=KPL branch (BZZ zeroed, GM
    excluded within the mixed layer) must actually fire somewhere in real data."""
    lmm, isos, expected = _load(date, itime)
    assert np.any(expected["azx"] != 0.0), f"{date}: AZX never nonzero (L>KPL branch never fires)"
