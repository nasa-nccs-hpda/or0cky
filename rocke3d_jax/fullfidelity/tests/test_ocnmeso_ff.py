"""Tests for ocnmeso_ff.py (OCNDYN2.f's ocnstate_derived + OCNMESO_DRV.f's
densgrad_vertical/get_1d_mesodiff) against real Fortran dumps -- Stage 2 of the DYNSI/ocean
port, D47. Prerequisite pieces for the Gent-McWilliams mesoscale-mixing scheme (GMKDIF/GMFEXP,
scoped D46, not yet ported). Plain-Python only."""
import glob

import numpy as np
import pytest

from ocnmeso_ff import (ocnstate_derived, densgrad_vertical, get_1d_mesodiff,
                         MESO_DIFFUSIVITY_CONST, IM, JM, LMO)
from ocnmeso_compare import load_ocnstate_derived, load_densgrad, FF_DEFAULT
from odhorz_compare import load_lmm
from odhorz0_compare import load_record as load_odhorz0_record

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
OCNMESO_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_ocnstate_derived_*.bin"))

pytestmark = pytest.mark.skipif(not OCNMESO_PATHS, reason="ff_data ocnmeso dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    osd = load_ocnstate_derived(f"{ff}/ffz_ocnstate_derived_{itime}.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    dh = load_odhorz0_record(f"{ff}/ffz_odhorz0_{itime}.bin")["dh3d"]
    return lmm, osd, dg, dh


@pytest.mark.parametrize("date,itime", DATES)
def test_ocnstate_derived_matches_real_fortran(date, itime):
    lmm, osd, dg, dh = _load(date, itime)
    g3d, s3d, p3d, vbar, rho = ocnstate_derived(
        osd["mo0"], osd["g0m0"], osd["gzm0"], osd["s0m0"], osd["szm0"],
        osd["opress0"], lmm, osd["vup"], osd["vdn"])
    for name, computed, expected in [("G3D", g3d, osd["g3d"]), ("S3D", s3d, osd["s3d"]),
                                      ("P3D", p3d, osd["p3d"]), ("VBAR", vbar, osd["vbar"]),
                                      ("RHO", rho, osd["rho"])]:
        assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_densgrad_vertical_matches_real_fortran(date, itime):
    lmm, osd, dg, dh = _load(date, itime)
    dzv, bydzv, bydh, rhomz, byrhoz = densgrad_vertical(
        lmm, dh, dg["vbar"], dg["vup"], dg["vdn"], dg["vupu"], dg["vdnu"])
    for name, computed, expected in [("DZV", dzv, dg["dzv"]), ("BYDZV", bydzv, dg["bydzv"]),
                                      ("BYDH", bydh, dg["bydh"]), ("RHOMZ", rhomz, dg["rhomz"]),
                                      ("BYRHOZ", byrhoz, dg["byrhoz"])]:
        assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_get_1d_mesodiff_matches_rundeck_constant(date, itime):
    """No real Fortran dump needed (this is a pure constant broadcast, same precedent as D29's
    RADIUS/GRAV) -- checks k3d equals meso_diffusivity_const=800 at every active cell and 0
    elsewhere, using this date's real LMM geometry."""
    lmm, osd, dg, dh = _load(date, itime)
    k3d = get_1d_mesodiff(lmm)
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            lm = lmm[i, j]
            if lm > 0:
                assert np.all(k3d[i, j, 1:lm + 1] == MESO_DIFFUSIVITY_CONST)
            assert np.all(k3d[i, j, lm + 1:] == 0.0)


@pytest.mark.parametrize("date,itime", DATES)
def test_ocnstate_derived_pole_uniformity(date, itime):
    """Both poles must be uniform across longitude after the real Fortran's explicit
    copy-to-all-longitudes step."""
    lmm, osd, dg, dh = _load(date, itime)
    g3d, s3d, p3d, vbar, rho = ocnstate_derived(
        osd["mo0"], osd["g0m0"], osd["gzm0"], osd["s0m0"], osd["szm0"],
        osd["opress0"], lmm, osd["vup"], osd["vdn"])
    for j in (1, JM):
        lm = lmm[1, j]
        if lm == 0:
            continue
        for name, arr in [("G3D", g3d), ("S3D", s3d), ("P3D", p3d), ("VBAR", vbar), ("RHO", rho)]:
            assert np.allclose(arr[1:IM + 1, j, 1:lm + 1], arr[1, j, 1:lm + 1][None, :]), \
                f"{date} {name} not uniform at pole J={j}"


@pytest.mark.parametrize("date,itime", DATES)
def test_ocnstate_derived_nonvacuous(date, itime):
    lmm, osd, dg, dh = _load(date, itime)
    assert np.any(lmm[1:, 1:] > 0), f"{date}: no active ocean columns"
    assert np.any(osd["g3d"] != 0.0), f"{date}: G3D never nonzero"
    assert np.any(osd["rho"] > 0.0), f"{date}: RHO never positive"
