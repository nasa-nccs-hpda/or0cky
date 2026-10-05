"""Tests for the batched OCNMESO inputs (Stage 2, D82) against the scalar port."""
import glob

import numpy as np
import pytest

from ocnmeso_ff import ocnstate_derived, densgrad_vertical, get_1d_mesodiff
from ocnmeso_vec import ocnstate_derived_vec, densgrad_vertical_vec, get_1d_mesodiff_vec
from ocnmeso_compare import load_ocnstate_derived, load_densgrad
from odhorz_compare import load_lmm, FF_DEFAULT
from odhorz0_compare import load_record as load_odhorz0_record

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]

pytestmark = pytest.mark.skipif(not glob.glob(f"{FF_DEFAULT}/*/ffz_densgrad_*.bin"),
                                reason="ff_data ocnmeso dumps not present on this host")


@pytest.mark.parametrize("date,itime", DATES)
def test_matches_scalar_port(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    osd = load_ocnstate_derived(f"{ff}/ffz_ocnstate_derived_{itime}.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    dh = load_odhorz0_record(f"{ff}/ffz_odhorz0_{itime}.bin")["dh3d"]
    a1 = (osd["mo0"], osd["g0m0"], osd["gzm0"], osd["s0m0"], osd["szm0"], osd["opress0"], lmm,
          osd["vup"], osd["vdn"])
    a2 = (lmm, dh, dg["vbar"], dg["vup"], dg["vdn"], dg["vupu"], dg["vdnu"])
    for g, r in zip(ocnstate_derived_vec(*a1), ocnstate_derived(*a1)):
        assert np.allclose(g, r, rtol=1e-12, atol=0.0), "ocnstate_derived"
    for g, r in zip(densgrad_vertical_vec(*a2), densgrad_vertical(*a2)):
        assert np.allclose(g, r, rtol=1e-12, atol=0.0), "densgrad_vertical"
    assert np.array_equal(get_1d_mesodiff_vec(lmm), get_1d_mesodiff(lmm))
