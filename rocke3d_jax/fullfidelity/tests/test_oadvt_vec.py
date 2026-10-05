"""Tests for the batched OADVTX2/OADVTY2/OADVTZ2 (Stage 2, D80-D81) against the scalar port and
the real Fortran OADVT2 dumps."""
import glob

import numpy as np
import pytest

from oadvt2_ff import oadvtx2, oadvty2, oadvtz2
from oadvt_vec import oadvtx2_vec, oadvty2_vec, oadvtz2_vec, oadvt2_vec
from oadvt2_compare import load_oadvt2_before, load_oadvt2_after, load_mmi, _load_smfinal
from odhorz_compare import load_lmm, load_lmv, load_lmu, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
TRACERS = [("g0m", "gxmo", "gymo", "gzmo", False), ("s0m", "sxmo", "symo", "szmo", True)]

pytestmark = pytest.mark.skipif(not glob.glob(f"{FF_DEFAULT}/*/ffz_oadvt2_before_*.bin"),
                                reason="ff_data oadvt2 dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    before = load_oadvt2_before(f"{ff}/ffz_oadvt2_before_{itime}.bin")
    after = load_oadvt2_after(f"{ff}/ffz_oadvt2_after_{itime}.bin")
    smu, smv = _load_smfinal(ff, itime)
    return lmm, lmv, lmu, before, after, load_mmi(date, itime), smu, smv


@pytest.mark.parametrize("date,itime", DATES)
@pytest.mark.parametrize("tracer", TRACERS, ids=["g0m", "s0m"])
def test_x_y_z_match_scalar_port(date, itime, tracer):
    lmm, lmv, lmu, before, _, mmi, smu, smv = _load(date, itime)
    *names, ql = tracer
    dt = before["dtdum"]
    rm, rx, ry, rz = [before[n] for n in names]
    ref_x = oadvtx2(rm, rx, ry, rz, mmi.copy(), smu, 0.5 * dt, ql, lmu, lmm)
    got_x = oadvtx2_vec(rm, rx, ry, rz, mmi.copy(), smu, 0.5 * dt, ql, lmu, lmm)
    for g, r in zip(got_x, ref_x):
        assert np.allclose(g, r, rtol=1e-12, atol=0.0), "X sweep"
    rm, rx, ry, rz, ma = ref_x
    ref_y = oadvty2(rm, rx, ry, rz, ma, smv, dt, ql, lmm, lmv)
    got_y = oadvty2_vec(rm, rx, ry, rz, ma, smv, dt, ql, lmm, lmv)
    for g, r in zip(got_y, ref_y):
        assert np.allclose(g, r, rtol=1e-12, atol=0.0), "Y sweep"
    ref_z = oadvtz2(*ref_y, before["smw"], dt, ql, lmm)
    got_z = oadvtz2_vec(*ref_y, before["smw"], dt, ql, lmm)
    for g, r in zip(got_z, ref_z):
        assert np.allclose(g, r, rtol=1e-12, atol=0.0), "Z sweep"


@pytest.mark.parametrize("date,itime", DATES)
@pytest.mark.parametrize("tracer", TRACERS, ids=["g0m", "s0m"])
def test_full_oadvt2_vec_matches_real_fortran(date, itime, tracer):
    lmm, lmv, lmu, before, after, mmi, smu, smv = _load(date, itime)
    *names, ql = tracer
    _, rm, rx, ry, rz = oadvt2_vec(mmi, *[before[n] for n in names], before["dtdum"], ql,
                                   smu, smv, before["smw"], lmu, lmv, lmm)
    for got, n in zip((rm, rx, ry, rz), names):
        assert np.allclose(got, after[n], atol=1e-6, rtol=1e-6), n
