"""Tests for the jitted OADVT2 sweeps (Stage 2, D84) against oadvt_vec and the real Fortran dumps."""
import glob

import numpy as np
import pytest

from oadvt_vec import oadvt2_vec
from oadvt_jax import oadvt2_jax
from oadvt2_compare import load_oadvt2_before, load_oadvt2_after, load_mmi, _load_smfinal
from odhorz_compare import load_lmm, load_lmv, load_lmu, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
TRACERS = [("g0m", "gxmo", "gymo", "gzmo", False), ("s0m", "sxmo", "symo", "szmo", True)]

pytestmark = pytest.mark.skipif(not glob.glob(f"{FF_DEFAULT}/*/ffz_oadvt2_before_*.bin"),
                                reason="ff_data oadvt2 dumps not present on this host")


@pytest.mark.parametrize("date,itime", DATES)
@pytest.mark.parametrize("tracer", TRACERS, ids=["g0m", "s0m"])
def test_jax_matches_vec_and_real_fortran(date, itime, tracer):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    before = load_oadvt2_before(f"{ff}/ffz_oadvt2_before_{itime}.bin")
    after = load_oadvt2_after(f"{ff}/ffz_oadvt2_after_{itime}.bin")
    smu, smv = _load_smfinal(ff, itime)
    *names, ql = tracer
    args = (load_mmi(date, itime), *[before[n] for n in names], before["dtdum"], ql, smu, smv,
            before["smw"], lmu, lmv, lmm)
    got = oadvt2_jax(*args)
    ref = oadvt2_vec(*args)
    for g, r in zip(got, ref):
        g = np.asarray(g)
        assert np.max(np.abs(g - r)) <= 1e-12 * max(np.max(np.abs(r)), 1e-300), "jax vs vec"
    for g, n in zip(got[1:], names):
        assert np.allclose(np.asarray(g), after[n], atol=1e-6, rtol=1e-6), n
