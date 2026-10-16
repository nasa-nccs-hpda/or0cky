"""Tests for the batched Gent-McWilliams port (Stage 2, D83) against the scalar gmredi_ff port
on the real dumps. Skipped if ff_data is absent."""
import glob

import numpy as np
import pytest

from gmredi_ff import isoslope4, gmkdif, gmfexp
from gm_vec import isoslope4_vec, gmkdif_vec, gmfexp_vec
from gm_vec_compare import load_all, iso_args, kdif_args, fexp_args, ISO_ORDER
from gmkdif_compare import GMKDIF_FIELDS
from ocnmeso_compare import FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]

pytestmark = pytest.mark.skipif(not glob.glob(f"{FF_DEFAULT}/*/ffz_gmfexp_*.bin"),
                                reason="ff_data gm dumps not present on this host")


@pytest.mark.parametrize("date,itime", DATES)
def test_isoslope4_matches_scalar(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    a = iso_args(lmm, dg, k3d)
    rs, rv = isoslope4(*a), isoslope4_vec(*a)
    for n in ISO_ORDER:
        assert np.allclose(rv[n], rs[n], rtol=1e-12, atol=0.0), n
    assert any(np.any(rs[n] != 0.0) for n in ISO_ORDER)


@pytest.mark.parametrize("date,itime", DATES)
def test_gmkdif_matches_scalar(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    a = kdif_args(lmm, gmk, isos)
    rs, rv = gmkdif(*a), gmkdif_vec(*a)
    for n in GMKDIF_FIELDS:
        assert np.allclose(rv[n], rs[n], rtol=1e-12, atol=0.0), n


@pytest.mark.parametrize("date,itime", DATES)
def test_gmfexp_matches_scalar_both_calls(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    assert sorted(bool(r["qlimit"]) for r in recs) == [False, True]
    for rec in recs:
        a = fexp_args(lmm, lmu, lmv, rec, gmk, dg)
        rs, rv = gmfexp(*a), gmfexp_vec(*a)
        for name, g, r in zip(("TRM", "TXM", "TYM", "TZM"), rv, rs):
            assert np.allclose(g, r, rtol=1e-12, atol=0.0), f"{name} qlimit={rec['qlimit']}"
        assert np.any(~np.isclose(rv[0], rec["trm0"]))
