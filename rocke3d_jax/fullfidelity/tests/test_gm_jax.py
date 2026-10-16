"""Tests for the JAX Gent-McWilliams port (Stage 2, D85) against gm_vec (numpy) on the real dumps.
Skipped if ff_data is absent."""
import glob

import numpy as np
import pytest

from gm_vec import isoslope4_vec, gmkdif_vec, gmfexp_vec
from gm_jax import isoslope4_jax, gmkdif_jax, gmfexp_jax
from gm_vec_compare import load_all, iso_args, kdif_args, fexp_args, ISO_ORDER
from gm_jax_compare import to_jax
from gmkdif_compare import GMKDIF_FIELDS
from ocnmeso_compare import FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
TOL = 1e-10

pytestmark = pytest.mark.skipif(not glob.glob(f"{FF_DEFAULT}/*/ffz_gmfexp_*.bin"),
                                reason="ff_data gm dumps not present on this host")


def max_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a - b)
    rel = np.where(b != 0.0, d / np.where(b != 0.0, np.abs(b), 1.0), np.where(d == 0.0, 0.0, np.inf))
    return float(rel.max())


@pytest.mark.parametrize("date,itime", DATES)
def test_isoslope4_jax_matches_vec(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    a = iso_args(lmm, dg, k3d)
    rv, rj = isoslope4_vec(*a), isoslope4_jax(*to_jax(a))
    for n in ISO_ORDER:
        assert max_rel(rj[n], rv[n]) <= TOL, n
    assert any(np.any(rv[n] != 0.0) for n in ISO_ORDER)


@pytest.mark.parametrize("date,itime", DATES)
def test_gmkdif_jax_matches_vec(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    a = kdif_args(lmm, gmk, isos)
    rv, rj = gmkdif_vec(*a), gmkdif_jax(*to_jax(a))
    for n in GMKDIF_FIELDS:
        assert max_rel(rj[n], rv[n]) <= TOL, n


@pytest.mark.parametrize("date,itime", DATES)
def test_gmfexp_jax_matches_vec_both_qlimit(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    assert sorted(bool(r["qlimit"]) for r in recs) == [False, True]
    for rec in recs:
        a = fexp_args(lmm, lmu, lmv, rec, gmk, dg)
        rv, rj = gmfexp_vec(*a), gmfexp_jax(*to_jax(a))
        for name, g, r in zip(("TRM", "TXM", "TYM", "TZM"), rj, rv):
            assert max_rel(g, r) <= TOL, f"{name} qlimit={rec['qlimit']}"
        assert np.any(~np.isclose(np.asarray(rj[0]), rec["trm0"]))
