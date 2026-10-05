"""Tests for icedyn_vec.py (D87): batched numpy PLAST/FORM/RELAX/TRIDIAG/VPICEDYN vs the scalar port
(icedyn_dynsi_ff) on real recorded inputs, and the full batched VPICEDYN vs the real Fortran outputs
with the same tolerances as test_dynsi_ff.py. All 18 real records (3 dates x 6 steps) are run: the
scalar VPICEDYN takes ~1.5 s per record, so the full parametrization is cheap."""
import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import icedyn_dynsi_ff as D
import icedyn_vec as V
import icedyn_vec_compare as K
import dynsi_compare as C

FF_DATA = K.FF_DATA
DATES = {"nov26": range(33312, 33318), "dec01": range(33552, 33558), "jan01": range(17520, 17526)}
CASES = [(d, it) for d, its in DATES.items() for it in its]

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(FF_DATA) and glob.glob(f"{FF_DATA}/nov26/ffz_geom.bin")),
    reason="ff_data dumps not present on this host")


@pytest.mark.parametrize("date,itime", [(d, its[0]) for d, its in DATES.items()])
def test_stages_match_scalar(date, itime):
    out = K.stage_checks(date, itime, verbose=False)
    for name in ("form", "plast", "relax", "thomas", "cyclic"):
        assert out[name][0] < 1e-12, f"{date} {name} {out[name][0]:.2e}"


@pytest.mark.parametrize("date,itime", CASES)
def test_vpicedyn_vec_matches_scalar_and_real(date, itime):
    r = K.full_checks(date, itime, verbose=False)
    assert r["kki_s"] == r["kki_v"]
    assert r["vs_scalar"] < 1e-12
    for name, (max_rel, mean_rel) in r["real"].items():
        assert max_rel < 1e-6, f"{date}/{itime} {name} max_relerr={max_rel:.3e}"
        assert mean_rel < 1e-8, f"{date}/{itime} {name} mean_relerr={mean_rel:.3e}"


def test_tridiag_batch_matches_dense_solve():
    rng = np.random.default_rng(3)
    n, L = 9, 5
    a, b, c, r = (rng.random((n, L)) + 2, rng.random((n, L)) + 10, rng.random((n, L)) + 2,
                  rng.random((n, L)))
    ut, uc = V.tridiag_thomas_batch(a, b, c, r), V.tridiag_cyclic_batch(a, b, c, r)
    for k in range(L):
        A = np.diag(b[:, k]) + np.diag(a[1:, k], -1) + np.diag(c[:-1, k], 1)
        assert np.max(np.abs(A @ ut[:, k] - r[:, k])) < 1e-10
        A[0, n - 1] = a[0, k]
        A[n - 1, 0] = c[n - 1, k]
        assert np.max(np.abs(A @ uc[:, k] - r[:, k])) < 1e-10


def test_cyclic_masked_line_branch():
    """A fully masked line (b==1, a=c=r=0) takes the b[0]==1 doubling branch per line."""
    n = 8
    a = np.zeros((n, 2)); b = np.ones((n, 2)); c = np.zeros((n, 2)); r = np.zeros((n, 2))
    b[:, 1] = 10.0; a[:, 1] = 1.0; c[:, 1] = 1.0; r[:, 1] = 1.0
    ref = np.stack([D.tridiag_cyclic(a[:, k], b[:, k], c[:, k], r[:, k]) for k in range(2)], 1)
    assert np.max(np.abs(V.tridiag_cyclic_batch(a, b, c, r) - ref)) < 1e-14
