"""Tests for icedyn_jax.py (D88): jitted PLAST/FORM/RELAX/TRIDIAG/VPICEDYN vs icedyn_vec (to ~1e-10) and
the real Fortran outputs (same tolerances as test_dynsi_ff.py), on all 18 real records."""
import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import icedyn_jax_compare as JC
import icedyn_vec_compare as K

DATES = {"nov26": range(33312, 33318), "dec01": range(33552, 33558), "jan01": range(17520, 17526)}
CASES = [(d, it) for d, its in DATES.items() for it in its]

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(K.FF_DATA) and glob.glob(f"{K.FF_DATA}/nov26/ffz_geom.bin")),
    reason="ff_data dumps not present on this host")


@pytest.mark.parametrize("date,itime", [(d, its[0]) for d, its in DATES.items()])
def test_stages_match_vec(date, itime):
    out = JC.stage_checks(date, itime, verbose=False)
    for name, w in out.items():
        assert w < 1e-10, f"{date} {name} {w:.2e}"


@pytest.mark.parametrize("date,itime", CASES)
def test_vpicedyn_jax_matches_vec_and_real(date, itime):
    r = JC.full_checks(date, itime, verbose=False)
    assert r["kki_v"] == r["kki_j"]
    assert r["vs_vec"] < 1e-10
    for name, (max_rel, mean_rel) in r["real"].items():
        assert max_rel < 1e-6, f"{date}/{itime} {name} max_relerr={max_rel:.3e}"
        assert mean_rel < 1e-8, f"{date}/{itime} {name} mean_relerr={mean_rel:.3e}"
