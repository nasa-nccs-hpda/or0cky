"""Tests for ovdiffs_ff.py (OCNKPP.f OVDIFFS + plain TRIDIAG) against real Fortran per-call
dumps -- Stage 2, D55. Plain-Python only."""
import glob

import numpy as np
import pytest

from ovdiffs_ff import ovdiffs, tridiag
from ovdiffs_compare import load_ovdiffs_records
from odhorz0_compare import FF_DEFAULT

OVDIFFS_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_ovdiffs_*.bin"))

pytestmark = pytest.mark.skipif(not OVDIFFS_PATHS, reason="ff_data ovdiffs dumps not present on this host")


@pytest.mark.parametrize("path", OVDIFFS_PATHS)
def test_ovdiffs_matches_real_fortran(path):
    recs = load_ovdiffs_records(path)
    max_u = max_fl = 0.0
    for r in recs:
        u, fl = ovdiffs(r["k"], r["ghat"], r["dtp4"], r["dtbydz"], r["bydz2"], r["dt"], r["lmij"], r["u0"])
        n = r["lmij"]
        max_u = max(max_u, float(np.max(np.abs(u[1:n + 1] - r["u_real"][1:n + 1]))))
        if n > 1:
            max_fl = max(max_fl, float(np.max(np.abs(fl[1:n] - r["fl_real"][1:n]))))
    assert max_u == 0.0 and max_fl == 0.0, f"{path}: u {max_u}, fl {max_fl}"


def test_tridiag_solves_known_system():
    n = 4
    a = np.zeros(14); b = np.zeros(14); c = np.zeros(14); r = np.zeros(14)
    a[1:n + 1] = [0, -1, -1, -1]
    b[1:n + 1] = [2, 2, 2, 2]
    c[1:n + 1] = [-1, -1, -1, 0]
    x_true = np.zeros(14); x_true[1:n + 1] = [1.0, 2.0, 3.0, 4.0]
    r[1] = b[1] * x_true[1] + c[1] * x_true[2]
    for j in range(2, n):
        r[j] = a[j] * x_true[j - 1] + b[j] * x_true[j] + c[j] * x_true[j + 1]
    r[n] = a[n] * x_true[n - 1] + b[n] * x_true[n]
    u = tridiag(a, b, c, r, n)
    assert np.allclose(u[1:n + 1], x_true[1:n + 1])


def test_dtp4_is_zero_in_real_records():
    """Regression pin: DTP4G/DTP4S are only set inside OCNKPP.f's `#ifdef OCN_GISS_SM` block
    (dead for this build), so the real OVDIFFS calls always receive all-zero DTP4. That makes the
    source's `DTP4(LMIJ-1)` index in the bottom RHS (OCNKPP.f:3511) unexercised by this build, not
    live -- pinned against the real dumps rather than assumed from the source."""
    recs = load_ovdiffs_records(OVDIFFS_PATHS[0])
    assert all(np.all(r["dtp4"][1:r["lmij"] + 1] == 0.0) for r in recs)
