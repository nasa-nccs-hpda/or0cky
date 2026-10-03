"""Tests for ovdiff_ff.py (OCNKPP.f momentum OVDIFF) against real Fortran per-call dumps --
Stage 2, D56. Plain-Python only."""
import glob

import numpy as np
import pytest

from ovdiff_ff import ovdiff
from ovdiffs_compare import load_ovdiffs_records
from odhorz0_compare import FF_DEFAULT

OVDIFF_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_ovdiff_*.bin"))

pytestmark = pytest.mark.skipif(not OVDIFF_PATHS, reason="ff_data ovdiff dumps not present on this host")


@pytest.mark.parametrize("path", OVDIFF_PATHS)
def test_ovdiff_matches_real_fortran(path):
    max_u = 0.0
    tags = set()
    for r in load_ovdiffs_records(path):
        tags.add(r["tag"])
        n = r["lmij"]
        u = ovdiff(r["k"], r["ghat"], r["dtp4"], r["dtbydz"], r["bydz2"], n, r["u0"])
        max_u = max(max_u, float(np.max(np.abs(u[1:n + 1] - r["u_real"][1:n + 1]))))
    assert max_u == 0.0, f"{path}: max abs u error {max_u}"
    assert tags <= {2, 3}


def test_ovdiff_both_uv_grids_present():
    """Regression pin: the real record set exercises both the U-point (tag 2) and D-grid (tag 3)
    momentum calls from OCONV, not just one."""
    tags = set()
    for path in OVDIFF_PATHS[:3]:
        tags |= {r["tag"] for r in load_ovdiffs_records(path)}
    assert tags == {2, 3}
