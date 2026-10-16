"""Tests for reduce_fig_ff.py (OCNKPP.f REDUCE_FIG) against real Fortran per-call dumps --
Stage 2, D57. Plain-Python only."""
import glob

import numpy as np
import pytest

from reduce_fig_ff import reduce_fig, _exponent
from odhorz0_compare import FF_DEFAULT

RF_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_rf_*.bin"))

pytestmark = pytest.mark.skipif(not RF_PATHS, reason="ff_data REDUCE_FIG dumps not present on this host")


def _records(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 4)
    return [(int(round(n)), float(a), float(b)) for _, n, a, b in raw]


@pytest.mark.parametrize("path", RF_PATHS)
def test_reduce_fig_matches_real_fortran(path):
    for nsig, rin, rout in _records(path):
        assert reduce_fig(nsig, rin) == rout


def test_reduce_fig_changes_some_real_values():
    """Non-vacuous: on real records the routine must actually modify values, not just pass them
    through."""
    changed = sum(1 for nsig, rin, rout in _records(RF_PATHS[0]) if rin != rout)
    assert changed > 0


def test_reduce_fig_leaves_high_exponent_values_alone():
    """Guard branch: when NSIG+30 <= EXPONENT(x), the value is returned unchanged."""
    x = 2.0**45 + 0.5
    assert reduce_fig(10, x) == x
    assert reduce_fig(10, 0.0) == 0.0


def test_reduce_fig_rounds_small_values():
    """Guard branch: when NSIG+30 > EXPONENT(x), the value is reduced to NSIG significant bits."""
    assert reduce_fig(10, 3000.7) == 3072.0


def test_exponent_matches_fortran_convention():
    assert _exponent(1.0) == 1  # 1.0 = 0.5 * 2**1
    assert _exponent(0.0) == 0
