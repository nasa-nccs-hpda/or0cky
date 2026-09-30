"""Tests for oadvt2_ff.py (OCNDYN2.f's OADVT2 dispatcher + OADVTX2/OADVTY2/OADVTZ2 -- the
long-timestep advection of potential enthalpy G0M and salt S0M) against real Fortran dumps --
Stage 2 of the DYNSI/ocean port, D45. Plain-Python only; JAX deferred (same "new architecture"
discipline as D42's ODHORZ)."""
import glob

import numpy as np
import pytest

from oadvt2_ff import oadvt2, _get_i1i2, _sign, _fortran_sum, IM, JM, LMO
from oadvt2_compare import load_mmi, load_oadvt2_before, load_oadvt2_after, _load_smfinal, FF_DEFAULT
from odhorz_compare import load_lmm, load_lmv, load_lmu

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
OADVT2_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_oadvt2_before_*.bin"))

pytestmark = pytest.mark.skipif(not OADVT2_PATHS, reason="ff_data oadvt2 dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    before = load_oadvt2_before(f"{ff}/ffz_oadvt2_before_{itime}.bin")
    after = load_oadvt2_after(f"{ff}/ffz_oadvt2_after_{itime}.bin")
    mmi = load_mmi(date, itime)
    smu, smv = _load_smfinal(ff, itime)
    return lmm, lmu, lmv, before, after, mmi, smu, smv


@pytest.mark.parametrize("date,itime", DATES)
def test_g0m_matches_real_fortran(date, itime):
    lmm, lmu, lmv, before, after, mmi, smu, smv = _load(date, itime)
    _, g0m, gxmo, gymo, gzmo = oadvt2(
        mmi, before["g0m"], before["gxmo"], before["gymo"], before["gzmo"],
        before["dtdum"], False, smu, smv, before["smw"], lmu, lmv, lmm)
    for name, computed, expected in [("G0M", g0m, after["g0m"]), ("GXMO", gxmo, after["gxmo"]),
                                      ("GYMO", gymo, after["gymo"]), ("GZMO", gzmo, after["gzmo"])]:
        assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_s0m_and_ma_match_real_fortran(date, itime):
    lmm, lmu, lmv, before, after, mmi, smu, smv = _load(date, itime)
    ma, s0m, sxmo, symo, szmo = oadvt2(
        mmi, before["s0m"], before["sxmo"], before["symo"], before["szmo"],
        before["dtdum"], True, smu, smv, before["smw"], lmu, lmv, lmm)
    for name, computed, expected in [("S0M", s0m, after["s0m"]), ("SXMO", sxmo, after["sxmo"]),
                                      ("SYMO", symo, after["symo"]), ("SZMO", szmo, after["szmo"]),
                                      ("MA", ma, after["mo1"])]:
        assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_ma_identical_for_g0m_and_s0m_calls(date, itime):
    """MA's evolution depends only on SMU/SMV/SMW/geometry, never on RM's own moments -- the two
    OADVT2 calls (G0M, S0M) must produce bitwise-identical MA despite advecting different
    tracers, matching the real Fortran's redundant-but-consistent recomputation."""
    lmm, lmu, lmv, before, after, mmi, smu, smv = _load(date, itime)
    ma_g, *_ = oadvt2(mmi, before["g0m"], before["gxmo"], before["gymo"], before["gzmo"],
                       before["dtdum"], False, smu, smv, before["smw"], lmu, lmv, lmm)
    ma_s, *_ = oadvt2(mmi, before["s0m"], before["sxmo"], before["symo"], before["szmo"],
                       before["dtdum"], True, smu, smv, before["smw"], lmu, lmv, lmm)
    assert np.array_equal(ma_g, ma_s), f"{date}: MA differs between G0M and S0M calls"


def test_get_i1i2_linear_no_wraparound():
    """OCNDYN.f's get_i1i2 explicitly disables wraparound -- a single active run spanning I=1
    through I=IM must NOT be joined with a separate run at the array's other end."""
    active = np.zeros(IM + 1, dtype=bool)
    active[1:5] = True       # run at the west edge
    active[IM - 2:IM + 1] = True  # separate run at the east edge (near IM)
    segs = _get_i1i2(active)
    assert len(segs) == 2, "wraparound must not join west-edge and east-edge runs"
    assert segs[0] == (1, 4)
    assert segs[1] == (IM - 2, IM)


def test_get_i1i2_single_cell_segment():
    active = np.zeros(IM + 1, dtype=bool)
    active[5] = True
    segs = _get_i1i2(active)
    assert segs == [(5, 5)]


def test_sign_matches_fortran_semantics():
    assert _sign(3.0, -1.0) == -3.0
    assert _sign(-3.0, 1.0) == 3.0
    assert _sign(3.0, 0.0) == 3.0  # Fortran SIGN treats B=0 as non-negative


def test_fortran_sum_is_sequential_left_to_right():
    """Regression pin for the real bug this delta found: `np.sum()`'s reduction strategy (which
    can reorder terms, e.g. pairwise summation for larger arrays) does not always match ifort's
    `-fp-model strict` SUM intrinsic's strict left-to-right accumulation -- the actual root cause
    of a real (if small, ~1e-6 relative) pole-row mismatch in OADVTY2's averaging step
    (`mo(:,j,l)=sum(mo(:,j,l))/im`, summing IM=72 terms). `_fortran_sum` must accumulate strictly
    left-to-right regardless of array size, so its result is order-dependent and NOT invariant
    under reordering the same multiset of terms."""
    a = np.array([1e16, 1.0, -1e16, 1.0])
    # sequential: ((1e16+1)-1e16)+1 -- the first "+1" is lost to float64 rounding (1e16+1==1e16),
    # so the running total is 0 after the first three terms, then +1 gives 1.0.
    assert _fortran_sum(a) == 1.0
    # reordering the SAME terms (so the large values cancel FIRST) changes the result -- proving
    # _fortran_sum is genuinely order-sensitive, not silently equivalent to any reduction.
    b = np.array([1e16, -1e16, 1.0, 1.0])
    assert _fortran_sum(b) == 2.0


@pytest.mark.parametrize("date,itime", DATES)
def test_oadvtz2_pole_restricted_to_i_equals_1(date, itime):
    """Regression pin for the real bug this delta found via mismatch isolation: nbyzm restricts
    J=JM (the North Pole) to I=1 only (D40's established finding), so OADVTZ2's persistent
    cmup/fmup state at I=2..IM,JM must never accumulate real per-cell history the way I=1 does --
    a naive pointwise LMM(i,JM) mask (which processes every I at the pole) diverges from real
    Fortran output at the pole specifically, growing with layer depth."""
    lmm, lmu, lmv, before, after, mmi, smu, smv = _load(date, itime)
    ma, *_ = oadvt2(
        mmi, before["s0m"], before["sxmo"], before["symo"], before["szmo"],
        before["dtdum"], True, smu, smv, before["smw"], lmu, lmv, lmm)
    assert np.allclose(ma[:, JM, :], after["mo1"][:, JM, :], atol=1e-6, rtol=1e-6), \
        f"{date}: MA mismatch at the pole row (J=JM)"


@pytest.mark.parametrize("date,itime", DATES)
def test_mmi_is_recorded_not_rederived(date, itime):
    """Regression pin: an earlier attempt derived MMI as MO0*DXYPO(J) from ODHORZ0's own mo0
    input, which does NOT match the real MMI (widespread ~1e-5 relative mismatches found by
    debugging) -- MMI is a persistent OCEAN_DYN module array only partially overwritten by
    ODHORZ0, not a dense recomputation. This test just confirms the recorded MMI is nonzero and
    plausible in scale (loud failure if the dump format ever silently changes)."""
    mmi = load_mmi(date, itime)
    assert np.any(mmi != 0.0), f"{date}: MMI is all zero"
    assert mmi.max() > 1e10, f"{date}: MMI implausibly small"


@pytest.mark.parametrize("date,itime", DATES)
def test_advection_is_nonvacuous(date, itime):
    """G0M must actually change somewhere -- confirms the advection genuinely fires, not just
    the untouched/identity path."""
    lmm, lmu, lmv, before, after, mmi, smu, smv = _load(date, itime)
    changed = ~np.isclose(after["g0m"], before["g0m"])
    assert np.any(changed), f"{date}: G0M never changed anywhere -- advection untested"
