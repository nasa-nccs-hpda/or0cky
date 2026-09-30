"""Tests for ofluxv_ff.py / ofluxv_jax.py (OCNDYN2.f OFLUXV + OADVUZ, long-timestep vertical mass
redistribution and simplest-upstream vertical advection of U/V) against real Fortran dumps --
Stage 2 of the DYNSI/ocean port, D43."""
import glob

import jax
import numpy as np
import pytest

from ofluxv_ff import ofluxv, oadvuz, IM, JM, LMO, GRAV, ZE, DZO
from ofluxv_jax import ofluxv_jax
from ofluxv_compare import load_record, load_lmm, load_lmv, load_lmu, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_ofluxv_*.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data ofluxv dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    rec = load_record(f"{ff}/ffz_ofluxv_{itime}.bin")
    return lmm, lmu, lmv, rec


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def test_ze_dzo_derivation():
    assert GRAV == 9.80665
    assert ZE[LMO] == pytest.approx(sum(DZO[1:]), rel=1e-12)
    assert ZE[0] == 0.0


@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    lmm, lmu, lmv, rec = _load(date, itime)
    mo, uo, vo = ofluxv(lmm, lmu, lmv, rec["dtolf"], rec["opbot0"], rec["opress0"],
                        rec["mo0"], rec["uo0"], rec["vo0"])
    for name, computed, expected in [("MO", mo, rec["mo1"]), ("UO", uo, rec["uo1"]),
                                      ("VO", vo, rec["vo1"])]:
        assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_real_fortran(date, itime):
    lmm, lmu, lmv, rec = _load(date, itime)
    args0 = [to0_2d(lmm).astype(np.float64), to0_2d(lmu).astype(np.float64),
             to0_2d(lmv).astype(np.float64), rec["dtolf"],
             to0_2d(rec["opbot0"]), to0_2d(rec["opress0"]),
             to0_3d(rec["mo0"]), to0_3d(rec["uo0"]), to0_3d(rec["vo0"])]
    mo, uo, vo = ofluxv_jax(*args0)
    for name, computed, expected1 in [("MO", mo, rec["mo1"]), ("UO", uo, rec["uo1"]),
                                       ("VO", vo, rec["vo1"])]:
        assert np.allclose(np.asarray(computed), to0_3d(expected1), atol=1e-6, rtol=1e-6), f"{date} {name}"


def test_jit_compiles_and_matches_eager():
    lmm, lmu, lmv, rec = _load(*DATES[0])
    args0 = [to0_2d(lmm).astype(np.float64), to0_2d(lmu).astype(np.float64),
             to0_2d(lmv).astype(np.float64), rec["dtolf"],
             to0_2d(rec["opbot0"]), to0_2d(rec["opress0"]),
             to0_3d(rec["mo0"]), to0_3d(rec["uo0"]), to0_3d(rec["vo0"])]
    eager = ofluxv_jax(*args0)
    jitted = jax.jit(ofluxv_jax)(*args0)
    for e, j in zip(eager, jitted):
        rel = np.abs(np.asarray(e) - np.asarray(j)) / np.maximum(np.abs(np.asarray(e)), 1e-10)
        assert np.max(rel) < 1e-9


# ---------------------------------------------------------------------------
# The single-layer-column edge case (verified from source, D43's key subtlety) and the
# OADVUZ NaN regression (caught during this delta's own JAX validation).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_no_single_layer_columns_in_real_data(date, itime):
    """Checked explicitly, not assumed: this rundeck's real bathymetry has NO columns with
    LMM==1 in any of the 3 test dates -- the single-layer-untouched branch (verified from
    source: OFLUXV's redistribution is gated on 'column reaches layer 2' for both the layer-1
    rescale and the bottom-layer update) is genuinely never exercised by real data. Cross-checked
    against a synthetic case instead (test_single_layer_column_synthetic), the same honest-
    scoping pattern as D28's rain branch."""
    lmm, _, _, _ = _load(date, itime)
    assert not np.any(lmm[1:, 1:] == 1), f"{date}: unexpectedly found a real single-layer column"


def test_single_layer_column_synthetic():
    """Synthetic cross-check of the single-layer-column edge case (never exercised by real data,
    confirmed above): a column with LMM=1 must have MO(I,J,1) byte-identical before/after, while
    a neighboring multi-layer column in the same synthetic grid does change."""
    lmm = np.zeros((IM + 1, JM + 1), dtype=int)
    lmu = np.zeros((IM + 1, JM + 1), dtype=int)
    lmv = np.zeros((IM + 1, JM + 1), dtype=int)
    lmm[1, 2] = 1   # single-layer column
    lmm[2, 2] = 3   # multi-layer neighbor
    lmu[1, 2] = 1
    lmu[2, 2] = 3
    lmv[1, 2] = 1
    lmv[2, 2] = 3
    opbot0 = np.zeros((IM + 1, JM + 1))
    opress0 = np.zeros((IM + 1, JM + 1))
    opbot0[1, 2] = 12000.0 * GRAV
    opbot0[2, 2] = 57000.0 * GRAV
    mo0 = np.zeros((IM + 1, JM + 1, LMO + 1))
    mo0[1, 2, 1] = 11000.0
    mo0[2, 2, 1] = 18000.0
    mo0[2, 2, 2] = 20000.0
    mo0[2, 2, 3] = 19000.0
    uo0 = np.zeros((IM + 1, JM + 1, LMO + 1))
    vo0 = np.zeros((IM + 1, JM + 1, LMO + 1))

    mo, uo, vo = ofluxv(lmm, lmu, lmv, 1800.0, opbot0, opress0, mo0, uo0, vo0)
    assert mo[1, 2, 1] == mo0[1, 2, 1], "single-layer column's MO changed"
    assert mo[2, 2, 1] != mo0[2, 2, 1], "multi-layer neighbor's MO never changed"


@pytest.mark.parametrize("date,itime", DATES)
def test_oadvuz_no_nan_at_inactive_cells(date, itime):
    """Regression pin for the real NaN bug caught during this delta's JAX validation: OADVUZ
    (dense JAX version) must never produce NaN/Inf, including at genuinely-inactive (I,J)
    columns where naive dense computation divides 0/0."""
    lmm, lmu, lmv, rec = _load(date, itime)
    args0 = [to0_2d(lmm).astype(np.float64), to0_2d(lmu).astype(np.float64),
             to0_2d(lmv).astype(np.float64), rec["dtolf"],
             to0_2d(rec["opbot0"]), to0_2d(rec["opress0"]),
             to0_3d(rec["mo0"]), to0_3d(rec["uo0"]), to0_3d(rec["vo0"])]
    mo, uo, vo = ofluxv_jax(*args0)
    for name, arr in [("MO", mo), ("UO", uo), ("VO", vo)]:
        arr_np = np.asarray(arr)
        assert np.all(np.isfinite(arr_np)), f"{date}: {name} contains NaN/Inf"


@pytest.mark.parametrize("date,itime", DATES)
def test_mass_redistribution_is_nonvacuous(date, itime):
    """MO must actually change in real multi-layer columns -- confirms the rescale-to-L13
    fraction branch genuinely fires, not just the untouched-single-layer path."""
    lmm, lmu, lmv, rec = _load(date, itime)
    multi_layer = lmm >= 2
    assert np.any(multi_layer[1:, 1:]), f"{date}: no multi-layer columns"
    changed = ~np.isclose(rec["mo1"], rec["mo0"])
    assert np.any(changed), f"{date}: MO never changed anywhere -- redistribution untested"
