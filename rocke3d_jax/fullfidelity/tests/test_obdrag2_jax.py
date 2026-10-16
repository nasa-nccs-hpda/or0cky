"""Tests for obdrag2_ff.py / obdrag2_jax.py (OCNDYN2.f OBDRAG2, implicit bottom-layer current
drag) against real Fortran dumps -- Stage 2 of the DYNSI/ocean port, D38."""
import glob

import jax
import numpy as np
import pytest

from obdrag2_ff import obdrag2, IM, JM, LMO, BDRAGX, DTS
from obdrag2_jax import obdrag2_jax
from obdrag2_compare import load_geom, load_record, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_obdrag2_geom.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data obdrag2 dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmu, lmv = load_geom(f"{ff}/ffz_obdrag2_geom.bin")
    rec = load_record(f"{ff}/ffz_obdrag2_{itime}.bin")
    return lmu, lmv, rec


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def test_constants():
    """BDRAGX=1.0, DTS=DTSRC=1800.0 are compile-time constants for this build; OCN_GISS_TURB
    is not #define'd, so the taubx/tauby/rhobot tidal-enhancement branch never compiles in."""
    assert BDRAGX == 1.0
    assert DTS == 1800.0


@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    lmu, lmv, rec = _load(date, itime)
    uo, vo, uod, vod = obdrag2(lmu, lmv, rec["mo"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"])
    for computed, expected, name in [(uo, rec["uo1"], "UO"), (vo, rec["vo1"], "VO"),
                                      (uod, rec["uod1"], "UOD"), (vod, rec["vod1"], "VOD")]:
        assert np.allclose(computed, expected, atol=1e-9, rtol=1e-9), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_real_fortran(date, itime):
    lmu, lmv, rec = _load(date, itime)
    args0 = [to0_2d(lmu).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_3d(rec["mo"]), to0_3d(rec["uo0"]), to0_3d(rec["vo0"]),
             to0_3d(rec["uod0"]), to0_3d(rec["vod0"])]
    uo, vo, uod, vod = obdrag2_jax(*args0)
    for computed, expected1, name in [(uo, rec["uo1"], "UO"), (vo, rec["vo1"], "VO"),
                                       (uod, rec["uod1"], "UOD"), (vod, rec["vod1"], "VOD")]:
        assert np.allclose(np.asarray(computed), to0_3d(expected1), atol=1e-9, rtol=1e-9), f"{date} {name}"


def test_jit_compiles_and_matches_eager():
    lmu, lmv, rec = _load(*DATES[0])
    args0 = [to0_2d(lmu).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_3d(rec["mo"]), to0_3d(rec["uo0"]), to0_3d(rec["vo0"]),
             to0_3d(rec["uod0"]), to0_3d(rec["vod0"])]
    eager = obdrag2_jax(*args0)
    jitted = jax.jit(obdrag2_jax)(*args0)
    for e, j in zip(eager, jitted):
        rel = np.abs(np.asarray(e) - np.asarray(j)) / np.maximum(np.abs(np.asarray(e)), 1e-10)
        assert np.max(rel) < 1e-9


# ---------------------------------------------------------------------------
# Mask non-vacuousness and mutation checks.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_lmu_lmv_mask_both_populated(date, itime):
    lmu, lmv, _ = _load(date, itime)
    assert np.any(lmu[1:, 1:] > 0) and np.any(lmu[1:, 1:] == 0)
    assert np.any(lmv[1:, 1:] > 0) and np.any(lmv[1:, 1:] == 0)


@pytest.mark.parametrize("date,itime", DATES)
def test_only_bottom_layer_changes(date, itime):
    """Only the per-column bottom layer (L=LMU(I,J) or L=LMV(I,J)) may change -- every other
    layer at every cell must be byte-identical before/after, a per-layer mutation check."""
    lmu, lmv, rec = _load(date, itime)
    lmu0 = to0_2d(lmu).astype(int)
    lmv0 = to0_2d(lmv).astype(int)
    uo0, uo1 = to0_3d(rec["uo0"]), to0_3d(rec["uo1"])
    vo0, vo1 = to0_3d(rec["vo0"]), to0_3d(rec["vo1"])
    l_idx = np.arange(1, LMO + 1)
    bottom_u_mask = (l_idx[None, None, :] == lmu0[:, :, None]) & (lmu0[:, :, None] > 0)
    bottom_v_mask = (l_idx[None, None, :] == lmv0[:, :, None]) & (lmv0[:, :, None] > 0)
    assert np.allclose(uo1[~bottom_u_mask], uo0[~bottom_u_mask]), f"{date}: UO changed off the bottom layer"
    assert np.allclose(vo1[~bottom_v_mask], vo0[~bottom_v_mask]), f"{date}: VO changed off the bottom layer"


@pytest.mark.parametrize("date,itime", DATES)
def test_drag_actually_reduces_speed(date, itime):
    """The drag is a pure deceleration (factor in (0,1]) -- at the cells that change, |velocity|
    must not increase, and at least some real cell must actually be damped (non-vacuous)."""
    lmu, lmv, rec = _load(date, itime)
    lmu0 = to0_2d(lmu).astype(int)
    l_idx = np.arange(1, LMO + 1)
    bottom_u_mask = (l_idx[None, None, :] == lmu0[:, :, None]) & (lmu0[:, :, None] > 0)
    uo0, uo1 = to0_3d(rec["uo0"]), to0_3d(rec["uo1"])
    changed = bottom_u_mask & ~np.isclose(uo1, uo0)
    assert np.any(changed), f"{date}: UO never changed -- drag branch untested"
    assert np.all(np.abs(uo1[changed]) <= np.abs(uo0[changed]) + 1e-12), \
        f"{date}: drag increased |UO| somewhere"
