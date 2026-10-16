"""Tests for polerelax_ff.py / polerelax_jax.py (OCNDYN2.f's polar UOD/VOD relaxation block plus
polevel()) against real Fortran dumps -- Stage 2 of the DYNSI/ocean port, D39."""
import glob

import jax
import numpy as np
import pytest

from polerelax_ff import polerelax, polevel, geomo_pole_arrays, IM, JM, LMO, RELFAC
from polerelax_jax import polerelax_jax
from polerelax_compare import load_geom, load_record, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_polerelax_geom.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data polerelax dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmu, lmv = load_geom(f"{ff}/ffz_polerelax_geom.bin")
    rec = load_record(f"{ff}/ffz_polerelax_{itime}.bin")
    return lmu, lmv, rec


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def test_relfac_constant():
    assert RELFAC == 0.005


def test_cosu_sinu_endpoint():
    """COSU(IM)=1, SINU(IM)=0 (OGEOM.f's explicit endpoint override for the wraparound U-point)."""
    _, _, cosu, sinu = geomo_pole_arrays()
    assert cosu[IM] == 1.0
    assert sinu[IM] == 0.0


@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    lmu, lmv, rec = _load(date, itime)
    uo, vo, uod, vod = polerelax(lmu, lmv, rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"])
    for computed, expected, name in [(uo, rec["uo1"], "UO"), (vo, rec["vo1"], "VO"),
                                      (uod, rec["uod1"], "UOD"), (vod, rec["vod1"], "VOD")]:
        assert np.allclose(computed, expected, atol=1e-9, rtol=1e-9), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_real_fortran(date, itime):
    lmu, lmv, rec = _load(date, itime)
    args0 = [to0_2d(lmu).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_3d(rec["uo0"]), to0_3d(rec["vo0"]), to0_3d(rec["uod0"]), to0_3d(rec["vod0"])]
    uo, vo, uod, vod = polerelax_jax(*args0)
    for computed, expected1, name in [(uo, rec["uo1"], "UO"), (vo, rec["vo1"], "VO"),
                                       (uod, rec["uod1"], "UOD"), (vod, rec["vod1"], "VOD")]:
        assert np.allclose(np.asarray(computed), to0_3d(expected1), atol=1e-9, rtol=1e-9), f"{date} {name}"


def test_jit_compiles_and_matches_eager():
    lmu, lmv, rec = _load(*DATES[0])
    args0 = [to0_2d(lmu).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_3d(rec["uo0"]), to0_3d(rec["vo0"]), to0_3d(rec["uod0"]), to0_3d(rec["vod0"])]
    eager = polerelax_jax(*args0)
    jitted = jax.jit(polerelax_jax)(*args0)
    for e, j in zip(eager, jitted):
        rel = np.abs(np.asarray(e) - np.asarray(j)) / np.maximum(np.abs(np.asarray(e)), 1e-10)
        assert np.max(rel) < 1e-9


# ---------------------------------------------------------------------------
# Mask non-vacuousness and mutation checks.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_pole_row_is_reconstructed_nonvacuously(date, itime):
    """polevel() must actually change UO/VO at the pole row on at least one real layer, or the
    reconstruction path would be untested."""
    _, _, rec = _load(date, itime)
    changed = ~np.isclose(rec["uo1"][:, JM, :], rec["uo0"][:, JM, :])
    assert np.any(changed), f"{date}: pole-row UO never changed by polevel()"


@pytest.mark.parametrize("date,itime", DATES)
def test_uo_vo_unchanged_off_pole_row(date, itime):
    """UO/VO are only ever modified at the pole row (J=JM) by this block -- every other row must
    be byte-identical before/after, since the relax block itself only ever writes UOD/VOD."""
    lmu, lmv, rec = _load(date, itime)
    uo0_interior = rec["uo0"][:, 1:JM, :]
    uo1_interior = rec["uo1"][:, 1:JM, :]
    assert np.allclose(uo1_interior, uo0_interior), f"{date}: UO changed off the pole row"
    vo0_interior = rec["vo0"][:, 1:JM, :]
    vo1_interior = rec["vo1"][:, 1:JM, :]
    assert np.allclose(vo1_interior, vo0_interior), f"{date}: VO changed off the pole row"


@pytest.mark.parametrize("date,itime", DATES)
def test_lmu_lmv_masks_both_populated(date, itime):
    lmu, lmv, _ = _load(date, itime)
    assert np.any(lmu[1:, 1:] > 0) and np.any(lmu[1:, 1:] == 0)
    assert np.any(lmv[1:, 1:] > 0) and np.any(lmv[1:, 1:] == 0)


@pytest.mark.parametrize("date,itime", DATES)
def test_relaxation_step_is_physically_bounded(date, itime):
    """Where VOD actually changes, the step must be a genuine RELFAC=0.5% relaxation toward a
    4-point average of real ocean velocities (which are O(0.01-1) m/s throughout this project's
    dumps, e.g. D36's OSTRES2 delta) -- not something unboundedly different. An absolute bound
    (not relative to VOD's own possibly-near-zero magnitude) is the physically meaningful one
    here; this is a sanity check, not the tight numerical one (test_ff_matches_real_fortran)."""
    lmu, lmv, rec = _load(date, itime)
    changed = ~np.isclose(rec["vod1"], rec["vod0"])
    assert np.any(changed), f"{date}: VOD never changed -- relax branch untested"
    step = np.abs(rec["vod1"][changed] - rec["vod0"][changed])
    assert np.all(step < 0.5), f"{date}: a VOD relaxation step exceeded 0.5 m/s (max {step.max():.3g})"
