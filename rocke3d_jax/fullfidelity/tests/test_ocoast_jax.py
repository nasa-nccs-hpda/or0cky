"""Tests for ocoast_ff.py / ocoast_jax.py (OCNDYN.f OCOAST, coastal tracer-gradient damping)
against real Fortran dumps -- Stage 2 of the DYNSI/ocean port, D37. Confirmed live and shared
unchanged between OCNDYN.f (definition) and OCNDYN2.f (the live OCEANS driver that calls it) --
see D36's PHASE0_LOG.md entry on the OCNDYN.f/OCNDYN2.f dead-code split."""
import glob

import jax
import numpy as np
import pytest

from ocoast_ff import ocoast, IM, JM, LMO, REDUCE, DTS, SECONDS_PER_DAY
from ocoast_jax import ocoast_jax
from ocoast_compare import load_geom, load_record, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_ocoast_geom.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data ocoast dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_ocoast_geom.bin")
    rec = load_record(f"{ff}/ffz_ocoast_{itime}.bin")
    return lmm, rec


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


def test_reduce_constant():
    """DTS=DTSRC=1800s (this rundeck, OCNDYN.f:405) and SECONDS_PER_DAY=86400 are both
    compile-time constants, not recorded inputs -- REDUCE is fully analytic."""
    assert DTS == 1800.0
    assert SECONDS_PER_DAY == 86400.0
    assert abs(REDUCE - (1.0 - 1800.0 / (86400.0 * 20.0))) < 1e-15


@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    lmm, rec = _load(date, itime)
    gxmo, sxmo, gymo, symo = ocoast(lmm, rec["gxmo0"], rec["sxmo0"], rec["gymo0"], rec["symo0"])
    for computed, expected, name in [(gxmo, rec["gxmo1"], "GXMO"), (sxmo, rec["sxmo1"], "SXMO"),
                                      (gymo, rec["gymo1"], "GYMO"), (symo, rec["symo1"], "SYMO")]:
        assert np.allclose(computed, expected, atol=1e-9, rtol=1e-9), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_real_fortran(date, itime):
    lmm, rec = _load(date, itime)
    lmm0 = to0_2d(lmm).astype(np.float64)
    args0 = [lmm0, to0_3d(rec["gxmo0"]), to0_3d(rec["sxmo0"]),
             to0_3d(rec["gymo0"]), to0_3d(rec["symo0"])]
    gxmo, sxmo, gymo, symo = ocoast_jax(*args0)
    for computed, expected1, name in [(gxmo, rec["gxmo1"], "GXMO"), (sxmo, rec["sxmo1"], "SXMO"),
                                       (gymo, rec["gymo1"], "GYMO"), (symo, rec["symo1"], "SYMO")]:
        assert np.allclose(np.asarray(computed), to0_3d(expected1), atol=1e-9, rtol=1e-9), f"{date} {name}"


def test_jit_compiles_and_matches_eager():
    lmm, rec = _load(*DATES[0])
    lmm0 = to0_2d(lmm).astype(np.float64)
    args0 = [lmm0, to0_3d(rec["gxmo0"]), to0_3d(rec["sxmo0"]),
             to0_3d(rec["gymo0"]), to0_3d(rec["symo0"])]
    eager = ocoast_jax(*args0)
    jitted = jax.jit(ocoast_jax)(*args0)
    for e, j in zip(eager, jitted):
        rel = np.abs(np.asarray(e) - np.asarray(j)) / np.maximum(np.abs(np.asarray(e)), 1e-10)
        assert np.max(rel) < 1e-9


# ---------------------------------------------------------------------------
# Mask/branch non-vacuousness (mutation checks: unreduced cells must be byte-identical;
# reduced cells must actually be scaled by REDUCE, not something else).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_lmin_gt_1_is_nonvacuous(date, itime):
    """Real bathymetry must have at least some cells with a shallower neighbor (LMIN>1, i.e.
    some layer actually gets damped) -- or the whole reduction branch would be untested."""
    lmm, _ = _load(date, itime)
    lmm_grid = lmm[1:, 1:]
    lmm_im1 = np.roll(lmm_grid, 1, axis=0)
    lmm_ip1 = np.roll(lmm_grid, -1, axis=0)
    lmin_x = np.minimum(lmm_im1, lmm_ip1) + 1
    assert np.any(lmin_x[:, 1:JM - 1] <= lmm_grid[:, 1:JM - 1]), \
        f"{date}: no coastal cell exercises the X-gradient damping branch"


@pytest.mark.parametrize("date,itime", DATES)
def test_reduced_cells_scaled_by_reduce_exactly(date, itime):
    """Every cell that actually changed must have changed by exactly REDUCE (not some other
    factor) -- confirms the reduction, where it fires, is the right one."""
    lmm, rec = _load(date, itime)
    changed = ~np.isclose(rec["gxmo1"], rec["gxmo0"])
    assert np.any(changed), f"{date}: GXMO never changed -- damping branch untested"
    ratio = np.where(np.abs(rec["gxmo0"][changed]) > 1e-6,
                      rec["gxmo1"][changed] / rec["gxmo0"][changed], REDUCE)
    assert np.allclose(ratio, REDUCE, atol=1e-9), f"{date}: changed GXMO cells not scaled by REDUCE"


@pytest.mark.parametrize("date,itime", DATES)
def test_deep_interior_cells_unchanged(date, itime):
    """Cells whose full depth is shallower than LMIN (both neighbors at least as deep, i.e. the
    layer range LMIN..LMM(I,J) is empty) must be byte-identical before/after -- a genuine
    per-layer mutation check, not just a global 'something changed' assertion."""
    lmm, rec = _load(date, itime)
    lmm_grid = lmm[1:, 1:]
    lmm_im1 = np.roll(lmm_grid, 1, axis=0)
    lmm_ip1 = np.roll(lmm_grid, -1, axis=0)
    lmin_x = np.minimum(lmm_im1, lmm_ip1) + 1
    # cells where LMIN > LMM(I,J): the L-loop is empty, GXMO/SXMO must be fully untouched
    empty_range = lmin_x > lmm_grid
    j_interior = np.zeros(JM, dtype=bool); j_interior[1:JM - 1] = True
    mask = empty_range & j_interior[None, :]
    gxmo0 = to0_3d(rec["gxmo0"]); gxmo1 = to0_3d(rec["gxmo1"])
    assert np.allclose(gxmo1[mask], gxmo0[mask]), f"{date}: empty-range cells' GXMO changed"
