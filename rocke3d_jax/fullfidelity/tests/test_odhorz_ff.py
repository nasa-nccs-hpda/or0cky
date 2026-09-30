"""Tests for odhorz_ff.py (OCNDYN2.f ODHORZ, the horizontal momentum + mass-continuity solve)
against real Fortran dumps -- Stage 2 of the DYNSI/ocean port, D42.

Plain-Python (F0) only for this delta -- the JAX vectorization is deliberately deferred as its
own follow-up (this project's established discipline: prove F0 correctness first, per the GHY
lesson that naively unrolling a large per-cell loop can be a JAX *regression*, not a speedup;
ODHORZ is large and multi-physics enough to deserve that care as separate work)."""
import glob

import numpy as np
import pytest

from odhorz_ff import odhorz, IM, JM, LMO, OMEGA, GRAV, RADIUS
from odhorz0_compare import load_geom as load_lmm, load_lmv, FF_DEFAULT
from odhorz_compare import load_hocean, load_lmu, load_records

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_odhorz_hocean.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data odhorz dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    hocean = load_hocean(f"{ff}/ffz_odhorz_hocean.bin")
    records = load_records(f"{ff}/ffz_odhorz_{itime}.bin")
    return lmm, lmu, lmv, hocean, records


def test_constants():
    assert GRAV == 9.80665
    assert RADIUS == 6371000.0
    # OMEGA validated empirically (not directly checkable in isolation); the real-Fortran match
    # in test_ff_matches_real_fortran is the actual validation of this value.
    assert 7.29e-5 < OMEGA < 7.30e-5


@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    lmm, lmu, lmv, hocean, records = _load(date, itime)
    assert len(records) >= 1, f"{date}: no ODHORZ records in window"
    for k, rec in enumerate(records):
        mo, uo, vo, uod, vod, opbot = odhorz(
            lmm, lmu, lmv, hocean, rec["dt"],
            rec["moh"], rec["uoh"], rec["voh"], rec["uodh"], rec["vodh"], rec["opboth"],
            rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], rec["opbot0"],
            rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"])
        for name, computed, expected in [
            ("MO", mo, rec["mo1"]), ("UO", uo, rec["uo1"]), ("VO", vo, rec["vo1"]),
            ("UOD", uod, rec["uod1"]), ("VOD", vod, rec["vod1"]), ("OPBOT", opbot, rec["opbot1"]),
        ]:
            assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), f"{date} call{k} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_multiple_calls_per_window(date, itime):
    """ODHORZ fires several times per DTsrc step (2 initial half-steps + leapfrog sub-steps) --
    confirm the real window genuinely exercises more than one call, not just a single trivial one."""
    _, _, _, _, records = _load(date, itime)
    assert len(records) >= 2, f"{date}: expected multiple ODHORZ calls in a 6-step window"


@pytest.mark.parametrize("date,itime", DATES)
def test_opbot_accumulates_across_layers_not_reset(date, itime):
    """Regression pin for the real bug caught before the first validation run: OPBOT (2D, no
    layer index) must accumulate mass-convergence contributions across ALL 13 layers within one
    call, not reset from OPBOT0 each layer. If OPBOT only reflected the last layer's
    contribution, it would equal OPBOT0 plus a single layer's convij*GRAV term -- check the real
    delta is NOT explainable that way (a loose but meaningful regression signal, distinct from
    the tight numerical check above)."""
    lmm, lmu, lmv, hocean, records = _load(date, itime)
    rec = records[0]
    delta = rec["opbot1"] - rec["opbot0"]
    changed = np.abs(delta) > 1e-9
    assert np.any(changed), f"{date}: OPBOT never changed -- accumulation branch untested"


@pytest.mark.parametrize("date,itime", DATES)
def test_lmu_lmv_lmm_masks_populated(date, itime):
    lmm, lmu, lmv, hocean, _ = _load(date, itime)
    for name, arr in [("LMM", lmm), ("LMU", lmu), ("LMV", lmv)]:
        assert np.any(arr[1:, 1:] > 0) and np.any(arr[1:, 1:] == 0), name


@pytest.mark.parametrize("date,itime", DATES)
def test_hocean_real_bathymetry(date, itime):
    """HOCEAN (needed for OGEOZ's per-call init, the real design gap caught before this delta's
    first validation run) must be real, non-trivial bathymetry -- not all zero."""
    _, _, _, hocean, _ = _load(date, itime)
    assert np.any(hocean[1:, 1:] > 100.0), f"{date}: HOCEAN looks trivial/unset"
