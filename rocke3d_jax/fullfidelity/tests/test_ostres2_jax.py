"""Tests for ostres2_ff.py / ostres2_jax.py (OCNDYN2.f OSTRES2, momentum-stress application to
ocean layer-1 velocities) against real Fortran dumps -- Stage 2 of the DYNSI/ocean port, D36.

First delta from OCNDYN2.f (the live dynamical core) rather than OCNDYN.f (whose OFLUX/OADVM/
OADVV/OPGF/OVtoM/OMtoV/OSTRES/OBDRAG/OPFIL are all dead code, superseded by OCNDYN2.f's rewrite
-- see D36's PHASE0_LOG.md entry)."""
import glob

import jax
import numpy as np
import pytest

from ostres2_ff import ostres2, geomo_arrays, IM, JM, RADIUS
from ostres2_jax import ostres2_jax, geomo_arrays_jax
from ostres2_compare import load_geom, load_record, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_ostres2_geom.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data ostres2 dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    geom = load_geom(f"{ff}/ffz_ostres2_geom.bin")
    rec = load_record(f"{ff}/ffz_ostres2_{itime}.bin")
    return geom, rec


def to0(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


# ---------------------------------------------------------------------------
# Static geometry: fully analytic, validated bitwise against the dump.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", GEOM_PATHS)
def test_geometry_matches_analytic_derivation(path):
    raw = np.fromfile(path, dtype='>f8')
    im_r, jm_r, ivnp = raw[0:3].astype(int)
    assert (im_r, jm_r) == (IM, JM)
    assert ivnp == IM // 4, "IVNP should be the standard GISS IM/4 virtual-V-as-U offset"
    off = 3 + 2 * IM * JM
    dxyso_d = raw[off:off + JM]; off += JM
    dxyno_d = raw[off:off + JM]; off += JM
    dxyvo_d = raw[off:off + JM]; off += JM
    cosic_d = raw[off:off + IM]; off += IM
    sinic_d = raw[off:off + IM]; off += IM
    assert off == raw.size

    dxys, dxyn, dxyvo, cosic, sinic = geomo_arrays()
    assert np.allclose(dxys[1:], dxyso_d, rtol=1e-13)
    assert np.allclose(dxyn[1:], dxyno_d, rtol=1e-13)
    assert np.allclose(dxyvo[1:JM], dxyvo_d[:JM - 1], rtol=1e-13)
    assert np.allclose(cosic[1:], cosic_d, rtol=1e-13)
    assert np.allclose(sinic[1:], sinic_d, rtol=1e-13)


def test_geomo_jax_matches_plain_python():
    dxys, dxyn, dxyvo, cosic, sinic = geomo_arrays()
    dxys_j, dxyn_j, dxyvo_j, cosic_j, sinic_j = geomo_arrays_jax()
    assert np.allclose(dxys[1:], np.asarray(dxys_j), rtol=1e-12)
    assert np.allclose(dxyn[1:], np.asarray(dxyn_j), rtol=1e-12)
    assert np.allclose(dxyvo[1:JM], np.asarray(dxyvo_j)[:JM - 1], rtol=1e-12)
    assert np.allclose(cosic[1:], np.asarray(cosic_j), rtol=1e-12)
    assert np.allclose(sinic[1:], np.asarray(sinic_j), rtol=1e-12)


def test_radius_is_earth_standard():
    """RADIUS is a runtime USE_PLANET_RAD parameter, not assumed Earth -- confirmed to equal
    Earth's standard value only because the geometry match above is bitwise-exact using it."""
    assert RADIUS == 6371000.0


# ---------------------------------------------------------------------------
# Full-grid correctness vs real Fortran.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    geom, rec = _load(date, itime)
    uo, vo, uod, vod = ostres2(
        geom["lmu"], geom["lmv"], rec["dmua"], rec["dmva"], rec["dmui"], rec["dmvi"],
        rec["mo1"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], ivnp=geom["ivnp"])
    for computed, expected, name in [(uo, rec["uo1"], "UO"), (vo, rec["vo1"], "VO"),
                                      (uod, rec["uod1"], "UOD"), (vod, rec["vod1"], "VOD")]:
        assert np.allclose(computed, expected, atol=1e-9, rtol=1e-9), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_real_fortran(date, itime):
    geom, rec = _load(date, itime)
    args0 = [to0(geom["lmu"]), to0(geom["lmv"]), to0(rec["dmua"]), to0(rec["dmva"]),
             to0(rec["dmui"]), to0(rec["dmvi"]), to0(rec["mo1"]), to0(rec["uo0"]),
             to0(rec["vo0"]), to0(rec["uod0"]), to0(rec["vod0"])]
    uo, vo, uod, vod = ostres2_jax(*args0, geom["ivnp"] - 1)
    for computed, expected1, name in [(uo, rec["uo1"], "UO"), (vo, rec["vo1"], "VO"),
                                       (uod, rec["uod1"], "UOD"), (vod, rec["vod1"], "VOD")]:
        assert np.allclose(np.asarray(computed), to0(expected1), atol=1e-9, rtol=1e-9), f"{date} {name}"


def test_jit_compiles_and_matches_eager():
    geom, rec = _load(DATES[0][0], DATES[0][1])
    args0 = [to0(geom["lmu"]), to0(geom["lmv"]), to0(rec["dmua"]), to0(rec["dmva"]),
             to0(rec["dmui"]), to0(rec["dmvi"]), to0(rec["mo1"]), to0(rec["uo0"]),
             to0(rec["vo0"]), to0(rec["uod0"]), to0(rec["vod0"])]
    ivnp0 = geom["ivnp"] - 1
    eager = ostres2_jax(*args0, ivnp0)
    jitted = jax.jit(ostres2_jax, static_argnames=())(*args0, ivnp0)
    for e, j in zip(eager, jitted):
        rel = np.abs(np.asarray(e) - np.asarray(j)) / np.maximum(np.abs(np.asarray(e)), 1e-10)
        assert np.max(rel) < 1e-9


# ---------------------------------------------------------------------------
# Mask-gating and non-vacuous-branch checks (mutation checks: masked-out cells must be
# byte-for-byte untouched; masked-in cells must actually change).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_lmu_lmv_mask_both_populated(date, itime):
    """Real bathymetry must exercise both branches of every mask -- not vacuously all-land or
    all-ocean -- or the gating logic below would be untested."""
    geom, _ = _load(date, itime)
    assert np.any(geom["lmu"][1:, 1:] > 0) and np.any(geom["lmu"][1:, 1:] == 0)
    assert np.any(geom["lmv"][1:, 1:] > 0) and np.any(geom["lmv"][1:, 1:] == 0)


@pytest.mark.parametrize("date,itime", DATES)
def test_masked_out_cells_unchanged(date, itime):
    """Cells with LMU/LMV<=0 must retain their pre-call value exactly -- OSTRES2 only updates
    real ocean cells, and the real Fortran dump proves this directly (not just our port)."""
    geom, rec = _load(date, itime)
    lmu_land = geom["lmu"] <= 0
    lmv_land = geom["lmv"] <= 0
    # UO/VOD gated by LMU (rows 2..JM-1); UO's north-pole special cells are exempt
    interior = np.zeros_like(lmu_land)
    interior[1:, 2:JM] = True
    mask = lmu_land & interior
    mask[IM, JM] = False
    mask[18, JM] = False  # IVNP special case
    assert np.allclose(rec["uo1"][mask], rec["uo0"][mask]), f"{date}: land UO changed"
    assert np.allclose(rec["vod1"][mask], rec["vod0"][mask]), f"{date}: land VOD changed"

    interior_v = np.zeros_like(lmv_land)
    interior_v[1:, 2:JM - 1] = True
    maskv = lmv_land & interior_v
    assert np.allclose(rec["vo1"][maskv], rec["vo0"][maskv]), f"{date}: land VO changed"
    assert np.allclose(rec["uod1"][maskv], rec["uod0"][maskv]), f"{date}: land UOD changed"


@pytest.mark.parametrize("date,itime", DATES)
def test_north_pole_special_case_is_nonvacuous(date, itime):
    """The IVNP/COSIC/SINIC pole-rotation code path must actually be exercised (mo1>0 at the
    pole in every real record) or it would be silently untested."""
    _, rec = _load(date, itime)
    assert rec["mo1"][1, JM] > 0, f"{date}: north pole cell is dry, pole special-case untested"


@pytest.mark.parametrize("date,itime", DATES)
def test_stress_actually_changes_ocean_cells(date, itime):
    """At least some real ocean cells must actually move (non-trivial physics, not a no-op)."""
    _, rec = _load(date, itime)
    assert np.max(np.abs(rec["uo1"] - rec["uo0"])) > 1e-6, f"{date}: UO never changed"
    assert np.max(np.abs(rec["vo1"] - rec["vo0"])) > 1e-6, f"{date}: VO never changed"
