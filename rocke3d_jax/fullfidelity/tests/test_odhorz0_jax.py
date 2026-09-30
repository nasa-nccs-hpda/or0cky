"""Tests for odhorz0_ff.py / odhorz0_jax.py (OCNDYN2.f ODHORZ0, pressure/equation-of-state prep
for the horizontal pressure-gradient solve) against real Fortran dumps -- Stage 2 of the
DYNSI/ocean port, D40."""
import glob

import jax
import numpy as np
import pytest

from odhorz0_ff import odhorz0, IM, JM, LMO, GRAV, Z12EH
from odhorz0_jax import odhorz0_jax
from odhorz0_compare import load_geom, load_lmv, load_record, FF_DEFAULT

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GEOM_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_odhorz0_geom.bin"))

pytestmark = pytest.mark.skipif(not GEOM_PATHS, reason="ff_data odhorz0 dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    rec = load_record(f"{ff}/ffz_odhorz0_{itime}.bin")
    return lmm, lmv, rec


def to0_2d(a1):
    return np.asarray(a1[1:, 1:], dtype=np.float64)


def to0_3d(a1):
    return np.asarray(a1[1:, 1:, 1:], dtype=np.float64)


DUMP_KEY = {"opbot": "opbot", "gup": "gup", "gdn": "gdn", "sup": "sup", "sdn": "sdn",
            "dzgdp": "dzgdp", "vbar": "vbar", "dh3d": "dh3d", "mo": "mo1", "uo": "uo1", "vo": "vo1"}


def test_constants():
    assert GRAV == 9.80665
    assert abs(Z12EH - 1.0 / np.sqrt(12)) < 1e-8


@pytest.mark.parametrize("date,itime", DATES)
def test_ff_matches_real_fortran(date, itime):
    lmm, lmv, rec = _load(date, itime)
    out = odhorz0(lmm, lmv, rec["opress"], rec["g0m"], rec["gzm"], rec["s0m"], rec["szm"],
                  rec["mo0"], rec["uo0"], rec["vo0"], rec["vup"], rec["vdn"])
    for name, dkey in DUMP_KEY.items():
        assert np.allclose(out[name], rec[dkey], atol=1e-6, rtol=1e-6), f"{date} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_real_fortran(date, itime):
    lmm, lmv, rec = _load(date, itime)
    args0 = [to0_2d(lmm).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_2d(rec["opress"]), to0_3d(rec["g0m"]), to0_3d(rec["gzm"]),
             to0_3d(rec["s0m"]), to0_3d(rec["szm"]), to0_3d(rec["mo0"]),
             to0_3d(rec["uo0"]), to0_3d(rec["vo0"]), to0_3d(rec["vup"]), to0_3d(rec["vdn"])]
    out = odhorz0_jax(*args0)
    for name, dkey in DUMP_KEY.items():
        expected = to0_2d(rec[dkey]) if rec[dkey].ndim == 2 else to0_3d(rec[dkey])
        assert np.allclose(np.asarray(out[name]), expected, atol=1e-6, rtol=1e-6), f"{date} {name}"


def test_jit_compiles_and_matches_eager():
    lmm, lmv, rec = _load(*DATES[0])
    args0 = [to0_2d(lmm).astype(np.float64), to0_2d(lmv).astype(np.float64),
             to0_2d(rec["opress"]), to0_3d(rec["g0m"]), to0_3d(rec["gzm"]),
             to0_3d(rec["s0m"]), to0_3d(rec["szm"]), to0_3d(rec["mo0"]),
             to0_3d(rec["uo0"]), to0_3d(rec["vo0"]), to0_3d(rec["vup"]), to0_3d(rec["vdn"])]
    eager = odhorz0_jax(*args0)
    jitted = jax.jit(odhorz0_jax)(*args0)
    for k in eager:
        e, j = np.asarray(eager[k]), np.asarray(jitted[k])
        rel = np.abs(e - j) / np.maximum(np.abs(e), 1e-10)
        assert np.max(rel) < 1e-9, k


# ---------------------------------------------------------------------------
# The North Pole one-cell-only mask override (the real bug caught during this delta's
# validation) -- a dedicated regression test so it can never silently regress.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date,itime", DATES)
def test_north_pole_only_cell_one_is_active(date, itime):
    """OPBOT/GUP/GDN/SUP/SDN must be EXACTLY zero at (I>1, J=JM) -- nbyzm hard-restricts the
    North Pole row to I=1 only, regardless of LMM's value at other longitudes there (the design
    gap this delta's validation caught: a naive LMM(I,J)>=L mask gives nonzero, wrong values
    there instead)."""
    lmm, lmv, rec = _load(date, itime)
    assert np.any(lmm[2:, JM] > 0), f"{date}: test is vacuous -- LMM(I>1,JM) is all zero anyway"
    for name in ["opbot"]:
        assert np.all(rec[name][2:, JM] == 0.0), f"{date}: real dump has nonzero {name} at (I>1,JM)"
    for name in ["gup", "gdn", "sup", "sdn"]:
        assert np.all(rec[name][2:, JM, 1:] == 0.0), f"{date}: real dump has nonzero {name} at (I>1,JM)"


@pytest.mark.parametrize("date,itime", DATES)
def test_pole_copy_fields_uniform_at_pole_row(date, itime):
    """DH3D/VBAR/dZGdP/MO (the fields the Fortran DOES copy across all longitudes at the North
    Pole) must be uniform across I at J=JM, for every layer LMM(1,JM) reaches."""
    lmm, lmv, rec = _load(date, itime)
    lmax = int(lmm[1, JM])
    assert lmax > 0, f"{date}: test is vacuous -- LMM(1,JM)=0"
    for name in ["dh3d", "vbar", "dzgdp", "mo1"]:
        row = rec[name][1:, JM, 1:lmax + 1]
        assert np.allclose(row, row[0:1, :]), f"{date}: {name} not uniform across I at the pole row"


@pytest.mark.parametrize("date,itime", DATES)
def test_lmm_mask_both_populated(date, itime):
    lmm, _, _ = _load(date, itime)
    assert np.any(lmm[1:, 1:] > 0) and np.any(lmm[1:, 1:] == 0)
