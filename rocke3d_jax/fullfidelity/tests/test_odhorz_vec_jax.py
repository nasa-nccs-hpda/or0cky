"""Tests for the batched ODHORZ ports (Stage 2): odhorz_vec (D78, numpy, grid vectorized) and
odhorz_jax (D79, jitted layer body), against the scalar port and the real Fortran dumps."""
import glob

import numpy as np
import pytest

from odhorz_ff import odhorz
from odhorz_vec import odhorz_vec
from odhorz_jax import odhorz_jax, make_layer_step
from odhorz_compare import load_records, load_hocean, load_lmu, FF_DEFAULT
from odhorz0_compare import load_geom, load_lmv

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]

pytestmark = pytest.mark.skipif(not glob.glob(f"{FF_DEFAULT}/*/ffz_odhorz_[0-9]*.bin"),
                                reason="ff_data odhorz dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    hocean = load_hocean(f"{ff}/ffz_odhorz_hocean.bin")
    recs = load_records(f"{ff}/ffz_odhorz_{itime}.bin")
    return lmm, lmu, lmv, hocean, recs


def _args(lmm, lmu, lmv, hocean, rec):
    return (lmm, lmu, lmv, hocean, rec["dt"], rec["moh"], rec["uoh"], rec["voh"], rec["uodh"],
            rec["vodh"], rec["opboth"], rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"],
            rec["vod0"], rec["opbot0"], rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"])


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)


@pytest.mark.parametrize("date,itime", DATES)
def test_vec_matches_scalar_port(date, itime):
    lmm, lmu, lmv, hocean, recs = _load(date, itime)
    assert len(recs) == 5
    with np.errstate(all="ignore"):
        for rec in recs:
            args = _args(lmm, lmu, lmv, hocean, rec)
            for g, r in zip(odhorz_vec(*args), odhorz(*args)[:6]):
                assert _rel(g, r) < 1e-12, f"{date} vec vs scalar"


@pytest.mark.parametrize("date,itime", DATES)
def test_jax_matches_vec_and_real_fortran(date, itime):
    lmm, lmu, lmv, hocean, recs = _load(date, itime)
    step = make_layer_step(lmm, lmu, lmv)
    with np.errstate(all="ignore"):
        for rec in recs:
            args = _args(lmm, lmu, lmv, hocean, rec)
            real = [rec["mo1"], rec["uo1"], rec["vo1"], rec["uod1"], rec["vod1"], rec["opbot1"]]
            got = odhorz_jax(*args, step=step)
            for g, r, f in zip(got, odhorz_vec(*args), real):
                assert _rel(g, r) < 1e-9, f"{date} jax vs vec"
                assert _rel(g, f) < 1e-6, f"{date} jax vs real Fortran"
