"""Tests for gmredi_ff.py's gmfexp port against real Fortran dumps -- Stage 2 of the DYNSI/ocean
port, D51 (GMFEXP + computeFluxes + wrapAdjustFluxes/addFluxes -- the actual Gent-McWilliams
skew-flux application to G0M/S0M, the last piece of the mesoscale-mixing family). Plain-Python
only."""
import glob

import numpy as np
import pytest

from gmredi_ff import gmfexp, IM, JM, LMO
from gmfexp_compare import load_gmfexp, FF_DEFAULT
from gmkdif_compare import load_gmkdif
from ocnmeso_compare import load_densgrad
from odhorz_compare import load_lmm, load_lmv, load_lmu

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
GMFEXP_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_gmfexp_*.bin"))

pytestmark = pytest.mark.skipif(not GMFEXP_PATHS, reason="ff_data gmfexp dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    gmk = load_gmkdif(f"{ff}/ffz_gmkdif_{itime}.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    records = load_gmfexp(f"{ff}/ffz_gmfexp_{itime}.bin")
    return lmm, lmu, lmv, gmk, dg, records


@pytest.mark.parametrize("date,itime", DATES)
def test_gmfexp_matches_real_fortran_both_calls(date, itime):
    lmm, lmu, lmv, gmk, dg, records = _load(date, itime)
    assert len(records) == 2, f"{date}: expected 2 GMFEXP calls (G0M, S0M), got {len(records)}"
    for rec in records:
        trm, txm, tym, tzm = gmfexp(
            lmm, lmu, lmv, rec["mo0"], rec["trm0"], rec["txm0"], rec["tym0"], rec["tzm0"],
            bool(rec["qlimit"]),
            gmk["bxx"], gmk["byy"], gmk["bzz"],
            gmk["azx"], gmk["bzx"], gmk["czx"], gmk["aezx"], gmk["ezx"], gmk["cezx"],
            gmk["azy"], gmk["bzy"], gmk["czy"], gmk["aezy"], gmk["ezy"], gmk["cezy"],
            gmk["kpl"], dg["bydh"], dg["bydzv"])
        qlimit = bool(rec["qlimit"])
        for name, computed, expected in [("TRM", trm, rec["trm1"]), ("TXM", txm, rec["txm1"]),
                                          ("TYM", tym, rec["tym1"]), ("TZM", tzm, rec["tzm1"])]:
            assert np.allclose(computed, expected, atol=1e-6, rtol=1e-6), \
                f"{date} qlimit={qlimit} {name}"


@pytest.mark.parametrize("date,itime", DATES)
def test_qlimit_true_and_false_both_present(date, itime):
    """The two GMFEXP calls per OCEANS invocation must be exactly one QLIMIT=False (G0M) and
    one QLIMIT=True (S0M) -- a structural fact about the call site, verified from the dump."""
    lmm, lmu, lmv, gmk, dg, records = _load(date, itime)
    qlimits = sorted(bool(r["qlimit"]) for r in records)
    assert qlimits == [False, True], f"{date}: expected [False, True], got {qlimits}"


@pytest.mark.parametrize("date,itime", DATES)
def test_tzm_is_never_updated(date, itime):
    """Regression pin for a real source-reading finding: GMFEXP's own TZM-update code is
    commented out in full in the real Fortran -- TZM must pass through completely unchanged,
    not by coincidence but because the port deliberately never touches it."""
    lmm, lmu, lmv, gmk, dg, records = _load(date, itime)
    for rec in records:
        assert np.array_equal(rec["tzm0"], rec["tzm1"]), \
            f"{date}: real TZM changed despite the commented-out update code"
        trm, txm, tym, tzm = gmfexp(
            lmm, lmu, lmv, rec["mo0"], rec["trm0"], rec["txm0"], rec["tym0"], rec["tzm0"],
            bool(rec["qlimit"]),
            gmk["bxx"], gmk["byy"], gmk["bzz"],
            gmk["azx"], gmk["bzx"], gmk["czx"], gmk["aezx"], gmk["ezx"], gmk["cezx"],
            gmk["azy"], gmk["bzy"], gmk["czy"], gmk["aezy"], gmk["ezy"], gmk["cezy"],
            gmk["kpl"], dg["bydh"], dg["bydzv"])
        assert np.array_equal(tzm, rec["tzm0"]), f"{date}: port's TZM changed unexpectedly"


@pytest.mark.parametrize("date,itime", DATES)
def test_pole_txm_tym_unchanged(date, itime):
    """GMFEXP's own main loop never touches the pole rows (J range 2..JM-1) -- TXM/TYM at J=1
    and J=JM must pass through completely unchanged."""
    lmm, lmu, lmv, gmk, dg, records = _load(date, itime)
    for rec in records:
        for j in (1, JM):
            assert np.array_equal(rec["txm0"][:, j, :], rec["txm1"][:, j, :]), \
                f"{date}: real TXM changed at pole J={j}"
            assert np.array_equal(rec["tym0"][:, j, :], rec["tym1"][:, j, :]), \
                f"{date}: real TYM changed at pole J={j}"


@pytest.mark.parametrize("date,itime", DATES)
def test_gmfexp_is_nonvacuous(date, itime):
    lmm, lmu, lmv, gmk, dg, records = _load(date, itime)
    for rec in records:
        changed = ~np.isclose(rec["trm1"], rec["trm0"])
        assert np.any(changed), f"{date}: TRM never changed anywhere (qlimit={bool(rec['qlimit'])})"
