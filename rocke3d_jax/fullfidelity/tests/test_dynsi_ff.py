"""Tests for icedyn_geom_ff.py (GEOMICDYN/ICDYN_MASKS) and icedyn_dynsi_ff.py (FORM/PLAST/RELAX/
VPICEDYN) against real Fortran dumps -- Stage 1 of the DYNSI/ocean port, D29.

Validated bitwise/float64-exact across all 18 real records (6 steps x 3 dates): geometry against
ffz_geom.bin (per date), VPICEDYN end-to-end (inputs -> UICE/VICE/DMU/DMV/USI/VSI) against
ffy_<itime>_{in,out}.bin. RADIUS is inferred exactly from each date's own dump (USE_PLANET_RAD
runtime parameter) rather than assumed; DTsrc=1800s (decks/P2SAoM40.R) -- DYNSI runs once per full
DTsrc step, not the 900s NIsurf-substep timestep used elsewhere in this project.
"""
import glob
import os

import numpy as np
import pytest

import dynsi_compare as C
import icedyn_dynsi_ff as D
from icedyn_geom_ff import geomicdyn, icdyn_masks
from icedyn_geom_compare import read_geom

FF_DATA = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
DATES = {
    "nov26": range(33312, 33318),
    "dec01": range(33552, 33558),
    "jan01": range(17520, 17526),
}
DTS = 1800.0
OIPHI = np.deg2rad(25.0)
SINWAT, COSWAT = np.sin(OIPHI), np.cos(OIPHI)


def _have_data():
    return os.path.isdir(FF_DATA) and glob.glob(f"{FF_DATA}/nov26/ffz_geom.bin")


pytestmark = pytest.mark.skipif(not _have_data(), reason="ff_data dumps not present on this host")


@pytest.mark.parametrize("date", list(DATES))
def test_geometry_bitwise_exact(date):
    gd = read_geom(f"{FF_DATA}/{date}/ffz_geom.bin")
    dlon = 2.0 * np.pi / gd["imicdyn"]
    radius = gd["dxt"][1] / dlon
    geom = geomicdyn(gd["imicdyn"], gd["jmicdyn"], radius)
    for name in ("dxt", "dxu", "bydx2", "bydxr", "dyt", "dyu", "bydy2", "bydyr",
                 "cst", "csu", "tngt", "tng", "bycsu", "sinen", "bydxdy"):
        assert np.max(np.abs(gd[name] - geom[name])) < 1e-6, name
    heffm, uvm = icdyn_masks(gd["gfocean"], gd["nx1"], gd["jmicdyn"])
    assert np.array_equal(gd["heffm"], heffm)
    assert np.array_equal(gd["uvm"], uvm)


@pytest.mark.parametrize("date,itime", [(d, it) for d, its in DATES.items() for it in its])
def test_vpicedyn_matches_real_fortran(date, itime):
    base = f"{FF_DATA}/{date}"
    res, kki = C.compare_one(f"{base}/ffz_geom.bin", f"{base}/ffy_{itime}_in.bin",
                              f"{base}/ffy_{itime}_out.bin", SINWAT, COSWAT, DTS, verbose=False)
    assert 1 <= kki <= 20
    for name, (max_rel, mean_rel) in res.items():
        assert max_rel < 1e-6, f"{date}/{itime} {name} max_relerr={max_rel:.3e}"
        assert mean_rel < 1e-8, f"{date}/{itime} {name} mean_relerr={mean_rel:.3e}"


def test_tridiag_thomas_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 8
    a = rng.random(n) + 2
    b = rng.random(n) + 10
    c = rng.random(n) + 2
    r = rng.random(n)
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    u = D.tridiag_thomas(a, b, c, r)
    assert np.max(np.abs(A @ u - r)) < 1e-10


def test_tridiag_cyclic_matches_dense_solve():
    rng = np.random.default_rng(1)
    n = 8
    a = rng.random(n) + 2
    b = rng.random(n) + 10
    c = rng.random(n) + 2
    r = rng.random(n)
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    A[0, n - 1] = a[0]
    A[n - 1, 0] = c[n - 1]
    u = D.tridiag_cyclic(a, b, c, r)
    assert np.max(np.abs(A @ u - r)) < 1e-10


def test_relax_is_not_a_vacuous_identity():
    """Mutation check: RELAX must actually move the velocity field, not just echo the input."""
    base = f"{FF_DATA}/nov26"
    gd = read_geom(f"{base}/ffz_geom.bin")
    dlon = 2.0 * np.pi / gd["imicdyn"]
    radius = gd["dxt"][1] / dlon
    D.RADIUS = radius
    D.BYRAD2 = 1.0 / (radius * radius)
    geom = geomicdyn(gd["imicdyn"], gd["jmicdyn"], radius)
    heffm, uvm = icdyn_masks(gd["gfocean"], gd["nx1"], gd["jmicdyn"])
    D.init_geometry(geom, heffm, uvm)
    din = C.read_dynsi_in(f"{base}/ffy_33312_in.bin")
    nx1, ny1 = 74, 46
    uice1_in, vice1_in = din["uice0"].copy(), din["vice0"].copy()
    f = D.form(nx1, ny1, uice1_in, vice1_in, din["gairx"], din["gairy"], din["gwatx"],
               din["gwaty"], din["heff"], din["area"], din["amass"], din["cor"],
               (SINWAT, COSWAT), 1, din["pgfub"], din["pgfvb"])
    uice = {1: uice1_in.copy(), 2: D._pad(nx1, ny1), 3: uice1_in.copy()}
    vice = {1: vice1_in.copy(), 2: D._pad(nx1, ny1), 3: vice1_in.copy()}
    uice2, vice2 = D.relax(nx1, ny1, uice, vice, uice1_in.copy(), vice1_in.copy(),
                            f["forcex"], f["forcey"], f["draga"], f["drags"], f["eta"], f["zeta"],
                            din["amass"], din["cor"], 1.0 / 1800.0)
    assert np.max(np.abs(uice2[1] - uice1_in)) > 1e-6
    assert np.max(np.abs(vice2[1] - vice1_in)) > 1e-6
