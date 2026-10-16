"""Tests for dyn_aflux_ff.py (AFLUX, ADVECM, MAtoP, AVRX, FFT72; D96) against the real-Fortran per-call
dumps ffd_aflux_* (3 dates x 6 steps x 5 leapfrog passes = 90 calls).  Skipped if the dumps are absent."""
import copy

import numpy as np
import pytest

import dyn_aflux_ff as ff
import dyn_aflux_compare as cm
import dyn_fft72_ff as fft72
import intel_libm_ff
from dyn_aflux_ff import IM, JM, LM

HAVE = all(cm.available(d) for d, _ in cm.DATES)
pytestmark = pytest.mark.skipif(not HAVE, reason="ff_data ffd_aflux dumps not present on this host")

CALLS = cm.calls()
_G = {}


def geom(date):
    if date not in _G:
        g = cm.load_geom_date(date)
        _G[date] = (g, ff.avrx_tables(g))
    return _G[date]


@pytest.mark.parametrize("date,itime,pas", CALLS)
def test_aflux_bitwise(date, itime, pas):
    g, tab = geom(date)
    i, o, r = cm.run_aflux(date, itime, pas, g, tab=tab)
    d = cm.aflux_compare(i, o, r)
    for k in ('mu', 'mv_J2..JM', 'mw', 'conv', 'spa', 'spa0'):
        assert d[k] == 0.0, (k, d[k])
    assert d['mv_J1_unwritten_real'] == 0.0     # row J=1 of MV is never written by AFLUX (stays 0)


@pytest.mark.parametrize("date,itime,pas", CALLS)
def test_aflux_logic_with_recorded_avrx(date, itime, pas):
    """Isolates AFLUX from the FFT: with the real post-AVRX SPA substituted the result is exact too."""
    g, tab = geom(date)
    i, o, r = cm.run_aflux(date, itime, pas, g, recorded_avrx=True, tab=tab)
    d = cm.aflux_compare(i, o, r)
    assert max(d['mu'], d['mv_J2..JM'], d['mw'], d['conv'], d['spa']) == 0.0


@pytest.mark.parametrize("date,itime,pas", CALLS)
def test_advecm_matop(date, itime, pas):
    g, _ = geom(date)
    i, o, r = cm.run_advecm(date, itime, pas, g)
    d = cm.advecm_compare(o, r)
    for k in ('mnew', 'msum', 'pedn', 'pmid', 'pdsig', 'p'):
        assert d[k] == 0.0, (k, d[k])
    # PK = PMID**KAPA with numpy pow: 1 ulp (<= 8.9e-16 for PK in [4,8)) in rare cells
    assert d['pk'] <= 1e-15
    assert r['n_exception'] == 0


@pytest.mark.skipif(not intel_libm_ff.available(), reason="Intel libimf not available")
@pytest.mark.parametrize("date,itime,pas", CALLS[::7])
def test_matop_pk_bitwise_with_libimf_pow(date, itime, pas):
    g, _ = geom(date)
    i, o, r = cm.run_advecm(date, itime, pas, g, imf_pow=True)
    assert cm.advecm_compare(o, r)['pk'] == 0.0


def test_pk_numpy_pow_is_not_always_exact():
    """Documents the finding: numpy pow differs from the real PK in a few cells (so the exact mode matters)."""
    g, _ = geom("nov26")
    i, o, r = cm.run_advecm("nov26", 33312, 1, g)
    assert np.sum(r['pk'] != o['pk']) > 0
    assert np.max(np.abs(r['pk'] - o['pk']) / o['pk']) < 2.3e-16


# ---------------------------------------------------------------- FFT72 / AVRX
def test_fft_roundtrip():
    x = np.random.RandomState(0).rand(72, 6)
    A, B = fft72.fft_rows(x)
    assert np.max(np.abs(fft72.ffti_rows(A, B) - x)) < 1e-14


def test_fft_matches_numpy_spectrum():
    """Independent check of the transliteration: A,B are 2/KM-scaled cos/sin coefficients with the
    sample index K=1..KM (so a one-sample phase shift against numpy's rfft)."""
    x = np.random.RandomState(1).rand(72, 3)
    A, B = fft72.fft_rows(x)
    k = np.arange(1, 73)[:, None]
    for n in (1, 5, 17, 35):
        a = (x * np.cos(2 * np.pi * n * k / 72)[:, :]).sum(0) * 2 / 72
        b = (x * np.sin(2 * np.pi * n * k / 72)[:, :]).sum(0) * 2 / 72
        assert np.allclose(A[n], a, atol=1e-13) and np.allclose(B[n], b, atol=1e-13)


@pytest.mark.parametrize("date", [d for d, _ in cm.DATES])
def test_fft_table_sequential_overwrite_matters(date):
    """FFT0 overwrites C(18),S(0),S(36),C(54),S(72) in sequence at N=KM/4; a table built from the
    initial cos value alone (a plausible port error) changes the AVRX result."""
    g, tab = geom(date)
    i, o, r = cm.run_aflux(date, dict(cm.DATES)[date], 1, g, tab=tab)
    saveC, saveS = dict(fft72.C), dict(fft72.S)
    try:
        c18 = fft72.C[18]
        fft72.C[18] = -c18                      # undo the final overwrite semantics
        fft72.S[36] = -fft72.S[36]
        r2 = ff.aflux(i['ns'], i['u'], i['v'], i['ma'], i['masum'], i['me'], i['mesum'], g, tab=tab)
        assert np.max(np.abs(r2['spa'] - o['spa'])) > 0
    finally:
        fft72.C.clear(); fft72.C.update(saveC); fft72.S.clear(); fft72.S.update(saveS)


@pytest.mark.parametrize("date,itime,pas", CALLS[::10])
def test_avrx_alone(date, itime, pas):
    g, tab = geom(date)
    i, o, r = cm.run_aflux(date, itime, pas, g, tab=tab)
    out = ff.avrx_field(o['spa0'], range(1, JM - 1), g, tab)
    assert np.array_equal(out[:, 1:JM - 1, :], o['spa'][:, 1:JM - 1, :])


# ---------------------------------------------------------------- non-vacuity (branches exercised)
def test_branch_coverage_over_all_calls():
    st_tot = {}
    for date, it0 in cm.DATES:
        g, tab = geom(date)
        for p in range(1, 6):
            st = {}
            cm.run_aflux(date, it0, p, g, stats=st, tab=tab)
            for a, b in st.items():
                st_tot[a] = st_tot.get(a, 0) + b
    # topography adjustment: cells with a height step in both directions, flux really moved up
    assert st_tot['ew_cells'] > 1000 and st_tot['ns_cells'] > 1000
    assert st_tot['ew_moved'] > 500 and st_tot['ns_moved'] > 500
    assert st_tot['ew_levels'] > st_tot['ew_moved']          # some levels visited without moving flux


def test_avrx_active_rows_and_effect():
    g, tab = geom("nov26")
    bysn, drat, nmin = tab
    act = [j + 1 for j in range(1, JM - 1) if drat[j] <= 1]
    assert act == list(range(2, 15)) + list(range(33, 46))       # 13 rows per hemisphere (Fortran J)
    i, o, r = cm.run_aflux("nov26", 33312, 1, g, tab=tab)
    dd = np.abs(o['spa'] - o['spa0'])
    for j in act:
        assert dd[:, j - 1, :].max() > 0, j
    for j in range(15, 33):
        assert dd[:, j - 1, :].max() == 0, j                      # rows with DRAT>1 are skipped


def test_polar_rows_nonzero_and_mu_pole_scaling():
    g, tab = geom("dec01")
    i, o, r = cm.run_aflux("dec01", 33552, 3, g, tab=tab)
    assert np.max(np.abs(o['mu'][:, 0, :])) > 0 and np.max(np.abs(o['mu'][:, JM - 1, :])) > 0
    assert np.max(np.abs(o['conv'][0, 0, :])) > 0
    # DO_POLEFIX=1: pole MU is 2/3 of the 3*(...) polar value; reproduce without the scaling -> differs
    r0 = ff.aflux(i['ns'], i['u'], i['v'], i['ma'], i['masum'], i['me'], i['mesum'], g, tab=tab, polefix=False)
    assert np.max(np.abs(r0['mu'][:, 0, :] - o['mu'][:, 0, :])) > 1e-3 * np.max(np.abs(o['mu'][:, 0, :]))


def test_mw_pole_columns_replicate():
    g, tab = geom("jan01")
    i, o, r = cm.run_aflux("jan01", 17520, 5, g, tab=tab)
    assert np.all(o['mw'][:, 0, :] == o['mw'][0, 0, :][None, :])
    assert np.max(np.abs(o['mw'])) > 0


def test_never_exercised_branches_documented():
    """Branches that the 90 real calls cannot exercise (and so are NOT validated): topography patches with
    mode 0 (the rundeck uses the single default patch, mode 1); the ADVECM column-mass exception paths."""
    g, _ = geom("nov26")
    assert list(g['md']) == [1] and int(g['npatch']) == 1
    for date, it0 in cm.DATES:
        gg, _ = geom(date)
        for p in (1, 3, 5):
            i, o, r = cm.run_advecm(date, it0 + 5, p, gg)
            assert r['n_exception'] == 0


# ---------------------------------------------------------------- mutation tests
def _base(date="nov26", itime=33312, pas=1):
    g, tab = geom(date)
    i, o, r = cm.run_aflux(date, itime, pas, g, tab=tab)
    return g, tab, i, o


def _run(i, g, tab, **kw):
    return ff.aflux(i['ns'], i['u'], i['v'], i['ma'], i['masum'], i['me'], i['mesum'], g, tab=tab, **kw)


def test_mutation_no_topography_adjustment():
    g, tab, i, o = _base()
    r = _run(i, g, tab, topo=False)
    assert max(np.max(np.abs(r['mu'] - o['mu'])), np.max(np.abs(r['mv'] - o['mv']))) > 1.0


def test_mutation_topography_mode0():
    g, tab, i, o = _base()
    g2 = copy.deepcopy(g); g2['md'] = np.array([0])
    r = _run(i, g2, tab)
    assert np.max(np.abs(r['mu'] - o['mu'])) > 1.0


def test_mutation_no_avrx():
    g, tab, i, o = _base()
    t2 = (tab[0], np.full(JM, 2.0), tab[2])                      # DRAT>1 everywhere -> AVRX skipped
    r = _run(i, g, t2)
    assert np.max(np.abs(r['spa'] - o['spa'])) > 1e-3


def test_mutation_wrong_ns():
    g, tab, i, o = _base()
    i2 = dict(i); i2['ns'] = i['ns'] + 1
    r = _run(i2, g, tab)
    assert np.max(np.abs(r['mw'] - o['mw'])) > 1e-6        # only the (ME-MA) term depends on NS


def test_mutation_polwt():
    g, tab, i, o = _base()
    g2 = copy.deepcopy(g); g2['polwt'] = 0.5
    r = _run(i, g2, tab)
    assert np.max(np.abs(r['mv'] - o['mv'])) > 1.0


def test_mutation_mw_recursion_direction():
    g, tab, i, o = _base()
    g2 = copy.deepcopy(g); g2['mfrac'] = g['mfrac'][::-1].copy()
    r = _run(i, g2, tab)
    assert np.max(np.abs(r['mw'] - o['mw'])) > 1.0


def test_mutation_advecm_no_polar_copy():
    g, _ = geom("nov26")
    i, o, r = cm.run_advecm("nov26", 33312, 1, g)
    assert np.all(o['mnew'][:, 1:, 0] == o['mnew'][:, 0, 0][:, None])
    bad = r['mnew'].copy(); bad[:, 1:, 0] = i['mold'][:, 1:, 0]
    assert np.max(np.abs(bad - o['mnew'])) > 0


def test_mutation_advecm_dt():
    g, _ = geom("nov26")
    i, o, r = cm.run_advecm("nov26", 33312, 1, g)
    r2 = ff.advecm(i['dt1'] * 1.0000001, i['mold'], i['conv'], i['mw'], g)
    assert np.max(np.abs(r2['mnew'] - o['mnew'])) > 0
