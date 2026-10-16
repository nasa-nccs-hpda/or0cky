"""Tests for dyn_isotropuv_ff.py (isotropuv + shap1, D95) against real Fortran per-row dumps
(ffd_isotr*).  Dump-based tests skip if the dumps are absent."""
import numpy as np
import pytest

import dyn_avrx_ff as fa
import dyn_geom_ff as gm
import dyn_isotropuv_ff as fi
import dyn_isotropuv_compare as cm
from dyn_isotropuv_compare import DATES, NSTEP

HAVE = all(cm.available(d) for d, _ in DATES)
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data ffd_isotr dumps not present on this host")
CALLS = [(d, it0 + k, n) for d, it0 in DATES for k in range(NSTEP) for n in range(1, 6)]


def _setup(date):
    return cm.load_geom(date), cm.load_rows(date), cm.load_stat(date), cm.ac.load_consts(date)


@needs_dumps
@pytest.mark.parametrize("date,itime,ncall", CALLS)
def test_bitwise_per_call(date, itime, ncall):
    geo, r, st, k = _setup(date)
    m = (r['itime'] == itime) & (r['ncall'] == ncall)
    assert m.sum() > 0
    Uo, Vo, aux = fi.iso_rows(r['u'][m], r['v'][m], r['j'][m], geo, k['C'], k['S'], return_all=True)
    assert np.array_equal(aux['k'], r['k'][m]) and np.array_equal(aux['fac'], r['fac'][m])
    assert np.array_equal(aux['n'], r['n'][m])
    assert np.array_equal(aux['ua'], r['ua'][m]) and np.array_equal(aux['va'], r['va'][m])   # after shap1
    assert np.array_equal(Uo, r['uo'][m]) and np.array_equal(Vo, r['vo'][m])


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_bitwise_analytic_geometry_and_numpy_tables(date):
    geo, r, st, k = _setup(date)
    a = gm.geometry(geo['radius'])
    geo2 = dict(cosv=a['cosv'], dxv=a['dxv'], cosiv=a['cosiv'], siniv=a['siniv'], fjeq=a['fjeq'])
    C, S = fa.make_tables()
    Uo, Vo = fi.iso_rows(r['u'], r['v'], r['j'], geo2, C, S)
    assert np.array_equal(Uo, r['uo']) and np.array_equal(Vo, r['vo'])


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_all_rows_k_fac_n_from_stat_file(date):
    res = cm.run_date(date)
    assert res['stat_k'] == 0.0 and res['stat_fac'] == 0.0 and res['stat_n_ok']


@needs_dumps
def test_subiteration_counts_exercised():
    seen = set()
    for d, _ in DATES:
        h, byrow, st = cm.n_histogram(d)
        seen |= set(h)
        assert len(st['j']) == 4 * 40 * 5 * NSTEP                # every (j,l) row of every call is in the stat file
        assert set(st['ncall'].tolist()) == {1, 2, 3, 4, 5}
        assert set(st['j'].tolist()) == {2, 3, 45, 46}
        assert max(byrow[46]) == 12                              # khi branch (k>1 -> 1e7) at the north pole row
        assert max(max(v) for v in byrow.values()) == 12
    assert seen == set(range(1, 13))                             # n = 1..12 all occur in the windows
    # the full-row dump holds every row with n>1, so multi-iteration shap1 is validated bitwise above
    for d, _ in DATES:
        r = cm.load_rows(d)
        assert (r['n'] > 1).sum() > 100 and r['n'].max() == 12
        st = cm.load_stat(d)
        assert (r['n'] > 1).sum() == (st['n'] > 1).sum()


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_non_vacuous_and_pole_rows(date):
    geo, r, st, k = _setup(date)
    assert np.max(np.abs(r['uo'] - r['u'])) > 0.5
    pole = np.isin(r['j'], (2, 46))
    assert pole.sum() > 0 and (~pole).sum() > 0
    # pole rows: harmonics >=2 of the x-y components removed by the FFT truncation -> output spectrum is low-order
    # in the x-y frame: convert back and check the harmonics of (ua,va)_final
    cosi, sini = geo['cosiv'], geo['siniv']
    hemi = r['hemi'][:, None]
    ua_f = cosi * r['uo'] - hemi * sini * r['vo']
    va_f = cosi * r['vo'] + hemi * sini * r['uo']
    A, B = fa.fft72(ua_f[pole], k['C'], k['S'])
    assert np.max(np.abs(A[:, 2:])) < 1e-9 * max(1.0, np.max(np.abs(A[:, :2])))
    A2, B2 = fa.fft72(r['ua'][pole], k['C'], k['S'])
    assert np.max(np.abs(A2[:, 2:])) > 1e-3                       # before truncation they are not small
    # non-pole rows are not FFT-truncated
    A3, _ = fa.fft72((cosi * r['uo'] - hemi * sini * r['vo'])[~pole], k['C'], k['S'])
    assert np.max(np.abs(A3[:, 2:])) > 1e-3
    assert set(r['hemi'].tolist()) == {-1, 1}


# ---- shap1 unit/property tests (no dump) -------------------------------------------------------
def test_shap1_constant_row_unchanged_and_sum_conserved():
    x = np.full((1, 72), 3.5)
    y, n = fi.shap1_rows(x, np.array([7.3]))
    assert np.array_equal(y, x) and n[0] == 8
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 72))
    y, _ = fi.shap1_rows(x, np.array([0.3, 2.5, 11.0]))
    assert np.max(np.abs(y.sum(axis=1) - x.sum(axis=1))) < 1e-12


@pytest.mark.parametrize("fac", [0.01, 0.9, 1.0, 2.5, 11.9])
def test_shap1_nyquist_damping_closed_form(fac):
    x = np.where(np.arange(72) % 2 == 0, 1.0, -1.0)[None, :]
    y, n = fi.shap1_rows(x, np.array([fac]))
    nn = int(fac) + 1
    assert n[0] == nn
    assert np.max(np.abs(y - x * (1 - fac / nn) ** nn)) < 1e-12


def test_shap1_n_is_trunc_plus_one():
    _, n = fi.shap1_rows(np.zeros((4, 72)), np.array([0.0, 0.999, 1.0, 5.5]))
    assert list(n) == [1, 1, 2, 6]


def test_hemisphere_pole_helpers():
    assert fi.hemisphere(2, 23.5) == -1 and fi.hemisphere(23, 23.5) == -1 and fi.hemisphere(24, 23.5) == 1
    assert fi.at_pole(2) and fi.at_pole(46) and not fi.at_pole(3)
    g = gm.geometry(6.371e6)
    far = [j for j in range(2, 47) if fi.far_from_pole(j, g['cosv'])]
    assert [j for j in range(2, 47) if j not in far] == [2, 3, 45, 46]


# ---- mutation tests ----------------------------------------------------------------------------
def _run(date="jan01", geo_edit=None, **kw):
    geo, r, st, k = _setup(date)
    if geo_edit:
        geo = geo_edit(dict(geo))
    return fi.iso_rows(r['u'], r['v'], r['j'], geo, k['C'], k['S'], **kw), r


@needs_dumps
def test_mutation_n_without_plus_one(monkeypatch):
    orig = fi.shap1_rows
    monkeypatch.setattr(fi, "shap1_rows", lambda x, fac: (_shap_n_trunc(x, fac), orig(x, fac)[1]))
    (Uo, Vo), r = _run()
    assert np.max(np.abs(Uo - r['uo'])) > 1e-6


def _shap_n_trunc(x, fac):
    """shap1 with n=int(fac) (the +1 dropped; at least 1) -- wrong on purpose."""
    x = np.array(x, dtype=float)
    n = np.maximum(fac.astype(int), 1)
    facby4 = fac * .25 / n
    for nn in range(1, int(n.max()) + 1):
        act = n >= nn
        xs = x[act]
        x[act] = xs + facby4[act][:, None] * (((np.roll(xs, 1, axis=1) - xs) - xs) + np.roll(xs, -1, axis=1))
    return x


@needs_dumps
def test_mutation_khi_changed(monkeypatch):
    monkeypatch.setattr(fi, "KHI", 5e6)
    (Uo, Vo), r = _run()
    assert np.max(np.abs(Uo - r['uo'])) > 1e-6


@needs_dumps
def test_mutation_no_pole_fft_truncation(monkeypatch):
    monkeypatch.setattr(fi, "at_pole", lambda j, jm=46: False)
    (Uo, Vo), r = _run()
    assert np.max(np.abs(Uo - r['uo'])) > 1e-3


@needs_dumps
def test_mutation_hemisphere_sign(monkeypatch):
    monkeypatch.setattr(fi, "hemisphere", lambda j, fjeq: 1)
    (Uo, Vo), r = _run()
    assert np.max(np.abs(Uo - r['uo'])) > 1e-3


@needs_dumps
def test_mutation_dt_changed():
    geo, r, st, k = _setup("jan01")
    Uo, Vo = fi.iso_rows(r['u'], r['v'], r['j'], geo, k['C'], k['S'], dt=449.0)
    assert np.max(np.abs(Uo - r['uo'])) > 1e-6
