"""Tests for dyn_sdrag_ff.py (SDRAG, D95).  Dump-based tests (ffd_sdrag*) validate the default non-linear
branch with cd_lin / non-cd_lin levels and the ANG_SDRAG=1 momentum return against real Fortran columns.
The wl>wmaxj clamp branch never occurs in the windows (max wl 64-71 m/s vs wmax 200): its tests are
hand-derived and are NOT validated against real Fortran.  The linear (rtau) branch is dead and not ported."""
import numpy as np
import pytest

import dyn_geom_ff as gm
import dyn_sdrag_ff as sd
import dyn_sdrag_compare as cm
import dyn_isotropuv_compare as icm
from dyn_sdrag_compare import DATES

HAVE = all(cm.available(d) for d, _ in DATES) and all(icm.available(d) for d, _ in DATES)
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data ffd_sdrag dumps not present on this host")
CALLS = [(d, it0 + k, n) for d, it0 in DATES for k in range(6) for n in (1, 2)]


def _setup(date):
    p, geo = cm.load_consts(date)
    return p, geo, cm.load_cols(date)


def _sel(cols, m):
    return {k: v[m] for k, v in cols.items()}


@needs_dumps
@pytest.mark.parametrize("date,itime,ncall", CALLS)
def test_bitwise_per_call(date, itime, ncall):
    p, geo, cols = _setup(date)
    c = _sel(cols, (cols['itime'] == itime) & (cols['ncall'] == ncall))
    assert len(c['j']) == 135                                  # 45 rows x I=1,25,49
    uo, vo, ncl = cm.run(c, p, geo)
    assert np.array_equal(uo, c['uo']) and np.array_equal(vo, c['vo'])
    assert ncl == 0


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_bitwise_with_analytic_geometry(date):
    p, geo, cols = _setup(date)
    a = gm.geometry(icm.load_geom(date)['radius'])
    geo2 = {k: a[k] for k in geo}
    uo, vo, _ = cm.run(cols, p, geo2)
    assert np.array_equal(uo, cols['uo']) and np.array_equal(vo, cols['vo'])


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_regimes_and_non_vacuous(date):
    p, geo, cols = _setup(date)
    assert p['ls1'] == 24 and p['lsdrag'] == 37 and p['lpsdrag'] == 37 and p['ang_sdrag'] == 1 and not p['linear']
    # both drag regimes are exercised: constant-C levels LS1..LSDRAG-1 and linear-in-wind levels >= LSDRAG
    du = np.abs(cols['uo'] - cols['u'])
    assert du[:, 23:36].max() > 1e-6 and du[:, 36:].max() > 1e-6
    # stratospheric drag really acts on the winds; below LS1 only the uniform angular-momentum return changes U
    assert du.max() > 0.3 and np.abs(cols['vo'] - cols['v']).max() > 0.1
    below = (cols['uo'] - cols['u'])[:, :23]
    assert np.all(np.abs(below - below[:, :1]) < 1e-12)       # same du added at every level L<LS1
    assert np.array_equal(cols['vo'][:, :23], cols['v'][:, :23])
    st_in, st_out = cm.load_stat(date)
    assert np.all(st_in[:, 4] == 0)                            # wl>wmaxj never occurs in the windows (documented)
    assert np.all(st_in[:, 5] == 12960) and np.all(st_in[:, 3] == 55080) and np.all(st_in[:, 9] == 0)
    assert np.max(st_in[:, 6]) < 75.0 and len(st_in) == 12


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_pole_rows_use_wmaxp_and_cosv_branch(date):
    """Rows with COSV<=.15 (J=2,3,45,46; J=2,3,46 selected here) are in the sampled columns and reproduce exactly."""
    p, geo, cols = _setup(date)
    pol = np.isin(cols['j'], (2, 3, 46))
    assert pol.sum() > 0
    uo, vo, _ = cm.run(_sel(cols, pol), p, geo)
    assert np.array_equal(uo, cols['uo'][pol])


# ---- property / hand-derived tests (no dump) ---------------------------------------------------
def _column(j=20, wind=(10.0, 5.0), nrep=1):
    g = gm.geometry(6.371e6)
    lm = 40
    p = dict(ls1=24, lsdrag=37, lpsdrag=37, ang_sdrag=1, wc_jdrag=30.0, wmax=200.0, x_sdrag=(.002, .0002),
             csdragl=np.full(lm, 2e-4), vsdragl=np.ones(lm), rgas=287.04873038614903)
    ones = np.ones((nrep, lm))
    u = ones * wind[0]
    v = ones * wind[1]
    t = ones * 220.0
    pk = ones * 0.95
    pedn1 = ones * np.linspace(1000, 1, lm)[None, :] * 0.5
    ma = ones * 5000.0
    return g, p, [u, v, t, pk, pedn1, ma, ma, ma, ma], np.full(nrep, j)


def test_angular_momentum_returned_uniformly_below_ls1():
    g, p, a, jc = _column()
    u, v, ex = sd.sdrag_columns(*a, jc, 900.0, p, g)
    d = u - a[0]
    assert np.all(np.abs(d[:, :23] - d[0, 0]) < 1e-12) and d[0, 0] > 0           # drag removes eastward momentum -> returned
    # mass-weighted conservation: sum_L<LS1 MMUV*dU equals ANG_MOM = -sum_L>=LS1 DUT (the momentum the drag removed)
    mm = .5 * ((a[5] + a[6]) * g['dxyn'][jc[0] - 2] + (a[7] + a[8]) * g['dxys'][jc[0] - 1])
    assert ex['ang'][0] > 0 and abs(np.sum(mm[:, :23] * d[:, :23]) - ex['ang'][0]) < 1e-9 * ex['ang'][0]


def test_zero_wind_gives_zero_drag():
    g, p, a, jc = _column(wind=(0.0, 0.0))
    u, v, ex = sd.sdrag_columns(*a, jc, 900.0, p, g)
    assert np.all(u == 0.0) and np.all(v == 0.0) and ex['nclamp'] == 0


def test_temperature_bound_check_raises():
    g, p, a, jc = _column()
    a[2][:] = 50.0
    with pytest.raises(RuntimeError):
        sd.sdrag_columns(*a, jc, 900.0, p, g)


def test_clamp_branch_hand_derived_NOT_VALIDATED_AGAINST_FORTRAN():
    """wl > wmaxj: the code sets X = 1-(1-X0)*wmaxj/wl so the new speed is exactly (1-X0)*wmaxj, where X0 is the
    unclamped drag evaluated with min(wl,wmaxj)=wmaxj.  Derived by hand here; this regime never occurs in the
    real-Fortran windows (max wl 71 m/s), so it is NOT validated against real Fortran output."""
    g, p, a, jc = _column(wind=(300.0, 0.0))
    p['ang_sdrag'] = 0
    u, v, ex = sd.sdrag_columns(*a, jc, 900.0, p, g)
    assert ex['nclamp'] > 0
    j = jc[0]
    wmaxj = 200.0                                   # J=20 is not a polar row
    for L in range(24, 41):
        l = L - 1
        rho = 100. * a[4][0, l] / (p['rgas'] * a[2][0, l] * a[3][0, l])
        xjud = (30.0 / (30.0 + wmaxj)) ** 2
        cdn = 2e-4 * xjud if L < 37 else (.002 + .0002 * wmaxj) * xjud
        mauv = (a[5][0, l] + a[6][0, l]) * g['rapvn'][j - 2] + (a[7][0, l] + a[8][0, l]) * g['rapvs'][j - 1]
        x0 = 900.0 * rho * cdn * wmaxj * 1.0 / mauv
        assert abs(u[0, l] - (1 - x0) * wmaxj) < 1e-9
    assert np.all(u[:, :23] == 300.0)               # ang_sdrag=0: levels below LS1 untouched


def test_polar_row_uses_three_quarter_wmax_NOT_VALIDATED_AGAINST_FORTRAN():
    g, p, a, jc = _column(j=3, wind=(160.0, 0.0))   # J=3: cosv<=.15 -> wmaxj=150 < 160 -> clamped (never in windows)
    p['ang_sdrag'] = 0
    u, v, ex = sd.sdrag_columns(*a, jc, 900.0, p, g)
    assert ex['nclamp'] > 0 and np.all(u[:, 23:] < 150.0)


# ---- mutation tests (real data) ----------------------------------------------------------------
def _mut(date="nov26", **pe):
    p, geo, cols = _setup(date)
    p = dict(p); p.update(pe)
    uo, vo, _ = cm.run(cols, p, geo)
    return uo, vo, cols


@needs_dumps
def test_mutation_no_angular_momentum_return():
    uo, vo, cols = _mut(ang_sdrag=0)
    assert np.max(np.abs(uo - cols['uo'])) > 1e-3


@needs_dumps
def test_mutation_lsdrag_boundary():
    uo, vo, cols = _mut(lsdrag=38, lpsdrag=38)
    assert np.max(np.abs(uo - cols['uo'])) > 1e-6


@needs_dumps
def test_mutation_x_sdrag():
    uo, vo, cols = _mut(x_sdrag=(.002, .00021))
    assert np.max(np.abs(uo - cols['uo'])) > 1e-9
    uo, vo, cols = _mut(wc_jdrag=0.0)
    assert np.max(np.abs(uo - cols['uo'])) > 1e-6


@needs_dumps
def test_mutation_rgas_ulp():
    p, geo, cols = _setup("dec01")
    p = dict(p); p['rgas'] = np.nextafter(p['rgas'], 1e9)
    uo, vo, _ = cm.run(cols, p, geo)
    assert not (np.array_equal(uo, cols['uo']) and np.array_equal(vo, cols['vo']))


@needs_dumps
def test_mutation_cosv_literal_float32_vs_double_not_distinguished_here():
    """Documented limitation: no velocity row has cosv within 1e-7 of .15, so float32(.15) vs .15d0 for the
    `COSV(J).LE..15` test cannot be distinguished by these data (the port uses the float32 value, as Fortran does)."""
    p, geo, cols = _setup("nov26")
    assert np.min(np.abs(geo['cosv'][1:] - 0.15)) > 1e-3
