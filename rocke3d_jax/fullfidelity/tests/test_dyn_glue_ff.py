"""Tests for dyn_glue_ff.py (D114-D117 coupling glue: CALC_TROP/tropwmo, COMPUTE_WSAVE, calc_kea_3d,
regrid_btoa_3d, DISSIP, CONSERV_SE + energy-fix block, PGRAD_PBL, recalc_agrid_uv, DAILY_ATMDYN) against
real-Fortran dumps ffd_glue_* (3 dates x 6 steps; DAILY: one real end-of-day call per date).  Dump tests skip
if the dumps are absent.  Tests labelled HAND / NOT VALIDATED cover branches that never occur in the real
windows; they check definitions, not the real Fortran."""
import numpy as np
import pytest

import dyn_glue_io as io
import dyn_glue_ff as gf
import dyn_glue_compare as cm
import dyn_filter_ff as ff
import intel_libm_ff
from dyn_glue_io import IM, JM, LM, DATES, NSTEP

HAVE = all(io.available(d) for d, _ in DATES)
needs_dump = pytest.mark.skipif(not HAVE, reason="ff_data ffd_glue dumps not present")
needs_imf = pytest.mark.skipif(not intel_libm_ff.available(), reason="Intel libimf not available")
CALLS = [(d, it0 + k) for d, it0 in DATES for k in range(NSTEP)]
_G = {}


def G(date, analytic=False):
    k = (date, analytic)
    if k not in _G:
        _G[k] = cm.geom(date, analytic)
    return _G[k]


def exact_keys(r):
    return [k for k in r if k.endswith('_nbad')] + ['trop_p', 'wsave', 'dissip_dke', 'dissip_t', 'recalc',
            'kea1', 'kea2', 'rg3_1', 'rg3_2', 'kea1_pre', 'kea2_pre', 'se_init', 'ke_init', 'se_final', 'ke_final',
            'efix_dse', 'efix_mmg', 'efix_sef', 'efix_kef', 'efix_t', 'pgrad_dpdx', 'pgrad_dpdy', 'pgrad_dpdx0',
            'pgrad_dpdy0', 'dissip_kea_saved']


@needs_dump
@pytest.mark.parametrize("date,itime", CALLS)
def test_all_routines_bitwise_recorded_geometry(date, itime):
    r = cm.run_step(date, itime, G(date))
    for k in exact_keys(r):
        assert r[k] == 0, (k, r[k])


@needs_dump
@pytest.mark.parametrize("date,itime", CALLS[::3])
def test_analytic_geometry_also_bitwise(date, itime):
    r = cm.run_step(date, itime, G(date, True))
    for k in exact_keys(r):
        assert r[k] == 0, (k, r[k])


@needs_dump
@needs_imf
@pytest.mark.parametrize("date,itime", CALLS[::6])
def test_tropwmo_libimf_pow_bitwise(date, itime):
    r = cm.run_step(date, itime, G(date), imf_pow=True)
    assert r['trop_p'] == 0 and r['trop_l_nbad'] == 0


@needs_dump
@pytest.mark.parametrize("date,it0", DATES)
def test_recalc_call_sites(date, it0):
    r = cm.run_step(date, it0, G(date))
    # SURFACE(1), ATURB(2) x2, CLOUDS2_DRV (3,4); DIAGA (5) on some steps; none unlabelled (0)
    assert 0 not in r['recalc_sites'] and set(r['recalc_sites']) >= {1, 2, 3, 4}


@needs_dump
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_daily_atmdyn_real_end_of_day_call(date):
    rows = io.load_daily_calls(date)
    assert any(eod for _, _, eod in rows)
    out = cm.run_daily(date, G(date))
    assert len(out) == 1
    r = out[0]
    for k in ('smass_diff', 'mdryanow_diff', 'deltam_diff', 'ma', 'masum', 'pedn'):
        assert r[k] == 0.0, (k, r[k])
    assert r['ma_nbad'] == 0 and r['chg'] > 0 and abs(r['deltam']) > 1e-12      # non-vacuous


# ---------------------------------------------------------------- non-vacuity
@needs_dump
@pytest.mark.parametrize("date,itime", CALLS[::6])
def test_non_vacuous(date, itime):
    r = cm.run_step(date, itime, G(date))
    lo, hi = r['trop_lrange']
    assert hi - lo >= 5                      # tropopause level varies over the grid
    assert r['wsave_scale'] > 1e-3 and r['pgrad_scale'] > 1e-4
    assert r['dissip_dke_scale'] > 1.0 and r['dissip_dT'] > 1e-4
    assert abs(r['efix_dse_val']) > 1e-3 and r['efix_dT'] > 1e-6
    assert r['trop_ierr'] == 0


@needs_dump
def test_tropwmo_branch_coverage_report():
    st = {}
    for d, it0 in DATES:
        cm.run_step(d, it0, G(d), stats=st)
    for k in ('failsafe_set', 'wmo_candidate', 'zptf_zero', 'ldtdz_false', 'jj_cycle', 'jj_valid_exit',
              'jj_discard', 'limit_exit', 'iplimb_gt1'):
        assert st[k] > 0, k
    # never exercised in real data (covered by HAND tests below or reported as uncovered)
    for k in ('limit_noexit', 'jj_loop_end', 'default_ltropp', 'ltropp_failsafe_only'):
        assert st[k] == 0, k


# ---------------------------------------------------------------- mutation checks
@needs_dump
def test_mutation_detected():
    d, it = DATES[0][0], DATES[0][1]
    g = dict(G(d))
    ke = io.load_kea(io.path(d, f"kea_{it}.bin"))[0]
    base = gf.calc_kea_3d(ke['u'], ke['v'], g)
    assert np.array_equal(base, ke['kea'])
    g2 = dict(g); g2['byim'] = g['byim'] * (1 + 1e-12)
    assert not np.array_equal(gf.calc_kea_3d(ke['u'], ke['v'], g2), ke['kea'])      # pole rows depend on byim
    rc = io.load_recalc(io.path(d, f"recalc_{it}.bin"))[0]
    g3 = dict(g); idij = g['idij'].copy(); idij[0, :, 5] = idij[1, :, 5]; g3['idij'] = idij
    ua, _ = gf.recalc_agrid_uv(rc['u'], rc['v'], g3)
    assert not np.array_equal(ua[:, :, 5], rc['ua'][:, :, 5])
    e = io.load_efix(io.path(d, f"efix_{it}.bin"))
    f = gf.energy_fix(e[1]['a'], e[1]['b'], e[2]['a'], e[2]['b'], e[2]['masum'], e[2]['t'], e[2]['pk'], g)
    assert np.array_equal(f['t'], e[3]['t'])
    f2 = gf.energy_fix(e[1]['a'], e[1]['b'], e[2]['a'], e[2]['b'] * (1 + 1e-14), e[2]['masum'], e[2]['t'],
                       e[2]['pk'], g)
    assert not np.array_equal(f2['t'], e[3]['t'])
    w = io.load_wsave(io.path(d, f"wsave_{it}.bin"))
    g4 = dict(g); g4['dtsrc'] = g['dtsrc'] * 2
    assert not np.array_equal(gf.compute_wsave(w['mws'], w['t'], w['pk'], w['pedn'], g4), w['wsave'])


# ---------------------------------------------------------------- HAND tests (NOT VALIDATED against Fortran)
def _hand_g():
    rgas, sha = 287.05, 1004.64
    return dict(psf=984., kapa=rgas / sha, bykapa=sha / rgas, grav=9.80665, rgas=rgas, sha=sha)


def _col(pmid, theta):
    pk = pmid ** _hand_g()['kapa']
    return (theta * pk)[None], pmid[None], pk[None]


def test_hand_tropwmo_default_branch_dry_adiabat():
    """HAND: constant potential temperature (lapse ~9.8 K/km > 3 K/km everywhere) -> no level qualifies:
    ltropp = iplimt-1, ptropo = papm1(ltropp), ierr=1."""
    g = _hand_g(); p = np.geomspace(1000., 0.05, LM)
    tl, pm, pk = _col(p, 300.)
    st = {}
    pt, lt, ierr = gf.tropwmo_columns(tl, pm, pk, g, stats=st)
    iplimt = int(np.argmax(p < 30.)) + 1               # first level (1-based) below ptropmin, exit value of jk
    assert st['default_ltropp'] == 1 and ierr[0] == 1
    assert lt[0] == iplimt - 1 and pt[0] == p[lt[0] - 1]


def test_hand_tropwmo_no_pressure_limit_exit():
    """HAND: all pressures above ptropmin (30 mb) -> the limit loop runs to completion, iplimt = LM."""
    g = _hand_g(); p = np.geomspace(1000., 40., LM)
    tl, pm, pk = _col(p, 300.)
    st = {}
    pt, lt, ierr = gf.tropwmo_columns(tl, pm, pk, g, stats=st)
    assert st['limit_noexit'] == 1 and st['limit_exit'] == 0
    assert 1 < lt[0] <= LM - 1 and pt[0] == p[lt[0] - 1]


def test_hand_tropwmo_isothermal_stratosphere_valid():
    """HAND: troposphere with 6.5 K/km lapse below 11 km then isothermal: the tropopause lands near 11 km level."""
    g = _hand_g(); p = np.geomspace(1000., 5., LM)
    z = -7000. * np.log(p / 1000.)
    T = np.where(z < 11000., 288. - 6.5e-3 * z, 288. - 6.5e-3 * 11000.)
    pk = p ** g['kapa']
    pt, lt, ierr = gf.tropwmo_columns((T)[None], p[None], pk[None], g)
    assert ierr[0] == 0 and abs(z[lt[0] - 1] - 11000.) < 2500.


def test_hand_daily_early_returns_and_initial_time():
    """HAND: DAILY_ATMDYN branches absent from the real windows: not end-of-day and not initial -> None;
    initial time with |DELTAM|<1e-9 -> None; initial time with a large mass error -> applied."""
    g = dict(axyp=np.ones((IM, JM)), areag=float(IM * JM), mtop=0.0, mfrac=np.linspace(0, 1, LM))
    ma = np.ones((LM, IM, JM)); masum = ma.sum(axis=0)
    mdrya = float(masum.sum() / (IM * JM))
    assert gf.daily_atmdyn(ma, masum, g, mdrya, False, False) is None
    assert gf.daily_atmdyn(ma, masum, g, mdrya, True, False) is None
    o = gf.daily_atmdyn(ma, masum, g, mdrya + 5.0, True, False)
    assert o is not None and abs(o['deltam'] - 5.0) < 1e-9
    assert np.array_equal(o['ma'][0], ma[0]) and o['ma'][-1, 0, 0] == ma[-1, 0, 0] + o['deltam'] * g['mfrac'][-1]


def test_hand_regrid_btoa_3d_constant_field():
    """HAND: a constant field is preserved by the interior (.25*(4c)) and by both pole averages."""
    g = dict(byim=1.0 / IM)
    x = np.full((IM, JM, LM), 3.0)
    assert np.allclose(gf.regrid_btoa_3d(x, g), 3.0, rtol=0, atol=1e-15)
