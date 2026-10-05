"""Tests for dyn_advecv_ff.py (MOMEN2ND.f ADVECV, D98) against real-Fortran per-call dumps (ffd_advecv_*,
3 dates x 6 steps x 5 passes = 90 calls).  Skipped if the dumps are absent."""
import copy

import numpy as np
import pytest

import dyn_advecv_ff as fv
import dyn_advecv_compare as vc
import dyn_aflux_compare as cm
from dyn_aflux_ff import IM, JM, LM

HAVE = all(vc.available(d) for d, _ in cm.DATES) and all(cm.available(d) for d, _ in cm.DATES)
pytestmark = pytest.mark.skipif(not HAVE, reason="ff_data ffd_advecv/ffd_aflux dumps not present on this host")

CALLS = cm.calls()
_G = {}


def geom(date):
    if date not in _G:
        _G[date] = cm.load_geom_date(date)
    return _G[date]


@pytest.mark.parametrize("date,itime,pas", CALLS)
def test_advecv_bitwise(date, itime, pas):
    i, o, r = vc.run_advecv(date, itime, pas, geom(date))
    d = vc.advecv_compare(i, o, r)
    assert d['ut'] == 0.0 and d['vt'] == 0.0
    assert d['ut_rows2_JM_n_exact'] == d['n_total']


def test_row1_unchanged():
    i, o, r = vc.run_advecv("nov26", 33312, 1, geom("nov26"))
    assert np.array_equal(o['ut'][:, 0, :], i['ut'][:, 0, :]) and np.array_equal(r['ut'][:, 0, :], i['ut'][:, 0, :])
    assert np.array_equal(o['vt'][:, 0, :], i['vt'][:, 0, :])


# ---------------------------------------------------------------- non-vacuity
@pytest.mark.parametrize("date,itime,pas", CALLS[::6])
def test_non_vacuous(date, itime, pas):
    st = {}
    i, o, r = vc.run_advecv(date, itime, pas, geom(date), stages=st)
    assert np.max(np.abs(o['ut'] - i['ut'])) > 1e-3 and np.max(np.abs(o['vt'] - i['vt'])) > 1e-3
    # every stage contributes: horizontal advection, polar replacement, vertical advection, Coriolis
    assert np.max(np.abs(st['dut_hadv'])) > 0
    for row in (1, JM - 1):
        assert np.max(np.abs(st['dut_pole'][:, row, :] - st['dut_hadv'][:, row, :])) > 0   # polar fix changes rows
    assert np.max(np.abs(st['ut_adv'] - i['ut'] * 0)) > 0
    assert np.max(np.abs(st['dut_cor'])) > 0 and np.max(np.abs(st['dvt_cor'])) > 0


def test_polar_override_effect_size():
    st = {}
    i, o, r = vc.run_advecv("dec01", 33552, 2, geom("dec01"), stages=st)
    for row in (1, JM - 1):
        a = np.max(np.abs(st['dut_pole'][:, row, :] - st['dut_hadv'][:, row, :]))
        b = np.max(np.abs(st['dut_hadv'][:, row, :]))
        assert a > 1e-3 * b, (row, a, b)


def test_unexercised_branches_documented():
    """The real runs are serial (one domain with both poles), so the MPI-halo branches of ADVECV
    (J_0STG>2 lower-boundary corner fluxes, haveLatitude(J=1)=false) never occur and are not ported."""
    g = geom("nov26")
    assert g['do_polefix'] == 1                           # polefix branch is the live one
    assert np.max(np.abs(g['polwt'] - 1.0)) < 1e-14       # POLWT=1-eps: the interpolation is nearly the identity


# ---------------------------------------------------------------- mutations
def _call(date="jan01", itime=17522, pas=4):
    g = geom(date)
    i, o, r = vc.run_advecv(date, itime, pas, g)
    return g, i, o


def _run(i, g, **kw):
    return fv.advecv(i['dt1'], i['u'], i['v'], i['mmean'], i['mbefor'], i['ut'], i['vt'], i['mafter'],
                     i['mu'], i['mv'], i['mw'], i['spa'], g, **kw)


def _differs(ut, vt, o, tol=0.0):
    return max(np.max(np.abs(ut - o['ut'])), np.max(np.abs(vt - o['vt']))) > tol


def test_mutation_no_polefix():
    g, i, o = _call()
    assert _differs(*_run(i, g, polefix=False), o, 1e-6)


def test_mutation_wrong_dt():
    g, i, o = _call()
    i2 = dict(i); i2['dt1'] = i['dt1'] * (1 + 1e-9)
    assert _differs(*_run(i2, g), o, 0.0)


def test_mutation_zero_spa_removes_metric_term():
    g, i, o = _call()
    i2 = dict(i); i2['spa'] = i['spa'] * 0
    assert _differs(*_run(i2, g), o, 1e-9)


def test_mutation_polwt():
    g, i, o = _call()
    g2 = copy.deepcopy(g); g2['polwt'] = 0.75
    assert _differs(*_run(i, g2), o, 1e-6)


def test_mutation_acor():
    g, i, o = _call()
    g2 = copy.deepcopy(g); g2['acor'] = 1.0
    assert _differs(*_run(i, g2), o, 1e-6)


def test_mutation_accumulation_order_matters(monkeypatch):
    """The per-cell order of the DUT/DVT updates (IP1 role before I role) is needed for the last bit:
    reversing it for all cells is detected."""
    g, i, o = _call()
    orig = fv._hadv_component

    def reversed_order(wE, nS, sW, sE, d0):
        a = orig(wE, nS, sW, sE, d0)
        # recompute with I-role first for every cell
        carry = lambda x: np.concatenate([np.zeros((IM, 1, x.shape[2])), x], axis=1)
        last = np.zeros((IM, 1, wE.shape[2]))
        n2 = np.concatenate([nS, last], axis=1); w2 = np.concatenate([sW, last], axis=1)
        e2 = np.concatenate([sE, last], axis=1)
        r = d0 - wE
        r = r + carry(nS); r = r + carry(sE); r = r - n2; r = r - w2
        r = r + fv._roll_p(wE); r = r + fv._roll_p(carry(sW)); r = r - fv._roll_p(e2)
        return r
    monkeypatch.setattr(fv, "_hadv_component", reversed_order)
    ut, vt = _run(i, g)
    assert _differs(ut, vt, o, 0.0)
    assert not _differs(ut, vt, o, 1e-9)                 # ...but only at rounding level


def test_mutation_cell_im_order(monkeypatch):
    """Cell IM is special (I-role of iteration 1 precedes its IP1-role).  Using the general order there is
    detected in at least one of the three samples."""
    bad = 0
    for date, it, p in (("nov26", 33312, 1), ("dec01", 33553, 3), ("jan01", 17524, 5)):
        g = geom(date)
        i, o, r = vc.run_advecv(date, it, p, g)
        orig = fv._hadv_component

        def general_order(wE, nS, sW, sE, d0, _orig=orig):
            zero = np.zeros((IM, 1, wE.shape[2]))
            cn = np.concatenate([zero, nS], axis=1); cw = np.concatenate([zero, sW], axis=1)
            ce = np.concatenate([zero, sE], axis=1)
            last = np.zeros((IM, 1, wE.shape[2]))
            n2 = np.concatenate([nS, last], axis=1); w2 = np.concatenate([sW, last], axis=1)
            e2 = np.concatenate([sE, last], axis=1)
            r_ = d0 + fv._roll_p(wE); r_ = r_ + fv._roll_p(cw); r_ = r_ - fv._roll_p(e2)
            r_ = r_ - wE; r_ = r_ + cn; r_ = r_ + ce; r_ = r_ - n2; r_ = r_ - w2
            return r_
        fv._hadv_component = general_order
        try:
            ut, vt = _run(i, g)
        finally:
            fv._hadv_component = orig
        bad += int(_differs(ut, vt, o, 0.0))
    assert bad >= 1


def test_coriolis_cell_im_order_is_numerically_irrelevant():
    """Coriolis updates start from DUT=0 and add exactly two terms per cell, so the special order of cell IM
    (IM1 role first) is bit-identical to the general order (a+b == b+a); the port keeps the Fortran order anyway."""
    ur = np.random.RandomState(3).rand(IM, JM - 1, LM)
    alph = np.random.RandomState(4).rand(IM, JM - 1, LM) * 1e3
    ud, vd = fv._coriolis_acc(alph, ur, ur)
    an = np.roll(alph, -1, axis=0)
    gen = (0. + alph * ur) + an * ur
    assert np.array_equal(ud, gen)
