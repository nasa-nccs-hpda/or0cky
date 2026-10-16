"""Tests for dyn_fltruv_ff.py (ATMDYN.f end-of-DYNAM velocity filter chain, D90) against
real Fortran per-call dumps (ffd_fltruv_*). Skipped if the dumps are not present."""
import os

import numpy as np
import pytest

import dyn_fltruv_ff as ff
from dyn_fltruv_ff import IM, JM, LM
import dyn_fltruv_compare as cmpm
from dyn_fltruv_compare import FF_DEFAULT, DATES, NSTEP

HAVE = all(cmpm.available(d) for d, _ in DATES)
pytestmark = pytest.mark.skipif(not HAVE, reason="ff_data ffd_fltruv dumps not present on this host")

CALLS = [(d, it0 + k) for d, it0 in DATES for k in range(NSTEP)]


def _setup(date, itime):
    g = cmpm.load_geom(f"{FF_DEFAULT}/{date}/ffd_fltruv_geom.bin")
    omega = g['omega']
    rin, rflt, rny, rout = cmpm.load_call(date, itime)
    return g, omega, rin, rflt, rny, rout


@pytest.mark.parametrize("date,itime", CALLS)
@pytest.mark.parametrize("analytic", [False, True])
def test_chain_bitwise(date, itime, analytic):
    r = cmpm.run_call(date, itime, analytic=analytic)
    for k in ('flt_u', 'flt_v', 'am1', 'ny_u', 'ny_v', 'damsum', 'out_u', 'out_v'):
        assert r[k] == 0.0, (k, r[k])
    assert r['n_out_u_exact'] == r['n_total']


@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_geometry_analytic_matches_dump(date):
    err = cmpm.geometry_check(date)
    assert max(err.values()) == 0.0, err


@pytest.mark.parametrize("date,itime", CALLS[::6])
def test_non_vacuous(date, itime):
    g, omega, rin, rflt, rny, rout = _setup(date, itime)
    assert np.max(np.abs(rflt['u'] - rin['u'])) > 1e-3
    assert np.max(np.abs(rny['u'] - rflt['u'])) > 1e-4
    assert np.max(np.abs(rout['u'] - rny['u'])) > 1e-9      # solid-body fix is tiny but nonzero
    assert rny['damsum'] != 0.0
    # pole rows are touched by the filters (J=2 and J=JM)
    assert np.max(np.abs(rout['u'][:, 1, :] - rin['u'][:, 1, :])) > 0
    assert np.max(np.abs(rout['v'][:, JM - 1, :] - rin['v'][:, JM - 1, :])) > 0


def test_row_j1_unchanged():
    g, omega, rin, *_ = _setup("nov26", 33312)
    u, v, _ = ff.filter_chain(rin['u'], rin['v'], rin['ma'], rin['masum'], g, omega)
    assert np.array_equal(u[:, 0, :], rin['u'][:, 0, :])
    assert np.array_equal(v[:, 0, :], rin['v'][:, 0, :])


def test_chain_end_matches_pre_condse_dump():
    """Independent cross-check: U,V after the chain equal the pre-CONDSE state dump of the
    base instrumentation (ffd_<itime>_pre_condse.bin)."""
    from ffdump_reader import read_dump
    p = f"{FF_DEFAULT}/nov26/ffd_33312_pre_condse.bin"
    if not os.path.exists(p):
        pytest.skip("pre_condse dump absent")
    d = read_dump(p)
    g, omega, rin, rflt, rny, rout = _setup("nov26", 33312)
    u, v, _ = ff.filter_chain(rin['u'], rin['v'], rin['ma'], rin['masum'], g, omega)
    assert np.array_equal(u, d['U']) and np.array_equal(v, d['V'])


def test_ang_uv_conservation_after_ew_filter():
    """FLTRUV with ANG_UV=1 conserves the per-row, per-layer angular-momentum sum of U
    (to rounding); without the fix the E-W filter really changes it."""
    g, omega, rin, *_ = _setup("dec01", 33552)
    ma = rin['ma']
    nxt = np.roll(np.arange(IM), -1)

    def resid(arr):
        out = []
        for j in range(1, JM):
            mm = .5 * ((ma[:, nxt, j - 1] + ma[:, :, j - 1]) * g['dxyn'][j - 1]
                       + (ma[:, nxt, j] + ma[:, :, j]) * g['dxys'][j])
            out.append(np.sum(mm.T * (arr[:, j, :] - rin['u'][:, j, :]), axis=0))
        return np.max(np.abs(out))
    r1 = resid(ff.fltruv(rin['u'], rin['v'], ma, g, ang_uv=1)[0])
    r0 = resid(ff.fltruv(rin['u'], rin['v'], ma, g, ang_uv=0)[0])
    assert r0 > 1e3 * max(r1, 1e-300), (r0, r1)


# ---- mutation tests: each deliberately-wrong variant must be detected -----------------------
def _chain_out(date, itime, **kw):
    g, omega, rin, rflt, rny, rout = _setup(date, itime)
    return g, omega, rin, rout


def test_mutation_no_angular_momentum_fix():
    g, omega, rin, rout = _chain_out("nov26", 33312)
    u, v, _ = ff.filter_chain(rin['u'], rin['v'], rin['ma'], rin['masum'], g, omega, ang_uv=0)
    assert np.max(np.abs(u - rout['u'])) > 1e-6


def test_mutation_wrong_nshap(monkeypatch):
    g, omega, rin, rout = _chain_out("nov26", 33312)
    monkeypatch.setattr(ff, "NSHAP", 6)
    u, v, _ = ff.filter_chain(rin['u'], rin['v'], rin['ma'], rin['masum'], g, omega)
    assert np.max(np.abs(u - rout['u'])) > 1e-9 or np.max(np.abs(v - rout['v'])) > 1e-9


def test_mutation_sign_of_pole_crossing():
    """Dropping the pole-crossing sign in fltry2 changes the result at J=2."""
    g, omega, rin, rflt, rny, rout = _setup("jan01", 17520)
    q = rflt['u']
    good = ff.fltry2(q)
    # emulate wrong crossing: plain reflection without the sign
    yv = ff.BY4TON
    yn = q[:, 1:, :].copy()
    h = IM // 2
    for _ in range(8):
        new = np.empty_like(yn)
        south = np.concatenate([yn[h:, 0, :], yn[:h, 0, :]], axis=0)  # sign dropped
        left = np.concatenate([south[:, None, :], yn[:, :-1, :]], axis=1)
        new[:, :-1, :] = ((left[:, :-1, :] - yn[:, :-1, :]) - yn[:, :-1, :]) + yn[:, 1:, :]
        yj = yn[:, -1, :]
        new[:, -1, :] = ((left[:, -1, :] - yj) - yj) - np.concatenate([yj[h:], yj[:h]], axis=0)
        yn = new
    bad = q.copy()
    bad[:, 1:, :] = q[:, 1:, :] - yn * yv
    assert np.max(np.abs(bad[:, 1, :] - good[:, 1, :])) > 1e-9
