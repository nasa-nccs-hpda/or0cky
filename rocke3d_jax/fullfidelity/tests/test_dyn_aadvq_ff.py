"""Tests for dyn_aadvq_ff.py (QUS3D.f AADVQ0/AADVQ/aadvqx/y/z/checkflux/XSTEP/ZSTEP/aadvqz_column + ATMDYN.f QDYNAM; D103-D106).
Real-dump tests need ff_data/<date>/ffd_qdyn_* (18 calls) and skip otherwise; stress tests need ffd_qdynS8_*; harness tests need
ff_data/qus3d_harness (SYNTHETIC fluxes on a real state, validated against a standalone ifort build of the real QUS3D.f).
Not exercised by the real windows: ncyc>1, ncycxy>1, z-extra columns, x-checkflux firing (see test_real_windows_coverage)."""
import numpy as np
import pytest

import dyn_aadvq_ff as aq
import dyn_qdynam_io as io
import dyn_qdynam_compare as cm
import dyn_qus3d_harness as hs
from dyn_qdynam_io import IM, JM, LM

HAVE = io.available()
STRESS = io.calls("qdynS8")
HAVE_S = len(STRESS) == 18
HAVE_H = hs.available()
skip_real = pytest.mark.skipif(not HAVE, reason="ffd_qdyn dumps not present")
skip_s = pytest.mark.skipif(not HAVE_S, reason="ffd_qdynS8 stress dumps not present")
skip_h = pytest.mark.skipif(not HAVE_H, reason="qus3d harness data not present")
CALLS = [(d, it0 + k) for d, it0 in io.DATES for k in range(io.NSTEP)]
ZERO_KEYS = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float)) and k not in ('n_exact', 'n_total', 'ck_records') and v != 0}


# ------------------------------------------------------------------ real windows
@skip_real
@pytest.mark.parametrize("date,itime", CALLS)
def test_qdynam_bitwise(date, itime):
    d, _ = cm.run_call(date, itime)
    assert not ZERO_KEYS(d)
    assert d['n_exact'] == d['n_total']
    assert d['ck_records'] == 199


@skip_real
def test_real_windows_coverage():
    """Documented coverage: all 18 real calls have ncyc=1, ncycxy=1 on every level, no z-extra column; X nstep reaches 2
    in some rows; checkflux fires in y and z but never in x."""
    nx2 = 0
    for d, it in CALLS:
        q0 = io.load_q0(io.fname(d, it, 'q0'))
        assert q0['ncyc'] == 1 and not q0['do_z_extra'] and q0['ncycxy'].max() == 1 and q0['nstepz_extra'].max() == 0
        nx2 += int(np.sum(q0['nstepx'][1:JM - 1] >= 2))
    assert nx2 > 0
    st = {}
    cm.run_call("dec01", 33552, stats=st)
    assert st['y_checkflux_fired'] > 0 and st['z_checkflux_fired'] > 0 and st['x_checkflux_fired'] == 0
    assert st['x_lim_pos_neg'] > 0 and st['y_lim_neg_pos'] > 0 and st['z_lim_pos_gt_rm'] > 0


@skip_real
@pytest.mark.parametrize("date,itime", CALLS[::6])
def test_non_vacuous(date, itime):
    d, x = cm.run_call(date, itime)
    assert np.max(np.abs(x['fin']['q'] - x['i']['q'])) > 0 and np.max(np.abs(x['fin']['qmom'] - x['i']['qmom'])) > 0
    ck = x['ck']
    # every sweep changes the level slice
    r1 = [r for r in ck if r['stage'] == 1 and r['l'] == 20][0]; r2 = [r for r in ck if r['stage'] == 2 and r['l'] == 20][0]
    ain = io.load_ain(io.fname(date, itime, 'ain'))
    assert np.max(np.abs(r1['rm'] - ain['rm'][:, :, 19])) > 0 and np.max(np.abs(r2['rm'] - r1['rm'])) > 0
    assert np.max(np.abs(x['out']['scf3d'])) > 0 and np.max(np.abs(x['out']['sbf'])) > 0
    # the carry is non-trivial
    c4 = [r for r in ck if r['stage'] == 4]
    assert len(c4) == LM and max(np.max(np.abs(r['fdn'])) for r in c4) > 0


@skip_real
def test_q0_exports_and_scaling_identity():
    """AADVQ0 on ncyc=1: MV is shifted one row, MU poles zeroed, MW(:, :, LM)=0, nothing else scaled."""
    d, it = "nov26", 33312
    i = io.load_in(io.fname(d, it, 'in')); q0 = io.load_q0(io.fname(d, it, 'q0'))
    assert np.array_equal(q0['mv'][:, :JM - 1], i['mv'][:, 1:]) and np.all(q0['mv'][:, JM - 1] == 0)
    assert np.array_equal(q0['pv_south'], i['mv'][:, 0, :])
    assert np.all(q0['mu'][:, 0] == 0) and np.all(q0['mu'][:, JM - 1] == 0) and np.all(q0['mw'][:, :, LM - 1] == 0)
    assert np.array_equal(q0['mu'][:, 1:JM - 1], i['mu'][:, 1:JM - 1])


@skip_real
def test_mutations_detected():
    date, itime = "dec01", 33552
    i = io.load_in(io.fname(date, itime, 'in')); g = io.load_geom(io.gname(date)); o = io.load_fin(io.fname(date, itime, 'fin'))
    args = (i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'], g['axyp'], g['imaxj'], g['kg2mb'], g['byim_geom'], g['byim_qus'])
    base = aq.qdynam(*args)
    assert np.array_equal(base['q'], o['q'])
    # 1: checkflux disabled
    orig = aq.checkflux
    try:
        aq.checkflux = lambda aml, amr, m, rm, rxm, rxxm: (rxm, rxxm, np.zeros(np.shape(rm), bool))
        r = aq.qdynam(*args)
    finally:
        aq.checkflux = orig
    assert not np.array_equal(r['qmom'], o['qmom'])
    # 2: X and Y direction specs swapped
    ox = aq.XSPEC
    try:
        aq.XSPEC = aq.YSPEC
        r = aq.qdynam(*args)
    finally:
        aq.XSPEC = ox
    assert not np.array_equal(r['q'], o['q'])
    # 3: wrong carry (pass flux fdn0 replaced by the limited flux)
    oz = aq.aadvqz

    def bad(*a, **k):
        z = oz(*a, **k); z['fdn0'] = z['fdn']; return z
    try:
        aq.aadvqz = bad
        r = aq.qdynam(*args)
    finally:
        aq.aadvqz = oz
    assert not np.array_equal(r['q'], o['q']) or not np.array_equal(r['qmom'], o['qmom'])
    # 4: wrong MB conversion constant
    a2 = list(args); a2[8] = g['kg2mb'] * (1 + 1e-15)
    assert not np.array_equal(aq.qdynam(*a2)['q'], o['q'])


def test_single_precision_constants():
    """byn=1./ncyc is REAL*4, mrat_limy is a REAL*4 literal (hand values)."""
    assert aq.MRAT_LIMY == 0.20000000298023224 and aq.MRAT_LIMY != 0.2
    assert aq.inv32(3) == 0.3333333432674408 and aq.inv32(3) != 1. / 3.
    assert aq.inv32(1) == 1.0 and aq.inv32(2) == 0.5


# ------------------------------------------------------------------ stress (real Fortran, scaled fluxes)
@skip_s
@pytest.mark.parametrize("date,itime", CALLS[::3])
def test_stress_s8_bitwise(date, itime):
    d, _ = cm.run_call(date, itime, tag="qdynS8")
    assert not ZERO_KEYS(d)
    assert d['n_exact'] == d['n_total']


@skip_s
def test_stress_s8_coverage():
    st = {}
    zx = 0; ncs = set()
    for d, it in CALLS[:6] + CALLS[6:8]:
        q0 = io.load_q0(io.fname(d, it, 'q0', tag="qdynS8"))
        ncs.add(q0['ncyc']); zx += int(q0['nstepz_extra'].max() > 0)
    assert max(ncs) >= 4 and zx > 0
    cm.run_call("nov26", 33312, tag="qdynS8", stats=st)
    assert st['z_extra_columns'] > 0 and st['ncyc_horizontal_fail'] > 0 and st['y_checkflux_fired'] > 0
    # the byn single-precision mutation is invisible at ncyc=1 (and 4: 1/4 exact) but visible at ncyc=6 (jan01 17520)
    orig = aq.inv32
    i = io.load_in(io.fname("jan01", 17520, 'in', tag="qdynS8")); g = io.load_geom(io.gname("jan01"))
    o = io.load_fin(io.fname("jan01", 17520, 'fin', tag="qdynS8"))
    try:
        aq.inv32 = lambda n: 1.0 / n
        r = aq.qdynam(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'], g['axyp'], g['imaxj'], g['kg2mb'], g['byim_geom'], g['byim_qus'])
    finally:
        aq.inv32 = orig
    assert not (np.array_equal(r['q'], o['q']) and np.array_equal(r['qmom'], o['qmom']))


# ------------------------------------------------------------------ standalone-Fortran harness (SYNTHETIC fluxes)
@skip_h
@pytest.mark.parametrize("k,name", list(enumerate(hs.NAMES, 1)))
def test_harness_bitwise(k, name):
    i, ain, g = hs.base_state()
    c = hs.make_case(name, i, ain)
    d, q0, out = hs.compare_case(c, g, hs.FF_H, k)
    assert not ZERO_KEYS(d) and d['n_exact'] == d['n_total']
    if name == 'ncyc2':
        assert q0['ncyc'] == 2                                    # hand-derived
    if name == 'ncycxy2':
        assert q0['ncyc'] == 1 and q0['ncycxy'][14] == 2 and q0['ncycxy'].sum() == LM + 1   # hand-derived
    if name == 'nstepx4':
        assert q0['nstepx'][24, 11] >= 4
    if name == 'zero':
        assert q0['ncyc'] == 1 and q0['ncycxy'].max() == 1 and q0['nstepx'][1:JM - 1].max() == 1


@skip_h
def test_harness_error_exit():
    import os
    assert 'ncyc>ncmax' in open(os.path.join(hs.FF_H, 'err_stop.txt')).read()
    i, ain, g = hs.base_state()
    c = hs.make_case('err', i, ain)
    with pytest.raises(aq.QusError, match='ncyc>ncmax'):
        aq.aadvq0(c['mu'], c['mv'], c['mw'], c['mb'], g['imaxj'], g['byim_geom'])


# ------------------------------------------------------------------ unit / hand-derived tests (no dumps needed)
def test_checkflux_hand():
    # flow out both sides empties the cell: moments are zeroed; mild flow leaves them
    m = np.array([1.0, 1.0]); rm = np.array([0.1, 1.0]); rx = np.array([0.5, 0.5]); rxx = np.array([0.2, 0.2])
    a, b, fire = aq.checkflux(np.array([-0.6, -0.01]), np.array([0.6, 0.01]), m, rm, rx, rxx)
    assert fire.tolist() == [True, False] and a.tolist() == [0.0, 0.5] and b.tolist() == [0.0, 0.2]


def test_zstep_hand():
    # three layers masses [1,.5,1], fluxes .9 through both interfaces: nstep=1 fails at interface 2 ((.5-.9)*(1.9)<0),
    # nstep=2 passes both sub-steps (hand-derived); final masses [.1,.5,1.9]
    ns, m = aq.zstep(np.array([1.0, 0.5, 1.0]), np.array([0.9, 0.9, 0.0]))
    assert ns == 2 and np.allclose(m, [0.1, 0.5, 1.9], rtol=0, atol=1e-15)
    ns, m = aq.zstep(np.array([1.0, 1.0]), np.array([0.5, 0.0]))
    assert ns == 1 and m.tolist() == [0.5, 1.5]


def test_xstep_rows_vs_loop_and_hand():
    rng = np.random.default_rng(3)
    m = 1.0 + 0.2 * rng.random((6, IM)); mu = (1 + 0.05 * rng.normal(0, 1, (6, IM))) * np.array([0.1, 0.3, 0.5, 0.8, 1, 1.5])[:, None]
    ns, mi, ie = aq.xstep_rows(m, mu)
    assert ie == 0
    for r in range(6):
        n2, mi2, ie2 = aq.xstep_row_loop(m[r], mu[r])
        assert n2 == ns[r] and np.array_equal(mi2, mi[r]) and ie2 == 0
    # uniform flux 3.7 x mass: Courant 3.7 -> nstep 4, mass unchanged
    ns, mi, ie = aq.xstep_rows(np.ones((1, IM)), 3.7 * np.ones((1, IM)))
    assert ns[0] == 4 and np.array_equal(mi[0], np.ones(IM))
    # nstep reaching 60 sets ierr (Fortran: stop_model 'too many steps in xstep')
    ns, mi, ie = aq.xstep_rows(np.ones((1, IM)), 100. * np.ones((1, IM)))
    assert ie == 1 and aq.xstep_row_loop(np.ones(IM), 100. * np.ones(IM))[2] == 1


def _toy_state(seed=5):
    rng = np.random.default_rng(seed)
    mass = 1e3 * (1 + 0.1 * rng.random((IM, JM)))
    mass[:, 0] = mass[0, 0]; mass[:, JM - 1] = mass[0, JM - 1]
    rm = mass * rng.random((IM, JM)) * 0.01
    rm[:, 0] = rm[0, 0]; rm[:, JM - 1] = rm[0, JM - 1]
    mom = rng.normal(0, 1, (9, IM, JM)) * rm[None] * 0.2
    mom[:, :, 0] = mom[:, 0:1, 0]; mom[:, :, JM - 1] = mom[:, 0:1, JM - 1]
    return rm, mom, mass, rng


def test_sweeps_vector_equals_literal_loops():
    rm, mom, mass, rng = _toy_state()
    mu = rng.normal(0, 150, (IM, JM)); mu[:, 0] = 0; mu[:, JM - 1] = 0
    mv = rng.normal(0, 150, (IM, JM)); mv[:, JM - 1] = 0
    ns = np.ones(JM, dtype=int); ns[10] = 2; ns[11] = 3
    z = np.zeros(JM)
    r1 = aq.aadvqy(rm, mom, mass, mv, 1 / 72., z, z, z)
    r2 = aq.aadvqy_loop(rm, mom, mass, mv, 1 / 72., z, z, z)
    for a, b in zip(r1, r2):
        assert np.array_equal(a, b)
    rx, mx_, mass_x = aq.aadvqx(rm, mom, mass, mu, ns)
    for j in (1, 10, 11, 30, JM - 2):
        r, m_, ma = aq.aadvqx_rowloop(rm, mom, mass, mu[:, j], int(ns[j]), j)
        assert np.array_equal(r[:, j], rx[:, j]) and np.array_equal(m_[:, :, j], mx_[:, :, j]) and np.array_equal(ma[:, j], mass_x[:, j])
    # z sweep with a non-trivial carry
    rm2, mom2, mass2, _ = _toy_state(6)
    mw = rng.normal(0, 150, (IM, JM))
    carry = [rng.normal(0, 50, (IM, JM)), rng.normal(0, 3, (IM, JM)), rng.normal(0, 3, (IM, JM)), rng.normal(0, 3, (9, IM, JM))]
    imaxj = np.where((np.arange(JM) == 0) | (np.arange(JM) == JM - 1), 1, IM)
    for j in (0, JM - 1):
        for a in carry:
            a[..., 1:, j] = 0
    args = (rm, mom, mass, rm2, mom2, mass2, mw, *carry, imaxj, z, z, z)
    za = aq.aadvqz(*args); zb = aq.aadvqz_loop(*args)
    for k in za:
        assert np.array_equal(za[k], zb[k]), k


def test_aadvqz_column_matches_interface_sweep():
    """Synthetic consistency: a column pass = successive interface sweeps with the carry (one column in an otherwise zero field)."""
    rng = np.random.default_rng(9)
    nl = 6
    mass = 1e3 * (1 + rng.random(nl)); rm = mass * rng.random(nl) * 0.01
    mom = rng.normal(0, 1, (9, nl)) * rm[None] * 0.2
    mw = rng.normal(0, 100, nl); mw[nl - 1] = 0.
    r, m, ma = aq.aadvqz_column(rm, mom, mass, mw)
    R = np.zeros((IM, JM, nl)); R[5, 7] = rm; Mo = np.zeros((9, IM, JM, nl)); Mo[:, 5, 7] = mom; Ma = np.ones((IM, JM, nl)); Ma[5, 7] = mass
    carry = [np.zeros((IM, JM)) for _ in range(3)] + [np.zeros((9, IM, JM))]
    imaxj = np.full(JM, IM); z = np.zeros(JM)
    W = np.zeros((IM, JM, nl)); W[5, 7] = mw
    for l in range(nl):
        l2 = min(l + 1, nl - 1)
        o = aq.aadvqz(R[:, :, l], Mo[:, :, :, l], Ma[:, :, l], R[:, :, l2], Mo[:, :, :, l2], Ma[:, :, l2], W[:, :, l], *carry, imaxj, z, z, z)
        R[:, :, l], Mo[:, :, :, l], Ma[:, :, l] = o['rm'], o['rmom'], o['mass']
        carry = [o['mwdn'], o['fdn'], o['fdn0'], o['fmomdn']]
    assert np.array_equal(R[5, 7], r) and np.array_equal(Mo[:, 5, 7], m) and np.array_equal(Ma[5, 7], ma)


def test_aadvq0_hand_cases():
    mb = np.full((IM, JM, LM), 1e6); z = np.zeros((IM, JM, LM)); imaxj = np.where((np.arange(JM) == 0) | (np.arange(JM) == JM - 1), 1, IM)
    byim = 1 / 72.
    r = aq.aadvq0(z, z, z, mb, imaxj, byim)
    assert r['ncyc'] == 1 and r['ncycxy'].max() == 1 and r['nstepx'][1:JM - 1].max() == 1 and not r['do_z_extra']
    # pure zonal outflow of 0.9 mass with no vertical refill: mass ratio < 0.25 at every ncyc -> stop_model
    mu = z.copy(); mu[29, 19, 9] = 0.9e6
    with pytest.raises(aq.QusError, match='ncyc>ncmax'):
        aq.aadvq0(mu, z, z, mb, imaxj, byim)
    # same outflow refilled by a vertical flux 0.9 mass: ncyc=2 (hand-derived, see dyn_qus3d_harness)
    mw = z.copy(); mw[29, 19, 9] = 0.9e6
    assert aq.aadvq0(mu, z, mw, mb, imaxj, byim)['ncyc'] == 2
    # meridional outflow 0.9 + zonal inflow 0.3: ncycxy = 2 on that level only
    mv = z.copy(); mv[29, 20, 14] = 0.9e6; mu2 = z.copy(); mu2[28, 19, 14] = 0.3e6
    r = aq.aadvq0(mu2, mv, z, mb, imaxj, byim)
    assert r['ncyc'] == 1 and r['ncycxy'][14] == 2 and r['ncycxy'].sum() == LM + 1
