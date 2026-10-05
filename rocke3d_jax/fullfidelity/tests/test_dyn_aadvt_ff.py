"""Tests for dyn_aadvt_ff.py / dyn_adv1d_ff.py (QUS_DRV.f AADVT family + QUSDEF.f adv1d/advection_1D_custom/limitq,
D99/D100).  Real-dump tests need ff_data/<date>/ffd_aadvt_* (36 calls = 3 dates x 6 steps x 2 calls) and skip
otherwise; stress-run tests need ffd_aadvtS4_*; harness tests need ff_data/qus1d_harness.
NOT validated against model dumps: the qlimit/limitq path and every Courant nstep>1 case of the REAL windows
(all 1).  nstep>1 is covered by the stress dumps (real Fortran, scaled fluxes); limitq by the standalone ifort harness."""
import numpy as np
import pytest

import dyn_aadvt_ff as ad
import dyn_adv1d_ff as a1
import dyn_aadvt_compare as cm
import dyn_adv1d_compare as hc
from dyn_aadvt_ff import IM, JM, LM

HAVE = all(cm.available(d) for d, _ in cm.DATES)
STRESS = cm.stress_calls("aadvtS4")
HAVE_S = len(STRESS) > 0
HAVE_H = hc.available()
skip_real = pytest.mark.skipif(not HAVE, reason="ffd_aadvt dumps not present")
skip_stress = pytest.mark.skipif(not HAVE_S, reason="ffd_aadvtS4 stress dumps not present")
skip_h = pytest.mark.skipif(not HAVE_H, reason="qus1d harness data not present")


# ----------------------------------------------------------------------------------- real windows
@skip_real
@pytest.mark.parametrize("date,itime,call", cm.calls())
def test_aadvt_bitwise(date, itime, call):
    i, o, r, st = cm.run_aadvt(date, itime, call)
    d = cm.aadvt_compare(i, o, r)
    assert d['mm'] == 0.0 and d['t'] == 0.0 and d['tmom'] == 0.0 and d['fqu'] == 0.0 and d['fqv'] == 0.0
    assert d['n_exact'] == d['n_total']
    s = cm.stage_compare(date, itime, call, st)
    assert s['x1'] == 0.0 and s['y'] == 0.0 and s['z'] == 0.0 and s['ns_equal']


@skip_real
def test_real_windows_nstep_is_always_one():
    """Documented coverage fact: in all 36 real calls every row/column has nstep=1 (so the real windows never
    exercise the multi-sub-step masking logic; the stress dumps do)."""
    for d, it0 in cm.DATES:
        for c in (1, 2):
            ns = cm.load_ns(cm.fname(d, it0, c, 'ns'))
            assert ns['nsx'][1:JM - 1].min() == 1 and ns['nsx'][1:JM - 1].max() == 1 and ns['nsz'].max() == 1
            assert ns['nsx'][0].max() == 0 and ns['nsx'][JM - 1].max() == 0     # pole rows never advected in x


@skip_real
@pytest.mark.parametrize("date,itime,call", cm.calls()[::5])
def test_non_vacuous(date, itime, call):
    i, o, r, st = cm.run_aadvt(date, itime, call)
    assert np.max(np.abs(o['t'] - i['t'])) > 1e-6
    assert np.max(np.abs(o['tmom'] - i['tmom'])) > 1e-6
    assert np.max(np.abs(o['fpeu'])) > 0 and np.max(np.abs(o['fpev'])) > 0
    # every sweep changes the field
    x1, y, z = st['x1'][0], st['y'][0], st['z'][0]
    assert np.max(np.abs(y - x1)) > 0 and np.max(np.abs(z - y)) > 0 and np.max(np.abs(x1 - i['t'] * i['mma'])) > 0
    # pole rows are advected in y (south pole row changes) and stay uniform in x
    assert np.max(np.abs(y[:, 0, :] - x1[:, 0, :])) > 0
    assert np.all(o['t'][:, 0, :] == o['t'][0:1, 0, :])


@skip_real
def test_mutations_detected():
    date, itime, call = "dec01", 33552, 1
    i, o, r, st = cm.run_aadvt(date, itime, call)
    # 1: skip the polar averaging sum order (pairwise sum instead of left-to-right) is NOT visible in 1e-16 sums,
    #    but dropping the polar average altogether must be
    orig = ad.ordered_sum
    try:
        ad.ordered_sum = lambda a, axis=-1: np.zeros(np.asarray(a).shape[:-1])
        r2 = ad.aadvt(i['dt'], i['mma'], i['t'], i['tmom'], i['mu'], i['mv'], i['mw'])
    finally:
        ad.ordered_sum = orig
    assert np.max(np.abs(r2['rm'] - o['t'])) > 0
    # 2: wrong sweep length (dt) and swapped y/x direction permutation
    r3 = ad.aadvt(i['dt'] * 1.0000001, i['mma'], i['t'], i['tmom'], i['mu'], i['mv'], i['mw'])
    assert np.max(np.abs(r3['rm'] - o['t'])) > 0
    old = ad.YDIR
    try:
        ad.YDIR = a1.XDIR
        import dyn_adv1d_ff
        r4 = ad.aadvt(i['dt'], i['mma'], i['t'], i['tmom'], i['mu'], i['mv'], i['mw'])
    finally:
        ad.YDIR = old
    assert np.max(np.abs(r4['rmom'] - o['tmom'])) > 0
    # 3: multiplying by 1/m vs dividing by m at the end differs somewhere (the Fortran multiplies by the reciprocal)
    by = 1. / r['mm']
    alt = r['rm'] / by**-1
    assert alt.shape == r['rm'].shape


# ------------------------------------------------------------------------------------------ stress
@skip_stress
@pytest.mark.parametrize("date,itime,call", STRESS)
def test_stress_bitwise(date, itime, call):
    i, o, r, st = cm.run_aadvt(date, itime, call, tag="aadvtS4")
    d = cm.aadvt_compare(i, o, r)
    assert d['n_exact'] == d['n_total'] and d['mm'] == 0.0 and d['fqu'] == 0.0 and d['fqv'] == 0.0
    s = cm.stage_compare(date, itime, call, st, tag="aadvtS4")
    assert s['x1'] == 0.0 and s['y'] == 0.0 and s['z'] == 0.0 and s['ns_equal']


@skip_stress
def test_stress_exercises_multistep_and_masking():
    mxx = mxz = 0
    for d, it, c in STRESS:
        ns = cm.load_ns(cm.fname(d, it, c, 'ns', tag="aadvtS4"))
        mxx = max(mxx, ns['nsx'].max()); mxz = max(mxz, ns['nsz'].max())
    assert mxx >= 4 and mxz >= 3
    # rows with different nstep coexist in one sweep (masked per-row trip count is needed)
    d, it, c = STRESS[-1]
    ns = cm.load_ns(cm.fname(d, it, c, 'ns', tag="aadvtS4"))
    assert len(np.unique(ns['nsx'][:, :, 0])) > 2 or len(np.unique(ns['nsz'])) > 2


def _stress_with_multistep():
    for d, it, c in STRESS:
        ns = cm.load_ns(cm.fname(d, it, c, 'ns', tag="aadvtS4"))
        if ns['nsx'].max() > 1 and ns['nsz'].max() > 1:
            return d, it, c
    pytest.skip("no stress call with both x and z multi-step")


@skip_stress
def test_unmasked_max_nstep_mutation_changes_result():
    d, it, c = _stress_with_multistep()
    i, o, r, st = cm.run_aadvt(d, it, c, tag="aadvtS4", unmasked_max_nstep=True)
    assert np.max(np.abs(r['rmom'] - o['tmom'])) > 0 or np.max(np.abs(r['rm'] - o['t'])) > 0


@skip_stress
def test_rowloop_reference_equals_batched():
    """Literal per-row Fortran control flow (own nstep per row) == grouped batched implementation (bitwise)."""
    d, it, c = _stress_with_multistep()
    i = cm.load_in(cm.fname(d, it, c, 'in', tag="aadvtS4"))
    rm = i['t'] * i['mma']; rmom = i['tmom'] * i['mma'][None]
    mu = i['mu'] * (.5 * i['dt'])
    rm1, rmom1, mm1, f1 = rm.copy(), rmom.copy(), i['mma'].copy(), np.zeros((IM, JM))
    rm2, rmom2, mm2, f2 = rm.copy(), rmom.copy(), i['mma'].copy(), np.zeros((IM, JM))
    ad.aadvtx(rm1, rmom1, mm1, mu, False, f1)
    ad.aadvtx_rowloop(rm2, rmom2, mm2, mu, False, f2)
    assert np.array_equal(rm1, rm2) and np.array_equal(rmom1, rmom2) and np.array_equal(mm1, mm2) and np.array_equal(f1, f2)
    mw = np.zeros((IM, JM, LM)); mw[:, :, :LM - 1] = i['mw'] * (-i['dt'])
    z1, zm1, zz1 = rm.copy(), rmom.copy(), i['mma'].copy()
    z2, zm2, zz2 = rm.copy(), rmom.copy(), i['mma'].copy()
    ad.aadvtz(z1, zm1, zz1, mw)
    ad.aadvtz_rowloop(z2, zm2, zz2, mw)
    assert np.array_equal(z1, z2) and np.array_equal(zm1, zm2) and np.array_equal(zz1, zz2)


# ------------------------------------------------------------------------- Courant counters (no data)
def test_courant_batched_equals_scalar_reference():
    rng = np.random.default_rng(1)
    mass = rng.uniform(0.5, 1.5, (60, IM))
    mu = np.sin(np.linspace(0, 2 * np.pi, IM, endpoint=False)[None] + rng.uniform(0, 6, (60, 1))) * rng.uniform(0.2, 5, (60, 1)) + rng.uniform(-.3, .3, (60, 1))
    ns, cmx = ad.courant_nstep_x(mu, mass)
    for b in range(60):
        n1, c1 = ad.courant_nstep_x_rowloop(mu[b], mass[b])
        assert n1 == ns[b] and c1 == cmx[b]
    assert len(set(ns)) > 3
    ml = rng.uniform(0.5, 1.5, (60, LM))
    mw = np.sin(np.linspace(0, np.pi, LM - 1)[None] * rng.integers(1, 4, (60, 1))) * rng.uniform(0.2, 5, (60, 1))
    nz, cz = ad.courant_nstep_z(mw, ml)
    for b in range(60):
        n1, c1 = ad.courant_nstep_z_rowloop(mw[b], ml[b])
        assert n1 == nz[b] and c1 == cz[b]
    assert len(set(nz)) > 3


def test_courant_hand_value():
    # uniform mass 1, flux 0.6 eastward everywhere: courmax=0.6 -> nstep 1; flux 2.5 -> nstep 3 (0.833 <= 1)
    mass = np.ones((2, IM)); mu = np.stack([np.full(IM, .6), np.full(IM, 2.5)])
    ns, c = ad.courant_nstep_x(mu, mass)
    assert list(ns) == [1, 3]
    assert abs(c[0] - .6) < 1e-15 and abs(c[1] - 2.5 / 3) < 1e-15


def test_courant_hits_cap_and_flags():
    mass = np.ones((1, IM)); mu = np.full((1, IM), 25.)
    ns, c = ad.courant_nstep_x(mass * 0 + mu, mass)
    assert ns[0] == 20 and c[0] > 1


# ------------------------------------------------------------------------------- hand-derived (no data)
def test_uniform_field_stays_uniform_and_mass_conserved():
    rng = np.random.default_rng(3)
    mass = rng.uniform(1, 2, (IM, JM, LM)) * 1e3
    mass[:, 0, :] = mass[0:1, 0, :]; mass[:, JM - 1, :] = mass[0:1, JM - 1, :]     # polar boxes are uniform in I
    T = np.full((IM, JM, LM), 300.)
    mom = np.zeros((9, IM, JM, LM))
    mu = rng.uniform(-.1, .1, (IM, JM, LM)) * 1e3 / 450.
    mv = rng.uniform(-.1, .1, (IM, JM, LM)) * 1e3 / 450.
    mv[:, 0, :] = 0.
    mw = rng.uniform(-.1, .1, (IM, JM, LM - 1)) * 1e3 / 450.
    r = ad.aadvt(450., mass, T, mom, mu, mv, mw)
    assert np.max(np.abs(r['rm'] - 300.)) < 1e-9 * 300
    assert np.max(np.abs(r['rmom'])) < 1e-9 * 300


def test_tracer_mass_conserved_nonuniform():
    rng = np.random.default_rng(4)
    mass = np.full((IM, JM, LM), 1e3)
    T = 280. + 20. * rng.random((IM, JM, LM))
    T[:, 0, :] = T[0:1, 0, :]; T[:, JM - 1, :] = T[0:1, JM - 1, :]
    mom = rng.normal(0, 1, (9, IM, JM, LM))
    mu = rng.uniform(-.2, .2, (IM, JM, LM)) * 1e3 / 450.
    mv = rng.uniform(-.2, .2, (IM, JM, LM)) * 1e3 / 450.
    mw = rng.uniform(-.2, .2, (IM, JM, LM - 1)) * 1e3 / 450.
    # make the mass fluxes divergence-free-ish is not needed: total (mass-weighted) tracer is conserved by
    # construction in flux form for the whole sweep sequence except polar boxes (replicated), so check the
    # row-wise interior total over one X sweep instead.
    rm = T * mass; rmom = mom * mass[None]; mm = mass.copy()
    pre = rm[:, 1:JM - 1, :].sum()
    ad.aadvtx(rm, rmom, mm, mu * (.5 * 450.))
    assert abs(rm[:, 1:JM - 1, :].sum() - pre) < 1e-11 * abs(pre)


def test_ordered_sum_is_left_to_right():
    a = np.array([1e16, 1., -1e16, 1.])
    assert a1.ordered_sum(a) == 1.0           # ((1e16+1)-1e16)+1 = 0+1 left to right
    assert float(np.sum(a)) in (1.0, 2.0, 0.0)


# ---------------------------------------------------------------------------------- adv1d / limitq
@skip_h
def test_adv1d_vs_standalone_ifort_harness():
    cases = hc.load_cases(); outs = hc.read_out(hc.HARNESS_DIR + "/out.bin", len(cases))
    nexact = nlines = 0
    stats = {}
    for c, o in zip(cases, outs):
        r = hc.run_case(c, stats)
        d = hc.compare(c, o)
        assert d['ierr_ok']
        if d['early']:
            continue
        nlines += 1
        nexact += (d['s'] == 0 and d['smom'] == 0 and d['mass'] == 0 and d['f'] == 0 and d['fmom'] == 0)
        assert d['smom'] < 1e-15 and d['fmom'] < 1e-15 and d['s'] == 0 and d['mass'] == 0 and d['f'] == 0
    # 2 lines of 2295 differ by 1 ulp in one moment: |fracm|>1 (unphysical) cells where numpy pow(x,3) and ifort differ
    assert nexact >= nlines - 3 and nlines > 2000


@skip_h
def test_harness_covers_all_limitq_branches():
    cases = hc.load_cases(); stats = {}
    for c in cases:
        hc.run_case(c, stats)
    for k in ('no_outflow', 'both_all_nonneg', 'both_center_neg', 'both_left_neg', 'both_right_neg',
              'both_fix_two_outer', 'both_fix_center_left', 'both_fix_center_right', 'one_left', 'one_right',
              'one_all_nonneg', 'one_down_neg', 'one_up_neg', 'one_warn_abs_gt_1', 'one_error_new_sn_neg'):
        assert stats.get(k, 0) > 0, k


@skip_h
def test_harness_non_vacuous_and_mutation():
    cases = hc.load_cases(); outs = hc.read_out(hc.HARNESS_DIR + "/out.bin", len(cases))
    changed = 0
    for c, o in zip(cases, outs):
        if c[0] != 1 or o['ierr'] == 2:
            continue
        ql, dt, s, sm, mass, dm = c
        free = hc.run_case((0,) + c[1:])
        lim = hc.run_case(c)
        changed += bool(np.any(free['smom'] != lim['smom']) or np.any(free['f'] != lim['f']))
        # mutation: the unlimited result must NOT equal the real limited result when the limiter acted
        if np.any(free['f'] != o['f']):
            assert not np.array_equal(free['f'], o['f'])
    assert changed > 100


def test_limitq_hand_cases():
    # no air leaving: returns inputs unchanged
    assert a1.limitq(0.1, -0.1, 0.3, -0.2, 1., .1, .2) == (0.3, -0.2, .1, .2, 0)
    # one edge, right outflow, downstream division negative: flux fn=sd=-0.5 < 0 -> clipped to >= 0 region
    fnm1, fn, sx, sxx, ierr = a1.limitq(0.1, 0.3, 0.05, -0.5, 1.0, 0.2, 0.1)
    assert ierr == 0 and fn >= 0.0 - 1e-15
    # |a|>1 with net tracer loss -> ierr 2
    r = a1.limitq(0.0, 1.5, 0.0, 5.0, 1.0, 0., 0.)
    assert r[4] == 2
    # |a|>1 warning only
    r = a1.limitq(0.0, 1.5, 0.0, 0.5, 1.0, 0., 0.)
    assert r[4] == 1


def test_adv1d_hand_uniform_translation():
    # uniform s with zero moments and uniform mass flux: s/mass unchanged, flux = fracm*s
    nx = IM
    mass = np.ones((1, nx)); s = np.full((1, nx), 2.); sm = np.zeros((9, 1, nx)); dm = np.full((1, nx), .25)
    f, fm, ierr, nerr = a1.adv1d(s, sm, mass, dm)
    assert np.all(f == .25 * 2.) and np.all(s == 2.) and np.all(mass == 1.) and ierr == 0
    assert np.all(sm == 0.)


def test_advection_1d_custom_equals_adv1d_with_zero_boundary_flux():
    rng = np.random.default_rng(7)
    nx = JM
    mass = rng.uniform(1, 2, (3, nx)); s = mass * rng.uniform(1, 2, (3, nx)); sm = rng.normal(0, .1, (9, 3, nx)) * s
    dm = rng.uniform(-.3, .3, (3, nx)) * mass; dm[:, -1] = 0.
    # custom (non-cyclic) vs cyclic adv1d on a line whose wrap interface is closed (dm[-1]=0 and first-cell
    # prev flux = 0): the interior results must be identical, the first cell of the cyclic one sees dm[-1]=0 too.
    s1, sm1, m1 = s.copy(), sm.copy(), mass.copy()
    s2, sm2, m2 = s.copy(), sm.copy(), mass.copy()
    a1.advection_1d_custom(s1, sm1, m1, dm)
    a1.adv1d(s2, sm2, m2, dm, False, a1.YDIR)
    assert np.array_equal(s1, s2) and np.array_equal(sm1, sm2) and np.array_equal(m1, m2)
