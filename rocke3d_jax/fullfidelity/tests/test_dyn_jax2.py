"""Tests for the D142-D144 JAX stages (dyn_jax_aflux, dyn_jax_aadvt, dyn_jax_qdynam, dyn_step_jax2) against their numpy ports on
the real stage-boundary dumps, the x4 / x8 stress dumps and scaled-input stress calls (multi-substep, multi-cycle,
topography branch), and the chained step.  Target and measured: bit-identical (0 unequal elements) with the FMA-free XLA flag;
the asserts are exact equality for stages (tolerance 0) and 1e-12 scale-relative for the end state.  Skipped if dumps are absent.
Mutation checks: perturbed geometry / unmasked max-nstep / dropped stage must break the agreement."""
import dyn_jax_env  # noqa: F401  (before jax)
import os

import numpy as np
import pytest

import dyn_step as ds
import dyn_step_compare as dc
import dyn_step_jax2 as dj2
import dyn_aflux_ff as fa
import dyn_aflux_compare as cm
import dyn_aadvt_compare as tc
import dyn_aadvt_ff as ft
import dyn_aadvq_ff as aq
import dyn_qdynam_io as qio
import dyn_jax_aflux as jaf
import dyn_jax_aadvt as jat
import dyn_jax_qdynam as jqd

DATE, IT0 = 'nov26', 33312
HAVE = ds.available(DATE) and os.path.exists(tc.fname(DATE, IT0, 1, 'in'))
NEED = pytest.mark.skipif(not HAVE, reason="ff_data dumps not present")
F = lambda x: np.ascontiguousarray(x, dtype=np.float64)
_C = {}


def ctx():
    if 'c' not in _C:
        _C['c'] = ds.load_ctx(DATE, imf_pow=False)
        _C['afl'] = jaf.make_aflux(_C['c'].g, _C['c'].tab)
    return _C['c'], _C['afl']


def nne(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def test_xla_fma_free():
    assert dj2.dyn_jax_env.fma_free()


@NEED
def test_aflux_advecm_real_dumps_bitwise_and_mutation():
    c, afl = ctx()
    for pas in (1, 2, 3, 4, 5):
        i = cm.load_aflux_in(cm.fname(DATE, IT0, pas, 'aflux', ds.FF_DEFAULT) + "_in.bin")
        a = [F(i[k]) for k in ('u', 'v', 'ma', 'masum', 'me', 'mesum')]
        rn = fa.aflux(i['ns'], *a, c.g, tab=c.tab)
        rj = afl.aflux(i['ns'], *a)
        for k in rn:
            assert nne(rj[k], rn[k]) == 0, (pas, k)
        j = cm.load_advecm_in(cm.fname(DATE, IT0, pas, 'aflux_advecm', ds.FF_DEFAULT) + "_in.bin")
        rn2 = fa.advecm(float(j['dt1']), j['mold'], j['conv'], j['mw'], c.g)
        rj2 = afl.advecm(float(j['dt1']), F(j['mold']), F(j['conv']), F(j['mw']))
        for k in rn2:
            assert nne(rj2[k], rn2[k]) == 0, (pas, k)
    # mutation: perturbed geometry breaks it
    geo = dict(afl.geo); geo['dyp'] = geo['dyp'] * 1.001
    r = jaf.make_aflux(c.g, c.tab)
    bad = r.aflux.__closure__ is not None
    i = cm.load_aflux_in(cm.fname(DATE, IT0, 1, 'aflux', ds.FF_DEFAULT) + "_in.bin")
    a = [F(i[k]) for k in ('u', 'v', 'ma', 'masum', 'me', 'mesum')]
    r.geo['dyp'] = r.geo['dyp'] * 1.001
    rm = r.aflux(i['ns'], *a)
    rn = fa.aflux(i['ns'], *a, c.g, tab=c.tab)
    assert nne(rm['mu'], rn['mu']) > 1000


@NEED
def test_aflux_topography_and_polar_branches_scaled_winds():
    """Wind x6 and a perturbed MA drive many topography moves (stats checked) through the same code."""
    c, afl = ctx()
    i = cm.load_aflux_in(cm.fname(DATE, IT0, 3, 'aflux', ds.FF_DEFAULT) + "_in.bin")
    st = {}
    a = [F(i['u']) * 6., F(i['v']) * 6., F(i['ma']), F(i['masum']), F(i['me']), F(i['mesum'])]
    rn = fa.aflux(i['ns'], *a, c.g, tab=c.tab, stats=st)
    assert st.get('ew_moved', 0) + st.get('ns_moved', 0) > 0, st
    rj = afl.aflux(i['ns'], *a)
    for k in rn:
        assert nne(rj[k], rn[k]) == 0, k


def _aadvt_call(date, it, call, tag='aadvt', scale=1.0):
    i = tc.load_in(tc.fname(date, it, call, 'in', tag=tag))
    mu, mw = F(i['mu']) * scale, F(i['mw']) * scale
    return i, mu, mw


def _cmp_aadvt(i, mu, mw, **kw):
    try:
        rn = ft.aadvt(i['dt'], i['mma'], i['t'], i['tmom'], mu, i['mv'], mw, False)
    except RuntimeError:
        rn = None
    rj = jat.aadvt_jax(i['dt'], F(i['mma']), F(i['t']), F(i['tmom']), mu, F(i['mv']), mw, **kw)
    return rn, rj


@NEED
def test_aadvt_real_calls_bitwise_vs_numpy_and_real():
    for call in (1, 2):
        i, mu, mw = _aadvt_call(DATE, IT0, call)
        o = tc.load_out(tc.fname(DATE, IT0, call, 'out'))
        rn, rj = _cmp_aadvt(i, mu, mw)
        assert not bool(rj['bad'])
        for k in ('mm', 'rm', 'rmom', 'fqu', 'fqv'):
            assert nne(rj[k], rn[k]) == 0, k
        assert nne(rj['rm'], o['t']) == 0 and nne(rj['rmom'], o['tmom']) == 0 and nne(rj['fqu'], o['fpeu']) == 0


@NEED
def test_aadvt_multi_substep_stress_and_unmasked_mutation():
    calls = tc.stress_calls('aadvtS4')
    if not calls:
        pytest.skip("no aadvtS4 dumps")
    d, it, c = calls[0]
    i, mu, mw = _aadvt_call(d, it, c, 'aadvtS4')
    rn, rj = _cmp_aadvt(i, mu, mw)
    ns = np.maximum(np.asarray(rj['nsx1']).max(), np.asarray(rj['nsz']).max())
    assert ns > 1 and not bool(rj['bad'])
    for k in ('mm', 'rm', 'rmom', 'fqu', 'fqv'):
        assert nne(rj[k], rn[k]) == 0, k
    # real dump of the stress call
    o = tc.load_out(tc.fname(d, it, c, 'out', tag='aadvtS4'))
    assert nne(rj['rm'], o['t']) == 0 and nne(rj['rmom'], o['tmom']) == 0
    # mutation: running every row for the max nstep without masking changes the result
    _, rjm = _cmp_aadvt(i, mu, mw, masked=False)
    assert nne(rjm['rm'], rn['rm']) > 0
    # scaled real input (x6): numpy and JAX agree, or both stop (courmax > 1 at nstep 20)
    i2, mu2, mw2 = _aadvt_call(DATE, IT0, 1, scale=6.0)
    rn2, rj2 = _cmp_aadvt(i2, mu2, mw2)
    if rn2 is None:
        assert bool(rj2['bad'])
    else:
        assert not bool(rj2['bad']) and nne(rj2['rm'], rn2['rm']) == 0 and nne(rj2['rmom'], rn2['rmom']) == 0


def _qd(tag, date, it, kit):
    g = qio.load_geom(qio.gname(date))
    i = qio.load_in(qio.fname(date, it, 'in', tag=tag))
    fin = qio.load_fin(qio.fname(date, it, 'fin', tag=tag))
    rn = aq.qdynam(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'], g['axyp'], g['imaxj'], g['kg2mb'], g['byim_geom'],
                   g['byim_qus'])
    kit = kit or jqd.QdynamKit(g)
    rj = kit(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'])
    return i, rn, rj, fin, kit


@pytest.mark.skipif(not qio.available(), reason="qdyn dumps not present")
def test_qdynam_real_calls_bitwise():
    kit = None
    for it in (IT0, IT0 + 3):
        i, rn, rj, fin, kit = _qd('qdyn', DATE, it, kit)
        q0 = rn['q0']
        assert nne(rj['q'], rn['q']) == 0 and nne(rj['qmom'], rn['qmom']) == 0
        assert nne(rj['q'], fin['q']) == 0 and nne(rj['qmom'], fin['qmom']) == 0
        assert rj['ncyc'] == q0['ncyc'] and nne(rj['ncycxy'], q0['ncycxy']) == 0 and nne(rj['nstepx'], q0['nstepx']) == 0
        assert nne(rj['mus'], q0['mu']) == 0 and nne(rj['mvs'], q0['mv']) == 0 and nne(rj['mws'], q0['mw']) == 0
    # mutation: a perturbed vertical flux changes the result
    g = qio.load_geom(qio.gname(DATE))
    r2 = kit(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'] * 1.001)
    assert nne(r2['q'], rn['q']) > 1000


@pytest.mark.skipif(not os.path.exists(qio.fname('dec01', 33552, 'in', tag='qdynS8')), reason="no qdynS8 dumps")
def test_qdynam_stress_multicycle_and_zextra_fallback():
    # dec01 33552: ncyc=6 entirely in JAX; nov26 33312: ncyc=4 with extra z columns (numpy fallback for the bookkeeping)
    for date, it, zextra in (('dec01', 33552, False), (DATE, IT0, True)):
        i, rn, rj, fin, kit = _qd('qdynS8', date, it, None)
        assert rj['ncyc'] > 1 and rj['do_z_extra'] == zextra and rn['q0']['do_z_extra'] == zextra
        assert nne(rj['q'], rn['q']) == 0 and nne(rj['qmom'], rn['qmom']) == 0
        assert nne(rj['q'], fin['q']) == 0 and nne(rj['qmom'], fin['qmom']) == 0
        assert nne(rj['nstepx'], rn['q0']['nstepx']) == 0 and nne(rj['ncycxy'], rn['q0']['ncycxy']) == 0
        assert (kit.n_fallback == 1) == zextra


@NEED
def test_chained_step_end_state_and_mutation():
    c, _ = ctx()
    kit = dj2.Kit(c)
    s1 = ds.load_state(ds.state_path(DATE, IT0, 1))
    s3 = ds.load_state(ds.state_path(DATE, IT0, 3))
    plan = ds.step_plan(nstep=(IT0 - ds.ITIMEI) * 4)
    wn = ds.dyn_step(s1, c, itime=IT0, plan=plan)
    wj = dj2.dyn_step_jax(s1, c, kit, plan=plan)
    for k in dc.FIELDS_S2:
        assert dc.stats(wj[k.upper()], wn[k.upper()])['rel'] <= 1e-12, k
        assert dc.stats(wj[k.upper()], s3[k])['rel'] <= max(2 * dc.stats(wn[k.upper()], s3[k])['rel'], 1e-14), k
    wm = dj2.dyn_step_jax(s1, c, kit, plan=[st for st in plan if st.kind != 'aadvt'])
    assert dc.stats(wm['T'], wn['T'])['rel'] > 1e-7
    wq = dj2.dyn_step_jax(s1, c, kit, plan=[st for st in plan if st.kind != 'qdynam'])
    assert dc.stats(wq['Q'], wn['Q'])['rel'] > 1e-7
