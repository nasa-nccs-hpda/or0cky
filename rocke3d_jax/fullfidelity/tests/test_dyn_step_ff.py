"""Tests for dyn_step.py (D122/D123): the whole dynamics block of one physics step chained from the ports, validated against
the real-Fortran state dumps ffd_state_<itime>_s1..s4 (ATM_DRV_dynG.f.patch) and the per-call stage dumps.

* plan-structure tests need no dumps (the call order of DYNAM: 5 leapfrog passes, 2 AADVT, 2 SDRAG, ...);
* dump-based tests (skipped if the ff_data dumps are absent): bitwise chain of all 18 steps (3 dates x 6) with the Intel libimf
  pow (skipped without the Intel runtime), numpy-pow tolerance test (bounds with margin over the measured worst scale-relative
  differences, see the test docstring and dyn_step_compare.py --numpy-pow), per-stage input/output checks, and mutation checks: dropping or
  reordering a stage must make the chained result differ from the real state (or raise).
"""
import copy

import numpy as np
import pytest

import dyn_step as ds
import dyn_step_compare as sc
import intel_libm_ff

HAVE = all(ds.available(d) for d, _ in ds.DATES)
NEED = pytest.mark.skipif(not HAVE, reason="ff_data ffd_state_* dumps not present on this host")
NO_IMF = not intel_libm_ff.available()
NEEDI = pytest.mark.skipif(not HAVE or NO_IMF, reason="needs the ff_data state dumps and the Intel libimf (bitwise mode)")
STEPS = [(d, it0 + k) for d, it0 in ds.DATES for k in range(ds.NSTEP)]
_CTX = {}


def ctx(date, imf):
    if (date, imf) not in _CTX:
        _CTX[(date, imf)] = ds.load_ctx(date, imf_pow=imf)
    return _CTX[(date, imf)]


# ----------------------------------------------------------------------------- plan structure (no dumps)
def kinds(plan):
    return [s.kind for s in plan]


def test_plan_call_counts():
    p = ds.dynam_plan()
    k = kinds(p)
    assert [k.count(x) for x in ('aflux', 'advecm', 'advecv', 'pgf', 'iso')] == [5] * 5
    assert k.count('aadvt') == 2 and k.count('sdrag') == 2 and k.count('filter_chain') == 1
    assert k.count('accum') == 2 and k.count('flux_zero') == 1 and k.count('reinit') == 1
    assert k[0] == 'flux_zero' and k[-1] == 'filter_chain' and k[-3:-1] == ['matopmb', 'flux_scale']


def test_plan_pass_order_and_ns():
    p = ds.dynam_plan()
    af = [s for s in p if s.kind == 'aflux']
    assert [s.pas for s in af] == [1, 2, 3, 4, 5]
    assert [s.a['ns'] for s in af] == [4, 4, 4, 3, 2]
    # argument bindings = actual arguments of the Fortran CALLs (ATMDYN.f:303-345)
    assert (af[1].a['u'], af[1].a['ma'], af[1].a['me']) == ('UX', 'MODD3', 'MA')
    assert (af[3].a['u'], af[3].a['ma'], af[3].a['me'], af[3].a['mesum']) == ('U', 'MA', 'MODD1', 'MSUMODD')
    assert (af[4].a['u'], af[4].a['ma'], af[4].a['me'], af[4].a['mesum']) == ('UT', 'MODD1', 'MEVEN', 'MASUM')
    pg = [s for s in p if s.kind == 'pgf']
    assert [(s.a['s0'], s.a['sz']) for s in pg] == [('T', 'TZ'), ('T', 'TZ'), ('TT', 'TZT'), ('T', 'TZ'), ('TT', 'TZT')]
    assert [s.a['dt'] for s in pg] == [300.0, 450.0, 900.0, 900.0, 900.0]
    # AADVT sits after the even-pass ADVECV and before its PGF
    for call, pas in ((1, 3), (2, 5)):
        names = [s.name for s in p if s.pas == pas]
        assert names.index(f'p{pas}.advecv') < names.index(f'p{pas}.aadvt') < names.index(f'p{pas}.pgf') \
            < names.index(f'p{pas}.iso') < names.index(f'p{pas}.sdrag')


def test_plan_diaga_selection():
    # ATMDYN.f:352-357: MODDA = Mod(NSTEP+4-NS+NDAA*NIDYN, NDAA*NIDYN+2) < 2 in an even pass (NDAA=13, ITIMEI=16032)
    fire = {it for _, it0 in ds.DATES for it in range(it0, it0 + 6)
            if 'diaga' in kinds(ds.dynam_plan(nstep=(it - ds.ITIMEI) * 4))}
    assert fire == {33312, 33555}
    assert 'diaga' not in kinds(ds.dynam_plan())


def test_step_plan_order():
    k = kinds(ds.step_plan())
    assert k[:3] == ['save_old', 'se_init', 'ke_init']
    assert k[-9:] == ['wsave', 'qscale', 'qdynam', 'se_final', 'ke_final', 'efix', 'trop', 'pgrad', 'kea']


# ----------------------------------------------------------------------------- bitwise chain
@NEED
@pytest.mark.skipif(NO_IMF, reason="Intel libimf not available")
@pytest.mark.parametrize("date,itime", STEPS)
def test_chain_bitwise_libimf(date, itime):
    r = sc.run_step_compare(date, itime, ctx(date, True), per_stage=True)
    for tag in ('s2cmp', 's3cmp', 's4cmp'):
        for f, s in r[tag].items():
            assert s['nne'] == 0, (tag, f, s)
    for f, s in r.get('precond', {}).items():
        assert s['nne'] == 0, ('pre_condse', f, s)
    for f, s in r['stages'].items():
        assert s['nne'] == 0, (f, s)


@NEED
@pytest.mark.skipif(NO_IMF, reason="Intel libimf not available")
@pytest.mark.parametrize("date,itime", [(d, it0) for d, it0 in ds.DATES])
def test_boundary_replay_bitwise_libimf(date, itime):
    r = sc.run_step_compare(date, itime, ctx(date, True), boundary=True)
    assert len(r['stages']) >= 100
    for f, s in r['stages'].items():
        assert s['nne'] == 0, (f, s)


@NEED
@pytest.mark.parametrize("date,itime", [(d, it0) for d, it0 in ds.DATES])
def test_chain_numpy_pow_tolerance(date, itime):
    """Without libimf pow the 1-ulp pow differences propagate.  Measured worst scale-relative differences over all 18 steps
    (dyn_step_compare.py --numpy-pow): prognostic and s3 fields <= 3.8e-13 (u, v), s4 exports <= 5.6e-14 except the
    near-cancelling PGRAD_PBL pressure-gradient terms dpdx/dpdy/dpdx0/dpdy0 <= 2.7e-11.  Asserted bounds: 2e-12 and, for
    the four PGRAD terms, 1e-10.  ltropo (integer) must match exactly."""
    r = sc.run_step_compare(date, itime, ctx(date, False), per_stage=False)
    for tag in ('s3cmp', 's4cmp'):
        for f, s in r[tag].items():
            bound = 1e-10 if f in ('dpdx', 'dpdy', 'dpdx0', 'dpdy0') else 2e-12
            assert s['rel'] < bound, (tag, f, s)
    assert r['s4cmp']['ltropo']['nne'] == 0


@NEED
def test_state_changes_during_step():
    """Non-vacuity: the dumped end state differs from the start state in every prognostic field."""
    s1 = ds.load_state(ds.state_path('nov26', 33312, 1))
    s3 = ds.load_state(ds.state_path('nov26', 33312, 3))
    for f in ('u', 'v', 't', 'q', 'ma', 'tmom', 'qmom', 'qcl'):
        assert np.max(np.abs(s1[f] - s3[f])) > 0, f


# ----------------------------------------------------------------------------- mutation checks
MUT_DATE, MUT_ITIME = 'nov26', 33313            # a step on which DIAGA does not fire
MUT_FIELDS = ('u', 'v', 't', 'q', 'ma', 'tmom', 'qmom', 'mus', 'mvs', 'mws', 'pk', 'qcl', 'qci')


def mutated_differs(plan, date=MUT_DATE, itime=MUT_ITIME, use_s2=False):
    """True if the mutated plan raises or its final state differs from the real s3 (or, use_s2, DYNAM-exit s2) state."""
    c = ctx(date, True)
    s1 = ds.load_state(ds.state_path(date, itime, 1))
    ref = ds.load_state(ds.state_path(date, itime, 2 if use_s2 else 3))
    snap = {}

    def hook(when, st, w, cx):
        if when == 'post' and st.kind == 'filter_chain':
            snap.update({k: v.copy() for k, v in w.items() if isinstance(v, np.ndarray)})
    try:
        w = ds.dyn_step(s1, c, itime=itime, plan=plan, hook=hook)
    except Exception:
        return True
    src = snap if use_s2 else w
    return any(not np.array_equal(src[f.upper()], ref[f]) for f in MUT_FIELDS)


def base_plan(itime=MUT_ITIME):
    return ds.step_plan(nstep=(itime - ds.ITIMEI) * 4)


@NEEDI
def test_unmutated_plan_matches():
    assert not mutated_differs(base_plan())


DROP = [('aflux', 2), ('advecm', 3), ('advecv', 4), ('pgf', 1), ('pgf', 5), ('iso', 2), ('iso', 5), ('sdrag', 3),
        ('sdrag', 5), ('aadvt', 3), ('accum', 3), ('avg', 3), ('tz', 5), ('pscale', 3), ('pscale', 5), ('mma', 5), ('copy', 4),
        ('reinit', 1), ('matopmb', 0), ('flux_scale', 0), ('filter_chain', 0), ('qscale', 0), ('qdynam', 0),
        ('efix', 0), ('wsave', 0), ('trop', 0), ('kea', 0)]


@NEEDI
@pytest.mark.parametrize("kind,pas", DROP)
def test_dropping_a_stage_fails(kind, pas):
    plan = base_plan()
    idx = [i for i, s in enumerate(plan) if s.kind == kind and s.pas == pas]
    assert len(idx) >= 1, (kind, pas)
    del plan[idx[0]]
    if kind in ('wsave', 'trop', 'kea'):          # exports are not in MUT_FIELDS: the dropped stage leaves a key unset
        c = ctx(MUT_DATE, True)
        s1 = ds.load_state(ds.state_path(MUT_DATE, MUT_ITIME, 1))
        w = ds.dyn_step(s1, c, itime=MUT_ITIME, plan=plan)
        assert {'wsave': 'WSAVE', 'trop': 'PTROPO', 'kea': 'KEA'}[kind] not in w
    else:
        assert mutated_differs(plan)


def _swap(plan, i):
    plan = list(plan)
    plan[i], plan[i + 1] = plan[i + 1], plan[i]
    return plan


def _find(plan, name):
    return [i for i, s in enumerate(plan) if s.name == name][0]


# (swaps of independent neighbours are benign by construction and are not listed: e.g. advecv<->pscale, qscale<->qdynam)
SWAPS = ['p3.advecm', 'p3.pgf', 'p3.iso', 'p5.aadvt', 'p4.aflux', 'p2.advecv', 'p1.iso']


@NEEDI
@pytest.mark.parametrize("name", SWAPS)
def test_reordering_two_stages_fails(name):
    plan = base_plan()
    i = _find(plan, name)
    if name == 'p5.aadvt':                          # swap with the following 'tz' (z-moment copy must follow AADVT)
        assert plan[i + 1].kind == 'tz'
    assert mutated_differs(_swap(plan, i))


@NEEDI
def test_reordering_post_dynam_stages_fails():
    plan = base_plan()
    # QCL/QCI rescale moved after se_final (CONSERV_SE reads QCI): the energy fix changes
    j = _find(plan, 'qscale')
    st = plan.pop(j)
    plan.insert(_find(plan, 'se_final') + 1, st)
    assert mutated_differs(plan)
    # se_final before QDYNAM: SEFINAL computed with the old Q
    plan = base_plan()
    assert mutated_differs(_swap(plan, _find(plan, 'qdynam')))
    # efix before ke_final: KEFINAL undefined
    plan = base_plan()
    assert mutated_differs(_swap(plan, _find(plan, 'ke_final')))
    # efix before se_final
    plan = base_plan()
    j, k = _find(plan, 'efix'), _find(plan, 'se_final')
    plan[j], plan[k] = plan[k], plan[j]
    assert mutated_differs(plan)


@NEEDI
def test_diaga_stage_matters_on_firing_step():
    """On dec01 33555 DIAGA (inside DYNAM, pass 5) homogenises Q at the poles: without the stage the DYNAM-exit Q differs from
    the real s2 state (this was found as the only s1->s2 change of Q in the 18 steps); with it, s2 is reproduced exactly."""
    date, itime = 'dec01', 33555
    plan = base_plan(itime)
    assert 'diaga' in kinds(plan)
    assert not mutated_differs(plan, date, itime, use_s2=True)
    no = [s for s in plan if s.kind != 'diaga']
    assert mutated_differs(no, date, itime, use_s2=True)
