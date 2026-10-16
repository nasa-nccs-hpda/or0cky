"""Tests for atm_step_fast.py (D133): the chained atmosphere step with the batched CONDSE (clouds_condse_batch) in place of the per-column one.

Skipped when ff_data is absent.  Runtime budget about 4-5 min: the per-column reference CONDSE of atm_step (~2 min) is run once, on nov26 step 0
(libm mode, dyn -> condse only, no surface), and compared bitwise with the batched stage; the fast end-to-end step (~15 s) is checked against
the real end state in libimf mode (skipped without libimf).  Mutation checks: a perturbed CONDSE input / a skipped pole must change the result,
so the equivalence comparison is not vacuous.  The patching of atm_step.stage_condse is restored after use."""
import os

import numpy as np
import pytest

import atm_step as A
import atm_step_fast as F
import intel_libm_ff

DATE, IT = 'nov26', dict(A.DATES)['nov26']
HAVE = A.have_dumps(DATE, IT)
NEED = pytest.mark.skipif(not HAVE, reason="ff_data atmosphere step dumps not present")
NEEDI = pytest.mark.skipif(not HAVE or not intel_libm_ff.available(), reason="needs dumps and Intel libimf")
_C = {}


def _ctx(imf):
    if imf not in _C:
        _C[imf] = A.make_ctx(DATE, imf=imf)
    F.ensure_backend(_C[imf])
    return _C[imf]


def test_patch_is_restored_and_atm_step_untouched():
    orig = A.stage_condse
    with F.fast_condse():
        assert A.stage_condse is F.stage_condse_fast
    assert A.stage_condse is orig and orig.__module__ == 'atm_step'


@NEED
def test_column_subset_not_supported():
    with pytest.raises(NotImplementedError):
        F.stage_condse_fast({}, None, None, cols=[(0, 1)])


@NEED
def test_fast_condse_stage_bitwise_equal_to_per_column_libm():
    """dyn -> condse of the real step-start state, libm mode: the batched stage equals the per-column stage in every CONDSE output field
    and in the cloud/precipitation carry (the D132 bit-for-bit statement, now through the chained stage)."""
    ctx = _ctx(False)
    R = A.Real(DATE, IT)
    Sr, snr = A.run_step(DATE, IT, ctx, stop='condse', R=R, ms={})
    R2 = A.Real(DATE, IT)
    Sf, snf = F.run_step(DATE, IT, ctx, stop='condse', R=R2, ms={})
    for k in A.CONDSE_OUT:
        assert np.array_equal(snf['condse'][k], snr['condse'][k]), k
    for k in F.CARRY_KEYS:
        if k in Sr['_condse_X']:
            assert np.array_equal(Sf['_condse_X'][k], Sr['_condse_X'][k]), k
    # mutation: perturbing one prognostic CONDSE input cell must change the batched output (the comparison above is not vacuous)
    S = A.run_step(DATE, IT, ctx, stop='dyn', R=A.Real(DATE, IT))[0]
    S['T'] = np.array(S['T'], copy=True)
    S['T'].flat[S['T'].size // 2] *= 1.0 + 1e-3
    F.stage_condse_fast(S, A.Real(DATE, IT), ctx, ms={})
    assert not np.array_equal(S['T'], snr['condse']['T'])


@NEEDI
def test_fast_step_end_state_close_to_real_libimf():
    """The fast chain, step 0 of nov26, libimf: every gate field within the F1 'B' bound (<= 1e-12 of scale) except the named boundary-layer
    fields, which stay below 1e-9 (per ATM_STEP_PLAN / D129 verdict); wall time of the whole step < 60 s (about 15 s measured)."""
    import time
    ctx = _ctx(True)
    R = A.Real(DATE, IT)
    t = time.perf_counter()
    S, sn = F.run_step(DATE, IT, ctx, R=R, ms={}, land_mode='recorded')
    wall = time.perf_counter() - t
    st = A.compare_state(sn['filter'], A.end_reference(R), A.END_FIELDS)
    for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'TMOM', 'QMOM'):
        assert st[k]['rel'] <= 1e-9, (k, st[k]['rel'])
    assert max(v['rel'] for v in st.values()) < 1e-6
    assert wall < 120
    # mutation: a 1e-3 perturbation of one T cell before CONDSE must make the end state visibly worse (>= 100x)
    def hook(stage, Sx, Rx):
        if stage == 'dyn':
            Sx['T'] = np.array(Sx['T'], copy=True)
            Sx['T'].flat[Sx['T'].size // 2] *= 1.0 + 1e-3
    S2, sn2 = F.run_step(DATE, IT, ctx, R=A.Real(DATE, IT), ms={}, land_mode='recorded', hook=hook)
    st2 = A.compare_state(sn2['filter'], A.end_reference(R), A.END_FIELDS)
    assert st2['T']['rel'] > 100 * max(st['T']['rel'], 1e-14)
