"""D191 quick unit tests of jax_phase2 (unit J3: DISSIP + FILTER as jnp, libimf host callback for the pow sites).  Real nov26 step-0 data; skipped when absent
or when the Intel libimf is not on the host.  Own process, at least TWO cores (host callbacks inside jit deadlock on one core, D185):
  taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 python -m pytest tests/test_jax_phase2.py -q      (from fullfidelity/; about 1-2 min with the compilation)
"""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clouds_jax_env  # noqa: E402,F401
import intel_libm_ff  # noqa: E402

DATE, IT0 = 'nov26', 33312
pytestmark = [pytest.mark.skipif(not intel_libm_ff.available(), reason='Intel libimf not on this host'),
              pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason='host callbacks inside jit need >= 2 cores'),
              pytest.mark.skipif(not os.path.exists('/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffa_step_33312_a.bin'),
                                 reason='real step-0 data absent')]


@pytest.fixture(scope='module')
def env():
    import d187_imf_p1 as W
    W.install()
    import jax
    import jax.numpy as jnp
    import atm_step as A
    import jax_phase2 as P2
    ctx = A.make_ctx(DATE, imf=True)
    R = A.Real(DATE, IT0)
    S = A._native(A.real_state_at(R, 'dissip', ctx))
    K, static = P2.make_consts(ctx)
    j3 = P2.make_unit(static)
    Sd = {k: jnp.asarray(S[k]) for k in P2.IN_KEYS}
    out, extra = j3(Sd, K)
    jax.block_until_ready(out)
    # NumPy reference stages (same input copy)
    Sn = copy.deepcopy(S)
    A.stage_dissip(Sn, ctx)
    td = np.array(Sn['T'])
    A.stage_filter(Sn, ctx)
    return dict(P2=P2, A=A, ctx=ctx, S=S, K=K, j3=j3, Sd=Sd, out=out, extra=extra, Sn=Sn, td=td, jnp=jnp)


def test_dissip_bitwise(env):
    assert np.array_equal(np.asarray(env['extra']['T_dissip']), env['td'])
    assert np.array_equal(np.asarray(env['out']['KEA_NEW']), env['Sn']['KEA_NEW'])


def test_filter_outputs_bitwise(env):
    bad = [k for k in env['P2'].OUT_KEYS if not np.array_equal(np.asarray(env['out'][k]), np.asarray(env['Sn'][k]))]
    assert not bad, bad


def test_exp_branch_not_selected(env):
    assert int(env['extra']['n_slp_exp_branch']) == 0


def test_mutation_detected(env):
    """a perturbed input must change the result (the comparison can fail)."""
    Sd = dict(env['Sd'])
    t = np.array(Sd['T'])
    t.flat[1234] = np.nextafter(t.flat[1234], np.inf)
    Sd['T'] = env['jnp'].asarray(t)
    out, _ = env['j3'](Sd, env['K'])
    assert not np.array_equal(np.asarray(out['T']), np.asarray(env['Sn']['T']))
