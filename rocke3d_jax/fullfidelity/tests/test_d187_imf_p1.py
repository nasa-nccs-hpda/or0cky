"""D187 quick unit tests of d187_imf_p1 (the libimf host callback wired into the D186 device glue).  Real nov26 step-0 data under ff_data; skipped when absent
or when the Intel libimf is not on the host.  Run in its OWN process with at least TWO cores (host callbacks inside jit deadlock on one core, D185):
  taskset -c 0-2 env OMP_NUM_THREADS=1 python -m pytest tests/test_d187_imf_p1.py -q      (from fullfidelity/; about 1-2 min incl. a few compilations)
It does not run a model step."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clouds_jax_env  # noqa: E402,F401
import intel_libm_ff  # noqa: E402

DATE = 'nov26'
pytestmark = [pytest.mark.skipif(not intel_libm_ff.available(), reason='Intel libimf not on this host'),
              pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason='host callbacks inside jit need >= 2 cores'),
              pytest.mark.skipif(not os.path.exists('/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffa_step_33312_a.bin'),
                                 reason='real step-0 data absent')]


@pytest.fixture(scope='module')
def env():
    import d187_imf_p1 as W
    W.install()
    import jax_p1_glue as G
    import atm_step as A
    ctx = A.make_ctx(DATE, imf=True)
    R = A.Real(DATE, 33312)
    S = A._native(A.init_state(R))
    return dict(W=W, G=G, A=A, ctx=ctx, S=S, K=G.make_consts(ctx))


def eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_matopmb_and_pek_equal_numpy_libimf(env):
    import jax.numpy as jnp
    import dyn_filter_ff as ffl
    G, ctx, S, K = env['G'], env['ctx'], env['S'], env['K']
    ma = np.asarray(S['MA'])
    r = ffl.matopmb(ma, ctx.dyn.g, imf_pow=True)
    o = G.matopmb(jnp.asarray(ma), K)
    assert all(eq(o[k], r[k]) for k in r)
    # non-vacuity: the libm (numpy) result differs from libimf in PK, so the equality above can fail
    r0 = ffl.matopmb(ma, ctx.dyn.g, imf_pow=False)
    assert not eq(r0['pk'], r['pk'])
    pek = G.pek_of(jnp.asarray(S['PEDN']), K)
    assert eq(pek, env['A']._pow(ctx, S['PEDN'], ctx.kapa))


def test_calc_trop_equals_numpy_libimf(env):
    import jax.numpy as jnp
    import dyn_glue_ff as gf
    G, ctx, S, K = env['G'], env['ctx'], env['S'], env['K']
    pt, lt = G.calc_trop(*(jnp.asarray(S[k]) for k in ('T', 'PK', 'PMID')), K)
    pn, ln, _, _ = gf.calc_trop(S['T'], S['PK'], S['PMID'], ctx.gg, imf_pow=True)
    assert eq(pt, pn) and eq(lt, ln)


def test_shims_route_to_imf(env):
    import clouds_lscond_size_ff as sz
    import jax_p1_condse as CS
    import libimf_ops as L
    CS.cf.set_backend('numpy')               # what PoleHost.run does; must keep the imf backend
    assert sz._IMF['on'] is True
    assert L.mode() == 'libimf'
    # the shim forwards attributes unchanged
    import clouds_condse_ff as cf
    assert CS.cf.GRAV == cf.GRAV and CS.lj.make_K is not None


def test_callbacks_are_counted(env):
    import jax.numpy as jnp
    import libimf_ops as L
    G, K, S = env['G'], env['K'], env['S']
    L.reset_counters()
    G.pek_of(jnp.asarray(S['PEDN']), K).block_until_ready()
    c = L.counters()
    assert c['pow']['calls'] >= 1 and c['pow']['elements'] == S['PEDN'].size
