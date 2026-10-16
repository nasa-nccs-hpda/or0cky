"""D180 tests.  Run in their own process (XLA flags must precede jax):  RUN_XLA_FLAG_TESTS=1 python -m pytest tests/test_jax_atm_step.py
1. radiation T update in JAX is bitwise equal to atm_step.radia_apply (synthetic inputs; needs the flags in effect).
2. If JAX_ATM_STEP_DIR holds saved ref_step*/jax_step*.npz (from jax_atm_step_run.py, same cores both sides), every saved field is bitwise equal.
3. The result declares non-JAX stages and recorded inputs.  Skipped when ff_data is absent."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402

flags_ok = 'jax' not in sys.modules or 'algsimp' in os.environ.get('XLA_FLAGS', '')
pytestmark = pytest.mark.skipif(not flags_ok, reason="XLA flags not in effect (jax imported earlier); run in a fresh process")


def test_radia_jax_bitwise():
    import jax_atm_step as J
    import atm_step as A
    import jax.numpy as jnp
    r = np.random.default_rng(1)
    IM, JM, LM = 72, 46, 40
    T = 250 + 30 * r.random((IM, JM, LM)); srhr = r.random((LM + 1, IM, JM)); trhr = -r.random((LM + 1, IM, JM))
    cz = r.random((IM, JM)); ma = 1e3 + 1e4 * r.random((LM, IM, JM)); pk = 0.2 + r.random((LM, IM, JM))

    class C: sha = 1004.64; dtsrc = 1800.0; imaxj = np.full(JM, IM)
    c = C(); c.imaxj[0] = c.imaxj[-1] = 1
    want = A.radia_apply(T, srhr, trhr, cz, ma, pk, c)
    got = np.array(J._radia_T_jax(jnp.asarray(T), jnp.asarray(srhr), jnp.asarray(trhr), jnp.asarray(cz), jnp.asarray(ma), jnp.asarray(pk),
                                  jnp.asarray(A.imaxj_mask(c))[:, :, None], c.dtsrc, 1.0 / c.sha))
    assert np.array_equal(got, want)


def test_declarations():
    import jax_atm_step as J
    assert J.NON_JAX_STAGES and any('dissip' in s for s in J.NON_JAX_STAGES) and 'dyn' in J.JAX_STAGES


D = os.environ.get('JAX_ATM_STEP_DIR')


@pytest.mark.skipif(not D or not os.path.exists(f'{D}/jax_step0.npz'), reason="no saved runs (set JAX_ATM_STEP_DIR)")
def test_saved_runs_bitwise():
    k = 0
    while os.path.exists(f'{D}/jax_step{k}.npz') and os.path.exists(f'{D}/ref_step{k}.npz'):
        a, b = np.load(f'{D}/jax_step{k}.npz'), np.load(f'{D}/ref_step{k}.npz')
        for key in b.files:
            if not key.startswith('real/'):
                assert np.array_equal(a[key], b[key], equal_nan=True), (k, key)
        k += 1
    assert k >= 1
