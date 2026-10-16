"""D120: tests for the chained ocean step (ocean_step.py). Dump-based tests skip if the ffo_* dumps are missing.
Tolerances are about two orders above the errors measured in D119/D120 (see ledger), not loosened beyond that."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocean_chain_io as C  # noqa: E402

D = C.FF_DEFAULT + '/nov26'
HAVE = os.path.exists(D + '/ffo_state_33312.bin') and os.path.exists(D + '/ffo_opcoef.bin') and os.path.exists(D + '/ffo_geom.bin')
pytestmark = pytest.mark.skipif(not HAVE, reason='ffo_* ocean-chain dumps missing')


@pytest.fixture(scope='module')
def env():
    import ocean_step as O
    from ocean_step_compare import ctx_for
    import ocean_step_chain_compare as K
    ctx = ctx_for(D)
    sn = C.load_step(D, 33312)
    return O, K, ctx, sn


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_omega_is_earth365_sidereal():
    import ocean_odhorz
    assert ocean_odhorz.OMEGA == 2.0 * np.pi / (86400.0 * 365.0 / 366.0)


def test_stage_order_and_names(env):
    O = env[0]
    assert [n for n, _, _ in O.STAGES] == O.ORDER


def test_stages_replay_from_real_snapshots(env):
    O, K, ctx, sn = env
    from ocean_step_compare import replay
    tol = dict(precip=0.0, ground=1e-14, ostres=1e-7, oconv=1e-7, drag=1e-12, polar=1e-12, dynamics=1e-10,
               straits=1e-12, post=1e-12, odiff=0.0, meso=1e-12)
    rows = dict(replay(D, 33312, ctx))
    assert set(rows) == set(tol)
    for name, r in rows.items():
        for f, (_, rl) in r.items():
            assert rl <= tol[name], (name, f, rl)


@pytest.fixture(scope='module')
def chained(env):
    O, K, ctx, sn = env
    return K.one_step(D, 33312, ctx, sn=sn)


def test_full_step_exit_state(chained, env):
    out, rows, _, sn = chained
    e = K = env[1].errs(out, sn[14])
    assert max(e.values()) < 1e-9, e
    # the chained (not replayed) boundaries must not drift past the single-stage errors
    for name, r in rows:
        assert max(r.values()) < 1e-8, (name, r)


def test_chain_is_not_vacuous(chained, env):
    out, _, _, sn = chained
    # the step changes the state at ~1e-2..1 relative scale, far above the error bounds above
    assert rel(sn[0]['g0m'], sn[14]['g0m']) > 1e-6
    assert rel(sn[0]['uo'], sn[14]['uo']) > 1e-3


def _exit_err(env, stages):
    O, K, ctx, sn = env
    fx = K.fx_step(sn, 33312)
    try:
        out = O.ocean_step(sn[0], fx, ctx, stages=stages)
    except Exception:           # a reordered chain may fail outright: that also detects the mutation
        return float('inf')
    e = K.errs(out, sn[14])
    return max(v if v == v else float('inf') for v in e.values())


@pytest.mark.parametrize('drop', ['straits', 'oconv', 'ostres', 'dynamics', 'meso', 'ground'])
def test_dropping_a_stage_is_detected(env, drop):
    O = env[0]
    stages = [s for s in O.STAGES if s[0] != drop]
    assert _exit_err(env, stages) > 1e-6


def test_reordering_is_detected(env):
    O = env[0]
    d = {n: (n, f, t) for n, f, t in O.STAGES}
    order = list(O.ORDER)
    i, j = order.index('ostres'), order.index('oconv')
    order[i], order[j] = order[j], order[i]
    assert _exit_err(env, [d[n] for n in order]) > 1e-9
