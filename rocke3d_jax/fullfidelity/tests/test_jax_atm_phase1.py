"""D186 quick unit tests of the phase-1 device modules (glue, MELT_SI, pole slices, execution counter, replay server).  They read the real nov26 step-0
data under ff_data and skip when absent.  The XLA flags must be set before jax: run this file in its OWN process:
  taskset -c 0-1 env OMP_NUM_THREADS=1 python -m pytest tests/test_jax_atm_phase1.py -q   (from fullfidelity/; ~1 min, no model step, no radiation server)
The whole-step comparison against the NumPy libm reference is jax_atm_phase1_run.py (minutes), not a unit test."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clouds_jax_env_fast  # noqa: E402,F401
import jax_p1_count as CNT  # noqa: E402
CNT.install()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import atm_step as A  # noqa: E402

DATE = 'nov26'
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(A.FF, DATE, 'ffa_step_33312_a.bin')), reason='real step-0 data absent')


@pytest.fixture(scope='module')
def env():
    import jax_p1_glue as G
    ctx = A.make_ctx(DATE, imf=False)
    R = A.Real(DATE, 33312)
    S = A._native(A.init_state(R))
    return dict(G=G, ctx=ctx, R=R, S=S, K=G.make_consts(ctx))


def eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_matopmb_se_ke_efix(env):
    import dyn_filter_ff as ffl, dyn_glue_ff as gf
    G, ctx, S, K = env['G'], env['ctx'], env['S'], env['K']
    ma = np.asarray(S['MA'])
    r = ffl.matopmb(ma, ctx.dyn.g)
    o = G.matopmb(jnp.asarray(ma), K)
    assert all(eq(o[k], r[k]) for k in r)
    se = gf.conserv_se(ma, S['MASUM'], S['PK'], S['T'], S['Q'], S['QCI'], ctx.gg)
    assert eq(G.conserv_se(*(jnp.asarray(x) for x in (ma, S['MASUM'], S['PK'], S['T'], S['Q'], S['QCI'])), K), se)
    ke = ffl.conserv_ke(ma, S['U'], S['V'], ctx.gg)[0]
    assert eq(G.conserv_ke(jnp.asarray(ma), jnp.asarray(S['U']), jnp.asarray(S['V']), K), ke)
    e = gf.energy_fix(se, ke, se * 1.0001, ke * 0.9999, S['MASUM'], S['T'], S['PK'], ctx.gg)
    t, dse, mmg = G.energy_fix(*(jnp.asarray(x) for x in (se, ke, se * 1.0001, ke * 0.9999, S['MASUM'], S['T'], S['PK'])), K)
    assert eq(t, e['t']) and eq(dse, e['dsepke']) and eq(mmg, e['mmglob'])


def test_trop_pgrad(env):
    import dyn_glue_ff as gf
    G, ctx, S, K = env['G'], env['ctx'], env['S'], env['K']
    pt, lt = G.calc_trop(*(jnp.asarray(S[k]) for k in ('T', 'PK', 'PMID')), K)
    pn, ln, _, _ = gf.calc_trop(S['T'], S['PK'], S['PMID'], ctx.gg)
    assert eq(pt, pn) and eq(lt, ln)
    phi = np.asarray(S['GZ'])
    o = gf.pgrad_pbl(S['T'][:, :, 0], S['PK'][0], S['PMID'][0], S['PEDN'][0], phi[:, :, 0], ctx.gg['zatmo'], ctx.gg)
    j = G.pgrad_pbl(*(jnp.asarray(x) for x in (S['T'][:, :, 0], S['PK'][0], S['PMID'][0], S['PEDN'][0], phi[:, :, 0])), K)
    assert all(eq(a, b) for a, b in zip(j, o))


def test_uv_replicate_avg_recalc(env):
    import clouds_condse_ff as cf, dyn_glue_ff as gf
    G, ctx, S = env['G'], env['ctx'], env['S']
    U, V = np.asarray(S['U']), np.asarray(S['V'])
    rn = cf.replicate_uv_to_agrid(U, V)
    rj = G.replicate_uv_to_agrid(jnp.asarray(U), jnp.asarray(V))
    assert all(eq(a, b) for a, b in zip(rj, rn))
    rng = np.random.default_rng(1)
    d = [rng.standard_normal(rn[0].shape) * 1e-3 for _ in range(2)] + [rng.standard_normal(rn[2].shape) * 1e-3 for _ in range(4)]
    un, vn = U.copy(), V.copy()
    a = [x.copy() for x in d]
    cf.avg_replicated_duv_to_vgrid(un, vn, *a)
    uj, vj, ukm, vkm = G.avg_replicated_duv_to_vgrid(jnp.asarray(U), jnp.asarray(V), *(jnp.asarray(x) for x in d))
    assert eq(uj, un) and eq(vj, vn) and eq(ukm, a[0]) and eq(vkm, a[1])      # pole-row copies the NumPy version writes in place
    C = G.make_agrid_consts(ctx.cfg['glue_geom'])
    ua, va = gf.recalc_agrid_uv(U, V, ctx.cfg['glue_geom'])
    uj, vj = G.recalc_agrid_uv(jnp.asarray(U), jnp.asarray(V), C)
    assert eq(uj, ua) and eq(vj, va)


def test_mutation_is_detected(env):
    """A perturbed constant must change the result (the comparison above cannot be vacuous)."""
    G, S, K = env['G'], env['S'], dict(env['K'])
    ma = jnp.asarray(S['MA'])
    base = G.matopmb(ma, K)
    K['kg2mb'] = K['kg2mb'] * (1 + 2.3e-16)
    mut = G.matopmb(ma, K)
    assert not eq(base['pedn'], mut['pedn'])


def test_melt_si_matches_numpy_and_recorded_rsi(env):
    import surface_loop as L
    import jax_p1_melt as M
    SS = L.init_surface_state(DATE)
    geo = L.load_statics(DATE)['geo']
    ice_n, melt_n = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], geo)
    ice_o, melt_o = M.melt_si({k: jnp.asarray(v) for k, v in SS['ice'].items()}, jnp.asarray(SS['atm']['gtemp']), jnp.asarray(SS['atm']['sss']),
                              jnp.asarray(SS['atm']['mlhc']), M.geo_device(geo), 1800.0)
    assert all(eq(ice_o[k], ice_n[k]) for k in ice_n)
    assert all(eq(melt_o[k], melt_n[k]) for k in ('melti', 'emelti', 'smelti', 'dop'))
    assert eq(ice_o['rsi'], env['R'].cse_in['RSI'])


def test_pole_slice_roundtrip():
    import jax_p1_condse as CS
    rng = np.random.default_rng(0)
    for shp in [(72, 46), (72, 46, 40), (40, 72, 46), (41, 72, 46), (3, 72, 46), (9, 72, 46, 40), (3, 40, 72, 46), (72, 40)]:
        x = rng.standard_normal(shp)
        for j in (0, 45):
            sl = CS.pole_slice(x, j)
            y = np.zeros(shp)
            CS._pole_put(y, np.asarray(sl), j)
            assert np.array_equal(CS.pole_slice(y, j), sl)
            assert np.array_equal(np.asarray(CS.pole_slice(jnp.asarray(x), j)), sl)


def test_execution_counter():
    @jax.jit
    def f(x):
        return x * 2
    x = jnp.ones(3)
    with CNT.stage('t') as r:
        y = f(x)
        z = f(y)
        jax.block_until_ready(z)
    assert r['jit_calls'] == 2
    @jax.jit
    def g(x):
        return f(x) + 1                      # nested call under trace is not an execution
    with CNT.stage('t2') as r2:
        jax.block_until_ready(g(x))
    assert r2['jit_calls'] == 1


def test_replay_server_echo():
    import jax_atm_phase1 as P1
    import radiation_server as rs
    r = dict(SRHR=np.ones((41, 72, 46)), TRHR=np.full((41, 72, 46), 2.0), COSZ1=np.full((72, 46), .5))
    srv = P1.ReplayServer(r)
    st = {k: np.full(shp, 3.0) for k, shp in rs.INPUT_FIELDS.items()}
    out = srv.request(st, 33312, 0)
    assert out['SRHR'].shape == (41, 72, 46) and out['COSZ1'].max() == .5 and out['T'].max() == 3.0 and out['AIJ'].shape[2] >= 391
    assert out['FSF'].max() == 0.0                      # surface-facing outputs are zeros in replay (declared)
