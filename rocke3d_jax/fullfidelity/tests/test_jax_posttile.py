"""D190: quick tests of the post-tile / ocean device program (jax_seaice_lake, jax_riverf, jax_dynsi, jax_advsi, jax_ocean, jax_posttile).  Own unit only.
Each test is a bitwise comparison with the existing NumPy path on real data (nov26 step 0) or on synthetic stress inputs; the whole-program test (about 6 minutes:
NumPy path 1 min + XLA compile 4 min) runs only with D190_FULL=1.
Run pinned: taskset -c 3-5 env OMP_NUM_THREADS=1 pytest tests/test_jax_posttile.py   (host callbacks inside jit need >= 2 cores)"""
import glob
import os
import re
import sys

import numpy as np
import pytest


def eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: F401,E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data'
DATE, IT = 'nov26', 33312
have = os.path.exists(f'{FF}/{DATE}/ffc_cse_in_{IT}.bin') and os.path.exists(f'{FF}/advsi_dumps/{DATE}/ffadv_in_{IT}.bin')
needs_data = pytest.mark.skipif(not have, reason='real nov26 dumps not available')
NEW_FILES = ['jax_seaice_lake.py', 'jax_riverf.py', 'jax_dynsi.py', 'jax_advsi.py', 'jax_ocean.py', 'jax_posttile.py']


def test_no_integer_power_above_two_in_new_modules():
    """D188: NumPy x**n (n >= 3) is libm pow, jnp x**n multiplies; the new modules must use jnp.power(x, float(n)) for those."""
    bad = []
    for f in NEW_FILES:
        for k, line in enumerate(open(os.path.join(HERE, f)).read().splitlines(), 1):
            code = line.split('#')[0]
            for m in re.finditer(r'\*\*\s*([0-9]+(?:\.[0-9]+)?)', code):
                if float(m.group(1)) >= 3:
                    bad.append((f, k, line.strip()))
    assert not bad, bad


def test_jnp_cumsum_is_not_left_to_right_but_seqsum_is():
    import jax_dynsi as JD
    rng = np.random.default_rng(0)
    f = jax.jit(lambda x: jnp.cumsum(x)[-1])
    g = jax.jit(JD.seqsum)
    nd_cum = nd_seq = 0
    for _ in range(100):
        x = rng.normal(size=72) * 10 ** rng.uniform(-3, 3, 72)
        s = 0.0
        for v in x:
            s += v
        nd_cum += float(f(x)) != s
        nd_seq += float(g(x)) != s
    assert nd_seq == 0
    assert nd_cum > 0           # non-vacuity: the reason jax_dynsi does not use jnp.cumsum


def test_np_sum72_equals_numpy_pairwise_sum():
    import jax_dynsi as JD
    rng = np.random.default_rng(1)
    f = jax.jit(JD.np_sum72)
    for _ in range(100):
        x = rng.normal(size=72) * 10 ** rng.uniform(-3, 3, 72)
        assert float(f(x)) == float(np.sum(x))


def test_ti2b_double_double_equals_binary128_emulation_float64_does_not():
    import advsi_ff as A
    import seaice_core_ff as S
    import jax_advsi as JA
    rng = np.random.default_rng(2)
    n = 3000
    eit = rng.uniform(-3.5e5, -2.5e5, n)
    si = rng.uniform(0, 40, n) * (rng.random(n) < 0.97)
    snowl = rng.uniform(0, 50, n) * (rng.random(n) < 0.5)
    mice = rng.uniform(1e-3, 400, n)
    dd = np.asarray(jax.jit(JA.ti2b_dd)(jnp.asarray(eit), jnp.asarray(si), jnp.asarray(snowl), jnp.asarray(mice)))
    q = np.array([A.ti2b_quad(*map(float, v)) for v in zip(eit, si, snowl, mice)])
    f64 = np.array([S.Ti2b(*map(float, v)) for v in zip(eit, si, snowl, mice)])
    assert eq(dd, q)
    assert (f64 != q).sum() > 100       # non-vacuity: float64 is NOT the Fortran result


@needs_data
def test_advsi_bitwise_vs_numpy_and_real_dump():
    import advsi_ff as A
    import jax_advsi as JA
    d, o = A.read_dump(f'{FF}/advsi_dumps/{DATE}/ffadv_in_{IT}.bin', f'{FF}/advsi_dumps/{DATE}/ffadv_out_{IT}.bin')
    st = {k: d[k] for k in ('rsi', 'rsix', 'rsiy', 'rsisave', 'msi', 'snowi', 'hsi', 'ssi')}
    new_np, out_np = A.advsi(st, d['ausi'], d['avsi'], d['focean'], d['geo'])
    K = JA.make_static(d['focean'], d['geo'])
    new_j, out_j = jax.jit(lambda s, a, b: JA.advsi(K, s, a, b))({k: jnp.asarray(v) for k, v in st.items()}, jnp.asarray(d['ausi']), jnp.asarray(d['avsi']))
    for k, v in new_np.items():
        assert eq(np.asarray(new_j[k]), v), k
    for k, v in out_np.items():
        assert eq(np.asarray(out_j[k]), v), k
    oc = (d['focean'] > 0).copy()
    oc[1:, 0] = oc[1:, -1] = False
    for k in ('rsi', 'msi', 'snowi', 'rsix', 'rsiy'):
        assert eq(np.where(oc, np.asarray(new_j[k]), 0.0), np.where(oc, o[k], 0.0)), k
    assert np.abs(np.asarray(new_j['rsi']) - d['rsi']).max() > 1e-4        # non-vacuity: the ice really moves


@pytest.fixture(scope='module')
def world():
    import surface_loop as L
    st = L.load_statics(DATE)
    st['ctx'] = L.make_ocean_ctx(DATE)
    S = L.init_surface_state(DATE, st=st)
    return dict(st=st, S=S, L=L)


@needs_data
def test_dynsi_bitwise_incl_multiple_iterations(world):
    import dynsi_ff as DF
    import jax_dynsi as JD
    from odhorz_ff import geomo_dyn_arrays
    import netCDF4 as nc
    L, st, S = world['L'], world['st'], world['S']
    geo = st['geo']
    sinvo, sinpo = geomo_dyn_arrays()[:2]
    Kd = JD.make_static(geo['focean'], sinpo, sinvo)
    oo = geo['is_ocean']
    ice = S['ice']
    rsi, msi, snowi = (np.where(oo, ice[k], 0.0) for k in ('rsi', 'msi', 'snowi'))
    oc = S['ocean']
    oga = L.toc2sst(oc, st['ctx'])['ogeoza']
    us, vs = DF.uosurf_from_ocean(oc['uo'][:, :, 0], oc['vo'][:, :, 0], sinpo, sinvo)
    R = nc.Dataset(f"{L.FF}/_pristine_restarts/{L.RESTART[DATE]}")
    usi = np.array(R.variables['usi'][:], float).T.copy()
    vsi = np.array(R.variables['vsi'][:], float).T.copy()
    rng = np.random.default_rng(5)
    G = DF.Geom(geo['focean'])
    f = jax.jit(lambda *a: JD.dynsi(Kd, *a))
    kkis = []
    for amp in (1.0, 40.0):
        dmua = amp * rng.normal(size=rsi.shape) * 100.0
        dmva = amp * rng.normal(size=rsi.shape) * 100.0
        ref = DF.dynsi(G, dmua, dmva, rsi, msi, snowi, oga, us, vs, usi, vsi)
        got = f(*[jnp.asarray(a) for a in (dmua, dmva, rsi, msi, snowi, oga, oc['uo'][:, :, 0], oc['vo'][:, :, 0], usi, vsi)])
        for k in ('odmui', 'odmvi', 'ui2rho', 'ustar', 'usi', 'vsi', 'uisurf', 'visurf'):
            assert eq(np.asarray(got[k]), ref[k]), (amp, k)
        assert int(got['kki']) == ref['kki']
        kkis.append(ref['kki'])
    assert max(kkis) >= 3          # non-vacuity: the convergence loop iterated more than the minimum


@needs_data
def test_riverf_bitwise_incl_synthetic_emergency_and_kd9(world):
    import riverf_ff as RF
    import jax_riverf as JR
    st, S = world['st'], world['S']
    geo = st['geo']
    rs = RF.load_statics(st['axyp'][0])
    rng = np.random.default_rng(7)
    # synthetic tables: adjacent KDIREC = 9 lake pairs and lake cells with kdirec = 0 and a valid emergency outlet (neither exists on this grid)
    rs9 = dict(rs)
    kd9 = rs['kdirec'].copy()
    n9 = 0
    for (i, j) in np.argwhere(geo['flake'] > 0):
        if 2 < j < 43 and i + 1 < 72 and geo['flake'][i + 1, j] > 0:
            kd9[i, j] = kd9[i + 1, j] = 9
            n9 += 1
            if n9 >= 4:
                break
    rs9['kdirec'] = kd9
    rse = dict(rs)
    kde = rs['kdirec'].copy()
    pure = (geo['flake'] > .949 * (geo['flake'] + st['fearth'])) & (rs['kd911'] > 0) & (geo['flake'] > 0)
    for (i, j) in np.argwhere(pure)[:12]:
        kde[i, j] = 0
    rse['kdirec'] = kde
    seen = dict(kd9=0, emergency=0, backwash=0)
    for tag, rsx in (('plain', rs), ('kd9', rs9), ('emergency', rse)):
        R = JR.make_static(rsx, geo['flake'], st['fland'], st['fearth'])
        f = jax.jit(lambda lk, R=R: JR.riverf(R, lk, None, None, None))
        for trial in range(3):
            lk = {k: np.array(S['lake'][k]) for k in ('mwl', 'gml', 'tlake', 'mldlk')}
            sel = rng.random(lk['mwl'].shape) < 0.5
            fac = np.where(sel, rng.uniform(0.2, 60.0, lk['mwl'].shape), 1.0)
            if tag == 'emergency':
                fac = fac * np.where((rsx['kdirec'] == 0) & (rsx['kd911'] > 0) & (geo['flake'] > 0), 800.0, 1.0)
            lk['mwl'] = lk['mwl'] * fac
            lk['gml'] = lk['gml'] * fac
            new, flowo, eflowo, gtm, gtr, mlh, diag = RF.riverf(lk, geo['flake'], st['fland'], st['fearth'], rsx)
            seen['kd9'] += diag['n_kd9']
            seen['emergency'] += diag['n_emergency']
            seen['backwash'] += diag['n_backwash']
            got = f({k: jnp.asarray(v) for k, v in lk.items()})
            for k in ('mwl', 'gml', 'tlake', 'mldlk', 'dlake', 'glake'):
                assert eq(np.asarray(got[0][k]), new[k]), (tag, trial, k)
            for a, b in zip(got[1:], (flowo, eflowo, gtm, gtr, mlh)):
                assert eq(np.asarray(a), b), (tag, trial)
    assert seen['kd9'] > 0 and seen['emergency'] > 0 and seen['backwash'] > 0      # all three branches exercised


@needs_data
def test_pre_half_bitwise_vs_surface_pre(world):
    import jax_seaice_lake as SL
    import jax_posttile as PT
    L, st, S = world['L'], world['st'], world['S']
    inp = L.replay_inputs(DATE, IT)
    K = SL.make_static(st)
    ice_m, melt = L.melt_si(S['ice'], S['atm']['gtemp'], S['atm']['sss'], S['atm']['mlhc'], st['geo'])
    S1, mid = L.surface_pre(S, st, inp, melt_done=(ice_m, melt))
    dev = lambda t: jax.tree_util.tree_map(jnp.asarray, t)       # noqa: E731
    f = jax.jit(lambda S_, mi, me, i_: PT.surface_pre_dev(K, S_, mi, me, i_))
    S1j, midj = f(dev({k: S[k] for k in ('ocean', 'ice', 'lake', 'li', 'atm')}), dev(ice_m), dev(melt),
                  dev(dict(prec=inp['prec'], eprec=inp['eprec'], irrig_act=inp['irrig_act'])))
    for g in ('ocean', 'ice', 'lake', 'li', 'atm'):
        for k, v in S1[g].items():
            if isinstance(v, np.ndarray) and v.dtype.kind in 'fiub':
                assert eq(np.asarray(S1j[g][k]), v), (g, k)
    for k in ('fxp', 'pi', 'pli', 'dl', 'ag', 'irr'):
        for kk, v in mid[k].items():
            assert eq(np.asarray(midj[k][kk]), v), (k, kk)


@needs_data
def test_new_ice_tile_pbl_columns_follow_the_loadbl_rule():
    """PBL_DRV.f loadbl: a tile whose PBL did not run at the previous step is initialised from the same cell's donor tile (ice <- ocean): profile columns bitwise
    equal to the ocean row at the PBL entry of substep 1; dskin_in / khs_in / z0m_in are the previous processed tile's outputs.  Needs the 54-step day."""
    import pbl_compare as PC
    d = f'{FF}/nov26_day'
    files = sorted(glob.glob(f'{d}/ffp_[0-9]*.bin'), key=lambda p: int(os.path.basename(p)[4:-4]))
    if len(files) < 54:
        pytest.skip('54-step nov26 day not available')

    def rows1(p):
        a = PC.load(p)
        a = a[:len(a) // 2]
        return {(int(r[0]), int(r[1]), int(r[2])): r for r in a}, a
    prev, _ = rows1(files[0])
    n_new = 0
    for p in files[1:]:
        cur, arr = rows1(p)
        for (i, j, t), r in cur.items():
            if t == 2 and (i, j, 2) not in prev and (i, j, 1) in cur and (i, j, 1) in prev:
                o = cur[(i, j, 1)]
                cols = [33, 34, 35] + list(range(50, 89))
                assert eq(r[cols], o[cols])
                n_new += 1
        # leftover columns: input of row k = output of row k-1 of the same record
        assert (arr[1:, 28] == arr[:-1, 97]).all()
        assert (arr[1:, 30] == arr[:-1, 101]).mean() > 0.999
        prev = cur
    assert n_new >= 5


@needs_data
@pytest.mark.skipif(os.environ.get('D190_FULL') != '1', reason='set D190_FULL=1 (about 6 minutes)')
def test_whole_program_bitwise_vs_numpy_path(world):
    """surface_pre_dev + surface_post_dev (one jit) against surface_pre + surface_post_v2 on the nov26 step-0 entry, every output of the step."""
    import surface_loop_v2 as V2
    import jax_posttile as PT
    L, st, S = world['L'], world['st'], world['S']
    inp = V2.replay_inputs_v2(DATE, IT)
    V = V2.V2State(st, DATE, L.FF, IT)
    ice_m, melt = L.melt_si(S['ice'], S['atm']['gtemp'], S['atm']['sss'], S['atm']['mlhc'], st['geo'])
    V0 = dict(rsix=V.adv['rsix'].copy(), rsiy=V.adv['rsiy'].copy(), usi=V.dyn.usi.copy(), vsi=V.dyn.vsi.copy())
    S1, mid = L.surface_pre(S, st, inp, melt_done=(ice_m, melt))
    S2, post = V2.surface_post_v2(S1, st, dict(acc=inp['acc'], srfp=inp['srfp'], itime=IT), mid, V)
    K, Kb = PT.make_static_all(st, DATE, IT)
    dev = lambda t: jax.tree_util.tree_map(jnp.asarray, t)       # noqa: E731

    @jax.jit
    def run(S_, mi, me, i_, acc, srfp, itime, V_, Kb_):
        S1_, mid_ = PT.surface_pre_dev(K, S_, mi, me, i_)
        return PT.surface_post_dev(K, Kb_, S1_, mid_, acc, srfp, itime, V_)
    S2j, V2j, pj = run(dev({k: S[k] for k in ('ocean', 'ice', 'lake', 'li', 'atm')}), dev(ice_m), dev(melt),
                       dev(dict(prec=inp['prec'], eprec=inp['eprec'], irrig_act=inp['irrig_act'])), dev(inp['acc']), jnp.asarray(inp['srfp']), jnp.asarray(IT),
                       dev(V0), dev(Kb))
    n = 0
    for g in ('ocean', 'ice', 'lake', 'li', 'atm'):
        for k, v in S2[g].items():
            if isinstance(v, np.ndarray) and v.dtype.kind in 'fiub':
                assert eq(np.asarray(S2j[g][k]), v), (g, k)
                n += 1
    for k, v in (('rsix', V.adv['rsix']), ('rsiy', V.adv['rsiy']), ('usi', V.dyn.usi), ('vsi', V.dyn.vsi)):
        assert eq(np.asarray(V2j[k]), v), k
        n += 1
    assert n > 60
