"""D181 (stage S1) tests: the state pytree, the round-trip converters, the fixed-shape tile layout and the tile masks.  Skip when the data are absent.

Environment (optional): JAX_STATE_DIR = directory written by jax_state_capture.py (state_<k>.pkl, s1_<k>.pkl, k = 0..5 of the nov26 chained run) enables
the round trip on the REAL driver states of steps 0-5 and the tile-mask comparison at steps 1-5; JAX_STATE_S1_DIR = directory with s1_<k>.pkl for k = 0..53
enables the mask comparison over the whole nov26 day.  Without them only the step-0 tests of the three dates run (a few seconds)."""
import glob
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
cio = pytest.importorskip('clouds_condse_io')
J = pytest.importorskip('jax_state')
FF = cio.FF_DEFAULT
DATES = (('nov26', 'nov26', 33312), ('dec01', 'dec01', 33552), ('jan01', 'jan01', 17520))
HAVE = os.path.exists(f"{FF}/nov26/ffa_step_33312_a.bin") and os.path.exists(f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc")
needs_data = pytest.mark.skipif(not HAVE, reason="dumps / restart absent")
STATE_DIR = os.environ.get('JAX_STATE_DIR')
S1_DIR = os.environ.get('JAX_STATE_S1_DIR')


# ---------------------------------------------------------------------------------------------------------------- no data needed
def test_strict_diff_detects_everything():
    a = dict(x=np.arange(5.0), i=3, f=1.5, b=True, n=None, l=[1.0, 2.0], t=(1, 2), z=np.array([0.0]))
    assert J.strict_diff(a, dict(a)) == []
    for mut in (lambda d: d.update(x=np.nextafter(d['x'], 9)),           # one ulp
                lambda d: d.update(i=3.0),                               # int -> float
                lambda d: d.update(b=1),                                 # bool -> int
                lambda d: d.update(z=np.array([-0.0])),                  # signed zero
                lambda d: d.update(t=[1, 2]),                            # tuple -> list
                lambda d: d.update(n=0.0),
                lambda d: d.update(x=d['x'].astype(np.float32))):
        b = dict(a)
        mut(b)
        assert J.strict_diff(a, b) != [], mut


def test_generic_roundtrip_scalars_lists_none_tuples():
    a = dict(i=7, f=-0.0, b=False, n=None, np_i=np.int32(4), np_f=np.float64(2.5), ll=[0.1, 0.2, 0.3], l2=[[1.0, 2.0], [3.0, 4.0]], li=[1, 2],
             mixed=[1.0, 2, None], t=(np.ones(2), 3), e={}, el=[], z=np.zeros((2, 3), bool), k=np.arange(4, dtype=np.int64))
    dev, sch = J.to_device(a)
    import jax
    assert all(isinstance(l, jax.Array) for l in jax.tree_util.tree_leaves(dev))
    back = J.to_host(dev, sch)
    assert J.strict_diff(a, back) == []


def test_big_endian_arrays_are_converted_value_identically():
    a = dict(x=np.arange(4.0).astype('>f8'))
    dev, sch = J.to_device(a)
    back = J.to_host(dev, sch)
    assert back['x'].dtype == np.float64 and np.array_equal(back['x'], np.arange(4.0)) and sch.byteswapped == ['/x']


def test_host_only_objects_are_rejected():
    class X:
        pass
    with pytest.raises(TypeError):
        J.to_device(dict(a=X()))


def test_tile_masks_rule_on_synthetic_cells():
    shape = (J.IM, J.JM)
    foc, fl, fli, fe = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    rsi = np.zeros(shape)
    valid = np.ones(shape, bool)
    foc[0, 5], foc[1, 5], foc[2, 5], foc[3, 5] = 1.0, 1.0, 1.0, 0.5       # ocean cells
    rsi[1, 5], rsi[2, 5], rsi[3, 5] = 0.3, 1.0, 1e-300                    # partly iced, fully iced, tiny ice
    fl[4, 5] = 0.2                                                        # lake cell
    fli[5, 5] = 1.0                                                       # land ice
    fe[6, 5] = 1.0                                                        # land
    m = J.tile_masks(rsi, foc, fl, fli, fe, valid)
    assert m.shape == (4, J.IM, J.JM) and m.dtype == bool
    assert m[0, 0, 5] and not m[1, 0, 5]                                  # open ocean only
    assert m[0, 1, 5] and m[1, 1, 5]                                      # both
    assert not m[0, 2, 5] and m[1, 2, 5]                                  # fully iced: no ocean tile
    assert m[0, 3, 5] and m[1, 3, 5]                                      # tiny ice still has a tile
    assert m[0, 4, 5] and not m[1, 4, 5]                                  # lake without ice
    assert m[2, 5, 5] and m[3, 6, 5] and m.sum() == 9
    assert np.array_equal(m[:, :, 5].sum(axis=0).nonzero()[0], np.arange(7))
    import jax.numpy as jnp
    mj = J.tile_masks(jnp.asarray(rsi), jnp.asarray(foc), jnp.asarray(fl), jnp.asarray(fli), jnp.asarray(fe), jnp.asarray(valid))
    assert np.array_equal(np.asarray(mj), m)                              # the same function works on device arrays (jit-able)


def test_rows_grid_roundtrip_and_record_order():
    rng = np.random.default_rng(1)
    mask = rng.random((J.IM, J.JM)) < 0.3
    ij = J.row_order_ij(mask)
    assert len(ij) == mask.sum() and list(ij[:, 1]) == sorted(ij[:, 1])
    rows = rng.random((len(ij), 5))
    g = J.rows_to_grid(rows, ij)
    assert np.array_equal(J.grid_to_rows(g, ij), rows) and g[~mask].sum() == 0


def test_jit_on_the_pytree_runs_and_matches_numpy():
    import jax
    import jax.numpy as jnp
    sd = dict(itime=1, k=0, seed0=5, S=dict(T=np.random.default_rng(0).random((J.IM, J.JM, J.LM))), ms={}, provider=dict(rad=None), timing_log=[])
    js = J.driver_to_pytree(sd)

    @jax.jit
    def f(tree):
        return jnp.sum(tree['atm']['T'] * 2.0) + tree['clock']['itime']
    assert float(f(js.tree)) == float(np.sum(sd['S']['T'] * 2.0) + 1) or abs(float(f(js.tree)) - (np.sum(sd['S']['T'] * 2.0) + 1)) < 1e-9


def test_land_prev_rows_to_grid_identity():
    rng = np.random.default_rng(2)
    mask = rng.random((J.IM, J.JM)) < 0.25
    ij = J.row_order_ij(mask)
    n = len(ij)
    lp = dict(p4_ij=(ij + 1).astype(np.int64), patch=dict(a=rng.random(n), b=rng.random((n, 3))), ghy=dict(nsn=rng.random((n, 2)), s=np.float64(3.0)),
              rho=rng.random(n), scalar=2)
    g, rl = J.land_prev_to_grid(lp)
    back = J.grid_to_land_prev(g, rl)
    assert J.strict_diff(lp, back) == []
    assert g['mask'].sum() == n and g['tree']['patch']['b'].shape == (J.IM, J.JM, 3)


# ---------------------------------------------------------------------------------------------------------------- records <-> fixed layout
def _have_step(d, it):
    return all(os.path.exists(f"{FF}/{d}/{n}_{it}.bin") for n in ('ffp', 'ffs', 'ffl', 'ffg', 'fft'))


def test_record_order_is_j_then_i_and_layout_roundtrip_is_bitwise():
    """On every available step of the three dates: the ten record arrays -> fixed layout -> record arrays are bitwise identical (same rows, same order),
    so the row sets (ffp 4,5xx rows, ffs 3,4xx rows, ...) carry no information beyond the (type, IM, JM) masks."""
    import atm_step as A
    nrec = 0
    for d, it0, n in (('nov26_day', 33312, 54), ('dec01', 33552, 6), ('jan01', 17520, 6)):
        for it in range(it0, it0 + n):
            if not _have_step(d, it):
                continue
            rec = {k: np.asarray(v) for k, v in A.surface_records(A.Real(d, it, FF)).items()}
            lay = J.records_to_layout(rec)
            assert J.strict_diff(rec, J.layout_to_records(lay)) == [], (d, it)
            nrec += 1
    if nrec == 0:
        pytest.skip('no complete record step')
    assert nrec >= 1


def test_layout_mask_changes_with_rowset_size_but_shapes_do_not():
    """D174: ten distinct row counts over the 54 steps; the layout has ONE shape."""
    import atm_step as A
    sizes, shapes = set(), set()
    for it in range(33312, 33312 + 54):
        if not _have_step('nov26_day', it):
            pytest.skip('day records absent')
        rec = A.surface_records(A.Real('nov26_day', it, FF))
        sizes.add((len(rec['pa']), len(rec['ta'])))
        lay = J.records_to_layout({k: np.asarray(v) for k, v in rec.items()})
        shapes.add(tuple(lay[f].shape for f in ('pbl', 'tile', 'landice', 'land', 'blk')))
    assert len(sizes) >= 5 and len(shapes) == 1


def test_layout_roundtrip_mutation_is_detected():
    import atm_step as A
    if not _have_step('nov26_day', 33312):
        pytest.skip('records absent')
    rec = {k: np.asarray(v) for k, v in A.surface_records(A.Real('nov26_day', 33312, FF)).items()}
    lay = J.records_to_layout(rec)
    lay['pbl'][1, 1, lay['masks']['pbl'][1][1].nonzero()[0][0], lay['masks']['pbl'][1][1].nonzero()[1][0], 20] += 1e-9
    assert J.strict_diff(rec, J.layout_to_records(lay)) != []


# ---------------------------------------------------------------------------------------------------------------- tile masks = real rowsets
def _record_masks(d, it):
    import atm_step as A
    rec = {k: np.asarray(v) for k, v in A.surface_records(A.Real(d, it, FF)).items()}
    lay = J.records_to_layout(rec)
    M = lay['masks']
    return np.stack([M['tile'][0][0], M['tile'][0][1], M['landice'][0], M['land'][0]]), M, lay


@needs_data
@pytest.mark.parametrize('date,daydir,it0', DATES)
def test_masks_equal_record_rowsets_step0(date, daydir, it0):
    """MELT_SI + PRECIP_SI on the restart state, rule tile_masks, vs the rows of ffs/ffl/ffg of step 0: 0 differing cells per type, three dates."""
    import atm_step as A
    import surface_loop as L
    import jax_static as S
    st = L.load_statics(date, FF)
    geo = st['geo']
    SS = L.init_surface_state(date, FF, st=dict(st, ctx=L.make_ocean_ctx(date, FF)))
    ice, _ = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], geo)
    co = A.Real(daydir, it0, FF).cse_out
    ice2, _ = L.precip_si(ice, np.asarray(co['PREC']), np.asarray(co['EPREC']), geo)
    m = J.tile_masks(ice2['rsi'], geo['focean'], geo['flake'], st['flice'], st['fearth'], geo['valid'])
    recm, M, _ = _record_masks(daydir, it0)
    assert (m != recm).sum(axis=(1, 2)).tolist() == [0, 0, 0, 0], (date, m.sum(axis=(1, 2)), recm.sum(axis=(1, 2)))
    # the masks of the two substeps and of all three tile record families agree with each other
    assert np.array_equal(M['tile'][0], M['tile'][1]) and np.array_equal(M['pbl'][0][:2], M['tile'][0]) and np.array_equal(M['pbl'][0][2], M['landice'][0])
    # the RSI of the state BETWEEN steps (before MELT_SI) would NOT give the real tile set -- the rule needs the surface-call RSI
    m0 = J.tile_masks(SS['ice']['rsi'], geo['focean'], geo['flake'], st['flice'], st['fearth'], geo['valid'])
    if date == 'nov26':
        assert (m0 != recm).sum() > 0
    # mutation: one more ice cell appears in the rule -> detected
    rsi = ice2['rsi'].copy()
    i, j = np.argwhere(geo['is_ocean'] & (rsi == 0) & geo['valid'])[0]
    rsi[i, j] = 0.5
    mm = J.tile_masks(rsi, geo['focean'], geo['flake'], st['flice'], st['fearth'], geo['valid'])
    assert (mm != recm).sum() >= 1


@pytest.mark.skipif(not STATE_DIR, reason="JAX_STATE_DIR not set (run jax_state_capture.py)")
def test_masks_equal_record_rowsets_nov26_steps_0_to_5():
    import surface_loop as L
    st = L.load_statics('nov26', FF)
    geo = st['geo']
    for k in range(6):
        s1 = pickle.load(open(f'{STATE_DIR}/s1_{k}.pkl', 'rb'))
        m = J.tile_masks(s1['ice']['rsi'], geo['focean'], geo['flake'], st['flice'], st['fearth'], geo['valid'])
        recm, _, _ = _record_masks('nov26_day', 33312 + k)
        assert (m != recm).sum(axis=(1, 2)).tolist() == [0, 0, 0, 0], k


@pytest.mark.skipif(not S1_DIR, reason="JAX_STATE_S1_DIR not set")
def test_masks_equal_record_rowsets_whole_day():
    import surface_loop as L
    st = L.load_statics('nov26', FF)
    geo = st['geo']
    files = sorted(glob.glob(f'{S1_DIR}/s1_*.pkl'), key=lambda p: int(os.path.basename(p)[3:-4]))
    assert len(files) >= 6
    bad = {}
    for p in files:
        k = int(os.path.basename(p)[3:-4])
        s1 = pickle.load(open(p, 'rb'))
        rsi = s1['rsi'] if 'rsi' in s1 else s1['ice']['rsi']
        m = J.tile_masks(rsi, geo['focean'], geo['flake'], st['flice'], st['fearth'], geo['valid'])
        recm, _, _ = _record_masks('nov26_day', 33312 + k)
        n = (m != recm).sum(axis=(1, 2)).tolist()
        if any(n):
            bad[k] = n
    # reported, not hidden: after the day boundary (k >= 48) FLAKE/FEARTH change (daily_LAKE is not ported); anything before must be 0
    assert not [k for k in bad if k < 48], bad
    print('mask mismatches by step (type counts):', bad)


# ---------------------------------------------------------------------------------------------------------------- round trip on the dict state
@needs_data
@pytest.mark.parametrize('date,daydir,it0', DATES)
def test_roundtrip_identity_step0_three_dates(date, daydir, it0):
    sd = J.initial_driver_state(date, daydir, it0, FF)
    js = J.driver_to_pytree(sd)
    import jax
    assert all(isinstance(l, jax.Array) for l in jax.tree_util.tree_leaves(js.tree))
    back = J.pytree_to_driver(js)
    assert J.strict_diff(sd, back) == []
    # mutation: one ulp in one leaf of the pytree is seen
    import jax.numpy as jnp
    js.tree['ocean']['g0m'] = js.tree['ocean']['g0m'].at[3, 3, 3].set(jnp.nextafter(js.tree['ocean']['g0m'][3, 3, 3], 1e9))
    assert J.strict_diff(sd, J.pytree_to_driver(js)) != []


@needs_data
def test_initial_state_equals_driver_state_if_saved():
    if not STATE_DIR or not os.path.exists(f'{STATE_DIR}/state_0.pkl'):
        pytest.skip('no saved driver state')
    ref = pickle.load(open(f'{STATE_DIR}/state_0.pkl', 'rb'))
    assert J.strict_diff(ref, J.initial_driver_state('nov26', 'nov26_day', 33312, FF)) == []


@pytest.mark.skipif(not STATE_DIR, reason="JAX_STATE_DIR not set (run jax_state_capture.py)")
@pytest.mark.parametrize('k', range(6))
def test_roundtrip_identity_nov26_driver_states_steps_0_to_5(k):
    sd = pickle.load(open(f'{STATE_DIR}/state_{k}.pkl', 'rb'))
    js = J.driver_to_pytree(sd)
    back = J.pytree_to_driver(js)
    assert J.strict_diff(sd, back) == []
    if k >= 1:                                   # land_prev (row arrays over the 753 land cells) exists from step 1: moved to the fixed layout
        assert sd['surface']['land_prev'] is not None and js.tree['land_carry']['mask'].sum() == 753
        assert js.schema.rowleaves
    rep = J.residency_report(js)
    assert rep['total_arrays'] > 100 and rep['total_bytes'] > 50e6


@needs_data
def test_pbl_carry_roundtrip_and_fixed_shape():
    import drv_state_cols as D
    for date, _, _ in DATES:
        C = D.PBLCarry.from_restart(date)
        g = J.pbl_carry_to_grid(C.st)
        assert g['u'].shape == (4, J.IM, J.JM, 8) and g['e'].shape == (4, J.IM, J.JM, 7) and g['ipbl'].shape == (4, J.IM, J.JM)
        back = J.grid_to_pbl_carry(g)
        assert J.strict_diff(C.st, back) == [], date
        sd = J.initial_driver_state(date, date, dict(nov26=33312, dec01=33552, jan01=17520)[date], FF)
        js = J.driver_to_pytree(sd, pbl_carry=C)
        assert J.strict_diff(C.st, J.pytree_to_pbl_carry(js).st) == []


@needs_data
def test_tile_group_in_pytree_from_statics():
    import jax_static as S
    b = S.build_static('nov26', FF, with_ent=False)
    sd = J.initial_driver_state('nov26', 'nov26_day', 33312, FF)
    js = J.driver_to_pytree(sd, statics=S.tile_statics(b['fields']))
    t = js.tree['tiles']
    assert t['mask'].shape == (4, J.IM, J.JM) and t['frac'].shape == (4, J.IM, J.JM)
    # fractions of the water tiles sum to FWATER, land tiles to FLICE / FEARTH
    fr = np.asarray(t['frac'])
    fw = b['fields']['focean'] + b['fields']['flake']
    assert np.allclose((fr[0] + fr[1])[b['cand']['ocean']], fw[b['cand']['ocean']], rtol=0, atol=1e-15)


def test_residency_report_vocabulary():
    sd = dict(itime=1, k=0, seed0=5, S=dict(T=np.zeros((J.IM, J.JM, J.LM))), ms={}, provider=dict(rad=None), timing_log=[{'a': 1}])
    js = J.driver_to_pytree(sd)
    rep = J.residency_report(js)
    assert 'atm' in rep['device'] and rep['device']['atm'][0] == 1
    assert any('timing_log' in h for h in rep['host'])
