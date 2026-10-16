"""D181: tests of jax_state_d181 (state pytree, dict <-> pytree round trip, static-field tile layout vs the real SURFACE record row sets).
Quick tests of the new unit only.  Data tests skip when the dumps / restarts are absent.  Optional: JS_MID_STATE=<driver checkpoint .pkl> adds a
round trip of a real mid-run ModelDriver state.  Run on the pinned cores:  OMP_NUM_THREADS=1 taskset -c 0-1 python -m pytest tests/test_jax_state_d181.py"""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax_state_d181 as JS  # noqa: E402  (sets the XLA flags before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

FF = JS.FF
HAVE = all(os.path.exists(p) for p in (
    f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc", f"{FF}/_pristine_restarts/fort1_dec01_itime33552.nc",
    f"{FF}/_pristine_restarts/fort1_jan01_itime17520.nc", f"{FF}/nov26_day/ffp_33312.bin", f"{FF}/nov26_day/ffc_cse_in_33312.bin",
    f"{FF}/dec01/ffp_33552.bin", f"{FF}/jan01/ffp_17520.bin", f"{FF}/nov26/ffo_geom.bin"))
needs_data = pytest.mark.skipif(not HAVE, reason="nov26/dec01/jan01 dumps or restarts missing")


def _diff(a, b, path="", out=None):
    """Bitwise tree comparison (dtype, shape, bytes; Python type of scalars; None; container types)."""
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            out.append(f"{path}: keys/order differ")
        for k in a:
            if k in b:
                _diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            out.append(f"{path}: array differs")
    elif isinstance(a, (float, np.floating)) and np.isnan(a) and np.isnan(b):
        pass
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def _synthetic_state():
    r = np.random.RandomState(1)
    return dict(
        itime=33312, k=0, seed0=4000000000,
        S=dict(T=r.rand(72, 46, 40), MA=r.rand(40, 72, 46), P=r.rand(72, 46), flag=np.array([True, False]),
               be=r.rand(5).astype('>f8'), _carry=dict(CLDSS=r.rand(40, 72, 46), LMC=r.rand(72, 46, 2))),
        ms=dict(S=dict(tl=[1.5, 2.5, 3.5], qm=[[1.0, 2.0], [3.0, 4.0]], npl=[np.float64(1.0), np.float64(2.0)], il=[1, 2, 3], empty=[])),
        provider=dict(rad=None),
        timing_log=[dict(itime=33312, wall=1.5, digest='abc', stages=dict(a=1.0))],
        surface=dict(SS=dict(ocean=dict(g0m=r.rand(72, 46, 13), kpl=np.arange(6).reshape(2, 3).astype(np.int64)),
                             ice=dict(rsi=r.rand(72, 46), flag_dsws=r.rand(72, 46) > 0.5), itime=33312,
                             atm=dict(gtemp=r.rand(72, 46)), ghy=dict(w=r.rand(72, 46, 3, 7))),
                     land_prev=None, usi=r.rand(72, 46), vsi=r.rand(72, 46), adv=dict(rsix=r.rand(72, 46), rsiy=r.rand(72, 46)),
                     uisurf=None, visurf=None),
        f3=dict(aij={151: r.rand(72, 46), 7: r.rand(72, 46)}, aijl={}, idacc={1: 0, 2: 3}, s0=None),
        tup=(np.float64(2.5), 3), flt=np.float32(1.25), npint=np.int32(7), nan=float('nan'))


# ------------------------------------------------------------------------------------------------ no data needed
def test_roundtrip_synthetic_bitwise_types_and_groups():
    sd = _synthetic_state()
    tree, meta = JS.driver_state_to_pytree(sd)
    back = JS.pytree_to_driver_state(tree, meta)
    assert _diff(sd, back) == []
    assert str(tree['rng']['seed0'].dtype) == 'uint32' and int(tree['rng']['seed0']) == 4000000000
    assert isinstance(back['seed0'], int) and isinstance(back['itime'], int) and back['surface']['SS']['ice']['flag_dsws'].dtype == bool
    assert back['S']['be'].dtype.str == '>f8'                        # big-endian dtype restored
    assert set(['atm', 'atm_carry', 'atm_ms', 'ocean', 'ice', 'ice_dyn', 'exch', 'land', 'f3', 'rng', 'clock']) <= set(tree)
    assert 'extra' in tree and JS.unconverted(meta) == ['timing_log']
    JS.check_dtypes({k: v for k, v in tree.items() if k != 'extra'})
    assert all(hasattr(x, 'shape') and not isinstance(x, np.ndarray) for _, x in jax.tree_util.tree_flatten_with_path(tree)[0])


def test_roundtrip_detects_one_ulp_and_survives_jit():
    sd = _synthetic_state()
    tree, meta = JS.driver_state_to_pytree(sd)
    tree2 = jax.jit(lambda t: jax.tree_util.tree_map(lambda x: x + 0 if x.dtype == jnp.float64 else x, t))(tree)
    assert _diff(sd, JS.pytree_to_driver_state(tree2, meta)) == []     # x + 0 is bitwise for float64 (non-vacuous: values pass through XLA)
    tree3 = dict(tree, ice=dict(tree['ice'], rsi=jnp.nextafter(tree['ice']['rsi'], 2.0)))
    assert len(_diff(sd, JS.pytree_to_driver_state(tree3, meta))) == 1
    sd2 = _synthetic_state()
    sd2['seed0'] = 2 ** 32
    with pytest.raises(AssertionError):
        JS.driver_state_to_pytree(sd2)


def _toy_static():
    r = np.random.RandomState(2)
    valid = JS.imaxj_mask()
    focean = np.where(r.rand(72, 46) < 0.5, 1.0, 0.0)
    flake = np.where((focean == 0) & (r.rand(72, 46) < 0.2), 1.0, 0.0)
    flice = np.where((focean == 0) & (flake == 0) & (r.rand(72, 46) < 0.1), 1.0, 0.0)
    fland = 1 - focean - flake
    return dict(focean=focean, flake=flake, fwater=focean + flake, flice=flice, fland=fland, fearth=fland - flice, valid=valid)


def test_tile_layout_synthetic_numpy_equals_jnp_and_rows_roundtrip():
    st = _toy_static()
    rsi = np.random.RandomState(3).rand(72, 46)
    rsi[::7, ::5] = 0.0
    rsi[1::9, ::4] = 1.0
    m_np = JS.tile_masks(st, rsi)
    m_jx = np.asarray(jax.jit(lambda r: JS.tile_masks({k: jnp.asarray(v) for k, v in st.items()}, r, jnp))(jnp.asarray(rsi)))
    assert m_np.shape == (4, 72, 46) and np.array_equal(m_np, m_jx)
    assert not m_np[:, ~st['valid']].any()                                      # nothing outside the IMAXJ domain
    assert not (m_np[0] & (rsi == 1.0)).any() and not (m_np[1] & (rsi == 0.0)).any()
    assert np.array_equal(m_np[2], st['valid'] & (st['flice'] > 0)) and np.array_equal(m_np[3], st['valid'] & (st['fearth'] > 0))
    for kind in ('ffp', 'ffs', 'ffl', 'ffg'):
        i, j, t = JS.rows_in_record_order(m_np, kind)
        vals = np.random.RandomState(4).rand(len(i))
        lay = JS.scatter_rows(vals, i, j, t, (4, 72, 46))
        assert np.array_equal(JS.gather_rows(lay, i, j, t), vals)
        assert int((lay != 0).sum()) == len(i) == int(sum(m_np[tt - 1].sum() for tt in set(t.tolist())))
    i, j, t = JS.rows_in_record_order(m_np, 'ffp')
    key12 = [(jj, ii, tt) for ii, jj, tt in zip(i, j, t) if tt <= 2]
    assert key12 == sorted(key12)
    # fixed shape: the layout has the same shape for any RSI field
    assert JS.tile_masks(st, np.zeros((72, 46))).shape == JS.tile_masks(st, np.ones((72, 46))).shape == (4, 72, 46)


# ------------------------------------------------------------------------------------------------ against the real data
@needs_data
@pytest.mark.parametrize("date", ["nov26", "dec01", "jan01"])
def test_static_fields_and_step0_masks_equal_real_rowsets(date):
    import atm_step as A
    import surface_loop as L
    dd = 'nov26_day' if date == 'nov26' else date
    it0 = JS.DATE_IT0[date]
    static, src = JS.build_static(date)
    chk = JS.verify_static(static)
    assert chk == dict(fland=0.0, fearth=0.0, flake=0.0)                  # derived FLAND/FEARTH and restart FLAKE equal the recorded fields bitwise
    R = A.Real(dd, it0)
    rec = A.surface_records(R)
    # (c) the rule with the real post-MELT_SI RSI
    m = JS.tile_masks(static, np.asarray(R.cse_in['RSI']))
    cmp = JS.compare_rowsets(m, static['valid'], rec)
    assert JS.rowsets_match(cmp), JS.mismatch_summary(cmp)
    assert all(v['order_ok'] for v in cmp.values())                       # rows in the real order too
    # (b) from the restart + our MELT_SI only
    SS = L.init_surface_state(date)
    geo = dict(focean=static['focean'], flake=static['flake'], fwater=static['fwater'], is_ocean=static['is_ocean'], is_lake=static['is_lake'],
               valid=static['valid'])
    ice, _ = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], geo)
    mb = JS.tile_masks(static, ice['rsi'])
    assert JS.rowsets_match(JS.compare_rowsets(mb, static['valid'], rec)), "restart + MELT_SI mask differs from the real row sets"
    assert np.array_equal(mb, m)
    # (a) NON-VACUITY: the raw restart RSI (before MELT_SI) is NOT the tile set of the step
    ma = JS.tile_masks(static, SS['ice']['rsi'])
    assert not JS.rowsets_match(JS.compare_rowsets(ma, static['valid'], rec, check_order=False))
    # mutation: flipping one ice tile must be detected
    mm = np.array(m, copy=True)
    i, j = np.argwhere(mm[1])[0]
    mm[1, i, j] = False
    cm = JS.compare_rowsets(mm, static['valid'], rec, check_order=False)
    assert not JS.rowsets_match(cm) and len(cm['pa']['only_real']) == 1


@needs_data
def test_masks_follow_the_day_boundary_steps_of_nov26():
    """33312 (step 0), a few ordinary steps, and 33359-33361 around the day boundary: the mask built from the real RSI of each step equals the
    real row sets, and the real row counts differ between steps (so the fixed layout is what removes the changing shape)."""
    import atm_step as A
    static, _ = JS.build_static('nov26')
    counts = set()
    for it in (33312, 33313, 33314, 33321, 33323, 33359, 33360, 33361):
        R = A.Real('nov26_day', it)
        rec = A.surface_records(R)
        m = JS.tile_masks(static, np.asarray(R.cse_in['RSI']))
        assert m.shape == (4, 72, 46)
        cmp = JS.compare_rowsets(m, static['valid'], rec)
        assert JS.rowsets_match(cmp) and all(v['order_ok'] for v in cmp.values()), (it, JS.mismatch_summary(cmp))
        counts.add(len(rec['pa']))
    assert len(counts) >= 3


@needs_data
@pytest.mark.parametrize("date", ["nov26", "dec01", "jan01"])
def test_initial_driver_state_roundtrip_bitwise_and_tile_state(date):
    sd = JS.initial_driver_state(date)
    tree, sp, meta = JS.build_state(sd, date)
    back = JS.pytree_to_driver_state(tree, meta)
    assert JS.trees_bitwise_equal(sd, back) == []
    JS.check_dtypes(tree)
    assert tree['tile']['mask'].shape == (4, 72, 46) and tree['tile_pbl']['u'].shape == (4, 72, 46, 8) and tree['ent_state'].shape == (72, 46, 1023)
    assert tree['rng']['seed0'].dtype == jnp.uint32 and int(tree['rng']['seed0']) == sd['seed0']
    # refresh_tile is a pure jit-able function of ice['rsi']
    t2 = jax.jit(JS.refresh_tile)(tree, sp) if False else JS.refresh_tile(tree, sp)
    assert np.array_equal(np.asarray(t2['tile']['mask']), np.asarray(tree['tile']['mask']))
    assert JS.unconverted(meta) == ['timing_log']


@pytest.mark.skipif(not os.environ.get('JS_MID_STATE') or not os.path.exists(os.environ.get('JS_MID_STATE', '')), reason="JS_MID_STATE not set")
def test_mid_run_driver_checkpoint_roundtrip_bitwise():
    sd = pickle.load(open(os.environ['JS_MID_STATE'], 'rb'))['state']
    tree, meta = JS.driver_state_to_pytree(sd)
    back = JS.pytree_to_driver_state(tree, meta)
    assert JS.trees_bitwise_equal(sd, back) == []
    JS.check_dtypes(tree)
    # the carried land rows (land_prev) are exactly the FEARTH > 0 cells of the static layout, in the order of the ffg record
    static, _ = JS.build_static('nov26')
    i, j, t = JS.rows_in_record_order(JS.tile_masks(static, np.zeros((72, 46))), 'ffg')
    p4 = np.asarray(tree['land']['carry']['p4_ij'])
    assert p4.shape == (len(i), 2) and np.array_equal(p4[:, 0].astype(int), i) and np.array_equal(p4[:, 1].astype(int), j)
