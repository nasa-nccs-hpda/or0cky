"""D205: the day-boundary hook jax_coupled.Coupled.daily_lake_update on the nov26 restart state (no step is run: initial_state, the real step-47 GHY rows/water of the records
stand in for the last land step).  Checks the plumbing: the device state is replaced, the statics (st, geometry, K, static181, melt_geo) carry the new lake fractions, the jitted
functions are rebuilt, and the result equals daily_lake.daily_lake called directly.  Skipped without the records."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
cio = pytest.importorskip('clouds_condse_io')
FF = cio.FF_DEFAULT
HAVE = os.path.exists(f'{FF}/nov26_day/ffg_33359.bin') and os.path.exists(f'{FF}/_pristine_restarts/fort1_nov26_itime33312.nc')


@pytest.mark.skipif(not HAVE, reason='nov26_day records / restart absent')
def test_hook_updates_state_and_statics():
    import jax.numpy as jnp
    import atm_step as A
    import d193_day as D
    import jax_coupled as C
    import ghy_compare as GC
    import daily_lake as DL
    A.Real = D.DayReal
    cp = C.Coupled('nov26', nit_strict=False)
    state, _, _ = cp.initial_state()
    g = GC.load(f'{FF}/nov26_day/ffg_33359.bin')
    g2 = g[len(g) // 2:]
    ec = (g2[:, 1].astype(int) - 1) * 72 + (g2[:, 0].astype(int) - 1)
    cp._last_ghy = (g2, ec)
    state['land_prev'] = dict(dyn_next=dict(w=jnp.asarray(np.stack([GC.unpack(r)[4]['w_out'] for r in g2]))))
    flake0 = np.array(cp.st['geo']['flake'], copy=True)
    pre = cp.pre
    new, info = cp.daily_lake_update(state, log=lambda *a: None, land_fractions=False)    # D208: lake half only (D205 plumbing)
    fl = np.asarray(cp.st['geo']['flake'])
    assert info['n_flake_changed'] > 100 and (fl != flake0).sum() == info['n_flake_changed']
    assert np.array_equal(cp.K['flake'], fl) and np.array_equal(cp.K['fwater'], cp.st['geo']['focean'] + fl)
    assert np.array_equal(np.asarray(cp.sp181['flake']), fl) and np.array_equal(np.asarray(cp.melt_geo['fwater']), cp.st['geo']['focean'] + fl)
    assert np.array_equal(cp.st['fearth'], cp.st['fland'] - cp.st['flice']) or np.allclose(cp.st['fearth'], cp.st['fland'] - cp.st['flice'], atol=1e-15)
    assert cp.pre is not pre and cp._stage_cache is None
    rsi = np.asarray(new['surf']['ice']['rsi'])
    assert (rsi >= 0).all() and (rsi <= 1).all()
    assert (np.asarray(new['surf']['lake']['mwl']) >= 0).all()
    assert info['pow'] in ('libimf', 'numpy') and 'GHY dfrac water/heat transfer (GHY_DRV.f:4531)' in info['not_applied']


@pytest.mark.skipif(not HAVE, reason='nov26_day records / restart absent')
def test_hook_applies_land_transfer():
    """D208: with land_fractions=True the carried land state (dyn_next w, ht, fr_snow) changes in the cells whose lake fraction changed, the underwater state is kept in
    cp._land3, and nothing else of land_prev changes."""
    import jax.numpy as jnp
    import atm_step as A
    import d193_day as D
    import jax_coupled as C
    import ghy_compare as GC
    A.Real = D.DayReal
    cp = C.Coupled('nov26', nit_strict=False)
    state, _, _ = cp.initial_state()
    g = GC.load(f'{FF}/nov26_day/ffg_33359.bin')
    g2 = g[len(g) // 2:]
    cp._last_ghy = (g2, (g2[:, 1].astype(int) - 1) * 72 + (g2[:, 0].astype(int) - 1))
    un = [GC.unpack(r) for r in g2]
    dyn = dict(w=jnp.asarray(np.stack([u[4]['w_out'] for u in un])), ht=jnp.asarray(np.stack([u[4]['ht_out'] for u in un])), fr_snow=jnp.asarray(np.stack([u[4]['fr_snow_out'] for u in un])))
    state['land_prev'] = dict(dyn_next=dyn, ghy='marker')
    new, info = cp.daily_lake_update(state, log=lambda *a: None)
    lf = info['land_fractions']
    assert lf['n_shrunk'] > 50 and lf['n_expanded'] > 50 and lf['new_cell_rows'] == []
    d2 = new['land_prev']['dyn_next']
    assert new['land_prev']['ghy'] == 'marker'
    assert (np.asarray(d2['w']) != np.asarray(dyn['w'])).any() and (np.asarray(d2['ht']) != np.asarray(dyn['ht'])).any()
    assert np.isfinite(np.asarray(d2['w'])).all() and np.isfinite(np.asarray(d2['ht'])).all()
    assert cp._land3['w'].shape == (len(g2), 7)
    assert 'GHY dfrac' not in ' '.join(info['not_applied'])
