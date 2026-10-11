"""D219: ent_jax (batched JAX Ent export path) against ent_ff (scalar reference, D169) on the real nov26 restart cells with randomised forcings.
Reference with libimf OFF (math.exp / pow); the JAX side uses the XLA exp/pow: agreement is judged at rounding level (rtol 1e-9), the number of failures is reported."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import atm_step as A  # noqa: E402
import ent_ff as E  # noqa: E402
import ent_jax as EJ  # noqa: E402

RESTART = f'{A.FF}/_pristine_restarts/fort1_nov26_itime33312.nc'
pytestmark = pytest.mark.skipif(not os.path.exists(RESTART), reason='nov26 restart not available')


def _forcing(rng, n):
    f = dict(TairC=rng.uniform(-20, 35, n), TcanopyC=rng.uniform(-20, 35, n), Qf=rng.uniform(0, 0.02, n), P_mbar=rng.uniform(900, 1020, n),
             Ch=rng.uniform(5e-4, 5e-3, n), U=rng.uniform(0.5, 12, n))
    vis = rng.uniform(0, 450, n)
    vis[rng.random(n) < 0.25] = 0.0
    dvis = vis * rng.uniform(0, 1, n)
    f.update(IPARdif=vis - dvis, IPARdir=dvis, CosZen=np.where(vis > 0, rng.uniform(0.02, 1, n), 0.0), fwet=rng.uniform(0, 0.4, n))
    f['Ca_ppm'] = rng.uniform(270, 420, n)
    f['Ca'] = f['Ca_ppm'] * 1e-6 * f['P_mbar'] * E.F32(100.0) / E.GASC / (f['TcanopyC'] + E.TFRZ)
    wsat = rng.uniform(0.05, 0.4, (n, 6))
    f['w_veg'] = wsat * rng.uniform(0.0, 1.0, (n, 6))
    f['ws_veg'] = wsat
    f['fice'] = np.where(rng.random((n, 6)) < 0.2, rng.uniform(0, 1, (n, 6)), 0.0)
    f['Soilmoist'] = np.where(f['ws_veg'] > 0, f['w_veg'] / np.where(f['ws_veg'] > 0, f['ws_veg'], 1.0), 0.0)
    return f


def test_ent_step_matches_reference():
    E.set_use_imf(False)
    ref_cells = E.load_restart_cells(RESTART)
    keys = sorted(ref_cells)[:120]
    const, state = EJ.pack_cells(ref_cells, keys)
    rng = np.random.default_rng(11)
    n = len(keys)
    f = _forcing(rng, n)
    dts = rng.choice([60.0, 150.0, 450.0, 900.0], n)
    upd = rng.random(n) < 0.3
    # reference
    exp = []
    for i, k in enumerate(keys):
        c = ref_cells[k]
        E.set_forcings(c, f['TairC'][i] + E.TFRZ, f['TcanopyC'][i], f['Qf'][i], f['P_mbar'][i], f['Ca_ppm'][i], f['Ch'][i], f['U'][i],
                       f['IPARdif'][i] + f['IPARdir'][i], f['IPARdir'][i], f['CosZen'][i], f['fwet'][i], f['w_veg'][i], f['ws_veg'][i], f['fice'][i])
        E.ent_run(c, dts[i], bool(upd[i]))
        exp.append(E.get_exports(c))
    cj = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in const.items()}
    sj = {k: jnp.asarray(v) for k, v in state.items()}
    fj = {k: jnp.asarray(f[k]) for k in ('TairC', 'TcanopyC', 'Qf', 'P_mbar', 'Ca', 'Ch', 'U', 'IPARdif', 'IPARdir', 'CosZen', 'fwet', 'Soilmoist', 'fice')}
    st, out = jax.jit(lambda s, ff, d, u, a: EJ.ent_step(cj, s, ff, d, u, a))(sj, fj, jnp.asarray(dts), jnp.asarray(upd), jnp.ones(n, dtype=bool))
    assert not bool(out['fail'])
    bad = []
    for name in ('cnc', 'trans_sw', 'ci', 'gpp', 'lai', 'ipp'):
        a = np.asarray([e[name] for e in exp])
        b = np.asarray(out[name])
        if not np.allclose(a, b, rtol=1e-9, atol=1e-18):
            bad.append((name, float(np.max(np.abs(a - b) / (np.abs(a) + 1e-30)))))
    a = np.asarray([e['betadl'] for e in exp])
    if not np.allclose(a, np.asarray(out['betadl']), rtol=1e-9, atol=1e-15):
        bad.append(('betadl', float(np.max(np.abs(a - np.asarray(out['betadl']))))))
    for name, getter in (('airtemp_10d', lambda c: c.airtemp_10d), ('par_10d', lambda c: c.par_10d), ('gdd', lambda c: c.gdd), ('ncd', lambda c: c.ncd)):
        a = np.asarray([getter(ref_cells[k]) for k in keys])
        if not np.allclose(a, np.asarray(st[name]), rtol=1e-12, atol=1e-12):
            bad.append((name, float(np.max(np.abs(a - np.asarray(st[name]))))))
    sacc = np.asarray([c.Sacclim for k in keys for p in ref_cells[k].patches for c in p.cohorts])
    if not np.allclose(sacc, np.asarray(st['Sacclim']), rtol=1e-12, atol=1e-12):
        bad.append(('Sacclim', float(np.max(np.abs(sacc - np.asarray(st['Sacclim']))))))
    assert not bad, bad
