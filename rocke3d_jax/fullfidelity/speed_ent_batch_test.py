"""ent_step outputs and state, ROCKE_ENT_BATCH 0 vs 1, bitwise (CPU)."""
import os, sys, subprocess, pickle
import numpy as np
if len(sys.argv) > 1 and sys.argv[1] == 'run':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import ent_ff as E, ent_jax as EJ, test_ent_jax as T
    cells = E.load_restart_cells(T.RESTART)
    keys = sorted(cells)
    const, state = EJ.pack_cells(cells, keys)
    n = len(keys)
    f = T._forcing(np.random.default_rng(5), n)
    cj = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in const.items()}
    sj = {k: jnp.asarray(v) for k, v in state.items()}
    fj = {k: jnp.asarray(f[k]) for k in ('TairC', 'TcanopyC', 'Qf', 'P_mbar', 'Ca', 'Ch', 'U', 'IPARdif', 'IPARdir', 'CosZen', 'fwet', 'Soilmoist', 'fice')}
    dts = jnp.asarray(np.random.default_rng(5).choice([60.0, 150.0, 450.0, 900.0], n))
    st, out = jax.jit(lambda s, ff, d, u, a: EJ.ent_step(cj, s, ff, d, u, a))(sj, fj, dts, jnp.zeros(n, bool), jnp.ones(n, bool))
    pickle.dump({k: np.asarray(v) for k, v in {**out, **{'s_' + k: v for k, v in st.items()}}.items()}, open(sys.argv[2], 'wb'))
else:
    me = os.path.abspath(__file__)
    res = {}
    for v in ('0', '1'):
        p = f'/tmp/ent_batch_{v}.pkl'
        subprocess.run([sys.executable, me, 'run', p], env=dict(os.environ, ROCKE_ENT_BATCH=v), check=True, capture_output=True)
        res[v] = pickle.load(open(p, 'rb'))
    bad = [k for k in res['0'] if not np.array_equal(res['0'][k], res['1'][k], equal_nan=True)]
    print('keys', len(res['0']), 'mismatching:', bad or 'none')
