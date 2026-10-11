"""D219: timing of ent_jax.ent_step (one GHY sub-iteration worth of Ent for all cells) on the default JAX backend.  python gpu_ent_bench.py [NREP]
Cells: all Ent cells of the nov26 restart (1146; the model has 753 land cells, so this is an upper bound for the batch).  Randomised forcings as tests/test_ent_jax.py."""
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
import ent_ff as E
import ent_jax as EJ
import test_ent_jax as T

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
print('backend', jax.default_backend(), jax.devices(), flush=True)
cells = E.load_restart_cells(T.RESTART)
keys = sorted(cells)
const, state = EJ.pack_cells(cells, keys)
n = len(keys)
rng = np.random.default_rng(5)
f = T._forcing(rng, n)
cj = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in const.items()}
sj = {k: jnp.asarray(v) for k, v in state.items()}
fj = {k: jnp.asarray(f[k]) for k in ('TairC', 'TcanopyC', 'Qf', 'P_mbar', 'Ca', 'Ch', 'U', 'IPARdif', 'IPARdir', 'CosZen', 'fwet', 'Soilmoist', 'fice')}
dts = jnp.asarray(rng.choice([60.0, 150.0, 450.0, 900.0], n))
upd = jnp.zeros(n, dtype=bool)
act = jnp.ones(n, dtype=bool)
fn = jax.jit(lambda s, ff, d, u, a: EJ.ent_step(cj, s, ff, d, u, a))
t0 = time.perf_counter()
st, out = fn(sj, fj, dts, upd, act)
jax.block_until_ready(out)
print(f'cells {n} cohorts {const["C"]} patches {const["P"]}  compile+first call {time.perf_counter() - t0:.1f} s  fail {bool(out["fail"])}', flush=True)
ts = []
for _ in range(NREP):
    t0 = time.perf_counter()
    st, out = fn(st, fj, dts, upd, act)
    jax.block_until_ready(out)
    ts.append(time.perf_counter() - t0)
print(f'steady per call: median {np.median(ts) * 1e3:.2f} ms  min {np.min(ts) * 1e3:.2f}  max {np.max(ts) * 1e3:.2f}', flush=True)
print('cnc sample', float(out['cnc'][0]), 'sum', float(jnp.sum(out['cnc'])), flush=True)
