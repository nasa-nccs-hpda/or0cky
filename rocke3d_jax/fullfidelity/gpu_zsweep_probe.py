"""The OADVT2 z sweep (oadvt_jax._z_sweep) alone on this device, from the captured ocean state (D214): standalone steady time and the compiled-program structure.

usage: python -u gpu_zsweep_probe.py IN.pkl       (IN.pkl from gpu_ocean_dyn_bench.py capture)
"""
import pickle
import sys
import time

_A = list(sys.argv)
sys.argv = [_A[0], 'nov26', '/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/_import_scratch']
import gpu_step_run as G   # noqa: E402  (sets the backend flags; parses argv, hence the replacement above)
sys.argv = _A
import numpy as np   # noqa: E402
import jax   # noqa: E402
import jax.numpy as jnp   # noqa: E402
import jax_ocean as JO   # noqa: E402
import oadvt_jax as OJ   # noqa: E402

d = pickle.load(open(sys.argv[1], 'rb'))
oc = jax.tree_util.tree_map(jnp.asarray, d['oc'])
cp = G.C.Coupled('nov26')
Ko = cp.K['oc']
rm, rx, ry, rz = (JO._to1(oc[k]) for k in ('g0m', 'gxmo', 'gymo', 'gzmo'))
mo = JO._to1(oc['mmi'])
mw = JO._to1(oc['smw'])
lmm = jnp.asarray(Ko['lmm1'])
dt = JO.DTOLF
print('backend', jax.default_backend(), 'shapes', rm.shape, mo.shape, mw.shape, lmm.shape, flush=True)
f = jax.jit(lambda rm, rx, ry, rz, mo, mw: OJ._z_sweep(rm, rx, ry, rz, mo, mw, dt, 0.0, lmm))
ts = []
for r in range(6):
    t0 = time.perf_counter()
    out = f(rm, rx, ry, rz, mo, mw)
    jax.block_until_ready(out)
    ts.append(time.perf_counter() - t0)
print(f'z sweep alone: first (compile) {ts[0]:.2f} s   steady {np.mean(ts[1:]) * 1000:.1f} ms  ({", ".join(f"{x * 1000:.1f}" for x in ts[1:])})', flush=True)
# a single layer-like elementwise chain for scale: 100 dependent elementwise ops on one (IM+1, JM+1) plane
x = jnp.ones((73, 47))
g = jax.jit(lambda x: [x := x * 1.0001 + 0.5 for _ in range(100)][-1])
for r in range(3):
    t0 = time.perf_counter()
    jax.block_until_ready(g(x))
    t1 = time.perf_counter() - t0
print(f'100 chained elementwise ops on one plane: {t1 * 1000:.2f} ms (fused: one kernel expected)', flush=True)
