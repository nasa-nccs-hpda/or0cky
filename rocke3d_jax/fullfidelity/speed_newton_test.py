"""pbl_ff._newton / griddr: plain while loop against the K-unrolled masked loop (ROCKE_NEWTON_UNROLL), bitwise, CPU."""
import importlib, os, sys
import numpy as np
res = {}
for k in ('1', '4'):
    os.environ['ROCKE_NEWTON_UNROLL'] = k
    if 'pbl_ff' in sys.modules:
        del sys.modules['pbl_ff']
    import pbl_ff as P
    import jax, jax.numpy as jnp
    rng = np.random.default_rng(3)
    z1 = jnp.asarray(rng.uniform(0.5, 3.0, 64)); zn = jnp.asarray(rng.uniform(50.0, 900.0, 64))
    f = jax.jit(jax.vmap(P.griddr))
    out = f(z1, zn)
    res[k] = [np.asarray(o) for o in out]
ok = all(np.array_equal(a, b) for a, b in zip(res['1'], res['4']))
print('griddr outputs bitwise equal (K=1 vs K=4):', ok, [a.shape for a in res['1']])
