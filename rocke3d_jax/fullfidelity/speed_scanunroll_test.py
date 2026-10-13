import os
os.environ['ROCKE_SCAN_UNROLL'] = '80'
import numpy as np
import scan_unroll as SU
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax
f = jax.jit(lambda x: lax.scan(lambda c, v: (c + v * 1.0000001, c), 0.0, x)[0])
x = jnp.asarray(np.random.default_rng(0).normal(size=40))
a = f(x)
SU.install()
g = jax.jit(lambda x: lax.scan(lambda c, v: (c + v * 1.0000001, c), 0.0, x)[0])
b = g(x)
h = jax.jit(lambda x: lax.fori_loop(0, 40, lambda i, c: c + x[i] * 1.0000001, 0.0))
print('scan equal', bool(a == b), ' fori equal', bool(h(x) == a), ' hlo has while (scan)', 'while' in g.lower(x).as_text(), '(fori)', 'while' in h.lower(x).as_text())
