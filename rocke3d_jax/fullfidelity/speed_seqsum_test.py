import os
os.environ['ROCKE_SEQSUM_BLK'] = '16'
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax
import seqsum_fast as F
rng = np.random.default_rng(1)
def ref(a, axis=0):
    a = jnp.moveaxis(a, axis, 0)
    return lax.scan(lambda c, x: (c + x, None), a[0], a[1:])[0]
ok = True
for shape, axis in (((3404,), 0), ((72,), 0), ((72, 46), 0), ((46, 72, 3), 1), ((40,), 0), ((17,), 0), ((33,), 0)):
    a = jnp.asarray(rng.normal(size=shape) * 10.0 ** rng.integers(-8, 8, size=shape))
    r, f = jax.jit(lambda x: ref(x, axis))(a), jax.jit(lambda x: F._fast(x, axis, 16))(a)
    e = bool(jnp.array_equal(r, f)); ok &= e
    print(shape, axis, 'bitwise equal', e)
print('ALL EQUAL' if ok else 'MISMATCH')
