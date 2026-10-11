"""What does one jax.pure_callback cost on this device, and what does the cost depend on? (D214)

usage: python -u gpu_callback_probe.py
Times, inside a jit, one host callback in several shapes: (a) one scalar in/out; (b) the real OADVT2 pre-pass shapes (2 inputs 14x47x73 f64; outputs 14x47x73 f64,
14x47 i64, 14x47 bool) with a body that only returns zeros; (c) the same with the real NumPy function oadvt_vec._x_prepass on real-like data; (d) the same as (b)
with jax.experimental.io_callback (ordered False).  Each variant: first call (compile), then 5 steady calls; seconds per call, and with 4 callbacks chained in one jit.
"""
import time

import clouds_jax_env  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback

jax.config.update('jax_enable_x64', True)
LMO, JM, IM = 13, 46, 72
shp3 = (LMO + 1, JM + 1, IM + 1)
print('backend', jax.default_backend(), jax.devices(), flush=True)
rng = np.random.default_rng(0)
shp_in = (IM + 1, JM + 1, LMO + 1)       # the pre-pass INPUTS are indexed (i, j, l); its outputs (l, j, i)
mm = jnp.asarray(rng.uniform(1e13, 1e14, shp_in))
mu = jnp.asarray(rng.normal(size=shp_in) * 1e12)


def timeit(name, f, *args, n=5):
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    first = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    print(f'{name:62s} first {first:7.2f} s   steady {np.mean(ts) * 1000:9.2f} ms per jit call', flush=True)


# (a) scalar
spec_s = jax.ShapeDtypeStruct((), jnp.float64)
fa = jax.jit(lambda x: jax.pure_callback(lambda v: np.float64(v) + 1.0, spec_s, x))
timeit('(a) pure_callback, scalar in / scalar out, no-op body', fa, jnp.float64(1.0))
fa4 = jax.jit(lambda x: jax.pure_callback(lambda v: np.float64(v) + 1.0, spec_s, jax.pure_callback(lambda v: np.float64(v) + 1.0, spec_s, jax.pure_callback(lambda v: np.float64(v) + 1.0, spec_s, jax.pure_callback(lambda v: np.float64(v) + 1.0, spec_s, x)))))
timeit('(a4) four chained scalar callbacks', fa4, jnp.float64(1.0))
# (b) real shapes, no-op body
spec = (jax.ShapeDtypeStruct(shp3, jnp.float64), jax.ShapeDtypeStruct(shp3[:2], jnp.int64), jax.ShapeDtypeStruct(shp3[:2], jnp.bool_))


def noop(a, b):
    return np.zeros(shp3), np.ones(shp3[:2], np.int64), np.ones(shp3[:2], bool)


fb = jax.jit(lambda a, b: jax.pure_callback(noop, spec, a, b))
timeit('(b) pure_callback, real pre-pass shapes, no-op body', fb, mm, mu)
# (c) real function
import oadvt_vec as OV
lmu1 = np.full((IM + 1, JM + 1), LMO)
lmm1 = np.full((IM + 1, JM + 1), LMO)
t0 = time.perf_counter()
OV._x_prepass(np.asarray(mm), np.asarray(mu), 100.0, lmu1, lmm1)
print(f'    (host function alone on this data: {time.perf_counter() - t0:.3f} s)', flush=True)
fc = jax.jit(lambda a, b: jax.pure_callback(lambda x, y: tuple(np.asarray(z) for z in OV._x_prepass(np.asarray(x), np.asarray(y), 100.0, lmu1, lmm1)), spec, a, b))
timeit('(c) pure_callback, real pre-pass shapes, real NumPy body', fc, mm, mu)
# (d) io_callback unordered, no-op body
fd = jax.jit(lambda a, b: io_callback(noop, spec, a, b, ordered=False))
timeit('(d) io_callback (unordered), real shapes, no-op body', fd, mm, mu)
# (e) host-side work only: device_get of both inputs, no callback
fe = lambda a, b: (np.asarray(a), np.asarray(b))
t0 = time.perf_counter()
for _ in range(5):
    fe(mm, mu)
print(f'(e) np.asarray of the two real-size inputs (device_get), no jit: {(time.perf_counter() - t0) / 5 * 1000:.2f} ms per pair', flush=True)
