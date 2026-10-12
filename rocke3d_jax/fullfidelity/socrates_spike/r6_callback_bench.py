"""Cost of the host callbacks of the on-demand mode (DayStream, ZenithStream) alone and inside a jit, on the device of the process.
usage: python r6_callback_bench.py products.npz RUNDIR   (env R6_STATIC)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
import radia_jax as RJ
import radia_native as RN
import daily_products as DP
import drv_zenith as Z

prod = RJ.Products(np.load(sys.argv[1]), np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
t = time.time()
dp = DP.DailyProducts(sys.argv[2], prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust, solar_constant=prod.solar_constant,
                      hlb0=prod.c60["hlb0"][0])
print("DailyProducts init %.2f s" % (time.time() - t), jax.devices()[0], flush=True)
ds = DP.DayStream(dp, prod)
t = time.time(); d = ds(694); print("day_dict cold %.3f s" % (time.time() - t), flush=True)
t = time.time(); ds(694); print("day_dict warm %.5f s" % (time.time() - t), flush=True)
zs = RN.ZenithStream(Z.Zenith())
t = time.time()
for k in range(10):
    zs(16032 + 5 * k)
print("zenith per call %.4f s" % ((time.time() - t) / 10), flush=True)
proto = {k: jax.ShapeDtypeStruct(np.shape(v), np.asarray(v).dtype) for k, v in d.items()}
f = jax.jit(lambda day: jax.pure_callback(lambda x: {k: np.asarray(ds(int(x))[k], dtype=proto[k].dtype) for k in proto}, proto, day))
g = jax.jit(lambda it: jax.pure_callback(lambda x: zs(int(x)), jax.ShapeDtypeStruct((3, 72, 46), jnp.float64), it))
for name, fn, arg in (("day callback in jit", f, jnp.int32(694)), ("zenith callback in jit", g, jnp.int32(33312))):
    jax.block_until_ready(fn(arg))
    t = time.time()
    for _ in range(5):
        jax.block_until_ready(fn(arg))
    print("%s %.4f s per call" % (name, (time.time() - t) / 5), flush=True)
