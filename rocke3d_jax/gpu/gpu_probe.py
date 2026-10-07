"""What does JAX see on this node, and how do its float64 math functions compare with NumPy's?

Run it inside the container the GPU job will use, e.g.:
  CMD="python -u gpu/gpu_probe.py" sbatch gpu/run_gpu_job.sbatch
Prints a short report; nothing is written. It answers: is JAX installed with CUDA, which devices, is float64 on,
how fast is a float64 matmul and a jit+scan loop on each device, and how big is the difference between the device's
exp / pow / sin / log (float64) and NumPy's (relevant to the bitwise-versus-libimf discussion: device math is not expected to be bitwise).
"""
import os
import platform
import sys
import time

import numpy as np


def main():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    print("python", sys.version.split()[0], platform.platform())
    print("jax", jax.__version__, "| jaxlib", getattr(__import__("jaxlib"), "__version__", "?"))
    print("backend:", jax.default_backend(), "| devices:", [f"{d.platform}:{d.device_kind}" for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "(unset)"), "| JAX_COMPILATION_CACHE_DIR:", os.environ.get("JAX_COMPILATION_CACHE_DIR", "(unset)"))
    print("x64 enabled:", jax.config.jax_enable_x64, "| float64 array dtype:", jnp.ones(2, dtype=jnp.float64).dtype)

    rng = np.random.default_rng(0)
    n = 2_000_000
    x = rng.uniform(-5.0, 5.0, n)
    xp = rng.uniform(0.1, 5.0, n)
    for d in jax.devices():
        with jax.default_device(d):
            xj, xpj = jnp.asarray(x), jnp.asarray(xp)
            out = {}
            for name, f, ref in (("exp", jnp.exp, np.exp(x)), ("sin", jnp.sin, np.sin(x)), ("log", jnp.log, np.log(xp)), ("pow(x,0.25)", lambda a: a ** 0.25, xp ** 0.25)):
                arg = xpj if name in ("log", "pow(x,0.25)") else xj
                got = np.asarray(jax.jit(f)(arg))
                with np.errstate(all="ignore"):
                    ulp = np.abs(got - ref) / np.maximum(np.spacing(np.abs(ref)), 1e-300)
                out[name] = (float(ulp.max()), float((got != ref).mean()))
            a = jnp.asarray(rng.standard_normal((2048, 2048)))
            f = jax.jit(lambda m: m @ m)
            f(a).block_until_ready()
            t = time.perf_counter()
            for _ in range(5):
                f(a).block_until_ready()
            dt = (time.perf_counter() - t) / 5

            def body(c, _):
                return c * 1.0000001 + 1e-9, None
            g = jax.jit(lambda c: jax.lax.scan(body, c, None, length=100000)[0])
            g(jnp.ones(1000)).block_until_ready()
            t = time.perf_counter()
            g(jnp.ones(1000)).block_until_ready()
            ds = time.perf_counter() - t
            print(f"[{d.platform}:{d.device_kind}] f64 matmul 2048^2: {dt*1e3:.1f} ms | jit+scan 1e5 steps: {ds*1e3:.1f} ms")
            for k, (mx, fr) in out.items():
                print(f"    {k:12s} vs NumPy: max {mx:.1f} ulp, {100*fr:.2f}% of values not bitwise equal")


if __name__ == "__main__":
    main()
