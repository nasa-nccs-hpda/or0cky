"""Standalone steady time of the OADVT2 x sweep on this device from the captured step-0 inputs (D214)."""
import sys
import time

import clouds_jax_env  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
import oadvt_jax as OJ   # noqa: E402

z = np.load(sys.argv[1])
names = ('rm', 'rx', 'ry', 'rz', 'mm', 'mudt', 'nc', 'lane_pass', 'lmu_l', 'lmm_l')
a = [jnp.asarray(z[n]) for n in names]
rl = float(z['rxlimit'])
print('backend', jax.default_backend(), flush=True)
for tag, lim in (('G0M (qlimit off)', 0.0), ('S0M (qlimit on)', 1.0)):
    ts = []
    for r in range(6):
        t0 = time.perf_counter()
        out = OJ._x_sweep(*a, lim)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    print(f'x sweep {tag}: first {ts[0]:.2f} s   steady {np.mean(ts[1:]) * 1000:.1f} ms  ({", ".join(f"{x * 1000:.1f}" for x in ts[1:])})  nc max {int(z["nc"].max())}', flush=True)
