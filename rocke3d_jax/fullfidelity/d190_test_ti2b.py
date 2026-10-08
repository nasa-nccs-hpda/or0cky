"""D190: double-double Ti2b (jax_advsi.ti2b_dd) vs the mpmath 113-bit REAL*16 emulation (advsi_ff.ti2b_quad) and vs float64 (seaice_core_ff.Ti2b): number of differing results."""
import sys, time
import d190_util as U
import numpy as np, jax, jax.numpy as jnp
import advsi_ff as A, seaice_core_ff as S
import jax_advsi as JA
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
rng = np.random.default_rng(11)
res = []
for scen in range(4):
    if scen == 0:      # generic
        eit = rng.uniform(-3.4e5, -2.0e5, n); si = rng.uniform(0, 40, n) * (rng.random(n) < 0.97); snowl = rng.uniform(0, 100, n) * (rng.random(n) < 0.5); mice = rng.uniform(1e-3, 400, n)
    elif scen == 1:    # cold, salty, thin
        eit = rng.uniform(-3.5e5, -3.3e5, n); si = rng.uniform(1, 10, n); snowl = rng.uniform(0, 2, n); mice = rng.uniform(1e-2, 5, n)
    elif scen == 2:    # near melting: Eit + LHM tiny / positive (cancellation in b + sqrt(det))
        eit = -3.34e5 + rng.uniform(-2e3, 2e3, n) * 10.0 ** rng.uniform(-8, 0, n); si = rng.uniform(1e-3, 8, n); snowl = rng.uniform(0, 10, n); mice = rng.uniform(1e-2, 100, n)
    else:              # tiny salinity
        eit = rng.uniform(-3.4e5, -3.0e5, n); si = 10.0 ** rng.uniform(-8, -1, n); snowl = rng.uniform(0, 10, n); mice = rng.uniform(1e-2, 100, n)
    f = jax.jit(JA.ti2b_dd)
    dd = np.asarray(f(jnp.asarray(eit), jnp.asarray(si), jnp.asarray(snowl), jnp.asarray(mice)))
    t0 = time.perf_counter()
    q = np.array([A.ti2b_quad(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(eit, si, snowl, mice)])
    f64 = np.array([S.Ti2b(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(eit[:20000], si[:20000], snowl[:20000], mice[:20000])])
    print('scenario', scen, 'n', n, 'dd != quad:', int((dd != q).sum()), '| float64 != quad (first 20000):', int((f64 != q[:20000]).sum()), 'mpmath %.1fs' % (time.perf_counter() - t0), flush=True)
