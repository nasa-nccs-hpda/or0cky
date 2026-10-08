"""D190: stage-by-stage check of jax_ocean against the NumPy reference trace (input of stage k = the recorded output of stage k-1 of the same run)."""
import sys, time
import d190_util as U
import numpy as np, jax, jax.numpy as jnp
import surface_loop as L
import jax_seaice_lake as SL
import jax_ocean as JO

date, refp = sys.argv[1], sys.argv[2]
only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
ref = dict(np.load(refp))
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
K = SL.make_static(st)
Ko, Kb = JO.make_static(st, K)
K['oc'] = Ko
fx = U.to_dev(U.tree(ref, 'post/fx/'))
order = ['ground', 'ostres', 'oconv', 'drag', 'polar', 'dynamics', 'straits', 'post', 'odiff', 'meso']
prev = U.to_dev(U.tree(ref, 'S1/ocean/'))
for name in order:
    trace = U.tree(ref, f'ocean_trace/{name}/')
    fn = getattr(JO, 'stage_' + name, None)
    if fn is None or (only and name not in only):
        prev = U.to_dev(trace); continue
    f = jax.jit(lambda s, fx, Kb, fn=fn: fn(K, Kb, s, fx))
    Kbd = U.to_dev(Kb)
    t0 = time.perf_counter(); out = f(prev, fx, Kbd); U.sync(out); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); out = f(prev, fx, Kbd); U.sync(out); t2 = time.perf_counter() - t0
    res = U.compare(f'{name} ', U.to_np(out), trace, out={}, skip=())
    print(f'   {name}: first {t1:.1f}s steady {t2:.3f}s', flush=True)
    prev = U.to_dev(trace)       # next stage starts from the recorded state
