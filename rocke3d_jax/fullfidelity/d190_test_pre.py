"""D190 quick check of the pre-tile half (PRECIP_*, TOC2SST, IRRIG_LK, seaice_to_atmgrid) of jax_posttile against the NumPy reference npz."""
import sys, time
import d190_util as U
import numpy as np, jax
import surface_loop as L
import jax_seaice_lake as SL
import jax_posttile as PT

date, refp = sys.argv[1], sys.argv[2]
ref = dict(np.load(refp))
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
K = SL.make_static(st)
S0 = U.to_dev(U.tree(ref, 'S0/'))
mi = U.to_dev(U.tree(ref, 'melt/ice/')); me = U.to_dev(U.tree(ref, 'melt/melt/'))
inp = U.to_dev(U.tree(ref, 'inp/'))
f = jax.jit(lambda S, mi, me, inp: PT.surface_pre_dev(K, S, mi, me, inp))
t0 = time.perf_counter(); S1, mid = f(S0, mi, me, inp); U.sync(S1); print('first', time.perf_counter()-t0)
t0 = time.perf_counter(); S1, mid = f(S0, mi, me, inp); U.sync(S1); print('steady', time.perf_counter()-t0)
S1n, midn = U.to_np(S1), U.to_np(mid)
def flatten(d, p=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, p + k + '/'))
        else: out[p + k] = v
    return out
res = {}
for g in ('ocean', 'ice', 'lake', 'li', 'atm'):
    U.compare('S1/' + g + '/', flatten(S1n[g]), flatten(U.tree(ref, 'S1/' + g + '/')), out=res)
U.compare('mid/', flatten(midn), flatten(U.tree(ref, 'mid/')), out=res)
