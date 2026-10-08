"""D190 C1: the whole post-tile program (surface_pre_dev + surface_post_dev) against the NumPy reference npz (isolated mode: acc from the npz).  usage: DATE REF.npz"""
import sys, time, json
import d190_util as U
import numpy as np, jax, jax.numpy as jnp
import surface_loop as L
import jax_posttile as PT

date, refp = sys.argv[1], sys.argv[2]
ref = dict(np.load(refp))
it0 = U.DATE_IT0[date]
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
t0 = time.perf_counter(); K, Kb = PT.make_static_all(st, date, it0); print('make_static_all', round(time.perf_counter() - t0, 1), flush=True)
Kbd = U.to_dev(Kb)
S0 = U.to_dev(U.tree(ref, 'S0/')); mi = U.to_dev(U.tree(ref, 'melt/ice/')); me = U.to_dev(U.tree(ref, 'melt/melt/'))
inp = U.to_dev(U.tree(ref, 'inp/')); acc = U.to_dev(U.tree(ref, 'acc/')); V0 = U.to_dev(U.tree(ref, 'V0/'))
pre = jax.jit(lambda S, mi, me, inp: PT.surface_pre_dev(K, S, mi, me, inp))
post = jax.jit(lambda S1, mid, acc, srfp, itime, V, Kb: PT.surface_post_dev(K, Kb, S1, mid, acc, srfp, itime, V))
t0 = time.perf_counter(); S1, mid = pre(S0, mi, me, inp); U.sync(S1); print('pre first', round(time.perf_counter() - t0, 1), flush=True)
t0 = time.perf_counter(); S2, V2, pst = post(S1, mid, acc, inp['srfp'], jnp.asarray(it0), V0, Kbd); U.sync(S2); print('post first', round(time.perf_counter() - t0, 1), flush=True)
for r in range(2):
    t0 = time.perf_counter(); S1, mid = pre(S0, mi, me, inp); S2, V2, pst = post(S1, mid, acc, inp['srfp'], jnp.asarray(it0), V0, Kbd); U.sync(S2); print('steady pre+post', round(time.perf_counter() - t0, 3), flush=True)
def flatten(d, p=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, p + k + '/'))
        else: out[p + k] = v
    return out
S2n, V2n, pn = U.to_np(S2), U.to_np(V2), U.to_np(pst)
res = {}
for g in ('ocean', 'ice', 'lake', 'li', 'atm'):
    U.compare('S2/' + g + '/', flatten(S2n[g]), flatten(U.tree(ref, 'S2/' + g + '/')), out=res)
U.compare('V2/', flatten(V2n), flatten(U.tree(ref, 'V2/')), out=res)
for k in ('flowo', 'eflowo', 'apress'):
    U.compare('post/' + k + '/', {'x': pn[k]}, {'x': ref[f'post/{k}']}, out=res)
for k in ('ice_after_gsi', 'ice_pre_adv', 'advsi_out', 'gs_l', 'gs_o', 'gli', 'dm', 'fx'):
    U.compare(f'post/{k}/', flatten(pn[k]), flatten(U.tree(ref, f'post/{k}/')), out=res)
dd = {k: pn['dyn'][k] for k in ('odmui', 'odmvi', 'ui2rho', 'ustar', 'usi', 'vsi', 'uisurf', 'visurf')}
U.compare('post/dyn/', dd, flatten(U.tree(ref, 'post/dyn/')), out=res)
cnt = {}
for v in res.values(): cnt[v[0]] = cnt.get(v[0], 0) + 1
print('TOTAL', cnt)
