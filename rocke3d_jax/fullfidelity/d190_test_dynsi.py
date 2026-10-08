"""D190: jax_dynsi vs dynsi_ff.dynsi (NumPy) / the reference npz."""
import sys, time
import d190_util as U
import numpy as np, jax, jax.numpy as jnp
import surface_loop as L
import dynsi_ff as DF
import jax_dynsi as JD
import jax_seaice_lake as SL
from odhorz_ff import geomo_dyn_arrays

date, refp = sys.argv[1], sys.argv[2]
ref = dict(np.load(refp))
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
geo = st['geo']
K = SL.make_static(st)
sinvo, sinpo = geomo_dyn_arrays()[:2]
# np_sum72 vs np.sum
rng = np.random.default_rng(0)
bad = 0
f72 = jax.jit(JD.np_sum72)
for t in range(300):
    x = rng.normal(size=72) * 10 ** rng.uniform(-3, 3, 72)
    bad += (np.sum(x) != float(f72(x)))
print('np_sum72 != np.sum:', bad, 'of 300')
Kd = JD.make_static(geo['focean'], sinpo, sinvo)
S1 = U.tree(ref, 'S1/'); acc = U.tree(ref, 'acc/'); V0 = U.tree(ref, 'V0/')
oo = geo['is_ocean']
rsi, msi, snowi = (np.where(oo, S1['ice'][k], 0.0) for k in ('rsi', 'msi', 'snowi'))
oc = S1['ocean']
oga = L.toc2sst(oc, st['ctx'])['ogeoza']
us, vs = DF.uosurf_from_ocean(oc['uo'][:, :, 0], oc['vo'][:, :, 0], sinpo, sinvo)
G = DF.Geom(geo['focean'])
t0 = time.perf_counter()
out = DF.dynsi(G, acc['dmua_i'], acc['dmva_i'], rsi, msi, snowi, oga, us, vs, V0['usi'], V0['vsi'])
print('numpy dynsi', round(time.perf_counter() - t0, 3), 'kki', out['kki'])
@jax.jit
def fj(dmua, dmva, rsi, msi, snowi, oga, uo1, vo1, usi, vsi):
    return JD.dynsi(Kd, dmua, dmva, rsi, msi, snowi, oga, uo1, vo1, usi, vsi)
args = [jnp.asarray(a) for a in (acc['dmua_i'], acc['dmva_i'], rsi, msi, snowi, oga, oc['uo'][:, :, 0], oc['vo'][:, :, 0], V0['usi'], V0['vsi'])]
t0 = time.perf_counter(); r = fj(*args); U.sync(r); print('jax first', round(time.perf_counter() - t0, 2))
t0 = time.perf_counter(); r = fj(*args); U.sync(r); print('jax steady', round(time.perf_counter() - t0, 4), 'kki', int(r['kki']))
rn = U.to_np(r)
res = {}
for k in ('odmui', 'odmvi', 'ui2rho', 'ustar', 'usi', 'vsi', 'uisurf', 'visurf'):
    res[k] = U.cat(rn[k], out[k])
res['us'] = U.cat(rn['us'], us); res['vs'] = U.cat(rn['vs'], vs)
for k in ('gairx', 'gairy', 'gwatx', 'gwaty', 'pgfub', 'pgfvb', 'heff', 'area', 'amass', 'cor'):
    res['in.' + k] = U.cat(rn['inp'][k], out['inputs'][k])
res['uice1'] = U.cat(rn['uice1'], out['uice1']); res['vice1'] = U.cat(rn['vice1'], out['vice1']); res['dwatn'] = U.cat(rn['dwatn'], out['dwatn'])
for k, v in res.items(): print(f'{k:12s}', v)
print('kki equal', int(rn['kki']) == out['kki'])
