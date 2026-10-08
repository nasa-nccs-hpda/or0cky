"""D202: single-cell JAX (ghy_jax, the port) versus NumPy (ghy_ref) first-iteration intermediates for cell 112 at a captured step/substep, SAME inputs.
    python d202_diag.py CAPDIR STEP NS"""
import sys, json
import numpy as np
import clouds_jax_env  # noqa
import jax, jax.numpy as jnp
import ghy_jax as J
import ghy_compare as GC
import ghy_ref as G
import d202_cell112 as C

cap, k, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
QM1 = float(sys.argv[4]) if len(sys.argv) > 4 else None      # override of q1*ma1 (the capture of q1 in run1 is the wrong cell)
GUSTI = sys.argv[5] if len(sys.argv) > 5 else 'rec'       # 'rec' = recorded (what the port does) | 'pbl' = the gusti the PBL used
z = np.load(f'{cap}/in_{k}.npz')
CELL = C.CELL
g = 'ghy/g%d/' % ns
sl = slice(CELL, CELL + 1)
get = lambda p: jnp.asarray(z[p][sl])
static = {kk.split('/')[-1]: get(kk) for kk in z.files if kk.startswith(g + 'static/')}
dyn_src = 'ghy/g0/dyn0/' if ns == 0 else 'res/r1/dyn_next/'
dyn = {kk: get(dyn_src + kk) for kk in ('w', 'ht', 'nsn', 'dzsn', 'wsn', 'hsn', 'fr_snow')}
forcing = {kk.split('/')[-1]: get(kk) for kk in z.files if kk.startswith(g + 'forcing/')}
tag = 'r1' if ns == 0 else 'r2'
rows = z['rows/rows1' if ns == 0 else 'rows/rows2'][CELL]
pb = {kk: float(z[f'res/{tag}/pbl/{kk}'][CELL]) for kk in ('tsv', 'qsrf', 'ch', 'ws')}
ddml = rows[23] > 0.5
RG = C.RGAS
rho = 100.0 * rows[16] / (RG * pb['tsv'])
q1 = float(z['misc/q1'][CELL]); ma1 = float(z['misc/ma1'][CELL])
qm1v = QM1 if QM1 is not None else q1 * ma1
forcing.update(ts=jnp.asarray([pb['tsv']]), qs=jnp.asarray([pb['qsrf']]), rho=jnp.asarray([rho]), ch=jnp.asarray([pb['ch']]), vs=jnp.asarray([pb['ws']]),
               tprime=jnp.asarray([(rows[25] - rows[7]) if ddml else 0.0]), qprime=jnp.asarray([(rows[26] - rows[39]) if ddml else 0.0]), qm1=jnp.asarray([qm1v]))
if GUSTI == 'pbl':
    forcing['gusti'] = jnp.asarray([rows[24]])
print('forcing', {kk: float(np.asarray(v)[0]) for kk, v in forcing.items()})
args = (static, dyn, forcing, get(g + 'edts'), get(g + 'ecnc'), get(g + 'ebet'), get(g + 'elai'), get(g + 'nsub'), get(g + 'dt'), get(g + 'snowm'))
out = jax.jit(J.advnc, static_argnames=('max_substeps',))(*args, max_substeps=int(z[g + 'edts'].shape[1]))
print('JAX advnc  tsns %.6f ashg %.4f alhg %.4f aevap %.6e' % tuple(float(np.asarray(out[x])[0]) for x in ('tsns', 'ashg', 'alhg', 'aevap')))
print('port saved tsns %.6f ashg %.4f alhg %.4f aevap %.6e' % tuple(float(z[f'res/{tag}/ghy/{x}'][CELL]) for x in ('tsns', 'ashg', 'alhg', 'aevap')))
# JAX first-iteration evap_limits
st = dict(static, process_bare=forcing['fb'] > 0, process_vege=forcing['fv'] > 0)
st = dict(st)
reth0 = J.reth(st, dyn['w'], dyn['nsn'], dyn['wsn'], dyn['fr_snow'], get(g + 'snowm'))
retp0 = J.retp(st, dyn['w'], dyn['ht'], dyn['wsn'], dyn['hsn'])
hy = J.hydra(st, reth0['theta'], retp0['fice'])
ev = J.evap_limits(st, dyn['w'], reth0['theta'], hy['d'], retp0['tp'], retp0['fice'], retp0['tsn1'], dyn['nsn'], dyn['wsn'], dyn['fr_snow'], get(g + 'dt'), forcing['pr'],
                   get(g + 'ebet')[:, 0], get(g + 'ecnc')[:, 0], forcing['ch'], forcing['vs'], forcing['rho'], forcing['pres'], forcing['qs'], forcing['gusti'], forcing['qprime'],
                   forcing['qm1'], get(g + 'elai')[:, 0], reth0['fm'])
print('JAX evap_limits it0:', {kk: np.asarray(v)[0].tolist() for kk, v in ev.items() if kk in ('evapb', 'evapvw', 'evapvd', 'evapvg', 'evap_min', 'abetad')})
print('JAX tp', np.asarray(retp0['tp'])[0].tolist(), 'fw', np.asarray(reth0['fw']), 'fm', np.asarray(reth0['fm']))
# NumPy
FF = C.FF
gg = GC.load(f'{FF}/ffg_{33312+k}.bin'); n = len(gg) // 2
rec = gg[ns * n + CELL]
dyn_np = {kk: np.asarray(dyn[kk])[0] for kk in dyn}
over = {kk: float(np.asarray(forcing[kk])[0]) for kk in ('ts', 'qs', 'rho', 'ch', 'vs', 'tprime', 'qprime', 'qm1')}
col, ent, refs, snowm = C.column(rec, dyn_np, over)
print('NumPy forcing gusti %.4g pr %.4g qm1 %.4g | JAX gusti %.4g pr %.4g' % (col.gusti, col.pr, col.qm1, float(forcing['gusti'][0]), float(forcing['pr'][0])))
col.dt = 900.0; col.snowm = snowm; col._bounds(); col.accm_zero(); col.reth(); col.retp()
col.evapb = col.epb = 1.0; col.evapvw = col.evapvd = col.epv = 1.0
col.dts = ent[0]['dts']; col.cnc = ent[0]['cnc'] if col.process_vege else 0.0
col.betadl = np.asarray(ent[0]['betadl']) if col.process_vege else np.zeros(col.n); col.lai = ent[0]['lai'] if col.process_vege else 0.0
col.hydra(); col.xklh(); col.evap_limits(True)
print('NumPy evap_limits it0:', dict(evapb=col.evapb, evapvw=col.evapvw, evapvd=col.evapvd, evapvg=col.evapvg, evap_min=col.evap_min), 'tp', col.tp.T.tolist())
print('fb fv', col.fb, col.fv, 'ent0', ent[0], 'JAX ecnc', float(get(g + 'ecnc')[0, 0]), 'elai', float(get(g + 'elai')[0, 0]))
