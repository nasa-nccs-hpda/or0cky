"""D190 (stage S5): run the post-tile device program on the step-0 entry of one date and (1) compare with the NumPy reference npz of d190_ref_numpy.py (C1, every
saved quantity, categories A-D of ACCEPTANCE section 3), (2) compare the resulting surface state with the real records (C2, d187_common.surface_c2, the same code and the
same records as D170/D187), (3) measure compile time, steady time per unit, jit executions, eager dispatches, host<->device transfers and host callbacks.
    taskset -c 3-5 env OMP_NUM_THREADS=1 python d190_run.py DATE REF.npz OUT.json [mono|split] [steady_repeats]
Counters are installed before any jitted kernel module is imported."""
import clouds_jax_env  # noqa: F401
import sys, os, json, time
import jax_p1_count as CNT
CNT.install()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import d190_util as U
import jax_harness as H
import surface_loop as L
import jax_posttile as PT
import jax_ocean as JO

date, refp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else 'split'
nrep = int(sys.argv[5]) if len(sys.argv) > 5 else 5
IT0 = U.DATE_IT0[date]
res = dict(date=date, itime=IT0, mode=mode, ref=os.path.basename(refp))
res['header'] = H.provenance_header(extra=dict(task='D190 post-tile + ocean device program', date=date), require_xla_flags=True)
ref = dict(np.load(refp))
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
t0 = time.perf_counter(); K, Kb = PT.make_static_all(st, date, IT0); res['t_make_static_s'] = time.perf_counter() - t0
Kbd = U.to_dev(Kb)
S0 = U.to_dev(U.tree(ref, 'S0/')); mi = U.to_dev(U.tree(ref, 'melt/ice/')); me = U.to_dev(U.tree(ref, 'melt/melt/'))
inp = U.to_dev(U.tree(ref, 'inp/')); acc = U.to_dev(U.tree(ref, 'acc/')); V0 = U.to_dev(U.tree(ref, 'V0/'))
itime = jnp.asarray(IT0)
C = H.Counters()
C.listen_compiles()
pre = jax.jit(lambda S, mi, me, inp: PT.surface_pre_dev(K, S, mi, me, inp))
if mode == 'mono':
    post = jax.jit(lambda S1, mid, acc, srfp, itime, V, Kb: PT.surface_post_dev(K, Kb, S1, mid, acc, srfp, itime, V))
    units = ['pre', 'post']
else:
    pa = jax.jit(lambda S1, mid, acc, srfp, itime, V: PT.post_a(K, S1, mid, acc, srfp, itime, V))
    oa = jax.jit(lambda oc, fx, Kb: JO.ocean_stages(K, Kb, oc, fx, which='a'))
    ob = jax.jit(lambda oc, fx, Kb: JO.ocean_stages(K, Kb, oc, fx, which='b'))
    pb = jax.jit(lambda S1, c, oc, V: PT.post_b(K, S1, c, oc, V))
    units = ['pre', 'post_a', 'ocean_a', 'ocean_b', 'post_b']
tm = {}
def timed(name, f, *a):
    t0 = time.perf_counter()
    o = f(*a)
    U.sync(o)
    tm[name] = tm.get(name, 0.0) + time.perf_counter() - t0
    return o
def step(acc=acc):
    S1, mid = timed('pre', pre, S0, mi, me, inp)
    if mode == 'mono':
        S2, V2, pst = timed('post', post, S1, mid, acc, inp['srfp'], itime, V0, Kbd)
    else:
        c = timed('post_a', pa, S1, mid, acc, inp['srfp'], itime, V0)
        oc_a = timed('ocean_a', oa, S1['ocean'], c['fx'], Kbd)
        oc_b = timed('ocean_b', ob, oc_a, c['fx'], Kbd)
        S2, V2, pst = timed('post_b', pb, S1, c, oc_b, V0)
    return S2, V2, pst
# ---- cold
cold0 = C.snapshot()
t0 = time.perf_counter(); S2, V2, pst = step(); res['cold_total_s'] = time.perf_counter() - t0
res['cold_unit_s'] = dict(tm)
res['cold_compiles'] = C.totals()['compiles']; res['cold_compile_seconds'] = C.totals()['compile_seconds']
# ---- steady
tm.clear()
ts = []
CNT.reset()
with C.instrument_transfers():
    for r in range(nrep):
        t0 = time.perf_counter(); S2, V2, pst = step(); ts.append(time.perf_counter() - t0)
snap = CNT.snapshot()
res['steady_s'] = ts
res['steady_median_s'] = float(np.median(ts))
res['steady_unit_s'] = {k: v / nrep for k, v in tm.items()}
res['jit_executions_per_step'] = snap['jit_calls'] / nrep
res['eager_primitive_dispatches_per_step'] = (snap['eager_primitive_calls'] or 0) / nrep
res['jit_by_name'] = snap['by_name']
res['transfers_counted'] = {k: v for k, v in C.totals().items() if k.startswith(('h2d', 'd2h'))}
res['host_callbacks'] = dict(JO.XPRE_STATS)
# guarded run: no implicit host<->device transfer between the units
try:
    with jax.transfer_guard('disallow'):
        step()
    res['transfer_guard_disallow'] = 'passed'
except Exception as e:
    res['transfer_guard_disallow'] = 'raised: ' + str(e)[:300]
# ---- C1
def flatten(d, p=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, p + k + '/'))
        else: out[p + k] = v
    return out
S2n, V2n, pn = U.to_np(S2), U.to_np(V2), U.to_np(pst)
c1 = {}
for g in ('ocean', 'ice', 'lake', 'li', 'atm'):
    U.compare('S2/' + g + '/', flatten(S2n[g]), flatten(U.tree(ref, 'S2/' + g + '/')), out=c1, quiet=True)
U.compare('V2/', flatten(V2n), flatten(U.tree(ref, 'V2/')), out=c1, quiet=True)
for k in ('flowo', 'eflowo', 'apress'):
    U.compare('post/' + k + '/', {'x': pn[k]}, {'x': ref[f'post/{k}']}, out=c1, quiet=True)
for k in ('ice_after_gsi', 'ice_pre_adv', 'advsi_out', 'gs_l', 'gs_o', 'gli', 'dm', 'fx'):
    U.compare(f'post/{k}/', flatten(pn[k]), flatten(U.tree(ref, f'post/{k}/')), out=c1, quiet=True)
dd = {k: pn['dyn'][k] for k in ('odmui', 'odmvi', 'ui2rho', 'ustar', 'usi', 'vsi', 'uisurf', 'visurf')}
U.compare('post/dyn/', dd, flatten(U.tree(ref, 'post/dyn/')), out=c1, quiet=True)
cnt = {}
for v in c1.values(): cnt[v[0]] = cnt.get(v[0], 0) + 1
res['c1_counts'] = cnt
res['c1_not_A'] = {k: v for k, v in c1.items() if v[0] != 'A'}
res['c1_n'] = len(c1)
print('C1 counts', cnt, 'not A:', res['c1_not_A'], flush=True)
# ---- C2 (same code and records as D187)
import d187_common as K187
class Shim: pass
def shim(S2x, post, V2x):
    s = Shim(); s.st = st; s.SS = {g: S2x[g] for g in ('ocean', 'ice', 'lake', 'li', 'atm')}
    s.last = dict(post=dict(ice_pre_adv=post['ice_pre_adv'], dyn=post['dyn'], flowo=post['flowo'], eflowo=post['eflowo']))
    return s
t0 = time.perf_counter()
c2 = K187.surface_c2(date, IT0, shim(S2n, pn, V2n))
res['c2'] = c2
res['c2_categories'] = K187.summarize_surface_c2(c2)
res['t_c2_s'] = time.perf_counter() - t0
refS2 = U.tree(ref, 'S2/')
refpost = dict(ice_pre_adv=U.tree(ref, 'post/ice_pre_adv/'), dyn=U.tree(ref, 'post/dyn/'), flowo=ref['post/flowo'], eflowo=ref['post/eflowo'])
c2ref = K187.surface_c2(date, IT0, shim(refS2, refpost, None))
res['c2_numpy_path'] = c2ref
res['c2_numpy_path_categories'] = K187.summarize_surface_c2(c2ref)
res['c2_identical_to_numpy_path'] = (json.dumps(c2, sort_keys=True, default=float) == json.dumps(c2ref, sort_keys=True, default=float))
print('C2 categories', res['c2_categories'], 'identical to NumPy path:', res['c2_identical_to_numpy_path'])
# ---- chained: the DEVICE tile outputs of D188 (jax_surface) feed the post-tile program without leaving the device; reference = NumPy stage + surface_post_v2
if len(sys.argv) > 6:
    import atm_step as A
    import jax_surface as JS
    ref2 = dict(np.load(sys.argv[6]))
    reg = H.RecordedInputRegistry(); H.declare_d174_inputs(reg, date, IT0)
    R = A.Real(date, IT0)
    rec = reg.read('surface_records', lambda: A.surface_records(R), stage='d190_chain')
    S_atm = A.real_state_at(R, 'surface', None); A._native(S_atm)
    tpl, host = JS.build_template(rec)
    Sd = JS.state_from_numpy(S_atm)
    stage = JS.make_stage(host)
    t0 = time.perf_counter(); out_s, aux = stage(Sd, tpl); U.sync(out_s); res['chain_d188_stage_cold_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); out_s, aux = stage(Sd, tpl); U.sync(out_s); res['chain_d188_stage_steady_s'] = time.perf_counter() - t0
    accf = jax.jit(lambda aux_: PT.tile_accumulators_dev(host, tpl, aux_))
    accj = accf(aux); U.sync(accj)
    accn = U.to_np(accj)
    ca = {}
    for k, v in accn.items():
        ca['acc/' + k] = U.cat(v, ref2['acc/' + k])
    res['chain_acc_vs_numpy_stage'] = ca
    ref_acc2 = U.to_dev(U.tree(ref2, 'acc/'))
    S2c, V2c, pstc = step(accj)
    S2cn, V2cn, pcn = U.to_np(S2c), U.to_np(V2c), U.to_np(pstc)
    cc = {}
    for g in ('ocean', 'ice', 'lake', 'li', 'atm'):
        U.compare('S2/' + g + '/', flatten(S2cn[g]), flatten(U.tree(ref2, 'S2/' + g + '/')), out=cc, quiet=True)
    U.compare('V2/', flatten(V2cn), flatten(U.tree(ref2, 'V2/')), out=cc, quiet=True)
    for k in ('flowo', 'eflowo', 'apress'):
        U.compare('post/' + k + '/', {'x': pcn[k]}, {'x': ref2[f'post/{k}']}, out=cc, quiet=True)
    for k in ('ice_after_gsi', 'ice_pre_adv', 'advsi_out', 'gs_l', 'gs_o', 'gli', 'dm', 'fx'):
        U.compare(f'post/{k}/', flatten(pcn[k]), flatten(U.tree(ref2, f'post/{k}/')), out=cc, quiet=True)
    cnt2 = {}
    for v in list(cc.values()) + list(ca.values()):
        cnt2[v[0]] = cnt2.get(v[0], 0) + 1
    res['chain_c1_counts'] = cnt2
    res['chain_c1_not_A'] = {k: v for k, v in list(cc.items()) + list(ca.items()) if v[0] != 'A'}
    print('CHAIN C1 (D188 device tile outputs -> S5, vs NumPy stage + surface_post_v2):', cnt2, 'not A:', res['chain_c1_not_A'], flush=True)
res['xla_flags'] = os.environ.get('XLA_FLAGS')
json.dump(res, open(outp, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print({k: res[k] for k in ('cold_total_s', 'cold_unit_s', 'cold_compiles', 'steady_median_s', 'steady_unit_s', 'jit_executions_per_step', 'eager_primitive_dispatches_per_step',
                           'transfers_counted', 'host_callbacks', 'transfer_guard_disallow')})
