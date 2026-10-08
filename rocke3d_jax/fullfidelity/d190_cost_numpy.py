"""D190 cost of the EXISTING NumPy path (surface_pre + surface_post_v2) on the same entry: first call (compile), steady seconds, jit executions, eager primitive
dispatches, host<->device transfers (jax_harness.Counters.instrument_transfers).  Counters are installed BEFORE the modules that define jitted kernels are imported.
usage: d190_cost_numpy.py DATE OUT.json [steady_repeats]"""
import clouds_jax_env  # noqa: F401
import sys, json, time, os
import jax_p1_count as CNT
CNT.install()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax_harness as H
import surface_loop as L
import surface_loop_v2 as V2
import copy

date, out = sys.argv[1], sys.argv[2]
nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}[date]
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
S = L.init_surface_state(date, st=st)
inp = V2.replay_inputs_v2(date, IT0)
C = H.Counters()
C.listen_compiles()
res = dict(date=date)
PROF = {}
def wrap(obj, name, label):
    f = getattr(obj, name)
    def w(*a, **k):
        t0 = time.perf_counter()
        before = CNT.snapshot()
        r = f(*a, **k)
        after = CNT.snapshot()
        d = PROF.setdefault(label, dict(seconds=0.0, jit=0, eager=0, calls=0))
        d['seconds'] += time.perf_counter() - t0
        d['jit'] += after['jit_calls'] - before['jit_calls']
        d['eager'] += (after['eager_primitive_calls'] or 0) - (before['eager_primitive_calls'] or 0)
        d['calls'] += 1
        return r
    setattr(obj, name, w)
import riverf_ff as RF
import surface_loop_advsi as AL
import dynsi_loop_ff as DL
profile = os.environ.get('D190_PROFILE', '1') == '1'
if profile:
    for nm in ('precip_si', 'seaice_to_atmgrid', 'precip_li', 'irrig_lk', 'precip_lk', 'ground_li', 'underice', 'ground_si', 'ground_lk', 'form_si', 'calc_apress', 'toc2sst', 'melt_si'):
        wrap(L, nm, 'L.' + nm)
    wrap(RF, 'riverf', 'riverf')
    wrap(AL, 'advsi_on_state', 'advsi_on_state')
    wrap(DL.Dynsi, '__call__', 'dynsi')
    import ocean_step as _O
    for k, (nm, fn, tag) in enumerate(_O.STAGES):
        pass
    _orig_stages = L.ocean_stages_after_precip
    def _staged():
        out = []
        for nm, fn, tag in _orig_stages():
            def w(oc, fx, ctx, fn=fn, nm=nm):
                t0 = time.perf_counter(); b = CNT.snapshot()
                r = fn(oc, fx, ctx)
                a = CNT.snapshot()
                d = PROF.setdefault('ocean.' + nm, dict(seconds=0.0, jit=0, eager=0, calls=0))
                d['seconds'] += time.perf_counter() - t0; d['jit'] += a['jit_calls'] - b['jit_calls']; d['eager'] += (a['eager_primitive_calls'] or 0) - (b['eager_primitive_calls'] or 0); d['calls'] += 1
                return r
            out.append((nm, w, tag))
        return out
    L.ocean_stages_after_precip = _staged
    _o_stage_precip = _O.stage_precip
def one(tag):
    S0 = copy.deepcopy(S)
    V = V2.V2State(st, date, L.FF, IT0)
    ice_m, melt = L.melt_si(S0['ice'], S0['atm']['gtemp'], S0['atm']['sss'], S0['atm']['mlhc'], st['geo'])
    CNT.reset()
    t0 = time.perf_counter()
    with C.instrument_transfers():
        with C.stage(tag):
            S1, mid = L.surface_pre(S0, st, inp, melt_done=(ice_m, melt))
            S2, post = V2.surface_post_v2(S1, st, dict(acc=inp['acc'], srfp=inp['srfp'], itime=IT0), mid, V)
    dt = time.perf_counter() - t0
    snap = CNT.snapshot()
    return dt, snap
dt0, snap0 = one('cold')
PROF.clear()
res['cold_s'] = dt0; res['cold_counts'] = snap0
steady = []
for k in range(nrep):
    dt, snap = one('steady%d' % k)
    steady.append(dt)
res['steady_s'] = steady; res['steady_counts'] = snap
res['transfer_counters'] = C.snapshot()
res['profile_steady_per_step'] = {k: dict(seconds=v['seconds'] / nrep, jit=v['jit'] / nrep, eager=v['eager'] / nrep, calls=v['calls'] / nrep) for k, v in PROF.items()}
res['compiles_total'] = C.totals()
json.dump(res, open(out, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print(json.dumps({k: v for k, v in res.items() if k != 'transfer_counters'}, indent=1, default=str))
print(C.report_text())
