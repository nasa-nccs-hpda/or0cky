"""D190 (stage S5) C1 reference: the EXISTING NumPy post-tile path surface_loop_v2.surface_post_v2 (with surface_loop.surface_pre before it) on the
step-0 entry state of one date, with every intermediate saved, so that jax_posttile.py can be compared stage by stage.

    taskset -c 3-5 env OMP_NUM_THREADS=1 python d190_ref_numpy.py DATE OUT.npz [replay|stage]

Entry state: the real restart (surface_loop.init_surface_state), MELT_SI of surface_loop.melt_si (NumPy glue + the jitted simelt), the real PREC/EPREC
of the CONDSE exit, SRFP = PEDN(1) of the pre-surface state and the irrigation flux of the ffg record (surface_loop.replay_inputs, D164/D170).
Tile accumulators (acc): 'replay' = the real SURFACE tile outputs of the step (surface_loop.tile_accumulators); 'stage' = the NumPy SURFACE stage of D188
(atm_step.stage_surface, land_mode 'ghy', recorded Ent) run from the real entry state of that stage, reduced with surface_loop.tile_outputs_to_acc.
Nothing existing is modified; the ocean stages are wrapped (not edited) to record the ocean state after each stage.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: F401,E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import surface_loop as L  # noqa: E402
import surface_loop_v2 as V2  # noqa: E402

DATE_IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}
ACC_KEYS = ('e0_o', 'evap_o', 'solar_o', 'dmua_o', 'dmva_o', 'e0_i', 'e1_i', 'evap_i', 'solar_i', 'e0_li', 'e1_li', 'evap_li', 'rune', 'erune',
            'dmua_i', 'dmva_i')


def flat(prefix, d, out):
    if isinstance(d, dict):
        for k, v in d.items():
            flat(f'{prefix}{k}/', v, out)
    elif d is None:
        return
    else:
        a = np.asarray(d)
        if a.dtype.kind in 'fiub':
            out[prefix[:-1]] = a
    return out


def stage_acc(date, st):
    """acc from the NumPy SURFACE stage of D188 (real entry state of the stage, recorded Ent)."""
    import atm_step as A
    it = DATE_IT0[date]
    R = A.Real(date, it)
    S = A.real_state_at(R, 'surface', None)
    A._native(S)
    rec = A.surface_records(R)
    S = L.stage_surface_closed(S, R, None, rec)        # = atm_step.stage_surface (land_mode 'ghy', recorded Ent) that also returns the tile rows
    sd = S['_surface']
    acc = L.tile_outputs_to_acc(sd, st)
    acc['dmua_i'], acc['dmva_i'] = V2.ice_tile_dmua_from_tiles(sd)
    return acc


def run(date, acc_mode='replay', log=print):
    it0 = DATE_IT0[date]
    tm = {}
    t0 = time.perf_counter()
    st = L.load_statics(date)
    st['ctx'] = L.make_ocean_ctx(date)
    S = L.init_surface_state(date, st=st)
    V = V2.V2State(st, date, L.FF, it0)
    inp = V2.replay_inputs_v2(date, it0)
    tm['setup'] = time.perf_counter() - t0
    if acc_mode == 'stage':
        t0 = time.perf_counter()
        inp['acc'] = stage_acc(date, st)
        tm['tile_stage'] = time.perf_counter() - t0
    out = {}
    flat('S0/', {k: S[k] for k in ('ocean', 'ice', 'lake', 'li', 'atm')}, out)
    flat('V0/', dict(rsix=V.adv['rsix'], rsiy=V.adv['rsiy'], usi=V.dyn.usi, vsi=V.dyn.vsi), out)
    flat('inp/', dict(prec=inp['prec'], eprec=inp['eprec'], srfp=inp['srfp'], irrig_act=inp['irrig_act']), out)
    flat('acc/', {k: inp['acc'][k] for k in ACC_KEYS}, out)
    t0 = time.perf_counter()
    ice_m, melt = L.melt_si(S['ice'], S['atm']['gtemp'], S['atm']['sss'], S['atm']['mlhc'], st['geo'])
    tm['melt_si'] = time.perf_counter() - t0
    flat('melt/ice/', ice_m, out)
    flat('melt/melt/', melt, out)
    t0 = time.perf_counter()
    S1, mid = L.surface_pre(S, st, inp, melt_done=(ice_m, melt))
    tm['surface_pre'] = time.perf_counter() - t0
    flat('S1/', {k: S1[k] for k in ('ocean', 'ice', 'lake', 'li', 'atm')}, out)
    flat('mid/', mid, out)
    trace = {}
    stages = []
    ts = {}
    for name, fn, tag in L.ocean_stages_after_precip():
        def w(oc, fx, ctx, fn=fn, name=name):
            t1 = time.perf_counter()
            r = fn(oc, fx, ctx)
            ts[name] = time.perf_counter() - t1
            trace[name] = {k: np.array(v) for k, v in r.items() if isinstance(v, np.ndarray)}
            return r
        stages.append((name, w, tag))
    post_in = dict(acc=inp['acc'], srfp=inp['srfp'], itime=it0)
    t0 = time.perf_counter()
    S2, post = V2.surface_post_v2(S1, st, post_in, mid, V, ocean_stages=stages)
    tm['surface_post_v2'] = time.perf_counter() - t0
    tm['ocean_stages'] = ts
    flat('S2/', {k: S2[k] for k in ('ocean', 'ice', 'lake', 'li', 'atm')}, out)
    flat('V2/', dict(rsix=V.adv['rsix'], rsiy=V.adv['rsiy'], usi=V.dyn.usi, vsi=V.dyn.vsi, uisurf=V.uisurf, visurf=V.visurf), out)
    for k in ('flowo', 'eflowo', 'apress'):
        flat(f'post/{k}/', post[k], out)
    for k in ('ice_after_gsi', 'ice_pre_adv', 'advsi_out', 'gs_l', 'gs_o', 'gli', 'dm', 'fx'):
        flat(f'post/{k}/', post[k], out)
    d = post['dyn']
    flat('post/dyn/', {k: d[k] for k in ('odmui', 'odmvi', 'ui2rho', 'ustar', 'usi', 'vsi', 'uisurf', 'visurf')}, out)
    for name, dd in trace.items():
        flat(f'ocean_trace/{name}/', dd, out)
    out['_times'] = np.array([tm['setup'], tm['melt_si'], tm['surface_pre'], tm['surface_post_v2']])
    log('times', {k: (round(v, 3) if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()}) for k, v in tm.items()})
    return out, tm


if __name__ == '__main__':
    date, path = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'replay'
    o, tm = run(date, mode)
    np.savez(path, **o)
    print('saved', path, len(o), 'arrays')
