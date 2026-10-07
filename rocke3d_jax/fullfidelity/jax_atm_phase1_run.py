"""D186 runner: python jax_atm_phase1_run.py DATE OUTDIR REFDIR [--rad replay|server] [--fused-dyn] [--mutate NAME ...] [--steady N]
Run under:  taskset -c 0-1 env OMP_NUM_THREADS=1 python jax_atm_phase1_run.py ...   (the libm reference is valid for cores 0-1 only)
Runs phase 1 of step 0 of DATE (cold, with compile), compares every stage snapshot with the NumPy libm phase-1 reference (jax_p1_ref.py) with the A/B/C/D
categories of jax_harness, repeats the step for the steady-state time, optional mutation runs, writes OUTDIR/<date>_phase1[_tag].json."""
import clouds_jax_env_fast  # noqa: F401
import argparse
import json
import os
import sys
import time

import numpy as np

import jax_atm_phase1 as P1
import jax_harness as H
import jax
import jax.numpy as jnp
import jax_p1_count as CNT
import atm_step as A


def to_np(d):
    return {k: np.asarray(v) for k, v in d.items() if hasattr(v, 'shape')}


def compare_stage(cand, ref, prefix, only=None):
    keys = sorted(k for k in cand if f'{prefix}/{k}' in ref and (only is None or k in only))
    res = {}
    for k in keys:
        res[k] = H.field_category(np.asarray(cand[k]), ref[f'{prefix}/{k}'])
    missing = sorted(k[len(prefix) + 1:] for k in ref if k.startswith(prefix + '/') and k[len(prefix) + 1:] not in cand)
    return res, missing


def summarize(res):
    cats = {}
    for v in res.values():
        cats[v['cat']] = cats.get(v['cat'], 0) + 1
    notA = {k: dict(cat=v['cat'], max_abs=v['max_abs'], rel=v['rel'], n_diff=v['n_diff'], worst=v['worst'], cols=v['n_cols_over']) for k, v in res.items() if v['cat'] != 'A'}
    return cats, notA


def run(date, outdir, refdir, rad='replay', fused=False, mutate=(), steady=1, tag='', server=None, live_packet=None):
    os.makedirs(outdir, exist_ok=True)
    hdr = H.provenance_header(extra=dict(task='D186 S2 atmosphere phase 1 (device-resident)', date=date, rad=rad, fused_dyn=fused, mutate=list(mutate)),
                              require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    ref = H.load_npz(os.path.join(refdir, f'{date}_p1_1.npz'))
    ph = P1.Phase1(date, rad=rad, fused_dyn=fused, server=server, live_packet=live_packet)
    ph.ct.listen_compiles()
    it0 = ph.it0
    out = dict(header=hdr, date=date, itime=it0, rad_mode=rad, fused_dyn=fused)
    t0 = time.perf_counter()
    with ph.ct.instrument_transfers():
        with ph.ct.stage('record_load'):
            rec = ph.load_records(it0, first=True)
            dev = ph.to_device(rec, first=True)
            jax.block_until_ready(dev)
    out['record_load_seconds'] = time.perf_counter() - t0
    S0 = dev['S']
    carry, ms = {}, P1.CS.ms_zero()
    hold = ph.new_hold(dev['rad'])
    runs = []
    res_first = None
    for rep in range(1 + steady):
        CNT.reset()
        c0 = ph.ct.snapshot()
        t0 = time.perf_counter()
        with ph.ct.instrument_transfers():
            with ph.ct.stage('step'):
                r = ph.step(it0, dev, S0, carry, ms, hold, dev['ice'], timed=False, rec=rec)
                jax.block_until_ready((r['S'], r['X'], r['flags']))
                with ph.ct.stage('flag_read'):
                    fl = P1.D.flags_to_host(r['flags'])
        wall = time.perf_counter() - t0
        snap = ph.ct.snapshot()
        cnt = CNT.snapshot()
        runs.append(dict(rep=rep, wall_seconds=wall, flags=fl, jit_executions=cnt['jit_calls'], eager_primitive_executions=cnt['eager_primitive_calls'],
                         by_stage_jit=r['info']['stage_jit_calls'], stage_seconds_untimed_dispatch=r['info']['stage_seconds'],
                         compiles_total=snap.get('step', {}).get('compiles', 0), compile_seconds_total=snap.get('step', {}).get('compile_seconds', 0.0)))
        if rep == 0:
            res_first = r
        print('run', rep, 'wall %.1f s' % wall, 'flags', fl, 'jit executions', cnt['jit_calls'], 'eager prims', cnt['eager_primitive_calls'], flush=True)
    out['runs'] = runs
    # ---- per-stage timing run with a block after every stage (adds syncs; labelled)
    CNT.reset()
    t0 = time.perf_counter()
    r2 = ph.step(it0, dev, S0, carry, ms, hold, dev['ice'], timed=True, rec=rec)
    jax.block_until_ready((r2['S'], r2['X']))
    out['timed_run'] = dict(wall_seconds=time.perf_counter() - t0, stage_seconds=r2['info']['stage_seconds'], note='block_until_ready after every stage (extra host syncs)')
    # ---- comparison with the NumPy libm phase-1 reference (the C1 comparison)
    comp = {}
    allcats = {}
    for st in ('dyn', 'condse', 'radia'):
        cand = to_np(res_first['snaps'][st])
        res, missing = compare_stage(cand, ref, st)
        cats, notA = summarize(res)
        comp[st] = dict(n_fields=len(res), categories=cats, not_A=notA, missing_in_candidate=missing)
        allcats[st] = cats
    resX, miss = compare_stage(to_np(res_first['X']), ref, 'X')
    cats, notA = summarize(resX)
    comp['X'] = dict(n_fields=len(resX), categories=cats, not_A=notA, missing_in_candidate=miss)
    if 'rad/CLDSS' in ref and A.is_radiation_step(it0):
        for k in ('CLDSS', 'CLDMC'):
            cm = np.asarray(res_first['cloud_masked'][k])
            comp.setdefault('rad_cloud_mask', {})[k] = H.field_category(cm, ref['rad/' + k])
    # gate: category A on all atmosphere fields (radia-stage snapshot + X)
    gate_fields = sorted(comp['radia']['not_A'].keys()) if comp['radia']['not_A'] else []
    out['comparison'] = comp
    out['melt_si'] = dict(rsi_equals_recorded_condse_entry=bool(np.array_equal(np.asarray(res_first['ice']['rsi']), rec['cse_in']['RSI'])))
    out['non_jax'] = ph.stages.non_jax_text()
    out['declared_host'] = P1.DECLARED_HOST
    out['units'] = P1.UNITS
    out['pole_callbacks'] = dict(calls=ph.cs.host.calls, seconds=ph.cs.host.seconds, bytes=ph.cs.host.bytes, log=ph.cs.host.log[:4])
    out['mstcnv'] = ph.cs_stats
    out['qus_subsidence_callback'] = dict(ph.qus, note='all steps run in this process so far (cold step, steady step, timed step; mutation runs come after this line is written)')
    out['transfers_counters'] = ph.ct.snapshot()
    if rad == 'server':
        out['radiation'] = dict(mode=rad, sentence=ph._handoff.log.sentence(), totals=ph._handoff.log.totals())
    else:
        out['radiation'] = dict(mode=rad, sentence=str(ph.rad_log.sentence),
                                note='NO radiation was computed in this result: SRHR/TRHR/COSZ1 are the recorded RADIA outputs of ffa_step_<it>_r served through the D185 hand-off '
                                     'interface by a stand-in server; the counters below are those of the INTERFACE (packet assembly and callback), not of a Fortran call',
                                interface_counters={k: v for k, v in ph._handoff.log.totals().items() if k not in ('radiation',)})
    out['recorded_inputs'] = ph.reg.emit()
    # ---- mutation tests
    mut = {}
    for name in mutate:
        saved = None
        if name == 'dyn_kg2mb':
            saved = ph.dyn.K['kg2mb']
            ph.dyn.K['kg2mb'] = saved * (1 + 2.2e-16)
        elif name == 'condse_ls_RGAS':
            saved = ph.cs.Kls['RGAS']
            ph.cs.Kls['RGAS'] = saved * (1 + 1e-12)
        elif name == 'radia_cosz':
            saved = (dev['rad']['COSZ1'], rec['rad']['COSZ1'])
            dev['rad']['COSZ1'] = saved[0] * (1 + 1e-12)
            rec['rad']['COSZ1'] = saved[1] * (1 + 1e-12)
        else:
            raise ValueError(name)
        rm = ph.step(it0, dev, S0, carry, ms, hold, dev['ice'], timed=False, rec=rec)
        jax.block_until_ready(rm['S'])
        cm = {}
        for st in ('dyn', 'condse', 'radia'):
            res, _ = compare_stage(to_np(rm['snaps'][st]), ref, st)
            c, nA = summarize(res)
            cm[st] = dict(categories=c, n_not_A=len(nA), examples=dict(list(nA.items())[:4]))
        mut[name] = dict(detected=any(v['n_not_A'] > 0 for v in cm.values()), by_stage=cm)
        if name == 'dyn_kg2mb':
            ph.dyn.K['kg2mb'] = saved
        elif name == 'condse_ls_RGAS':
            ph.cs.Kls['RGAS'] = saved
        else:
            dev['rad']['COSZ1'], rec['rad']['COSZ1'] = saved
        print('mutation', name, 'detected', mut[name]['detected'], flush=True)
    out['mutation_tests'] = mut
    fn = os.path.join(outdir, f'{date}_phase1{tag}.json')
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    print('wrote', fn)
    for st in ('dyn', 'condse', 'radia', 'X'):
        print(st, comp[st]['categories'], 'not A:', list(comp[st]['not_A'])[:12], 'missing:', comp[st]['missing_in_candidate'][:6])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('date'); ap.add_argument('outdir'); ap.add_argument('refdir')
    ap.add_argument('--rad', default='replay'); ap.add_argument('--fused-dyn', action='store_true')
    ap.add_argument('--mutate', nargs='*', default=[]); ap.add_argument('--steady', type=int, default=1); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    run(a.date, a.outdir, a.refdir, a.rad, a.fused_dyn, tuple(a.mutate), a.steady, a.tag)
