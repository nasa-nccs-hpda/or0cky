"""D191: run the ASSEMBLED, device-resident coupled step (jax_coupled.Coupled) at step 0 of one date and compare it
  C1  with the NumPy libimf chain with the same closed surface (d187_ref_numpy.py; same pinning), field by field with the harness categories,
  C2  with the real Fortran step-boundary dumps (same fields and verdict logic as D187: 14 gate fields, 30 end fields, surface records),
and measure: cold compile and steady seconds per step, per-stage shares (blocked run), jit executions, eager dispatches, transfers, host callbacks.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> python d191_run.py DATE OUTDIR REFDIR [--server] [--runs 3]
Runs in one process: run 1 = cold + validation arrays (compile), run 2 = steady, run 3 = steady with a device block after every stage, [server run, nov26 only]."""
import clouds_jax_env  # noqa: F401
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import jax_coupled as C
import jax
import jax.numpy as jnp
import jax_atm_phase1 as P1
import jax_p1_count as CNT
import jax_harness as H
import jax_ocean as JO
import libimf_ops as LI
import libimf_fused as FX
import clouds_mstcnv_dev as md
import atm_step as A
import surface_loop as L
import d187_common as K
import d187_analyze as AN


def build_registry(rad_mode):
    sr = H.StageRegistry()
    for n, (k, d) in P1.STAGE_TABLE.items():
        if n == 'melt_si':
            continue
        if n == 'radia':
            k = {'replay': 'JJ/REC', 'server': 'JJ/FORT'}[rad_mode]
        if n == 'condse_mstcnv':
            d = ('MSTCNV (D189): cloud-base loop and event block as ONE device program; libimf exp/pow through FUSED host callbacks (C shim over libimf.so); '
                 'QUS ADV1D subsidence = NumPy host callback (176 calls)')
        if n in ('dyn', 'condse_setup', 'condse_post'):
            d += ' [libimf pow/exp: labelled host callback inside this stage, time included]'
        sr.register(n, k, d)
    for n, k, d in C.STAGES2:
        sr.register(n, k, d)
    return sr


def snap_np(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, 'shape'):
            a = np.asarray(v)
            if a.dtype.kind in 'fiub':
                out[k] = a
    return out


def run_once(cp, label, timed, keep, rad_mode='replay', server=None, lv=None, guard=False):
    sr = build_registry(rad_mode)
    ct = H.Counters()
    ct.listen_compiles()
    LI.reset_counters()
    FX.reset_counters()
    md.reset_qus()
    for kk in JO.XPRE_STATS:
        JO.XPRE_STATS[kk] = 0 if kk != 'seconds' else 0.0
    CNT.reset()
    hp0 = cp.ph.cs.host
    p0 = (hp0.calls, hp0.seconds, hp0.bytes) if hp0 is not None else (0, 0.0, 0)
    if getattr(cp.ph, '_handoff', None) is not None:
        cp.ph._handoff.log.reset()
    out = dict(label=label, rad_mode=rad_mode, timed=timed)
    try:
        with ct.instrument_transfers():
            t0 = time.perf_counter()
            c0 = CNT.snapshot()
            with ct.stage('record_load'), sr.time('record_load'):
                state, rec, dev = cp.initial_state()
                jax.block_until_ready((state, dev))
            c1 = CNT.snapshot()
            out['record_load_seconds'] = time.perf_counter() - t0
            out['record_load_jit_eager'] = dict(jit=c1['jit_calls'] - c0['jit_calls'], eager=(c1['eager_primitive_calls'] or 0) - (c0['eager_primitive_calls'] or 0))
            t1 = time.perf_counter()
            cs0 = CNT.snapshot()
            with ct.stage('step'):
                new, info, arrays = cp.step(0, state, rec, dev, sr, timed=timed, keep=keep, server=server, lv=lv)
                jax.block_until_ready(new)
            cs1 = CNT.snapshot()
            out['step_seconds'] = time.perf_counter() - t1
            out['wall_seconds'] = time.perf_counter() - t0
            out['step_jit_executions'] = cs1['jit_calls'] - cs0['jit_calls']
            out['step_eager_primitive_dispatches'] = (cs1['eager_primitive_calls'] or 0) - (cs0['eager_primitive_calls'] or 0)
            out['jit_by_stage'] = info['jit_by_stage']
            out['eager_by_stage'] = info['eager_by_stage']
            out['jit_by_name_all'] = cs1['by_name']
        out['info'] = {k: v for k, v in info.items() if k not in ('jit_by_stage', 'eager_by_stage')}
        out['counters_by_stage'] = ct.snapshot()
        out['counters_totals'] = ct.totals()
        out['libimf_ops'] = LI.counters()
        out['libimf_fused'] = FX.counters()
        out['qus_callbacks'] = dict(md.QUS)
        out['oadvt2_prepass_callbacks'] = dict(JO.XPRE_STATS)
        hp = cp.ph.cs.host
        out['pole_callbacks'] = dict(calls=hp.calls - p0[0], seconds=hp.seconds - p0[1], bytes=hp.bytes - p0[2])
        hf = getattr(cp.ph, '_handoff', None)
        if hf is not None:
            out['radiation_interface'] = {k: v for k, v in hf.log.totals().items() if rad_mode == 'server' or k != 'radiation'}
            out['radiation_sentence'] = hf.log.sentence() if rad_mode == 'server' else H.SENTENCE_RADIATION_REPLAY
        out['stage_table'] = sr.table()
        out['non_jax_text'] = sr.non_jax_text()
        out['groups_bytes'] = None
        if guard:
            pass
        return out, new, arrays, rec, dev, sr
    finally:
        ct.stop_listening()


def validate(cp, date, it0, out, new, arrays, rec, refdir, outdir, label):
    """C1 / C2 / mask checks; arrays from a keep=True run."""
    R = rec['R']
    res = {}
    allarr = {}
    for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter', 'X'):
        for k, v in snap_np(arrays[stg]).items():
            allarr[f'{stg}/{k}'] = v
    SS2 = C.tree_np(arrays['SS2'])
    V2n = C.tree_np(arrays['V2'])
    for grp in ('ocean', 'ice', 'lake', 'li', 'atm'):
        allarr.update(K.flatten(SS2[grp], f'surf/{grp}/'))
    allarr.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy'], usi=V2n['usi'], vsi=V2n['vsi']), 'surf/v2/'))
    allarr.update(K.flatten(C.tree_np(arrays['aux_land']), 'land/'))
    np.savez(os.path.join(outdir, f'{date}_{label}.npz'), **allarr)
    res['n_arrays'] = len(allarr)
    # ---- C1
    if refdir:
        ref = H.load_npz(os.path.join(refdir, f'{date}_ref.npz'))
        rows = AN.compare(allarr, ref)
        res['c1'] = rows
        res['c1_total'] = {c: sum(r['categories'][c] for r in rows.values()) for c in 'ABCD'}
        res['c1_n'] = sum(r['n_fields'] for r in rows.values())
        k0 = 'filter/T'
        pert = ref[k0].copy()
        pert.flat[0] = np.nextafter(pert.flat[0], np.inf)
        res['non_vacuity_1ulp_detected_as'] = H.field_category(pert, ref[k0])['cat']
    # ---- C2 atmosphere
    end = K.numeric_state(snap_np(arrays['filter']))
    refe = {k: np.asarray(v) for k, v in A.end_reference(R).items() if isinstance(v, (np.ndarray, np.generic))}
    gate = H.compare_end_state(end, refe)
    end30 = {f: K.slim(H.field_category(end[f], refe[f])) for f in A.END_FIELDS if f in end and f in refe}
    res['atm_gate'] = dict(verdict=gate['verdict'], categories=gate['categories'], gate_fields={f: K.slim(v) for f, v in gate['gate_fields'].items()},
                           exception_columns=gate['exception_columns'])
    res['atm_end30'] = dict(categories={c: sum(1 for v in end30.values() if v['cat'] == c) for c in 'ABCD'}, fields=end30, n_compared=len(end30),
                            missing=[f for f in A.END_FIELDS if f not in end or f not in refe])
    # ---- C2 surface
    post = C.tree_np(arrays['post'])

    class Shim:
        pass
    s = Shim()
    s.st = cp.st
    s.SS = {g: SS2[g] for g in ('ocean', 'ice', 'lake', 'li', 'atm')}
    s.last = dict(post=dict(ice_pre_adv=post['ice_pre_adv'], dyn=post['dyn'], flowo=post['flowo'], eflowo=post['eflowo']))
    t0 = time.perf_counter()
    res['surface_c2'] = K.surface_c2(date, it0, s)
    res['surface_c2_categories'] = K.summarize_surface_c2(res['surface_c2'])
    res['seconds_surface_c2_compare'] = time.perf_counter() - t0
    # ---- masks and misc
    m = np.asarray(arrays['masks'])
    res['tile_mask_check'] = dict(mismatch_ocean_type=int(m[0]), mismatch_ice_type=int(m[1]), mask_ocean_tiles=int(m[2]), mask_ice_tiles=int(m[3]),
                                  template_ocean_slots=int(m[4]), template_ice_slots=int(m[5]))
    res['n_slp_exp_branch_cells'] = int(np.asarray(arrays['n_slp_exp']))
    res['host_template'] = out['info']['host']
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date')
    ap.add_argument('outdir')
    ap.add_argument('refdir')
    ap.add_argument('--server', action='store_true')
    ap.add_argument('--runs', type=int, default=3)
    a = ap.parse_args()
    date, outdir = a.date, a.outdir
    os.makedirs(outdir, exist_ok=True)
    assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
    assert len(os.sched_getaffinity(0)) >= 2, 'host callbacks inside jit need >= 2 cores'
    it0 = dict(A.DATES)[date]
    hdr = H.provenance_header(extra=dict(task='D191 assembled device-resident coupled step, step 0', date=date, libimf=LI.status(), installed=C.INST), require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    io = K.IOAudit()
    t_all = time.perf_counter()
    with io.on('setup'):
        cp = C.Coupled(date)
    print(f'setup {cp.setup_seconds:.1f} s', flush=True)
    runs = []
    registry_cold = None
    labels = [('cold', False, True), ('steady', False, False), ('timed', True, False)][:a.runs]
    val = None
    first_end = None
    for lab, timed, keep in labels:
        with io.on('step'):
            out, new, arrays, rec, dev, sr = run_once(cp, lab, timed, keep)
        if lab == 'cold':
            registry_cold = cp.ph.reg.emit()
        if keep:
            val = validate(cp, date, it0, out, new, arrays, rec, a.refdir, outdir, 'asm')
            out['validation'] = val
            first_end = {k: np.asarray(v) for k, v in new['S'].items() if hasattr(v, 'shape')}
        else:
            same = {k: bool(np.array_equal(np.asarray(v), first_end[k])) for k, v in new['S'].items() if hasattr(v, 'shape') and k in first_end}
            out['end_state_bitwise_equal_to_cold_run'] = dict(n_fields=len(same), n_equal=sum(same.values()), unequal=[k for k, v in same.items() if not v])
        runs.append(out)
        line = f'[{lab}] step {out["step_seconds"]:.1f} s (load {out["record_load_seconds"]:.1f}) jit {out["step_jit_executions"]} eager {out["step_eager_primitive_dispatches"]}'
        if keep:
            line += f' | gate {val["atm_gate"]["verdict"]} {val["atm_gate"]["categories"]} end30 {val["atm_end30"]["categories"]} | C1 {val.get("c1_total")}'
        print(line, flush=True)
        json.dump(dict(header=hdr, date=date, runs=runs), open(os.path.join(outdir, f'{date}_asm.json'), 'w'), indent=1, default=str)
    # ---- real radiation server (nov26 only)
    server_run = None
    if a.server:
        assert date == 'nov26'
        import radiation_server as rs
        import radiation_server_persist as RP
        lv = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it0}_in.bin')
        cp.ph.reg.read('live_radiation_packet', lambda: lv, stage='record_load')
        with RP.PersistentServer('nov26') as srv:
            cp.ph.rad_mode, cp.ph.server, cp.ph.live_packet = 'server', srv, lv
            for att in ('_handoff', '_hstep'):
                if hasattr(cp.ph, att):
                    delattr(cp.ph, att)
            with io.on('step'):
                out, new, arrays, rec, dev, sr = run_once(cp, 'server', False, True, rad_mode='server', server=srv, lv=lv)
            out['server'] = dict(start_wall_s=getattr(srv, 'start_wall', None), binary=getattr(srv, 'binary', None), rundir=getattr(srv, 'rundir', None))
            out['validation'] = validate(cp, date, it0, out, new, arrays, rec, a.refdir, outdir, 'asm_server')
            out_srv = cp.ph._handoff.last_out
            rv = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it0}_out.bin')
            out['server_outputs_vs_recorded_server_output'] = {k: K.slim(H.field_category(np.asarray(out_srv[k]), rv[k])) for k in ('SRHR', 'TRHR', 'COSZ1', 'T', 'Q', 'CLDSS', 'CLDMC')}
            fe = {k: np.asarray(v) for k, v in new['S'].items() if hasattr(v, 'shape')}
            out['server_vs_replay_end_state'] = {c: sum(1 for k in first_end if k in fe and first_end[k].dtype.kind == 'f' and H.field_category(fe[k], first_end[k])['cat'] == c) for c in 'ABCD'}
            server_run = out
            v = out['validation']
            print(f'[server] step {out["step_seconds"]:.1f} s gate {v["atm_gate"]["verdict"]} {v["atm_gate"]["categories"]} end30 {v["atm_end30"]["categories"]} C1(vs replay ref) {v.get("c1_total")}', flush=True)
        alive = subprocess.run(['pgrep', '-f', 'P2SAoM40.bin'], capture_output=True, text=True).stdout.split()
        server_run['model_processes_left_after_stop'] = [p for p in alive if p != str(os.getpid())]
    d = dict(header=hdr, date=date, itime=it0, libimf_install=C.INST, taskset=sorted(os.sched_getaffinity(0)),
             seconds=dict(setup=cp.setup_seconds, total_process=time.perf_counter() - t_all), runs=runs, server_run=server_run,
             recorded_inputs_registry=registry_cold, io_audit=io.report(),
             state_groups_bytes=None)
    json.dump(d, open(os.path.join(outdir, f'{date}_asm.json'), 'w'), indent=1, default=str)
    print('done', flush=True)


if __name__ == '__main__':
    main()
