"""D187 (stage S7): the BEST COUPLED STEP THAT EXISTS NOW, assembled from validated pieces, step 0 of a date.  A HYBRID step, NOT end-to-end JAX.

  device (JAX, D186)      MELT_SI, DYNAM (+QDYNAM, energy fix, TROP, PGRAD_PBL, KEA), CONDSE, RADIA apply -- device-resident arrays, with the libimf host callback
                          (d187_imf_p1: Intel libimf pow/exp for the same operations the NumPy imf chain routes through libimf)
  radiation               replay of the recorded SRHR/TRHR/COSZ1 through the D185 hand-off interface (all dates), and for nov26 ALSO the real persistent
                          Fortran server through the D185 io_callback ("radiation computed by the original Fortran (hybrid component)")
  host (existing code)    SURFACE (PBL, ATURB, tile fluxes, GHY land with recorded Ent exports, land ice), PRECIP_*, GROUND_*, RIVERF, DYNSI, ocean step, FORM_SI,
                          ADVSI (surface_loop_v2.Loop2), then DISSIP and FILTER (NumPy, atm_step)
The state crosses to the host after phase 1 (device -> NumPy) and stays on the host for the rest of the step.

Run (cores 0-2 only; host callbacks inside jit need >= 2 cores):
    taskset -c 0-2 env OMP_NUM_THREADS=1 python d187_coupled_step.py DATE OUTDIR [--server] [--runs 3]
Runs in one process: run 1 = cold (compile), run 2 = steady, run 3 = steady with a device block after every stage (stage shares), [run 4 = real radiation server,
nov26 only].  Writes OUTDIR/<date>_hyb.npz (state of run 1: stage snapshots, X, surface state), <date>_hyb_server.npz, OUTDIR/<date>_hyb.json.
Comparison with the NumPy reference: d187_analyze.py."""
import clouds_jax_env  # noqa: F401  (XLA flags BEFORE jax)
import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

import d187_imf_p1 as W        # imports jax_atm_phase1 (installs the execution counters) and wires libimf into the D186 modules
INST = W.install()
import jax
import jax.numpy as jnp
import jax_atm_phase1 as P1
import jax_harness as H
import jax_p1_count as CNT
import jax_radiation as JR
import libimf_ops as LI
import atm_step as A
import surface_loop as L
import surface_loop_v2 as V2
import riverf_ff as RF
import surface_loop_advsi as AL
import d187_common as K

KINDS_RAD = {'replay': 'JJ/REC', 'server': 'JJ/FORT'}


def build_registry(rad_mode):
    sr = H.StageRegistry()
    for n, (k, d) in P1.STAGE_TABLE.items():
        if n == 'radia':
            k = KINDS_RAD[rad_mode]
        sr.register(n, k, d + ('' if n not in ('dyn', 'condse_setup', 'condse_mstcnv', 'condse_post') else
                               ' [libimf pow/exp: labelled host callback inside this stage, time included]'))
    sr.register('handoff_to_host', 'NP', 'device -> host copy of the phase-1 state (np.array of every device array) and of the MELT_SI result; the state stays on the host from here')
    sr.register('surface_pre', 'EJ/NP', 'PRECIP_SI, PRECIP_OC (+TOC2SST), PRECIP_LI, PRECIP_LK, seaice_to_atmgrid (surface_loop.surface_pre; NumPy dict glue, jax-importing modules)')
    sr.register('surface_records', 'REC/NP', 'read of the recorded SURFACE templates ffp/ffs/ffl/ffg/fft and rewriting of their state columns from OUR surface state (apply_state_to_records)')
    sr.register('surface_tiles_land', 'EJ/NP/REC', 'PBL, ATURB, tile fluxes, aggregation, GHY land (batched JAX kernels called eagerly from NumPy glue), land ice, 2 substeps; '
                'Ent exports, land forcing columns and tile radiation columns are RECORDED inputs')
    sr.register('surface_post_glue', 'NP', 'rest of surface_post_v2 (state copies, flux dict assembly, pole replicate)')
    sr.register('ground_li', 'NP', 'GROUND_LI')
    sr.register('underice_ground_si', 'NP', 'UNDERICE + GROUND_SI (lake and ocean domains)')
    sr.register('ground_lk', 'NP', 'GROUND_LK')
    sr.register('riverf', 'NP', 'RIVERF (NumPy)')
    sr.register('form_si', 'NP', 'FORM_SI')
    sr.register('dynsi', 'NP', 'DYNSI input assembly + VPICEDYN (NumPy batch)')
    sr.register('ocean_step', 'EJ/NP', 'ocean step stages after PRECIP (ocean_step.py stage functions: NumPy dict glue around JAX kernels)')
    sr.register('toc2sst', 'NP', 'TOC2SST')
    sr.register('advsi', 'NP', 'ADVSI (NumPy + mpmath binary128 Ti2b)')
    sr.register('dissip', 'NP', 'DISSIP (NumPy)')
    sr.register('filter', 'NP', 'FILTER (SLP filter, MAtoPMB; NumPy, libimf pow through ctypes)')
    return sr


@contextlib.contextmanager
def cnt_phase(name, store):
    """jit executions and eager primitive dispatches made while the block runs (differences of the global execution counters)."""
    a = CNT.snapshot()
    try:
        yield
    finally:
        b = CNT.snapshot()
        store[name] = dict(jit_executions=b['jit_calls'] - a['jit_calls'],
                           eager_primitive_dispatches=None if a['eager_primitive_calls'] is None else b['eager_primitive_calls'] - a['eager_primitive_calls'])


@contextlib.contextmanager
def surface_timers(sr, loop, ocean_detail):
    """Wrap the functions of the existing surface half with timers (results untouched; restored on exit)."""
    saved = []

    def wrap(mod, attr, name):
        orig = getattr(mod, attr)

        def w(*a, **k):
            with sr.time(name):
                return orig(*a, **k)
        w.__wrapped__ = orig
        setattr(mod, attr, w)
        saved.append((mod, attr, orig))
    wrap(L, 'surface_pre', 'surface_pre')
    wrap(L, 'apply_state_to_records', 'surface_records')
    wrap(L, 'stage_surface_closed', 'surface_tiles_land')
    wrap(V2, 'surface_post_v2', 'surface_post_glue')
    wrap(L, 'ground_li', 'ground_li')
    wrap(L, 'underice', 'underice_ground_si')
    wrap(L, 'ground_si', 'underice_ground_si')
    wrap(L, 'ground_lk', 'ground_lk')
    wrap(RF, 'riverf', 'riverf')
    wrap(L, 'form_si', 'form_si')
    wrap(L, 'toc2sst', 'toc2sst')
    wrap(AL, 'advsi_on_state', 'advsi')
    orig_stages = L.ocean_stages_after_precip

    def timed_stages():
        out = []
        for name, fn, t in orig_stages():
            def f(oc, fx, ctx, _fn=fn, _n=name):
                t0 = time.perf_counter()
                with sr.time('ocean_step'):
                    r = _fn(oc, fx, ctx)
                ocean_detail[_n] = ocean_detail.get(_n, 0.0) + time.perf_counter() - t0
                return r
            out.append((name, f, t))
        return out
    L.ocean_stages_after_precip = timed_stages
    saved.append((L, 'ocean_stages_after_precip', orig_stages))

    class _Timed:
        def __init__(self, obj):
            self.__dict__['_o'] = obj

        def __call__(self, *a, **k):
            with sr.time('dynsi'):
                return self._o(*a, **k)

        def __getattr__(self, n):
            return getattr(self._o, n)

        def __setattr__(self, n, v):
            setattr(self._o, n, v)
    orig_dyn = loop.V.dyn
    loop.V.dyn = _Timed(orig_dyn)
    try:
        yield
    finally:
        loop.V.dyn = orig_dyn
        for mod, attr, o in reversed(saved):
            setattr(mod, attr, o)


def host_state(r):
    return {k: np.array(v) for k, v in r['S'].items() if hasattr(v, 'shape')}


def snap(S):
    return {k: np.array(v, copy=True) for k, v in S.items() if not k.startswith('_') and isinstance(v, np.ndarray) and v.dtype.kind in 'fiub'}


def melt_to_loop(r, SS):
    """MELT_SI result of the DEVICE program -> the (ice, melt) pair the NumPy surface loop expects, cast to the dtypes of the NumPy melt_si; the NumPy melt_si is
    run on a copy as a check (equality is reported, not assumed)."""
    ice_n, melt_n = L.melt_si(copy.deepcopy(SS['ice']), SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], loop_geo)
    ice_d = {k: np.asarray(v) for k, v in r['ice'].items()}
    melt_d = {k: np.asarray(v) for k, v in r['melt'].items()}
    chk = dict(ice_keys_equal=sorted(ice_d) == sorted(ice_n), melt_keys_equal=sorted(melt_d) == sorted(melt_n))
    ice_c = {k: ice_d[k].astype(np.asarray(ice_n[k]).dtype, copy=False) for k in ice_n if k in ice_d}
    melt_c = {k: melt_d[k].astype(np.asarray(melt_n[k]).dtype, copy=False) if k in melt_n else melt_d[k] for k in melt_d}
    chk['ice_bitwise_equal_numpy_melt_si'] = {k: bool(np.array_equal(ice_c[k], ice_n[k])) for k in ice_n if k in ice_c}
    chk['melt_bitwise_equal_numpy_melt_si'] = {k: bool(np.array_equal(melt_c[k], melt_n[k])) for k in melt_n if k in melt_c}
    chk['all_equal'] = all(chk['ice_bitwise_equal_numpy_melt_si'].values()) and all(chk['melt_bitwise_equal_numpy_melt_si'].values()) \
        and chk['ice_keys_equal'] and chk['melt_keys_equal']
    return (ice_c, melt_c), chk


loop_geo = None


def one_run(ph, date, it0, st, SS0, rad_mode, timed, io, label, keep_arrays=False, lv=None):
    """One complete coupled step 0.  Returns a dict of results (+ arrays when keep_arrays)."""
    global loop_geo
    loop_geo = st['geo']
    sr = build_registry(rad_mode)
    ct = H.Counters()
    ct.listen_compiles()
    ocean_detail = {}
    SS = copy.deepcopy(SS0)
    loop = V2.Loop2(date, date, L.FF, st, SS, flags=None, it0=it0)
    ctx = ph.ctx
    out = dict(label=label, rad_mode=rad_mode, timed=timed)
    LI.reset_counters()
    CNT.reset()
    hp0 = ph.cs.host
    c_pole0, c_qus0 = ((hp0.calls, hp0.seconds, hp0.bytes) if hp0 is not None else (0, 0.0, 0)), dict(ph.qus)
    if getattr(ph, '_handoff', None) is not None:
        ph._handoff.log.reset()
    jc = {}
    arrays = {}
    t_wall0 = time.perf_counter()
    try:
        with ct.instrument_transfers(), surface_timers(sr, loop, ocean_detail):
            with ct.stage('coupled_step'):
                # ---------------- records + device_put
                with ct.stage('record_load'), sr.time('record_load'), io.on('record_load'), cnt_phase('record_load', jc):
                    rec = ph.load_records(it0, first=True)
                    dev = ph.to_device(rec, first=True)
                    jax.block_until_ready(dev)
                hold = ph.new_hold(dev['rad'])
                if rad_mode == 'server':
                    for k in ('RQT', 'KLIQ', 'SNOAGE'):
                        hold[k] = jnp.asarray(lv[k])
                # ---------------- phase 1 on the device
                with ct.stage('phase1'), cnt_phase('phase1', jc):
                    t0 = time.perf_counter()
                    r = ph.step(it0, dev, dev['S'], {}, P1.CS.ms_zero(), hold, dev['ice'], timed=timed, rec=rec)
                    jax.block_until_ready((r['S'], r['X'], r['flags']))
                    with sr.time('flag_read'):
                        fl = P1.D.flags_to_host(r['flags'])
                    t_p1 = time.perf_counter() - t0
                for n, s in r['info']['stage_seconds'].items():
                    if timed and n in ('melt_si', 'dyn', 'condse_entry', 'condse_setup', 'condse_mstcnv', 'condse_post', 'radia'):
                        sr.add_seconds(n, s)
                out['phase1_seconds'] = t_p1
                out['phase1_stage_seconds_dispatch_or_timed'] = r['info']['stage_seconds']
                out['phase1_stage_jit_executions'] = r['info']['stage_jit_calls']
                out['phase1_stage_eager_primitives'] = r['info']['stage_eager_prims']
                out['flags'] = fl
                # ---------------- hand-over to the host
                with ct.stage('handoff_to_host'), sr.time('handoff_to_host'), cnt_phase('handoff_to_host', jc):
                    S = host_state(r)
                    A._native(S)
                    loop.melt, melt_chk = melt_to_loop(r, SS0)
                    snaps = dict(dyn={k: np.asarray(v) for k, v in r['snaps']['dyn'].items() if hasattr(v, 'shape')} if keep_arrays else None)
                    if keep_arrays:
                        snaps['condse'] = {k: np.asarray(v) for k, v in r['snaps']['condse'].items() if hasattr(v, 'shape')}
                        snaps['radia'] = {k: np.asarray(v) for k, v in r['snaps']['radia'].items() if hasattr(v, 'shape')}
                        snaps['X'] = {k: np.asarray(v) for k, v in r['X'].items() if hasattr(v, 'shape')}
                out['melt_si_device_vs_numpy'] = melt_chk
                out['handoff_bytes'] = int(sum(v.nbytes for v in S.values() if isinstance(v, np.ndarray)))
                out['handoff_arrays'] = int(sum(1 for v in S.values() if isinstance(v, np.ndarray)))
                R = rec['R']
                # ---------------- host surface half (existing code)
                orig_sr = A.surface_records
                A.surface_records = ph.reg.guard_function('surface_records', orig_sr, stage='surface')
                tm = {}
                try:
                    with ct.stage('surface_half'), cnt_phase('surface_half', jc), io.on('surface_half'):
                        t0 = time.perf_counter()
                        loop.stage_surface(S, R, ctx, tm=tm, land_mode='ghy')
                        t_surf = time.perf_counter() - t0
                finally:
                    A.surface_records = orig_sr
                if keep_arrays:
                    snaps['surface'] = snap(S)
                with ct.stage('dissip'), sr.time('dissip'), cnt_phase('dissip', jc):
                    A.stage_dissip(S, ctx)
                if keep_arrays:
                    snaps['dissip'] = snap(S)
                with ct.stage('filter'), sr.time('filter'), cnt_phase('filter', jc):
                    A.stage_filter(S, ctx)
                A._native(S)
                if keep_arrays:
                    snaps['filter'] = snap(S)
        out['wall_seconds'] = time.perf_counter() - t_wall0
        out['surface_half_seconds'] = t_surf
        out['surface_tm'] = tm
        out['ocean_detail_seconds'] = ocean_detail
        # ---------------- counters
        cn = ct.snapshot()
        out['counters_by_phase'] = cn
        out['counters_totals'] = ct.totals()
        cs = CNT.snapshot()
        out['jit_executions_total'] = cs['jit_calls']
        out['eager_primitive_dispatches_total'] = cs['eager_primitive_calls']
        out['jit_and_eager_by_phase'] = jc
        out['libimf_callbacks'] = LI.counters()
        hp = ph.cs.host
        out['pole_callbacks'] = dict(calls=hp.calls - c_pole0[0], seconds=hp.seconds - c_pole0[1], bytes=hp.bytes - c_pole0[2])
        out['qus_callbacks'] = {k: ph.qus[k] - c_qus0[k] for k in ph.qus}
        hf = getattr(ph, '_handoff', None)
        if hf is not None:
            out['radiation_interface'] = {k: v for k, v in hf.log.totals().items() if rad_mode == 'server' or k != 'radiation'}
            out['radiation_sentence'] = hf.log.sentence() if rad_mode == 'server' else H.SENTENCE_RADIATION_REPLAY
        out['mstcnv_stats'] = dict(ph.cs_stats)
        out['stage_table'] = sr.table()
        out['non_jax_text'] = sr.non_jax_text()
        # ---------------- end state vs the real records (C2)
        end = K.numeric_state(S)
        ref = {k: np.asarray(v) for k, v in A.end_reference(R).items() if isinstance(v, (np.ndarray, np.generic))}
        gate = H.compare_end_state(end, ref)
        end30 = {f: K.slim(H.field_category(end[f], ref[f])) for f in A.END_FIELDS if f in end and f in ref}
        out['atm_gate'] = dict(verdict=gate['verdict'], categories=gate['categories'], gate_fields={f: K.slim(v) for f, v in gate['gate_fields'].items()},
                               exception_columns=gate['exception_columns'])
        out['atm_end30'] = dict(categories={c: sum(1 for v in end30.values() if v['cat'] == c) for c in 'ABCD'}, fields=end30,
                                n_compared=len(end30), missing=[f for f in A.END_FIELDS if f not in end or f not in ref])
        if keep_arrays or label.startswith('server'):
            t0 = time.perf_counter()
            out['surface_c2'] = K.surface_c2(date, it0, loop)
            out['surface_c2_categories'] = K.summarize_surface_c2(out['surface_c2'])
            out['seconds_surface_c2_compare'] = time.perf_counter() - t0
        arrays = dict(end=end)
        if keep_arrays:
            arrs = {}
            for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter', 'X'):
                for k, v in snaps[stg].items():
                    v = np.asarray(v)
                    if v.dtype.kind in 'fiub':
                        arrs[f'{stg}/{k}'] = v
            arrs.update(K.surface_snapshot(loop))
            arrays['all'] = arrs
        return out, arrays, loop
    finally:
        ct.stop_listening()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date')
    ap.add_argument('outdir')
    ap.add_argument('--server', action='store_true', help='nov26 only: add a run with the real persistent radiation server')
    ap.add_argument('--runs', type=int, default=3)
    a = ap.parse_args()
    date, outdir = a.date, a.outdir
    os.makedirs(outdir, exist_ok=True)
    assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
    assert len(os.sched_getaffinity(0)) >= 2, 'host callbacks inside jit need >= 2 cores'
    it0 = dict(A.DATES)[date]
    hdr = H.provenance_header(extra=dict(task='D187 coupled step (hybrid), step 0', date=date, libimf=LI.status(), installed=INST), require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    io = K.IOAudit()
    t_all = time.perf_counter()
    # ---------------- set-up (everything that is not the step)
    with io.on('setup'):
        t0 = time.perf_counter()
        ph = W.make_phase1(date, rad='replay')
        t_ph = time.perf_counter() - t0
        H.declare_d174_surface_items(ph.reg, date)
        ph.reg.declare('surface_init', f'ff_data/{date}: ffc_cse_in (FLAKE...), ffp (coriolis), ffl2 (lake depth), ffo_geom.bin, ffo_state_<it0> tag 0 (MMST), '
                       f'restart nc (ocean, ice, lake, land ice, ADVSI rsix/rsiy), advsi_dumps/{date}/ffadv_in_<it0> (ADVSI geometry)',
                       description='statics and initial state of the surface half (surface_loop.load_statics, init_surface_state, make_ocean_ctx, V2State)',
                       origin='D187; D170 section 6; D174 items mmst, advsi_geometry, straits_start')
    with io.on('surface_init'):
        t0 = time.perf_counter()
        st = L.load_statics(date)
        st['ctx'] = L.make_ocean_ctx(date)
        SS0 = L.init_surface_state(date, st=st)
        V2.V2State(st, date, L.FF, it0)                          # audit of the files the Loop2 constructor reads (ADVSI geometry dump, restart rsix/rsiy, DYNSI/RIVERF tables)
        t_surf_init = time.perf_counter() - t0
    restart_path = f'{L.FF}/_pristine_restarts/{L.RESTART[date]}'
    nb = sum(os.path.getsize(p) for p, tag in io.files.items() if tag == 'surface_init') + os.path.getsize(restart_path)
    ph.reg.read('surface_init', None, stage='setup', nbytes=nb)
    runs, arrays_first, loops = [], None, []
    rad_mode_ph = 'replay'
    labels = [('cold', False), ('steady', False), ('timed', True)][:a.runs]
    keep_first = True
    for lab, timed in labels:
        res, arrs, loop = one_run(ph, date, it0, st, SS0, 'replay', timed, io, lab, keep_arrays=keep_first)
        runs.append(res)
        if lab == 'cold':
            registry_cold = ph.reg.emit()
        if keep_first:
            first_end = arrs['end']
            np.savez(os.path.join(outdir, f'{date}_hyb.npz'), **arrs['all'])
            keep_first = False
        else:
            # in-process repeatability: the end state of this run vs run 1 (bitwise)
            same = {k: bool(np.array_equal(arrs['end'][k], first_end[k])) for k in first_end if k in arrs['end']}
            res['end_state_bitwise_equal_to_cold_run'] = dict(n_fields=len(same), n_equal=sum(same.values()), unequal=[k for k, v in same.items() if not v])
        print(f'[{lab}] wall {res["wall_seconds"]:.1f} s phase1 {res["phase1_seconds"]:.1f} s surface {res["surface_half_seconds"]:.1f} s jit {res["jit_executions_total"]} '
              f'gate {res["atm_gate"]["verdict"]} {res["atm_gate"]["categories"]} end30 {res["atm_end30"]["categories"]}', flush=True)
        json.dump(dict(header=hdr, date=date, runs=runs), open(os.path.join(outdir, f'{date}_hyb.json'), 'w'), indent=1, default=str)
    # ---------------- recorded-input registry (after the replay runs)
    reg = registry_cold
    # ---------------- real radiation server run (nov26 only)
    server_run = None
    if a.server:
        assert date == 'nov26'
        import radiation_server as rs
        import radiation_server_persist as RP
        lv = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it0}_in.bin')
        ph.reg.read('live_radiation_packet', lambda: lv, stage='record_load')
        with RP.PersistentServer('nov26') as srv:
            ph.rad_mode, ph.server, ph.live_packet = 'server', srv, lv
            if hasattr(ph, '_handoff'):
                del ph._handoff
            if hasattr(ph, '_hstep'):
                del ph._hstep
            res, arrs, loop = one_run(ph, date, it0, st, SS0, 'server', False, io, 'server', keep_arrays=True, lv=lv)
            res['server'] = dict(start_wall_s=getattr(srv, 'start_wall', None), binary=getattr(srv, 'binary', None), rundir=getattr(srv, 'rundir', None))
            np.savez(os.path.join(outdir, f'{date}_hyb_server.npz'), **arrs['all'])
            # compare server-run end state with the replay-run end state (isolates server vs record)
            same = {k: H.field_category(arrs['end'][k], first_end[k]) for k in first_end if k in arrs['end'] and first_end[k].dtype.kind == 'f'}
            res['server_vs_replay_end_state'] = dict(categories={c: sum(1 for v in same.values() if v['cat'] == c) for c in 'ABCD'},
                                                     worst={k: K.slim(v) for k, v in sorted(same.items(), key=lambda kv: -kv[1]['rel'])[:8]})
            # radiation outputs of the server vs the recorded real-server output
            out_srv = ph._handoff.last_out
            rv = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it0}_out.bin')
            res['server_outputs_vs_recorded_server_output'] = {k: K.slim(H.field_category(np.asarray(out_srv[k]), rv[k])) for k in ('SRHR', 'TRHR', 'COSZ1', 'T', 'Q', 'CLDSS', 'CLDMC')}
            server_run = res
            print(f'[server] wall {res["wall_seconds"]:.1f} s gate {res["atm_gate"]["verdict"]} {res["atm_gate"]["categories"]} end30 {res["atm_end30"]["categories"]}', flush=True)
        alive = subprocess.run(['pgrep', '-f', 'P2SAoM40.bin'], capture_output=True, text=True).stdout.split()
        server_run['model_processes_left_after_stop'] = [p for p in alive if p != str(os.getpid())]
    io_rows = io.report()
    out = dict(header=hdr, date=date, itime=it0, libimf_install=INST, taskset=sorted(os.sched_getaffinity(0)),
               seconds=dict(phase1_object_build=t_ph, surface_init=t_surf_init, total_process=time.perf_counter() - t_all),
               runs=runs, server_run=server_run, recorded_inputs_registry=reg, io_audit=io_rows,
               ocean_stage_names=list(runs[0].get('ocean_detail_seconds', {}).keys()))
    json.dump(out, open(os.path.join(outdir, f'{date}_hyb.json'), 'w'), indent=1, default=str)
    print('done', flush=True)


if __name__ == '__main__':
    main()
