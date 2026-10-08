"""D193 (S8, first attempt): ONE MODEL DAY (nov26, steps 33312..33365, 54 steps) with the assembled device-resident coupled step (jax_coupled.Coupled),
state device-resident between the steps, per-step end state kept in the layout of multiday_score.py, scored against the five real one-ulp members.
Recorded inputs (all listed in the JSON): per step the records of ff_data/nov26_day (CONDSE entry/exit ffc_cse_in/out, radiation ffa_step_<it>_r, SURFACE templates
ffp/ffs/ffl/ffg/fft; ffd_state/ffa_step_a of the FIRST step), the statics/restart of nov26; at the day boundary (step 48, it 33360) the DAILY_ATMDYN/ch4ox
increments of atm_day_open_loop (MDRYA from the real ffd_glue_daily dump, dH2O mass from ffa_step_<it-1>_e and ffa_step_<it>_a) and the recorded SNOAGE (the D157/D171
convention).  Radiation: replay of the recorded SRHR/TRHR (held between radiation steps) and COSZ1 of each step; NO Fortran callback in this result.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d193_day.py OUTDIR [--nsteps 54] [--c1dir DIR]
Writes OUTDIR/ours_d193/step_<it>.npz (T U V Q P QCL QCI), OUTDIR/d193_run.json (per step timings, counters), and with --c1dir the full per-step arrays for the
C1 comparison (d193_ref_day.py reads them)."""
import clouds_jax_env  # noqa: F401
import argparse
import json
import os
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
import atm_day_open_loop as OL
import d191_run as R191
import d187_common as K

DATE = 'nov26'
DAYDIR = 'nov26_day'
NDAYSTEPS = 48
SAVE = ('T', 'U', 'V', 'Q', 'P', 'QCL', 'QCI')


class DayReal(A.Real):
    """A.Real whose files come from ff_data/nov26_day (the 54-step record set) for every step."""

    def __init__(self, date, itime, ff=A.FF):
        super().__init__(DAYDIR, itime, ff)


def allarr_of(arrays):
    allarr = {}
    for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter', 'X'):
        for kk, v in R191.snap_np(arrays[stg]).items():
            allarr[f'{stg}/{kk}'] = v
    SS2, V2n = C.tree_np(arrays['SS2']), C.tree_np(arrays['V2'])
    for grp in ('ocean', 'ice', 'lake', 'li', 'atm'):
        allarr.update(K.flatten(SS2[grp], f'surf/{grp}/'))
    allarr.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy'], usi=V2n['usi'], vsi=V2n['vsi']), 'surf/v2/'))
    allarr.update(K.flatten(C.tree_np(arrays['aux_land']), 'land/'))
    return allarr


def daily_on_state(state, dev, ctx_imf, mdrya, it):
    """Day boundary (it % 48 == 0): DAILY_ATMDYN + ch4ox on the end state of the previous step, the SNOAGE reset to the recorded value, exactly as
    atm_day_open_loop.run_day.  ONE declared host round trip of 9 atmosphere arrays (MA MASUM Q in, MA MASUM PEDN PMID PK PDSIG P PEK Q out)."""
    S = state['S']
    h = {k: np.array(np.asarray(S[k]), copy=True) for k in ('MA', 'MASUM', 'Q')}
    rec_daily = OL.apply_daily(h, ctx_imf, mdrya)
    dm, spread = OL.ch4ox_increment(A.FF, DAYDIR, it)
    OL.apply_ch4ox(h, ctx_imf, dm)
    S2 = dict(S)
    for k in ('MA', 'MASUM', 'PEDN', 'PMID', 'PK', 'PDSIG', 'P', 'PEK', 'Q'):
        S2[k] = jnp.asarray(h[k])
    carry = dict(state['carry'])
    if 'SNOAGE' in carry:
        new = jnp.asarray(np.asarray(dev['cse_in']['SNOAGE']))
        rec_daily['snoage_change_recorded'] = float(np.abs(np.asarray(new) - np.asarray(carry['SNOAGE'])).max())
        carry['SNOAGE'] = new
    rec_daily['ch4ox_dm_i_spread_rel'] = spread
    out = dict(state)
    out['S'], out['carry'] = S2, carry
    return out, rec_daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--nsteps', type=int, default=NDAYSTEPS + 6)
    ap.add_argument('--c1dir', default=None, help='save the full per-step arrays here (for the C1 comparison against d193_ref_day.py)')
    ap.add_argument('--timed', type=int, default=1, help='block the device after every stage (per-stage shares; adds host syncs)')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ours = os.path.join(a.outdir, 'ours_d193')
    os.makedirs(ours, exist_ok=True)
    if a.c1dir:
        os.makedirs(a.c1dir, exist_ok=True)
    assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
    assert len(os.sched_getaffinity(0)) >= 2, 'host callbacks inside jit need >= 2 cores'
    A.Real = DayReal                                    # records of every step from nov26_day (patched before Coupled/Phase1 call it)
    it0 = dict(A.DATES)[DATE]
    hdr = H.provenance_header(extra=dict(task='D193 one model day, assembled coupled step', date=DATE, libimf=LI.status(), installed=C.INST), require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    cp = C.Coupled(DATE)
    print(f'setup {cp.setup_seconds:.1f} s', flush=True)
    mdrya, deltam_real, it_daily = OL.daily_mdrya(A.FF, DATE)
    sr = R191.build_registry('replay')
    ct = H.Counters()
    ct.listen_compiles()
    LI.reset_counters()
    FX.reset_counters()
    md.reset_qus()
    CNT.reset()
    res = dict(date=DATE, it0=it0, nsteps_requested=a.nsteps, header=hdr, taskset=sorted(os.sched_getaffinity(0)), steps=[], mdrya=mdrya, completed=0,
               error=None, radiation='replay of the recorded SRHR/TRHR (held between radiation steps) and COSZ1; no Fortran callback',
               record_dir=f'{A.FF}/{DAYDIR}')
    ctx_imf = None
    state, rec, dev = None, None, None
    t_run = time.perf_counter()
    compiles_prev = 0
    cs_prev = CNT.snapshot()
    for k in range(a.nsteps):
        it = it0 + k
        try:
            t0 = time.perf_counter()
            with ct.stage('record_load'), sr.time('record_load'):
                if k == 0:
                    state, rec, dev = cp.initial_state()
                else:
                    rec = cp.ph.load_records(it, first=False)
                    dev = cp.ph.to_device(rec, first=False)
                jax.block_until_ready((state, dev))
            t_load = time.perf_counter() - t0
            rec_daily = None
            if k > 0 and it % NDAYSTEPS == 0:
                if ctx_imf is None:
                    ctx_imf = A.make_ctx(DATE, imf=True)
                state, rec_daily = daily_on_state(state, dev, ctx_imf, mdrya, it)
                print(f'  daily (DAILY_ATMDYN + ch4ox) at it {it}: {rec_daily}', flush=True)
            t1 = time.perf_counter()
            with ct.stage('step'):
                state, info, arrays = cp.step(k, state, rec, dev, sr, timed=bool(a.timed), keep=True)
                jax.block_until_ready(state)
            wall = time.perf_counter() - t1
            cs1 = CNT.snapshot()
            tot = ct.totals()
            end = {f: np.asarray(arrays['filter'][f], dtype=np.float64) for f in SAVE}
            np.savez(os.path.join(ours, f'step_{it}.npz'), **end)
            if a.c1dir:
                np.savez(os.path.join(a.c1dir, f'{DATE}_day_step{k}.npz'), **allarr_of(arrays))
            m = np.asarray(arrays['masks'])
            row = dict(k=k, itime=it, wall=wall, record_load=t_load, radiation_step=bool(A.is_radiation_step(it)), flags=info['flags'],
                       flags_sum=int(sum(info['flags'].values())), stage_rebuilt=bool(info.get('stage_rebuilt', False)),
                       jit_executions=cs1['jit_calls'] - cs_prev['jit_calls'], eager_primitives=(cs1['eager_primitive_calls'] or 0) - (cs_prev['eager_primitive_calls'] or 0),
                       compiles_cum=tot['compiles'], compile_seconds_cum=tot['compile_seconds'], compiles_this_step=tot['compiles'] - compiles_prev,
                       tile_mask_mismatch=[int(m[0]), int(m[1])], masks_tiles=[int(m[2]), int(m[3])], host=info['host'], daily=rec_daily,
                       n_slp_exp_branch=int(np.asarray(arrays['n_slp_exp'])), jit_by_stage=info['jit_by_stage'])
            compiles_prev, cs_prev = tot['compiles'], cs1
            res['steps'].append(row)
            res['completed'] = k + 1
            print(f'step {k} it {it} wall {wall:.1f} s (load {t_load:.1f}) jit {row["jit_executions"]} compiles+{row["compiles_this_step"]} rebuilt {row["stage_rebuilt"]} '
                  f'flags {row["flags_sum"]} mask_mismatch {row["tile_mask_mismatch"]} rad {row["radiation_step"]}', flush=True)
        except Exception as e:                       # report how far the run could go; no silent substitute
            import traceback
            res['error'] = dict(step=k, itime=it, text=traceback.format_exc())
            print(f'STOP at step {k} it {it}: {e!r}', flush=True)
            traceback.print_exc()
            break
        finally:
            arrays = None
        res['total_wall'] = time.perf_counter() - t_run
        json.dump(res, open(os.path.join(a.outdir, 'd193_run.json'), 'w'), indent=1, default=str)
    res['total_wall'] = time.perf_counter() - t_run
    res['counters_totals'] = ct.totals()
    res['counters_by_stage'] = ct.snapshot()
    res['libimf_ops'] = LI.counters()
    res['libimf_fused'] = FX.counters()
    res['qus_callbacks'] = dict(md.QUS)
    res['oadvt2_prepass_callbacks'] = dict(JO.XPRE_STATS)
    hp = cp.ph.cs.host
    res['pole_callbacks'] = dict(calls=hp.calls, seconds=hp.seconds, bytes=hp.bytes)
    hf = getattr(cp.ph, '_handoff', None)
    if hf is not None:
        res['radiation_interface'] = {k: v for k, v in hf.log.totals().items() if k != 'radiation'}
    res['radiation_sentence'] = H.SENTENCE_RADIATION_REPLAY
    res['stage_table'] = sr.table()
    res['non_jax_text'] = sr.non_jax_text()
    res['recorded_inputs'] = cp.ph.reg.emit()
    ct.stop_listening()
    json.dump(res, open(os.path.join(a.outdir, 'd193_run.json'), 'w'), indent=1, default=str)
    print('done', res['completed'], 'steps', flush=True)


if __name__ == '__main__':
    main()
