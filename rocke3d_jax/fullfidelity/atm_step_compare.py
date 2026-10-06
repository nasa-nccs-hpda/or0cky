"""D129: compare the chained atmosphere step (atm_step.py) with the real model, field by field.

For every date/step available (ff_data/<date>/, see atm_step.Real.available) and every stage the script reports

  chained   : the step started from the REAL state at the start of the step; after each stage the state is compared with the real
              boundary dump of that stage (dyn: ffd_state s3/s4; condse: ffc_cse_out; radia: ffa_step_r; surface: ffd post_surface +
              the ffa_step_e hidden state; dissip: ffa_step_d; filter / end of step: ffa_step_e = next step's start).
  isolated  : each stage started from the REAL state at its entry (start=stage, stop=stage) so that its own error is not mixed
              with the propagation of earlier ones.

and prints per field the category (A bitwise, B <= 1e-12 of the field scale, C <= 1e-6, D worse; thresholds fixed in advance, not
loosened), max abs, scale-relative max, rms, number of differing cells / columns beyond 1e-12 scale, and the worst cell.  The F1 gate
verdict is computed from the END state of step 0 of each date (see `verdict`).

Usage (from fullfidelity/, conda python):
  python3 atm_step_compare.py [--imf] [--dates nov26,dec01,jan01] [--steps 0,1,2,3,4,5] [--no-isolated] [--condse-real]
                              [--json out.json] [--timing]
--imf routes pow/exp through the Intel libimf (the real build's runtime; needed for bitwise dynamics/FILTER/CONDSE).
"""
import argparse
import json
import sys
import time

import numpy as np

import atm_step as A

DYN_FIELDS = ['U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'MASUM', 'TMOM', 'QMOM', 'MUS', 'MVS', 'MWS', 'GZ',
              'KEA', 'PTROPO', 'LTROPO', 'WSAVE', 'DPDX', 'DPDY', 'DPDX0', 'DPDY0', 'PHI']
CONDSE_FIELDS = list(A.CONDSE_OUT)
RADIA_FIELDS = ['T', 'Q', 'U', 'V']
SURFACE_FIELDS = ['U', 'V', 'T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA',
                  'U1AA', 'V1AA', 'USTARPBL', 'LMONINPBL', 'TSAVG', 'QSAVG']
DISSIP_FIELDS = ['T', 'U', 'V', 'Q']
END_FIELDS = list(A.END_FIELDS)
STAGE_FIELDS = dict(dyn=DYN_FIELDS, condse=CONDSE_FIELDS, radia=RADIA_FIELDS, surface=SURFACE_FIELDS, dissip=DISSIP_FIELDS,
                    filter=END_FIELDS)
# fields of the end state that the chain does not produce (not ported / not chained): listed, not compared
NOT_CHAINED = ('USAVG', 'VSAVG', 'TGVAVG', 'QGAVG')


def stage_ref(R, stage):
    if stage == 'dyn':
        r = {k.upper(): v for k, v in R.s3.items()}
        r.update({k.upper(): v for k, v in R.s4.items()})
        return r
    if stage == 'condse':
        return dict(R.cse_out)
    if stage == 'radia':
        return dict(R.r)
    if stage == 'surface':
        ps = R.post_surface()
        r = {k: ps[k] for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI')}
        r['TMOM'] = R.e['TMOM']
        r['QMOM'] = R.filt_in()['qmom']
        for k in ('UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'USTARPBL', 'LMONINPBL',
                  'TSAVG', 'QSAVG'):
            r[k] = R.e[k]
        return r
    if stage == 'dissip':
        return dict(R.d)
    return A.end_reference(R)


def summarize(stats):
    cats = {}
    for v in stats.values():
        cats[v['cat']] = cats.get(v['cat'], 0) + 1
    return cats


def show(label, stats, only_inexact=True, out=sys.stdout):
    cats = summarize(stats)
    out.write(f"  {label:22s} fields {len(stats):2d}  " + " ".join(f"{k}:{cats.get(k, 0)}" for k in 'ABCD') + "\n")
    if only_inexact:
        for k, v in stats.items():
            if v['cat'] != 'A':
                out.write(f"      {k:10s} {v['cat']}  max {v['max_abs']:.3e}  scale {v['scale']:.3e}  rel {v['rel']:.2e}  rms {v['rms']:.2e}  "
                          f"cells>{1e-12:.0e}s {v['n_over']}  cols>1e-12 {v['cols_over']} cols>1e-6 {v.get('cols_over6', 0)}  worst {v['worst']}\n")
    out.flush()


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


GATE_FIELDS = ['U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'TMOM', 'QMOM', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP']
LAND_MODES = ('ghy', 'recorded')


def _copy_state(S):
    return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in S.items()}


def _condse_inputs_exact(S_dyn, R):
    inp = A.condse_inputs(S_dyn, R)
    ref = R.cse_in
    ok = all(np.array_equal(inp[k], ref[k]) for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'PK', 'PMID', 'PEDN', 'PDSIG',
                                                      'PMIDOLD', 'GZ', 'MWS', 'PEK', 'UKMSP', 'VKMSP', 'UKMNP', 'VKMNP'))
    # UKM/VKM rows J=1 and J=JM are never read (the poles use UKMSP/UKMNP) and hold stale values scaled by .5 from the previous step
    return ok and all(np.array_equal(inp[k][:, :, :, 1:A.JM - 1], ref[k][:, :, :, 1:A.JM - 1]) for k in ('UKM', 'VKM'))


def run_step_compare(date, itime, ctx, ms_chain=None, ms_iso=None, isolated=True, condse='run', quiet=False,
                     land_modes=LAND_MODES):
    """One step.  res['common'] holds the stages shared by all land modes (dyn, condse, radia); res['runs'][land_mode] the
    surface/dissip/filter stages of that land mode (+ the isolated stage runs)."""
    import copy
    R = A.Real(date, itime)
    res = dict(date=date, itime=itime, imf=ctx.imf, condse=condse, common=dict(chained={}, isolated={}), runs={}, timing={})
    tm = {}
    ms_before = copy.deepcopy(ms_chain) if ms_chain is not None else None      # the isolated CONDSE run needs the same module-array carry
    t0 = time.perf_counter()
    S0, snaps = A.run_step(date, itime, ctx, start='dyn', stop='radia', R=R, ms=ms_chain, condse=condse, timing=tm)
    res['timing_common'] = dict(tm)
    if '_condse_counts' in S0:
        res['condse_counts'] = {k: int(v) for k, v in S0['_condse_counts'].items()}
    for st in ('dyn', 'condse', 'radia'):
        res['common']['chained'][st] = A.compare_state(snaps[st], stage_ref(R, st), STAGE_FIELDS[st])
        if not quiet:
            show(f"chained {st}", res['common']['chained'][st])
    if isolated:
        res['common']['isolated']['dyn'] = res['common']['chained']['dyn']
        if condse == 'run':
            S_dyn = A.init_state(R)
            A.stage_dyn(S_dyn, R, ctx)
            same = _condse_inputs_exact(S_dyn, R)
            res['common']['condse_inputs_bitwise_real'] = bool(same)
            if same:
                res['common']['isolated']['condse'] = res['common']['chained']['condse']     # identical inputs: identical run
                res['common']['isolated_condse_reused'] = True
            else:
                S1, sn1 = A.run_step(date, itime, ctx, start='condse', stop='condse', R=R, ms=ms_before if ms_before is not None else {}, condse='run')
                res['common']['isolated']['condse'] = A.compare_state(sn1['condse'], stage_ref(R, 'condse'), CONDSE_FIELDS)
            if not quiet:
                show('isolated condse', res['common']['isolated']['condse'])
        S1, sn1 = A.run_step(date, itime, ctx, start='radia', stop='radia', R=R, condse=condse)
        res['common']['isolated']['radia'] = A.compare_state(sn1['radia'], stage_ref(R, 'radia'), RADIA_FIELDS)
        if not quiet:
            show('isolated radia', res['common']['isolated']['radia'])
    for lm in land_modes:
        tm2 = {}
        S, sn = A.run_step(date, itime, ctx, start='surface', stop='filter', R=R, S=_copy_state(S0), land_mode=lm, timing=tm2)
        run = dict(chained={}, isolated={}, timing=dict(tm2))
        for st in ('surface', 'dissip', 'filter'):
            run['chained'][st] = A.compare_state(sn[st], stage_ref(R, st), STAGE_FIELDS[st])
            if not quiet:
                show(f"chained {st} [land {lm}]", run['chained'][st])
        if isolated:
            for st in ('surface', 'dissip', 'filter'):
                if st != 'surface' and lm != land_modes[0]:
                    run['isolated'][st] = res['runs'][land_modes[0]]['isolated'][st]      # no land dependence
                    continue
                S1, sn1 = A.run_step(date, itime, ctx, start=st, stop=st, R=R, land_mode=lm)
                run['isolated'][st] = A.compare_state(sn1[st], stage_ref(R, st), STAGE_FIELDS[st])
                if not quiet:
                    show(f"isolated {st} [land {lm}]", run['isolated'][st])
        res['runs'][lm] = run
    res['wall'] = time.perf_counter() - t0
    return res


def gate_summary(end_stats, max_exception_columns=10, exception_rel=1e-9):
    """Category of the gate fields of an end-of-step comparison and the verdict rule of scoping/ATM_STEP_PLAN.md section 5:
    MET = every gate field A or B; MET with named exceptions = the fields beyond B are confined to at most `max_exception_columns`
    horizontal columns each and below `exception_rel` of scale (the columns are listed); PARTLY MET = every gate field <= 1e-6 of
    scale; NOT MET otherwise.  The bounds are fixed here and are not tuned per run."""
    g = {k: end_stats[k] for k in GATE_FIELDS if k in end_stats}
    cats = summarize(g)
    worst = max(g.values(), key=lambda v: v['rel'])
    beyond = {k: dict(cat=x['cat'], rel=x['rel'], max_abs=x['max_abs'], cols=x['cols_over'], worst=x['worst'])
              for k, x in g.items() if x['cat'] not in 'AB'}
    if not beyond:
        v = 'MET'
    elif all(x['cols'] <= max_exception_columns and x['rel'] <= exception_rel for x in beyond.values()):
        v = 'MET with named exception columns'
    elif all(x['cat'] in 'ABC' for x in g.values()):
        v = 'PARTLY MET (all gate fields <= 1e-6 of scale; named fields beyond 1e-12)'
    else:
        v = 'NOT MET (gate fields beyond 1e-6 of scale: ' + ",".join(k for k, x in g.items() if x['cat'] == 'D') + ')'
    return dict(verdict=v, categories=cats, worst_field=max(g, key=lambda k: g[k]['rel']), worst_rel=worst['rel'], beyond_B=beyond)


def verdict(results):
    """F1 gate (FULL_FIDELITY_PLAN.md:48), from the END state of step 0 of each date, per land mode."""
    out = {}
    for r in results:
        if r['itime'] not in [it0 for _, it0 in A.DATES]:
            continue
        for lm, run in r['runs'].items():
            out.setdefault(lm, {})[r['date']] = gate_summary(run['chained']['filter'])
    return out


def columns_over_bound(S, ref, fields, bound=1e-12):
    """(IM,JM) boolean mask of columns where any of `fields` differs by more than bound * field scale."""
    m = np.zeros((A.IM, A.JM), bool)
    for f in fields:
        a, b = np.asarray(S[f], float), np.asarray(ref[f], float)
        over = np.abs(a - b) > bound * max(float(np.abs(b).max()), 1e-300)
        ax = A._ij_axes(over.shape)
        other = tuple(k for k in range(over.ndim) if k not in ax)
        o = over.any(axis=other) if other else over
        m |= o if ax[0] < ax[1] else o.T
    return m


def run_condse_only(date, itime, ctx, ms_c):
    """dyn -> CONDSE only: chained (inputs from our dynamics, module arrays carried from our previous step) and isolated (the REAL
    entry state and the same module-array carry).  Reports the fields and the number of columns beyond 1e-12 of the field scale."""
    import copy
    R = A.Real(date, itime)
    ms_before = copy.deepcopy(ms_c)
    t0 = time.perf_counter()
    S0, snaps = A.run_step(date, itime, ctx, start='dyn', stop='condse', R=R, ms=ms_c)
    ref = stage_ref(R, 'condse')
    out = dict(date=date, itime=itime, imf=ctx.imf, chained=A.compare_state(snaps['condse'], ref, CONDSE_FIELDS))
    colf = ['T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'U', 'V']
    out['chained_columns_over_bound'] = int(columns_over_bound(snaps['condse'], ref, colf).sum())
    S1, sn1 = A.run_step(date, itime, ctx, start='condse', stop='condse', R=R, ms=ms_before)
    out['isolated'] = A.compare_state(sn1['condse'], ref, CONDSE_FIELDS)
    out['isolated_columns_over_bound'] = int(columns_over_bound(sn1['condse'], ref, colf).sum())
    out['seconds'] = time.perf_counter() - t0
    return out


def time_stages(imf):
    """Warm single-thread-pool CPU timing of each stage, nov26 step 0 (surface: cold = first call incl. XLA compile, warm = second)."""
    date, it = 'nov26', 33312
    ctx = A.make_ctx(date, imf=imf)
    R = A.Real(date, it)
    tm = {}
    S, sn = A.run_step(date, it, ctx, R=R, ms={}, land_mode='recorded', timing=tm)
    cold = dict(tm)
    tm2 = {}
    S1 = A.init_state(R)
    A.stage_dyn(S1, R, ctx, tm2)
    t0 = time.perf_counter(); A.stage_dyn(A.init_state(R), R, ctx); dyn_warm = time.perf_counter() - t0
    S2, _ = A.run_step(date, it, ctx, start='surface', stop='filter', R=R, land_mode='recorded', timing=tm2)
    S3, _ = A.run_step(date, it, ctx, start='surface', stop='filter', R=R, land_mode='ghy', timing=tm2)
    tm3, tm4 = {}, {}
    A.run_step(date, it, ctx, start='surface', stop='filter', R=R, land_mode='recorded', timing=tm3)
    A.run_step(date, it, ctx, start='surface', stop='filter', R=R, land_mode='ghy', timing=tm4)
    print("first full chain (condse runs, surface cold incl. jit): " + " ".join(f"{k}={v:.2f}" for k, v in cold.items() if k.startswith('stage_')))
    print(f"dyn warm {dyn_warm:.2f}")
    print("warm surface+dissip+filter (land recorded): " + " ".join(f"{k}={v:.2f}" for k, v in tm3.items() if k.startswith('stage_')))
    print("warm surface+dissip+filter (land ghy):      " + " ".join(f"{k}={v:.2f}" for k, v in tm4.items() if k.startswith('stage_')))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--imf', action='store_true')
    ap.add_argument('--dates', default='nov26,dec01,jan01')
    ap.add_argument('--steps', default='0,1,2,3,4,5')
    ap.add_argument('--no-isolated', action='store_true')
    ap.add_argument('--condse-real', action='store_true', help='replay the real CONDSE exit instead of running the CONDSE port')
    ap.add_argument('--land', default='ghy,recorded', help="land modes: 'ghy' (our GHY port), 'recorded' (recorded GHY outputs)")
    ap.add_argument('--free', type=int, default=0, help='also run an N-step free-running chain from the first step of each date')
    ap.add_argument('--condse-only', action='store_true', help='only dyn -> CONDSE, chained and isolated, with column counts')
    ap.add_argument('--timing', action='store_true', help='warm CPU timing of the stages (nov26 step 0) and exit')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    if a.timing:
        return time_stages(a.imf)
    allres = []
    if a.condse_only:
        for date, it0 in A.DATES:
            if date not in a.dates.split(','):
                continue
            ctx = A.make_ctx(date, imf=a.imf)
            ms_c = {}
            for s in (int(x) for x in a.steps.split(',')):
                r = run_condse_only(date, it0 + s, ctx, ms_c)
                print(f"== {date} itime {it0 + s} [{'imf' if a.imf else 'libm'}] CONDSE only ({r['seconds']:.0f} s): columns beyond 1e-12: chained "
                      f"{r['chained_columns_over_bound']}/3170, isolated {r['isolated_columns_over_bound']}/3170")
                show('chained condse', r['chained'])
                show('isolated condse', r['isolated'])
                allres.append(r)
                if a.json:
                    json.dump(jsonable(allres), open(a.json, 'w'))
        return 0
    for date, it0 in A.DATES:
        if date not in a.dates.split(','):
            continue
        ctx = A.make_ctx(date, imf=a.imf)
        ms_c, ms_i = {}, {}
        for s in (int(x) for x in a.steps.split(',')):
            it = it0 + s
            if not A.have_dumps(date, it):
                print(f"{date} itime {it}: dumps missing, skipped")
                continue
            print(f"== {date} itime {it} [{'imf' if a.imf else 'libm'}]")
            sys.stdout.flush()
            r = run_step_compare(date, it, ctx, ms_c, ms_i, isolated=not a.no_isolated, condse='real' if a.condse_real else 'run',
                                 land_modes=tuple(a.land.split(',')))
            tc = r['timing_common']
            print("  timing s (common): " + " ".join(f"{k}={v:.1f}" for k, v in tc.items() if k.startswith('stage_')))
            for lm, run in r['runs'].items():
                print(f"  timing s (land {lm}): " + " ".join(f"{k}={v:.1f}" for k, v in run['timing'].items() if k.startswith('stage_')))
            allres.append(r)
            if a.json:
                json.dump(jsonable(allres), open(a.json, 'w'))
    if a.free:
        for date, it0 in A.DATES:
            if date not in a.dates.split(',') or not A.have_dumps(date, it0):
                continue
            ctx = A.make_ctx(date, imf=a.imf)
            print(f"== free-running chain {date} from {it0}, {a.free} steps, land {a.land.split(',')[-1]}")
            sys.stdout.flush()
            fr = A.run_free(date, it0, a.free, ctx, land_mode=a.land.split(',')[-1], condse='real' if a.condse_real else 'run',
                            log=lambda it, st: show(f"free step {it}", st, out=sys.stdout))
            allres.append(dict(date=date, itime=it0, free=fr, runs={}))
    v = verdict([r for r in allres if 'runs' in r and r['runs']])
    for lm, d in v.items():
        for date, x in d.items():
            print(f"F1 gate [{lm} land] {date}: {x['verdict']}; categories {x['categories']}; worst {x['worst_field']} rel {x['worst_rel']:.2e}")
            for k, y in x['beyond_B'].items():
                print(f"      {k}: {y}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
