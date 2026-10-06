"""D133/D134: validate atm_step_fast.py (batched CONDSE in the chained atmosphere step) and run it for several steps.

  --equiv : (D133) step 0 of each date: atm_step.run_step (per-column CONDSE) vs atm_step_fast.run_step (batched CONDSE), both from the real
            step-start state; per-field differences of the CONDSE-exit state and of the end-of-step state (bitwise count, max abs) and
            the wall time of each.  Reference side runs the per-column CONDSE (~2 min).
  --chain : (D134) all steps of each date; step k+1 starts from OUR end state of step k (atm_step_fast.run_chain).  After every stage the
            state is compared with the real boundary dump of that stage of the same step (atm_step_compare.stage_ref); the end state
            is the real start state of the next step.  Per step: F1 categories (A bitwise, B <=1e-12 of scale, C <=1e-6, D worse)
            per stage, the first stage with any inexact / beyond-B field, the gate fields' relative error (growth), and the flip
            analysis: columns beyond 1e-12 at the dynamics exit (inherited smooth drift) vs at the CONDSE exit (threshold flips
            appear as NEW columns), and the worst relative error inside the new columns vs outside them.
Usage (fullfidelity/, conda python):  python3 atm_step_fast_compare.py --equiv|--chain [--imf] [--dates nov26,dec01,jan01] [--steps 6]
                                      [--land recorded|ghy] [--json out.json]
"""
import argparse
import json
import sys
import time

import numpy as np

import atm_step as A
import atm_step_compare as C
import atm_step_fast as F

COLF = ['T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'U', 'V']
STAGES_CHAIN = ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter')


def diff_states(Sa, Sb, fields):
    """Per-field bitwise comparison of two state snapshots: (n differing cells, max abs, size)."""
    out = {}
    for k in fields:
        if k not in Sa or k not in Sb:
            continue
        a, b = np.asarray(Sa[k], float), np.asarray(Sb[k], float)
        ne = int((a != b).sum())
        out[k] = dict(n_diff=ne, size=int(a.size), max_abs=float(np.abs(a - b).max()) if ne else 0.0)
    return out


def equiv(date, it0, imf, land):
    ctx = A.make_ctx(date, imf=imf)
    R = A.Real(date, it0)
    t = time.perf_counter()
    F.ensure_backend(ctx)
    Sr, snr = A.run_step(date, it0, ctx, R=R, ms={}, land_mode=land)
    t_ref = time.perf_counter() - t
    R = A.Real(date, it0)
    t = time.perf_counter()
    tm = {}
    Sf, snf = F.run_step(date, it0, ctx, R=R, ms={}, land_mode=land, timing=tm)
    t_fast = time.perf_counter() - t
    res = dict(date=date, itime=it0, imf=imf, t_ref=t_ref, t_fast=t_fast, stage_timing={k: v for k, v in tm.items() if k.startswith('stage_')})
    res['condse'] = diff_states(snf['condse'], snr['condse'], C.CONDSE_FIELDS)
    res['end'] = diff_states(snf['filter'], snr['filter'], A.END_FIELDS)
    ref_end = A.end_reference(R)
    res['end_vs_real'] = {'ref_chain': C.summarize(A.compare_state(snr['filter'], ref_end, A.END_FIELDS)),
                          'fast_chain': C.summarize(A.compare_state(snf['filter'], ref_end, A.END_FIELDS))}
    res['gate_fast'] = C.gate_summary(A.compare_state(snf['filter'], ref_end, A.END_FIELDS))['verdict']
    res['gate_ref'] = C.gate_summary(A.compare_state(snr['filter'], ref_end, A.END_FIELDS))['verdict']
    return res


def print_equiv(r):
    mode = 'imf' if r['imf'] else 'libm'
    nd_c = {k: v for k, v in r['condse'].items() if v['n_diff']}
    nd_e = {k: v for k, v in r['end'].items() if v['n_diff']}
    print(f"== D133 {r['date']} step 0 [{mode}]: per-column chain {r['t_ref']:.1f} s, fast chain {r['t_fast']:.1f} s "
          f"(stages: {' '.join(f'{k[6:]}={v:.1f}' for k, v in r['stage_timing'].items())})")
    print(f"   CONDSE-exit fields {len(r['condse'])}: differing from per-column {len(nd_c)}; end-state fields {len(r['end'])}: differing {len(nd_e)}")
    for k, v in {**{'condse.' + a: b for a, b in nd_c.items()}, **{'end.' + a: b for a, b in nd_e.items()}}.items():
        print(f"      {k:18s} cells {v['n_diff']}/{v['size']} max abs {v['max_abs']:.3e}")
    print(f"   vs REAL end state categories: per-column {r['end_vs_real']['ref_chain']}  fast {r['end_vs_real']['fast_chain']}; "
          f"gate: per-column '{r['gate_ref']}' / fast '{r['gate_fast']}'")
    sys.stdout.flush()


def chain(date, it0, nsteps, imf, land):
    ctx = A.make_ctx(date, imf=imf)
    rows = []

    def on_step(k, it, R, sn, S, tm):
        row = dict(itime=it, stages={}, cols={})
        first_inexact = first_beyond_B = None
        for st in STAGES_CHAIN:
            ref = C.stage_ref(R, st)
            stats = A.compare_state(sn[st], ref, C.STAGE_FIELDS[st])
            row['stages'][st] = dict(cats=C.summarize(stats), fields={f: dict(cat=v['cat'], rel=v['rel'], max_abs=v['max_abs'], cols=v['cols_over'])
                                                                  for f, v in stats.items() if v['cat'] != 'A'})
            if first_inexact is None and any(v['cat'] != 'A' for v in stats.values()):
                first_inexact = st
            if first_beyond_B is None and any(v['cat'] in 'CD' for v in stats.values()):
                first_beyond_B = st
        row['first_inexact_stage'], row['first_beyond_B_stage'] = first_inexact, first_beyond_B
        # flip analysis on the primary fields
        mD = C.columns_over_bound(sn['dyn'], C.stage_ref(R, 'dyn'), COLF)
        mC = C.columns_over_bound(sn['condse'], C.stage_ref(R, 'condse'), COLF)
        new = mC & ~mD
        row['cols'] = dict(dyn_exit=int(mD.sum()), condse_exit=int(mC.sum()), new_at_condse=int(new.sum()),
                           end=int(C.columns_over_bound(sn['filter'], A.end_reference(R), COLF).sum()))
        # worst relative error inside / outside the columns newly beyond bound at CONDSE
        ref = C.stage_ref(R, 'condse')
        w_in = w_out = 0.0
        for f in COLF:
            a, b = np.asarray(sn['condse'][f], float), np.asarray(ref[f], float)
            sc = max(float(np.abs(b).max()), 1e-300)
            d = np.abs(a - b) / sc
            ax = A._ij_axes(d.shape)
            other = tuple(i for i in range(d.ndim) if i not in ax)
            dd = d.max(axis=other) if other else d
            dd = dd if ax[0] < ax[1] else dd.T
            if new.any():
                w_in = max(w_in, float(dd[new].max()))
            w_out = max(w_out, float(dd[~new].max()))
        row['condse_worst_rel_in_new_cols'], row['condse_worst_rel_outside_new_cols'] = w_in, w_out
        rel = {f: v['rel'] for f, v in A.compare_state(sn['filter'], A.end_reference(R), C.GATE_FIELDS).items()}
        row['end_gate_rel'] = rel
        row['end_cats'] = row['stages']['filter']['cats']
        row['timing'] = {a: b for a, b in tm.items() if a.startswith('stage_')}
        rows.append(row)
        print(f"-- {date} {it} (step {k}) [{'imf' if imf else 'libm'}] first inexact stage: {first_inexact}; first beyond-B stage: {first_beyond_B}")
        for st in STAGES_CHAIN:
            print(f"   {st:8s} {row['stages'][st]['cats']}")
        print(f"   columns>1e-12 (T,Q,QCL,QCI,TMOM,QMOM,U,V): dyn exit {row['cols']['dyn_exit']}, condse exit {row['cols']['condse_exit']} "
              f"(new at condse {row['cols']['new_at_condse']}), end of step {row['cols']['end']}; condse worst rel in new cols {w_in:.2e}, outside {w_out:.2e}")
        print("   end gate rel: " + " ".join(f"{f}={v:.1e}" for f, v in rel.items()) + "  | step wall " + f"{sum(row['timing'].values()):.1f} s")
        sys.stdout.flush()

    out = F.run_chain(date, it0, nsteps, ctx, land_mode=land, on_step=on_step)
    return dict(date=date, imf=imf, steps=rows, wall=[o['wall'] for o in out])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--equiv', action='store_true')
    ap.add_argument('--chain', action='store_true')
    ap.add_argument('--imf', action='store_true')
    ap.add_argument('--dates', default='nov26,dec01,jan01')
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--land', default='recorded')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    res = []
    for date, it0 in A.DATES:
        if date not in a.dates.split(','):
            continue
        if a.equiv:
            r = equiv(date, it0, a.imf, a.land)
            print_equiv(r)
            res.append(r)
        if a.chain:
            res.append(chain(date, it0, a.steps, a.imf, a.land))
        if a.json:
            json.dump(C.jsonable(res), open(a.json, 'w'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
