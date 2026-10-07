"""D181: report of the fixed-layout tile masks (jax_state_d181) against the REAL row sets of the SURFACE records (ffp/ffs/ffl/ffg/fft).

  A. step 0 of nov26, dec01, jan01, with three sources of the ice fraction RSI:
       a  the RESTART RSI as it is (before MELT_SI)                           -> expected NOT to match (MELT_SI runs before SURFACE)
       b  OUR MELT_SI (surface_loop.melt_si) applied to the restart ice       -> computed from the restart + static fields only
       c  the RSI of the real ffc_cse_in dump of the step (after MELT_SI)     -> the rule itself with the real RSI
  B. the 54 steps of the nov26 day, RSI from the real ffc_cse_in of each step (rule check, independent of our own ice evolution), with the
     per-step change of the tile sets and the number of distinct tile row counts.
  C. optional: our own closed-loop ice carried by the driver (a ModelDriver checkpoint) -> MELT_SI -> mask vs the real record of that step.
Usage: python jax_state_d181_masks_report.py OUT.json [--ckpt driver_checkpoint.pkl]
Run with OMP_NUM_THREADS=1 and taskset (the module imports clouds_jax_env first)."""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_state_d181 as JS  # noqa: E402  (sets the XLA flags before jax)
import numpy as np  # noqa: E402

import atm_step as A  # noqa: E402
import surface_loop as L  # noqa: E402

DAY0, NSTEP = 33312, 54


def _geo(static):
    return dict(focean=static['focean'], flake=static['flake'], fwater=static['fwater'], is_ocean=static['is_ocean'], is_lake=static['is_lake'],
                valid=static['valid'])


def step0(dates=('nov26', 'dec01', 'jan01')):
    out = {}
    for date in dates:
        dd = 'nov26_day' if date == 'nov26' else date
        it0 = JS.DATE_IT0[date]
        static, _ = JS.build_static(date)
        rec = A.surface_records(A.Real(dd, it0))
        SS = L.init_surface_state(date)
        ice_b, _ = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], _geo(static))
        rsi_c = np.asarray(A.Real(dd, it0).cse_in['RSI'])
        res = dict(static_check=JS.verify_static(static))
        for nm, rsi in (('a_restart_rsi', SS['ice']['rsi']), ('b_our_melt_si', ice_b['rsi']), ('c_cse_in_rsi', rsi_c)):
            m = JS.tile_masks(static, rsi)
            cmp = JS.compare_rowsets(m, static['valid'], rec)
            r = dict(match=JS.rowsets_match(cmp), mask_counts=m.sum(axis=(1, 2)).tolist(),
                     real_counts={k: v['n_real'] for k, v in cmp.items()}, mismatch=JS.mismatch_summary(cmp),
                     order_ok={k: v['order_ok'] for k, v in cmp.items()})
            if not r['match']:
                rr = cmp['pa']
                r['pa_only_real_first10'] = rr['only_real'][:10]
                r['pa_only_mask_first10'] = rr['only_mask'][:10]
                post = ice_b['rsi']
                pre = SS['ice']['rsi']
                ch = [(i - 1, j - 1) for (i, j, t) in rr['only_real']] + [(i - 1, j - 1) for (i, j, t) in rr['only_mask']]
                r['cells_mismatch'] = len(set(ch))
                r['cells_mismatch_rsi_pre_post_examples'] = [(int(i) + 1, int(j) + 1, float(pre[i, j]), float(post[i, j])) for (i, j) in sorted(set(ch))[:6]]
                r['cells_where_melt_si_changed_rsi'] = int((pre != post).sum())
                r['cells_where_melt_si_changed_tile_set'] = int((JS.tile_masks(static, pre) != JS.tile_masks(static, post)).any(axis=0).sum())
            res[nm] = r
        res['rsi_max_diff'] = dict(b_minus_c=float(np.abs(ice_b['rsi'] - rsi_c).max()), a_minus_c=float(np.abs(SS['ice']['rsi'] - rsi_c).max()))
        out[date] = res
    return out


def day54(it0=DAY0, n=NSTEP):
    static, _ = JS.build_static('nov26')
    rows, prev = [], None
    for k in range(n):
        it = it0 + k
        R = A.Real('nov26_day', it)
        rec = A.surface_records(R)
        rsi = np.asarray(R.cse_in['RSI'])
        m = JS.tile_masks(static, rsi)
        cmp = JS.compare_rowsets(m, static['valid'], rec)
        row = dict(step=k, itime=it, match=JS.rowsets_match(cmp), n_ocean_tiles=int(m[0].sum()), n_ice_tiles=int(m[1].sum()),
                   rows_ffp=int(len(rec['pa'])), rows_ffs=int(len(rec['ta'])), rows_ffl=int(len(rec['la'])), rows_ffg=int(len(rec['g1'])),
                   rows_fft=int(len(rec['blk1'])), mismatch=JS.mismatch_summary(cmp), order_ok=all(v['order_ok'] for v in cmp.values()))
        w = static['valid'] & (static['fwater'] > 0)
        row['cells_rsi_within_1e-6_of_0_or_1'] = int((w & (((rsi > 0) & (rsi <= 1e-6)) | ((1 - rsi > 0) & (1 - rsi <= 1e-6)))).sum())
        if prev is not None:
            row['ocean_tiles_gained'] = int((m[0] & ~prev[0]).sum()); row['ocean_tiles_lost'] = int((~m[0] & prev[0]).sum())
            row['ice_tiles_gained'] = int((m[1] & ~prev[1]).sum()); row['ice_tiles_lost'] = int((~m[1] & prev[1]).sum())
            ch = []
            for t, nm in ((0, 'ocean'), (1, 'ice')):
                for (i, j) in zip(*np.nonzero(m[t] != prev[t])):
                    ch.append(dict(tile=nm, gained=bool(m[t][i, j]), i=int(i) + 1, j=int(j) + 1, lake=bool(static['is_lake'][i, j]),
                                   rsi_prev=float(prev_rsi[i, j]), rsi=float(rsi[i, j])))
            row['changes'] = ch[:120]
            row['rsi_changed_cells_lake_domain'] = int(((rsi != prev_rsi) & static['is_lake']).sum())
        prev = m
        prev_rsi = rsi
        rows.append(row)
    first = next((r['step'] for r in rows if not r['match']), None)
    return dict(rows=rows, first_mismatch_step=first, steps_matching=sum(r['match'] for r in rows), steps=n,
                distinct_ffp_row_counts=len({r['rows_ffp'] for r in rows}), distinct_ffs_row_counts=len({r['rows_ffs'] for r in rows}),
                distinct_ffl_row_counts=len({r['rows_ffl'] for r in rows}), distinct_ffg_row_counts=len({r['rows_ffg'] for r in rows}),
                distinct_fft_row_counts=len({r['rows_fft'] for r in rows}))


def evolved(ckpt, date='nov26', daydir='nov26_day'):
    """Our own closed-loop ice after the steps of a driver checkpoint -> MELT_SI -> mask vs the real record of the checkpoint's next step."""
    st = pickle.load(open(ckpt, 'rb'))['state']
    SS = st['surface']['SS']
    it = int(st['itime'])
    static, _ = JS.build_static(date)
    ice, _ = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], _geo(static))
    rec = A.surface_records(A.Real(daydir, it))
    rsi_real = np.asarray(A.Real(daydir, it).cse_in['RSI'])
    m = JS.tile_masks(static, ice['rsi'])
    cmp = JS.compare_rowsets(m, static['valid'], rec)
    out = dict(itime=it, steps_carried=int(st['k']), match=JS.rowsets_match(cmp), mismatch=JS.mismatch_summary(cmp),
               rsi_ours_minus_real_cse_in_max=float(np.abs(ice['rsi'] - rsi_real).max()),
               cells_rsi_differ_bitwise=int((ice['rsi'] != rsi_real).sum()))
    if not out['match']:
        rr = cmp['pa']
        out['pa_only_real'] = rr['only_real'][:20]
        out['pa_only_mask'] = rr['only_mask'][:20]
        cells = sorted({(i - 1, j - 1) for (i, j, t) in rr['only_real'] + rr['only_mask']})
        out['mismatch_cells_rsi_ours_vs_real'] = [(i + 1, j + 1, float(ice['rsi'][i, j]), float(rsi_real[i, j])) for (i, j) in cells[:20]]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = argv[0]
    ck = argv[argv.index('--ckpt') + 1] if '--ckpt' in argv else None
    rep = dict(step0=step0(), day54=day54())
    if ck:
        rep['evolved'] = evolved(ck)
    json.dump(rep, open(out_path, 'w'), indent=1, default=str)
    s = rep['step0']
    for d in s:
        print(d, {k: (v['match'] if isinstance(v, dict) and 'match' in v else None) for k, v in s[d].items()}, s[d]['static_check'])
    D = rep['day54']
    print('day54: steps matching', D['steps_matching'], '/', D['steps'], 'first mismatch', D['first_mismatch_step'],
          'distinct row counts ffp/ffs/ffl/ffg/fft', D['distinct_ffp_row_counts'], D['distinct_ffs_row_counts'], D['distinct_ffl_row_counts'],
          D['distinct_ffg_row_counts'], D['distinct_fft_row_counts'])
    if ck:
        print('evolved', {k: v for k, v in rep['evolved'].items() if k not in ('pa_only_real', 'pa_only_mask')})
    return 0


if __name__ == '__main__':
    sys.exit(main())
