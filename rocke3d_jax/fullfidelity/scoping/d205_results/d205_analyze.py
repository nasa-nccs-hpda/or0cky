"""D205 analysis of a d205_day.py run.  python d205_analyze.py OUT C1 D203OUT   (OUT = d205_day.py outdir, C1 = its --c1dir, D203OUT = ours_d193 dir of the D203 run).
(1) steps 0-47 of the run bitwise equal to the stored D203 steps (T U V Q P QCL QCI);  (2) the hook's FLAKE/FEARTH/FLAND against the recorded CONDSE entry of step 48 (33360);
(3) end-of-step lake/ice state against the next step's record, steps 44-52: lake RSI, lake mass (ffs lake-tile mwl, gml);  (4) tile-mask mismatches per step from d193_run.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import clouds_condse_io as cio  # noqa: E402
import surface_tile_ff as ST  # noqa: E402

FF = os.environ.get('FF_DATA', '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data') + '/nov26_day'
IT0 = 33312
out, c1, d203 = sys.argv[1:4]
rep = {}
# (1) bitwise against D203
same, first_diff = [], None
for k in range(54):
    a = np.load(f'{out}/ours_d193/step_{IT0 + k}.npz')
    b = np.load(f'{d203}/step_{IT0 + k}.npz')
    eq = all(np.array_equal(a[f], b[f]) for f in ('T', 'U', 'V', 'Q', 'P', 'QCL', 'QCI'))
    same.append(eq)
    if not eq and first_diff is None:
        first_diff = k
rep['bitwise_vs_d203'] = dict(steps_equal=[k for k, e in enumerate(same) if e], first_differing_step=first_diff)
# (2) fractions
h = np.load(f'{out}/d205_lake_hook.npz')
c47, c48 = cio.read_cse(f'{FF}/ffc_cse_in_33359.bin'), cio.read_cse(f'{FF}/ffc_cse_in_33360.bin')
lake = c47['FLAKE'] > 0
d = h['flake'] - c48['FLAKE']
chg_rec, chg_us = (c48['FLAKE'] != c47['FLAKE']) & lake, (h['flake'] != c47['FLAKE']) & lake
rel = np.abs(d)[chg_rec & chg_us] / np.abs(c48['FLAKE'] - c47['FLAKE'])[chg_rec & chg_us]
rep['flake_vs_record_33360'] = dict(max_abs=float(np.abs(d[lake]).max()), n_gt_1e9=int((np.abs(d[lake]) > 1e-9).sum()), n_gt_1e6=int((np.abs(d[lake]) > 1e-6).sum()),
                                    changed_record=int(chg_rec.sum()), changed_ours=int(chg_us.sum()), both=int((chg_rec & chg_us).sum()),
                                    err_over_change_median=float(np.median(rel)), err_over_change_max=float(rel.max()),
                                    flake_unchanged_noop_max_abs=float(np.abs(c47['FLAKE'] - c48['FLAKE'])[lake].max()),
                                    fearth_max_abs=float(np.abs(h['fearth'] - c48['FEARTH'])[lake].max()), fland_max_abs=float(np.abs(h['fland'] - c48['FLAND'])[lake].max()))
# (3) end of step k vs record of step k+1
rows = []
for k in range(44, 53):
    z = np.load(f'{c1}/nov26_day_step{k}.npz')
    it1 = IT0 + k + 1
    cn = cio.read_cse(f'{FF}/ffc_cse_in_{it1}.bin')
    lk = cn['FLAKE'] > 0
    dr = np.abs(z['surf/ice/rsi'] - cn['RSI'])[lk]
    t = ST.load(f'{FF}/ffs_{it1}.bin')
    t1 = t[:len(t) // 2]
    r = t1[t1[:, 2] == 1]
    ii, jj = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
    sel = (cn['FOCEAN'][ii, jj] == 0) & (cn['FLAKE'][ii, jj] > 0)
    mr = np.abs(z['surf/lake/mwl'][ii, jj] - r[:, ST.IN['mwl']])[sel] / np.abs(r[:, ST.IN['mwl']][sel])
    gr = np.abs(z['surf/lake/gml'][ii, jj] - r[:, ST.IN['gml']])[sel] / np.maximum(np.abs(r[:, ST.IN['gml']][sel]), 1e-30)
    rows.append(dict(step=k, lake_rsi_max=float(dr.max()), lake_rsi_n_gt_1e3=int((dr > 1e-3).sum()), lake_rsi_n_gt_1e6=int((dr > 1e-6).sum()),
                     mwl_rel_median=float(np.median(mr)), mwl_rel_max=float(mr.max()), gml_rel_median=float(np.median(gr)), gml_rel_max=float(gr.max()), n_tiles=int(sel.sum())))
rep['end_of_step_vs_next_record'] = rows
run = json.load(open(f'{out}/d193_run.json'))
rep['tile_mask_mismatch'] = {s['k']: s['tile_mask_mismatch'] for s in run['steps'] if s['k'] >= 44}
rep['daily_step48'] = run['steps'][48]['daily'] if len(run['steps']) > 48 else None
rep['completed'] = run['completed']
rep['error'] = run['error']
print(json.dumps(rep, indent=1, default=str))
json.dump(rep, open(f'{out}/d205_analysis.json', 'w'), indent=1, default=str)
