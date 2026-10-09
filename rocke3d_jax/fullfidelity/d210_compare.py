"""D210: per-step difference of the server-radiation day (ours_d193 of d210_day.py --rad server) from the replay-driven day (D209 run), as mass-weighted rms
(atm_day_report.diff_metrics, the scorer's metric), also as a multiple of the largest real one-ulp member distance rms(member - real) (cache of multiday_score).
    python d210_compare.py SERVER_OURS_DIR REPLAY_OURS_DIR MEMBERS_CACHE.json OUT.json [nsteps]"""
import json
import sys

import numpy as np

import atm_day_report as RP
import dyn_glue_io as gio

F = ('T', 'U', 'V', 'Q', 'P', 'QCL', 'QCI')
a, b, cache, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 54
mem = json.load(open(cache))['mem']
axyp = gio.load_g('nov26', RP.FF)['axyp']
rows = []
for k in range(n):
    it = 33312 + k
    try:
        sa, sb = np.load(f'{a}/step_{it}.npz'), np.load(f'{b}/step_{it}.npz')
    except FileNotFoundError:
        break
    real = RP.load_real_e(RP.FF, RP.DAYDIR, it)
    w = RP.weights(real['MA'], axyp)
    A = {f: np.asarray(sa[f], float) for f in F}
    B = {f: np.asarray(sb[f], float) for f in F}
    m = RP.state_metrics(A, B, w, fields=F)
    row = dict(k=k, it=it)
    for f in F:
        fl = max(mem[p][k][f]['rms'] for p in mem if mem[p][k] is not None)
        row[f] = dict(rms=m[f]['rms'], n_diff=int((A[f] != B[f]).sum()), max_abs=float(np.abs(A[f] - B[f]).max()), member_max_rms=fl, ratio_to_member_max=m[f]['rms'] / fl if fl > 0 else None,
                      finite=bool(np.isfinite(A[f]).all()))
    rows.append(row)
json.dump(rows, open(out, 'w'), indent=1)
first = next((r['k'] for r in rows if any(r[f]['n_diff'] for f in F)), None)
print('first step with any difference:', first)
for r in rows:
    if r['k'] in (0, 1, 2, 3, 5, 10, 20, 30, 40, 47, 48, 53) or r['k'] == len(rows) - 1:
        print(r['k'], {f: ('%.1e' % r[f]['rms'], '%.2f' % (r[f]['ratio_to_member_max'] or 0)) for f in F})
