"""D201: ON (D199 elhx fix) vs OFF per step from the d201_day.py captures (aux_land of the 2nd surface substep, 753 land cells) + real rows."""
import sys, json, numpy as np
S = sys.argv[1]
real = json.load(open('scoping/d201_results/real_rows.json'))
res = []
print('k | first ON/OFF diff: n cells !=, max|dtsns| cell | cell112 tsns OFF ON real | aevap OFF ON real | qsrf OFF ON | n nonfinite ON/OFF')
for k in range(41):
    a = np.load(f'{S}/on/cap/land_{k}.npz'); b = np.load(f'{S}/off/cap/land_{k}.npz')
    ts_a, ts_b = a['land/ghy/tsns'], b['land/ghy/tsns']
    neq = int(sum(((a[x] != b[x]) & ~(np.isnan(a[x]) & np.isnan(b[x]))).reshape(753, -1).any(1).sum() * 0 for x in ['land/ghy/tsns']))
    diffcells = np.where(((a['land/ghy/tsns'] != b['land/ghy/tsns'])))[0]
    d = np.abs(ts_a - ts_b); d = np.where(np.isnan(d), 1e9, d)
    c = int(d.argmax())
    r = real[f'{k}/112/2']
    row = dict(k=k, ncells_diff=len(diffcells), maxd=float(d.max()), cell=c, ts_off=float(ts_b[112]), ts_on=float(ts_a[112]), ts_real=r['tsns'],
               ae_off=float(b['land/ghy/aevap'][112]), ae_on=float(a['land/ghy/aevap'][112]), ae_real=r['aevap'],
               qs_off=float(b['land/pbl/qsrf'][112]), qs_on=float(a['land/pbl/qsrf'][112]),
               nf_on=int((~np.isfinite(ts_a)).sum()), nf_off=int((~np.isfinite(ts_b)).sum()), first_cells=[int(x) for x in diffcells[:6]])
    res.append(row)
    print(k, row['ncells_diff'], '%.2e' % row['maxd'], c, '| %.4f %.4f %.4f | %.4e %.4e %.4e | %.6f %.6f | %d %d' % (row['ts_off'], row['ts_on'], row['ts_real'], row['ae_off'], row['ae_on'], row['ae_real'], row['qs_off'], row['qs_on'], row['nf_on'], row['nf_off']), row['first_cells'])
json.dump(res, open('scoping/d201_results/on_off_by_step.json', 'w'), indent=1)
