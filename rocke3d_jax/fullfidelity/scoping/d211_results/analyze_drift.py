"""D211: per-cell comparison of our carried CONDSE-exit SNOAGE/PREC (OUT/carry/carry_KK.npz) with the recorded CONDSE exit (nov26_day ffc_cse_out_*).
usage: python analyze_drift.py OUTDIR [nsteps]   (run from fullfidelity/)"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import clouds_condse_io as cio
ff = cio.FF_DEFAULT
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
res = []
prev_so = None
for k in range(n):
    c = np.load(f'{out}/carry/carry_{k:02d}.npz')
    r = cio.read_cse(f"{ff}/nov26_day/ffc_cse_out_{33312 + k}.bin")
    ri = cio.read_cse(f"{ff}/nov26_day/ffc_cse_in_{33312 + k}.bin")
    so, ro, rin = c['SNOAGE'], np.asarray(r['SNOAGE']), np.asarray(ri['SNOAGE'])
    rp = np.asarray(r['PREC']); po = c['PREC'] if 'PREC' in c.files else rp*np.nan
    d = np.abs(so - ro)[0]
    # snow-event classification: SNOAGE changed relative to this step's entry. ours entry = our previous exit (k>0)
    ours_in = prev_so if prev_so is not None else rin
    ours_snow = (so != ours_in).any(0); rec_snow = (ro != rin).any(0)
    flips = ours_snow != rec_snow
    dp = np.abs(po - rp)
    e = dict(k=k, max_sno=float(d.max()), n_cells_sno=int((d > 0).sum()), n_snow_ours=int(ours_snow.sum()), n_snow_rec=int(rec_snow.sum()), n_class_flip=int(flips.sum()),
             max_dprec=float(np.nanmax(dp)) if np.isfinite(dp).any() else None, n_dprec_gt_1em12=int((dp > 1e-12).sum()),
             n_cells_sno_gt_1em6=int((d > 1e-6).sum()))
    if d.max() > 0:
        i, j = np.unravel_index(np.argmax(d), d.shape)
        e.update(argmax=[int(i), int(j)], ours=float(so[0, i, j]), rec=float(ro[0, i, j]), ours_prec=float(po[i, j]), rec_prec=float(rp[i, j]), ours_in=float(ours_in[0, i, j]), rec_in=float(rin[0, i, j]),
                 flip_at_argmax=bool(flips[i, j]))
    if flips.any():
        e['flip_cells'] = [[int(a), int(b), bool(ours_snow[a, b]), float(po[a, b]), float(rp[a, b]), float(so[0, a, b]), float(ro[0, a, b])] for a, b in zip(*np.nonzero(flips))][:20]
    res.append(e); prev_so = so
    print({kk: v for kk, v in e.items() if kk != 'flip_cells'}, flush=True)
json.dump(res, open(f'{out}/analysis.json', 'w'), indent=1)
