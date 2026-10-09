"""D211: (1) self-consistency of the chain's SNOAGE update: ours_exit(k) == ours_exit(k-1) * exp(-ours PREC(k)) where ours EPREC(k) < 0 (all other cells unchanged),
(2) rain/snow classification (EPREC<0) ours vs record per step, (3) relevance of the drift to the albedo: cells with |dSNOAGE| > thresholds split by surface type present
(type 1 ocean ice: RSI>0, type 2 land ice: FLICE>0, type 3 land: FEARTH>0).  usage (from fullfidelity/): python scoping/d211_results/analyze_drift2.py OUTDIR [nsteps]"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import clouds_condse_io as cio
ff = cio.FF_DEFAULT
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
res = []; prev = None
for k in range(n):
    c = np.load(f'{out}/carry/carry_{k:02d}.npz')
    rin = cio.read_cse(f"{ff}/nov26_day/ffc_cse_in_{33312 + k}.bin"); rout = cio.read_cse(f"{ff}/nov26_day/ffc_cse_out_{33312 + k}.bin")
    so, po, eo = c['SNOAGE'], c['PREC'], c['EPREC']
    ro, rp, re = np.asarray(rout['SNOAGE']), np.asarray(rout['PREC']), np.asarray(rout['EPREC'])
    entry = prev if prev is not None else np.asarray(rin['SNOAGE'])
    # (1) self-consistency of the chain
    snow = eo < 0
    pred = np.where(snow[None], entry * np.exp(-po)[None], entry)
    sc = np.abs(so - pred)
    # (2) classification
    rsnow = re < 0
    flips = snow != rsnow
    d = np.abs(so - ro)
    fe, fl, rsi = np.asarray(rin['FEARTH']), np.asarray(rin['FLICE']), np.asarray(rin['RSI'])
    present = np.stack([rsi > 0, fl > 0, fe > 0])          # types 1,2,3
    e = dict(k=k, selfcons_max=float(sc.max()), selfcons_n_gt_1e_13=int((sc > 1e-13).sum()), n_snow_ours=int(snow.sum()), n_snow_rec=int(rsnow.sum()), n_flip=int(flips.sum()),
             max_dprec=float(np.abs(po - rp).max()), n_dprec_gt_1e_6=int((np.abs(po - rp) > 1e-6).sum()),
             max_dsno=float(d.max()), n_dsno_gt_1=int((d > 1).sum()), n_dsno_gt_1_with_surface=int(((d > 1) & present).sum()), n_dsno_gt_1e_3_with_surface=int(((d > 1e-3) & present).sum()),
             n_present=int(present.sum()))
    # drift in cells that never flipped so far vs flipped (cumulative)
    res.append(e); prev = so
    print(e, flush=True)
fl_cum = None
json.dump(res, open(f'{out}/analysis2.json', 'w'), indent=1)
