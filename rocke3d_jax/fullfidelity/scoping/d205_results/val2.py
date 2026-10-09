import sys, os
sys.path.insert(0, '.')
import numpy as np
exec(open(os.environ['SC']+'/val1.py').read().split("print('ffl2 files'")[0])
import surface_tile_ff as ST
St = dict(mwl=z['surf/lake/mwl'], gml=z['surf/lake/gml'])
o = np.load(f'{S}/val1_out.npz'); dm = np.load(f'{S}/val1_dm.npy')
f47, f48, fp = c47['FLAKE'], c48['FLAKE'], o['flake']
lake = f47 > 0
err = np.abs(fp - f48)
idx = np.argsort(-err * lake, axis=None)[:12]
t = ST.load(f'{FF}/ffs_33360.bin'); n = len(t)//2; t1 = t[:n]; r = t1[t1[:, 2] == 1]
mwl_rec = np.full((IM, JM), np.nan); gml_rec = np.full((IM, JM), np.nan)
mwl_rec[r[:, 0].astype(int)-1, r[:, 1].astype(int)-1] = r[:, ST.IN['mwl']]; gml_rec[r[:, 0].astype(int)-1, r[:, 1].astype(int)-1] = r[:, ST.IN['gml']]
print('cell | flake47 rec48 port | dmwldf | mwl_ours47 mwl_rec48 mwl_port | frac chg rec/port')
for k in idx:
    i, j = np.unravel_index(k, err.shape)
    print((i+1, j+1), '%.7g %.7g %.7g' % (f47[i, j], f48[i, j], fp[i, j]), '%.1f' % dm[i, j], '%.6g %.6g %.6g' % (St['mwl'][i, j], mwl_rec[i, j], o['mwl'][i, j]), '%.3g %.3g' % ((f48[i, j]-f47[i, j])/f47[i, j], (fp[i, j]-f47[i, j])/f47[i, j]))
# mwl-after comparison, per cell where mwl changes (expanding) : compare per-area mass mwl/flake
ok = lake & np.isfinite(mwl_rec)
print('lake tiles with ffs lake row:', ok.sum())
rel_port = np.abs(o['mwl'] - mwl_rec)[ok] / mwl_rec[ok]
rel_noop = np.abs(St['mwl'] - mwl_rec)[ok] / mwl_rec[ok]
print('mwl: |port-rec|/rec median %.3g max %.3g ; |ours47-rec|/rec median %.3g max %.3g' % (np.median(rel_port), rel_port.max(), np.median(rel_noop), rel_noop.max()))
relg = np.abs(o['gml'] - gml_rec)[ok] / np.abs(gml_rec[ok]); relg0 = np.abs(St['gml'] - gml_rec)[ok] / np.abs(gml_rec[ok])
print('gml: port %.3g/%.3g  noop %.3g/%.3g' % (np.median(relg), relg.max(), np.median(relg0), relg0.max()))
ch = ok & (o['mwl'] != St['mwl'])
print('cells where port changes mwl', ch.sum(), ' median |port-rec|/rec there %.3g, noop %.3g' % (np.median(np.abs(o['mwl']-mwl_rec)[ch]/mwl_rec[ch]), np.median(np.abs(St['mwl']-mwl_rec)[ch]/mwl_rec[ch])))
# ffs at 33359 for baseline noise of the state: mwl rec 47 vs rec 48 in unchanged-lake cells
t = ST.load(f'{FF}/ffs_33359.bin'); n = len(t)//2; r = t[:n][t[:n][:, 2] == 1]
m47 = np.full((IM, JM), np.nan); m47[r[:, 0].astype(int)-1, r[:, 1].astype(int)-1] = r[:, ST.IN['mwl']]
print('real mwl step47 entry vs step48 entry rel change (all lake): median %.3g max %.3g' % (np.nanmedian(np.abs(mwl_rec-m47)[ok]/mwl_rec[ok]), np.nanmax(np.abs(mwl_rec-m47)[ok]/mwl_rec[ok])))
