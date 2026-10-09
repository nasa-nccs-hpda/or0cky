import sys, os
sys.path.insert(0, '.')
import numpy as np
import clouds_condse_io as cio, ghy_compare as GC, jax_static as JST, ghy_ref as R
import ocean_step as O
import daily_lake as DL
FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
S = os.environ['SC']
IM, JM = 72, 46
c47, c48 = cio.read_cse(f'{FF}/ffc_cse_in_33359.bin'), cio.read_cse(f'{FF}/ffc_cse_in_33360.bin')
z = np.load(f'{S}/../d196/c1/nov26_day_step47.npz')
topo = JST.topography()
axyp = np.repeat(O._DXYPO[None, :], IM, axis=0)
import lakes_compare as LCm
import glob
l2f = sorted(glob.glob(FF.replace('nov26_day','nov26')+'/ffl2_*.bin'))[:1]
print('ffl2 files', l2f)
hl = np.zeros((IM, JM))
l2 = LCm.load(l2f[0]); hl[l2[:, 0].astype(int) - 1, l2[:, 1].astype(int) - 1] = l2[:, 14]
print('hlake rec vs topo(clamped) on lake cells:', np.abs(hl - np.where(topo['flake'] + c47['FLAKE'] > 0, np.maximum(topo['hlake'], 1.0), 0))[c47['FLAKE'] > 0].max())
tn = DL.tanlk(topo['flake'], np.maximum(topo['hlake'], 1.0), axyp)
# DMWLDF from the real substep-2 GHY output of step 47
g = GC.load(f'{FF}/ffg_33359.bin'); n = len(g) // 2; g2 = g[n:]
ecells = (g2[:, 1].astype(int) - 1) * IM + (g2[:, 0].astype(int) - 1)
w_out = np.stack([GC.unpack(r)[4]['w_out'] for r in g2])
dm = DL.water_deficit(g2, w_out, ecells, c47['FEARTH'], R.THM[0, :])
lake = c47['FLAKE'] > 0
print('lake cells', lake.sum(), 'DMWLDF on lake cells: n>0', (dm[lake] > 0).sum(), 'max', dm[lake].max())
St = dict(flake=c47['FLAKE'], fearth=c47['FEARTH'], fland=c47['FLAND'], rsi=z['surf/ice/rsi'], msi=z['surf/ice/msi'], snowi=z['surf/ice/snowi'], hsi=z['surf/ice/hsi'],
          mwl=z['surf/lake/mwl'], gml=z['surf/lake/gml'], tlake=z['surf/lake/tlake'], mldlk=z['surf/lake/mldlk'])
out = DL.daily_lake(St, c47['FLICE'], c47['FOCEAN'], tn, hl, axyp, dm)
print(out['counters'], out['pow'])
f48 = c48['FLAKE']
d = out['flake'] - f48
print('FLAKE port vs record33360: max abs', np.abs(d[lake]).max(), ' cells > 1e-9:', (np.abs(d[lake]) > 1e-9).sum(), ' >1e-6:', (np.abs(d[lake]) > 1e-6).sum())
chg_rec = (f48 != c47['FLAKE']) & lake
chg_us = (out['flake'] != c47['FLAKE']) & lake
print('changed cells record', chg_rec.sum(), 'port', chg_us.sum(), 'both', (chg_rec & chg_us).sum())
rel = np.abs(d)[chg_rec] / np.abs(f48 - c47['FLAKE'])[chg_rec]
print('error relative to the change: median %.3g max %.3g' % (np.median(rel), rel.max()))
np.save(f'{S}/val1_dm.npy', dm)
for nm, a, b in (('FEARTH', out['fearth'], c48['FEARTH']), ('FLAND', out['fland'], c48['FLAND'])):
    print(nm, 'max abs vs record', np.abs(a - b)[lake].max())
dr = out['rsi'] - c48['RSI']
print('RSI lake cells: max abs vs record 33360 %.3g; cells > 1e-6: %d; > 1e-3: %d' % (np.abs(dr[lake]).max(), (np.abs(dr[lake]) > 1e-6).sum(), (np.abs(dr[lake]) > 1e-3).sum()))
pure = lake & (c47['RSI'] == 1.0)
print('pure-ice cells', pure.sum(), 'record rsi<1:', (c48['RSI'][pure] < 1).sum(), 'port rsi<1:', (out['rsi'][pure] < 1).sum(), 'set equal', np.array_equal(c48['RSI'][pure] < 1, out['rsi'][pure] < 1))
np.savez(f'{S}/val1_out.npz', **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
