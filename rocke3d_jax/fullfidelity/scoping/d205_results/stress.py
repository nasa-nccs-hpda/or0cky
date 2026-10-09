import sys, os
sys.path.insert(0, '.')
import numpy as np
import daily_lake_harness as H
import daily_lake as DL
S = os.environ['SC']
c = dict(np.load(f'{S}/fhar/case.npz'))
rng = np.random.default_rng(205)
tot = {}
for trial in range(12):
    d = {k: np.array(v, copy=True) for k, v in c.items()}
    lake = d['flake'] > 0
    shp = d['mwl'].shape
    f = np.exp(rng.normal(0, [0.05, 0.3, 1.0, 2.5][trial % 4], shp))          # mass factors up to e^(+-2.5)
    d['mwl'] = d['mwl'] * f
    d['gml'] = d['gml'] * f
    d['rsi'] = np.where(lake, rng.uniform(0, 1, shp) * (rng.uniform(0, 1, shp) < 0.6), d['rsi'])
    d['rsi'] = np.where(lake & (rng.uniform(0, 1, shp) < 0.2), 1.0, d['rsi'])
    d['msi'] = np.where(lake, np.exp(rng.normal(5, 1.5, shp)), d['msi'])         # ~150 kg/m2 ; some above 5000 (ice dump)
    d['snowi'] = np.where(lake, rng.uniform(0, 50, shp), d['snowi'])
    d['hsi'] = d['hsi'] * np.exp(rng.normal(0, 0.3, d['hsi'].shape))
    d['dmwldf'] = np.where(rng.uniform(0, 1, shp) < 0.8, rng.uniform(0, 1500, shp), 0.0) * (d['fearth'] > 0)
    d['mldlk'] = np.where(lake, rng.uniform(0.5, 4, shp), d['mldlk'])
    d['tlake'] = np.where(lake, rng.uniform(0, 25, shp), d['tlake'])
    rep, po, fo = H.compare(f'{S}/fhar', d)
    bad = {k: v for k, v in rep.items() if v[1] != 0}
    for k, v in po['counters'].items():
        tot[k] = tot.get(k, 0) + v
    print('trial', trial, 'differing fields:', bad if bad else 'none (bitwise)', flush=True)
print('branch counts over all trials', tot)
