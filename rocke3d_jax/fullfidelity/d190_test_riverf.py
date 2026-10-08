"""D190: jax_riverf vs riverf_ff.riverf (NumPy) on the reference lake state and on random perturbations of it (stress for the emergency / back-water / KDIREC=9 branches)."""
import sys, time
import d190_util as U
import numpy as np, jax
import surface_loop as L
import riverf_ff as RF
import jax_riverf as JR

date, refp = sys.argv[1], sys.argv[2]
nrand = int(sys.argv[3]) if len(sys.argv) > 3 else 6
ref = dict(np.load(refp))
st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
rs = RF.load_statics(st['axyp'][0])
geo = st['geo']
t0 = time.perf_counter()
R = JR.make_static(rs, geo['flake'], st['fland'], st['fearth'])
print('make_static', round(time.perf_counter() - t0, 2), 'K flow/ocean', R['tab_flow'].shape[1], R['tab_ocean'].shape[1])
f = jax.jit(lambda lake: JR.riverf(R, lake, None, None, None))
lake0 = U.tree(ref, 'S1/lake/')
rng = np.random.default_rng(7)
tot = {}
# synthetic KDIREC = 9 pairs (no such cell exists on this grid): the pair rule is exercised on both implementations with the same modified tables
rs9 = dict(rs); kd9 = rs['kdirec'].copy()
lakecells = np.argwhere((geo['flake'] > 0))
pairs = []
for (i, j) in lakecells:
    if i + 1 < 72 and geo['flake'][i + 1, j] > 0 and j > 2 and j < 43:
        kd9[i, j] = 9; kd9[i + 1, j] = 9; pairs.append((i, j))
        if len(pairs) >= 4: break
    elif j + 1 < 44 and geo['flake'][i, j + 1] > 0 and j > 2:
        kd9[i, j] = 9; kd9[i, j + 1] = 9; pairs.append((i, j))
        if len(pairs) >= 8: break
rs9['kdirec'] = kd9
R9 = JR.make_static(rs9, geo['flake'], st['fland'], st['fearth'])
f9 = jax.jit(lambda lake: JR.riverf(R9, lake, None, None, None))
print('synthetic kd9 cells', int((kd9 == 9).sum()), 'k2_act', int(R9['k2_act'].sum()), 'k8_act', int(R9['k8_act'].sum()))
# synthetic emergency cells: lake cells (almost pure lake) with kd911 > 0 get kdirec = 0 (no direction), then a huge water mass
rse = dict(rs); kde = rs['kdirec'].copy()
pure = (geo['flake'] > .949 * (geo['flake'] + st['fearth'])) & (rs['kd911'] > 0) & (geo['flake'] > 0)
ii = np.argwhere(pure)[:12]
for (i, j) in ii: kde[i, j] = 0
rse['kdirec'] = kde
Re = JR.make_static(rse, geo['flake'], st['fland'], st['fearth'])
fe = jax.jit(lambda lake: JR.riverf(Re, lake, None, None, None))
print('synthetic emergency cells', len(ii))
for trial in range(nrand + 1 + 6 + 4):
    lk = {k: np.array(v) for k, v in lake0.items()}
    use9 = nrand < trial <= nrand + 6
    use_e = trial > nrand + 6
    rsx, Rx, fx = (rs9, R9, f9) if use9 else ((rse, Re, fe) if use_e else (rs, R, f))
    if trial > 0:
        sel = rng.random(lk['mwl'].shape) < 0.5
        fac = np.where(sel, rng.uniform(0.2, 4.0, lk['mwl'].shape), 1.0) if trial % 2 else np.where(sel, rng.uniform(1.0, 60.0, lk['mwl'].shape), 1.0)
        lk['mwl'] = lk['mwl'] * fac; lk['gml'] = lk['gml'] * fac
        lk['mldlk'] = lk['mldlk'] * np.where(rng.random(lk['mwl'].shape) < 0.3, rng.uniform(0.3, 1.5, lk['mwl'].shape), 1.0)
    if use_e or trial == nrand - 1 or trial == nrand:        # emergency overflow: huge water mass in lakes without outflow direction (kdirec = 0, kd911 > 0)
        em = (rsx['kdirec'] == 0) & (rs['kd911'] > 0) & (geo['flake'] > 0)
        big = np.where(em, 800.0, 1.0)
        lk['mwl'] = lk['mwl'] * big; lk['gml'] = lk['gml'] * big
    new, flowo, eflowo, gtm, gtr, mlh, diag = RF.riverf(lk, geo['flake'], st['fland'], st['fearth'], rsx)
    got = fx(U.to_dev(lk))
    gn, gfo, gefo, ggt, ggr, gml_ = [U.to_np(x) for x in got]
    res = {}
    for k in ('mwl', 'gml', 'tlake', 'mldlk', 'dlake', 'glake'):
        res['lake.' + k] = U.cat(gn[k], new[k])
    res['flowo'] = U.cat(gfo, flowo); res['eflowo'] = U.cat(gefo, eflowo)
    res['gtemp'] = U.cat(ggt, gtm); res['gtempr'] = U.cat(ggr, gtr); res['mlhc'] = U.cat(gml_, mlh)
    bad = {k: v for k, v in res.items() if v[0] != 'A'}
    print('trial', trial, 'diag', {k: v for k, v in diag.items() if not isinstance(v, list)}, 'nonA:', bad if bad else 'none', 'flowo nz', int((flowo != 0).sum()))
