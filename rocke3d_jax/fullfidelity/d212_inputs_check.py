"""D212: step-0 (and chain) validation of the three recorded inputs of the radiation packet against their recorded values, for nov26 / dec01 / jan01.
  surface fields : drv_radpacket (NumPy) and jax_radiation.surface_fields_jax (device) built from the RESTART state vs rsv_n26_<it0>_in.bin
  seed           : jax_radiation.seed_chain_radia / seed_chain_next (D174 chain, from the recorded SEEDS[0] of the first step) vs recorded SEEDS[1] of every step
  COSZ1          : drv_zenith.Zenith (libimf) vs the recorded COSZ1 of ffa_step_<it>_r.bin of every step
Usage: python d212_inputs_check.py OUT.json [dates...]"""
import clouds_jax_env  # noqa: F401
import json
import sys

import numpy as np

import radiation_server as rs
import drv_radpacket as RP
import drv_rng
import surface_loop as L
import atm_step as A

FF = rs.FF
DAYDIR = {'nov26': 'nov26_day', 'dec01': 'dec01', 'jan01': 'jan01'}
NST = {'nov26': 54, 'dec01': 6, 'jan01': 6}


def cmp(a, b, sel):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim == 3 and sel.ndim == 2:
        d = np.stack([np.abs(a[k] - b[k])[sel] for k in range(a.shape[0])])
        nd = int(sum((a[k][sel] != b[k][sel]).sum() for k in range(a.shape[0])))
        sc = float(np.abs(b[:, sel]).max()) if d.size else 0.0
    else:
        d = np.abs(a - b)[sel]
        nd = int((a[sel] != b[sel]).sum())
        sc = float(np.abs(b[sel]).max()) if d.size else 0.0
    return dict(max=float(d.max()) if d.size else 0.0, ndiff=nd, n=int(d.size), scale=sc)


def cat(r):
    if r['ndiff'] == 0:
        return 'A'
    rel = r['max'] / max(r['scale'], 1e-300)
    return 'B' if rel <= 1e-12 else ('C' if rel <= 1e-6 else 'D')


def surface(date, it0):
    import ghy_compare as GC
    day = DAYDIR[date]
    lv = rs.read_packet(f'{FF}/{day}/rsv_n26_{it0}_in.bin')
    st = L.load_statics(date, L.FF)
    st['ctx'] = L.make_ocean_ctx(date, L.FF)
    S0 = L.init_surface_state(date, L.FF, st=st)
    pole = RP.pole_mask(); ok = ~pole
    res = {}
    f = RP.ice_lake_landice_fields(S0, st)
    geo = st['geo']
    poice = (f['RSI'] * geo['fwater'] > 0) & ok
    fw = (geo['fwater'] > 0) & ok
    fli = (st['flice'] > 0) & ok
    sel = dict(RSI=ok, SNOWI=ok, POND_MELT=ok, FLAG_DSWS=ok, ZSI=poice, ZSNOWI=poice, GTEMPR2=poice, FLAKE=ok, DLAKE=ok, FLICE=ok, FLAND=ok, FEARTH=ok, GTEMPR1=fw, GTEMPR3=fli, SNOWLI=fli)
    for k, s in sel.items():
        res[k] = cmp(f[k], lv[k], s)
    g = GC.load(f'{FF}/{day}/ffg_{it0}.bin')
    g2 = g[:len(g) // 2]
    rst = f'{FF}/_pristine_restarts/fort1_{date}_itime{it0}.nc'
    ra = RP.restart_land_arrays(rst, g2)
    lf, sbv = RP.land_fields(g2, ra['tbcs'], ra['w'], ra['nsn'], ra['dzsn'], ra['wsn'], ra['fr_snow'], ra['snowbv'], update_snowbv=False)
    land = np.zeros((72, 46), bool); land[g2[:, 0].astype(int) - 1, g2[:, 1].astype(int) - 1] = True; land &= ok
    fvv = np.zeros((72, 46)); fvv[g2[:, 0].astype(int) - 1, g2[:, 1].astype(int) - 1] = np.where(g2[:, 169] < 1e-6, 0.0, np.where(g2[:, 169] > 1 - 1e-6, 1.0, g2[:, 169]))
    for k in ('GTEMPR4', 'BARESW', 'SNOWD'):
        res[k] = cmp(lf[k], lv[k], land)
    res['FRSNOW_bare(fb>0)'] = cmp(lf['FRSNOW'][0], lv['FRSNOW'][0], land & (fvv < 1.0))
    res['FRSNOW_bare(fb=0)'] = cmp(lf['FRSNOW'][0], lv['FRSNOW'][0], land & (fvv >= 1.0))
    res['FRSNOW_veg(fv>0)'] = cmp(lf['FRSNOW'][1], lv['FRSNOW'][1], land & (fvv > 0.0))
    res['FRSNOW_veg(fv=0)'] = cmp(lf['FRSNOW'][1], lv['FRSNOW'][1], land & (fvv <= 0.0))
    res['TSAVG(restart)'] = cmp(ra['tsavg'], lv['TSAVG'], ok)
    res['WSAVG(restart)'] = cmp(ra['wsavg'], lv['WSAVG'], ok)
    # device version of the ice/lake/land-ice group (same inputs, jax functions)
    import jax.numpy as jnp
    import jax_radiation as JR
    import jax_p1_melt as M
    import jax_seaice_lake as SL
    import jax_state_d181 as _  # noqa: F401
    import jax_posttile as PT
    K, _kb = PT.make_static_all(st, date, it0)
    dv = lambda d: {k: jnp.asarray(v) for k, v in d.items()}
    ice_new, _m = M.melt_si(dv(S0['ice']), jnp.asarray(S0['atm']['gtemp']), jnp.asarray(S0['atm']['sss']), jnp.asarray(S0['atm']['mlhc']), M.geo_device(st['geo']), 1800.0)
    ag = SL.seaice_to_atmgrid(K, ice_new)
    li = S0['li']
    jf = JR.surface_fields_jax(ice_new['rsi'], ice_new['snowi'], ice_new['pond_melt'], ice_new['flag_dsws'], ag['zsi'], ag['zsnowi'], ag['gtempr'],
                               jnp.asarray(geo['fwater']), jnp.asarray(geo['flake']), jnp.asarray(S0['lake']['mwl']), jnp.asarray(st['axyp']),
                               jnp.asarray(st['flice']), jnp.asarray(st['fland']), jnp.asarray(st['fearth']), jnp.asarray(S0['atm']['gtempr']),
                               jnp.asarray(li['tlandi'][..., 0]), jnp.asarray(li['snowli']))
    for k, s in sel.items():
        res['jax:' + k] = cmp(np.asarray(jf[k]), lv[k], s)
    return res


def seeds(date, it0, n):
    import clouds_condse_io as cio
    day = DAYDIR[date]
    rec = [cio.read_cse(f'{FF}/{day}/ffc_cse_out_{it0 + k}.bin')['SEEDS'] for k in range(n)]
    s0 = int(round(float(rec[0][0]))) % drv_rng.M
    out = []
    for k in range(n):
        it = it0 + k
        mine1 = drv_rng.radia_seed(s0)
        r1 = int(round(float(rec[k][1])))
        r0 = int(round(float(rec[k][0]))) % drv_rng.M
        out.append(dict(k=k, s0_equal=(s0 == r0), seed_equal=(mine1 % drv_rng.M == r1 % drv_rng.M)))
        s0 = drv_rng.next_seed(s0, A.is_radiation_step(it))
    return dict(n=n, all_s0_equal=all(o['s0_equal'] for o in out), all_seed_equal=all(o['seed_equal'] for o in out), bad=[o for o in out if not (o['s0_equal'] and o['seed_equal'])][:5])


def cosz(date, it0, n):
    import drv_zenith as Z
    z = Z.Zenith()
    day = DAYDIR[date]
    mx, nd, bad = 0.0, 0, []
    for k in range(n):
        it = it0 + k
        rr = A.Real(day, it, FF)
        ref = np.asarray(rr.r['COSZ1'])
        c = np.asarray(z.cosz1(it))
        d = np.abs(c - ref)
        mx = max(mx, float(d.max())); nd += int((c != ref).sum())
        if (c != ref).any():
            bad.append(k)
    return dict(n=n, max=mx, n_unequal=nd, bad_steps=bad[:10])


if __name__ == '__main__':
    out = sys.argv[1]
    dates = sys.argv[2:] or ['nov26', 'dec01', 'jan01']
    res = {}
    for d in dates:
        it0 = dict(A.DATES)[d]
        r = dict(it0=it0)
        r['seed'] = seeds(d, it0, NST[d])
        print(d, 'seed', r['seed'], flush=True)
        r['cosz1'] = cosz(d, it0, NST[d])
        print(d, 'cosz1', r['cosz1'], flush=True)
        import os
        if not os.path.exists(f'{FF}/{DAYDIR[d]}/rsv_n26_{it0}_in.bin'):
            r['surface'] = 'no recorded radiation packet at step 0 (radiation steps of this date start at it %s)' % sorted(f for f in os.listdir(f'{FF}/{DAYDIR[d]}') if f.startswith('rsv_n26') and f.endswith('_in.bin'))
            print(d, r['surface'], flush=True)
            res[d] = r
            json.dump(res, open(out, 'w'), indent=1)
            continue
        r['surface'] = surface(d, it0)
        for k, v in r['surface'].items():
            v['cat'] = cat(v)
        print(d, {k: (v['cat'], v['max'], v['ndiff']) for k, v in r['surface'].items()}, flush=True)
        res[d] = r
        json.dump(res, open(out, 'w'), indent=1)
