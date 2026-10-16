"""Validate dyn_glue_ff.py (D114-D117 coupling glue) against real-Fortran dumps ffd_glue_* (layouts in
dyn_glue_io.py; instrumentation ATM_DRV_dynF.f.patch + ATMDYN_glue / ATM_UTILS_glue / ATMDYN_COM_glue /
SURFACE_glue / ATURB_glue / CLOUDS2_DRV_glue / DIAG_glue patches, units 1240-1251), 3 dates x 6 steps.

Usage (from fullfidelity/): python3 dyn_glue_compare.py [--imf-pow] [--analytic-geom]
  --imf-pow        tropwmo pow via the Intel libimf bridge (bitwise to ifort)
  --analytic-geom  use dyn_glue_ff.build_geom (analytic, numpy sin/cos) instead of the recorded geometry
"""
import glob
import sys
import numpy as np

import dyn_glue_io as io
import dyn_glue_ff as gf
import dyn_filter_ff as ff
import dyn_filter_compare as fcm
from dyn_glue_io import IM, JM, LM, DATES, NSTEP, FF_DEFAULT


def mx(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def nbad(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def geom(date, analytic=False, ff_dir=FF_DEFAULT):
    rec = io.load_g(date, ff_dir)
    if not analytic:
        return rec
    radius = fcm.ff_load(f"{ff_dir}/{date}/ffd_filt_consts.bin")['radius']
    return gf.build_geom(radius, rec)


def run_step(date, itime, g, imf_pow=False, stats=None, ff_dir=FF_DEFAULT):
    P = lambda n: io.path(date, f"{n}_{itime}.bin", ff_dir)
    res = {}
    # ---- D114 CALC_TROP / tropwmo
    d = io.load_trop(P('trop'))
    pt, lt, ierr, tt = gf.calc_trop(d['t'], d['pk'], d['pmid'], g, imf_pow=imf_pow, stats=stats)
    res['trop_p'] = mx(pt, d['ptropo']); res['trop_l_nbad'] = nbad(lt, d['ltropo'])
    res['trop_ierr'] = ierr
    res['trop_lrange'] = (int(d['ltropo'].min()), int(d['ltropo'].max()))
    # ---- D114 COMPUTE_WSAVE
    d = io.load_wsave(P('wsave'))
    w = gf.compute_wsave(d['mws'], d['t'], d['pk'], d['pedn'], g)
    res['wsave'] = mx(w, d['wsave']); res['wsave_nbad'] = nbad(w, d['wsave']); res['wsave_scale'] = float(np.max(np.abs(d['wsave'])))
    # ---- D116 PGRAD_PBL (valid cells: J=2..JM-1 and the two pole cells I=1)
    d = io.load_pgrad(P('pgrad'))
    o = gf.pgrad_pbl(d['t1'], d['pk1'], d['pmid1'], d['pedn1'], d['phi1'], d['zatmo'], g)
    mask = np.zeros((IM, JM), bool); mask[:, 1:JM - 1] = True; mask[0, 0] = mask[0, JM - 1] = True
    for n, a in zip(('dpdx', 'dpdy', 'dpdx0', 'dpdy0'), o):
        res['pgrad_' + n] = mx(a[mask], d[n][mask]); res['pgrad_' + n + '_nbad'] = nbad(a[mask], d[n][mask])
    res['pgrad_scale'] = float(np.max(np.abs(d['dpdx'][mask])))
    # ---- D115 calc_kea_3d / regrid_btoa_3d / DISSIP
    ke = io.load_kea(P('kea')); rg = io.load_rg3(P('rg3')); ds = io.load_dissip(P('dissip'))
    assert len(ke) == len(rg) == 2 and [k['site'] for k in ke] == [1, 2], ([k['site'] for k in ke], len(rg))
    for k, r in zip(ke, rg):
        s = k['site']
        kk = gf.calc_kea_3d(k['u'], k['v'], g)
        res[f'kea{s}'] = mx(kk, k['kea']); res[f'kea{s}_nbad'] = nbad(kk, k['kea'])
        b = gf.kea_bgrid(k['u'], k['v'])
        res[f'kea{s}_pre'] = mx(b[:, 1:], r['x_in'][:, 1:])                    # row 1 undefined
        o = gf.regrid_btoa_3d(r['x_in'], g)
        res[f'rg3_{s}'] = mx(o, r['x_out']); res[f'rg3_{s}_nbad'] = nbad(o, r['x_out'])
    dd = ds[0]
    dke, tn, _ = gf.dissip(ke[1]['u'], ke[1]['v'], ke[0]['kea'], dd['t_in'], dd['pk'], g)
    res['dissip_kea_saved'] = mx(ke[0]['kea'], dd['kea'])
    res['dissip_dke'] = mx(dke, dd['dke']); res['dissip_t'] = mx(tn, dd['t_out'])
    res['dissip_t_nbad'] = nbad(tn, dd['t_out'])
    res['dissip_dT'] = mx(dd['t_out'], dd['t_in']); res['dissip_dke_scale'] = float(np.max(np.abs(dd['dke'])))
    # ---- D116 recalc_agrid_uv (every call, valid cells I<=IMAXJ(J))
    rc = io.load_recalc(P('recalc'))
    m = np.zeros((IM, JM), bool)
    for j in range(JM):
        m[:g['imaxj'][j], j] = True
    worst = 0.0; nb = 0; sites = []
    for r in rc:
        ua, va = gf.recalc_agrid_uv(r['u'], r['v'], g)
        mm = np.broadcast_to(m[None], (LM, IM, JM))
        worst = max(worst, mx(ua[mm], r['ua'][mm]), mx(va[mm], r['va'][mm]))
        nb += nbad(ua[mm], r['ua'][mm]) + nbad(va[mm], r['va'][mm]); sites.append(r['site'])
    res['recalc'] = worst; res['recalc_nbad'] = nb; res['recalc_sites'] = sites
    # ---- D115 energy-fix block
    e = io.load_efix(P('efix'))
    e1, e2, e3 = e[1], e[2], e[3]
    for tag, ee in (('init', e1), ('final', e2)):
        se = gf.conserv_se(ee['ma'], ee['masum'], ee['pk'], ee['t'], ee['q'], ee['qci'], g)
        kea, _ = ff.conserv_ke(ee['ma'], ee['u'], ee['v'], fcm_g(g))
        res[f'se_{tag}'] = mx(se, ee['a']); res[f'ke_{tag}'] = mx(kea, ee['b'])
        res[f'se_{tag}_nbad'] = nbad(se, ee['a']); res[f'ke_{tag}_nbad'] = nbad(kea, ee['b'])
    f = gf.energy_fix(e1['a'], e1['b'], e2['a'], e2['b'], e2['masum'], e2['t'], e2['pk'], g)
    res['efix_dse'] = abs(f['dsepke'] - e3['dse']); res['efix_mmg'] = abs(f['mmglob'] - e3['mmg'])
    res['efix_sef'] = mx(f['sef2'], e3['a']); res['efix_kef'] = mx(f['kef2'], e3['b'])
    res['efix_t'] = mx(f['t'], e3['t']); res['efix_t_nbad'] = nbad(f['t'], e3['t'])
    res['efix_dse_val'] = e3['dse']; res['efix_dT'] = mx(e3['t'], e2['t'])
    return res


def fcm_g(g):
    """dyn_filter_ff.conserv_ke needs rapvs,rapvn,dxyp,dxyv,byim,dxyn,dxys,byaxyp: all present in g."""
    return g


def run_daily(date, g, ff_dir=FF_DEFAULT, imf_pow=False):
    """D117: validate the real end-of-day DAILY_ATMDYN call(s) (itime -> ffd_glue_daily_<itime>.bin)."""
    out = []
    mpg = fcm.ff_load(f"{ff_dir}/{date}/ffd_filt_consts.bin")
    for p in io.daily_files(date, ff_dir):
        d = io.load_daily(p)
        d1, d2 = d[1], d[2]
        o = gf.daily_atmdyn(d1['ma'], d1['masum'], g, d1['mdrya'], False, True, imf_pow, mp_g=mpg)
        r = dict(itime=d1['itime'], deltam=d1['deltam'], smass_diff=abs(o['smass'] - d1['smass']),
                 mdryanow_diff=abs(o['mdryanow'] - d1['mdryanow']), deltam_diff=abs(o['deltam'] - d1['deltam']),
                 ma=mx(o['ma'], d2['ma']), ma_nbad=nbad(o['ma'], d2['ma']),
                 masum=mx(o['matopmb']['masum'], d2['masum']), pedn=mx(o['matopmb']['pedn'], d2['pedn']),
                 chg=mx(d2['ma'], d1['ma']))
        out.append(r)
    return out


if __name__ == "__main__":
    imf = "--imf-pow" in sys.argv; ana = "--analytic-geom" in sys.argv
    worst = {}; stats = {}
    for date, it0 in DATES:
        g = geom(date, ana)
        for k in range(NSTEP):
            r = run_step(date, it0 + k, g, imf_pow=imf, stats=stats)
            bad = {n: v for n, v in r.items() if isinstance(v, (int, float)) and not n.endswith(('_scale', '_val', 'dT'))
                   and not n.startswith('trop_lrange') and v != 0 and not n.endswith('_dke_scale')}
            print(f"{date} itime={it0 + k}: nonzero {len(bad)}", {n: (f'{v:.2e}' if isinstance(v, float) else v)
                                                                   for n, v in bad.items()},
                  "recalc sites", r['recalc_sites'])
            for n, v in r.items():
                if isinstance(v, (int, float)):
                    worst[n] = max(worst.get(n, 0), abs(v))
        for r in run_daily(date, g, imf_pow=imf):
            print(f"{date} DAILY itime={r['itime']}:", {n: v for n, v in r.items()})
    print("imf_pow =", imf, "analytic_geom =", ana)
    print("WORST:", {k: f"{v:.2e}" for k, v in worst.items() if v != 0})
    print("tropwmo branch stats:", stats)
