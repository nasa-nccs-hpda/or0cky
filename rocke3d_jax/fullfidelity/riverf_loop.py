"""D167: hook of riverf_ff into the D164 surface loop (NEW file; surface_loop.py is not modified).

`surface_post_riverf` is a copy of surface_loop.surface_post with two stated differences:
  1. GROUND_LK is applied with the full-domain geo (D164 passed the lake-only geo, so the land-runoff addition of GROUND_LK, LAKES.f:3380-3390,
     was done for lake cells only; in the real code it is done for every FLAND>0 cell, and RIVERF then routes the MWL of all land cells);
  2. RIVERF is called between GROUND_LK and FORM_SI (SURFACE.f:1232-1236); oflowo/oeflowo are COMPUTED (not taken from inp['flows']);
     RIVERF's exports to the atmosphere (GTEMP, GTEMPR, MLHC of lake cells) are applied.
Everything else is identical to surface_post.  `run_free` runs the replay-mode (R1: real tile outputs as flux input) free run of D164 for N steps
with either variant and returns, per step, the computed-vs-recorded flows and the lake comparison against the real tile records of the next step.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface_loop as L  # noqa: E402
import riverf_ff as RF  # noqa: E402
import ocean_step as O  # noqa: E402

IM, JM = L.IM, L.JM


def surface_post_riverf(S, st, inp, mid, rs, ocean_stages=None, dynsi=None, record_flows=False, runoff_all=True, apply_riverf=True):
    """runoff_all=False reproduces D164's lake-only GROUND_LK runoff; apply_riverf=False skips the RIVERF state update (isolation switches)."""
    geo, ctx = st['geo'], st['ctx']
    acc = inp['acc']
    gl, gol = L.geo_sub(geo, 'lake'), L.geo_sub(geo, 'ocean')
    ice, atm, lake, li = S['ice'], {k: v.copy() for k, v in S['atm'].items()}, S['lake'], S['li']
    li, gli = L.ground_li(li, acc['e0_li'], acc['e1_li'] + mid['pli']['e1'], acc['evap_li'], st['flice'], geo)
    fm, fh, fs = L.underice(ice, atm['gtemp'], atm['sss'], atm['mlhc'], np.zeros((IM, JM)), st['coriol'], lake['mldlk'], mid['dl']['dlake'],
                            mid['dl']['glake'], gl)
    ice, gs_l = L.ground_si(ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], gl)
    lake, gt_l, dm = L.ground_lk(lake, ice, st['hlake'], st['axyp'], st['fland'], st['flice'], st['fearth'], st['fgeotherm'], gli['runo'],
                                 acc['rune'], acc['erune'], acc['e0_o'], acc['evap_o'], acc['solar_o'], gs_l['runosi'], gs_l['erunosi'],
                                 gs_l['solar_io'], geo if runoff_all else gl)
    for k in ('gtemp', 'gtemp2', 'gtempr'):
        atm[k] = np.where(geo['is_lake'], gt_l[k], atm[k])
    # RIVERF ---------------------------------------------------------------------------------------------------------
    lk_rv, flowo, eflowo, gtm, gtr, mlh, rdiag = RF.riverf(lake, geo['flake'], st['fland'], st['fearth'], rs)
    if apply_riverf:
        lake = {k: lk_rv[k] for k in ('mwl', 'gml', 'tlake', 'mldlk')}
        isl = geo['is_lake']
        atm['gtemp'] = np.where(isl, gtm, atm['gtemp']); atm['gtempr'] = np.where(isl, gtr, atm['gtempr']); atm['mlhc'] = np.where(isl, mlh, atm['mlhc'])
    ice = L.form_si(ice, dm['dmsi'], dm['dhsi'], dm['dssi'], gl)
    oc = S['ocean']
    dyn = inp['dynsi'] if dynsi is None else dynsi
    fm, fh, fs = L.underice(ice, atm['gtemp'], atm['sss'], atm['mlhc'], dyn.get('ui2rho', np.zeros((IM, JM))), st['coriol'], lake['mldlk'],
                            mid['dl']['dlake'], mid['dl']['glake'], gol, ustar_override=dyn.get('ustar'))
    ice, gs_o = L.ground_si(ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], gol)
    apress = L.calc_apress(inp['srfp'], ice, geo)
    rsi = ice['rsi']
    foc = geo['focean']
    sel = geo['is_ocean'] & geo['valid']
    inv_f = 1.0 / np.where(foc > 0, foc, 1.0)
    mt = mid['melt']
    fx = dict(mid['fxp'])
    fl = inp['flows'] if record_flows else dict(oflowo=flowo, oeflowo=eflowo)
    fx.update(oflowo=fl['oflowo'], oeflowo=fl['oeflowo'],
              omelti=np.where(sel, mt['melti'] * inv_f, 0.0), oemelti=np.where(sel, mt['emelti'] * inv_f, 0.0),
              osmelti=np.where(sel, mt['smelti'] * inv_f, 0.0),
              oevapor=np.where(sel, acc['evap_o'], 0.0), oe0=np.where(sel, acc['e0_o'], 0.0), osolarw=np.where(sel, acc['solar_o'], 0.0),
              orunosi=np.where(sel, gs_o['runosi'], 0.0), oerunosi=np.where(sel, gs_o['erunosi'], 0.0),
              osrunosi=np.where(sel, gs_o['srunosi'], 0.0), osolari=np.where(sel, gs_o['solar_io'], 0.0),
              oapress=np.where(geo['is_ocean'], apress, 0.0),
              odmua=np.where(sel, acc['dmua_o'] * (1.0 - rsi), 0.0), odmva=np.where(sel, acc['dmva_o'] * (1.0 - rsi), 0.0),
              odmui=dyn['odmui'], odmvi=dyn['odmvi'], orsi=rsi.copy(), itime=inp['itime'])
    ogeoz_entry = oc['ogeoz'].copy()
    for name, fn, _t in (ocean_stages or L.ocean_stages_after_precip()):
        oc = fn(oc, fx, ctx)
    oc['ogeoz_sv'] = ogeoz_entry
    ice = L.form_si(ice, oc['odmsi'], oc['odhsi'], oc['odssi'], gol)
    t = L.toc2sst(oc, ctx)
    for k in ('gtemp', 'gtemp2', 'gtempr', 'sss', 'mlhc'):
        atm[k] = np.where(geo['is_ocean'], t[k], atm[k])
    S2 = dict(S, ocean=oc, ice=ice, lake=lake, li=li, atm=atm, itime=S['itime'] + 1)
    return S2, dict(fx=fx, flowo=flowo, eflowo=eflowo, rdiag=rdiag, lake_pre_riverf=None)


def run_free(nsteps=6, variant='riverf', date='nov26', it0=33312, rs=None, log=print):
    # variants also: 'runoff_only' (all-land runoff, no RIVERF state update, recorded flows), 'riverf_lakeonly_runoff' (RIVERF on, D164 lake-only runoff)
    """variant 'd164': L.surface_post (recorded flows, lake-only runoff); 'riverf': surface_post_riverf (computed flows);
    'riverf_recflows': RIVERF applied to the lake state but the ocean receives the recorded flows (isolates the lake effect)."""
    st = L.load_statics(date); st['ctx'] = L.make_ocean_ctx(date)
    rs = rs or RF.load_statics(st['axyp'][0])
    S = L.init_surface_state(date, st=st)
    rows = []
    for k in range(nsteps):
        it = it0 + k
        inp = L.replay_inputs(date, it)
        S1, mid = L.surface_pre(S, st, inp)
        cmp = L.compare_pre_records(S1, mid, st, date, it)
        row = dict(it=it, pre={n: cmp[n] for n in ('lake.tg1', 'lake.mwl', 'lake.gml')})
        if variant == 'd164':
            S, post = L.surface_post(S1, st, inp, mid)
        else:
            S, post = surface_post_riverf(S1, st, inp, mid, rs, record_flows=(variant in ('riverf_recflows', 'runoff_only')),
                                          runoff_all=(variant != 'riverf_lakeonly_runoff'), apply_riverf=(variant != 'runoff_only'))
            fo, efo = post['flowo'], post['eflowo']
            r1, r2 = inp['flows']['oflowo'], inp['flows']['oeflowo']
            nz = r1 != 0
            row['flows'] = dict(bitwise_oflowo=int((fo == r1).sum()), bitwise_oeflowo=int((efo == r2).sum()), cells=int(fo.size), nonzero_real=int(nz.sum()),
                                nonzero_ours=int((fo != 0).sum()), maxabs_o=float(abs(fo - r1).max()), maxabs_eo=float(abs(efo - r2).max()),
                                scale_o=float(abs(r1).max()), scale_eo=float(abs(r2).max()), diag=post['rdiag'])
        rows.append(row)
        log(row)
    return rows, S
