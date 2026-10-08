"""D190 (stage S5): the post-tile half of the SURFACE stage and the ocean as ONE device-resident JAX program (composition module).

Pieces (all new modules, nothing existing edited): jax_seaice_lake (PRECIP_*, GROUND_*, FORM_SI, UNDERICE, IRRIG_LK, TOC2SST...), jax_riverf (RIVERF),
jax_dynsi (DYNSI + VPICEDYN loop), jax_advsi (ADVSI, with a double-double REAL*16 Ti2b), jax_ocean (OCEANS).  This module mirrors, line for line,
surface_loop.surface_pre + surface_loop_v2.surface_post_v2 (the NumPy reference of C1).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import jax_seaice_lake as SL  # noqa: E402
import jax_ocean as JO  # noqa: E402
import jax_riverf as JR  # noqa: E402
import jax_dynsi as JD  # noqa: E402
import jax_advsi as JA  # noqa: E402

IM, JM = 72, 46


def surface_pre_dev(K, S, melt_ice, melt, inp):
    """surface_loop.surface_pre on device arrays.  S: dict ocean, ice, lake, li, atm; (melt_ice, melt): result of MELT_SI (D186 / surface_loop.melt_si);
    inp: prec, eprec, irrig_act.  Returns (S1, mid)."""
    prec, eprec = inp['prec'], inp['eprec']
    ice, atm = melt_ice, S['atm']
    ice, pi = SL.precip_si(K, ice, prec, eprec)
    fxp = SL.ag2og_precip(K, prec, eprec, ice, pi)
    oc = JO.stage_precip(K, None, S["ocean"], fxp)
    t = SL.toc2sst(K, oc)
    atm = dict(atm)
    oo = jnp.asarray(K['is_ocean'])
    for k in ('gtemp', 'gtemp2', 'gtempr', 'sss', 'mlhc'):
        atm[k] = jnp.where(oo, t[k], atm[k])
    li, pli = SL.precip_li(K, S['li'], prec, eprec)
    lake0, irr = SL.irrig_lk(K, S['lake'], inp['irrig_act'])
    lake, gt_l, dl = SL.precip_lk(K, lake0, ice, prec, eprec, pi['runpsi'], pli['runo'], melt['melti'], melt['emelti'], atm)
    il = jnp.asarray(K['is_lake'])
    for k in ('gtemp', 'gtemp2', 'gtempr'):
        atm[k] = jnp.where(il, gt_l[k], atm[k])
    ag = SL.seaice_to_atmgrid(K, ice)
    S1 = dict(S, ocean=oc, ice=ice, lake=lake, li=li, atm=atm)
    return S1, dict(melt=melt, pi=pi, fxp=fxp, pli=pli, dl=dl, ag=ag, irr=irr)


OCEAN_STATE_KEYS = ('g0m', 's0m', 'gxmo', 'gymo', 'gzmo', 'sxmo', 'symo', 'szmo', 'mo', 'uo', 'vo', 'uod', 'vod', 'ogeoz', 'ogeoz_sv', 'kpl', 'must', 'g0mst', 'gxmst',
                    'gzmst', 's0mst', 'sxmst', 'szmst', 'mmst', 'mmi', 'smu', 'smv', 'smw', 'opbot', 'opress', 'vonp')


def make_static_all(st, date, it0, ff=None):
    """All host-side constants of the post-tile program (numpy; closed over by the jitted functions) and the big arrays passed as arguments.
    st: surface_loop.load_statics(date) with st['ctx'] = make_ocean_ctx(date).  Recorded static inputs (declared in the ledger entry): the five ADVSI geometry
    vectors (a real ffadv_in dump), the OFTAB tables, the river-direction and topography files read by riverf_ff.load_statics."""
    import surface_loop as L
    import surface_loop_advsi as AL
    import riverf_ff as RF
    from odhorz_ff import geomo_dyn_arrays
    ff = ff or L.FF
    geo = st['geo']
    K = SL.make_static(st)
    Ko, Kb = JO.make_static(st, K)
    K['oc'] = Ko
    sinvo, sinpo = geomo_dyn_arrays()[:2]
    K['dyn'] = JD.make_static(geo['focean'], sinpo, sinvo)
    rs = RF.load_statics(st['axyp'][0])
    K['riv'] = JR.make_static(rs, geo['flake'], st['fland'], st['fearth'])
    advgeo = AL.advsi_geo_from_dump(date, it0, ff)
    K['adv'] = JA.make_static(geo['focean'], advgeo)
    return K, Kb


def post_a(K, S, mid, acc, srfp, itime, V):
    """First part of surface_loop_v2.surface_post_v2 up to (excluding) the OCEANS stages: DYNSI, GROUND_LI, the lake chain (UNDERICE, GROUND_SI, GROUND_LK, RIVERF,
    FORM_SI), the ocean-driver head (UNDERICE, GROUND_SI, CALC_APRESS) and the flux dictionary of the ocean.  Returns a carry dict."""
    oo = jnp.asarray(K['is_ocean'])
    il = jnp.asarray(K['is_lake'])
    ice, atm, lake, li = S['ice'], dict(S['atm']), S['lake'], S['li']
    rsisave = ice['rsi']
    oc0 = S['ocean']
    # ---- DYNSI
    rsi_o, msi_o, snowi_o = (jnp.where(oo, ice[k], 0.0) for k in ('rsi', 'msi', 'snowi'))
    oga = SL.toc2sst(K, oc0)['ogeoza']
    d = JD.dynsi(K['dyn'], acc['dmua_i'], acc['dmva_i'], rsi_o, msi_o, snowi_o, oga, oc0['uo'][:, :, 0], oc0['vo'][:, :, 0], V['usi'], V['vsi'])
    dyn = dict(odmui=d['odmui'], odmvi=d['odmvi'], ustar=d['ustar'], ui2rho=d['ui2rho'])
    # ---- land ice, lake chain
    li, gli = SL.ground_li(K, li, acc['e0_li'], acc['e1_li'] + 0.0, acc['evap_li'])
    z = jnp.zeros((IM, JM))
    vl, vo_ = K['valid_lake'], K['valid_ocean']
    fm, fh, fs = SL.underice(K, ice, atm['gtemp'], atm['sss'], atm['mlhc'], z, lake['mldlk'], mid['dl']['dlake'], mid['dl']['glake'], vl)
    ice, gs_l = SL.ground_si(K, ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], vl)
    lake, gt_l, dm = SL.ground_lk(K, lake, ice, gli['runo'], acc['rune'], acc['erune'], acc['e0_o'], acc['evap_o'], acc['solar_o'], gs_l['runosi'],
                                  gs_l['erunosi'], gs_l['solar_io'], K['valid'])
    for k in ('gtemp', 'gtemp2', 'gtempr'):
        atm[k] = jnp.where(il, gt_l[k], atm[k])
    lk_rv, flowo, eflowo, gtm, gtr, mlh = JR.riverf(K['riv'], lake, None, None, None)
    lake = {k: lk_rv[k] for k in ('mwl', 'gml', 'tlake', 'mldlk')}
    atm['gtemp'] = jnp.where(il, gtm, atm['gtemp'])
    atm['gtempr'] = jnp.where(il, gtr, atm['gtempr'])
    atm['mlhc'] = jnp.where(il, mlh, atm['mlhc'])
    ice = SL.form_si(K, ice, dm['dmsi'], dm['dhsi'], dm['dssi'], vl)
    # ---- ocean driver
    fm, fh, fs = SL.underice(K, ice, atm['gtemp'], atm['sss'], atm['mlhc'], dyn['ui2rho'], lake['mldlk'], mid['dl']['dlake'], mid['dl']['glake'], vo_,
                             ustar_override=dyn['ustar'])
    ice, gs_o = SL.ground_si(K, ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], vo_)
    ice_after_gsi = ice
    apress = SL.calc_apress(K, srfp, ice)
    rsi = ice['rsi']
    foc = jnp.asarray(K['focean'])
    sel = jnp.asarray(K['valid_ocean'])
    inv_f = 1.0 / jnp.where(foc > 0, foc, 1.0)
    mt = mid['melt']
    fx = dict(mid['fxp'])
    fx.update(oflowo=flowo, oeflowo=eflowo,
              omelti=jnp.where(sel, mt['melti'] * inv_f, 0.0), oemelti=jnp.where(sel, mt['emelti'] * inv_f, 0.0),
              osmelti=jnp.where(sel, mt['smelti'] * inv_f, 0.0),
              oevapor=jnp.where(sel, acc['evap_o'], 0.0), oe0=jnp.where(sel, acc['e0_o'], 0.0), osolarw=jnp.where(sel, acc['solar_o'], 0.0),
              orunosi=jnp.where(sel, gs_o['runosi'], 0.0), oerunosi=jnp.where(sel, gs_o['erunosi'], 0.0),
              osrunosi=jnp.where(sel, gs_o['srunosi'], 0.0), osolari=jnp.where(sel, gs_o['solar_io'], 0.0),
              oapress=jnp.where(oo, apress, 0.0),
              odmua=jnp.where(sel, acc['dmua_o'] * (1.0 - rsi), 0.0), odmva=jnp.where(sel, acc['dmva_o'] * (1.0 - rsi), 0.0),
              odmui=dyn['odmui'], odmvi=dyn['odmvi'], orsi=rsi, itime=itime)
    return dict(fx=fx, ice=ice, atm=atm, lake=lake, li=li, rsisave=rsisave, d=d, ogeoz_entry=oc0['ogeoz'],
                diag=dict(flowo=flowo, eflowo=eflowo, ice_after_gsi=ice_after_gsi, gs_l=gs_l, gs_o=gs_o, gli=gli, apress=apress, dm=dm))


def post_b(K, S, c, oc, V):
    """Last part: ogeoz_sv, FORM_SI (ocean domain), TOC2SST, ADVSI and the pole replication.  c: carry of post_a; oc: ocean state after the OCEANS stages."""
    oo = jnp.asarray(K['is_ocean'])
    vo_ = K['valid_ocean']
    atm, ice = dict(c['atm']), c['ice']
    d = c['d']
    oc = dict(oc)
    oc['ogeoz_sv'] = c['ogeoz_entry']
    ice = SL.form_si(K, ice, oc['odmsi'], oc['odhsi'], oc['odssi'], vo_)
    ice_pre_adv = ice
    t = SL.toc2sst(K, oc)
    for k in ('gtemp', 'gtemp2', 'gtempr', 'sss', 'mlhc'):
        atm[k] = jnp.where(oo, t[k], atm[k])
    stt = dict(rsi=ice['rsi'], rsix=V['rsix'], rsiy=V['rsiy'], rsisave=c['rsisave'], msi=ice['msi'], snowi=ice['snowi'], hsi=ice['hsi'], ssi=ice['ssi'])
    new, aout = JA.advsi(K['adv'], stt, d['usi'], d['vsi'])
    ice2 = dict(ice)
    for k in ('rsi', 'msi', 'snowi', 'hsi', 'ssi'):
        ice2[k] = jnp.where(oo[..., None], new[k], ice[k]) if new[k].ndim == 3 else jnp.where(oo, new[k], ice[k])
    ice2 = SL.pole_replicate(K, ice2)
    V2 = dict(usi=d['usi'], vsi=d['vsi'], rsix=new['rsix'], rsiy=new['rsiy'], uisurf=d['uisurf'], visurf=d['visurf'])
    S2 = dict(S, ocean=oc, ice=ice2, lake=c['lake'], li=c['li'], atm=atm)
    post = dict(fx=c['fx'], dyn=d, ice_pre_adv=ice_pre_adv, advsi_out=aout, **c['diag'])
    return S2, V2, post


def surface_post_dev(K, Kb, S, mid, acc, srfp, itime, V):
    """surface_loop_v2.surface_post_v2 (dynsi='computed', riverf and advsi on) as ONE traceable function (post_a, the ten OCEANS stages, post_b).
    S: state after surface_pre_dev (dict ocean, ice, lake, li, atm); mid: its intermediates; acc: tile accumulators (device); srfp = PEDN(1);
    itime: traced int; V: dict usi, vsi, rsix, rsiy.  Returns (S2, V2, post)."""
    c = post_a(K, S, mid, acc, srfp, itime, V)
    oc = JO.ocean_stages(K, Kb, S['ocean'], c['fx'])
    return post_b(K, S, c, oc, V)


def tile_accumulators_dev(host, tpl, aux):
    """surface_loop.tile_outputs_to_acc + surface_loop_v2.ice_tile_dmua_from_tiles on the DEVICE outputs of the D188 stage (jax_surface.make_stage):
    the per-slot tile fluxes of both substeps are reduced to the (IM, JM) accumulators of the post-tile stage.  Invalid water slots (masked by wvalid) contribute
    nothing, exactly as the rows that do not exist in the record sets.  host/tpl: from jax_surface.build_template; aux: second return value of the stage."""
    import surface_tile_ff as ST
    Nw = host['Nw']
    wcells, lcells, ecells = (jnp.asarray(host[k]) for k in ('wcells', 'lcells', 'ecells'))

    def grid(cells, vals):
        return jnp.zeros(IM * JM).at[cells].set(vals).reshape(JM, IM).T
    parts = []
    for ns, (r, rows) in enumerate(((aux['r1'], aux['rows1']), (aux['r2'], aux['rows2']))):
        wv = tpl['ns'][ns]['wvalid']
        tw = rows['tw']
        dts = tw[:, ST.IN['dtsurf']]
        srh = tw[:, ST.IN['srheat']] * dts
        t = r['tile']

        def water(vals, ty, wv=wv):
            return grid(wcells, jnp.where(wv[ty], vals[ty * Nw:(ty + 1) * Nw], 0.0))
        gl = r['li']
        gh = r['land']['ghy']
        parts.append(dict(
            e0_o=water(t['f0dt'], 0), evap_o=water(t['evap'], 0), solar_o=water(srh, 0), dmua_o=water(t['dmua'] * dts, 0), dmva_o=water(t['dmva'] * dts, 0),
            e0_i=water(t['f0dt'], 1), e1_i=water(t['f1dt'], 1), evap_i=water(t['evap'], 1), solar_i=water(srh, 1),
            dmua_i=water(t['dmua'] * dts, 1), dmva_i=water(t['dmva'] * dts, 1),
            e0_li=grid(lcells, gl['f0dt']), e1_li=grid(lcells, gl['f1dt']), evap_li=grid(lcells, gl['evap']),
            rune=grid(ecells, gh['aruns'] + gh['arunu']), erune=grid(ecells, gh['aeruns'] + gh['aerunu'])))
    return {k: parts[0][k] + parts[1][k] for k in parts[0]}
