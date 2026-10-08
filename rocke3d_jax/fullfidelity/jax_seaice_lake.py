"""D190 (stage S5): the sea-ice / lake / land-ice glue of surface_loop.py (PRECIP_SI, AG2OG_precip, TOC2SST, PRECIP_LI, GROUND_LI, IRRIG_LK,
PRECIP_LK, GROUND_LK, UNDERICE, GROUND_SI, FORM_SI, CALC_APRESS, seaice_to_atmgrid) as device-resident jax.numpy functions on the FULL (IM, JM) grid.

NEW module; nothing existing is edited.  The NumPy originals gather the cells of a mask with np.nonzero, call the (already jitted, elementwise) kernels of
seaice_core_jax / lakes_core_jax / underice_jax / landice_precip_jax on the gathered vectors and scatter the results back.  Here the same kernels run on the
whole grid and the result is selected with jnp.where(mask, new, old): every kernel is elementwise, so each selected cell sees the same operations in the
same order.  The expressions of the NumPy glue (lndice, irrig_lk, ground_lk's runoff part, toc2sst) are transcribed term by term.
Exponents: the only integer powers in this glue are `** 2` (equal in NumPy and jnp); jnp.power(x, 4.0) is used for anything above 2 (D188).
Static masks / constants are passed in a dict `K` built by `make_static` (numpy arrays closed over by the jitted functions: compile-time constants).
Pole rows (ocean domain) are replicated from i = 1 as `_pole_replicate` of surface_loop, with Python booleans fixed by the static geometry.
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

import ocean_step as O  # noqa: E402
import seaice_core_jax as SI  # noqa: E402
import landice_precip_jax as LIP  # noqa: E402
import lakes_core_jax as LK  # noqa: E402
import underice_jax as UI  # noqa: E402
import apress_jax as AP  # noqa: E402
import surface_loop as L  # noqa: E402
from seaice_to_atmgrid_jax import seaice_to_atmgrid_cell  # noqa: E402

IM, JM, LMO = 72, 46, 13
TF = 273.15
DTSRC = 1800.0
ACE1I, XSI = SI.ACE1I, SI.XSI
SHW, RHOW, RHOI, RHOWS = SI.SHW, 1000.0, SI.RHOI, SI.RHOWS
FLEADOC, FLEADLK = 0.06, 0.0
LHM, SHI = 3.34e5, 2060.0
ACE1LI, ACE2LI, HC2LI = L.ACE1LI, L.ACE2LI, L.HC2LI
TEENY = 1e-30
MINMLD = LK.MINMLD


def make_static(st):
    """Static fields of the post-tile stage: numpy copies (closed over by jit) from surface_loop.load_statics' dict `st`."""
    geo = st['geo']
    K = {}
    for k in ('focean', 'flake', 'fwater'):
        K[k] = np.asarray(geo[k], np.float64)
    for k in ('is_ocean', 'is_lake', 'valid'):
        K[k] = np.asarray(geo[k], bool)
    K['valid_lake'] = K['valid'] & K['is_lake']
    K['valid_ocean'] = K['valid'] & K['is_ocean']
    for k in ('axyp', 'fland', 'flice', 'fearth', 'fgeotherm', 'coriol', 'hlake'):
        K[k] = np.asarray(st[k], np.float64)
    K['dxypo'] = np.asarray(O._DXYPO, np.float64)
    # OFTAB tables (record 2 of TEMGS, record 3 of SHCGS)
    K['tgsp'] = O.oftab_record(1).reshape((43, 41, 40), order='F')[:, :, 0].copy()
    K['cgs'] = O.oftab_record(2).reshape((43, 41), order='F').copy()
    K['pole_ocean'] = tuple(bool(K['is_ocean'][0, j]) for j in (0, JM - 1))
    K['pole_ocean_f'] = tuple(bool(K['focean'][0, j] > 0) for j in (0, JM - 1))
    return K


# ------------------------------------------------------------------------------------------------ seawater tables
def _lookup(tab, g, s, two_d=False):
    gg = g * 2.5e-4
    ss = s * 1000.0
    ig = jnp.clip(jnp.trunc(gg + 2.0).astype(jnp.int64) - 2, -2, 39)
    js = jnp.trunc(ss).astype(jnp.int64)
    js = jnp.where(js >= 40, 39, js)
    t = jnp.asarray(tab)
    c = lambda i, j: t[i + 2, j]   # noqa: E731
    return ((js - ss + 1) * ((ig - gg + 1) * c(ig, js) + (gg - ig) * c(ig + 1, js))
            + (ss - js) * ((ig - gg + 1) * c(ig, js + 1) + (gg - ig) * c(ig + 1, js + 1)))


def temgs(K, g, s):
    return _lookup(K['tgsp'], g, s)


def shcgs(K, g, s):
    return _lookup(K['cgs'], g, s)


def toc2sst(K, s):
    """surface_loop.toc2sst on device arrays (s: ocean dict with mo, g0m, s0m, ogeoz[, ogeoz_sv])."""
    foc = jnp.asarray(K['focean'])
    sel = foc > 0
    dx = jnp.asarray(K['dxypo'])[None, :]
    mo1 = jnp.where(sel, s['mo'][:, :, 0], 1.0)
    g1 = jnp.where(sel, s['g0m'][:, :, 0] / (mo1 * dx), 0.0)
    s1 = jnp.where(sel, s['s0m'][:, :, 0] / (mo1 * dx), 0.0)
    out = dict(gtemp=jnp.where(sel, temgs(K, g1, s1), 0.0), sss=jnp.where(sel, 1e3 * s1, 0.0),
               mlhc=jnp.where(sel, mo1 * shcgs(K, g1, s1), 0.0))
    mo2 = jnp.where(sel, s['mo'][:, :, 1], 1.0)
    g2 = jnp.where(sel, s['g0m'][:, :, 1] / (mo2 * dx), 0.0)
    s2 = jnp.where(sel, s['s0m'][:, :, 1] / (mo2 * dx), 0.0)
    out['gtemp2'] = jnp.where(sel, temgs(K, g2, s2), 0.0)
    out['ogeoza'] = jnp.where(sel, 0.5 * (s['ogeoz'] + s.get('ogeoz_sv', s['ogeoz'])), 0.0)
    for k in ('gtemp', 'gtemp2', 'sss', 'mlhc', 'ogeoza'):
        for j, flag in zip((0, JM - 1), K['pole_ocean_f']):
            if flag:
                out[k] = out[k].at[1:, j].set(out[k][0, j])
    out['gtempr'] = jnp.where(sel, out['gtemp'] + TF, 0.0)
    return out


# ------------------------------------------------------------------------------------------------ ice helpers
def pole_replicate(K, ice):
    ice = dict(ice)
    for j, flag in zip((0, JM - 1), K['pole_ocean']):
        if flag:
            for k in ('rsi', 'snowi', 'msi'):
                ice[k] = ice[k].at[1:, j].set(ice[k][0, j])
            for k in ('hsi', 'ssi'):
                ice[k] = ice[k].at[1:, j, :].set(ice[k][0, j, :][None, :])
    return ice


def precip_si(K, ice, prec, eprec):
    """surface_loop.precip_si.  Returns (ice', dict runpsi, srunpsi, erunpsi)."""
    fw = jnp.asarray(K['fwater'])
    poice = jnp.where(jnp.asarray(K['valid']), ice['rsi'] * fw, 0.0)
    sel = poice > 0
    r = SI.prec_si(ice['snowi'], ice['msi'], ice['hsi'], ice['ssi'], prec, eprec)
    new = dict(ice)
    s3 = sel[:, :, None]
    new['snowi'] = jnp.where(sel, r['snow'], ice['snowi'])
    new['msi'] = jnp.where(sel, r['msi2'], ice['msi'])
    new['hsi'] = jnp.where(s3, r['hsil'], ice['hsi'])
    new['ssi'] = jnp.where(s3, r['ssil'], ice['ssi'])
    z = jnp.zeros((IM, JM))
    runpsi = jnp.where(sel, r['run0'], z)
    srunpsi = jnp.where(sel, r['srun0'], z)
    erunpsi = jnp.where(sel, r['erun0'], z)
    wet = r['wetsnow'].astype(bool)
    fl = ice['flag_dsws'] | wet
    fl = jnp.where((~wet) & (prec > 0.0), False, fl)
    new['flag_dsws'] = jnp.where(sel, fl, ice['flag_dsws'])
    new['pond_melt'] = jnp.where(sel, ice['pond_melt'] + 0.3 * r['run0'], ice['pond_melt'])
    return new, dict(runpsi=runpsi, srunpsi=srunpsi, erunpsi=erunpsi)


def ag2og_precip(K, prec, eprec, ice, pi):
    rsi = ice['rsi']
    w0 = (1.0 - rsi) > 0
    z = 0.0
    return dict(oprec=jnp.where(w0, prec, z), oeprec=jnp.where(w0, eprec, z), orsi=rsi,
                orunpsi=jnp.where(rsi > 0, pi['runpsi'], z), oerunpsi=jnp.where(rsi > 0, pi['erunpsi'], z),
                osrunpsi=jnp.where(rsi > 0, pi['srunpsi'], z))


def seaice_to_atmgrid(K, ice):
    return seaice_to_atmgrid_cell(ice['rsi'], ice['snowi'], ice['msi'], ice['hsi'][..., 0], ice['hsi'][..., 1],
                                  ice['ssi'][..., 0], ice['ssi'][..., 1], ice['ssi'][..., 2], ice['ssi'][..., 3])


def underice(K, ice, gtemp, sss, mlhc, ui2rho, mldlk, dlake, glake, valid_mask, ustar_override=None):
    """surface_loop.underice for the domain `valid_mask` (static bool, lake or ocean sub-domain of geo_sub)."""
    fw = jnp.asarray(K['fwater'])
    oc = jnp.asarray(K['is_ocean'])
    cor = jnp.asarray(K['coriol'])
    sel = (ice['rsi'] * fw > 0) & jnp.asarray(valid_mask)
    msi = ice['msi']
    h4 = ice['hsi'][:, :, 3]
    s4 = ice['ssi'][:, :, 3]
    dh = 0.5 * (XSI[3] * msi) / RHOI
    tm = gtemp
    si_ = 1e3 * s4 / (XSI[3] * msi)
    tic_o = SI.Ti(h4 / (XSI[3] * msi), si_)
    ustar = jnp.maximum(5e-4, jnp.sqrt(ui2rho / RHOWS)) if ustar_override is None else ustar_override
    ro = UI.iceocean_fluxes(tic_o, si_, tm, sss, dh, ustar, cor, DTSRC, mlhc)
    tic_l = SI.Ti(h4 / (XSI[3] * msi), jnp.zeros_like(msi))
    mlsh = SHW * mldlk * RHOW
    rl = UI.icelake_fluxes_limited(tic_l, tm, dh, DTSRC, mlsh, dlake, glake)
    mflux = jnp.where(oc, ro['mflux'], rl['mflux'])
    hflux = jnp.where(oc, ro['hflux'], rl['hflux'])
    sflux = jnp.where(oc, ro['sflux'], 0.0)
    z = 0.0
    return (jnp.where(sel, mflux * DTSRC, z), jnp.where(sel, hflux * DTSRC, z), jnp.where(sel, sflux * DTSRC, z))


def ground_si(K, ice, e0, e1, evapor, solar, fmsi, fhsi, fssi, gtemp, sss, valid_mask):
    """surface_loop.ground_si on the sub-domain valid_mask.  Returns (ice', dict runosi, erunosi, srunosi, solar_io)."""
    fw = jnp.asarray(K['fwater'])
    oc = jnp.asarray(K['is_ocean'])
    sel = (ice['rsi'] * fw > 0) & jnp.asarray(valid_mask)
    snow, msi2, hsil, ssil, wet = ice['snowi'], ice['msi'], ice['hsi'], ice['ssi'], ice['flag_dsws']
    tm = gtemp
    sm = jnp.where(oc, sss, 0.0)
    si = SI.sea_ice(DTSRC, snow, hsil, ssil, msi2, e0, e1, evapor, solar, fmsi, fhsi, fssi, wet)
    dec = SI.ssidec(si['snow'], si['msi2'], si['hsil'], si['ssil'], DTSRC, si['melt12'])
    sic = SI.snowice(tm, sm, dec['snow'], dec['msi2'], dec['hsil'], dec['ssil'], False)
    m = oc
    snow_n = jnp.where(m, sic['snow'], si['snow'])
    msi2_n = jnp.where(m, sic['msi2'], si['msi2'])
    hsil_n = jnp.where(m[..., None], sic['hsil'], si['hsil'])
    ssil_n = jnp.where(m[..., None], sic['ssil'], si['ssil'])
    run = jnp.where(m, fmsi + si['run'] + dec['mflux'] + sic['msnwic'], fmsi + si['run'])
    erun = jnp.where(m, fhsi + si['erun'] + dec['hflux'] + sic['hsnwic'], fhsi + si['erun'])
    srun = jnp.where(m, fssi + si['srun'] + dec['sflux'] + sic['ssnwic'], fssi + si['srun'])
    new = dict(ice)
    s3 = sel[:, :, None]
    new['snowi'] = jnp.where(sel, snow_n, ice['snowi'])
    new['msi'] = jnp.where(sel, msi2_n, ice['msi'])
    new['hsi'] = jnp.where(s3, hsil_n, ice['hsi'])
    new['ssi'] = jnp.where(s3, ssil_n, ice['ssi'])
    new['flag_dsws'] = jnp.where(sel, si['wetsnow'].astype(bool), ice['flag_dsws'])
    z = 0.0
    out = dict(runosi=jnp.where(sel, run, z), erunosi=jnp.where(sel, erun, z), srunosi=jnp.where(sel, srun, z), solar_io=jnp.where(sel, si['srox2'], z))
    # pond_melt (SEAICE_DRV.f:~735-755)
    melt12 = si['melt12']
    msi1 = snow_n + ACE1I
    mice1 = jnp.where(ACE1I > XSI[1] * msi1, ACE1I - XSI[1] * msi1, 0.0)
    snowl1 = jnp.where(ACE1I > XSI[1] * msi1, snow_n, XSI[0] * msi1)
    m1s = jnp.where(mice1 != 0.0, mice1, 1.0)
    ti1a = SI.Ti2b(hsil_n[..., 0] / (XSI[0] * msi1), 1e3 * ssil_n[..., 0] / m1s, snowl1, m1s)
    ti1b = SI.Ti(hsil_n[..., 0] / (XSI[0] * (snow_n + ACE1I)), jnp.zeros_like(snow_n))
    ti1 = jnp.where(mice1 != 0.0, ti1a, ti1b)
    pm = ice['pond_melt'] + 0.3 * melt12
    pm = jnp.minimum(pm, 0.5 * (msi2_n + snow_n + ACE1I))
    pm = jnp.where(melt12 > 0, pm * (1.0 - DTSRC / (30.0 * 86400.0)), pm * (1.0 - DTSRC / (10.0 * 86400.0)))
    pm = jnp.where(ti1 < -10.0, 0.0, pm)
    new['pond_melt'] = jnp.where(sel, pm, ice['pond_melt'])
    return new, out


def form_si(K, ice, dmsi, dhsi, dssi, valid_mask):
    """surface_loop.form_si (ADDICE) on the sub-domain valid_mask; dmsi/dhsi/dssi (2, IM, JM)."""
    fw = jnp.asarray(K['fwater'])
    oc = jnp.asarray(K['is_ocean'])
    sel = (fw > 0) & jnp.asarray(valid_mask)
    flead = jnp.where(oc, FLEADOC, FLEADLK)
    r = SI.addice(ice['snowi'], ice['rsi'], ice['hsi'], ice['ssi'], ice['msi'], dhsi[0], dmsi[0], dmsi[1], dhsi[1], dssi[0], dssi[1], flead, False)
    new = dict(ice)
    s3 = sel[:, :, None]
    new['snowi'] = jnp.where(sel, r['snow'], ice['snowi'])
    new['msi'] = jnp.where(sel, r['msi2'], ice['msi'])
    new['hsi'] = jnp.where(s3, r['hsil'], ice['hsi'])
    new['ssi'] = jnp.where(s3, r['ssil'], ice['ssi'])
    new['rsi'] = jnp.where(sel, r['roice'], ice['rsi'])
    return pole_replicate(K, new)


def calc_apress(K, srfp, ice, grav=O.GRAV):
    ap = AP.calc_apress(srfp, ice['rsi'], ice['snowi'], ice['msi'], grav)
    for j in (0, JM - 1):
        ap = ap.at[1:, j].set(ap[0, j])
    return ap


# ------------------------------------------------------------------------------------------------ land ice
def precip_li(K, li, prec, eprec):
    r = LIP.precip_li(jnp.asarray(K['flice']), prec, eprec, li['snowli'], li['tlandi'][..., 0], li['tlandi'][..., 1])
    act = (jnp.asarray(K['flice']) > 0) & (prec > 0) & jnp.asarray(K['valid'])
    new = dict(li)
    new['snowli'] = jnp.where(act, r['snow'], li['snowli'])
    t = li['tlandi']
    new['tlandi'] = jnp.stack([jnp.where(act, r['tg1'], t[..., 0]), jnp.where(act, r['tg2'], t[..., 1])], axis=-1)
    z = 0.0
    return new, dict(runo=jnp.where(act, r['runo'], z), implm=jnp.where(act, r['implm'], z), implh=jnp.where(act, r['implh'], z),
                     e1=jnp.where(act, r['e1'], z))


def lndice(snow, tg1, tg2, f0dt, f1dt, evap):
    snandi = snow + ACE1LI - evap
    hc1 = snandi * SHI
    enrg1 = f0dt + evap * (tg1 * SHI - LHM) - f1dt
    hot = enrg1 > -tg1 * hc1
    run0 = jnp.where(hot, (enrg1 + tg1 * hc1) / LHM, 0.0)
    tg1n = jnp.where(hot, 0.0, tg1 + enrg1 / jnp.where(hc1 == 0, 1.0, hc1))
    snandi = jnp.where(hot, snandi - run0, snandi)
    thin = snandi < ACE1LI
    difs = jnp.where(thin, snandi - ACE1LI, 0.0)
    tg1n = jnp.where(thin, (tg1n * snandi - tg2 * difs) / ACE1LI, tg1n)
    edifs = jnp.where(thin, difs * (tg2 * SHI - LHM), 0.0)
    snow_n = jnp.where(thin, 0.0, snandi - ACE1LI)
    tg2n = tg2 + f1dt / HC2LI
    return snow_n, tg1n, tg2n, edifs, difs, run0


def ground_li(K, li, e0, e1, evapor):
    act = (jnp.asarray(K['flice']) > 0) & jnp.asarray(K['valid'])
    s, t1, t2, edifs, difs, run0 = lndice(li['snowli'], li['tlandi'][..., 0], li['tlandi'][..., 1], e0, e1, evapor)
    new = dict(li)
    new['snowli'] = jnp.where(act, s, li['snowli'])
    t = li['tlandi']
    new['tlandi'] = jnp.stack([jnp.where(act, t1, t[..., 0]), jnp.where(act, t2, t[..., 1])], axis=-1)
    return new, dict(runo=jnp.where(act, run0, 0.0), implm=jnp.where(act, difs, 0.0), implh=jnp.where(act, edifs, 0.0),
                     e1=jnp.where(act, edifs + e1, 0.0))


# ------------------------------------------------------------------------------------------------ lakes
def irrig_lk(K, lake, irrig_act):
    flake = jnp.asarray(K['flake'])
    axyp = jnp.asarray(K['axyp'])
    fland = jnp.asarray(K['fland'])
    m_to_kg = RHOW * axyp
    act = (fland > 0) & jnp.asarray(K['valid']) & (irrig_act > TEENY)
    mwl, gml, mld, tl = lake['mwl'], lake['gml'], lake['mldlk'], lake['tlake']
    isl = flake > 0
    m_avail = jnp.where(isl, jnp.maximum(mwl - MINMLD * flake * m_to_kg, 0.0), mwl)
    t_irr = jnp.where(isl, tl, gml / (mwl * SHW + TEENY))
    t_irr2 = jnp.where(isl, jnp.where(mwl > flake * mld * m_to_kg + TEENY,
                                      (gml - mld * m_to_kg * flake * tl * SHW) / (mwl - mld * m_to_kg * flake + TEENY) / SHW, 0.0), t_irr)
    t_irr = jnp.maximum(t_irr, 0.0)
    t_irr2 = jnp.maximum(t_irr2, 0.0)
    m_irr_pot = irrig_act * m_to_kg * DTSRC
    m_irr = jnp.where(m_avail <= TEENY, 0.0, jnp.minimum(m_irr_pot, m_avail))
    m_irr = jnp.where(act, m_irr, 0.0)
    l1 = mld * m_to_kg * flake
    two = isl & (m_irr > l1)
    g_irr = jnp.where(two, l1 * SHW * t_irr + (m_irr - l1) * SHW * t_irr2, m_irr * SHW * t_irr)
    do = m_irr > 0
    mwl_n = jnp.where(do, mwl - m_irr, mwl)
    gml_n = jnp.where(do, gml - g_irr, gml)
    fsafe = jnp.where(isl, flake, 1.0)
    one = do & isl & (m_irr < l1)
    mld1 = mld - m_irr / (fsafe * axyp * RHOW)
    m1 = mld1 * RHOW * flake * axyp
    m2 = jnp.maximum(mwl_n - m1, 0.0)
    up = one & (mld1 < MINMLD) & (m2 > 0)
    e1 = tl * SHW * m1
    e2 = gml_n - e1
    dm = jnp.maximum(MINMLD * RHOW * flake * axyp - m1, 0.0)
    de = dm * e2 / (m2 + TEENY)
    tl_up = (e1 + de) / ((m1 + dm) * SHW)
    mld_up = mld1 + dm / (fsafe * axyp * RHOW)
    allgone = do & isl & ~(m_irr < l1)
    mld_all = mwl_n / (fsafe * axyp * RHOW)
    tl_all = gml_n / (mwl_n * SHW + TEENY)
    mld_new = jnp.where(up, mld_up, jnp.where(one, mld1, jnp.where(allgone, mld_all, mld)))
    tl_new = jnp.where(up, tl_up, jnp.where(allgone, tl_all, tl))
    new = dict(lake)
    new.update(mwl=mwl_n, gml=gml_n, mldlk=mld_new, tlake=tl_new)
    return new, dict(mwl_to_irrig=m_irr, gml_to_irrig=jnp.where(do, g_irr, 0.0))


def precip_lk(K, lake, ice, prec, eprec, runpsi, runo_li, melti, emelti, atm_g):
    flake = jnp.asarray(K['flake'])
    flice = jnp.asarray(K['flice'])
    r = LK.precip_lk(flake, flice, ice['rsi'], prec, eprec, runpsi, runo_li, melti, emelti, jnp.asarray(K['axyp']),
                     lake['mwl'], lake['gml'], lake['tlake'], lake['mldlk'], atm_g['gtemp'], atm_g['gtemp2'], atm_g['gtempr'])
    active = ((flake + flice) > 0) & jnp.asarray(K['valid'])
    out = {k: jnp.where(active, r[k], lake[k]) for k in ('mwl', 'gml', 'tlake', 'mldlk')}
    lk = active & (flake > 0)
    g = {k: jnp.where(lk, r[k], atm_g[k]) for k in ('gtemp', 'gtemp2', 'gtempr')}
    return out, g, dict(dlake=jnp.where(active, r['dlake'], 0.0), glake=jnp.where(active, r['glake'], 0.0))


def ground_lk(K, lake, ice, runo_li, rune, erune, e0_o, evap_o, solar_o, runosi, erunosi, solar_io, valid_mask):
    """surface_loop.ground_lk (no tracers, TKE = 0).  valid_mask: the `geo['valid']` of the call (full domain with RIVERF, lake-only without)."""
    flake = jnp.asarray(K['flake'])
    axyp = jnp.asarray(K['axyp'])
    fland, flice, fearth = (jnp.asarray(K[k]) for k in ('fland', 'flice', 'fearth'))
    valid = jnp.asarray(valid_mask)
    hlake = jnp.asarray(K['hlake'])
    mwl, gml, tlake, mldlk = lake['mwl'], lake['gml'], lake['tlake'], lake['mldlk']
    fl_ok = (fland > 0) & valid
    run0 = runo_li * flice + rune * fearth
    erun0 = erune * fearth
    egeo = jnp.asarray(K['fgeotherm']) * flake * DTSRC
    mwl = jnp.where(fl_ok, mwl + run0 * axyp, mwl)
    gml = jnp.where(fl_ok, gml + erun0 * axyp + egeo * axyp, gml)
    lk = fl_ok & (flake > 0)
    fsafe = jnp.where(flake > 0, flake, 1.0)
    hlk1 = tlake * mldlk * RHOW * SHW
    mld_new = mldlk + run0 / (fsafe * RHOW)
    tl_new = (hlk1 * fsafe + erun0 + egeo) / (mld_new * fsafe * RHOW * SHW)
    nl = fl_ok & ~(flake > 0)
    tl_nl = gml / (mwl * SHW + TEENY)
    mldlk = jnp.where(lk, mld_new, mldlk)
    tlake = jnp.where(lk, tl_new, jnp.where(nl, tl_nl, tlake))
    act = (flake > 0) & valid
    roice = ice['rsi']
    fk, ax = flake, axyp
    ml1 = mldlk * RHOW
    ml2 = jnp.maximum(mwl / (fk * ax) - ml1, 0.0)
    el1 = tlake * SHW * ml1
    el2 = gml / (fk * ax) - el1
    thin = ml2 < 1e-10
    ml1 = jnp.where(thin, ml1 + ml2, ml1)
    el1 = jnp.where(thin, el1 + el2, el1)
    ml2 = jnp.where(thin, 0.0, ml2)
    el2 = jnp.where(thin, 0.0, el2)
    fsr2 = jnp.minimum(jnp.exp(-mldlk * LK.BYZETA), ml2 / (ml1 + ml2))
    src = LK.lksourc_full(roice, ml1, ml2, el1, el2, runosi, e0_o, erunosi, solar_o, solar_io, fsr2, evap_o)
    mix = LK.lkmix(src['mlake0'], src['mlake1'], src['elake0'], src['elake1'], hlake, jnp.zeros_like(roice), roice, DTSRC)
    m0, m1, e0_, e1_ = mix['mlake0'], mix['mlake1'], mix['elake0'], mix['elake1']
    mwl_n = jnp.where(act, (m0 + m1) * (fk * ax), mwl)
    gml_n = jnp.where(act, (e0_ + e1_) * (fk * ax), gml)
    mld = m0 / RHOW
    mld = jnp.where(m1 == 0.0, jnp.minimum(MINMLD, mld), mld)
    mldlk_n = jnp.where(act, mld, mldlk)
    tl = e0_ / (SHW * m0)
    tlake_n = jnp.where(act, tl, tlake)
    tlk2 = jnp.where(m1 > 0, e1_ / (SHW * jnp.where(m1 > 0, m1, 1.0)), tl)
    z = jnp.zeros((IM, JM))
    gt = dict(gtemp=jnp.where(act, tl, z), gtemp2=jnp.where(act, tlk2, z), gtempr=jnp.where(act, tl + TF, z))
    dmsi = jnp.stack([jnp.where(act, src['acefo'], z), jnp.where(act, src['acefi'], z)])
    dhsi = jnp.stack([jnp.where(act, src['enrgfo'], z), jnp.where(act, src['enrgfi'], z)])
    dssi = jnp.zeros((2, IM, JM))
    new = dict(lake)
    new.update(mwl=mwl_n, gml=gml_n, tlake=tlake_n, mldlk=mldlk_n)
    return new, gt, dict(dmsi=dmsi, dhsi=dhsi, dssi=dssi)
