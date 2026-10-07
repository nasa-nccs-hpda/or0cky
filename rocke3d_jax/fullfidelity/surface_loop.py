"""D164: surface loop -- ocean / sea-ice / lake / land-ice / land surface state evolved from OUR computation (new module).

Scope and honesty (details and every recorded input: scoping/D164_SURFACE_LOOP_ENTRY.md):
  * Only pieces that already have validated ports are wired here (ocean_step chained ocean, seaice_core_jax, lakes_core_jax,
    precip_*_jax, apress_jax, seaice_to_atmgrid_jax, land_chain/ghy_jax); the glue between them (the drivers' gating,
    scaling and accumulation, TOC2SST, the exchange-grid copies) is transcribed here from the pristine ModelE source.
  * Nothing else is invented: ADVSI, the DYNSI input assembly, RIVERF, IRRIG_LK, GROUND_LI, GLMELT and Ent are NOT ported
    and appear as named boundaries (see the ledger entry).
Array convention: 0-based numpy, [i, j(, l)] with i = longitude (IM = 72), j = latitude (JM = 46), like ocean_step.py.
The restart file (ff_data/_pristine_restarts/fort1_<date>_itime<it>.nc) stores (J, I) arrays; they are transposed on load.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import ocean_step as O  # noqa: E402
import seaice_core_jax as SI  # noqa: E402
import seaice_core_ff as SIF  # noqa: E402
import atm_step as A  # noqa: E402

IM, JM, LMO = 72, 46, 13
FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data'
TF = 273.15
DTSRC = 1800.0
ACE1I = SI.ACE1I
XSI = SI.XSI

RESTART = {'nov26': 'fort1_nov26_itime33312.nc', 'dec01': 'fort1_dec01_itime33552.nc', 'jan01': 'fort1_jan01_itime17520.nc'}


# ------------------------------------------------------------------------------------------------ seawater functions
_TGSP = None


def temgs(g, s):
    """OCNFUNTAB.f TEMGS(G,S): bilinear lookup in the OFTAB temperature table (record 2, p index 0). Arrays any shape.
    Non-finite / out-of-range inputs must be masked by the caller (the table index is clamped like the Fortran)."""
    global _TGSP
    if _TGSP is None:
        _TGSP = O.oftab_record(1).reshape((43, 41, 40), order='F')[:, :, 0]
    gg = g * 2.5e-4
    ss = s * 1000.0
    ig = np.clip(np.trunc(gg + 2.0).astype(np.int64) - 2, -2, 39)
    js = np.trunc(ss).astype(np.int64)
    js = np.where(js >= 40, 39, js)
    t = lambda i, j: _TGSP[i + 2, j]   # noqa: E731
    return ((js - ss + 1) * ((ig - gg + 1) * t(ig, js) + (gg - ig) * t(ig + 1, js))
            + (ss - js) * ((ig - gg + 1) * t(ig, js + 1) + (gg - ig) * t(ig + 1, js + 1)))


def ocean_cell_sets(ctx):
    """focean>0 mask on the (IM, JM) grid."""
    return np.asarray(ctx['focean']) > 0


def toc2sst(s, ctx, g_prev=None):
    """OCNDYN.f:5577 TOC2SST -> get_exports_layer1 + OG2AG_TOC2SST (identity regrid on this grid; verified bitwise below):
    ATMOCN GTEMP, SSS, MLHC from layer 1, GTEMP2 from layer 2, OGEOZA, UOSURF/VOSURF (UOdrag=0: unused by the tiles).
    Returns dict of (IM, JM) arrays, defined where FOCEAN > 0 (zeros elsewhere); poles copied from i = 1 like the Fortran."""
    foc = np.asarray(ctx['focean'])
    sel = foc > 0
    dx = O._DXYPO[None, :]
    mo1 = np.where(sel, s['mo'][:, :, 0], 1.0)
    g1 = np.where(sel, s['g0m'][:, :, 0] / (mo1 * dx), 0.0)
    s1 = np.where(sel, s['s0m'][:, :, 0] / (mo1 * dx), 0.0)
    out = dict(gtemp=np.where(sel, temgs(g1, s1), 0.0), sss=np.where(sel, 1e3 * s1, 0.0),
               mlhc=np.where(sel, mo1 * O.shcgs(g1, s1), 0.0))
    mo2 = np.where(sel, s['mo'][:, :, 1], 1.0)
    g2 = np.where(sel, s['g0m'][:, :, 1] / (mo2 * dx), 0.0)
    s2 = np.where(sel, s['s0m'][:, :, 1] / (mo2 * dx), 0.0)
    out['gtemp2'] = np.where(sel, temgs(g2, s2), 0.0)
    out['ogeoza'] = np.where(sel, 0.5 * (s['ogeoz'] + s.get('ogeoz_sv', s['ogeoz'])), 0.0)
    for k in ('gtemp', 'gtemp2', 'sss', 'mlhc', 'ogeoza'):
        for j in (0, JM - 1):
            if foc[0, j] > 0:
                out[k][1:, j] = out[k][0, j]
    out['gtempr'] = np.where(sel, out['gtemp'] + TF, 0.0)
    return out


# ------------------------------------------------------------------------------------------------ restart state
def _r2(R, k):
    """(J, I) restart array -> (I, J)"""
    return np.array(R.variables[k][:], dtype=np.float64).T


def _r3(R, k):
    """(J, I, n) restart array -> (I, J, n)"""
    return np.transpose(np.array(R.variables[k][:], dtype=np.float64), (1, 0, 2))


def load_restart(date='nov26', ff=FF):
    """Full surface state at the start of the window, from the real restart (NOT a dump of an intermediate point):
    ocean (state dict of ocean_step: bitwise equal to ffo tag 0, checked), sea ice of the ocean domain (si_ocn) and of the
    atmosphere/lake domain (si_atm), lakes, land ice, GHY land state, and the atmosphere-grid exchange arrays that the restart stores."""
    import netCDF4 as nc
    R = nc.Dataset(f"{ff}/_pristine_restarts/{RESTART[date]}")
    oc = {}
    for k in ('g0m', 's0m', 'gxmo', 'gymo', 'gzmo', 'sxmo', 'symo', 'szmo', 'mo', 'uo', 'vo', 'uod', 'vod'):
        oc[k] = np.transpose(np.array(R.variables[k][:], dtype=np.float64), (2, 1, 0)).copy()
    oc['ogeoz'] = _r2(R, 'ogeoz').copy()
    oc['ogeoz_sv'] = _r2(R, 'ogeoz_sv').copy()
    oc['kpl'] = np.rint(_r2(R, 'kpl')).astype(np.int64)
    st = {}
    for k in ('must', 'g0mst', 'gxmst', 'gzmst', 's0mst', 'sxmst', 'szmst'):
        st[k] = np.array(R.variables[k][:], dtype=np.float64).T.copy()      # (LMO, NMST)
    oc.update(st)
    si_o = dict(rsi=_r2(R, 'rsi'), snowi=_r2(R, 'snowi'), msi=_r2(R, 'msi'), hsi=_r3(R, 'hsi'), ssi=_r3(R, 'ssi'),
                pond_melt=_r2(R, 'pond_melt'), flag_dsws=_r2(R, 'flag_dsws') > 0.5)
    si_a = dict(rsi=_r2(R, 'rsi_atm'), snowi=_r2(R, 'snowi_atm'), msi=_r2(R, 'msi_atm'), hsi=_r3(R, 'hsi_atm'),
                ssi=_r3(R, 'ssi_atm'), pond_melt=_r2(R, 'pond_melt_atm'), flag_dsws=_r2(R, 'flag_dsws_atm') > 0.5)
    lake = {k: _r2(R, k) for k in ('mwl', 'gml', 'tlake', 'mldlk', 'flake', 'T2Lbot')}
    li = dict(snowli=np.array(R.variables['snowli'][:], float)[0].T, tlandi=np.transpose(np.array(R.variables['tlandi'][:], float)[0], (1, 0, 2)))
    atm = {k: _r2(R, k) for k in ('asst', 'atempr', 'sss', 'uosurf', 'vosurf', 'mlhc', 'ogeoza')}
    ghy = dict(w=np.transpose(np.array(R.variables['w_ij'][:], float), (1, 0, 2, 3)),
               ht=np.transpose(np.array(R.variables['ht_ij'][:], float), (1, 0, 2, 3)),
               nsn=np.transpose(np.array(R.variables['nsn_ij'][:], float), (1, 0, 2)),
               dzsn=np.transpose(np.array(R.variables['dzsn_ij'][:], float), (1, 0, 2, 3)),
               wsn=np.transpose(np.array(R.variables['wsn_ij'][:], float), (1, 0, 2, 3)),
               hsn=np.transpose(np.array(R.variables['hsn_ij'][:], float), (1, 0, 2, 3)),
               fr_snow=np.transpose(np.array(R.variables['fr_snow_ij'][:], float), (1, 0, 2)))
    return dict(ocean=oc, si_ocn=si_o, si_atm=si_a, lake=lake, landice=li, atm=atm, ghy=ghy, itime=int(np.array(R.variables['itime'][:])))


# ------------------------------------------------------------------------------------------------ sea ice / lake ice (both domains)
# One combined set of (IM, JM) arrays: ocean-domain cells (FOCEAN > 0) hold si_ocn, lake-domain cells (FLAKE > 0) hold si_atm.
# FWATER = FOCEAN (ocean domain) or FLAKE (lake domain); the two never overlap (checked on the dumps).
ICE_KEYS = ('rsi', 'snowi', 'msi', 'hsi', 'ssi', 'pond_melt', 'flag_dsws')


def make_geo(ctx, flake, fland=None):
    """Static fields of the exchange grid. ctx: ocean ctx (focean, ...); flake: (IM, JM)."""
    foc = np.asarray(ctx['focean'], dtype=np.float64)
    fl = np.asarray(flake, dtype=np.float64)
    assert not ((foc > 0) & (fl > 0)).any()
    return dict(focean=foc, flake=fl, fwater=foc + fl, is_ocean=foc > 0, is_lake=fl > 0, valid=O.imaxj_mask())


def ice_from_restart(S, geo):
    o, a = S['si_ocn'], S['si_atm']
    oc = geo['is_ocean']
    ice = {}
    for k in ICE_KEYS:
        if o[k].ndim == 3:
            ice[k] = np.where(oc[:, :, None], o[k], a[k]).copy()
        else:
            ice[k] = np.where(oc, o[k], a[k]).copy()
    return ice


def copy_ice(ice):
    return {k: v.copy() for k, v in ice.items()}


def _pole_replicate(ice, geo):
    """ocean-domain pole rows are replicated from i = 1 (MELT_SI/FORM_SI end)."""
    for j in (0, JM - 1):
        if geo['is_ocean'][0, j]:
            for k in ('rsi', 'snowi', 'msi'):
                ice[k][1:, j] = ice[k][0, j]
            for k in ('hsi', 'ssi'):
                ice[k][1:, j, :] = ice[k][0, j, :]


def _J(a):
    return jnp.asarray(a)


def melt_si(ice, gtemp, sss, mlhc, geo):
    """ATM_DRV.f:257-258 MELT_SI(si_ocn) then MELT_SI(si_atm) (SEAICE_DRV.f:377-562) for both domains.
    Returns (new ice, dict melti/emelti/smelti)."""
    ice = copy_ice(ice)
    fw, oc = geo['fwater'], geo['is_ocean']
    roice = ice['rsi']
    dop = (fw * roice > 0) & (oc | (roice < 1.0)) & geo['valid']
    melti = np.zeros((IM, JM)); emelti = np.zeros((IM, JM)); smelti = np.zeros((IM, JM))
    ii, jj = np.nonzero(dop)
    if len(ii):
        pocean = np.where(oc, fw, 0.0)[ii, jj]
        tfo = np.where(oc, np.asarray(SI.tfrez(_J(sss))), 0.0)[ii, jj]
        tm = gtemp[ii, jj]
        enrgmax = np.maximum(tm - tfo, 0.0) * mlhc[ii, jj]
        r = SI.simelt(DTSRC, _J(roice[ii, jj]), _J(ice['snowi'][ii, jj]), _J(ice['msi'][ii, jj]), _J(ice['hsi'][ii, jj]),
                      _J(ice['ssi'][ii, jj]), _J(pocean), _J(tm), _J(tfo), _J(enrgmax))
        r = {k: np.asarray(v) for k, v in r.items()}
        ice['rsi'][ii, jj] = r['roice']; ice['msi'][ii, jj] = r['msi2']; ice['snowi'][ii, jj] = r['snow']
        ice['hsi'][ii, jj, :] = r['hsil']; ice['ssi'][ii, jj, :] = r['ssil']
        pw = fw[ii, jj]
        melti[ii, jj] = r['run0'] * pw; emelti[ii, jj] = -r['enrgused'] * pw; smelti[ii, jj] = r['salt'] * pw
    _pole_replicate(ice, geo)
    return ice, dict(melti=melti, emelti=emelti, smelti=smelti, dop=dop)


def seaice_to_atmgrid(ice, geo):
    """SEAICE_DRV.f:1716 seaice_to_atmgrid, second loop: GTEMP, GTEMP2, GTEMPR (atmice) from the ice state. Valid where there is
    ice; elsewhere the Fortran keeps stale values (not used)."""
    from seaice_to_atmgrid_jax import seaice_to_atmgrid_cell
    r = seaice_to_atmgrid_cell(_J(ice['rsi']), _J(ice['snowi']), _J(ice['msi']), _J(ice['hsi'][..., 0]), _J(ice['hsi'][..., 1]),
                               _J(ice['ssi'][..., 0]), _J(ice['ssi'][..., 1]), _J(ice['ssi'][..., 2]), _J(ice['ssi'][..., 3]))
    return {k: np.asarray(v) for k, v in r.items()}


def precip_si(ice, prec, eprec, geo):
    """MODELE.f:327 / SURFACE.f:298 PRECIP_SI (SEAICE_DRV.f:45-185) for both domains. prec/eprec: atmosphere PREC/EPREC (kg/m2, J/m2 per step)."""
    ice = copy_ice(ice)
    fw = geo['fwater']
    poice = np.where(geo['valid'], ice['rsi'] * fw, 0.0)
    runpsi = np.zeros((IM, JM)); srunpsi = np.zeros((IM, JM)); erunpsi = np.zeros((IM, JM))
    ii, jj = np.nonzero(poice > 0)
    if len(ii):
        pr = prec[ii, jj]; en = eprec[ii, jj]
        r = SI.prec_si(_J(ice['snowi'][ii, jj]), _J(ice['msi'][ii, jj]), _J(ice['hsi'][ii, jj]), _J(ice['ssi'][ii, jj]), _J(pr), _J(en))
        r = {k: np.asarray(v) for k, v in r.items()}
        ice['snowi'][ii, jj] = r['snow']; ice['msi'][ii, jj] = r['msi2']
        ice['hsi'][ii, jj, :] = r['hsil']; ice['ssi'][ii, jj, :] = r['ssil']
        runpsi[ii, jj] = r['run0']; srunpsi[ii, jj] = r['srun0']; erunpsi[ii, jj] = r['erun0']
        wet = r['wetsnow'].astype(bool)
        fl = ice['flag_dsws'][ii, jj] | wet
        fl = np.where((~wet) & (pr > 0.0), False, fl)
        ice['flag_dsws'][ii, jj] = fl
        ice['pond_melt'][ii, jj] = ice['pond_melt'][ii, jj] + 0.3 * r['run0']
    return ice, dict(runpsi=runpsi, srunpsi=srunpsi, erunpsi=erunpsi)


# ------------------------------------------------------------------------------------------------ atmosphere -> ocean exchange (AG2OG): identity regrid
def ag2og_precip(prec, eprec, ice, pi, geo):
    """OCN_Interp.f:128 AG2OG_precip. The atmosphere and ocean grids are the same (72 x 46), so the weighted regrid is the identity
    (verified bitwise against ffo tag 0: oprec, oeprec, orsi, orunpsi, oerunpsi, osrunpsi). Weight-zero cells (RSI = 1 for the
    (1-RSI)-weighted fields, RSI = 0 for the RSI-weighted ones) are set to 0: this branch is not exercised in any dump (no cell
    with RSI == 1) and is therefore UNVERIFIED."""
    rsi = ice['rsi']
    w0 = (1.0 - rsi) > 0
    return dict(oprec=np.where(w0, prec, 0.0), oeprec=np.where(w0, eprec, 0.0), orsi=rsi.copy(),
                orunpsi=np.where(rsi > 0, pi['runpsi'], 0.0), oerunpsi=np.where(rsi > 0, pi['erunpsi'], 0.0),
                osrunpsi=np.where(rsi > 0, pi['srunpsi'], 0.0))


# ------------------------------------------------------------------------------------------------ UNDERICE, GROUND_SI, FORM_SI, CALC_APRESS
import underice_jax as UI  # noqa: E402
import apress_jax as AP  # noqa: E402

SHW, RHOW, RHOI, RHOWS = SI.SHW, 1000.0, SI.RHOI, SI.RHOWS
FLEADOC, FLEADLK = 0.06, 0.0


def underice(ice, gtemp, sss, mlhc, ui2rho, coriol, mldlk, dlake, glake, geo, ustar_override=None):
    """SEAICE_DRV.f:187 UNDERICE for both domains (KOCEAN = 1). Returns fmsi, fhsi, fssi (kg/m2, J/m2, kg/m2 per step, positive down)."""
    fw, oc = geo['fwater'], geo['is_ocean']
    fm = np.zeros((IM, JM)); fh = np.zeros((IM, JM)); fs = np.zeros((IM, JM))
    sel = (ice['rsi'] * fw > 0) & geo['valid']
    ii, jj = np.nonzero(sel)
    if len(ii) == 0:
        return fm, fh, fs
    msi = ice['msi'][ii, jj]
    h4 = ice['hsi'][ii, jj, 3]; s4 = ice['ssi'][ii, jj, 3]
    dh = 0.5 * (XSI[3] * msi) / RHOI
    tm = gtemp[ii, jj]
    isoc = oc[ii, jj]
    # ocean branch
    si_ = 1e3 * s4 / (XSI[3] * msi)
    tic_o = SI.Ti(_J(h4 / (XSI[3] * msi)), _J(si_))
    ustar = np.maximum(5e-4, np.sqrt(ui2rho[ii, jj] / RHOWS)) if ustar_override is None else ustar_override[ii, jj]
    ro = UI.iceocean_fluxes(tic_o, _J(si_), _J(tm), _J(sss[ii, jj]), _J(dh), _J(ustar), _J(coriol[ii, jj]), DTSRC, _J(mlhc[ii, jj]))
    # lake branch
    tic_l = SI.Ti(_J(h4 / (XSI[3] * msi)), _J(np.zeros_like(msi)))
    mlsh = SHW * mldlk[ii, jj] * RHOW
    rl = UI.icelake_fluxes_limited(tic_l, _J(tm), _J(dh), DTSRC, _J(mlsh), _J(dlake[ii, jj]), _J(glake[ii, jj]))
    mflux = np.where(isoc, np.asarray(ro['mflux']), np.asarray(rl['mflux']))
    hflux = np.where(isoc, np.asarray(ro['hflux']), np.asarray(rl['hflux']))
    sflux = np.where(isoc, np.asarray(ro['sflux']), 0.0)
    fm[ii, jj] = mflux * DTSRC; fh[ii, jj] = hflux * DTSRC; fs[ii, jj] = sflux * DTSRC
    return fm, fh, fs


def ground_si(ice, e0, e1, evapor, solar, fmsi, fhsi, fssi, gtemp, sss, geo):
    """SEAICE_DRV.f:564 GROUND_SI (both domains) on the accumulated ice-tile fluxes of the step: e0 = sum F0DT, e1 = sum F1DT,
    evapor = sum EVAP, solar = sum SRHEAT*DTSURF (J/m2, kg/m2). Returns (new ice, dict runosi, erunosi, srunosi, solar_io)."""
    ice = copy_ice(ice)
    fw, oc = geo['fwater'], geo['is_ocean']
    runosi = np.zeros((IM, JM)); erunosi = np.zeros((IM, JM)); srunosi = np.zeros((IM, JM)); solar_io = np.zeros((IM, JM))
    sel = (ice['rsi'] * fw > 0) & geo['valid']
    ii, jj = np.nonzero(sel)
    if len(ii) == 0:
        return ice, dict(runosi=runosi, erunosi=erunosi, srunosi=srunosi, solar_io=solar_io)
    isoc = oc[ii, jj]
    snow, msi2 = ice['snowi'][ii, jj], ice['msi'][ii, jj]
    hsil, ssil = ice['hsi'][ii, jj], ice['ssi'][ii, jj]
    wet = ice['flag_dsws'][ii, jj]
    fmoc, fhoc, fsoc = fmsi[ii, jj], fhsi[ii, jj], fssi[ii, jj]
    tm = gtemp[ii, jj]; sm = np.where(isoc, sss[ii, jj], 0.0)
    si = SI.sea_ice(DTSRC, _J(snow), _J(hsil), _J(ssil), _J(msi2), _J(e0[ii, jj]), _J(e1[ii, jj]), _J(evapor[ii, jj]),
                    _J(solar[ii, jj]), _J(fmoc), _J(fhoc), _J(fsoc), _J(wet))
    dec = SI.ssidec(si['snow'], si['msi2'], si['hsil'], si['ssil'], DTSRC, si['melt12'])
    sic = SI.snowice(_J(tm), _J(sm), dec['snow'], dec['msi2'], dec['hsil'], dec['ssil'], False)
    m = _J(isoc)
    snow_n = jnp.where(m, sic['snow'], si['snow']); msi2_n = jnp.where(m, sic['msi2'], si['msi2'])
    hsil_n = jnp.where(m[..., None], sic['hsil'], si['hsil']); ssil_n = jnp.where(m[..., None], sic['ssil'], si['ssil'])
    run = jnp.where(m, fmoc + si['run'] + dec['mflux'] + sic['msnwic'], fmoc + si['run'])
    erun = jnp.where(m, fhoc + si['erun'] + dec['hflux'] + sic['hsnwic'], fhoc + si['erun'])
    srun = jnp.where(m, fsoc + si['srun'] + dec['sflux'] + sic['ssnwic'], fsoc + si['srun'])
    snow_n, msi2_n, hsil_n, ssil_n, run, erun, srun = [np.asarray(x) for x in (snow_n, msi2_n, hsil_n, ssil_n, run, erun, srun)]
    ice['snowi'][ii, jj] = snow_n; ice['msi'][ii, jj] = msi2_n; ice['hsi'][ii, jj, :] = hsil_n; ice['ssi'][ii, jj, :] = ssil_n
    ice['flag_dsws'][ii, jj] = np.asarray(si['wetsnow']).astype(bool)
    runosi[ii, jj] = run; erunosi[ii, jj] = erun; srunosi[ii, jj] = srun; solar_io[ii, jj] = np.asarray(si['srox2'])
    # pond_melt (SEAICE_DRV.f:~735-755)
    melt12 = np.asarray(si['melt12'])
    msi1 = snow_n + ACE1I
    mice1 = np.where(ACE1I > XSI[1] * msi1, ACE1I - XSI[1] * msi1, 0.0)
    snowl1 = np.where(ACE1I > XSI[1] * msi1, snow_n, XSI[0] * msi1)
    m1s = np.where(mice1 != 0.0, mice1, 1.0)
    ti1a = np.asarray(SI.Ti2b(_J(hsil_n[:, 0] / (XSI[0] * msi1)), _J(1e3 * ssil_n[:, 0] / m1s), _J(snowl1), _J(m1s)))
    ti1b = np.asarray(SI.Ti(_J(hsil_n[:, 0] / (XSI[0] * (snow_n + ACE1I))), _J(np.zeros_like(snow_n))))
    ti1 = np.where(mice1 != 0.0, ti1a, ti1b)
    pm = ice['pond_melt'][ii, jj] + 0.3 * melt12
    pm = np.minimum(pm, 0.5 * (msi2_n + snow_n + ACE1I))
    pm = np.where(melt12 > 0, pm * (1.0 - DTSRC / (30.0 * 86400.0)), pm * (1.0 - DTSRC / (10.0 * 86400.0)))
    pm = np.where(ti1 < -10.0, 0.0, pm)
    ice['pond_melt'][ii, jj] = pm
    return ice, dict(runosi=runosi, erunosi=erunosi, srunosi=srunosi, solar_io=solar_io)


def form_si(ice, dmsi, dhsi, dssi, geo):
    """SEAICE_DRV.f:877 FORM_SI (ADDICE) for both domains. dmsi/dhsi/dssi: (2, IM, JM) = (ACEFO, ACEFI), (ENRGFO, ENRGFI), (SALTO, SALTI)."""
    ice = copy_ice(ice)
    fw, oc = geo['fwater'], geo['is_ocean']
    sel = (fw > 0) & geo['valid']
    ii, jj = np.nonzero(sel)
    isoc = oc[ii, jj]
    flead = np.where(isoc, FLEADOC, FLEADLK)
    r = SI.addice(_J(ice['snowi'][ii, jj]), _J(ice['rsi'][ii, jj]), _J(ice['hsi'][ii, jj]), _J(ice['ssi'][ii, jj]), _J(ice['msi'][ii, jj]),
                  _J(dhsi[0][ii, jj]), _J(dmsi[0][ii, jj]), _J(dmsi[1][ii, jj]), _J(dhsi[1][ii, jj]), _J(dssi[0][ii, jj]),
                  _J(dssi[1][ii, jj]), _J(flead), False)
    r = {k: np.asarray(v) for k, v in r.items()}
    ice['snowi'][ii, jj] = r['snow']; ice['msi'][ii, jj] = r['msi2']; ice['hsi'][ii, jj, :] = r['hsil']; ice['ssi'][ii, jj, :] = r['ssil']
    ice['rsi'][ii, jj] = r['roice']
    _pole_replicate(ice, geo)
    return ice


def calc_apress(srfp, ice, geo, grav=O.GRAV):
    """SEAICE_DRV.f:10 CALC_APRESS: APRESS = 100*(SRFP-1013.25) + RSI*(SNOWI+ACE1I+MSI)*GRAV, poles replicated from i = 1."""
    ap = np.array(AP.calc_apress(_J(srfp), _J(ice["rsi"]), _J(ice["snowi"]), _J(ice["msi"]), grav))
    for j in (0, JM - 1):
        ap[1:, j] = ap[0, j]
    return ap


# ------------------------------------------------------------------------------------------------ land ice (PRECIP_LI, GROUND_LI), lakes (PRECIP_LK, GROUND_LK)
import landice_precip_jax as LIP  # noqa: E402
import lakes_core_jax as LK  # noqa: E402

LHM, SHI = 3.34e5, 2060.0
Z1E, Z2LI = 0.1, 2.9
ACE1LI = Z1E * RHOI
ACE2LI = Z2LI * RHOI
HC2LI = ACE2LI * SHI
TEENY = 1e-30
MINMLD = LK.MINMLD


def precip_li(li, prec, eprec, flice, geo):
    """LANDICE_DRV.f:507 PRECIP_LI (one height class; ftype = FLICE).  Returns (new li, dict runo, implm, implh, e1)."""
    li = {k: v.copy() for k, v in li.items()}
    r = LIP.precip_li(_J(flice), _J(prec), _J(eprec), _J(li['snowli']), _J(li['tlandi'][..., 0]), _J(li['tlandi'][..., 1]))
    r = {k: np.asarray(v) for k, v in r.items()}
    act = (flice > 0) & (prec > 0) & geo['valid']
    li['snowli'] = np.where(act, r['snow'], li['snowli'])
    li['tlandi'][..., 0] = np.where(act, r['tg1'], li['tlandi'][..., 0])
    li['tlandi'][..., 1] = np.where(act, r['tg2'], li['tlandi'][..., 1])
    return li, dict(runo=np.where(act, r['runo'], 0.0), implm=np.where(act, r['implm'], 0.0), implh=np.where(act, r['implh'], 0.0),
                    e1=np.where(act, r['e1'], 0.0))


def lndice(snow, tg1, tg2, f0dt, f1dt, evap):
    """LANDICE.f:162 LNDICE (NEW small port, no tracers). Vectorised over cells (numpy). Returns snow, tg1, tg2, edifs, difs, run0."""
    snow = np.asarray(snow, float); tg1 = np.asarray(tg1, float); tg2 = np.asarray(tg2, float)
    snandi = snow + ACE1LI - evap
    hc1 = snandi * SHI
    enrg1 = f0dt + evap * (tg1 * SHI - LHM) - f1dt
    hot = enrg1 > -tg1 * hc1
    run0 = np.where(hot, (enrg1 + tg1 * hc1) / LHM, 0.0)
    tg1n = np.where(hot, 0.0, tg1 + enrg1 / np.where(hc1 == 0, 1.0, hc1))
    snandi = np.where(hot, snandi - run0, snandi)
    thin = snandi < ACE1LI
    difs = np.where(thin, snandi - ACE1LI, 0.0)
    tg1n = np.where(thin, (tg1n * snandi - tg2 * difs) / ACE1LI, tg1n)
    edifs = np.where(thin, difs * (tg2 * SHI - LHM), 0.0)
    snow_n = np.where(thin, 0.0, snandi - ACE1LI)
    tg2n = tg2 + f1dt / HC2LI
    return snow_n, tg1n, tg2n, edifs, difs, run0


def ground_li(li, e0, e1, evapor, flice, geo):
    """LANDICE_DRV.f:654 GROUND_LI (one height class): fluxes accumulated over the step's two substeps (e1 includes PRECIP_LI's EDIFS).
    Returns (new li, dict runo = RUN0 (kg/m2), implm, implh)."""
    li = {k: v.copy() for k, v in li.items()}
    act = (flice > 0) & geo['valid']
    s, t1, t2, edifs, difs, run0 = lndice(li['snowli'], li['tlandi'][..., 0], li['tlandi'][..., 1], e0, e1, evapor)
    li['snowli'] = np.where(act, s, li['snowli'])
    li['tlandi'][..., 0] = np.where(act, t1, li['tlandi'][..., 0])
    li['tlandi'][..., 1] = np.where(act, t2, li['tlandi'][..., 1])
    return li, dict(runo=np.where(act, run0, 0.0), implm=np.where(act, difs, 0.0), implh=np.where(act, edifs, 0.0),
                    e1=np.where(act, edifs + e1, 0.0))


def irrig_lk(lake, irrig_act, fland, axyp, geo):
    """LAKES.f:3153 IRRIG_LK / IRRIGMOD.f:212 irrigate_extract, RECONSTRUCTED from the recorded ACTUAL irrigation flux (m/s, the GHY forcing
    `irrig*fearth` of the ffg record).  The irrigation DEMAND (irrig_water_pot, a data file) is NOT read: the mass taken from the lake is
    min(irrig_act*rho_w*A*dt, m_avail) with m_avail = max(MWL - MINMLD*FLAKE*rho_w*A, 0) for lakes (MWL for lake-free cells), which equals the
    Fortran in all three branches (full, partial, groundwater-only; checked on the nov26 window).  The energy and the layer adjustments are
    transcribed from the source.  Returns (new lake, dict mwl_to_irrig)."""
    lake = {k: v.copy() for k, v in lake.items()}
    flake = geo['flake']
    m_to_kg = RHOW * axyp
    act = (fland > 0) & geo['valid'] & (irrig_act > TEENY)
    mwl, gml, mld, tl = lake['mwl'], lake['gml'], lake['mldlk'], lake['tlake']
    isl = flake > 0
    m_avail = np.where(isl, np.maximum(mwl - MINMLD * flake * m_to_kg, 0.0), mwl)
    t_irr = np.where(isl, tl, gml / (mwl * SHW + TEENY))
    t_irr2 = np.where(isl, np.where(mwl > flake * mld * m_to_kg + TEENY,
                                    (gml - mld * m_to_kg * flake * tl * SHW) / (mwl - mld * m_to_kg * flake + TEENY) / SHW, 0.0), t_irr)
    t_irr = np.maximum(t_irr, 0.0); t_irr2 = np.maximum(t_irr2, 0.0)
    m_irr_pot = irrig_act * m_to_kg * DTSRC
    m_irr = np.where(m_avail <= TEENY, 0.0, np.minimum(m_irr_pot, m_avail))
    m_irr = np.where(act, m_irr, 0.0)
    l1 = mld * m_to_kg * flake
    two = isl & (m_irr > l1)
    g_irr = np.where(two, l1 * SHW * t_irr + (m_irr - l1) * SHW * t_irr2, m_irr * SHW * t_irr)
    do = m_irr > 0
    mwl_n = np.where(do, mwl - m_irr, mwl); gml_n = np.where(do, gml - g_irr, gml)
    # layer / temperature adjustments for lakes
    fsafe = np.where(isl, flake, 1.0)
    one = do & isl & (m_irr < l1)
    mld1 = mld - m_irr / (fsafe * axyp * RHOW)
    m1 = mld1 * RHOW * flake * axyp
    m2 = np.maximum(mwl_n - m1, 0.0)
    up = one & (mld1 < MINMLD) & (m2 > 0)
    e1 = tl * SHW * m1
    e2 = gml_n - e1
    dm = np.maximum(MINMLD * RHOW * flake * axyp - m1, 0.0)
    de = dm * e2 / (m2 + TEENY)
    tl_up = (e1 + de) / ((m1 + dm) * SHW)
    mld_up = mld1 + dm / (fsafe * axyp * RHOW)
    allgone = do & isl & ~(m_irr < l1)
    mld_all = mwl_n / (fsafe * axyp * RHOW)
    tl_all = gml_n / (mwl_n * SHW + TEENY)
    mld_new = np.where(up, mld_up, np.where(one, mld1, np.where(allgone, mld_all, mld)))
    tl_new = np.where(up, tl_up, np.where(allgone, tl_all, tl))
    lake.update(mwl=mwl_n, gml=gml_n, mldlk=mld_new, tlake=tl_new)
    return lake, dict(mwl_to_irrig=m_irr, gml_to_irrig=np.where(do, g_irr, 0.0))


def precip_lk(lake, ice, prec, eprec, runpsi, runo_li, melti, emelti, flice, axyp, atm_g, geo):
    """LAKES.f:3012 PRECIP_LK. lake: dict mwl, gml, tlake, mldlk (IM, JM). atm_g: dict gtemp, gtemp2, gtempr of the atmocn arrays (lake
    cells get TLAKE).  Returns (new lake, new gtemp dict, dict dlake, glake, run0, erun0)."""
    r = LK.precip_lk(_J(geo['flake']), _J(flice), _J(ice['rsi']), _J(prec), _J(eprec), _J(runpsi), _J(runo_li), _J(melti), _J(emelti),
                     _J(axyp), _J(lake['mwl']), _J(lake['gml']), _J(lake['tlake']), _J(lake['mldlk']), _J(atm_g['gtemp']),
                     _J(atm_g['gtemp2']), _J(atm_g['gtempr']))
    r = {k: np.asarray(v) for k, v in r.items()}
    valid = geo['valid']
    active = ((geo['flake'] + flice) > 0) & valid
    out = {k: v.copy() for k, v in lake.items()}
    for k in ('mwl', 'gml', 'tlake', 'mldlk'):
        out[k] = np.where(active, r[k], lake[k])
    g = {k: np.where(active & (geo['flake'] > 0), r[k], atm_g[k]) for k in ('gtemp', 'gtemp2', 'gtempr')}
    return out, g, dict(dlake=np.where(active, r['dlake'], 0.0), glake=np.where(active, r['glake'], 0.0))


def ground_lk(lake, ice, hlake, axyp, fland, flice, fearth, fgeotherm, runo_li, rune, erune, e0_o, evap_o, solar_o, runosi, erunosi,
              solar_io, geo):
    """LAKES.f:3285 GROUND_LK (no tracers, TKE = 0). Inputs per cell: land runoff rune/erune (atmlnd%RUNO/ERUNO summed over the
    substeps), land-ice runoff runo_li (GROUND_LI), open-lake tile accumulators e0_o/evap_o/solar_o, GROUND_SI(lake) outputs
    runosi/erunosi/solar_io.  Returns (new lake, dict gtemp, gtemp2, gtempr (lake cells), dmsi, dhsi, dssi (2, IM, JM))."""
    lake = {k: v.copy() for k, v in lake.items()}
    valid = geo['valid']; flake = geo['flake']
    mwl, gml, tlake, mldlk = lake['mwl'], lake['gml'], lake['tlake'], lake['mldlk']
    # (a) land runoff and geothermal heat into cells with FLAND > 0
    fl_ok = (fland > 0) & valid
    run0 = runo_li * flice + rune * fearth
    erun0 = erune * fearth
    egeo = fgeotherm * flake * DTSRC
    mwl = np.where(fl_ok, mwl + run0 * axyp, mwl)
    gml = np.where(fl_ok, gml + erun0 * axyp + egeo * axyp, gml)
    lk = fl_ok & (flake > 0)
    fsafe = np.where(flake > 0, flake, 1.0)
    hlk1 = tlake * mldlk * RHOW * SHW
    mld_new = mldlk + run0 / (fsafe * RHOW)
    tl_new = (hlk1 * fsafe + erun0 + egeo) / (mld_new * fsafe * RHOW * SHW)
    nl = fl_ok & ~(flake > 0)
    tl_nl = gml / (mwl * SHW + TEENY)
    mldlk = np.where(lk, mld_new, mldlk)
    tlake = np.where(lk, tl_new, np.where(nl, tl_nl, tlake))
    # (b) lake dynamics
    act = (flake > 0) & valid
    ii, jj = np.nonzero(act)
    dmsi = np.zeros((2, IM, JM)); dhsi = np.zeros((2, IM, JM)); dssi = np.zeros((2, IM, JM))
    gt = dict(gtemp=np.zeros((IM, JM)), gtemp2=np.zeros((IM, JM)), gtempr=np.zeros((IM, JM)))
    if len(ii):
        fk = flake[ii, jj]; ax = axyp[ii, jj]
        roice = ice['rsi'][ii, jj]
        ml1 = mldlk[ii, jj] * RHOW
        ml2 = np.maximum(mwl[ii, jj] / (fk * ax) - ml1, 0.0)
        el1 = tlake[ii, jj] * SHW * ml1
        el2 = gml[ii, jj] / (fk * ax) - el1
        thin = ml2 < 1e-10
        ml1 = np.where(thin, ml1 + ml2, ml1); el1 = np.where(thin, el1 + el2, el1)
        ml2 = np.where(thin, 0.0, ml2); el2 = np.where(thin, 0.0, el2)
        fsr2 = np.minimum(np.exp(-mldlk[ii, jj] * LK.BYZETA), ml2 / (ml1 + ml2))
        src = LK.lksourc_full(_J(roice), _J(ml1), _J(ml2), _J(el1), _J(el2), _J(runosi[ii, jj]), _J(e0_o[ii, jj]),
                              _J(erunosi[ii, jj]), _J(solar_o[ii, jj]), _J(solar_io[ii, jj]), _J(fsr2), _J(evap_o[ii, jj]))
        src = {k: np.asarray(v) for k, v in src.items()}
        mix = LK.lkmix(_J(src['mlake0']), _J(src['mlake1']), _J(src['elake0']), _J(src['elake1']), _J(hlake[ii, jj]),
                       _J(np.zeros_like(roice)), _J(roice), DTSRC)
        mix = {k: np.asarray(v) for k, v in mix.items()}
        m0, m1, e0_, e1_ = mix['mlake0'], mix['mlake1'], mix['elake0'], mix['elake1']
        mwl[ii, jj] = (m0 + m1) * (fk * ax)
        gml[ii, jj] = (e0_ + e1_) * (fk * ax)
        mld = m0 / RHOW
        mld = np.where(m1 == 0.0, np.minimum(MINMLD, mld), mld)
        mldlk[ii, jj] = mld
        tl = e0_ / (SHW * m0)
        tlake[ii, jj] = tl
        tlk2 = np.where(m1 > 0, e1_ / (SHW * np.where(m1 > 0, m1, 1.0)), tl)
        gt['gtemp'][ii, jj] = tl; gt['gtemp2'][ii, jj] = tlk2; gt['gtempr'][ii, jj] = tl + TF
        dmsi[0][ii, jj] = src['acefo']; dmsi[1][ii, jj] = src['acefi']
        dhsi[0][ii, jj] = src['enrgfo']; dhsi[1][ii, jj] = src['enrgfi']
    lake.update(mwl=mwl, gml=gml, tlake=tlake, mldlk=mldlk)
    return lake, gt, dict(dmsi=dmsi, dhsi=dhsi, dssi=dssi)


# ------------------------------------------------------------------------------------------------ statics and recorded boundaries of the replay mode
def load_statics(date='nov26', ff=FF, it0=None):
    """Static fields of the exchange grid. Sources named (nothing guessed): FOCEAN/FLAKE/FLAND/FLICE/FEARTH from the CONDSE entry dump
    (ffc_cse_in), AXYP = DXYPO(j) (equal to the tile records' AXYP, checked), CORIOL = |PBL-record coriol| (iceocn%coriol = ABS(2*OMEGA*SINLAT)),
    HLAKE (DLAKE0, lake depth) from the GROUND_LK records ffl2 (static, recorded), FGEOTHERM = land geothermal flux from the ffg records
    (all zero in the records)."""
    import clouds_condse_io as cio
    import pbl_compare as PC
    import lakes_compare as LCm
    from ocean_step_compare import ctx_for
    d = f"{ff}/{date}"
    ctx = ctx_for(d)
    it0 = it0 or int(sorted(glob_its(d, 'ffc_cse_in'))[0])
    ci = cio.read_cse(f"{d}/ffc_cse_in_{it0}.bin")
    geo = make_geo(ctx, ci['FLAKE'])
    st = dict(ctx=ctx, geo=geo, axyp=np.repeat(O._DXYPO[None, :], IM, axis=0), fland=ci['FLAND'], flice=ci['FLICE'], fearth=ci['FEARTH'])
    p = PC.load(f"{d}/ffp_{it0}.bin")
    cor = np.zeros((IM, JM))
    cor[p[:, 0].astype(int) - 1, p[:, 1].astype(int) - 1] = np.abs(p[:, 36])
    st['coriol'] = cor
    l2 = LCm.load(f"{d}/ffl2_{it0}.bin") if os.path.exists(f"{d}/ffl2_{it0}.bin") else None
    hl = np.zeros((IM, JM))
    if l2 is not None:
        hl[l2[:, 0].astype(int) - 1, l2[:, 1].astype(int) - 1] = l2[:, 14]
    st['hlake'] = hl
    st['fgeotherm'] = np.zeros((IM, JM))
    return st


def glob_its(d, prefix):
    import glob
    return [int(os.path.basename(p)[len(prefix) + 1:-4]) for p in glob.glob(f"{d}/{prefix}_[0-9]*.bin")]


# ------------------------------------------------------------------------------------------------ recorded real tile outputs (replay mode) -> accumulators
def tile_accumulators(date, it, ff=FF):
    """Real SURFACE tile OUTPUT columns of step `it` (ffs ocean/ice tiles, ffl land-ice tiles, ffg land) summed over the two substeps,
    in the form the drivers use.  Replay-mode substitute for OUR tile chain; the closed loop replaces it by the chain's own outputs
    (D18-D22).  Identities checked bitwise against ffo tag 1 (oe0, oevapor, osolarw, odmua, odmva)."""
    import surface_tile_ff as ST
    import landice_tile_ff as LIT
    import ghy_compare as GC
    d = f"{ff}/{date}"
    t = ST.load(f"{d}/ffs_{it}.bin"); n = len(t) // 2
    acc = {k: np.zeros((IM, JM)) for k in ('e0_o', 'evap_o', 'solar_o', 'dmua_o', 'dmva_o', 'e0_i', 'e1_i', 'evap_i', 'solar_i',
                                          'e0_li', 'e1_li', 'evap_li', 'rune', 'erune')}
    for ns in (0, 1):
        tt = t[ns * n:(ns + 1) * n]
        for typ, names in ((1, ('e0_o', 'evap_o', 'solar_o', 'dmua_o', 'dmva_o')), (2, ('e0_i', 'e1_i', 'evap_i', 'solar_i'))):
            r = tt[tt[:, 2] == typ]
            i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
            dts = r[:, ST.IN['dtsurf']]
            vals = {'e0_o': r[:, ST.OUT['f0dt']], 'evap_o': r[:, ST.OUT['evap']], 'solar_o': r[:, ST.IN['srheat']] * dts,
                    'dmua_o': r[:, ST.OUT['dmua']] * dts, 'dmva_o': r[:, ST.OUT['dmva']] * dts,
                    'e0_i': r[:, ST.OUT['f0dt']], 'e1_i': r[:, ST.OUT['f1dt']], 'evap_i': r[:, ST.OUT['evap']],
                    'solar_i': r[:, ST.IN['srheat']] * dts}
            for k in names:
                acc[k][i, j] += vals[k]
    l = LIT.load(f"{d}/ffl_{it}.bin"); nl = len(l) // 2
    for ns in (0, 1):
        r = l[ns * nl:(ns + 1) * nl]
        i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
        acc['e0_li'][i, j] += r[:, LIT.OUT['f0dt']]; acc['e1_li'][i, j] += r[:, LIT.OUT['f1dt']]; acc['evap_li'][i, j] += r[:, LIT.OUT['evap']]
    g = GC.load(f"{d}/ffg_{it}.bin"); ng = len(g) // 2
    for ns in (0, 1):
        r = g[ns * ng:(ns + 1) * ng]
        i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
        acc['rune'][i, j] += r[:, 250] + r[:, 251]          # aruns + arunu (1-based 251, 252)
        acc['erune'][i, j] += r[:, 252] + r[:, 253]         # aeruns + aerunu (1-based 253, 254)
    return acc


def recorded_dynsi(date, it, ff=FF):
    """Replay-mode DYNSI boundary: the real DMUI/DMVI (ice -> ocean stress, ocean grid) and the ustar used by UNDERICE (ocean domain)."""
    import ocean_chain_io as C
    import underice_compare as UC
    d = f"{ff}/{date}"
    sn = C.load_step(d, it)
    u = UC.read_undocn(f"{d}/ffz_undocn_{it}.bin")
    ustar = np.zeros((IM, JM))
    ustar[u['i'].astype(int) - 1, u['j'].astype(int) - 1] = u['ustar']
    return dict(odmui=sn[1]['odmui'], odmvi=sn[1]['odmvi'], ustar=ustar, undocn=u)


# ------------------------------------------------------------------------------------------------ the loop
def make_ocean_ctx(date='nov26', ff=FF, computed_opfil=True):
    """Ocean ctx (D119): geometry/straits tables from ffo_geom.bin (static); the OPFIL2 coefficient vector is COMPUTED (D138) and written to
    a scratch file (the recorded ffo_opcoef.bin is not read)."""
    import tempfile
    import ocean_chain_io as C
    from ocean_step_compare import ctx_for
    d = f"{ff}/{date}"
    ctx = ctx_for(d)
    if computed_opfil:
        from ocean_opfil2_coeffs import calc_opfil2_coeffs
        v, _ = calc_opfil2_coeffs(ctx['lmu'])
        tmp = tempfile.mkdtemp(prefix='opcoef_') + '/opcoef_computed.bin'
        v.astype('>f8').tofile(tmp)
        ctx['opcoef'] = tmp
    return ctx


def init_surface_state(date='nov26', ff=FF, st=None):
    """Surface state at the start of the window from the real restart + the one recorded static of init_STRAITS (MMST) from ffo tag 0."""
    import ocean_chain_io as C
    S = load_restart(date, ff)
    st = st or load_statics(date, ff)
    ctx = st['ctx']
    oc = S['ocean']
    sn0 = C.load_step(f"{ff}/{date}", C.list_steps(f"{ff}/{date}")[0])[0]
    oc['mmst'] = sn0['mmst'].copy()                       # recorded static (init_STRAITS)
    for k in ('mmi', 'smu', 'smv', 'smw'):
        oc[k] = np.zeros((IM, JM, LMO))
    oc['opbot'] = np.zeros((IM, JM)); oc['opress'] = np.zeros((IM, JM)); oc['vonp'] = np.zeros(LMO)
    geo = st['geo']
    ice = ice_from_restart(S, geo)
    lake = {k: S['lake'][k].copy() for k in ('mwl', 'gml', 'tlake', 'mldlk')}
    t = toc2sst(oc, ctx)
    gtemp = np.where(geo['is_ocean'], t['gtemp'], lake['tlake'])
    atm = dict(gtemp=gtemp, gtemp2=np.where(geo['is_ocean'], t['gtemp2'], S['atm']['asst']), gtempr=np.where(geo['is_ocean'], t['gtempr'], lake['tlake'] + TF),
               sss=t['sss'], mlhc=np.where(geo['is_ocean'], t['mlhc'], S['atm']['mlhc']))
    return dict(ocean=oc, ice=ice, lake=lake, li={k: v.copy() for k, v in S['landice'].items()}, atm=atm, itime=S['itime'],
                ghy={k: v.copy() for k, v in S['ghy'].items()})


def geo_sub(geo, which):
    g = dict(geo)
    g['valid'] = geo['valid'] & (geo['is_ocean'] if which == 'ocean' else geo['is_lake'])
    return g


def ocean_stages_after_precip():
    import ocean_step_odiff as OD
    return [s for s in OD.STAGES_ODIFF if s[0] != 'precip']


def surface_pre(S, st, inp, melt_done=None):
    """Everything of step k that happens BEFORE the SURFACE tile loop: MELT_SI, PRECIP_SI, PRECIP_OC (+TOC2SST), PRECIP_LI, PRECIP_LK,
    seaice_to_atmgrid.  inp: prec, eprec.  Returns (state', dict of intermediates used by the tiles and the post stage)."""
    geo, ctx = st['geo'], st['ctx']
    ice, atm = S['ice'], S['atm']
    ice, melt = melt_done if melt_done is not None else melt_si(ice, atm['gtemp'], atm['sss'], atm['mlhc'], geo)
    ice, pi = precip_si(ice, inp['prec'], inp['eprec'], geo)
    fxp = ag2og_precip(inp['prec'], inp['eprec'], ice, pi, geo)
    oc = O.stage_precip(S['ocean'], fxp, ctx)
    t = toc2sst(oc, ctx)
    atm = {k: v.copy() for k, v in atm.items()}
    oo = geo['is_ocean']
    for k in ('gtemp', 'gtemp2', 'gtempr', 'sss', 'mlhc'):
        atm[k] = np.where(oo, t[k], atm[k])
    li, pli = precip_li(S['li'], inp['prec'], inp['eprec'], st['flice'], geo)
    lake0, irr = irrig_lk(S['lake'], inp['irrig_act'], st['fland'], st['axyp'], geo)
    lake, gt_l, dl = precip_lk(lake0, ice, inp['prec'], inp['eprec'], pi['runpsi'], pli['runo'], melt['melti'], melt['emelti'],
                               st['flice'], st['axyp'], atm, geo)
    for k in ('gtemp', 'gtemp2', 'gtempr'):
        atm[k] = np.where(geo['is_lake'], gt_l[k], atm[k])
    ag = seaice_to_atmgrid(ice, geo)
    S2 = dict(S, ocean=oc, ice=ice, lake=lake, li=li, atm=atm)
    return S2, dict(melt=melt, pi=pi, fxp=fxp, pli=pli, dl=dl, ag=ag, irr=irr)


def surface_post(S, st, inp, mid, ocean_stages=None, dynsi=None):
    """Everything AFTER the tile loop: GROUND_LI, the lake chain (UNDERICE, GROUND_SI, GROUND_LK, FORM_SI), then ocean_driver (DYNSI boundary,
    UNDERICE, GROUND_SI, CALC_APRESS, OCEANS, FORM_SI).  inp: acc (tile accumulators), srfp, flows (oflowo, oeflowo recorded: RIVERF).
    Returns (state', dict of intermediates: fx (ocean flux dict), runosi, ...)."""
    geo, ctx = st['geo'], st['ctx']
    acc = inp['acc']
    gl, gol = geo_sub(geo, 'lake'), geo_sub(geo, 'ocean')
    ice, atm, lake, li = S['ice'], {k: v.copy() for k, v in S['atm'].items()}, S['lake'], S['li']
    li, gli = ground_li(li, acc['e0_li'], acc['e1_li'] + mid['pli']['e1'], acc['evap_li'], st['flice'], geo)
    # lakes ------------------------------------------------------------------------------------------------------------
    fm, fh, fs = underice(ice, atm['gtemp'], atm['sss'], atm['mlhc'], np.zeros((IM, JM)), st['coriol'], lake['mldlk'], mid['dl']['dlake'],
                          mid['dl']['glake'], gl)
    ice, gs_l = ground_si(ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], gl)
    lake, gt_l, dm = ground_lk(lake, ice, st['hlake'], st['axyp'], st['fland'], st['flice'], st['fearth'], st['fgeotherm'], gli['runo'],
                               acc['rune'], acc['erune'], acc['e0_o'], acc['evap_o'], acc['solar_o'], gs_l['runosi'], gs_l['erunosi'],
                               gs_l['solar_io'], gl)
    for k in ('gtemp', 'gtemp2', 'gtempr'):
        atm[k] = np.where(geo['is_lake'], gt_l[k], atm[k])
    ice = form_si(ice, dm['dmsi'], dm['dhsi'], dm['dssi'], gl)
    # ocean_driver ---------------------------------------------------------------------------------------------------
    oc = S['ocean']
    dyn = inp['dynsi'] if dynsi is None else dynsi
    fm, fh, fs = underice(ice, atm['gtemp'], atm['sss'], atm['mlhc'], dyn.get('ui2rho', np.zeros((IM, JM))), st['coriol'], lake['mldlk'],
                          mid['dl']['dlake'], mid['dl']['glake'], gol, ustar_override=dyn.get('ustar'))
    ice, gs_o = ground_si(ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], gol)
    apress = calc_apress(inp['srfp'], ice, geo)
    rsi = ice['rsi']
    foc = geo['focean']
    sel = geo['is_ocean'] & geo['valid']
    inv_f = 1.0 / np.where(foc > 0, foc, 1.0)
    mt = mid['melt']
    fx = dict(mid['fxp'])
    fx.update(oflowo=inp['flows']['oflowo'], oeflowo=inp['flows']['oeflowo'],
              omelti=np.where(sel, mt['melti'] * inv_f, 0.0), oemelti=np.where(sel, mt['emelti'] * inv_f, 0.0),
              osmelti=np.where(sel, mt['smelti'] * inv_f, 0.0),
              oevapor=np.where(sel, acc['evap_o'], 0.0), oe0=np.where(sel, acc['e0_o'], 0.0), osolarw=np.where(sel, acc['solar_o'], 0.0),
              orunosi=np.where(sel, gs_o['runosi'], 0.0), oerunosi=np.where(sel, gs_o['erunosi'], 0.0),
              osrunosi=np.where(sel, gs_o['srunosi'], 0.0), osolari=np.where(sel, gs_o['solar_io'], 0.0),
              oapress=np.where(geo['is_ocean'], apress, 0.0),
              odmua=np.where(sel, acc['dmua_o'] * (1.0 - rsi), 0.0), odmva=np.where(sel, acc['dmva_o'] * (1.0 - rsi), 0.0),
              odmui=dyn['odmui'], odmvi=dyn['odmvi'], orsi=rsi.copy(), itime=inp['itime'])
    ogeoz_entry = oc['ogeoz'].copy()
    stages = ocean_stages or ocean_stages_after_precip()
    for name, fn, _t in stages:
        oc = fn(oc, fx, ctx)
    oc['ogeoz_sv'] = ogeoz_entry
    ice = form_si(ice, oc['odmsi'], oc['odhsi'], oc['odssi'], gol)
    t = toc2sst(oc, ctx)
    for k in ('gtemp', 'gtemp2', 'gtempr', 'sss', 'mlhc'):
        atm[k] = np.where(geo['is_ocean'], t[k], atm[k])
    S2 = dict(S, ocean=oc, ice=ice, lake=lake, li=li, atm=atm, itime=S['itime'] + 1)
    return S2, dict(fx=fx, gs_l=gs_l, gs_o=gs_o, gli=gli, apress=apress, fm=fm, fh=fh, fs=fs)


# ------------------------------------------------------------------------------------------------ replay driver (real atmosphere-side inputs)
def replay_inputs(date, it, ff=FF):
    """Real atmosphere-side boundary of step `it`: PREC/EPREC (CONDSE exit), SRFP = PEDN(1) at SURFACE, tile output accumulators,
    recorded RIVERF outflow (oflowo, oeflowo of ffo tag 1), recorded DYNSI boundary (odmui, odmvi, ustar)."""
    import clouds_condse_io as cio
    import ocean_chain_io as C
    d = f"{ff}/{date}"
    co = cio.read_cse(f"{d}/ffc_cse_out_{it}.bin")
    ps = cio.old_state(date, it, 'pre_surface', ff)
    sn = C.load_step(d, it)
    import ghy_compare as GC
    import clouds_condse_io as _c
    g = GC.load(f"{d}/ffg_{it}.bin"); ng = len(g) // 2
    irrig = np.zeros((IM, JM)); irrig[g[:ng, 0].astype(int) - 1, g[:ng, 1].astype(int) - 1] = g[:ng, 147]
    fearth = _c.read_cse(f"{d}/ffc_cse_in_{it}.bin")['FEARTH']
    return dict(prec=co['PREC'], eprec=co['EPREC'], srfp=ps['PEDN'][0], acc=tile_accumulators(date, it, ff), irrig_act=irrig * fearth,
                flows=dict(oflowo=sn[1]['oflowo'], oeflowo=sn[1]['oeflowo']), dynsi=recorded_dynsi(date, it, ff), itime=it)


# ------------------------------------------------------------------------------------------------ comparison with the real SURFACE-entry records
def compare_pre_records(S1, mid, st, date, it, ff=FF):
    """State columns of the real substep-1 tile records of step `it` (ffs: open-water/lake and sea-ice tiles, ffl: land-ice tiles) against
    the state produced by OUR surface_pre.  Returns {name: (max abs diff, scale (max abs of ref), number of tiles)} and the tile-set check."""
    import surface_tile_ff as ST
    import landice_tile_ff as LIT
    d = f"{ff}/{date}"
    geo = st['geo']
    t = ST.load(f"{d}/ffs_{it}.bin"); n = len(t) // 2; t1 = t[:n]
    out = {}

    def add(name, got, ref):
        got = np.asarray(got, float); ref = np.asarray(ref, float)
        out[name] = (float(np.abs(got - ref).max()) if len(ref) else 0.0, float(np.abs(ref).max()) if len(ref) else 0.0, int(len(ref)))
    atm = S1['atm']; ice = S1['ice']; lake = S1['lake']; ag = mid['ag']
    # ocean-type tiles
    r = t1[t1[:, 2] == 1]
    i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
    oo = geo['is_ocean'][i, j]
    tfz = np.asarray(SI.tfrez(_J(atm['sss'][i, j])))
    tg1 = np.maximum(atm['gtemp'][i, j], tfz)
    add('ocean.tg1', tg1[oo], r[oo, ST.IN['tg1']]); add('ocean.sss', atm['sss'][i, j][oo], r[oo, ST.IN['sss']])
    add('ocean.tr4', (atm['gtempr'][i, j] ** 4)[oo], r[oo, ST.IN['tr4']])
    add('ocean.ptype', ((1.0 - ice['rsi']) * geo['fwater'])[i, j], r[:, ST.IN['ptype']])
    lk = ~oo
    add('lake.tg1', tg1[lk], r[lk, ST.IN['tg1']]); add('lake.tr4', (atm['gtempr'][i, j] ** 4)[lk], r[lk, ST.IN['tr4']])
    add('lake.mwl', lake['mwl'][i, j][lk], r[lk, ST.IN['mwl']]); add('lake.gml', lake['gml'][i, j][lk], r[lk, ST.IN['gml']])
    # sea-ice tiles
    r = t1[t1[:, 2] == 2]
    i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
    add('ice.tg1', ag['gtemp'][i, j], r[:, ST.IN['tg1']]); add('ice.tg2', ag['gtemp2'][i, j], r[:, ST.IN['tg2']])
    add('ice.tr4', ag['gtempr'][i, j] ** 4, r[:, ST.IN['tr4']])
    add('ice.snow', ice['snowi'][i, j], r[:, ST.IN['snow']]); add('ice.msi2', ice['msi'][i, j], r[:, ST.IN['msi2']])
    add('ice.ssi1', ice['ssi'][i, j, 0], r[:, ST.IN['ssi1']]); add('ice.ssi2', ice['ssi'][i, j, 1], r[:, ST.IN['ssi2']])
    add('ice.flag_dsws', ice['flag_dsws'][i, j].astype(float), r[:, ST.IN['flag_dsws']])
    add('ice.ptype', (ice['rsi'] * geo['fwater'])[i, j], r[:, ST.IN['ptype']])
    tfo = np.maximum(atm['gtemp'][i, j], np.asarray(SI.tfrez(_J(atm['sss'][i, j]))))
    add('ice.tgo', tfo, r[:, ST.IN['tgo']])
    out['tiles.ocean+lake'] = (0.0, 0.0, int((t1[:, 2] == 1).sum())); out['tiles.ice'] = (0.0, 0.0, int((t1[:, 2] == 2).sum()))
    # tile-set check: cells with an open-water tile / ice tile in the record vs in our state
    ocean_tile = ((1.0 - ice['rsi']) * geo['fwater'] > 0) & geo['valid']
    ice_tile = (ice['rsi'] * geo['fwater'] > 0) & geo['valid']
    rec_o = np.zeros((IM, JM), bool); rec_i = np.zeros((IM, JM), bool)
    ro = t1[t1[:, 2] == 1]; rec_o[ro[:, 0].astype(int) - 1, ro[:, 1].astype(int) - 1] = True
    ri = t1[t1[:, 2] == 2]; rec_i[ri[:, 0].astype(int) - 1, ri[:, 1].astype(int) - 1] = True
    out['tileset.ocean_mismatch'] = (float((ocean_tile != rec_o).sum()), 0.0, 0)
    out['tileset.ice_mismatch'] = (float((ice_tile != rec_i).sum()), 0.0, 0)
    # land ice
    l = LIT.load(f"{d}/ffl_{it}.bin"); nl = len(l) // 2; l1 = l[:nl]
    i, j = l1[:, 0].astype(int) - 1, l1[:, 1].astype(int) - 1
    li = S1['li']
    add('landice.tg1', li['tlandi'][i, j, 0], l1[:, LIT.IN['tg1']]); add('landice.tg2', li['tlandi'][i, j, 1], l1[:, LIT.IN['tg2']])
    add('landice.snow', li['snowli'][i, j], l1[:, LIT.IN['snow']])
    return out


# ------------------------------------------------------------------------------------------------ state -> SURFACE records (closed loop)
def _rows_ij(rec):
    return rec[:, 0].astype(int) - 1, rec[:, 1].astype(int) - 1


def apply_state_to_records(rec, S1, mid, st, ps_ij):
    """Overwrite the SURFACE-record columns that are functions of the surface STATE with the values derived from OUR state (S1 = state after
    surface_pre, mid = its intermediates).  rec: dict of surface_records(R) (pa, pb, ta, tb, la, lb, blk1, blk2, g1, g2; real records used as
    templates for everything that is static, atmospheric or radiative).  ps_ij: surface pressure PEDN(1) (IM, JM) for qg_sat.
    Columns overwritten (0-based, from the instrumentation patches): ffs ocean/lake tile tg1 tg2 tr4 sss evaplim htlim mwl gml ptype; ffs ice tile tg1 tg2 tr4 snow msi2
    ssi1 ssi2 flag_dsws tgo ptype; ffl tg1 tg2 tr4 snow; ffp (itype 1-3) tg tgv qg_sat qg_aver tr4 sss_loc snow; fft ftype.
    Returns (new rec, report with the tile-set counts)."""
    import surface_tile_ff as ST
    import landice_tile_ff as LIT
    import pbl_ff as P
    geo = st['geo']
    atm, ice, lake, li = S1['atm'], S1['ice'], S1['lake'], S1['li']
    ag = mid['ag']
    rec = {k: np.array(v, dtype=np.float64) for k, v in rec.items()}
    tfz = np.asarray(SI.tfrez(_J(atm['sss'])))
    tg1_oc = np.maximum(atm['gtemp'], tfz)
    flake, axyp = geo['flake'], st['axyp']
    # evap limit / heat limit of lake tiles (SURFACE.f:~440)
    with np.errstate(all='ignore'):
        mwl, gml = lake['mwl'], lake['gml']
        small = mwl < MINMLD * RHOW * flake * axyp
        evl = np.where(small, np.maximum(0.5 * (mwl / (flake * axyp) - 0.4 * RHOW), 0.0), mwl / (flake * axyp) - (0.5 * MINMLD + 0.2) * RHOW)
        htl = gml / (flake * axyp) + 0.5 * LHM * evl
    pocean = (1.0 - ice['rsi']) * geo['fwater']
    poice = ice['rsi'] * geo['fwater']
    for key in ('ta', 'tb'):
        t = rec[key]
        i, j = _rows_ij(t)
        oc = t[:, 2] == 1
        t[oc, ST.IN['tg1']] = tg1_oc[i, j][oc]; t[oc, ST.IN['tg2']] = atm['gtemp2'][i, j][oc]
        t[oc, ST.IN['tr4']] = (atm['gtempr'][i, j] ** 4)[oc]; t[oc, ST.IN['sss']] = atm['sss'][i, j][oc]
        lk = oc & (flake[i, j] > 0)
        t[lk, ST.IN['evaplim']] = evl[i, j][lk]; t[lk, ST.IN['htlim']] = htl[i, j][lk]
        t[oc, ST.IN['mwl']] = lake['mwl'][i, j][oc]; t[oc, ST.IN['gml']] = lake['gml'][i, j][oc]
        t[oc, ST.IN['ptype']] = pocean[i, j][oc]
        ic = t[:, 2] == 2
        if key == 'ta':     # substep-2 ice state is predicted from substep 1 (predict_ns2) from this template's columns only for constants
            t[ic, ST.IN['tg1']] = ag['gtemp'][i, j][ic]; t[ic, ST.IN['tg2']] = ag['gtemp2'][i, j][ic]
            t[ic, ST.IN['tr4']] = (ag['gtempr'][i, j] ** 4)[ic]
        t[ic, ST.IN['snow']] = ice['snowi'][i, j][ic]; t[ic, ST.IN['msi2']] = ice['msi'][i, j][ic]
        t[ic, ST.IN['ssi1']] = ice['ssi'][i, j, 0][ic]; t[ic, ST.IN['ssi2']] = ice['ssi'][i, j, 1][ic]
        t[ic, ST.IN['flag_dsws']] = ice['flag_dsws'][i, j][ic].astype(float)
        t[ic, ST.IN['tgo']] = tg1_oc[i, j][ic]; t[ic, ST.IN['ptype']] = poice[i, j][ic]
    for key in ('la', 'lb'):
        l = rec[key]
        i, j = _rows_ij(l)
        if key == 'la':     # substep-2 land-ice ground state is predicted from substep 1 (predict_ns2)
            l[:, LIT.IN['tg1']] = li['tlandi'][i, j, 0]; l[:, LIT.IN['tg2']] = li['tlandi'][i, j, 1]
            l[:, LIT.IN['tr4']] = (li['tlandi'][i, j, 0] + TF) ** 4
        l[:, LIT.IN['snow']] = li['snowli'][i, j]
    for key in ('pa', 'pb'):
        p = rec[key]
        i, j = _rows_ij(p)
        it = p[:, 2]
        psv = ps_ij[i, j]
        for itype in (1, 2):
            m = it == itype
            if not m.any():
                continue
            if itype == 2 and key == 'pb':
                p[m, 21] = atm['sss'][i, j][m]; p[m, 27] = ice['snowi'][i, j][m]
                continue
            if itype == 1:
                tg1 = tg1_oc[i, j][m]; tr4 = (atm['gtempr'][i, j] ** 4)[m]
            else:
                tg1 = ag['gtemp'][i, j][m]; tr4 = (ag['gtempr'][i, j] ** 4)[m]
            tg = tg1 + TF
            qs = np.asarray(P.qsat(_J(tg), _J(p[m, 19]), _J(psv[m])))
            qs = np.where(p[m, 22] > 0.5, 0.98 * qs, qs)
            p[m, 18] = tg; p[m, 6] = tg; p[m, 8] = qs; p[m, 9] = qs; p[m, 11] = tr4
            p[m, 21] = atm['sss'][i, j][m]
            if itype == 2:
                p[m, 27] = ice['snowi'][i, j][m]
        m = (it == 3) & (key == 'pa')
        if m.any():
            tg = li['tlandi'][i, j, 0][m] + TF
            qs = np.asarray(P.qsat(_J(tg), _J(p[m, 19]), _J(psv[m])))
            p[m, 18] = tg; p[m, 6] = tg; p[m, 8] = qs; p[m, 9] = qs; p[m, 11] = tg ** 4; p[m, 27] = li['snowli'][i, j][m]
    for key in ('blk1', 'blk2'):
        b = rec[key]
        i, j = _rows_ij(b)
        b[:, 2] = pocean[i, j]; b[:, 9] = poice[i, j]
    # tile-set report
    rep = {}
    t = rec['ta']
    have_o = np.zeros((IM, JM), bool); have_i = np.zeros((IM, JM), bool)
    i, j = _rows_ij(t[t[:, 2] == 1]); have_o[i, j] = True
    i, j = _rows_ij(t[t[:, 2] == 2]); have_i[i, j] = True
    ours_o = (pocean > 0) & geo['valid']; ours_i = (poice > 0) & geo['valid']
    rep['ocean_tiles_template'] = int(have_o.sum()); rep['ice_tiles_template'] = int(have_i.sum())
    rep['ocean_tiles_ours_without_template'] = int((ours_o & ~have_o).sum()); rep['ice_tiles_ours_without_template'] = int((ours_i & ~have_i).sum())
    rep['ocean_tiles_template_without_ours'] = int((have_o & ~ours_o).sum()); rep['ice_tiles_template_without_ours'] = int((have_i & ~ours_i).sum())
    rep['ptype_dropped_max'] = float(max(pocean[ours_o & ~have_o].max() if (ours_o & ~have_o).any() else 0.0,
                                          poice[ours_i & ~have_i].max() if (ours_i & ~have_i).any() else 0.0))
    return rec, rep


# ------------------------------------------------------------------------------------------------ closed SURFACE stage (copy of atm_step.stage_surface with the surface-state hooks)
def stage_surface_closed(S, R, ctx, rec, land_dyn=None, tm=None):
    """atm_step.stage_surface (D128), land_mode 'ghy', with exactly three differences: (1) `rec` (records already built from OUR surface state
    by apply_state_to_records) is passed in, (2) the substep-1 GHY state is `land_dyn` (our carried state; None = the recorded one), (3) the
    tile/land-ice/land inputs and outputs of both substeps are returned in S['_surface'] for the post-tile stage.  Nothing else is changed."""
    import time as _t
    M = A._surf_mods()
    C2, LC, TA, jnp_, SC = M['C2'], M['LC'], M['TA'], M['jnp'], M['SC']
    dt = 900.0
    atm1 = A.atm_layout(S, ctx)
    pa, pb = rec['pa'], rec['pb']
    a12, a3, a4 = pa[pa[:, 2] <= 2], pa[pa[:, 2] == 3], pa[pa[:, 2] == 4]
    b12, b3, b4 = pb[pb[:, 2] <= 2], pb[pb[:, 2] == 3], pb[pb[:, 2] == 4]
    coriol = np.zeros(atm1['T'].shape[:2])
    coriol[pb[:, 1].astype(int) - 1, pb[:, 0].astype(int) - 1] = pb[:, 36]
    cell1 = A.cell_inputs(S, atm1, ctx, S['USTARPBL'].T, S['LMONINPBL'].T, S['PBLHT'].T, S['DCLEV'].T, coriol, S['T1AA'].T, S['U1AA'].T, S['V1AA'].T)
    a12c, a3c, a4c = A.override_pbl(a12, cell1), A.override_pbl(a3, cell1), A.override_pbl(a4, cell1)
    ta, la = A.override_tiles(rec['ta'], rec['la'], atm1)
    g1, g2, blk1, blk2 = rec['g1'], rec['g2'], rec['blk1'], rec['blk2']
    _, patch1, _ = TA.unpack(blk1)
    lut1 = {(int(a), int(b)): k for k, (a, b) in enumerate(blk1[:, :2])}
    idx1 = np.array([lut1[(int(a), int(b))] for a, b in g1[:, :2]])
    trup = LC.infer_trup(g1, patch1['dth1'][idx1, 3], dt)
    t0 = _t.perf_counter()
    r1 = A._substep(a12c, ta, a3c, la, blk1, atm1, dt, dict(p4=a4c, g=g1, trup=trup, dyn=land_dyn))
    dth1, dq1 = A._grid(r1['comp']['dth1'], r1, 0.0), A._grid(r1['comp']['dq1'], r1, 0.0)
    S['TMOM'], S['QMOM'] = A.first_layer_update(S['TMOM'], S['QMOM'], dth1, dq1, S['T'][:, :, 0], S['Q'][:, :, 0], S['PK'][0])
    fl = M['CA'].aturb_flux_arrays(r1['comp'], atm1['MA1'][r1['cj'], r1['ci']], dt)
    ex1 = A._aturb(atm1, fl, blk1, dt)
    atm2 = dict(atm1)
    for k in ('T', 'Q', 'U', 'V', 'UA', 'VA'):
        atm2[k] = A._merge_valid(ex1[k], atm1[k], ex1['m'], k)
    atm2['E'] = A._merge_valid(ex1['E'], atm1['E'], ex1['m'], 'E')
    atm2['pblht'], atm2['dclev'] = ex1['pblht'], ex1['dclev']
    p12, p3, tnew, lnew, p4 = C2.predict_ns2(a12c, a3c, a4c, b12, b3, ta, rec['tb'], la, rec['lb'], r1, blk1, atm2, r1['ftype'], pb, b4)
    cell2 = dict(cell1)
    t1aa = atm2['T'][..., 0]
    cell2['dtdt'] = (A._tkv2(atm2) - t1aa * atm1['PEK1']) / dt
    cell2['u1aa'], cell2['v1aa'] = atm2['UA'][..., 0], atm2['VA'][..., 0]
    skip = ('zs1', 'tkv', 'dbl', 'ug', 'vg', 'utop', 'vtop', 'qtop', 'ztop')
    p12, p3 = (A.override_pbl(x, cell2, skip) for x in (p12, p3))
    p4 = A.override_pbl(p4, cell2, skip) if p4 is not None else None
    tnew2, lnew2 = A.override_tiles(tnew, lnew, atm2)
    r2 = A._substep(p12, tnew2, p3, lnew2, blk2, atm2, dt, dict(p4=p4, g=g2, trup=trup, dyn=r1['land']['dyn_next']))
    dth1, dq1 = A._grid(r2['comp']['dth1'], r2, 0.0), A._grid(r2['comp']['dq1'], r2, 0.0)
    S['TMOM'], S['QMOM'] = A.first_layer_update(S['TMOM'], S['QMOM'], dth1, dq1, np.transpose(atm2['T'], (1, 0, 2))[:, :, 0],
                                                np.transpose(atm2['Q'], (1, 0, 2))[:, :, 0], S['PK'][0])
    fl2 = M['CA'].aturb_flux_arrays(r2['comp'], atm2['MA1'][r2['cj'], r2['ci']], dt)
    ex2 = A._aturb(atm2, fl2, blk2, dt)
    if tm is not None:
        tm['surface'] = tm.get('surface', 0.0) + _t.perf_counter() - t0
    m = ex2['m']
    to_ijl = lambda x: np.transpose(np.asarray(x), (1, 0, 2))     # noqa: E731
    to_lij = lambda x: np.transpose(np.asarray(x), (2, 1, 0))     # noqa: E731
    S['T'] = to_ijl(A._merge_valid(ex2['T'], atm2['T'], m, 'T')); S['Q'] = to_ijl(A._merge_valid(ex2['Q'], atm2['Q'], m, 'Q'))
    S['U'] = to_ijl(A._merge_valid(ex2['U'], atm2['U'], m, 'U')); S['V'] = to_ijl(A._merge_valid(ex2['V'], atm2['V'], m, 'V'))
    S['UALIJ'] = to_lij(A._merge_valid(ex2['UA'], atm2['UA'], m, 'UA')); S['VALIJ'] = to_lij(A._merge_valid(ex2['VA'], atm2['VA'], m, 'VA'))
    S['EGCM'] = to_lij(A._merge_valid(ex2['E'], atm2['E'], m, 'E'))
    S['W2GCM'] = to_lij(A._merge_valid(ex2['W2'], np.transpose(S['W2GCM'], (2, 1, 0)), m, 'W2'))
    for k, kk in (('PBLHT', 'pblht'), ('DCLEV', 'dclev'), ('PBLPTOP', 'pblptop')):
        S[k] = np.where(m.T, np.asarray(ex2[kk]).T, S[k])
    S['T1AA'] = np.where(m.T, S['T'][:, :, 0], S['T1AA'])
    S['U1AA'], S['V1AA'] = np.where(m.T, S['UALIJ'][0], S['U1AA']), np.where(m.T, S['VALIJ'][0], S['V1AA'])
    S['TSAVG'] = A._grid_ij(r2['comp']['tsavg'], r2, S['TSAVG']); S['QSAVG'] = A._grid_ij(r2['comp']['qsavg'], r2, S['QSAVG'])
    ust, lmo = A.composite_ustar_lmonin(r2, p12, p3, p4 if p4 is not None else b4, pb)
    S['USTARPBL'], S['LMONINPBL'] = ust.T, lmo.T
    S['_surface'] = dict(r1=r1, r2=r2, ex1=ex1, ex2=ex2, ta=ta, tnew2=tnew2, la=la, lnew2=lnew2, g1=g1, g2=g2, p4_sub1=a4c, trup=trup)
    return S


def tile_outputs_to_acc(sd, st):
    """Accumulators of the post-tile stage from OUR tile outputs (both substeps), same names as tile_accumulators (replay)."""
    import surface_tile_ff as ST
    acc = {k: np.zeros((IM, JM)) for k in ('e0_o', 'evap_o', 'solar_o', 'dmua_o', 'dmva_o', 'e0_i', 'e1_i', 'evap_i', 'solar_i',
                                          'e0_li', 'e1_li', 'evap_li', 'rune', 'erune')}
    for rr, rows, lrows in ((sd['r1'], sd['ta'], sd['la']), (sd['r2'], sd['tnew2'], sd['lnew2'])):
        got = {k: np.asarray(v) for k, v in rr['tile'].items()}
        i, j = _rows_ij(rows)
        dts = rows[:, ST.IN['dtsurf']]
        srh = rows[:, ST.IN['srheat']] * dts
        o = rows[:, 2] == 1
        np.add.at(acc['e0_o'], (i[o], j[o]), got['f0dt'][o]); np.add.at(acc['evap_o'], (i[o], j[o]), got['evap'][o])
        np.add.at(acc['solar_o'], (i[o], j[o]), srh[o])
        np.add.at(acc['dmua_o'], (i[o], j[o]), got['dmua'][o] * dts[o]); np.add.at(acc['dmva_o'], (i[o], j[o]), got['dmva'][o] * dts[o])
        q = rows[:, 2] == 2
        np.add.at(acc['e0_i'], (i[q], j[q]), got['f0dt'][q]); np.add.at(acc['e1_i'], (i[q], j[q]), got['f1dt'][q])
        np.add.at(acc['evap_i'], (i[q], j[q]), got['evap'][q]); np.add.at(acc['solar_i'], (i[q], j[q]), srh[q])
        gl = {k: np.asarray(v) for k, v in rr['li'].items()}
        li_, lj_ = _rows_ij(lrows)
        np.add.at(acc['e0_li'], (li_, lj_), gl['f0dt']); np.add.at(acc['e1_li'], (li_, lj_), gl['f1dt']); np.add.at(acc['evap_li'], (li_, lj_), gl['evap'])
        gh = rr['land']['ghy']
        g = sd['g1']
        gi, gj = _rows_ij(g)
        np.add.at(acc['rune'], (gi, gj), np.asarray(gh['aruns']) + np.asarray(gh['arunu']))
        np.add.at(acc['erune'], (gi, gj), np.asarray(gh['aeruns']) + np.asarray(gh['aerunu']))
    return acc


# ------------------------------------------------------------------------------------------------ coupled run: atmosphere chain + closed surface
def recorded_boundary(date, it, ff=FF):
    """The recorded boundaries that remain in the coupled run: RIVERF outflow (oflowo, oeflowo, ffo tag 1) and the DYNSI result (odmui, odmvi, ustar)."""
    import ocean_chain_io as C
    sn = C.load_step(f"{ff}/{date}", it)
    return dict(flows=dict(oflowo=sn[1]['oflowo'], oeflowo=sn[1]['oeflowo']), dynsi=recorded_dynsi(date, it, ff))


class Loop:
    """Holds the evolving surface state and the per-step diagnostics of a coupled run (used by the monkey-patched atm_step.stage_surface)."""

    def __init__(self, date, daydir, ff, st, SS, rec_dir=None, closed_land=True, bnd=None):
        self.date, self.daydir, self.ff, self.st, self.SS = date, daydir, ff, st, SS
        self.closed_land = closed_land
        self.land_prev = None          # r2['land'] of the previous step (GHY state + PBL results) = our carried land state
        self.rows = []                 # per-step diagnostics
        self.melt = None
        self.bnd = bnd or (lambda it: recorded_boundary(date, it, ff))

    def pre_cse(self, R):
        """MELT_SI (ATM_DRV.f:257) before CONDSE; our RSI replaces the recorded CONDSE-entry RSI."""
        SS = self.SS
        ice, melt = melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], self.st['geo'])
        self.melt = (ice, melt)
        ci = R.cse_in
        self.rsi_recorded = np.array(ci['RSI'], copy=True)
        ci['RSI'] = np.array(ice['rsi'], copy=True)
        return ice['rsi']

    def stage_surface(self, S, R, ctx, rec=None, tm=None, land_mode='ghy'):
        import ghy_compare as GC
        import land_chain as LCm
        st = self.st
        it = R.itime
        inp = dict(prec=np.asarray(S['PREC']), eprec=np.asarray(S['EPREC']))
        rec0 = A.surface_records(R)
        g1 = rec0['g1']
        irrig = np.zeros((IM, JM)); irrig[g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1] = g1[:, 147]
        inp['irrig_act'] = irrig * st['fearth']
        S1, mid = surface_pre(self.SS, st, inp, melt_done=self.melt)
        rec, rep = apply_state_to_records(rec0, S1, mid, st, np.asarray(S['PEDN'][0]))
        # GHY precipitation forcing from OUR CONDSE (identity checked on the dumps: pr = PREC/(dtsrc*rhow), htpr = EPREC/dtsrc, prs = PRECSS/(dtsrc*rhow))
        for key in ('g1', 'g2'):
            g = rec[key]
            i, j = _rows_ij(g)
            g[:, 143] = np.asarray(S['PREC'])[i, j] / (DTSRC * RHOW)
            g[:, 144] = np.asarray(S['EPREC'])[i, j] / DTSRC
            g[:, 145] = np.asarray(S['PRECSS'])[i, j] / (DTSRC * RHOW)
        land_dyn = None
        if self.closed_land and self.land_prev is not None:
            lp = self.land_prev
            pa = rec['pa']
            m4 = pa[:, 2] == 4
            assert np.array_equal(lp['p4_ij'], pa[m4][:, :2]), 'land tile set changed between steps'
            pa4 = LCm.next_land_pbl_columns(pa[m4], lp)
            pa[m4] = pa4
            land_dyn = lp['dyn_next']
        S = stage_surface_closed(S, R, ctx, rec, land_dyn=land_dyn, tm=tm)
        sd = S['_surface']
        acc = tile_outputs_to_acc(sd, st)
        post_in = dict(acc=acc, srfp=np.asarray(S['PEDN'][0]), itime=it, **self.bnd(it))
        SSn, post = surface_post(S1, st, post_in, mid)
        land2 = dict(sd['r2']['land'])
        land2['p4_ij'] = sd['r2']['land']['p4_ij'] if 'p4_ij' in sd['r2']['land'] else rec['pa'][rec['pa'][:, 2] == 4][:, :2]
        self.land_prev = land2
        self.last = dict(S1=S1, mid=mid, post=post, rec_report=rep, acc=acc, rec=rec, SSn=SSn)
        self.SS = SSn
        S['_surf_state'] = SSn
        return S


def run_coupled(nsteps=6, date='nov26', daydir='nov26_day', it0=33312, ff=FF, imf=True, closed_land=True, out_dir=None, log=print, bnd=None):
    """Atmosphere chain (atm_step_fast, numpy dynamics, libimf pow, batched CONDSE, recorded radiation frozen as in D150) COUPLED to the
    closed surface loop of this module for `nsteps` consecutive steps from the real restart state.  Surface state (ocean, sea ice, lake,
    land ice, land GHY) is carried from OUR computation; the surface records of every step are built from it (apply_state_to_records).
    Recorded inputs that remain: radiation (SRHR/TRHR/COSZ1 frozen, tile SRHEAT/TRHR0 columns), Ent exports + land forcing columns of
    the ffg records, TRUP_in_rad, PBL profile columns, RIVERF outflow, DYNSI result, init_STRAITS MMST.
    Returns (rows, loop); rows = per-step atmosphere divergence from the real end state + surface diagnostics."""
    import time
    import atm_step as A_
    import atm_step_fast as F
    import atm_day_open_loop as OL
    import atm_day_report as RP
    ctx = A_.make_ctx(date, imf=imf, ff=ff)
    F.ensure_backend(ctx)
    st = load_statics(date, ff)
    st['ctx'] = make_ocean_ctx(date, ff)
    SS = init_surface_state(date, ff, st)
    loop = Loop(date, daydir, ff, st, SS, bnd=bnd)
    A_.stage_surface = loop.stage_surface
    axyp = ctx.gg['axyp']
    S, ms, rad, rows = None, {}, None, []
    for k in range(nsteps):
        it = it0 + k
        if rad is None or A_.is_radiation_step(it):
            rr = A_.Real(daydir, it, ff).site('r')
            rad = dict(SRHR=np.array(rr['SRHR']), TRHR=np.array(rr['TRHR']))
        R = OL.RealRad(daydir, it, ff, rad)
        loop.pre_cse(R)
        tm = {}
        t0 = time.perf_counter()
        with F.fast_condse():
            S, sn = A_.run_step(date, it, ctx, R=R, S=S, ms=ms, land_mode='ghy', timing=tm)
        wall = time.perf_counter() - t0
        real_e = R.e
        w = RP.weights(real_e['MA'], axyp)
        end = {f: S[f] for f in RP.FIELDS + ('P',)}
        stt = RP.state_metrics(end, real_e, w)
        row = dict(step=k, itime=it, wall=wall, stats=stt, tile_report=loop.last['rec_report'])
        rows.append(row)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            np.savez(f"{out_dir}/step_{it}.npz", **{f: np.asarray(end[f], float) for f in end})
        X = S.get('_condse_X')
        carry = {key: np.array(X[key], copy=True) for key in A_.CARRY_KEYS if X is not None and key in X}
        if '_cloud_rad' in S:
            carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
        S = {key: v for key, v in S.items() if not key.startswith('_')}
        if carry:
            S['_carry'] = carry
        log(f"step {k:2d} it {it} {wall:5.1f}s  rms T {stt['T']['rms']:.2e} Q {stt['Q']['rms']:.2e} U {stt['U']['rms']:.2e} V {stt['V']['rms']:.2e} "
            f"P {stt['P']['rms']:.2e} | tiles {loop.last['rec_report']}")
        sys.stdout.flush()
    return rows, loop
