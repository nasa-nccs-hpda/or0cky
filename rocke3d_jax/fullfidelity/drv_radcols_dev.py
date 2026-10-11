"""D218: device (jnp) version of drv_radcols (D176): the radiation-derived columns of the SURFACE template computed from OUR radiation state instead of read from the
real tile records, inside the assembled step (jax_coupled.Coupled, ROCKE_RADCOLS=1).

State (all jnp, (IM,JM) layout of drv_radcols.RadSurf): FSF(4,IM,JM), TRSURF(4,IM,JM), TRHR0, SRVIS, FSRDIR.  Radiation steps: loaded from the server output;
other steps: the ice-fraction legs of RESET_SURF_FLUXES (a) after MELT_SI (non-radiation steps only) and (b), (c), (F) at the end of every step (drv_radcols_couple.CoupledRad).
Columns written into the template `tpl` (jax_surface.build_template layout), exactly the ones drv_radcols.fill_records writes into the records:
  tw (ffs rows):  srheat[15] = FSF(type) * COSZ1, trhr0[24] = TRHR0, trup[80] = TRSURF(type)      (water slots, both types; valid slots only)
  pw (ffp rows):  trhr0[17] = TRHR0, qsol[20] = FSF(type) * COSZ1
  pl/pe (ffp rows of land ice / land): same with type 3 / 4
  tl (ffl rows):  srheat[8] = FSF(3) * COSZ1, flong[12] = TRHR0, trup[14] = TRSURF(3)
  ghy forcing:    srht = FSF(4) * COSZ1, trht = TRHR0 (the land patch); tpl['trup'] = TRSURF(4)
Ca, cosz1, vis_rad, dvis of the ffg rows enter only the Ent computation, which is not ported (Ent exports stay recorded), so they are not written here.
Not done here (declared): the day-boundary RESET_SURF_FLUXES of daily_LAKE (host RadSurf.daily_lake, applied by the caller at the boundary).
"""
import jax
import jax.numpy as jnp
import numpy as np

import drv_radcols as DR

FFS, FFP, FFL = DR.FFS, DR.FFP, DR.FFL
LAST = {}          # last radiation-server output (set by the server wrapper on the host: LAST['out'], LAST['it'])


def rs_from_server(out):
    """device radiation state from a radiation-server output dict"""
    return dict(FSF=jnp.asarray(out['FSF'], dtype=jnp.float64), TRSURF=jnp.asarray(out['TRSURF'], dtype=jnp.float64),
                TRHR0=jnp.asarray(np.asarray(out['TRHR'])[0], dtype=jnp.float64), SRVIS=jnp.asarray(out['SRVISSURF'], dtype=jnp.float64),
                FSRDIR=jnp.asarray(out['FSRDIR'], dtype=jnp.float64))


def _leg(F, rsi_old, rsi_new, mask):
    """drv_radcols.reset_surf_fluxes_ice on (4,IM,JM) device arrays (same formulas, same operation order)"""
    up = (rsi_new > rsi_old) & mask
    dn = (rsi_new < rsi_old) & mask
    f1 = jnp.where(up, (F[1] * rsi_old + F[0] * (rsi_new - rsi_old)) / jnp.where(up, rsi_new, 1.0), F[1])
    fo, fn = 1.0 - rsi_old, 1.0 - rsi_new
    f0 = jnp.where(dn, (F[0] * fo + F[1] * (fn - fo)) / jnp.where(dn, fn, 1.0), F[0])
    return F.at[1].set(f1).at[0].set(f0)


@jax.jit
def _ice_leg_jit(rs, rsi_old, rsi_new, mask):
    out = dict(rs)
    out['FSF'] = _leg(rs['FSF'], rsi_old, rsi_new, mask)
    out['TRSURF'] = _leg(rs['TRSURF'], rsi_old, rsi_new, mask)
    return out


def ice_leg(rs, rsi_old, rsi_new, mask):
    """RESET_SURF_FLUXES for the ice fraction change on `mask` cells (FSF and TRSURF); one jit program"""
    return _ice_leg_jit(rs, rsi_old, rsi_new, mask)


def fill_tpl(tpl, rs, cosz1, host):
    """tpl with the radiation columns computed from `rs` and `cosz1` (IM,JM); a new pytree, tpl is not modified.  One jit program; the index arrays are arguments
    (so the program survives the daily_LAKE rebuild)."""
    idx = {k: jnp.asarray(host[k]) for k in ('wi', 'wj', 'li', 'lj', 'ei', 'ej')}
    return _fill_jit(tpl, rs, cosz1, idx, Nw=int(host['Nw']))


def _fill_impl(tpl, rs, cosz1, idx, Nw):
    wi, wj, li, lj, ei, ej = (idx[k] for k in ('wi', 'wj', 'li', 'lj', 'ei', 'ej'))
    FSF, TRS, TRH = rs['FSF'], rs['TRSURF'], rs['TRHR0']
    out = dict(tpl)
    nsl = []
    for ns, d in enumerate(tpl['ns']):
        d = dict(d)
        tw, pw = d['tw'], d['pw']
        for ty in (0, 1):
            sl = slice(ty * Nw, (ty + 1) * Nw)
            v = d['wvalid'][ty]
            cz = cosz1[wi, wj]
            tw = tw.at[sl, FFS['srheat']].set(jnp.where(v, FSF[ty, wi, wj] * cz, tw[sl, FFS['srheat']]))
            tw = tw.at[sl, FFS['trhr0']].set(jnp.where(v, TRH[wi, wj], tw[sl, FFS['trhr0']]))
            tw = tw.at[sl, FFS['trup']].set(jnp.where(v, TRS[ty, wi, wj], tw[sl, FFS['trup']]))
            pw = pw.at[sl, FFP['trhr0']].set(jnp.where(v, TRH[wi, wj], pw[sl, FFP['trhr0']]))
            pw = pw.at[sl, FFP['qsol']].set(jnp.where(v, FSF[ty, wi, wj] * cz, pw[sl, FFP['qsol']]))
        pl, pe, tl = d['pl'], d['pe'], d['tl']
        pl = pl.at[:, FFP['trhr0']].set(TRH[li, lj]).at[:, FFP['qsol']].set(FSF[2, li, lj] * cosz1[li, lj])
        pe = pe.at[:, FFP['trhr0']].set(TRH[ei, ej]).at[:, FFP['qsol']].set(FSF[3, ei, ej] * cosz1[ei, ej])
        tl = tl.at[:, FFL['srheat']].set(FSF[2, li, lj] * cosz1[li, lj]).at[:, FFL['flong']].set(TRH[li, lj]).at[:, FFL['trup']].set(TRS[2, li, lj])
        d.update(tw=tw, pw=pw, pl=pl, pe=pe, tl=tl)
        nsl.append(d)
    out['ns'] = nsl
    ghy = []
    for b in tpl['ghy']:
        b = dict(b)
        f = dict(b['forcing'])
        f['srht'] = FSF[3, ei, ej] * cosz1[ei, ej]
        f['trht'] = TRH[ei, ej]
        b['forcing'] = f
        if 'entf' in b:                                  # inputs of the live Ent call (GHY_DRV.f:1191-1193): cosz1, vis_rad = SRVISSURF*COSZ1*.82 (REAL*4), direct_vis_rad = vis_rad*FSRDIR
            ef = dict(b['entf'])
            cz = cosz1[ei, ej]
            vis = rs['SRVIS'][ei, ej] * cz * DR.F82
            ef.update(cosz1=cz, vis=vis, dvis=vis * rs['FSRDIR'][ei, ej])
            b['entf'] = ef
        ghy.append(b)
    out['ghy'] = ghy
    out['trup'] = TRS[3, ei, ej]
    return out


_fill_jit = jax.jit(_fill_impl, static_argnames=('Nw',))
