"""D191 (stage S6): the STATE-dependent columns of the SURFACE templates on the device.

In the NumPy chain (surface_loop_v2.Loop2.stage_surface) the SURFACE record rows of a step are read from the real record (templates for everything static,
atmospheric or radiative) and then rewritten on the host from OUR surface state: `surface_loop.apply_state_to_records` (ground temperatures, humidity at the
ground, sss, snow, ice state, tile fractions) and the GHY precipitation forcing (ffg columns 143-145 from PREC / EPREC / PRECSS).  This module is the
device version for the fixed-shape slot templates of jax_surface (D188): the recorded template is uploaded ONCE per step (the declared record read), and
`apply_state` is one jitted function from (template, surface state after PRECIP/MELT_SI, PREC/EPREC/PRECSS, PEDN(1)) to the rewritten template, so the surface
state never visits the host.  It also carries the previous step's land state into the template (substep-1 GHY state `dyn_next` and the land PBL columns of
land_chain.next_land_pbl_columns) for chained steps.

Same arithmetic as the NumPy function: `x ** 4` of a float64 array is libm pow(x, 4.0) in NumPy, here jnp.power(x, 4.0) (D188).  Invalid slots (no tile) keep the
dummy row they were given (finite); only valid slots are rewritten.  Not ported: the tile-set report (host diagnostics of apply_state_to_records).

NEW module; nothing existing is edited.
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

import seaice_core_jax as SI  # noqa: E402
import pbl_ff as P  # noqa: E402
import surface_tile_ff as ST  # noqa: E402
import landice_tile_ff as LIT  # noqa: E402
import surface_loop as L  # noqa: E402
import jax_surface as JS  # noqa: E402

IM, JM = 72, 46
TF, DTSRC, RHOW, LHM, MINMLD = L.TF, L.DTSRC, L.RHOW, L.LHM, L.MINMLD


def _pow4(x):
    return jnp.power(x, 4.0)


def make_apply_state(host, K):
    """host: static index arrays of jax_surface.build_template; K: post-tile statics (jax_posttile.make_static_all, numpy).  Returns a jitted function
    apply(tpl, S1, ag, ps_ij, prec, eprec, precss, dyn_prev=None, land_prev=None) -> new tpl (same pytree structure as `tpl`)."""
    Nw = host['Nw']
    wj, wi = jnp.asarray(host['wj']), jnp.asarray(host['wi'])
    lj, li = jnp.asarray(host['lj']), jnp.asarray(host['li'])
    ej, ei = jnp.asarray(host['ej']), jnp.asarray(host['ei'])
    bcells = jnp.asarray(host['bcells'])
    flake = jnp.asarray(K['flake'])
    fwater = jnp.asarray(K['fwater'])
    axyp = jnp.asarray(K['axyp'])
    IN, LIN = ST.IN, LIT.IN

    def setcols(rows, cols, valid):
        for c, v in cols.items():
            rows = rows.at[:, c].set(jnp.where(valid, v, rows[:, c]))
        return rows

    def one_ns(ns, d, S1, ag, ps_ij, with_ice_ground, tg1_oc, pocean, poice, evl, htl):
        atm, ice, lake, lic = S1['atm'], S1['ice'], S1['lake'], S1['li']
        v1, v2 = d['wvalid'][0], d['wvalid'][1]
        tw = d['tw']
        tw1, tw2 = tw[:Nw], tw[Nw:]
        # ---- ffs ocean / lake water tile rows
        f = lambda fld: fld[wi, wj]                                   # (IM,JM) field at the water slots' cells        # noqa: E731
        lk = f(flake) > 0
        tw1 = setcols(tw1, {IN['tg1']: f(tg1_oc), IN['tg2']: f(atm['gtemp2']), IN['tr4']: f(_pow4(atm['gtempr'])), IN['sss']: f(atm['sss']),
                            IN['mwl']: f(lake['mwl']), IN['gml']: f(lake['gml']), IN['ptype']: f(pocean)}, v1)
        tw1 = setcols(tw1, {IN['evaplim']: f(evl), IN['htlim']: f(htl)}, v1 & lk)
        # ---- ffs ice tile rows
        cols2 = {IN['snow']: f(ice['snowi']), IN['msi2']: f(ice['msi']), IN['ssi1']: f(ice['ssi'][..., 0]), IN['ssi2']: f(ice['ssi'][..., 1]),
                 IN['flag_dsws']: f(ice['flag_dsws'].astype(jnp.float64)), IN['tgo']: f(tg1_oc), IN['ptype']: f(poice)}
        if with_ice_ground:
            cols2.update({IN['tg1']: f(ag['gtemp']), IN['tg2']: f(ag['gtemp2']), IN['tr4']: f(_pow4(ag['gtempr']))})
        tw2 = setcols(tw2, cols2, v2)
        out = dict(d, tw=jnp.concatenate([tw1, tw2], axis=0))
        # ---- land-ice tile rows (ffl)
        tl = d['tl']
        lcols = {LIN['snow']: lic['snowli'][li, lj]}
        if with_ice_ground:
            lcols.update({LIN['tg1']: lic['tlandi'][li, lj, 0], LIN['tg2']: lic['tlandi'][li, lj, 1], LIN['tr4']: _pow4(lic['tlandi'][li, lj, 0] + TF)})
        out['tl'] = setcols(tl, lcols, jnp.ones(tl.shape[0], bool))
        # ---- PBL rows (ffp): water slots
        pw = d['pw']
        p1, p2 = pw[:Nw], pw[Nw:]
        psv = f(ps_ij)

        def ground(p, tg1, tr4, valid, snow=None):
            tg = tg1 + TF
            qs = P.qsat(tg, p[:, 19], psv)
            qs = jnp.where(p[:, 22] > 0.5, 0.98 * qs, qs)
            cols = {18: tg, 6: tg, 8: qs, 9: qs, 11: tr4, 21: f(atm['sss'])}
            if snow is not None:
                cols[27] = snow
            return setcols(p, cols, valid)
        p1 = ground(p1, f(tg1_oc), f(_pow4(atm['gtempr'])), v1)
        if with_ice_ground:
            p2 = ground(p2, f(ag['gtemp']), f(_pow4(ag['gtempr'])), v2, snow=f(ice['snowi']))
        else:
            p2 = setcols(p2, {21: f(atm['sss']), 27: f(ice['snowi'])}, v2)
        out['pw'] = jnp.concatenate([p1, p2], axis=0)
        # ---- land-ice PBL rows (substep 1 only)
        if with_ice_ground:
            pl = d['pl']
            tg = lic['tlandi'][li, lj, 0] + TF
            qs = P.qsat(tg, pl[:, 19], ps_ij[li, lj])
            out['pl'] = setcols(pl, {18: tg, 6: tg, 8: qs, 9: qs, 11: _pow4(tg), 27: lic['snowli'][li, lj]}, jnp.ones(pl.shape[0], bool))
        # ---- tile fractions of the cells (blk columns 2 and 9)
        ft = d['ftype']
        pf = lambda x: x.T.reshape(-1)                                 # (IM,JM) -> flat j*IM+i          # noqa: E731
        ft = ft.at[bcells, 0].set(pf(pocean)[bcells]).at[bcells, 1].set(pf(poice)[bcells])
        out['ftype'] = ft
        return out

    @jax.jit
    def apply(tpl, S1, ag, ps_ij, prec, eprec, precss, dyn_prev, land_prev):
        atm, ice, lake = S1['atm'], S1['ice'], S1['lake']
        tfz = SI.tfrez(atm['sss'])
        tg1_oc = jnp.maximum(atm['gtemp'], tfz)
        mwl, gml = lake['mwl'], lake['gml']
        small = mwl < MINMLD * RHOW * flake * axyp
        evl = jnp.where(small, jnp.maximum(0.5 * (mwl / (flake * axyp) - 0.4 * RHOW), 0.0), mwl / (flake * axyp) - (0.5 * MINMLD + 0.2) * RHOW)
        htl = gml / (flake * axyp) + 0.5 * LHM * evl
        pocean = (1.0 - ice['rsi']) * fwater
        poice = ice['rsi'] * fwater
        nsl = [one_ns(k, tpl['ns'][k], S1, ag, ps_ij, k == 0, tg1_oc, pocean, poice, evl, htl) for k in (0, 1)]
        # previous step's land state: substep-1 land PBL columns (land_chain.next_land_pbl_columns) on the template rows, and GHY start state
        if land_prev is not None:
            nsl[0] = dict(nsl[0], pe=JS.next_land_pbl_columns(nsl[0]['pe'], land_prev))
        # GHY precipitation forcing (identity checked on the dumps: pr = PREC/(dtsrc*rhow), htpr = EPREC/dtsrc, prs = PRECSS/(dtsrc*rhow))
        pr0 = prec[ei, ej] / (DTSRC * RHOW)
        htpr = eprec[ei, ej] / DTSRC
        prs0 = precss[ei, ej] / (DTSRC * RHOW)
        pr = jnp.maximum(pr0, 0.0)
        prs = jnp.minimum(jnp.maximum(prs0, 0.0), pr)
        htprs = jnp.where(pr <= 0.0, 0.0, htpr / jnp.where(pr <= 0.0, 1.0, pr) * prs)
        ghy = []
        for k in (0, 1):
            g = dict(tpl['ghy'][k])
            fc = dict(g['forcing'])
            fc.update(pr=pr, prs=prs, htpr=htpr, htprs=htprs)
            g['forcing'] = fc
            if k == 0 and dyn_prev is not None:
                g['dyn0'] = dyn_prev
            ghy.append(g)
        return dict(tpl, ns=nsl, ghy=ghy)
    return apply
