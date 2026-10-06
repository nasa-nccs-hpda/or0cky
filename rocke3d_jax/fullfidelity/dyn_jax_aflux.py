"""D142: AFLUX + ADVECM + MAtoP in JAX (jitted port of dyn_aflux_ff.aflux / advecm / matop, numpy-pow semantics).

Same statement order as the numpy port (left-to-right products, strictly sequential reductions: the Fortran SUM loops are
lax.scan carries, never jnp.sum), AVRX through the traced FFT72 of dyn_jax_fft (avrx_field_jax), the topography
adjustment as an unrolled level loop on the (static) patch blocks with per-cell masks, the MW column recursion as a
reverse lax.scan.  Geometry/constants are TRACED arguments (dict `geo`), not closure constants, because XLA rewrites
operations on closure constants (see dyn_jax_env / D141).  Needs the FMA-free XLA flag (dyn_jax_env).

make_aflux(g, tab) -> AfluxKit with .aflux(ns, u, v, ma, masum, me, mesum) -> dict(mu, mv, mw, conv, spa, spa0) and
.advecm(dt1, mold, conv, mw) -> dict(mnew, msum, pedn, pmid, pdsig, pk, p, n_exception) (jax arrays; n_exception int array).
PK = PMID**KAPA is jnp.power (numpy-pow semantics, as in the numpy chain; the Intel libimf pow is not available here).
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import dyn_jax_fft as jf

IM, JM, LM = 72, 46, 40
TWOBY3 = 2 / 3.0


def seqsum(a, axis=0):
    """Strictly sequential (left-to-right) sum along `axis` as a scan carry (jnp.sum may reassociate)."""
    a = jnp.moveaxis(a, axis, 0)
    s, _ = lax.scan(lambda c, x: (c + x, None), a[0], a[1:])
    return s


_GEO_VEC = ('dyp', 'dxv', 'dxyp', 'bydxyp', 'mfix', 'mfrac')
_GEO_SC = ('polwt', 'byim', 'dt', 'mfixs', 'kg2mb', 'mtop', 'kapa')


def aflux_geo(g, tab):
    d = {k: jnp.asarray(np.asarray(g[k], dtype=np.float64)) for k in _GEO_VEC}
    d.update({k: jnp.asarray(float(g[k])) for k in _GEO_SC})
    d['maxcolmass'] = jnp.asarray(float(g['maxcolmass']))
    d['mincolmass'] = jnp.asarray(float(g['mincolmass']))
    act, coef, mask = jf.avrx_plan(range(1, JM - 1), g, tab)
    d['avrx_coef'] = jnp.asarray(coef)
    d['avrx_mask'] = jnp.asarray(mask)
    d['imaxj'] = jnp.asarray(np.asarray(g['imaxj'], dtype=np.int64))
    return d, act


def _patch_plan(g):
    """Static (host) tables of the topography patches: [(mode, 'ew'|'ns', I, J, iup, idn, xx, act)] in the numpy order."""
    zat = g['zatmo']
    plan = []
    for adjmode in (0, 1):
        for nn in range(int(g['npatch'])):
            if g['md'][nn] != adjmode:
                continue
            ip = g['ipatch'][:, nn]
            jp = g['jpatch'][:, nn]
            for d, jlo, jhi in (('ew', max(jp[0], 2), min(jp[1], JM - 1)), ('ns', max(jp[0], max(3, 2)), min(jp[1], JM - 1))):
                I = np.arange(ip[0] - 1, ip[1])
                J = np.arange(jlo - 1, jhi)
                if J.size == 0:
                    continue
                Ig, Jg = np.meshgrid(I, J, indexing='ij')
                if d == 'ew':
                    Ip1 = (Ig + 1) % IM
                    z0 = zat[Ig, Jg]; z1 = zat[Ip1, Jg]
                    lt = z0 < z1; gt = z0 > z1
                    iup = np.where(lt, Ip1, Ig); idn = np.where(lt, Ig, Ip1)
                    jup = Jg; jdn = Jg
                else:
                    z0 = zat[Ig, Jg - 1]; z1 = zat[Ig, Jg]
                    lt = z0 < z1; gt = z0 > z1
                    jup = np.where(lt, Jg, Jg - 1); jdn = np.where(lt, Jg - 1, Jg)
                    iup = Ig; idn = Ig
                plan.append(dict(mode=int(adjmode), dir=d, i0=int(ip[0] - 1), i1=int(ip[1]), j0=int(jlo - 1), j1=int(jhi),
                                 iup=iup, jup=jup, idn=idn, jdn=jdn, xx=np.where(lt, 1.0, -1.0), act=lt | gt))
    return plan


def _topo_block(blk, ma_t, masum, p):
    """One patch (ATMDYN.f 616-704) on the block blk = mu[I,J,:] (or mv): per-cell state over the level loop."""
    mup = masum[p['iup'], p['jup']]
    mdn = masum[p['idn'], p['jdn']]
    mad = ma_t[p['idn'], p['jdn'], :]
    xx = jnp.asarray(p['xx'])
    alive = jnp.asarray(p['act'])
    for l in range(LM - 1):
        mdn = mdn - mad[:, :, l]
        alive = alive & ~(mdn < mup)
        cur = blk[:, :, l]
        if p['mode'] == 0:
            blk = blk.at[:, :, l].set(jnp.where(alive, 0.0, cur))
        else:
            m = alive & (xx * cur > 0.0)
            blk = blk.at[:, :, l + 1].set(jnp.where(m, blk[:, :, l + 1] + cur, blk[:, :, l + 1]))
            blk = blk.at[:, :, l].set(jnp.where(m, 0.0, cur))
    return blk


def matop_jax(ma, masum, geo):
    """MAtoP (numpy: dyn_aflux_ff.matop with numpy pow).  ma (LM,IM,JM), masum (IM,JM)."""
    kg2mb, mtop, kapa = geo['kg2mb'], geo['mtop'], geo['kapa']
    p = kg2mb * (masum - geo['mfixs'])

    def body(m, mal):
        pedn = kg2mb * (m + mal)
        pmid = kg2mb * (m + mal * .5)
        pdsig = kg2mb * mal
        pk = jnp.power(pmid, kapa)
        return m + mal, (pedn, pmid, pdsig, pk)
    _, out = lax.scan(body, jnp.full(ma.shape[1:], mtop), ma[::-1])
    pedn, pmid, pdsig, pk = (o[::-1] for o in out)
    return dict(pedn=pedn, pmid=pmid, pdsig=pdsig, pk=pk, p=p)


def make_aflux(g, tab, polefix=True, topo=True):
    geo, act = aflux_geo(g, tab)
    plan = _patch_plan(g) if (g['aflux_topo'] and topo) else []
    do_polefix = (g['do_polefix'] == 1 and polefix)

    @jax.jit
    def aflux_jax(ns, u0, v0, ma, masum, me, mesum, geo):
        dyp, dxv, dxyp = geo['dyp'], geo['dxv'], geo['dxyp']
        polwt, byim = geo['polwt'], geo['byim']
        zNSxDT = 1 / (ns * geo['dt'])
        u = u0.at[:, 1, :].set(polwt * u0[:, 1, :] + (1 - polwt) * u0[:, 2, :])
        v = v0.at[:, 1, :].set(polwt * v0[:, 1, :] + (1 - polwt) * v0[:, 2, :])
        u = u.at[:, JM - 1, :].set(polwt * u[:, JM - 1, :] + (1 - polwt) * u[:, JM - 2, :])
        v = v.at[:, JM - 1, :].set(polwt * v[:, JM - 1, :] + (1 - polwt) * v[:, JM - 2, :])
        ma_t = ma.transpose(1, 2, 0)
        spa0 = jnp.zeros((IM, JM, LM)).at[:, 1:JM - 1, :].set(u[:, 1:JM - 1, :] + u[:, 2:JM, :])
        spa = jf.avrx_field_jax(spa0, act, geo['avrx_coef'], geo['avrx_mask'])
        mav = ma_t + jnp.roll(ma_t, -1, axis=0)
        mu = jnp.zeros((IM, JM, LM)).at[:, 1:JM - 1, :].set(
            (.25 * dyp[None, 1:JM - 1, None] * spa[:, 1:JM - 1, :]) * mav[:, 1:JM - 1, :])
        vsum = v[:, 1:JM, :] + jnp.roll(v, 1, axis=0)[:, 1:JM, :]
        mv = jnp.zeros((IM, JM, LM)).at[:, 1:JM, :].set(
            ((.25 * dxv[None, 1:JM, None]) * vsum) * (ma_t[:, 1:JM, :] + ma_t[:, 0:JM - 1, :]))
        mvsa = mvna = None
        for pole in (0, 1):
            if pole == 0:
                jv, jma, jdyp = 1, 0, dyp[1]
            else:
                jv, jma, jdyp = JM - 1, JM - 1, dyp[JM - 2]
            m1 = ma_t[0, jma, :]
            mus = seqsum(u0[:, jv, :], 0) * byim * .25 * jdyp * m1
            mvs = seqsum(mv[:, jv, :], 0) * byim
            d = mv[1:, jv, :] - mvs
            _, dtail = lax.scan(lambda c, x: (c + x, c + x), jnp.zeros(LM), d)
            dum = jnp.concatenate([jnp.zeros((1, LM)), dtail], axis=0)          # dum[0]=0, dum[i]=dum[i-1]+(mv[i]-mvs)
            pb = seqsum(dum, 0) * byim
            if pole == 0:
                x = (pb[None, :] - dum) + mus[None, :]
                mvsa = mvs
            else:
                x = (dum - pb[None, :]) + mus[None, :]
                mvna = mvs
            pj = 0 if pole == 0 else JM - 1
            spa = spa.at[:, pj, :].set((4 * x) / (jdyp * m1)[None, :])
            mu = mu.at[:, pj, :].set(3 * x)
        if do_polefix:
            mu = mu.at[:, 0, :].set(mu[:, 0, :] * TWOBY3)
            mu = mu.at[:, JM - 1, :].set(mu[:, JM - 1, :] * TWOBY3)
        for p in plan:
            sl = (slice(p['i0'], p['i1']), slice(p['j0'], p['j1']), slice(None))
            if p['dir'] == 'ew':
                mu = mu.at[sl].set(_topo_block(mu[sl], ma_t, masum, p))
            else:
                mv = mv.at[sl].set(_topo_block(mv[sl], ma_t, masum, p))
        conv = jnp.zeros((IM, JM, LM)).at[:, 1:JM - 1, :].set(
            (((jnp.roll(mu, 1, axis=0)[:, 1:JM - 1, :] - mu[:, 1:JM - 1, :]) + mv[:, 1:JM - 1, :]) - mv[:, 2:JM, :]))
        conv = conv.at[0, 0, :].set(-mvsa)
        conv = conv.at[0, JM - 1, :].set(mvna)
        mfix, mfrac = geo['mfix'], geo['mfrac']
        convs = seqsum(conv, 2)
        mvars = mesum - geo['mfixs']
        me_t = me.transpose(1, 2, 0)
        dxyp_b = dxyp[None, :]
        mwl = ((conv[:, :, LM - 1] - convs * mfrac[LM - 1])
               + (((me_t[:, :, LM - 1] - (mfix[LM - 1] + mvars * mfrac[LM - 1])) * dxyp_b) * zNSxDT))

        def body(mwn, xs):
            cl, ml, mf, mfr = xs
            r = (((mwn + cl) - convs * mfr) + (((ml - (mf + mvars * mfr)) * dxyp_b) * zNSxDT))
            return r, r
        xs = (jnp.moveaxis(conv[:, :, 1:LM - 1], 2, 0), jnp.moveaxis(me_t[:, :, 1:LM - 1], 2, 0), mfix[1:LM - 1], mfrac[1:LM - 1])
        _, rest = lax.scan(body, mwl, xs, reverse=True)                              # levels 0..LM-3
        mw = jnp.concatenate([jnp.moveaxis(rest, 0, 2), mwl[:, :, None]], axis=2)     # (IM,JM,LM-1)
        mw = mw.at[1:, 0, :].set(mw[0, 0, :][None, :])
        mw = mw.at[1:, JM - 1, :].set(mw[0, JM - 1, :][None, :])
        return dict(mu=mu, mv=mv, mw=mw, conv=conv, spa=spa, spa0=spa0)

    @jax.jit
    def advecm_jax(dt1, mold, conv, mw, geo):
        b = geo['bydxyp'][None, :]
        mtop = geo['mtop']
        conv_t = conv.transpose(2, 0, 1)
        mw_t = mw.transpose(2, 0, 1)
        top = mold[LM - 1] + (dt1 * (conv_t[LM - 1] - mw_t[LM - 2])) * b
        mid_old = mold[1:LM - 1]
        mid_c, mid_w, mid_wm = conv_t[1:LM - 1], mw_t[1:LM - 1], mw_t[0:LM - 2]
        # levels LM-2..1 (0-based), msum accumulates top-down: scan in reverse over l
        def body(msum, xs):
            mo, c, w, wm = xs
            mn = mo + (dt1 * ((c + w) - wm)) * b
            return msum + mn, mn
        msum, mid = lax.scan(body, top, (mid_old, mid_c, mid_w, mid_wm), reverse=True)
        bot = mold[0] + (dt1 * (conv_t[0] + mw_t[0])) * b
        msum = msum + bot
        mnew = jnp.concatenate([bot[None], mid, top[None]], axis=0)
        tot = msum + mtop
        imx = geo['imaxj']
        valid = jnp.arange(IM)[:, None] < imx[None, :]
        e1 = jnp.any(valid & ((tot > geo['maxcolmass']) | (tot < geo['mincolmass'])))
        e2 = jnp.any(valid & ((tot > geo['maxcolmass'] * (1200. / 1160.)) | (tot < geo['mincolmass'] * (250. / 350.))))
        ex = jnp.where(e1, jnp.where(e2, 2, 1), 0)
        mnew = mnew.at[:, 1:, 0].set(mnew[:, 0, 0][:, None])
        mnew = mnew.at[:, 1:, JM - 1].set(mnew[:, 0, JM - 1][:, None])
        msum = msum.at[1:, 0].set(msum[0, 0])
        msum = msum.at[1:, JM - 1].set(msum[0, JM - 1])
        out = matop_jax(mnew, msum, geo)
        out.update(mnew=mnew, msum=msum, n_exception=ex)
        return out

    class AfluxKit:
        pass
    kit = AfluxKit()
    kit.geo = geo
    kit.aflux = lambda ns, u, v, ma, masum, me, mesum: aflux_jax(float(ns), u, v, ma, masum, me, mesum, geo)
    kit.advecm = lambda dt1, mold, conv, mw: advecm_jax(float(dt1), mold, conv, mw, geo)
    return kit
