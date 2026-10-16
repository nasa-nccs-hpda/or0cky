"""D139: SDRAG and isotropuv in JAX.

sdrag_field_jax: the column port dyn_sdrag_ff.sdrag_columns applied to all B-grid columns J=2..JM as whole-array jnp
(level loop L=LS1..LM unrolled, columns are independent), the T-range stop_model check is returned as a flag and
raised on the host.
make_iso: isotropuv on the rows with COSV<0.15 (shap1 with per-row sub-iteration counts through lax.while_loop, pole
rows through the traced FFT72 of dyn_jax_fft).
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import dyn_jax_fft as jf
import dyn_isotropuv_ff as fi

IM, JM, LM = 72, 46, 40
F32_015 = float(np.float32(.15))


def make_sdrag(p, geo):
    ii, jj = np.meshgrid(np.arange(IM), np.arange(1, JM), indexing='ij')
    I = ii.ravel(); J = jj.ravel(); ip1 = (I + 1) % IM
    jcol = J + 1
    g = {k: np.asarray(geo[k], dtype=np.float64) for k in ('cosv', 'rapvn', 'rapvs', 'dxyv', 'dxyn', 'dxys')}
    cosv = g['cosv'][jcol - 1]; rapvn = g['rapvn'][jcol - 2]; rapvs = g['rapvs'][jcol - 1]
    dxyv = g['dxyv'][jcol - 1]; dxyn = g['dxyn'][jcol - 2]; dxys = g['dxys'][jcol - 1]
    ls1, lsdrag, lpsdrag = p['ls1'], p['lsdrag'], p['lpsdrag']
    wmax = p['wmax']; wmaxp = wmax * 3. / 4.
    polar = cosv <= F32_015
    wmaxj = np.where(polar, wmaxp, wmax)
    assert p['ang_sdrag'] == 1

    @jax.jit
    def sdrag_jax(u, v, t, pk, pedn, ma, dt1):
        uc = u[I, J, :]; vc = v[I, J, :]
        tc = t[I, J, :]; pkc = pk[:, I, J].T; pe1 = pedn[1:, I, J].T
        m_pj = ma[:, ip1, J - 1].T; m_ij = ma[:, I, J - 1].T; m_pJ = ma[:, ip1, J].T; m_iJ = ma[:, I, J].T
        R = uc.shape[0]
        ang = jnp.zeros(R)
        bad = jnp.zeros((), bool)
        for L in range(ls1, LM + 1):
            l = L - 1
            cd_lin = (L >= lsdrag) | ((L >= lpsdrag) & polar)
            tl = tc[:, l] * pkc[:, l]
            bad = bad | jnp.any((tl < 100.) | (tl > 373.))
            rho = 100. * pe1[:, l] / (p['rgas'] * tl)
            wl = jnp.sqrt(uc[:, l] * uc[:, l] + vc[:, l] * vc[:, l])
            q = p['wc_jdrag'] / (p['wc_jdrag'] + jnp.minimum(wl, wmaxj))
            xjud = q * q
            cdn = p['csdragl'][l] * xjud
            cdn_lin = (p['x_sdrag'][0] + p['x_sdrag'][1] * jnp.minimum(wl, wmaxj)) * xjud
            cdn = jnp.where(cd_lin, cdn_lin, cdn)
            mauv = (m_pj[:, l] + m_ij[:, l]) * rapvn + (m_pJ[:, l] + m_iJ[:, l]) * rapvs
            x = dt1 * rho * cdn * jnp.minimum(wl, wmaxj) * p['vsdragl'][l] / mauv
            over = wl > wmaxj
            xc = 1. - (1. - x) * wmaxj / wl
            x = jnp.where(over, xc, x)
            dut = -x * mauv * dxyv * uc[:, l]
            ang = ang - dut
            uc = uc.at[:, l].set(uc[:, l] * (1. - x))
            vc = vc.at[:, l].set(vc[:, l] * (1. - x))
        lmax = ls1 - 1
        s = jnp.zeros(R)
        for l in range(lmax):
            mm = .5 * ((m_pj[:, l] + m_ij[:, l]) * dxyn + (m_pJ[:, l] + m_iJ[:, l]) * dxys)
            s = s + mm
        du = ang / s
        uc = uc.at[:, :lmax].set(uc[:, :lmax] + du[:, None])
        return u.at[I, J, :].set(uc), v.at[I, J, :].set(vc), bad
    return sdrag_jax


def make_iso(geo, dt=fi.DT, coscut=fi.COS_LIMIT):
    rows = [j for j in range(2, JM + 1) if not fi.far_from_pole(j, geo['cosv'], coscut)]
    jj = np.array([j for l in range(LM) for j in rows])
    ll = np.array([l for l in range(LM) for j in rows])
    hemi = np.array([fi.hemisphere(j, geo['fjeq']) for j in jj], dtype=float)[:, None]
    dxv = np.asarray(geo['dxv'], dtype=np.float64)[jj - 1]
    pole = np.array([fi.at_pole(j) for j in jj])
    pidx = np.nonzero(pole)[0]
    cosi = jnp.asarray(np.asarray(geo['cosiv'], dtype=np.float64))[None, :]
    sini = jnp.asarray(np.asarray(geo['siniv'], dtype=np.float64))[None, :]

    def shap1(x, fac):
        n = fac.astype(int) + 1
        facby4 = fac * .25 / n
        nmax = jnp.max(n)

        def cond(c):
            return c[0] <= nmax

        def body(c):
            nn, xx = c
            new = xx + facby4[:, None] * (((jnp.roll(xx, 1, axis=1) - xx) - xx) + jnp.roll(xx, -1, axis=1))
            return nn + 1, jnp.where((n >= nn)[:, None], new, xx)
        return lax.while_loop(cond, body, (jnp.asarray(1), x))[1]

    consts = dict(cosi=cosi, sini=sini, hemi=jnp.asarray(hemi), dxv=jnp.asarray(dxv), dt=jnp.asarray(float(dt)))

    @jax.jit
    def _iso(u, v, c):
        # constants are traced arguments, not closure constants: XLA's simplifier rewrites ops on constants
        # (e.g. x / const, (x*c1)*c2) in ways that are not bit-identical
        cosi, sini, hemi, dxv, dt = c['cosi'], c['sini'], c['hemi'], c['dxv'], c['dt']
        U = u[:, jj - 1, ll].T
        V = v[:, jj - 1, ll].T
        ua = cosi * U - (hemi * sini) * V
        va = cosi * V + (hemi * sini) * U
        umax = jnp.max(jnp.abs(U), axis=1)
        k = umax * 2. * dt / dxv
        kk = jnp.where(k < 0.5, fi.KLO, jnp.where(k > 1.0, fi.KHI, fi.KLO + 2. * (k - 0.5) * (fi.KHI - fi.KLO)))
        fac = kk * dt / (dxv * dxv)
        ua = shap1(ua, fac)
        va = shap1(va, fac)
        if len(pidx):
            def pole_filter(a):
                A, B = jf.fft_rows_jax(a[pidx].T)
                A = A.at[2:jf.IMH + 1].set(0.); B = B.at[2:jf.IMH + 1].set(0.)
                return a.at[pidx].set(jf.ffti_rows_jax(A, B).T)
            ua = pole_filter(ua); va = pole_filter(va)
        Uo = cosi * ua + (hemi * sini) * va
        Vo = cosi * va - (hemi * sini) * ua
        return u.at[:, jj - 1, ll].set(Uo.T), v.at[:, jj - 1, ll].set(Vo.T)
    return lambda u, v: _iso(u, v, consts)
