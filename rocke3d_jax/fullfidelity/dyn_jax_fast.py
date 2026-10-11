"""D182: compile-time-friendly drop-in variants of the D139-D145 JAX dynamics units.  Arithmetic (operation order, constants, dtype)
is the same as in the originals, only the data movement around it differs (static slices instead of gathers/scatters over
constant index arrays, per-level column lists instead of `.at[:, l].set` chains), so results are bitwise identical to the
originals under the same XLA flags and core count (checked in tests/test_dyn_fast.py and d182_unit.py digests).
Import dyn_jax_env (or clouds_jax_env) BEFORE this module.
"""
import numpy as np
import dyn_jax_env  # noqa: F401
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

IM, JM, LM = 72, 46, 40
F32_015 = float(np.float32(.15))


def make_sdrag2(p, geo):
    """Same signature/outputs as dyn_jax_pointwise.make_sdrag(...)(u, v, t, pk, pedn, ma, dt1) -> (u, v, bad)."""
    ii, jj = np.meshgrid(np.arange(IM), np.arange(1, JM), indexing='ij')
    J = jj.ravel()
    jcol = J + 1
    g = {k: np.asarray(geo[k], dtype=np.float64) for k in ('cosv', 'rapvn', 'rapvs', 'dxyv', 'dxyn', 'dxys')}
    cosv = g['cosv'][jcol - 1]; rapvn = g['rapvn'][jcol - 2]; rapvs = g['rapvs'][jcol - 1]
    dxyv = g['dxyv'][jcol - 1]; dxyn = g['dxyn'][jcol - 2]; dxys = g['dxys'][jcol - 1]
    ls1, lsdrag, lpsdrag = p['ls1'], p['lsdrag'], p['lpsdrag']
    wmax = p['wmax']; wmaxp = wmax * 3. / 4.
    polar = cosv <= F32_015
    wmaxj = np.where(polar, wmaxp, wmax)
    assert p['ang_sdrag'] == 1
    R = IM * (JM - 1)
    NJ = JM - 1

    def flat(a):                       # (IM, NJ, X) -> (R, X), row = i*NJ + (j-1): the raveled (I, J) order of the original
        return a.reshape(R, a.shape[-1])

    @jax.jit
    def sdrag_jax(u, v, t, pk, pedn, ma, dt1):
        uc = flat(u[:, 1:JM, :]); vc = flat(v[:, 1:JM, :])
        tc = flat(t[:, 1:JM, :])
        pkc = flat(jnp.moveaxis(pk[:, :, 1:JM], 0, 2))
        pe1 = flat(jnp.moveaxis(pedn[1:, :, 1:JM], 0, 2))
        mr = jnp.roll(ma, -1, axis=1)               # ma[:, (I+1) % IM, :]
        m_pj = flat(jnp.moveaxis(mr[:, :, 0:JM - 1], 0, 2)); m_ij = flat(jnp.moveaxis(ma[:, :, 0:JM - 1], 0, 2))
        m_pJ = flat(jnp.moveaxis(mr[:, :, 1:JM], 0, 2)); m_iJ = flat(jnp.moveaxis(ma[:, :, 1:JM], 0, 2))
        ang = jnp.zeros(R)
        bad = jnp.zeros((), bool)
        ucs = [uc[:, l] for l in range(LM)]
        vcs = [vc[:, l] for l in range(LM)]
        for L in range(ls1, LM + 1):
            l = L - 1
            cd_lin = (L >= lsdrag) | ((L >= lpsdrag) & polar)
            tl = tc[:, l] * pkc[:, l]
            bad = bad | jnp.any((tl < 100.) | (tl > 373.))
            rho = 100. * pe1[:, l] / (p['rgas'] * tl)
            wl = jnp.sqrt(ucs[l] * ucs[l] + vcs[l] * vcs[l])
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
            dut = -x * mauv * dxyv * ucs[l]
            ang = ang - dut
            ucs[l] = ucs[l] * (1. - x)
            vcs[l] = vcs[l] * (1. - x)
        lmax = ls1 - 1
        s = jnp.zeros(R)
        for l in range(lmax):
            mm = .5 * ((m_pj[:, l] + m_ij[:, l]) * dxyn + (m_pJ[:, l] + m_iJ[:, l]) * dxys)
            s = s + mm
        du = ang / s
        for l in range(lmax):
            ucs[l] = ucs[l] + du
        uo = jnp.stack(ucs, axis=1).reshape(IM, NJ, LM); vo = jnp.stack(vcs, axis=1).reshape(IM, NJ, LM)
        return u.at[:, 1:JM, :].set(uo), v.at[:, 1:JM, :].set(vo), bad
    return sdrag_jax


def make_sdrag2b(p, geo):
    """Same signature/outputs as dyn_jax_pointwise.make_sdrag(...)(u, v, t, pk, pedn, ma, dt1) -> (u, v, bad)."""
    ii, jj = np.meshgrid(np.arange(IM), np.arange(1, JM), indexing='ij')
    J = jj.ravel()
    jcol = J + 1
    g = {k: np.asarray(geo[k], dtype=np.float64) for k in ('cosv', 'rapvn', 'rapvs', 'dxyv', 'dxyn', 'dxys')}
    cosv = g['cosv'][jcol - 1]; rapvn = g['rapvn'][jcol - 2]; rapvs = g['rapvs'][jcol - 1]
    dxyv = g['dxyv'][jcol - 1]; dxyn = g['dxyn'][jcol - 2]; dxys = g['dxys'][jcol - 1]
    ls1, lsdrag, lpsdrag = p['ls1'], p['lsdrag'], p['lpsdrag']
    wmax = p['wmax']; wmaxp = wmax * 3. / 4.
    polar = cosv <= F32_015
    wmaxj = np.where(polar, wmaxp, wmax)
    assert p['ang_sdrag'] == 1
    R = IM * (JM - 1)
    NJ = JM - 1

    def flat(a):                       # (IM, NJ, X) -> (R, X), row = i*NJ + (j-1): the raveled (I, J) order of the original
        return a.reshape(R, a.shape[-1])

    @jax.jit
    def sdrag_jax(u, v, t, pk, pedn, ma, dt1):
        uc = flat(u[:, 1:JM, :]); vc = flat(v[:, 1:JM, :])
        tc = flat(t[:, 1:JM, :])
        pkc = flat(jnp.moveaxis(pk[:, :, 1:JM], 0, 2))
        pe1 = flat(jnp.moveaxis(pedn[1:, :, 1:JM], 0, 2))
        mr = jnp.roll(ma, -1, axis=1)               # ma[:, (I+1) % IM, :]
        m_pj = flat(jnp.moveaxis(mr[:, :, 0:JM - 1], 0, 2)); m_ij = flat(jnp.moveaxis(ma[:, :, 0:JM - 1], 0, 2))
        m_pJ = flat(jnp.moveaxis(mr[:, :, 1:JM], 0, 2)); m_iJ = flat(jnp.moveaxis(ma[:, :, 1:JM], 0, 2))
        ang = jnp.zeros(R)
        bad = jnp.zeros((), bool)
        ucs = [uc[:, l] for l in range(LM)]
        vcs = [vc[:, l] for l in range(LM)]
        for L in range(ls1, LM + 1):
            l = L - 1
            cd_lin = (L >= lsdrag) | ((L >= lpsdrag) & polar)
            tl = tc[:, l] * pkc[:, l]
            bad = bad | jnp.any((tl < 100.) | (tl > 373.))
            rho = 100. * pe1[:, l] / (p['rgas'] * tl)
            wl = jnp.sqrt(ucs[l] * ucs[l] + vcs[l] * vcs[l])
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
            dut = -x * mauv * dxyv * ucs[l]
            ang = ang - dut
            ucs[l] = ucs[l] * (1. - x)
            vcs[l] = vcs[l] * (1. - x)
            ucs[l], vcs[l], ang, bad = jax.lax.optimization_barrier((ucs[l], vcs[l], ang, bad))
        lmax = ls1 - 1
        s = jnp.zeros(R)
        for l in range(lmax):
            mm = .5 * ((m_pj[:, l] + m_ij[:, l]) * dxyn + (m_pJ[:, l] + m_iJ[:, l]) * dxys)
            s = s + mm
        du = ang / s
        for l in range(lmax):
            ucs[l] = ucs[l] + du
        uo = jnp.stack(ucs, axis=1).reshape(IM, NJ, LM); vo = jnp.stack(vcs, axis=1).reshape(IM, NJ, LM)
        return u.at[:, 1:JM, :].set(uo), v.at[:, 1:JM, :].set(vo), bad
    return sdrag_jax
