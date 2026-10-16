"""D191 (stage S6, unit J3): DISSIP and FILTER of the coupled step as ONE device program (jnp), the port of atm_step.stage_dissip + atm_step.stage_filter.

NEW module; nothing existing is edited; SOCRATES/RADIA untouched.  Same statement order as the NumPy originals (dyn_glue_ff.dissip, dyn_filter_ff.filter_slp /
slp / shap1d / isotropslp / row_loop / matopmb / total_energy / add_energy_as_diffuse_heat); every strictly sequential Fortran sum is a lax.scan carry
(jax_p1_glue.seqsum); the libimf `pow` sites of the NumPy imf chain (SLP pow, MAtoPMB PK, PEK) go through libimf_ops (labelled host callback) when the
process runs in libimf mode, otherwise through jnp.power (numpy-pow semantics of the libm-mode chain).  Constants of the filter come from ctx.fg exactly as in
the NumPy stage (ctx.gg for PEK); they are passed as traced arguments, not closed over.

Declared limits: the SLP "exp" branch (BETA <= 1e-6) uses jnp.exp; it is selected in no cell of the three dates (counted by the test) and numpy's exp is not
bitwise-guaranteed to equal XLA's; the isotropslp row counts and factors are static (they depend on constants only), computed on the host with the NumPy
expressions of dyn_isotropuv_ff.shap1_rows.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401  (XLA flags BEFORE jax)
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import jax_p1_glue as G  # noqa: E402
import dyn_jax_filter as DF  # noqa: E402
import libimf_ops as LI  # noqa: E402

IM, JM, LM = 72, 46, 40
F64 = np.float64
NSHAP = 8

SCALARS = ('grav', 'rgas', 'sha', 'kapa', 'bygrav', 'bmoist', 'by3', 'mb2kg', 'kg2mb', 'mtop', 'mfixs', 'psf', 'pmtop', 'areag', 'byim')
ARRAYS = ('dxyp', 'dxyv', 'rapvs', 'rapvn', 'dxyn', 'dxys', 'mfix', 'mfrac', 'zatmo', 'axyp', 'byaxyp')


def make_consts(ctx):
    """Traced-argument constants of the filter (ctx.fg) and of DISSIP / PEK (ctx.gg); plus the static python data of isotropslp and the layer range."""
    fg, gg = ctx.fg, ctx.gg
    K = {k: jnp.asarray(F64(fg[k])) for k in SCALARS}
    K.update({k: jnp.asarray(np.asarray(fg[k], dtype=np.float64)) for k in ARRAYS})
    K['two_by3'] = jnp.asarray(F64(2. * fg['by3']))
    # DISSIP uses ctx.gg (sha, byim) and IMAXJ
    K['gg_sha'] = jnp.asarray(F64(gg['sha']))
    K['gg_byim'] = jnp.asarray(F64(gg['byim']))
    K['gg_kapa'] = jnp.asarray(F64(gg['kapa']))
    imaxj = np.asarray(gg['imaxj'], dtype=int)
    mask = np.zeros((IM, JM), bool)
    for j in range(JM):
        mask[:imaxj[j], j] = True
    K['imaxj_mask'] = jnp.asarray(mask)
    # static data of isotropslp (host, numpy expressions of dyn_isotropuv_ff.shap1_rows / dyn_filter_ff.isotropslp)
    cosp, dxp = np.asarray(fg['cosp']), np.asarray(fg['dxp'])
    rows = []
    for j in range(2, JM):
        if cosp[j - 1] >= fg['cos_limit']:
            continue
        fac = np.array([1e3 * fg['dt'] / (dxp[j - 1] * dxp[j - 1])])
        n = int(fac.astype(int)[0] + 1)
        facby4 = float((fac * .25 / (fac.astype(int) + 1))[0])
        rows.append((j - 1, n, facby4))
    static = dict(iso_rows=tuple(rows), l1=int(fg['lmfrac1']) - 1, l2=int(fg['lmfrac2']) - 1)
    return K, static


def _pow(x, y, mask=None):
    return LI.pow(x, y, mask)


# ----------------------------------------------------------------------------- DISSIP
def dissip(U, V, KEA, T, PK, K):
    ke = DF.calc_kea_3d_jax(U, V, K['gg_byim'])
    dke = ke - KEA
    pkt = jnp.transpose(PK, (1, 2, 0))
    tn = jnp.where(K['imaxj_mask'][:, :, None], T - dke / (K['gg_sha'] * pkt), T)
    return tn, ke


# ----------------------------------------------------------------------------- FILTER pieces
def slp(ps, tas, zs, K):
    bmoist, grav, rgas = K['bmoist'], K['grav'], K['rgas']
    m = zs != 0.0
    tsl = tas + bmoist * zs
    tasn = tas
    beta = jnp.full(ps.shape, bmoist)
    c1 = m & (tas < 290.5) & (tsl > 290.5)
    beta = jnp.where(c1, (290.5 - tas) / zs, beta)
    c2 = m & (tas > 290.5) & (tsl > 290.5)
    tasn = jnp.where(c2, 0.5 * (290.5 + tas), tasn)
    c3 = m & (tas < 255)
    tasn = jnp.where(c3, 0.5 * (255.0 + tas), tasn)
    bzbyt = beta * zs / tasn
    gbyrb = grav / (rgas * beta)
    cb = m & (beta > 1e-6)
    ce = m & ~cb
    a = _pow(1. + bzbyt, gbyrb, cb)
    r_pow = ps * a
    r_exp = ps * jnp.exp((1. - 0.5 * bzbyt + _pow(bzbyt, K['two_by3'], ce)) * gbyrb * bzbyt)
    return jnp.where(cb, r_pow, jnp.where(ce, r_exp, ps)), ce


def shap1d(x, norder=NSHAP):
    by4ton = 1. / 4. ** norder
    xs = x[:, 1:JM - 1]
    for _ in range(norder):
        xs = ((jnp.roll(xs, 1, axis=0) - xs) - xs) + jnp.roll(xs, -1, axis=0)
    return x.at[:, 1:JM - 1].set(x[:, 1:JM - 1] - xs * by4ton)


def isotropslp(x, static):
    for jj, n, facby4 in static['iso_rows']:
        row = x[:, jj]
        for _ in range(n):
            row = row + facby4 * (((jnp.roll(row, 1) - row) - row) + jnp.roll(row, -1))
        x = x.at[:, jj].set(row)
    return x


def row_loop(x, y, pednold, K):
    sl = slice(1, JM - 1)
    po = pednold[:, sl]
    v = x[:, sl] / y[:, sl]
    v = jnp.maximum(v, 0.9882 * po)
    v = jnp.minimum(v, 1.0118 * po)
    psumo = G.seqsum(po, axis=0)
    psumn = G.seqsum(v, axis=0)
    pdif = (psumn - psumo) * K['byim']
    return pednold.at[:, sl].set(v - pdif)


def conserv_pe(masum, t, pk, ma, K):
    tt = jnp.transpose(t, (2, 0, 1))
    s = G.seqsum((tt * pk) * ma, axis=0)
    tpe = K['zatmo'] * (masum + K['mtop']) + K['sha'] * s
    tpe = tpe.at[1:, 0].set(tpe[0, 0])
    return tpe.at[1:, JM - 1].set(tpe[0, JM - 1])


def total_energy(masum, ma, pk, t, u, v, K):
    kea = G.conserv_ke(ma, u, v, K)
    pe = conserv_pe(masum, t, pk, ma, K)
    te = ((kea + pe) * K['axyp']) / K['areag']
    return G.global_sum_ij(te)


def filter_slp(pedn1, tsavg, ma, pk, t, q, qcl, qci, qmom, u, v, K, static):
    l1, l2 = static['l1'], static['l2']
    mfix, mfrac, mtop, mfixs, mb2kg = K['mfix'], K['mfrac'], K['mtop'], K['mfixs'], K['mb2kg']
    masum0 = G.seqsum(ma[::-1], axis=0)                       # the MAtoPMB recurrence (masum = ma[l] + masum, l = LM..1)
    e0 = total_energy(masum0, ma, pk, t, u, v, K)
    mabef, pkold, pednold = ma, pk, pedn1
    zs = K['zatmo'] * K['bygrav']
    x, ce = slp(pednold, tsavg, zs, K)
    y = x / pednold
    x = shap1d(x, NSHAP)
    x = isotropslp(x, static)
    pn = row_loop(x, y, pednold, K)
    sl = slice(1, JM - 1)
    man = ma
    mvar = pn[:, sl] * mb2kg - mfixs - mtop
    for l in range(l1, l2 + 1):
        man = man.at[l, :, sl].set(mfix[l] + mvar * mfrac[l])
    mp = G.matopmb(man, K)
    tn, qn, qcln, qcin, qmn = t, q, qcl, qci, qmom
    for l in range(l1, l2 + 1):
        zmrat = mabef[l][:, sl] / man[l][:, sl]
        tn = tn.at[:, sl, l].set(t[:, sl, l] * pkold[l][:, sl] / mp['pk'][l][:, sl])
        qn = qn.at[:, sl, l].set(q[:, sl, l] * zmrat)
        qcln = qcln.at[:, sl, l].set(qcl[:, sl, l] * zmrat)
        qcin = qcin.at[:, sl, l].set(qci[:, sl, l] * zmrat)
        qmn = qmn.at[:, :, sl, l].set(qmom[:, :, sl, l] * zmrat[None])
    e1 = total_energy(mp['masum'], man, mp['pk'], tn, u, v, K)
    ediff = (e1 - e0) / ((K['psf'] - K['pmtop']) * K['sha'] * mb2kg)
    tn2 = tn - ediff / jnp.transpose(mp['pk'], (1, 2, 0))
    return dict(pedn=mp['pedn'], pmid=mp['pmid'], pk=mp['pk'], ma=man, masum=mp['masum'], t=tn2, q=qn, qcl=qcln, qci=qcin, qmom=qmn,
                p=mp['p'], pdsig=mp['pdsig']), ce


# ----------------------------------------------------------------------------- the unit J3
IN_KEYS = ('U', 'V', 'KEA', 'T', 'PK', 'PEDN', 'TSAVG', 'MA', 'Q', 'QCL', 'QCI', 'QMOM')
OUT_KEYS = ('T', 'Q', 'QCL', 'QCI', 'QMOM', 'PEDN', 'PMID', 'PK', 'MA', 'MASUM', 'P', 'PDSIG', 'PEK', 'KEA_NEW')


def make_unit(static):
    """Returns the jitted J3: (S_in dict of device arrays, K) -> (updated entries, dissip-stage extras)."""
    @jax.jit
    def j3(Sin, K):
        tn, ke = dissip(Sin['U'], Sin['V'], Sin['KEA'], Sin['T'], Sin['PK'], K)
        o, ce = filter_slp(Sin['PEDN'][0], Sin['TSAVG'], Sin['MA'], Sin['PK'], tn, Sin['Q'], Sin['QCL'], Sin['QCI'], Sin['QMOM'], Sin['U'], Sin['V'], K, static)
        pedn = o['pedn']
        # PEK = PEDN ** KAPA (atm_step.stage_filter: ctx.kapa from gg)
        pek = _pow(pedn, K['gg_kapa'])
        out = dict(T=o['t'], Q=o['q'], QCL=o['qcl'], QCI=o['qci'], QMOM=o['qmom'], PEDN=pedn, PMID=o['pmid'], PK=o['pk'], MA=o['ma'], MASUM=o['masum'],
                   P=o['p'], PDSIG=o['pdsig'], PEK=pek, KEA_NEW=ke)
        return out, dict(T_dissip=tn, n_slp_exp_branch=jnp.sum(ce))
    return j3
