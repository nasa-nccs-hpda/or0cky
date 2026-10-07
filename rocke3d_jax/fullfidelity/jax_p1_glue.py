"""D186 (stage S2): device ports of the dynamics GLUE that was NumPy in D180 (TROP, MAtoPMB, SE/KE bookkeeping, energy fix, PGRAD_PBL, QCL/QCI rescale,
the pole fix of DIAGA, the RADIA cloud masking, the CONDSE momentum back-transfer and recalc_agrid_uv).

Every function is a pure jnp function on device arrays with the SAME statement order as the NumPy original (dyn_glue_ff, dyn_filter_ff, atm_step,
clouds_condse_ff); strictly sequential Fortran sums are lax.scan carries; constants come in as traced float64 arguments (dict K from `make_consts`),
never as closure constants.  Nothing here changes a result by design: the tests compare each function bitwise with its NumPy original on the real
step-0 state.  Needs the XLA flags of clouds_jax_env_fast (imported first).

Declared limits: the Intel libimf `pow` is not available in these functions (jnp.power, numpy-pow semantics, the libm-mode reference); stop_model
checks that the Fortran has are returned as flags, they are not raised inside the device program.
"""
import clouds_jax_env_fast  # noqa: F401  (XLA flags BEFORE jax)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

IM, JM, LM = 72, 46, 40
F64 = np.float64


def seqsum(a, axis=0):
    """Strictly sequential (left-to-right) sum along `axis` (ifort -fp-model strict SUM) as a scan carry."""
    a = jnp.moveaxis(a, axis, 0)
    s, _ = lax.scan(lambda c, x: (c + x, None), a[0], a[1:])
    return s


def global_sum_ij(a):
    return seqsum(seqsum(a, axis=0), axis=0)


def make_consts(ctx):
    """Device constants of the glue from an atm_step ctx (ctx.gg = glue consts, ctx.g = aflux consts)."""
    gg, g = ctx.gg, ctx.dyn.g
    f = lambda x: jnp.asarray(F64(x))                                         # noqa: E731
    a = lambda x: jnp.asarray(np.asarray(x, dtype=np.float64))               # noqa: E731
    K = dict(sha=f(gg['sha']), lhe=f(gg['lhe']), lhm=f(gg['lhm']), mtop=f(gg['mtop']), byim=f(gg['byim']), rgas=f(gg['rgas']),
             grav=f(gg['grav']), kapa=f(gg['kapa']), bykapa=f(gg['bykapa']), psf=f(gg['psf']), bygrav=f(gg['bygrav']), dtsrc=f(gg['dtsrc']),
             kg2mb=f(g['kg2mb']), mfixs=f(g['mfixs']),
             zatmo=a(gg['zatmo']), axyp=a(gg['axyp']), byaxyp=a(gg['byaxyp']), dxyn=a(gg['dxyn']), dxys=a(gg['dxys']), dxyp=a(gg['dxyp']),
             dxyv=a(gg['dxyv']), rapvs=a(gg['rapvs']), rapvn=a(gg['rapvn']), bydyp=a(gg['bydyp']), bydxp=a(gg['bydxp']),
             cosip=a(gg['cosip']), sinip=a(gg['sinip']))
    return K


# ----------------------------------------------------------------------------- MAtoPMB (dyn_filter_ff.matopmb)
@jax.jit
def matopmb(ma, K):
    """ma (LM,IM,JM) -> dict(masum, pedn (LM+1,..), pmid, pk, pdsig, p).  Same recurrence as dyn_filter_ff.matopmb (numpy-pow PK)."""
    kg2mb, kapa = K['kg2mb'], K['kapa']
    pedn_top = K['mtop'] * kg2mb

    def body(c, mal):
        masum, pup = c
        masum = mal + masum
        pdsig = mal * kg2mb
        pmid = pup + pdsig * .5
        pedn = pup + pdsig
        pk = jnp.power(pmid, kapa)
        return (masum, pedn), (pedn, pmid, pk, pdsig)
    z = jnp.zeros(ma.shape[1:])
    (masum, _), (pedn, pmid, pk, pdsig) = lax.scan(body, (z, jnp.full(ma.shape[1:], pedn_top)), ma[::-1])
    pedn, pmid, pk, pdsig = pedn[::-1], pmid[::-1], pk[::-1], pdsig[::-1]
    pedn = jnp.concatenate([pedn, jnp.full((1,) + ma.shape[1:], pedn_top)], axis=0)
    return dict(masum=masum, pedn=pedn, pmid=pmid, pk=pk, pdsig=pdsig, p=(masum - K['mfixs']) * kg2mb)


@jax.jit
def pek_of(pedn, K):
    return jnp.power(pedn, K['kapa'])


# ----------------------------------------------------------------------------- CONSERV_SE, CONSERV_KE, energy fix
@jax.jit
def conserv_se(ma, masum, pk, t, q, qci, K):
    """dyn_glue_ff.conserv_se: L = LM..1 sequential, poles replicated from I=1."""
    sha, lhe, lhm = K['sha'], K['lhe'], K['lhm']
    se = jnp.zeros((IM, JM))
    for l in range(LM - 1, -1, -1):
        mi = ma[l] * qci[:, :, l]
        mv = ma[l] * q[:, :, l]
        md = ma[l]
        se = ((se + ((sha * md) * t[:, :, l]) * pk[l]) + lhe * mv) - lhm * mi
    se = se + K['zatmo'] * (masum + K['mtop'])
    se = se.at[1:, 0].set(se[0, 0])
    return se.at[1:, JM - 1].set(se[0, JM - 1])


@jax.jit
def conserv_ke(ma, u, v, K):
    """dyn_filter_ff.conserv_ke(...)[0] * 1 : returns KEA * BYAXYP (what the plan stores as KEINIT / KEFINAL)."""
    dxyn, dxys = K['dxyn'], K['dxys']
    mj0 = ma[:, :, 0:JM - 1]
    mj1 = ma[:, :, 1:JM]
    a = (mj0 + jnp.roll(mj0, -1, axis=1)) * dxyn[None, None, 0:JM - 1] + (mj1 + jnp.roll(mj1, -1, axis=1)) * dxys[None, None, 1:JM]
    ut = jnp.transpose(u, (2, 0, 1))[:, :, 1:JM]
    vt = jnp.transpose(v, (2, 0, 1))[:, :, 1:JM]
    rke = jnp.zeros((IM, JM)).at[:, 1:JM].set(seqsum(a * (ut * ut + vt * vt), axis=0) * .25)
    # regrid_btoa_ext
    byim, dxyp, dxyv, rapvs, rapvn = K['byim'], K['dxyp'], K['dxyv'], K['rapvs'], K['rapvn']
    new = rke
    new = new.at[:, 0].set(seqsum(rke[:, 1]) * byim * (dxyp[0] / dxyv[1]))
    r0 = rke[:, 1:JM - 1]
    r1 = rke[:, 2:JM]
    mid = (jnp.roll(r0, 1, axis=0) + r0) * rapvs[None, 1:JM - 1] + (jnp.roll(r1, 1, axis=0) + r1) * rapvn[None, 1:JM - 1]
    new = new.at[:, 1:JM - 1].set(mid)
    new = new.at[:, JM - 1].set(seqsum(rke[:, JM - 1]) * byim * (dxyp[JM - 1] / dxyv[JM - 1]))
    return new * K['byaxyp']


@jax.jit
def energy_fix(sei, kei, sef, kef, masum, t, pk, K):
    """dyn_glue_ff.energy_fix: returns (t_new, dsepke, mmglob)."""
    sef2 = ((sef - sei) + (kef - kei)) * K['axyp']
    kef2 = masum * K['axyp']
    dse = global_sum_ij(sef2)
    mmg = global_sum_ij(kef2)
    dse = dse / mmg
    pkt = jnp.transpose(pk, (1, 2, 0))
    return t - dse / (pkt * K['sha']), dse, mmg


# ----------------------------------------------------------------------------- PGRAD_PBL
@jax.jit
def pgrad_pbl(t1, pk1, pmid1, pedn1, phi1, K):
    """dyn_glue_ff.pgrad_pbl (inputs (IM,JM)); only the cells the Fortran sets are filled, the others 0."""
    rgas, byim, zatmo = K['rgas'], K['byim'], K['zatmo']
    bydyp, bydxp, cosip, sinip = K['bydyp'], K['bydxp'], K['cosip'], K['sinip']
    sl, up, dn = slice(1, JM - 1), slice(2, JM), slice(0, JM - 2)
    by_rho1 = ((rgas * t1[:, sl]) * pk1[:, sl]) / (100. * pmid1[:, sl])
    dpdy_m = ((((100 * (pmid1[:, up] - pmid1[:, dn])) * by_rho1 + phi1[:, up]) - phi1[:, dn]) * bydyp[None, sl]) * .5
    dpdy0_m = ((((100 * (pedn1[:, up] - pedn1[:, dn])) * by_rho1 + zatmo[:, up]) - zatmo[:, dn]) * bydyp[None, sl]) * .5
    pm_ip, pm_im = jnp.roll(pmid1, -1, axis=0)[:, sl], jnp.roll(pmid1, 1, axis=0)[:, sl]
    ph_ip, ph_im = jnp.roll(phi1, -1, axis=0)[:, sl], jnp.roll(phi1, 1, axis=0)[:, sl]
    pe_ip, pe_im = jnp.roll(pedn1, -1, axis=0)[:, sl], jnp.roll(pedn1, 1, axis=0)[:, sl]
    za_ip, za_im = jnp.roll(zatmo, -1, axis=0)[:, sl], jnp.roll(zatmo, 1, axis=0)[:, sl]
    dpdx_m = ((((100 * (pm_ip - pm_im)) * by_rho1 + ph_ip) - ph_im) * bydxp[None, sl]) * .5
    dpdx0_m = ((((100 * (pe_ip - pe_im)) * by_rho1 + za_ip) - za_im) * bydxp[None, sl]) * .5
    dpdx = jnp.zeros((IM, JM)).at[:, sl].set(dpdx_m)
    dpdy = jnp.zeros((IM, JM)).at[:, sl].set(dpdy_m)
    dpdx0 = jnp.zeros((IM, JM)).at[:, sl].set(dpdx0_m)
    dpdy0 = jnp.zeros((IM, JM)).at[:, sl].set(dpdy0_m)
    for jp, j1, hemi in ((0, 1, -1.0), (JM - 1, JM - 2, 1.0)):
        def body(c, xs):
            a1, b1, a0, b0 = c
            px, py, px0, py0, co, si = xs
            a1 = a1 + (px * co - (hemi * py) * si)
            b1 = b1 + (py * co + (hemi * px) * si)
            a0 = a0 + (px0 * co - (hemi * py0) * si)
            b0 = b0 + (py0 * co + (hemi * px0) * si)
            return (a1, b1, a0, b0), None
        z = jnp.zeros(())
        (a1, b1, a0, b0), _ = lax.scan(body, (z, z, z, z), (dpdx[:, j1], dpdy[:, j1], dpdx0[:, j1], dpdy0[:, j1], cosip, sinip))
        dpdx = dpdx.at[0, jp].set(a1 * byim)
        dpdy = dpdy.at[0, jp].set(b1 * byim)
        dpdx0 = dpdx0.at[0, jp].set(a0 * byim)
        dpdy0 = dpdy0.at[0, jp].set(b0 * byim)
    return dpdx, dpdy, dpdx0, dpdy0


# ----------------------------------------------------------------------------- QCL/QCI rescale, DIAGA pole fix, masks
@jax.jit
def qscale(x, maold, ma):
    """x (IM,JM,LM) * (MAOLD/MA) per level (dyn_step 'qscale')."""
    return x * jnp.transpose(maold / ma, (1, 2, 0))


@jax.jit
def diaga_poles(q):
    q = q.at[1:, 0, :].set(q[0, 0, :][None, :])
    return q.at[1:, JM - 1, :].set(q[0, JM - 1, :][None, :])


TAULIM = 1.0e-3


@jax.jit
def radia_cloud_masking(cldss, cldmc, tauss, taumc):
    """atm_step.radia_cloud_masking (RAD_DRV.f:2611-2615)."""
    return jnp.where(tauss <= TAULIM, 0.0, cldss), jnp.where(taumc <= TAULIM, 0.0, cldmc)


# ----------------------------------------------------------------------------- CALC_TROP (dyn_glue_ff.calc_trop / tropwmo_columns)
def _trop_params(K):
    psf = K['psf']
    zgwmo = -2e-3 * psf / 984.0
    zgwmo2 = -3e-3 * psf / 984.0
    zfaktor = -K['grav'] / K['rgas']
    zplimb = 500.0 * psf / 984.0
    ptropmax = 600.0 * psf / 984.0
    ptropmin = 30.0 * psf / 984.0
    return zgwmo, zgwmo2, zfaktor, zplimb, ptropmax, ptropmin


def _trop_column(zpm, zpmk, zd, zt, pap, p):
    """One column.  All arrays (LM+1,) with the 1-based level jk as index (index 0 unused); pap[0] = 0.  p = scalars tuple.
    Statement-for-statement port of dyn_glue_ff.tropwmo_columns for one column (python loops -> while_loops)."""
    zgwmo, zgwmo2, zfaktor, zplimb, ptropmax, ptropmin, zzkap, zdeltaz = p

    def c1(s):
        jk, iplimb, ex = s
        return (jk < LM) & ~ex

    def b1(s):
        jk, iplimb, ex = s
        gt = pap[jk - 1] > ptropmax
        brk = (~gt) & (pap[jk] < ptropmin)
        iplimb = jnp.where(gt, jk, iplimb)
        return jnp.where(brk, jk, jk + 1), iplimb, brk
    jk_e, iplimb, ex = lax.while_loop(c1, b1, (jnp.int32(2), jnp.int32(1), jnp.asarray(False)))
    iplimt = jk_e                                    # break value, or LM when the scan ran to the end

    def c2(s):
        jk, ltset, ltropp, done = s
        return (jk < iplimt) & ~done

    def b2(s):
        jk, ltset, ltropp, done = s
        c_fail = (zd[jk] > zgwmo2) & (ltset != 1)
        ltropp = jnp.where(c_fail, jk, ltropp)
        ltset = jnp.where(c_fail, 1, ltset)
        cand = (zd[jk] > zgwmo) & (zpm[jk] <= zplimb)
        ltropp = jnp.where(cand, jk, ltropp)
        ltset = jnp.where(cand, 1, ltset)
        zag = (zd[jk] - zd[jk + 1]) / (zpmk[jk] - zpmk[jk + 1])
        zbg = zd[jk + 1] - zag * zpmk[jk + 1]
        q = (zgwmo - zbg) / zag
        zptf = jnp.where(q < 0.0, 0.0, 1.0)
        zptph = zptf * jnp.power(jnp.abs(q), zzkap)
        zptph = jnp.where(zd[jk + 1] < zgwmo, zptph, zpm[jk])
        zp2km = zptph + zdeltaz * zpm[jk] / zt[jk] * zfaktor

        def ci(t):
            jj, zasum, kc, st = t
            return (jj < iplimt) & (st == 0)

        def bi(t):
            jj, zasum, kc, st = t
            skip = zpm[jj] > zptph
            vexit = (~skip) & (zpm[jj] < zp2km)
            zas = zasum + zd[jj]
            kc2 = kc + 1
            zaq = zas / kc2.astype(jnp.float64)
            disc = zaq <= zgwmo
            acc = (~skip) & (~vexit)
            zasum = jnp.where(acc, zas, zasum)
            kc = jnp.where(acc, kc2, kc)
            st = jnp.where(vexit, 1, jnp.where(acc & disc, 2, st))
            return jj + 1, zasum, kc, st
        _, _, _, st = lax.while_loop(ci, bi, (jk, jnp.zeros(()), jnp.int32(0), jnp.int32(0)))
        st = jnp.where(st == 0, 1, st)               # loop ran to its end without a break (the for-else branch): done
        disc = cand & (st == 2)
        done = cand & ~disc
        return jnp.where(done, jk, jk + 1), ltset, ltropp, done

    _, ltset, ltropp, done = lax.while_loop(c2, b2, (iplimb + 1, jnp.int32(-999), jnp.int32(0), jnp.asarray(False)))
    ltropp = jnp.where(ltset == -999, iplimt - 1, ltropp)
    return pap[ltropp], ltropp


@jax.jit
def calc_trop(t, pk, pmid, K):
    """dyn_glue_ff.calc_trop: t (IM,JM,LM), pk,pmid (LM,IM,JM) -> (ptropo (IM,JM) float64, ltropo (IM,JM) int64 1-based).  Columns i < IMAXJ(j) (all i for
    j=2..JM-1, i=1 at the poles); poles replicated from I=1."""
    zgwmo, zgwmo2, zfaktor, zplimb, ptropmax, ptropmin = _trop_params(K)
    zkappa, zzkap = K['kapa'], K['bykapa']
    N = IM * JM
    ptm1 = (t * jnp.transpose(pk, (1, 2, 0))).reshape(N, LM)
    papm1 = jnp.transpose(pmid, (1, 2, 0)).reshape(N, LM)
    pkc = jnp.transpose(pk, (1, 2, 0)).reshape(N, LM)
    zpmk = 0.5 * (pkc[:, 0:LM - 1] + pkc[:, 1:LM])
    zpm = jnp.power(zpmk, zzkap)
    za = (ptm1[:, 0:LM - 1] - ptm1[:, 1:LM]) / (pkc[:, 0:LM - 1] - pkc[:, 1:LM])
    zb = ptm1[:, 1:LM] - (za * pkc[:, 1:LM])
    ztm = za * zpmk + zb
    zdtdz = zfaktor * zkappa * za * zpmk / ztm
    pad = lambda x: jnp.concatenate([jnp.zeros((N, 2)), x], axis=1)                  # index jk = 0..LM  (entries 2..LM)
    zpm_, zpmk_, zd_, zt_ = pad(zpm), pad(zpmk), pad(zdtdz), pad(ztm)
    pap_ = jnp.concatenate([jnp.zeros((N, 1)), papm1], axis=1)
    p = (zgwmo, zgwmo2, zfaktor, zplimb, ptropmax, ptropmin, zzkap, 2000.0)
    pt, lt = jax.vmap(lambda a, b, c, d, e: _trop_column(a, b, c, d, e, p))(zpm_, zpmk_, zd_, zt_, pap_)
    pt = pt.reshape(IM, JM)
    lt = lt.reshape(IM, JM).astype(jnp.int64)
    for j in (0, JM - 1):
        pt = pt.at[1:, j].set(pt[0, j])
        lt = lt.at[1:, j].set(lt[0, j])
    return pt, lt


# ----------------------------------------------------------------------------- CONDSE entry: A-grid replication of U,V; momentum back-transfer; recalc_agrid_uv
@jax.jit
def replicate_uv_to_agrid(u, v):
    """clouds_condse_ff.replicate_uv_to_agrid: ukm,vkm (4,LM,IM,JM) (rows 1 and JM zero), ukmsp,vkmsp = u,v(:,2,:), ukmnp,vkmnp = u,v(:,JM,:) (IM,LM)."""
    def rep(w):
        wt = jnp.transpose(w, (2, 0, 1))                         # (LM,IM,JM)
        r = jnp.roll(wt, 1, axis=1)                              # W[im1, j]
        k0 = r[:, :, 1:JM - 1]
        k1 = wt[:, :, 1:JM - 1]
        k2 = r[:, :, 2:JM]
        k3 = wt[:, :, 2:JM]
        out = jnp.zeros((4, LM, IM, JM))
        return out.at[0, :, :, 1:JM - 1].set(k0).at[1, :, :, 1:JM - 1].set(k1).at[2, :, :, 1:JM - 1].set(k2).at[3, :, :, 1:JM - 1].set(k3)
    return rep(u), rep(v), u[:, 1, :], v[:, 1, :], u[:, JM - 1, :], v[:, JM - 1, :]


@jax.jit
def avg_replicated_duv_to_vgrid(u, v, ukm, vkm, ukmsp, vkmsp, ukmnp, vkmnp):
    """clouds_condse_ff.avg_replicated_duv_to_vgrid (functional): returns (u, v, ukm, vkm); ukm,vkm carry the pole-row copies the NumPy version writes in place."""
    outs = []
    xs_out = []
    for X, Xs, Xn, W in ((ukm, ukmsp, ukmnp, u), (vkm, vkmsp, vkmnp, v)):
        im1 = jnp.arange(IM) - 1
        X = X.at[2, :, :, 0].set(Xs[im1, :].T * 0.5)
        X = X.at[3, :, :, 0].set(Xs[jnp.arange(IM), :].T * 0.5)
        X = X.at[0, :, :, JM - 1].set(Xn[im1, :].T * 0.5)
        X = X.at[1, :, :, JM - 1].set(Xn[jnp.arange(IM), :].T * 0.5)
        ip1 = (jnp.arange(IM) + 1) % IM
        s = ((X[3, :, :, 0:JM - 1] + X[2][:, ip1][:, :, 0:JM - 1]) + X[1, :, :, 1:JM]) + X[0][:, ip1][:, :, 1:JM]      # (LM, IM, JM-1) rows j=1..JM-1
        outs.append(W.at[:, 1:JM, :].set(W[:, 1:JM, :] + jnp.transpose(s, (1, 2, 0))))
        xs_out.append(X)
    return outs[0], outs[1], xs_out[0], xs_out[1]


def make_agrid_consts(glue_geom):
    g = glue_geom
    return dict(rapj=jnp.asarray(np.asarray(g['rapj'], dtype=np.float64)), cosiv=jnp.asarray(np.asarray(g['cosiv'], dtype=np.float64)),
                siniv=jnp.asarray(np.asarray(g['siniv'], dtype=np.float64)),
                idik=jnp.asarray(np.asarray(g['idij'], dtype=np.int64) - 1), idjk=jnp.asarray(np.asarray(g['idjj'], dtype=np.int64) - 1))


@jax.jit
def recalc_agrid_uv(u, v, C):
    """dyn_glue_ff.recalc_agrid_uv: u,v (IM,JM,LM) -> (ua, va) (LM,IM,JM), cells I<=IMAXJ(J) only (others 0).
    Polar boxes: sequential sum over the 72 neighbours; rows 2..JM-1: sequential sum over the 4 neighbours (u * rak, left to right)."""
    rapj, cosiv, siniv, idik, idjk = C['rapj'], C['cosiv'], C['siniv'], C['idik'], C['idjk']      # rapj (IM,JM) [k,j], idik (K=IM,I=IM,J=JM) 0-based, idjk (IM,JM)
    ua = jnp.zeros((LM, IM, JM))
    va = jnp.zeros((LM, IM, JM))
    for jp, hemi in ((0, -1.0), (JM - 1, 1.0)):
        def body(c, k):
            ut, vt = c
            ik = idik[k, 0, jp]
            jk = idjk[k, jp]
            rak = rapj[k, jp]
            ck, sk = cosiv[k], siniv[k]
            uk = u[ik, jk, :]
            vk = v[ik, jk, :]
            ut = ut + rak * (uk * ck - (hemi * vk) * sk)
            vt = vt + rak * (vk * ck + (hemi * uk) * sk)
            return (ut, vt), None
        (ut, vt), _ = lax.scan(body, (jnp.zeros(LM), jnp.zeros(LM)), jnp.arange(IM))
        ua = ua.at[:, 0, jp].set(ut)
        va = va.at[:, 0, jp].set(vt)
    ii = jnp.arange(IM)
    ut = jnp.zeros((IM, JM - 2, LM))
    vt = jnp.zeros((IM, JM - 2, LM))
    for k in range(4):
        ik = idik[k, :, 1:JM - 1]                                  # (IM, JM-2)
        jk = idjk[k, 1:JM - 1]                                     # (JM-2,)
        rak = rapj[k, 1:JM - 1]
        ut = ut + u[ik, jk[None, :], :] * rak[None, :, None]
        vt = vt + v[ik, jk[None, :], :] * rak[None, :, None]
    ua = ua.at[:, :, 1:JM - 1].set(jnp.transpose(ut, (2, 0, 1)))
    va = va.at[:, :, 1:JM - 1].set(jnp.transpose(vt, (2, 0, 1)))
    return ua, va
