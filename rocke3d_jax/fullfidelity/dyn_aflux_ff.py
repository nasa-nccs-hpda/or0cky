"""D96: AFLUX + ADVECM + MAtoP (+ AVRX) of the atmospheric dynamics ported to numpy.

Fortran references (model/ of modelE2_planet_2.0, rundeck P2SAoM40):
  AFLUX   ATMDYN.f 483-745   mass fluxes MU,MV,MW,CONV and SPA (polar rules, AVRX smoothing,
                             topography adjustment, MW column recursion)
  ADVECM  ATMDYN.f 748-844   MNEW = MOLD + DT1*(CONV+MW diff)*byDXYP, column sums, polar copy
  MAtoP   ATMDYN_COM.F90 147-191  PEDN,PMID,PDSIG,PK,P from MA
  AVRX    ATMDYN.f 1327-1415 zonal Fourier truncation of near-polar rows (uses FFT72.f, ported in
                             dyn_fft72_ff.py)

Array conventions (numpy, 0-based; Fortran J is index J-1): U,V,MU,MV,CONV,SPA (IM,JM,LM);
MW (IM,JM,LM-1); MA,ME,MOLD,MNEW,PEDN,PMID,PDSIG,PK (LM,IM,JM); MASUM,MSUM,P (IM,JM).

Operation order follows the Fortran statement by statement (left-to-right products, strictly
sequential reductions via `seqsum`; the real build uses -O2 -fp-model strict -assume protect_parens
with no FMA).  Branches that need care: DO_POLEFIX=1 (MU at the poles scaled by 2/3), the
topography adjustment (one global patch, mode 1, from init_ATMDYN defaults), AVRX only on rows with
DRAT(J) <= 1.

Fixed geometry, constants and the AVRX tables (DRAT, BYSN, NMIN) are NOT hard-coded: geometry and
constants are read from the recorded one-time dump ffd_aflux_geom.bin (`load_geom`, written by
ffdb_geom in instrumentation/ATM_DRV_dynB.f.patch); DRAT/BYSN/NMIN are recomputed from it exactly as
AVRX's init part does.
"""
import numpy as np

import dyn_fft72_ff as fft72
import intel_libm_ff

IM, JM, LM = 72, 46, 40
IMH = IM // 2
TWOBY3 = 2 / 3.0


def seqsum(a, axis=0):
    """Strictly sequential (left-to-right) sum along `axis` (ifort SUM under -fp-model strict)."""
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    s = a[0].copy()
    for k in range(1, a.shape[0]):
        s = s + a[k]
    return s


# ---------------------------------------------------------------- geometry / constants
_SCALARS = ['im', 'jm', 'lm', 'grav', 'rgas', 'kapa', 'bygrav', 'bykapa', 'bykapap1', 'bykapap2',
            'kg2mb', 'mtop', 'mfixs', 'mincolmass', 'maxcolmass', 'dt', 'dlon', 'byim', 'do_polefix',
            'nidyn', 'aflux_topo', 'npatch', 'polwt', 'acor', 'acor2', 'mdrya', 'sha', 'radius',
            'omega', 'pad1', 'pad2', 'pad3']
_JARR = ['dxyp', 'bydxyp', 'dyp', 'bydyp', 'dxp', 'dxv', 'dyv', 'dxyv', 'dxyn', 'dxys', 'ravpn',
         'ravps', 'rapvn', 'rapvs', 'fcor', 'cosv', 'imaxj']


def load_geom(path):
    """Recorded one-time dump -> dict of scalars, JM-arrays, IM-arrays (siniv,cosiv), LM-arrays
    (mfix,mfrac), zatmo (IM,JM) and the AFLUX topography patch tables."""
    raw = np.fromfile(path, dtype='>f8')
    g = {k: raw[i] for i, k in enumerate(_SCALARS)}
    o = len(_SCALARS)
    for k in _JARR:
        g[k] = raw[o:o + JM].copy(); o += JM
    g['imaxj'] = g['imaxj'].astype(int)
    for k, n in (('siniv', IM), ('cosiv', IM), ('mfix', LM), ('mfrac', LM)):
        g[k] = raw[o:o + n].copy(); o += n
    g['zatmo'] = raw[o:o + IM * JM].reshape((IM, JM), order='F').copy(); o += IM * JM
    npatch = int(g['npatch'])
    g['ipatch'] = raw[o:o + 2 * npatch].reshape((2, npatch), order='F').astype(int); o += 2 * npatch
    g['jpatch'] = raw[o:o + 2 * npatch].reshape((2, npatch), order='F').astype(int); o += 2 * npatch
    g['md'] = raw[o:o + npatch].astype(int); o += npatch
    assert o == raw.size, (o, raw.size)
    for k in ('do_polefix', 'nidyn', 'aflux_topo'):
        g[k] = int(g[k])
    return g


# ---------------------------------------------------------------- AVRX
def avrx_tables(g):
    """AVRX init part (ATMDYN.f 1352-1368) with xAVRX=1 (moment order 2): BYSN, DRAT, NMIN."""
    dlon = g['dlon']
    n = np.arange(1, IMH + 1)
    bysn = 1.0 / np.sin(.5 * dlon * n)                       # BYSN(N)=xAVRX/SIN(.5*DLON*N)
    drat = g['dxp'] * g['bydyp'][2]                          # DRAT(J)=DXP(J)*BYDYP(3)
    nmin = np.zeros(JM, dtype=int)
    for j in range(JM):
        for nn in range(IMH, 0, -1):
            if bysn[nn - 1] * drat[j] > 1.0:
                nmin[j] = nn + 1
                break
    return bysn, drat, nmin


def avrx_rows(x, rows, g, tab=None):
    """AVRX on X(IM,JM) for Fortran-J rows `rows` (iterable of 0-based j): x is (IM, JM) array
    (one layer), modified copy returned.  Rows with DRAT>1 are skipped."""
    bysn, drat, nmin = tab if tab is not None else avrx_tables(g)
    x = x.copy()
    sel = [j for j in rows if not drat[j] > 1]
    for j in sel:
        A, B = fft72.fft_rows(x[:, j][:, None])
        A = A[:, 0].copy(); B = B[:, 0].copy()
        for n in range(nmin[j], IMH):                        # DO N=NMIN(J),IMH-1
            A[n] = bysn[n - 1] * drat[j] * A[n]
            B[n] = bysn[n - 1] * drat[j] * B[n]
        A[IMH] = bysn[IMH - 1] * drat[j] * A[IMH]
        x[:, j] = fft72.ffti_rows(A[:, None], B[:, None])[:, 0]
    return x


def avrx_field(spa, rows, g, tab=None):
    """AVRX applied to every layer of spa (IM,JM,LM); batched over layers (rows are independent)."""
    bysn, drat, nmin = tab if tab is not None else avrx_tables(g)
    out = spa.copy()
    for j in rows:
        if drat[j] > 1:
            continue
        A, B = fft72.fft_rows(spa[:, j, :])                  # (IM,LM) -> (37,LM)
        A = A.copy(); B = B.copy()
        for n in range(nmin[j], IMH):
            A[n] = bysn[n - 1] * drat[j] * A[n]
            B[n] = bysn[n - 1] * drat[j] * B[n]
        A[IMH] = bysn[IMH - 1] * drat[j] * A[IMH]
        B[IMH] = bysn[IMH - 1] * drat[j] * B[IMH]
        out[:, j, :] = fft72.ffti_rows(A, B)
    return out


# ---------------------------------------------------------------- AFLUX
def _topo_ew(mu, ma_t, masum, zatmo, ip, jlo, jhi, mode, stats=None):
    """Uphill E-W flux adjustment for one patch (ATMDYN.f 616-657), vectorised over (I,J): every
    cell only modifies its own column mu(i,j,:), so the Fortran (j,i) loop order is immaterial."""
    i1, i2 = ip
    I = np.arange(i1 - 1, i2)
    J = np.arange(jlo - 1, jhi)
    if J.size == 0:
        return
    Ig, Jg = np.meshgrid(I, J, indexing='ij')
    Ip1 = (Ig + 1) % IM                                       # i.eq.im -> ip1=1
    z0 = zatmo[Ig, Jg]; z1 = zatmo[Ip1, Jg]
    lt = z0 < z1
    gt = z0 > z1
    act = lt | gt
    iup = np.where(lt, Ip1, Ig)
    idn = np.where(lt, Ig, Ip1)
    xx = np.where(lt, 1.0, -1.0)
    mup = masum[iup, Jg]
    mdn = masum[idn, Jg].copy()
    alive = act.copy()
    if stats is not None:
        stats['ew_cells'] = stats.get('ew_cells', 0) + int(act.sum())
    for l in range(LM - 1):                                   # DO L=1,LM-1
        mdn = mdn - ma_t[idn, Jg, l]
        alive = alive & ~(mdn < mup)
        if mode == 0:
            m = alive
            mu[Ig[m], Jg[m], l] = 0.0
        else:
            cur = mu[Ig, Jg, l]
            m = alive & (xx * cur > 0.0)
            mu[Ig[m], Jg[m], l + 1] = mu[Ig[m], Jg[m], l + 1] + cur[m]
            mu[Ig[m], Jg[m], l] = 0.0
            if stats is not None:
                stats['ew_moved'] = stats.get('ew_moved', 0) + int(m.sum())
        if stats is not None:
            stats['ew_levels'] = stats.get('ew_levels', 0) + int(alive.sum())


def _topo_ns(mv, ma_t, masum, zatmo, ip, jlo, jhi, mode, stats=None):
    """Uphill N-S flux adjustment (ATMDYN.f 659-704): cell (i,j) compares zatmo(i,j-1), zatmo(i,j)."""
    i1, i2 = ip
    I = np.arange(i1 - 1, i2)
    J = np.arange(jlo - 1, jhi)
    if J.size == 0:
        return
    Ig, Jg = np.meshgrid(I, J, indexing='ij')
    z0 = zatmo[Ig, Jg - 1]; z1 = zatmo[Ig, Jg]
    lt = z0 < z1
    gt = z0 > z1
    act = lt | gt
    jup = np.where(lt, Jg, Jg - 1)
    jdn = np.where(lt, Jg - 1, Jg)
    xx = np.where(lt, 1.0, -1.0)
    mup = masum[Ig, jup]
    mdn = masum[Ig, jdn].copy()
    alive = act.copy()
    if stats is not None:
        stats['ns_cells'] = stats.get('ns_cells', 0) + int(act.sum())
    for l in range(LM - 1):
        mdn = mdn - ma_t[Ig, jdn, l]
        alive = alive & ~(mdn < mup)
        if mode == 0:
            m = alive
            mv[Ig[m], Jg[m], l] = 0.0
        else:
            cur = mv[Ig, Jg, l]
            m = alive & (xx * cur > 0.0)
            mv[Ig[m], Jg[m], l + 1] = mv[Ig[m], Jg[m], l + 1] + cur[m]
            mv[Ig[m], Jg[m], l] = 0.0
            if stats is not None:
                stats['ns_moved'] = stats.get('ns_moved', 0) + int(m.sum())


def aflux(ns, u, v, ma, masum, me, mesum, g, avrx_post=None, tab=None, stats=None,
          polefix=True, topo=True):
    """AFLUX.  Returns dict(mu, mv, mw, conv, spa, spa0).
    avrx_post: if given (IM,JM,LM array of the real post-AVRX SPA) it replaces the ported AVRX
    (isolates the FFT); stats: optional dict that collects branch counters."""
    dyp, dxv, dxyp = g['dyp'], g['dxv'], g['dxyp']
    polwt = g['polwt']
    byim = g['byim']
    zNSxDT = 1 / (ns * g['dt'])
    u0, v0 = u, v
    u = u.copy(); v = v.copy()
    # interpolated polar velocities (restored afterwards: u0/v0 are the restored values)
    u[:, 1, :] = polwt * u[:, 1, :] + (1 - polwt) * u[:, 2, :]
    v[:, 1, :] = polwt * v[:, 1, :] + (1 - polwt) * v[:, 2, :]
    u[:, JM - 1, :] = polwt * u[:, JM - 1, :] + (1 - polwt) * u[:, JM - 2, :]
    v[:, JM - 1, :] = polwt * v[:, JM - 1, :] + (1 - polwt) * v[:, JM - 2, :]
    ma_t = ma.transpose(1, 2, 0)                              # (IM,JM,LM) view of MA(L,I,J)
    # SPA(I,J,L)=U(I,J,L)+U(I,J+1,L), J=2..JM-1, then AVRX
    spa0 = np.zeros((IM, JM, LM))
    spa0[:, 1:JM - 1, :] = u[:, 1:JM - 1, :] + u[:, 2:JM, :]
    if avrx_post is None:
        spa = avrx_field(spa0, range(1, JM - 1), g, tab)
    else:
        spa = spa0.copy()
        spa[:, 1:JM - 1, :] = avrx_post[:, 1:JM - 1, :]
    mu = np.zeros((IM, JM, LM))
    mv = np.zeros((IM, JM, LM))
    mav = ma_t + np.roll(ma_t, -1, axis=0)                    # MA(L,I,J)+MA(L,Ip1,J)
    mu[:, 1:JM - 1, :] = (.25 * dyp[None, 1:JM - 1, None] * spa[:, 1:JM - 1, :]) * mav[:, 1:JM - 1, :]
    # MV(I,J,L), J=2..JM, Im1 = cyclic previous
    vsum = v[:, 1:JM, :] + np.roll(v, 1, axis=0)[:, 1:JM, :]
    mv[:, 1:JM, :] = ((.25 * dxv[None, 1:JM, None]) * vsum) * (ma_t[:, 1:JM, :] + ma_t[:, 0:JM - 1, :])
    # polar mass fluxes from the unmodified U,V (restored)
    mvsa = np.zeros(LM); mvna = np.zeros(LM)
    for pole in (0, 1):
        if pole == 0:
            jv, jma, jdyp = 1, 0, dyp[1]
        else:
            jv, jma, jdyp = JM - 1, JM - 1, dyp[JM - 2]
        m1 = ma_t[0, jma, :]
        mus = seqsum(u0[:, jv, :], 0) * byim * .25 * jdyp * m1
        mvs = seqsum(mv[:, jv, :], 0) * byim
        dum = np.zeros((IM, LM))
        for i in range(1, IM):
            dum[i] = dum[i - 1] + (mv[i, jv, :] - mvs)
        pb = seqsum(dum, 0) * byim
        if pole == 0:
            x = (pb[None, :] - dum) + mus[None, :]
            mvsa = mvs
        else:
            x = (dum - pb[None, :]) + mus[None, :]
            mvna = mvs
        pj = 0 if pole == 0 else JM - 1
        spa[:, pj, :] = (4 * x) / (jdyp * m1)[None, :]
        mu[:, pj, :] = 3 * x
    if g['do_polefix'] == 1 and polefix:
        mu[:, 0, :] = mu[:, 0, :] * TWOBY3
        mu[:, JM - 1, :] = mu[:, JM - 1, :] * TWOBY3
    if g['aflux_topo'] and topo:
        for adjmode in (0, 1):
            for nn in range(int(g['npatch'])):
                if g['md'][nn] != adjmode:
                    continue
                ip = g['ipatch'][:, nn]
                jp = g['jpatch'][:, nn]
                _topo_ew(mu, ma_t, masum, g['zatmo'], ip, max(jp[0], 2), min(jp[1], JM - 1), adjmode, stats)
                _topo_ns(mv, ma_t, masum, g['zatmo'], ip, max(jp[0], max(3, 2)), min(jp[1], JM - 1),
                         adjmode, stats)
    # CONV, J=2..JM-1
    conv = np.zeros((IM, JM, LM))
    conv[:, 1:JM - 1, :] = (((np.roll(mu, 1, axis=0)[:, 1:JM - 1, :] - mu[:, 1:JM - 1, :])
                             + mv[:, 1:JM - 1, :]) - mv[:, 2:JM, :])
    conv[0, 0, :] = -mvsa
    conv[0, JM - 1, :] = mvna
    # MW downward flux, recursion in L (vector over columns I<=IMAXJ(J))
    mw = np.zeros((IM, JM, LM - 1))
    mfix, mfrac = g['mfix'], g['mfrac']
    convs = seqsum(conv, 2)                                   # Sum(CONV(I,J,:))
    mvars = mesum - g['mfixs']
    me_t = me.transpose(1, 2, 0)
    dxyp_b = dxyp[None, :]
    L = LM - 1                                                # Fortran LM -> 0-based LM-1
    mw[:, :, LM - 2] = ((conv[:, :, LM - 1] - convs * mfrac[LM - 1])
                        + (((me_t[:, :, LM - 1] - (mfix[LM - 1] + mvars * mfrac[LM - 1])) * dxyp_b) * zNSxDT))
    for l in range(LM - 3, -1, -1):                           # DO L=LM-2,1,-1 (Fortran)
        mw[:, :, l] = (((mw[:, :, l + 1] + conv[:, :, l + 1]) - convs * mfrac[l + 1])
                       + (((me_t[:, :, l + 1] - (mfix[l + 1] + mvars * mfrac[l + 1])) * dxyp_b) * zNSxDT))
    # only I<=IMAXJ(J) are computed in Fortran; poles replicate I=1
    mw[1:, 0, :] = mw[0, 0, :][None, :]
    mw[1:, JM - 1, :] = mw[0, JM - 1, :][None, :]
    return dict(mu=mu, mv=mv, mw=mw, conv=conv, spa=spa, spa0=spa0)


# ---------------------------------------------------------------- MAtoP / ADVECM
def matop(ma, masum, g, imf_pow=False):
    """MAtoP: returns dict(pedn,pmid,pdsig,pk,p).  ma (LM,IM,JM), masum (IM,JM).
    PK = PMID**KAPA: numpy pow (default, within 1 ulp of ifort's) or, with imf_pow=True, the Intel libimf
    scalar pow the real build calls (bitwise; needs the Intel runtime, see intel_libm_ff.py)."""
    kg2mb, mtop, kapa = g['kg2mb'], g['mtop'], g['kapa']
    p = kg2mb * (masum - g['mfixs'])
    pedn = np.zeros_like(ma); pmid = np.zeros_like(ma); pdsig = np.zeros_like(ma); pk = np.zeros_like(ma)
    m = np.full(ma.shape[1:], mtop)
    for l in range(LM - 1, -1, -1):
        pedn[l] = kg2mb * (m + ma[l])
        pmid[l] = kg2mb * (m + ma[l] * .5)
        pdsig[l] = kg2mb * ma[l]
        pk[l] = intel_libm_ff.pow_imf(pmid[l], kapa) if imf_pow else pmid[l] ** kapa
        m = m + ma[l]
    return dict(pedn=pedn, pmid=pmid, pdsig=pdsig, pk=pk, p=p)


def advecm(dt1, mold, conv, mw, g, stats=None, imf_pow=False):
    """ADVECM (+ MAtoP).  mold (LM,IM,JM), conv (IM,JM,LM), mw (IM,JM,LM-1).
    Returns dict(mnew, msum, pedn, pmid, pdsig, pk, p, n_exception)."""
    bydxyp = g['bydxyp']
    mtop = g['mtop']
    mnew = np.zeros((LM, IM, JM))
    conv_t = conv.transpose(2, 0, 1)                          # (LM,IM,JM)
    mw_t = mw.transpose(2, 0, 1)                              # (LM-1,IM,JM)
    b = bydxyp[None, None, :]
    mnew[LM - 1] = mold[LM - 1] + (dt1 * (conv_t[LM - 1] - mw_t[LM - 2])) * b
    msum = mnew[LM - 1].copy()
    for l in range(LM - 2, 0, -1):                            # DO L=LM-1,2,-1 (Fortran), 0-based l = L-1
        mnew[l] = mold[l] + (dt1 * ((conv_t[l] + mw_t[l]) - mw_t[l - 1])) * b
        msum = msum + mnew[l]
    mnew[0] = mold[0] + (dt1 * (conv_t[0] + mw_t[0])) * b
    msum = msum + mnew[0]
    # exception diagnostics (the Fortran stops the run if the column mass is out of range)
    ex = 0
    tot = msum + mtop
    imx = g['imaxj']
    valid = np.arange(IM)[:, None] < imx[None, :]
    if np.any(valid & ((tot > g['maxcolmass']) | (tot < g['mincolmass']))):
        ex = 1
        if np.any(valid & ((tot > g['maxcolmass'] * (1200. / 1160.)) | (tot < g['mincolmass'] * (250. / 350.)))):
            ex = 2
    # polar replication of I=1 (cells I>IMAXJ at the poles are not computed in Fortran)
    mnew[:, 1:, 0] = mnew[:, 0, 0][:, None]
    mnew[:, 1:, JM - 1] = mnew[:, 0, JM - 1][:, None]
    msum[1:, 0] = msum[0, 0]
    msum[1:, JM - 1] = msum[0, JM - 1]
    out = matop(mnew, msum, g, imf_pow=imf_pow)
    out.update(mnew=mnew, msum=msum, n_exception=ex)
    return out
