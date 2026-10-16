"""D114-D117: coupling glue of atm_phase1/atm_phase2 ported to numpy (loop-faithful for every sum).

Fortran (model/ of modelE2_planet_2.0; line numbers of the pristine files):
  CALC_TROP + tropwmo      ATM_UTILS.f:329-388 / 390-551        (D114)  WMO tropopause per column
  COMPUTE_WSAVE            ATM_UTILS.f:689-716                  (D114)  vertical velocity export
  calc_kea_3d              ATMDYN.f:2284-2304                   (D115)  .5*(U*U+V*V) + regrid_btoa_3d
  regrid_btoa_3d           ATMDYN.f:2710-2742                   (D115/D116)
  DISSIP + addEnergyAsLocalHeat  ATM_UTILS.f:635-645 / 647-679  (D115)  dissipated KE added back as heat
  CONSERV_SE               DIAG.f:1170-1215 (calc part 1200-1214) (D115) column static energy
  energy-fix block         ATM_DRV.f:176-205 (atm_phase1)       (D115)  uniform dSEpKE removed from T
  PGRAD_PBL (live, non-cubed)  ATM_UTILS.f:69-174               (D116)
  recalc_agrid_uv          ATMDYN.f:2306-2415                   (D116)  B-grid winds to A-grid UALIJ,VALIJ
  DAILY_ATMDYN             ATMDYN_COM.F90:438-473               (D117)  daily dry-mass fixer
Reused by import from dyn_filter_ff (not duplicated): seqsum, conserv_ke (CONSERV_KE + regrid_btoa_ext),
global_sum_ij (GLOBALSUM), matopmb, _pow_arr.

Conventions as in dyn_filter_ff: 2-D (IM,JM), 3-D (IM,JM,LM) for T,Q,U,V,KEA,WSAVE ..., (LM,IM,JM) for MA, PK, PMID,
PEDN, UALIJ, VALIJ.  Index 0 = Fortran 1.  Geometry `g` is a dict (dyn_glue_io.load_consts: recorded values, or
build_geom: analytic from dyn_geom_ff).  Build flags (-fp-model strict, protect_parens) give strict left-to-right
evaluation, which every statement below reproduces.  `**` calls (tropwmo: zpmk**zzkap and abs(..)**zzkap) go to
Intel libimf in the real build: imf_pow=True uses the ctypes bridge, else numpy (may differ by 1 ulp).
"""
import numpy as np

import dyn_geom_ff
import dyn_filter_ff as ff
from dyn_filter_ff import seqsum, global_sum_ij

IM, JM, LM = 72, 46, 40


# ----------------------------------------------------------------------------- geometry
def build_geom(radius, rec=None):
    """Analytic geometry for the glue routines (GEOM_B.f 191, 246-262, 308-313, 329-368).  `rec` (recorded
    consts, optional) only supplies the non-geometric scalars (constants, areag, mtop ...)."""
    G = dyn_geom_ff.geometry(radius)
    jm, im = JM, IM
    dxp = G['dxp'].copy()
    bydxp = np.zeros(jm)
    bydxp[1:jm - 1] = 1.0 / dxp[1:jm - 1]
    dxyp = G['dxyp']
    ravps = np.zeros(jm); ravpn = np.zeros(jm)
    for j in range(2, jm + 1):
        ravps[j - 1] = .5 * G['dxys'][j - 1] / dxyp[j - 1]
        ravpn[j - 2] = .5 * G['dxyn'][j - 2] / dxyp[j - 2]
    dlon = G['dlon']
    sinip = np.array([np.sin(dlon * (i - .5)) for i in range(1, im + 1)])
    cosip = np.array([np.cos(dlon * (i - .5)) for i in range(1, im + 1)])
    kmaxj = np.zeros(jm, dtype=int); imaxj = np.zeros(jm, dtype=int)
    rapj = np.zeros((im, jm)); idjj = np.zeros((im, jm), dtype=int)
    idij = np.zeros((im, im, jm), dtype=int)           # idij[K-1, I-1, J-1]
    for j in (1, jm):
        jvpo = 2 if j == 1 else jm
        kmaxj[j - 1] = im; imaxj[j - 1] = 1
        rapj[:, j - 1] = dyn_geom_ff.BYIM
        idjj[:, j - 1] = jvpo
        for k in range(1, im + 1):
            idij[k - 1, :, j - 1] = k
    for j in range(2, jm):
        kmaxj[j - 1] = 4; imaxj[j - 1] = im
        for k in (1, 2):
            rapj[k - 1, j - 1] = ravps[j - 1]
            idjj[k - 1, j - 1] = j
            rapj[k + 1, j - 1] = ravpn[j - 1]
            idjj[k + 1, j - 1] = j + 1
        im1 = im
        for i in range(1, im + 1):
            idij[0, i - 1, j - 1] = im1; idij[1, i - 1, j - 1] = i
            idij[2, i - 1, j - 1] = im1; idij[3, i - 1, j - 1] = i
            im1 = i
    bydxyp = 1. / dxyp
    g = dict(bydxp=bydxp, bydyp=G['bydyp'], cosip=cosip, sinip=sinip, cosiv=G['cosiv'], siniv=G['siniv'],
             kmaxj=kmaxj, imaxj=imaxj, rapj=rapj, idjj=idjj, idij=idij, dxyp=dxyp, dxyn=G['dxyn'],
             dxys=G['dxys'], dxyv=G['dxyv'], rapvs=G['rapvs'], rapvn=G['rapvn'],
             byaxyp=np.repeat(bydxyp[None, :], im, axis=0), axyp=np.repeat(dxyp[None, :], im, axis=0),
             byim=dyn_geom_ff.BYIM)
    if rec is not None:
        for n in ('im', 'jm', 'lm', 'grav', 'rgas', 'kapa', 'bykapa', 'psf', 'bygrav', 'sha', 'dtsrc', 'mtop',
                  'pmtop', 'areag', 'lhe', 'lhm', 'mfrac', 'zatmo'):
            g[n] = rec[n]
    return g


# ----------------------------------------------------------------------------- tropopause (D114)
def _scalar_pow(x, y, imf):
    if imf:
        f = ff.intel_libm_ff._load()
        if f is None:
            raise RuntimeError("Intel libimf not available")
        return np.float64(f(float(x), float(y)))
    return np.float64(x) ** np.float64(y)


def _tropwmo_params(g):
    psf = g['psf']
    zgwmo = -2e-3 * psf / 984.0
    zgwmo2 = -3e-3 * psf / 984.0
    zfaktor = -g['grav'] / g['rgas']
    zplimb = 500.0 * psf / 984.0
    ptropmax = 600.0 * psf / 984.0
    ptropmin = 30.0 * psf / 984.0
    return zgwmo, zgwmo2, 2000.0, zfaktor, zplimb, ptropmax, ptropmin


STAT_KEYS = ['columns', 'iplimb_gt1', 'limit_exit', 'limit_noexit', 'failsafe_set', 'wmo_candidate',
             'zptf_zero', 'ldtdz_false', 'jj_cycle', 'jj_valid_exit', 'jj_discard', 'jj_loop_end',
             'default_ltropp', 'ltropp_failsafe_only', 'ltropp_overridden']


def tropwmo_columns(ptm1, papm1, pk, g, imf_pow=False, stats=None):
    """tropwmo for N columns: ptm1, papm1, pk are (N,LM) arrays (level 1 first).  Returns (ptropo (N,),
    ltropp (N,) 1-based, ierr (N,)).  The elementwise vertical arrays (zpmk, zpm, za, zb, ztm, zdtdz) are
    computed once for all levels with numpy (every operation is a plain IEEE op except the pow); the
    data-dependent scans are plain Python loops, statement by statement."""
    N = ptm1.shape[0]
    zkappa, zzkap = g['kapa'], g['bykapa']
    zgwmo, zgwmo2, zdeltaz, zfaktor, zplimb, ptropmax, ptropmin = _tropwmo_params(g)
    st = stats if stats is not None else {}
    for k in STAT_KEYS:
        st.setdefault(k, 0)
    with np.errstate(all='ignore'):
        zpmk = np.zeros((N, LM + 1)); zpm = np.zeros_like(zpmk); za = np.zeros_like(zpmk)
        zb = np.zeros_like(zpmk); ztm = np.zeros_like(zpmk); zdtdz = np.zeros_like(zpmk)
        # 1-based level jk -> column jk (columns 0 unused); formulas for jk = 2..LM
        zpmk[:, 2:] = 0.5 * (pk[:, 0:LM - 1] + pk[:, 1:LM])
        zpm[:, 2:] = ff._pow_arr(zpmk[:, 2:], zzkap, imf_pow)
        za[:, 2:] = (ptm1[:, 0:LM - 1] - ptm1[:, 1:LM]) / (pk[:, 0:LM - 1] - pk[:, 1:LM])
        zb[:, 2:] = ptm1[:, 1:LM] - (za[:, 2:] * pk[:, 1:LM])
        ztm[:, 2:] = za[:, 2:] * zpmk[:, 2:] + zb[:, 2:]
        zdtdz[:, 2:] = zfaktor * zkappa * za[:, 2:] * zpmk[:, 2:] / ztm[:, 2:]
    ptropo = np.zeros(N); ltrop = np.zeros(N, dtype=int); ierr = np.zeros(N, dtype=int)
    zpm_l = zpm.tolist(); zpmk_l = zpmk.tolist(); zdt_l = zdtdz.tolist(); ztm_l = ztm.tolist()
    pap_l = papm1.tolist()
    for c in range(N):
        pap = [0.0] + pap_l[c]; zp = zpm_l[c]; zpk = zpmk_l[c]; zd = zdt_l[c]; zt = ztm_l[c]
        st['columns'] += 1
        ltset = -999; ltropp = None
        iplimb = 1; exited = False; jk = 2
        for jk in range(2, LM):
            if pap[jk - 1] > ptropmax:
                iplimb = jk
            else:
                if pap[jk] < ptropmin:
                    exited = True
                    break
        iplimt = jk if exited else LM
        st['limit_exit' if exited else 'limit_noexit'] += 1
        if iplimb > 1:
            st['iplimb_gt1'] += 1
        done = False
        for jk in range(iplimb + 1, iplimt):              # do 1000 jk=iplimb+1,iplimt-1
            if zd[jk] > zgwmo2 and ltset != 1:
                ltropp = jk; ltset = 1
                st['failsafe_set'] += 1
            if zd[jk] > zgwmo and zp[jk] <= zplimb:
                if ltropp is not None and ltset == 1 and ltropp != jk:
                    st['ltropp_overridden'] += 1
                ltropp = jk; ltset = 1
                st['wmo_candidate'] += 1
                with np.errstate(all='ignore'):
                    zag = (np.float64(zd[jk]) - zd[jk + 1]) / (np.float64(zpk[jk]) - zpk[jk + 1])
                    zbg = zd[jk + 1] - zag * zpk[jk + 1]
                    q = (zgwmo - zbg) / zag
                    zptf = 0.0 if q < 0.0 else 1.0
                    if zptf == 0.0:
                        st['zptf_zero'] += 1
                    zptph = zptf * _scalar_pow(abs(q), zzkap, imf_pow)
                ldtdz = zd[jk + 1] < zgwmo
                if not ldtdz:
                    zptph = zp[jk]
                    st['ldtdz_false'] += 1
                zp2km = zptph + zdeltaz * zp[jk] / zt[jk] * zfaktor
                zasum = 0.0; kcount = 0
                discard = False
                for jj in range(jk, iplimt):               # do jj=jk,iplimt-1
                    if zp[jj] > zptph:
                        st['jj_cycle'] += 1
                        continue
                    if zp[jj] < zp2km:
                        st['jj_valid_exit'] += 1
                        done = True
                        break
                    zasum = zasum + zd[jj]
                    kcount += 1
                    zaquer = zasum / float(np.float32(kcount))
                    if zaquer <= zgwmo:
                        st['jj_discard'] += 1
                        discard = True
                        break
                else:
                    st['jj_loop_end'] += 1
                    done = True
                if discard:
                    continue                               # goto 1000 (next level)
                break                                      # goto 2000
        if ltset == -999:
            ltropp = iplimt - 1
            st['default_ltropp'] += 1
            ierr[c] = 1
        elif not done:
            st['ltropp_failsafe_only'] += 1
        ltrop[c] = ltropp
        ptropo[c] = pap[ltropp]
    return ptropo, ltrop, ierr


def calc_trop(t, pk, pmid, g, imf_pow=False, stats=None):
    """CALC_TROP: t (IM,JM,LM), pk,pmid (LM,IM,JM).  TL(L)=T*PK per column, columns I<=IMAXJ(J) only, poles
    replicated from I=1.  Returns (ptropo (IM,JM), ltropo (IM,JM) 1-based, ierr count, tl_at_trop (IM,JM))."""
    imaxj = g['imaxj']
    cols = [(i, j) for j in range(JM) for i in range(imaxj[j])]
    ii = np.array([c[0] for c in cols]); jj = np.array([c[1] for c in cols])
    tl = t[ii, jj, :] * pk[:, ii, jj].T
    p = pmid[:, ii, jj].T; k = pk[:, ii, jj].T
    ptr, ltr, ierr = tropwmo_columns(tl, p, k, g, imf_pow, stats)
    ptropo = np.zeros((IM, JM)); ltropo = np.zeros((IM, JM), dtype=int); ttrop = np.zeros((IM, JM))
    ptropo[ii, jj] = ptr; ltropo[ii, jj] = ltr
    ttrop[ii, jj] = tl[np.arange(len(cols)), ltr - 1]
    for j in (0, JM - 1):
        ptropo[1:, j] = ptropo[0, j]; ltropo[1:, j] = ltropo[0, j]; ttrop[1:, j] = ttrop[0, j]
    return ptropo, ltropo, int(ierr.sum()), ttrop


# ----------------------------------------------------------------------------- COMPUTE_WSAVE (D114)
def compute_wsave(mws, t, pk, pedn, g):
    """ATM_UTILS.f:704-713: wsave(i,j,l)=MWs*byaxyp*rgas*0.5*(T(l)*PK(l)+T(l+1)*PK(l+1))*bygrav/(DTsrc*pedn(l+1)),
    l=1..LM-1.  0.5 is a REAL(4) literal (exact).  mws,t (IM,JM,LM); pk (LM,IM,JM); pedn (LM+1,IM,JM)."""
    out = np.zeros((IM, JM, LM - 1))
    by = g['byaxyp']
    for l in range(LM - 1):
        s = t[:, :, l] * pk[l] + t[:, :, l + 1] * pk[l + 1]
        out[:, :, l] = (((((mws[:, :, l] * by) * g['rgas']) * 0.5) * s) * g['bygrav']) / (g['dtsrc'] * pedn[l + 1])
    return out


# ----------------------------------------------------------------------------- regrid_btoa_3d, calc_kea_3d (D115/116)
def regrid_btoa_3d(x, g):
    """ATMDYN.f:2710-2742 on x (IM,JM,LM): south pole row = sum(old row 2)*byim; rows J=2..JM-1 = .25*(((x(im1,j)+
    x(i,j))+x(im1,j+1))+x(i,j+1)) from OLD values; north pole row = sum(row JM)*byim.  Row 1 of the input is
    not used."""
    byim = g['byim']
    xo = np.array(x, dtype=float)
    new = xo.copy()
    new[:, 0, :] = seqsum(xo[:, 1, :], axis=0) * byim
    for j in range(1, JM - 1):
        r0 = xo[:, j, :]; r1 = xo[:, j + 1, :]
        new[:, j, :] = .25 * (((np.roll(r0, 1, axis=0) + r0) + np.roll(r1, 1, axis=0)) + r1)
    new[:, JM - 1, :] = seqsum(xo[:, JM - 1, :], axis=0) * byim
    return new


def kea_bgrid(u, v):
    """KEA(I,J,L)=.5*(U*U+V*V) for J=2..JM (J_STRT_STGR..J_STOP_STGR); row 1 is undefined (returned 0)."""
    k = np.zeros_like(u)
    k[:, 1:, :] = .5 * (u[:, 1:, :] * u[:, 1:, :] + v[:, 1:, :] * v[:, 1:, :])
    return k


def calc_kea_3d(u, v, g):
    return regrid_btoa_3d(kea_bgrid(u, v), g)


# ----------------------------------------------------------------------------- DISSIP (D115)
def add_energy_as_local_heat(dke, t, pk, g):
    """ATM_UTILS.f:665-678: T(I,J,L) -= deltaKE(I,J,L)/(SHA*PK(L,I,J)) for I<=IMAXJ(J) (poles: I=1 only)."""
    tn = t.copy()
    imaxj = g['imaxj']
    for j in range(JM):
        n = imaxj[j]
        for l in range(LM):
            tn[:n, j, l] = t[:n, j, l] - dke[:n, j, l] / (g['sha'] * pk[l][:n, j])
    return tn


def dissip(u, v, kea_saved, t, pk, g):
    """DISSIP: new KE = calc_kea_3d(U,V); DKE = new - saved KEA (whole array); T updated locally.
    Returns (dke, t_new, kea_new)."""
    ke_new = calc_kea_3d(u, v, g)
    dke = ke_new - kea_saved
    return dke, add_energy_as_local_heat(dke, t, pk, g), ke_new


# ----------------------------------------------------------------------------- energy-fix block (D115)
def conserv_se(ma, masum, pk, t, q, qci, g):
    """CONSERV_SE (DIAG.f:1200-1214), I<=IMAXJ(J), L = LM..1 sequential, poles replicated from I=1:
    SE = SE + SHA*MD*T*PK + LHE*MV - LHM*MI with MI=MA*QCI, MV=MA*Q, MD=MA; finally + ZATMO*(MASUM+MTOP)."""
    sha, lhe, lhm = g['sha'], g['lhe'], g['lhm']
    se = np.zeros((IM, JM))
    for l in range(LM - 1, -1, -1):
        mi = ma[l] * qci[:, :, l]; mv = ma[l] * q[:, :, l]; md = ma[l]
        se = ((se + ((sha * md) * t[:, :, l]) * pk[l]) + lhe * mv) - lhm * mi
    se = se + g['zatmo'] * (masum + g['mtop'])
    se[1:, 0] = se[0, 0]
    se[1:, JM - 1] = se[0, JM - 1]
    return se


def energy_fix(sei, kei, sef, kef, masum, t, pk, g):
    """ATM_DRV.f:176-205 after the two CONSERV calls.  Returns dict(sef2, kef2, dsepke, mmglob, t)."""
    sef2 = ((sef - sei) + (kef - kei)) * g['axyp']
    kef2 = masum * g['axyp']
    dse = global_sum_ij(sef2)
    mmg = global_sum_ij(kef2)
    dse = dse / mmg
    tn = t.copy()
    for l in range(LM):
        tn[:, :, l] = t[:, :, l] - dse / (pk[l] * g['sha'])
    return dict(sef2=sef2, kef2=kef2, dsepke=dse, mmglob=mmg, t=tn)


# ----------------------------------------------------------------------------- PGRAD_PBL (D116)
def pgrad_pbl(t1, pk1, pmid1, pedn1, phi1, zatmo, g):
    """ATM_UTILS.f:69-174.  All inputs (IM,JM).  Returns (dpdx, dpdy, dpdx0, dpdy0), (IM,JM) with only the
    cells the Fortran sets filled (J=2..JM-1 all I; the pole cell (1,1) and (1,JM)); others 0."""
    rgas, byim = g['rgas'], g['byim']
    bydyp, bydxp, cosip, sinip = g['bydyp'], g['bydxp'], g['cosip'], g['sinip']
    dpdx = np.zeros((IM, JM)); dpdy = np.zeros((IM, JM)); dpdx0 = np.zeros((IM, JM)); dpdy0 = np.zeros((IM, JM))
    for j in range(1, JM - 1):
        by_rho1 = ((rgas * t1[:, j]) * pk1[:, j]) / (100. * pmid1[:, j])
        dpdy[:, j] = ((((100 * (pmid1[:, j + 1] - pmid1[:, j - 1])) * by_rho1 + phi1[:, j + 1]) - phi1[:, j - 1])
                      * bydyp[j]) * .5
        dpdy0[:, j] = ((((100 * (pedn1[:, j + 1] - pedn1[:, j - 1])) * by_rho1 + zatmo[:, j + 1])
                        - zatmo[:, j - 1]) * bydyp[j]) * .5
        ip = np.roll(np.arange(IM), -1); im1 = np.roll(np.arange(IM), 1)
        dpdx[:, j] = ((((100 * (pmid1[ip, j] - pmid1[im1, j])) * by_rho1 + phi1[ip, j]) - phi1[im1, j])
                      * bydxp[j]) * .5
        dpdx0[:, j] = ((((100 * (pedn1[ip, j] - pedn1[im1, j])) * by_rho1 + zatmo[ip, j]) - zatmo[im1, j])
                       * bydxp[j]) * .5
    for jp, j1, hemi in ((0, 1, -1.0), (JM - 1, JM - 2, 1.0)):
        a1 = b1 = a0 = b0 = 0.0
        for k in range(IM):
            a1 = a1 + (dpdx[k, j1] * cosip[k] - (hemi * dpdy[k, j1]) * sinip[k])
            b1 = b1 + (dpdy[k, j1] * cosip[k] + (hemi * dpdx[k, j1]) * sinip[k])
            a0 = a0 + (dpdx0[k, j1] * cosip[k] - (hemi * dpdy0[k, j1]) * sinip[k])
            b0 = b0 + (dpdy0[k, j1] * cosip[k] + (hemi * dpdx0[k, j1]) * sinip[k])
        dpdx[0, jp] = a1 * byim; dpdy[0, jp] = b1 * byim
        dpdx0[0, jp] = a0 * byim; dpdy0[0, jp] = b0 * byim
    return dpdx, dpdy, dpdx0, dpdy0


# ----------------------------------------------------------------------------- recalc_agrid_uv (D116)
def recalc_agrid_uv(u, v, g):
    """ATMDYN.f:2306-2415.  u,v (IM,JM,LM).  Returns (ua, va) as (LM,IM,JM); only I<=IMAXJ(J) cells are set by
    the Fortran (poles: I=1), the rest are returned as 0."""
    ua = np.zeros((LM, IM, JM)); va = np.zeros((LM, IM, JM))
    idij, idjj, rapj, kmaxj = g['idij'], g['idjj'], g['rapj'], g['kmaxj']
    cosiv, siniv = g['cosiv'], g['siniv']
    for jp, hemi in ((0, -1.0), (JM - 1, 1.0)):                  # polar boxes, I=1..IMAXJ(J)=1
        for i in range(g['imaxj'][jp]):
            ut = np.zeros(LM); vt = np.zeros(LM)
            for k in range(kmaxj[jp]):
                idik = idij[k, i, jp] - 1; idjk = idjj[k, jp] - 1; rak = rapj[k, jp]
                ck = cosiv[k]; sk = siniv[k]
                uk = u[idik, idjk, :]; vk = v[idik, idjk, :]
                ut = ut + rak * (uk * ck - (hemi * vk) * sk)
                vt = vt + rak * (vk * ck + (hemi * uk) * sk)
            ua[:, i, jp] = ut; va[:, i, jp] = vt
    for j in range(1, JM - 1):
        n = g['imaxj'][j]
        ut = np.zeros((n, LM)); vt = np.zeros((n, LM))
        for k in range(kmaxj[j]):
            idik = idij[k, :n, j] - 1; idjk = idjj[k, j] - 1; rak = rapj[k, j]
            ut = ut + u[idik, idjk, :] * rak
            vt = vt + v[idik, idjk, :] * rak
        ua[:, :n, j] = ut.T; va[:, :n, j] = vt.T
    return ua, va


# ----------------------------------------------------------------------------- DAILY_ATMDYN (D117)
def daily_atmdyn(ma, masum, g, mdrya, itime_is_itimei=False, end_of_day=True, imf_pow=False, mp_g=None):
    """ATMDYN_COM.F90:438-473.  Returns None when the routine returns early (not end of day and not the initial
    time, or initial time with |DELTAM|<1e-9); else dict(smass, mdryanow, deltam, ma, masum, pedn, ...).
    MFRAC(L)=0 layers are skipped; MAtoPMB is dyn_filter_ff.matopmb (PEDN, MASUM, PK ...)."""
    if not (end_of_day or itime_is_itimei):
        return None
    cmass = masum * g['axyp']
    smass = global_sum_ij(cmass)
    mdryanow = smass / g['areag'] + g['mtop']
    deltam = mdrya - mdryanow
    if itime_is_itimei and abs(deltam) < 1e-9:
        return None
    man = ma.copy()
    for l in range(LM):
        if g['mfrac'][l] == 0.:
            continue
        man[l] = ma[l] + deltam * g['mfrac'][l]
    out = dict(smass=smass, mdryanow=mdryanow, deltam=deltam, ma=man)
    if mp_g is not None:
        out['matopmb'] = ff.matopmb(man, mp_g, imf_pow=imf_pow)
    return out
