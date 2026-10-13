"""numba (nopython) port of the per-column MSTCNV scalar port `clouds_mstcnv_ff.mstcnv_column` -- speed track
(branch speed/step-time).  Same statement order and expressions as the validated NumPy port; the differences are
only what nopython needs: no dict/locals() checkpoints or branch counters, scalar helpers instead of the
broadcast-array ones, explicit loops for the XYM/ZM moment index lists, one fixed LM = 40.

Not a validated replacement by itself: `speed_mstcnv_nb_compare.py` compares it to the NumPy port column by
column (rounding-level differences are possible: libm pow/exp through LLVM, no FMA contraction requested).
Needs numba (graphcast-env cannot import it: NumPy 2.2.4; DEV-Kernel has numba 0.60 with NumPy 2.0.2).

Layout: inputs are 0-based per-column arrays as in clouds_mstcnv_io IN_FIELDS; `mstcnv_batch` takes them with a
leading column axis N and writes outputs (see OUT_L order below) with the same leading axis.
"""
import math

import numpy as np
from numba import njit, prange

import clouds_mstcnv_ff as src

LM = 40
NMOM = 9
RGAS, GRAV, TEENY, PI = src.RGAS, src.GRAV, src.TEENY, src.PI
LHE, LHS, TF, BYSHA, BYGRAV = src.LHE, src.LHS, src.TF, src.BYSHA, src.BYGRAV
DELTX, SLHE, TI = src.DELTX, src.SLHE, src.TI
CN0, CN0I, CN0G, RHOG, RHOIP = src.CN0, src.CN0I, src.CN0G, src.RHOG, src.RHOIP
ITMAX, FITMAX, WMAX = src.ITMAX, src.FITMAX, src.WMAX
SPH = src.SECONDS_PER_HOUR
CLDMIN, FDDET, DTMIN1, COETAU, WMU, WMUL = src.CLDMIN, src.FDDET, src.DTMIN1, src.COETAU, src.WMU, src.WMUL
CCMUL, CCMUL2 = src.CCMUL, src.CCMUL2
U00A_DEFAULT = src.U00A_DEFAULT
RHOW, LHM = src.hp.RHOW, src.hp.LHM
BY3, BY6, TWOPI = src.hp.BY3, src.hp.BY6, src.hp.TWOPI
F193, F1172, F27, F2439, F04, F1E20 = src.hp._F193, src.hp._F1172, src.hp._F27, src.hp._F2439, src.hp._F04, src.hp._F1E20
F001, F95, PGRAD = src.f4(0.001), src.f4(0.95), src.f4(0.7)
F20_001, F20_050, F20_33 = src.f4(.001), src.f4(.050), src.f4(.33)
POW222 = float(np.power(np.float64(222.0), np.float64(F20_33)))
QSAT_A, QSAT_B, QSAT_C = src.dq._A, src.dq._B, src.dq._C
TH_A, TH_B, TH_C, TH_D, TH_E, TH_F, TH_G = (src.mf._A, src.mf._B, src.mf._C, src.mf._D, src.mf._E, src.mf._F,
                                            src.mf._G)
QSATRE = float(src.dq.qsat(np.float64(283.16), LHE, np.float64(920.0)))
XYM = np.array([0, 1, 3, 6, 4], dtype=np.int64)       # MX MY MXX MXY MYY
ZM = np.array([2, 5, 8, 7], dtype=np.int64)           # MZ MZZ MYZ MZX
# ZDIR = (MZ, MY, MX, MZZ, MYY, MXX, MYZ, MXY, MZX) -> local (mx,my,mz,mxx,myy,mzz,mxy,myz,mzx)
Z_MX, Z_MY, Z_MZ, Z_MXX, Z_MYY, Z_MZZ, Z_MXY, Z_MYZ, Z_MZX = 2, 1, 0, 5, 4, 3, 8, 6, 7

OUT_SC = ("ierr lerr lmcmin lmcmax prcpmc cldslwij clddepij airxl prheat wmsum wmctwp wmclwp fmc1 mccont "
          "negcover").split()
OUT_L = ("tl sm qm fssl cldmcl taumcl svlatl svlat1 svwmxl csizel condpt vsubl tpsav mcflx dgdsm dgdeep dgshlw "
         "dphase dphadeep dphashlw dtotw dqcond dgdqm dqmtotal dqmshlw dqmdeep dqctotal dqcshlw dqcdeep ddmflx "
         "tdnl qdnl u00l qlmc qimc cnvmmrl condmmr").split()
TUNE_NAMES = ("entrainment_cont1 entrainment_cont2 radiusl_multiplier radiusi_multiplier u00a u00b wmu_multiplier "
              "rwcldox rimax mc_fddrt mc_entr_mass_lim_plume mc_new_ddrft_thetav mc_revp_abv_cldbase").split()


# ------------------------------------------------------------------------------------------------ helpers
@njit(cache=True, error_model='numpy')
def qsat(tm, lh, pr):
    return QSAT_A * math.exp(lh * (QSAT_B - QSAT_C / max(130.0, tm))) / pr


@njit(cache=True, error_model='numpy')
def dqsatdt(tm, lh):
    return lh * QSAT_C / (tm * tm)


@njit(cache=True, error_model='numpy')
def thbar(x, y):
    q = x / y
    al = (TH_A + q * (TH_B + q * (TH_C + q * (TH_D + q)))) / (TH_E + q * (TH_F + TH_G * q))
    return x * al


@njit(cache=True, error_model='numpy')
def adjust(sm, qm, plk, mass, lhx, pl, sign):
    slh = lhx * BYSHA
    qmt = qm
    tp = sm * plk / mass
    dqsum = 0.0
    for _ in range(3):
        qst = qsat(tp, lhx, pl)
        dq = (qmt - mass * qst) / (1.0 + slh * qst * dqsatdt(tp, lhx))
        tp = tp + slh * dq / mass
        qmt = qmt - dq
        dqsum = dqsum + sign * dq
    return dqsum


@njit(cache=True, error_model='numpy')
def get_dq_cond(sm, qm, plk, mass, lhx, pl):
    dq = adjust(sm, qm, plk, mass, lhx, pl, 1.0)
    dq = max(0.0, min(dq, qm))
    fc = dq / qm
    if qm > 0.0:
        return dq, fc
    return 0.0, 0.0


@njit(cache=True, error_model='numpy')
def get_dq_evap(sm, qm, plk, mass, lhx, pl, cond):
    dq = adjust(sm, qm, plk, mass, lhx, pl, -1.0)
    dq = max(0.0, min(dq, cond))
    fe = dq / cond
    if cond > 0.0:
        return dq, fe
    return 0.0, 0.0


@njit(cache=True, error_model='numpy')
def mass_flux(lhx, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg, airm0, airm1, byam0, byam1, byam2, sm2, qm2,
              plk0, plk1, pl0, pl1):
    fplume = 0.25
    dfp = 0.25
    fmp2 = 0.0
    dqsum = 0.0
    for it in range(1, 9):
        dfp = dfp * 0.5
        fmp2 = fplume * airm0
        frat1 = fmp2 * byam1
        frat2 = fmp2 * byam2
        smn1 = smo1 * (1.0 - fplume) + frat1 * smo2
        qmn1 = qmo1 * (1.0 - fplume) + frat1 * qmo2
        smn2 = smo2 * (1.0 - frat1) + frat2 * sm2
        qmn2 = qmo2 * (1.0 - frat1) + frat2 * qm2
        smp = smo1 * fplume
        qmp = qmo1 * fplume
        tp = smo1 * plk1 * byam0
        qsatmp = fmp2 * qsat(tp, lhx, pl1)
        gama = slh * qsatmp * dqsatdt(tp, lhx) / fmp2
        dqs = (qmp - qsatmp) / (1.0 + gama)
        fevap = 0.5 * fplume
        if dqs > 0.0:
            mcloud = fevap * airm1
            tnx1 = smo2 * plk1 * byam1
            qnx1 = qmo2 * byam1
            qsatc = qsat(tnx1, lhx, pl1)
            dqe = mcloud * (qsatc - qnx1) / (1.0 + slh * qsatc * dqsatdt(tnx1, lhx))
            if dqe > dqs:
                dqe = dqs
            smn2 = smn2 - slh * dqe / plk1
            qmn2 = qmn2 + dqe
            dqs = dqs - dqe
            if dqs > 0.0:
                mcloud2 = fevap * airm0
                tnx0 = smo1 * plk0 * byam0
                qnx0 = qmo1 * byam0
                qsatc2 = qsat(tnx0, lhx, pl0)
                dq2 = mcloud2 * (qsatc2 - qnx0) / (1.0 + slh * qsatc2 * dqsatdt(tnx0, lhx))
                if dq2 > dqs:
                    dq2 = dqs
                smn1 = smn1 - slh * dq2 / plk0
                qmn1 = qmn1 + dq2
        sdn = smn1 * byam0
        sup = smn2 * byam1
        sedge = thbar(sup, sdn)
        qdn = qmn1 * byam0
        qup = qmn2 * byam1
        svdn = sdn * (1.0 + DELTX * qdn - wmdn)
        svup = sup * (1.0 + DELTX * qup - wmup)
        qedge = 0.5 * (qup + qdn)
        svedg = sedge * (1.0 + DELTX * qedge - wmedg)
        dmse = ((svup - svedg) * plk1 + (svedg - svdn) * plk0 + SLHE * (qsat(sup * plk1, lhx, pl1) - qdn))
        dqsum = dqs
        if abs(dmse) <= 1.0e-3:
            break
        if dmse > 1.0e-3:
            fplume = fplume - dfp
        if dmse < -1.0e-3:
            fplume = fplume + dfp
    return fplume, fmp2, dqsum


@njit(cache=True, error_model='numpy')
def dcw_search(wv, pl, ddcw, wmax, nmax):
    dcw = 0.0
    pfac = (1000.0 / pl) ** 0.4
    for k in range(nmax):
        vt = (-0.267 + dcw * (5.15e3 - dcw * (1.0225e6 - 7.55e7 * dcw))) * pfac
        if vt >= 0.0 and vt >= wv:
            break
        if vt > wmax:
            break
        dcw = dcw + ddcw
    return dcw


@njit(cache=True, error_model='numpy')
def dcg(wv, pl):
    return min(((wv / F193) * (pl / 1000.0) ** F04) ** F27, 1e-2)


@njit(cache=True, error_model='numpy')
def dci(wv, pl):
    return min(((wv / F1172) * (pl / 1000.0) ** F04) ** F2439, 1e-2)


@njit(cache=True, error_model='numpy')
def precip_mp(rho, flam, dc, cn):
    f4p = math.pow(flam, 4.0)
    return (rho * (PI * BY6) * cn * math.exp(-flam * dc)
            * (dc * dc * dc / flam + 3.0 * dc * dc / (flam * flam) + 6.0 * dc / (flam * flam * flam) + 6.0 / f4p))


@njit(cache=True, error_model='numpy')
def conv_micro(pl, wcu, dwcu, lfrz, wcufrz, tp, ti, pland, flamw, flamg, flami, tlmin, tlmin1, condip_in, condgp_in):
    wv = max(wcu - dwcu, 0.0)
    wvu = wcu + dwcu
    if lfrz == 0:
        tig = TF
    else:
        tig = TF - 4.0 * wcufrz
    if tig < ti - 10.0:
        tig = ti - 10.0
    nmax = ITMAX - 1
    if tp >= TF:
        if pland < 0.5:
            ddcw = 1.5e-3 * FITMAX
        else:
            ddcw = 6e-3 * FITMAX
        dcw1 = dcw_search(wv, pl, ddcw, WMAX, nmax)
        condp1 = precip_mp(RHOW, flamw, dcw1, CN0)
        dcw2 = dcw_search(wvu, pl, ddcw, WMAX, nmax)
        condp = precip_mp(RHOW, flamw, dcw2, CN0)
        return condp, condp1, condip_in, condgp_in
    if tp <= tig:
        dci1 = dci(wv, pl)
        condp1 = precip_mp(RHOIP, flami, dci1, CN0I)
        dci_u = dci(wvu, pl)
        condp = precip_mp(RHOIP, flami, dci_u, CN0I)
        return condp, condp1, condip_in, condgp_in
    dcg1 = dcg(wv, pl)
    dci1 = dci(wv, pl)
    fg = (tp - tig) / ((TF - tig) + TEENY)
    if fg > 1.0:
        fg = 1.0
    if fg < 0.0:
        fg = 0.0
    if tlmin <= TF or tlmin1 <= TF:
        fg = 0.0
    fi = 1.0 - fg
    cip_l = precip_mp(RHOIP, flami, dci1, CN0I)
    cgp_l = precip_mp(RHOG, flamg, dcg1, CN0G)
    condp1 = fg * cgp_l + fi * cip_l
    dci_u = dci(wvu, pl)
    dcg_u = dcg(wvu, pl)
    cip_u = precip_mp(RHOIP, flami, dci_u, CN0I)
    cgp_u = precip_mp(RHOG, flamg, dcg_u, CN0G)
    condp = fg * cgp_u + fi * cip_u
    return condp, condp1, cip_u, cgp_u


@njit(cache=True, error_model='numpy')
def cloud_fraction(L, tl, pl, ccm, wcu, plemin, plel2, plemax, ccmmin, wcumin, lmin, lmax, dtsrc):
    rho = pl / (RGAS * tl)
    fcloud = CCMUL * ccm / (rho * GRAV * wcu * dtsrc + TEENY)
    if plemin - plel2 >= 450.0:
        fcloud = 5.0 * fcloud
    below = L < lmin
    if below:
        fcloud = CCMUL * ccmmin / (rho * GRAV * wcumin * dtsrc + TEENY)
    shallow = (plemin - plemax) < 450.0
    if shallow and L == lmax - 1:
        fcloud = CCMUL2 * ccm / (rho * GRAV * wcu * dtsrc + TEENY)
    if shallow and below:
        fcloud = 0.0
    if fcloud > 1.0:
        fcloud = 1.0
    return fcloud


@njit(cache=True, error_model='numpy')
def precip_phase(L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1_in, mc_revp):
    heat1 = heat1_in
    if mc_revp == 0:
        mcloud = 2.0 * fevap * airm if L <= lmin else 0.0
    else:
        mcloud = 2.0 * fevap * airm
    if mcloud > airm:
        mcloud = airm
    lhp = lhp1
    if lhp == LHS and told > TF and told1 <= TF:
        heat1 = heat1 + LHM * prcp * BYSHA
        lhp = LHE
    if lhp == LHE and told <= TF and told1 > TF:
        heat1 = heat1 - LHM * prcp * BYSHA
        lhp = LHS
    if lhp != vlat and cond > 0:
        heat1 = heat1 + (vlat - lhp) * cond * BYSHA
    return lhp, mcloud, heat1


@njit(cache=True, error_model='numpy')
def anvil(svlatl, rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem):
    r_liq = rcldlx * 100.0 * (wtem / (2.0 * BY3 * TWOPI * mcdncw)) ** BY3
    r_ice = rcldix * 100.0 * (wtem / (2.0 * BY3 * TWOPI * mcdnci)) ** BY3
    r_ice = min(r_ice, rimax)
    if svlatl == LHE:
        rcld = r_liq
    else:
        rcld = r_ice
    rclde = rcld / bybr
    taumc = 1.5 * tem / (fcld * rclde + F1E20)
    if taumc > 100.0:
        taumc = 100.0
    return rcld, taumc


@njit(cache=True, error_model='numpy')
def limitq(anm1, an, fnm1, fn, sn, sx, sxx):
    ierr = 0
    if anm1 >= 0. and an <= 0.:
        return fnm1, fn, sx, sxx, ierr
    if anm1 < 0. and an > 0.:
        sl = -fnm1
        sr = +fn
        sc = sn - sl
        sc = sc - sr
        if sl >= 0. and sr >= 0. and sc >= 0.:
            return fnm1, fn, sx, sxx, ierr
        frl = anm1
        frl1 = frl + 1.
        frr = an
        frr1 = frr - 1.
        if sl >= 0. and sr >= 0.:
            gamma = 1. + (frl - frr)
            t = (frl + frr)
            g13ab = gamma * gamma - 1. + 3. * (t * t)
            den = gamma * (12. * (t * t) + 5. * g13ab * g13ab)
            sxx = sxx - sc * 10. * g13ab / den
            sx = sx + sc * 12. * (frl + frr) / den
            sl = -frl * (sn - frl1 * (sx - (frl + frl1) * sxx))
            sr = sn - sl
        elif sr >= 0.:
            u = (frl + frl1)
            sxx = sxx + sl * (frl + frl1) / (frl * frl1 * (.6 + u * u))
            sx = sn / frl1 + (frl + frl1) * sxx
            sr = frr * (sn - frr1 * (sx - (frr + frr1) * sxx))
            sl = 0.
        elif sl >= 0.:
            u = (frr + frr1)
            sxx = sxx - sr * (frr + frr1) / (frr * frr1 * (.6 + u * u))
            sx = sn / frr1 + (frr + frr1) * sxx
            sl = -frl * (sn - frl1 * (sx - (frl + frl1) * sxx))
            sr = 0.
        sc = sn - sl
        sc = sc - sr
        if sl <= 0. and sr <= 0.:
            gamma = 1. + (frl - frr)
            sxx = sn * (1. + gamma) / (2. * gamma * frl1 * frr1)
            sx = sn * (frl + frr) * (1. + 2. * gamma) / (2. * gamma * frl1 * frr1)
            sl = 0.
            sr = 0.
        elif sl <= 0. and sc <= 0.:
            sxx = sn / (2. * frr * frl1)
            sx = sn * (frl + frr + .5) / (frr * frl1)
            sl = 0.
            sr = sn
        elif sr <= 0. and sc <= 0.:
            sxx = sn / (2. * frl * frr1)
            sx = sn * (frl + frr - .5) / (frl * frr1)
            sl = sn
            sr = 0.
        fnm1 = -sl
        fn = +sr
        return fnm1, fn, sx, sxx, ierr
    if an > 0.:
        fr = an
        sd = fn
        fsign = -1.
    else:
        fr = anm1
        sd = -fnm1
        fsign = 1.
    if abs(fr) > 1.:
        ierr = 1
        if sn + fnm1 - fn < 0:
            ierr = 2
            return fnm1, fn, sx, sxx, ierr
    su = sn - sd
    if sd >= 0. and su >= 0.:
        return fnm1, fn, sx, sxx, ierr
    fr1 = fr + fsign
    w = (fr + fr1)
    if sd < 0.:
        sxx = sxx + fsign * sd * (fr + fr1) / (fr * fr1 * (.6 + w * w))
        sx = sn / fr1 + (fr + fr1) * sxx
        su = sn
    else:
        sxx = sxx - fsign * su * (fr + fr1) / (fr * fr1 * (.6 + w * w))
        sx = sn / fr + (fr + fr1) * sxx
        su = 0.
    sd = sn - su
    if an > 0.:
        fn = sd
    else:
        fnm1 = -sd
    return fnm1, fn, sx, sxx, ierr


@njit(cache=True, error_model='numpy')
def adv1d_z(s, smom, mass, dm, nx, qlimit):
    """One cyclic line, ZDIR, in place on s (nx), smom (9, nx), mass (nx); dm (nx) read only.
    Returns (ierr, nerr).  Mirrors dyn_adv1d_ff.adv1d on a single line."""
    f = np.zeros(nx)
    fracm = np.zeros(nx)
    frac1 = np.zeros(nx)
    nn = np.zeros(nx, np.int64)
    for k in range(nx):
        nn[k] = (k + 1) % nx if dm[k] < 0. else k
        fr1 = 1. if dm[k] < 0. else -1.
        ms = mass[nn[k]]
        fm = dm[k] / ms
        if ms <= 0.:
            fm = 0.
        fracm[k] = fm
        frac1[k] = fm + fr1
        sn = s[nn[k]]
        sx = smom[Z_MX, nn[k]]
        sxx = smom[Z_MXX, nn[k]]
        f[k] = fm * (sn - frac1[k] * (sx - (frac1[k] + fm) * sxx))
    ierr_out = 0
    nerr_out = 0
    if qlimit:
        nm1 = nx - 1
        for n in range(nx):
            an = fracm[n]
            anm1 = fracm[nm1]
            fn = f[n]
            fnm1 = f[nm1]
            sn = s[n]
            sxn = smom[Z_MX, n]
            sxxn = smom[Z_MXX, n]
            fnm1, fn, sxn, sxxn, ierr = limitq(anm1, an, fnm1, fn, sn, sxn, sxxn)
            ierr_out = ierr
            if ierr > 0:
                nerr_out = n + 1
                if ierr == 2:
                    return ierr_out, nerr_out
            f[n] = fn
            f[nm1] = fnm1
            smom[Z_MX, n] = sxn
            smom[Z_MXX, n] = sxxn
            nm1 = n
    # slope stage
    fm = np.zeros((NMOM, nx))
    for k in range(nx):
        n = nn[k]
        sx = smom[Z_MX, n]
        sxx = smom[Z_MXX, n]
        d = dm[k]
        fmk = fracm[k]
        f1 = frac1[k]
        fm[Z_MX, k] = d * (fmk * fmk * (sx - 3. * f1 * sxx) - 3. * f[k])
        fm[Z_MXX, k] = d * (d * math.pow(fmk, 3.0) * sxx - 5. * (d * f[k] + fm[Z_MX, k]))
        sxy = smom[Z_MXY, n]
        fm[Z_MY, k] = fmk * (smom[Z_MY, n] - f1 * sxy)
        fm[Z_MXY, k] = d * (fmk * fmk * sxy - 3. * fm[Z_MY, k])
        szx = smom[Z_MZX, n]
        fm[Z_MZ, k] = fmk * (smom[Z_MZ, n] - f1 * szx)
        fm[Z_MZX, k] = d * (fmk * fmk * szx - 3. * fm[Z_MZ, k])
        fm[Z_MYY, k] = fmk * smom[Z_MYY, n]
        fm[Z_MZZ, k] = fmk * smom[Z_MZZ, n]
        fm[Z_MYZ, k] = fmk * smom[Z_MYZ, n]
    # update stage (cyclic prev)
    newmass = np.zeros(nx)
    for k in range(nx):
        p = k - 1 if k > 0 else nx - 1
        dmm = dm[p]
        fmm = f[p]
        m0 = mass[k]
        tmp = m0 + dmm
        mnew = tmp - dm[k]
        bymnew = 1. / mnew
        dm2 = dmm + dm[k]
        tmp = s[k] + fmm
        s[k] = tmp - f[k]
        fmx_m = fm[Z_MX, p]
        fmxx_m = fm[Z_MXX, p]
        smom[Z_MX, k] = (smom[Z_MX, k] * m0 - 3. * (-dm2 * s[k] + m0 * (fmm + f[k])) + (fmx_m - fm[Z_MX, k])) * bymnew
        smom[Z_MXX, k] = (smom[Z_MXX, k] * m0 * m0
                          + 2.5 * s[k] * (m0 * m0 - mnew * mnew - 3. * dm2 * dm2)
                          + 5. * (m0 * (m0 * (fmm - f[k]) - fmx_m - fm[Z_MX, k]) + dm2 * smom[Z_MX, k] * mnew)
                          + (fmxx_m - fm[Z_MXX, k])) * (bymnew * bymnew)
        fmy_m = fm[Z_MY, p]
        smom[Z_MY, k] = smom[Z_MY, k] + fmy_m - fm[Z_MY, k]
        smom[Z_MXY, k] = (smom[Z_MXY, k] * m0 - 3. * (-dm2 * smom[Z_MY, k] + m0 * (fmy_m + fm[Z_MY, k]))
                          + (fm[Z_MXY, p] - fm[Z_MXY, k])) * bymnew
        fmz_m = fm[Z_MZ, p]
        smom[Z_MZ, k] = smom[Z_MZ, k] + fmz_m - fm[Z_MZ, k]
        smom[Z_MZX, k] = (smom[Z_MZX, k] * m0 - 3. * (-dm2 * smom[Z_MZ, k] + m0 * (fmz_m + fm[Z_MZ, k]))
                          + (fm[Z_MZX, p] - fm[Z_MZX, k])) * bymnew
        smom[Z_MYY, k] = smom[Z_MYY, k] + fm[Z_MYY, p] - fm[Z_MYY, k]
        smom[Z_MZZ, k] = smom[Z_MZZ, k] + fm[Z_MZZ, p] - fm[Z_MZZ, k]
        smom[Z_MYZ, k] = smom[Z_MYZ, k] + fm[Z_MYZ, p] - fm[Z_MYZ, k]
        newmass[k] = mnew
    for k in range(nx):
        mass[k] = newmass[k]
        if mass[k] <= 0.:
            s[k] = 0.
            for q in range(NMOM):
                smom[q, k] = 0.
    return ierr_out, nerr_out


# ------------------------------------------------------------------------------------------------ the column
@njit(cache=True, error_model='numpy')
def _pad1(x, n):
    a = np.zeros(LM + 2)
    for i in range(n):
        a[1 + i] = x[i]
    return a


@njit(cache=True, error_model='numpy')
def _padm(x, nr):
    a = np.zeros((nr, LM + 2))
    for q in range(nr):
        for i in range(LM):
            a[q, 1 + i] = x[q, i]
    return a


@njit(cache=True, error_model='numpy')
def mstcnv_col(pearth, pland, dcl, lmcm, xmass, bydtsrc, dtsrc, bybr, pl_i, ple_i, plk_i, airm_i, byam_i, etal_i,
               tl_i, tvl_i, sm_i, qm_i, qcll_i, qcil_i, sdl_i, wturb_i, gzl_i, smom_i, qmom_i, ra, um_i, vm_i,
               u0_i, v0_i, tune, nks, o_sc, o_l, o_l1, o_m, o_k):
    pl = _pad1(pl_i, LM)
    ple = _pad1(ple_i, LM + 1)
    plk = _pad1(plk_i, LM)
    airm = _pad1(airm_i, LM)
    byam = _pad1(byam_i, LM)
    etal = _pad1(etal_i, LM)
    tl = _pad1(tl_i, LM)
    tvl = _pad1(tvl_i, LM)
    sm = _pad1(sm_i, LM)
    qm = _pad1(qm_i, LM)
    qcll = _pad1(qcll_i, LM)
    qcil = _pad1(qcil_i, LM)
    sdl = _pad1(sdl_i, LM)
    wturb = _pad1(wturb_i, LM)
    gzl = _pad1(gzl_i, LM)
    smom = _padm(smom_i, NMOM)
    qmom = _padm(qmom_i, NMOM)
    um = _padm(um_i, nks)
    vm = _padm(vm_i, nks)
    u_0 = _padm(u0_i, nks)
    v_0 = _padm(v0_i, nks)

    contce1, contce2, rcldlx, rcldix, u00a, u00b, wmu_mult, rwcldox, rimax, fddrt = (
        tune[0], tune[1], tune[2], tune[3], tune[4], tune[5], tune[6], tune[7], tune[8], tune[9])
    mc_entr_lim, mc_newthv, mc_revp = int(tune[10]), int(tune[11]), int(tune[12])

    ierr = 0
    lerr = 0
    lmcmin = 0
    lmcmax = 0
    mccont = 0
    fmc1 = 0.0
    negcover = 0
    fssl = np.ones(LM + 2)
    tadj = 1.0
    qsatre = QSATRE
    taumcl = np.zeros(LM + 2)
    condpt = np.zeros(LM + 2)
    svwmxl = np.zeros(LM + 2)
    svlatl = np.zeros(LM + 2)
    svlat1 = np.zeros(LM + 2)
    vsubl = np.zeros(LM + 2)
    precnvl = np.zeros(LM + 2)
    cldmcl = np.zeros(LM + 2)
    tpsav = np.zeros(LM + 2)
    lhp = np.zeros(LM + 2)
    cldslwij = 0.0
    clddepij = 0.0
    prcpmc = 0.0
    csizel = np.zeros(LM + 2)
    csizel[1:LM + 1] = rwcldox * 10. * (1. - pearth) + 10. * pearth
    vlat = np.full(LM + 2, LHE)
    condmmr = np.zeros(LM + 2)
    mcflx = np.zeros(LM + 2)
    dgdsm = np.zeros(LM + 2)
    dgdeep = np.zeros(LM + 2)
    dgshlw = np.zeros(LM + 2)
    dphase = np.zeros(LM + 2)
    dphadeep = np.zeros(LM + 2)
    dphashlw = np.zeros(LM + 2)
    dtotw = np.zeros(LM + 2)
    dqcond = np.zeros(LM + 2)
    dgdqm = np.zeros(LM + 2)
    dqmtotal = np.zeros(LM + 2)
    dqmshlw = np.zeros(LM + 2)
    dqmdeep = np.zeros(LM + 2)
    dqctotal = np.zeros(LM + 2)
    dqcshlw = np.zeros(LM + 2)
    dqcdeep = np.zeros(LM + 2)
    ddmflx = np.zeros(LM + 2)
    tdnl = np.zeros(LM + 2)
    qdnl = np.zeros(LM + 2)
    sm1 = sm.copy()
    qm1 = qm.copy()
    smold = sm.copy()
    smomold = smom.copy()
    qmold = qm.copy()
    qmomold = qmom.copy()
    u00l = np.zeros(LM + 2)
    x_u00 = u00b * F20_001 * F20_050 * 3. * POW222
    u00_pbl = 1.0 - 2. * x_u00 / qsatre
    u00_mc = 1.0 - 2. * (u00b * 2.0e-4 / qsatre)
    for L in range(1, LM + 1):
        u00l[L] = 0.
        if pl[L] >= pl[dcl]:
            u00l[L] = u00_pbl
    dwcu = 0.0
    for L in range(1, lmcm + 1):
        dwcu = dwcu + airm[L] * tl[L] * RGAS / (GRAV * pl[L])
    dwcu = 0.5 * dwcu * bydtsrc / float(lmcm)

    # per-partition state (re-zeroed each (ic, nppl))
    dm = np.zeros(LM + 2)
    dmr = np.zeros(LM + 2)
    ddr = np.zeros(LM + 2)
    ccm = np.zeros(LM + 2)
    cond = np.zeros(LM + 2)
    condp = np.zeros(LM + 2)
    condp1 = np.zeros(LM + 2)
    condgp = np.zeros(LM + 2)
    condip = np.zeros(LM + 2)
    condv = np.zeros(LM + 2)
    taumc1 = np.zeros(LM + 2)
    cdheat = np.zeros(LM + 2)
    ent = np.zeros(LM + 2)
    det = np.zeros(LM + 2)
    buoy = np.zeros(LM + 2)
    wcu = np.zeros(LM + 2)
    wcu2 = np.zeros(LM + 2)
    heat1 = np.zeros(LM + 2)
    ddm = np.zeros(LM + 2)
    smdnl = np.zeros(LM + 2)
    qmdnl = np.zeros(LM + 2)
    dsm = np.zeros(LM + 2)
    dsmr = np.zeros(LM + 2)
    dqm = np.zeros(LM + 2)
    dqmr = np.zeros(LM + 2)
    smomdnl = np.zeros((NMOM, LM + 2))
    qmomdnl = np.zeros((NMOM, LM + 2))
    dsmom = np.zeros((NMOM, LM + 2))
    dsmomr = np.zeros((NMOM, LM + 2))
    dqmom = np.zeros((NMOM, LM + 2))
    dqmomr = np.zeros((NMOM, LM + 2))
    umdnl = np.zeros((nks, LM + 2))
    vmdnl = np.zeros((nks, LM + 2))
    dum = np.zeros((nks, LM + 2))
    dvm = np.zeros((nks, LM + 2))
    umdn = np.zeros(nks)
    vmdn = np.zeros(nks)
    ump = np.zeros(nks)
    vmp = np.zeros(nks)
    umtemp = np.zeros(nks)
    vmtemp = np.zeros(nks)
    smomp = np.zeros(NMOM)
    qmomp = np.zeros(NMOM)
    smompmax = np.zeros(NMOM)
    qmompmax = np.zeros(NMOM)
    smomdn = np.zeros(NMOM)
    qmomdn = np.zeros(NMOM)
    mpmax = 0.0
    smpmax = 0.0
    qmpmax = 0.0
    lmax = 0
    lmin = 0
    ldraft = 0
    ldmin = 0
    llmin = 0
    lfrz = 0
    etadn = 0.0
    cdhsum = 0.0
    cdhsum1 = 0.0
    cdhdrt = 0.0
    cdhm = 0.0
    evpsum = 0.0
    prcp = 0.0
    prheat = 0.0
    airxl = 0.0
    mc1 = False
    mplume = 0.0
    fplume = 0.0
    smp = 0.0
    qmp = 0.0
    ddraft = 0.0
    fctype = 0.0
    mplum1 = 0.0
    told = 0.0
    cm = np.zeros(LM + 2)
    cmneg = np.zeros(LM + 2)
    fmp2 = 0.0
    tp = 0.0
    flamw = 0.0
    flamg = 0.0
    flami = 0.0
    maxlvl = 0
    minlvl = LM

    for lmin in range(1, lmcm):
        maxlvl = 0
        minlvl = LM
        fmp0 = -(10. * 1.0 * sdl[lmin + 1] * BYGRAV * xmass)
        if fmp0 <= 0.:
            fmp0 = 0.0
        smo1, qmo1, smo2, qmo2 = sm[lmin], qm[lmin], sm[lmin + 1], qm[lmin + 1]
        sdn = smo1 * byam[lmin]
        sup = smo2 * byam[lmin + 1]
        sedge = thbar(sup, sdn)
        qdn = qmo1 * byam[lmin]
        qup = qmo2 * byam[lmin + 1]
        wmdn = qcll[lmin] + qcil[lmin]
        wmup = qcll[lmin + 1] + qcil[lmin + 1]
        svdn = sdn * (1. + DELTX * qdn - wmdn)
        svup = sup * (1. + DELTX * qup - wmup)
        qedge = .5 * (qup + qdn)
        wmedg = .5 * (wmup + wmdn)
        svedg = sedge * (1. + DELTX * qedge - wmedg)
        lhx = LHE
        slh = lhx * BYSHA
        dmse = (svup - svedg) * plk[lmin + 1] + (svedg - svdn) * plk[lmin] + \
            slh * (qsat(sup * plk[lmin + 1], lhx, pl[lmin + 1]) - qdn)
        if dmse > -1e-10:
            continue
        fplume, fmp2, dqsum = mass_flux(lhx, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg,
                                        airm[lmin], airm[lmin + 1], byam[lmin], byam[lmin + 1], byam[lmin + 2],
                                        sm[lmin + 2], qm[lmin + 2], plk[lmin], plk[lmin + 1], pl[lmin],
                                        pl[lmin + 1])
        if fplume <= F001:
            continue
        fmp2 = fmp2 * min(1.0, dtsrc / (tadj * SPH))
        for ic in range(1, 3):
            mc1 = False
            lhx = LHE
            mplume = min(airm[lmin], airm[lmin + 1], fmp2)
            fctype = 1.0
            if mplume > fmp0:
                fctype = fmp0 / mplume
            if ic == 2:
                fctype = 1. - fctype
            if fctype < F001:
                continue
            mplum1 = mplume
            cycle_types = False
            for nppl in range(1, 3):
                if nppl == 2 and (not mc1 or mccont < 2):
                    continue
                mplume = mplum1 * fctype
                if mccont == 0 or mc1:
                    if mccont == 0:
                        fsub_tmp = 1.0 + (airm[lmin + 1] - 100.0) / 200.0
                    else:
                        fsub_tmp = 1.0 + (pl[lmin] - pl[lmax] - 100.0) / 200.0
                    fconv_tmp = min(mplum1 * byam[lmin + 1] * (1800.0 / dtsrc), 1.0)
                    if fsub_tmp > 1.0 / (fconv_tmp + 1.0e-20) - 1.0:
                        fsub_tmp = 1.0 / (fconv_tmp + 1.0e-20) - 1.0
                    fsub_tmp = max(1.0, min(fsub_tmp, 5.0))
                    fssl_tmp = 1.0 - (1.0 + fsub_tmp) * fconv_tmp
                    fssl_tmp = max(CLDMIN, min(fssl_tmp, 1.0 - CLDMIN))
                    fmc1 = (1.0 - fssl_tmp) + TEENY
                if mc1 or mccont > 0:
                    mplume = min(0.95 * airm[lmin] * fmc1, mplume)
                cond[:] = 0.0
                cdheat[:] = 0.0
                condp[:] = 0.0
                condp1[:] = 0.0
                condgp[:] = 0.0
                condip[:] = 0.0
                condv[:] = 0.0
                heat1[:] = 0.0
                dm[:] = 0.0
                dmr[:] = 0.0
                ddr[:] = 0.0
                ccm[:] = 0.0
                ddm[:] = 0.0
                taumc1[:] = 0.0
                ent[:] = 0.0
                det[:] = 0.0
                buoy[:] = 0.0
                wcu[:] = 0.0
                smdnl[:] = 0.0
                qmdnl[:] = 0.0
                smomdnl[:, :] = 0.0
                qmomdnl[:, :] = 0.0
                umdn[:] = 0.0
                vmdn[:] = 0.0
                umdnl[:, :] = 0.0
                vmdnl[:, :] = 0.0
                dum[:, :] = 0.0
                dvm[:, :] = 0.0
                dsm[:] = 0.0
                dsmom[:, :] = 0.0
                dsmr[:] = 0.0
                dsmomr[:, :] = 0.0
                dqm[:] = 0.0
                dqmom[:, :] = 0.0
                dqmr[:] = 0.0
                dqmomr[:, :] = 0.0
                mplume = min(mplume / fmc1, airm[lmin] * 0.95 * qm[lmin] / (qmold[lmin] + TEENY))
                if mplume <= F001 * airm[lmin]:
                    cycle_types = True
                    break
                fplume = mplume * byam[lmin]
                smp = smold[lmin] * fplume
                smomp[:] = 0.0
                qmomp[:] = 0.0
                for ii in range(5):
                    q = XYM[ii]
                    smomp[q] = smomold[q, lmin] * fplume
                qmp = qmold[lmin] * fplume
                for ii in range(5):
                    q = XYM[ii]
                    qmomp[q] = qmomold[q, lmin] * fplume
                if tpsav[lmin] == 0:
                    tpsav[lmin] = smp * plk[lmin] / mplume
                dmr[lmin] = -mplume
                dsmr[lmin] = -smp
                for ii in range(5):
                    q = XYM[ii]
                    dsmomr[q, lmin] = -smomp[q]
                for ii in range(4):
                    q = ZM[ii]
                    dsmomr[q, lmin] = -smomold[q, lmin] * fplume
                dqmr[lmin] = -qmp
                for ii in range(5):
                    q = XYM[ii]
                    dqmomr[q, lmin] = -qmomp[q]
                for ii in range(4):
                    q = ZM[ii]
                    dqmomr[q, lmin] = -qmomold[q, lmin] * fplume
                for k in range(nks):
                    ump[k] = um[k, lmin] * fplume
                    dum[k, lmin] = -ump[k]
                    vmp[k] = vm[k, lmin] * fplume
                    dvm[k, lmin] = -vmp[k]
                cdhsum = 0.0
                cdhsum1 = 0.0
                cdhdrt = 0.0
                etadn = 0.0
                ldraft = LM
                evpsum = 0.0
                ddraft = 0.0
                lfrz = 0
                lmax = lmin
                contce = contce1
                if ic == 2:
                    contce = contce2
                wcu[lmin] = max(.5, wturb[lmin + 1])
                if ic == 1:
                    wcu[lmin] = max(.5, 2.0 * wturb[lmin + 1])
                wcu2[lmin] = wcu[lmin] * wcu[lmin]
                mpmax = 0.0
                smpmax = 0.0
                qmpmax = 0.0
                smompmax[:] = 0.0
                qmompmax[:] = 0.0

                # ---------------------------------------------------------------- plume ascent
                for L in range(lmin + 1, LM + 1):
                    if mplume <= F001 * airm[L]:
                        break
                    sdn = smp / mplume
                    sup = sm1[L] * byam[L]
                    qdn = qmp / mplume
                    qup = qm1[L] * byam[L]
                    wmdn = 0.0
                    wmup = qcll[L] + qcil[L]
                    svdn = sdn * (1. + DELTX * qdn - wmdn)
                    svup = sup * (1. + DELTX * qup - wmup)
                    if plk[L - 1] * (svup - svdn) + SLHE * (qup - qdn) >= 0.:
                        break
                    if tpsav[L] == 0:
                        tpsav[L] = smp * plk[L] / mplume
                    tp = tpsav[L]
                    if tpsav[L - 1] >= TF and tpsav[L] < TF:
                        lfrz = L - 1
                    lhx = LHE
                    if tp < TI:
                        lhx = LHS
                    qsatmp = mplume * qsat(tp, lhx, pl[L])
                    if qmp < qsatmp:
                        break
                    if tp < TF and lhx == LHE:
                        lhx = LHS
                        qsatmp = mplume * qsat(tp, lhx, pl[L])
                    if vlat[L] == LHS:
                        lhx = LHS
                    vlat[L] = lhx
                    slh = lhx * BYSHA
                    if tl[L] >= TF and u00l[L] != u00a:
                        u00l[L] = u00_mc
                    mccont += 1
                    if mccont == 1:
                        mc1 = True
                    if mplume > F95 * airm[L]:
                        delta = (mplume - F95 * airm[L]) / mplume
                        dm[L - 1] = dm[L - 1] + delta * mplume
                        mplume = F95 * airm[L]
                        dsm[L - 1] = dsm[L - 1] + delta * smp
                        smp = smp * (1. - delta)
                        for ii in range(5):
                            q = XYM[ii]
                            dsmom[q, L - 1] = dsmom[q, L - 1] + delta * smomp[q]
                            smomp[q] = smomp[q] * (1. - delta)
                        dqm[L - 1] = dqm[L - 1] + delta * qmp
                        qmp = qmp * (1. - delta)
                        for ii in range(5):
                            q = XYM[ii]
                            dqmom[q, L - 1] = dqmom[q, L - 1] + delta * qmomp[q]
                            qmomp[q] = qmomp[q] * (1. - delta)
                        for k in range(nks):
                            dum[k, L - 1] = dum[k, L - 1] + ump[k] * delta
                            dvm[k, L - 1] = dvm[k, L - 1] + vmp[k] * delta
                            ump[k] = ump[k] - ump[k] * delta
                            vmp[k] = vmp[k] - vmp[k] * delta
                    work = mplume * (sup - sdn) * (plk[L - 1] - plk[L]) / plk[L - 1]
                    dsm[L - 1] = dsm[L - 1] - work
                    ccm[L - 1] = mplume
                    dqsum, fqcond = get_dq_cond(smp, qmp, plk[L], mplume, lhx, pl[L])
                    if dqsum > 0. and qmp > TEENY:
                        for ii in range(5):
                            q = XYM[ii]
                            qmomp[q] = qmomp[q] * (1. - fqcond)
                        smp = smp + slh * dqsum / plk[L]
                        qmp = qmp - dqsum
                    cond[L] = dqsum
                    condmmr[L] = condmmr[L] + cond[L] * byam[L] * fmc1
                    cdheat[L] = slh * cond[L]
                    cdhsum = cdhsum + cdheat[L]
                    cond[L] = cond[L] + condv[L - 1]
                    if vlat[L - 1] != vlat[L]:
                        smp = smp - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA / plk[L]
                        cdheat[L] = cdheat[L] - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA
                        cdhsum = cdhsum - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA
                    condmu = 100. * cond[L] * pl[L] / (ccm[L - 1] * tl[L] * RGAS)
                    flamw = (1000.0 * PI * CN0 / (condmu + TEENY)) ** .25
                    flamg = (400.0 * PI * CN0G / (condmu + TEENY)) ** .25
                    flami = (100.0 * PI * CN0I / (condmu + TEENY)) ** .25
                    taumc1[L] = taumc1[L] + cond[L] * fmc1
                    tvp = (smp / mplume) * plk[L] * (1. + DELTX * qmp / mplume)
                    buoy[L] = (tvp - tvl[L]) / tvl[L] - cond[L] / mplume
                    ent[L] = .16667 * contce * GRAV * buoy[L] / (wcu[L - 1] * wcu[L - 1] + TEENY)
                    if ent[L] < 0.:
                        det[L] = -ent[L]
                        ent[L] = 0.
                    if ent[L] > 0.:
                        fentr = 1000. * ent[L] * gzl[L] * fplume
                        if fentr + fplume > 1.:
                            fentr = 1. - fplume
                            ent[L] = 0.001 * fentr / (gzl[L] * fplume)
                        if fentr >= TEENY:
                            mpold = mplume
                            fpold = fplume
                            etal1 = fentr / (fplume + TEENY)
                            eplume = mplume * etal1
                            if eplume > airm[L] * 0.975 - mplume:
                                eplume = airm[L] * 0.975 - mplume
                            mplume = mplume + eplume
                            etal1 = eplume / mpold
                            fentr = etal1 * fpold
                            ent[L] = 0.001 * fentr / (gzl[L] * fpold)
                            if mc_entr_lim == 0:
                                fplume = fplume + fentr
                            else:
                                fplume = mplume * byam[L]
                            fentra = eplume * byam[L]
                            dsmr[L] = dsmr[L] - eplume * sup
                            for q in range(NMOM):
                                dsmomr[q, L] = dsmomr[q, L] - smom[q, L] * fentra
                            dqmr[L] = dqmr[L] - eplume * qup
                            for q in range(NMOM):
                                dqmomr[q, L] = dqmomr[q, L] - qmom[q, L] * fentra
                            dmr[L] = dmr[L] - eplume
                            smp = smp + eplume * sup
                            for ii in range(5):
                                q = XYM[ii]
                                smomp[q] = smomp[q] + smom[q, L] * fentra
                            qmp = qmp + eplume * qup
                            for ii in range(5):
                                q = XYM[ii]
                                qmomp[q] = qmomp[q] + qmom[q, L] * fentra
                            for k in range(nks):
                                umtemp[k] = PGRAD * mplume ** 2 * (u_0[k, L + 1] - u_0[k, L]) / (pl[L] - pl[L + 1])
                                vmtemp[k] = PGRAD * mplume ** 2 * (v_0[k, L + 1] - v_0[k, L]) / (pl[L] - pl[L + 1])
                                ump[k] = ump[k] + u_0[k, L] * eplume + umtemp[k]
                                dum[k, L] = dum[k, L] - u_0[k, L] * eplume - umtemp[k]
                                vmp[k] = vmp[k] + v_0[k, L] * eplume + vmtemp[k]
                                dvm[k, L] = dvm[k, L] - v_0[k, L] * eplume - vmtemp[k]
                    if det[L] > 0.:
                        delta = 1000. * det[L] * gzl[L]
                        if delta > .95:
                            delta = .95
                            det[L] = .001 * delta / gzl[L]
                        dm[L] = dm[L] + delta * mplume
                        mplume = mplume * (1. - delta)
                        dsm[L] = dsm[L] + delta * smp
                        smp = smp * (1. - delta)
                        for ii in range(5):
                            q = XYM[ii]
                            dsmom[q, L] = dsmom[q, L] + delta * smomp[q]
                            smomp[q] = smomp[q] * (1. - delta)
                        dqm[L] = dqm[L] + delta * qmp
                        qmp = qmp * (1. - delta)
                        for ii in range(5):
                            q = XYM[ii]
                            dqmom[q, L] = dqmom[q, L] + delta * qmomp[q]
                            qmomp[q] = qmomp[q] * (1. - delta)
                        for k in range(nks):
                            umtemp[k] = PGRAD * mplume ** 2 * (u_0[k, L + 1] - u_0[k, L]) / (pl[L] - pl[L + 1])
                            vmtemp[k] = PGRAD * mplume ** 2 * (v_0[k, L + 1] - v_0[k, L]) / (pl[L] - pl[L + 1])
                            dum[k, L] = dum[k, L] + ump[k] * delta - umtemp[k]
                            dvm[k, L] = dvm[k, L] + vmp[k] * delta - vmtemp[k]
                            ump[k] = ump[k] - ump[k] * delta + umtemp[k]
                            vmp[k] = vmp[k] - vmp[k] * delta + vmtemp[k]
                    if L - lmin > 1:
                        smix = .5 * (sup + smp / mplume)
                        qmix = .5 * (qup + qmp / mplume)
                        wmix = .5 * (wmup + cond[L] / mplume)
                        if mc_newthv == 0:
                            svmix = smix
                            svup = sup
                            dmmix = (svup - svmix) * plk[L]
                        else:
                            svmix = smix * (1. + DELTX * qmix - wmix)
                            svup = sup * (1. + DELTX * qup - wmup)
                            dmmix = (svup - svmix) * plk[L] + SLHE * (qsat(sup * plk[L], lhx, pl[L]) - qmix)
                        if dmmix < 1e-10:
                            cdhdrt = cdhdrt + cdheat[L]
                        if dmmix >= 1e-10:
                            ldraft = L
                            etadn = 1.0 / 3.0
                            fleft = 1. - .5 * etadn
                            ddraft = etadn * mplume
                            ddr[L] = ddraft
                            cdhsum1 = cdhsum1 + cdhdrt * .5 * etadn
                            cdhdrt = cdhdrt - cdhdrt * .5 * etadn + cdheat[L]
                            fddp = .5 * ddraft
                            fddp = fddp / mplume
                            fddl = .5 * ddraft * byam[L]
                            mplume = fleft * mplume
                            smdnl[L] = ddraft * smix
                            for ii in range(5):
                                q = XYM[ii]
                                smomdnl[q, L] = smom[q, L] * fddl + smomp[q] * fddp
                            smp = fleft * smp
                            for ii in range(5):
                                q = XYM[ii]
                                smomp[q] = smomp[q] * fleft
                            qmdnl[L] = ddraft * qmix
                            for ii in range(5):
                                q = XYM[ii]
                                qmomdnl[q, L] = qmom[q, L] * fddl + qmomp[q] * fddp
                            qmp = fleft * qmp
                            for ii in range(5):
                                q = XYM[ii]
                                qmomp[q] = qmomp[q] * fleft
                            dmr[L] = dmr[L] - .5 * ddraft
                            dsmr[L] = dsmr[L] - .5 * ddraft * sup
                            for q in range(NMOM):
                                dsmomr[q, L] = dsmomr[q, L] - smom[q, L] * fddl
                            dqmr[L] = dqmr[L] - .5 * ddraft * qup
                            for q in range(NMOM):
                                dqmomr[q, L] = dqmomr[q, L] - qmom[q, L] * fddl
                            for k in range(nks):
                                umdnl[k, L] = .5 * (etadn * ump[k] + ddraft * u_0[k, L])
                                ump[k] = ump[k] * fleft
                                dum[k, L] = dum[k, L] - .5 * ddraft * u_0[k, L]
                                vmdnl[k, L] = .5 * (etadn * vmp[k] + ddraft * v_0[k, L])
                                vmp[k] = vmp[k] * fleft
                                dvm[k, L] = dvm[k, L] - .5 * ddraft * v_0[k, L]
                    w2tem = .16667 * GRAV * buoy[L] - wcu[L - 1] * wcu[L - 1] * (.66667 * det[L] + ent[L])
                    hdep = airm[L] * tl[L] * RGAS / (GRAV * pl[L])
                    wcu2[L] = wcu2[L - 1] + 2. * hdep * w2tem
                    wcu[L] = 0.
                    if wcu2[L] > 0.:
                        wcu[L] = math.sqrt(wcu2[L])
                    if wcu[L] >= 0.:
                        wcu[L] = min(50., wcu[L])
                    if wcu[L] < 0.:
                        wcu[L] = max(-50., wcu[L])
                    smpmax = smp
                    for q in range(NMOM):
                        smompmax[q] = smomp[q]
                        qmompmax[q] = qmomp[q]
                    qmpmax = qmp
                    mpmax = mplume
                    lmax = lmax + 1
                    if wcu2[L] < 0.:
                        break
                    wcufrz = 0.0
                    if lfrz > 0:
                        wcufrz = wcu[lfrz]
                    cp, cp1, cip, cgp = conv_micro(pl[L], wcu[L], dwcu, float(lfrz), wcufrz, tp, TI, pland,
                                                   flamw, flamg, flami, tl[lmin], tl[lmin + 1], condip[L],
                                                   condgp[L])
                    condp[L] = cp
                    condp1[L] = cp1
                    condip[L] = cip
                    condgp[L] = cgp
                    condp[L] = .01 * condp[L] * ccm[L - 1] * tl[L] * RGAS / pl[L]
                    condp1[L] = .01 * condp1[L] * ccm[L - 1] * tl[L] * RGAS / pl[L]
                    if condp1[L] > cond[L]:
                        condp1[L] = cond[L]
                    if condp[L] > condp1[L]:
                        condp[L] = condp1[L]
                    condv[L] = cond[L] - condp1[L]
                    cond[L] = cond[L] - condv[L]
                    taumc1[L] = taumc1[L] - condv[L] * fmc1
                if lmin == lmax:
                    cycle_types = True
                    break
            if cycle_types:
                continue
            # ---------------------------------------------------------------- after the plume partitions
            for L in range(lmin, lmax + 1):
                taumcl[L] = taumcl[L] + taumc1[L]
            if pl[lmin] < pl[dcl]:
                u00l[lmin] = u00_mc
            else:
                for L in range(1, lmin + 1):
                    u00l[L] = u00_mc
            if tpsav[lmax] >= TF:
                lfrz = lmax
            u00l[lmax] = u00a
            dm[lmax] = dm[lmax] + mpmax
            dsm[lmax] = dsm[lmax] + smpmax
            for ii in range(5):
                q = XYM[ii]
                dsmom[q, lmax] = dsmom[q, lmax] + smompmax[q]
            dqm[lmax] = dqm[lmax] + qmpmax
            for ii in range(5):
                q = XYM[ii]
                dqmom[q, lmax] = dqmom[q, lmax] + qmompmax[q]
            ccm[lmax] = 0.
            for k in range(nks):
                dum[k, lmax] = dum[k, lmax] + ump[k]
                dvm[k, lmax] = dvm[k, lmax] + vmp[k]
            cdhm = 0.0
            if minlvl > lmin:
                minlvl = lmin
            if maxlvl < lmax:
                maxlvl = lmax
            if lmcmin == 0:
                lmcmin = lmin
            if lmcmax < maxlvl:
                lmcmax = maxlvl

            # ---------------------------------------------------------------- downdraft
            ldmin = ldraft - 1
            llmin = ldmin
            edraft = 0.0
            smdn = 0.0
            qmdn = 0.0
            ddrold = 0.0
            if etadn > 1e-10:
                ddraft = ddr[ldraft]
                ddrold = ddraft
                smdn = smdnl[ldraft]
                qmdn = qmdnl[ldraft]
                smomdn[:] = 0.0
                qmomdn[:] = 0.0
                for ii in range(5):
                    q = XYM[ii]
                    smomdn[q] = smomdnl[q, ldraft]
                    qmomdn[q] = qmomdnl[q, ldraft]
                for k in range(nks):
                    umdn[k] = umdnl[k, ldraft]
                    vmdn[k] = vmdnl[k, ldraft]
                for L in range(ldraft, 0, -1):
                    lhx = vlat[L]
                    slh = lhx * BYSHA
                    dqe, fq1 = get_dq_evap(smdn, qmdn, plk[L], ddraft, lhx, pl[L], cond[L])
                    dqsum = dqe
                    dqevp = fddrt * cond[L]
                    if dqevp > dqsum:
                        dqevp = dqsum
                    if dqevp > smdn * plk[L] / slh:
                        dqevp = smdn * plk[L] / slh
                    if L < lmin:
                        dqevp = 0.0
                    fsevp = 0.0
                    if plk[L] * smdn > TEENY:
                        fsevp = slh * dqevp / (plk[L] * smdn)
                    smdn = smdn - slh * dqevp / plk[L]
                    for ii in range(5):
                        q = XYM[ii]
                        smomdn[q] = smomdn[q] * (1. - fsevp)
                    qmdn = qmdn + dqevp
                    cond[L] = cond[L] - dqevp
                    taumcl[L] = taumcl[L] - dqevp * fmc1
                    cdheat[L] = cdheat[L] - dqevp * slh
                    evpsum = evpsum + dqevp * slh
                    if L < ldraft and L > 1:
                        ddrup = ddraft
                        ddraft = ddraft + ddrold * etal[L]
                        if ddrup > ddraft:
                            ddraft = ddrup
                        smix = smdn / (ddrup + TEENY)
                        qmix = qmdn / (ddrup + TEENY)
                        wmix = cond[L] / (ddrup + TEENY)
                        if mc_newthv == 0:
                            svmix = smix * plk[L - 1]
                            svm1 = sm1[L - 1] * byam[L - 1] * plk[L - 1]
                        else:
                            svmix = smix * plk[L - 1] * (1. + DELTX * qmix - wmix)
                            svm1 = sm1[L - 1] * byam[L - 1] * plk[L - 1] * (
                                1. + DELTX * qm1[L - 1] * byam[L - 1] - qcll[L - 1] - qcil[L - 1])
                        if svmix - svm1 >= DTMIN1:
                            ddraft = FDDET * ddrup
                        if ddraft > .95 * (airm[L - 1] + dmr[L - 1]):
                            ddraft = .95 * (airm[L - 1] + dmr[L - 1])
                        edraft = ddraft - ddrup
                        if edraft > 0:
                            fentra = edraft * byam[L]
                            senv = sm[L] * byam[L]
                            qenv = qm[L] * byam[L]
                            smdn = smdn + edraft * senv
                            qmdn = qmdn + edraft * qenv
                            for ii in range(5):
                                q = XYM[ii]
                                smomdn[q] = smomdn[q] + smom[q, L] * fentra
                                qmomdn[q] = qmomdn[q] + qmom[q, L] * fentra
                            dsmr[L] = dsmr[L] - edraft * senv
                            for q in range(NMOM):
                                dsmomr[q, L] = dsmomr[q, L] - smom[q, L] * fentra
                            dqmr[L] = dqmr[L] - edraft * qenv
                            for q in range(NMOM):
                                dqmomr[q, L] = dqmomr[q, L] - qmom[q, L] * fentra
                            dmr[L] = dmr[L] - edraft
                            for k in range(nks):
                                umtemp[k] = PGRAD * ddraft ** 2 * (u_0[k, L + 1] - u_0[k, L]) / (pl[L] - pl[L + 1])
                                vmtemp[k] = PGRAD * ddraft ** 2 * (v_0[k, L + 1] - v_0[k, L]) / (pl[L] - pl[L + 1])
                                umdn[k] = umdn[k] + fentra * um[k, L] - umtemp[k]
                                vmdn[k] = vmdn[k] + fentra * vm[k, L] - vmtemp[k]
                                dum[k, L] = dum[k, L] - fentra * um[k, L] + umtemp[k]
                                dvm[k, L] = dvm[k, L] - fentra * vm[k, L] + vmtemp[k]
                        else:
                            fentra = edraft / (ddrup + TEENY)
                            dsm[L] = dsm[L] - fentra * smdn
                            for ii in range(5):
                                q = XYM[ii]
                                dsmom[q, L] = dsmom[q, L] - smomdn[q] * fentra
                            dqm[L] = dqm[L] - fentra * qmdn
                            for ii in range(5):
                                q = XYM[ii]
                                dqmom[q, L] = dqmom[q, L] - qmomdn[q] * fentra
                            smdn = smdn * (1 + fentra)
                            qmdn = qmdn * (1 + fentra)
                            for ii in range(5):
                                q = XYM[ii]
                                smomdn[q] = smomdn[q] * (1 + fentra)
                                qmomdn[q] = qmomdn[q] * (1 + fentra)
                            dm[L] = dm[L] - edraft
                            for k in range(nks):
                                umtemp[k] = PGRAD * ddraft ** 2 * (u_0[k, L + 1] - u_0[k, L]) / (pl[L] - pl[L + 1])
                                vmtemp[k] = PGRAD * ddraft ** 2 * (v_0[k, L + 1] - v_0[k, L]) / (pl[L] - pl[L + 1])
                                dum[k, L] = dum[k, L] - fentra * umdn[k] + umtemp[k]
                                dvm[k, L] = dvm[k, L] - fentra * vmdn[k] + vmtemp[k]
                                umdn[k] = umdn[k] * (1. + fentra) - umtemp[k]
                                vmdn[k] = vmdn[k] * (1. + fentra) - vmtemp[k]
                    ldmin = L
                    llmin = ldmin
                    if L > 1:
                        smix = smdn / (ddraft + TEENY)
                        qmix = qmdn / (ddraft + TEENY)
                        wmix = cond[L - 1] / (ddraft + TEENY)
                        svmix = smix * plk[L - 1] * (1. + DELTX * qmix - wmix)
                        svm1 = sm1[L - 1] * byam[L - 1] * plk[L - 1] * (
                            1. + DELTX * qm1[L - 1] * byam[L - 1] - qcll[L - 1] - qcil[L - 1])
                        if L <= lmin and svmix >= svm1:
                            break
                        ddm[L - 1] = ddraft
                        ddrold = ddraft
                        ddraft = ddraft + ddr[L - 1]
                        smdn = smdn + smdnl[L - 1]
                        qmdn = qmdn + qmdnl[L - 1]
                        for ii in range(5):
                            q = XYM[ii]
                            smomdn[q] = smomdn[q] + smomdnl[q, L - 1]
                            qmomdn[q] = qmomdn[q] + qmomdnl[q, L - 1]
                        for k in range(nks):
                            umdn[k] = umdn[k] + umdnl[k, L - 1]
                            vmdn[k] = vmdn[k] + vmdnl[k, L - 1]
                dsm[ldmin] = dsm[ldmin] + smdn
                for ii in range(5):
                    q = XYM[ii]
                    dsmom[q, ldmin] = dsmom[q, ldmin] + smomdn[q]
                dqm[ldmin] = dqm[ldmin] + qmdn
                for ii in range(5):
                    q = XYM[ii]
                    dqmom[q, ldmin] = dqmom[q, ldmin] + qmomdn[q]
                tdnl[ldmin] = smdn * plk[ldmin] / (ddraft + TEENY)
                qdnl[ldmin] = qmdn / (ddraft + TEENY)
                for k in range(nks):
                    dum[k, ldmin] = dum[k, ldmin] + umdn[k]
                    dvm[k, ldmin] = dvm[k, ldmin] + vmdn[k]
                dm[ldmin] = dm[ldmin] + ddraft

            # ---------------------------------------------------------------- subsidence
            if ldmin > lmin:
                ldmin = lmin
            cm[:] = 0.0
            smt = np.zeros(LM + 2)
            qmt = np.zeros(LM + 2)
            for L in range(ldmin, lmax + 1):
                cm[L] = cm[L - 1] - dm[L] - dmr[L]
                smt[L] = sm[L]
                qmt[L] = qm[L]
            for L in range(lmax, LM + 1):
                cm[L] = 0.0
            ksub = 1
            for l in range(ldmin, lmax):
                if +cm[l] > airm[l + 1] + dmr[l + 1]:
                    ksub = max(ksub, 1 + int((+cm[l] - dmr[l + 1]) / airm[l + 1]))
                elif -cm[l] > airm[l] + dmr[l]:
                    ksub = max(ksub, 1 + int((-cm[l] - dmr[l]) / airm[l]))
            ksub = min(ksub, 2)
            byksub = 1.0 / ksub
            sumu = np.zeros(nks)
            sumv = np.zeros(nks)
            for l in range(ldmin, lmax + 1):
                for k in range(nks):
                    sumu[k] = sumu[k] + um[k, l]
                    sumv[k] = sumv[k] + vm[k, l]
            sumdp = 0.0
            for L in range(ldmin, lmax + 1):
                sumdp = sumdp + airm[L]
            alpha = 0.0
            for L in range(ldmin, lmax + 1):
                cldm = ccm[L]
                if L < ldraft and L >= llmin and etadn > 1e-10:
                    cldm = ccm[L] - ddm[L]
                if mc1:
                    vsubl[L] = 100. * cldm * RGAS * tl[L] / (pl[L] * GRAV * dtsrc)
                beta = cldm * byam[L + 1]
                if cldm < 0.:
                    beta = cldm * byam[L]
                betau = beta
                alphau = alpha
                if beta < 0.:
                    betau = 0.0
                if alpha < 0.:
                    alphau = 0.0
                for k in range(nks):
                    um[k, L] = um[k, L] + ra[k] * (-alphau * um[k, L] + betau * um[k, L + 1] + dum[k, L])
                    vm[k, L] = vm[k, L] + ra[k] * (-alphau * vm[k, L] + betau * vm[k, L + 1] + dvm[k, L])
                alpha = beta
            for k in range(nks):
                s1u = 0.0
                s1v = 0.0
                for l in range(ldmin, lmax + 1):
                    s1u = s1u + um[k, l]
                    s1v = s1v + vm[k, l]
                for l in range(ldmin, lmax + 1):
                    um[k, l] = um[k, l] - (s1u - sumu[k]) * airm[l] / sumdp
                    vm[k, l] = vm[k, l] - (s1v - sumv[k]) * airm[l] / sumdp
            nsub = lmax - ldmin + 1
            for L in range(ldmin, lmax):
                cmneg[L] = -cm[L] * byksub
            cmneg[lmax] = 0.
            s_l = np.zeros(nsub)
            q_l = np.zeros(nsub)
            sm_m = np.zeros((NMOM, nsub))
            qm_m = np.zeros((NMOM, nsub))
            ml = np.zeros(nsub)
            cmn = np.zeros(nsub)
            for i in range(nsub):
                cmn[i] = cmneg[ldmin + i]
            for it in range(1, ksub + 1):
                for i in range(nsub):
                    Lc = ldmin + i
                    ml[i] = airm[Lc] + dmr[Lc] * byksub
                    sm[Lc] = sm[Lc] + dsmr[Lc] * byksub
                    for q in range(NMOM):
                        smom[q, Lc] = smom[q, Lc] + dsmomr[q, Lc] * byksub
                    s_l[i] = sm[Lc]
                    for q in range(NMOM):
                        sm_m[q, i] = smom[q, Lc]
                ierrt, lerrt = adv1d_z(s_l, sm_m, ml.copy(), cmn, nsub, False)
                for i in range(nsub):
                    Lc = ldmin + i
                    sm[Lc] = s_l[i]
                    for q in range(NMOM):
                        smom[q, Lc] = sm_m[q, i]
                for i in range(nsub):
                    Lc = ldmin + i
                    sm[Lc] = sm[Lc] + dsm[Lc] * byksub
                    for q in range(NMOM):
                        smom[q, Lc] = smom[q, Lc] + dsmom[q, Lc] * byksub
                ierr = max(ierrt, ierr)
                lerr = max(lerrt + ldmin - 1, lerr)
                for i in range(nsub):
                    Lc = ldmin + i
                    ml[i] = airm[Lc] + dmr[Lc] * byksub
                    qm[Lc] = qm[Lc] + dqmr[Lc] * byksub
                    for q in range(NMOM):
                        qmom[q, Lc] = qmom[q, Lc] + dqmomr[q, Lc] * byksub
                    q_l[i] = qm[Lc]
                    for q in range(NMOM):
                        qm_m[q, i] = qmom[q, Lc]
                ierrt, lerrt = adv1d_z(q_l, qm_m, ml.copy(), cmn, nsub, True)
                for i in range(nsub):
                    Lc = ldmin + i
                    qm[Lc] = q_l[i]
                    for q in range(NMOM):
                        qmom[q, Lc] = qm_m[q, i]
                for i in range(nsub):
                    Lc = ldmin + i
                    qm[Lc] = qm[Lc] + dqm[Lc] * byksub
                    for q in range(NMOM):
                        qmom[q, Lc] = qmom[q, Lc] + dqmom[q, Lc] * byksub
                ierr = max(ierrt, ierr)
                lerr = max(lerrt + ldmin - 1, lerr)
            # ---------------------------------------------------------------- diagnostics
            for L in range(ldmin, lmax + 1):
                fcdh = 0.0
                if L == lmax:
                    fcdh = cdhsum - cdhsum1 + cdhm
                fcdh1 = 0.0
                if L == llmin:
                    fcdh1 = cdhsum1 - evpsum
                mcflx[L] = mcflx[L] + ccm[L] * fmc1
                dgdsm[L] = dgdsm[L] + (plk[L] * (sm[L] - smt[L]) - fcdh - fcdh1) * fmc1
                dgdqm[L] = dgdqm[L] + SLHE * (qm[L] - qmt[L]) * fmc1
                dqmtotal[L] = dqmtotal[L] + (qm[L] - qmt[L]) * byam[L] * fmc1
                if ple[lmax + 1] > 700.:
                    dgshlw[L] = dgshlw[L] + (plk[L] * (sm[L] - smt[L]) - fcdh - fcdh1) * fmc1
                    dqmshlw[L] = dqmshlw[L] + (qm[L] - qmt[L]) * byam[L] * fmc1
                if ple[lmin] - ple[lmax + 1] >= 450.:
                    dgdeep[L] = dgdeep[L] + (plk[L] * (sm[L] - smt[L]) - fcdh - fcdh1) * fmc1
                    dqmdeep[L] = dqmdeep[L] + (qm[L] - qmt[L]) * byam[L] * fmc1
                dtotw[L] = dtotw[L] + SLHE * (qm[L] - qmt[L] + cond[L]) * fmc1
                ddmflx[L] = ddmflx[L] + ddm[L] * fmc1
            for L in range(1, LM + 1):
                sm1[L] = sm[L]
                qm1[L] = qm[L]

            # ---------------------------------------------------------------- precipitation
            cond[lmax] = cond[lmax] + condv[lmax]
            if ple[lmin] - ple[lmax + 1] >= 450.:
                for L in range(lmax, lmin - 1, -1):
                    if cond[L] < condp[L]:
                        condp[L] = cond[L]
                    fclw = 0.0
                    if cond[L] > 0:
                        fclw = (cond[L] - condp[L]) / cond[L]
                    if svlatl[L] > 0 and svlatl[L] != vlat[L]:
                        heat1[L] = heat1[L] + (svlatl[L] - vlat[L]) * svwmxl[L] * airm[L] * BYSHA / fmc1
                    svlatl[L] = vlat[L]
                    svwmxl[L] = svwmxl[L] + fclw * cond[L] * byam[L] * fmc1
                    cond[L] = condp[L]
                    condpt[L] = condpt[L] + condp[L]
            prcp = cond[lmax]
            prheat = cdheat[lmax]
            told = smold[lmax] * plk[lmax] * byam[lmax]
            lhp[lmax] = LHE
            if told <= TF:
                lhp[lmax] = LHS
            if (told > TF and vlat[lmax] == LHS) or (told <= TF and vlat[lmax] == LHE):
                fssum = 0.0
                if abs(plk[lmax] * sm[lmax]) > TEENY and (lhp[lmax] - vlat[lmax]) * prcp * BYSHA < 0:
                    fssum = -((lhp[lmax] - vlat[lmax]) * prcp * BYSHA / (plk[lmax] * sm[lmax]))
                sm[lmax] = sm[lmax] + (lhp[lmax] - vlat[lmax]) * prcp * BYSHA / plk[lmax]
                for q in range(NMOM):
                    smom[q, lmax] = smom[q, lmax] * (1. - fssum)
            dphase[lmax] = dphase[lmax] + (cdhsum - cdhsum1 + cdhm) * fmc1
            if ple[lmax + 1] > 700:
                dphashlw[lmax] = dphashlw[lmax] + (cdhsum - cdhsum1 + cdhm) * fmc1
            if ple[lmin] - ple[lmax + 1] >= 450:
                dphadeep[lmax] = dphadeep[lmax] + (cdhsum - cdhsum1 + cdhm) * fmc1
            for L in range(lmax - 1, 0, -1):
                fcloud = cloud_fraction(float(L), tl[L], pl[L], ccm[L], wcu[L], ple[lmin], ple[L + 2],
                                        ple[lmax + 1], ccm[lmin], wcu[lmin], float(lmin), float(lmax), dtsrc)
                if fcloud < 0.:
                    negcover = 1
                fevap = .5 * ccm[L] * byam[L + 1]
                if L < lmin:
                    fevap = .5 * ccm[lmin] * byam[lmin + 1]
                if fevap > .5:
                    fevap = .5
                cldmcl[L + 1] = min(cldmcl[L + 1] + fcloud * fmc1, fmc1)
                cldref = cldmcl[L + 1]
                if ple[lmax + 1] > 700 and cldref > cldslwij:
                    cldslwij = cldref
                if ple[lmin] - ple[lmax + 1] >= 450 and cldref > clddepij:
                    clddepij = cldref
                told = smold[L] * plk[L] * byam[L]
                told1 = smold[L + 1] * plk[L + 1] * byam[L + 1]
                precnvl[L + 1] = precnvl[L + 1] + prcp * BYGRAV
                lh, mcloud, h1 = precip_phase(float(L), airm[L], fevap, lhp[L + 1], prcp, told, told1, vlat[L],
                                              cond[L], float(lmin), heat1[L], mc_revp)
                lhp[L] = lh
                heat1[L] = h1
                lhx = lhp[L]
                dqsum = 0.0
                fprcp = 0.0
                slh = lhx * BYSHA
                if prcp > 0.:
                    if mcloud > 0:
                        dqs, fprcp = get_dq_evap(smold[L], qmold[L], plk[L], airm[L], lhx, pl[L],
                                                 prcp * airm[L] / mcloud)
                        dqsum = dqs
                    dqsum = dqsum * mcloud * byam[L]
                    prcp = prcp - dqsum
                    qm[L] = qm[L] + dqsum
                fssum = 0.0
                if abs(plk[L] * sm[L]) > TEENY and slh * dqsum + heat1[L] > 0:
                    fssum = (slh * dqsum + heat1[L]) / (plk[L] * sm[L])
                sm[L] = sm[L] - (slh * dqsum + heat1[L]) / plk[L]
                for q in range(NMOM):
                    smom[q, L] = smom[q, L] * (1. - fssum)
                fcdh1 = 0.0
                if L == llmin:
                    fcdh1 = cdhsum1 - evpsum
                dphase[L] = dphase[L] - (slh * dqsum - fcdh1 + heat1[L]) * fmc1
                dqcond[L] = dqcond[L] - slh * dqsum * fmc1
                dqctotal[L] = dqctotal[L] - dqsum * byam[L] * fmc1
                if ple[lmax + 1] > 700.:
                    dphashlw[L] = dphashlw[L] - (slh * dqsum - fcdh1 + heat1[L]) * fmc1
                    dqcshlw[L] = dqcshlw[L] - dqsum * byam[L] * fmc1
                if ple[lmin] - ple[lmax + 1] >= 450.:
                    dphadeep[L] = dphadeep[L] - (slh * dqsum - fcdh1 + heat1[L]) * fmc1
                    dqcdeep[L] = dqcdeep[L] - dqsum * byam[L] * fmc1
                prheat = cdheat[L] + slh * prcp
                prcp = prcp + cond[L]
            if prcp > 0.:
                if ple[lmin] - ple[lmax + 1] < 450.:
                    cldmcl[1] = min(cldmcl[1], fmc1)
                else:
                    rho = pl[1] / (RGAS * tl[1])
                    cldmcl[1] = min(cldmcl[1] + fmc1 * ccm[lmin] / (rho * GRAV * wcu[lmin] * dtsrc + TEENY), fmc1)
            prcpmc = prcpmc + prcp * fmc1
            if lmcmin > ldmin:
                lmcmin = ldmin
            mc1 = False

    # ---------------------------------------------------------------- after the cloud-base loop
    if lmcmin > 0:
        for L in range(1, lmcmax + 1):
            fssl[L] = 1 - fmc1
        sumaj = 0.0
        sumdp = 0.0
        for L in range(lmcmin, lmcmax + 1):
            sumdp = sumdp + airm[L] * fmc1
            sumaj = sumaj + dgdsm[L]
        for L in range(lmcmin, lmcmax + 1):
            dgdsm[L] = dgdsm[L] - sumaj * airm[L] * fmc1 / sumdp
            sm[L] = sm[L] - sumaj * airm[L] / (sumdp * plk[L])
        airxl = 0.0
        for L in range(lmcmin, lmcmax + 1):
            airxl = airxl + mcflx[L]

    # ---------------------------------------------------------------- optical thickness
    wconst = WMU * (1. - pearth) + WMUL * pearth
    wconst = wmu_mult * wconst
    wmsum = 0.0
    qlmc = np.zeros(LM + 2)
    qimc = np.zeros(LM + 2)
    wmctwp = 0.0
    wmclwp = 0.0
    for L in range(1, lmcmax + 1):
        tl[L] = (sm[L] * byam[L]) * plk[L]
        temwm = (taumcl[L] - svwmxl[L] * airm[L]) * 1e2 * BYGRAV
        if tl[L] >= TF:
            wmsum = wmsum + temwm
        wmctwp = wmctwp + temwm
        if tl[L] >= TF:
            wmclwp = wmclwp + temwm
        temwm = taumcl[L] - condpt[L] * fmc1 - svwmxl[L] * airm[L]
        if svlatl[L] == LHE:
            qlmc[L] = temwm / airm[L] + svwmxl[L]
        elif svlatl[L] == LHS:
            qimc[L] = temwm / airm[L] + svwmxl[L]
        if lhp[L] == LHE:
            qlmc[L] = qlmc[L] + condpt[L] * fmc1 / airm[L]
        elif lhp[L] == LHS:
            qimc[L] = qimc[L] + condpt[L] * fmc1 / airm[L]
        if cldmcl[L] > 0.:
            taumcl[L] = airm[L] * COETAU
            if L == lmcmax and ple[lmcmin] - ple[lmcmax + 1] < 450:
                taumcl[L] = airm[L] * .02
            if L <= lmcmin and ple[lmcmin] - ple[lmcmax + 1] >= 450:
                taumcl[L] = airm[L] * .02
        svlat1[L] = svlatl[L]
        if svlatl[L] == 0.:
            svlatl[L] = LHE
            if (tpsav[L] > 0. and tpsav[L] < TF) or (tpsav[L] == 0. and tl[L] < TF):
                svlatl[L] = LHS
        if svwmxl[L] > 0.:
            fcld = cldmcl[L] + F1E20
            tem = 1e5 * svwmxl[L] * airm[L] * BYGRAV
            wtem = 1e5 * svwmxl[L] * pl[L] / (fcld * tl[L] * RGAS)
            if svlatl[L] == LHE and svwmxl[L] / fcld >= wconst * 1e-3:
                wtem = 1e2 * wconst * pl[L] / (tl[L] * RGAS)
            if wtem < 1e-10:
                wtem = 1e-10
            mndo = 59.68 / (rwcldox * rwcldox * rwcldox)
            mndl = 174.0
            mndi = 0.06417127
            mcdncw = mndo * (1. - pearth) + mndl * pearth
            mcdnci = mndi
            rcld, tau = anvil(svlatl[L], rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem)
            taumcl[L] = tau
            csizel[L] = rcld / bybr
        if taumcl[L] < 0. and cldmcl[L] <= 0.:
            taumcl[L] = 0.
    if lmcmax <= 1:
        for L in range(1, LM + 1):
            if pl[L] < pl[dcl]:
                u00l[L] = 0.
    cnvmmrl = np.zeros(LM + 2)
    for L in range(1, LM + 1):
        cnvmmrl[L] = condmmr[L] if cldmcl[L] > 0.0 else 0.0

    o_sc[0] = ierr
    o_sc[1] = lerr
    o_sc[2] = lmcmin
    o_sc[3] = lmcmax
    o_sc[4] = prcpmc
    o_sc[5] = cldslwij
    o_sc[6] = clddepij
    o_sc[7] = airxl
    o_sc[8] = prheat
    o_sc[9] = wmsum
    o_sc[10] = wmctwp
    o_sc[11] = wmclwp
    o_sc[12] = fmc1
    o_sc[13] = mccont
    o_sc[14] = negcover
    for L in range(LM):
        i = L + 1
        o_l[0, L] = tl[i]
        o_l[1, L] = sm[i]
        o_l[2, L] = qm[i]
        o_l[3, L] = fssl[i]
        o_l[4, L] = cldmcl[i]
        o_l[5, L] = taumcl[i]
        o_l[6, L] = svlatl[i]
        o_l[7, L] = svlat1[i]
        o_l[8, L] = svwmxl[i]
        o_l[9, L] = csizel[i]
        o_l[10, L] = condpt[i]
        o_l[11, L] = vsubl[i]
        o_l[12, L] = tpsav[i]
        o_l[13, L] = mcflx[i]
        o_l[14, L] = dgdsm[i]
        o_l[15, L] = dgdeep[i]
        o_l[16, L] = dgshlw[i]
        o_l[17, L] = dphase[i]
        o_l[18, L] = dphadeep[i]
        o_l[19, L] = dphashlw[i]
        o_l[20, L] = dtotw[i]
        o_l[21, L] = dqcond[i]
        o_l[22, L] = dgdqm[i]
        o_l[23, L] = dqmtotal[i]
        o_l[24, L] = dqmshlw[i]
        o_l[25, L] = dqmdeep[i]
        o_l[26, L] = dqctotal[i]
        o_l[27, L] = dqcshlw[i]
        o_l[28, L] = dqcdeep[i]
        o_l[29, L] = ddmflx[i]
        o_l[30, L] = tdnl[i]
        o_l[31, L] = qdnl[i]
        o_l[32, L] = u00l[i]
        o_l[33, L] = qlmc[i]
        o_l[34, L] = qimc[i]
        o_l[35, L] = cnvmmrl[i]
        o_l[36, L] = condmmr[i]
        for q in range(NMOM):
            o_m[0, q, L] = smom[q, i]
            o_m[1, q, L] = qmom[q, i]
        for k in range(nks):
            o_k[0, k, L] = um[k, i]
            o_k[1, k, L] = vm[k, i]
    for L in range(LM + 1):
        o_l1[0, L] = lhp[L + 1]
        o_l1[1, L] = precnvl[L + 1]


@njit(parallel=True, cache=True, error_model='numpy')
def mstcnv_batch(pearth, pland, dcl, pl, ple, plk, airm, byam, etal, tl, tvl, sm, qm, qcll, qcil, sdl, wturb, gzl,
                 smom, qmom, ra, um, vm, u0, v0, scal, tune, o_sc, o_l, o_l1, o_m, o_k):
    """Column-last inputs as the device set-up produces them: pearth/pland/dcl (N,), pl.. (LM,N), ple (LM+1,N),
    smom/qmom (9,LM,N), um/vm/u0/v0 (nks,LM,N), ra (nks,N); scal = (lmcm, xmass, bydtsrc, dtsrc, bybr); tune (13,).
    Outputs (written column by column): o_sc (15,N), o_l (37,N,LM), o_l1 (2,N,LM+1), o_m (2,N,9,LM),
    o_k (2,N,nks,LM) -- the shapes the device post-processing consumes after a transpose-free read."""
    n = pl.shape[1]
    nks = ra.shape[0]
    lmcm = int(scal[0])
    for j in prange(n):
        mstcnv_col(pearth[j], pland[j], dcl[j], lmcm, scal[1], scal[2], scal[3], scal[4], pl[:, j], ple[:, j],
                   plk[:, j], airm[:, j], byam[:, j], etal[:, j], tl[:, j], tvl[:, j], sm[:, j], qm[:, j],
                   qcll[:, j], qcil[:, j], sdl[:, j], wturb[:, j], gzl[:, j], smom[:, :, j], qmom[:, :, j], ra[:, j],
                   um[:, :, j], vm[:, :, j], u0[:, :, j], v0[:, :, j], tune, nks, o_sc[:, j], o_l[:, j, :],
                   o_l1[:, j, :], o_m[:, j], o_k[:, j])
