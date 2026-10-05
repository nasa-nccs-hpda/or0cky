"""Full-fidelity port of the LSCOND column physics (CLOUDS2.F90:3211-5286) -- D108 (main layer loop), D109 (CTEI + remainder);
the particle size / optical thickness tail (D107) is in clouds_lscond_size_ff.py and is called by `lscond()`.

One COLUMN per call; explicit sequential L loops exactly as in the Fortran (top-down `CLOUD_FORMATION: do L=LMCLD,1,-1` with
the downward precipitation carry PREBAR/PREICE/LHP, then `CLOUD_TOP_ENTRAINMENT: do L=LMCLD-1,1,-1` with its bounded
`do ITER=1,9` mixing iteration and `exit`).  Plain Python floats (IEEE double), strict left-to-right operation order as in
the Fortran source (ifort -fp-model strict, no FMA).  Arrays are lists indexed 0..LM-1 for layers 1..LM (PREBAR, PREICE,
LHP, PRECNVL have LM+1 entries); QMOM/SMOM are [LM][NMOM] lists; UM/VM [LM][4] (K=1..4 only, as recorded).

Live arm: `use_vmp=1` (rundeck P2SAoM40).  The non-VMP phase/autoconversion arm (random-number ice formation, B-F process)
is NOT ported (`use_vmp=False` raises NotImplementedError): it is dead in this rundeck.  `do_blU00` (=0), all
`CLD_AER_CDNC`/`AIE_DIAG_FIX_MET`/TRACERS blocks and `debug` prints are not compiled / not live and are not ported.

Single-precision literal audit (D54/D91 hazard; un-suffixed REAL literals are REAL(4) and promoted to double AFTER
rounding), every non-exactly-representable one in the live code is reproduced with f4():
    -0.2   (cm0*10.**(-0.2*vdef))                       -> _F02
    238.16 (`TL(L).lt.238.16`, RH1 ice branch)           -> _F23816
    207.83 (`2.583d0-TL(L)/207.83`)                      -> _F20783
    .999999 (`RH(L).lt..999999`)                         -> _F999999
(.001 in WMUI=WMUIX*.001 is handled by taking WMUI from the dump.)  Exactly representable literals (1., 2., 5., .5, 10.,
35., 100., 1000., ERP=2 integer) need nothing; `**5` (integer power, SIGK in CTEI) is a libm pow(x,5.) call (checked against the dumps), `**ERP` is x*x.
pow/exp calls go through the selectable backend of clouds_lscond_size_ff (use_imf).
"""
import math

import numpy as np

import clouds_dq_ff as dq
import clouds_helpers_ff as ch
import clouds_lscond_size_ff as sz
import clouds_massflux_ff as mf

f4 = ch.f4
GRAV = ch.GRAV
RGAS = ch.RGAS
TEENY = ch.TEENY
LHE = ch.LHE
LHS = ch.LHS
LHM = ch.LHM
TF = ch.TF
SHA = ch.SHA
BYSHA = ch.BYSHA
BYGRAV = sz.BYGRAV
BY3 = ch.BY3
TWOPI = ch.TWOPI
RVAP = dq.RVAP
BYMRAT = 1.0 / dq.MRAT
DELTX = BYMRAT - 1.0
SLHE = LHE * BYSHA
WMU, WMUL = 0.25, 0.5
AIRM0 = 100.0
GBYAIRM0 = GRAV / AIRM0
COESIG = 1.0e-3
COEEC = 1000.0
TMAX_ICE = TF - 5.0
TMIN_WATER = TF - 35.0
CM00LIQ = 1.0e-4
CM00ICE = 3.0e-4
_F02, _F23816, _F20783, _F999999 = f4(0.2), f4(238.16), f4(207.83), f4(0.999999)
NMOM = 9
ZMOMS = (2, 5, 8, 7)          # MZ, MZZ, MYZ, MZX (0-based)
XYMOMS = (0, 1, 3, 6, 4)      # MX, MY, MXX, MXY, MYY (0-based)

# ------------------------------------------------------------------------------------------ scalar helpers
_A, _B, _C = dq._A, dq._B, dq._C


def qsat(tm, lh, pr):
    return _A * sz.ex(lh * (_B - _C / max(130.0, tm))) / pr


def dqsatdt(tm, lh):
    return lh * _C / (tm * tm)


def _dq_adjust(sm, qm, plk, mass, lhx, pl, sign):
    slh = lhx * BYSHA
    qmt = qm
    tp = sm * plk / mass
    dqsum = 0.0
    for _ in range(dq.NITER):
        qst = qsat(tp, lhx, pl)
        d = (qmt - mass * qst) / (1.0 + slh * qst * dqsatdt(tp, lhx))
        tp = tp + slh * d / mass
        qmt = qmt - d
        dqsum = dqsum + sign * d
    return dqsum


def get_dq_cond(sm, qm, plk, mass, lhx, pl):
    """-> (dqsum, fcond).  Imported helper (numpy) in libm mode; scalar copy (same algorithm) with libimf exp in imf mode."""
    if sz._IMF["on"]:
        if not qm > 0.0:
            return 0.0, 0.0
        d = _dq_adjust(sm, qm, plk, mass, lhx, pl, +1.0)
        d = max(0.0, min(d, qm))
        return d, d / qm
    a, b = dq.get_dq_cond(sm, qm, plk, mass, lhx, pl)
    return float(a), float(b)


def get_dq_evap(sm, qm, plk, mass, lhx, pl, cond):
    if sz._IMF["on"]:
        if not cond > 0.0:
            return 0.0, 0.0
        d = _dq_adjust(sm, qm, plk, mass, lhx, pl, -1.0)
        d = max(0.0, min(d, cond))
        return d, d / cond
    a, b = dq.get_dq_evap(sm, qm, plk, mass, lhx, pl, cond)
    return float(a), float(b)


def _pow5(x):
    """`**5` of the CTEI SIGK formula.  ifort -O2 -fp-model strict compiles the integer power as a libm `pow(x, 5.)` call: the
    real CTEI dumps are matched bit-for-bit only with pow (x*x*x*x*x or ((x*x)*(x*x))*x differ in ~1 of 100 mixed layers)."""
    return sz.pw(x, 5.0)


def _clear_from_rh(rh, rh00, c):
    """The recurring `if(RH.le.1.) then ... CLEARA=DSQRT(...)/1.; CLEARA>1 -> 1; RH>1 -> 0` sequence."""
    if rh <= 1.0:
        if rh00 < 1.0:
            c = math.sqrt((1.0 - rh) / ((1.0 - rh00) + TEENY))
        else:
            c = 1.0
    if c > 1.0:
        c = 1.0
    if rh > 1.0:
        c = 0.0
    return c


def ctmix(rm, rmom, fmair, fmix, frat):
    """QUSDEF.f CTMIX on layer pair (rm = [lower, upper], rmom = [[NMOM],[NMOM]]) in place."""
    rtemp = rm[0] * (1.0 - fmix) + frat * rm[1]
    rm[1] = rm[1] * (1.0 - frat) + fmix * rm[0]
    rm[0] = rtemp
    for m in XYMOMS:
        rtemp = rmom[0][m] * (1.0 - fmix) + frat * rmom[1][m]
        rmom[1][m] = rmom[1][m] * (1.0 - frat) + fmix * rmom[0][m]
        rmom[0][m] = rtemp
    for m in ZMOMS:
        rmom[0][m] = rmom[0][m] * (1.0 - fmair)
        rmom[1][m] = rmom[1][m] * (1.0 - fmair)


def new_counters():
    return {k: 0 for k in (
        "layers", "lhx_ice_forced", "lhx_water", "lhp_ice_tl_lt_tf", "phase_le_to_ls", "phase_ls_to_le", "oldlat_le_ls",
        "oldlat_ls_le", "qcx_pos", "wtliq_1", "wtliq_0", "wtliq_interp", "wtliq_det_ice", "vdef_cm0", "tem_cap10",
        "cm_cap", "form_clouds", "no_form", "er_clip_max", "er_clip_zero", "er_qcl", "er_ice_precip", "er_rh",
        "cleara_pos", "wtem_wconst", "drhdt_zero", "qcl_neg", "qnew_neg", "ierr", "fqtow_pos", "ls_branch_clear",
        "rhw_branch", "rh1_lt238", "cond_rh1", "cond_dqsum_pos", "rainout_liq", "rainout_ice", "rainout_melt",
        "prebar_er_eq_ermax", "hphase_melt_icep", "hphase_freeze_liq", "hphase_lhp_ne_lhx", "lhp_zero",
        "preice_set", "qcx_teeny_zero", "no_form_evap_liq", "no_form_evap_ice", "no_form_evap_zero_cond",
        "ctei_visits", "ctei_cycle_qcl", "ctei_cycle_clear", "ctei_cycle_ckr_gt_ckm", "ctei_cycle_ck_lt_ckr",
        "ctei_cycle_fpmax", "ctei_cycle_dse", "ctei_mixed", "ctei_iter_exit", "ctei_iter_exhaust", "ctei_fsslrat_ne1",
        "ctei_fmass_clip", "ctei_ckij_set", "ctei_iters_total", "ctei_lhx_ice")}


def lscond_main(S, P, cnt=None, mut=None):
    """Initialisation + `CLOUD_FORMATION` main layer loop + PRCPSS.  `S` (dict of lists, modified in place): the module state
    tl ql th rh qclx qcix svlhxl cldsavl cldssl taussl tausslip csizel csizelip cldsal cldsv1 dqlsc sm qm rh1 wmpr qlss qiss sshr
    dctei lhp(LM+1) prebar1(LM+1) qmom smom um vm and the inputs qcll qcil svlatl svlat1 svwmxl sdl vsubl fssl ttoldl aq dpdt pl
    plk airm byam u00l pdsigl00 taumcl precnvl(LM+1).  `P`: scalars pearth dcl lmcld kmax wconst wmui cmx u00a rtemp scdncw scdnci
    bydtsrc dtsrc use_vmp.  Returns the dict W of local arrays/scalars at the end of the loop (cleara rhf rh00 er ec prep
    qheatl qheati qheat prebar preice sq ath prcpss hcndss ierr lerr wmerr ckij)."""
    if cnt is None:
        cnt = new_counters()
    if not P["use_vmp"]:
        raise NotImplementedError("non-VMP arm of LSCOND is not live in P2SAoM40 and is not ported")
    LM = len(S["tl"])
    dtsrc, bydtsrc = P["dtsrc"], P["bydtsrc"]
    lmcld, dcl = int(P["lmcld"]), int(P["dcl"])
    pearth, wconst, wmui, cmx = P["pearth"], P["wconst"], P["wmui"], P["cmx"]
    u00a = P["u00a"]
    scdncw, scdnci, rimax, rcldlx, rcldix = P["scdncw"], P["scdnci"], P["rimax"], P["rcldlx"], P["rcldix"]
    z = lambda n: [0.0] * n  # noqa: E731
    er, ec, prep, qheat, qheatl, qheati = z(LM), z(LM), z(LM), z(LM), z(LM), z(LM)
    prebar, preice = z(LM + 1), z(LM + 1)
    lhp = z(LM + 1)
    S["cldssl"] = z(LM)
    S["taussl"] = z(LM)
    S["wmpr"] = z(LM)
    S["prebar1"] = z(LM + 1)
    S["rh1"] = z(LM)
    S["qlss"] = z(LM)
    S["qiss"] = z(LM)
    qcinew = 0.0
    qclnew = 0.0
    cleara = z(LM)
    rhf, rh00, qsatl, sq, ath = z(LM), z(LM), z(LM), z(LM), z(LM)
    for i in range(lmcld):
        cleara[i] = 1.0 - S["cldsavl"][i]
        if S["qclx"][i] + S["qcix"][i] <= 0.0:
            cleara[i] = 1.0
    S["sshr"] = z(LM)
    S["dctei"] = z(LM)
    hcndss = 0.0
    ierr, lerr, wmerr = 0, 0, 0.0
    tl, ql, th, rh = S["tl"], S["ql"], S["th"], S["rh"]
    qclx, qcix, svlhxl = S["qclx"], S["qcix"], S["svlhxl"]
    fssl, airm, byam, pl, plk = S["fssl"], S["airm"], S["byam"], S["pl"], S["plk"]
    for L in range(lmcld, 0, -1):
        i = L - 1
        ip = i + 1
        cnt["layers"] += 1
        told = tl[i]
        qold = ql[i]
        oldlhx = svlhxl[i]
        oldlat = S["svlatl"][i]
        temp = 100.0 * RGAS * tl[i] / (pl[i] * GRAV)
        if L == 1:
            vvel = -S["sdl"][ip] * temp
        elif L == LM:
            vvel = -S["sdl"][i] * temp
        else:
            vvel = -0.5 * (S["sdl"][i] + S["sdl"][ip]) * temp
        vdef = vvel - S["vsubl"][i]
        fcld = (1.0 - cleara[i]) * fssl[i] + TEENY
        rh00[i] = u00a
        if pl[i] < pl[dcl - 1]:
            rh00[i] = rh00[i] / (rh00[i] + (1.0 - rh00[i]) * S["pdsigl00"][i] / 35.0)
        if S["u00l"][i] > rh00[i]:
            rh00[i] = S["u00l"][i]
        if rh00[i] < 0.0:
            rh00[i] = 0.0
        if rh00[i] > 1.0:
            rh00[i] = 1.0
        rhf[i] = rh00[i] + (1.0 - cleara[i]) * (1.0 - rh00[i])
        # ---- phase (use_vmp)
        if tl[i] <= TMIN_WATER:
            lhx = LHS
            lhp[i] = LHS
            cnt["lhx_ice_forced"] += 1
        else:
            lhx = LHE
            lhp[i] = LHE
            cnt["lhx_water"] += 1
            if tl[i] < TF:
                lhp[i] = LHS
                cnt["lhp_ice_tl_lt_tf"] += 1
        qsatl[i] = qsat(tl[i], lhx, pl[i])
        S["rh1"][i] = ql[i] / qsatl[i]
        if lhx == LHS and qclx[i] + qcix[i] <= 0.0:
            qsate = qsat(tl[i], LHE, pl[i])
            rhw = (2.583 - tl[i] / _F20783) * (qsat(tl[i], LHS, pl[i]) / qsate)
            cnt["rhw_branch"] += 1
            if tl[i] < _F23816:
                S["rh1"][i] = ql[i] / (qsate * rhw)
                cnt["rh1_lt238"] += 1
        hchang = 0.0
        if oldlhx == LHE and lhx == LHS:
            hchang = S["qcll"][i] * LHM
            qcix[i] = qcix[i] + S["qcll"][i]
            qclx[i] = qclx[i] - S["qcll"][i]
            cnt["phase_le_to_ls"] += 1
        if oldlhx == LHS and lhx == LHE:
            hchang = -S["qcil"][i] * LHM
            qclx[i] = qclx[i] + S["qcil"][i]
            qcix[i] = qcix[i] - S["qcil"][i]
            cnt["phase_ls_to_le"] += 1
        if oldlat == LHE and lhx == LHS:
            hchang = hchang + S["svwmxl"][i] * LHM
            qcix[i] = qcix[i] + S["svwmxl"][i]
            qclx[i] = qclx[i] - S["svwmxl"][i]
            cnt["oldlat_le_ls"] += 1
        if oldlat == LHS and lhx == LHE:
            hchang = hchang - S["svwmxl"][i] * LHM
            qclx[i] = qclx[i] + S["svwmxl"][i]
            qcix[i] = qcix[i] - S["svwmxl"][i]
            cnt["oldlat_ls_le"] += 1
        if oldlhx == LHE and lhx == LHS:
            qcix[i] = S["qcll"][i] + S["svwmxl"][i]
            qclx[i] = 0.0
        elif oldlhx == LHS and lhx == LHE:
            qclx[i] = S["qcil"][i] + S["svwmxl"][i]
            qcix[i] = 0.0
        elif lhx == LHS:
            qcix[i] = S["qcil"][i] + S["svwmxl"][i]
            qclx[i] = 0.0
        elif lhx == LHE:
            qclx[i] = S["qcll"][i] + S["svwmxl"][i]
            qcix[i] = 0.0
        svlhxl[i] = lhx
        tl[i] = tl[i] + hchang / (SHA * fssl[i] + TEENY)
        th[i] = tl[i] / plk[i]
        rhi = ql[i] / qsat(tl[i], LHS, pl[i])
        if lhp[ip] == LHS and tl[i] < TF + dtsrc * LHM * preice[ip] * GRAV * byam[i] * BYSHA:
            lhp[i] = LHS
        # ---- autoconversion
        qcx = qclx[i] + qcix[i]
        if qcx > 0.0:
            cnt["qcx_pos"] += 1
            rho = 1.0e5 * pl[i] / (RGAS * tl[i])
            if tl[i] > TMAX_ICE:
                wtliq = 1.0
                cnt["wtliq_1"] += 1
            elif tl[i] < TMIN_WATER:
                wtliq = 0.0
                cnt["wtliq_0"] += 1
            else:
                wtliq = (tl[i] - TMIN_WATER) / (TMAX_ICE - TMIN_WATER)
                cnt["wtliq_interp"] += 1
            if oldlat == LHS and S["svwmxl"][i] > 0.0:
                wtliq = 0.0
                cnt["wtliq_det_ice"] += 1
            tem = wconst * wtliq + wmui * (1.0 - wtliq)
            cm0 = CM00LIQ * wtliq + CM00ICE * (1.0 - wtliq)
            if vdef > 0.0 and rho * qcx < 10.0:
                cm0 = cm0 * sz.pw(10.0, -_F02 * vdef)
                cnt["vdef_cm0"] += 1
            tem = rho * qcx / (tem * fcld + TEENY)
            tem = tem * tem
            if tem > 10.0:
                tem = 10.0
                cnt["tem_cap10"] += 1
            cm1 = cm0
            cm = cm1 * (1.0 - 1.0 / sz.ex(tem * tem)) + 100.0 * (prebar[ip] + S["precnvl"][ip] * bydtsrc)
            cm = cm * cmx
            if cm > bydtsrc:
                cm = bydtsrc
                cnt["cm_cap"] += 1
            prep[i] = qcx * cm
        else:
            cm = 0.0
        # ---- form clouds?
        if S["rh1"][i] < rh00[i]:
            form = False
        else:
            sq[i] = lhx * qsatl[i] * dqsatdt(tl[i], lhx) * BYSHA
            tem = -lhx * S["dpdt"][i] / pl[i]
            ath[i] = (th[i] - S["ttoldl"][i]) * bydtsrc
            qconv = lhx * S["aq"][i] - rh[i] * sq[i] * SHA * plk[i] * ath[i] - tem * qsatl[i] * rh[i]
            if lhx == LHE:
                form = (qconv > 0.0 or qclx[i] > 0.0)
            else:
                form = (qconv > 0.0 or qcix[i] > 0.0)
        ermax = lhx * prebar[ip] * GRAV * byam[i]
        if form:
            cnt["form_clouds"] += 1
            rhn = min(rh[i], rhf[i])
            if lhx == LHE:
                if qclx[i] > 0.0:
                    er[i] = (1.0 - rhn) * (1.0 - rhn) * lhx * prebar[ip] * GBYAIRM0
                    cnt["er_qcl"] += 1
                else:
                    if preice[ip] > 0.0 and tl[i] < TF:
                        er[i] = (1.0 - rhi) * (1.0 - rhi) * lhx * prebar[ip] * GBYAIRM0
                        cnt["er_ice_precip"] += 1
                    else:
                        er[i] = (1.0 - rh[i]) * (1.0 - rh[i]) * lhx * prebar[ip] * GBYAIRM0
                        cnt["er_rh"] += 1
            if lhx == LHS:
                if qcix[i] > 0.0:
                    er[i] = (1.0 - rhn) * (1.0 - rhn) * lhx * prebar[ip] * GBYAIRM0
                    cnt["er_qcl"] += 1
                else:
                    if preice[ip] > 0.0 and tl[i] < TF:
                        er[i] = (1.0 - rhi) * (1.0 - rhi) * lhx * prebar[ip] * GBYAIRM0
                        cnt["er_ice_precip"] += 1
                    else:
                        er[i] = (1.0 - rh[i]) * (1.0 - rh[i]) * lhx * prebar[ip] * GBYAIRM0
                        cnt["er_rh"] += 1
            e0 = er[i]
            er[i] = max(0.0, min(er[i], ermax))
            if er[i] != e0:
                cnt["er_clip_max" if e0 > ermax else "er_clip_zero"] += 1
            if cleara[i] > 0.0:
                cnt["cleara_pos"] += 1
                if lhx == LHE:
                    wtem = 1.0e5 * qclx[i] * pl[i] / (fcld * tl[i] * RGAS + TEENY)
                else:
                    wtem = 1.0e5 * qcix[i] * pl[i] / (fcld * tl[i] * RGAS + TEENY)
                if lhx == LHE and qclx[i] / fcld >= wconst * 1.0e-3:
                    wtem = 1.0e2 * wconst * pl[i] / (tl[i] * RGAS)
                    cnt["wtem_wconst"] += 1
                if wtem < 1.0e-10:
                    wtem = 1.0e-10
                if lhx == LHE:
                    rcld = rcldlx * 1.0e-6 * 100.0 * sz.pw(wtem / (2.0 * BY3 * TWOPI * scdncw), BY3)
                else:
                    rcld = rcldix * 100.0e-6 * sz.pw(wtem / (2.0 * BY3 * TWOPI * scdnci), BY3)
                    rcld = min(rcld, rimax)
                ck1 = 1000.0 * lhx * lhx / (2.4e-2 * RVAP * tl[i] * tl[i])
                ck2 = 1000.0 * RGAS * tl[i] / (2.4e-3 * qsatl[i] * pl[i])
                tevap = COEEC * (ck1 + ck2) * rcld * rcld
                if lhx == LHE:
                    wmx1 = qclx[i] - prep[i] * dtsrc
                else:
                    wmx1 = qcix[i] - prep[i] * dtsrc
                S["wmpr"][i] = prep[i] * dtsrc
                ecrate = (1.0 - rhf[i]) / (tevap * fcld + TEENY)
                if ecrate > bydtsrc:
                    ecrate = bydtsrc
                ec[i] = wmx1 * ecrate * lhx
            if lhx == LHE:
                drhdt = 2.0 * cleara[i] * cleara[i] * (1.0 - rh00[i]) * (qconv + er[i]) / lhx / (
                    qclx[i] / (fcld + TEENY) + 2.0 * cleara[i] * qsatl[i] * (1.0 - rh00[i]) + TEENY)
                if er[i] == 0.0 and qclx[i] <= 0.0:
                    drhdt = 0.0
                    cnt["drhdt_zero"] += 1
                qheatl[i] = fssl[i] * (qconv - lhx * drhdt * qsatl[i]) / (1.0 + rh[i] * sq[i])
                dwdt = qheatl[i] / lhx - prep[i] + cleara[i] * fssl[i] * er[i] / lhx
                qclnew = qclx[i] + dwdt * dtsrc
                if qclnew < 0.0:
                    qclnew = 0.0
                    qheatl[i] = (-qclx[i] * bydtsrc + prep[i]) * lhx - cleara[i] * fssl[i] * er[i]
                    cnt["qcl_neg"] += 1
            if lhx == LHS:
                drhdt = 2.0 * cleara[i] * cleara[i] * (1.0 - rh00[i]) * (qconv + er[i]) / lhx / (
                    qcix[i] / (fcld + TEENY) + 2.0 * cleara[i] * qsatl[i] * (1.0 - rh00[i]) + TEENY)
                if er[i] == 0.0 and qcix[i] <= 0.0:
                    drhdt = 0.0
                    cnt["drhdt_zero"] += 1
                qheati[i] = fssl[i] * (qconv - lhx * drhdt * qsatl[i]) / (1.0 + rh[i] * sq[i])
                dwdt = qheati[i] / lhx - prep[i] + cleara[i] * fssl[i] * er[i] / lhx
                qcinew = qcix[i] + dwdt * dtsrc
                if qcinew < 0.0:
                    qcinew = 0.0
                    qheati[i] = (-qcix[i] * bydtsrc + prep[i]) * lhx - cleara[i] * fssl[i] * er[i]
                    cnt["qcl_neg"] += 1
        else:
            cnt["no_form"] += 1
            qheatl[i] = 0.0
            qheati[i] = 0.0
            if lhx == LHE and qclx[i] > 0.0:
                dqsum, _ = get_dq_evap(tl[i] * rh00[i] / plk[i], ql[i] * rh00[i], plk[i], rh00[i], lhx, pl[i],
                                       qclx[i] / (fssl[i] * rh00[i]))
                dwdt = dqsum * rh00[i] * fssl[i]
                qheatl[i] = -dwdt * lhx * bydtsrc
                prep[i] = max(0.0, (qclx[i] - dwdt) * bydtsrc)
                S["wmpr"][i] = prep[i] * dtsrc
                cnt["no_form_evap_liq"] += 1
            if lhx == LHS and qcix[i] > 0.0:
                dqsum, _ = get_dq_evap(tl[i] * rh00[i] / plk[i], ql[i] * rh00[i], plk[i], rh00[i], lhx, pl[i],
                                       qcix[i] / (fssl[i] * rh00[i]))
                dwdt = dqsum * rh00[i] * fssl[i]
                qheati[i] = -dwdt * lhx * bydtsrc
                prep[i] = max(0.0, (qcix[i] - dwdt) * bydtsrc)
                S["wmpr"][i] = prep[i] * dtsrc
                cnt["no_form_evap_ice"] += 1
            er[i] = (1.0 - rh[i]) * (1.0 - rh[i]) * lhx * prebar[ip] * GBYAIRM0
            if preice[ip] > 0.0 and tl[i] < TF:
                er[i] = (1.0 - rhi) * (1.0 - rhi) * lhx * prebar[ip] * GBYAIRM0
            e0 = er[i]
            er[i] = max(0.0, min(er[i], ermax))
            if er[i] != e0:
                cnt["er_clip_max" if e0 > ermax else "er_clip_zero"] += 1
            if lhx == LHE:
                qheatl[i] = qheatl[i] - cleara[i] * fssl[i] * er[i]
                qclnew = 0.0
            else:
                qheati[i] = qheati[i] - cleara[i] * fssl[i] * er[i]
                qcinew = 0.0
        # ---- phase of precipitation, precip carry
        hphase = 0.0
        if lhp[ip] == LHS and lhp[i] == LHE and preice[ip] > 0.0:
            hphase = hphase + LHM * preice[ip] * GRAV * byam[i]
            preice[ip] = 0.0
            cnt["hphase_melt_icep"] += 1
        if lhp[ip] == LHE and lhp[i] == LHS and prebar[ip] > 0.0:
            hphase = hphase - LHM * prebar[ip] * GRAV * byam[i]
            cnt["hphase_freeze_liq"] += 1
        if lhp[i] != lhx:
            hphase = hphase + (er[i] * cleara[i] * fssl[i] / lhx - prep[i]) * LHM
            cnt["hphase_lhp_ne_lhx"] += 1
        if er[i] == ermax:
            prebar[i] = prebar[ip] * (1.0 - cleara[i] * fssl[i]) + airm[i] * prep[i] * BYGRAV
            cnt["prebar_er_eq_ermax"] += 1
        else:
            prebar[i] = max(0.0, prebar[ip] + airm[i] * (prep[i] - er[i] * cleara[i] * fssl[i] / lhx) * BYGRAV)
        if lhx == LHE:
            qnew = ql[i] - dtsrc * qheatl[i] / (lhx * fssl[i] + TEENY)
            if qnew < 0.0:
                qnew = 0.0
                qheatl[i] = ql[i] * lhx * bydtsrc * fssl[i]
                dwdt1 = qheatl[i] / lhx - prep[i] + cleara[i] * fssl[i] * er[i] / lhx
                qclnew = qclx[i] + dwdt1 * dtsrc
                cnt["qnew_neg"] += 1
                if qclnew < 0.0:
                    ierr, lerr, wmerr = 1, L, qclnew
                    qclnew = 0.0
                    cnt["ierr"] += 1
        else:
            qnew = ql[i] - dtsrc * qheati[i] / (lhx * fssl[i] + TEENY)
            if qnew < 0.0:
                qnew = 0.0
                qheati[i] = ql[i] * lhx * bydtsrc * fssl[i]
                dwdt1 = qheati[i] / lhx - prep[i] + cleara[i] * fssl[i] * er[i] / lhx
                qcinew = qcix[i] + dwdt1 * dtsrc
                cnt["qnew_neg"] += 1
                if qcinew < 0.0:
                    ierr, lerr, wmerr = 1, L, qcinew
                    qcinew = 0.0
                    cnt["ierr"] += 1
        fqtow = 0.0
        if fssl[i] > 0.0:
            qheat[i] = qheatl[i] if lhx == LHE else qheati[i]
            if qheat[i] + cleara[i] * fssl[i] * er[i] > 0.0:
                if lhx * ql[i] + dtsrc * cleara[i] * er[i] > 0.0:
                    fqtow = (qheat[i] + cleara[i] * fssl[i] * er[i]) * dtsrc / (
                        (lhx * ql[i] + dtsrc * cleara[i] * er[i]) * fssl[i])
                    cnt["fqtow_pos"] += 1
        ql[i] = qnew
        S["qmom"][i] = [v * (1.0 - fqtow) for v in S["qmom"][i]]
        if lhx == LHE:
            qclx[i] = qclnew
        else:
            qcix[i] = qcinew
        if lhx == LHE:
            tl[i] = tl[i] + dtsrc * (qheatl[i] - hphase) / (SHA * fssl[i] + TEENY)
        else:
            tl[i] = tl[i] + dtsrc * (qheati[i] - hphase) / (SHA * fssl[i] + TEENY)
        th[i] = tl[i] / plk[i]
        qsatc = qsat(tl[i], lhx, pl[i])
        rh[i] = ql[i] / qsatc
        S["rh1"][i] = ql[i] / qsatc
        if lhx == LHS:
            cnt["ls_branch_clear"] += 1
            cleara[i] = _clear_from_rh(rh[i], rh00[i], cleara[i])
            if qclx[i] + qcix[i] <= 0.0:
                cleara[i] = 1.0
            qf = (ql[i] - qsatc * (1.0 - cleara[i])) / (cleara[i] + TEENY)
            qsate = qsat(tl[i], LHE, pl[i])
            rhw = (2.583 - tl[i] / _F20783) * (qsat(tl[i], LHS, pl[i]) / qsate)
            if tl[i] < _F23816 and qclx[i] + qcix[i] <= 0.0:
                S["rh1"][i] = qf / (qsate * rhw)
        if S["rh1"][i] > 1.0:
            cnt["cond_rh1"] += 1
            slh = lhx * BYSHA
            dqsum, fcond = get_dq_cond(tl[i], ql[i], 1.0, 1.0, lhx, pl[i])
            if dqsum > 0.0:
                cnt["cond_dqsum_pos"] += 1
                tl[i] = tl[i] + slh * dqsum
                ql[i] = ql[i] - dqsum
                if lhx == LHE:
                    qclx[i] = qclx[i] + dqsum * fssl[i]
                else:
                    qcix[i] = qcix[i] + dqsum * fssl[i]
                S["qmom"][i] = [v * (1.0 - fcond) for v in S["qmom"][i]]
            rh[i] = ql[i] / qsat(tl[i], lhx, pl[i])
            th[i] = tl[i] / plk[i]
        cleara[i] = _clear_from_rh(rh[i], rh00[i], cleara[i])
        if lhx == LHE:
            if qclx[i] <= 0.0:
                cleara[i] = 1.0
        else:
            if qcix[i] <= 0.0:
                cleara[i] = 1.0
        if cleara[i] < 0.0:
            cleara[i] = 0.0
        rhf[i] = rh00[i] + (1.0 - cleara[i]) * (1.0 - rh00[i])
        if lhx == LHE:
            if rh[i] <= rhf[i] and rh[i] < _F999999 and qclx[i] > 0.0:
                cnt["rainout_liq"] += 1
                prebar[i] = prebar[i] + qclx[i] * airm[i] * BYGRAV * bydtsrc
                if lhp[i] == LHS and lhx == LHE:
                    hchang = qclx[i] * LHM
                    tl[i] = tl[i] + hchang / (SHA * fssl[i] + TEENY)
                    th[i] = tl[i] / plk[i]
                    cnt["rainout_melt"] += 1
                qclx[i] = 0.0
        else:
            if rh[i] <= rhf[i] and rh[i] < _F999999 and qcix[i] > 0.0:
                cnt["rainout_ice"] += 1
                prebar[i] = prebar[i] + qcix[i] * airm[i] * BYGRAV * bydtsrc
                qcix[i] = 0.0
        S["prebar1"][i] = prebar[i]
        preice[i] = 0.0
        if prebar[i] > 0.0 and lhp[i] == LHS:
            preice[i] = prebar[i]
            cnt["preice_set"] += 1
        if prebar[i] <= 0.0:
            lhp[i] = 0.0
            cnt["lhp_zero"] += 1
        cleara[i] = _clear_from_rh(rh[i], rh00[i], cleara[i])
        if lhx == LHE:
            if qclx[i] <= TEENY:
                qclx[i] = 0.0
                cnt["qcx_teeny_zero"] += 1
            if qclx[i] <= 0.0:
                cleara[i] = 1.0
        else:
            if qcix[i] <= TEENY:
                qcix[i] = 0.0
                cnt["qcx_teeny_zero"] += 1
            if qcix[i] <= 0.0:
                cleara[i] = 1.0
        if cleara[i] < 0.0:
            cleara[i] = 0.0
        S["cldssl"][i] = fssl[i] * (1.0 - cleara[i])
        S["cldsavl"][i] = 1.0 - cleara[i]
        hcndss = hcndss + fssl[i] * (tl[i] - told) * airm[i]
        S["sshr"][i] = S["sshr"][i] + fssl[i] * (tl[i] - told) * airm[i]
        S["dqlsc"][i] = S["dqlsc"][i] + fssl[i] * (ql[i] - qold)
    prcpss = max(0.0, prebar[0] * GRAV * dtsrc)
    S["lhp"] = lhp
    W = dict(cleara=cleara, rhf=rhf, rh00=rh00, er=er, ec=ec, prep=prep, qheatl=qheatl, qheati=qheati, qheat=qheat,
             prebar=prebar, preice=preice, prcpss=prcpss, hcndss=hcndss, ierr=ierr, lerr=lerr, wmerr=wmerr, ckij=1.0)
    return W


def lscond_ctei(S, W, P, cnt=None, mut=None):
    """`CLOUD_TOP_ENTRAINMENT: do L=LMCLD-1,1,-1` (S, W modified in place)."""
    if cnt is None:
        cnt = new_counters()
    LM = len(S["tl"])
    lmcld = int(P["lmcld"])
    kmax = min(int(P["kmax"]), 4)
    dtsrc = P["dtsrc"]
    tl, ql, th, rh = S["tl"], S["ql"], S["th"], S["rh"]
    qclx, qcix, svlhxl = S["qclx"], S["qcix"], S["svlhxl"]
    fssl, airm, byam, pl, plk = S["fssl"], S["airm"], S["byam"], S["pl"], S["plk"]
    cleara, rh00 = W["cleara"], W["rh00"]
    sm, qm = S["sm"], S["qm"]
    wmxm = [0.0] * (LM + 1)
    ra = P["ra"]
    ckij = 1.0
    hcndss = W["hcndss"]
    for L in range(lmcld - 1, 0, -1):
        i = L - 1
        ip = i + 1
        cnt["ctei_visits"] += 1
        lhx = svlhxl[i]
        sm[i] = th[i] * airm[i]
        qm[i] = ql[i] * airm[i]
        wmxm[i] = qclx[i] * airm[i] if lhx == LHE else qcix[i] * airm[i]
        sm[ip] = th[ip] * airm[ip]
        qm[ip] = ql[ip] * airm[ip]
        wmxm[ip] = qclx[ip] * airm[i] if lhx == LHE else qcix[ip] * airm[i]
        if qclx[ip] + qcix[ip] > TEENY:
            cnt["ctei_cycle_qcl"] += 1
            continue
        told, toldu, qold, qoldu = tl[i], tl[ip], ql[i], ql[ip]
        fcld = (1.0 - cleara[i]) * fssl[i] + TEENY
        if cleara[i] == 1.0 or (cleara[i] < 1.0 and cleara[ip] < 1.0):
            cnt["ctei_cycle_clear"] += 1
            continue
        sedge = mf.thbar(th[ip], th[i])
        sedge = float(sedge)
        dse = (th[ip] - sedge) * plk[ip] + (sedge - th[i]) * plk[i] + SLHE * (ql[ip] - ql[i])
        if lhx == LHE:
            dwm = ql[ip] - ql[i] + (qclx[ip] - qclx[i]) / fcld
        else:
            dwm = ql[ip] - ql[i] + (qcix[ip] - qcix[i]) / fcld
            cnt["ctei_lhx_ice"] += 1
        dqsdt = dqsatdt(tl[i], LHE) * ql[i] / (rh[i] + 1.0e-30)
        beta = (1.0 + BYMRAT * tl[i] * dqsdt) / (1.0 + SLHE * dqsdt)
        ckm = (1.0 + SLHE * dqsdt) * (1.0 + (1.0 - DELTX) * tl[i] / SLHE) / (
            2.0 + (1.0 + BYMRAT * tl[i] / SLHE) * SLHE * dqsdt)
        ckr = tl[i] / (beta * SLHE)
        ck = dse / (SLHE * dwm + TEENY)
        sigk = 0.0
        if ckr > ckm:
            cnt["ctei_cycle_ckr_gt_ckm"] += 1
            continue
        if ck > ckr:
            sigk = COESIG * _pow5((ck - ckr) / ((ckm - ckr) + TEENY))
        expst = sz.ex(-sigk * dtsrc)
        if L <= 1:
            ckij = expst
            cnt["ctei_ckij_set"] += 1
        dsec = dwm * tl[i] / beta
        if ck < ckr:
            cnt["ctei_cycle_ck_lt_ckr"] += 1
            continue
        fpmax = min(1.0, 1.0 - expst)
        if fpmax <= 0.0:
            cnt["ctei_cycle_fpmax"] += 1
            continue
        if dse >= dsec:
            cnt["ctei_cycle_dse"] += 1
            continue
        cnt["ctei_mixed"] += 1
        airmr = (airm[ip] + airm[i]) * byam[ip] * byam[i]
        smo1, qmo1, wmo1 = sm[i], qm[i], wmxm[i]
        smo2, qmo2, wmo2 = sm[ip], qm[ip], wmxm[ip]
        smo12 = smo1 * plk[i] + smo2 * plk[ip]
        umo1 = [S["um"][i][k] for k in range(kmax)]
        vmo1 = [S["vm"][i][k] for k in range(kmax)]
        umo2 = [S["um"][ip][k] for k in range(kmax)]
        vmo2 = [S["vm"][ip][k] for k in range(kmax)]
        fplume = fpmax * fssl[i]
        dfx = fplume
        it = 0
        exited = False
        for it in range(1, 10):
            dfx = dfx * 0.5
            fmix = fplume * fcld
            fmass = fmix * airm[i]
            fmass = min(fmass, (airm[ip] * airm[i]) / (airm[ip] + airm[i]))
            fmix = fmass * byam[i]
            frat = fmass * byam[ip]
            smn1 = smo1 * (1.0 - fmix) + frat * smo2
            qmn1 = qmo1 * (1.0 - fmix) + frat * qmo2
            wmn1 = wmo1 * (1.0 - fmix) + frat * wmo2
            smn2 = smo2 * (1.0 - frat) + fmix * smo1
            qmn2 = qmo2 * (1.0 - frat) + fmix * qmo1
            wmn2 = wmo2 * (1.0 - frat) + fmix * wmo1
            tht1 = smn1 * byam[i] / plk[i]
            qlt1 = qmn1 * byam[i]
            tlt1 = tht1 * plk[i]
            lhx = svlhxl[i]
            rht1 = qlt1 / (qsat(tlt1, lhx, pl[i]))
            wmt1 = wmn1 * byam[i]
            tht2 = smn2 * byam[ip] / plk[ip]
            qlt2 = qmn2 * byam[ip]
            wmt2 = wmn2 * byam[ip]
            sedge = float(mf.thbar(tht2, tht1))
            dse = (tht2 - sedge) * plk[ip] + (sedge - tht1) * plk[i] + SLHE * (qlt2 - qlt1)
            dwm = qlt2 - qlt1 + (wmt2 - wmt1) / fcld
            dqsdt = dqsatdt(tlt1, LHE) * qlt1 / (rht1 + 1.0e-30)
            beta = (1.0 + BYMRAT * tlt1 * dqsdt) / (1.0 + SLHE * dqsdt)
            ckm = (1.0 + SLHE * dqsdt) * (1.0 + (1.0 - DELTX) * tlt1 / SLHE) / (
                2.0 + (1.0 + BYMRAT * tlt1 / SLHE) * SLHE * dqsdt)
            dsec = dwm * tlt1 / beta
            dsedif = dse - dsec
            if dsedif > 1.0e-3:
                fplume = fplume - dfx
            if dsedif < -1.0e-3:
                fplume = fplume + dfx
            if abs(dsedif) <= 1.0e-3 or fplume > fpmax * fssl[i]:
                exited = True
                break
        else:
            it = 10
        cnt["ctei_iters_total"] += it
        cnt["ctei_iter_exit" if exited else "ctei_iter_exhaust"] += 1
        if fmass == (airm[ip] * airm[i]) / (airm[ip] + airm[i]):
            cnt["ctei_fmass_clip"] += 1
        smn12 = smn1 * plk[i] + smn2 * plk[ip]
        smn1 = smn1 - (smn12 - smo12) * airm[i] / ((airm[i] + airm[ip]) * plk[i])
        smn2 = smn2 - (smn12 - smo12) * airm[ip] / ((airm[i] + airm[ip]) * plk[ip])
        th[i] = smn1 * byam[i]
        tl[i] = th[i] * plk[i]
        ql[i] = qmn1 * byam[i]
        lhx = svlhxl[i]
        rh[i] = ql[i] / qsat(tl[i], lhx, pl[i])
        if lhx == LHE:
            qclx[i] = wmn1 * byam[i]
        else:
            qcix[i] = wmn1 * byam[i]
        fsslrat = fssl[i] / fssl[ip]
        if fsslrat != 1.0:
            cnt["ctei_fsslrat_ne1"] += 1
            smn2 = sm[ip] + (smn2 - sm[ip]) * fsslrat
            qmn2 = qm[ip] + (qmn2 - qm[ip]) * fsslrat
            smom2_sv = list(S["smom"][ip])
            qmom2_sv = list(S["qmom"][ip])
        smp = [sm[i], sm[ip]]
        smm = [S["smom"][i], S["smom"][ip]]
        ctmix(smp, smm, fmass * airmr, fmix, frat)
        sm[i], sm[ip] = smp
        qmp = [qm[i], qm[ip]]
        qmm = [S["qmom"][i], S["qmom"][ip]]
        ctmix(qmp, qmm, fmass * airmr, fmix, frat)
        qm[i], qm[ip] = qmp
        if fsslrat != 1.0:
            sm[ip] = smn2
            qm[ip] = qmn2
            S["smom"][ip] = [smom2_sv[m] + (S["smom"][ip][m] - smom2_sv[m]) * fsslrat for m in range(NMOM)]
            S["qmom"][ip] = [qmom2_sv[m] + (S["qmom"][ip][m] - qmom2_sv[m]) * fsslrat for m in range(NMOM)]
        th[ip] = smn2 * byam[ip]
        ql[ip] = qmn2 * byam[ip]
        if lhx == LHE:
            qclx[ip] = wmn2 * byam[ip]
        else:
            qcix[ip] = wmn2 * byam[ip]
        for k in range(kmax):
            umn1 = (umo1[k] * (1.0 - fmix) + frat * umo2[k])
            vmn1 = (vmo1[k] * (1.0 - fmix) + frat * vmo2[k])
            umn2 = (umo2[k] * (1.0 - frat) + fmix * umo1[k])
            vmn2 = (vmo2[k] * (1.0 - frat) + fmix * vmo1[k])
            S["um"][i][k] = S["um"][i][k] + (umn1 - umo1[k]) * ra[k]
            S["vm"][i][k] = S["vm"][i][k] + (vmn1 - vmo1[k]) * ra[k]
            S["um"][ip][k] = S["um"][ip][k] + (umn2 - umo2[k]) * ra[k]
            S["vm"][ip][k] = S["vm"][ip][k] + (vmn2 - vmo2[k]) * ra[k]
        if lhx == LHE:
            ql[ip] = ql[ip] + qclx[ip] / (fssl[ip] + TEENY)
        else:
            ql[ip] = ql[ip] + qcix[ip] / (fssl[ip] + TEENY)
        th[ip] = th[ip] - ((svlhxl[ip] - svlhxl[i]) * BYSHA) * wmxm[ip] / airm[ip] / (plk[ip] * fssl[ip] + TEENY)
        if lhx == LHE:
            th[ip] = th[ip] - (lhx * BYSHA) * qclx[ip] / (plk[ip] * fssl[ip] + TEENY)
        else:
            th[ip] = th[ip] - (lhx * BYSHA) * qcix[ip] / (plk[ip] * fssl[ip] + TEENY)
        tl[ip] = th[ip] * plk[ip]
        rh[ip] = ql[ip] / qsat(tl[ip], lhx, pl[ip])
        if lhx == LHE:
            qclx[ip] = 0.0
        else:
            qcix[ip] = 0.0
        cleara[i] = _clear_from_rh(rh[i], rh00[i], cleara[i])
        S["cldssl"][i] = fssl[i] * (1.0 - cleara[i])
        S["cldsavl"][i] = 1.0 - cleara[i]
        tnew, tnewu, qnew, qnewu = tl[i], tl[ip], ql[i], ql[ip]
        hcndss = hcndss + fssl[i] * (tnew - told) * airm[i] + fssl[ip] * (tnewu - toldu) * airm[ip]
        S["sshr"][i] = S["sshr"][i] + fssl[i] * (tnew - told) * airm[i]
        S["sshr"][ip] = S["sshr"][ip] + fssl[ip] * (tnewu - toldu) * airm[ip]
        S["dqlsc"][i] = S["dqlsc"][i] + fssl[i] * (qnew - qold)
        S["dqlsc"][ip] = S["dqlsc"][ip] + fssl[ip] * (qnewu - qoldu)
        S["dctei"][i] = S["dctei"][i] + fssl[i] * (qnew - qold) * airm[i] * lhx * BYSHA
        S["dctei"][ip] = S["dctei"][ip] + fssl[ip] * (qnewu - qoldu) * airm[ip] * lhx * BYSHA
    W["hcndss"] = hcndss
    W["ckij"] = ckij


def lscond_tail(S, W, P, cnt=None):
    """D107 block via clouds_lscond_size_ff.size_tail; updates S, W (wmsum)."""
    a = {k: S[k] for k in ("cldssl", "qclx", "qcix", "svlhxl", "tl", "pl", "airm", "fssl", "wmpr", "taumcl", "taussl",
                           "tausslip", "csizel", "csizelip", "cldsal", "cldsv1", "qlss", "qiss", "lhp")}
    a.update(cleara=W["cleara"], qheatl=W["qheatl"], ec=W["ec"], er=W["er"], prep=W["prep"])
    par = dict(pearth=P["pearth"], lmcld=P["lmcld"], dcl=P["dcl"], ckij=W["ckij"], bybr=P["bybr"], rimax=P["rimax"],
               rwmax=P["rwmax"], rwcldox=P["rwcldox"], rcldlx=P["rcldlx"], rcldix=P["rcldix"])
    s, wmsum, c = sz.size_tail(a, par, use_vmp=True)
    for k in ("cldssl", "qclx", "qcix", "svlhxl", "taussl", "tausslip", "csizel", "csizelip", "cldsal", "cldsv1", "qlss",
              "qiss", "wmpr"):
        S[k] = s[k]
    W["wmsum"] = wmsum
    return c


def lscond(S, P, cnt=None):
    """Whole LSCOND for one column: main loop, CTEI, tail.  -> (S, W, counters)."""
    if cnt is None:
        cnt = new_counters()
    W = lscond_main(S, P, cnt)
    lscond_ctei(S, W, P, cnt)
    lscond_tail(S, W, P, cnt)
    return S, W, cnt
