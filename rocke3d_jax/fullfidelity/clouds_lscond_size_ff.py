"""Full-fidelity port of the LSCOND tail block 'COMPUTE CLOUD PARTICLE SIZE AND OPTICAL THICKNESS' -- D107 (scoping D-C5).

CLOUDS2.F90:4927-5285 (the block from `SNdO = 59.68d0/(RWCLDOX**3)` to the end of LSCOND): the OPTICAL_THICKNESS loop
(`do L=1,LMCLD`, per layer independent given CLDSSL, QCLX/QCIX, WMPR), `CLDSAL = CLDSSL`, and the final
`do L=1,LMCLD` loop (CLDSV1 save, SVLHXL reset, boundary-layer / non-convective cloud-fraction rescaling, QLss/QIss
accumulation).  The `use_vmp` arm is the live one in P2SAoM40 (rundeck `use_vmp=1`); the non-VMP arm of this block
only differs in the TAUSSL/TAUSSLIP update and is also coded (use_vmp=False).  The `#ifdef CLD_AER_CDNC`,
`AIE_DIAG_FIX_MET` and `CACHED_SUBDD` COSP sub-blocks are not compiled in this rundeck and are not ported.

One column per call, explicit sequential L loops exactly as in the Fortran, plain Python floats (IEEE double, strict
left-to-right operation order as in the Fortran source, which is compiled -fp-model strict without FMA).
Single-precision-literal audit (D54 hazard) of every un-suffixed literal in the block: 100.d0, 2.d0, 1.d5, 1d-10,
1.d2, 1.5d3, 25.d0, .3d0, .95d0, 1d0 are double; `5.`, `0.`, `1.`, `2.` (in 2.*BY3), `100.` (TAUSSL>100.) are exactly
representable in REAL(4); `2d+0/3d+0` is the double quotient 2/3.  No literal needs f4().
pow/exp: `**BY3` and `**(2.*BY3)` are libm pow calls, `exp(-(x/.3d0))` an exp call; `use_imf(True)` routes them through
Intel libimf (see intel_libm_ff.py) when the runtime is present.
"""
import ctypes
import math
import os

import clouds_helpers_ff as ch

TEENY = ch.TEENY
LHE = ch.LHE
LHS = ch.LHS
BY3 = ch.BY3
TWOPI = ch.TWOPI
RGAS = ch.RGAS
BYGRAV = 1.0 / ch.GRAV

_IMF = {"lib": None, "on": False}


def _load_imf():
    if _IMF["lib"] is None:
        import intel_libm_ff as il
        lib = None
        if il.available():
            d = os.environ.get("INTEL_LIBIMF_DIR", il._DEFAULT)
            lib = ctypes.CDLL(os.path.join(d, "libimf.so"))
            lib.exp.restype = ctypes.c_double
            lib.exp.argtypes = [ctypes.c_double]
            lib.pow.restype = ctypes.c_double
            lib.pow.argtypes = [ctypes.c_double, ctypes.c_double]
        _IMF["lib"] = lib if lib is not None else False
    return _IMF["lib"]


def imf_available():
    return bool(_load_imf())


def use_imf(flag):
    """Route exp/pow through Intel libimf (True) or the platform libm (False). Returns the mode actually set."""
    _IMF["on"] = bool(flag) and imf_available()
    return _IMF["on"]


def pw(x, y):
    if _IMF["on"]:
        return _IMF["lib"].pow(x, y)
    return x ** y


def ex(x):
    if _IMF["on"]:
        return _IMF["lib"].exp(x)
    return math.exp(x)


def size_tail(a, par, use_vmp=True, lhp=None):
    """Port of the tail block for ONE column.  `a` is a dict of per-layer lists (index 0..LM-1) copied in/out:
    cldssl qclx qcix svlhxl tl pl airm fssl cleara qheatl ec er prep wmpr taumcl taussl tausslip csizel csizelip cldsal
    cldsv1 qlss qiss, plus `lhp` (LM+1).  `par`: pearth lmcld dcl ckij bybr rimax rwmax rwcldox rcldlx rcldix.
    Returns (state dict after the block, wmsum, counters).  Inputs are not modified."""
    s = {k: list(v) for k, v in a.items()}
    lhp = s["lhp"]
    pearth, lmcld, dcl, ckij = par["pearth"], int(par["lmcld"]), int(par["dcl"]), par["ckij"]
    bybr, rimax, rwmax, rwcldox = par["bybr"], par["rimax"], par["rwmax"], par["rwcldox"]
    rcldlx, rcldix = par["rcldlx"], par["rcldix"]
    cnt = dict(liq=0, ice=0, wtem_floor=0, rcld_rwmax=0, vmp_ip_liq=0, csize_cap=0, tau_lhp_eq=0, tau_ip=0,
               tau_ip_cap=0, fcld_teeny=0, tau_cap=0, neg_tau=0, rescale_bl=0, rescale_free=0, svlhx_reset=0,
               skip_taumcl=0, bmax_095=0, ip_fcld_zero=0, lhp_lhe_stop=0)
    sndo = 59.68 / (rwcldox ** 3)
    sndl = 174.0
    sndi = 0.06417127
    scdncw = sndo * (1.0 - pearth) + sndl * pearth
    scdnci = sndi
    wmsum = 0.0
    for L in range(1, lmcld + 1):
        i = L - 1
        fcld = s["cldssl"][i] + TEENY
        lhx = s["svlhxl"][i]
        if lhx == LHE:
            wtem = 1.0e5 * s["qclx"][i] * s["pl"][i] / (fcld * s["tl"][i] * RGAS + TEENY)
        else:
            wtem = 1.0e5 * s["qcix"][i] * s["pl"][i] / (fcld * s["tl"][i] * RGAS + TEENY)
        if wtem < 1.0e-10:
            wtem = 1.0e-10
            cnt["wtem_floor"] += 1
        s["wmpr"][i] = max(s["wmpr"][i], 0.0)
        if lhx == LHE:
            cnt["liq"] += 1
            rcld = rcldlx * 100.0 * pw(wtem / (2.0 * BY3 * TWOPI * scdncw), BY3)
            qheatc = (s["qheatl"][i] + s["fssl"][i] * s["cleara"][i] * (s["ec"][i] + s["er"][i])) / lhx
            if rcld > rwmax and s["prep"][i] > qheatc:
                rcld = rwmax
                cnt["rcld_rwmax"] += 1
            rclde = rcld / bybr
            rclde1 = rclde
            if use_vmp and lhp[i] == LHS and s["wmpr"][i] > 0.0:
                rclde1 = 1.0e5 * s["wmpr"][i] * s["pl"][i] / (fcld * s["tl"][i] * RGAS + TEENY)
                rclde1 = rcldix * 100.0 * pw(rclde1 / (2.0 * BY3 * TWOPI * scdnci), BY3)
                rclde1 = min(rclde1, rimax) / bybr
                s["csizelip"][i] = rclde1
                cnt["vmp_ip_liq"] += 1
        else:
            cnt["ice"] += 1
            rcld = rcldix * 100.0 * pw(wtem / (2.0 * BY3 * TWOPI * scdnci), BY3)
            rcld = min(rcld, rimax)
            rclde = rcld / bybr
            rclde1 = rclde
            if use_vmp and s["cldssl"][i] > 0:
                s["csizelip"][i] = rclde1
        rclde1 = 5.0 * rclde1
        s["csizel"][i] = rclde
        if fcld <= TEENY and s["csizel"][i] > 25.0:
            s["csizel"][i] = 25.0
            cnt["csize_cap"] += 1
        if lhx == LHE:
            tem = s["airm"][i] * s["qclx"][i] * 1.0e2 * BYGRAV
        else:
            tem = s["airm"][i] * s["qcix"][i] * 1.0e2 * BYGRAV
        s["taussl"][i] = 1.5e3 * tem / (fcld * rclde + TEENY)
        tem1 = s["airm"][i] * s["wmpr"][i] * 1.0e2 * BYGRAV
        tem1 = 1.5e3 * tem1 / (fcld * rclde1 + TEENY)
        if use_vmp:
            if lhp[i] == lhx:
                s["taussl"][i] = s["taussl"][i] + tem1
                s["tausslip"][i] = 0.0
                cnt["tau_lhp_eq"] += 1
            elif lhp[i] == LHE:
                cnt["lhp_lhe_stop"] += 1
                raise RuntimeError("VMP: should not be here")
            else:
                s["tausslip"][i] = tem1
                cnt["tau_ip"] += 1
                if s["tausslip"][i] > 100.0:
                    s["tausslip"][i] = 100.0
                    cnt["tau_ip_cap"] += 1
                if fcld <= TEENY:
                    s["tausslip"][i] = 0.0
                    cnt["ip_fcld_zero"] += 1
        else:
            s["taussl"][i] = s["taussl"][i] + tem1
        if fcld <= TEENY:
            s["taussl"][i] = 0.0
            cnt["fcld_teeny"] += 1
        if s["taussl"][i] > 100.0:
            s["taussl"][i] = 100.0
            cnt["tau_cap"] += 1
        if lhx == LHE:
            wmsum = wmsum + tem
    # CLDSAL = CLDSSL (whole array)
    s["cldsal"] = list(s["cldssl"])
    by3_2 = 2.0 * BY3
    for L in range(1, lmcld + 1):
        i = L - 1
        s["cldsv1"][i] = s["cldssl"][i]
        lhx = s["svlhxl"][i]
        if lhx == LHE:
            if s["qclx"][i] <= 0.0:
                s["svlhxl"][i] = 0.0
                cnt["svlhx_reset"] += 1
        else:
            if s["qcix"][i] <= 0.0:
                s["svlhxl"][i] = 0.0
                cnt["svlhx_reset"] += 1
        if s["taumcl"][i] == 0.0 or ckij != 1.0:
            bmax = 1.0 - ex(-(s["cldsv1"][i] / 0.3))
            if s["cldsv1"][i] >= 0.95:
                bmax = s["cldsv1"][i]
                cnt["bmax_095"] += 1
            if L == 1 or L <= dcl:
                s["cldssl"][i] = min(s["cldssl"][i] + (bmax - s["cldssl"][i]) * ckij, s["fssl"][i])
                s["taussl"][i] = s["taussl"][i] * s["cldsv1"][i] / (s["cldssl"][i] + TEENY)
                if use_vmp:
                    s["tausslip"][i] = s["tausslip"][i] * s["cldsv1"][i] / (s["cldssl"][i] + TEENY)
                s["cldsal"][i] = s["cldssl"][i]
                cnt["rescale_bl"] += 1
            if s["taussl"][i] <= 0.0:
                s["cldssl"][i] = 0.0
            if L > dcl and s["taumcl"][i] <= 0.0:
                s["cldssl"][i] = min(pw(s["cldssl"][i], by3_2), s["fssl"][i])
                s["taussl"][i] = s["taussl"][i] * pw(s["cldsv1"][i], BY3)
                if use_vmp:
                    s["tausslip"][i] = s["tausslip"][i] * pw(s["cldsv1"][i], BY3)
                s["cldsal"][i] = pw(s["cldsal"][i], 2.0 / 3.0)
                cnt["rescale_free"] += 1
        else:
            cnt["skip_taumcl"] += 1
        if s["taussl"][i] < 0.0:
            cnt["neg_tau"] += 1
            s["taussl"][i] = 0.0
            s["cldssl"][i] = 0.0
            if lhx == LHE:
                s["qclx"][i] = 0.0
            else:
                s["qcix"][i] = 0.0
        else:
            if lhx == LHE:
                s["qlss"][i] = s["qlss"][i] + s["qclx"][i]
            else:
                s["qiss"][i] = s["qiss"][i] + s["qcix"][i]
            if lhp[i] == LHE:
                s["qlss"][i] = s["qlss"][i] + s["wmpr"][i]
            else:
                s["qiss"][i] = s["qiss"][i] + s["wmpr"][i]
        if use_vmp and s["tausslip"][i] < 0.0:
            s["tausslip"][i] = 0.0
    return s, wmsum, cnt
