"""Full-fidelity numpy port of CLOUDS2.F90 `MSTCNV` (moist convection; lines 432-3207 of the source, 903 live code
lines for P2SAoM40) -- D110-D113, scoping item D-C7.

One call = one atmospheric column, exactly as the Fortran: `mstcnv_column(r, c)` takes the entry state of one real
call (the `IN_FIELDS` of clouds_mstcnv_io.py: PL/PLE/PLK/AIRM/BYAM/ETAL/TL/TVL/SM/QM/QCLL/QCIL/SDL/WTURB/GZL, the
9-moment SMOM/QMOM, the K-sampled UM/VM/U_0/V_0/RA, and the scalars PEARTH/PLAND/DCL/LMCM/XMASS/BYDTsrc/DTsrc/BYBR)
and returns every output the Fortran writes (`OUT_FIELDS`), optionally with the stage checkpoints of
`CK_FIELDS`.  The control flow (cloud-base loop x 2 plume types x 2 area partitions x ascent loop, downdraft,
subsidence, precip/evap loop, optical-thickness loop, every cycle/exit) is reproduced statement by statement with
Python scalar control flow; layer arrays are 1-based with a pad cell (index 0, and LM+1) so the code reads like the
Fortran.  The Fortran evaluation order is kept exactly (strict left-to-right, no regrouping; ifort -fp-model
strict -assume protect_parens, no FMA).

Reused validated pieces (imported): QSAT/DQSATDT/get_dq_cond/get_dq_evap (clouds_dq_ff), THBAR/MASS_FLUX
(clouds_massflux_ff), PRECIP_MP/MC_CLOUD_FRACTION/MC_PRECIP_PHASE/CONVECTIVE_MICROPHYSICS/ANVIL_OPTICAL_THICKNESS
(clouds_helpers_ff), adv1d (QUS vertical advection, dyn_adv1d_ff; the qlimit=.true. path for Q is the real limitq).

REAL(4)-literal audit of MSTCNV (D54 hazard: an un-suffixed literal is rounded to 24 bits, then promoted):
  .001 (FPLUME/MPLUME/FCTYPE tests), .95 (plume mass cap, three places), 0.7 (PGRAD), .33 (U00L: 222.d0**.33),
  .001*.050*3. (U00L), 1.E-20 (FCLD) are NOT exactly representable and are replicated with f4().
  Exactly representable (no treatment): .5 .25 1. 2. 3. 10. 50. 100. 450. 700. .02? (no: .02d0 is double) 0.
  d0 literals (.16667D0, .66667D0, 0.95d0, 0.975d0, 1.d-10, .01d0, .08d0, ...) are double.
Intrinsic risks: exp (QSAT), **.25 (FLAMW/G/I), 222.d0**.33 and the exp/pow inside the helpers may differ in the
last bit between numpy/glibc and Intel libimf; `backend="imf"` routes np.exp/np.power calls inside the reused modules
through libimf (intel_libm_ff.py path) when available (the `**` operator inside the helper formulas is not
intercepted).

Module-state notes: values the Fortran leaves in module variables from earlier calls (AIRXL, PRHEAT when no
convection occurs; the uninitialised entries of SMOMP/QMOMP/SMOMPMAX/QMOMPMAX outside xymoms) are not reproducible
from the inputs and are excluded from comparisons by the compare script.  Out-of-range reads in the Fortran
(U_0(K,LM+1), BYAM(LM+1), ...) read zero here.
"""
import ctypes
import os
from collections import defaultdict

import numpy as np

import clouds_dq_ff as dq
import clouds_helpers_ff as hp
import clouds_massflux_ff as mf
import dyn_adv1d_ff as adv
from clouds_helpers_ff import f4

LM = 40
NMOM = 9
MX, MY, MZ, MXX, MYY, MZZ, MXY, MZX, MYZ = range(9)      # QUSDEF.f order (mzx=8, myz=9 in 1-based)
XYM = [MX, MY, MXX, MXY, MYY]                             # xymoms
ZM = [MZ, MZZ, MYZ, MZX]                                  # zmoms
ZDIR = adv.ZDIR

# ---- constants (Constants_mod.F90, non-PLANET_PARAMS), same derivations as clouds_dq_ff / clouds_helpers_ff
RGAS, GRAV, TEENY, PI = hp.RGAS, hp.GRAV, hp.TEENY, hp.PI
LHE, LHS, TF, BYSHA = hp.LHE, hp.LHS, hp.TF, hp.BYSHA
BYGRAV = 1.0 / GRAV
DELTX = mf.DELTX
SLHE = mf.SLHE
TI = 233.16
CN0 = CN0I = CN0G = 8.0e6
RHOG, RHOIP = 400.0, 100.0
ITMAX = 50
FITMAX = 1.0 / ITMAX
WMAX = 50.0
SECONDS_PER_HOUR = 3600.0
# CLOUDS2.F90 module parameters
CLDMIN, FDDET, DTMIN1, COETAU, WMU, WMUL = 0.10, 0.25, 1.0, 0.08, 0.25, 0.5
CCMUL, CCMUL2 = 2.0, 3.0
U00A_DEFAULT = 0.55

DEFAULT_TUNE = dict(entrainment_cont1=0.4, entrainment_cont2=0.6, radiusl_multiplier=1.0, radiusi_multiplier=1.0,
                    u00a=0.55, u00b=1.0, wmu_multiplier=1.0, rwcldox=1.0, rimax=100.0, mc_fddrt=0.5,
                    mc_entr_mass_lim_plume=1, mc_new_ddrft_thetav=1, mc_revp_abv_cldbase=1)


def tune_from_consts(c):
    """Tunables of the real run (ffc_mc_consts.txt as dict)."""
    t = dict(DEFAULT_TUNE)
    for k in t:
        if k in c:
            t[k] = c[k]
    for k in ("mc_entr_mass_lim_plume", "mc_new_ddrft_thetav", "mc_revp_abv_cldbase"):
        t[k] = int(t[k])
    return t


# ------------------------------------------------------------------------------------------- libimf backend
_IMF = None


def _imf():
    global _IMF
    if _IMF is None:
        d = os.environ.get("INTEL_LIBIMF_DIR", "/panfs/ccds02/app/modules/intel/platform/x86_64/rhel/8.6/2020Update4/"
                           "compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin")
        try:
            ctypes.CDLL(os.path.join(d, "libintlc.so.5"), mode=ctypes.RTLD_GLOBAL)
            lib = ctypes.CDLL(os.path.join(d, "libimf.so"))
            lib.pow.restype = ctypes.c_double
            lib.pow.argtypes = [ctypes.c_double, ctypes.c_double]
            lib.exp.restype = ctypes.c_double
            lib.exp.argtypes = [ctypes.c_double]
            _IMF = lib
        except OSError:
            _IMF = False
    return _IMF


def imf_available():
    return bool(_imf())


class _NpProxy:
    """numpy with exp / power routed through Intel libimf (scalar calls)."""

    def __getattr__(self, name):
        return getattr(np, name)

    def exp(self, x):
        f = _imf().exp
        a = np.asarray(x, dtype=np.float64)
        return np.array([f(v) for v in a.ravel()]).reshape(a.shape)[()] if a.shape else np.float64(f(float(a)))

    def power(self, x, y):
        f = _imf().pow
        a, b = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        out = np.array([f(u, v) for u, v in zip(a.ravel(), b.ravel())]).reshape(a.shape)
        return out[()] if a.shape == () else out


_PROXY = _NpProxy()
_MODS = (dq, hp)
_pow = np.power
_HP_ORIG = dict(dcg=hp._dcg, dci=hp._dci, dcw=hp._dcw_search)


def _dcg_imf(wv, pl, gr=True):
    return np.minimum(_pow((wv / hp._F193) * _pow(pl / 1000.0, hp._F04), hp._F27), 1e-2)


def _dci_imf(wv, pl):
    return np.minimum(_pow((wv / hp._F1172) * _pow(pl / 1000.0, hp._F04), hp._F2439), 1e-2)


def _dcw_search_imf(wv, pl, ddcw, wmax, nmax):
    wv, pl, ddcw, wmax = np.broadcast_arrays(wv, pl, ddcw, wmax)
    dcw = np.zeros(wv.shape)
    active = np.ones(wv.shape, bool)
    kind = np.zeros(wv.shape, int)
    nit = np.zeros(wv.shape, int)
    pfac = _pow(1000.0 / pl, 0.4)
    for k in range(int(np.max(nmax))):
        act = active & (k < nmax)
        vt = (-0.267 + dcw * (5.15e3 - dcw * (1.0225e6 - 7.55e7 * dcw))) * pfac
        ex1 = (vt >= 0.0) & (vt >= wv)
        ex2 = ~ex1 & (vt > wmax)
        ex = act & (ex1 | ex2)
        kind = np.where(ex & ex1, 1, np.where(ex & ex2, 2, kind))
        nit = np.where(act, k + 1, nit)
        dcw = np.where(act & ~ex, dcw + ddcw, dcw)
        active = active & ~ex
    return dcw, nit, kind


def _anvil_imf(svlatl, rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem):
    """ANVIL_OPTICAL_THICKNESS with the two `**BY3` evaluated through libimf pow (same formula as the validated
    clouds_helpers_ff.anvil_optical_thickness)."""
    r_liq = rcldlx * 100.0 * _pow(wtem / (2.0 * hp.BY3 * hp.TWOPI * mcdncw), hp.BY3)
    r_ice = rcldix * 100.0 * _pow(wtem / (2.0 * hp.BY3 * hp.TWOPI * mcdnci), hp.BY3)
    r_ice = np.minimum(r_ice, rimax)
    rcld = np.where(svlatl == hp.LHE, r_liq, r_ice)
    rclde = rcld / bybr
    taumc = 1.5 * tem / (fcld * rclde + hp._F1E20)
    taumc = np.where(taumc > 100.0, 100.0, taumc)
    return rcld, taumc


_ANVIL = hp.anvil_optical_thickness


def set_backend(name):
    """'numpy' (default) or 'imf': route exp / pow through Intel libimf everywhere the port reaches them
    (np.exp/np.power inside clouds_dq_ff and clouds_helpers_ff, the `**` of the microphysics size searches and of
    the anvil radius, and this module's own pow calls)."""
    global _pow, _ANVIL
    if name == "imf":
        if not imf_available():
            raise RuntimeError("Intel libimf not available")
        for m in _MODS:
            m.np = _PROXY
        _pow = _PROXY.power
        hp._dcg, hp._dci, hp._dcw_search = _dcg_imf, _dci_imf, _dcw_search_imf
        _ANVIL = _anvil_imf
    else:
        for m in _MODS:
            m.np = np
        _pow = np.power
        hp._dcg, hp._dci, hp._dcw_search = _HP_ORIG["dcg"], _HP_ORIG["dci"], _HP_ORIG["dcw"]
        _ANVIL = hp.anvil_optical_thickness


# ------------------------------------------------------------------------------------------- helpers
def _a1(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.zeros(LM + 2)
    a[1:1 + x.size] = x
    return a


def _m1(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.zeros((x.shape[0], LM + 2))
    a[:, 1:1 + x.shape[1]] = x
    return a


def _seqsum(v):
    s = np.float64(0.0)
    for t in v:
        s = s + t
    return s


def _seqsum_rows(a, lo, hi):
    """sum(A(K,lo:hi)) for every K, left to right."""
    s = np.zeros(a.shape[0])
    for l in range(lo, hi + 1):
        s = s + a[:, l]
    return s


def _scal(x):
    return np.float64(x)


# fields whose checkpoint value is forced to zero when the Fortran left them undefined
def snapshot(stage, lmin, ic, nppl, v, fields):
    """Record dict for a checkpoint: stage payload fields from the locals dict `v` (1-based padded arrays)."""
    out = {}
    lmax, lmin_, etadn = v.get("lmax", 0), v.get("lmin", 0), v.get("etadn", 0.0)
    for name, _, shape in fields:
        if name == "mc1":
            out[name] = float(v["mc1"])
            continue
        x = v[name]
        if shape == ():
            val = float(x)
        elif shape == (LM,):
            val = np.array(x[1:LM + 1], dtype=np.float64)
        elif shape == (LM + 1,) and name == "cm":
            val = np.array(x[0:LM + 1], dtype=np.float64)
        elif shape == (LM + 1,):
            val = np.array(x[1:LM + 2], dtype=np.float64)
        elif len(shape) == 2:
            val = np.array(x[:, 1:LM + 1], dtype=np.float64)
        else:
            val = np.array(x, dtype=np.float64)
        out[name] = val
    if stage == 4 and not lmax > lmin_:
        for n in ("mpmax", "smpmax", "qmpmax"):
            out[n] = 0.0
        out["smompmax"] = np.zeros(NMOM)
        out["qmompmax"] = np.zeros(NMOM)
    if stage == 5 and not etadn > 1e-10:
        for n in ("edraft", "ddraft", "smdn", "qmdn"):
            out[n] = 0.0
    if stage == 9 and not v["lmcmin"] > 0:
        out["airxl"] = 0.0
    return dict(stage=stage, lmin=lmin, ic=ic, nppl=nppl, f=out)


# ------------------------------------------------------------------------------------------- the port
def mstcnv_column(r, c, ck=None, br=None, mut=None):
    """Port of MSTCNV for one column.

    r: dict of one record's inputs (clouds_mstcnv_io IN_FIELDS names; arrays 0-based [layer-1]).
    c: tunables dict (tune_from_consts).  ck: if a list, checkpoint events are appended (same stages as the dump).
    br: dict-like counter of branch outcomes.  mut: dict of mutation switches (tests only).
    Returns dict of outputs named as OUT_FIELDS (0-based arrays)."""
    mut = mut or {}
    if br is None:
        br = defaultdict(int)
    from clouds_mstcnv_io import CK_FIELDS
    with np.errstate(all="ignore"):
        return _mstcnv(r, c, ck, br, mut, CK_FIELDS)


def _mstcnv(r, c, ck, br, mut, CK):
    F = np.float64
    pearth, pland = F(r["pearth"]), F(r["pland"])
    dcl, lmcm = int(r["dcl"]), int(r["lmcm"])
    xmass, bydtsrc, dtsrc, bybr = F(r["xmass"]), F(r["bydtsrc"]), F(r["dtsrc"]), F(r["bybr"])
    pl, ple, plk, airm, byam = (_a1(r[k]) for k in ("pl", "ple", "plk", "airm", "byam"))
    etal, tl, tvl = _a1(r["etal"]), _a1(r["tl"]), _a1(r["tvl"])
    sm, qm = _a1(r["sm"]), _a1(r["qm"])
    qcll, qcil, sdl, wturb, gzl = (_a1(r[k]) for k in ("qcll", "qcil", "sdl", "wturb", "gzl"))
    smom, qmom = _m1(r["smom"]), _m1(r["qmom"])
    ra = np.asarray(r["ra"], dtype=np.float64)
    um, vm, u_0, v_0 = (_m1(r[k]) for k in ("um", "vm", "u0", "v0"))
    nks = ra.size

    contce1, contce2 = F(c["entrainment_cont1"]), F(c["entrainment_cont2"])
    rcldlx, rcldix = F(c["radiusl_multiplier"]), F(c["radiusi_multiplier"])
    u00a, u00b = F(c["u00a"]), F(c["u00b"])
    wmu_mult, rwcldox, rimax = F(c["wmu_multiplier"]), F(c["rwcldox"]), F(c["rimax"])
    fddrt = F(c["mc_fddrt"])
    mc_entr_lim, mc_newthv, mc_revp = c["mc_entr_mass_lim_plume"], c["mc_new_ddrft_thetav"], c["mc_revp_abv_cldbase"]
    f001 = F(0.001) if mut.get("f001_double") else F(f4(0.001))
    f95 = F(0.95) if mut.get("f95_double") else F(f4(0.95))
    pgrad = F(0.7) if mut.get("pgrad_double") else F(f4(0.7))
    zl = lambda: np.zeros(LM + 2)  # noqa: E731
    zm = lambda: np.zeros((NMOM, LM + 2))  # noqa: E731
    zk = lambda: np.zeros((nks, LM + 2))  # noqa: E731

    ierr = lerr = 0
    lmcmin = lmcmax = 0
    mccont = 0
    fmc1 = F(0.0)
    fssl = np.ones(LM + 2)
    tadj = F(1.0)
    qsatre = dq.qsat(F(283.16), LHE, F(920.0))
    taumcl, condpt, svwmxl, svlatl, svlat1, vsubl = zl(), zl(), zl(), zl(), zl(), zl()
    precnvl, cldmcl, tpsav, lhp = zl(), zl(), zl(), zl()
    cldslwij = clddepij = prcpmc = F(0.0)
    csizel = zl()
    csizel[1:LM + 1] = rwcldox * 10. * (1. - pearth) + 10. * pearth
    vlat = np.full(LM + 2, LHE)
    condmmr = zl()
    mcflx, dgdsm, dgdeep, dgshlw, dphase, dphadeep, dphashlw, dtotw = (zl() for _ in range(8))
    dqcond, dgdqm, dqmtotal, dqmshlw, dqmdeep, dqctotal, dqcshlw, dqcdeep = (zl() for _ in range(8))
    ddmflx, tdnl, qdnl = zl(), zl(), zl()
    sm1, qm1 = sm.copy(), qm.copy()
    smold, smomold, qmold, qmomold = sm.copy(), smom.copy(), qm.copy(), qmom.copy()
    u00l = zl()
    # U00 for PBL stratiform clouds: 1.d0-2.*(U00b*.001*.050*3.*222.d0**.33)/QSATRE (REAL(4) literals)
    x_u00 = u00b * F(f4(.001)) * F(f4(.050)) * F(3.) * _pow(F(222.0), F(f4(.33)))
    u00_pbl = 1.0 - 2. * x_u00 / qsatre
    u00_mc = 1.0 - 2. * (u00b * 2.0e-4 / qsatre)
    for L in range(1, LM + 1):
        u00l[L] = 0.
        if pl[L] >= pl[dcl]:
            u00l[L] = u00_pbl
    dwcu = F(0.0)
    for L in range(1, lmcm + 1):
        dwcu = dwcu + airm[L] * tl[L] * RGAS / (GRAV * pl[L])
    dwcu = 0.5 * dwcu * bydtsrc / F(lmcm)

    # state variables that live across loops (Fortran locals)
    dm = dmr = ddr = ccm = cond = condp = condp1 = condv = taumc1 = cdheat = ent = det = buoy = wcu = wcu2 = None
    smdnl = qmdnl = dsm = dsmr = dqm = dqmr = ddm = heat1 = condgp = condip = None
    dsmom = dsmomr = dqmom = dqmomr = smomdnl = qmomdnl = None
    dum = dvm = umdnl = vmdnl = None
    ump = vmp = np.zeros(nks)
    smomp = qmomp = smompmax = qmompmax = np.zeros(NMOM)
    mpmax = smpmax = qmpmax = F(0.0)
    lmax = lmin = ldraft = ldmin = llmin = lfrz = 0
    etadn = F(0.0)
    cdhsum = cdhsum1 = cdhdrt = cdhm = evpsum = F(0.0)
    prcp = prheat = airxl = F(0.0)
    mc1 = False
    mplume = fplume = smp = qmp = ddraft = F(0.0)
    fctype = mplum1 = told = F(0.0)
    cm = zl()
    cmneg = zl()
    dmse = fmp0 = dqsum = fmp2 = F(0.0)
    ksub = 1

    def rec(stage, ic, nppl, lm_):
        if ck is not None:
            ck.append(snapshot(stage, lm_, ic, nppl, loc, CK[stage]))

    def bump(k):
        br[k] += 1

    loc = locals()

    # =========================================================================== cloud-base loop
    for lmin in range(1, lmcm):
        maxlvl = 0
        minlvl = LM
        fmp0 = -(10. * 1.0 * sdl[lmin + 1] * BYGRAV * xmass)
        if fmp0 <= 0.:
            fmp0 = F(0.0)
        smo1, qmo1, smo2, qmo2 = sm[lmin], qm[lmin], sm[lmin + 1], qm[lmin + 1]
        sdn = smo1 * byam[lmin]
        sup = smo2 * byam[lmin + 1]
        sedge = mf.thbar(sup, sdn)
        qdn = qmo1 * byam[lmin]
        qup = qmo2 * byam[lmin + 1]
        wmdn = qcll[lmin] + qcil[lmin]
        wmup = qcll[lmin + 1] + qcil[lmin + 1]
        svdn = sdn * (1. + DELTX * qdn - wmdn)
        svup = sup * (1. + DELTX * qup - wmup)
        qedge = .5 * (qup + qdn)
        wmedg = .5 * (wmup + wmdn)
        svedg = sedge * (1. + DELTX * qedge - wmedg)
        lhx = F(LHE)
        slh = lhx * BYSHA
        dmse = (svup - svedg) * plk[lmin + 1] + (svedg - svdn) * plk[lmin] + \
            slh * (dq.qsat(sup * plk[lmin + 1], lhx, pl[lmin + 1]) - qdn)
        loc = locals()
        rec(1, 0, 0, lmin)
        if dmse > -1e-10:
            bump("base_cycle_dmse")
            continue
        mfo = mf.mass_flux(F(lmin), lhx, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg,
                           airm[lmin], airm[lmin + 1], byam[lmin], byam[lmin + 1], byam[lmin + 2],
                           sm[lmin + 2], qm[lmin + 2], plk[lmin], plk[lmin + 1], pl[lmin], pl[lmin + 1])
        fplume, fmp2, dqsum = F(mfo["fplume"]), F(mfo["fmp2"]), F(mfo["dqsum"])
        loc = locals()
        rec(2, 0, 0, lmin)
        if fplume <= f001:
            bump("base_cycle_fplume")
            continue
        bump("base_pass")
        fmp2 = fmp2 * min(1.0, dtsrc / (tadj * SECONDS_PER_HOUR))
        for ic in (1, 2):
            mc1 = False
            lhx = F(LHE)
            mplume = min(airm[lmin], airm[lmin + 1], fmp2)
            fctype = F(1.0)
            if mplume > fmp0:
                fctype = fmp0 / mplume
            if ic == 2:
                fctype = 1. - fctype
            if fctype < f001:
                bump("type_cycle_fctype")
                continue
            mplum1 = mplume
            cycle_types = False
            for nppl in (1, 2):
                if nppl == 2 and (not mc1 or mccont < 2):
                    continue
                if nppl == 2:
                    bump("area_partition_2")
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
                cond, cdheat, condp, condp1, condgp = zl(), zl(), zl(), zl(), zl()
                condip, condv, heat1, dm, dmr, ddr = zl(), zl(), zl(), zl(), zl(), zl()
                ccm, ddm, taumc1, ent, det, buoy = zl(), zl(), zl(), zl(), zl(), zl()
                wcu, smdnl, qmdnl = zl(), zl(), zl()
                smomdnl, qmomdnl = zm(), zm()
                umdn, vmdn = np.zeros(nks), np.zeros(nks)
                umdnl, vmdnl, dum, dvm = zk(), zk(), zk(), zk()
                dsm, dsmom, dsmr, dsmomr = zl(), zm(), zl(), zm()
                dqm, dqmom, dqmr, dqmomr = zl(), zm(), zl(), zm()
                if wcu2 is None:
                    wcu2 = zl()
                loc = locals()
                rec(3, ic, nppl, lmin)
                mplume = min(mplume / fmc1, airm[lmin] * 0.95 * qm[lmin] / (qmold[lmin] + TEENY))
                if mplume <= f001 * airm[lmin]:
                    bump("type_cycle_mplume_small")
                    cycle_types = True
                    break
                fplume = mplume * byam[lmin]
                smp = smold[lmin] * fplume
                smomp = np.zeros(NMOM)
                qmomp = np.zeros(NMOM)
                smomp[XYM] = smomold[XYM, lmin] * fplume
                qmp = qmold[lmin] * fplume
                qmomp[XYM] = qmomold[XYM, lmin] * fplume
                if tpsav[lmin] == 0:
                    tpsav[lmin] = smp * plk[lmin] / mplume
                dmr[lmin] = -mplume
                dsmr[lmin] = -smp
                dsmomr[XYM, lmin] = -smomp[XYM]
                dsmomr[ZM, lmin] = -smomold[ZM, lmin] * fplume
                dqmr[lmin] = -qmp
                dqmomr[XYM, lmin] = -qmomp[XYM]
                dqmomr[ZM, lmin] = -qmomold[ZM, lmin] * fplume
                ump = um[:, lmin] * fplume
                dum[:, lmin] = -ump
                vmp = vm[:, lmin] * fplume
                dvm[:, lmin] = -vmp
                cdhsum = cdhsum1 = cdhdrt = F(0.0)
                etadn = F(0.0)
                ldraft = LM
                evpsum = F(0.0)
                ddraft = F(0.0)
                lfrz = 0
                lmax = lmin
                contce = contce1
                if ic == 2:
                    contce = contce2
                wcu[lmin] = max(.5, wturb[lmin + 1])
                if ic == 1:
                    wcu[lmin] = max(.5, 2.0 * wturb[lmin + 1])
                wcu2[lmin] = wcu[lmin] * wcu[lmin]
                mpmax = smpmax = qmpmax = F(0.0)
                smompmax = np.zeros(NMOM)
                qmompmax = np.zeros(NMOM)

                # ------------------------------------------------------------------ plume ascent
                for L in range(lmin + 1, LM + 1):
                    if mplume <= f001 * airm[L]:
                        bump("top_exit_mplume_small")
                        break
                    sdn = smp / mplume
                    sup = sm1[L] * byam[L]
                    qdn = qmp / mplume
                    qup = qm1[L] * byam[L]
                    wmdn = F(0.0)
                    wmup = qcll[L] + qcil[L]
                    svdn = sdn * (1. + DELTX * qdn - wmdn)
                    svup = sup * (1. + DELTX * qup - wmup)
                    if plk[L - 1] * (svup - svdn) + SLHE * (qup - qdn) >= 0.:
                        bump("top_exit_stable")
                        break
                    if tpsav[L] == 0:
                        tpsav[L] = smp * plk[L] / mplume
                    tp = tpsav[L]
                    if tpsav[L - 1] >= TF and tpsav[L] < TF:
                        lfrz = L - 1
                    lhx = F(LHE)
                    if tp < TI:
                        lhx = F(LHS)
                    qsatmp = mplume * dq.qsat(tp, lhx, pl[L])
                    if qmp < qsatmp:
                        bump("top_exit_noplume")
                        break
                    if tp < TF and lhx == LHE:
                        lhx = F(LHS)
                        qsatmp = mplume * dq.qsat(tp, lhx, pl[L])
                        bump("lhx_switch_ice")
                    if vlat[L] == LHS:
                        lhx = F(LHS)
                    vlat[L] = lhx
                    slh = lhx * BYSHA
                    if tl[L] >= TF and u00l[L] != u00a:
                        u00l[L] = u00_mc
                    mccont += 1
                    if mccont == 1:
                        mc1 = True
                    if mplume > f95 * airm[L]:
                        bump("plume_mass_cap")
                        delta = (mplume - f95 * airm[L]) / mplume
                        dm[L - 1] = dm[L - 1] + delta * mplume
                        mplume = f95 * airm[L]
                        dsm[L - 1] = dsm[L - 1] + delta * smp
                        smp = smp * (1. - delta)
                        dsmom[XYM, L - 1] = dsmom[XYM, L - 1] + delta * smomp[XYM]
                        smomp[XYM] = smomp[XYM] * (1. - delta)
                        dqm[L - 1] = dqm[L - 1] + delta * qmp
                        qmp = qmp * (1. - delta)
                        dqmom[XYM, L - 1] = dqmom[XYM, L - 1] + delta * qmomp[XYM]
                        qmomp[XYM] = qmomp[XYM] * (1. - delta)
                        dum[:, L - 1] = dum[:, L - 1] + ump * delta
                        dvm[:, L - 1] = dvm[:, L - 1] + vmp * delta
                        ump = ump - ump * delta
                        vmp = vmp - vmp * delta
                    work = mplume * (sup - sdn) * (plk[L - 1] - plk[L]) / plk[L - 1]
                    dsm[L - 1] = dsm[L - 1] - work
                    ccm[L - 1] = mplume
                    dqs, fqcond = dq.get_dq_cond(smp, qmp, plk[L], mplume, lhx, pl[L])
                    dqsum, fqcond = F(dqs), F(fqcond)
                    if dqsum > 0. and qmp > TEENY:
                        qmomp[XYM] = qmomp[XYM] * (1. - fqcond)
                        smp = smp + slh * dqsum / plk[L]
                        qmp = qmp - dqsum
                        bump("cond_applied")
                    cond[L] = dqsum
                    condmmr[L] = condmmr[L] + cond[L] * byam[L] * fmc1
                    cdheat[L] = slh * cond[L]
                    cdhsum = cdhsum + cdheat[L]
                    cond[L] = cond[L] + condv[L - 1]
                    if vlat[L - 1] != vlat[L]:
                        bump("vlat_phase_change")
                        smp = smp - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA / plk[L]
                        cdheat[L] = cdheat[L] - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA
                        cdhsum = cdhsum - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA
                    condmu = 100. * cond[L] * pl[L] / (ccm[L - 1] * tl[L] * RGAS)
                    flamw = _pow(1000.0 * PI * CN0 / (condmu + TEENY), F(.25))
                    flamg = _pow(400.0 * PI * CN0G / (condmu + TEENY), F(.25))
                    flami = _pow(100.0 * PI * CN0I / (condmu + TEENY), F(.25))
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
                            bump("fentr_cap")
                            fentr = 1. - fplume
                            ent[L] = 0.001 * fentr / (gzl[L] * fplume)
                        if fentr >= TEENY:
                            mpold = mplume
                            fpold = fplume
                            etal1 = fentr / (fplume + TEENY)
                            eplume = mplume * etal1
                            if eplume > airm[L] * 0.975 - mplume:
                                bump("eplume_cap")
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
                            dsmomr[:, L] = dsmomr[:, L] - smom[:, L] * fentra
                            dqmr[L] = dqmr[L] - eplume * qup
                            dqmomr[:, L] = dqmomr[:, L] - qmom[:, L] * fentra
                            dmr[L] = dmr[L] - eplume
                            smp = smp + eplume * sup
                            smomp[XYM] = smomp[XYM] + smom[XYM, L] * fentra
                            qmp = qmp + eplume * qup
                            qmomp[XYM] = qmomp[XYM] + qmom[XYM, L] * fentra
                            umtemp = pgrad * mplume ** 2 * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                            vmtemp = pgrad * mplume ** 2 * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                            ump = ump + u_0[:, L] * eplume + umtemp
                            dum[:, L] = dum[:, L] - u_0[:, L] * eplume - umtemp
                            vmp = vmp + v_0[:, L] * eplume + vmtemp
                            dvm[:, L] = dvm[:, L] - v_0[:, L] * eplume - vmtemp
                            bump("entrain_applied")
                    if det[L] > 0.:
                        delta = 1000. * det[L] * gzl[L]
                        if delta > .95:
                            bump("det_cap")
                            delta = F(.95)
                            det[L] = .001 * delta / gzl[L]
                        dm[L] = dm[L] + delta * mplume
                        mplume = mplume * (1. - delta)
                        dsm[L] = dsm[L] + delta * smp
                        smp = smp * (1. - delta)
                        dsmom[XYM, L] = dsmom[XYM, L] + delta * smomp[XYM]
                        smomp[XYM] = smomp[XYM] * (1. - delta)
                        dqm[L] = dqm[L] + delta * qmp
                        qmp = qmp * (1. - delta)
                        dqmom[XYM, L] = dqmom[XYM, L] + delta * qmomp[XYM]
                        qmomp[XYM] = qmomp[XYM] * (1. - delta)
                        umtemp = pgrad * mplume ** 2 * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                        vmtemp = pgrad * mplume ** 2 * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                        dum[:, L] = dum[:, L] + ump * delta - umtemp
                        dvm[:, L] = dvm[:, L] + vmp * delta - vmtemp
                        ump = ump - ump * delta + umtemp
                        vmp = vmp - vmp * delta + vmtemp
                        bump("detrain_applied")
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
                            dmmix = (svup - svmix) * plk[L] + SLHE * (dq.qsat(sup * plk[L], lhx, pl[L]) - qmix)
                        if dmmix < 1e-10:
                            cdhdrt = cdhdrt + cdheat[L]
                        if dmmix >= 1e-10:
                            bump("downdraft_trigger")
                            ldraft = L
                            etadn = F(1.0) / F(3.0)
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
                            smomdnl[XYM, L] = smom[XYM, L] * fddl + smomp[XYM] * fddp
                            smp = fleft * smp
                            smomp[XYM] = smomp[XYM] * fleft
                            qmdnl[L] = ddraft * qmix
                            qmomdnl[XYM, L] = qmom[XYM, L] * fddl + qmomp[XYM] * fddp
                            qmp = fleft * qmp
                            qmomp[XYM] = qmomp[XYM] * fleft
                            dmr[L] = dmr[L] - .5 * ddraft
                            dsmr[L] = dsmr[L] - .5 * ddraft * sup
                            dsmomr[:, L] = dsmomr[:, L] - smom[:, L] * fddl
                            dqmr[L] = dqmr[L] - .5 * ddraft * qup
                            dqmomr[:, L] = dqmomr[:, L] - qmom[:, L] * fddl
                            umdnl[:, L] = .5 * (etadn * ump + ddraft * u_0[:, L])
                            ump = ump * fleft
                            dum[:, L] = dum[:, L] - .5 * ddraft * u_0[:, L]
                            vmdnl[:, L] = .5 * (etadn * vmp + ddraft * v_0[:, L])
                            vmp = vmp * fleft
                            dvm[:, L] = dvm[:, L] - .5 * ddraft * v_0[:, L]
                    w2tem = .16667 * GRAV * buoy[L] - wcu[L - 1] * wcu[L - 1] * (.66667 * det[L] + ent[L])
                    hdep = airm[L] * tl[L] * RGAS / (GRAV * pl[L])
                    wcu2[L] = wcu2[L - 1] + 2. * hdep * w2tem
                    wcu[L] = 0.
                    if wcu2[L] > 0.:
                        wcu[L] = np.sqrt(wcu2[L])
                    if wcu[L] >= 0.:
                        wcu[L] = min(50., wcu[L])
                    if wcu[L] < 0.:
                        wcu[L] = max(-50., wcu[L])
                    smpmax = smp
                    smompmax = smomp.copy()
                    qmpmax = qmp
                    qmompmax = qmomp.copy()
                    mpmax = mplume
                    lmax = lmax + 1
                    if wcu2[L] < 0.:
                        bump("top_exit_wcu2")
                        break
                    wcufrz = F(0.0)
                    if lfrz > 0:
                        wcufrz = wcu[lfrz]
                    cp, cp1, cip, cgp = hp.convective_microphysics(
                        pl[L], wcu[L], dwcu, F(lfrz), wcufrz, tp, TI, FITMAX, pland, CN0, CN0I, CN0G, flamw, flamg,
                        flami, RHOIP, RHOG, ITMAX, tl[lmin], tl[lmin + 1], WMAX, condip[L], condgp[L])
                    condp[L], condp1[L], condip[L], condgp[L] = cp, cp1, cip, cgp
                    condp[L] = .01 * condp[L] * ccm[L - 1] * tl[L] * RGAS / pl[L]
                    condp1[L] = .01 * condp1[L] * ccm[L - 1] * tl[L] * RGAS / pl[L]
                    if condp1[L] > cond[L]:
                        condp1[L] = cond[L]
                    if condp[L] > condp1[L]:
                        condp[L] = condp1[L]
                    condv[L] = cond[L] - condp1[L]
                    cond[L] = cond[L] - condv[L]
                    taumc1[L] = taumc1[L] - condv[L] * fmc1
                loc = locals()
                rec(4, ic, nppl, lmin)
                if lmin == lmax:
                    bump("type_cycle_lmin_eq_lmax")
                    cycle_types = True
                    break
            if cycle_types:
                continue
            # ---------------------------------------------------------------------- after the plume partitions
            taumcl[lmin:lmax + 1] = taumcl[lmin:lmax + 1] + taumc1[lmin:lmax + 1]
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
            dsmom[XYM, lmax] = dsmom[XYM, lmax] + smompmax[XYM]
            dqm[lmax] = dqm[lmax] + qmpmax
            dqmom[XYM, lmax] = dqmom[XYM, lmax] + qmompmax[XYM]
            ccm[lmax] = 0.
            dum[:, lmax] = dum[:, lmax] + ump
            dvm[:, lmax] = dvm[:, lmax] + vmp
            cdhm = F(0.0)
            if minlvl > lmin:
                minlvl = lmin
            if maxlvl < lmax:
                maxlvl = lmax
            if lmcmin == 0:
                lmcmin = lmin
            if lmcmax < maxlvl:
                lmcmax = maxlvl

            # ---------------------------------------------------------------------- downdraft
            ldmin = ldraft - 1
            llmin = ldmin
            edraft = F(0.0)
            smdn = qmdn = ddrold = F(0.0)
            if etadn > 1e-10:
                bump("downdraft_active")
                ddraft = ddr[ldraft]
                ddrold = ddraft
                smdn = smdnl[ldraft]
                qmdn = qmdnl[ldraft]
                smomdn = np.zeros(NMOM)
                qmomdn = np.zeros(NMOM)
                smomdn[XYM] = smomdnl[XYM, ldraft]
                qmomdn[XYM] = qmomdnl[XYM, ldraft]
                umdn = umdnl[:, ldraft].copy()
                vmdn = vmdnl[:, ldraft].copy()
                for L in range(ldraft, 0, -1):
                    lhx = vlat[L]
                    slh = lhx * BYSHA
                    dqe, fq1 = dq.get_dq_evap(smdn, qmdn, plk[L], ddraft, lhx, pl[L], cond[L])
                    dqsum = F(dqe)
                    dqevp = fddrt * cond[L]
                    if dqevp > dqsum:
                        dqevp = dqsum
                    if dqevp > smdn * plk[L] / slh:
                        dqevp = smdn * plk[L] / slh
                    if L < lmin:
                        dqevp = F(0.0)
                    fsevp = F(0.0)
                    if plk[L] * smdn > TEENY:
                        fsevp = slh * dqevp / (plk[L] * smdn)
                    smdn = smdn - slh * dqevp / plk[L]
                    smomdn[XYM] = smomdn[XYM] * (1. - fsevp)
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
                            bump("dd_detrain_buoyant")
                            ddraft = FDDET * ddrup
                        if ddraft > .95 * (airm[L - 1] + dmr[L - 1]):
                            bump("dd_cap_095")
                            ddraft = .95 * (airm[L - 1] + dmr[L - 1])
                        edraft = ddraft - ddrup
                        if edraft > 0:
                            bump("dd_entrain")
                            fentra = edraft * byam[L]
                            senv = sm[L] * byam[L]
                            qenv = qm[L] * byam[L]
                            smdn = smdn + edraft * senv
                            qmdn = qmdn + edraft * qenv
                            smomdn[XYM] = smomdn[XYM] + smom[XYM, L] * fentra
                            qmomdn[XYM] = qmomdn[XYM] + qmom[XYM, L] * fentra
                            dsmr[L] = dsmr[L] - edraft * senv
                            dsmomr[:, L] = dsmomr[:, L] - smom[:, L] * fentra
                            dqmr[L] = dqmr[L] - edraft * qenv
                            dqmomr[:, L] = dqmomr[:, L] - qmom[:, L] * fentra
                            dmr[L] = dmr[L] - edraft
                            umtemp = pgrad * ddraft ** 2 * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                            vmtemp = pgrad * ddraft ** 2 * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                            umdn = umdn + fentra * um[:, L] - umtemp
                            vmdn = vmdn + fentra * vm[:, L] - vmtemp
                            dum[:, L] = dum[:, L] - fentra * um[:, L] + umtemp
                            dvm[:, L] = dvm[:, L] - fentra * vm[:, L] + vmtemp
                        else:
                            bump("dd_detrain_env")
                            fentra = edraft / (ddrup + TEENY)
                            dsm[L] = dsm[L] - fentra * smdn
                            dsmom[XYM, L] = dsmom[XYM, L] - smomdn[XYM] * fentra
                            dqm[L] = dqm[L] - fentra * qmdn
                            dqmom[XYM, L] = dqmom[XYM, L] - qmomdn[XYM] * fentra
                            smdn = smdn * (1 + fentra)
                            qmdn = qmdn * (1 + fentra)
                            smomdn[XYM] = smomdn[XYM] * (1 + fentra)
                            qmomdn[XYM] = qmomdn[XYM] * (1 + fentra)
                            dm[L] = dm[L] - edraft
                            umtemp = pgrad * ddraft ** 2 * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                            vmtemp = pgrad * ddraft ** 2 * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                            dum[:, L] = dum[:, L] - fentra * umdn + umtemp
                            dvm[:, L] = dvm[:, L] - fentra * vmdn + vmtemp
                            umdn = umdn * (1. + fentra) - umtemp
                            vmdn = vmdn * (1. + fentra) - vmtemp
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
                            bump("dd_exit_buoyant")
                            break
                        ddm[L - 1] = ddraft
                        ddrold = ddraft
                        ddraft = ddraft + ddr[L - 1]
                        smdn = smdn + smdnl[L - 1]
                        qmdn = qmdn + qmdnl[L - 1]
                        smomdn[XYM] = smomdn[XYM] + smomdnl[XYM, L - 1]
                        qmomdn[XYM] = qmomdn[XYM] + qmomdnl[XYM, L - 1]
                        umdn = umdn + umdnl[:, L - 1]
                        vmdn = vmdn + vmdnl[:, L - 1]
                dsm[ldmin] = dsm[ldmin] + smdn
                dsmom[XYM, ldmin] = dsmom[XYM, ldmin] + smomdn[XYM]
                dqm[ldmin] = dqm[ldmin] + qmdn
                dqmom[XYM, ldmin] = dqmom[XYM, ldmin] + qmomdn[XYM]
                tdnl[ldmin] = smdn * plk[ldmin] / (ddraft + TEENY)
                qdnl[ldmin] = qmdn / (ddraft + TEENY)
                dum[:, ldmin] = dum[:, ldmin] + umdn
                dvm[:, ldmin] = dvm[:, ldmin] + vmdn
                dm[ldmin] = dm[ldmin] + ddraft
            loc = locals()
            rec(5, ic, 0, lmin)

            # ---------------------------------------------------------------------- subsidence
            if ldmin > lmin:
                ldmin = lmin
            cm = zl()
            smt, qmt = zl(), zl()
            for L in range(ldmin, lmax + 1):
                cm[L] = cm[L - 1] - dm[L] - dmr[L]
                smt[L] = sm[L]
                qmt[L] = qm[L]
            cm[lmax:LM + 1] = 0
            ksub = 1
            for l in range(ldmin, lmax):
                if +cm[l] > airm[l + 1] + dmr[l + 1]:
                    ksub = max(ksub, 1 + int((+cm[l] - dmr[l + 1]) / airm[l + 1]))
                elif -cm[l] > airm[l] + dmr[l]:
                    ksub = max(ksub, 1 + int((-cm[l] - dmr[l]) / airm[l]))
            ksub = min(ksub, 2)
            if mut.get("ksub1"):
                ksub = 1
            if ksub == 2:
                bump("ksub_2")
            byksub = 1.0 / ksub
            sumu = _seqsum_rows(um, ldmin, lmax)
            sumv = _seqsum_rows(vm, ldmin, lmax)
            sumdp = F(0.0)
            for L in range(ldmin, lmax + 1):
                sumdp = sumdp + airm[L]
            alpha = F(0.0)
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
                    betau = F(0.0)
                if alpha < 0.:
                    alphau = F(0.0)
                um[:, L] = um[:, L] + ra * (-alphau * um[:, L] + betau * um[:, L + 1] + dum[:, L])
                vm[:, L] = vm[:, L] + ra * (-alphau * vm[:, L] + betau * vm[:, L + 1] + dvm[:, L])
                alpha = beta
            sumu1 = _seqsum_rows(um, ldmin, lmax)
            sumv1 = _seqsum_rows(vm, ldmin, lmax)
            for k in range(nks):
                um[k, ldmin:lmax + 1] = um[k, ldmin:lmax + 1] - (sumu1[k] - sumu[k]) * airm[ldmin:lmax + 1] / sumdp
                vm[k, ldmin:lmax + 1] = vm[k, ldmin:lmax + 1] - (sumv1[k] - sumv[k]) * airm[ldmin:lmax + 1] / sumdp
            nsub = lmax - ldmin + 1
            cmneg = zl()
            cmneg[ldmin:lmax] = -cm[ldmin:lmax] * byksub
            cmneg[lmax] = 0.
            sl = slice(ldmin, lmax + 1)
            for it in range(1, ksub + 1):
                ml = zl()
                ml[sl] = airm[sl] + dmr[sl] * byksub
                sm[sl] = sm[sl] + dsmr[sl] * byksub
                smom[:, sl] = smom[:, sl] + dsmomr[:, sl] * byksub
                _, _, ierrt, lerrt = adv.adv1d(sm[sl], smom[:, sl], ml[sl].copy(), cmneg[sl].copy(), False, ZDIR)
                sm[sl] = sm[sl] + dsm[sl] * byksub
                smom[:, sl] = smom[:, sl] + dsmom[:, sl] * byksub
                ierr = max(ierrt, ierr)
                lerr = max(lerrt + ldmin - 1, lerr)
                ml = zl()
                ml[sl] = airm[sl] + dmr[sl] * byksub
                qm[sl] = qm[sl] + dqmr[sl] * byksub
                qmom[:, sl] = qmom[:, sl] + dqmomr[:, sl] * byksub
                mlq = ml[sl].copy()
                lqs = {}
                _, _, ierrt, lerrt = adv.adv1d(qm[sl], qmom[:, sl], mlq, cmneg[sl].copy(),
                                               not mut.get("noqlimit"), ZDIR, lqs)
                for kk, vv in lqs.items():
                    br["limitq_" + kk] += vv
                qm[sl] = qm[sl] + dqm[sl] * byksub
                qmom[:, sl] = qmom[:, sl] + dqmom[:, sl] * byksub
                ierr = max(ierrt, ierr)
                lerr = max(lerrt + ldmin - 1, lerr)
            loc = locals()
            rec(6, ic, 0, lmin)
            # ---------------------------------------------------------------------- diagnostics
            for L in range(ldmin, lmax + 1):
                fcdh = F(0.0)
                if L == lmax:
                    fcdh = cdhsum - cdhsum1 + cdhm
                fcdh1 = F(0.0)
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
            sm1[1:LM + 1] = sm[1:LM + 1]
            qm1[1:LM + 1] = qm[1:LM + 1]

            # ---------------------------------------------------------------------- precipitation
            cond[lmax] = cond[lmax] + condv[lmax]
            if ple[lmin] - ple[lmax + 1] >= 450.:
                bump("deep_precip_partition")
                for L in range(lmax, lmin - 1, -1):
                    if cond[L] < condp[L]:
                        condp[L] = cond[L]
                    fclw = F(0.0)
                    if cond[L] > 0:
                        fclw = (cond[L] - condp[L]) / cond[L]
                    if svlatl[L] > 0 and svlatl[L] != vlat[L]:
                        heat1[L] = heat1[L] + (svlatl[L] - vlat[L]) * svwmxl[L] * airm[L] * BYSHA / fmc1
                        bump("svlat_phase_diff")
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
                bump("top_phase_fix")
                fssum = F(0.0)
                if abs(plk[lmax] * sm[lmax]) > TEENY and (lhp[lmax] - vlat[lmax]) * prcp * BYSHA < 0:
                    fssum = -((lhp[lmax] - vlat[lmax]) * prcp * BYSHA / (plk[lmax] * sm[lmax]))
                sm[lmax] = sm[lmax] + (lhp[lmax] - vlat[lmax]) * prcp * BYSHA / plk[lmax]
                smom[:, lmax] = smom[:, lmax] * (1. - fssum)
            dphase[lmax] = dphase[lmax] + (cdhsum - cdhsum1 + cdhm) * fmc1
            if ple[lmax + 1] > 700:
                dphashlw[lmax] = dphashlw[lmax] + (cdhsum - cdhsum1 + cdhm) * fmc1
            if ple[lmin] - ple[lmax + 1] >= 450:
                dphadeep[lmax] = dphadeep[lmax] + (cdhsum - cdhsum1 + cdhm) * fmc1
            loc = locals()
            rec(7, ic, 0, lmin)
            for L in range(lmax - 1, 0, -1):
                fcloud = F(hp.mc_cloud_fraction(F(L), tl[L], pl[L], ccm[L], wcu[L], ple[lmin], ple[L + 2],
                                                ple[lmax + 1], ccm[lmin], wcu[lmin], F(lmin), F(lmax), CCMUL, CCMUL2,
                                                dtsrc=dtsrc))
                if fcloud < 0.:
                    raise RuntimeError("MSTCNV: negative cloud cover")
                fevap = .5 * ccm[L] * byam[L + 1]
                if L < lmin:
                    fevap = .5 * ccm[lmin] * byam[lmin + 1]
                if fevap > .5:
                    fevap = F(.5)
                cldmcl[L + 1] = min(cldmcl[L + 1] + fcloud * fmc1, fmc1)
                cldref = cldmcl[L + 1]
                if ple[lmax + 1] > 700 and cldref > cldslwij:
                    cldslwij = cldref
                if ple[lmin] - ple[lmax + 1] >= 450 and cldref > clddepij:
                    clddepij = cldref
                told = smold[L] * plk[L] * byam[L]
                told1 = smold[L + 1] * plk[L + 1] * byam[L + 1]
                precnvl[L + 1] = precnvl[L + 1] + prcp * BYGRAV
                lh, mcloud, h1 = hp.mc_precip_phase(F(L), airm[L], fevap, lhp[L + 1], prcp, told, told1, vlat[L],
                                                     cond[L], F(lmin), heat1[L], mc_revp_abv_cldbase=mc_revp)
                lhp[L], mcloud, heat1[L] = F(lh), F(mcloud), F(h1)
                lhx = lhp[L]
                dqsum = F(0.0)
                fprcp = F(0.0)
                slh = lhx * BYSHA
                if prcp > 0.:
                    if mcloud > 0:
                        dqs, fprcp = dq.get_dq_evap(smold[L], qmold[L], plk[L], airm[L], lhx, pl[L],
                                                    prcp * airm[L] / mcloud)
                        dqsum = F(dqs)
                    dqsum = dqsum * mcloud * byam[L]
                    prcp = prcp - dqsum
                    qm[L] = qm[L] + dqsum
                fssum = F(0.0)
                if abs(plk[L] * sm[L]) > TEENY and slh * dqsum + heat1[L] > 0:
                    fssum = (slh * dqsum + heat1[L]) / (plk[L] * sm[L])
                sm[L] = sm[L] - (slh * dqsum + heat1[L]) / plk[L]
                smom[:, L] = smom[:, L] * (1. - fssum)
                fcdh1 = F(0.0)
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
            loc = locals()
            rec(8, ic, 0, lmin)
            mc1 = False

    # =========================================================================== after the cloud-base loop
    cdhm = cdhm
    if lmcmin > 0:
        fssl[1:lmcmax + 1] = 1 - fmc1
        sumaj = F(0.0)
        sumdp = F(0.0)
        for L in range(lmcmin, lmcmax + 1):
            sumdp = sumdp + airm[L] * fmc1
            sumaj = sumaj + dgdsm[L]
        for L in range(lmcmin, lmcmax + 1):
            dgdsm[L] = dgdsm[L] - sumaj * airm[L] * fmc1 / sumdp
            sm[L] = sm[L] - sumaj * airm[L] / (sumdp * plk[L])
        airxl = F(0.0)
        for L in range(lmcmin, lmcmax + 1):
            airxl = airxl + mcflx[L]
    lmin = 0
    loc = locals()
    rec(9, 0, 0, 0)

    # =========================================================================== optical thickness
    wconst = WMU * (1. - pearth) + WMUL * pearth
    wconst = wmu_mult * wconst
    wmsum = F(0.0)
    qlmc, qimc = zl(), zl()
    wmctwp = wmclwp = F(0.0)
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
            bump("optical_anvil")
            fcld = cldmcl[L] + F(f4(1.0e-20))
            tem = 1e5 * svwmxl[L] * airm[L] * BYGRAV
            wtem = 1e5 * svwmxl[L] * pl[L] / (fcld * tl[L] * RGAS)
            if svlatl[L] == LHE and svwmxl[L] / fcld >= wconst * 1e-3:
                wtem = 1e2 * wconst * pl[L] / (tl[L] * RGAS)
            if wtem < 1e-10:
                wtem = F(1e-10)
            mndo = 59.68 / (rwcldox * rwcldox * rwcldox)
            mndl = F(174.0)
            mndi = F(0.06417127)
            mcdncw = mndo * (1. - pearth) + mndl * pearth
            mcdnci = mndi
            rcld, tau = _ANVIL(svlatl[L], rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem,
                                                    wtem)
            taumcl[L] = F(tau)
            rclde = F(rcld) / bybr
            csizel[L] = rclde
        if taumcl[L] < 0. and cldmcl[L] <= 0.:
            taumcl[L] = 0.
    if lmcmax <= 1:
        for L in range(1, LM + 1):
            if pl[L] < pl[dcl]:
                u00l[L] = 0.
    cnvmmrl = zl()
    for L in range(1, LM + 1):
        cnvmmrl[L] = condmmr[L] if cldmcl[L] > 0.0 else 0.0

    o = dict(ierr=ierr, lerr=lerr, lmcmin=lmcmin, lmcmax=lmcmax, prcpmc=prcpmc, cldslwij=cldslwij,
             clddepij=clddepij, airxl=airxl, prheat=prheat, wmsum=wmsum, wmctwp=wmctwp, wmclwp=wmclwp, fmc1=fmc1,
             mccont=mccont)
    named = dict(tl=tl, sm=sm, qm=qm, fssl=fssl, cldmcl=cldmcl, taumcl=taumcl, svlatl=svlatl, svlat1=svlat1,
                 svwmxl=svwmxl, csizel=csizel, condpt=condpt, vsubl=vsubl, tpsav=tpsav, mcflx=mcflx, dgdsm=dgdsm,
                 dgdeep=dgdeep, dgshlw=dgshlw, dphase=dphase, dphadeep=dphadeep, dphashlw=dphashlw, dtotw=dtotw,
                 dqcond=dqcond, dgdqm=dgdqm, dqmtotal=dqmtotal, dqmshlw=dqmshlw, dqmdeep=dqmdeep,
                 dqctotal=dqctotal, dqcshlw=dqcshlw, dqcdeep=dqcdeep, ddmflx=ddmflx, tdnl=tdnl, qdnl=qdnl,
                 u00l=u00l, qlmc=qlmc, qimc=qimc, cnvmmrl=cnvmmrl, condmmr=condmmr)
    for k, a in named.items():
        o[k] = np.array(a[1:LM + 1])
    o["lhp"] = np.array(lhp[1:LM + 2])
    o["precnvl"] = np.array(precnvl[1:LM + 2])
    o["smom"] = np.array(smom[:, 1:LM + 1])
    o["qmom"] = np.array(qmom[:, 1:LM + 1])
    o["um"] = np.array(um[:, 1:LM + 1])
    o["vm"] = np.array(vm[:, 1:LM + 1])
    return o
