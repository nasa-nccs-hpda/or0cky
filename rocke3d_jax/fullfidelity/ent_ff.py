"""D169: Ent (vegetation) per-sub-iteration exports for GHY, ported from the real source (read-only
modelE2_planet_2.0/model/Ent), stage 1 of the Ent port.

Scope (see scoping/D169_ENT_ENTRY.md for the call-graph evidence): in the P2SAoM40 build (OPTS_Ent = ONLINE=YES
PS_MODEL=FBB PFT_MODEL=ENT; no RAD_MODEL=GORT, no PS_BVOC, no ENT_WATER_STRESS_4, do_soilresp=1,
do_phenology_activegrowth=0, do_frost_hardiness=1) one GHY sub-iteration calls

    ent_set_forcings -> ent_run -> ent_integrate:
        clim_stats                       (10-day running means; Sacclim)          [ported here: export-relevant part]
        photosynth_cond  (canopyspitters.f + FBBphotosynthesis.f + water_stress3) [ported here]
        soil_bgc                         (soil respiration, carbon pools)          [NOT ported: no feedback to exports]
        summarize_entcell / summarize_patch                                         [ported here]
    ent_get_exports -> cnc, betadl(6), TRANS_SW, Ci, GPP, lai, IPP

and once per GHY call (before the loop) ent_get_exports -> ws_can, shc_can, fv, height, albedo(6).

Source files and lines this module mirrors (all under model/Ent unless noted):
    canopyspitters.f   photosynth_cond 57-334, canopyfluxes 336-432, photosynth_sunshd 434-494,
                       canopy_rad_setup 687-762, canopy_rad 764-833, canopy_transmittance 835-908, qsimp 910-978, trapzd 980-1046
    FBBphotosynthesis.f  pscondleaf 97-128, Photosynth_analyticsoln 152-318, calc_CO2compp 496-513, Q10fn 658,
                       calc_Pspar 1016-1082, frost_hardiness 1105-1135, BallBerry 1137-1150, ci_cubic (Newton/bisection,
                       USE_NR_SOLVER_FOR_FBB default) 1381-1470, A_eqn 1472, A_eqn_0 1505, rtsafe 1535-1610
    respauto_physio.f  water_stress3 334-372, Rdark 35
    phenology.f        clim_stats, running_mean 2699, photosyn_acclim 277
    patches.f          summarize_patch;  entcells.f summarize_entcell 157-378, entcell_update_shc_mosaicveg 414
    ent_mod.f          ent_set_forcings_r8_0 2498, ent_get_exports_r8_0 2807, ent_cell_unpack/copy_*_vars (restart layout)
    ent_pfts_ENT.f (pfpar), FBBpfts_ENT.f (pftpar), ent_const.f, physutil.f (QSAT)

Fortran evaluation order is kept term by term (left to right), the real-literal traps are reproduced
(0.21 in calc_CO2compp and 2.22222e-6 in photosyn_acclim are single precision), and exp/pow call the Intel libimf
scalar functions when that runtime exists (same mechanism as intel_libm_ff.py), else math.exp/math.pow.

What is NOT ported (documented in the ledger): soil_bgc, Respauto_NPP_Clabile (carbon pools), the llspan/turnover_amp,
betad_10d, soiltemp_10d, sgdd bookkeeping of clim_stats, and the daily prescribed update (update_vegetation_data).
None of those changes cnc/betadl/TRANS_SW/Ci/GPP/lai/IPP in this configuration (read the code: LAI is prescribed,
calc_Pspar ignores llspan, the carbon pools are not read by photosynth_cond).
"""
import ctypes
import math
import os

import numpy as np

# --------------------------------------------------------------------------- libm (Intel libimf if present)
_DEFAULT_IMF = ("/panfs/ccds02/app/modules/intel/platform/x86_64/rhel/8.6/2020Update4/"
                "compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin")
_imf = {"tried": False, "exp": None, "pow": None}


def _load_imf():
    if _imf["tried"]:
        return
    _imf["tried"] = True
    d = os.environ.get("INTEL_LIBIMF_DIR", _DEFAULT_IMF)
    try:
        ctypes.CDLL(os.path.join(d, "libintlc.so.5"), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(os.path.join(d, "libimf.so"))
        lib.exp.restype = ctypes.c_double
        lib.exp.argtypes = [ctypes.c_double]
        lib.pow.restype = ctypes.c_double
        lib.pow.argtypes = [ctypes.c_double, ctypes.c_double]
        _imf["exp"], _imf["pow"] = lib.exp, lib.pow
    except OSError:
        pass


def imf_available():
    _load_imf()
    return _imf["exp"] is not None


_USE_IMF = True


def set_use_imf(flag):
    global _USE_IMF
    _USE_IMF = bool(flag)


def fexp(x):
    if _USE_IMF:
        _load_imf()
        if _imf["exp"] is not None:
            return _imf["exp"](x)
    return math.exp(x)


def fpow(x, y):
    if _USE_IMF:
        _load_imf()
        if _imf["pow"] is not None:
            return _imf["pow"](x, y)
    return math.pow(x, y)


F32 = lambda v: float(np.float32(v))   # a Fortran real*4 literal promoted to real*8

# --------------------------------------------------------------------------- constants (ent_const.f, GHY)
KELVIN = 273.15        # ent_const tfrz
TFRZ = 273.15
GASC = 8.314510
EPS = 1.0e-8
EPS2 = 1.0e-12
SWTOPAR = 4.05
O2FRAC = F32(0.20900)   # canopyspitters.f: real*8,parameter :: O2frac=.20900 (NO d0: single precision literal; confirmed in canopyspitters.o .rodata)
N_DEPTH = 6
N_PFT = 16
N_BANDS = 6
UNDEF = -1.0e30

# canopy radiation constants (canopy_rad_setup)
SIGMA = 0.2
KDF = 0.71

# photosynthesis constants (photcondmod)
KC = 30.0
KO = 3.0e4
KCQ10 = 2.1
KOQ10 = 1.2

# ---- pfpar (ent_pfts_ENT.f:144-178; pftype constructor order: pst, woody, leaftype, hwilt, sstar, swilt, nf, sla, r,
# lrage, woodage, lit_C2N, lignin, croot_ratio, phenotype, b1Cf..b2Ht).  Only the fields the export path reads are kept.
# NOTE (found while transcribing): pst is 1 (C3) for ALL 16 entries, including the C4 grass (12) and crops (15): the C4
# branch of Photosynth_analyticsoln, which tests pfpar(pft)%pst, is dead in this build.
PFPAR_PST = [1] * 16
PFPAR_LEAFTYPE = [1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 3, 3, 3, 3, 1, 1]
PFPAR_SSTAR = [.60, .60, .50, .50, .50, .50, .45, .55, .50, .40, .30, .30, .30, .60, .45, .50]
PFPAR_SWILT = [.29, .29, .25, .25, .29, .29, .22, .25, .30, .22, .10, .10, .10, .27, .27, .29]
PFPAR_PHENOTYPE = [1, 1, 1, 1, 2, 2, 3, 2, 4, 4, 4, 4, 5, 4, 4, 4]
# ---- pftpar (FBBpfts_ENT.f): pst, PARabsorb, Vcmax, m, b, Nleaf
PFTPAR_VCMAX = [75., 50., 51., 43., 60., 51., 56.4, 43., 33., 17., 43., 24., 60., 43., 50., 51.]
PFTPAR_M = [9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 11., 5., 11., 9., 9., 9.]
PFTPAR_B = [.002] * 10 + [.008, .002, .008, .002, .002, .002]
# ---- alamax/alamin (ent_pfts_ENT.f:461-477) for the canopy heat capacity
ALAMAX = [6., 6., 8., 8., 6., 6., 4., 6., 1.5, 2.5, 2., 2., 2., 2., 4.5, 6., 0., 0.]
ALAMIN = [5., 5., 6., 6., 1., 1., 1., 1., 1., 1., 1., 1., 0.1, 1., 1., 1., 0., 0.]
SHW = F32(4185.)       # ent_const: real*8,parameter :: shw = 4185.  (real*4 literal, exactly representable)
RHOW = 1.0e3
COVER_SAND, COVER_DIRT = 17, 18


# --------------------------------------------------------------------------- data structures
class Cohort:
    __slots__ = ("pft", "n", "lai", "h", "fracroot", "Sacclim", "llspan", "turnover_amp", "betad_10d", "stressH2O",
                 "stressH2Ol", "GCANOPY", "Ci", "GPP", "IPP", "Vcmax")


class Patch:
    __slots__ = ("area", "albedo", "soil_type", "cohorts", "TRANS_SW", "LAI", "h", "GCANOPY", "GPP", "IPP", "Ci",
                 "betadl", "pcLAI")


class EntCell:
    """The part of entcelltype/patch/cohort that the export path reads or writes."""

    def __init__(self):
        self.patches = []     # oldest -> youngest
        self.fall = 0
        self.airtemp_10d = 0.0
        self.par_10d = 0.0
        self.gdd = 0.0
        self.ncd = 0.0
        self.daylength = [0.0, 0.0]
        self.Qf = 0.0
        # forcings (ent_set_forcings)
        self.TairC = self.TcanopyC = self.P_mbar = self.Ca = self.Ch = self.U = 0.0
        self.IPARdif = self.IPARdir = self.CosZen = self.fwet_canopy = 0.0
        self.Soilmoist = np.zeros(N_DEPTH)
        self.fice = np.zeros(N_DEPTH)
        # summaries (ecp%...)
        self.LAI = self.h = self.fv = self.Ci = self.GCANOPY = self.GPP = self.IPP = self.TRANS_SW = 0.0
        self.betadl = np.zeros(N_DEPTH)
        self.albedo = np.zeros(N_BANDS)
        self.area = 0.0
        self.heat_capacity = 0.0


def _read_vars(buf, dc, n):
    return buf[dc:dc + n], dc + n


def unpack_cell(buf):
    """ent_cell_unpack (ent_mod.f) for one restart column of `ent_state` (1023 doubles).  Returns None for an absent cell.
    Layout: np, nc(1:np), cell vars (25), then per patch: patch vars (52) and per cohort cohort vars (46); see
    copy_cell_vars / copy_patch_vars / copy_cohort_vars (ent_mod.f).  Fields the export path does not read are skipped."""
    np_ = int(round(buf[0]))
    if np_ <= 0:
        return None
    nc = [int(round(v)) for v in buf[1:1 + np_]]
    dc = 1 + np_
    cell = EntCell()
    cv = buf[dc:dc + 25]
    # soil_texture(5) soiltemp_10d airtemp_10d paw_10d par_10d gdd ncd daylength1 daylength2 fall lai soil_Phi soil_dry Qf Soilmp(6) sgdd
    cell.airtemp_10d = float(cv[6])
    cell.par_10d = float(cv[8])
    cell.gdd = float(cv[9])
    cell.ncd = float(cv[10])
    cell.daylength = [float(cv[11]), float(cv[12])]
    cell.fall = int(round(cv[13]))
    cell.Qf = float(cv[17])
    dc += 25
    for ip in range(np_):
        pv = buf[dc:dc + 52]
        p = Patch()
        # age area Ci Tpool(1,:,1)(12) Tpool(2,:,1)(12) soil_type GCANOPY albedo(6) Reproduction(16) lai
        p.area = float(pv[1])
        p.soil_type = int(round(pv[27]))
        p.albedo = np.array(pv[29:35], dtype=float)
        p.cohorts = []
        p.TRANS_SW = 1.0
        p.LAI = p.h = p.GCANOPY = p.GPP = p.IPP = p.Ci = 0.0
        p.betadl = np.zeros(N_DEPTH)
        dc += 52
        for ic in range(nc[ip]):
            cvv = buf[dc:dc + 46]
            c = Cohort()
            # pft n nm lai h dbh C_fol C_froot C_hw fracroot(6) Ci gcanopy C_fol N_fol C_sw N_sw C_hw N_hw C_lab N_lab
            # C_froot N_froot C_croot N_croot C_growth C_growth_flux C_total llspan turnover_amp Sacclim Ntot crown_dx
            # phenofactor phenofactor_c phenofactor_d phenostatus betad_10d CB_d senescefrac stressH2O NPP
            c.pft = int(round(cvv[0]))
            c.n = float(cvv[1])
            c.lai = float(cvv[3])
            c.h = float(cvv[4])
            c.fracroot = np.array(cvv[9:15], dtype=float)
            c.llspan = float(cvv[32])
            c.turnover_amp = float(cvv[33])
            c.Sacclim = float(cvv[34])
            c.betad_10d = float(cvv[41])
            c.stressH2O = float(cvv[44])
            c.stressH2Ol = np.zeros(N_DEPTH)
            c.GCANOPY = c.GPP = c.IPP = 0.0
            c.Ci = 0.0
            c.Vcmax = 0.0
            p.cohorts.append(c)
            dc += 46
        cell.patches.append(p)
    summarize_entcell(cell)
    return cell


# --------------------------------------------------------------------------- summaries
def summarize_patch(p):
    """patches.f summarize_patch (only the fields the exports read)."""
    p.LAI = 0.0
    p.h = 0.0
    p.Ci = 0.0
    p.GCANOPY = 0.0
    p.GPP = 0.0
    p.IPP = 0.0
    betad = 0.0
    p.betadl = np.zeros(N_DEPTH)
    if not p.cohorts:
        return
    nsum = 0.0
    for c in p.cohorts:
        nc = c.n
        nsum = nsum + nc
        p.LAI = p.LAI + c.lai
        p.h = p.h + c.h * nc
        p.Ci = p.Ci + c.Ci * c.lai
        p.GCANOPY = p.GCANOPY + c.GCANOPY
        p.GPP = p.GPP + c.GPP
        p.IPP = p.IPP + c.IPP
        betad = betad + c.stressH2O * c.lai
        for ia in range(N_DEPTH):
            p.betadl[ia] = p.betadl[ia] + c.stressH2Ol[ia] * c.lai
    if nsum > 0.0:
        p.h = p.h / nsum
    if p.LAI > 0.0:
        p.Ci = p.Ci / p.LAI
        p.betadl = p.betadl / p.LAI
    else:
        p.Ci = 0.0
        p.betadl = np.zeros(N_DEPTH)


def _extract_pfts(p):
    """patch_extract_pfts: vfraction(18)"""
    v = np.zeros(18)
    density = 0.0
    for c in p.cohorts:
        v[c.pft - 1] = v[c.pft - 1] + c.n
        density = density + c.n
    if density > 0.0:
        v = v / density
    else:
        if p.soil_type == 1:
            v[COVER_SAND - 1] = 1.0
        else:
            v[COVER_DIRT - 1] = 1.0
    return v


def giss_shc(mean_lai):
    return (.010 + .002 * mean_lai + .001 * (mean_lai * mean_lai)) * SHW * RHOW


def summarize_entcell(e):
    """entcells.f summarize_entcell 157-378 + entcell_update_shc_mosaicveg (do_geo false)."""
    e.fv = 0.0
    e.LAI = 0.0
    e.h = 0.0
    e.Ci = 0.0127
    e.GCANOPY = 0.0
    e.GPP = 0.0
    e.IPP = 0.0
    e.TRANS_SW = 0.0
    e.betadl = np.zeros(N_DEPTH)
    e.albedo = np.zeros(N_BANDS)
    fa = 0.0
    laifasum = 0.0
    for p in e.patches:
        summarize_patch(p)
        fa = fa + p.area
        laifa = p.area * p.LAI
        laifasum = laifasum + laifa
        e.LAI = e.LAI + p.LAI * p.area
        e.h = e.h + p.h * laifa
        e.Ci = e.Ci + p.Ci * p.area
        e.GCANOPY = e.GCANOPY + p.GCANOPY * p.area
        e.GPP = e.GPP + p.GPP * p.area
        e.IPP = e.IPP + p.IPP * p.area
        for ia in range(N_BANDS):
            e.albedo[ia] = e.albedo[ia] + p.albedo[ia] * p.area
        for ia in range(N_DEPTH):
            e.betadl[ia] = e.betadl[ia] + p.betadl[ia] * p.area
        e.TRANS_SW = e.TRANS_SW + p.TRANS_SW * p.area
        if p.cohorts:
            e.fv = e.fv + p.area
    if e.patches:
        if laifasum > 0.0:
            e.h = e.h / laifasum
        else:
            e.h = 0.0
        e.LAI = e.LAI / fa
        e.Ci = e.Ci / fa
        e.GCANOPY = e.GCANOPY / fa
        e.GPP = e.GPP / fa
        e.IPP = e.IPP / fa
        for ia in range(N_BANDS):
            e.albedo[ia] = e.albedo[ia] / fa
        e.TRANS_SW = e.TRANS_SW / fa
        for ia in range(N_DEPTH):
            e.betadl[ia] = e.betadl[ia] / fa
        e.area = fa
    # entcell_update_shc_mosaicveg
    vfraction = np.zeros(18)
    for p in e.patches:
        vp = _extract_pfts(p)
        vfraction = vfraction + vp * p.area
    lai = 0.0
    fsum = 0.0
    for pft in range(1, N_PFT + 1):
        lai = lai + .5 * (ALAMAX[pft - 1] + ALAMIN[pft - 1]) * vfraction[pft - 1]
        fsum = fsum + vfraction[pft - 1]
    lai = lai / fsum if fsum > EPS else 0.0
    e.heat_capacity = giss_shc(lai)


# --------------------------------------------------------------------------- photosynthesis (FBBphotosynthesis.f)
class _PsPar:
    """module-level `pspar` (photcondmod) plus the SAVEd locals of ci_cubic and Photosynth_analyticsoln."""

    def __init__(self):
        self.pft = 0
        self.Vcmax = self.Kc = self.Ko = self.Gammastar = self.m = self.b = 0.0
        self.first_call = True
        self.reset_ci_cubic1 = True
        self.Ac = self.As = 0.0
        # saved locals
        self.a1c = 1.0e30
        self.f1c = -1.0e30
        self.Ra = self.bb = self.K = self.gamol = self.A_d_asymp = self.x1 = self.x2save = self.xacc = 0.0


def q10fn(q10, t):
    return fpow(q10, (t - 25.0) / 10.0)


def calc_co2compp(o2, kc, ko, tl):
    # Gammastar = 0.5d0*(Kc/Ko)*0.21*O2 ; the literal 0.21 is single precision (no PS_BVOC)
    return 0.5 * (kc / ko) * F32(0.21) * o2


def frost_hardiness(sacclim):
    tacclim = -5.93
    a_const = 0.1
    if sacclim > tacclim:
        f = a_const * (sacclim - tacclim)
        if f > 1.0:
            f = 1.0
        return f
    elif sacclim == UNDEF:
        return 1.0
    return 0.01


def calc_pspar(ps, pft, Pa, Tl, O2pres, stressH2O, sacclim):
    facclim = frost_hardiness(sacclim)
    fparlimit = 1.0
    p = pft - 1
    ps.pft = pft
    ps.Vcmax = PFTPAR_VCMAX[p] * q10fn(2.21, Tl) * facclim * fparlimit
    ps.Kc = KC * q10fn(KCQ10, Tl)
    ps.Ko = KO * q10fn(KOQ10, Tl)
    ps.Gammastar = calc_co2compp(O2pres, ps.Kc, ps.Ko, Tl)
    ps.m = stressH2O * PFTPAR_M[p]
    ps.b = PFTPAR_B[p]
    ps.first_call = True
    ps.reset_ci_cubic1 = True


def ball_berry(Anet, rh, cs, ps):
    if cs <= 0.0:
        raise RuntimeError("BallBerry: cs <= 0")
    gsw = ps.m * Anet * rh / cs + ps.b
    if gsw < ps.b:
        gsw = ps.b
    return gsw


def _a_eqn(A, Ra, b, K1, gamol, ca, a1, f1, Rd, want_df):
    K = K1 if A > 0.0 else 0.0
    cs = ca - A * Ra
    byAKbcs = 1.0 / (A * K + b * cs)
    ci = cs * (1.0 - A * byAKbcs)
    bycif1 = 1.0 / (ci + f1)
    f = A - (a1 * (ci - gamol) * bycif1 - Rd)
    if not want_df:
        return f, 0.0
    dci = -Ra * (1.0 - A * byAKbcs) + cs * (-byAKbcs + A * byAKbcs * byAKbcs * (K - b * Ra))
    df = 1 - a1 * (f1 + gamol) * bycif1 * bycif1 * dci
    return f, df


def _rtsafe(x1, x2, xacc, Ra, b, K, gamol, ca, a1, f1, Rd):
    MAXIT = 100
    fl, _ = _a_eqn(x1, Ra, b, K, gamol, ca, a1, f1, Rd, False)
    fh, _ = _a_eqn(x2, Ra, b, K, gamol, ca, a1, f1, Rd, False)
    if (fl > 0.0 and fh > 0.0) or (fl < 0.0 and fh < 0.0):
        return -1.0e30
    if fl == 0.0:
        return x1
    elif fh == 0.0:
        return x2
    elif fl < 0.0:
        xl, xh = x1, x2
    else:
        xh, xl = x1, x2
    rts = x1
    dxold = abs(x2 - x1)
    dx = dxold
    f, df = _a_eqn(rts, Ra, b, K, gamol, ca, a1, f1, Rd, True)
    for j in range(1, MAXIT + 1):
        if (((rts - xh) * df - f) * ((rts - xl) * df - f) >= 0.0) or (abs(2.0 * f) > abs(dxold * df)):
            dxold = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return rts
        else:
            dxold = dx
            dx = f / df
            temp = rts
            rts = rts - dx
            if temp == rts:
                return rts
        if abs(dx) < xacc:
            return rts
        f, df = _a_eqn(rts, Ra, b, K, gamol, ca, a1, f1, Rd, True)
        if f < 0.0:
            xl = rts
        else:
            xh = rts
    raise RuntimeError("rtsafe exceeding maximum iterations")


def ci_cubic(ps, ca, rh, gb, Pa, Rd, a1, f1):
    S_ATM = 1.37
    S_STOM = 1.65
    if ps.reset_ci_cubic1:
        ps.reset_ci_cubic1 = False
        ps.Ra = 1 / gb * S_ATM
        ps.bb = ps.b / S_STOM
        ps.K = ps.m * rh / S_STOM
        ps.gamol = ps.Gammastar * 1.0e06 / Pa
        ps.A_d_asymp = -ps.bb * ca / (ps.K - ps.bb * ps.Ra)
        ps.x1 = -Rd
        ps.x2save = ca / ps.Ra
        x2tmp = ps.bb * ca / (1.0 - ps.K + ps.bb * ps.Ra)
        if x2tmp > 0.0:
            ps.x2save = min(ps.x2save, x2tmp)
        x2tmp = ps.A_d_asymp
        if x2tmp > 0.0:
            ps.x2save = min(ps.x2save, x2tmp)
        ps.x2save = ps.x2save - .0000001
        ps.xacc = .0001
    x2 = min(ps.x2save, a1 - Rd)
    return _rtsafe(ps.x1, x2, ps.xacc, ps.Ra, ps.bb, ps.K, ps.gamol, ca, a1, f1, Rd)


def photosynth_analyticsoln(ps, IPAR, ca, Tl, Pa, rh, gb):
    """Returns (ci, gs, Atot, Rd, isp).  pspar%first_call state lives in `ps`."""
    O2pres = 20900.0
    alpha = .08
    Rd = 0.015 * ps.Vcmax
    if IPAR < .000001:
        Atot = 0.0
        Anet = -Rd
        cs = ca - Anet * 1.37 / gb
        gs = ball_berry(Anet, rh, cs, ps)
        ci = cs - Anet / (gs / 1.65)
        return ci, gs, Atot, Rd, 0.0
    if ps.first_call:
        # pfpar(pft)%pst == C3 for every pft (see PFPAR_PST note)
        ps.a1c = ps.Vcmax
        ps.f1c = ps.Kc * (1.0 + O2pres / ps.Ko) * 1.0e06 / Pa
        Ac = ci_cubic(ps, ca, rh, gb, Pa, Rd, ps.a1c, ps.f1c)
        ps.Ac = Ac
    else:
        Ac = ps.Ac
    a1e = IPAR * alpha
    f1e = 2 * ps.Gammastar * 1.0e06 / Pa
    if a1e < ps.a1c or f1e > ps.f1c:
        Ae = ci_cubic(ps, ca, rh, gb, Pa, Rd, a1e, f1e)
    else:
        Ae = 1.0e30
    if ps.first_call:
        As = ps.Vcmax / 2.0 - Rd
        ps.As = As
        ps.first_call = False
    else:
        As = ps.As
    Anet = min(Ae, Ac, As, 0.8 * ca * gb / 1.37)
    Atot = Anet + Rd
    if Atot < 0.0:
        Atot = 0.0
        Anet = -Rd
    cs = ca - Anet * 1.37 / gb
    gs = ball_berry(Anet, rh, cs, ps)
    ci = cs - Anet / (gs / 1.65)
    return ci, gs, Atot, Rd, 0.0


# --------------------------------------------------------------------------- canopy radiation / integration (canopyspitters.f)
class _Crp:
    __slots__ = ("sigma", "sqrtexpr", "kdf", "rhor", "kbl", "pft", "canalbedo", "LAI", "CosZen", "I0df", "I0dr")


def canopy_rad_setup(pft, CosZen, fdir, IPAR, LAI, canalbedo):
    crp = _Crp()
    sbeta = CosZen
    crp.CosZen = CosZen
    crp.I0dr = fdir * IPAR
    crp.I0df = (1.0 - fdir) * IPAR
    crp.sigma = SIGMA
    crp.kdf = KDF
    crp.sqrtexpr = math.sqrt(1.0 - crp.sigma)
    crp.kbl = 0.5 * crp.kdf / (0.8 * crp.sqrtexpr * sbeta + EPS)
    if canalbedo == 0.0:
        crp.rhor = ((1.0 - crp.sqrtexpr) / (1.0 + crp.sqrtexpr)) * (2.0 / (1.0 + 1.6 * sbeta))
        crp.canalbedo = crp.rhor
    else:
        crp.rhor = canalbedo
        crp.canalbedo = crp.rhor
    crp.pft = pft
    crp.LAI = LAI
    return crp


def canopy_rad(Lc, crp):
    sbeta = crp.CosZen
    I0df = crp.I0df
    I0dr = crp.I0dr
    if sbeta > EPS:
        Idfa = (1.0 - crp.rhor) * I0df * crp.kdf * fexp(-crp.kdf * Lc)
        Idra = (1.0 - crp.rhor) * I0dr * crp.sqrtexpr * crp.kbl * fexp(-crp.sqrtexpr * crp.kbl * Lc)
        Idrdra = (1.0 - crp.sigma) * I0dr * crp.kbl * fexp(-crp.sqrtexpr * crp.kbl * Lc)
        Isha = Idfa + (Idra - Idrdra)
        Isla = Isha + (1.0 - crp.sigma) * crp.kbl * I0dr
        fsl = fexp(-crp.kbl * Lc)
        if Isha <= 0.0:
            Isha = 0.0
        if Isla <= 0.0:
            Isla = 0.0
        if fsl < 0.0:
            fsl = 0.0
    else:
        Isha = 0.0
        Isla = 0.0
        fsl = 0.0
    return Isla, Isha, fsl


def canopy_transmittance(sbeta, fdir, crp):
    sigma, sqrtexpr, rhor, kbl, kdf, alai = crp.sigma, crp.sqrtexpr, crp.rhor, crp.kbl, crp.kdf, crp.LAI
    if sbeta > 0.0:
        absdf = (1 - fdir) * (1.0 - rhor) * kdf * fexp(-kdf * alai)
        absdr = fdir * (1.0 - rhor) * sqrtexpr * kbl * fexp(-sqrtexpr * kbl * alai)
        absdrdr = absdr * (1.0 - sigma) * kbl * fexp(-sqrtexpr * kbl * alai)
        abssh = absdf + (absdr - absdrdr)
        abssl = abssh + (1.0 - sigma) * kbl * fdir
        fracsl = fexp(-kbl * alai)
    else:
        abssh = (1.0 - rhor) * kdf * fexp(-kdf * alai)
        abssl = 0.0
        fracsl = 0.0
    if abssh < 0.0:
        abssh = 0.0
    if abssl < 0.0:
        abssl = 0.0
    if abssh > 1.0:
        abssh = 1.0
    if abssl > 1.0:
        abssl = 1.0
    return (1.0 - fracsl) * abssh + fracsl * abssl


class _Psd:
    __slots__ = ("ca", "ci", "Tc", "Pa", "rh")


def photosynth_sunshd(Lcum, crp, psd, Gb, ps):
    Isl, Ish, fsl = canopy_rad(Lcum, crp)
    ci, gssl, Asl, Rdsl, Iel = _pscondleaf(ps, Isl, psd, Gb)
    psd.ci = ci
    ci, gssh, Ash, Rdsh, Ies = _pscondleaf(ps, Ish, psd, Gb)
    psd.ci = ci
    Aleaf = fsl * Asl + (1.0 - fsl) * Ash
    gsleaf = fsl * gssl + (1.0 - fsl) * gssh
    Rdleaf = fsl * Rdsl + (1.0 - fsl) * Rdsh
    Ileaf = fsl * Iel + (1.0 - fsl) * Ies
    return Aleaf, gsleaf, Rdleaf, Ileaf


def _pscondleaf(ps, IPAR, psd, Gb):
    ci, gs, Atot, Rd, isp = photosynth_analyticsoln(ps, IPAR, psd.ca, psd.Tc, psd.Pa, psd.rh, Gb)
    return ci, gs, Atot, Rd, isp


def _trapzd(L1, L2, L1c, L2c, S, Sg, Sr, Si, N, layers, crp, psd, Gb, ps):
    if N == 1:
        A1, g1, R1, I1 = photosynth_sunshd(L1, crp, psd, Gb, ps)
        A2, g2, R2, I2 = photosynth_sunshd(L2, crp, psd, Gb, ps)
        S = 0.5 * (L2c - L1c) * (A1 + A2)
        Sg = 0.5 * (L2c - L1c) * (g1 + g2)
        Sr = 0.5 * (L2c - L1c) * (R1 + R2)
        Si = 0.5 * (L2c - L1c) * (I1 + I2)
    else:
        RCL = float(layers)
        DEL = (L2 - L1) / RCL
        X = L1 + 0.5 * DEL
        SUM = SUMg = SUMr = SUMi = 0.0
        for _ in range(layers):
            a, g, r, i_ = photosynth_sunshd(X, crp, psd, Gb, ps)
            SUM = SUM + a
            SUMg = SUMg + g
            SUMr = SUMr + r
            SUMi = SUMi + i_
            X = X + DEL
        S = 0.5 * (S + (L2 - L1) * SUM / RCL)
        Sg = 0.5 * (Sg + (L2 - L1) * SUMg / RCL)
        Sr = 0.5 * (Sr + (L2 - L1) * SUMr / RCL)
        Si = 0.5 * (Si + (L2 - L1) * SUMi / RCL)
    return S, Sg, Sr, Si


def qsimp(Xlim, crp, psd, Gb, ps):
    MAXIT = 6
    ERRLIM = 0.1
    A = 0.0
    B = Xlim
    Ac = 0.0
    Bc = crp.LAI
    OST = OS = OSTg = OSg = OSTr = OSr = OSTi = OSi = -1.0e30
    ST = STg = STr = STi = 0.0
    layers = 1
    S = Sg = Sr = Si = 0.0
    for IT in range(1, MAXIT + 1):
        ST, STg, STr, STi = _trapzd(A, B, Ac, Bc, ST, STg, STr, STi, IT, layers, crp, psd, Gb, ps)
        S = (4.0 * ST - OST) / 3.0
        Sg = (4.0 * STg - OSTg) / 3.0
        Sr = (4.0 * STr - OSTr) / 3.0
        Si = (4.0 * STi - OSTi) / 3.0
        if abs(S - OS) < ERRLIM:
            return S, Sg, Sr, Si
        OS = S
        OST = ST
        OSg = Sg
        OSTg = STg
        OSr = Sr
        OSTr = STr
        OSi = Si
        OSTi = STi
        if IT > 1:
            layers = layers * 2
    return S, Sg, Sr, Si


def qsat(TM, LH, PR):
    MWAT = 18.015
    MAIR = 28.9655
    MRAT = MWAT / MAIR
    RVAP = 1e3 * GASC / MWAT
    A = 6.108 * MRAT
    B = 1.0 / (RVAP * TFRZ)
    C = 1.0 / RVAP
    return A * fexp(LH * (B - C / max(130.0, TM))) / PR


def water_stress3(pft, thetarel, fracroot, fice):
    betad = 0.0
    betadl = np.zeros(N_DEPTH)
    sstar, swilt = PFPAR_SSTAR[pft - 1], PFPAR_SWILT[pft - 1]
    for k in range(N_DEPTH):
        s = thetarel[k]
        if s >= sstar:
            betak = 1.0
        elif s < sstar and s > swilt:
            betak = (s - swilt) / (sstar - swilt)
        else:
            betak = 0.0
        betadl[k] = (1.0 - fice[k]) * fracroot[k] * betak
        betad = betad + (1.0 - fice[k]) * fracroot[k] * betak
    if betad < EPS2:
        betad = 0.0
    return betad, betadl


def photosynth_cond(dtsec, p, cell, ps):
    """canopyspitters.f photosynth_cond for one patch (carbon-pool outputs are not computed)."""
    if not p.cohorts:
        p.TRANS_SW = 1.0
        return
    p.TRANS_SW = 1.0
    IPAR = cell.IPARdir + cell.IPARdif
    fdir = 0.0 if cell.IPARdir == 0.0 else cell.IPARdir / IPAR
    CosZen = cell.CosZen
    Pa = cell.P_mbar * 100.0
    Gb = cell.Ch * cell.U * Pa / (GASC * (cell.TairC + KELVIN))
    molconc_to_umol = GASC * (cell.TcanopyC + KELVIN) / Pa * 1.0e6
    ca_umol = cell.Ca * molconc_to_umol
    ci_umol = 0.7 * ca_umol
    TcanK = cell.TcanopyC + KELVIN
    TsurfK = cell.TairC + KELVIN
    psd = _Psd()
    psd.ca = ca_umol
    psd.ci = ci_umol
    psd.Tc = cell.TcanopyC
    psd.Pa = Pa
    psd.rh = min(1.0, max(cell.Qf, 0.0) / qsat(TcanK, 2500800.0 - 2360.0 * (TsurfK - KELVIN), Pa / 100.0))
    # patch LAI (as summarized before this call: pp%LAI is the sum over cohorts of the restart/prescribed LAI)
    pLAI = 0.0
    for c in p.cohorts:
        pLAI = pLAI + c.lai
    for c in p.cohorts:
        if c.lai > 0.0:
            c.stressH2O, c.stressH2Ol = water_stress3(c.pft, cell.Soilmoist, c.fracroot, cell.fice)
            calc_pspar(ps, c.pft, psd.Pa, psd.Tc, O2FRAC * psd.Pa, c.stressH2O, c.Sacclim)
            c.Vcmax = ps.Vcmax
            # canopyfluxes
            crp = canopy_rad_setup(c.pft, CosZen, fdir, IPAR * SWTOPAR, pLAI, p.albedo[0])
            Atot, Gsint, Rdint, Iint = qsimp(crp.LAI, crp, psd, Gb, ps)
            Gs = Gsint
            Iemis = Iint
            if PFPAR_LEAFTYPE[c.pft - 1] == 1:
                fdry = 1.0
            elif PFPAR_LEAFTYPE[c.pft - 1] == 2:
                fdry = 1.0 - min(cell.fwet_canopy, 0.333)
            else:
                fdry = 1.0
            c.GCANOPY = Gs * fdry * (GASC * TsurfK) / Pa
            c.Ci = psd.ci * psd.Pa / (GASC * (psd.Tc + KELVIN))
            c.GPP = Atot * fdry * 0.012e-6
            c.IPP = Iemis * 0.0600e-6
        else:
            c.GCANOPY = 0.0
            c.Ci = EPS
            c.GPP = 0.0
            c.IPP = 0.0
    crp = canopy_rad_setup(p.cohorts[0].pft, CosZen, fdir, IPAR, pLAI, p.albedo[0])
    p.TRANS_SW = canopy_transmittance(CosZen, fdir, crp)


# --------------------------------------------------------------------------- clim_stats (export-relevant part)
def running_mean(dtsec, numd, var, var_mean):
    zweight = fexp(-1.0 / (numd * 86400.0 / dtsec))
    return zweight * var_mean + (1.0 - zweight) * var


def photosyn_acclim(dtsec, Ta, Sacc):
    tau_inv = F32(2.22222e-6)
    return Sacc + dtsec * tau_inv * (Ta - Sacc)


def clim_stats(dtsec, cell, update_day, do_frost_hardiness=True):
    """phenology.f clim_stats: advances airtemp_10d, par_10d, daylength, gdd/ncd/fall (daily) and every cohort's Sacclim.
    llspan/turnover_amp, betad_10d, soiltemp_10d, sgdd, CB_d are not advanced (they do not enter the exports)."""
    airtemp = cell.TairC
    airtemp_10d = running_mean(dtsec, 10.0, airtemp, cell.airtemp_10d)
    par = cell.IPARdif + cell.IPARdir
    par_10d = running_mean(dtsec, 10.0, par, cell.par_10d)
    if cell.CosZen > 0.0:
        cell.daylength[1] = cell.daylength[1] + dtsec / 60.0
    gdd, ncd = cell.gdd, cell.ncd
    if update_day:
        if airtemp_10d >= 5.0:
            gdd = gdd + (airtemp_10d - 5.0)
        if airtemp_10d < 5.0:
            ncd = ncd + 1.0
        if int(np.round(cell.daylength[1])) < int(np.round(cell.daylength[0])):
            cell.fall = 1
        elif int(np.round(cell.daylength[1])) > int(np.round(cell.daylength[0])):
            cell.fall = 0
    for p in cell.patches:
        for c in p.cohorts:
            ph = PFPAR_PHENOTYPE[c.pft - 1]
            lt = PFPAR_LEAFTYPE[c.pft - 1]
            if do_frost_hardiness:
                if (ph == 1 and lt == 2) or ph == 2 or ph == 4:
                    c.Sacclim = photosyn_acclim(dtsec, airtemp_10d, c.Sacclim)
                else:
                    c.Sacclim = 25.0
    cell.airtemp_10d = airtemp_10d
    cell.par_10d = par_10d
    cell.gdd = gdd
    cell.ncd = ncd


# --------------------------------------------------------------------------- driver entry points
def set_forcings(cell, ts, tcan, Qf, pres, Ca, ch, vs, vis_rad, direct_vis_rad, cosz1, fw, w_veg, ws_veg, fice_veg,
                 tfrz=TFRZ):
    """GHY.f 2434-2467 + ent_set_forcings_r8_0.  w_veg/ws_veg/fice_veg: GHY w(1:ngm,2), ws(1:ngm,2), fice(1:ngm,2)."""
    cell.TairC = ts - tfrz
    cell.TcanopyC = tcan
    cell.Qf = Qf
    cell.P_mbar = pres
    cell.Ca = Ca * (1.0e-06) * pres * F32(100.0) / GASC / (tcan + tfrz)
    cell.Ch = ch
    cell.U = vs
    cell.IPARdif = vis_rad - direct_vis_rad
    cell.IPARdir = direct_vis_rad
    cell.CosZen = cosz1
    cell.fwet_canopy = fw
    sm = np.zeros(N_DEPTH)
    for k in range(N_DEPTH):
        if ws_veg[k] > 0.0:
            sm[k] = w_veg[k] / ws_veg[k]
    cell.Soilmoist = sm
    cell.fice = np.array(fice_veg, dtype=float)


_PS = None


def ent_run(cell, dts, update_day, ps=None, do_frost_hardiness=True):
    """ent_integrate (soil_bgc and carbon pools omitted).  `ps` defaults to ONE module-level _PsPar (Fortran module
    variable `pspar` and the SAVEd locals persist across calls and cells)."""
    global _PS
    if ps is None:
        if _PS is None:
            _PS = _PsPar()
        ps = _PS
    clim_stats(dts, cell, update_day, do_frost_hardiness)
    for p in cell.patches:
        photosynth_cond(dts, p, cell, ps)
    summarize_entcell(cell)


def get_exports(cell):
    """ent_get_exports for the per-iteration set: dict(cnc, betadl(6), trans_sw, ci, gpp, lai, ipp)."""
    return dict(cnc=cell.GCANOPY, betadl=np.array(cell.betadl), trans_sw=cell.TRANS_SW, ci=cell.Ci, gpp=cell.GPP,
                lai=cell.LAI, ipp=cell.IPP)


def call_exports(cell):
    """exports read once per GHY call (before the iteration loop): ws_can, shc_can, fv, height, albedo(6)."""
    return dict(ws_can=cell.LAI * .0001, shc_can=cell.heat_capacity, fv=cell.fv, height=cell.h,
                albedo=np.array(cell.albedo))


def load_restart_cells(restart_nc):
    """All Ent cells of a restart file as {(i, j) 1-based: EntCell}."""
    import netCDF4 as nc
    d = nc.Dataset(restart_nc)
    es = np.array(d["ent_state"][:])
    d.close()
    out = {}
    for j in range(es.shape[0]):
        for i in range(es.shape[1]):
            c = unpack_cell(es[j, i])
            if c is not None:
                out[(i + 1, j + 1)] = c
    return out
