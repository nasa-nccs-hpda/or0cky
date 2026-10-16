"""Full-fidelity port of OCNKPP.f's KPP vertical-mixing scheme -- Stage 2, D54+.

Scoped in D52 (corrected D53): `bldepth` is dead for this build (its two call sites are gated
`#ifndef OCN_GISS_TURB -> CALL KPPMIX / #else -> call bldepth`, and OCN_GISS_TURB is confirmed
not `#define`d) -- `KPPMIX` (OCNKPP.f:225-836) is the live routine at both call sites, and it
inlines its own copy of the boundary-layer-depth search (what a separate `bldepth` would have
done) and the boundary-layer mixing-coefficient computation (what a separate `blmix` would have
done); neither is called out to. `ddmix` is also dead (`LDD` is the file's `.false.` compile-time
constant), so the double-diffusion branch and its `alphaDT`/`betaDS`/`Coriol` inputs never affect
`KPPMIX`'s output -- they are accepted here for signature completeness but unused.

`KPPMIX` is called from `OCONV`'s per-column, per-iteration loop (`OCNKPP.f:1978-2334`, up to
ITER=4 times per column per `OCEANS` call -- a fixed-point iteration on HBL where `G0ML`/`S0ML`/
`UL`/`ULD` get re-diffused between iterations using the previous iteration's coefficients). This
module ports `KPPMIX` (+`z121`, its only real subroutine call) as the pure per-call function it
is: given one iteration's real `Shsq`/`dVsq`/`Ustar`/`Bo`/`Bosol`/`dbloc`/`Ritop`/`zgrid`/`hwide`/
`byhwide`/`LMIJ`, it deterministically produces `visc`/`difs`/`dift`/`ghats`/`hbl`/`kbl` -- the
outer OCONV iteration/re-diffusion orchestration is out of scope here (belongs to OCONV's own
future port).

`kmixinit` (OCNKPP.f:1178-1256) and `SW2OCEAN`'s `init_solar` (OCEAN_COM.f:321-349) are ported
directly rather than dumped: both are pure closed-form functions of fixed physical constants and
the fixed vertical grid `ZE`, computed once at model startup, not per-step -- the same
"derive fixed setup data directly" precedent as D29's RADIUS/GRAV and D47's
MESO_DIFFUSIVITY_CONST. `ocean_fkph=0.6` (decks/P2SAoM40.R, overrides the 0.1 default) feeds
`kmixinit`'s `difmiw`/`difsiw`.
"""
import numpy as np
from oadvt2_ff import _sign

LMO = 13

# KPPE module constants (OCNKPP.f:41-222)
EPSL = 1.0e-20
EPSILON = 0.1
VONK = 0.4
CONC1 = 5.0
CONAM = 1.257
CONCM = 8.380
CONC2 = 16.0
ZETAM = -0.2
CONAS = -28.86
CONCS = 98.96
CONC3 = 16.0
ZETAS = -1.0
RICR = 0.3  # OCN_GISS_TURB not defined -> Ricr=0.3d0 branch
CONCV = 1.8
CSTAR = 10.0
NNI = 890
NNJ = 480
ZMIN = -4e-7
ZMAX = 0.0
UMIN = 0.0
UMAX = 4e-2
DELTAZ = (ZMAX - ZMIN) / (NNI + 1)
DELTAU = (UMAX - UMIN) / (NNJ + 1)
RDELTAZ = 1.0 / DELTAZ
RDELTAU = 1.0 / DELTAU
# OCNKPP.f's `**(1./3.)` literals have no `d0` suffix on either operand, so Fortran computes
# 1./3. in single precision (~0.33333334326744079 once promoted to double) before using it as
# the exponent -- NOT double precision's 0.3333333333333333. Confirmed empirically: an ifort
# test program reproduces the real dumped `cg` bit-for-bit only with this single-precision
# exponent, not a "clean" double-precision 1.0/3.0 (D54 finding).
_ONE_THIRD_SP = np.float64(np.float32(1.0) / np.float32(3.0))
BVSQCON = -1e-7
RBVSQCON = 1.0 / BVSQCON
RIINFTY = 0.7
RRIINFTY = 1.0 / RIINFTY
VVCRIC = 50.0
VDCRIC = 50.0
VVCLIM = 1000.0
VDCLIM = 1000.0
DIFM0 = VVCRIC * 1e-4
DIFS0 = VDCRIC * 1e-4
DIFMCON = VVCLIM * 1e-4
DIFSCON = VDCLIM * 1e-4
DIFTOP = 0.0
NUM_V_SMOOTH_RI = 1

# SW2OCEAN module constants (OCEAN_COM.f:305-351)
ZMAX_SOLAR = 92.0
RFRAC = 0.62
ZETA1 = 1.5
ZETA2 = 20.0

OCEAN_FKPH = 0.6  # decks/P2SAoM40.R: ocean_fkph=0.6 (overrides KPP_COM's 0.1 default)


def kmixinit(ze, fkph=OCEAN_FKPH):
    """OCNKPP.f:1178-1256. Pure function of the fixed vertical grid `ze` (ZE(0:LMO), length
    LMO+1) and the rundeck's ocean_fkph. Returns wmt/wst (the (NNI+2,NNJ+2) velocity-scale
    lookup tables), fz500 ((LMO+1,LMO+1), 0-indexed, only [l,lbot] for lbot=LMO/2..LMO-1,
    l=lbot-2..lbot-1 populated), vtc, cg, difmiw, difsiw."""
    vtc = CONCV * np.sqrt(0.2 / CONCS / EPSILON) / VONK**2 / RICR
    cg = CSTAR * VONK * (CONCS * VONK * EPSILON) ** _ONE_THIRD_SP

    fkpm = 10.0 * fkph
    difmiw = fkpm * 1e-4
    difsiw = fkph * 1e-4

    i_idx = np.arange(0, NNI + 2)
    j_idx = np.arange(0, NNJ + 2)
    zehat = DELTAZ * i_idx + ZMIN
    usta = DELTAU * j_idx + UMIN
    ZEHAT, USTA = np.meshgrid(zehat, usta, indexing="ij")
    zeta = ZEHAT / (USTA**3 + EPSL)

    wmt = np.zeros_like(ZEHAT)
    wst = np.zeros_like(ZEHAT)

    pos = ZEHAT >= 0.0
    wmt[pos] = VONK * USTA[pos] / (1.0 + CONC1 * zeta[pos])
    wst[pos] = wmt[pos]

    neg = ~pos
    negA = neg & (zeta > ZETAM)
    negB = neg & ~negA
    wmt[negA] = VONK * USTA[negA] * np.sqrt(np.sqrt(1.0 - CONC2 * zeta[negA]))
    wmt[negB] = VONK * (CONAM * USTA[negB] ** 3 - CONCM * ZEHAT[negB]) ** _ONE_THIRD_SP

    negC = neg & (zeta > ZETAS)
    negD = neg & ~negC
    wst[negC] = VONK * USTA[negC] * np.sqrt(1.0 - CONC3 * zeta[negC])
    wst[negD] = VONK * (CONAS * USTA[negD] ** 3 - CONCS * ZEHAT[negD]) ** _ONE_THIRD_SP

    fz500 = np.zeros((LMO + 1, LMO + 1))
    for lbot in range(LMO // 2, LMO):
        for l in range(lbot - 2, lbot):
            fz500[l, lbot] = np.exp(-(ze[lbot] - ze[l]) / 500.0)

    return dict(wmt=wmt, wst=wst, fz500=fz500, vtc=vtc, cg=cg,
                difmiw=difmiw, difsiw=difsiw)


def init_solar(ze):
    """SW2OCEAN's `init_solar` (OCEAN_COM.f:321-349). Pure function of the fixed vertical grid
    `ze`. Returns lsrpd, fsr, dfsrdz, dfsrdzb (all 1-indexed: index 0 unused, valid 1..lsrpd)."""
    lsrpd = LMO - 1
    for l in range(1, LMO):
        if ze[l + 1] > ZMAX_SOLAR:
            lsrpd = l
            break

    def ef(z):
        return RFRAC * np.exp(-z / ZETA1) + (1.0 - RFRAC) * np.exp(-z / ZETA2)

    fsr = np.zeros(lsrpd + 1)
    dfsrdzb = np.zeros(lsrpd + 1)
    dfsrdz = np.zeros(lsrpd + 1)
    for l in range(1, lsrpd + 1):
        fsr[l] = ef(ze[l - 1])
        dfsrdzb[l] = fsr[l] / (ze[l] - ze[l - 1])
    for l in range(1, lsrpd):
        dfsrdz[l] = (fsr[l] - fsr[l + 1]) / (ze[l] - ze[l - 1])
    dfsrdz[lsrpd] = 0.0
    return lsrpd, fsr, dfsrdz, dfsrdzb


def z121(v, kmtj):
    """OCNKPP.f:1292-1312. `v` indices 0..kmtj+1 valid (len >= kmtj+2, matching Fortran's
    V(0:km+1)). Returns a new smoothed array; does not mutate the input."""
    v = v.copy()
    v[0] = 0.25 * v[1]
    v[kmtj + 1] = v[kmtj]
    for k in range(1, kmtj + 1):
        tmp = v[k]
        v[k] = v[0] + 0.5 * v[k] + 0.25 * v[k + 1]
        v[0] = 0.25 * tmp
    return v


def _wscale(sigma, hbl_or_caseA, ustar, bfsfc, wmt, wst):
    """OCNKPP.f's repeated inline `wscale` lookup (4 occurrences in KPPMIX, all identical).
    `hbl_or_caseA` is the depth argument (`caseA`=-zgrid(kl) during the search loop, `hbl`
    afterward)."""
    zehat = VONK * sigma * hbl_or_caseA * bfsfc
    if zehat <= ZMAX:
        zdiff = zehat - ZMIN
        iz = int(zdiff * RDELTAZ)
        iz = min(iz, NNI)
        iz = max(iz, 0)
        izp1 = iz + 1

        udiff = ustar - UMIN
        ju = int(udiff * RDELTAU)
        ju = min(ju, NNJ)
        ju = max(ju, 0)
        jup1 = ju + 1

        zfrac = zdiff * RDELTAZ - float(iz)
        ufrac = udiff * RDELTAU - float(ju)
        fzfrac = 1.0 - zfrac

        wam = fzfrac * wmt[iz, jup1] + zfrac * wmt[izp1, jup1]
        wbm = fzfrac * wmt[iz, ju] + zfrac * wmt[izp1, ju]
        wm = (1.0 - ufrac) * wbm + ufrac * wam
        if ju == NNJ and wm < wam:
            wm = wam

        was = fzfrac * wst[iz, jup1] + zfrac * wst[izp1, jup1]
        wbs = fzfrac * wst[iz, ju] + zfrac * wst[izp1, ju]
        ws = (1.0 - ufrac) * wbs + ufrac * was
        if ju == NNJ and ws < was:
            ws = was
    else:
        u3 = ustar * ustar * ustar
        wm = VONK * ustar * u3 / (u3 + CONC1 * zehat)
        ws = wm
    return wm, ws


def _fort_nint(x):
    """Fortran NINT(0.5d0 + SIGN(0.5d0,-(y))) as it appears twice in KPPMIX -- always rounds
    to 0 or 1 since the argument to NINT is always in [0,1]."""
    return round(x)


def _bfsfc_search(ze, zgrid_kl, kl, kmax, fsr, dfsrdz, dfsrdzb):
    """OCNKPP.f:437-449 (the inline `swfrac` call inside the bulk-Ri search loop):
    `kt = kl + NINT(0.5d0 + SIGN(0.5d0,-(ZE(kl)+zgrid(kl))))`."""
    kt = kl + _fort_nint(0.5 + _sign(0.5, -(ze[kl] + zgrid_kl)))
    if kt > kmax:
        return 0.0
    if kt == kmax:
        return fsr[kt] + (zgrid_kl + ze[kt - 1]) * dfsrdzb[kt]
    return fsr[kt] + (zgrid_kl + ze[kt - 1]) * dfsrdz[kt]


def _bfsfc_at_hbl(ze, hbl, kbl, kmax, fsr, dfsrdz, dfsrdzb):
    """OCNKPP.f:532-544 (the inline `swfrac` call after the search loop, using `hbl`):
    `kt = kbl-1 + NINT(0.5d0 + SIGN(0.5d0,-(ZE(kbl-1)-hbl)))`."""
    km1 = kbl - 1
    kt = km1 + _fort_nint(0.5 + _sign(0.5, -(ze[km1] - hbl)))
    if kt > kmax:
        return 0.0
    if kt == kmax:
        return fsr[kt] + (ze[kt - 1] - hbl) * dfsrdzb[kt]
    return fsr[kt] + (ze[kt - 1] - hbl) * dfsrdz[kt]


def kppmix(ze, zgrid, hwide, byhwide, kmtj, shsq, dvsq, ustar, bo, bosol,
           dbloc, ritop, wmt, wst, fz500, vtc, cg, difmiw, difsiw,
           lsrpd, fsr, dfsrdz, dfsrdzb):
    """OCNKPP.f:225-834. `LDD` is always `.false.` for this build (D52/D53), so `ddmix` never
    fires and the real Fortran's `alphaDT`/`betaDS`/`Coriol` arguments (unused in that case) are
    omitted here. All arrays use literal Fortran indices: `ze` is (0:LMO) i.e. length LMO+1;
    `zgrid`/`hwide`/`byhwide` are (0:LMO+1) i.e. length LMO+2; `shsq`/`dvsq`/`dbloc`/`ritop` are
    1-indexed length LMO+1 (index 0 unused padding, matching Fortran's default 1-based `(km)`
    declaration). Returns `visc`/`difs`/`dift` (0-indexed length LMO+2, matching Fortran's
    `(0:km+1)`), `ghats` (1-indexed length LMO+1), `hbl` (float), `kbl` (int)."""
    kmax = min(lsrpd, kmtj)

    visc = np.zeros(LMO + 2)
    difs = np.zeros(LMO + 2)
    dift = np.zeros(LMO + 2)
    ghats = np.zeros(LMO + 1)

    for ki in range(1, kmtj + 1):
        visc[ki] = dbloc[ki] * (zgrid[ki] - zgrid[ki + 1]) / (shsq[ki] + EPSL)
        dift[ki] = dbloc[ki] * byhwide[ki + 1]

    for _ in range(NUM_V_SMOOTH_RI):
        visc = z121(visc, kmtj)

    for ki in range(1, kmtj + 1):
        rigg = max(dift[ki], BVSQCON)
        ratio = min((BVSQCON - rigg) * RBVSQCON, 1.0)
        fcon = (1.0 - ratio * ratio)
        fcon = fcon * fcon * fcon

        rigg = max(visc[ki], 0.0)
        ratio = min(rigg * RRIINFTY, 1.0)
        fri = (1.0 - ratio * ratio)
        fri = fri * fri * fri

        ftop = fz500[ki, kmtj]

        visc[ki] = difmiw + fcon * DIFMCON + fri * DIFM0
        difs[ki] = difsiw + fcon * DIFSCON + fri * DIFS0 + ftop * DIFTOP
        dift[ki] = difs[ki]

    visc[0] = 0.0
    dift[0] = 0.0
    difs[0] = 0.0

    # LDD is always .false. -- ddmix never fires (D52/D53)

    visc[kmtj:LMO + 2] = 0.0
    difs[kmtj:LMO + 2] = 0.0
    dift[kmtj:LMO + 2] = 0.0

    # bulk-Richardson-number search loop (OCNKPP.f:422-527, the "GOTO 10" loop)
    rib_ka = 0.0
    kbl = kmtj
    hbl = -zgrid[kmtj]
    kl = 1
    while True:
        kl += 1
        bfsfc = _bfsfc_search(ze, zgrid[kl], kl, kmax, fsr, dfsrdz, dfsrdzb)
        caseA_depth = -zgrid[kl]
        bfsfc = bo + bosol * (1.0 - bfsfc)
        stable = 0.5 + _sign(0.5, bfsfc)
        sigma = stable * 1.0 + (1.0 - stable) * EPSILON
        wm, ws = _wscale(sigma, caseA_depth, ustar, bfsfc, wmt, wst)
        bvsq = 0.5 * (dbloc[kl - 1] * byhwide[kl] + dbloc[kl] * byhwide[kl + 1])
        vtsq = -zgrid[kl] * ws * np.sqrt(abs(bvsq)) * vtc
        rib_ku = ritop[kl] / (dvsq[kl] + vtsq + EPSL)
        if kbl == kmtj and rib_ku > RICR:
            hbl = -zgrid[kl - 1] + (zgrid[kl - 1] - zgrid[kl]) * \
                (RICR - rib_ka) / (rib_ku - rib_ka)
            kbl = kl
            break
        else:
            rib_ka = rib_ku
            if kl < kmtj:
                continue
            break

    bfsfc = _bfsfc_at_hbl(ze, hbl, kbl, kmax, fsr, dfsrdz, dfsrdzb)
    bfsfc = bo + bosol * (1.0 - bfsfc)
    stable = 0.5 + _sign(0.5, bfsfc)
    bfsfc = bfsfc + stable * EPSL

    caseA = 0.5 + _sign(0.5, -zgrid[kbl] - 0.5 * hwide[kbl] - hbl)
    byhbl = 1.0 / hbl

    sigma = stable * 1.0 + (1.0 - stable) * EPSILON
    wm, ws = _wscale(sigma, hbl, ustar, bfsfc, wmt, wst)

    kn = int(caseA + EPSL) * (kbl - 1) + (1 - int(caseA + EPSL)) * kbl

    delhat = 0.5 * hwide[kn] - zgrid[kn] - hbl
    r = 1.0 - delhat * byhwide[kn]

    def _deriv(arr):
        dvdzup = (arr[kn - 1] - arr[kn]) * byhwide[kn]
        dvdzdn = (arr[kn] - arr[kn + 1]) * byhwide[kn + 1]
        return 0.5 * ((1.0 - r) * (dvdzup + abs(dvdzup)) + r * (dvdzdn + abs(dvdzdn)))

    viscp = _deriv(visc)
    difsp = _deriv(difs)
    diftp = _deriv(dift)

    visch = visc[kn] + viscp * delhat
    difsh = difs[kn] + difsp * delhat
    difth = dift[kn] + diftp * delhat

    f1 = stable * CONC1 * bfsfc / (ustar**4 + EPSL)
    bywm = 1.0 / (wm + EPSL)
    byws = 1.0 / (ws + EPSL)

    gat1_1 = visch * byhbl * bywm
    dat1_1 = min(-viscp * bywm + f1 * visch, 0.0)
    gat1_2 = difsh * byhbl * byws
    dat1_2 = min(-difsp * byws + f1 * difsh, 0.0)
    gat1_3 = difth * byhbl * byws
    dat1_3 = min(-diftp * byws + f1 * difth, 0.0)

    blmc = np.zeros((kmtj + 1, 4))  # columns 1,2,3; rows 1..kbl-1 valid
    for ki in range(1, kbl):
        sig = (-zgrid[ki] + 0.5 * hwide[ki]) * byhbl
        sigma = stable * sig + (1.0 - stable) * min(sig, EPSILON)
        wm, ws = _wscale(sigma, hbl, ustar, bfsfc, wmt, wst)

        sig = (-zgrid[ki] + 0.5 * hwide[ki]) * byhbl
        a1 = sig - 2.0
        a2 = 3.0 - 2.0 * sig
        a3 = sig - 1.0

        gm = a1 + a2 * gat1_1 + a3 * dat1_1
        gs = a1 + a2 * gat1_2 + a3 * dat1_2
        gt = a1 + a2 * gat1_3 + a3 * dat1_3

        blmc[ki, 1] = hbl * wm * sig * (1.0 + sig * gm)
        blmc[ki, 2] = hbl * ws * sig * (1.0 + sig * gs)
        blmc[ki, 3] = hbl * ws * sig * (1.0 + sig * gt)

        ghats[ki] = (1.0 - stable) * cg * byws * byhbl

    sig = -zgrid[kbl - 1] * byhbl
    sigma = stable * sig + (1.0 - stable) * min(sig, EPSILON)
    wm, ws = _wscale(sigma, hbl, ustar, bfsfc, wmt, wst)

    sig = -zgrid[kbl - 1] * byhbl
    a1 = sig - 2.0
    a2 = 3.0 - 2.0 * sig
    a3 = sig - 1.0
    gm = a1 + a2 * gat1_1 + a3 * dat1_1
    gs = a1 + a2 * gat1_2 + a3 * dat1_2
    gt = a1 + a2 * gat1_3 + a3 * dat1_3
    dkm1_1 = hbl * wm * sig * (1.0 + sig * gm)
    dkm1_2 = hbl * ws * sig * (1.0 + sig * gs)
    dkm1_3 = hbl * ws * sig * (1.0 + sig * gt)

    if kbl <= kmtj:
        ki = kbl - 1
        delta = (hbl + zgrid[ki]) * byhwide[ki + 1]

        dkmp5 = caseA * visc[ki] + (1.0 - caseA) * blmc[ki, 1]
        dstar = (1.0 - delta) ** 2 * dkm1_1 + delta**2 * dkmp5
        blmc[ki, 1] = (1.0 - delta) * visc[ki] + delta * dstar

        dkmp5 = caseA * difs[ki] + (1.0 - caseA) * blmc[ki, 2]
        dstar = (1.0 - delta) ** 2 * dkm1_2 + delta**2 * dkmp5
        blmc[ki, 2] = (1.0 - delta) * difs[ki] + delta * dstar

        dkmp5 = caseA * dift[ki] + (1.0 - caseA) * blmc[ki, 3]
        dstar = (1.0 - delta) ** 2 * dkm1_3 + delta**2 * dkmp5
        blmc[ki, 3] = (1.0 - delta) * dift[ki] + delta * dstar

        ghats[ki] = (1.0 - caseA) * ghats[ki]

    for ki in range(1, kbl):
        visc[ki] = blmc[ki, 1]
        difs[ki] = blmc[ki, 2]
        dift[ki] = blmc[ki, 3]
    ghats[kbl:LMO + 1] = 0.0

    return visc, difs, dift, ghats, hbl, kbl
