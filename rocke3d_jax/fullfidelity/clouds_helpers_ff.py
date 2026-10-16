"""Full-fidelity port of the stateless per-layer convective-cloud helpers in CLOUDS2.F90 -- D91 and D92.

D91: PRECIP_MP (CLOUDS2.F90:5658-5680), ANVIL_OPTICAL_THICKNESS (5290-5328), MC_CLOUD_FRACTION (5332-5392),
     MC_PRECIP_PHASE (5588-5654).
D92: CONVECTIVE_MICROPHYSICS (5396-5584; the CLD_AER_CDNC arms are not compiled in P2SAoM40 and are not ported).

All functions are elementwise over broadcastable float64 arrays (a flat record axis); the Fortran branches
and the data-dependent early-exit loops are reproduced with masks, not Python-level per-record branching.

Single-precision-literal audit (D54 hazard; un-suffixed REAL literals are REAL(4) and are promoted to double
AFTER rounding to 24 bits).  Literals that are NOT exactly representable in REAL(4) and appear in these routines:
    19.3, 11.72, 2.7, 2.439, .4 (DCG/DCI formulas), 1.E-20 (ANVIL_OPTICAL_THICKNESS).
They are reproduced through `f4()` (float32 round-trip).  Exactly representable ones (1., 3., 4., 6., .5, 100.,
450., 1000., 1.5, 2., 5., 50.) need no treatment; `.4d0`, `.267d0`, `5.15D3`, `1.0225D6`, `7.55D7`, `6d-3`,
`1.5d-3`, `1.D-2`, `10.d0`, `6.108d0` etc. carry d0 exponents and are double.
Constants (Constants_mod.F90, non-PLANET_PARAMS branch) are rebuilt from their definitions and checked against the
values written by the instrumented real model (ffc_h_consts.txt) in clouds_helpers_compare.py.
"""
import numpy as np

# ------------------------------------------------------------------ constants
GASC = 8.314510
MAIR = 28.9655
RGAS = 1e3 * GASC / MAIR
GRAV = 9.80665
TEENY = 1.0e-30
PI = 3.141592653589793
BY3 = 1.0 / 3.0
BY6 = 1.0 / 6.0
TWOPI = 2.0 * PI
LHE = 2.5e6
LHM = 3.34e5
LHS = LHE + LHM
TF = 273.15
RHOW = 1e3
SRAT = 1.401
KAPA = (SRAT - 1.0) / SRAT
SHA = RGAS / KAPA
BYSHA = 1.0 / SHA
DTSRC = 1800.0          # rundeck value (model_com DTsrc); the real value is read from ffc_h_consts.txt too


def f4(x):
    """A REAL(4) literal promoted to REAL(8) (gfortran/ifort semantics for an un-suffixed literal)."""
    return float(np.float32(x))


_F193, _F1172, _F27, _F2439, _F04 = f4(19.3), f4(11.72), f4(2.7), f4(2.439), f4(0.4)
_F1E20 = f4(1.0e-20)

_d = lambda *a: [np.asarray(x, dtype=np.float64) for x in a]  # noqa: E731


# ------------------------------------------------------------------ D91: PRECIP_MP
def precip_mp(rho, flam, dc, cn, pow4="pow"):
    """Marshall-Palmer mass density of the precipitating part (function PRECIP_MP).

    `FLAM**4.` has a REAL(4) exponent.  `pow4` selects the evaluation: "pow" = libm pow(x, 4.0) (what
    numpy.power does), "mul" = (x*x)*(x*x).  clouds_helpers_compare.py reports which one reproduces the real
    dumps bitwise (see ledger D91)."""
    rho, flam, dc, cn = _d(rho, flam, dc, cn)
    with np.errstate(all="ignore"):
        if pow4 == "pow":
            f4p = np.power(flam, 4.0)
        else:
            f2 = flam * flam
            f4p = f2 * f2
        return (rho * (PI * BY6) * cn * np.exp(-flam * dc)
                * (dc * dc * dc / flam + 3.0 * dc * dc / (flam * flam)
                   + 6.0 * dc / (flam * flam * flam) + 6.0 / f4p))


# ------------------------------------------------------------------ D91: ANVIL_OPTICAL_THICKNESS
def anvil_optical_thickness(svlatl, rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem,
                            lhe=LHE, one_e20=_F1E20, cap=100.0):
    """-> (rcld, taumc).  TAUMC is INOUT in the Fortran but is assigned unconditionally, so its input is unused."""
    svlatl, rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem = _d(
        svlatl, rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem)
    with np.errstate(all="ignore"):
        r_liq = rcldlx * 100.0 * (wtem / (2.0 * BY3 * TWOPI * mcdncw)) ** BY3
        r_ice = rcldix * 100.0 * (wtem / (2.0 * BY3 * TWOPI * mcdnci)) ** BY3
        r_ice = np.minimum(r_ice, rimax)
        rcld = np.where(svlatl == lhe, r_liq, r_ice)
        rclde = rcld / bybr
        taumc = 1.5 * tem / (fcld * rclde + one_e20)
        taumc = np.where(taumc > cap, cap, taumc)
    return rcld, taumc


# ------------------------------------------------------------------ D91: MC_CLOUD_FRACTION
def mc_cloud_fraction(L, tl, pl, ccm, wcu, plemin, plel2, plemax, ccmmin, wcumin, lmin, lmax, ccmul, ccmul2,
                      dtsrc=DTSRC, anvil5=5.0, virga=True, detr3=True):
    """-> fcloud.  `anvil5`/`virga`/`detr3` exist only for mutation tests."""
    L, tl, pl, ccm, wcu, plemin, plel2, plemax, ccmmin, wcumin, lmin, lmax, ccmul, ccmul2 = _d(
        L, tl, pl, ccm, wcu, plemin, plel2, plemax, ccmmin, wcumin, lmin, lmax, ccmul, ccmul2)
    with np.errstate(all="ignore"):
        rho = pl / (RGAS * tl)
        fcloud = ccmul * ccm / (rho * GRAV * wcu * dtsrc + TEENY)
        fcloud = np.where(plemin - plel2 >= 450.0, anvil5 * fcloud, fcloud)
        below = L < lmin
        if virga:
            fcloud = np.where(below, ccmul * ccmmin / (rho * GRAV * wcumin * dtsrc + TEENY), fcloud)
        shallow = (plemin - plemax) < 450.0
        if detr3:
            fcloud = np.where(shallow & (L == lmax - 1), ccmul2 * ccm / (rho * GRAV * wcu * dtsrc + TEENY), fcloud)
        fcloud = np.where(shallow & below, 0.0, fcloud)
        fcloud = np.where(fcloud > 1.0, 1.0, fcloud)
    return fcloud


# ------------------------------------------------------------------ D91: MC_PRECIP_PHASE
def mc_precip_phase(L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1_in, mc_revp_abv_cldbase=1,
                    lhs=LHS, lhe=LHE, bysha=BYSHA):
    """-> (lhp, mcloud, heat1).  HEAT1 is declared INTENT(OUT) but accumulated, so its entry value matters
    (the real value at the call is dumped and used).  PRCP is INOUT but never modified."""
    L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1 = _d(
        L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1_in)
    L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1 = np.broadcast_arrays(
        L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1)
    heat1 = heat1.copy()
    if mc_revp_abv_cldbase == 0:
        mcloud = np.where(L <= lmin, 2.0 * fevap * airm, 0.0)
    else:
        mcloud = 2.0 * fevap * airm
    mcloud = np.where(mcloud > airm, airm, mcloud)
    lhp = lhp1.copy()
    m1 = (lhp == lhs) & (told > TF) & (told1 <= TF)          # melt frozen precip
    heat1 = np.where(m1, heat1 + LHM * prcp * bysha, heat1)
    lhp = np.where(m1, lhe, lhp)
    m2 = (lhp == lhe) & (told <= TF) & (told1 > TF)          # refreeze rain (inversion)
    heat1 = np.where(m2, heat1 - LHM * prcp * bysha, heat1)
    lhp = np.where(m2, lhs, lhp)
    m3 = (lhp != vlat) & (cond > 0)                          # condensate phase differs from precip phase
    heat1 = np.where(m3, heat1 + (vlat - lhp) * cond * bysha, heat1)
    return lhp, mcloud, heat1


# ------------------------------------------------------------------ D92: CONVECTIVE_MICROPHYSICS
def _dcg(wv, pl, gr=True):
    return np.minimum(((wv / _F193) * (pl / 1000.0) ** _F04) ** _F27, 1e-2)


def _dci(wv, pl):
    return np.minimum(((wv / _F1172) * (pl / 1000.0) ** _F04) ** _F2439, 1e-2)


def _dcw_search(wv, pl, ddcw, wmax, nmax):
    """The `do ITER=1,ITMAX-1` critical-size search for liquid: DCW grows by DDCW until the fall speed VT
    reaches WV (VT>=0 .and. VT>=WV) or exceeds WMAX.  Returns (DCW, iterations_used, exit_kind) where
    exit_kind 1 = fall-speed exit, 2 = WMAX exit, 0 = loop ran to completion."""
    wv, pl, ddcw, wmax = np.broadcast_arrays(wv, pl, ddcw, wmax)
    dcw = np.zeros(wv.shape)
    active = np.ones(wv.shape, bool)
    kind = np.zeros(wv.shape, int)
    nit = np.zeros(wv.shape, int)
    pfac = (1000.0 / pl) ** 0.4
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


def convective_microphysics(pl, wcu, dwcu, lfrz, wcufrz, tp, ti, fitmax, pland, cn0, cn0i, cn0g, flamw, flamg,
                            flami, rhoip, rhog, itmax, tlmin, tlmin1, wmax, condip_in=0.0, condgp_in=0.0,
                            pow4="pow", return_diag=False):
    """-> (condp, condp1, condip, condgp).  CONDIP/CONDGP are only assigned in the mixed-phase branch, elsewhere
    the entry values pass through.  return_diag adds a dict with branch masks (water/ice/mixed) and loop exits."""
    (pl, wcu, dwcu, lfrz, wcufrz, tp, ti, fitmax, pland, cn0, cn0i, cn0g, flamw, flamg, flami, rhoip, rhog,
     itmax, tlmin, tlmin1, wmax, condip_in, condgp_in) = np.broadcast_arrays(*_d(
        pl, wcu, dwcu, lfrz, wcufrz, tp, ti, fitmax, pland, cn0, cn0i, cn0g, flamw, flamg, flami, rhoip, rhog,
        itmax, tlmin, tlmin1, wmax, condip_in, condgp_in))
    mp = lambda rho, flam, dc, cn: precip_mp(rho, flam, dc, cn, pow4)  # noqa: E731
    with np.errstate(all="ignore"):
        wv = np.maximum(wcu - dwcu, 0.0)
        dcg = _dcg(wv, pl)
        dci = _dci(wv, pl)
        tig = np.where(lfrz == 0, TF, TF - 4.0 * wcufrz)
        tig = np.where(tig < ti - 10.0, ti - 10.0, tig)
        water = tp >= TF
        ice = ~water & (tp <= tig)
        mixed = ~water & ~ice
        # ---- liquid
        ddcw = np.where(pland < 0.5, 1.5e-3 * fitmax, 6e-3 * fitmax)
        nmax = itmax.astype(int) - 1
        dcw1, nit1, k1 = _dcw_search(wv, pl, ddcw, wmax, nmax)
        condp1_w = mp(RHOW, flamw, dcw1, cn0)
        wvu = wcu + dwcu
        dcw2, nit2, k2 = _dcw_search(wvu, pl, ddcw, wmax, nmax)
        condp_w = mp(RHOW, flamw, dcw2, cn0)
        # ---- ice
        condp1_i = mp(rhoip, flami, dci, cn0i)
        dci_u = _dci(wvu, pl)
        condp_i = mp(rhoip, flami, dci_u, cn0i)
        # ---- mixed phase
        fg = (tp - tig) / ((TF - tig) + TEENY)
        fg = np.where(fg > 1.0, 1.0, fg)
        fg = np.where(fg < 0.0, 0.0, fg)
        fg = np.where((tlmin <= TF) | (tlmin1 <= TF), 0.0, fg)
        fi = 1.0 - fg
        cip_l = mp(rhoip, flami, dci, cn0i)
        cgp_l = mp(rhog, flamg, dcg, cn0g)
        condp1_m = fg * cgp_l + fi * cip_l
        dcg_u = _dcg(wvu, pl)
        cip_u = mp(rhoip, flami, dci_u, cn0i)
        cgp_u = mp(rhog, flamg, dcg_u, cn0g)
        condp_m = fg * cgp_u + fi * cip_u
    condp = np.where(water, condp_w, np.where(ice, condp_i, condp_m))
    condp1 = np.where(water, condp1_w, np.where(ice, condp1_i, condp1_m))
    condip = np.where(mixed, cip_u, condip_in)
    condgp = np.where(mixed, cgp_u, condgp_in)
    if return_diag:
        return condp, condp1, condip, condgp, dict(water=water, ice=ice, mixed=mixed, fg=fg, tig=tig,
                                                  exit1=np.where(water, k1, -1), exit2=np.where(water, k2, -1),
                                                  nit1=nit1, nit2=nit2)
    return condp, condp1, condip, condgp
