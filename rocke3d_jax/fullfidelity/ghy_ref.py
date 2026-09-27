"""Reference transcription (plain Python/NumPy, float64) of the ModelE land model GHY (giss_LSM/GHY.f
`advnc` and callees, SNOW.f, SNOW_DRV.f) -- Track B.

Written to mirror the Fortran control flow line by line (early exits, sub-cycling, variable snow layer
counts) so it can be validated against the real model first; a JAX-vectorized version is derived from
it afterwards. Indexing: soil layer index k keeps its Fortran numbering (arrays have 7 slots, 0..6);
the bare/vegetated fraction index ibv is 0-based (0 = bare soil, 1 = vegetated). Compiled options of the
P2SAoM40 build: EVAP_VEG_GROUND, GHY_FD_1_HACK, GHY_USE_LARGESCALE_PRECIP, INTERCEPT_TEMPORAL,
LARGE_SCALE_PRECIP_INTERCEPT, MELT_FRESH_SNOW_ON_WARM_GROUND (only if snow_cover_same_as_rad != 0);
no tracers, no SCM, no RAD_VEG_GROUND, no SNSH_VEG_GROUND, shv = 0.

The vegetation (Ent) exports (canopy conductance, soil-layer betas, ...) are INPUTS per sub-iteration
(recorded from the real model); porting Ent itself is separate work.
"""
import math
import numpy as np

# ------------------------------------------------------------------ constants
TFRZ = 273.15
SHA = 1002.88097573814161
LHE = 2.5e6
LHM = 3.34e5
RHOW = 1000.0
STBO = 5.67037321e-8
GRAV = 9.80664999999999942
SHW = 4185.0 * RHOW
SHI = 2060.0 * RHOW
FSN = LHM * RHOW
ELH = LHE * RHOW
SHV = 0.0
PRFR = 0.2
NGM, IMT, NLSN = 6, 5, 3
NTH = 64
MRAT, RVAP = 0.621946798777856413, 461.532611712461858
_QA = 6.108 * MRAT
_QB = 1.0 / (RVAP * TFRZ)
_QC = 1.0 / RVAP
SHC_SOIL_TEXTURE = np.array([2e6, 2e6, 2e6, 2.5e6, 2.4e6])
RHO_ICE = 916.6
RHO_WATER = RHOW
RHO_FRESH_SNOW = 150.0
LAT_FUSION = LHM * RHOW
LAT_EVAP = LHE * RHOW
MAX_FRACT_WATER = 0.055
EPS_SNOW = 1e-8
MIN_SNOW_THICKNESS = 0.1
MIN_FRACT_COVER = 0.0001
TOTAL_NL = 3


def qsat(tm, lh, pr):
    return _QA * math.exp(lh * (_QB - _QC / max(130.0, tm))) / pr


def dqsatdt(tm, lh):
    return lh * _QC / (tm * tm)


# ------------------------------------------------------------- soil tables (hl0)
def hl0():
    """Soil property tables: thm(0:64,4), hlm(0:64), xklm(0:64,4), dlm(0:64,4)."""
    nexp = 6
    c = 2.3025851
    a = np.array([[.2514, 0.0136, -2.8319, 0.5958],
                  [.1481, 1.8726, 0.1025, -3.6416],
                  [.2484, 2.4842, 0.4583, -3.9470],
                  [.8781, -5.1816, 13.2385, -11.9501]]).T       # a[k, i]
    b = np.array([[-0.4910, -9.8945, 9.7976, -3.2211],
                  [-0.3238, -12.9013, 3.4247, 4.4929],
                  [-0.5187, -13.4246, 2.8899, 5.0642],
                  [-3.0848, 9.5497, -26.2868, 16.6930]]).T
    p = np.array([[-0.1800, -7.9999, 5.5685, -1.8868],
                  [-0.1000, -10.0085, 3.6752, 1.2304],
                  [-0.1951, -9.7055, 2.7418, 2.0054],
                  [-2.1220, 5.9983, -16.9824, 8.7615]]).T
    sat = np.array([.394, .537, .577, .885])
    sxtn = 16.0
    nth = 2 ** nexp
    hlm = np.zeros(nth + 1)
    delh1 = -0.00625
    hmin = -1000.0
    delhn = delh1
    s = hmin / delh1
    alph0 = 1.0 / 8.0
    while True:
        alph0o = alph0
        alph0 = (s * alph0 + 1.0) ** (1.0 / nth) - 1.0
        if abs(alph0o - alph0) < 1e-8:
            break
    alpls1 = 1.0 + alph0
    for j in range(1, nth + 1):
        hlm[j] = hlm[j - 1] + delhn
        delhn = alpls1 * delhn
    mmax, xtol = 100, 1e-6
    thm = np.zeros((nth + 1, 4))
    for i in range(4):
        thm[0, i] = 1.0
        for j in range(1, nth + 1):
            hs = -math.exp(c * (a[0, i] + a[1, i] + a[2, i] + a[3, i]))
            a1 = a[2, i] / a[3, i]
            a2 = (a[1, i] - (math.log(-hlm[j] - hs)) / c) / a[3, i]
            a3 = a[0, i] / a[3, i]
            testh = thm[j - 1, i]
            for m in range(mmax):
                func = (testh ** 3) + (a1 * (testh ** 2)) + (a2 * testh) + a3
                dfunc = (3 * testh ** 2) + (2 * a1 * testh) + a2
                diff = func / dfunc
                testh = testh - diff
                if abs(diff) < xtol:
                    break
            thm[j, i] = testh
    xklm = np.zeros((nth + 1, 4))
    dlm = np.zeros((nth + 1, 4))
    for j in range(nth + 1):
        for i in range(4):
            arg = 0.0
            for k in range(-1, 3):
                arg = arg + b[k + 1, i] * thm[j, i] ** k
            arg = min(arg, sxtn); arg = max(arg, -sxtn)
            xklm[j, i] = math.exp(c * arg)
        for i in range(4):
            arg = 0.0
            for k in range(-1, 3):
                arg = arg + p[k + 1, i] * thm[j, i] ** k
            arg = min(arg, sxtn); arg = max(arg, -sxtn)
            dlm[j, i] = math.exp(c * arg)
    for j in range(nth + 1):
        for i in range(4):
            thm[j, i] = thm[j, i] * sat[i]
    return thm, hlm, xklm, dlm


THM, HLM, XKLM, DLM = hl0()


# -------------------------------------------------------------------- snow model
def pass_water(wsn, hsn, dz, nl, water_down, heat_down):
    """SNOW.f pass_water (arrays are 0-based, layers 0..nl-1)."""
    for n in range(nl):
        ice_old = min(wsn[n], -hsn[n] / LAT_FUSION)
        ice = ice_old
        wsn[n] = wsn[n] + water_down
        hsn[n] = hsn[n] + heat_down
        water_down = 0.0
        heat_down = 0.0
        if hsn[n] >= 0.0 or wsn[n] <= 0.0:
            water_down = wsn[n]
            heat_down = hsn[n]
            wsn[n] = 0.0
            hsn[n] = 0.0
            dz[n] = 0.0
        elif hsn[n] > -wsn[n] * LAT_FUSION:
            ice = -hsn[n] / LAT_FUSION
            free_water = wsn[n] - ice
            water_down = max(0.0, free_water - ice * MAX_FRACT_WATER)
            wsn[n] = wsn[n] - water_down
            dz[n] = min(dz[n], ice * RHO_WATER / RHO_FRESH_SNOW)
        else:
            if wsn[n] + EPS_SNOW < ice_old:
                dz[n] = dz[n] * wsn[n] / ice_old
            dz[n] = min(dz[n], wsn[n] * RHO_WATER / RHO_FRESH_SNOW)
        dz[n] = max(dz[n], wsn[n] * RHO_WATER / RHO_ICE)
    return water_down, heat_down


def snow_fraction(dz, nl, prsnow, dt, fract_cover):
    fresh_snow = RHO_WATER / RHO_FRESH_SNOW * prsnow * dt
    dz_aver = float(np.sum(dz[:nl])) * fract_cover + fresh_snow
    fnew = min(.95, dz_aver / MIN_SNOW_THICKNESS)
    if fnew < MIN_FRACT_COVER:
        fnew = 0.0
    return fnew


def snow_redistr(dz, wsn, hsn, nl, fract_cover_ratio, want_flux=False, dt=1.0):
    """SNOW.f snow_redistr; dz has TOTAL_NL+1 slots. Returns (nl, tr_flux)."""
    tr_flux = np.zeros(TOTAL_NL + 1)
    if dz[0] == 0.0:
        return nl, tr_flux
    dzo = dz[:nl].copy()
    wsno = wsn[:nl].copy()
    hsno = hsn[:nl].copy()
    nlo = nl
    total_dz = float(np.sum(dzo)) * fract_cover_ratio
    if total_dz > MIN_SNOW_THICKNESS * 1.5:
        nl = TOTAL_NL
        dz[0] = MIN_SNOW_THICKNESS
        for i in range(1, nl):
            dz[i] = (total_dz - dz[0]) / (nl - 1)
    else:
        nl = 1
        dz[0] = total_dz
    wsn[:TOTAL_NL] = 0.0
    hsn[:TOTAL_NL] = 0.0
    no = 0
    delta = 0.0
    fcr = fract_cover_ratio
    for n in range(nl):
        while delta < dz[n] and no < nlo:
            no += 1
            delta = delta + dzo[no - 1] * fcr
            wsn[n] = wsn[n] + wsno[no - 1] * fcr
            hsn[n] = hsn[n] + hsno[no - 1] * fcr
        ddz = delta - dz[n]
        fract = ddz / (dzo[no - 1] * fcr)
        if fract < -EPS_SNOW or fract > 1.0 + EPS_SNOW:
            raise RuntimeError('snow_redistr: internal error 3')
        wsn[n] = wsn[n] - fract * wsno[no - 1] * fcr
        hsn[n] = hsn[n] - fract * hsno[no - 1] * fcr
        if n < nl - 1:
            wsn[n + 1] = wsn[n + 1] + fract * wsno[no - 1] * fcr
            hsn[n + 1] = hsn[n + 1] + fract * hsno[no - 1] * fcr
            delta = ddz
        else:
            if abs(fract) > EPS_SNOW:
                raise RuntimeError('snow_redistr: internal error 1')
    if want_flux:
        wsno_full = np.zeros(TOTAL_NL)
        wsno_full[:nlo] = wsno
        tr_flux[0] = 0.0
        for i in range(1, TOTAL_NL + 1):
            tr_flux[i] = -(wsn[i - 1] - wsno_full[i - 1] * fcr) / dt + tr_flux[i - 1]
    return nl, tr_flux


def tridiag(a, b, c, r):
    """solvers/TRIDIAG.f: (a sub, b diag, c super)."""
    n = len(b)
    u = np.zeros(n)
    gam = np.zeros(n)
    bet = b[0]
    u[0] = r[0] / bet
    for j in range(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def heat_eq(dz, tsn, hsn, csn, ksn, nl, flux_in, flux_in_deriv, dt):
    """SNOW.f heat_eq. dz,tsn,ksn have nl+1 valid entries; returns (flux_corr, flux_in)."""
    eta = np.zeros(nl + 1)
    eta[:nl] = .5
    eta[nl] = 0.0
    gamma = .5
    if dz[0] < MIN_SNOW_THICKNESS * .5:
        eta[0] = 1.0
        gamma = 1.0
    itermax = 2
    tnew = np.zeros(nl)
    flux_corr = 0.0
    for it in range(1, itermax + 1):
        a = np.zeros(nl); b = np.zeros(nl); c = np.zeros(nl); f = np.zeros(nl)
        n = 0
        dt_to_cdz = dt / (csn[n] * dz[n])
        right = 2.0 * dt_to_cdz / (dz[n] / ksn[n] + dz[n + 1] / ksn[n + 1])
        a[n] = 1.0 + right * eta[n] - dt_to_cdz * flux_in_deriv * gamma
        b[n] = 0.0
        c[n] = -right * eta[n + 1]
        f[n] = (tsn[n] * (1.0 - right * (1.0 - eta[n]) - dt_to_cdz * flux_in_deriv * gamma)
                + tsn[n + 1] * right * (1.0 - eta[n + 1]) + dt_to_cdz * flux_in)
        for n in range(1, nl):
            dt_to_cdz = dt / (csn[n] * dz[n])
            right = 2.0 * dt_to_cdz / (dz[n] / ksn[n] + dz[n + 1] / ksn[n + 1])
            left = 2.0 * dt_to_cdz / (dz[n] / ksn[n] + dz[n - 1] / ksn[n - 1])
            a[n] = 1.0 + (left + right) * eta[n]
            b[n] = -left * eta[n - 1]
            c[n] = -right * eta[n + 1]
            f[n] = (tsn[n] * (1.0 - (left + right) * (1.0 - eta[n]))
                    + tsn[n - 1] * left * (1.0 - eta[n - 1])
                    + tsn[n + 1] * right * (1.0 - eta[n + 1]))
        tnew = tridiag(b, a, c, f)
        flux_corr = flux_in_deriv * (tnew[0] - tsn[0]) * gamma
        syst_flux_err = flux_in_deriv * (tnew[0] - 0.0) * gamma
        if it != itermax and tnew[0] > 0.0 and flux_in_deriv < 0.0:
            gamma = (1.0 - syst_flux_err / flux_corr) * gamma
        else:
            break
    for n in range(nl):
        hsn[n] = hsn[n] + (tnew[n] - tsn[n]) * csn[n] * dz[n]
    n = nl - 1
    flux_in = -(tsn[n + 1] - tnew[n] * eta[n] - tsn[n] * (1.0 - eta[n])) * 2.0 / (dz[n] / ksn[n] + dz[n + 1] / ksn[n + 1])
    return flux_corr, flux_in


def snow_adv_1(dz, wsn, hsn, nl, srht, trht, snht, htpr, evaporation, pr, dt, t_ground, dz_ground,
               snsh_dt, evap_dt, evap_min):
    """Returns (nl, snht, evaporation, water_to_ground, heat_to_ground, radiation_out)."""
    k_ground = 3.4
    wsn_o = wsn[:nl].copy()
    hsn_o = hsn[:nl].copy()
    nl_o = nl
    tsn = np.zeros(TOTAL_NL + 2)
    tsn[0] = 0.0

    def all_melted():
        w2g = float(np.sum(wsn_o[:nl_o])) / dt + pr - evaporation
        h2g = (float(np.sum(hsn_o[:nl_o])) / dt + htpr - LAT_EVAP * evaporation - snht + srht + trht
               - STBO * (tsn[0] + TFRZ) ** 4)
        rad = STBO * (tsn[0] + TFRZ) ** 4
        wsn[:nl] = 0.0
        hsn[:nl] = 0.0
        dz[:nl] = 0.0
        return 1, snht, evaporation, w2g, h2g, rad

    fresh_snow = RHO_WATER / RHO_FRESH_SNOW * min(pr * dt - evaporation * dt, -htpr * dt / LAT_FUSION)
    if fresh_snow > 0.0:
        dz[0] = dz[0] + fresh_snow
        nl = max(nl, 1)
    else:
        if wsn[0] < EPS_SNOW:
            return all_melted()
    water_down = (pr - evaporation) * dt
    heat_down = htpr * dt
    water_down, heat_down = pass_water(wsn, hsn, dz, nl, water_down, heat_down)
    heat_to_ground = heat_down / dt
    water_to_ground = water_down / dt
    if float(np.sum(wsn[:nl])) < EPS_SNOW:
        return all_melted()
    nl, _ = snow_redistr(dz, wsn, hsn, nl, 1.0)
    dz[nl] = dz_ground
    csn = np.zeros(TOTAL_NL)
    ksn = np.zeros(TOTAL_NL + 1)
    for n in range(nl):
        rho_snow = wsn[n] * RHO_WATER / dz[n]
        csn[n] = 2060.0 * rho_snow
        ksn[n] = 3.22e-6 * rho_snow ** 2
    ksn[nl] = k_ground
    for n in range(nl):
        if hsn[n] > 0.0:
            raise RuntimeError('snow_adv_1: empty snow layer found (1)')
        elif hsn[n] > -wsn[n] * LAT_FUSION:
            tsn[n] = 0.0
        else:
            tsn[n] = (hsn[n] + wsn[n] * LAT_FUSION) / (csn[n] * dz[n])
    tsn[nl] = t_ground
    flux_in = (srht + trht - STBO * (tsn[0] + TFRZ) ** 4 - LAT_EVAP * evaporation - snht
               - evaporation * SHV * tsn[0])
    flux_in_deriv = (-4.0 * STBO * (tsn[0] + TFRZ) ** 3 - LAT_EVAP * evap_dt - snsh_dt
                     - evap_dt * SHV * tsn[0] - evaporation * SHV)
    radiation_out = STBO * (tsn[0] + TFRZ) ** 4
    snht = snht + evaporation * SHV * tsn[0]
    flux_corr, flux_in = heat_eq(dz, tsn, hsn, csn, ksn, nl, flux_in, flux_in_deriv, dt)
    heat_to_ground = heat_to_ground + flux_in
    delta_tsn_impl = flux_corr / flux_in_deriv
    radiation_out = radiation_out - (-4.0 * STBO * (tsn[0] + TFRZ) ** 3) * delta_tsn_impl
    snht = snht + snsh_dt * delta_tsn_impl + (evap_dt * SHV * tsn[0] + evaporation * SHV) * delta_tsn_impl
    delta_evap = evap_dt * delta_tsn_impl
    if evaporation + delta_evap < evap_min:
        evap_corr = evap_min - (evaporation + delta_evap)
        delta_evap = delta_evap + evap_corr
        snht = snht - evap_corr * LAT_EVAP
    evaporation = evaporation + delta_evap
    water_down = -delta_evap * dt
    heat_down = 0.0
    water_down, heat_down = pass_water(wsn, hsn, dz, nl, water_down, heat_down)
    heat_to_ground = heat_to_ground + heat_down / dt
    water_to_ground = water_to_ground + water_down / dt
    if float(np.sum(wsn[:nl])) < EPS_SNOW:
        return all_melted()
    nl, _ = snow_redistr(dz, wsn, hsn, nl, 1.0)
    dz[nl] = dz_ground
    for n in range(nl):
        if hsn[n] > 0.0:
            raise RuntimeError('snow_adv_1: empty snow layer found (2)')
        elif hsn[n] > -wsn[n] * LAT_FUSION:
            tsn[n] = 0.0
        else:
            tsn[n] = (hsn[n] + wsn[n] * LAT_FUSION) / (csn[n] * dz[n])
    tsn[nl] = t_ground
    mass_above = 0.0
    for n in range(nl):
        if dz[n] > EPS_SNOW:
            mass_layer = wsn[n] * RHO_WATER
            mass_above = mass_above + .5 * mass_layer
            scale_rho = (.5e-7 * GRAV * mass_above
                         * math.exp(14.643 - 4000.0 / (tsn[n] + TFRZ) - .02 * mass_layer / dz[n]) * dt)
            scale_rho = 1.0 + scale_rho
            dz[n] = dz[n] / scale_rho
            if dz[n] < mass_layer / RHO_ICE:
                dz[n] = mass_layer / RHO_ICE
            mass_above = mass_above + .5 * mass_layer
    return nl, snht, evaporation, water_to_ground, heat_to_ground, radiation_out


def snow_adv(dz, wsn, hsn, nl, srht, trht, snht, htpr, evaporation, pr, dt, t_ground, dz_ground,
             snsh_dt, evap_dt, evap_min):
    return snow_adv_1(dz, wsn, hsn, nl, srht, trht, snht, htpr, evaporation, pr, dt, t_ground,
                      dz_ground, snsh_dt, evap_dt, evap_min)


def snow_drv(fm, evap, snsh, srht, trht, canht, drips, dripw, htdrips, htdripw, devap_dt, dsnsh_dt,
             evap_min, dts, tp_soil, dz_soil, dzsn, wsn, hsn, nsn, fr_snow):
    """SNOW_DRV.f snow_drv (snow_cover_same_as_rad == 0). dzsn has 4 slots. Returns dict of outputs."""
    epotsn = fm * evap
    snshsn = fm * snsh
    srhtsn = fm * srht
    trhtsn = fm * trht + (1.0 - fm) * canht
    devap_sn_dt = fm * devap_dt
    dsnsh_sn_dt = fm * dsnsh_dt
    fr_snow_old = fr_snow
    fr_snow = snow_fraction(dzsn, nsn, drips, dts, fr_snow_old)
    out = dict(flmlt=0.0, fhsng=0.0, flmlt_scale=0.0, fhsng_scale=0.0, thrmsn=0.0)
    if fr_snow <= 0.0:
        out['flmlt_scale'] = drips
        out['fhsng_scale'] = htdrips
        if fr_snow_old > 0.0:
            out['flmlt_scale'] = out['flmlt_scale'] + float(np.sum(wsn[:nsn])) * fr_snow_old / dts
            out['fhsng_scale'] = out['fhsng_scale'] + float(np.sum(hsn[:nsn])) * fr_snow_old / dts
            wsn[:nsn] = 0.0
            hsn[:nsn] = 0.0
            dzsn[:nsn] = 0.0
            fr_snow = 0.0
            nsn = 1
        out.update(evap=evap, snsh=snsh, nsn=nsn, fr_snow=fr_snow)
        return out
    nsn, _ = snow_redistr(dzsn, wsn, hsn, nsn, fr_snow_old / fr_snow, want_flux=True, dt=dts)
    prsn = drips / fr_snow + dripw
    htprsn = htdrips / fr_snow + htdripw
    nsn, snshsn, epotsn, w2g, h2g, rad = snow_adv(dzsn, wsn, hsn, nsn, srhtsn, trhtsn, snshsn, htprsn,
                                                   epotsn, prsn, dts, tp_soil, dz_soil, dsnsh_sn_dt,
                                                   devap_sn_dt, evap_min)
    out['flmlt'] = max(w2g, 0.0)
    out['fhsng'] = h2g
    out['thrmsn'] = rad
    if fm > 0.0:
        evap = epotsn / fm
        snsh = snshsn / fm
    out.update(evap=evap, snsh=snsh, nsn=nsn, fr_snow=fr_snow)
    return out


# ---------------------------------------------------------------- soil column state
class Cell:
    """One grid-cell's GHY prognostic + diagnostic state (mirrors GHY.f module vars).
    ibv index: 0=bare, 1=vegetated. k index: Fortran 0..NGM (0 = canopy, only used at ibv=1)."""

    def __init__(self):
        self.w = np.zeros((NGM + 1, 2))
        self.ht = np.zeros((NGM + 1, 2))
        self.nsn = np.zeros(2, dtype=int)
        self.dzsn = np.zeros((NLSN + 1, 2))
        self.wsn = np.zeros((NLSN, 2))
        self.hsn = np.zeros((NLSN, 2))
        self.fr_snow = np.zeros(2)
        self.top_index = 0.0
        self.top_stdev = 0.0
        self.dz = np.zeros(NGM)
        self.q = np.zeros((IMT, NGM))
        self.qk = np.zeros((IMT, NGM))
        self.sl = 0.0
        self.fb = 0.0
        self.fv = 0.0

    def init_step(self):
        n = 0
        for k in range(NGM):
            if self.dz[k] <= 0.0:
                break
            n = k + 1
        self.n = n
        self.zb = np.zeros(n + 1)
        for k in range(1, n + 1):
            self.zb[k] = self.zb[k - 1] - self.dz[k - 1]
        self.zc = np.array([.5 * (self.zb[k] + self.zb[k + 1]) for k in range(n)])
        self.thets = np.zeros((n + 1, 2))
        self.thetm = np.zeros((n + 1, 2))
        self.shc = np.zeros((n + 1, 2))
        self.ws = np.zeros((n + 1, 2))
        for ibv in range(2):
            for k in range(n):
                ts_, tm_, sh_ = 0.0, 0.0, 0.0
                for i in range(IMT - 1):
                    ts_ += self.q[i, k] * THM[0, i]
                    tm_ += self.q[i, k] * THM[NTH, i]
                for i in range(IMT):
                    sh_ += self.q[i, k] * SHC_SOIL_TEXTURE[i]
                sh_ = (1.0 - ts_) * sh_ * self.dz[k]
                self.thets[k + 1, ibv] = ts_
                self.thetm[k + 1, ibv] = tm_
                self.shc[k + 1, ibv] = sh_
                self.ws[k + 1, ibv] = ts_ * self.dz[k]
        self.htprs = 0.0


def hydra(cell, i_bare_lo, i_vege_hi, fice, theta):
    """GHY.f hydra: computes h(0:n,2), xk(0:n+1,2), d, xku, xkh helpers (xkus, xkusa) for one call.
    i_bare_lo/i_vege_hi select which ibv (0 or 1) columns to process (1-based Fortran i_bare..i_vege
    mapped here to a set of ibv in {0,1})."""
    n = cell.n
    h = np.zeros((n + 2, 2))
    xk = np.zeros((n + 2, 2))
    d = np.zeros((n + 1, 2))
    xku = np.zeros((n + 1, 2))
    xkus = np.zeros((n + 1, 2))
    xkusa = np.zeros(2)
    xkud = 2.78e-5
    jcm = round(math.log(NTH) / math.log(2.0))
    for ibv in range(i_bare_lo, i_vege_hi + 1):
        xk[n + 1, ibv] = 0.0
        xku[0, ibv] = 0.0
        for k in range(1, n + 1):
            j1, j2 = 0, NTH
            thr1 = cell.thets[k, ibv]
            thr2 = cell.thetm[k, ibv]
            thr0 = theta[k, ibv]
            thr0 = min(thr1, thr0); thr0 = max(thr2, thr0)
            found = False
            for _ in range(jcm):
                j = (j1 + j2) // 2
                thr = 0.0
                for i in range(IMT - 1):
                    thr += THM[j, i] * cell.q[i, k - 1]
                if thr - thr0 < 0.0:
                    j2, thr2 = j, thr
                elif thr - thr0 > 0.0:
                    j1, thr1 = j, thr
                else:
                    hl = HLM[j]
                    j1, thr1, thr2 = j, thr0, -10.0
                    found = True
                    break
            if not found:
                hl = (HLM[j1] * (thr1 - thr2) + HLM[j2] * (thr1 - thr0)) / (thr1 - thr2) if False else \
                     (HLM[j1] * (thr1 - thr2) + HLM[j2] * (thr1 - thr0)) / (thr1 - thr2)
                # NOTE: Fortran hl formula: hl=(hlm(j1)*(thr0-thr2)+hlm(j2)*(thr1-thr0))/(thr1-thr2)
                hl = (HLM[j1] * (thr0 - thr2) + HLM[j2] * (thr1 - thr0)) / (thr1 - thr2)
            h[k, ibv] = hl
            ith = j1
            temp = (thr1 - thr0) / (thr1 - thr2)
            d1 = d2 = xku1 = xku2 = 0.0
            xkus[k, ibv] = 0.0
            for i in range(IMT - 1):
                d1 += cell.q[i, k - 1] * DLM[ith, i]
                d2 += cell.q[i, k - 1] * DLM[ith + 1, i]
                xku1 += cell.q[i, k - 1] * XKLM[ith, i]
                xku2 += cell.q[i, k - 1] * XKLM[ith + 1, i]
                xkus[k, ibv] += cell.q[i, k - 1] * XKLM[0, i]
            dl = (1.0 - temp) * d1 + temp * d2
            dl = (1.0 - fice[k, ibv]) * dl
            d[k, ibv] = dl
            xklu = (1.0 - temp) * xku1 + temp * xku2
            xklu = (1.0 - fice[k, ibv]) * xklu
            xku[k, ibv] = xklu
            if k == 1:
                xk1 = 0.0
                for i in range(IMT - 1):
                    xk1 += cell.qk[i, 0] * XKLM[0, i]
                xkl = xk1
                xkl = xkl / (1.0 + xkl / (-cell.zc[0] * xkud))
                xkl = (1.0 - fice[1, ibv] * theta[1, ibv] / cell.thets[1, ibv]) * xkl
                xkl = max(0.0, xkl)
                xk[1, ibv] = xkl
            else:
                xk[k, ibv] = math.sqrt(xku[k - 1, ibv] * xku[k, ibv])
    for ibv in range(i_bare_lo, i_vege_hi + 1):
        dz_total = 0.0
        for k in range(1, n + 1):
            xkusa[ibv] += xkus[k, ibv] * cell.dz[k - 1]
            dz_total += cell.dz[k - 1]
        xkusa[ibv] /= dz_total
    for k in range(1, n + 1):
        for ibv in range(i_bare_lo, i_vege_hi + 1):
            h[k, ibv] = h[k, ibv] + cell.zc[k - 1] * GRAV / 9.80665
    return h, xk, d, xku, xkus, xkusa


# ============================================================ full column model
class GhyColumn:
    """One grid cell's full GHY.f `advnc` computation, transcribed subroutine by subroutine.
    Arrays w,ht,tp,fice,theta,h,xk,d,xku are indexed [0..NGM, ibv] with ibv in {0,1} (0=bare,1=vege);
    k=0 is only meaningful for ibv=1 (canopy). f,fh are edge fluxes indexed [1..NGM+1, ibv] here stored
    as [0..NGM, ibv] with index k meaning Fortran's f(k+1) (i.e. python f[k] == fortran f(k+1))."""

    def __init__(self, static, dynamic, forcing):
        c = static
        self.dz = np.asarray(c['dz'], float)          # (NGM,)
        self.q = np.asarray(c['q'], float)             # (IMT, NGM)
        self.qk = np.asarray(c['qk'], float)
        self.sl = c['sl']
        self.top_index = c['top_index']
        self.top_stdev = c['top_stdev']
        self.geothermal_heat = forcing['geothermal_heat']

        self.w = np.asarray(dynamic['w'], float).copy()      # (NGM+1, 2)
        self.ht = np.asarray(dynamic['ht'], float).copy()
        self.nsn = np.asarray(dynamic['nsn'], int).copy()
        self.dzsn = np.zeros((NLSN + 1, 2))
        self.dzsn[:NLSN] = dynamic['dzsn']
        self.wsn = np.asarray(dynamic['wsn'], float).copy()
        self.hsn = np.asarray(dynamic['hsn'], float).copy()
        self.fr_snow = np.asarray(dynamic['fr_snow'], float).copy()

        self.pr = max(forcing['pr'], 0.0)
        self.htpr = forcing['htpr']
        self.prs = min(max(forcing['prs'], 0.0), self.pr)
        self.htprs = forcing['htprs']
        self.irrig = np.zeros(2)
        self.htirrig = np.zeros(2)
        self.srht = forcing['srht']
        self.trht = forcing['trht']
        self.ts = forcing['ts']
        self.qs = forcing['qs']
        self.pres = forcing['pres']
        self.rho = forcing['rho']
        self.ch = forcing['ch']
        self.qm1 = forcing['qm1']
        self.vs = forcing['vs']
        self.vs0 = forcing['vs0']
        self.gusti = forcing['gusti']
        self.tprime = forcing['tprime']
        self.qprime = forcing['qprime']
        self.ws_can = forcing.get('ws_can', 0.0)
        self.shc_can = forcing.get('shc_can', 0.0)

        n = 0
        for k in range(NGM):
            if self.dz[k] <= 0.0:
                break
            n = k + 1
        self.n = n
        self.zb = np.zeros(n + 1)
        for k in range(1, n + 1):
            self.zb[k] = self.zb[k - 1] - self.dz[k - 1]
        self.zc = np.array([.5 * (self.zb[k] + self.zb[k + 1]) for k in range(n)])
        self.thets = np.zeros((n + 1, 2))
        self.thetm = np.zeros((n + 1, 2))
        self.shc = np.zeros((n + 1, 2))
        self.ws = np.zeros((n + 1, 2))
        for ibv in (0, 1):
            for k in range(1, n + 1):
                ts_ = tm_ = sh_ = 0.0
                for i in range(IMT - 1):
                    ts_ += self.q[i, k - 1] * THM[0, i]
                    tm_ += self.q[i, k - 1] * THM[NTH, i]
                for i in range(IMT):
                    sh_ += self.q[i, k - 1] * SHC_SOIL_TEXTURE[i]
                sh_ = (1.0 - ts_) * sh_ * self.dz[k - 1]
                self.thets[k, ibv] = ts_
                self.thetm[k, ibv] = tm_
                self.shc[k, ibv] = sh_
                self.ws[k, ibv] = ts_ * self.dz[k - 1]
        # ws(0,2)=ws_can; shc(0,2)=shc_can -- set once, from Ent's canopy exports (GHY.f advnc, before loop)
        self.ws[0, 1] = self.ws_can
        self.shc[0, 1] = self.shc_can
        self.htprs = 0.0 if self.pr <= 0.0 else self.htpr / self.pr * self.prs
        self._xklh_warm()

        self.tp = np.zeros((n + 1, 2))
        self.fice = np.zeros((n + 1, 2))
        self.theta = np.zeros((n + 1, 2))
        self.h = np.zeros((n + 2, 2))
        self.xk = np.zeros((n + 2, 2))
        self.d = np.zeros((n + 1, 2))
        self.xku = np.zeros((n + 1, 2))
        self.xkus = np.zeros((n + 1, 2))
        self.xkusa = np.zeros(2)
        self.tsn1 = np.zeros(2)
        self.f = np.zeros((n + 2, 2))
        self.fh = np.zeros((n + 2, 2))
        self.fc = np.zeros(2)
        self.fch = np.zeros(2)
        self.snowd = np.zeros(2)
        self.betadl = np.zeros(n)
        self.betad = 0.0
        # accumulators
        self.atrg = self.ashg = self.aevap = self.alhg = 0.0
        self.aruns = self.arunu = self.aeruns = self.aerunu = 0.0
        self.ae0 = self.af1dt = self.aedifs = 0.0
        self.abetad = 0.0
        self.tbcs = self.tsns = 0.0

    # ---- fraction bookkeeping ----
    def _bounds(self):
        self.i_bare = 0 if self.fb > 0.0 else 1
        self.i_vege = 1 if self.fv > 0.0 else 0
        self.process_bare = self.fb > 0.0
        self.process_vege = self.fv > 0.0

    def reth(self):
        n = self.n
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(1, n + 1):
                self.theta[k, ibv] = self.w[k, ibv] / self.dz[k - 1]
        if self.process_vege and self.ws[0, 1] > 0.0:
            self.theta[0, 1] = (self.w[0, 1] / self.ws[0, 1]) ** (2.0 / 3.0)
        else:
            self.theta[0, 1] = 0.0
        self.theta[0, 1] = min(self.theta[0, 1], 1.0)
        self.snowd[:] = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            for lsn in range(self.nsn[ibv]):
                self.snowd[ibv] += self.wsn[lsn, ibv] * self.fr_snow[ibv]
        self.fw = self.theta[0, 1]
        self.fm = 1.0 - math.exp(-self.snowd[1] / (self.snowm + 1e-12))
        if self.fm < 1e-3:
            self.fm = 0.0
        self.fd = 1.0  # GHY_FD_1_HACK
        self.fw0 = self.fw
        self.fd0 = 1.0 - self.fw

    def hydra(self):
        n = self.n
        xkud = 2.78e-5
        jcm = round(math.log(NTH) / math.log(2.0))
        for ibv in range(self.i_bare, self.i_vege + 1):
            self.xk[n + 1, ibv] = 0.0
            self.xku[0, ibv] = 0.0
            for k in range(1, n + 1):
                j1, j2 = 0, NTH
                thr1 = self.thets[k, ibv]
                thr2 = self.thetm[k, ibv]
                thr0 = self.theta[k, ibv]
                thr0 = min(thr1, thr0); thr0 = max(thr2, thr0)
                exact = False
                for _ in range(jcm):
                    j = (j1 + j2) // 2
                    thr = sum(THM[j, i] * self.q[i, k - 1] for i in range(IMT - 1))
                    if thr - thr0 < 0.0:
                        j2, thr2 = j, thr
                    elif thr - thr0 > 0.0:
                        j1, thr1 = j, thr
                    else:
                        j1, thr1, thr2 = j, thr0, -10.0
                        exact = True
                        break
                hl = (HLM[j1] * (thr0 - thr2) + HLM[j2] * (thr1 - thr0)) / (thr1 - thr2)
                self.h[k, ibv] = hl
                ith = j1
                temp = (thr1 - thr0) / (thr1 - thr2)
                d1 = d2 = xku1 = xku2 = 0.0
                self.xkus[k, ibv] = 0.0
                for i in range(IMT - 1):
                    d1 += self.q[i, k - 1] * DLM[ith, i]
                    d2 += self.q[i, k - 1] * DLM[ith + 1, i]
                    xku1 += self.q[i, k - 1] * XKLM[ith, i]
                    xku2 += self.q[i, k - 1] * XKLM[ith + 1, i]
                    self.xkus[k, ibv] += self.q[i, k - 1] * XKLM[0, i]
                dl = ((1.0 - temp) * d1 + temp * d2) * (1.0 - self.fice[k, ibv])
                self.d[k, ibv] = dl
                xklu = ((1.0 - temp) * xku1 + temp * xku2) * (1.0 - self.fice[k, ibv])
                self.xku[k, ibv] = xklu
                if k == 1:
                    xk1 = sum(self.qk[i, 0] * XKLM[0, i] for i in range(IMT - 1))
                    xkl = xk1
                    xkl = xkl / (1.0 + xkl / (-self.zc[0] * xkud))
                    xkl = (1.0 - self.fice[1, ibv] * self.theta[1, ibv] / self.thets[1, ibv]) * xkl
                    xkl = max(0.0, xkl)
                    self.xk[1, ibv] = xkl
                else:
                    self.xk[k, ibv] = math.sqrt(self.xku[k - 1, ibv] * self.xku[k, ibv])
        for ibv in range(self.i_bare, self.i_vege + 1):
            self.xkusa[ibv] = 0.0
            dz_total = 0.0
            for k in range(1, n + 1):
                self.xkusa[ibv] += self.xkus[k, ibv] * self.dz[k - 1]
                dz_total += self.dz[k - 1]
            self.xkusa[ibv] /= dz_total
        for k in range(1, n + 1):
            for ibv in range(self.i_bare, self.i_vege + 1):
                self.h[k, ibv] = self.h[k, ibv] + self.zc[k - 1] * GRAV / 9.80665

    def _xklh_warm(self):
        gabc = [.125, .125, 1.0 - .125 - .125]
        alamw, alami = .573345, 2.1762
        alams = np.array([8.8, 2.9, 2.9, .25])
        self.hcwtw = 1.0
        self.hcwti = sum(1.0 / (1.0 + (alami / alamw - 1.0) * g) for g in gabc) / 3.0
        self.hcwtb = 1.0
        self.hcwt = np.array([sum(1.0 / (1.0 + (alams[i] / alamw - 1.0) * g) for g in gabc) / 3.0
                              for i in range(IMT - 1)])
        n = self.n
        self.xsha = np.zeros((n + 1, 2))
        self.xsh = np.zeros((n + 1, 2))
        for ibv in (0, 1):
            for k in range(1, n + 1):
                for i in range(IMT - 1):
                    xs = (1.0 - THM[0, i]) * self.q[i, k - 1]
                    self.xsha[k, ibv] += xs * self.hcwt[i] * alams[i]
                    self.xsh[k, ibv] += xs * self.hcwt[i]
        self.ba = .025 / alamw - 1.0

    def xklh(self):
        n = self.n
        alamw, alami, alama, alambr = .573345, 2.1762, .025, 2.9
        self.xkh = np.zeros((n + 1, 2))
        self.xkhm = np.zeros((n + 1, 2))
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(1, n + 1):
                gaa = .298 * self.theta[k, ibv] / (self.thets[k, ibv] + 1e-6) + .035
                gca = 1.0 - 2.0 * gaa
                hcwta = (2.0 / (1.0 + self.ba * gaa) + 1.0 / (1.0 + self.ba * gca)) / 3.0
                xw = self.w[k, ibv] * (1.0 - self.fice[k, ibv]) / self.dz[k - 1]
                xi = self.w[k, ibv] * self.fice[k, ibv] / self.dz[k - 1]
                xa = self.thets[k, ibv] - self.theta[k, ibv]
                xb = self.q[IMT - 1, k - 1]
                xnum = (xw * self.hcwtw * alamw + xi * self.hcwti * alami + xa * hcwta * alama
                        + self.xsha[k, ibv] + xb * self.hcwtb * alambr)
                xden = xw * self.hcwtw + xi * self.hcwti + xa * hcwta + self.xsh[k, ibv] + xb * self.hcwtb
                self.xkh[k, ibv] = xnum / xden
            for k in range(2, n + 1):
                self.xkhm[k, ibv] = ((self.zb[k - 1] - self.zc[k - 2]) * self.xkh[k, ibv]
                                     + (self.zc[k - 1] - self.zb[k - 1]) * self.xkh[k - 1, ibv]
                                     ) / (self.zc[k - 1] - self.zc[k - 2])

    def retp(self):
        n = self.n
        self.tp[:] = 0.0
        self.fice[:] = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            kk = 1 - ibv  # ibv=0(bare)->kk=1 (fortran kk=2-ibv, ibv fortran 1,2)
            for k in range(kk, n + 1):
                w_, ht_, shc_ = self.w[k, ibv], self.ht[k, ibv], self.shc[k, ibv]
                if FSN * w_ + ht_ < 0.0:
                    self.tp[k, ibv] = (ht_ + w_ * FSN) / (shc_ + w_ * SHI)
                    self.fice[k, ibv] = 1.0
                elif ht_ > 0.0:
                    self.tp[k, ibv] = ht_ / (shc_ + w_ * SHW)
                elif w_ >= 1e-12:
                    self.fice[k, ibv] = -ht_ / (FSN * w_)
        self.tsn1[:] = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            if self.wsn[0, ibv] > 1e-6 and self.hsn[0, ibv] + self.wsn[0, ibv] * FSN < 0.0:
                self.tsn1[ibv] = (self.hsn[0, ibv] + self.wsn[0, ibv] * FSN) / (self.wsn[0, ibv] * SHI)

    def evap_limits(self, compute_evap):
        n = self.n
        evap_max = np.zeros(2); evap_max_snow = np.zeros(2)
        evap_max_wet = np.zeros(2); evap_max_dry = np.zeros(2)
        self.betad = 0.0; self.abetad = 0.0; self.acna = 0.0; self.acnc = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(1, n + 1):
                evap_max[ibv] += (self.w[k, ibv] - self.dz[k - 1] * self.thetm[k, ibv]) / self.dt
        for ibv in range(self.i_bare, self.i_vege + 1):
            evap_max_snow[ibv] = self.pr
            for k in range(self.nsn[ibv]):
                evap_max_snow[ibv] += self.wsn[k, ibv] / self.dt
        if self.process_bare:
            ibv = 0
            evap_max_wet[ibv] = evap_max[ibv] + self.pr
            evap_max_dry[ibv] = min(evap_max[ibv],
                                    2.467 * self.d[1, ibv] * (self.theta[1, ibv] - self.thetm[1, ibv]) / self.dz[0] + self.pr)
        evap_max_vegsoil = 0.0
        cna = self.ch * self.vs
        if self.process_vege:
            ibv = 1
            evap_max_wet[ibv] = self.w[0, ibv] / self.dt
            evap_max_vegsoil = min(evap_max[ibv],
                                   2.467 * self.d[1, ibv] * (self.theta[1, ibv] - self.thetm[1, ibv]) / self.dz[0] + self.pr)
            self.betad = float(np.sum(self.betadl[:n]))
            if self.betad < 1e-12:
                self.betad = 0.0
            self.abetad = self.betad
            betat = self.cnc / (self.cnc + cna + 1e-12)
            self.acna = cna; self.acnc = self.cnc
            pot_evap_can = betat * (self.rho / RHOW) * self.ch * (
                self.vs * (qsat(self.tp[0, 1] + TFRZ, LHE, self.pres) - self.qs) - self.gusti * self.qprime)
            evap_max_dry[ibv] = 0.0
            if self.betad > 0.0 and pot_evap_can > 0.0:
                for k in range(1, n + 1):
                    evap_max_dry[ibv] += min(pot_evap_can * self.betadl[k - 1] / self.betad,
                                             (self.w[k, ibv] - self.dz[k - 1] * self.thetm[k, ibv]) / self.dt)
        evap_max_sat = self.fb * self.fr_snow[0] * evap_max_snow[0]
        evap_max_nsat = self.fb * (1.0 - self.fr_snow[0]) * evap_max_dry[0]
        if self.process_vege:
            evap_max_sat += self.fv * (self.fr_snow[1] * self.fm * evap_max_snow[1]
                                       + (1.0 - self.fr_snow[1] * self.fm) * self.theta[0, 1] * evap_max_wet[1])
            evap_max_nsat += self.fv * ((1.0 - self.fr_snow[1] * self.fm) * (1.0 - self.theta[0, 1]) * evap_max_dry[1])
        fr_sat = self.fb * self.fr_snow[0]
        if self.process_vege:
            fr_sat += self.fv * (self.fr_snow[1] * self.fm + (1.0 - self.fr_snow[1] * self.fm) * self.theta[0, 1])
        self.evap_max_out = evap_max_nsat
        self.fr_sat = fr_sat
        if not compute_evap:
            return
        qm1dt = .001 * self.qm1 / self.dt
        self.evap_min = -qm1dt
        qb = qsat(self.tp[1, 0] + TFRZ, LHE, self.pres)
        qv = qsat(self.tp[0, 1] + TFRZ, LHE, self.pres)
        qbs = qsat(self.tsn1[0] + TFRZ, LHE, self.pres)
        qvs = qsat(self.tsn1[1] + TFRZ, LHE, self.pres)
        qvg = qsat(self.tp[1, 1] + TFRZ, LHE, self.pres)
        rho3 = self.rho / RHOW
        v_qprime = self.gusti * self.qprime
        epb = rho3 * self.ch * (self.vs * (qb - self.qs) - v_qprime)
        epbs = rho3 * self.ch * (self.vs * (qbs - self.qs) - v_qprime)
        epv = rho3 * self.ch * (self.vs * (qv - self.qs) - v_qprime)
        epvs = rho3 * self.ch * (self.vs * (qvs - self.qs) - v_qprime)
        epv1 = epv * (1.0 - self.fw) / (self.fd + 1e-12)
        f_clump = 1.0; lai_ = self.lai; sai = 0.0
        ch_dense_veg = 0.01 * cna
        eta = math.exp(-f_clump * (lai_ + sai))
        ch_vg = self.ch * eta + ch_dense_veg * (1.0 - eta)
        epvg = rho3 * ch_vg * (self.vs * (qvg - self.qs) - v_qprime)
        if self.process_bare:
            self.evapb = max(min(epb, evap_max_dry[0]), -qm1dt)
            self.evapbs = max(min(epbs, evap_max_snow[0]), -qm1dt)
        else:
            self.evapb = 0.0; self.evapbs = 0.0
        if self.process_vege:
            self.evapvw = max(min(epv, evap_max_wet[1]), -qm1dt)
            self.evapvd = max(min(epv1, evap_max_dry[1]), 0.0)
            self.evapvs = max(min(epvs, evap_max_snow[1]), -qm1dt)
            evapvg = min(epvg, evap_max_vegsoil)
            evapvg = min(evapvg, evap_max[1] - self.evapvd * self.fd)
            evapvg = min(evapvg, epv - self.evapvd * self.fd - self.evapvw * self.fw)
            self.evapvg = max(evapvg, 0.0)
            if self.evapvw < 0.0:
                self.fw = 1.0; self.fd = 0.0
        else:
            self.evapvw = self.evapvd = self.evapvs = self.evapvg = 0.0
        self.devapbs_dt = rho3 * cna * qsat(self.tsn1[0] + TFRZ, LHE, self.pres) * dqsatdt(self.tsn1[0] + TFRZ, LHE)
        self.devapvs_dt = rho3 * cna * qsat(self.tsn1[1] + TFRZ, LHE, self.pres) * dqsatdt(self.tsn1[1] + TFRZ, LHE)
        self.evapdl = np.zeros(n)
        if self.betad > 0.0:
            for k in range(n):
                self.evapdl[k] = self.evapvd * self.betadl[k] / self.betad

    def sensible_heat(self):
        cna = self.ch * self.vs
        v_tprime = self.gusti * self.tprime
        eta = 0.0  # SNSH_VEG_GROUND not defined
        self.snshg = np.array([
            SHA * self.rho * self.ch * (self.vs * (self.tp[1, 0] - self.ts + TFRZ) - v_tprime),
            SHA * self.rho * self.ch * (self.vs * (self.tp[1, 1] - self.ts + TFRZ) - v_tprime) * eta])
        self.snshv = np.array([0.0,
            SHA * self.rho * self.ch * (self.vs * (self.tp[0, 1] - self.ts + TFRZ) - v_tprime) * (1.0 - eta)])
        self.snshs = np.array([
            SHA * self.rho * self.ch * (self.vs * (self.tsn1[0] - self.ts + TFRZ) - v_tprime),
            SHA * self.rho * self.ch * (self.vs * (self.tsn1[1] - self.ts + TFRZ) - v_tprime)])
        self.dsnsh_dt = SHA * self.rho * cna

    def drip_from_canopy(self):
        can_evap = 0.0
        snowf = min(-self.htpr / FSN, self.pr) if self.htpr < 0.0 else 0.0
        snowfs = min(-self.htprs / FSN, self.prs) if self.htprs < 0.0 else 0.0
        self.dripw = np.zeros(2); self.htdripw = np.zeros(2)
        self.drips = np.zeros(2); self.htdrips = np.zeros(2)
        self.dripw_scale = np.zeros(2)
        if self.process_vege:
            can_evap = self.evapvw * self.fw * (1.0 - self.fm * self.fr_snow[1])
            ptmps = max(self.prs - snowfs - can_evap, 0.0)
            ptmp = self.pr - self.prs - (snowf - snowfs)
            pr_dry = self.fd0 * ptmps
            wc_add = self.ws[0, 1] - self.w[0, 1]
            wc_new = self.w[0, 1] + min(pr_dry * self.dts, wc_add)
            fw_new = (wc_new / self.ws[0, 1]) ** (2.0 / 3.0) if self.ws[0, 1] > 1e-12 else 0.0
            fd_new = 1.0 - fw_new
            dr_scale = ptmps - (wc_new - self.w[0, 1]) / self.dts
            tau_storm = 3600.0
            f_prev_wet = 1.0 - (self.dts / tau_storm)
            if fw_new > PRFR:
                pr_dry = (1.0 - f_prev_wet) * fd_new * ptmp
            else:
                pr_dry = (1.0 - f_prev_wet * fw_new / PRFR) * fd_new * ptmp
            wc_add = (1.0 - f_prev_wet) * PRFR * (self.ws[0, 1] - wc_new)
            wc_new = wc_new + min(pr_dry * self.dts, wc_add)
            dr = ptmp - (wc_new - self.w[0, 1]) / self.dts
            dr = min(dr, self.pr - snowf - can_evap)
            dr = max(dr, self.pr - snowf - can_evap - (self.ws[0, 1] - self.w[0, 1]) / self.dts)
            dr = max(dr, 0.0)
            self.dripw[1] = dr
            self.dripw_scale[1] = dr_scale
            self.htdripw[1] = SHW * dr * max(self.tp[0, 1], 0.0)
            self.drips[1] = snowf
            self.htdrips[1] = min(self.htpr, 0.0)
        self.drips[0] = snowf
        self.htdrips[0] = min(self.htpr, 0.0)
        self.dripw[0] = self.pr - self.drips[0]
        self.dripw_scale[0] = self.prs - snowfs
        self.htdripw[0] = self.htpr - self.htdrips[0]
        # snow_cover_same_as_rad == 0 by default -> MELT_FRESH_SNOW_ON_WARM_GROUND not applied

    def flg(self):
        if self.process_bare:
            self.f[0, 0] = (-self.flmlt[0] * self.fr_snow[0] - self.flmlt_scale[0]
                            - (self.dripw[0] - self.evapb) * (1.0 - self.fr_snow[0]) - self.irrig[0])
        if self.process_vege:
            self.fc[0] = -self.pr + self.evapvw * self.fw * (1.0 - self.fm * self.fr_snow[1])
            self.fc[1] = -self.dripw[1] - self.drips[1]
            self.f[0, 1] = (-self.flmlt[1] * self.fr_snow[1] - self.flmlt_scale[1]
                            - (self.dripw[1] - self.evapvg) * (1.0 - self.fr_snow[1]) - self.irrig[1])
        self.evap_tot = np.zeros(2)
        self.evap_tot[0] = self.evapb * (1.0 - self.fr_snow[0]) + self.evapbs * self.fr_snow[0]
        self.evap_tot[1] = ((self.evapvw * self.fw + self.evapvd * self.fd) * (1.0 - self.fr_snow[1] * self.fm)
                            + self.evapvs * self.fr_snow[1] * self.fm + self.evapvg * (1.0 - self.fr_snow[1]))

    def flhg(self):
        thrm_can = STBO * (self.tp[0, 1] + TFRZ) ** 4
        thrm_soil = np.array([STBO * (self.tp[1, 0] + TFRZ) ** 4, STBO * (self.tp[1, 1] + TFRZ) ** 4])
        snsh_vapor = np.zeros((2, 2))
        if self.process_bare:
            self.fh[0, 0] = (-self.fhsng[0] * self.fr_snow[0] - self.fhsng_scale[0]
                             + (-self.htdripw[0] + self.evapb * (ELH + SHV * self.tp[1, 0]) + self.snshg[0]
                                + thrm_soil[0] - self.srht - self.trht) * (1.0 - self.fr_snow[0])
                             - self.htirrig[0])
            snsh_vapor[1, 0] = self.evapb * SHV * self.tp[1, 0]
        if self.process_vege:
            self.fh[0, 1] = (-self.fhsng[1] * self.fr_snow[1] - self.fhsng_scale[1]
                             + (-self.htdripw[1] + self.evapvg * (ELH + SHV * self.tp[1, 1]) + self.snshg[1]
                                + thrm_soil[1] - thrm_can) * (1.0 - self.fr_snow[1])
                             - self.htirrig[1])
            snsh_vapor[1, 1] = self.evapvg * SHV * self.tp[1, 1]
            self.fch[0] = (-self.htpr
                          + (self.evapvw * (ELH + SHV * self.tp[0, 1]) * self.fw + self.snshv[1] + thrm_can
                             - self.srht - self.trht + self.evapvd * (ELH + SHV * self.tp[0, 1]) * self.fd)
                          * (1.0 - self.fm * self.fr_snow[1]))
            snsh_vapor[0, 1] = (self.evapvw * self.fw + self.evapvd * self.fd) * SHV * self.tp[0, 1]
            self.fch[1] = (-(thrm_can - thrm_soil[1]) * (1.0 - self.fr_snow[1])
                          - (thrm_can - self.thrmsn[1]) * self.fr_snow[1] * (1.0 - self.fm)
                          - self.htdripw[1] - self.htdrips[1])
        self.thrm_tot = np.array([
            thrm_soil[0] * (1.0 - self.fr_snow[0]) + self.thrmsn[0] * self.fr_snow[0],
            thrm_can * (1.0 - self.fr_snow[1] * self.fm) + self.thrmsn[1] * self.fr_snow[1] * self.fm])
        self.snsh_tot = np.array([
            (self.snshg[0] + snsh_vapor[1, 0]) * (1.0 - self.fr_snow[0]) + self.snshs[0] * self.fr_snow[0],
            (self.snshv[1] + snsh_vapor[0, 1]) * (1.0 - self.fr_snow[1] * self.fm)
            + self.snshs[1] * self.fr_snow[1] * self.fm
            + (self.snshg[1] + snsh_vapor[1, 1]) * (1.0 - self.fr_snow[1])])

    def fl(self):
        n = self.n
        for ibv in range(self.i_bare, self.i_vege + 1):
            self.f[n, ibv] = 0.0  # f(n+1,ibv) fortran -> python f[n,ibv] (1-based n+1 -> 0-based n)
            for k in range(2, n + 1):
                self.f[k - 1, ibv] = -self.xk[k, ibv] * (self.h[k - 1, ibv] - self.h[k, ibv]) / (self.zc[k - 2] - self.zc[k - 1])
            self.xinfc = getattr(self, 'xinfc', np.zeros(2))
            self.xinfc[ibv] = self.xk[1, ibv] * self.h[1, ibv] / self.zc[0]

    def flh(self):
        n = self.n
        for ibv in range(self.i_bare, self.i_vege + 1):
            self.fh[n, ibv] = self.geothermal_heat
            for k in range(2, n + 1):
                val = -self.xkhm[k, ibv] * (self.tp[k - 1, ibv] - self.tp[k, ibv]) / (self.zc[k - 2] - self.zc[k - 1])
                if self.f[k - 1, ibv] > 0:
                    val += self.f[k - 1, ibv] * self.tp[k, ibv] * SHW
                else:
                    val += self.f[k - 1, ibv] * self.tp[k - 1, ibv] * SHW
                self.fh[k - 1, ibv] = val

    def runoff(self):
        n = self.n
        rosmp = 8.0
        self.rnff = np.zeros((n + 1, 2))
        self.rnf = np.array([self.pr + self.irrig[0], self.pr + self.irrig[1]])
        f1_conv = np.zeros(2)
        if self.process_bare:
            f1_conv[0] = -(self.dripw[0] - self.evapb - self.dripw_scale[0])
        if self.process_vege:
            f1_conv[1] = -(self.dripw[1] - self.evapvg - self.dripw_scale[1])
        for ibv in range(self.i_bare, self.i_vege + 1):
            satfrac = min((self.w[1, ibv] / self.ws[1, ibv]) ** rosmp, 0.6)
            self.rnf[ibv] = satfrac * max(-self.f[0, ibv], 0.0)
            water_down = max(0.0, -f1_conv[ibv])
            if water_down > 1e-30:
                self.rnf[ibv] += (1.0 - self.fr_snow[ibv]) * (1.0 - satfrac) * water_down * math.exp(
                    -self.xinfc[ibv] * PRFR / water_down)
            water_down = max(-self.f[0, ibv] - water_down, 0.0)
            if water_down > 1e-30:
                self.rnf[ibv] += (1.0 - satfrac) * water_down * math.exp(-self.xinfc[ibv] / water_down)
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(1, n + 1):
                self.rnff[k - 1, ibv] = self.xku[k, ibv] * self.sl * self.dz[k - 1] / 100.0

    def fllmt(self):
        n = self.n
        trunc = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(n, 1, -1):
                wn = (self.w[k, ibv] + (self.f[k, ibv] - self.f[k - 1, ibv] - self.rnff[k - 1, ibv]
                      - self.fd * (1.0 - self.fr_snow[1] * self.fm) * (self.evapdl[k - 1] if ibv == 1 else 0.0)) * self.dts)
                if wn - self.ws[k, ibv] > trunc:
                    self.f[k - 1, ibv] += (wn - self.ws[k, ibv] + trunc) / self.dts
                if wn - self.dz[k - 1] * self.thetm[k, ibv] < trunc:
                    self.rnff[k - 1, ibv] += (wn - self.dz[k - 1] * self.thetm[k, ibv] - trunc) / self.dts
                    if self.rnff[k - 1, ibv] < 0.0:
                        self.f[k - 1, ibv] += self.rnff[k - 1, ibv]
                        self.rnff[k - 1, ibv] = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            evd1 = self.evapdl[0] if ibv == 1 else 0.0
            wn = (self.w[1, ibv] + (self.f[1, ibv] - self.f[0, ibv] - self.rnf[ibv] - self.rnff[0, ibv]
                  - self.fd * (1.0 - self.fr_snow[1] * self.fm) * evd1) * self.dts)
            if wn - self.ws[1, ibv] > trunc:
                self.rnf[ibv] += (wn - self.ws[1, ibv] + trunc) / self.dts
            if wn - self.dz[0] * self.thetm[1, ibv] < trunc:
                self.rnf[ibv] += (wn - self.dz[0] * self.thetm[1, ibv] - trunc) / self.dts
        for ibv in range(self.i_bare, self.i_vege + 1):
            k = 1
            while self.rnf[ibv] < 0.0 and k <= n:
                if k > 1:
                    evdl = self.evapdl[k - 1] if ibv == 1 else 0.0
                    dflux = (self.f[k, ibv] + (self.w[k, ibv] - self.dz[k - 1] * self.thetm[k, ibv]) / self.dts
                            - self.f[k - 1, ibv] - self.rnff[k - 1, ibv]
                            - self.fd * (1.0 - self.fr_snow[1] * self.fm) * evdl)
                    self.f[k - 1, ibv] -= self.rnf[ibv]
                    self.rnf[ibv] += min(-self.rnf[ibv], dflux)
                drnf = min(-self.rnf[ibv], self.rnff[k - 1, ibv])
                self.rnf[ibv] += drnf
                self.rnff[k - 1, ibv] -= drnf
                k += 1
            self.rnf[ibv] = max(self.rnf[ibv], 0.0)

    def apply_fluxes(self):
        n = self.n
        self.w[0, 1] = self.w[0, 1] + (self.fc[1] - self.fc[0]) * self.dts
        self.ht[0, 1] = self.ht[0, 1] + (self.fch[1] - self.fch[0]) * self.dts
        for ibv in range(self.i_bare, self.i_vege + 1):
            self.w[1, ibv] -= self.rnf[ibv] * self.dts
            self.ht[1, ibv] -= SHW * max(self.tp[1, ibv], 0.0) * self.rnf[ibv] * self.dts
            for k in range(1, n + 1):
                evdl = self.evapdl[k - 1] if ibv == 1 else 0.0
                self.w[k, ibv] += (self.f[k, ibv] - self.f[k - 1, ibv] - self.rnff[k - 1, ibv]
                                  - self.fd * (1.0 - self.fr_snow[1] * self.fm) * evdl) * self.dts
                self.ht[k, ibv] += (self.fh[k, ibv] - self.fh[k - 1, ibv]
                                    - SHW * max(self.tp[k, ibv], 0.0) * self.rnff[k - 1, ibv]) * self.dts
        if self.w[0, 1] < 0.0:
            self.w[0, 1] = 0.0
        for ibv in (0, 1):
            for k in range(1, n + 1):
                self.w[k, ibv] = max(min(self.w[k, ibv], self.ws[k, ibv]), self.dz[k - 1] * self.thetm[k, ibv])

    def gdtm(self, nit):
        n = self.n
        t450 = 450.0
        dqdt = dqsatdt(self.ts, LHE) * qsat(self.ts, LHE, self.pres)
        dldz2 = 0.0
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(1, n + 1):
                dldz2 = max(dldz2, self.d[k, ibv] / self.dz[k - 1] ** 2)
        dtm = 1.0 / (dldz2 + 1e-12)
        if self.q[3, 0] > 0.0:
            dtm = min(dtm, t450)
        for ibv in range(self.i_bare, self.i_vege + 1):
            for k in range(1, n + 1):
                xk1 = self.xkh[k, ibv]
                ak1 = (self.shc[k, ibv] + ((1.0 - self.fice[k, ibv]) * SHW + self.fice[k, ibv] * SHI)
                       * self.w[k, ibv]) / self.dz[k - 1]
                dtm = min(dtm, .5 * ak1 * self.dz[k - 1] ** 2 / (xk1 + 1e-12))
        cna = self.ch * self.vs
        rho3 = .001 * self.rho
        betas = [1.0, 1.0]
        if self.epb > 0.0:
            betas[0] = self.evapb / self.epb
        if self.epv > 0.0:
            betas[1] = (self.evapvw * self.fw + self.evapvd * (1.0 - self.fw)) / self.epv
        for ibv in range(self.i_bare, self.i_vege + 1):
            k = 1 - ibv  # fortran k=2-ibv, ibv=1,2 -> here ibv 0,1 -> k=1,0
            xk2 = (SHA * self.rho * cna + betas[ibv] * rho3 * cna * ELH * dqdt
                   + 8.0 * STBO * (self.tp[k, ibv] + TFRZ) ** 3)
            ak2 = self.shc[k, ibv] + ((1.0 - self.fice[k, ibv]) * SHW + self.fice[k, ibv] * SHI) * self.w[k, ibv]
            dtm = min(dtm, 0.5 * ak2 / (xk2 + 1e-12))
        return dtm

    def snow(self):
        fmask = [1.0, self.fm]
        evapsn = [self.evapbs, self.evapvs]
        devapsn_dt = [self.devapbs_dt, self.devapvs_dt]
        canht = STBO * (self.tp[0, 1] + TFRZ) ** 4
        self.flmlt = np.zeros(2); self.fhsng = np.zeros(2)
        self.flmlt_scale = np.zeros(2); self.fhsng_scale = np.zeros(2)
        self.thrmsn = np.zeros(2)
        for ibv in range(self.i_bare, self.i_vege + 1):
            out = snow_drv(fmask[ibv], evapsn[ibv], self.snshs[ibv], self.srht, self.trht, canht,
                           self.drips[ibv], self.dripw[ibv], self.htdrips[ibv], self.htdripw[ibv],
                           devapsn_dt[ibv], self.dsnsh_dt, self.evap_min, self.dts,
                           self.tp[1, ibv], self.dz[0], self.dzsn[:, ibv], self.wsn[:, ibv], self.hsn[:, ibv],
                           int(self.nsn[ibv]), self.fr_snow[ibv])
            evapsn[ibv] = out['evap']
            self.snshs[ibv] = out['snsh']
            self.nsn[ibv] = out['nsn']
            self.fr_snow[ibv] = out['fr_snow']
            self.flmlt[ibv] = out['flmlt']; self.fhsng[ibv] = out['fhsng']
            self.flmlt_scale[ibv] = out['flmlt_scale']; self.fhsng_scale[ibv] = out['fhsng_scale']
            self.thrmsn[ibv] = out['thrmsn']
        self.evapbs, self.evapvs = evapsn

    def accm_zero(self):
        (self.atrg, self.ashg, self.aevap, self.aruns, self.aeruns, self.arunu, self.aerunu,
         self.ae0, self.af1dt, self.aedifs, self.abetad) = (0.0,) * 11

    def accm(self):
        self.atrht = self.trht - (self.thrm_tot[0] * self.fb + self.thrm_tot[1] * self.fv)
        self.asrht = self.srht
        self.atrg += (self.thrm_tot[0] * self.fb + self.thrm_tot[1] * self.fv) * self.dts
        self.ashg += (self.snsh_tot[0] * self.fb + self.snsh_tot[1] * self.fv) * self.dts
        self.aevap += (self.evap_tot[0] * self.fb + self.evap_tot[1] * self.fv) * self.dts
        self.alhg = ELH * self.aevap
        self.aruns += (self.fb * self.rnf[0] + self.fv * self.rnf[1]) * self.dts
        self.aeruns += SHW * (self.fb * max(self.tp[1, 0], 0.0) * self.rnf[0]
                              + self.fv * max(self.tp[1, 1], 0.0) * self.rnf[1]) * self.dts
        for k in range(1, self.n + 1):
            self.arunu += (self.rnff[k - 1, 0] * self.fb + self.rnff[k - 1, 1] * self.fv) * self.dts
            self.aerunu += SHW * (max(self.tp[k, 0], 0.0) * self.rnff[k - 1, 0] * self.fb
                                  + max(self.tp[k, 1], 0.0) * self.rnff[k - 1, 1] * self.fv) * self.dts
        dedifs = self.f[1, 0] * (self.tp[2, 0] if self.f[1, 0] >= 0 else self.tp[1, 0]) if self.n >= 2 else 0.0
        self.aedifs -= self.dts * SHW * dedifs * self.fb
        dedifs = self.f[1, 1] * (self.tp[2, 1] if self.f[1, 1] >= 0 else self.tp[1, 1]) if self.n >= 2 else 0.0
        self.aedifs -= self.dts * SHW * dedifs * self.fv
        self.ae0 -= self.dts * (
            -self.srht - self.trht - self.htpr
            + (self.thrm_tot[0] + self.snsh_tot[0] + ELH * self.evap_tot[0]) * self.fb
            + (self.thrm_tot[1] + self.snsh_tot[1] + ELH * self.evap_tot[1]) * self.fv)
        self.af1dt -= self.dts * (self.fb * self.fh[1, 0] + self.fv * self.fh[1, 1])

    def accm_final(self):
        self.aruns *= RHOW; self.arunu *= RHOW; self.aevap *= RHOW
        self.af1dt -= self.aedifs
        h0 = self.fb * (self.snsh_tot[0] + ELH * self.evap_tot[0]) + self.fv * (self.snsh_tot[1] + ELH * self.evap_tot[1])
        self.tbcs = math.sqrt(math.sqrt(self.atrg / (self.dt * STBO))) - TFRZ
        self.tsns = ((self.snsh_tot[0] * self.fb + self.snsh_tot[1] * self.fv) / (SHA * self.rho * self.ch)
                    + self.gusti * self.tprime) / self.vs + self.ts - TFRZ

    # -------------------------------------------------------------- main driver
    def advnc(self, ent_iters, dt, snowm, cnc0=0.0):
        """ent_iters: list of dicts, one per Fortran `nit` (1-based order), each with
        cnc,betadl(6),TRANS_SW,Ci,GPP,lai,IPP,dts (as recorded by the instrumented model).
        Uses the RECORDED dts per iteration rather than recomputing gdtm, to isolate this
        port's physics from any difference in adaptive-step selection."""
        self.dt = dt
        self.snowm = snowm
        self._bounds()
        self.accm_zero()
        self.reth()
        self.retp()
        self.tb0 = self.tp[1, 0]; self.tc0 = self.tp[0, 1]
        self.evapb = self.epb = 1.0
        self.evapvw = self.evapvd = self.epv = 1.0
        for it in ent_iters:
            self.dts = it['dts']
            self.cnc = it['cnc'] if self.process_vege else 0.0
            self.betadl = np.asarray(it['betadl']) if self.process_vege else np.zeros(self.n)
            self.lai = it['lai'] if self.process_vege else 0.0
            self.hydra()
            self.xklh()
            self.evap_limits(True)
            self.drip_from_canopy()
            self.sensible_heat()
            self.snow()
            self.fl()
            self.flg()
            self.runoff()
            self.fllmt()
            self.flh()
            self.flhg()
            self.apply_fluxes()
            self.accm()
            self.reth()
            self.retp()
        self.accm_final()
        self.hydra()
