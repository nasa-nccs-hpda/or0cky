"""Daily products of the radiation code for an arbitrary calendar day, from the ModelE input files (R6, owner request 2026-10-12).

Replaces, for the dates other than the dump's, the products that `radia_jax.Products` takes from a dump: the Fortran readers they port are
RCOMPT (RADIATION.f) = UPDGHG / UPDO3D / updateAerosol2 / UPDDST2 / UPDVOL2 / UPDEPS / UPDSUR, updBCdalbsn (READ_AERO.f), DAILY_orbit (RAD_DRV.f),
the time interpolation of `timestream_mod` (linm2m, ppm) and REPART (RAD_UTILS.f).  Input files are those of the rundeck (names O3file, TAero_*,
DUSTaer, BCdalbsn, GHG, RADN7, RADN8 in the run directory or a directory holding links of that name).

Rundeck assumptions (checked against dumps, README / report 12.5 and 14): master_yr = 1850 for the ghg/o3/aerosol/dust/albsn years (a year earlier
than the first file of a decadal stream selects the first file), volc_yr = -1 (long-term mean), calendar = 365-day Julian, orbit parameters of
1850 (variable_orb_par = 0), MADAER = 3, MADVOL = 2, GHG reference (XREF/PPMV80) of the run's start date.

usage:  dp = DailyProducts(rundir, ref_jday=330);  P = dp.at(jday)   # dict of the products of the day (grid fields (72, 46, ...))
"""
import os
import numpy as np

JDEND = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
JDMID = np.array([-15, 16, 45, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350, 381])        # index 0..13
BY12 = 1.0 / 12.0
SEVBY12 = 7.0 * BY12
IM, JM = 72, 46


def _nc(path):
    import netCDF4
    return netCDF4.Dataset(path)


# ------------------------------------------------------------------------------------------------------------------ timestream_mod
def stream_files(d):
    """{year: path} of the files 'YYYY.nc' of a multi-file stream directory."""
    out = {}
    for f in os.listdir(d):
        if f.endswith(".nc") and f[:4].isdigit():
            out[int(f[:4])] = os.path.join(d, f)
    return dict(sorted(out.items()))


def year_weights(years, jyear):
    """get_wts: (index of the file at or after jyear, weight of the file before it)."""
    jj = 0
    while jj < len(years) and years[jj] < jyear:
        jj += 1
    if jj == 0:
        wtl = 0.0
    elif jj >= len(years):
        wtl = 1.0
    else:
        wtl = float(years[jj] - jyear) / float(years[jj] - years[jj - 1])
    return jj, wtl


def _read(path, var):
    """A float32 variable as float64 with denormal float32 values flushed to zero (the Fortran build runs with flush-to-zero / denormals-are-zero)."""
    v = np.array(_nc(path).variables[var][:])
    if v.dtype == np.float32:
        v = np.where(np.abs(v) < np.finfo(np.float32).tiny, np.float32(0.0), v)
    return v.astype(np.float64)


def load_monthly(path_or_dir, var, jyear):
    """The 12 monthly fields of `var` of the stream for the (fixed) year jyear: (12, ..., lat, lon) float64 as stored.
    A directory with several yearly files is interpolated in year the way read_stream_netcdf does (wtl*file before + wtr*file after)."""
    if os.path.isdir(path_or_dir):
        files = stream_files(path_or_dir)
        yrs = list(files)
        jj, wtl = year_weights(yrs, jyear)
        if jj == 0:
            q = _read(files[yrs[0]], var)
        elif jj >= len(yrs):
            q = _read(files[yrs[-1]], var)
        else:
            q = wtl * _read(files[yrs[jj - 1]], var) + (1.0 - wtl) * _read(files[yrs[jj]], var)
    else:
        q = _read(path_or_dir, var)
    return q


def _mon_of(jday):
    jmon = 1
    while jday > JDEND[jmon]:
        jmon += 1
    return jmon


def padded(q12, m):
    """qty(:,:,:,m) of the cyclic padded year, m = -1..14 (m=0: December, m=13: January)."""
    return q12[(m - 1) % 12]


def linm2m(q12, jday):
    """Monthly data, linear between mid-month days (timestream_mod, method 'linm2m')."""
    jmon = _mon_of(jday)
    imon = jmon if jday <= JDMID[jmon] else jmon + 1
    frac = float(JDMID[imon] - jday) / float(JDMID[imon] - JDMID[imon - 1])
    return frac * padded(q12, imon - 1) + (1.0 - frac) * padded(q12, imon)


def ppm_nonneg(q12, jday, qmin=0.0, qmax=1e30):
    """Piecewise-parabolic monthly-to-daily (timestream_mod, method 'ppm' with qmin = 0, qmax >= 1e20: ppm_nonneg)."""
    jmon = _mon_of(jday)
    jdate = jday - JDEND[jmon - 1]
    # time=(jdate-.5)/(jdendofm(jmon)-jdendofm(jmon-1))-.5 : default-real literals and integers, so the expression is evaluated in REAL*4
    f4 = np.float32
    time = float(f4(f4(f4(jdate) - f4(0.5)) / f4(JDEND[jmon] - JDEND[jmon - 1])) - f4(0.5))

    def eom(me):
        x0, x1, x2, x3 = padded(q12, me - 1), padded(q12, me), padded(q12, me + 1), padded(q12, me + 2)
        xe = -BY12 * (x0 + x3) + SEVBY12 * (x1 + x2)
        xe = np.minimum(np.maximum(qmin, xe), qmax)
        xe = np.where(np.minimum(x1, x2) <= qmin, qmin, xe)
        xe = np.where(np.maximum(x1, x2) >= qmax, qmax, xe)
        return xe
    a = padded(q12, jmon)
    e0, e1 = eom(jmon - 1), eom(jmon)
    b = e1 - e0
    c = 3.0 * (e1 + e0) - 6.0 * a
    out = a + b * time + c * (time ** 2 - BY12)
    with np.errstate(divide="ignore", invalid="ignore"):
        csq = c * (a * c - 0.25 * b ** 2 - c ** 2 * BY12)
        bq = 0.5 * (e0 ** 2 + e1 ** 2) / a
        lin = np.where(e0 - bq * (time + 0.5) > 0.0, e0 - bq * (time + 0.5),
                       np.where(e1 - bq * (0.5 - time) > 0.0, e1 - bq * (0.5 - time), 0.0))
    fix = (np.abs(c) > np.abs(b)) & (csq < 0.0)
    out = np.where(fix, lin, out)
    return np.where(a <= 0.0, a, out)


# ------------------------------------------------------------------------------------------------------------------ REPART
def repart_weights(xlb, ylb):
    """Overlap fractions of REPARTINT(oper='repart') (RAD_UTILS.f labels 100-160 for increasing, 200-260 for decreasing coordinates; the two
    blocks are the same logic with the comparison sense reversed) as, per target layer J, the list [(I, PART), ...] in the order of
    accumulation, so that sum(PART*FXL) has the rounding of the Fortran.  xlb (source edges), ylb (target edges): pressures, decreasing."""
    nxf, nyg = len(xlb) - 1, len(ylb) - 1
    res = [[] for _ in range(nyg)]
    dec = xlb[1] < xlb[0]
    after = (lambda p, q: p < q) if dec else (lambda p, q: p > q)               # IF(YB > XA), IF(XB > YA), IF(YA > XA), IF(YB > XB)
    i, j = 0, 0
    xa, ya = xlb[0], ylb[0]
    yb = ylb[1]
    while not after(yb, xa):                                                    # 100 / 200: target layers outside the source range
        j += 1
        if j >= nyg:
            return res
        ya = yb
        yb = ylb[j + 1]
    while True:                                                                 # 110 / 210: skip source layers outside the target layer
        xb = xlb[i + 1]
        if after(xb, ya):
            break
        i += 1
        if i >= nxf:
            return res
        xa = xb
    while True:                                                                 # 120 / 220
        xaya = ya if after(ya, xa) else xa
        if not after(yb, xb):                                                   # target layer ends inside the source layer
            res[j].append((i, (yb - xaya) / (xb - xa)))
            j += 1
            if j >= nyg:
                return res
            ya = yb
            yb = ylb[j + 1]
            continue
        res[j].append((i, (xb - xaya) / (xb - xa)))                             # 130 / 230: source layer consumed
        i += 1
        if i >= nxf:
            return res
        xa = xb
        xb = xlb[i + 1]


def repart_apply(weights, f):
    """gyl(J) = sum PART*FXL(I) in accumulation order; f has the source layer index on axis 0; returns (nyg, ...)."""
    out = np.zeros((len(weights),) + f.shape[1:])
    for j, lst in enumerate(weights):
        s = np.zeros(f.shape[1:])
        for i, part in lst:
            s = s + part * f[i]
        out[j] = s
    return out


# ------------------------------------------------------------------------------------------------------------------ small readers
def read_eps(path):
    """RADN8: Fortran sequential unformatted, big-endian records; tag(80) + REAL*4 EPLMHC(72,46,12,4); RADIATION.f 1260-1288 fills the
    negative entries (extend the end values to the poles, linear interpolation across gaps) in place, REAL*8 intermediates stored to REAL*4."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(raw[:4].tobytes(), dtype=">u4")[0])
    body = raw[4:4 + n]
    ep = np.frombuffer(body[80:].tobytes(), dtype=">f4").reshape((72, 46, 12, 4), order="F").astype(np.float32)
    for N in range(4):
        for M in range(12):
            for I in range(72):
                v = ep[I, :, M, N]
                j = 45
                while v[j] < 0:
                    j -= 1
                if j < 45:
                    v[j + 1:] = v[j]
                j = 0
                while v[j] < 0:
                    j += 1
                if j > 0:
                    v[:j] = v[j]
                if j == 45:
                    continue
                while True:
                    j += 1
                    stop = False
                    while v[j] >= 0:
                        j += 1
                        if j > 45:
                            stop = True
                            break
                    if stop:
                        break
                    k = j - 1
                    epk = np.float64(v[k])
                    j += 1
                    while v[j] < 0:
                        j += 1
                    l = j
                    epl = np.float64(v[l])
                    dep = (epl - epk) / (l - k)
                    for nn in range(1, l - k):
                        v[k + nn] = np.float32(epk + nn * dep)
                    if not j < 45:
                        break
    return ep


def read_ghg(path):
    """GHG table: 4 header lines (5 if the 5th starts with '--'), then one line per year: year, CO2 N2O CH4 CFC11 CFC12 others."""
    lines = open(path).read().splitlines()
    nh = 4
    if lines[4].startswith("--"):
        nh = 5
    rows = [l.split() for l in lines[nh:] if l.strip()]
    yrs = np.array([int(r[0]) for r in rows])
    am = np.array([[float(x) for x in r[1:7]] for r in rows])
    for n in range(1, len(am)):
        am[n][am[n] < 0] = am[n - 1][am[n] < 0]
    return int(yrs[0]), int(yrs[-1]), am.T                        # (6, nyears)


def gtrend(tab, tnow):
    """GTREND: xnow(1..6) at time tnow (decimal year)."""
    y1, y2, am = tab
    year = tnow
    if tnow <= y1 + 0.5:
        year = y1 + 0.5
    if tnow >= y2 + 0.49999:
        year = y2 + 0.49999
    dy = year - (y1 + 0.5)
    iy = int(dy)
    frac = dy - iy
    return am[:, iy] + frac * (am[:, iy + 1] - am[:, iy])


# ------------------------------------------------------------------------------------------------------------------ the day
AERNAMES = ("SUL", "SSA", "NIT", "OCA", "BCA", "BCB")
PSF = 984.0


class DailyProducts:
    """Per-day products on the (72, 46) radiation grid from the rundeck's input files.
    rundir: directory with the files (or links) O3file, TAero_SUL..BCB, DUSTaer, BCdalbsn, GHG, RADN8.
    plb0: reference layer-edge pressures of the radiation layering (44 values, decreasing), as in RAD_COM / the dumps.
    drym2g (8): the aerosol mass-to-optical-depth factors DRYM2G of AerParam_mod AFTER the radiation set-up (0.75/DENAER*Q55/AREFF of SETAER; a
    static of the run, in the dump's tag 60);  ppmv80 (13): PPMV80 of the dump (the XREF of the GHG trend are these, not the table's run start).
    The streams use the year master_yr (a year before the first file of a decadal stream selects the first file)."""

    def __init__(self, rundir, plb0, drym2g, ppmv80, taucon_dust, solar_constant=None, master_yr=1850, ghg_yr=None, ghg_tab=None, ghg_day=182,
                 albvnd=None):
        self.dir = rundir
        self.plb0 = np.asarray(plb0, dtype=np.float64)
        self.year = master_yr
        self.nl = len(self.plb0) - 1
        self.drym2g = np.asarray(drym2g, dtype=np.float64)
        self.taucon = np.asarray(taucon_dust, dtype=np.float64)                              # 0.75e3*QDST55/(RODUST*REDUST), static
        self.solar_constant = solar_constant                                                 # S0 of the run (tag 173, second element)
        self.ppmv80 = np.asarray(ppmv80, dtype=np.float64)
        self.albvnd = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "albvnd.npz"))["albvnd"] if albvnd is None else albvnd
        self._cache = {}
        j = lambda n: os.path.join(rundir, n)
        # ozone
        self.o3 = load_monthly(j("O3file"), "O3", master_yr)                              # (12, plm, lat, lon)
        ple = np.array(_nc(sorted(stream_files(j("O3file")).values())[0]).variables["ple"][:], dtype=np.float64)
        if ple[0] < PSF:
            ple[0] = PSF
        self.w_o3 = repart_weights(ple, self.plb0)
        # tropospheric aerosols (A6JDAY, 1000*DRYM2G scaling)
        self.aer = [load_monthly(j("TAero_" + n), n, master_yr) for n in AERNAMES]      # (12, lev, lat, lon)
        first = sorted(stream_files(j("TAero_SSA")).values())[0]
        plbaer = np.array(_nc(first).variables["plbaer"][:], dtype=np.float64)
        self.w_aer = repart_weights(plbaer, self.plb0)
        # dust
        dfirst = sorted(stream_files(j("DUSTaer")).values())[0]
        dn = _nc(dfirst)
        self.dust = load_monthly(j("DUSTaer"), "DUST", master_yr)                          # (12, nsized, lev, lat, lon)
        self.plbdust = np.array(dn.variables["plbdust"][:], dtype=np.float64)
        self.redust = np.array(dn.variables["redust"][:], dtype=np.float64)
        self.rodust = np.array(dn.variables["rodust"][:], dtype=np.float64)
        self.w_dust = repart_weights(self.plbdust, self.plb0)
        # black-carbon snow albedo change (percent -> fraction)
        self.bcd = load_monthly(j("BCdalbsn"), "BCdalbsn", master_yr)                      # (12, lat, lon)
        # cloud heterogeneity
        self.eps = read_eps(j("RADN8"))                                                    # (72, 46, 12, 4) float32
        # greenhouse gases
        self.ghg = read_ghg(j("GHG")) if ghg_tab is None else ghg_tab
        self.ghg_yr = master_yr if ghg_yr is None else ghg_yr
        self.ghg_day = ghg_day                                                             # GHG_day of RAD_COM (default 182); the 1980 reference of SETGHG is at day 184
        # XREF: GTREND at the default reference mid 1980 (SETGHG(1980, 184)), floored at 1e-6; XREF(7) = 1e-13
        xr = gtrend(self.ghg, 1980 + (184 - 0.999) / 366.0)
        self.xref = np.concatenate([np.maximum(xr, 1.0e-6), [1.0e-13]])

    # ------------------------------------------------------------------------------------------------------------- fields on the grid
    def ozone(self, jday):
        a = ppm_nonneg(self.o3, jday)                                                      # (plm, lat, lon)
        return repart_apply(self.w_o3, a.reshape(a.shape[0], -1)).reshape((self.nl,) + a.shape[1:])

    def aerosol(self, jday):
        out = []
        for n, q in enumerate(self.aer):
            a = linm2m(q, jday) * (1000.0 * self.drym2g[n])                                    # (1000*DRYM2G) * A6JDAY
            out.append(repart_apply(self.w_aer, a.reshape(a.shape[0], -1)).reshape((self.nl,) + a.shape[1:]))
        return np.stack(out, axis=-1)                                                     # (nl, lat, lon, 6)

    def dust_cols(self, jday):
        a = linm2m(self.dust, jday)                                                       # (nsized, lev, lat, lon)
        ns = a.shape[0]
        out = [repart_apply(self.w_dust, a[n].reshape(a.shape[1], -1)).reshape((self.nl,) + a.shape[2:]) for n in range(ns)]
        return np.stack(out, axis=-1) * self.taucon                                       # DTAULX(:,n)*TAUCON_dust(n): (nl, lat, lon, nsized)

    def dalbsn(self, jday):
        return linm2m(self.bcd, jday) / 100.0                                              # (lat, lon)

    def epsilon(self, jday):
        xjday = jday - 0.999
        xmo = xjday / 30.5 + 0.5
        mi = int(xmo)
        wtmj = xmo - mi
        wtmi = 1.0 - wtmj
        if mi < 1:
            mi = 12
        mj = mi + 1
        if mj > 12:
            mj = 1
        e = self.eps.astype(np.float64)
        return [wtmi * e[:, :, mi - 1, k] + wtmj * e[:, :, mj - 1, k] for k in range(3)]  # EPLOW, EPMID, EPHIG: (72, 46)

    def fulgas_ppmv(self, jday, ppmv80_template, fulgas_template, co2x=1.0, n2ox=1.0, ch4x=1.0, cfc11x=1.0, cfc12x=1.0, xghgx=1.0, yghgx=2.0,
                    daily=False):
        """UPDGHG: FULGAS(2,6,7,8,9,11,12) = XNOW/XREF * X; the other elements of FULGAS come from the template (PPMV80 does not change).
        With a fixed GHG year (ghg_yr > 0, as here) RCOMPT does not call UPDGHG again (MADGHG = 0): the values of the call at start-up, at
        day ghg_day (182), hold for every day (daily=False); daily=True is the trend of the day for ghg_yr = 0 (unverified)."""
        tnow = self.ghg_yr + ((jday if daily else self.ghg_day) - 0.999) / 366.0
        xn = np.concatenate([gtrend(self.ghg, tnow), [1.0e-20]])
        f = np.array(fulgas_template, dtype=np.float64).copy()
        p = np.array(ppmv80_template, dtype=np.float64).copy()
        f[1] = xn[0] / self.xref[0] * co2x
        f[5] = xn[1] / self.xref[1] * n2ox
        f[6] = xn[2] / self.xref[2] * ch4x
        f[7] = xn[3] / self.xref[3] * cfc11x
        f[8] = xn[4] / self.xref[4] * cfc12x
        f[10] = xn[5] / self.xref[5] * xghgx
        f[11] = xn[6] / self.xref[6] * yghgx
        return f, p


# ------------------------------------------------------------------------------------------------------------------ orbit, surface
def rsdist(jday, year=1850.0):
    """RSDIST = distance**2 at noon of the day (DAILY_orbit); the orbit has a 365-day period, so only the day of year enters."""
    from fractions import Fraction
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import drv_zenith as Z
    orb = Z.EarthOrbit(year)
    t = Fraction(1 * Z.SECONDS_PER_YEAR + (jday - 1) * Z.SECONDS_PER_DAY + Z.SECONDS_PER_DAY // 2)
    ma = orb.mean_anomaly(t)
    m, e = orb.m, orb.ecc
    x = ma + e * (m["sin"](ma) + e * m["sin"](2.0 * ma) / 2.0)
    while True:
        f = x - e * m["sin"](x) - ma
        df = 1.0 - e * m["cos"](x)
        dx = -f / df
        x = x + dx
        if abs(dx) < 1.0e-10:
            break
    r = 1.0 * (1.0 - e * m["cos"](x))
    return r * r


SEASON = (15.0, 105.0, 196.0, 288.0)


def albvnh(jday, albvnd):
    """UPDSUR: (12, 6, 2) vegetation albedo of the day from ALBVND(12, 4, 6) (season) by linear interpolation in day."""
    xjday = float(jday)
    seasn1 = -77.0
    k = 0
    for k in range(4):
        seasn2 = SEASON[k]
        if xjday <= seasn2:
            break
        seasn1 = seasn2
    else:
        k = 0
        seasn2 = 380.0
    wt2 = (xjday - seasn1) / (seasn2 - seasn1)
    wt1 = 1.0 - wt2
    k1 = k + 1                                                                    # 1-based K of the Fortran
    ks1, ks2 = 1 + (k1 % 4), 1 + ((k1 + 1) % 4)
    kn1, kn2 = 1 + ((k1 + 2) % 4), k1
    out = np.zeros((12, 6, 2))
    out[:, :, 0] = wt1 * albvnd[:, ks1 - 1, :] + wt2 * albvnd[:, ks2 - 1, :]
    out[:, :, 1] = wt1 * albvnd[:, kn1 - 1, :] + wt2 * albvnd[:, kn2 - 1, :]
    return out


def day_dict(dp, jday, ilon, jlat, fulgas_template):
    """The day-varying per-column products of day-of-year jday (keys of radia_jax.RadiaJax.DAY_KEYS) for the columns (ilon, jlat) (1-based
    72 x 46 box indices of the radiation columns)."""
    il, jl = np.asarray(ilon, int) - 1, np.asarray(jlat, int) - 1
    eps = dp.epsilon(jday)
    fu, pp = dp.fulgas_ppmv(jday, dp.ppmv80, fulgas_template)
    out = dict(o3col=dp.ozone(jday)[:, jl, il].T, ataulx=dp.aerosol(jday)[:, jl, il, :].transpose(1, 0, 2),
               dtaulx=dp.dust_cols(jday)[:, jl, il, :].transpose(1, 0, 2), dalbsn=dp.dalbsn(jday)[jl, il],
               epl=eps[0][il, jl], epm=eps[1][il, jl], eph=eps[2][il, jl], albvnh=albvnh(jday, dp.albvnd), fulgas=fu, ppmv80=pp)
    if dp.solar_constant is not None:
        out["solar_irrad"] = np.float64(dp.solar_constant / rsdist(jday))
    return out


def jday_of_itime(itime, nsteps_per_day=48, days_per_year=365):
    """Day of year (1..365) of the model clock `itime` (itime 0 = 1949-01-01 00:00 of the 365-day calendar)."""
    return (int(itime) // nsteps_per_day) % days_per_year + 1


def day_products_for_run(dp, prod, itime_first, itime_last, nsteps_per_day=48):
    """(list of day dicts, day0) for the model days spanned by itime_first..itime_last, to be passed as
    RadiaJax(..., day_products=list, day0=day0); `prod` is the radia_jax.Products of the dump supplying the static content.
    One entry costs about N*(43*(1 + 6 + 7) + 5) doubles (N = 3170 columns: 16 MB); the arrays are indexed by the day inside the jit."""
    day0 = int(itime_first) // nsteps_per_day
    day1 = int(itime_last) // nsteps_per_day
    out = [day_dict(dp, jday_of_itime(d * nsteps_per_day), prod.ilon, prod.jlat, prod.g["fulgas"]) for d in range(day0, day1 + 1)]
    return out, day0
