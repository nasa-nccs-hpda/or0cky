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


def _seqsum(a, axis=0):
    """Sum in index order (the Fortran SUM intrinsic without reassociation); numpy's pairwise sum differs in the last bit."""
    return np.cumsum(a, axis=axis).take(-1, axis=axis)


def reterp_apply(weights, f):
    """RETERP (REPARTINT oper='interp'): gyl(J) = sum PART*FXL(I) / sum PART, both in accumulation order; layers without overlap are 0."""
    out = np.zeros((len(weights),) + f.shape[1:])
    for j, lst in enumerate(weights):
        if not lst:
            continue
        s = np.zeros(f.shape[1:])
        y = 0.0
        for i, part in lst:
            s = s + part * f[i]
            y = y + part
        out[j] = s / y
    return out


class VolcanicAerosol:
    """UPDVOL2 / get_volc_column (READ_AERO.f VolcParam_mod): stratospheric aerosol optical depth per layer and effective radius per
    latitude zone from the RADN7 file (Fortran sequential, big-endian: header record tag(80) + NVOLMON, first year, last year, JMv, LMv;
    records VTau4(NVOLMON, JMv, LMv), VReff4(NVOLMON, JMv), HVOLKM(LMv+1), ELATVOL(JMv+1), all REAL*4).
    hlb0: the reference layer-edge heights (km) of the radiation layering (dump tag 60 `hlb0`, 44 values, the same sense as the file's
    layer edges); kyearvm: the range of years of the long-term mean (volc_mean, default 1850-2014).
    at(jyearv, jjdayv): jyearv = -1 selects the long-term mean (volc_yr = -1 of the rundeck), otherwise the monthly data are interpolated to
    jyearv + jjdayv/366.  Returns (vtau (46, nl) bottom first, vreff (46,)) for the 46 latitude zones of the 4x5 grid."""

    def __init__(self, path, hlb0, kyearvm=(1850, 2014)):
        raw = np.fromfile(path, dtype=np.uint8)
        pos, rec = 0, []
        while pos < len(raw):
            n = int(np.frombuffer(raw[pos:pos + 4].tobytes(), ">u4")[0])
            rec.append(raw[pos + 4:pos + 4 + n])
            pos += 8 + n
        nm, self.y0, self.y1, jm, lm = [int(v) for v in np.frombuffer(rec[0][80:100].tobytes(), ">i4")]
        self.nmon, self.jm, self.lm = nm, jm, lm
        f4 = lambda r: np.frombuffer(r[80:].tobytes(), ">f4").astype(np.float32)
        self.vtau = f4(rec[1]).reshape((nm, jm, lm), order="F").astype(np.float64)
        self.vreff = f4(rec[2]).reshape((nm, jm), order="F").astype(np.float64)
        self.hvolkm = f4(rec[3]).astype(np.float64)
        self.elat = f4(rec[4]).astype(np.float64)
        e46 = np.empty(47)
        for j in range(2, 47):
            e46[j - 1] = -90.0 + (j - 1.5) * 180.0 / 45
        e46[0], e46[46] = -90.0, 90.0
        self.w_lat = repart_weights(self.elat, e46)
        self.hlb0 = np.asarray(hlb0, dtype=np.float64)
        self.w_h = repart_weights(self.hvolkm, self.hlb0)
        self.kyearvm = tuple(kyearvm)
        self._mean = None

    def _mean_fields(self):
        if self._mean is None:
            vmi = (self.kyearvm[0] - self.y0) * 12 + 1
            vme = (self.kyearvm[1] - self.y0 + 1) * 12
            if self.kyearvm[0] < self.y0 or self.kyearvm[1] > self.y1:
                raise ValueError("volc_mean %s outside the file's years %d-%d" % (self.kyearvm, self.y0, self.y1))
            tau, reff = self.vtau.copy(), self.vreff.copy()
            for j in range(self.jm):
                vt = _seqsum(self.vtau[:, j, :], axis=1)                              # (nmon): sum over layers of ALL months
                reff[0, j] = _seqsum(self.vreff[vmi - 1:vme, j] * vt[vmi - 1:vme]) / _seqsum(vt[vmi - 1:vme])
                tau[0, j, :] = _seqsum(self.vtau[vmi - 1:vme, j, :], axis=0) / (vme - vmi + 1)
            self._mean = (tau, reff)
        return self._mean

    def at(self, jyearv=-1, jjdayv=182):
        tau, reff = self._mean_fields() if jyearv == -1 else (self.vtau, self.vreff)
        xyyear = jyearv + jjdayv / 366.0
        if xyyear < self.y0:
            xyyear = float(self.y0)
        xyi = (xyyear - self.y0) * 12.0 + 1.0
        if xyi > self.nmon - 0.001:
            xyi = self.nmon - 0.001
        mi = int(xyi)
        wmj = xyi - mi
        wmi = 1.0 - wmj
        mj = mi + 1
        g = wmi * reff[mi - 1] + wmj * reff[mj - 1]                                   # (jm,)
        vreffj = reterp_apply(self.w_lat, g[:, None])[:, 0]
        gt = wmi * tau[mi - 1] + wmj * tau[mj - 1]                                    # (jm, lm)
        vtaujl = reterp_apply(self.w_lat, gt)                                         # (46, lm)
        vtaulx = repart_apply(self.w_h, vtaujl.T).T                                   # (46, nl)
        return vtaulx, vreffj


def spline4(x, f, xx, cuspwm=1.0, cuspwe=1.0, kxtrap=1):
    """SPLINEVector (RAD_UTILS.f): 4-point cubic spline value at xx of f (axis 0 along x); f may carry further axes (NVEC)."""
    x = np.asarray(x, float)
    f = np.asarray(f, float)
    nxf = len(x)
    k = 2                                                                       # 1-based as in the Fortran
    x2, x3 = x[k - 1], x[nxf - 2]
    betw = (xx - x2) * (x3 - xx)
    if betw > 0.0:                                                              # label 100/110
        while True:
            k += 1
            x3 = x[k - 1]
            betw = (xx - x2) * (x3 - xx)
            if betw >= 0.0:
                break
            x2 = x3
        f3, f4, x4 = f[k - 1], f[k], x[k]
        f2, x2 = f[k - 2], x[k - 2]
        f1, x1 = f[k - 3], x[k - 3]
        x21, x31, x32, x43, x42 = x2 - x1, x3 - x1, x3 - x2, x4 - x3, x4 - x2
        f21 = (f2 - f1) / (x21 * x21)
        f32 = (f3 - f2) / (x32 * x32)
        f43 = (f4 - f3) / (x43 * x43)
        f3221 = (f32 + f21) / x31 * x21
        f4332 = (f43 + f32) / x42 * x43
        a = f2
        b = x32 * f3221
        c = 3.0 * f32 - f3221 - f3221 - f4332
        d = (f3221 + f4332 - f32 - f32) / x32
        xf = xx - x2
        ffcusp = a + xf * (b + xf * (c + xf * d))
        xe = (x3 + x2 - xx - xx) / x32
        if xe < 0.0:
            xe = -xe
        xexm = xe ** 2
        w = (1.0 - xexm) * cuspwm + xexm * cuspwe
        fflinr = a + xf * f32 * x32
        return ffcusp * w + fflinr * (1.0 - w)
    raise NotImplementedError("spline4: only the interior 4-point branch is ported (the volcanic tables never leave it)")


class VolcanicTables:
    """GETQVA's Q55H2S (the 0.55 micron extinction efficiency of the H2SO4 aerosol at effective radius vreff, variance VEFF0 = 0.35) from the
    Mie tables of RADN3 (miescatpar.abcdv2: DATA Q55VA1/2/3/5, REFVA1) as SETQVA builds them: spline over the variances 0.1..0.5 (M = 4 is the
    mean of 3 and 5), then over the sizes for REFV(N) = N/10, N = 9..20."""

    def __init__(self, miefile, veff0=0.35):
        import re
        txt = open(miefile).read()

        def data(name):
            m = re.search(r"DATA\s+" + name + r"\s*/?\s*([^/]*)/", txt)
            return np.array([float(v) for v in re.sub(r"\n\s*[A-Z0-9]\s+", " ", m.group(1)).replace(",", " ").split()])
        cols = {1: data("Q55VA1"), 2: data("Q55VA2"), 3: data("Q55VA3"), 5: data("Q55VA5")}
        cols[4] = (cols[3] + cols[5]) / 2.0
        q20 = np.stack([cols[m] for m in (1, 2, 3, 4, 5)], axis=1)                  # (20, 5)
        rv20 = data("REFVA1")
        v5 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        q55v = np.array([spline4(v5, q20[n], veff0) for n in range(20)])         # at VEFF0, original sizes rv20
        qv20 = q55v.copy()
        for n in range(9, 21):                                                      # sizes N/10 (N = 9..20)
            q55v[n - 1] = spline4(rv20, qv20, n / 10.0)
        self.q55v = q55v

    def q55(self, vreff):
        radx = np.asarray(vreff, float) * 10.0
        radx = np.minimum(np.maximum(radx, 1.000001), 19.99999)
        lo = radx.astype(int)
        whi = radx - lo
        return (1.0 - whi) * self.q55v[lo - 1] + whi * self.q55v[lo]


class DailyProducts:
    """Per-day products on the (72, 46) radiation grid from the rundeck's input files.
    rundir: directory with the files (or links) O3file, TAero_SUL..BCB, DUSTaer, BCdalbsn, GHG, RADN8.
    plb0: reference layer-edge pressures of the radiation layering (44 values, decreasing), as in RAD_COM / the dumps.
    drym2g (8): the aerosol mass-to-optical-depth factors DRYM2G of AerParam_mod AFTER the radiation set-up (0.75/DENAER*Q55/AREFF of SETAER; a
    static of the run, in the dump's tag 60);  ppmv80 (13): PPMV80 of the dump (the XREF of the GHG trend are these, not the table's run start).
    The streams use the year master_yr (a year before the first file of a decadal stream selects the first file)."""

    def __init__(self, rundir, plb0, drym2g, ppmv80, taucon_dust, solar_constant=None, master_yr=1850, ghg_yr=None, ghg_tab=None, ghg_day=182,
                 albvnd=None, hlb0=None, volc_yr=-1, volc_day=182, volc_mean=(1850, 2014)):
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
        # stratospheric (volcanic) aerosol, RADN7: volc_yr = -1 long-term mean over volc_mean; > 0 the monthly data at (volc_yr, volc_day);
        # 0 the data of the model year of the day (volcanic(jday, jyear)).  Needs hlb0 (dump tag 60) for the vertical repartition.
        self.volc_yr, self.volc_day = volc_yr, volc_day
        self.volc = None
        if hlb0 is not None and os.path.exists(j("RADN7")):
            self.volc = VolcanicAerosol(j("RADN7"), hlb0, volc_mean)
            self.volc_tables = VolcanicTables(j("RADN3")) if os.path.exists(j("RADN3")) else None     # Q55H2S(vreff)

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

    def volcanic(self, jday=None, jyear=None):
        """(vtaulx (46, nl), vreff (46,)) of the day: constant for volc_yr = -1 or > 0 (volc_day), the model day (jday, jyear) for volc_yr = 0."""
        if self.volc is None:
            raise RuntimeError("DailyProducts was built without hlb0 / RADN7: no volcanic reader")
        key = ("volc", jyear, jday) if self.volc_yr == 0 else ("volc", self.volc_yr, self.volc_day)
        if key not in self._cache:
            self._cache[key] = self.volc.at(jyear, jday) if self.volc_yr == 0 else self.volc.at(self.volc_yr, self.volc_day)
        return self._cache[key]

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


def day_dict(dp, jday, ilon, jlat, fulgas_template, jyear=None):
    """The day-varying per-column products of day-of-year jday (keys of radia_jax.RadiaJax.DAY_KEYS) for the columns (ilon, jlat) (1-based
    72 x 46 box indices of the radiation columns)."""
    il, jl = np.asarray(ilon, int) - 1, np.asarray(jlat, int) - 1
    eps = dp.epsilon(jday)
    fu, pp = dp.fulgas_ppmv(jday, dp.ppmv80, fulgas_template)
    out = dict(o3col=dp.ozone(jday)[:, jl, il].T, ataulx=dp.aerosol(jday)[:, jl, il, :].transpose(1, 0, 2),
               dtaulx=dp.dust_cols(jday)[:, jl, il, :].transpose(1, 0, 2), dalbsn=dp.dalbsn(jday)[jl, il],
               epl=eps[0][il, jl], epm=eps[1][il, jl], eph=eps[2][il, jl], albvnh=albvnh(jday, dp.albvnd), fulgas=fu, ppmv80=pp)
    if dp.volc is not None:
        vt, vr = dp.volcanic(jday, jyear)
        out["vtaulx"], out["vreff"] = vt[jl], vr[jl]
        if dp.volc_tables is not None:
            out["q55h2s"] = dp.volc_tables.q55(vr[jl])
    if dp.solar_constant is not None:
        out["solar_irrad"] = np.float64(dp.solar_constant / rsdist(jday, float(dp.year)))
    return out


def jday_of_itime(itime, nsteps_per_day=48, days_per_year=365):
    """Day of year (1..365) of the model clock `itime` (itime 0 = 1949-01-01 00:00 of the 365-day calendar)."""
    return (int(itime) // nsteps_per_day) % days_per_year + 1


def jyear_of_itime(itime, nsteps_per_day=48, days_per_year=365, year0=1949):
    """Calendar year of the model clock `itime` (itime 0 = 1949-01-01 00:00)."""
    return year0 + int(itime) // (nsteps_per_day * days_per_year)


def day_products_for_run(dp, prod, itime_first, itime_last, nsteps_per_day=48):
    """(list of day dicts, day0) for the model days spanned by itime_first..itime_last, to be passed as
    RadiaJax(..., day_products=list, day0=day0); `prod` is the radia_jax.Products of the dump supplying the static content.
    One entry costs about N*(43*(1 + 6 + 7) + 5) doubles (N = 3170 columns: 16 MB); the arrays are indexed by the day inside the jit."""
    day0 = int(itime_first) // nsteps_per_day
    day1 = int(itime_last) // nsteps_per_day
    out = [day_dict(dp, jday_of_itime(d * nsteps_per_day), prod.ilon, prod.jlat, prod.g["fulgas"], jyear_of_itime(d * nsteps_per_day))
           for d in range(day0, day1 + 1)]
    return out, day0


class DayStream:
    """Day products on demand for RadiaJax(day_source=...): called with the model day number (itime // 48) returns the day dict (keys of
    RadiaJax.DAY_KEYS and, when the reader has the volcanic file, vtaulx / vreff).  Keeps the last `keep` days (a run needs one day at a time)."""

    def __init__(self, dp, prod, nsteps_per_day=48, keep=3):
        self.dp, self.prod, self.nspd, self.keep = dp, prod, nsteps_per_day, keep
        self._cache = {}

    def __call__(self, day):
        day = int(day)
        if day not in self._cache:
            if len(self._cache) >= self.keep:
                self._cache.pop(next(iter(self._cache)))
            it = day * self.nspd
            self._cache[day] = day_dict(self.dp, jday_of_itime(it, self.nspd), self.prod.ilon, self.prod.jlat, self.prod.g["fulgas"],
                                        jyear_of_itime(it, self.nspd))
        return self._cache[day]
