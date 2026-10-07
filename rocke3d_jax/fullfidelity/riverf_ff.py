"""D167: RIVERF (river routing / lake outflow) and its initialisation, ported from the pristine ModelE source (LAKES.f, read-only).

Version ported: the ORIGINAL RIVERF (LAKES.f:1708-2212, `#else` branch). P2SAoM40.R defines neither RVR_ELEV nor TOPO_DIRECTED_RIVER_FLOW
(checked in decks/P2SAoM40.R preprocessor options), and has no TRACERS_WATER, SCM or TRACERS_OBIO_RIVERS, so those branches are absent.
Also ported: the river part of init_LAKES (LAKES.f:889-1050: IFLOW/JFLOW/KDIREC/KD911/IFL911/JFL911/DHORZ/RATE from the river-direction file
RVR=RD_modelE_M.nc, read the way the model reads it), get_dir, horzdist_2pts, lonlat_to_ij (GEOM_B.f:430).
Statics read from the real input files (TOPO=Z72X46N_gas.1_nocasp.nc: focean, flake, zatmo (x GRAV = geopotential, ATM_COM.f:212), hlake
(max(.,1 m), LAKES.f init)).  NAMERVR (named river mouths) only feeds diagnostics and is not read.
Array convention: 0-based numpy [i, j], i = longitude (IM=72), j = latitude (JM=46); Fortran indices are converted at the boundary only.
Single-domain (one PE) latlon grid, as the reference run (OMP_NUM_THREADS=1): halo = periodic wrap in i is NOT used by the original code
(I_0H = I_0), the pole rows keep IMAXJ = 1.
Floating point: the expressions are written in the Fortran evaluation order (left to right) with python floats (IEEE double, no FMA).
sin/cos/acos in the geometry (DHORZ only) use the Intel libimf scalar functions when available (the real build's libm), else numpy.
Assumed -r8 for REAL literals (config/compiler.intel.mk R8): DZDH1 = .00005 is taken as the double 5e-5 (a REAL*4 literal would differ at 1e-8 relative;
`DZDH1_REAL4` is provided and the comparison reports which matches the real outflow).
"""
import ctypes
import math
import os

import numpy as np

IM, JM = 72, 46
DTSRC = 1800.0
RHOW, SHW, TF, TEENY = 1000.0, 4185.0, 273.15, 1e-30
GRAV = 9.80665
BYGRAV = 1.0 / GRAV
RADIUS = 6371000.0
PI = 3.14159265358979323846264338327950288
TWOPI = 2.0 * PI
RADIAN = PI / 180.0
URATE = 1e-6
LAKE_RISE_MAX = 100.0
RIVER_FAC = 1.0
HLAKE_MIN = 1.0
DZDH1_R8 = 5e-5
DZDH1_REAL4 = float(np.float32(5e-5))
DZDH1 = DZDH1_REAL4
PROD = '/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files'
TOPO_FILE = PROD + '/Z72X46N_gas.1_nocasp.nc'
RVR_FILE = PROD + '/RD_modelE_M.nc'

_IMF = os.environ.get("INTEL_LIBIMF_DIR", "/panfs/ccds02/app/modules/intel/platform/x86_64/rhel/8.6/2020Update4/"
                      "compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin")
_lib = None


def _imf():
    global _lib
    if _lib is None:
        try:
            ctypes.CDLL(os.path.join(_IMF, "libintlc.so.5"), mode=ctypes.RTLD_GLOBAL)
            lib = ctypes.CDLL(os.path.join(_IMF, "libimf.so"))
            for n in ('sin', 'cos', 'acos'):
                f = getattr(lib, n); f.restype = ctypes.c_double; f.argtypes = [ctypes.c_double]
            _lib = lib
        except OSError:
            _lib = False
    return _lib


def libm_mode():
    return 'intel libimf' if _imf() else 'numpy/glibc (libimf not found)'


def _fn(name):
    lib = _imf()
    return getattr(lib, name) if lib else getattr(math, name)


def available():
    return os.path.exists(TOPO_FILE) and os.path.exists(RVR_FILE)


def nint(x):
    """Fortran NINT (round half away from zero)."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def lonlat_to_ij(lon, lat):
    """GEOM_B.f:430 (1-based result)."""
    dlon_dg = 360.0 / float(IM)
    dlat_dg = 180.0 / float(JM - 1)
    i = nint(0.5 * (IM + 1) + lon / dlon_dg)
    j = nint(0.5 * (JM + 1) + lat / dlat_dg)
    return i, min(max(j, 1), JM)


def get_dir(i, j, id_, jd, im=IM, jm=JM):
    """LAKES.f get_dir (1-based indices), single PE with both poles."""
    di = i - id_
    if di == im - 1:
        di = -1
    if di == 1 - im:
        di = 1
    dj = j - jd
    table = {(-1, -1): 1, (-1, 0): 8, (-1, 1): 7, (0, 1): 6, (0, 0): 0, (0, -1): 2, (1, -1): 3, (1, 0): 4, (1, 1): 5}
    g = table.get((di, dj), -99)
    if j == jm:
        g = 6 if di == 0 else 8
    elif j == 1:
        g = 2 if di == 0 else 8
    elif j == jm - 1:
        if jd == jm:
            g = 2
    elif j == 2:
        if jd == 1:
            g = 6
    if g == -99:
        g = 0
    return g


def _geom_trig():
    """lon2d(i) = DLON*(i-.5), lat(j) (GEOM_B.f), sinlat2d = sin(lat), coslat2d = cos(lat); 1-based arrays of length IM / JM."""
    sin, cos = _fn('sin'), _fn('cos')
    dlon = TWOPI * (1.0 / float(IM))
    dlat = (180.0 / float(JM - 1)) * RADIAN
    fjeq = 0.5 * (1 + JM)
    lon = [dlon * (i - 0.5) for i in range(1, IM + 1)]
    lat = [dlat * (j - fjeq) for j in range(1, JM + 1)]
    lat[0] = -0.25 * TWOPI
    lat[JM - 1] = -lat[0]
    return lon, [sin(a) for a in lat], [cos(a) for a in lat], sin, cos


def load_statics(axyp_j):
    """Everything RIVERF needs that is static, from the real input files.  axyp_j: DXYP(j) (JM,) (e.g. ocean_step._DXYPO).
    Returns dict of (IM, JM) arrays (focean, flake0, zatmo (geopotential), hlake) and the river tables (iflow, jflow, kdirec, ifl911, jfl911,
    kd911, dhorz, rate; Fortran 1-based index values stored in the int arrays)."""
    import netCDF4 as nc
    T = nc.Dataset(TOPO_FILE)
    foc = np.array(T.variables['focean'][:], dtype=np.float64).T.copy()
    flake0 = np.array(T.variables['flake'][:], dtype=np.float64).T.copy()
    zatmo = np.array(T.variables['zatmo'][:], dtype=np.float64).T.copy() * GRAV
    hlake = np.maximum(np.array(T.variables['hlake'][:], dtype=np.float64).T.copy(), HLAKE_MIN)
    fgrnd = np.array(T.variables['fgrnd'][:], dtype=np.float64).T.copy()
    fgice = np.array(T.variables['fgice'][:], dtype=np.float64).T.copy()
    R = nc.Dataset(RVR_FILE)
    dla = np.array(R.variables['down_lat'][:], dtype=np.float64).T
    dlo = np.array(R.variables['down_lon'][:], dtype=np.float64).T
    dla9 = np.array(R.variables['down_lat_911'][:], dtype=np.float64).T
    dlo9 = np.array(R.variables['down_lon_911'][:], dtype=np.float64).T
    lon, sinl, cosl, sin, cos = _geom_trig()
    acos = _fn('acos')

    def horz(i1, j1, i2, j2):
        if i1 == i2 and j1 == j2:
            return math.sqrt(float(axyp_j[j1 - 1]))
        x1 = cosl[j1 - 1] * cos(lon[i1 - 1]); y1 = cosl[j1 - 1] * sin(lon[i1 - 1]); z1 = sinl[j1 - 1]
        x2 = cosl[j2 - 1] * cos(lon[i2 - 1]); y2 = cosl[j2 - 1] * sin(lon[i2 - 1]); z2 = sinl[j2 - 1]
        return RADIUS * acos(x1 * x2 + y1 * y2 + z1 * z2)

    iflow = np.full((IM, JM), -99, int); jflow = iflow.copy(); kdirec = np.zeros((IM, JM), int)
    ifl9 = iflow.copy(); jfl9 = iflow.copy(); kd9 = kdirec.copy()
    dhorz = np.zeros((IM, JM))
    warn = []
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            a, b = i - 1, j - 1
            if dlo[a, b] > -1000.0:
                ii, jj = lonlat_to_ij(float(dlo[a, b]), float(dla[a, b]))
                iflow[a, b], jflow[a, b] = ii, jj
                if 1 <= ii <= IM and 1 <= jj <= JM:
                    dhorz[a, b] = horz(i, j, ii, jj)
            elif dlo[a, b] == -9999:
                kdirec[a, b] = 9
                # DHORZ is not set by the real code in this branch (stays 0 here; unused for KDIREC=9)
            else:
                # land box without direction: warning in the real code, IFLOW stays -99; otherwise local flow
                fl0 = fgrnd[a, b] + fgice[a, b] + flake0[a, b]
                if (fl0 > 0) and foc[a, b] <= 0:
                    warn.append((i, j))
                else:
                    iflow[a, b], jflow[a, b] = i, j
                dhorz[a, b] = horz(i, j, i, j)
            if dlo9[a, b] > -1000.0:
                ii, jj = lonlat_to_ij(float(dlo9[a, b]), float(dla9[a, b]))
                ifl9[a, b], jfl9[a, b] = ii, jj
            if iflow[a, b] > -99:
                kdirec[a, b] = get_dir(i, j, iflow[a, b], jflow[a, b])
            if ifl9[a, b] > -99:
                kd9[a, b] = get_dir(i, j, ifl9[a, b], jfl9[a, b])
    st = dict(focean=foc, flake0=flake0, zatmo=zatmo, hlake=hlake, iflow=iflow, jflow=jflow, kdirec=kdirec, ifl911=ifl9, jfl911=jfl9, kd911=kd9,
              dhorz=dhorz, no_direction_land=warn, axyp_j=np.asarray(axyp_j, float))
    st['rate'] = compute_rate(st)
    return st


def compute_rate(st, dzdh1=None):
    """LAKES.f:1027 RATE (per source time step); loop JU=1..JM-1 (J_1S), IU=1..IMAXJ(JU)."""
    dzdh1 = DZDH1 if dzdh1 is None else dzdh1
    rate = np.zeros((IM, JM))
    speed0, spmin, spmax = 0.35, 0.15, 5.0
    for ju in range(1, JM):
        for iu in range(1, (1 if ju == 1 else IM) + 1):
            a, b = iu - 1, ju - 1
            if st['focean'][a, b] < 1 and st['kdirec'][a, b] <= 8:
                if st['kdirec'][a, b] >= 1:
                    idd, jdd = st['iflow'][a, b] - 1, st['jflow'][a, b] - 1
                    dzdh = (st['zatmo'][a, b] - st['zatmo'][idd, jdd]) / (GRAV * st['dhorz'][a, b])
                else:
                    dzdh = st['zatmo'][a, b] / (GRAV * st['dhorz'][a, b])
                speed = speed0 * dzdh / dzdh1
                if speed < spmin:
                    speed = spmin
                if speed > spmax:
                    speed = spmax
                rate[a, b] = DTSRC * speed / st['dhorz'][a, b]
    return rate


def riverf(lake, flake, fland, fearth, st, rate=None, verbose=False):
    """LAKES.f RIVERF (original version).  lake: dict mwl, gml, tlake, mldlk (IM, JM); flake/fland/fearth (IM, JM); st: load_statics().
    Returns (new lake dict (mwl, gml, tlake, mldlk, dlake, glake), flowo, eflowo (kg/m2 and J/m2 of ocean area per step, IM x JM),
    gtemp, gtempr, mlhc (lake cells, else 0), diag dict (n_emergency, n_backwash, n_clip, rate_used))."""
    rate = st['rate'] if rate is None else rate
    mwl = lake['mwl'].astype(float).copy(); gml = lake['gml'].astype(float).copy()
    tlake = lake['tlake'].astype(float).copy(); mldlk = lake['mldlk'].astype(float).copy()
    focean = st['focean']; hlake = st['hlake']; zatmo = st['zatmo']; kdirec = st['kdirec']
    iflow, jflow = st['iflow'], st['jflow']
    axyp = np.repeat(st['axyp_j'][None, :], IM, axis=0)
    flow = np.zeros((IM, JM)); eflow = np.zeros((IM, JM)); flowo = np.zeros((IM, JM)); eflowo = np.zeros((IM, JM))
    diag = dict(n_emergency=0, n_backwash=0, n_clip95=0, n_kd9=0)
    I1 = lambda ju: 1 if ju == 1 or ju == JM else IM   # IMAXJ   # noqa: E731

    for ju in range(1, JM + 1):
        for iu in range(1, I1(ju) + 1):
            a, b = iu - 1, ju - 1
            if kdirec[a, b] == 9:
                _kd9(a, b, iu, ju, mwl, gml, tlake, flake, kdirec, hlake, axyp, flow, eflow, diag)
                continue
            if (kdirec[a, b] == 0 and flake[a, b] > .949 * (flake[a, b] + fearth[a, b])
                    and mwl[a, b] > (hlake[a, b] + LAKE_RISE_MAX) * flake[a, b] * RHOW * axyp[a, b] and st['kd911'][a, b] > 0):
                kd = st['kd911'][a, b]; jd = st['jfl911'][a, b]; id_ = st['ifl911'][a, b]
                mwlsill = RHOW * (hlake[a, b] + LAKE_RISE_MAX) * flake[a, b] * axyp[a, b]
                diag['n_emergency'] += 1
            else:
                kd = kdirec[a, b]; jd = jflow[a, b]; id_ = iflow[a, b]
                mwlsill = RHOW * hlake[a, b] * flake[a, b] * axyp[a, b]
            if kd == 0 and fland[a, b] * focean[a, b] == 0:
                continue
            if jd > JM or jd < 1 or id_ > IM or id_ < 1:
                continue
            ad, bd = id_ - 1, jd - 1
            do_back = False
            if not ((1 <= kdirec[ad, bd] <= 8) or flake[ad, bd] <= .949 * (flake[ad, bd] + fearth[ad, bd])):
                mwlsilld = RHOW * axyp[ad, bd] * flake[ad, bd] * (hlake[ad, bd] + BYGRAV * max(zatmo[a, b] - zatmo[ad, bd], 0.0))
                if mwl[ad, bd] > mwlsilld:
                    if flake[a, b] > 0:
                        dmm = URATE * DTSRC * (flake[ad, bd] * axyp[ad, bd] * (mwl[a, b] - mwlsill)
                                               - flake[a, b] * axyp[a, b] * (mwl[ad, bd] - mwlsilld)) / (flake[a, b] * axyp[a, b] + flake[ad, bd] * axyp[ad, bd])
                        do_back = dmm < 0
                    else:
                        dmm = -(mwl[ad, bd] - mwlsilld) * URATE * DTSRC
                        do_back = True
            if do_back:
                dgm = tlake[ad, bd] * dmm * SHW
                diag['n_backwash'] += 1
            else:
                if mwl[a, b] <= mwlsill:
                    continue
                dmm = (mwl[a, b] - mwlsill) * rate[a, b]
                if mwl[a, b] - dmm < 1e-6:
                    dmm = mwl[a, b]
                dmm = min(dmm, .5 * RHOW * axyp[a, b])
                if flake[a, b] > 0:
                    mlm = RHOW * mldlk[a, b] * flake[a, b] * axyp[a, b]
                    if dmm > .95 * mlm:
                        diag['n_clip95'] += 1
                    dmm = min(dmm, .95 * mlm)
                dgm = tlake[a, b] * dmm * SHW
            flow[a, b] = flow[a, b] - dmm
            eflow[a, b] = eflow[a, b] - dgm
            flfac = 1.0
            if ju == 1 or ju == JM:
                flfac = float(IM)
            if jd == 1 or jd == JM:
                flfac = 1.0 / IM
            if focean[ad, bd] == 0:
                flow[ad, bd] = flow[ad, bd] + dmm * flfac
                eflow[ad, bd] = eflow[ad, bd] + (dgm + 0.0) * flfac
            else:
                dmm = RIVER_FAC * dmm
                flowo[ad, bd] = flowo[ad, bd] + dmm * flfac
                eflowo[ad, bd] = eflowo[ad, bd] + (dgm + 0.0) * flfac

    # apply net river flow to the continental reservoirs
    for j in range(1, JM + 1):
        for i in range(1, I1(j) + 1):
            a, b = i - 1, j - 1
            if fland[a, b] + flake[a, b] > 0.0:
                mwl[a, b] = mwl[a, b] + flow[a, b]
                gml[a, b] = gml[a, b] + eflow[a, b]
                if mwl[a, b] < 1e-6:
                    mwl[a, b] = 0.0; gml[a, b] = 0.0
                if flake[a, b] > 0:
                    hlk1 = (mldlk[a, b] * RHOW) * tlake[a, b] * SHW
                    mldlk[a, b] = mldlk[a, b] + flow[a, b] / (RHOW * flake[a, b] * axyp[a, b])
                    tlake[a, b] = (hlk1 * flake[a, b] * axyp[a, b] + eflow[a, b]) / (mldlk[a, b] * RHOW * flake[a, b] * axyp[a, b] * SHW)
                else:
                    tlake[a, b] = gml[a, b] / (SHW * mwl[a, b] + TEENY)
    gtemp = np.zeros((IM, JM)); gtempr = np.zeros((IM, JM)); mlhc = np.zeros((IM, JM))
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            a, b = i - 1, j - 1
            if flake[a, b] > 0:
                gtemp[a, b] = tlake[a, b]; gtempr[a, b] = tlake[a, b] + TF; mlhc[a, b] = SHW * mldlk[a, b] * RHOW
    dlake = np.zeros((IM, JM)); glake = np.zeros((IM, JM))
    for j in range(1, JM + 1):
        for i in range(1, I1(j) + 1):
            a, b = i - 1, j - 1
            if flake[a, b] > 0.0:
                dlake[a, b] = mwl[a, b] / (RHOW * flake[a, b] * axyp[a, b]); glake[a, b] = gml[a, b] / (flake[a, b] * axyp[a, b])
            if focean[a, b] > 0.0:
                byoarea = 1.0 / (axyp[a, b] * focean[a, b])
                flowo[a, b] = flowo[a, b] * byoarea; eflowo[a, b] = eflowo[a, b] * byoarea
    new = dict(lake, mwl=mwl, gml=gml, tlake=tlake, mldlk=mldlk, dlake=dlake, glake=glake)
    return new, flowo, eflowo, gtemp, gtempr, mlhc, diag


def _kd9(a, b, iu, ju, mwl, gml, tlake, flake, kdirec, hlake, axyp, flow, eflow, diag):
    """label 400 of RIVERF: KDIREC = 9 internal seas, adjacent KD=2 (north, ID=IU, JD=JU+1) and KD=8 (east, ID=IU+1, JD=JU)."""
    for kd in (2, 8):
        if kd == 2:
            idd, jdd = iu, ju + 1
        else:
            idd, jdd = iu + 1, ju
        if idd > IM or jdd > JM:
            if kdirec[a, b] == 9:
                diag.setdefault('kd9_out_of_array', []).append((iu, ju, kd))
            continue
        ad, bd = idd - 1, jdd - 1
        if kdirec[ad, bd] != 9:
            continue
        if flake[a, b] + flake[ad, bd] == 0:
            continue
        diag['n_kd9'] += 1
        flakeu = max(flake[a, b], .01); flaked = max(flake[ad, bd], .01)
        mwlsill = RHOW * hlake[a, b] * flakeu * axyp[a, b]
        mwlsilld = RHOW * hlake[ad, bd] * flaked * axyp[ad, bd]
        dmm = URATE * DTSRC * (flaked * axyp[ad, bd] * (mwl[a, b] - mwlsill) - flakeu * axyp[a, b] * (mwl[ad, bd] - mwlsilld)) \
            / (flakeu * axyp[a, b] + flaked * axyp[ad, bd])
        if dmm > 0:
            if mwl[a, b] <= 1 * RHOW * flake[a, b] * axyp[a, b]:
                continue
            if dmm > mwl[a, b] - 1 * RHOW * flake[a, b] * axyp[a, b]:
                dmm = mwl[a, b] - 1 * RHOW * flake[a, b] * axyp[a, b]
            dgm = tlake[a, b] * dmm * SHW
        else:
            if mwl[ad, bd] <= 1 * RHOW * flake[ad, bd] * axyp[ad, bd]:
                continue
            if dmm < 1 * RHOW * flake[ad, bd] * axyp[ad, bd] - mwl[ad, bd]:
                dmm = 1 * RHOW * flake[ad, bd] * axyp[ad, bd] - mwl[ad, bd]
            dgm = tlake[ad, bd] * dmm * SHW
        flow[a, b] = flow[a, b] - dmm; flow[ad, bd] = flow[ad, bd] + dmm
        eflow[a, b] = eflow[a, b] - dgm; eflow[ad, bd] = eflow[ad, bd] + dgm
