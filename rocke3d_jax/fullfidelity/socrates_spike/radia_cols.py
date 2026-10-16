"""R6 stage A: the ModelE side of one radiation call for this rundeck (P2SAoM40, USE_PLANET_RAD + GISS_RAD_OFF): what RADIA (RAD_DRV.f) and
RCOMPX (RADIATION.f) do between the model state and run_planet_rad, i.e. the arguments / module inputs recorded in dump tags 50 and 60.

  column_state      PLB/TLM/SHL/RHL of the 43 radiation layers (RADIA 'SET UP VERTICAL ARRAYS'), p_level / t_layer for planet_rad
  getsur            GETSUR (ALBEDO.f): surface albedos PRNB/PRNX of the four surface types (with RXSNOW / GTSALB / SGPGXG of RAD_UTILS.f)
  getgas_ulgas      getgas (GHGMOD.f) + FPXSCALEGAS + seth2o (RADIATION.f): column amounts ULGAS of the 13 gases
  aerosol_inputs    SETAER / GETDST / SETVOL (RADIATION.f): the planet_rad aerosol arrays; KDELIQ hysteresis (KLIQ)
  cloudeps          GETEPS (KCLDEP = 4)

Daily / static products are INPUTS here (they depend on the date and the grid box only, not on the model state): the ozone column on the
radiation layering (o3col), the aerosol / dust / volcanic optical-depth columns, the cloud-heterogeneity fields, the surface vegetation
fractions, the GHG scaling.  Producing them from the climatology files for an arbitrary date is NOT ported (report section 12).
All arrays: leading axis N = columns; layer index 0 = bottom (GISS order) unless the name says 'top-first'.  float64, jax.numpy."""
import numpy as np
import jax
import jax.numpy as jnp
import planet_side as P

LM, LM_REQ, NL = 40, 3, 43
LHE = 2.5e6
MWAT, MAIR = 18.015, 28.9655
RVAP = 1.0e3 * P.GASC / MWAT
MRAT = MWAT / MAIR
TF = 273.15
RHOW = 1.0e3
PPMV_TO_CM_AT_STP = 1.0e-05 * P.AVOG / (P.GRAV * P.MAIR * P.LOSCHMIDT)
H2O_MMR_TO_CM_AT_STP = PPMV_TO_CM_AT_STP * 1.0e+06 * P.MAIR / 18.0153
_f32 = lambda a: np.array(a, dtype=np.float32).astype(np.float64)    # REAL*8 data statements written with default-real literals
CMANO2 = _f32([
    8.66E-06, 5.15E-06, 2.85E-06, 1.50E-06, 9.89E-07, 6.91E-07, 7.17E-07, 8.96E-07, 3.67E-06, 4.85E-06, 5.82E-06, 6.72E-06, 7.77E-06,
    8.63E-06, 8.77E-06, 8.14E-06, 6.91E-06, 5.45E-06, 4.00E-06, 2.67E-06, 1.60E-06, 8.36E-07, 3.81E-07, 1.58E-07, 6.35E-08, 2.57E-08,
    1.03E-08, 4.18E-09, 1.66E-09, 6.57E-10, 2.58E-10, 1.02E-10, 4.11E-11, 1.71E-11, 7.73E-12, 9.07E-12, 4.63E-12, 2.66E-12, 1.73E-12,
    1.28E-12, 1.02E-12, 1.00E-30])
# GHG module constants (GHGMOD.f)
ZH = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 30., 50., 30., 30., 0.0, 30., 30.])            # K = 1..12
PPGRAD = _f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0100, 0.0900, 0.0600, 0.0600, 0.0, 0.0600, 0.0600])
KLATZ0, KPGRAD = 1, 1
RHC = np.array([.38, .47, .28, .38])
RHD = np.array([.80, .75, .62, .80])


def imaxj():
    return np.array([1] + [72] * 44 + [1])


def column_index():
    """(i, j) 1-based of the columns in the order of the RADIA loops (J outer, I inner, I <= IMAXJ(J)): the order of the dumps."""
    ii, jj = [], []
    for j, m in enumerate(imaxj(), start=1):
        for i in range(1, m + 1):
            ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)


def qsat(tm, lh, pr):
    a = 6.108 * MRAT
    b = 1.0 / (RVAP * TF)
    c = 1.0 / RVAP
    return a * jnp.exp(lh * (b - c / jnp.maximum(130.0, tm))) / pr


# -------------------------------------------------------------------------------------------------------------------- vertical set-up
def column_state(T, Q, PK, PMID, PEDN, RQT, plb0_req, shl0_req, rhfix=-1.0):
    """T, Q, PK, PMID: (N, 40); PEDN (N, 41); RQT (N, 3); plb0_req/shl0_req (3,) -> dict PLB (N, 44), TLM, SHL, RHL (N, 43) (bottom first)."""
    N = T.shape[0]
    q = jnp.where(Q < 0.0, 0.0, Q)
    tlm = jnp.concatenate([T * PK, RQT], axis=1)
    shl = jnp.concatenate([q, jnp.broadcast_to(jnp.asarray(shl0_req), (N, LM_REQ))], axis=1)
    plb = jnp.concatenate([PEDN, jnp.broadcast_to(jnp.asarray(plb0_req), (N, LM_REQ))], axis=1)
    rh_m = shl[:, :LM] / qsat(tlm[:, :LM], LHE, PMID)
    pmid_r = 0.5 * (plb[:, LM:LM + LM_REQ] + plb[:, LM + 1:LM + LM_REQ + 1])
    rh_r = shl[:, LM:] / qsat(tlm[:, LM:], LHE, pmid_r)
    rhl = jnp.concatenate([rh_m, rh_r], axis=1)
    if rhfix >= 0.0:
        rhl = jnp.where(jnp.arange(NL)[None, :] < LM, rhfix, rhl)
    return dict(plb=plb, tlm=tlm, shl=shl, rhl=rhl, q=q)


# -------------------------------------------------------------------------------------------------------------------- snow albedo
def _compute(f1, f2, f3, f4, var):
    """RAD_UTILS.f 'compute' (cusp weights 0.5 / 0.5)."""
    f21 = f2 - f1
    f32 = f3 - f2
    f43 = f4 - f3
    f3221 = (f32 + f21) * 0.5
    f4332 = (f43 + f32) * 0.5
    a = f2
    b = f3221
    c = 3.0 * f32 - f3221 - f3221 - f4332
    d = (f3221 + f4332 - f32 - f32)
    xf = var
    ffcusp = a + xf * (b + xf * (c + xf * d))
    xexm = (1 - 2 * xf) ** 2
    cuspwt = (1.0 - xexm) * 0.5 + xexm * 0.5
    fflinr = a + xf * f32
    return ffcusp * cuspwt + fflinr * (1.0 - cuspwt)


_SPLIT = 134217729.0                                                   # 2**27 + 1 (Veltkamp)


def _two_prod(a, b):
    """(p, e): p = fl(a*b) and the Dekker error term; e is the guard that gives p a second use (no FMA contraction of p + q)."""
    p = a * b
    t = _SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _trunc(x):
    return jnp.trunc(x).astype(jnp.int32)


def gtsalb_tau(rbb_in, eg_in, tab):
    """GTSALB with KGTAUR = 2, GIN = TAUIN = 0 (the only call of RXSNOW): optical depth TAUOUT for the diffuse albedo rbb_in and the
    asymmetry eg_in.  tab: dict(SALBTG (768,14), TAUGSA (1001,14), TAUTGS (768,))."""
    SALBTG = jnp.asarray(tab["salbtg"]); TAUGSA = jnp.asarray(tab["taugsa"]); TAUTGS = jnp.asarray(tab["tautgs"])
    # --- label 300: albedo -> tau (first pass)
    rbbb = rbb_in
    ri = rbbb * 1000.0
    ir = _trunc(ri)
    wrj = ri - ir
    ir = ir + 1
    jr = ir + 1
    ei = eg_in * 20.0
    big = ei > 10.0
    ie = jnp.where(big, _trunc(ei) - 6, 2)
    wej = ei - _trunc(ei)
    iterpl = jnp.where(big, 4, 1)
    cap = big & (ie > 12)
    iterpl = jnp.where(cap, 2, iterpl)
    ie = jnp.where(cap, 12, ie)
    iem = ie - 1

    def tg(i, kg):                                                      # TAUGSA(i, kg), 1-based
        return TAUGSA[i - 1, kg - 1]

    ffk = []                                                            # FFKG(k, 1..3), k = 0..3
    for k in range(4):
        kg = iem + k
        f1 = tg(ir - 1, kg); f2 = tg(ir, kg); f3 = tg(jr, kg); f4 = tg(jr + 1, kg)
        f1 = jnp.where(ir == 1, -f3, f1)
        ffk.append((_compute(f1, f2, f3, f4, wrj), f2, f3))
    # ITERPL = 4: cubic in e; ITERPL < 4: parabola (G=0 in the expression for ITERPL == 2 as in the Fortran)
    cub = [_compute(ffk[0][c], ffk[1][c], ffk[2][c], ffk[3][c], wej) for c in range(3)]
    xg = eg_in * 2.0 - 0.5
    xg = jnp.where(iterpl == 2, 0.0 * eg_in + 0.0 * 10.0 - 9.0, xg)
    xm = 1.0 - xg - xg
    xp = 1.0 + xg + xg
    # FFKG(ITERPL+1) and FFKG(ITERPL) are 1-based k indices -> 0-based ITERPL and ITERPL-1
    def pick(c, k1):                                                    # FFKG(k1, c), k1 1-based, per-element k1
        arr = jnp.stack([ffk[k][c] for k in range(4)], axis=0)
        return jnp.take_along_axis(arr, (jnp.clip(k1, 1, 4) - 1)[None, ...], axis=0)[0]
    pa = [xm * xp * pick(c, iterpl + 1) - xg * xm * pick(c, iterpl) + xg * xp * ffk[3][c] for c in range(3)]
    tbb = jnp.where(iterpl == 4, cub[0], pa[0])
    # --- label 390 -> 200: tau -> albedo (second pass) with tau = tbb, g = eg_in
    tau = tbb
    g = eg_in
    deltau = jnp.full_like(tau, 0.05)
    ti = tau / deltau
    it = _trunc(ti)
    in1 = it <= 599
    wtj = ti - it
    it1 = it + 1
    d2 = 0.50
    ti2 = tau / d2
    it2 = _trunc(ti2)
    in2 = it2 <= 199
    d3 = 50.0
    ti3 = tau / d3
    it3 = _trunc(ti3)
    in3 = it3 <= 19
    d4 = 1000.0
    ti4 = tau / d4
    it4 = _trunc(ti4)
    deltau = jnp.where(in1, 0.05, jnp.where(in2, d2, jnp.where(in3, d3, d4)))
    ti_s = jnp.where(in1, ti, jnp.where(in2, ti2, jnp.where(in3, ti3, ti4)))
    it_s = jnp.where(in1, it1, jnp.where(in2, it2 + 541, jnp.where(in3, it3 + 649, it4 + 758)))
    wtj = ti_s - jnp.where(in1, it, jnp.where(in2, it2, jnp.where(in3, it3, it4)))
    wti = 1.0 - wtj
    it = it_s
    jt = it + 1
    gi = g * 20.0
    bigg = gi > 10.0
    ig = jnp.where(bigg, _trunc(gi) - 6, 2)
    wgj = gi - _trunc(gi)
    iterpl = jnp.where(bigg, 4, 1)
    capg = bigg & (ig > 12)
    iterpl = jnp.where(capg, 2, iterpl)
    ig = jnp.where(capg, 12, ig)
    igm = ig - 1

    def sa(i, kg):
        return SALBTG[i - 1, kg - 1]

    ffs = []
    for k in range(4):
        kg = igm + k
        f1 = sa(it - 1, kg); f2 = sa(it, kg); f3 = sa(jt, kg); f4 = sa(jt + 1, kg)
        f1 = jnp.where(it == 1, -f3, f1)
        ffs.append((_compute(f1, f2, f3, f4, wtj), f2, f3))
    cub2 = [_compute(ffs[0][c], ffs[1][c], ffs[2][c], ffs[3][c], wgj) for c in range(3)]
    xg2 = g * 2.0 - 0.5
    xg2 = jnp.where(iterpl == 2, g * 10.0 - 9.0, xg2)
    xm2 = 1.0 - xg2 - xg2
    xp2 = 1.0 + xg2 + xg2

    def pick2(c, k1):
        arr = jnp.stack([ffs[k][c] for k in range(4)], axis=0)
        return jnp.take_along_axis(arr, (jnp.clip(k1, 1, 4) - 1)[None, ...], axis=0)[0]

    rb = [jnp.where(iterpl == 4, cub2[c], xm2 * xp2 * pick2(c, iterpl + 1) - xg2 * xm2 * pick2(c, iterpl) + xg2 * xp2 * ffs[3][c])
          for c in range(3)]
    rbb, rb2, rb3 = rb
    # --- label 400
    small = jnp.abs(wti * wtj) < 0.1
    dtau_s = (rbbb - rb2) / (rb3 - rb2)
    c_ = (rb3 - rbb) / wti - (rbb - rb2) / wtj
    b_ = (rbb - rb2) / wtj - wtj * c_
    a_ = rb2
    bb = b_ * b_ + 4.0 * c_ * (rbbb - a_)
    dtau_l = (jnp.sqrt(jnp.maximum(bb, 0.0)) - b_) / (c_ + c_)
    dtau = jnp.where(small, dtau_s, dtau_l)
    return (it - 1 + dtau) * deltau


def sgpgxg(xmu, tau, g, GTAU):
    """RAD_UTILS.f SGPGXG: GTAU (51, 11, 143)."""
    G = jnp.asarray(GTAU)
    xi = xmu * 50.0 + 0.999999
    ix = _trunc(xi)
    jx = ix + 1
    wxj = xi - ix
    wxi = 1.0 - wxj
    gi = g * 10.0
    ig = _trunc(gi)
    wgj = gi - ig
    wgi = 1.0 - wgj
    ig = ig + 1
    jg = ig + 1
    c1 = tau < 6.0
    c2 = tau < 20.0
    c3 = tau < 100.0
    c4 = tau < 1000.0
    ti = jnp.where(c1, tau * 10.0 + 1.0, jnp.where(c2, (tau - 6.0) * 2.0 + 2.0, jnp.where(c3, (tau - 20.0) * 0.20 + 2.0,
                   jnp.where(c4, (tau - 100.0) * 0.02 + 2.0, tau * 0.001 + 1.0e-6))))
    it = _trunc(ti)
    wtj = ti - it
    it = jnp.where(c4, it, jnp.minimum(it, 9))
    it0 = jnp.where(c1, 0, jnp.where(c2, 62, jnp.where(c3, 93, jnp.where(c4, 112, 133))))
    wti = 1.0 - wtj
    it = it + it0
    jt = it + 1
    g_ = lambda a, b, c: G[a - 1, b - 1, c - 1]
    return (wgi * (wti * (wxi * g_(ix, ig, it) + wxj * g_(jx, ig, it)) + wtj * (wxi * g_(ix, ig, jt) + wxj * g_(jx, ig, jt)))
            + wgj * (wti * (wxi * g_(ix, jg, it) + wxj * g_(jx, jg, it)) + wtj * (wxi * g_(ix, jg, jt) + wxj * g_(jx, jg, jt))))


def rxsnow(rbsno, xcosz, ggsno, tab):
    """RXSNOW: direct albedo of snow / ice from the diffuse albedo (zenith-angle dependence); element-wise."""
    thin = rbsno < 0.05
    ggsn = jnp.minimum(ggsno, 0.9)
    rbsn = rbsno
    frtop = jnp.ones_like(rbsno)
    hi = rbsno > 0.5
    rbsn = jnp.where(hi, 0.5, rbsn)
    frtop = jnp.where(hi, ((1.0 - rbsno) / 0.5) ** 2, frtop)
    rbs = jnp.where(thin, 0.5, rbsn)                                    # keep the table lookups in range for the masked elements
    tausn = gtsalb_tau(rbs, ggsn, tab)
    gpff = sgpgxg(xcosz, tausn, ggsn, tab["gtau"])
    pr = 1.0 - gpff
    pt = 1.0 + gpff
    dbls = 10.0 + 1.44269 * jnp.log(tausn)
    ndbls = _trunc(dbls)
    tau = tausn / 2.0 ** ndbls.astype(jnp.float64)
    secz = 1.0 / xcosz
    xanb = jnp.exp(-tau - tau)
    xanx = jnp.exp(-tau * secz)
    tanb = pt * xanb
    xxt = (secz - 2.0) * tau
    tanx = pt * secz * (.5 + xxt * (.25 + xxt * (.0833333 + xxt * (.0208333 + xxt)))) * xanx
    rasb = pr * (1.0 - tau * (2.0 - 2.66667 * tau * (1.0 - tau)))
    xxt = (secz + 2.0) * tau
    rasx = pr * secz * (.5 - xxt * (.25 - xxt * (.0833333 - xxt * (.0208333 - xxt))))
    bnorm = (1.0 - xanb) / (rasb + tanb)
    xnorm = (1.0 - xanx) / (rasx + tanx)
    rasb = rasb * bnorm
    rasx = rasx * xnorm
    tanb = tanb * bnorm
    tanx = tanx * xnorm
    for nn in range(1, 31):                                            # NDBLS <= 10 + 1.44269 ln(tau) < 30 for tau < 1e8
        act = nn <= ndbls
        rarb = rasb * rasb
        rarx = xanx * rasx
        xatb = xanb + tanb
        denom = 1.0 - rarb
        db = (tanb + xanb * rarb) / denom
        dx = (tanx + rarx * rasb) / denom
        ub = rasb * (xanb + db)
        ux = rarx + rasb * dx
        rasb_n = rasb + xatb * ub
        rasx_n = rasx + xatb * ux
        tanb_n = xanb * tanb + xatb * db
        tanx_n = xanx * tanx + xatb * dx
        xanb_n = xanb * xanb
        xanx_n = xanx * xanx
        rasb = jnp.where(act, rasb_n, rasb); rasx = jnp.where(act, rasx_n, rasx)
        tanb = jnp.where(act, tanb_n, tanb); tanx = jnp.where(act, tanx_n, tanx)
        xanb = jnp.where(act, xanb_n, xanb); xanx = jnp.where(act, xanx_n, xanx)
    drbrat = rasx / rbsn - 1.0
    rx = rbsno * (1.0 + drbrat * frtop)
    return jnp.where(thin, rbsno, rx)


# -------------------------------------------------------------------------------------------------------------------- GETSUR
ASNALB = np.array([.60, .55, .55, .30, .10, .05, .35])
AOIALB = np.array([.55, .50, .45, .25, .10, .05, .30])
ALIALB = np.array([.60, .55, .50, .30, .10, .05, .35])
AGEXPF = np.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]])                   # (3, 2) (surface, hemisphere)
ALBDIF = np.array([[0.35, 0.35], [0.35, 0.35], [0.35, 0.35]])
AOIMIN = np.array([.05, .05, .05, .050, .05, .03])
AOIMAX = np.array([.62, .42, .30, .120, .05, .03])
ASNWET = np.array([.85, .75, .50, .175, .03, .01])
ASNDRY = np.array([.90, .85, .65, .450, .10, .10])
AMPMIN = np.array([.10, .05, .05, .050, .05, .03])
GZSNOW = np.array([.95, .94, .905, .896, .894, .89, .91])                 # same for all surface types / hemispheres
VTMASK = np.array([.1, .2, .2, .5, 2., 5., 10., 25., .2, .1, 1e-5, .2])
AVSCAT, ANSCAT, AVFOAM, ANFOAM = .0156, 0.0, .2197, .1514
ASHZOI = ANHZOI = 0.1
DMOICE = DMLICE = 10.0
WETSRA = 1.0
ZOCSRA = 1.0
SRFOAM = _f32([0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.014, 0.019, 0.025, 0.032, 0.041, 0.051, 0.063,
                   0.077, 0.094, 0.112, 0.138, 0.164, 0.191, 0.218, 0.246])


def _bvh2o(w):
    return .0488 + .0974 / (5.679 + w) + .0004 / (.3333 + w)


def _xvh2o(w, x):
    return .021 + x * x * (.0421 + x * (.1283 + x * (-.04 + x * (3.117 / (5.679 + w) + x * .025 / (.3333 + w)))))


def getsur(cosz, jlat, ilon, ageSN, pocean, poice, pearth, plice, plake, zlake, tgo, zoice, fmp, zsnwoi, zmp, snowoi, snowd, snowli,
           snow_frac, wearth, wmag, pvt, dalbsn, flags, loc_chl, albvnh, tab, ksialb=0, kzsnow=1, snoage_fac_max=0.5, mlat46=46, jnorth=23):
    """GETSUR for N columns -> PRNB, PRNX (N, 6, 4) (Fortran PRNB(J, type), J = 1..6 the SW bands high-to-low).
    albvnh: (12, 6, 2) vegetation albedo table of the day (SURF_ALBEDO ALBVNH)."""
    N = cosz.shape[0]
    ALB = jnp.asarray(albvnh)
    ALBDIF_, AGEXPF_ = jnp.asarray(ALBDIF), jnp.asarray(AGEXPF)
    jh = jnp.where(jlat > jnorth, 1, 0)                                 # 0-based hemisphere index
    kkz = jnp.where(cosz < 0.001, 0, kzsnow)
    # ---- ocean
    x = 0.5 + (0.5 - cosz) * ZOCSRA
    bvis = _bvh2o(wmag) + AVSCAT
    xvis = _xvh2o(wmag, x) + AVSCAT
    bnir = _bvh2o(wmag) + ANSCAT
    xnir = _xvh2o(wmag, x) + ANSCAT
    iwm = jnp.clip(_trunc(wmag), 1, 24)
    jwm = iwm + 1
    wmj = wmag - iwm
    wmi = 1.0 - wmj
    sr = jnp.asarray(SRFOAM)
    frfoam = wmi * sr[iwm - 1] + wmj * sr[jwm - 1]
    bvis = bvis * (1.0 - frfoam) + frfoam * AVFOAM
    xvis = xvis * (1.0 - frfoam) + frfoam * AVFOAM
    bnir = bnir * (1.0 - frfoam) + frfoam * ANFOAM
    xnir = xnir * (1.0 - frfoam) + frfoam * ANFOAM
    boc = jnp.stack([bvis] + [bnir] * 5, axis=1)                        # (N, 6)
    xoc = jnp.stack([xvis] + [xnir] * 5, axis=1)
    shallow = (plake > 0) & (zlake < 1)
    alb1 = 0.5 * ALB[0, :, 0][None, :]
    boc = jnp.where(shallow[:, None], (boc * (zlake[:, None] - .4) + (1 - zlake[:, None]) * alb1) / .6, boc)
    xoc = jnp.where(shallow[:, None], (xoc * (zlake[:, None] - .4) + (1 - zlake[:, None]) * alb1) / .6, xoc)
    oc_on = (pocean >= 1.0e-04)[:, None]
    boc = jnp.where(oc_on, boc, 0.0); xoc = jnp.where(oc_on, xoc, 0.0)
    # ---- land (soil + vegetation + snow)
    L = jnp.arange(1, 7)
    fsn = jnp.where(L > 2, 2.0 / L, 1.0)[None, :]
    asnage = ALBDIF_[0, jh] * jnp.exp(-AGEXPF_[0, jh] * ageSN[:, 0])
    bsn = jnp.asarray(ASNALB[:6])[None, :] + asnage[:, None] * fsn
    gz = jnp.asarray(GZSNOW[:6])[None, :]
    cz = jnp.where(kkz > 0, cosz, 0.5)
    xsn_r = rxsnow(bsn, cz[:, None], gz, tab)
    xsn = jnp.where((kkz > 0)[:, None], xsn_r, bsn)
    dsfrac = pvt[:, 0] + pvt[:, 9]
    nosnow = (snowd[:, 0] + snowd[:, 1]) <= 1.0e-04 * 10.0
    # table index [type, band, hemisphere]; ALBVNH(K, L, JH)
    alb_jh = jnp.stack([ALB[:, :, 0], ALB[:, :, 1]], axis=0)[jh]        # (N, 12, 6)
    # no-snow branch
    be0 = pvt[:, 0][:, None] * alb_jh[:, 0, :] * (1.0 - 0.5 * wearth[:, None] * WETSRA)
    for k in range(1, 12):
        if k in (9, 10):
            continue
        be0 = be0 + pvt[:, k][:, None] * alb_jh[:, k, :]
    # snow branch
    vt = pvt[:, 0] * (1.0 - snow_frac[:, 0] * (1.0 - jnp.exp(-snowd[:, 0] / VTMASK[0])))
    expsne = vt + pvt[:, 9] * (1.0 - snow_frac[:, 0] * (1.0 - jnp.exp(-snowd[:, 0] / VTMASK[9])))
    dsfrac_s = expsne
    be1 = vt[:, None] * alb_jh[:, 0, :] * (1.0 - 0.5 * wearth[:, None] * WETSRA)
    for k in range(1, 12):
        if k in (9, 10):
            continue
        vt = pvt[:, k] * (1.0 - snow_frac[:, 1] * (1.0 - jnp.exp(-snowd[:, 1] / VTMASK[k])))
        be1 = be1 + vt[:, None] * alb_jh[:, k, :]
        expsne = expsne + vt
    expsne = jnp.where(nosnow, 1.0, expsne)
    beavn = jnp.where(nosnow[:, None], be0, be1)
    xeavn = beavn
    dalb = dalbsn[:, None] / L[None, :2]
    snw = (1.0 - expsne)[:, None]
    be_lo = beavn[:, :2] + jnp.maximum(0.0, bsn[:, :2] + dalb) * snw
    xe_lo = xeavn[:, :2] + jnp.maximum(0.0, xsn[:, :2] + dalb) * snw
    be_hi = beavn[:, 2:] + bsn[:, 2:] * snw
    xe_hi = xeavn[:, 2:] + xsn[:, 2:] * snw
    bea = jnp.concatenate([be_lo, be_hi], axis=1)
    xea = jnp.concatenate([xe_lo, xe_hi], axis=1)
    ea_on = (pearth >= 1.0e-04)[:, None]
    bea = jnp.where(ea_on, bea, 0.0); xea = jnp.where(ea_on, xea, 0.0)
    # ---- ocean ice
    if int(ksialb) == 1:
        expsno = jnp.exp(-snowoi / DMOICE)
        asn2 = ALBDIF_[1, jh] * jnp.exp(-AGEXPF_[1, jh] * ageSN[:, 1])
        bsn2 = jnp.asarray(ASNALB[:6])[None, :] + asn2[:, None] * fsn
        ahm = jnp.where(jlat > jnorth, ANHZOI, ASHZOI)
        fdz = zoice / (zoice + ahm)
        boi = fdz[:, None] * jnp.asarray(AOIALB[:6])[None, :] * expsno[:, None] + bsn2 * (1.0 - expsno)[:, None]
        gz2 = jnp.asarray(GZSNOW[:6])[None, :]
        xoi = jnp.where((kkz > 0)[:, None], rxsnow(boi, cz[:, None], gz2, tab), boi)
    else:
        boi = jnp.broadcast_to(jnp.asarray(AOIMAX), (N, 6))
        thin = zoice < 1.0
        boi_t = jnp.concatenate([jnp.asarray(AOIMIN[:4])[None, :] + (jnp.asarray(AOIMAX[:4]) - jnp.asarray(AOIMIN[:4]))[None, :]
                                 * jnp.sqrt(zoice)[:, None], boi[:, 4:]], axis=1)
        boi = jnp.where(thin[:, None], boi_t, boi)
        sno = zsnwoi > 0.0
        patchy = jnp.where(sno, jnp.where(zsnwoi >= 0.1, 1.0, zsnwoi / 0.1), 0.0)
        alsf = jnp.where(flags[:, None] > 0.5, jnp.asarray(ASNWET)[None, :], jnp.asarray(ASNDRY)[None, :])
        snag = jnp.where(flags > 0.5, jnp.where(zsnwoi > 0.25, 0.015 / 0.7 * ageSN[:, 1], 0.071 / 0.7 * ageSN[:, 1]),
                         0.006 / 0.82 * ageSN[:, 1])
        snag = jnp.minimum(snoage_fac_max, snag)
        alsf = alsf * (1.0 - snag)[:, None]
        boi = jnp.where(sno[:, None], boi * (1.0 - patchy)[:, None] + alsf * patchy[:, None], boi)
        almp = jnp.broadcast_to(jnp.asarray(AMPMIN), (N, 6))
        mp = (zmp > 0.0) & (zmp < 0.5)
        almp = jnp.where(mp[:, None], almp + (jnp.asarray(AOIMAX)[None, :] - almp) * ((1.0 - 2.0 * zmp) ** 2)[:, None], almp)
        # strict-IEEE blend: with fused multiply-add the sum 0.05*(1-f) + 0.05*f can land one ulp above 0.05 and the threshold of RXSNOW
        # (rbsno < 0.05) then takes the other branch (a jump of up to 0.45 in the albedo); each product gets a second use so that it cannot be contracted
        p_a, e_a = _two_prod(boi, (1.0 - fmp)[:, None])
        p_b, e_b = _two_prod(almp, fmp[:, None])
        boi = (p_a + p_b) + (e_a * 0.0 + e_b * 0.0)
        gz2 = jnp.asarray(GZSNOW[:6])[None, :]
        xoi = jnp.where((kkz > 0)[:, None], rxsnow(boi, cz[:, None], gz2, tab), boi)
        expsno = 1.0 - patchy
        d2 = dalbsn[:, None] / L[None, :2]
        boi = jnp.concatenate([jnp.maximum(0.0, boi[:, :2] + d2), boi[:, 2:]], axis=1)
        xoi = jnp.concatenate([jnp.maximum(0.0, xoi[:, :2] + d2), xoi[:, 2:]], axis=1)
    oi_on = (poice >= 1.0e-04)[:, None]
    boi = jnp.where(oi_on, boi, 0.0); xoi = jnp.where(oi_on, xoi, 0.0)
    # ---- land ice
    asn3 = ALBDIF_[2, jh] * jnp.exp(-AGEXPF_[2, jh] * ageSN[:, 2])
    bsn3 = jnp.asarray(ASNALB[:6])[None, :] + asn3[:, None] * fsn
    expsnl = jnp.exp(-snowli / DMLICE)
    bli = jnp.asarray(ALIALB[:6])[None, :] * expsnl[:, None] + bsn3 * (1.0 - expsnl)[:, None]
    if int(ksialb) == 0:
        fix = (jlat < jnp.rint(mlat46 / 6.0)) | ((jlat < 45) & (jlat > 38) & (ilon < 33) & (ilon > 23))
        amean = jnp.where(kkz > 0, .78, .8)
        blivis = 0.95
        blinir = (amean - .585 * blivis) / .415
        bfix = jnp.concatenate([jnp.full((N, 1), blivis), jnp.broadcast_to(blinir[:, None], (N, 5))], axis=1)
        bli = jnp.where(fix[:, None], bfix, bli)
    xli_r = rxsnow(bli, cz[:, None], gz, tab)
    xli = jnp.where((kkz > 0)[:, None], xli_r, bli)
    d3 = dalbsn[:, None] / L[None, :2]
    bli = jnp.concatenate([jnp.maximum(0.0, bli[:, :2] + d3), bli[:, 2:]], axis=1)
    xli = jnp.concatenate([jnp.maximum(0.0, xli[:, :2] + d3), xli[:, 2:]], axis=1)
    li_on = (plice >= 1.0e-04)[:, None]
    bli = jnp.where(li_on, bli, 0.0); xli = jnp.where(li_on, xli, 0.0)
    # PRNB(J, type), J = 7 - L
    prnb = jnp.stack([boc[:, ::-1], bea[:, ::-1], boi[:, ::-1], bli[:, ::-1]], axis=2)       # (N, 6 [J], 4)
    prnx = jnp.stack([xoc[:, ::-1], xea[:, ::-1], xoi[:, ::-1], xli[:, ::-1]], axis=2)
    return prnb, prnx


# -------------------------------------------------------------------------------------------------------------------- gases
def getgas_ulgas(plb, shl, lat46, o3col, fulgas, ppmv80, plb0_ref, hlb0, fpxco2, ls1_loc, h2o_scaled=True):
    """getgas + FPXSCALEGAS + seth2o for N columns.  plb (N, 44) bottom-first; shl (N, 43); lat46 (N,) latitude of the 72x46 box (deg);
    o3col (N, 43) ozone amount on the radiation layering; fulgas/ppmv80 (13,); plb0_ref/hlb0 (44,) reference pressure / height of the
    radiation levels (RADPAR PLB0/HLB0); fpxco2 (N, 43); ls1_loc (N,) 1-based.  Returns ulgas (N, 13, 43) (gas, layer bottom-first) and u0gas."""
    N = plb.shape[0]
    dpl = plb[:, :-1] - plb[:, 1:]                                      # (N, 43)
    ppm = jnp.asarray(ppmv80)
    u0 = ppm[None, :, None] * PPMV_TO_CM_AT_STP * dpl[:, None, :]       # (N, 13, 43) (index 0 = gas 1 unused here)
    # latitude dependent scale height factor (K = 6..12 except 10)
    dl = lat46
    ptro = jnp.full_like(dl, 100.0)
    ptro = jnp.where(dl < -40.0, 189.0 - (dl + 40.0) * 2.22, ptro)
    ptro = jnp.where(dl > 40.0, 189.0 + (dl - 40.0) * 2.22, ptro)
    plb0 = np.asarray(plb0_ref); hlb = np.asarray(hlb0)
    ge = jnp.asarray(plb0[:NL])[None, :] >= ptro[:, None]               # (N, 43): PLB0(L) >= PTRO
    # Z0LAT = HLB0 of the last layer satisfying the condition (loop keeps overwriting)
    idx = jnp.max(jnp.where(ge, jnp.arange(NL)[None, :], -1), axis=1)
    z0lat = jnp.where(idx >= 0, jnp.asarray(hlb)[jnp.maximum(idx, 0)], 0.0)
    ok = plb0[0] >= ptro                                                # safety check PLB0(1) >= PTRO
    hl_lo = jnp.asarray(hlb[:NL])[None, :]
    hl_hi = jnp.asarray(hlb[1:NL + 1])[None, :]
    for k in (6, 7, 8, 9, 11, 12):
        zt = (hl_hi - z0lat[:, None]) / ZH[k - 1]
        zb = (hl_lo - z0lat[:, None]) / ZH[k - 1]
        expzt = jnp.exp(-zt)
        expzb = jnp.exp(-zb)
        expzb = jnp.where(zb < 0.0, 1.0 - zb, expzb)
        fac = (expzb - expzt) / jnp.maximum(zt - zb, 1.0e-6)
        apply = ok[:, None] & (zt > 0.0)
        u0 = u0.at[:, k - 1, :].set(jnp.where(apply, u0[:, k - 1, :] * fac, u0[:, k - 1, :]))
    # NO2
    no2 = _no2_column(hlb, NL)                                          # (43,) on the reference layering
    u0 = u0.at[:, 4, :].set(jnp.asarray(no2)[None, :] * dpl / jnp.asarray(plb0[:NL] - plb0[1:NL + 1])[None, :])
    # ozone from the climatology column
    u0 = u0.at[:, 2, :].set(o3col)
    # pole-to-pole gradient
    sinlat = jnp.sin(lat46 * np.pi / 180.0)
    for k in range(2, 13):
        if k == 3:
            continue
        part = 1.0 + 0.5 * PPGRAD[k - 1] * sinlat
        u0 = u0.at[:, k - 1, :].set(u0[:, k - 1, :] * part[:, None])
    # water vapour (seth2o): U0GAS(:,1) = h2o_mmr_to_cm_at_stp * DPL * SHL
    u0 = u0.at[:, 0, :].set(H2O_MMR_TO_CM_AT_STP * dpl * shl)
    ful = jnp.asarray(fulgas)
    ul = u0 * ful[None, :, None]
    # FPXSCALEGAS: CO2 vertical profile scaling
    ul = ul.at[:, 1, :].set(ul[:, 1, :] * fpxco2)
    # seth2o: stratospheric H2O scaled by FULGAS(1): ULGAS(1:LS1-1,1) = U0GAS, ULGAS(LS1:NL,1) = U0GAS*FULGAS(1)
    lay = jnp.arange(1, NL + 1)[None, :]
    ul = ul.at[:, 0, :].set(jnp.where(lay < ls1_loc[:, None], u0[:, 0, :], u0[:, 0, :] * ful[0]))
    return ul, u0


def _no2_column(hlb0, nl):
    """The NO2 column amounts (cm-atm) on the reference layering from CMANO2 (every 2 km), trapezoid integration (GETGAS)."""
    hl = np.asarray(hlb0)
    out = np.zeros(nl)
    acm = 0.0
    hi = 0.0
    hj = hi + 2.0
    fi = CMANO2[0]
    fj = CMANO2[1]
    hlv = hl[1]
    L = 1
    J = 2
    while True:
        dh = hj - hi
        if hj <= hlv:
            acm = acm + (fi + fj) * dh * 0.5
            hi = hj
            fi = fj
            J += 1
            if J > 42:
                break
            hj = hi + 2.0
            fj = CMANO2[J - 1]
        else:
            ff = fi + (fj - fi) * (hlv - hi) / dh
            dh = hlv - hi
            acm = acm + (fi + fj) * dh * 0.5
            out[L - 1] = acm
            acm = 0.0
            hi = hlv
            fi = ff
            if L == nl:
                break
            L += 1
            hlv = hl[L]
    out[L - 1] = acm
    return out


# -------------------------------------------------------------------------------------------------------------------- aerosols
def rhdtna(tk, na):
    """RAD_UTILS.f RHDTNA: temperature dependent deliquescence RH of aerosol type na (1..4)."""
    c = {1: 25.0, 2: 80.0, 3: 852.0, 4: 25.0}[na]
    r = {1: 0.80, 2: 0.75, 3: 0.62, 4: 0.80}[na]
    return jnp.minimum(1.0, r * jnp.exp(c * (298.0 - tk) / (298.0 * tk)))


def kdeliq_update(kdeliq, rhl, tlm, krhdtk=1):
    """SETAER label 200: deliquescence hysteresis.  kdeliq (N, 40, 4) int (layer, aerosol type); rhl, tlm (N, 43) (first 40 layers)."""
    out = []
    for na in range(1, 5):
        k = kdeliq[:, :, na - 1]
        rh = rhl[:, :LM]
        rhd = rhdtna(tlm[:, :LM], na) if krhdtk == 1 else RHD[na - 1]
        up = (k == 0) & (rh > rhd)
        dn = (k != 0) & (rh < RHC[na - 1])
        out.append(jnp.where(up, 1, jnp.where(dn, 0, k)))
    return jnp.stack(out, axis=2)


def aerosol_inputs(ataulx, dtaulx, vtaulx, vreff, q55h2s, taucon_dust, redust, rodust, c60):
    """planet_rad aerosol set-up of one RCOMPX call -> n_aer, component (n,), radius (N, n), mass_div_dens (N, n, 43), frac_sulph (N, n),
    scaling_sw / scaling_lw (N, n).  ataulx (N, 43, 6) (bottom first), dtaulx (N, 43, nsized) (already x TAUCON), vtaulx (N, 43);
    c60: dict with fstaer.., fs8opx (N, 8), ft8opx, denaer (N, 8), refdry, frsulf, drym2g."""
    N = ataulx.shape[0]
    i_aer_planet = [7, 14, 30, 25, 4, 4, 2, 6]                          # ip_ammonium_sulphate, sodium_chloride, nitrate, biogenic, soot, soot, dust, sulphuric
    den = c60["denaer"][0]; ref = c60["refdry"][0]; frs = c60["frsulf"][0]; dry = c60["drym2g"][0]
    fs8 = c60["fs8opx"]; ft8 = c60["ft8opx"]
    fsx_a = c60["fstaer"] * c60["fsaaer"]; ftx_a = c60["fttaer"] * c60["ftaaer"]
    comp, rad, mdd, fsu, ssw, slw = [], [], [], [], [], []
    for na in range(1, 7):
        comp.append(i_aer_planet[na - 1])
        rad.append(jnp.full((N,), ref[na - 1] * 1e-6))
        mdd.append(ataulx[:, ::-1, na - 1] / (dry[na - 1] * 1e3) / (den[na - 1] * 1e3))
        fsu.append(jnp.full((N,), frs[na - 1] if na < 5 else 0.0))
        ssw.append(fsx_a * fs8[:, na - 1]); slw.append(ftx_a * ft8[:, na - 1])
    fsx_d = c60["fstaer"] * c60["fsdaer"]; ftx_d = c60["fttaer"] * c60["ftdaer"]
    for n in range(dtaulx.shape[2]):
        comp.append(i_aer_planet[6])
        rad.append(jnp.full((N,), redust[n] * 1e-6))
        mdd.append(dtaulx[:, ::-1, n] / taucon_dust[n] / (rodust[n] * 1e3))
        fsu.append(jnp.zeros((N,)))
        ssw.append(fsx_d * fs8[:, 6]); slw.append(ftx_d * ft8[:, 6])
    fsx_v = c60["fstaer"] * c60["fsvaer"]; ftx_v = c60["fttaer"] * c60["ftvaer"]
    comp.append(i_aer_planet[7])
    rad.append(vreff * 1e-6)
    mdd.append(vtaulx[:, ::-1] / ((0.75 / den[7] * q55h2s / vreff) * 1e3)[:, None] / (den[7] * 1e3))
    fsu.append(jnp.zeros((N,)))
    ssw.append(fsx_v * fs8[:, 7]); slw.append(ftx_v * ft8[:, 7])
    return (np.array(comp), jnp.stack(rad, axis=1), jnp.stack(mdd, axis=1), jnp.stack(fsu, axis=1), jnp.stack(ssw, axis=1),
            jnp.stack(slw, axis=1))


def cloudeps(plb, epl, epm, eph):
    """GETEPS, KCLDEP = 4: CLDEPS(L) = EPMID, EPLOW where PLB(L) > 750, EPHIG where PLB(L) < 430 (PLB(L) = lower edge)."""
    pl = plb[:, :NL]
    e = jnp.broadcast_to(epm[:, None], pl.shape)
    e = jnp.where(pl > 750.0, epl[:, None], e)
    e = jnp.where(pl < 430.0, eph[:, None], e)
    return e
