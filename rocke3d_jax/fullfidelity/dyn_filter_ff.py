"""D101/D102: sea-level-pressure filter FILTER, SLP, SHAP1D, isotropslp, MAtoPMB and the energy
functions (getTotalEnergy = CONSERV_KE + REGRID_BtoA_EXT + conserv_PE + GLOBALSUM, and
addEnergyAsDiffuseHeat) ported to numpy.

Fortran (model/ of modelE2_planet_2.0):  FILTER ATMDYN.f:1418-1624 (live part: MFILTR=1 -> SLP branch, lines
1463-1596; the temperature-stratification part 1597-1623 is dead because MFILTR=1, the non-SLP pressure
branch 1517-1545 is dead because pfilter_using_slp=.true. for planet_name='Earth', the V2_PSURF_FILTER
blocks and the TRACERS_ON block are not compiled), SLP shared/Utilities.F90:98-127, SHAP1D ATMDYN.f:1958-1990,
isotropslp ATMDYN.f:1814-1836 (calls shap1, from dyn_isotropuv_ff), MAtoPMB ATM_UTILS.f:241-284,
getTotalEnergy ATM_UTILS.f:579-605, addEnergyAsDiffuseHeat ATM_UTILS.f:607-633, CONSERV_KE ATMDYN.f:2250-2281,
regrid_btoa_ext ATMDYN.f:2744-2779, conserv_PE DIAG.f:1131-1165, globalSum_IJ GlobalSum_mod.F90:166-248.

Conventions (as the dumps, see dyn_filter_compare.py): 2-D fields (IM,JM); MA, PK, PMID, PEDN (LM[+1],IM,JM);
T, Q, QCL, QCI, U, V (IM,JM,LM); QMOM (NMOM,IM,JM,LM).  Index 0 = Fortran 1.
Arithmetic follows the Fortran statement order (build flags -fp-model strict, -assume protect_parens: no FMA,
no reassociation); every reduction is strictly sequential.  The only non-IEEE-exact operations are
`**` (SLP: (1.+BZBYT)**GBYRB; MAtoPMB: PMID**KAPA) and `exp` (SLP, BETA<=1e-6 branch) which ifort sends to
Intel libimf; numpy can differ by 1 ulp in a few 0.01 % of cells.  With imf_pow=True the libimf scalar pow is
called through ctypes (intel_libm_ff.py) and both are reproduced bit for bit.
"""
import ctypes

import numpy as np

import dyn_isotropuv_ff as iso
import intel_libm_ff

IM, JM, LM = 72, 46, 40
NSHAP = 8


# ----------------------------------------------------------------------------- helpers
def seqsum(a, axis=0):
    """Strictly sequential (left-to-right) sum along `axis` (ifort -fp-model strict `Sum`)."""
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    s = a[0].copy()
    for k in range(1, a.shape[0]):
        s = s + a[k]
    return s


def _pow_arr(x, y, imf):
    """x**y; x and y arrays or scalars.  imf=True: libimf scalar pow (bitwise to ifort)."""
    x = np.asarray(x, dtype=float)
    if not imf:
        return x ** y
    f = intel_libm_ff._load()
    if f is None:
        raise RuntimeError("Intel libimf not available")
    xb, yb = np.broadcast_arrays(x, np.asarray(y, dtype=float))
    return np.array([f(a, b) for a, b in zip(xb.ravel(), yb.ravel())]).reshape(xb.shape)


def load_consts(path):
    """Read ffd_filt_consts.bin (layout: ATM_DRV_dynD.f.patch ffdd_consts)."""
    raw = np.fromfile(path, dtype='>f8')
    s = raw[:32]
    names = ['im', 'jm', 'lm', 'nmom', 'cos_limit', 'dt', 'mfiltr', 'r7', 'grav', 'rgas', 'sha', 'kapa',
             'bygrav', 'bmoist', 'by3', 'mb2kg', 'kg2mb', 'mtop', 'mfixs', 'psf', 'pmtop', 'areag', 'byim',
             'radius', 'lmfrac1', 'lmfrac2']
    g = {n: float(s[i]) for i, n in enumerate(names)}
    o = 32
    for n in ['cosp', 'dxp', 'dxyp', 'dxyv', 'rapvs', 'rapvn', 'dxyn', 'dxys', 'imaxj']:
        g[n] = raw[o:o + JM].copy(); o += JM
    g['imaxj'] = g['imaxj'].astype(int)
    for n in ['mfix', 'mfrac']:
        g[n] = raw[o:o + LM].copy(); o += LM
    for n in ['zatmo', 'axyp', 'byaxyp']:
        g[n] = raw[o:o + IM * JM].reshape(JM, IM).T.copy(); o += IM * JM
    assert o == raw.size, (o, raw.size)
    for n in ('im', 'jm', 'lm', 'nmom', 'lmfrac1', 'lmfrac2'):
        g[n] = int(g[n])
    return g


# ----------------------------------------------------------------------------- SLP
def slp(ps, tas, zs, g, imf_pow=False, stats=None):
    """SLP(PS,TAS,ZS) of Utilities.F90 elementwise (arrays of any shape).
    BMOIST, GRAV, RGAS, BY3 from g.  Literal types as in the source (290.5, 255, .5, 1., 2. exact in
    single precision, so no rounding issue).  `stats` (dict) accumulates branch counts."""
    bmoist, grav, rgas, by3 = g['bmoist'], g['grav'], g['rgas'], g['by3']
    ps = np.asarray(ps, dtype=float); tas = np.asarray(tas, dtype=float); zs = np.asarray(zs, dtype=float)
    out = ps.copy()                                         # ZS == 0: SLP=PS
    m = zs != 0.0
    with np.errstate(all='ignore'):
        tsl = tas + bmoist * zs
        tasn = tas.copy()
        beta = np.full(ps.shape, bmoist)
        c1 = m & (tas < 290.5) & (tsl > 290.5)
        beta = np.where(c1, (290.5 - tas) / zs, beta)
        c2 = m & (tas > 290.5) & (tsl > 290.5)
        tasn = np.where(c2, 0.5 * (290.5 + tas), tasn)
        c3 = m & (tas < 255)
        tasn = np.where(c3, 0.5 * (255.0 + tas), tasn)
        bzbyt = beta * zs / tasn
        gbyrb = grav / (rgas * beta)
        cb = m & (beta > 1e-6)
        ce = m & ~cb
        a = _pow_arr(1. + bzbyt, gbyrb, imf_pow)
        r_pow = ps * a
        r_exp = ps * np.exp((1. - 0.5 * bzbyt + _pow_arr(bzbyt, 2. * by3, imf_pow)) * gbyrb * bzbyt)
        out = np.where(cb, r_pow, np.where(ce, r_exp, out))
    if stats is not None:
        for k, v in (('zs0', ~m), ('beta_lapse', c1), ('tasn_warm', c2), ('tasn_cold', c3),
                     ('pow_branch', cb), ('exp_branch', ce)):
            stats[k] = stats.get(k, 0) + int(np.sum(v))
    return out


# ----------------------------------------------------------------------------- SHAP1D / isotropslp
def shap1d(x, norder=NSHAP):
    """SHAP1D(NORDER,X) on X (IM,JM): rows J=2..JM-1 (J_0S..J_1S), zonal cyclic Shapiro filter
    X := X - (norder-times  XS(i)=XS(i-1)-2XS(i)+XS(i+1)) / 4**norder  (Fortran evaluates
    ((XSIM1-XSI)-XSI)+XS(I+1) with the pre-pass neighbours)."""
    x = np.array(x, dtype=float)
    by4ton = 1. / 4. ** norder
    xs = x[:, 1:JM - 1].copy()
    for _ in range(norder):
        xs = ((np.roll(xs, 1, axis=0) - xs) - xs) + np.roll(xs, -1, axis=0)
    x[:, 1:JM - 1] = x[:, 1:JM - 1] - xs * by4ton
    return x


def isotropslp(x, g, coscut=None, dt=None, stats=None):
    """isotropslp(slp,coscut): near-polar rows (COSP(J) < coscut) get shap1 with fac=k*dt/dxp(j)**2,
    k=1e3.  Rows J=2..JM-1 only.  Returns a new array."""
    coscut = g['cos_limit'] if coscut is None else coscut
    dt = g['dt'] if dt is None else dt
    x = np.array(x, dtype=float)
    cosp, dxp = g['cosp'], g['dxp']
    for j in range(2, JM):                                  # 1-based J_0S..J_1S
        if cosp[j - 1] >= coscut:                           # far_from_pole
            continue
        fac = 1e3 * dt / (dxp[j - 1] * dxp[j - 1])
        row, n = iso.shap1_rows(x[None, :, j - 1], np.array([fac]))
        x[:, j - 1] = row[0]
        if stats is not None:
            stats.setdefault('iso_rows', []).append((j, int(n[0])))
    return x


# ----------------------------------------------------------------------------- MAtoPMB
def matopmb(ma, g, imf_pow=False):
    """MAtoPMB (ATM_UTILS.f:241-284): P-arrays from MA (LM,IM,JM).  Returns dict(masum, pedn (LM+1,..),
    pmid, pk, pdsig, p).  PEK, byMA and the ATMSRF copies are not ported (not part of the validated set)."""
    kg2mb, mtop, kapa, mfixs = g['kg2mb'], g['mtop'], g['kapa'], g['mfixs']
    masum = np.zeros(ma.shape[1:])
    pedn = np.zeros((LM + 1,) + ma.shape[1:])
    pmid = np.zeros_like(ma); pk = np.zeros_like(ma); pdsig = np.zeros_like(ma)
    pedn[LM] = mtop * kg2mb
    for l in range(LM - 1, -1, -1):
        masum = ma[l] + masum
        pdsig[l] = ma[l] * kg2mb
        pmid[l] = pedn[l + 1] + pdsig[l] * .5
        pedn[l] = pedn[l + 1] + pdsig[l]
        pk[l] = _pow_arr(pmid[l], kapa, imf_pow)
    p = (masum - mfixs) * kg2mb
    return dict(masum=masum, pedn=pedn, pmid=pmid, pk=pk, pdsig=pdsig, p=p)


# ----------------------------------------------------------------------------- energy functions
def regrid_btoa_ext(x, g):
    """regrid_btoa_ext (ATMDYN.f:2744-2779): x_bgrid*dxyv -> x_agrid*dxyp on x (IM,JM); row 1 input is
    not used (the south-pole row is rebuilt from row 2, the north-pole row from the B-grid row JM)."""
    rapvs, rapvn, dxyp, dxyv, byim = g['rapvs'], g['rapvn'], g['dxyp'], g['dxyv'], g['byim']
    xo = np.array(x, dtype=float)
    new = xo.copy()
    new[:, 0] = seqsum(xo[:, 1]) * byim * (dxyp[0] / dxyv[1])
    for j in range(2, JM):                                  # J=2..JM-1 (1-based), old rows j and j+1
        r0 = xo[:, j - 1]; r1 = xo[:, j]
        new[:, j - 1] = ((np.roll(r0, 1) + r0) * rapvs[j - 1]
                         + (np.roll(r1, 1) + r1) * rapvn[j - 1])
    new[:, JM - 1] = seqsum(xo[:, JM - 1]) * byim * (dxyp[JM - 1] / dxyv[JM - 1])
    return new


def conserv_ke_bgrid(ma, u, v, g):
    """B-grid column kinetic energy of CONSERV_KE before the regrid.  ma (LM,IM,JM), u,v (IM,JM,LM).
    Rows J=2..JM; row 1 is not defined (returned as 0)."""
    dxyn, dxys = g['dxyn'], g['dxys']
    rke = np.zeros((IM, JM))
    ut = u.transpose(2, 0, 1); vt = v.transpose(2, 0, 1)    # (LM,IM,JM)
    for j in range(2, JM + 1):
        jj = j - 1
        a = (ma[:, :, jj - 1] + np.roll(ma[:, :, jj - 1], -1, axis=1)) * dxyn[jj - 1] + \
            (ma[:, :, jj] + np.roll(ma[:, :, jj], -1, axis=1)) * dxys[jj]
        rke[:, jj] = seqsum(a * (ut[:, :, jj] ** 2 + vt[:, :, jj] ** 2), axis=0) * .25
    return rke


def conserv_ke(ma, u, v, g):
    """CONSERV_KE (J/m^2, A grid): returns (kea, keb) with keb the pre-regrid B-grid field."""
    keb = conserv_ke_bgrid(ma, u, v, g)
    kea = regrid_btoa_ext(keb, g)
    return kea * g['byaxyp'], keb


def conserv_pe(masum, t, pk, ma, g):
    """conserv_PE (DIAG.f:1131-1165): TPE = ZATMO*(MASUM+MTOP) + SHA*Sum_L(T*PK*MA).  t (IM,JM,LM)."""
    tt = t.transpose(2, 0, 1)                               # (LM,IM,JM)
    s = seqsum((tt * pk) * ma, axis=0)
    tpe = g['zatmo'] * (masum + g['mtop']) + g['sha'] * s
    # only I<=IMAXJ(J) is computed; the poles are then replicated from I=1
    tpe[1:, 0] = tpe[0, 0]
    tpe[1:, JM - 1] = tpe[0, JM - 1]
    return tpe


def global_sum_ij(a):
    """GLOBALSUM_IJ (all J): zonal sum over I per J, then sum over J."""
    return float(seqsum(seqsum(a, axis=0), axis=0))


def total_energy(masum, ma, pk, t, u, v, g, return_parts=False):
    """getTotalEnergy: GLOBALSUM_IJ of ((KE+PE)*AXYP)/AREAG."""
    kea, keb = conserv_ke(ma, u, v, g)
    pe = conserv_pe(masum, t, pk, ma, g)
    te = ((kea + pe) * g['axyp']) / g['areag']
    tot = global_sum_ij(te)
    if return_parts:
        return tot, dict(keb=keb, kea=kea, pe=pe, te=te)
    return tot


def add_energy_as_diffuse_heat(delta, t, pk, g):
    """addEnergyAsDiffuseHeat: EDIFF = DELTA/((PSF-PMTOP)*SHA*MB2KG); T(:,:,L) -= EDIFF/PK(L,:,:)
    for all I, J, L.  Returns (t_new, ediff)."""
    ediff = delta / ((g['psf'] - g['pmtop']) * g['sha'] * g['mb2kg'])
    tn = t.copy()
    for l in range(LM):
        tn[:, :, l] = t[:, :, l] - ediff / pk[l]
    return tn, ediff


# ----------------------------------------------------------------------------- FILTER
def row_loop(x, y, pednold, g, stats=None):
    """The per-row loop of FILTER (ATMDYN.f:1494-1514): PEDN(1,I,J)=X/Y, limit to [0.9882, 1.0118]*PEDNOLD,
    then remove the row-mean change (PDIF=(PSUMN-PSUMO)*BYIM) so each row keeps its mass.  Rows J=2..JM-1,
    sequential sums over I.  x,y,pednold (IM,JM); returns the new PEDN(1) (rows 1 and JM unchanged)."""
    pn = np.array(pednold, dtype=float)
    byim = g['byim']
    for j in range(2, JM):                                  # J1P..JNP
        jj = j - 1
        psumo = 0.; psumn = 0.
        row = np.zeros(IM)
        for i in range(IM):
            psumo = psumo + pednold[i, jj]
            v = x[i, jj] / y[i, jj]
            lo = 0.9882 * pednold[i, jj]
            hi = 1.0118 * pednold[i, jj]
            if stats is not None:
                stats['clip_lo'] = stats.get('clip_lo', 0) + int(v < lo)
                stats['clip_hi'] = stats.get('clip_hi', 0) + int(v > hi)
            v = max(v, lo)
            v = min(v, hi)
            row[i] = v
            psumn = psumn + v
        pdif = (psumn - psumo) * byim
        pn[:, jj] = row - pdif
    return pn


def slp_filter_pedn(pednold, tsavg, g, imf_pow=False, stats=None, return_stages=False):
    """SLP part of FILTER: new PEDN(1,:,:) (rows 2..JM-1; other rows returned unchanged).
    pednold, tsavg (IM,JM)."""
    zs = g['zatmo'] * g['bygrav']
    x = slp(pednold, tsavg, zs, g, imf_pow=imf_pow, stats=stats)
    y = x / pednold
    stages = {'x1': x.copy(), 'y1': y.copy()}
    x = shap1d(x, NSHAP)
    stages['x2'] = x.copy()
    x = isotropslp(x, g, stats=stats)
    stages['x3'] = x.copy()
    pn = row_loop(x, y, pednold, g, stats=stats)
    stages['pedn4'] = pn.copy()
    return (pn, stages) if return_stages else pn


def slp_filter_pedn_fast(pednold, tsavg, g, imf_pow=False):
    """Same as slp_filter_pedn but vectorised over I with a sequential cumulative sum (np.cumsum is
    sequential); used by the batch tests.  Bitwise identical to the loop version (tested)."""
    zs = g['zatmo'] * g['bygrav']
    x = slp(pednold, tsavg, zs, g, imf_pow=imf_pow)
    y = x / pednold
    x = shap1d(x, NSHAP)
    x = isotropslp(x, g)
    pn = pednold.copy()
    sl = slice(1, JM - 1)
    v = x[:, sl] / y[:, sl]
    v = np.maximum(v, 0.9882 * pednold[:, sl])
    v = np.minimum(v, 1.0118 * pednold[:, sl])
    psumo = np.cumsum(pednold[:, sl], axis=0)[-1]
    psumn = np.cumsum(v, axis=0)[-1]
    pdif = (psumn - psumo) * g['byim']
    pn[:, sl] = v - pdif
    return pn


def filter_slp(pedn1, tsavg, ma, pk, t, q, qcl, qci, qmom, u, v, g, masum0=None, imf_pow=False,
               stats=None, return_stages=False):
    """FILTER with MFILTR=1, pfilter_using_slp (the live path).  Inputs as in ffd_filt_<itime>_in.bin plus
    U,V (for the energy fix) and the entry MASUM (default: recomputed from MA by the MAtoPMB recurrence, which
    equals the last MAtoPMB value because MA is unchanged since).  Returns a dict with the outputs
    (pedn (LM+1,..), pmid, pk, ma, masum, t, q, qcl, qci, qmom) and, when requested, stages."""
    l1, l2 = g['lmfrac1'] - 1, g['lmfrac2'] - 1             # 0-based inclusive layer range
    mfix, mfrac, mtop, mfixs, mb2kg = g['mfix'], g['mfrac'], g['mtop'], g['mfixs'], g['mb2kg']
    if masum0 is None:
        masum0 = matopmb(ma, g, imf_pow=imf_pow)['masum']
    e0 = total_energy(masum0, ma, pk, t, u, v, g)
    mabef = ma.copy(); pkold = pk.copy(); pednold = pedn1.copy()
    pn, stg = slp_filter_pedn(pednold, tsavg, g, imf_pow=imf_pow, stats=stats, return_stages=True)
    # new MA from filtered PEDN(1), rows J=2..JM-1
    man = ma.copy()
    sl = slice(1, JM - 1)
    mvar = pn[:, sl] * mb2kg - mfixs - mtop
    for l in range(l1, l2 + 1):
        man[l, :, sl] = mfix[l] + mvar * mfrac[l]
    mp = matopmb(man, g, imf_pow=imf_pow)
    # scale mixing ratios / moments (rows J=2..JM-1, layers l1..l2)
    tn = t.copy(); qn = q.copy(); qcln = qcl.copy(); qcin = qci.copy(); qmn = qmom.copy()
    for l in range(l1, l2 + 1):
        zmrat = mabef[l][:, sl] / man[l][:, sl]
        tn[:, sl, l] = t[:, sl, l] * pkold[l][:, sl] / mp['pk'][l][:, sl]
        qn[:, sl, l] = q[:, sl, l] * zmrat
        qcln[:, sl, l] = qcl[:, sl, l] * zmrat
        qcin[:, sl, l] = qci[:, sl, l] * zmrat
        qmn[:, :, sl, l] = qmom[:, :, sl, l] * zmrat[None]
    e1 = total_energy(mp['masum'], man, mp['pk'], tn, u, v, g)
    tn2, ediff = add_energy_as_diffuse_heat(e1 - e0, tn, mp['pk'], g)
    out = dict(pedn=mp['pedn'], pmid=mp['pmid'], pk=mp['pk'], ma=man, masum=mp['masum'], t=tn2, q=qn,
               qcl=qcln, qci=qcin, qmom=qmn, e0=e0, e1=e1, ediff=ediff)
    if return_stages:
        out['stages'] = stg
        out['t_prediff'] = tn
    return out
