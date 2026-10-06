"""Batched MSTCNV (moist convection, CLOUDS2.F90:432-3207): all columns of a step at once with numpy -- D132.

The validated per-column port is clouds_mstcnv_ff.py.  This module repeats its statements, in the same order and with the same operation
order / REAL(4) literals, with every scalar replaced by a vector over the N columns of the batch.  The Fortran control flow is kept as
lock-step loops with per-column masks:

  * cloud-base loop `do LMIN=1,LMCM-1` : a Python loop shared by all columns; the cheap pre-test (DMSE) is vector arithmetic over ALL
    columns, MASS_FLUX (already vectorised in clouds_massflux_ff) runs on the columns that pass it, and everything below runs only on those
    compacted columns (gather of the persistent state, scatter back at the end of the LMIN iteration);
  * cloud types IC=1,2 and area partitions NPPL=1,2 : Python loops with per-column "run" masks (cycle/break = mask update);
  * plume ascent `do L=LMIN+1,LM` : Python loop on L, per-column `alive` mask (every `exit`/`cycle` of the Fortran clears the mask; state
    written between two exits is written only for the columns alive at that point);
  * downdraft `do L=LDRAFT,1,-1`, subsidence `do L=LDMIN,LMAX`, precipitation `do L=LMAX-1,1,-1`: lock-step loops over all layers with
    per-column layer-range masks (the per-column bounds LDRAFT/LDMIN/LMAX become masks);
  * the QUS vertical advection ADV1D of the subsidence is batched over the columns of equal slice length (adv1d is already vectorised
    over lines; the qlimit loop of the Q advection stays a scalar loop per line, exactly as in dyn_adv1d_ff).
Exactness rules are those of clouds_lscond_batch.py: IEEE arithmetic in the same order, Python max/min semantics (pymax/pymin), exp/pow
through the same functions as the per-column port (np.exp/np.power on the arrays in libm mode = the elementwise numpy functions the
per-column port calls on 0-d arrays; the libimf proxies of clouds_mstcnv_ff.set_backend in the imf mode), np.where evaluating both
branches only on finite, discarded data.
Array layout: layered arrays (LM+2, N) indexed by the 1-based layer L (rows 0 and LM+1 are the zero pad cells of the per-column port);
moments (9, LM+2, N); K-sampled winds (4, LM+2, N); (4, N) for RA.  Poles (KMAX=72) are not supported (the caller uses the per-column
port).
NOT reproduced (pure diagnostics, not read by any exit field of CONDSE): DGDEEP DGSHLW DPHASE DPHADEEP DPHASHLW DTOTW DQCOND DGDQM
DQMTOTAL DQMSHLW DQMDEEP DQCTOTAL DQCSHLW DQCDEEP, the branch counters `br` and the checkpoint records `ck`.
"""
import numpy as np

import clouds_dq_ff as dq
import clouds_helpers_ff as hp
import clouds_massflux_ff as mf
import clouds_mstcnv_ff as mc
import dyn_adv1d_ff as adv

F = np.float64
LM, NMOM = mc.LM, mc.NMOM
XYM, ZM = mc.XYM, mc.ZM
XYMa, ZMa = np.array(XYM), np.array(ZM)
RGAS, GRAV, TEENY, PI = mc.RGAS, mc.GRAV, mc.TEENY, mc.PI
LHE, LHS, TF, BYSHA, BYGRAV, DELTX, SLHE, TI = mc.LHE, mc.LHS, mc.TF, mc.BYSHA, mc.BYGRAV, mc.DELTX, mc.SLHE, mc.TI
CN0, CN0I, CN0G, RHOG, RHOIP = mc.CN0, mc.CN0I, mc.CN0G, mc.RHOG, mc.RHOIP
ITMAX, FITMAX, WMAX, SECONDS_PER_HOUR = mc.ITMAX, mc.FITMAX, mc.WMAX, mc.SECONDS_PER_HOUR
CLDMIN, FDDET, DTMIN1, COETAU, WMU, WMUL, CCMUL, CCMUL2 = mc.CLDMIN, mc.FDDET, mc.DTMIN1, mc.COETAU, mc.WMU, mc.WMUL, mc.CCMUL, mc.CCMUL2
f4 = mc.f4
NK = 4
_DBG = None          # test hook: callable(stage, lmin, ic, nppl, locals())
OUT_ARR = ("tl sm qm fssl cldmcl taumcl svlatl svlat1 svwmxl csizel condpt vsubl tpsav mcflx dgdsm ddmflx tdnl qdnl u00l qlmc qimc "
           "cnvmmrl condmmr").split()
OUT_SCAL = "ierr lerr lmcmin lmcmax prcpmc cldslwij clddepij airxl prheat wmsum wmctwp wmclwp fmc1 mccont".split()


def pymax(a, b):
    return np.where(b > a, b, a)


def pymin(a, b):
    return np.where(b < a, b, a)


# ---------------------------------------------------------------------------------------------- packing
def stack_r(r_list):
    """List of per-column input records (clouds_condse_ff r dicts) -> batch inputs (layer arrays (LM,N), moments (9,LM,N), winds (4,LM,N))."""
    R = {}
    for k in ("pearth", "pland", "xmass", "bydtsrc", "dtsrc", "bybr"):
        R[k] = np.array([float(r[k]) for r in r_list])
    R["dcl"] = np.array([int(r["dcl"]) for r in r_list])
    R["lmcm"] = np.array([int(r["lmcm"]) for r in r_list])
    for k in "pl plk airm byam etal tl tvl sm qm qcll qcil sdl wturb gzl".split():
        R[k] = np.array([r[k] for r in r_list], float).T.copy()
    R["ple"] = np.array([r["ple"] for r in r_list], float).T.copy()
    for k in ("smom", "qmom"):
        R[k] = np.array([r[k] for r in r_list], float).transpose(1, 2, 0).copy()
    for k in ("um", "vm", "u0", "v0"):
        R[k] = np.array([r[k] for r in r_list], float).transpose(1, 2, 0).copy()
    R["ra"] = np.array([r["ra"] for r in r_list], float).T.copy()
    return R


def _pad(x):
    """(LM, ...N) -> (LM+2, ...N) with the layer index 1..LM (rows 0 and LM+1 zero)."""
    x = np.asarray(x, float)
    out = np.zeros((LM + 2,) + x.shape[1:])
    out[1:1 + x.shape[0]] = x
    return out


def _padm(x):
    """(m, LM, N) -> (m, LM+2, N)."""
    x = np.asarray(x, float)
    out = np.zeros((x.shape[0], LM + 2, x.shape[2]))
    out[:, 1:1 + x.shape[1]] = x
    return out


# ---------------------------------------------------------------------------------------------- helpers
def _dcw_search_fast(wv, pl, ddcw, wmax, nmax):
    """hp._dcw_search with an early exit when every element has finished (identical elementwise results)."""
    wv, pl, ddcw, wmax = np.broadcast_arrays(wv, pl, ddcw, wmax)
    dcw = np.zeros(wv.shape)
    active = np.ones(wv.shape, bool)
    pfac = mc._pow(1000.0 / pl, 0.4)
    for k in range(int(np.max(nmax))):
        act = active & (k < nmax)
        if not act.any():
            break
        vt = (-0.267 + dcw * (5.15e3 - dcw * (1.0225e6 - 7.55e7 * dcw))) * pfac
        ex1 = (vt >= 0.0) & (vt >= wv)
        ex2 = ~ex1 & (vt > wmax)
        ex = act & (ex1 | ex2)
        dcw = np.where(act & ~ex, dcw + ddcw, dcw)
        active = active & ~ex
    return dcw


def conv_micro(pl, wcu, dwcu, lfrz, wcufrz, tp, pland, flamw, flamg, flami, tlmin, tlmin1, condip_in, condgp_in):
    """hp.convective_microphysics on 1-D arrays (same expressions; the two critical-size searches are run as one call)."""
    n = pl.shape
    one = lambda x: np.full(n, x, float)  # noqa: E731
    ti, fitmax, cn0, cn0i, cn0g, rhoip, rhog, wmax = one(TI), one(FITMAX), one(CN0), one(CN0I), one(CN0G), one(RHOIP), one(RHOG), one(WMAX)
    mp = lambda rho, flam, dc, cn: hp.precip_mp(rho, flam, dc, cn, "pow")  # noqa: E731
    with np.errstate(all="ignore"):
        wv = np.maximum(wcu - dwcu, 0.0)
        dcg = hp._dcg(wv, pl)
        dci = hp._dci(wv, pl)
        tig = np.where(lfrz == 0, TF, TF - 4.0 * wcufrz)
        tig = np.where(tig < ti - 10.0, ti - 10.0, tig)
        water = tp >= TF
        ice = ~water & (tp <= tig)
        mixed = ~water & ~ice
        ddcw = np.where(pland < 0.5, 1.5e-3 * fitmax, 6e-3 * fitmax)
        nmax = np.full(n, int(ITMAX) - 1)
        wvu = wcu + dwcu
        dcw_both = _dcw_search_fast(np.concatenate([wv, wvu]), np.concatenate([pl, pl]), np.concatenate([ddcw, ddcw]),
                                    np.concatenate([wmax, wmax]), np.concatenate([nmax, nmax]))
        dcw1, dcw2 = dcw_both[:n[0]], dcw_both[n[0]:]
        condp1_w = mp(hp.RHOW, flamw, dcw1, cn0)
        condp_w = mp(hp.RHOW, flamw, dcw2, cn0)
        condp1_i = mp(rhoip, flami, dci, cn0i)
        dci_u = hp._dci(wvu, pl)
        condp_i = mp(rhoip, flami, dci_u, cn0i)
        fg = (tp - tig) / ((TF - tig) + TEENY)
        fg = np.where(fg > 1.0, 1.0, fg)
        fg = np.where(fg < 0.0, 0.0, fg)
        fg = np.where((tlmin <= TF) | (tlmin1 <= TF), 0.0, fg)
        fi = 1.0 - fg
        cip_l = mp(rhoip, flami, dci, cn0i)
        cgp_l = mp(rhog, flamg, dcg, cn0g)
        condp1_m = fg * cgp_l + fi * cip_l
        dcg_u = hp._dcg(wvu, pl)
        cip_u = mp(rhoip, flami, dci_u, cn0i)
        cgp_u = mp(rhog, flamg, dcg_u, cn0g)
        condp_m = fg * cgp_u + fi * cip_u
    condp = np.where(water, condp_w, np.where(ice, condp_i, condp_m))
    condp1 = np.where(water, condp1_w, np.where(ice, condp1_i, condp1_m))
    condip = np.where(mixed, cip_u, condip_in)
    condgp = np.where(mixed, cgp_u, condgp_in)
    return condp, condp1, condip, condgp


def _mg(A, ms, rows, cols):
    """A (m, LM+2, n)[ms, rows[col], col] -> (len(ms), E) for the columns `cols` (rows aligned with cols)."""
    return A[np.asarray(ms)[:, None], rows[None, :], cols[None, :]]


def _ms(A, ms, rows, cols, val):
    A[np.asarray(ms)[:, None], rows[None, :], cols[None, :]] = val


def adv_batch(s, smom, mass, dm, qlimit):
    """adv1d (ZDIR, cyclic lines) for g lines of nx cells; s (g,nx), smom (9,g,nx), mass, dm (g,nx), updated in place.
    -> (ierr (g,), nerr (g,)) per line.  The qlimit loop is a scalar loop per line (adv._qlimit_cyclic), exactly as in dyn_adv1d_ff; a line
    with ierr==2 returns before the update stage and keeps s/smom/mass as they are after the limiter."""
    g, nx = s.shape
    dirv = adv.ZDIR
    k = np.arange(nx)
    nn = np.broadcast_to(np.where(dm < 0., (k + 1) % nx, k), dm.shape)
    f, fracm, frac1 = adv._flux_stage(s, smom, mass, dm, nn, dirv)
    ierr = np.zeros(g, int)
    nerr = np.zeros(g, int)
    skip = np.zeros(g, bool)
    if qlimit:
        mx, mxx = dirv[0], dirv[3]
        for l in range(g):
            ie, ne = adv._qlimit_cyclic(f[l], fracm[l], s[l], smom[:, l], mx, mxx, None)
            ierr[l], nerr[l] = ie, ne
            skip[l] = ie == 2
    if skip.any():
        s_pre, smom_pre, mass_pre = s.copy(), smom.copy(), mass.copy()
    fm = adv._slope_stage(smom, dm, f, fracm, frac1, nn, dirv)
    adv._update_stage(s, smom, mass, dm, f, fm, adv._cyc_prev, dirv)
    if skip.any():
        s[skip], smom[:, skip], mass[skip] = s_pre[skip], smom_pre[:, skip], mass_pre[skip]
    return ierr, nerr


# ---------------------------------------------------------------------------------------------- the batch
def mstcnv_batch(R, c, mut=None):
    """MSTCNV for N columns.  R: stack_r() output (all columns KMAX=4); c: tunables (clouds_mstcnv_ff.tune_from_consts / RUN_TUNE).
    -> dict of outputs (OUT_SCAL (N,), OUT_ARR (N, LM) like the per-column o but stacked, lhp/precnvl (N, LM+1), smom/qmom (N, 9, LM),
    um/vm (N, 4, LM))."""
    mut = mut or {}
    with np.errstate(all="ignore"):
        return _mstcnv_b(R, c, mut)


def _mstcnv_b(R, c, mut):
    N = R["pl"].shape[1]
    cols_all = np.arange(N)
    lmcm = int(R["lmcm"][0])
    assert (R["lmcm"] == lmcm).all()
    for k in ("xmass", "bydtsrc", "dtsrc", "bybr"):
        assert (R[k] == R[k][0]).all(), k
    xmass, bydtsrc, dtsrc, bybr = F(R["xmass"][0]), F(R["bydtsrc"][0]), F(R["dtsrc"][0]), F(R["bybr"][0])
    pearth, pland, dcl = R["pearth"], R["pland"], R["dcl"]
    pl, ple, plk, airm, byam = (_pad(R[k]) for k in ("pl", "ple", "plk", "airm", "byam"))
    etal, tl, tvl = _pad(R["etal"]), _pad(R["tl"]), _pad(R["tvl"])
    sm, qm = _pad(R["sm"]), _pad(R["qm"])
    qcll, qcil, sdl, wturb, gzl = (_pad(R[k]) for k in ("qcll", "qcil", "sdl", "wturb", "gzl"))
    smom, qmom = _padm(R["smom"]), _padm(R["qmom"])
    ra = np.asarray(R["ra"], float)
    um, vm, u_0, v_0 = (_padm(R[k]) for k in ("um", "vm", "u0", "v0"))
    assert ra.shape[0] == NK
    contce1, contce2 = F(c["entrainment_cont1"]), F(c["entrainment_cont2"])
    rcldlx, rcldix = F(c["radiusl_multiplier"]), F(c["radiusi_multiplier"])
    u00a, u00b = F(c["u00a"]), F(c["u00b"])
    wmu_mult, rwcldox, rimax = F(c["wmu_multiplier"]), F(c["rwcldox"]), F(c["rimax"])
    fddrt = F(c["mc_fddrt"])
    mc_entr_lim, mc_newthv, mc_revp = c["mc_entr_mass_lim_plume"], c["mc_new_ddrft_thetav"], c["mc_revp_abv_cldbase"]
    assert mc_newthv != 0, "mc_new_ddrft_thetav=0 arm not batched"
    f001, f95, pgrad = F(f4(0.001)), F(f4(0.95)), F(f4(0.7))
    zl = lambda: np.zeros((LM + 2, N))  # noqa: E731
    tadj = F(1.0)
    qsatre = dq.qsat(F(283.16), LHE, F(920.0))
    fssl = np.ones((LM + 2, N))
    taumcl, condpt, svwmxl, svlatl, svlat1, vsubl = zl(), zl(), zl(), zl(), zl(), zl()
    precnvl, cldmcl, tpsav, lhp = zl(), zl(), zl(), zl()
    csizel = zl()
    csizel[1:LM + 1] = (rwcldox * 10. * (1. - pearth) + 10. * pearth)[None, :]
    vlat = np.full((LM + 2, N), LHE)
    condmmr, mcflx, dgdsm, ddmflx, tdnl, qdnl = zl(), zl(), zl(), zl(), zl(), zl()
    sm1, qm1 = sm.copy(), qm.copy()
    smold, smomold, qmold, qmomold = sm.copy(), smom.copy(), qm.copy(), qmom.copy()
    u00l = zl()
    x_u00 = u00b * F(f4(.001)) * F(f4(.050)) * F(3.) * mc._pow(F(222.0), F(f4(.33)))
    u00_pbl = 1.0 - 2. * x_u00 / qsatre
    u00_mc = 1.0 - 2. * (u00b * 2.0e-4 / qsatre)
    pl_dcl = pl[dcl, cols_all]
    for L in range(1, LM + 1):
        u00l[L] = np.where(pl[L] >= pl_dcl, u00_pbl, 0.)
    dwcu = np.zeros(N)
    for L in range(1, lmcm + 1):
        dwcu = dwcu + airm[L] * tl[L] * RGAS / (GRAV * pl[L])
    dwcu = 0.5 * dwcu * bydtsrc / F(lmcm)
    wcu2 = zl()
    # persistent per-column scalars
    P_s = dict(mccont=np.zeros(N, int), lmcmin=np.zeros(N, int), lmcmax=np.zeros(N, int), lmax=np.zeros(N, int), fmc1=np.zeros(N),
               cldslwij=np.zeros(N), clddepij=np.zeros(N), prcpmc=np.zeros(N), ierr=np.zeros(N, int), lerr=np.zeros(N, int),
               prheat=np.zeros(N))
    P_a = dict(sm=sm, qm=qm, sm1=sm1, qm1=qm1, smom=smom, qmom=qmom, um=um, vm=vm, tpsav=tpsav, vlat=vlat, condmmr=condmmr, u00l=u00l,
               wcu2=wcu2, taumcl=taumcl, condpt=condpt, svwmxl=svwmxl, svlatl=svlatl, lhp=lhp, precnvl=precnvl, cldmcl=cldmcl,
               mcflx=mcflx, dgdsm=dgdsm, ddmflx=ddmflx, tdnl=tdnl, qdnl=qdnl, vsubl=vsubl)
    IN_NAMES = ("pl ple plk airm byam etal tl tvl qcll qcil wturb gzl smold qmold smomold qmomold u_0 v_0 ra sdl").split()
    IN = dict(pl=pl, ple=ple, plk=plk, airm=airm, byam=byam, etal=etal, tl=tl, tvl=tvl, qcll=qcll, qcil=qcil, wturb=wturb, gzl=gzl,
              smold=smold, qmold=qmold, smomold=smomold, qmomold=qmomold, u_0=u_0, v_0=v_0, ra=ra, sdl=sdl)

    for lmin in range(1, lmcm):
        # ================================================================== cloud-base test on every column (CLOUDS2 1:377-404)
        fmp0_all = -(10. * 1.0 * sdl[lmin + 1] * BYGRAV * xmass)
        fmp0_all = np.where(fmp0_all <= 0., 0.0, fmp0_all)
        smo1, qmo1, smo2, qmo2 = P_a["sm"][lmin], P_a["qm"][lmin], P_a["sm"][lmin + 1], P_a["qm"][lmin + 1]
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
        lhx0 = F(LHE)
        slh0 = lhx0 * BYSHA
        dmse = (svup - svedg) * plk[lmin + 1] + (svedg - svdn) * plk[lmin] + slh0 * (dq.qsat(sup * plk[lmin + 1], lhx0, pl[lmin + 1]) - qdn)
        sub = np.flatnonzero(~(dmse > -1e-10))
        if not sub.size:
            continue
        g = lambda a: a[sub]  # noqa: E731
        mfo = mf.mass_flux(F(lmin), lhx0, g(qmo1), g(qmo2), g(smo1), g(smo2), slh0, g(wmdn), g(wmup), g(wmedg),
                           g(airm[lmin]), g(airm[lmin + 1]), g(byam[lmin]), g(byam[lmin + 1]), g(byam[lmin + 2]),
                           g(P_a["sm"][lmin + 2]), g(P_a["qm"][lmin + 2]), g(plk[lmin]), g(plk[lmin + 1]), g(pl[lmin]), g(pl[lmin + 1]))
        fplume_b, fmp2_b = np.asarray(mfo["fplume"], float), np.asarray(mfo["fmp2"], float)
        ok = ~(fplume_b <= f001)
        sub = sub[ok]
        if not sub.size:
            continue
        fmp0 = fmp0_all[sub]
        fmp2 = fmp2_b[ok] * min(1.0, dtsrc / (tadj * SECONDS_PER_HOUR))
        _event_block(lmin, sub, fmp0, fmp2, P_a, P_s, IN, IN_NAMES, dict(
            dcl=dcl, dwcu=dwcu, pland=pland, contce=(contce1, contce2), u00a=u00a, u00_mc=u00_mc, f001=f001, f95=f95, pgrad=pgrad,
            fddrt=fddrt, mc_entr_lim=mc_entr_lim, mc_revp=mc_revp, dtsrc=dtsrc, bydtsrc=bydtsrc, lmcm=lmcm, N=N))

    # ====================================================================== after the cloud-base loop (CLOUDS2 1115-1132)
    sm, qm = P_a["sm"], P_a["qm"]
    lmcmin, lmcmax, fmc1 = P_s["lmcmin"], P_s["lmcmax"], P_s["fmc1"]
    conv = lmcmin > 0
    for L in range(1, LM + 1):
        fssl[L] = np.where(conv & (L <= lmcmax), 1 - fmc1, fssl[L])
    sumaj, sumdp = np.zeros(N), np.zeros(N)
    for L in range(1, LM + 1):
        m = conv & (L >= lmcmin) & (L <= lmcmax)
        sumdp = np.where(m, sumdp + airm[L] * fmc1, sumdp)
        sumaj = np.where(m, sumaj + P_a["dgdsm"][L], sumaj)
    for L in range(1, LM + 1):
        m = conv & (L >= lmcmin) & (L <= lmcmax)
        P_a["dgdsm"][L] = np.where(m, P_a["dgdsm"][L] - sumaj * airm[L] * fmc1 / sumdp, P_a["dgdsm"][L])
        sm[L] = np.where(m, sm[L] - sumaj * airm[L] / (sumdp * plk[L]), sm[L])
    airxl = np.zeros(N)
    for L in range(1, LM + 1):
        m = conv & (L >= lmcmin) & (L <= lmcmax)
        airxl = np.where(m, airxl + P_a["mcflx"][L], airxl)

    # ====================================================================== optical thickness (CLOUDS2 1134-1195)
    taumcl, svwmxl, svlatl, cldmcl = P_a["taumcl"], P_a["svwmxl"], P_a["svlatl"], P_a["cldmcl"]
    condpt, tpsav, lhp, condmmr, u00l = P_a["condpt"], P_a["tpsav"], P_a["lhp"], P_a["condmmr"], P_a["u00l"]
    wconst = WMU * (1. - pearth) + WMUL * pearth
    wconst = wmu_mult * wconst
    wmsum, wmctwp, wmclwp = np.zeros(N), np.zeros(N), np.zeros(N)
    qlmc, qimc = zl(), zl()
    for L in range(1, LM + 1):
        m = L <= lmcmax
        if not m.any():
            break
        tl[L] = np.where(m, (sm[L] * byam[L]) * plk[L], tl[L])
        temwm = (taumcl[L] - svwmxl[L] * airm[L]) * 1e2 * BYGRAV
        warm = m & (tl[L] >= TF)
        wmsum = np.where(warm, wmsum + temwm, wmsum)
        wmctwp = np.where(m, wmctwp + temwm, wmctwp)
        wmclwp = np.where(warm, wmclwp + temwm, wmclwp)
        temwm = taumcl[L] - condpt[L] * fmc1 - svwmxl[L] * airm[L]
        qlmc[L] = np.where(m & (svlatl[L] == LHE), temwm / airm[L] + svwmxl[L], qlmc[L])
        qimc[L] = np.where(m & (svlatl[L] == LHS), temwm / airm[L] + svwmxl[L], qimc[L])
        qlmc[L] = np.where(m & (lhp[L] == LHE), qlmc[L] + condpt[L] * fmc1 / airm[L], qlmc[L])
        qimc[L] = np.where(m & (lhp[L] == LHS), qimc[L] + condpt[L] * fmc1 / airm[L], qimc[L])
        cp = m & (cldmcl[L] > 0.)
        dpl = ple[lmcmin, cols_all] - ple[lmcmax + 1, cols_all]
        taumcl[L] = np.where(cp, airm[L] * COETAU, taumcl[L])
        taumcl[L] = np.where(cp & (L == lmcmax) & (dpl < 450), airm[L] * .02, taumcl[L])
        taumcl[L] = np.where(cp & (L <= lmcmin) & (dpl >= 450), airm[L] * .02, taumcl[L])
        svlat1[L] = np.where(m, svlatl[L], svlat1[L])
        z0 = m & (svlatl[L] == 0.)
        svlatl[L] = np.where(z0, np.where(((tpsav[L] > 0.) & (tpsav[L] < TF)) | ((tpsav[L] == 0.) & (tl[L] < TF)), LHS, LHE), svlatl[L])
        an = m & (svwmxl[L] > 0.)
        if an.any():
            ia = np.flatnonzero(an)
            fcld = cldmcl[L][ia] + F(f4(1.0e-20))
            tem = 1e5 * svwmxl[L][ia] * airm[L][ia] * BYGRAV
            wtem = 1e5 * svwmxl[L][ia] * pl[L][ia] / (fcld * tl[L][ia] * RGAS)
            wc = wconst[ia]
            wtem = np.where((svlatl[L][ia] == LHE) & (svwmxl[L][ia] / fcld >= wc * 1e-3), 1e2 * wc * pl[L][ia] / (tl[L][ia] * RGAS), wtem)
            wtem = np.where(wtem < 1e-10, F(1e-10), wtem)
            mndo = 59.68 / (rwcldox * rwcldox * rwcldox)
            mndl, mndi = F(174.0), F(0.06417127)
            mcdncw = mndo * (1. - pearth[ia]) + mndl * pearth[ia]
            mcdnci = mndi
            rcld, tau = mc._ANVIL(svlatl[L][ia], rcldlx, rcldix, mcdncw, mcdnci, rimax, bybr, fcld, tem, wtem)
            taumcl[L][ia] = np.asarray(tau, float)
            csizel[L][ia] = np.asarray(rcld, float) / bybr
        taumcl[L] = np.where(m & (taumcl[L] < 0.) & (cldmcl[L] <= 0.), 0., taumcl[L])
    low = lmcmax <= 1
    for L in range(1, LM + 1):
        u00l[L] = np.where(low & (pl[L] < pl_dcl), 0., u00l[L])
    cnvmmrl = np.where(cldmcl > 0.0, condmmr, 0.0)
    cnvmmrl[0] = 0.0
    cnvmmrl[LM + 1] = 0.0

    o = {k: P_s[k].copy() for k in ("ierr", "lerr", "lmcmin", "lmcmax", "prcpmc", "cldslwij", "clddepij", "prheat", "fmc1", "mccont")}
    o["airxl"] = airxl
    o.update(wmsum=wmsum, wmctwp=wmctwp, wmclwp=wmclwp)
    named = dict(tl=tl, sm=sm, qm=qm, fssl=fssl, cldmcl=cldmcl, taumcl=taumcl, svlatl=svlatl, svlat1=svlat1, svwmxl=svwmxl, csizel=csizel,
                 condpt=condpt, vsubl=P_a["vsubl"], tpsav=tpsav, mcflx=P_a["mcflx"], dgdsm=P_a["dgdsm"], ddmflx=P_a["ddmflx"],
                 tdnl=P_a["tdnl"], qdnl=P_a["qdnl"], u00l=u00l, qlmc=qlmc, qimc=qimc, cnvmmrl=cnvmmrl, condmmr=condmmr)
    for k, a in named.items():
        o[k] = np.ascontiguousarray(a[1:LM + 1].T)
    o["lhp"] = np.ascontiguousarray(lhp[1:LM + 2].T)
    o["precnvl"] = np.ascontiguousarray(P_a["precnvl"][1:LM + 2].T)
    o["smom"] = np.ascontiguousarray(P_a["smom"][:, 1:LM + 1].transpose(2, 0, 1))
    o["qmom"] = np.ascontiguousarray(P_a["qmom"][:, 1:LM + 1].transpose(2, 0, 1))
    o["um"] = np.ascontiguousarray(P_a["um"][:, 1:LM + 1].transpose(2, 0, 1))
    o["vm"] = np.ascontiguousarray(P_a["vm"][:, 1:LM + 1].transpose(2, 0, 1))
    return o


# ---------------------------------------------------------------------------------------------- one cloud-base level
def _event_block(lmin, sub, fmp0, fmp2, P_a, P_s, IN, IN_NAMES, K):
    """Everything below the base test/MASS_FLUX of one LMIN for the columns `sub` (CLOUDS2 1:411-1113 of the per-column port)."""
    n = sub.size
    cols = np.arange(n)
    S = {k: np.take(v, sub, axis=-1) for k, v in P_a.items()}
    Sc = {k: v[sub].copy() for k, v in P_s.items()}
    I = {k: np.take(IN[k], sub, axis=-1) for k in IN_NAMES}
    dcl = K["dcl"][sub]
    dwcu = K["dwcu"][sub]
    pland = K["pland"][sub]
    dtsrc, bydtsrc = K["dtsrc"], K["bydtsrc"]
    u00a, u00_mc, f001, f95, pgrad, fddrt = K["u00a"], K["u00_mc"], K["f001"], K["f95"], K["pgrad"], K["fddrt"]
    mc_entr_lim, mc_revp = K["mc_entr_mass_lim_plume"] if "mc_entr_mass_lim_plume" in K else K["mc_entr_lim"], K["mc_revp"]
    pl, ple, plk, airm, byam, etal = I["pl"], I["ple"], I["plk"], I["airm"], I["byam"], I["etal"]
    tl, tvl, qcll, qcil, wturb, gzl = I["tl"], I["tvl"], I["qcll"], I["qcil"], I["wturb"], I["gzl"]
    smold, qmold, smomold, qmomold, u_0, v_0, ra = I["smold"], I["qmold"], I["smomold"], I["qmomold"], I["u_0"], I["v_0"], I["ra"]
    sm, qm, sm1, qm1, smom, qmom, um, vm = S["sm"], S["qm"], S["sm1"], S["qm1"], S["smom"], S["qmom"], S["um"], S["vm"]
    tpsav, vlat, condmmr, u00l, wcu2 = S["tpsav"], S["vlat"], S["condmmr"], S["u00l"], S["wcu2"]
    taumcl, condpt, svwmxl, svlatl, lhp, precnvl, cldmcl = S["taumcl"], S["condpt"], S["svwmxl"], S["svlatl"], S["lhp"], S["precnvl"], S["cldmcl"]
    mcflx, dgdsm, ddmflx, tdnl, qdnl, vsubl = S["mcflx"], S["dgdsm"], S["ddmflx"], S["tdnl"], S["qdnl"], S["vsubl"]
    mccont, lmcmin, lmcmax, lmax, fmc1 = Sc["mccont"], Sc["lmcmin"], Sc["lmcmax"], Sc["lmax"], Sc["fmc1"]
    cldslwij, clddepij, prcpmc, ierr, lerr, prheat = Sc["cldslwij"], Sc["clddepij"], Sc["prcpmc"], Sc["ierr"], Sc["lerr"], Sc["prheat"]
    z2 = lambda: np.zeros((LM + 2, n))  # noqa: E731
    zm = lambda: np.zeros((NMOM, LM + 2, n))  # noqa: E731
    zk = lambda: np.zeros((NK, LM + 2, n))  # noqa: E731
    cond, cdheat, condp, condp1, condgp = z2(), z2(), z2(), z2(), z2()
    condip, condv, heat1, dm, dmr, ddr = z2(), z2(), z2(), z2(), z2(), z2()
    ccm, ddm, taumc1, ent, det, buoy = z2(), z2(), z2(), z2(), z2(), z2()
    wcu, smdnl, qmdnl = z2(), z2(), z2()
    smomdnl, qmomdnl = zm(), zm()
    umdnl, vmdnl, dum, dvm = zk(), zk(), zk(), zk()
    dsm, dsmom, dsmr, dsmomr = z2(), zm(), z2(), zm()
    dqm, dqmom, dqmr, dqmomr = z2(), zm(), z2(), zm()
    work = [cond, cdheat, condp, condp1, condgp, condip, condv, heat1, dm, dmr, ddr, ccm, ddm, taumc1, ent, det, buoy, wcu, smdnl, qmdnl,
            smomdnl, qmomdnl, umdnl, vmdnl, dum, dvm, dsm, dsmom, dsmr, dsmomr, dqm, dqmom, dqmr, dqmomr]
    # per-column iteration variables (persist across IC / NPPL iterations like the Fortran locals)
    mplume, fplume, smp, qmp, ddraft, etadn = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    ldraft, lfrz = np.full(n, LM), np.zeros(n, int)
    cdhsum, cdhsum1, cdhdrt, evpsum = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    mpmax, smpmax, qmpmax = np.zeros(n), np.zeros(n), np.zeros(n)
    smomp, qmomp, smompmax, qmompmax = np.zeros((NMOM, n)), np.zeros((NMOM, n)), np.zeros((NMOM, n)), np.zeros((NMOM, n))
    ump, vmp = np.zeros((NK, n)), np.zeros((NK, n))
    mc1 = np.zeros(n, bool)
    ldmin, llmin = np.zeros(n, int), np.zeros(n, int)
    fctype = np.zeros(n)
    wb = lambda m, x, y: np.where(m, x, y)  # noqa: E731

    for ic in (1, 2):
        mc1 = np.zeros(n, bool)
        mplume = pymin(pymin(airm[lmin], airm[lmin + 1]), fmp2)
        fctype = np.where(mplume > fmp0, fmp0 / mplume, 1.0)
        if ic == 2:
            fctype = 1. - fctype
        live = ~(fctype < f001)
        if not live.any():
            continue
        mplum1 = mplume.copy()
        cyc = np.zeros(n, bool)
        contce = K["contce"][0] if ic == 1 else K["contce"][1]
        for nppl in (1, 2):
            run = live & ~cyc
            if nppl == 2:
                run &= (mc1 & (mccont >= 2))
            if not run.any():
                continue
            mplume = wb(run, mplum1 * fctype, mplume)
            c1 = run & ((mccont == 0) | mc1)
            fsub_tmp = np.where(mccont == 0, 1.0 + (airm[lmin + 1] - 100.0) / 200.0, 1.0 + (pl[lmin] - pl[lmax, cols] - 100.0) / 200.0)
            fconv_tmp = pymin(mplum1 * byam[lmin + 1] * (1800.0 / dtsrc), 1.0)
            t2 = 1.0 / (fconv_tmp + 1.0e-20) - 1.0
            fsub_tmp = np.where(fsub_tmp > t2, t2, fsub_tmp)
            fsub_tmp = pymax(1.0, pymin(fsub_tmp, 5.0))
            fssl_tmp = 1.0 - (1.0 + fsub_tmp) * fconv_tmp
            fssl_tmp = pymax(CLDMIN, pymin(fssl_tmp, 1.0 - CLDMIN))
            fmc1 = wb(c1, (1.0 - fssl_tmp) + TEENY, fmc1)
            m2 = run & (mc1 | (mccont > 0))
            mplume = wb(m2, pymin(0.95 * airm[lmin] * fmc1, mplume), mplume)
            for w in work:
                w[..., run] = 0.0
            umdn, vmdn = np.zeros((NK, n)), np.zeros((NK, n))
            mplume = wb(run, pymin(mplume / fmc1, airm[lmin] * 0.95 * qm[lmin] / (qmold[lmin] + TEENY)), mplume)
            small = run & (mplume <= f001 * airm[lmin])
            cyc |= small
            r2 = run & ~small
            if not r2.any():
                continue
            # ---- plume set-up
            fplume = wb(r2, mplume * byam[lmin], fplume)
            smp = wb(r2, smold[lmin] * fplume, smp)
            smomp[:, r2] = 0.0
            qmomp[:, r2] = 0.0
            smomp[XYMa[:, None], np.flatnonzero(r2)[None, :]] = smomold[XYMa, lmin][:, r2] * fplume[r2]
            qmp = wb(r2, qmold[lmin] * fplume, qmp)
            qmomp[XYMa[:, None], np.flatnonzero(r2)[None, :]] = qmomold[XYMa, lmin][:, r2] * fplume[r2]
            tpsav[lmin] = wb(r2 & (tpsav[lmin] == 0), smp * plk[lmin] / mplume, tpsav[lmin])
            dmr[lmin] = wb(r2, -mplume, dmr[lmin])
            dsmr[lmin] = wb(r2, -smp, dsmr[lmin])
            for mm in XYM:
                dsmomr[mm, lmin] = wb(r2, -smomp[mm], dsmomr[mm, lmin])
                dqmomr[mm, lmin] = wb(r2, -qmomp[mm], dqmomr[mm, lmin])
            for mm in ZM:
                dsmomr[mm, lmin] = wb(r2, -smomold[mm, lmin] * fplume, dsmomr[mm, lmin])
                dqmomr[mm, lmin] = wb(r2, -qmomold[mm, lmin] * fplume, dqmomr[mm, lmin])
            dqmr[lmin] = wb(r2, -qmp, dqmr[lmin])
            ump = wb(r2[None, :], um[:, lmin] * fplume, ump)
            dum[:, lmin] = wb(r2[None, :], -ump, dum[:, lmin])
            vmp = wb(r2[None, :], vm[:, lmin] * fplume, vmp)
            dvm[:, lmin] = wb(r2[None, :], -vmp, dvm[:, lmin])
            cdhsum, cdhsum1, cdhdrt = wb(r2, 0.0, cdhsum), wb(r2, 0.0, cdhsum1), wb(r2, 0.0, cdhdrt)
            etadn = wb(r2, 0.0, etadn)
            ldraft = wb(r2, LM, ldraft)
            evpsum = wb(r2, 0.0, evpsum)
            ddraft = wb(r2, 0.0, ddraft)
            lfrz = wb(r2, 0, lfrz)
            lmax = wb(r2, lmin, lmax)
            wc0 = pymax(.5, wturb[lmin + 1])
            if ic == 1:
                wc0 = pymax(.5, 2.0 * wturb[lmin + 1])
            wcu[lmin] = wb(r2, wc0, wcu[lmin])
            wcu2[lmin] = wb(r2, wcu[lmin] * wcu[lmin], wcu2[lmin])
            mpmax, smpmax, qmpmax = wb(r2, 0.0, mpmax), wb(r2, 0.0, smpmax), wb(r2, 0.0, qmpmax)
            smompmax[:, r2] = 0.0
            qmompmax[:, r2] = 0.0
            # ---- plume ascent
            a = r2.copy()
            for L in range(lmin + 1, LM + 1):
                if not a.any():
                    break
                a &= ~(mplume <= f001 * airm[L])
                sdn = smp / mplume
                sup = sm1[L] * byam[L]
                qdn = qmp / mplume
                qup = qm1[L] * byam[L]
                wmdn = 0.0
                wmup = qcll[L] + qcil[L]
                svdn = sdn * (1. + DELTX * qdn - wmdn)
                svup = sup * (1. + DELTX * qup - wmup)
                a &= ~(plk[L - 1] * (svup - svdn) + SLHE * (qup - qdn) >= 0.)
                tpsav[L] = wb(a & (tpsav[L] == 0), smp * plk[L] / mplume, tpsav[L])
                tp = tpsav[L]
                lfrz = wb(a & (tpsav[L - 1] >= TF) & (tpsav[L] < TF), L - 1, lfrz)
                lhx = np.where(tp < TI, LHS, LHE)
                qsatmp = mplume * dq.qsat(tp, lhx, pl[L])
                a &= ~(qmp < qsatmp)
                ms_ = a & (tp < TF) & (lhx == LHE)
                lhx = np.where(ms_, LHS, lhx)
                qsatmp = np.where(ms_, mplume * dq.qsat(tp, lhx, pl[L]), qsatmp)
                lhx = np.where(a & (vlat[L] == LHS), LHS, lhx)
                vlat[L] = wb(a, lhx, vlat[L])
                slh = lhx * BYSHA
                u00l[L] = wb(a & (tl[L] >= TF) & (u00l[L] != u00a), u00_mc, u00l[L])
                mccont = np.where(a, mccont + 1, mccont)
                mc1 = mc1 | (a & (mccont == 1))
                cap = a & (mplume > f95 * airm[L])
                if cap.any():
                    delta = (mplume - f95 * airm[L]) / mplume
                    dm[L - 1] = wb(cap, dm[L - 1] + delta * mplume, dm[L - 1])
                    mplume = wb(cap, f95 * airm[L], mplume)
                    dsm[L - 1] = wb(cap, dsm[L - 1] + delta * smp, dsm[L - 1])
                    smp = wb(cap, smp * (1. - delta), smp)
                    dsmom[XYMa, L - 1] = wb(cap, dsmom[XYMa, L - 1] + delta * smomp[XYMa], dsmom[XYMa, L - 1])
                    smomp[XYMa] = wb(cap, smomp[XYMa] * (1. - delta), smomp[XYMa])
                    dqm[L - 1] = wb(cap, dqm[L - 1] + delta * qmp, dqm[L - 1])
                    qmp = wb(cap, qmp * (1. - delta), qmp)
                    dqmom[XYMa, L - 1] = wb(cap, dqmom[XYMa, L - 1] + delta * qmomp[XYMa], dqmom[XYMa, L - 1])
                    qmomp[XYMa] = wb(cap, qmomp[XYMa] * (1. - delta), qmomp[XYMa])
                    dum[:, L - 1] = wb(cap, dum[:, L - 1] + ump * delta, dum[:, L - 1])
                    dvm[:, L - 1] = wb(cap, dvm[:, L - 1] + vmp * delta, dvm[:, L - 1])
                    ump = wb(cap, ump - ump * delta, ump)
                    vmp = wb(cap, vmp - vmp * delta, vmp)
                wk = mplume * (sup - sdn) * (plk[L - 1] - plk[L]) / plk[L - 1]
                dsm[L - 1] = wb(a, dsm[L - 1] - wk, dsm[L - 1])
                ccm[L - 1] = wb(a, mplume, ccm[L - 1])
                dqsum, fqcond = dq.get_dq_cond(smp, qmp, plk[L], mplume, lhx, pl[L])
                m3 = a & (dqsum > 0.) & (qmp > TEENY)
                qmomp[XYMa] = wb(m3, qmomp[XYMa] * (1. - fqcond), qmomp[XYMa])
                smp = wb(m3, smp + slh * dqsum / plk[L], smp)
                qmp = wb(m3, qmp - dqsum, qmp)
                cond[L] = wb(a, dqsum, cond[L])
                condmmr[L] = wb(a, condmmr[L] + cond[L] * byam[L] * fmc1, condmmr[L])
                cdheat[L] = wb(a, slh * cond[L], cdheat[L])
                cdhsum = wb(a, cdhsum + cdheat[L], cdhsum)
                cond[L] = wb(a, cond[L] + condv[L - 1], cond[L])
                mv = a & (vlat[L - 1] != vlat[L])
                smp = wb(mv, smp - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA / plk[L], smp)
                cdheat[L] = wb(mv, cdheat[L] - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA, cdheat[L])
                cdhsum = wb(mv, cdhsum - (vlat[L - 1] - vlat[L]) * condv[L - 1] * BYSHA, cdhsum)
                ia = np.flatnonzero(a)
                condmu = 100. * cond[L] * pl[L] / (ccm[L - 1] * tl[L] * RGAS)
                flamw = np.zeros(n)
                flamg = np.zeros(n)
                flami = np.zeros(n)
                if ia.size:
                    cm_ = condmu[ia]
                    flamw[ia] = mc._pow(1000.0 * PI * CN0 / (cm_ + TEENY), F(.25))
                    flamg[ia] = mc._pow(400.0 * PI * CN0G / (cm_ + TEENY), F(.25))
                    flami[ia] = mc._pow(100.0 * PI * CN0I / (cm_ + TEENY), F(.25))
                taumc1[L] = wb(a, taumc1[L] + cond[L] * fmc1, taumc1[L])
                tvp = (smp / mplume) * plk[L] * (1. + DELTX * qmp / mplume)
                buoy[L] = wb(a, (tvp - tvl[L]) / tvl[L] - cond[L] / mplume, buoy[L])
                ent[L] = wb(a, .16667 * contce * GRAV * buoy[L] / (wcu[L - 1] * wcu[L - 1] + TEENY), ent[L])
                ng = a & (ent[L] < 0.)
                det[L] = wb(ng, -ent[L], det[L])
                ent[L] = wb(ng, 0., ent[L])
                pe = a & (ent[L] > 0.)
                if pe.any():
                    fentr = 1000. * ent[L] * gzl[L] * fplume
                    cap2 = pe & (fentr + fplume > 1.)
                    fentr = np.where(cap2, 1. - fplume, fentr)
                    ent[L] = wb(cap2, 0.001 * fentr / (gzl[L] * fplume), ent[L])
                    ee = pe & (fentr >= TEENY)
                    if ee.any():
                        mpold, fpold = mplume, fplume
                        etal1 = fentr / (fplume + TEENY)
                        eplume = mplume * etal1
                        cap3 = ee & (eplume > airm[L] * 0.975 - mplume)
                        eplume = np.where(cap3, airm[L] * 0.975 - mplume, eplume)
                        mplume = wb(ee, mplume + eplume, mplume)
                        etal1 = eplume / mpold
                        fentr = np.where(ee, etal1 * fpold, fentr)
                        ent[L] = wb(ee, 0.001 * fentr / (gzl[L] * fpold), ent[L])
                        if mc_entr_lim == 0:
                            fplume = wb(ee, fplume + fentr, fplume)
                        else:
                            fplume = wb(ee, mplume * byam[L], fplume)
                        fentra = eplume * byam[L]
                        dsmr[L] = wb(ee, dsmr[L] - eplume * sup, dsmr[L])
                        dsmomr[:, L] = wb(ee, dsmomr[:, L] - smom[:, L] * fentra, dsmomr[:, L])
                        dqmr[L] = wb(ee, dqmr[L] - eplume * qup, dqmr[L])
                        dqmomr[:, L] = wb(ee, dqmomr[:, L] - qmom[:, L] * fentra, dqmomr[:, L])
                        dmr[L] = wb(ee, dmr[L] - eplume, dmr[L])
                        smp = wb(ee, smp + eplume * sup, smp)
                        smomp[XYMa] = wb(ee, smomp[XYMa] + smom[XYMa, L] * fentra, smomp[XYMa])
                        qmp = wb(ee, qmp + eplume * qup, qmp)
                        qmomp[XYMa] = wb(ee, qmomp[XYMa] + qmom[XYMa, L] * fentra, qmomp[XYMa])
                        umtemp = pgrad * np.power(mplume, 2.0) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                        vmtemp = pgrad * np.power(mplume, 2.0) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                        ump = wb(ee, ump + u_0[:, L] * eplume + umtemp, ump)
                        dum[:, L] = wb(ee, dum[:, L] - u_0[:, L] * eplume - umtemp, dum[:, L])
                        vmp = wb(ee, vmp + v_0[:, L] * eplume + vmtemp, vmp)
                        dvm[:, L] = wb(ee, dvm[:, L] - v_0[:, L] * eplume - vmtemp, dvm[:, L])
                dd = a & (det[L] > 0.)
                if dd.any():
                    delta = 1000. * det[L] * gzl[L]
                    cap4 = dd & (delta > .95)
                    delta = np.where(cap4, 0.95, delta)
                    det[L] = wb(cap4, .001 * delta / gzl[L], det[L])
                    dm[L] = wb(dd, dm[L] + delta * mplume, dm[L])
                    mplume = wb(dd, mplume * (1. - delta), mplume)
                    dsm[L] = wb(dd, dsm[L] + delta * smp, dsm[L])
                    smp = wb(dd, smp * (1. - delta), smp)
                    dsmom[XYMa, L] = wb(dd, dsmom[XYMa, L] + delta * smomp[XYMa], dsmom[XYMa, L])
                    smomp[XYMa] = wb(dd, smomp[XYMa] * (1. - delta), smomp[XYMa])
                    dqm[L] = wb(dd, dqm[L] + delta * qmp, dqm[L])
                    qmp = wb(dd, qmp * (1. - delta), qmp)
                    dqmom[XYMa, L] = wb(dd, dqmom[XYMa, L] + delta * qmomp[XYMa], dqmom[XYMa, L])
                    qmomp[XYMa] = wb(dd, qmomp[XYMa] * (1. - delta), qmomp[XYMa])
                    umtemp = pgrad * np.power(mplume, 2.0) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                    vmtemp = pgrad * np.power(mplume, 2.0) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                    dum[:, L] = wb(dd, dum[:, L] + ump * delta - umtemp, dum[:, L])
                    dvm[:, L] = wb(dd, dvm[:, L] + vmp * delta - vmtemp, dvm[:, L])
                    ump = wb(dd, ump - ump * delta + umtemp, ump)
                    vmp = wb(dd, vmp - vmp * delta + vmtemp, vmp)
                if L - lmin > 1:
                    smix = .5 * (sup + smp / mplume)
                    qmix = .5 * (qup + qmp / mplume)
                    wmix = .5 * (wmup + cond[L] / mplume)
                    svmix = smix * (1. + DELTX * qmix - wmix)
                    svup = sup * (1. + DELTX * qup - wmup)
                    dmmix = (svup - svmix) * plk[L] + SLHE * (dq.qsat(sup * plk[L], lhx, pl[L]) - qmix)
                    cdhdrt = wb(a & (dmmix < 1e-10), cdhdrt + cdheat[L], cdhdrt)
                    tr = a & (dmmix >= 1e-10)
                    if tr.any():
                        ldraft = wb(tr, L, ldraft)
                        etad = F(1.0) / F(3.0)
                        etadn = wb(tr, etad, etadn)
                        fleft = 1. - .5 * etad
                        ddraft = wb(tr, etad * mplume, ddraft)
                        ddr[L] = wb(tr, ddraft, ddr[L])
                        cdhsum1 = wb(tr, cdhsum1 + cdhdrt * .5 * etad, cdhsum1)
                        cdhdrt = wb(tr, cdhdrt - cdhdrt * .5 * etad + cdheat[L], cdhdrt)
                        fddp = .5 * ddraft
                        fddp = fddp / mplume
                        fddl = .5 * ddraft * byam[L]
                        mplume = wb(tr, fleft * mplume, mplume)
                        smdnl[L] = wb(tr, ddraft * smix, smdnl[L])
                        smomdnl[XYMa, L] = wb(tr, smom[XYMa, L] * fddl + smomp[XYMa] * fddp, smomdnl[XYMa, L])
                        smp = wb(tr, fleft * smp, smp)
                        smomp[XYMa] = wb(tr, smomp[XYMa] * fleft, smomp[XYMa])
                        qmdnl[L] = wb(tr, ddraft * qmix, qmdnl[L])
                        qmomdnl[XYMa, L] = wb(tr, qmom[XYMa, L] * fddl + qmomp[XYMa] * fddp, qmomdnl[XYMa, L])
                        qmp = wb(tr, fleft * qmp, qmp)
                        qmomp[XYMa] = wb(tr, qmomp[XYMa] * fleft, qmomp[XYMa])
                        dmr[L] = wb(tr, dmr[L] - .5 * ddraft, dmr[L])
                        dsmr[L] = wb(tr, dsmr[L] - .5 * ddraft * sup, dsmr[L])
                        dsmomr[:, L] = wb(tr, dsmomr[:, L] - smom[:, L] * fddl, dsmomr[:, L])
                        dqmr[L] = wb(tr, dqmr[L] - .5 * ddraft * qup, dqmr[L])
                        dqmomr[:, L] = wb(tr, dqmomr[:, L] - qmom[:, L] * fddl, dqmomr[:, L])
                        umdnl[:, L] = wb(tr, .5 * (etad * ump + ddraft * u_0[:, L]), umdnl[:, L])
                        ump = wb(tr, ump * fleft, ump)
                        dum[:, L] = wb(tr, dum[:, L] - .5 * ddraft * u_0[:, L], dum[:, L])
                        vmdnl[:, L] = wb(tr, .5 * (etad * vmp + ddraft * v_0[:, L]), vmdnl[:, L])
                        vmp = wb(tr, vmp * fleft, vmp)
                        dvm[:, L] = wb(tr, dvm[:, L] - .5 * ddraft * v_0[:, L], dvm[:, L])
                w2tem = .16667 * GRAV * buoy[L] - wcu[L - 1] * wcu[L - 1] * (.66667 * det[L] + ent[L])
                hdep = airm[L] * tl[L] * RGAS / (GRAV * pl[L])
                wcu2[L] = wb(a, wcu2[L - 1] + 2. * hdep * w2tem, wcu2[L])
                wl = np.where(wcu2[L] > 0., np.sqrt(np.where(wcu2[L] > 0., wcu2[L], 1.0)), 0.)
                wl = np.where(wl >= 0., pymin(50., wl), wl)
                wl = np.where(wl < 0., pymax(-50., wl), wl)
                wcu[L] = wb(a, wl, wcu[L])
                smpmax = wb(a, smp, smpmax)
                smompmax = wb(a, smomp, smompmax)
                qmpmax = wb(a, qmp, qmpmax)
                qmompmax = wb(a, qmomp, qmompmax)
                mpmax = wb(a, mplume, mpmax)
                lmax = np.where(a, lmax + 1, lmax)
                a &= ~(wcu2[L] < 0.)
                ia = np.flatnonzero(a)
                if ia.size:
                    wcufrz = np.where(lfrz > 0, wcu[lfrz, cols], 0.0)
                    gi = lambda x: x[ia]  # noqa: E731
                    cp, cp1, cip, cgp = conv_micro(gi(pl[L]), gi(wcu[L]), gi(dwcu), gi(lfrz).astype(float), gi(wcufrz), gi(tp), gi(pland),
                                                   gi(flamw), gi(flamg), gi(flami), gi(I["tl"][lmin]), gi(I["tl"][lmin + 1]),
                                                   gi(condip[L]), gi(condgp[L]))
                    condp[L, ia], condp1[L, ia], condip[L, ia], condgp[L, ia] = cp, cp1, cip, cgp
                    condp[L] = wb(a, .01 * condp[L] * ccm[L - 1] * tl[L] * RGAS / pl[L], condp[L])
                    condp1[L] = wb(a, .01 * condp1[L] * ccm[L - 1] * tl[L] * RGAS / pl[L], condp1[L])
                    condp1[L] = wb(a & (condp1[L] > cond[L]), cond[L], condp1[L])
                    condp[L] = wb(a & (condp[L] > condp1[L]), condp1[L], condp[L])
                    condv[L] = wb(a, cond[L] - condp1[L], condv[L])
                    cond[L] = wb(a, cond[L] - condv[L], cond[L])
                    taumc1[L] = wb(a, taumc1[L] - condv[L] * fmc1, taumc1[L])
            if _DBG:
                _DBG('asc', lmin, ic, nppl, locals())
            cyc |= r2 & (lmax == lmin)
        ev = live & ~cyc
        if not ev.any():
            continue
        # ---------------------------------------------------------------------- after the plume partitions
        for L in range(lmin, LM + 1):
            taumcl[L] = wb(ev & (L <= lmax), taumcl[L] + taumc1[L], taumcl[L])
        mA = pl[lmin] < pl[dcl, cols]
        u00l[lmin] = wb(ev & mA, u00_mc, u00l[lmin])
        for L in range(1, lmin + 1):
            u00l[L] = wb(ev & ~mA, u00_mc, u00l[L])
        lfrz = wb(ev & (tpsav[lmax, cols] >= TF), lmax, lfrz)
        ei = np.flatnonzero(ev)
        lme = lmax[ei]
        u00l[lme, ei] = u00a
        dm[lme, ei] = dm[lme, ei] + mpmax[ei]
        dsm[lme, ei] = dsm[lme, ei] + smpmax[ei]
        _ms(dsmom, XYM, lme, ei, _mg(dsmom, XYM, lme, ei) + smompmax[XYMa][:, ei])
        dqm[lme, ei] = dqm[lme, ei] + qmpmax[ei]
        _ms(dqmom, XYM, lme, ei, _mg(dqmom, XYM, lme, ei) + qmompmax[XYMa][:, ei])
        ccm[lme, ei] = 0.
        _ms(dum, range(NK), lme, ei, _mg(dum, range(NK), lme, ei) + ump[:, ei])
        _ms(dvm, range(NK), lme, ei, _mg(dvm, range(NK), lme, ei) + vmp[:, ei])
        lmcmin = np.where(ev & (lmcmin == 0), lmin, lmcmin)
        lmcmax = np.where(ev & (lmcmax < lmax), lmax, lmcmax)
        if _DBG:
            _DBG('pp', lmin, ic, 0, locals())
        # ---------------------------------------------------------------------- downdraft
        ldmin = wb(ev, ldraft - 1, ldmin)
        llmin = wb(ev, ldmin, llmin)
        edraft = np.zeros(n)
        smdn, qmdn, ddrold = np.zeros(n), np.zeros(n), np.zeros(n)
        ddm_ev = ev & (etadn > 1e-10)
        if ddm_ev.any():
            di = np.flatnonzero(ddm_ev)
            ld = ldraft[di]
            ddraft[di] = ddr[ld, di]
            ddrold[di] = ddraft[di]
            smdn[di] = smdnl[ld, di]
            qmdn[di] = qmdnl[ld, di]
            smomdn, qmomdn = np.zeros((NMOM, n)), np.zeros((NMOM, n))
            smomdn[XYMa[:, None], di[None, :]] = _mg(smomdnl, XYM, ld, di)
            qmomdn[XYMa[:, None], di[None, :]] = _mg(qmomdnl, XYM, ld, di)
            umdn[:, di] = _mg(umdnl, range(NK), ld, di)
            vmdn[:, di] = _mg(vmdnl, range(NK), ld, di)
            alive_dd = ddm_ev.copy()
            for L in range(int(ldraft[di].max()), 0, -1):
                dact = alive_dd & (L <= ldraft)
                if not alive_dd.any():
                    break
                if not dact.any():
                    continue
                lhx = vlat[L]
                slh = lhx * BYSHA
                dqe, fq1 = dq.get_dq_evap(smdn, qmdn, plk[L], ddraft, lhx, pl[L], cond[L])
                dqsum = dqe
                dqevp = fddrt * cond[L]
                dqevp = np.where(dqevp > dqsum, dqsum, dqevp)
                dqevp = np.where(dqevp > smdn * plk[L] / slh, smdn * plk[L] / slh, dqevp)
                if L < lmin:
                    dqevp = np.zeros(n)
                fsevp = np.where(plk[L] * smdn > TEENY, slh * dqevp / (plk[L] * smdn), 0.0)
                smdn = wb(dact, smdn - slh * dqevp / plk[L], smdn)
                smomdn[XYMa] = wb(dact, smomdn[XYMa] * (1. - fsevp), smomdn[XYMa])
                qmdn = wb(dact, qmdn + dqevp, qmdn)
                cond[L] = wb(dact, cond[L] - dqevp, cond[L])
                taumcl[L] = wb(dact, taumcl[L] - dqevp * fmc1, taumcl[L])
                cdheat[L] = wb(dact, cdheat[L] - dqevp * slh, cdheat[L])
                evpsum = wb(dact, evpsum + dqevp * slh, evpsum)
                if 1 < L:
                    mu = dact & (L < ldraft)
                    if mu.any():
                        ddrup = ddraft.copy()
                        ddr_new = ddraft + ddrold * etal[L]
                        ddr_new = np.where(ddrup > ddr_new, ddrup, ddr_new)
                        smix = smdn / (ddrup + TEENY)
                        qmix = qmdn / (ddrup + TEENY)
                        wmix = cond[L] / (ddrup + TEENY)
                        svmix = smix * plk[L - 1] * (1. + DELTX * qmix - wmix)
                        svm1 = sm1[L - 1] * byam[L - 1] * plk[L - 1] * (1. + DELTX * qm1[L - 1] * byam[L - 1] - qcll[L - 1] - qcil[L - 1])
                        ddr_new = np.where(svmix - svm1 >= DTMIN1, FDDET * ddrup, ddr_new)
                        ddr_new = np.where(ddr_new > .95 * (airm[L - 1] + dmr[L - 1]), .95 * (airm[L - 1] + dmr[L - 1]), ddr_new)
                        ddraft = wb(mu, ddr_new, ddraft)
                        edraft = wb(mu, ddraft - ddrup, edraft)
                        e1 = mu & (edraft > 0)
                        e2 = mu & ~(edraft > 0)
                        if e1.any():
                            fentra = edraft * byam[L]
                            senv = sm[L] * byam[L]
                            qenv = qm[L] * byam[L]
                            smdn = wb(e1, smdn + edraft * senv, smdn)
                            qmdn = wb(e1, qmdn + edraft * qenv, qmdn)
                            smomdn[XYMa] = wb(e1, smomdn[XYMa] + smom[XYMa, L] * fentra, smomdn[XYMa])
                            qmomdn[XYMa] = wb(e1, qmomdn[XYMa] + qmom[XYMa, L] * fentra, qmomdn[XYMa])
                            dsmr[L] = wb(e1, dsmr[L] - edraft * senv, dsmr[L])
                            dsmomr[:, L] = wb(e1, dsmomr[:, L] - smom[:, L] * fentra, dsmomr[:, L])
                            dqmr[L] = wb(e1, dqmr[L] - edraft * qenv, dqmr[L])
                            dqmomr[:, L] = wb(e1, dqmomr[:, L] - qmom[:, L] * fentra, dqmomr[:, L])
                            dmr[L] = wb(e1, dmr[L] - edraft, dmr[L])
                            umtemp = pgrad * np.power(ddraft, 2.0) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                            vmtemp = pgrad * np.power(ddraft, 2.0) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                            umdn = wb(e1, umdn + fentra * um[:, L] - umtemp, umdn)
                            vmdn = wb(e1, vmdn + fentra * vm[:, L] - vmtemp, vmdn)
                            dum[:, L] = wb(e1, dum[:, L] - fentra * um[:, L] + umtemp, dum[:, L])
                            dvm[:, L] = wb(e1, dvm[:, L] - fentra * vm[:, L] + vmtemp, dvm[:, L])
                        if e2.any():
                            fentra = edraft / (ddrup + TEENY)
                            dsm[L] = wb(e2, dsm[L] - fentra * smdn, dsm[L])
                            dsmom[XYMa, L] = wb(e2, dsmom[XYMa, L] - smomdn[XYMa] * fentra, dsmom[XYMa, L])
                            dqm[L] = wb(e2, dqm[L] - fentra * qmdn, dqm[L])
                            dqmom[XYMa, L] = wb(e2, dqmom[XYMa, L] - qmomdn[XYMa] * fentra, dqmom[XYMa, L])
                            smdn = wb(e2, smdn * (1 + fentra), smdn)
                            qmdn = wb(e2, qmdn * (1 + fentra), qmdn)
                            smomdn[XYMa] = wb(e2, smomdn[XYMa] * (1 + fentra), smomdn[XYMa])
                            qmomdn[XYMa] = wb(e2, qmomdn[XYMa] * (1 + fentra), qmomdn[XYMa])
                            dm[L] = wb(e2, dm[L] - edraft, dm[L])
                            umtemp = pgrad * np.power(ddraft, 2.0) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
                            vmtemp = pgrad * np.power(ddraft, 2.0) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
                            dum[:, L] = wb(e2, dum[:, L] - fentra * umdn + umtemp, dum[:, L])
                            dvm[:, L] = wb(e2, dvm[:, L] - fentra * vmdn + vmtemp, dvm[:, L])
                            umdn = wb(e2, umdn * (1. + fentra) - umtemp, umdn)
                            vmdn = wb(e2, vmdn * (1. + fentra) - vmtemp, vmdn)
                ldmin = wb(dact, L, ldmin)
                llmin = wb(dact, ldmin, llmin)
                if L > 1:
                    smix = smdn / (ddraft + TEENY)
                    qmix = qmdn / (ddraft + TEENY)
                    wmix = cond[L - 1] / (ddraft + TEENY)
                    svmix = smix * plk[L - 1] * (1. + DELTX * qmix - wmix)
                    svm1 = sm1[L - 1] * byam[L - 1] * plk[L - 1] * (1. + DELTX * qm1[L - 1] * byam[L - 1] - qcll[L - 1] - qcil[L - 1])
                    brk = dact & (L <= lmin) & (svmix >= svm1)
                    cont = dact & ~brk
                    ddm[L - 1] = wb(cont, ddraft, ddm[L - 1])
                    ddrold = wb(cont, ddraft, ddrold)
                    ddraft = wb(cont, ddraft + ddr[L - 1], ddraft)
                    smdn = wb(cont, smdn + smdnl[L - 1], smdn)
                    qmdn = wb(cont, qmdn + qmdnl[L - 1], qmdn)
                    smomdn[XYMa] = wb(cont, smomdn[XYMa] + smomdnl[XYMa, L - 1], smomdn[XYMa])
                    qmomdn[XYMa] = wb(cont, qmomdn[XYMa] + qmomdnl[XYMa, L - 1], qmomdn[XYMa])
                    umdn = wb(cont, umdn + umdnl[:, L - 1], umdn)
                    vmdn = wb(cont, vmdn + vmdnl[:, L - 1], vmdn)
                    alive_dd = alive_dd & ~brk
            ld_ = ldmin[di]
            dsm[ld_, di] = dsm[ld_, di] + smdn[di]
            _ms(dsmom, XYM, ld_, di, _mg(dsmom, XYM, ld_, di) + smomdn[XYMa][:, di])
            dqm[ld_, di] = dqm[ld_, di] + qmdn[di]
            _ms(dqmom, XYM, ld_, di, _mg(dqmom, XYM, ld_, di) + qmomdn[XYMa][:, di])
            tdnl[ld_, di] = smdn[di] * plk[ld_, di] / (ddraft[di] + TEENY)
            qdnl[ld_, di] = qmdn[di] / (ddraft[di] + TEENY)
            _ms(dum, range(NK), ld_, di, _mg(dum, range(NK), ld_, di) + umdn[:, di])
            _ms(dvm, range(NK), ld_, di, _mg(dvm, range(NK), ld_, di) + vmdn[:, di])
            dm[ld_, di] = dm[ld_, di] + ddraft[di]
        if _DBG:
            _DBG('dd', lmin, ic, 0, locals())
        # ---------------------------------------------------------------------- subsidence
        ldmin = np.where(ev & (ldmin > lmin), lmin, ldmin)
        cm = z2()
        smt, qmt = z2(), z2()
        for L in range(1, LM + 1):
            m = ev & (L >= ldmin) & (L <= lmax)
            cm[L] = wb(m, cm[L - 1] - dm[L] - dmr[L], cm[L])
            smt[L] = wb(m, sm[L], smt[L])
            qmt[L] = wb(m, qm[L], qmt[L])
        for L in range(1, LM + 1):
            cm[L] = wb(ev & (L >= lmax), 0., cm[L])
        ksub = np.ones(n, int)
        for l in range(1, LM):
            m = ev & (l >= ldmin) & (l < lmax)
            c_a = m & (+cm[l] > airm[l + 1] + dmr[l + 1])
            c_b = m & ~c_a & (-cm[l] > airm[l] + dmr[l])
            va = np.where(c_a, (+cm[l] - dmr[l + 1]) / airm[l + 1], 0.0)
            vb = np.where(c_b, (-cm[l] - dmr[l]) / airm[l], 0.0)
            ksub = np.where(c_a, np.maximum(ksub, 1 + va.astype(int)), ksub)
            ksub = np.where(c_b, np.maximum(ksub, 1 + vb.astype(int)), ksub)
        ksub = np.minimum(ksub, 2)
        byksub = 1.0 / ksub
        sumu, sumv = np.zeros((NK, n)), np.zeros((NK, n))
        sumdp = np.zeros(n)
        for L in range(1, LM + 1):
            m = ev & (L >= ldmin) & (L <= lmax)
            sumu = wb(m, sumu + um[:, L], sumu)
            sumv = wb(m, sumv + vm[:, L], sumv)
            sumdp = wb(m, sumdp + airm[L], sumdp)
        alpha = np.zeros(n)
        for L in range(1, LM + 1):
            m = ev & (L >= ldmin) & (L <= lmax)
            cldm = ccm[L]
            cldm = np.where(m & (L < ldraft) & (L >= llmin) & (etadn > 1e-10), ccm[L] - ddm[L], cldm)
            vsubl[L] = wb(m & mc1, 100. * cldm * RGAS * tl[L] / (pl[L] * GRAV * dtsrc), vsubl[L])
            beta = cldm * byam[L + 1]
            beta = np.where(cldm < 0., cldm * byam[L], beta)
            betau = np.where(beta < 0., 0.0, beta)
            alphau = np.where(alpha < 0., 0.0, alpha)
            um[:, L] = wb(m, um[:, L] + ra * (-alphau * um[:, L] + betau * um[:, L + 1] + dum[:, L]), um[:, L])
            vm[:, L] = wb(m, vm[:, L] + ra * (-alphau * vm[:, L] + betau * vm[:, L + 1] + dvm[:, L]), vm[:, L])
            alpha = wb(m, beta, alpha)
        sumu1, sumv1 = np.zeros((NK, n)), np.zeros((NK, n))
        for L in range(1, LM + 1):
            m = ev & (L >= ldmin) & (L <= lmax)
            sumu1 = wb(m, sumu1 + um[:, L], sumu1)
            sumv1 = wb(m, sumv1 + vm[:, L], sumv1)
        for L in range(1, LM + 1):
            m = ev & (L >= ldmin) & (L <= lmax)
            um[:, L] = wb(m, um[:, L] - (sumu1 - sumu) * airm[L] / sumdp, um[:, L])
            vm[:, L] = wb(m, vm[:, L] - (sumv1 - sumv) * airm[L] / sumdp, vm[:, L])
        cmneg = z2()
        for L in range(1, LM + 1):
            cmneg[L] = np.where(ev & (L >= ldmin) & (L < lmax), -cm[L] * byksub, 0.0)
        for it in (1, 2):
            selm = ev & (ksub >= it)
            sel = np.flatnonzero(selm)
            if not sel.size:
                continue
            ml = z2()
            for L in range(1, LM + 1):
                m = selm & (L >= ldmin) & (L <= lmax)
                ml[L] = np.where(m, airm[L] + dmr[L] * byksub, 0.0)
                sm[L] = wb(m, sm[L] + dsmr[L] * byksub, sm[L])
                smom[:, L] = wb(m, smom[:, L] + dsmomr[:, L] * byksub, smom[:, L])
            # advection of SM (no limiter), grouped by slice
            _adv_groups(sel, ldmin, lmax, sm, smom, ml, cmneg, False, ierr, lerr)
            ml2 = z2()
            for L in range(1, LM + 1):
                m = selm & (L >= ldmin) & (L <= lmax)
                sm[L] = wb(m, sm[L] + dsm[L] * byksub, sm[L])
                smom[:, L] = wb(m, smom[:, L] + dsmom[:, L] * byksub, smom[:, L])
                ml2[L] = np.where(m, airm[L] + dmr[L] * byksub, 0.0)
                qm[L] = wb(m, qm[L] + dqmr[L] * byksub, qm[L])
                qmom[:, L] = wb(m, qmom[:, L] + dqmomr[:, L] * byksub, qmom[:, L])
            _adv_groups(sel, ldmin, lmax, qm, qmom, ml2, cmneg, True, ierr, lerr)
            for L in range(1, LM + 1):
                m = selm & (L >= ldmin) & (L <= lmax)
                qm[L] = wb(m, qm[L] + dqm[L] * byksub, qm[L])
                qmom[:, L] = wb(m, qmom[:, L] + dqmom[:, L] * byksub, qmom[:, L])
        if _DBG:
            _DBG('sub', lmin, ic, 0, locals())
        # ---------------------------------------------------------------------- diagnostics that feed exit fields
        for L in range(1, LM + 1):
            m = ev & (L >= ldmin) & (L <= lmax)
            fcdh = np.where(L == lmax, cdhsum - cdhsum1 + 0.0, 0.0)
            fcdh1 = np.where(L == llmin, cdhsum1 - evpsum, 0.0)
            mcflx[L] = wb(m, mcflx[L] + ccm[L] * fmc1, mcflx[L])
            dgdsm[L] = wb(m, dgdsm[L] + (plk[L] * (sm[L] - smt[L]) - fcdh - fcdh1) * fmc1, dgdsm[L])
            ddmflx[L] = wb(m, ddmflx[L] + ddm[L] * fmc1, ddmflx[L])
        for L in range(1, LM + 1):
            sm1[L] = wb(ev, sm[L], sm1[L])
            qm1[L] = wb(ev, qm[L], qm1[L])
        # ---------------------------------------------------------------------- precipitation
        cond[lme, ei] = cond[lme, ei] + condv[lme, ei]
        deep = (ple[lmin] - ple[lmax + 1, cols]) >= 450.
        for L in range(LM, 0, -1):
            m = ev & deep & (L <= lmax) & (L >= lmin)
            if not m.any():
                continue
            condp[L] = wb(m & (cond[L] < condp[L]), cond[L], condp[L])
            fclw = np.where(cond[L] > 0, (cond[L] - condp[L]) / cond[L], 0.0)
            ph = m & (svlatl[L] > 0) & (svlatl[L] != vlat[L])
            heat1[L] = wb(ph, heat1[L] + (svlatl[L] - vlat[L]) * svwmxl[L] * airm[L] * BYSHA / fmc1, heat1[L])
            svlatl[L] = wb(m, vlat[L], svlatl[L])
            svwmxl[L] = wb(m, svwmxl[L] + fclw * cond[L] * byam[L] * fmc1, svwmxl[L])
            cond[L] = wb(m, condp[L], cond[L])
            condpt[L] = wb(m, condpt[L] + condp[L], condpt[L])
        prcp = cond[lmax, cols]
        prheat = wb(ev, cdheat[lmax, cols], prheat)
        told = smold[lmax, cols] * plk[lmax, cols] * byam[lmax, cols]
        lh_new = np.where(told <= TF, LHS, LHE)
        lhp[lme, ei] = lh_new[ei]
        vl_top = vlat[lmax, cols]
        fix = ev & (((told > TF) & (vl_top == LHS)) | ((told <= TF) & (vl_top == LHE)))
        if fix.any():
            lhpl = lhp[lmax, cols]
            pks = plk[lmax, cols] * sm[lmax, cols]
            fs = np.where((np.abs(pks) > TEENY) & ((lhpl - vl_top) * prcp * BYSHA < 0), -((lhpl - vl_top) * prcp * BYSHA / pks), 0.0)
            fi_ = np.flatnonzero(fix)
            lf = lmax[fi_]
            sm[lf, fi_] = sm[lf, fi_] + ((lhpl - vl_top) * prcp * BYSHA / plk[lmax, cols])[fi_]
            for mm in range(NMOM):
                smom[mm, lf, fi_] = smom[mm, lf, fi_] * (1. - fs[fi_])
        for L in range(LM - 1, 0, -1):
            m = ev & (L <= lmax - 1)
            if not m.any():
                continue
            ia = np.flatnonzero(m)
            plemin, plel2, plemax = ple[lmin], ple[L + 2], ple[lmax + 1, cols]
            fcloud = np.asarray(hp.mc_cloud_fraction(F(L), tl[L], pl[L], ccm[L], wcu[L], plemin, plel2, plemax, ccm[lmin], wcu[lmin],
                                                      F(lmin), lmax.astype(float), CCMUL, CCMUL2, dtsrc=dtsrc), float)
            if (m & (fcloud < 0.)).any():
                raise RuntimeError("MSTCNV: negative cloud cover")
            fevap = .5 * ccm[L] * byam[L + 1]
            if L < lmin:
                fevap = .5 * ccm[lmin] * byam[lmin + 1]
            fevap = np.where(fevap > .5, 0.5, fevap)
            cldmcl[L + 1] = wb(m, pymin(cldmcl[L + 1] + fcloud * fmc1, fmc1), cldmcl[L + 1])
            cldref = cldmcl[L + 1]
            cldslwij = wb(m & (plemax > 700) & (cldref > cldslwij), cldref, cldslwij)
            clddepij = wb(m & (plemin - plemax >= 450) & (cldref > clddepij), cldref, clddepij)
            told = smold[L] * plk[L] * byam[L]
            told1 = smold[L + 1] * plk[L + 1] * byam[L + 1]
            precnvl[L + 1] = wb(m, precnvl[L + 1] + prcp * BYGRAV, precnvl[L + 1])
            lh, mcloud, h1 = hp.mc_precip_phase(F(L), airm[L], fevap, lhp[L + 1], prcp, told, told1, vlat[L], cond[L], F(lmin), heat1[L],
                                                mc_revp_abv_cldbase=mc_revp)
            lhp[L] = wb(m, np.asarray(lh, float), lhp[L])
            mcloud = np.asarray(mcloud, float)
            heat1[L] = wb(m, np.asarray(h1, float), heat1[L])
            lhx = lhp[L]
            slh = lhx * BYSHA
            pp = m & (prcp > 0.)
            dqsum = np.zeros(n)
            me = pp & (mcloud > 0)
            if me.any():
                dqs, _ = dq.get_dq_evap(smold[L], qmold[L], plk[L], airm[L], lhx, pl[L], prcp * airm[L] / mcloud)
                dqsum = np.where(me, dqs, dqsum)
            dqsum = np.where(pp, dqsum * mcloud * byam[L], dqsum)
            prcp = np.where(pp, prcp - dqsum, prcp)
            qm[L] = wb(pp, qm[L] + dqsum, qm[L])
            fssum = np.where((np.abs(plk[L] * sm[L]) > TEENY) & (slh * dqsum + heat1[L] > 0), (slh * dqsum + heat1[L]) / (plk[L] * sm[L]), 0.0)
            sm[L] = wb(m, sm[L] - (slh * dqsum + heat1[L]) / plk[L], sm[L])
            smom[:, L] = wb(m, smom[:, L] * (1. - fssum), smom[:, L])
            prheat = wb(m, cdheat[L] + slh * prcp, prheat)
            prcp = np.where(m, prcp + cond[L], prcp)
        pos = ev & (prcp > 0.)
        shal = (ple[lmin] - ple[lmax + 1, cols]) < 450.
        cldmcl[1] = wb(pos & shal, pymin(cldmcl[1], fmc1), cldmcl[1])
        rho = pl[1] / (RGAS * tl[1])
        cldmcl[1] = wb(pos & ~shal, pymin(cldmcl[1] + fmc1 * ccm[lmin] / (rho * GRAV * wcu[lmin] * dtsrc + TEENY), fmc1), cldmcl[1])
        if _DBG:
            _DBG('prec', lmin, ic, 0, locals())
        prcpmc = wb(ev, prcpmc + prcp * fmc1, prcpmc)
        lmcmin = np.where(ev & (lmcmin > ldmin), ldmin, lmcmin)
        mc1 = np.where(ev, False, mc1)
    # ---------------------------------------------------------------------- scatter back
    Sc.update(mccont=mccont, lmcmin=lmcmin, lmcmax=lmcmax, lmax=lmax, fmc1=fmc1, cldslwij=cldslwij, clddepij=clddepij, prcpmc=prcpmc,
              ierr=ierr, lerr=lerr, prheat=prheat)
    for k, v in S.items():
        P_a[k][..., sub] = v
    for k, v in Sc.items():
        P_s[k][sub] = v


def _adv_groups(sel, ldmin, lmax, arr, mom, ml, cmneg, qlimit, ierr, lerr):
    """Advect arr/mom (layers ldmin..lmax of each column of `sel`) with adv1d, batched over columns of equal slice."""
    keys = {}
    for cidx in sel:
        keys.setdefault((int(ldmin[cidx]), int(lmax[cidx])), []).append(cidx)
    for (l0, l1), cl in keys.items():
        cl = np.array(cl)
        sl = slice(l0, l1 + 1)
        s = arr[sl][:, cl].T.copy()                                  # (g, nx)
        mm = mom[:, sl][:, :, cl].transpose(0, 2, 1).copy()          # (9, g, nx)
        mass = ml[sl][:, cl].T.copy()
        dmv = cmneg[sl][:, cl].T.copy()
        ie, ne = adv_batch(s, mm, mass, dmv, qlimit)
        arr[l0:l1 + 1, cl] = s.T
        mom[:, l0:l1 + 1, cl] = mm.transpose(0, 2, 1)
        ierr[cl] = np.maximum(ie, ierr[cl])
        lerr[cl] = np.maximum(ne + l0 - 1, lerr[cl])
