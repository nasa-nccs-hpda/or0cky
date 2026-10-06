"""Batched CONDSE column chain (D132): clouds_condse_ff.condse_step with the 3,168 non-polar columns processed at once.

Chain (same statements and operation order as clouds_condse_ff.condse_column, vectorised over columns):
  column set-up -> clouds_mstcnv_batch.mstcnv_batch -> convective post-processing (TPRCP/PRCP/ENRGP, DDML search, DDM1/DDMS/TDN1/QDN1,
  state hand-over) -> LSCOND input set-up -> clouds_lscond_batch.lscond_batch -> precipitation/energy bookkeeping, snow age (exp per
  column through the scalar function of the per-column port) -> state stores -> radiation hand-off (vectorised branch table) -> final
  T/Q/moment/UKM merge.  The two pole columns (KMAX=72) stay on the per-column code (condse_column) and are run in the Fortran order
  (south pole first, the batch, then the north pole) so that the LSCOND module-array carry (csizelip etc.) passes through them as in
  condse_step.  The momentum back-transfer and recalc_agrid_uv are the per-column driver's own functions.
Not batched: the two poles; everything else is batched.  Not reproduced (diagnostic counters): the `cnt` branch counters except
columns/convecting.
Usage: condse_step_batch(inp, cfg, ms=None) -> (X, cnt) with the same X as clouds_condse_ff.condse_step.
"""
import numpy as np

import clouds_condse_ff as cf
import clouds_lscond_batch as lb
import clouds_lscond_size_ff as sz
import clouds_mstcnv_batch as mb

LM, IM, JM = cf.LM, cf.IM, cf.JM
GRAV, RGAS, TF, LHE, LHS, LHM = cf.GRAV, cf.RGAS, cf.TF, cf.LHE, cf.LHS, cf.LHM
BYGRAV, DELTX, ENTCON, TINY = cf.BYGRAV, cf.DELTX, cf.ENTCON, cf.TINY
f4 = cf.f4
F = np.float64


def _g(a, I, J):
    """(IM,JM,LM) -> (LM,N)."""
    return a[I, J, :].T.copy()


def _g2(a, I, J):
    """(LM,IM,JM) -> (LM,N)."""
    return a[:, I, J].copy()


def hand_off_b(X, I, J, cldmcl, cldssl, svlatl, svlhxl, qclx, qcix, cnvmmrl, cldsal):
    """CONDSE:1949-2080 for all columns; arrays (LM,N)."""
    X["W_CLOUD"][:, I, J] = cldmcl + cldssl - cldmcl * cldssl
    csizss = X["CSIZSS"][:, I, J]
    csizmc = X["CSIZMC"][:, I, J]
    cs = cldssl > 0.0
    den = cldmcl + cldssl + TINY
    w = cs & (svlhxl == LHE)
    ic_ = cs & (svlhxl == LHS)
    if (cs & ~w & ~ic_).any():
        raise RuntimeError("CONDSE: Error setting stratiform cloud properties (stop_model 255)")
    z = np.zeros_like(cldssl)
    fsw = np.where(w, cldssl / den, z)
    fsi = np.where(ic_, cldssl / den, z)
    X["FRAC_ST_WATER"][:, I, J] = fsw
    X["FRAC_ST_ICE"][:, I, J] = fsi
    X["MIX_ST_WATER"][:, I, J] = np.where(w, qclx, z)
    X["MIX_ST_ICE"][:, I, J] = np.where(ic_, qcix, z)
    X["DIM_ST_WATER"][:, I, J] = np.where(w, csizss * 1.0e-06, z)
    X["DIM_ST_ICE"][:, I, J] = np.where(ic_, csizss * 1.0e-06, z)
    X["FRAC_AREA_ST"][:, I, J] = np.where(cs, cldsal, z)
    cm = cldmcl > 0.0
    cw = cm & (svlatl == LHE)
    ci = cm & (svlatl == LHS)
    if (cm & ~cw & ~ci).any():
        raise RuntimeError("CONDSE: Error setting convective cloud properties (stop_model 255)")
    fcw = np.where(cw, cldmcl / den, z)
    fci = np.where(ci, cldmcl / den, z)
    mcw = np.where(cw, cnvmmrl, z)
    mci = np.where(ci, cnvmmrl, z)
    dcw = np.where(cw, csizmc * 1.0e-06, z)
    dci = np.where(ci, csizmc * 1.0e-06, z)
    fac_old = X["FRAC_AREA_CNV"][:, I, J]
    fac = np.where(cm, cldmcl, fac_old)                       # FRAC_AREA_CNV is deliberately not reset where there is no convective cloud
    zm = cm & (cnvmmrl == 0.0)
    fcw, fci, mcw, mci, dcw, dci, fac = (np.where(zm, z, v) for v in (fcw, fci, mcw, mci, dcw, dci, fac))
    wc = cldmcl + cldssl - cldmcl * cldssl
    wc = np.where(zm, cldssl, wc)
    renorm = zm & cs
    fw0 = fsw
    fw_n = fw0 / (fw0 + fsi)
    fi_n = fsi / (fw_n + fsi)
    X["FRAC_ST_WATER"][:, I, J] = np.where(renorm, fw_n, fsw)
    X["FRAC_ST_ICE"][:, I, J] = np.where(renorm, fi_n, fsi)
    X["W_CLOUD"][:, I, J] = wc
    X["FRAC_CNV_WATER"][:, I, J] = fcw
    X["FRAC_CNV_ICE"][:, I, J] = fci
    X["MIX_CNV_WATER"][:, I, J] = mcw
    X["MIX_CNV_ICE"][:, I, J] = mci
    X["DIM_CNV_WATER"][:, I, J] = dcw
    X["DIM_CNV_ICE"][:, I, J] = dci
    X["FRAC_AREA_CNV"][:, I, J] = fac
    return int(zm.sum())


def condse_step_batch(inp, cfg, ms=None, with_momentum=True, cnt=None):
    """-> (X, cnt).  ms: LSCOND module-array carry dict as in condse_step (updated)."""
    A = inp
    G = cfg["geom"]
    X = cf._fresh_outputs(inp)
    cnt = cnt if cnt is not None else cf.new_counts()
    ms = ms if ms is not None else cf._zeros_ms()
    dtsrc, bydtsrc, xmass, bybr = cfg["dtsrc"], cfg["bydtsrc"], cfg["xmass"], cfg["bybr"]
    lmcld = cfg["lmcld"]
    # ---------------- south pole (per column, Fortran order)
    for k in ("AIRX", "DDM1", "DDMS", "DDML", "TDN1", "QDN1"):
        X[k][:, :] = 0.0
    for i in range(int(G["IMAXJ"][0])):
        cf.condse_column(X, A, G, cfg, i, 0, ms, cnt)
    I = np.array([i for j in range(1, JM - 1) for i in range(int(G["IMAXJ"][j]))])
    J = np.array([j for j in range(1, JM - 1) for i in range(int(G["IMAXJ"][j]))])
    N = I.size
    assert (G["KMAXJ"][1:JM - 1] == 4).all()
    # ---------------- column set-up (CLOUDS2_DRV.F90:680-842)
    axyp, byaxyp = G["AXYP"][I, J], G["BYAXYP"][I, J]
    pearth, pland = A["FEARTH"][I, J], A["FLAND"][I, J]
    ts, qs = A["TSAVG"][I, J], A["QSAVG"][I, J]
    tsv = ts * (1 + qs * DELTX)
    dcl = (A["DCLEV"][I, J] + 0.5).astype(int)
    pl, ple, plk, airm = _g2(A["PMID"], I, J), _g2(A["PEDN"], I, J), _g2(A["PK"], I, J), _g2(A["PDSIG"], I, J)
    byam = 1.0 / airm
    wturb = np.sqrt(F(f4(0.6666667)) * _g2(A["EGCM"], I, J))
    ra = G["RAVJ"][:4, J].copy()
    dpdt = (pl - _g2(A["PMIDOLD"], I, J)) * bydtsrc
    t, q = _g(A["T"], I, J), _g(A["Q"], I, J)
    sm = t * airm
    smom = A["TMOM"][:, I, J, :].transpose(0, 2, 1) * airm[None]
    tl = t * plk
    qm = q * airm
    qmom = A["QMOM"][:, I, J, :].transpose(0, 2, 1) * airm[None]
    qcll, qcil = _g(A["QCL"], I, J), _g(A["QCI"], I, J)
    sdl = _g(A["MWS"], I, J) / dtsrc * byaxyp[None, :]
    tvl = tl * (1.0 + DELTX * q)
    etal, gzl = np.zeros((LM, N)), np.zeros((LM, N))
    gz = _g(A["GZ"], I, J)
    etal[1:LM - 1] = 0.5 * ENTCON * (gz[2:LM] - gz[0:LM - 2]) * 1.0e-3 * BYGRAV
    gzl[1:LM - 1] = etal[1:LM - 1] / ENTCON
    etal[LM - 1] = etal[LM - 2]
    gzl[LM - 1] = gzl[LM - 2]
    u0, v0 = A["UKM"][:, :, I, J].copy(), A["VKM"][:, :, I, J].copy()
    um, vm = u0 * airm[None], v0 * airm[None]
    tprcp = t[0] * plk[0] - TF
    # ---------------- MSTCNV
    R = dict(pearth=pearth, pland=pland, xmass=np.full(N, xmass), bydtsrc=np.full(N, bydtsrc), dtsrc=np.full(N, dtsrc), bybr=np.full(N, bybr),
             dcl=dcl, lmcm=np.full(N, cfg["lmcm"]), pl=pl, plk=plk, airm=airm, byam=byam, etal=etal, tl=tl, tvl=tvl, sm=sm, qm=qm, qcll=qcll,
             qcil=qcil, sdl=sdl, wturb=wturb, gzl=gzl, ple=ple, smom=smom, qmom=qmom, um=um, vm=vm, u0=u0, v0=v0, ra=ra)
    o = mb.mstcnv_batch(R, cfg["tune"])
    if (o["ierr"] == 2).any():
        raise RuntimeError("SUBSID ERROR: ABS(C) > 1 (stop_model 255)")
    cnt["mc_ierr"] += int((o["ierr"] > 0).sum())
    lmcmin, lmcmax = o["lmcmin"].astype(int), o["lmcmax"].astype(int)
    conv = lmcmin > 0
    cnt["convecting"] += int(conv.sum())
    Lr = np.arange(LM)[:, None]
    inmc = (Lr < lmcmax[None, :])                               # layers L < LMCMAX of convecting columns
    sm_m, qm_m = o["sm"].T, o["qm"].T                           # (LM,N)
    tmc = np.where(inmc, sm_m * byam, t)
    qmc = np.where(inmc, qm_m * byam, q)
    smom_mc = np.where(inmc[None], o["smom"].transpose(1, 2, 0), smom)
    qmom_mc = np.where(inmc[None], o["qmom"].transpose(1, 2, 0), qmom)
    um1 = np.where(inmc[None], o["um"].transpose(1, 2, 0), um)
    vm1 = np.where(inmc[None], o["vm"].transpose(1, 2, 0), vm)
    prcpmc = o["prcpmc"]
    prcp = np.where(conv, prcpmc * 100.0 * BYGRAV, 0.0)
    enrgp = np.zeros(N)
    pos = conv & (tprcp > 0)
    neg = conv & ~(tprcp > 0)
    cnt["tprcp_pos"] += int(pos.sum())
    cnt["tprcp_neg"] += int(neg.sum())
    enrgp = np.where(neg, (enrgp + 0.0) - prcp * LHM, enrgp)
    X["CSIZMC"][:, I, J] = np.where(inmc, o["csizel"].T, X["CSIZMC"][:, I, J])
    fssl = np.where(conv[None, :], o["fssl"].T, 1.0)
    X["FSS"][:, I, J] = np.where(conv[None, :], o["fssl"].T, X["FSS"][:, I, J])
    X["AIRX"][I, J] = np.where(conv, o["airxl"] * axyp, 0.0)
    ddm = o["ddmflx"]
    posd = (ddm > 0.0) & ((np.arange(1, LM + 1)[None, :]) <= dcl[:, None])
    ddml = np.where(posd.any(axis=1), posd.argmax(axis=1) + 1, dcl)
    cols = np.arange(N)
    X["DDML"][I, J] = np.where(conv, ddml, 0.0)
    X["TDN1"][I, J] = np.where(conv, o["tdnl"][cols, ddml - 1], 0.0)
    X["QDN1"][I, J] = np.where(conv, o["qdnl"][cols, ddml - 1], 0.0)
    X["DDMS"][I, J] = np.where(conv, -100.0 * ddm[cols, ddml - 1] / (GRAV * dtsrc), 0.0)
    X["DDM1"][I, J] = np.where(conv, ddm[:, 0] * RGAS * tsv / (GRAV * ple[0] * dtsrc), 0.0)
    X["LMC"][0, I, J] = lmcmin
    X["LMC"][1, I, J] = lmcmax + 1
    # ---------------- LSCOND set-up (1259-1284)
    svlatl, svwmxl = o["svlatl"].T, o["svwmxl"].T
    qclx = np.where(inmc & (svlatl == LHE), qcll + svwmxl, qcll)
    qcix = np.where(inmc & (svlatl == LHS), qcil + svwmxl, qcil)
    ql_ls = q.copy()
    aq = (ql_ls - _g2(A["QTOLD"], I, J)) * bydtsrc
    S = dict(qcll=qcll.copy(), qcil=qcil.copy(), svlatl=svlatl.copy(), svlat1=o["svlat1"].T.copy(), svwmxl=svwmxl.copy(), sdl=sdl, vsubl=o["vsubl"].T.copy(),
             fssl=fssl.copy(), ttoldl=_g2(A["TTOLD"], I, J), aq=aq, dpdt=dpdt, pl=pl, plk=plk, airm=airm, byam=byam, u00l=o["u00l"].T.copy(),
             taumcl=o["taumcl"].T.copy(), precnvl=o["precnvl"].T.copy(), tl=t * plk, ql=ql_ls, th=t.copy(), rh=_g2(A["RHSAV"], I, J), qclx=qclx.copy(),
             qcix=qcix.copy(), svlhxl=_g2(A["SVLHX"], I, J), cldsavl=_g2(A["CLDSAV"], I, J), csizel=o["csizel"].T.copy(), sm=sm_m.copy(),
             qm=qm_m.copy(), qmom=qmom.copy(), smom=smom.copy(), um=um.transpose(1, 0, 2).copy() * 0 + um.transpose(1, 0, 2),
             vm=vm.transpose(1, 0, 2).copy())
    S["qmom"] = qmom.transpose(1, 0, 2).copy()
    S["smom"] = smom.transpose(1, 0, 2).copy()
    S["um"] = um.transpose(1, 0, 2).copy()
    S["pdsigl00"] = list(cf.PDSIGL00)
    S["dqlsc"] = np.zeros((LM, N))
    prev = ms.get("S") or {}
    for k in ("tausslip", "csizelip", "cldsal", "cldsv1"):
        init = np.asarray(prev.get(k, np.zeros(LM)), float)
        S[k + "_init"] = init
    S["tausslip"] = np.tile(S["tausslip_init"][:, None], (1, N))
    S["cldsv1"] = np.tile(S["cldsv1_init"][:, None], (1, N))
    pe_ = pearth
    c = cf.LS_CONST
    sndo = 59.68 / (c["rwcldox"] ** 3)
    P = {k: c[k] for k in ("cmx", "u00a", "rimax", "rwmax", "rwcldox", "rcldix")}
    P.update(rcldlx=c["rcldlx"], wmui=cf.WMUIX * float(f4(0.001)), scdnci=0.06417127, bybr=bybr, bydtsrc=bydtsrc, dtsrc=dtsrc, lmcld=lmcld,
             pearth=pe_, wconst=lb.L0.WMU * (1.0 - pe_) + lb.L0.WMUL * pe_, scdncw=sndo * (1.0 - pe_) + 174.0 * pe_, dcl=dcl, ra=ra)
    S, W = lb.lscond_batch(S, P)
    # ---------------- bookkeeping (1442-1464)
    prcpss = W["prcpss"]
    lhp = S["lhp"]
    prcp = prcp + prcpss * 100.0 * BYGRAV
    ice0 = lhp[0] == LHS
    enrgp = np.where(ice0, (enrgp + 0.0) - prcpss * 100.0 * BYGRAV * LHM, enrgp + 0.0)
    cnt["lhp1_ice"] += int(ice0.sum())
    sa = enrgp < 0.0
    cnt["snow_age"] += int(sa.sum())
    if sa.any():
        ia = np.flatnonzero(sa)
        e = np.array([float(sz.ex(-v)) for v in prcp[ia].tolist()])
        for it in range(3):
            X["SNOAGE"][it, I[ia], J[ia]] = X["SNOAGE"][it, I[ia], J[ia]] * e
    # ---------------- stores (1932-1945)
    cldmcl = o["cldmcl"].T
    X["TAUMC"][:, I, J] = o["taumcl"].T
    X["CLDMC"][:, I, J] = cldmcl
    X["SVLAT"][:, I, J] = svlatl
    cldssl = S["cldssl"]
    X["TAUSS"][:, I, J] = S["taussl"]
    X["CLDSS"][:, I, J] = cldssl
    X["CLDSAV"][:, I, J] = S["cldsavl"]
    X["CLDSAV1"][:, I, J] = S["cldsv1"]
    X["SVLHX"][:, I, J] = S["svlhxl"]
    X["CSIZSS"][:, I, J] = S["csizel"]
    X["QLSS"][:, I, J] = S["qlss"]
    X["QISS"][:, I, J] = S["qiss"]
    X["QLMC"][:, I, J] = o["qlmc"].T
    X["QIMC"][:, I, J] = o["qimc"].T
    cnt["cnv_zero_mmr"] += hand_off_b(X, I, J, cldmcl, cldssl, svlatl, S["svlhxl"], S["qclx"], S["qcix"], o["cnvmmrl"].T, S["cldsal"])
    X["TAUSSIP"][:, I, J] = S["tausslip"]
    X["CSIZSSIP"][:, I, J] = S["csizelip"]
    X["RHSAV"][:, I, J] = S["rh"]
    X["TTOLD"][:, I, J] = S["th"]
    X["QTOLD"][:, I, J] = S["ql"]
    X["PREC"][I, J] = prcp
    X["EPREC"][I, J] = enrgp
    precss = prcpss * 100.0 * BYGRAV
    X["PRECSS"][I, J] = precss
    X["P_ACC"][I, J] = X["P_ACC"][I, J] + prcp
    X["PM_ACC"][I, J] = X["PM_ACC"][I, J] + prcp - precss
    # ---------------- final merge (2152-2185)
    fs = fssl
    X["T"][I, J, :] = (S["th"] * fs + tmc * (1.0 - fs)).T
    X["Q"][I, J, :] = (S["ql"] * fs + qmc * (1.0 - fs)).T
    smom_f = S["smom"].transpose(1, 0, 2) * fs[None] + smom_mc * (1.0 - fs[None])
    qmom_f = S["qmom"].transpose(1, 0, 2) * fs[None] + qmom_mc * (1.0 - fs[None])
    X["TMOM"][:, I, J, :] = (smom_f * byam[None]).transpose(0, 2, 1)
    X["QMOM"][:, I, J, :] = (qmom_f * byam[None]).transpose(0, 2, 1)
    X["QCI"][I, J, :] = S["qcix"].T
    X["QCL"][I, J, :] = S["qclx"].T
    X["TMC"][I, J, :] = tmc.T
    X["QMC"][I, J, :] = qmc.T
    ums = S["um"].transpose(1, 0, 2)
    vms = S["vm"].transpose(1, 0, 2)
    du = (ums * fs[None] + um1 * (1.0 - fs[None])) * byam[None] - u0
    dv = (vms * fs[None] + vm1 * (1.0 - fs[None])) * byam[None] - v0
    X["UKM"][:, :, I, J] = du
    X["VKM"][:, :, I, J] = dv
    cnt["columns"] += N
    # carry for the north pole: the last column's LSCOND arrays
    last = {k: np.array(S[k][:, -1]) for k in ("tausslip", "csizelip", "cldsal", "cldsv1")}
    ms["S"] = dict(last, dqlsc=[0.0] * LM, lhp=lhp[:, -1].tolist(), prebar1=S["prebar1"][:, -1].tolist())
    # ---------------- north pole (per column)
    for i in range(int(G["IMAXJ"][JM - 1])):
        cf.condse_column(X, A, G, cfg, i, JM - 1, ms, cnt)
    if with_momentum:
        cf.avg_replicated_duv_to_vgrid(X["U"], X["V"], X["UKM"], X["VKM"], X["UKMSP"], X["VKMSP"], X["UKMNP"], X["VKMNP"])
        import dyn_glue_ff as gf
        X["UALIJ"], X["VALIJ"] = gf.recalc_agrid_uv(X["U"], X["V"], cfg["glue_geom"])
    return X, cnt
