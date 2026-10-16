"""Full-fidelity numpy port of the CONDSE column-driver glue (CLOUDS2_DRV.F90:3-2854 live physics lines, about 480) and of the
derived constants of init_CLD (CLOUDS2_DRV.F90:3085-3098) -- D124-D126, scoping items D-C8 and D-C10.

`condse_step(inp, cfg)` takes the arrays recorded at CONDSE entry (clouds_condse_io.load_step) and runs every column in the order
of the Fortran (J outer, I inner, I<=IMAXJ(J): 3170 columns), calling the validated ports `clouds_mstcnv_ff.mstcnv_column` (MSTCNV)
and `clouds_lscond_ff.lscond` (LSCOND) by import.  Order of one column (CLOUDS2_DRV.F90 line numbers, see
scoping/CLOUDS_CONDSE_PLAN.md for the full table): column set-up 680-842; TPRCP 848; MSTCNV 896; convective post-processing 1119-1180
(LMCMIN>0 only); LMC 1254; LSCOND input set-up 1259-1284; LSCOND 1352; precipitation and energy bookkeeping 1442-1464; snow age
1485; state stores 1932-1945, 2084-2088, 2115-2125; radiation hand-off 1949-2080; final T/Q/moment/UKM merge 2152-2185.  After the
column loop: avg_replicated_duv_to_vgrid (ATMDYN.f:2606-2708) and recalc_agrid_uv (dyn_glue_ff).

Not ported (diagnostic or dead for P2SAoM40): all AIJ/AJL/ADIURN/SUBDD bookkeeping, ISCCP_CLOUD_TYPES (its only coupling is the
stop_model on JERR), the ISC=1 stratocumulus block (1288-1315; ISC=0, the entry dump records isc), RIS/RI1/RI2 (1319-1343; read only
under do_blU00==1, which is 0).  The intermediate T/Q stores of the convective block (1146-1147) are overwritten at 2152-2153 and are
not reproduced.

Single-precision literal audit (D54 hazard: an unsuffixed REAL literal is rounded to 24 bits, then promoted):
  .6666667   WTURB=sqrt(.6666667*EGCM)                       -> f4  (CLOUDS2_DRV.F90:737)
  .001       WMUI=WMUIX*.001 (LSCOND parameters)             -> f4
  everything else in the glue is exactly representable (1., .5, 100., 1.d-3 and 1d-6 are d0/E-notation DOUBLE or exact) or d0:
  .5*ENTCON*...*1.d-3*BYGRAV evaluates strictly left to right; TINY(CLDSSL) is the REAL(8) smallest normal number.
exp(-PRCP) (snow age) and the pow in BYBR go through the selectable backend of clouds_lscond_size_ff (use_imf).

Module-state emulation: the Fortran module CLOUDS carries some LSCOND arrays from one column to the next (CSIZELIP, CLDSAL, DQLSC ...);
they are carried here in the same column order (`ms` dict); the very first column of a step starts from zeros, and a mismatch that can
only come from that is reported by the compare script, not hidden.
"""
import math
import types

import numpy as np

import clouds_condse_io as cio
import clouds_helpers_ff as ch
import clouds_lscond_ff as ls0
import clouds_lscond_size_ff as sz
import clouds_massflux_ff as mf
import clouds_mstcnv_ff as mc

F = np.float64
f4 = ch.f4
LM, IM, JM, NMOM = 40, 72, 46, 9
GRAV, RGAS, TF = ch.GRAV, ch.RGAS, ch.TF
LHE, LHS, LHM = ch.LHE, ch.LHS, ch.LHM
BYGRAV = 1.0 / GRAV
DELTX = mf.DELTX
ENTCON = 0.2
TINY = np.finfo(np.float64).tiny
LMCLD_RUN = 29
LMCM_RUN = 23
SECONDS = 1800.0

# rundeck / default tunables of the real run (ffc_mc_consts.txt, ffc_ls_bnd records; P2SAoM40.R:205-215)
RUN_TUNE = dict(entrainment_cont1=0.4, entrainment_cont2=0.6, radiusl_multiplier=1.01, radiusi_multiplier=1.0, u00a=0.695, u00b=0.6,
                wmu_multiplier=1.0, rwcldox=1.0, rimax=100.0, mc_fddrt=0.5, mc_entr_mass_lim_plume=1, mc_new_ddrft_thetav=1,
                mc_revp_abv_cldbase=1)
LS_CONST = dict(rimax=100.0, rwmax=20.0, rwcldox=1.0, rcldlx=1.01, rcldix=1.0, cmx=1.0, u00a=0.695, u00b=0.6, rtemp=22.0, do_blu00=0.0)
WMUIX = 0.001            # wmui_multiplier (P2SAoM40.R)
PDSIGL00 = [20.000000000000004, 22.0, 25.000000000000004, 27.000000000000004, 30.0, 35.00000000000001, 40.00000000000001,
            45.00000000000001, 48.00000000000001, 50.00000000000001, 51.0, 52.00000000000001, 50.00000000000001, 48.00000000000001,
            45.00000000000001, 42.00000000000001, 38.00000000000001, 34.00000000000001, 31.000000000000004, 28.0, 26.000000000000004,
            24.000000000000004, 23.0, 22.0, 20.000000000000004, 18.0, 17.0, 16.0, 14.0, 12.0, 11.0, 10.000000000000002, 4.38, 2.46,
            1.3800000000000001, 0.78, 0.43799999999999994, 0.24600000000000005, 0.138, 0.078]   # ATM_COM PDSIGL00 (recorded, constant)


def init_cld_constants(dtsrc=SECONDS):
    """Derived constants of init_CLD (CLOUDS2_DRV.F90:3095-3098) that CONDSE/MSTCNV/LSCOND read: BYDTsrc, XMASS, BYBR.
    BYBR = ((1.-BRCLD)*(1.-2.*BRCLD))**BY3 with BRCLD=.2d0 (libm pow)."""
    brcld = 0.2
    bydtsrc = 1.0 / dtsrc
    xmass = 0.1 * dtsrc * GRAV
    bybr = sz.pw((1.0 - brcld) * (1.0 - 2.0 * brcld), ch.BY3)
    return dict(bydtsrc=bydtsrc, xmass=xmass, bybr=float(bybr))


def ls_params(pearth, dcl, kmax, ra, bydtsrc, dtsrc, bybr, lmcld=LMCLD_RUN):
    """Per-column LSCOND parameter set-up (CLOUDS2.F90:3441-3459): WCONST, WMUI, SCDNCW, SCDNCI plus the rundeck constants."""
    c = LS_CONST
    p = dict(c)
    p["wconst"] = ls0.WMU * (1.0 - pearth) + ls0.WMUL * pearth
    p["wmui"] = WMUIX * float(f4(0.001))
    sndo = 59.68 / (c["rwcldox"] ** 3)
    p["scdncw"] = sndo * (1.0 - pearth) + 174.0 * pearth
    p["scdnci"] = 0.06417127
    p.update(bybr=bybr, bydtsrc=bydtsrc, dtsrc=dtsrc, pearth=pearth, dcl=int(dcl), lmcld=int(lmcld), kmax=int(kmax), use_vmp=True,
             ra=[float(x) for x in ra])
    return p


_LS_ALLK = None


def _lscond_allk():
    """clouds_lscond_ff compiled from the same source with `kmax = min(kmax, 4)` replaced by `kmax = kmax` (needed for the two
    pole columns, KMAX = 72; the standalone LSCOND dump only recorded K <= 4).  The file itself is not modified."""
    global _LS_ALLK
    if _LS_ALLK is None:
        src = open(ls0.__file__).read()
        old = 'kmax = min(int(P["kmax"]), 4)'
        assert src.count(old) == 1
        mod = types.ModuleType("clouds_lscond_ff_allk")
        mod.__file__ = ls0.__file__
        exec(compile(src.replace(old, 'kmax = int(P["kmax"])'), ls0.__file__ + "#allk", "exec"), mod.__dict__)
        _LS_ALLK = mod
    return _LS_ALLK


def set_backend(name):
    """'numpy' (libm) or 'imf' (Intel libimf for exp/pow in MSTCNV and LSCOND, as the real ifort build)."""
    mc.set_backend(name)
    sz.use_imf(name == "imf")


# ------------------------------------------------------------------------------------------------ RANDU / RNDSS
def randu_stream(ix0, ncols_per_j, lmcld=LMCLD_RUN):
    """RNDSS(3,L,I,J) draws of CONDSE:525-540 for J=1..JM, I=1..IMAXJ(J), L=LMCLD..1, NR=1..3 from seed ix0 (RANDOM module:
    ix=ix*69069+1 in 32-bit wrap-around; RANDU = iand(ix,ffffff00)*2**-32, plus 1 if ix<0, all exact in REAL(4)).
    -> (rndss (3,LM,IM,JM) with L>LMCLD zero, final seed as signed 32-bit)."""
    ix = int(ix0) & 0xFFFFFFFF
    out = np.zeros((3, LM, IM, JM))
    for j in range(JM):
        for i in range(ncols_per_j[j]):
            for L in range(lmcld - 1, -1, -1):
                for nr in range(3):
                    ix = (ix * 69069 + 1) & 0xFFFFFFFF
                    s = ix - (1 << 32) if ix >= (1 << 31) else ix
                    v = (s & ~0xFF) * 2.0 ** -32
                    out[nr, L, i, j] = 1.0 + v if s < 0 else v
    seed = ix - (1 << 32) if ix >= (1 << 31) else ix
    return out, seed


def replicate_uv_to_agrid(u, v):
    """ATMDYN.f:2554-2604 (k=4): u,v (IM,JM,LM) -> ukm,vkm (4,LM,IM,JM) for J=2..JM-1 (rows 1 and JM are untouched, zero here),
    ukmsp,vkmsp = u,v(:,2,:) and ukmnp,vkmnp = u,v(:,JM,:) as (IM,LM)."""
    ukm = np.zeros((4, LM, IM, JM))
    vkm = np.zeros((4, LM, IM, JM))
    im1 = np.arange(IM) - 1
    for j in range(1, JM - 1):
        for X, W in ((ukm, u), (vkm, v)):
            X[0, :, :, j] = W[im1, j, :].T
            X[1, :, :, j] = W[:, j, :].T
            X[2, :, :, j] = W[im1, j + 1, :].T
            X[3, :, :, j] = W[:, j + 1, :].T
    return ukm, vkm, u[:, 1, :].copy(), v[:, 1, :].copy(), u[:, JM - 1, :].copy(), v[:, JM - 1, :].copy()


# ------------------------------------------------------------------------------------------------ momentum back-transfer
def avg_replicated_duv_to_vgrid(u, v, ukm, vkm, ukmsp, vkmsp, ukmnp, vkmnp):
    """ATMDYN.f:2606-2708 (latlon, ALT_CLDMIX_UV undefined): adds the A-grid tendencies ukm,vkm (4,LM,IM,JM; modified in place
    exactly as the Fortran modifies DU,DV: pole rows copied in and scaled by .5) to u,v (IM,JM,LM; modified in place)."""
    for X, Xs, Xn in ((ukm, ukmsp, ukmnp), (vkm, vkmsp, vkmnp)):
        im1 = np.arange(IM) - 1                      # i-1 with wrap (Fortran i=1 -> im)
        # south pole row J=1 (index 0): du(3,:,i,1)=dusp(i-1), du(4,:,i,1)=dusp(i), then *.5
        X[2, :, :, 0] = Xs[im1, :].T * 0.5
        X[3, :, :, 0] = Xs[np.arange(IM), :].T * 0.5
        # north pole row J=JM: du(1,:,i,jm)=dunp(i-1), du(2,:,i,jm)=dunp(i), then *.5
        X[0, :, :, JM - 1] = Xn[im1, :].T * 0.5
        X[1, :, :, JM - 1] = Xn[np.arange(IM), :].T * 0.5
    ip1 = (np.arange(IM) + 1) % IM
    for j in range(1, JM):                           # J_0STG=2..J_1STG=JM
        for X, W in ((ukm, u), (vkm, v)):
            s = ((X[3, :, :, j - 1] + X[2, :, ip1, j - 1].T) + X[1, :, :, j]) + X[0, :, ip1, j].T      # (LM, IM)
            W[:, j, :] = W[:, j, :] + s.T


# ------------------------------------------------------------------------------------------------ column driver
def _zeros_ms():
    return dict(S=None)


def _hand_off(X, i, j, cldmcl, cldssl, svlatl, svlhxl, qclx, qcix, cnvmmrl, cldsal, cnt, mut=None):
    mut = mut or {}
    """CONDSE:1949-2080 (radiation hand-off arrays of one column)."""
    cldmcl, cldssl = np.asarray(cldmcl), np.asarray(cldssl)
    X["W_CLOUD"][:, i, j] = cldmcl + cldssl - cldmcl * cldssl
    for l in range(LM):
        if cldssl[l] > 0.0:
            den = cldmcl[l] + cldssl[l] + TINY
            if svlhxl[l] == LHE:
                X["FRAC_ST_WATER"][l, i, j] = cldssl[l] / den
                X["FRAC_ST_ICE"][l, i, j] = 0.0
                X["MIX_ST_WATER"][l, i, j] = qclx[l]
                X["MIX_ST_ICE"][l, i, j] = 0.0
                X["DIM_ST_WATER"][l, i, j] = X["CSIZSS"][l, i, j] * 1.0e-06
                X["DIM_ST_ICE"][l, i, j] = 0.0
                X["FRAC_AREA_ST"][l, i, j] = cldsal[l]
                cnt["st_water"] += 1
            elif svlhxl[l] == LHS:
                X["FRAC_ST_WATER"][l, i, j] = 0.0
                X["FRAC_ST_ICE"][l, i, j] = cldssl[l] / den
                X["MIX_ST_WATER"][l, i, j] = 0.0
                X["MIX_ST_ICE"][l, i, j] = qcix[l]
                X["DIM_ST_WATER"][l, i, j] = 0.0
                X["DIM_ST_ICE"][l, i, j] = X["CSIZSS"][l, i, j] * 1.0e-06
                X["FRAC_AREA_ST"][l, i, j] = cldsal[l]
                cnt["st_ice"] += 1
            else:
                raise RuntimeError("CONDSE: Error setting stratiform cloud properties (stop_model 255)")
        else:
            for k in ("FRAC_ST_WATER", "FRAC_ST_ICE", "MIX_ST_WATER", "MIX_ST_ICE", "DIM_ST_WATER", "DIM_ST_ICE", "FRAC_AREA_ST"):
                X[k][l, i, j] = 0.0
        if cldmcl[l] > 0.0:
            den = cldmcl[l] + cldssl[l] + TINY
            if svlatl[l] == LHE:
                X["FRAC_CNV_WATER"][l, i, j] = cldmcl[l] / den
                X["FRAC_CNV_ICE"][l, i, j] = 0.0
                X["MIX_CNV_WATER"][l, i, j] = cnvmmrl[l]
                X["MIX_CNV_ICE"][l, i, j] = 0.0
                X["DIM_CNV_WATER"][l, i, j] = X["CSIZMC"][l, i, j] * 1.0e-06
                X["DIM_CNV_ICE"][l, i, j] = 0.0
                X["FRAC_AREA_CNV"][l, i, j] = cldmcl[l]
                cnt["cnv_water"] += 1
            elif svlatl[l] == LHS:
                X["FRAC_CNV_WATER"][l, i, j] = 0.0
                X["FRAC_CNV_ICE"][l, i, j] = cldmcl[l] / den
                X["MIX_CNV_WATER"][l, i, j] = 0.0
                X["MIX_CNV_ICE"][l, i, j] = cnvmmrl[l]
                X["DIM_CNV_WATER"][l, i, j] = 0.0
                X["DIM_CNV_ICE"][l, i, j] = X["CSIZMC"][l, i, j] * 1.0e-06
                X["FRAC_AREA_CNV"][l, i, j] = cldmcl[l]
                cnt["cnv_ice"] += 1
            else:
                raise RuntimeError("CONDSE: Error setting convective cloud properties (stop_model 255)")
            if cnvmmrl[l] == 0.0:
                for k in ("FRAC_CNV_WATER", "FRAC_CNV_ICE", "MIX_CNV_WATER", "MIX_CNV_ICE", "DIM_CNV_WATER", "DIM_CNV_ICE",
                          "FRAC_AREA_CNV"):
                    X[k][l, i, j] = 0.0
                X["W_CLOUD"][l, i, j] = cldssl[l]
                cnt["cnv_zero_mmr"] += 1
                if cldssl[l] > 0.0:
                    fw = X["FRAC_ST_WATER"][l, i, j]
                    fi = X["FRAC_ST_ICE"][l, i, j]
                    fw0 = fw
                    fw = fw / (fw + fi)                    # second statement sees the updated frac_st_water (Fortran order)
                    fi = fi / ((fw0 if mut.get("renorm_old") else fw) + fi)
                    X["FRAC_ST_WATER"][l, i, j] = fw
                    X["FRAC_ST_ICE"][l, i, j] = fi
        else:
            for k in ("FRAC_CNV_WATER", "FRAC_CNV_ICE", "MIX_CNV_WATER", "MIX_CNV_ICE", "DIM_CNV_WATER", "DIM_CNV_ICE"):
                X[k][l, i, j] = 0.0                        # FRAC_AREA_CNV deliberately NOT reset (CLOUDS2_DRV.F90:2068-2078)
            if mut.get("reset_area_cnv"):
                X["FRAC_AREA_CNV"][l, i, j] = 0.0
            cnt["cnv_none"] += 1


def new_counts():
    return {k: 0 for k in ("columns", "convecting", "mc_ierr", "ls_ierr", "snow_age", "ddml_exit_early", "ddml_full_loop",
                           "st_water", "st_ice", "cnv_water", "cnv_ice", "cnv_zero_mmr", "cnv_none", "lhp1_ice", "tprcp_pos",
                           "tprcp_neg")}


def condse_column(X, A, G, cfg, i, j, ms, cnt, br=None, mut=None):
    """One column (0-based i, j).  X: output arrays (mutated), A: entry arrays (read only), G: geometry, cfg: constants."""
    mut = mut or {}
    j1 = j + 1
    kmax = int(G["KMAXJ"][j])
    dtsrc, bydtsrc, xmass, bybr = cfg["dtsrc"], cfg["bydtsrc"], cfg["xmass"], cfg["bybr"]
    axyp, byaxyp = F(G["AXYP"][i, j]), F(G["BYAXYP"][i, j])
    pearth, pland = F(A["FEARTH"][i, j]), F(A["FLAND"][i, j])
    ts, qs = F(A["TSAVG"][i, j]), F(A["QSAVG"][i, j])
    tsv = ts * (1 + qs * DELTX)
    dcl = int(A["DCLEV"][i, j] + 0.5)
    pl, ple, plk, airm = A["PMID"][:, i, j], A["PEDN"][:, i, j], A["PK"][:, i, j], A["PDSIG"][:, i, j]
    byam = 1.0 / airm
    wturb = np.sqrt((F(0.6666667) if mut.get("wturb_double") else F(f4(0.6666667))) * A["EGCM"][:, i, j])
    ra = G["RAVJ"][:kmax, j]
    dpdt = (pl - A["PMIDOLD"][:, i, j]) * bydtsrc
    t, q = A["T"][i, j, :], A["Q"][i, j, :]
    sm = t * airm
    smom = A["TMOM"][:, i, j, :] * airm
    tl = t * plk
    qm = q * airm
    qmom = A["QMOM"][:, i, j, :] * airm
    qcll, qcil = A["QCL"][i, j, :], A["QCI"][i, j, :]
    sdl = A["MWS"][i, j, :] / dtsrc * byaxyp
    tvl = tl * (1.0 + DELTX * q)
    etal = np.zeros(LM)
    gzl = np.zeros(LM)
    gz = A["GZ"][i, j, :]
    for L in range(1, LM - 1):                                  # Fortran L=1..LM-2 -> ETAL(L+1)
        etal[L] = 0.5 * ENTCON * (gz[L + 1] - gz[L - 1]) * 1.0e-3 * BYGRAV
        gzl[L] = etal[L] / ENTCON
    etal[LM - 1] = etal[LM - 2]
    etal[0] = 0.0
    gzl[LM - 1] = gzl[LM - 2]
    gzl[0] = 0.0
    if j == 0:
        u0, v0 = A["UKMSP"][:kmax, :], A["VKMSP"][:kmax, :]
    elif j == JM - 1:
        u0, v0 = A["UKMNP"][:kmax, :], A["VKMNP"][:kmax, :]
    else:
        u0, v0 = A["UKM"][:kmax, :, i, j], A["VKM"][:kmax, :, i, j]
    um, vm = u0 * airm, v0 * airm
    um1, vm1 = um.copy(), vm.copy()
    prcp, enrgp = F(0.0), F(0.0)
    tprcp = t[0] * plk[0] - TF
    if mut.get("tprcp_post"):
        tprcp = tprcp + 1000.0          # test-only mutation: pretend the surface layer is warm
    for k in ("AIRX", "DDM1", "DDMS", "DDML", "TDN1", "QDN1"):
        X[k][i, j] = 0.0
    r = dict(pearth=pearth, pland=pland, dcl=dcl, lmcm=cfg["lmcm"], xmass=xmass, bydtsrc=bydtsrc, dtsrc=dtsrc, bybr=bybr, kmax=kmax,
             pl=pl, ple=ple, plk=plk, airm=airm, byam=byam, etal=etal, tl=tl, tvl=tvl, sm=sm, qm=qm, qcll=qcll, qcil=qcil, sdl=sdl,
             wturb=wturb, gzl=gzl, smom=smom, qmom=qmom, ra=ra, um=um, vm=vm, u0=u0, v0=v0)
    # ------------------------------------------------------------------ MSTCNV (CLOUDS2_DRV.F90:896)
    o = mc.mstcnv_column(r, cfg["tune"], br=br, mut=mut)
    tr = cfg.get("trace")
    if tr is not None and (i, j) in tr:
        tr[(i, j)] = dict(r=r, o=o)
    if o["ierr"] > 0:
        cnt["mc_ierr"] += 1
        if o["ierr"] == 2:
            raise RuntimeError("SUBSID ERROR: ABS(C) > 1 (stop_model 255)")
    lmcmin, lmcmax = int(o["lmcmin"]), int(o["lmcmax"])
    fssl = np.ones(LM)
    sm_m, qm_m = np.asarray(o["sm"]), np.asarray(o["qm"])
    smom_mc, qmom_mc = smom.copy(), qmom.copy()
    tmc, qmc = t.copy(), q.copy()
    csiz_mc = X["CSIZMC"]
    if lmcmin > 0:
        cnt["convecting"] += 1
        prcpmc = F(o["prcpmc"])
        prcp = prcpmc * 100.0 * BYGRAV
        if tprcp > 0:
            eprcp = 0.0
            enrgp = enrgp + eprcp
            cnt["tprcp_pos"] += 1
        else:
            eprcp = 0.0
            enrgp = enrgp + eprcp - prcp * LHM
            cnt["tprcp_neg"] += 1
        fo = np.asarray(o["fssl"])
        for L in range(lmcmax):
            tmc[L] = sm_m[L] * byam[L]
            qmc[L] = qm_m[L] * byam[L]
            smom_mc[:, L] = np.asarray(o["smom"])[:, L]
            qmom_mc[:, L] = np.asarray(o["qmom"])[:, L]
            um1[:, L] = np.asarray(o["um"])[:, L]
            vm1[:, L] = np.asarray(o["vm"])[:, L]
        csiz_mc[:lmcmax, i, j] = np.asarray(o["csizel"])[:lmcmax]
        fssl = fo.copy()
        X["FSS"][:, i, j] = fssl
        X["AIRX"][i, j] = F(o["airxl"]) * axyp
        ddmflx = np.asarray(o["ddmflx"])
        ddml = 0
        for L in range(1, dcl + (1 if mut.get("ddml_loop_long") else 0) + 1):
            ddml = L
            if ddmflx[L - 1] > 0.0:
                cnt["ddml_exit_early"] += 1
                break
        else:
            cnt["ddml_full_loop"] += 1
        X["DDML"][i, j] = ddml
        if ddml > 0:
            X["TDN1"][i, j] = np.asarray(o["tdnl"])[ddml - 1]
            X["QDN1"][i, j] = np.asarray(o["qdnl"])[ddml - 1]
            X["DDMS"][i, j] = -100.0 * ddmflx[ddml - 1] / (GRAV * dtsrc)
        X["DDM1"][i, j] = ddmflx[0] * RGAS * tsv / (GRAV * A["PEDN"][0, i, j] * dtsrc)
    X["LMC"][0, i, j] = lmcmin
    X["LMC"][1, i, j] = lmcmax + 1
    # ------------------------------------------------------------------ LSCOND set-up (1259-1284)
    tls = A["T"][i, j, :]
    qls = A["Q"][i, j, :]
    th = tls.copy()
    tl_ls = tls * plk
    ql_ls = qls.copy()
    qclx, qcix = qcll.copy(), qcil.copy()
    svlatl, svwmxl = np.asarray(o["svlatl"]), np.asarray(o["svwmxl"])
    for L in range(lmcmax):
        if svlatl[L] == LHE:
            qclx[L] = qclx[L] + svwmxl[L]
        elif svlatl[L] == LHS:
            qcix[L] = qcix[L] + svwmxl[L]
    aq = (ql_ls - A["QTOLD"][:, i, j]) * bydtsrc
    rndssl = np.zeros((LM, 3))
    rndssl[:cfg["lmcld"], :] = A["RNDSS"][:, :cfg["lmcld"], i, j].T
    fssl_ls = X["FSS"][:, i, j].copy()
    um_ls, vm_ls = u0 * airm, v0 * airm
    prev = ms.get("S") or {}
    keep = {k: prev[k] for k in ("tausslip", "csizelip", "cldsal", "cldsv1", "dqlsc", "lhp", "prebar1") if k in prev}
    z = lambda n: [0.0] * n  # noqa: E731
    S = dict(qcll=qcll.tolist(), qcil=qcil.tolist(), svlatl=svlatl.tolist(), svlat1=np.asarray(o["svlat1"]).tolist(),
             svwmxl=svwmxl.tolist(), sdl=sdl.tolist(), vsubl=np.asarray(o["vsubl"]).tolist(), fssl=fssl_ls.tolist(),
             ttoldl=A["TTOLD"][:, i, j].tolist(), aq=aq.tolist(), dpdt=dpdt.tolist(), pl=pl.tolist(), plk=plk.tolist(),
             airm=airm.tolist(), byam=byam.tolist(), u00l=np.asarray(o["u00l"]).tolist(), pdsigl00=list(PDSIGL00),
             taumcl=np.asarray(o["taumcl"]).tolist(), precnvl=np.asarray(o["precnvl"]).tolist(),
             rndssl=rndssl.tolist(),
             tl=tl_ls.tolist(), ql=ql_ls.tolist(), th=th.tolist(), rh=A["RHSAV"][:, i, j].tolist(), qclx=qclx.tolist(),
             qcix=qcix.tolist(), svlhxl=A["SVLHX"][:, i, j].tolist(), cldsavl=A["CLDSAV"][:, i, j].tolist(),
             cldssl=z(LM), taussl=z(LM), csizel=np.asarray(o["csizel"]).tolist(), sm=sm_m.tolist(), qm=qm_m.tolist(),
             rh1=z(LM), wmpr=z(LM), qlss=z(LM), qiss=z(LM), sshr=z(LM), dctei=z(LM),
             qmom=qmom.T.tolist(), smom=smom.T.tolist(),
             um=um_ls.T.tolist(), vm=vm_ls.T.tolist())
    # LSCOND works on SMOMLS/QMOMLS = the ENTRY moments (CONDSE:1263-1264), not the post-MSTCNV ones
    for k, v in keep.items():
        S[k] = v
    for k in ("tausslip", "csizelip", "cldsal", "cldsv1", "dqlsc"):
        S.setdefault(k, z(LM))
    S.setdefault("lhp", z(LM + 1))
    S.setdefault("prebar1", z(LM + 1))
    P = ls_params(float(pearth), dcl, kmax, ra, bydtsrc, dtsrc, bybr, cfg["lmcld"])
    if tr is not None and (i, j) in tr:
        import copy
        tr[(i, j)].update(S0=copy.deepcopy(S), P0=P)
    lsm = ls0 if kmax <= 4 else _lscond_allk()
    S, W, _ = lsm.lscond(S, P)
    ms["S"] = S
    if W["ierr"] != 0:
        cnt["ls_ierr"] += 1
    prcpss = F(W["prcpss"])
    lhp = np.asarray(S["lhp"])
    prcp = prcp + prcpss * 100.0 * BYGRAV
    if lhp[0] != LHS:
        eprcp = 0.0
        enrgp = enrgp + eprcp
    else:
        eprcp = 0.0
        enrgp = enrgp + eprcp - prcpss * 100.0 * BYGRAV * LHM
        cnt["lhp1_ice"] += 1
    if enrgp < 0.0 and not mut.get("no_snow_age"):
        cnt["snow_age"] += 1
        for it in range(3):
            X["SNOAGE"][it, i, j] = X["SNOAGE"][it, i, j] * float(sz.ex(-prcp))
    # ------------------------------------------------------------------ stores (1932-1945)
    cldmcl = np.asarray(o["cldmcl"])
    X["TAUMC"][:, i, j] = np.asarray(o["taumcl"])
    X["CLDMC"][:, i, j] = cldmcl
    X["SVLAT"][:, i, j] = svlatl
    cldssl = np.asarray(S["cldssl"])
    X["TAUSS"][:, i, j] = np.asarray(S["taussl"])
    X["CLDSS"][:, i, j] = cldssl
    X["CLDSAV"][:, i, j] = np.asarray(S["cldsavl"])
    X["CLDSAV1"][:, i, j] = np.asarray(S["cldsv1"])
    X["SVLHX"][:, i, j] = np.asarray(S["svlhxl"])
    X["CSIZSS"][:, i, j] = np.asarray(S["csizel"])
    X["QLSS"][:, i, j] = np.asarray(S["qlss"])
    X["QISS"][:, i, j] = np.asarray(S["qiss"])
    X["QLMC"][:, i, j] = np.asarray(o["qlmc"])
    X["QIMC"][:, i, j] = np.asarray(o["qimc"])
    qclx_o, qcix_o = np.asarray(S["qclx"]), np.asarray(S["qcix"])
    _hand_off(X, i, j, cldmcl, cldssl, svlatl, np.asarray(S["svlhxl"]), qclx_o, qcix_o, np.asarray(o["cnvmmrl"]),
              np.asarray(S["cldsal"]), cnt, mut)
    X["TAUSSIP"][:, i, j] = np.asarray(S["tausslip"])
    X["CSIZSSIP"][:, i, j] = np.asarray(S["csizelip"])
    X["RHSAV"][:, i, j] = np.asarray(S["rh"])
    th_o, ql_o = np.asarray(S["th"]), np.asarray(S["ql"])
    X["TTOLD"][:, i, j] = th_o
    X["QTOLD"][:, i, j] = ql_o
    X["PREC"][i, j] = prcp
    X["EPREC"][i, j] = enrgp
    precss = prcpss * 100.0 * BYGRAV
    X["PRECSS"][i, j] = precss
    X["P_ACC"][i, j] = X["P_ACC"][i, j] + prcp
    X["PM_ACC"][i, j] = X["PM_ACC"][i, j] + prcp - precss
    # ------------------------------------------------------------------ final merge (2152-2185)
    smom_ls, qmom_ls = np.asarray(S["smom"]).T, np.asarray(S["qmom"]).T       # (9, LM)
    if mut.get("merge_swap"):
        fssl_ls = 1.0 - fssl_ls
    X["T"][i, j, :] = th_o * fssl_ls + tmc * (1.0 - fssl_ls)
    X["Q"][i, j, :] = ql_o * fssl_ls + qmc * (1.0 - fssl_ls)
    smom_f = smom_ls * fssl_ls + smom_mc * (1.0 - fssl_ls)
    qmom_f = qmom_ls * fssl_ls + qmom_mc * (1.0 - fssl_ls)
    X["TMOM"][:, i, j, :] = smom_f * byam
    X["QMOM"][:, i, j, :] = qmom_f * byam
    X["QCI"][i, j, :] = qcix_o
    X["QCL"][i, j, :] = qclx_o
    X["TMC"][i, j, :] = tmc
    X["QMC"][i, j, :] = qmc
    ums = np.asarray(S["um"]).T                                                   # (K, LM)
    vms = np.asarray(S["vm"]).T
    if kmax > ums.shape[0]:
        raise RuntimeError("UM rows lost")
    du = (ums * fssl_ls + um1 * (1.0 - fssl_ls)) * byam - u0
    dv = (vms * fssl_ls + vm1 * (1.0 - fssl_ls)) * byam - v0
    if j == 0:
        X["UKMSP"][:, :] = du
        X["VKMSP"][:, :] = dv
    elif j == JM - 1:
        X["UKMNP"][:, :] = du
        X["VKMNP"][:, :] = dv
    else:
        X["UKM"][:kmax, :, i, j] = du
        X["VKM"][:kmax, :, i, j] = dv
    cnt["columns"] += 1
    return o, S, W


# ------------------------------------------------------------------------------------------------ whole step
def _fresh_outputs(inp):
    keys3 = ("TTOLD QTOLD SVLHX SVLAT RHSAV CLDSAV CLDSAV1 FSS TAUSS TAUSSIP TAUMC CLDSS CLDMC CSIZMC CSIZSS CSIZSSIP QLSS QISS QLMC "
             "QIMC W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE "
             "DIM_ST_WATER DIM_ST_ICE DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV LMC SNOAGE TMOM QMOM UKM VKM UKMSP VKMSP UKMNP "
             "VKMNP").split()
    keys2 = "P_ACC PM_ACC PREC EPREC PRECSS DDM1 DDMS TDN1 QDN1 DDML AIRX".split()
    keys_old = "T Q QCL QCI U V".split()
    X = {k: inp[k].copy() for k in keys3 + keys2 + keys_old}
    X["FSS"][:] = 1.0                      # FSS=1. at CONDSE:580
    X["TLS"] = inp["T"].copy()
    X["QLS"] = inp["Q"].copy()
    X["TMC"] = inp["T"].copy()
    X["QMC"] = inp["Q"].copy()
    return X


def condse_step(inp, cfg, cols=None, cnt=None, br=None, with_momentum=True, mut=None, ms=None):
    """Run CONDSE for one step from the entry arrays `inp`.  cols: optional iterable of (i, j) 0-based to restrict the work (the
    persistent module state emulation is then approximate).  ms: optional dict holding the LSCOND module arrays left by the last column of
    the previous step (the Fortran module keeps them across steps; the very first column of a window starts from zeros).  -> (X, counts)."""
    G = cfg["geom"]
    X = _fresh_outputs(inp)
    cnt = cnt if cnt is not None else new_counts()
    ms = ms if ms is not None else _zeros_ms()      # a dict passed in carries the module state of the previous step (and is updated)
    order = cols if cols is not None else [(i, j) for j in range(JM) for i in range(int(G["IMAXJ"][j]))]
    for (i, j) in order:
        condse_column(X, inp, G, cfg, i, j, ms, cnt, br=br, mut=mut)
    if with_momentum and cols is None:
        avg_replicated_duv_to_vgrid(X["U"], X["V"], X["UKM"], X["VKM"], X["UKMSP"], X["VKMSP"], X["UKMNP"], X["VKMNP"])
        import dyn_glue_ff as gf
        X["UALIJ"], X["VALIJ"] = gf.recalc_agrid_uv(X["U"], X["V"], cfg["glue_geom"])
    return X, cnt


def make_cfg(date, ff=cio.FF_DEFAULT, glue_geom=None):
    c = cio.load_consts(date, ff)
    d = init_cld_constants(c["dtsrc"])
    cfg = dict(dtsrc=c["dtsrc"], lmcld=c["lmcld"], lmcm=c["lmcm"], isc=c["isc"], tune=dict(RUN_TUNE), geom=cio.load_geom(date, ff),
               **d)
    if glue_geom is None:
        import dyn_glue_io as gio
        glue_geom = gio.load_g(date, ff)
    cfg["glue_geom"] = glue_geom
    return cfg
