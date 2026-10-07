"""D163: minimal AIJ-style diagnostics for the F3 monthly comparison, computed from the chained atmosphere state.

What this is.  The real model accumulates its monthly diagnostics (the `aij`, `aijl` arrays of the *.acc files) by calls scattered
through the step (list in scoping/D163_F3_DIAGNOSTICS_ENTRY.md).  This module re-implements the accumulation of a chosen subset of
them, each at the same call site and with the same sampling rule as the Fortran, as methods of `F3Acc` that take the model state at
that site:

  site            Fortran                                   sampling (idacc counter)                      method
  DIAGA           DIAG.f:98-856, called from DYNAM          MODDA<2 of the even leap-frog pass            F3Acc.diaga       (ia_dga=4)
                  (ATMDYN.f:352-356), 4 times per 54 steps  (dyn_step.dynam_plan finds the pass)
  accum_ma_ia_src DIAG.f:1408-1423 (ATM_DRV.f:504)          every step (ia_src=1)                         F3Acc.airmass
  RADIA           RAD_DRV.f:4750-4790 (radiation steps),    radiation step (ia_rad=2) / every step        F3Acc.radia
                  RAD_DRV.f:5479 (SRINCP0, every step)
  CONDSE          CLOUDS2_DRV.F90:1474 (IJ_PREC)            every step (ia_src=1)                         F3Acc.prec

Everything is kept in the model layout (IM,JM) per AIJ column (1-based column number of the 1660-column `aij` of this build, names
read from the real acc files, see AIJ_COLS); `to_nc_layout` gives the (KAIJ,JM,IM) array of the acc files.

What it is NOT (honest limits, details in the ledger entry): the SURFACE-site fields (tsurf, tgrnd, qsurf, usurf..., pblht, evap,
sensht, ...) are not ported (they need the per-substep PBL/tile composites of SURFACE.f:1700-1830, sampled on every third surface
substep); the ocean/ice/land-surface fields, the conservation (consrv) quantities and ISCCP/cloud-overlap fields are not ported;
the RADIA contributions are not computed here, they are the radiation server's AIJ output (the real, unmodified RADIA).
SOCRATES/RADIA is never ported.

Sampling-time caveat: DIAGA reads the state in the middle of the dynamics (after the PGF/SDRAG of an even leap-frog pass) and the
surface air temperature/humidity of the *previous* SURFACE step (atmsrf%TSAVG/QSAVG); `diaga_states` gets the first from
`dyn_step.dyn_step` through its stage hook (bitwise with the real model on the real step-start state) and the second from the
recorded ffa_step_<it>_a dump (step start) or from the chained step state.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IM, JM, LM = 72, 46, 40
KAIJ = 1660
KGZ = 20
FF = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"

# model constants (Constants_mod.F90, same values as clouds_dq_ff / clouds_helpers_ff)
TF = 273.15
LHE = 2.5e6

P1000 = 1000.0

PMB = np.array([1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 70., 50., 30., 20., 10., 5., 1., .5])
PMNAME = ["1000", "925", "850", "700", "600", "500", "400", "300", "250", "200", "150", "100", "70", "50", "30", "20", "10", "5", "1", "p5"]

# ------------------------------------------------------------------------------------------------------ AIJ column table
# 1-based column numbers of this build's aij (kaij=1660), names/ia/scale/denom as stored in the real acc files
# (JAN1950.accP2SAoM40.nc sname_aij/ia_aij/scale_aij/denom_aij); verified by tests/test_f3_diagnostics.py when the acc file is present.
AIJ_COLS = {
    "pres": None,           # IJ_PRES: no output name in the acc file ('prsurf' is the diagnostic that carries it, see below)
    "prsurf": 151, "prsurfq": 152, "slp": 153, "slpq": 154,
    "rh_layer1": 97, "qatm": 100,
    "prec": 315,
    "incsw_toa": 376, "srnf_toa": 377, "srnf_grnd": 385, "trnf_toa": 391,
    "tsurf": 182, "qsurf": 76, "trdn_surf": 395, "evap": 322, "tauus": 291, "tauvs": 292,
}
for _k, _n in enumerate(PMNAME):
    AIJ_COLS["p_freq_" + _n] = 13 + _k
    AIJ_COLS["q_" + _n] = 56 + _k
    AIJ_COLS["rh_" + _n] = 77 + _k
    AIJ_COLS["t_" + _n] = 162 + _k
    AIJ_COLS["z_" + _n] = 217 + _k
    AIJ_COLS["u_" + _n] = 246 + _k
    AIJ_COLS["v_" + _n] = 266 + _k
    AIJ_COLS["omega_" + _n] = 294 + _k
del AIJ_COLS["pres"]
IA = {"src": 1, "rad": 2, "srf": 3, "dga": 4}
# aijl columns (1-based) of the 24 of the acc file used here
AIJL_COLS = {"TempL": 19, "SpHuL": 20, "z": 21, "airmass": 22}

# fields we claim as ported by this module (the F3 minimal set), in the order used by the ledger table
PORTED_NAMES = (["prsurf", "prsurfq", "slp", "slpq", "rh_layer1", "qatm"]
                + [f"{v}_{p}" for v in ("t", "z", "u", "v", "q", "rh", "omega", "p_freq") for p in PMNAME]
                + ["prec", "incsw_toa", "srnf_toa", "srnf_grnd", "trnf_toa", "tsurf", "qsurf", "trdn_surf", "evap", "tauus", "tauvs"])


def _valid_mask():
    m = np.ones((IM, JM), bool)
    m[1:, 0] = False                         # pole rows: IMAXJ = 1
    m[1:, JM - 1] = False
    return m


VALID = _valid_mask()


def _consts():
    import clouds_dq_ff as dq
    import clouds_helpers_ff as ch
    return dq, ch


def aij_names_from_nc(path):
    """names/ia/scale/denom of the 1660 columns from a real acc file (1-based index -> dict)."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    sn = [b"".join(x).decode().strip() for x in d["sname_aij"][:]]
    ia = np.asarray(d["ia_aij"][:]); sc = np.asarray(d["scale_aij"][:]); dn = np.asarray(d["denom_aij"][:])
    d.close()
    return {i + 1: dict(name=sn[i], ia=int(ia[i]), scale=float(sc[i]), denom=int(dn[i])) for i in range(len(sn))}


# ------------------------------------------------------------------------------------------------------ the accumulator
class F3Acc:
    """AIJ/AIJL accumulators of the ported subset.  `aij[col]` is an (IM,JM) float64 array (col = 1-based), `aijl[col]` (IM,JM,LM)."""

    def __init__(self, geo=None):
        self.geo = geo or {}
        self.aij = {}
        self.aijl = {}
        self.idacc = {1: 0, 2: 0, 3: 0, 4: 0}
        self.s0 = None                       # solar constant factor of the last radiation step (RADIA's S0, persistent)

    # -- helpers
    def _a(self, col):
        if col not in self.aij:
            self.aij[col] = np.zeros((IM, JM))
        return self.aij[col]

    def _al(self, col):
        if col not in self.aijl:
            self.aijl[col] = np.zeros((IM, JM, LM))
        return self.aijl[col]

    def to_nc_layout(self):
        """(KAIJ,JM,IM) array like the acc files (columns not ported are zero) and (24,LM,JM,IM) aijl."""
        a = np.zeros((KAIJ, JM, IM))
        for c, x in self.aij.items():
            a[c - 1] = x.T
        al = np.zeros((24, LM, JM, IM))
        for c, x in self.aijl.items():
            al[c - 1] = np.transpose(x, (2, 1, 0))
        return a, al

    # -- every step, ia_src
    def airmass(self, ma):
        """accum_ma_ia_src (DIAG.f:1408-1423): AIJL(ijl_airmass) += MA at the end of the step.  ma (LM,IM,JM)."""
        al = self._al(AIJL_COLS["airmass"])
        al += np.where(VALID[:, :, None], np.transpose(ma, (1, 2, 0)), 0.0)
        self.idacc[1] += 1

    def prec(self, prcp):
        """CLOUDS2_DRV.F90:1474 AIJ(IJ_PREC) += PRCP (the CONDSE exit export PREC, kg m-2 per step)."""
        self._a(AIJ_COLS["prec"])[...] += np.where(VALID, prcp, 0.0)

    # -- RADIA
    def radia(self, cosz1, rad_step, aij_server=None, columns=None):
        """RAD_DRV.f:5479: AIJ(IJ_SRINCP0) += S0*COSZ1 on every step (S0 of the last radiation step, persistent module value).
        On a radiation step (MODRD==0) the whole RADIA increment is the real RADIA's own AIJ delta (the server output `AIJ`/`AIJD`,
        shape (IM,JM,KAIJ), the delta of the call, which already contains that step's SRINCP0 term); `columns` (1-based) restricts
        which columns are added (default: every column that is non-zero).  On other steps only the SRINCP0 term exists (the other RADIA
        diagnostics are inside the radiation block, RAD_DRV.f:2502)."""
        if rad_step:
            cols = columns or [c for c in range(1, aij_server.shape[2] + 1) if np.any(aij_server[:, :, c - 1])]
            for c in cols:
                self._a(c)[...] += np.where(VALID, aij_server[:, :, c - 1], 0.0)
            # S0 recovered from the server's own SRINCP0 (= S0*COSZ1) where the sun is up: a derived value, see the ledger
            sr = aij_server[:, :, AIJ_COLS["incsw_toa"] - 1]
            m = VALID & (cosz1 > 0)
            self.s0 = float(np.median(sr[m] / cosz1[m])) if m.any() else self.s0
            self.idacc[2] += 1
        else:
            if self.s0 is None:
                raise ValueError("S0 unknown: the first step must be a radiation step")
            self._a(AIJ_COLS["incsw_toa"])[...] += np.where(VALID, self.s0 * cosz1, 0.0)

    # -- SURFACE (sampled)
    @staticmethod
    def surface_samples(itime, nisurf=2, ndasf=1):
        """SURFACE.f:386-387: MODDSF = MOD(NSTEPS+NS-1, NDASF*NIsurf+1), NSTEPS = NIsurf*ITime; the diagnostics of the substep are
        accumulated (and IDACC(ia_srf) incremented) when MODDSF == 0.  Returns the list of sampled substeps NS (1-based)."""
        return [ns for ns in range(1, nisurf + 1) if (nisurf * itime + ns - 1) % (ndasf * nisurf + 1) == 0]

    def surface(self, itime, ns_dump, trhr0, dtsurf=900.0):
        """SURFACE.f:1745-1801 for: evap (IJ_EVAP -= DTSURF*QFLUX1, every substep, line 1749); on the sampled substeps (MODDSF==0)
        tsurf (IJ_TS += TSAVG-TF), qsurf (IJ_QS += QSAVG), tauus/tauvs (IJ_TAUUS/IJ_TAUVS += UFLUX1/VFLUX1), trdn_surf
        (IJ_TRSDN += TRHR(0)).  ns_dump: dict NS -> dict(TSAVG,QSAVG,QFLUX1,UFLUX1,VFLUX1) (the ATURB exit dumps of that substep);
        trhr0: TRHR(0,:,:) (IM,JM)."""
        c = AIJ_COLS
        for ns in sorted(ns_dump):
            self._a(c["evap"])[...] += np.where(VALID, -dtsurf * ns_dump[ns]["QFLUX1"], 0.0)
        for ns in self.surface_samples(itime):
            d = ns_dump[ns]
            self.idacc[3] += 1
            self._a(c["tsurf"])[...] += np.where(VALID, d["TSAVG"] - TF, 0.0)
            self._a(c["qsurf"])[...] += np.where(VALID, d["QSAVG"], 0.0)
            self._a(c["tauus"])[...] += np.where(VALID, d["UFLUX1"], 0.0)
            self._a(c["tauvs"])[...] += np.where(VALID, d["VFLUX1"], 0.0)
            self._a(c["trdn_surf"])[...] += np.where(VALID, trhr0, 0.0)

    # -- DIAGA
    def diaga(self, w, geo, tsavg, qsavg, slp_from_t1=False, imf_pow=False):
        """DIAG.f:98-856 restricted to the ported fields.  `w`: dyn_step workspace at the DIAGA stage (U V T Q QCL QCI PK PMID PEDN MA
        MASUM PHI MW, Fortran shapes: U..QCI (IM,JM,LM), MA/PK/PMID (LM,IM,JM), PEDN (LM+1,IM,JM), MW (IM,JM,LM-1)).
        `geo`: dict with zatmo (IM,JM), axyp (IM,JM), grav, kg2mb, fg (dyn_filter_ff constants for SLP) and `ua_va` function
        (u,v)->(UA,VA) (LM,IM,JM) = recalc_agrid_uv.  tsavg/qsavg: atmsrf%TSAVG/QSAVG at the time of the call (IM,JM)."""
        import dyn_filter_ff as ffl
        dq, ch = _consts()
        grav, kg2mb, bygrav = geo["grav"], geo["kg2mb"], geo["bygrav"]
        lhe = ch.LHE
        lhs = ch.LHS
        v = VALID
        # state in (L,IM,JM)
        T = np.array(w["T"], float); Q = np.array(w["Q"], float)
        T[1:, 0, :] = T[0, 0, :][None, :]; T[1:, JM - 1, :] = T[0, JM - 1, :][None, :]     # pole fill (DIAG.f:236-253)
        Q[1:, 0, :] = Q[0, 0, :][None, :]; Q[1:, JM - 1, :] = Q[0, JM - 1, :][None, :]     # (DIAG.f:221-233)
        pk = np.asarray(w["PK"], float); pmid = np.asarray(w["PMID"], float); pedn = np.asarray(w["PEDN"], float)
        ma = np.asarray(w["MA"], float)
        tx = np.transpose(T, (2, 0, 1)) * pk                                                  # (LM,IM,JM)
        qL = np.transpose(Q, (2, 0, 1))
        qcl = np.transpose(np.asarray(w["QCL"], float), (2, 0, 1)); qci = np.transpose(np.asarray(w["QCI"], float), (2, 0, 1))
        ua, va = geo["ua_va"](w["U"], w["V"])
        phi = np.transpose(np.asarray(w["PHI"], float), (2, 0, 1))
        zatmo = geo["zatmo"]
        ps = pedn[0]
        c = AIJ_COLS
        self.idacc[4] += 1
        # --- IJ_PRES, IJ_SLP, IJ_PRESQ, IJ_SLPQ, IJ_RH1 (DIAG.f:276-296)
        zs = bygrav * zatmo
        ts_slp = (T[:, :, 0] * np.asarray(w["PEK1"], float)) if slp_from_t1 else tsavg
        slp1 = ffl.slp(ps, ts_slp, zs, geo["fg"], imf_pow=imf_pow)
        self._a(c["prsurf"])[...] += np.where(v, ps, 0.0)
        self._a(c["slp"])[...] += np.where(v, slp1 - P1000, 0.0)
        acc = np.zeros((IM, JM))
        for L in range(LM):                                    # Sum((Q+QCL+QCI)*MA(:,I,J)): sequential in L
            acc = acc + (qL[L] + qcl[L] + qci[L]) * ma[L]
        psq = ps + acc * kg2mb
        self._a(c["prsurfq"])[...] += np.where(v, psq, 0.0)
        self._a(c["slpq"])[...] += np.where(v, ffl.slp(psq, ts_slp, zs, geo["fg"], imf_pow=imf_pow) - P1000, 0.0)
        self._a(c["rh_layer1"])[...] += np.where(v, qL[0] / dq.qsat(tx[0], lhe, pmid[0]), 0.0)
        # --- IJ_QM (DIAG.f:493) and the AIJL accumulations (DIAG.f:488-496), per layer
        qm = self._a(c["qatm"])
        al_t, al_q, al_z = self._al(AIJL_COLS["TempL"]), self._al(AIJL_COLS["SpHuL"]), self._al(AIJL_COLS["z"])
        for L in range(LM):
            qm += np.where(v, qL[L] * ma[L], 0.0)
            al_t[:, :, L] += np.where(v, tx[L], 0.0)
            al_q[:, :, L] += np.where(v, qL[L], 0.0)
            al_z[:, :, L] += np.where(v, phi[L] / grav, 0.0)
        # --- T, Q, Z, RH, U, V at constant pressure (DIAG.f:392-470): nodes = surface + layer midpoints
        def nodes(top, bot):
            return np.concatenate([bot[None], top], axis=0)                                   # (LM+1,IM,JM)
        pn = nodes(pmid, ps)
        tn = nodes(tx - TF, tsavg - TF)
        qn = nodes(qL, qsavg)
        zn = nodes(phi * bygrav, zatmo * bygrav)
        un = nodes(ua, np.zeros((IM, JM)))
        vn = nodes(va, np.zeros((IM, JM)))
        # --- omega nodes (DIAG.f:472-496): surface + layer edges PEDN(L+1), MW(L)*GRAV*byAXYP (0 at the top layer)
        ofac = grav * (1.0 / geo["axyp"])
        mw = np.asarray(w["MW"], float)                                                       # (IM,JM,LM-1)
        on = np.zeros((LM + 1, IM, JM))
        on[1:LM] = np.transpose(mw, (2, 0, 1)) * ofac
        po = nodes(pedn[1:], ps)
        for k in range(KGZ):
            pk_ = PMB[k]
            # ---- T,Q,U,V,Z,RH
            cnt = (pmid > pk_).sum(axis=0)                  # m = number of layers with PMID > PMB (monotone): UP = layer m+1
            ok = (ps >= pk_) & (cnt < LM) & v
            m = np.minimum(cnt, LM - 1)
            take = lambda a, idx: np.take_along_axis(a, idx[None], axis=0)[0]
            pdn, pup = take(pn, m), take(pn, m + 1)
            fr = (pk_ - pdn) / (pup - pdn)

            def lin(a):
                dn, up = take(a, m), take(a, m + 1)
                return dn + (up - dn) * (pk_ - pdn) / (pup - pdn)
            with np.errstate(all="ignore"):
                tij, qij, uij, vij = lin(tn), lin(qn), lin(un), lin(vn)
                zdn, zup = take(zn, m), take(zn, m + 1)
                zij = zdn + (zup - zdn) * np.log(pk_ / pdn) / np.log(pup / pdn)
                rh = np.where(tij >= 0, qij / dq.qsat(tij + TF, lhe, pk_), qij / dq.qsat(tij + TF, lhs, pk_))
            nm = PMNAME[k]
            for key, val in (("p_freq_", np.ones((IM, JM))), ("t_", tij), ("q_", qij), ("z_", zij), ("u_", uij), ("v_", vij), ("rh_", rh)):
                self._a(c[key + nm])[...] += np.where(ok, val, 0.0)
            # ---- omega
            cnto = (po[1:] > pk_).sum(axis=0)               # nodes l=1..LM with PEDN(l+1) > PMB
            oko = (ps >= pk_) & (cnto < LM) & v
            mo = np.minimum(cnto, LM - 1)
            pdn_o, pup_o = take(po, mo), take(po, mo + 1)
            with np.errstate(all="ignore"):
                dn, up = take(on, mo), take(on, mo + 1)
                oij = dn + (up - dn) * (pk_ - pdn_o) / (pup_o - pdn_o)
            self._a(c["omega_" + nm])[...] += np.where(oko, oij, 0.0)


# ------------------------------------------------------------------------------------------------------ monthly means
def field_from_aij(aij_col, idacc, meta, aij_all=None):
    """DIAG_PRT.f:3119-3131 (ij_mapk): numerator = aij*scale/(idacc(ia)+teeny); if denom>0 the denominator is
    aij(denom)/(idacc(ia(denom))+teeny); the map is numerator/denominator.  `meta`: dict(ia,scale,denom,denom_ia) for the column.
    Returns (anum, adenom).  teeny = 1e-30 (Constants_mod, value taken from the model, not re-derived here)."""
    teeny = 1e-30
    anum = aij_col * (meta["scale"] / (idacc[meta["ia"] - 1] + teeny))
    if meta["denom"] > 0:
        adenom = aij_all[meta["denom"] - 1] / (idacc[meta["denom_ia"] - 1] + teeny)
    else:
        adenom = np.ones_like(anum)
    return anum, adenom


def global_mean(anum, adenom, axyp):
    """area-weighted global mean of the ratio field (sum(anum*w)/sum(adenom*w)); a plain restatement of the ij_avg weighting
    (not verified line by line against DIAG_PRT.f ij_avg)."""
    w = axyp
    return float((anum * w).sum() / (adenom * w).sum())


# ------------------------------------------------------------------------------------------------------ the 54-step window driver
DAY = "nov26_day"
IT0, NSTEP_DAY = 33312, 54
ITIMEI = 16032
REAL_ACC = f"{FF}/{DAY}/real_acc54_nov26.npz"


def load_geo(ff=FF, date=DAY):
    """geometry/constants for DIAGA from the recorded one-time dumps."""
    import dyn_step as ds
    import dyn_filter_ff as _f
    import intel_libm_ff
    ctx = ds.load_ctx(date, ff, imf_pow=bool(intel_libm_ff.available()))
    g = ctx.g
    geo = dict(zatmo=np.asarray(ctx.gg["zatmo"]), axyp=np.asarray(ctx.gg["axyp"]), grav=float(g["grav"]), bygrav=float(g["bygrav"]),
               kg2mb=float(g["kg2mb"]), fg=_f.load_consts(f"{ff}/{date}/ffd_filt_consts.bin"))
    import dyn_glue_ff as gf
    geo["ua_va"] = lambda u, v: gf.recalc_agrid_uv(u, v, ctx.gg)
    return ctx, geo


def diaga_fire_steps(it0=IT0, n=NSTEP_DAY):
    import dyn_step as ds
    return [it for it in range(it0, it0 + n) if any(s.kind == "diaga" for s in ds.step_plan(nstep=(it - ds.ITIMEI) * 4))]


def diaga_state(ctx, it, ff=FF, date=DAY, state=None):
    """Run the (bitwise) chained dynamics of step `it` from its start state and return the workspace at the DIAGA stage."""
    import dyn_step as ds
    st = state or ds.load_state(ds.state_path(date, it, 1, ff))
    cap = {}

    def hook(when, stg, w, c):
        if when == "post" and stg.kind == "diaga":
            cap.update({k: np.array(v, copy=True) for k, v in w.items() if isinstance(v, np.ndarray)})
    ds.dyn_step(st, ctx, itime=it, hook=hook)
    return cap


def run_window(ff=FF, date=DAY, it0=IT0, nsteps=NSTEP_DAY, steps_diaga=None, log=print):
    """Accumulate the ported fields over the window from the REAL per-step states (start states, ffa dumps, CONDSE exits, radiation
    server packets).  Returns F3Acc."""
    import clouds_condse_io as cio
    import ffdump_reader as fr
    import radiation_server as rs
    ctx, geo = load_geo(ff, date)
    acc = F3Acc(geo)
    fires = set(diaga_fire_steps(it0, nsteps) if steps_diaga is None else steps_diaga)
    for it in range(it0, it0 + nsteps):
        a = cio.read_cse(f"{ff}/{date}/ffa_step_{it}_a.bin")
        r = cio.read_cse(f"{ff}/{date}/ffa_step_{it}_r.bin")
        if it in fires:
            w = diaga_state(ctx, it, ff, date)
            w["PEK1"] = None  # only used with SLP_FROM_T1
            acc.diaga(w, geo, a["TSAVG"], a["QSAVG"], imf_pow=bool(ctx.imf_pow))
            log(f"  DIAGA at itime {it}")
        # RADIA site
        rad_step = (it - ITIMEI) % 5 == 0
        srv = None
        if rad_step:
            pth = f"{ff}/{date}/rsv_n26_{it}_out.bin"
            if os.path.exists(pth):
                srv = rs.read_packet(pth)["AIJD"]
        if rad_step and srv is None:
            raise FileNotFoundError(f"radiation packet for {it}")
        acc.radia(r["COSZ1"], rad_step, srv)
        # SURFACE site: atmsrf TSAVG/QSAVG of each substep (ATURB exit dumps ffa_<it>_c1/c2_out), TRHR(0) of the step
        sub = {ns: fr.read_dump(f"{ff}/{date}/ffa_{it}_c{ns}_out.bin", header_f8=1) for ns in (1, 2)}
        acc.surface(it, sub, r["TRHR"][0])
        # CONDSE exit PREC and end-of-step MA
        o = cio.read_cse(f"{ff}/{date}/ffc_cse_out_{it}.bin")
        acc.prec(o["PREC"])
        e = cio.read_cse(f"{ff}/{date}/ffa_step_{it}_e.bin")
        acc.airmass(e["MA"])
    return acc


def real_window(path=REAL_ACC):
    z = np.load(path)
    return dict(daij=z["daij"], daijl=z["daijl"], idacc_delta=(z["idacc_after"] - z["idacc_before"]))


def compare(acc, real, cols):
    """per column: max |ported - real| (absolute), max |real|, relative to the real max."""
    a, al = acc.to_nc_layout()
    out = {}
    for name, c in cols.items():
        d = np.abs(a[c - 1] - real["daij"][c - 1])
        sc = np.abs(real["daij"][c - 1]).max()
        out[name] = dict(col=c, maxabs=float(d.max()), real_maxabs=float(sc), rel=float(d.max() / sc) if sc > 0 else (0.0 if d.max() == 0 else np.inf))
    return out
