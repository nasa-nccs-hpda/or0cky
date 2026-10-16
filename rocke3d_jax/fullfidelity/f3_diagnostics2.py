"""D172: second set of AIJ-style accumulators for the F3 monthly comparison (extends f3_diagnostics.F3Acc, which is not edited).

Groups (each validated against the real model's own 54-step accumulation `real_acc54_nov26.npz`, see scoping/D172_F3_DIAG2_ENTRY.md):

  surface-site, sampled on 1 substep in 3 (SURFACE.f:1759-1819, surface_diag1):
      usurf 286, vsurf 287, wsurf 288, gusti 289, pblht 237, tausmag 293, RHsurf 98, tgrnd 184, mccon 142 (DDMS<0, from CONDSE exit)
      = area-fraction (ftype) weighted sums over the four tile types of the PBL exports (PBL_DRV.f:420-430) / ground temperature.
  surface fluxes, every step (surface_diag1a, SURFACE.f:2010-2110; sums of the two substeps' tile fluxes, ftype weighted):
      sensht 356, sensht_lndice 357, sh_oice 358, evap_ocn 325, evap_oice 326, evap_lndice 324, lwd_oice 400, lwu_oice 401,
      trht_lndice 399, latht_lndice 360
  CONDSE (CLOUDS2_DRV.F90 :937-962, 1135-1137, 1463, 1537), every step:
      prec_mc 321 (PREC-PRECSS), snowfall 333 (-EPREC/LHM), cldw 101, cnvfrq 468, mccvbs 54, mccldbs 150
  dynamics (ATM_UTILS.f:364, ATM_DRV.f:186-202): ptrop 157, ttrop 158  (dSE_Dyn/dKE_Dyn/dTE_Dyn were tried: 5.5%/6.3%/98% of scale off, NOT included)

Inputs are 'tile packets' (see `packet_recorded`, `packet_chained`): per tile type the rows (i,j), PBL outputs and tile fluxes of one substep,
either the real dumped records (validation mode) or the values our chained port computed (chained mode).
NOT implemented (reasons in the ledger): sst/sss/ssh/sivol/simass and the land/ocean/ice/lake state columns, pr_*/evap_* land breakdown,
runoff, netht_* (e0 of ice/landice tiles incl. precipitation energy not dumped), srtrnf_grnd (solar/trheat composite did not match, 1.49
relative), clwp, cldi (needs ice-precip WMPR), pscld/pdcld (CLDREF), mccvtp/mccldtp (LMCMAX convention did not match), the land-model
diagnostics, ISCCP, aj/ajl/consrv.  SOCRATES/RADIA is not touched: RADIA columns are the real RADIA output (recorded packets).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f3_diagnostics as f3

IM, JM, LM = 72, 46, 40
VALID = f3.VALID
RGAS = 287.04873038614903
G = 9.80665
LHM = 2.834e6 - 2.5e6

COLS2 = {"usurf": 286, "vsurf": 287, "wsurf": 288, "gusti": 289, "pblht": 237, "tausmag": 293, "RHsurf": 98, "tgrnd": 184, "mccon": 142,
         "sensht": 356, "sensht_lndice": 357, "sh_oice": 358, "evap_ocn": 325, "evap_oice": 326, "evap_lndice": 324, "lwd_oice": 400,
         "lwu_oice": 401, "trht_lndice": 399, "latht_lndice": 360,
         "prec_mc": 321, "snowfall": 333, "cldw": 101, "cnvfrq": 468, "mccvbs": 54, "mccldbs": 150,
         "ptrop": 157, "ttrop": 158}
SURFACE_SITE = ["usurf", "vsurf", "wsurf", "gusti", "pblht", "tausmag", "RHsurf", "tgrnd", "mccon"]
SURFACE_FLUX = ["sensht", "sensht_lndice", "sh_oice", "evap_ocn", "evap_oice", "evap_lndice", "lwd_oice", "lwu_oice", "trht_lndice", "latht_lndice"]
CONDSE_COLS = ["prec_mc", "snowfall", "cldw", "cnvfrq", "mccvbs", "mccldbs"]
DYN_COLS = ["ptrop", "ttrop"]


# ------------------------------------------------------------------------------------------------------ tile packets
def _grid_ftype(blk):
    """(4,IM,JM) ftype of the tile types (ocean, ocean ice, land ice, land) from an fft block (N,40)."""
    F = np.zeros((4, IM, JM))
    i = blk[:, 0].astype(int) - 1
    j = blk[:, 1].astype(int) - 1
    for k in range(4):
        F[k, i, j] = blk[:, 2 + 7 * k]
    return F


def _qsat(t, lh, p):
    import clouds_dq_ff as dq
    return dq.qsat(t, lh, p)


def _tile(rows_ij, itype, us, vs, ws, gusti, dbl, tsv, qsrf, cm, psurf, elhx, tg1, shdt, evhdt, trhdt, evap):
    n = len(rows_ij)
    return dict(i=rows_ij[:, 0].astype(int), j=rows_ij[:, 1].astype(int), itype=itype,
                us=np.asarray(us, float), vs=np.asarray(vs, float), ws=np.asarray(ws, float), gusti=np.asarray(gusti, float),
                dbl=np.asarray(dbl, float), tsv=np.asarray(tsv, float), qsrf=np.asarray(qsrf, float), cm=np.asarray(cm, float),
                psurf=np.asarray(psurf, float), elhx=np.asarray(elhx, float), tg1=np.asarray(tg1, float),
                shdt=np.asarray(shdt, float), evhdt=np.asarray(evhdt, float), trhdt=np.asarray(trhdt, float), evap=np.asarray(evap, float))


def packet_recorded(rec, ns):
    """Tile packet of substep ns (1,2) from the REAL records of atm_step.surface_records(R): ffp (PBL), ffs (ocean/ice tile), ffl (land ice),
    ffg (land), fft (ftype).  Returns dict(tiles=[...], F=(4,IM,JM))."""
    p = rec["pa"] if ns == 1 else rec["pb"]
    t = rec["ta"] if ns == 1 else rec["tb"]
    l = rec["la"] if ns == 1 else rec["lb"]
    g = rec["g1"] if ns == 1 else rec["g2"]
    blk = rec["blk1"] if ns == 1 else rec["blk2"]
    tiles = []
    for it in (1, 2):
        m = p[:, 2] == it
        pp = p[m]
        tt = t[t[:, 2] == it]
        assert np.array_equal(pp[:, :2], tt[:, :2])
        tiles.append(_tile(pp, it, pp[:, 89], pp[:, 90], pp[:, 91], pp[:, 113], pp[:, 29], pp[:, 92], pp[:, 93], pp[:, 94], pp[:, 16],
                           pp[:, 19], tt[:, 59], tt[:, 66], tt[:, 67], tt[:, 68], tt[:, 69]))
    pp = p[p[:, 2] == 3]
    tiles.append(_tile(pp, 3, pp[:, 89], pp[:, 90], pp[:, 91], pp[:, 113], pp[:, 29], pp[:, 92], pp[:, 93], pp[:, 94], pp[:, 16], pp[:, 19],
                       l[:, 29], l[:, 34], l[:, 35], l[:, 36], l[:, 37]))
    pp = p[p[:, 2] == 4]
    assert np.array_equal(pp[:, :2], g[:, :2])
    tiles.append(_tile(pp, 4, pp[:, 89], pp[:, 90], pp[:, 91], pp[:, 113], pp[:, 29], pp[:, 92], pp[:, 93], pp[:, 94], pp[:, 16], pp[:, 19],
                       g[:, 246], -g[:, 247], -g[:, 248], g[:, 150] * 900.0 * 0 + 0.0, g[:, 249]))
    return dict(tiles=tiles, F=_grid_ftype(blk))


def packet_chained(call, blk):
    """Tile packet from one `atm_step._substep` call recorded by `ChainedHooks` (our PBL outputs, our tile fluxes, our GHY for the land)."""
    pbl12, pbl3, p4 = call["pbl12"], call["pbl3"], call["p4"]
    r = call["res"]
    tiles = []
    for it in (1, 2):
        m = pbl12[:, 2] == it
        pp = pbl12[m]
        o, got = r["pbl"], r["tile"]
        sel = lambda a: np.asarray(a)[m]
        tiles.append(_tile(pp, it, sel(o["us"]), sel(o["vs"]), sel(o["ws"]), pp[:, 113], pp[:, 29], sel(o["tsv"]), sel(o["qsrf"]), sel(o["cm"]),
                           pp[:, 16], pp[:, 19], sel(got["tg1"]), sel(got["shdt"]), sel(got["evhdt"]), sel(got["trhdt"]), sel(got["evap"])))
    o, got = r["pbl_li"], r["li"]
    tiles.append(_tile(pbl3, 3, o["us"], o["vs"], o["ws"], pbl3[:, 113], pbl3[:, 29], o["tsv"], o["qsrf"], o["cm"], pbl3[:, 16], pbl3[:, 19],
                       got["tg1"], got["shdt"], got["evhdt"], got["trhdt"], got["evap"]))
    lr = r["land"]
    o, gh = lr["pbl"], lr["ghy"]
    z = np.zeros(len(p4))
    tiles.append(_tile(p4, 4, o["us"], o["vs"], o["ws"], p4[:, 113], p4[:, 29], o["tsv"], o["qsrf"], o["cm"], p4[:, 16], p4[:, 19],
                       gh["tsns"], -np.asarray(gh["ashg"]), -np.asarray(gh["alhg"]), z, gh["aevap"]))
    return dict(tiles=tiles, F=_grid_ftype(blk))


def _csum(tile, F, vals):
    out = np.zeros((IM, JM))
    np.add.at(out, (tile["i"] - 1, tile["j"] - 1), F[tile["itype"] - 1, tile["i"] - 1, tile["j"] - 1] * vals)
    return out


def pbl_composites(pk):
    """ftype-weighted sums of the PBL exports over the tile types (PBL_DRV.f:420-430): usavg vsavg wsavg gustiwind dblavg tauavg rsavg and
    the ground temperature gtemps (SURFACE.f:1047-1050, GHY_DRV.f:1387)."""
    F = pk["F"]
    out = {k: np.zeros((IM, JM)) for k in ("usavg", "vsavg", "wsavg", "gusti", "dblavg", "tauavg", "rsavg", "gtemps")}
    for t in pk["tiles"]:
        rho = 100.0 * t["psurf"] / (RGAS * t["tsv"])
        for k, v in (("usavg", t["us"]), ("vsavg", t["vs"]), ("wsavg", t["ws"]), ("gusti", t["gusti"]), ("dblavg", t["dbl"]),
                     ("tauavg", t["cm"] * t["ws"] * t["ws"] * rho), ("rsavg", t["qsrf"] / _qsat(t["tsv"], t["elhx"], t["psurf"])),
                     ("gtemps", t["tg1"])):
            out[k] += _csum(t, F, v)
    return out


def flux_sums(pks):
    """Per tile type sums over the substeps of SHDT, EVHDT, TRHDT, EVAP and the ftype grid (the last substep's)."""
    S = {}
    for k in range(4):
        for q in ("shdt", "evhdt", "trhdt", "evap"):
            S[(q, k)] = np.zeros((IM, JM))
    for pk in pks:
        for t in pk["tiles"]:
            k = t["itype"] - 1
            for q in ("shdt", "evhdt", "trhdt", "evap"):
                np.add.at(S[(q, k)], (t["i"] - 1, t["j"] - 1), t[q])
    return S, pks[-1]["F"]


# ------------------------------------------------------------------------------------------------------ accumulator
class F3Acc2(f3.F3Acc):
    """F3Acc plus the D172 groups.  Counters: idacc[3] (surface samples) is incremented by the base `surface` (call it first)."""

    def surface_site(self, itime, packets_by_ns, ddms=None):
        """Sampled substeps only (MODDSF==0).  packets_by_ns: {ns: tile packet}; ddms: CONDSE exit DDMS (IM,JM) for mccon."""
        c = COLS2
        for ns in self.surface_samples(itime):
            comp = pbl_composites(packets_by_ns[ns])
            for name, key in (("usurf", "usavg"), ("vsurf", "vsavg"), ("wsurf", "wsavg"), ("gusti", "gusti"), ("pblht", "dblavg"),
                              ("RHsurf", "rsavg"), ("tgrnd", "gtemps")):
                self._a(c[name])[...] += np.where(VALID, comp[key], 0.0)
            self._a(c["tausmag"])[...] += np.where(VALID, comp["tauavg"], 0.0)
            if ddms is not None:
                self._a(c["mccon"])[...] += np.where(VALID & (ddms < 0), 1.0, 0.0)

    def surface_flux(self, packets, trhr0, dtsrc=1800.0):
        """Once per step: both substeps' packets, TRHR(0) of the step."""
        c = COLS2
        S, F = flux_sums(packets)
        a = lambda name, x: self._a(c[name]).__iadd__(np.where(VALID, x, 0.0))
        a("sensht", sum(F[k] * S[("shdt", k)] for k in range(4)))
        a("sensht_lndice", F[2] * S[("shdt", 2)])
        a("sh_oice", -F[1] * S[("shdt", 1)])
        a("evap_ocn", F[0] * S[("evap", 0)])
        a("evap_oice", F[1] * S[("evap", 1)])
        a("evap_lndice", F[2] * S[("evap", 2)])
        a("lwd_oice", F[1] * dtsrc * trhr0)
        a("lwu_oice", F[1] * (dtsrc * trhr0 - S[("trhdt", 1)]))
        a("trht_lndice", F[2] * S[("trhdt", 2)])
        a("latht_lndice", F[2] * S[("evhdt", 2)])

    def condse2(self, o, pdsig, pedn):
        """CONDSE exit dict o (PREC PRECSS EPREC QCL QCI LMC CLDMC), PDSIG (L,IM,JM), PEDN (L+1,IM,JM) of the CONDSE entry."""
        c = COLS2
        v = VALID
        self._a(c["prec_mc"])[...] += np.where(v, o["PREC"] - o["PRECSS"], 0.0)
        self._a(c["snowfall"])[...] += np.where(v, -o["EPREC"] / LHM, 0.0)
        ma = np.transpose(np.asarray(pdsig) * 100.0 / G, (1, 2, 0))
        self._a(c["cldw"])[...] += np.where(v, ((np.asarray(o["QCL"]) + np.asarray(o["QCI"])) * ma).sum(2), 0.0)
        lmin = np.asarray(o["LMC"])[0]
        on = lmin > 0
        cl = np.transpose(np.asarray(o["CLDMC"]), (1, 2, 0))
        pe = np.transpose(np.asarray(pedn), (1, 2, 0))
        ii, jj = np.meshgrid(range(IM), range(JM), indexing="ij")
        ln = np.clip(lmin.astype(int), 0, LM - 1)
        self._a(c["cnvfrq"])[...] += np.where(v & on, 1.0, 0.0)
        self._a(c["mccvbs"])[...] += np.where(v & on, cl[ii, jj, ln], 0.0)
        self._a(c["mccldbs"])[...] += np.where(v & on, pe[ii, jj, np.clip(lmin.astype(int), 0, LM)] * cl[ii, jj, ln], 0.0)

    def dyn2(self, w):
        """Dynamics workspace at the end of dyn_step (T after the energy fix, PK, PTROPO, LTROPO)."""
        c = COLS2
        v = VALID
        self._a(c["ptrop"])[...] += np.where(v, w["PTROPO"], 0.0)
        tl = np.asarray(w["T"]) * np.transpose(np.asarray(w["PK"]), (1, 2, 0))
        lt = np.asarray(w["LTROPO"]).astype(int)
        ii, jj = np.meshgrid(range(IM), range(JM), indexing="ij")
        self._a(c["ttrop"])[...] += np.where(v, tl[ii, jj, lt - 1], 0.0)


# ------------------------------------------------------------------------------------------------------ window drivers
def run_window2(groups=("surface_site", "surface_flux", "condse", "dyn"), ff=f3.FF, date=f3.DAY, it0=f3.IT0, nsteps=f3.NSTEP_DAY, log=print):
    """Real-input mode: accumulators fed with the REAL per-step records/states of the nov26 window."""
    import atm_step as A
    import clouds_condse_io as cio
    acc = F3Acc2()
    ctx = geo = None
    if "dyn" in groups:
        ctx, geo = f3.load_geo(ff, date)
        import dyn_step as ds
    for it in range(it0, it0 + nsteps):
        R = A.Real(date, it, ff)
        rec = A.surface_records(R)
        pk = {1: packet_recorded(rec, 1), 2: packet_recorded(rec, 2)}
        o = R.cse_out
        if "surface_site" in groups:
            acc.surface_site(it, pk, o["DDMS"])
            acc.idacc[3] += len(acc.surface_samples(it))      # the base `surface` (not called here) is what normally counts ia_srf
        if "surface_flux" in groups:
            r = cio.read_cse(f"{ff}/{date}/ffa_step_{it}_r.bin")
            acc.surface_flux([pk[1], pk[2]], r["TRHR"][0])
        if "condse" in groups:
            acc.condse2(o, R.cse_in["PDSIG"], R.cse_in["PEDN"])
        if "dyn" in groups:
            st = ds.load_state(ds.state_path(date, it, 1, ff))
            acc.dyn2(ds.dyn_step(st, ctx, itime=it))
        log(f"  window2 step {it}")
    return acc


def compare2(acc, real_daij, names=None):
    """per column: max|ours-real|, max|real|, relative (same convention as f3_diagnostics.compare)."""
    a, _ = acc.to_nc_layout()
    out = {}
    for n in (names or list(COLS2)):
        c = COLS2[n]
        if c not in acc.aij:
            continue
        d = np.abs(a[c - 1] - real_daij[c - 1])
        sc = np.abs(real_daij[c - 1]).max()
        out[n] = dict(col=c, maxabs=float(d.max()), real_maxabs=float(sc), rel=float(d.max() / sc) if sc > 0 else (0.0 if d.max() == 0 else np.inf))
    return out


# ------------------------------------------------------------------------------------------------------ chained driver
class ChainedHooks:
    """Collects, without editing atm_step.py, what the chained step does not export: the per-substep `_substep` inputs/results and the
    dynamics workspace.  `install()` swaps `atm_step.stage_dyn` / `atm_step._substep` for wrappers (same behaviour plus capture)."""

    def __init__(self, A):
        self.A = A
        self.calls = []
        self.w = None
        self.diaga_w = None
        self.start = {}
        self._orig_dyn, self._orig_sub = A.stage_dyn, A._substep

    def install(self):
        A, me = self.A, self

        def stage_dyn(S, R, ctx, tm=None):
            me.start = dict(TSAVG=np.array(S["TSAVG"], copy=True), QSAVG=np.array(S["QSAVG"], copy=True))
            cap = {}

            def hook(when, stg, w, c):
                if when == "post" and stg.kind == "diaga":
                    cap.update({k: np.array(v, copy=True) for k, v in w.items() if isinstance(v, np.ndarray)})
            w = A.ds.dyn_step({k: S[k.upper()] for k in A.ds.STATE_KEYS}, ctx.dyn, itime=R.itime, timing=tm, hook=hook)
            me.w = {k: np.array(v, copy=True) for k, v in w.items() if isinstance(v, np.ndarray)}
            me.diaga_w = cap or None
            for k in A.DYN_OUT:
                if k in w:
                    S[k] = np.array(w[k], copy=True)
            S["PEK"] = A._pow(ctx, S["PEDN"], ctx.kapa)
            return S

        def substep(pbl12, tile, pbl3, li, blk, atm, dt, land):
            res = me._orig_sub(pbl12, tile, pbl3, li, blk, atm, dt, land)
            me.calls.append(dict(pbl12=np.array(pbl12, copy=True), pbl3=np.array(pbl3, copy=True),
                                 p4=None if land is None else np.array(land["p4"], copy=True), res=res, blk=np.array(blk, copy=True)))
            return res
        A.stage_dyn, A._substep = stage_dyn, substep
        return self

    def uninstall(self):
        self.A.stage_dyn, self.A._substep = self._orig_dyn, self._orig_sub


def run_chained_window(nsteps=f3.NSTEP_DAY, it0=f3.IT0, date=f3.DAY, ff=f3.FF, log=print, out_npz=None):
    """Chained (not real-input) run from the real state at it0: dynamics, CONDSE, surface (PBL + tiles + GHY, Ent exports and land forcing
    recorded) from OUR previous end state (atm_step.run_step with S carried, as atm_step.run_free does), RADIA recorded (real packets).
    The F3Acc2 accumulators are called at their sites from OUR state: DIAGA (via the dynamics hook), accum_ma, CONDSE, SURFACE per substep,
    dynamics.  Returns (acc, per-step end-state stats)."""
    import atm_step as A
    import clouds_condse_io as cio
    import radiation_server as rs
    ctx = A.make_ctx(date, imf=True, ff=ff)
    _, geo = f3.load_geo(ff, date)
    acc = F3Acc2(geo)
    hk = ChainedHooks(A).install()
    S, ms, stats = None, {}, []
    try:
        for k in range(nsteps):
            it = it0 + k
            R = A.Real(date, it, ff)
            hk.calls.clear()
            S, sn = A.run_step(date, it, ctx, R=R, S=S, ms=ms, land_mode="ghy", condse="run")
            X = S.get("_condse_X")
            carry = {key: np.array(X[key], copy=True) for key in A.CARRY_KEYS if X is not None and key in X}
            if "_cloud_rad" in S:
                carry["CLDSS"], carry["CLDMC"] = (np.array(a, copy=True) for a in S["_cloud_rad"])
            # --- sites, from OUR state
            if hk.diaga_w is not None:
                w = dict(hk.diaga_w)
                w["PEK1"] = None
                acc.diaga(w, geo, hk.start["TSAVG"], hk.start["QSAVG"], imf_pow=bool(ctx.dyn.imf_pow if hasattr(ctx, "dyn") else True))
                log(f"  DIAGA at itime {it}")
            rad_step = (it - f3.ITIMEI) % 5 == 0
            srv = None
            if rad_step:
                srv = rs.read_packet(f"{ff}/{date}/rsv_n26_{it}_out.bin")["AIJD"]
            acc.radia(np.asarray(R.r["COSZ1"]), rad_step, srv)
            blks = [c["blk"] for c in hk.calls]
            pk = {ns: packet_chained(hk.calls[ns - 1], blks[ns - 1]) for ns in (1, 2)}
            sub = {}
            for ns in (1, 2):
                comp = hk.calls[ns - 1]["res"]["comp"]
                r_ = hk.calls[ns - 1]
                g = lambda v: A._grid(v, r_["res"], 0.0)
                ma1 = np.asarray(sn["filter"]["MA"])[0] if False else np.asarray(R.a["MA"])[0]
                dq1 = g(comp["dq1"])
                sub[ns] = dict(TSAVG=g(comp["tsavg"]), QSAVG=g(comp["qsavg"]), QFLUX1=-dq1 * ma1 / 900.0, UFLUX1=g(comp["uflux1"]),
                               VFLUX1=g(comp["vflux1"]))
            acc.surface(it, sub, np.asarray(R.r["TRHR"])[0])
            ddms = S["DDMS"] if "DDMS" in S else R.cse_out["DDMS"]
            acc.surface_site(it, pk, ddms)
            acc.surface_flux([pk[1], pk[2]], np.asarray(R.r["TRHR"])[0])
            acc.prec(S["PREC"])
            acc.condse2({**{kk: S[kk] for kk in ("PREC", "EPREC", "PRECSS", "QCL", "QCI")}, "LMC": X["LMC"], "CLDMC": X["CLDMC"]},
                        S["PDSIG"], R.cse_in["PEDN"] if False else np.asarray(sn["condse"]["PEDN"]))
            acc.dyn2(hk.w)
            acc.airmass(np.asarray(sn["filter"]["MA"]))
            stats.append(A.compare_state(sn["filter"], A.end_reference(R), A.END_FIELDS))
            S = {key: v for key, v in S.items() if not key.startswith("_")}
            if carry:
                S["_carry"] = carry
            log(f"  chained step {it} done")
    finally:
        hk.uninstall()
    if out_npz:
        a, al = acc.to_nc_layout()
        np.savez(out_npz, aij=a, aijl=al, idacc=np.array([acc.idacc[i] for i in (1, 2, 3, 4)]), cols=np.array(sorted(acc.aij)))
    return acc, stats
