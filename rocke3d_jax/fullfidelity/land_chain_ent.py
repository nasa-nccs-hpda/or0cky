"""D171: Ent (vegetation) computed from our own state inside the chained land path (stage 1b of the D169 plan).

What this module is
-------------------
`land_chain.land_substep` runs PBL (our port) and the batched JAX GHY with the Ent exports (cnc, betadl, lai, dts) taken from the
recorded `ffent` block.  The exports of one sub-iteration depend on the GHY-side state of that sub-iteration (canopy temperature,
soil moisture, fice, Qf, ...), and the next sub-iteration depends on the exports, so Ent cannot be precomputed for the batched
JAX scan.  Here GHY is therefore run cell by cell with the scalar GHY port (`ghy_ref.GhyColumn`, the D158 level, same loop as
GHY.f advnc), and Ent (`ent_ff`) is called inside the loop exactly as GHY.f does (GHY.f 2434-2467 -> ent_integrate -> exports):

  * per sub-iteration: hydra, xklh, gdtm (the time step is COMPUTED, not the recorded `dts`), Ent forcings from the GHY state,
    `ent_ff.ent_run`, exports cnc/betadl/lai -> evap_limits ... apply_fluxes, accm, reth, retp, Qf (GHY.f:2621);
  * per call (once, before the loop): ws_can, shc_can, fv(=1-fb), canopy height (snowm = 0.1*height), albedo come from OUR Ent state
    (`ent_ff.call_exports`), not from the record; irrigation is the record's irrig_tot/fv with our fv;
  * the Ent state of every cell (the restart `ent_state`) is carried from call to call (two calls per step) and Qf_ij (the
    persistent canopy-air humidity of GHY_DRV.f, `Qf_ij(i,j)`) is carried per cell (option `qf_mode='record'` takes the
    recorded entry value instead);
  * the daily prescribed LAI/albedo update (`ent_daily_ff`, D169 stage 3a) is applied before the first step of a new model day
    (end-of-day flag of the ffg record, same rule as `ent_ghy_compare.run_dataset`; jday of the new day from itime).

Inputs that stay RECORDED (stated plainly): the radiation inputs of Ent (vis_rad, direct_vis_rad, cosz1: GHY_DRV.f 1191-1193 from
SRVISSURF/FSRDIR/COSZ1) and Ca (CO2 ppm), the precipitation / radiation forcing of GHY (pr, htpr, prs, srheat, trheat), the
geothermal heat, the irrigation totals, vs0/gusti/pres, and in the open-loop day the GHY prognostic state at the start of every
step.  What is NOT ported (see D169): carbon/soil biogeochemistry, set_vegetation_data at the first day end of a segment, the
height/crop/structure updates.

Entry points
------------
  land_substep_ent(p4, g, q1, trup, dtsurf, dyn, ent)   drop-in for land_chain.land_substep (same return structure)
  install(A, ent)                                       make atm_step use it (land_mode 'ghy'): replaces atm_step._SURF['LC'] by a shim
  closed_day(tag, shard, nshard, ...)                   closed-mode validation of the GHY outputs vs the recorded ffg outputs per step
  python land_chain_ent.py closed_day <tag> <shard> <nshard> <out.json> [nsteps]
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ent_ff as E
import ghy_compare as GC
import ghy_ref as G

FF = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
RESTART = {"nov26": f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc",
           "dec01": f"{FF}/_pristine_restarts/fort1_dec01_itime33552.nc",
           "jan01": f"{FF}/_pristine_restarts/fort1_jan01_itime17520.nc",
           "nov26_day": f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc"}
RGAS = 287.048730386149032
EOD = 163           # 0-based index of the end-of-day flag in the ffg record (record slot 164)
GHY_REFS = ["tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad"]
EXPORTS = ["cnc", "betadl", "trans_sw", "ci", "gpp", "lai", "ipp"]


def _pad6(a, n=E.N_DEPTH):
    out = np.zeros(n)
    a = np.asarray(a, dtype=float)
    out[:len(a)] = a
    return out


# ------------------------------------------------------------------------------------------------------------ one GHY call with Ent
def ghy_ent_call(rec, cell, qf0=None, forcing_over=None, dyn_over=None, dt=900.0, dts_mode="computed", compare=False):
    """One GHY call (advnc over dt = 900 s) for one land cell with Ent computed from the GHY-side state.

    rec          ffg record (450 doubles): static soil data, recorded forcing, recorded dynamic state (used unless dyn_over), refs
    cell         ent_ff.EntCell of the cell (state advanced in place) or None for a cell without Ent (then fv must be 0)
    qf0          Qf at call entry (None: the record's entry value rec[7])
    forcing_over dict of GhyColumn forcing entries to replace (ts, qs, rho, ch, vs, tprime, qprime, qm1 ...)
    dyn_over     dict of dynamic entries (w, ht, nsn, dzsn(3,2), wsn, hsn, fr_snow) to replace the recorded state
    dts_mode     'computed' (GHY.f dtr loop with gdtm; the chained path) or 'recorded' (first nrec sub-iterations use the record's dts)
    compare      also return per-iteration (computed, recorded) export pairs for nit <= min(11, record length)
    Returns dict(col, nit, qf_exit, qf_entry, refs, comp, call, dts)."""
    static, dynamic, forcing, ent_iters, refs, snowm = GC.unpack(rec)
    forcing = dict(forcing)
    ce = None
    if cell is not None:
        ce = E.call_exports(cell)
        fv = ce["fv"]
        if fv < 1e-6:
            fv = 0.0
        if fv > 1.0 - 1e-6:
            fv = 1.0
        irrig_tot, htirrig_tot = rec[147], rec[148]
        forcing.update(fv=fv, fb=1.0 - fv, ws_can=ce["ws_can"], shc_can=ce["shc_can"],
                       irrig=(irrig_tot / fv if fv > 0 else 0.0), htirrig=(htirrig_tot / fv if fv > 0 else 0.0))
        snowm = ce["height"] * 0.1
    elif forcing["fv"] > 0.0:
        raise KeyError("vegetated land cell without Ent state")
    if forcing_over:
        forcing.update(forcing_over)
    if dyn_over:
        dynamic = dict(dynamic)
        dynamic.update(dyn_over)
    col = G.GhyColumn(static, dynamic, forcing)
    col.fb, col.fv = forcing["fb"], forcing["fv"]
    Ca, cosz1, vis_rad, dvis = rec[3], rec[4], rec[5], rec[6]
    Qf = rec[7] if qf0 is None else qf0
    qf_entry = Qf
    end_of_day = rec[EOD] == 1.0
    out = dict(comp=[], call=None, dts=[], qf_entry=qf_entry)
    col.dt = dt
    col.snowm = snowm
    col._bounds()
    col.accm_zero()
    col.reth()
    col.retp()
    col.tb0 = col.tp[1, 0]
    col.tc0 = col.tp[0, 1]
    col.evapb = col.epb = 1.0
    col.evapvw = col.evapvd = col.epv = 1.0
    if col.process_vege and compare:
        out["call"] = dict(ws_can=(ce["ws_can"], rec[167]), shc_can=(ce["shc_can"], rec[168]), fv=(ce["fv"], rec[169]),
                           height=(ce["height"], rec[170]), albedo=(ce["albedo"], rec[171:177]))
    dtr = dt
    nit = 0
    nrec = len(ent_iters)
    ffnit = refs["ffnit"]
    while dtr > 0.0:
        nit += 1
        col.hydra()
        col.xklh()
        dtm = col.gdtm(nit)
        if dts_mode == "recorded" and nit <= nrec:
            dts = ent_iters[nit - 1]["dts"]
            dtr = 0.0 if nit == ffnit else dtr - dts
        elif dtm >= dtr:
            dts = dtr
            dtr = 0.0
        else:
            dts = min(dtm, dtr * 0.5)
            dtr = dtr - dts
        col.dts = dts
        out["dts"].append(dts)
        if col.process_vege:
            E.set_forcings(cell, col.ts, col.tp[0, 1], Qf, col.pres, Ca, col.ch, col.vs, vis_rad, dvis, cosz1, col.fw,
                           _pad6(col.w[1:, 1]), _pad6(col.ws[1:, 1]), _pad6(col.fice[1:, 1]))
            E.ent_run(cell, dts, bool(end_of_day and nit == 1))
            ex = E.get_exports(cell)
            if compare and nit <= nrec:
                r = ent_iters[nit - 1]
                out["comp"].append({k: (ex[k], r[k]) for k in EXPORTS})
            col.cnc = ex["cnc"]
            col.betadl = np.asarray(ex["betadl"])[:col.n]
            col.lai = ex["lai"]
        else:
            col.cnc = 0.0
            col.betadl = np.zeros(col.n)
            col.lai = 0.0
        col.evap_limits(True)
        col.drip_from_canopy()
        col.sensible_heat()
        col.snow()
        col.fl()
        col.flg()
        col.runoff()
        col.fllmt()
        col.flh()
        col.flhg()
        col.apply_fluxes()
        col.accm()
        col.reth()
        col.retp()
        Qf = (col.evap_tot[1] / (col.rho / 1000.0 * col.ch) + col.gusti * col.qprime) / col.vs + col.qs     # GHY.f:2621
    col.accm_final()
    col.hydra()
    # GHY_DRV.f: evap_limits(.false.) on the final state with the last sub-iteration's cnc/betadl/lai -> evap_max_ij, fr_sat_ij (as ghy_jax.advnc)
    col.evap_limits(False)
    out.update(col=col, nit=nit, qf_exit=Qf, refs=refs)
    return out


# ------------------------------------------------------------------------------------------------------------ Ent state holder
class EntLand:
    """Ent state of all cells (from the restart), per-cell Qf_ij carry, daily update; statistics of what happened."""

    def __init__(self, restart, qf_mode="carry", dts_mode="computed", daily=True, keys=None):
        self.cells = E.load_restart_cells(restart)
        if keys is not None:
            self.cells = {k: v for k, v in self.cells.items() if k in keys}
        self.qf_mode, self.dts_mode, self.daily = qf_mode, dts_mode, daily
        self.qf = {}
        self.lai_qty = None
        self.itime = None
        self.sub = 0
        self.nsteps = 0
        self.log = []          # per step dict of diagnostics
        self.cur = None

    # called once per model step before its first land substep
    def begin_step(self, itime):
        self.itime = itime
        self.sub = 0
        self.cur = dict(itime=itime, ncalls=0, nit_total=0, nit_max=0, qf_dev_max=0.0, qf_dev_n=0, daily=False, wall=0.0)

    def end_step(self):
        if self.cur is not None:
            self.log.append(self.cur)
            self.cur = None
            self.nsteps += 1

    def daily_update(self, jday):
        import ent_daily_ff as D
        if self.lai_qty is None:
            self.lai_qty = D.read_lai_file()
        la = D.lai_linm2m(self.lai_qty, jday)
        for (ci, cj), cell in self.cells.items():
            D.daily_update(cell, la[:, cj - 1, ci - 1], D.hemi_of_j(cj), jday)

    def maybe_daily(self, g):
        """Day boundary: first substep of a step whose ffg records carry the end-of-day flag (not before the first step seen)."""
        if self.daily and self.sub == 0 and self.nsteps > 0 and len(g) and g[0, EOD] == 1.0:
            jday = (self.itime // 48) % 365 + 1
            self.daily_update(jday)
            self.cur["daily"] = jday

    def call(self, rec, forcing_over=None, dyn_over=None):
        key = (int(rec[0]), int(rec[1]))
        cell = self.cells.get(key)
        qf0 = None
        if self.qf_mode == "carry" and key in self.qf:
            qf0 = self.qf[key]
        t0 = time.perf_counter()
        o = ghy_ent_call(rec, cell, qf0, forcing_over, dyn_over, 900.0, self.dts_mode)
        c = self.cur
        if c is not None:
            c["wall"] += time.perf_counter() - t0
            c["ncalls"] += 1
            c["nit_total"] += o["nit"]
            c["nit_max"] = max(c["nit_max"], o["nit"])
            if key in self.qf and cell is not None:
                c["qf_dev_max"] = max(c["qf_dev_max"], abs(self.qf[key] - rec[7]))
                c["qf_dev_n"] += 1
        self.qf[key] = o["qf_exit"]
        return o


# ------------------------------------------------------------------------------------------------------------ drop-in land substep
def land_substep_ent(p4, g, q1, trup, dtsurf=900.0, dyn=None, ent=None):
    """Same contract as land_chain.land_substep (returns dict(patch, pbl, ghy, rho, dyn_next, evap_max_ij, fr_sat_ij)) with the
    GHY run cell by cell with Ent computed (see module docstring).  `dyn`: carried GHY state of the previous substep (dict of
    batched arrays: w (N,7,2), ht, nsn (N,2), dzsn (N,3,2), wsn (N,3,2), hsn (N,3,2), fr_snow (N,2)) or None (recorded)."""
    import land_chain as LC
    import pbl_compare as PC
    assert np.array_equal(p4[:, :2], g[:, :2])
    out = PC.run(p4)
    ps = p4[:, 16]
    tsv, qsrf = out["tsv"], out["qsrf"]
    rho = 100.0 * ps / (LC.RGAS * tsv)
    ddml = p4[:, 23] > 0.5
    ma1 = g[:, 165]
    fo = dict(ts=tsv / (1.0 + qsrf * LC.XDELT), qs=qsrf, rho=rho, ch=out["ch"], vs=out["ws"],
              tprime=np.where(ddml, p4[:, 25] - p4[:, 7], 0.0), qprime=np.where(ddml, p4[:, 26] - p4[:, 39], 0.0),
              qm1=q1 * ma1)
    ent.maybe_daily(g)
    N = len(g)
    res = {k: np.zeros(N) for k in GHY_REFS}
    w = np.zeros((N, 7, 2)); ht = np.zeros((N, 7, 2)); nsn = np.zeros((N, 2), int)
    dzsn = np.zeros((N, 3, 2)); wsn = np.zeros((N, 3, 2)); hsn = np.zeros((N, 3, 2)); frs = np.zeros((N, 2))
    emax = np.zeros(N); frsat = np.zeros(N)
    for n in range(N):
        fov = {k: float(v[n]) for k, v in fo.items()}
        dv = None
        if dyn is not None:
            dv = dict(w=dyn["w"][n][:7], ht=dyn["ht"][n][:7], nsn=dyn["nsn"][n], dzsn=dyn["dzsn"][n][:3], wsn=dyn["wsn"][n],
                      hsn=dyn["hsn"][n], fr_snow=dyn["fr_snow"][n])
        o = ent.call(g[n], fov, dv)
        c = o["col"]
        for k in GHY_REFS:
            res[k][n] = getattr(c, k)
        w[n], ht[n], nsn[n] = c.w[:7], c.ht[:7], c.nsn
        dzsn[n], wsn[n], hsn[n], frs[n] = c.dzsn[:3], c.wsn, c.hsn, c.fr_snow
        eo = c.evap_max_out
        emax[n] = 0.0 if np.isnan(eo) else eo
        frsat[n] = 0.0 if np.isnan(c.fr_sat) else c.fr_sat
    ghy = dict(res, w=w, ht=ht, nsn=nsn, dzsn=dzsn, wsn=wsn, hsn=hsn, fr_snow=frs, evap_max_ij=emax, fr_sat_ij=frsat)
    rcdmws = out["cm"] * out["ws"] * rho
    dlw = dtsurf * (trup - LC.STBO * (ghy["tbcs"] + LC.TF) ** 4)
    patch = dict(uflux1=rcdmws * out["us"], vflux1=rcdmws * out["vs"],
                 dth1=-(-ghy["ashg"] + dlw) / (LC.SHA * ma1), dq1=ghy["aevap"] / ma1, tsavg=tsv, qsavg=qsrf)
    ent.sub += 1
    return dict(patch=patch, pbl=out, ghy=ghy, rho=rho, dyn_next={k: ghy[k] for k in LC.DYN_KEYS},
                evap_max_ij=emax, fr_sat_ij=frsat, elhx=np.array(p4[:, 19]))


class _LCShim:
    """Looks like the land_chain module for atm_step (every attribute delegated) except land_substep, which runs Ent."""

    def __init__(self, base, ent):
        self._base, self._ent = base, ent

    def __getattr__(self, k):
        return getattr(self._base, k)

    def land_substep(self, p4, g, q1, trup, dtsurf=900.0, dyn=None):
        return land_substep_ent(p4, g, q1, trup, dtsurf, dyn, self._ent)


def install(A, ent):
    """Route atm_step's land_mode='ghy' through land_substep_ent and tell `ent` the step number (itime) before every step.
    Returns an undo function."""
    M = A._surf_mods()
    old_lc, old_run = M["LC"], A.run_step
    M["LC"] = _LCShim(old_lc, ent)

    def run_step(date, itime, *a, **kw):
        ent.begin_step(itime)
        try:
            return old_run(date, itime, *a, **kw)
        finally:
            ent.end_step()
    A.run_step = run_step

    def undo():
        M["LC"] = old_lc
        A.run_step = old_run
    return undo


# ------------------------------------------------------------------------------------------------------------ closed-mode validation
def _acc():
    return dict(n=0, maxabs=0.0, scale=0.0, ndiff=0, cell=None)


def _upd(a, d, scale, key):
    a["n"] += 1
    if d > a["maxabs"]:
        a["maxabs"], a["cell"] = d, list(key)
    if d != 0.0:
        a["ndiff"] += 1
    a["scale"] = max(a["scale"], scale)


def closed_day(tag="nov26_day", shard=0, nshard=1, nsteps=None, verbose=True, qf_mode="record", dts_mode="computed"):
    """Closed mode: every call of every step of `tag` (two per step) with Ent computed, GHY prognostic state and forcing reset from the
    record at every call (so this validates Ent + GHY inside the loop, not drift), the Ent state (and Qf if qf_mode='carry') carried.
    Cells are split over `nshard` processes by index (the Ent state of a cell is independent of the others).
    Returns per-step dict: per output max |ours - record|, scale (max |record|) and the number of calls with a non-zero difference."""
    import glob
    paths = sorted(glob.glob(f"{FF}/{tag}/ffg_[0-9]*.bin"))
    if nsteps:
        paths = paths[:nsteps]
    first = GC.load(paths[0])
    n0 = len(first) // 2
    allkeys = [(int(r[0]), int(r[1])) for r in first[:n0]]
    keys = set(allkeys[shard::nshard])
    ent = EntLand(RESTART[tag], qf_mode=qf_mode, dts_mode=dts_mode, keys=keys)
    rows = []
    for ip, p in enumerate(paths):
        itime = int(os.path.basename(p)[4:-4])
        rec = GC.load(p)
        ent.begin_step(itime)
        ng = len(rec) // 2
        acc = {k: _acc() for k in GHY_REFS + ["w_out", "ht_out", "tp_out", "fr_snow_out", "qf_exit"]}
        exa = {k: dict(n=0, neq=0, maxrel=0.0) for k in EXPORTS}
        nit_mismatch = nit_calls = 0
        dts_mismatch = 0
        nit_max = 0
        t0 = time.perf_counter()
        for sub in (0, 1):
            ent.sub = sub
            gsub = rec[sub * ng:(sub + 1) * ng]
            if sub == 0:
                ent.maybe_daily(gsub)
            for r in gsub:
                key = (int(r[0]), int(r[1]))
                if key not in keys:
                    continue
                cell = ent.cells.get(key)
                qf0 = ent.qf.get(key) if qf_mode == "carry" else None
                if key in ent.qf:
                    acc_qf = abs(ent.qf[key] - r[7])
                    ent.cur["qf_dev_max"] = max(ent.cur["qf_dev_max"], acc_qf)
                    ent.cur["qf_dev_n"] += 1
                o = ghy_ent_call(r, cell, qf0, None, None, 900.0, dts_mode, compare=True)
                ent.qf[key] = o["qf_exit"]
                c, refs = o["col"], o["refs"]
                for k in GHY_REFS:
                    _upd(acc[k], abs(getattr(c, k) - refs[k]), abs(refs[k]), key)
                _upd(acc["w_out"], float(np.abs(c.w[:, :2] - refs["w_out"]).max()), float(np.abs(refs["w_out"]).max()), key)
                _upd(acc["ht_out"], float(np.abs(c.ht[:, :2] - refs["ht_out"]).max()), float(np.abs(refs["ht_out"]).max()), key)
                _upd(acc["tp_out"], float(np.abs(c.tp[:7, :2] - refs["tp_out"]).max()), float(np.abs(refs["tp_out"]).max()), key)
                _upd(acc["fr_snow_out"], float(np.abs(c.fr_snow - refs["fr_snow_out"]).max()), 1.0, key)
                _upd(acc["qf_exit"], abs(o["qf_exit"] - r[179]), abs(r[179]), key)
                nit_calls += 1
                nit_max = max(nit_max, o["nit"])
                if o["nit"] != refs["ffnit"]:
                    nit_mismatch += 1
                for it in o["comp"]:
                    for f in EXPORTS:
                        got, ref = np.atleast_1d(it[f][0]), np.atleast_1d(it[f][1])
                        d = np.abs(got - ref)
                        a = exa[f]
                        a["n"] += got.size
                        a["neq"] += int(np.sum(got == ref))
                        a["maxrel"] = max(a["maxrel"], float((d / np.maximum(np.abs(ref), 1e-300)).max()))
        ent.end_step()
        row = dict(itime=itime, step=ip, shard=shard, calls=nit_calls, nit_mismatch=nit_mismatch, nit_max=nit_max,
                   daily=ent.log[-1]["daily"], qf_dev_max=ent.log[-1]["qf_dev_max"], wall=time.perf_counter() - t0,
                   out={k: v for k, v in acc.items()}, exports=exa)
        rows.append(row)
        if verbose:
            print(itime, "calls", nit_calls, "nit!=ffnit", nit_mismatch,
                  {k: "%.1e" % (acc[k]["maxabs"] / max(acc[k]["scale"], 1e-300)) for k in ("tbcs", "ashg", "aevap", "abetad", "w_out")},
                  "wall %.0fs" % row["wall"], flush=True)
    return rows


def run_day_ent(tag="d171_ent", nsteps=54, land="ent", qf_mode="carry", dts_mode="computed", it0=33312, **kw):
    """The D150 open-loop day (atm_day_open_loop.run_day: recorded radiation, step k+1 starts from OUR atmosphere end state) with the land
    patch from our PBL + GHY: land='ent' -> scalar GHY with Ent computed (this module), 'ghy' -> batched JAX GHY with the RECORDED
    Ent exports (land_chain, existing), 'recorded' -> the recorded land patch (D150).  GHY prognostic state restarts from the record at
    every step (open loop) and is carried between the two substeps; the Ent state (and Qf) is carried by us over all steps.
    Writes ff_data/nov26_day/ours_<tag>/ (step_<it>.npz, run.json) and, for 'ent', ent_log.json (per step diagnostics and Ent wall time)."""
    import atm_day_open_loop as OL
    import atm_step as A
    undo, ent = None, None
    if land == "ent":
        ent = EntLand(RESTART["nov26_day"], qf_mode=qf_mode, dts_mode=dts_mode)
        undo = install(A, ent)
        land_mode = "ghy"
    else:
        land_mode = land
    try:
        res = OL.run_day(OL.DATE, it0, nsteps, imf=True, dyn="numpy", land=land_mode, daydir=OL.DAYDIR, ff=FF, tag=tag, **kw)
    finally:
        if undo:
            undo()
    if ent is not None:
        json.dump(ent.log, open(f"{FF}/nov26_day/ours_{tag}/ent_log.json", "w"))
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "day":
        land, tag, ns = sys.argv[2], sys.argv[3], int(sys.argv[4])
        run_day_ent(tag, ns, land)
    elif len(sys.argv) >= 6 and sys.argv[1] == "closed_day":
        tag, sh, nsh, outp = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        ns = int(sys.argv[6]) if len(sys.argv) > 6 else None
        qm = sys.argv[7] if len(sys.argv) > 7 else "record"
        rows = closed_day(tag, sh, nsh, ns, qf_mode=qm)
        json.dump(rows, open(outp, "w"))
    else:
        print(__doc__)
