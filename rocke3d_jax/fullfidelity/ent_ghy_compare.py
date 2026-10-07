"""D169: validate ent_ff (Ent per-iteration exports) against the ffent block of the ffg dumps.

For every recorded land call (one ffg record = one GHY call of 900 s for one cell) the GHY port (ghy_ref, the validated
D158 loop) is run; at every sub-iteration the Ent forcings are taken from the GHY-side state exactly as GHY.f 2434-2467
does (air T, canopy T tp(0,2), Qf, pres, Ca, ch, vs, vis_rad, direct_vis_rad, cosz1, fw, soil moisture w/ws, fice), ent_ff
computes the exports, and they are compared with the recorded ffent(1:13, nit).

Two modes:
  teacher  (default): the GHY port is driven by the RECORDED exports (as in D158), Ent runs alongside on the GHY state and its
           outputs are only compared.  This isolates the Ent port (its inputs are the real-model trajectory to the GHY
           port's own residual).
  closed:  the GHY port is driven by the exports computed by ent_ff (the real replacement of the recorded boundary);
           the GHY outputs are compared with the record as well.

The Ent state of each cell (restart `ent_state`) is carried from call to call in file order (two calls per step, 753
land cells per call: records are written in cell order for substep 1 then again for substep 2).
Limits: the ffg record holds the exports of at most 11 sub-iterations; iterations > 11 are run (they advance the Ent
state) but not compared.
"""
import os
import sys

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
FIELDS = ["cnc", "betadl", "trans_sw", "ci", "gpp", "lai", "ipp"]
GHY_REFS = ["tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad"]


def _pad6(a, n=E.N_DEPTH):
    out = np.zeros(n)
    a = np.asarray(a, dtype=float)
    out[:len(a)] = a
    return out


def run_cell(rec, cell, mode="teacher", dt=900.0, ca_override=None):
    """One GHY call with Ent.  Returns dict(comp=list per iteration of dict(field -> (got, ref)), col, refs, call=dict)."""
    static, dynamic, forcing, ent_iters, refs, snowm = GC.unpack(rec)
    col = G.GhyColumn(static, dynamic, forcing)
    col.fb, col.fv = forcing["fb"], forcing["fv"]
    Ca, cosz1, vis_rad, dvis, Qf = rec[3], rec[4], rec[5], rec[6], rec[7]
    end_of_day = rec[163] == 1.0
    out = dict(comp=[], call=None, qf_exit=None)
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
    if col.process_vege:
        ce = E.call_exports(cell)
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
        if nit <= nrec:
            dts = ent_iters[nit - 1]["dts"]
            dtr = 0.0 if nit == ffnit else dtr - dts
        elif dtm >= dtr:
            dts = dtr
            dtr = 0.0
        else:
            dts = min(dtm, dtr * 0.5)
            dtr = dtr - dts
        col.dts = dts
        if col.process_vege:
            E.set_forcings(cell, col.ts, col.tp[0, 1], Qf, col.pres, Ca, col.ch, col.vs, vis_rad, dvis, cosz1, col.fw,
                           _pad6(col.w[1:, 1]), _pad6(col.ws[1:, 1]), _pad6(col.fice[1:, 1]))
            E.ent_run(cell, dts, bool(end_of_day and nit == 1))
            ex = E.get_exports(cell)
            if nit <= nrec:
                r = ent_iters[nit - 1]
                out["comp"].append({k: (ex[k], r[k]) for k in FIELDS})
            use = ex if mode == "closed" else (ent_iters[min(nit, nrec) - 1])
            col.cnc = use["cnc"]
            col.betadl = np.asarray(use["betadl"])[:col.n]
            col.lai = use["lai"]
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
        # GHY.f:2621
        Qf = (col.evap_tot[1] / (col.rho / 1000.0 * col.ch) + col.gusti * col.qprime) / col.vs + col.qs
    col.accm_final()
    col.hydra()
    out["qf_exit"] = (Qf, rec[179])
    out["col"] = col
    out["refs"] = refs
    out["nit"] = nit
    return out


def run_file(path, cells, mode="teacher", keys=None, ncalls=None):
    """All records of one ffg file (in order) carrying Ent state in `cells`.  `keys`: optional set of (i, j) cells to
    process (the Ent state of a cell does not depend on other cells).  Returns a list of (record index, (i, j), result)."""
    rec = GC.load(path)
    res = []
    for k in range(len(rec) if ncalls is None else min(ncalls, len(rec))):
        r = rec[k]
        key = (int(r[0]), int(r[1]))
        if keys is not None and key not in keys:
            continue
        cell = cells.get(key)
        if cell is None:
            res.append(None)
            continue
        res.append((k, key, run_cell(r, cell, mode)))
    return res


def summarize(res):
    """Max abs / relative residual per field over all compared iterations, bitwise-equal counts."""
    acc = {f: dict(n=0, neq=0, maxabs=0.0, maxrel=0.0) for f in FIELDS}
    for item in res:
        if item is None:
            continue
        k, key, o = item
        for it in o["comp"]:
            for f in FIELDS:
                got, ref = it[f]
                got = np.atleast_1d(got)
                ref = np.atleast_1d(ref)
                d = np.abs(got - ref)
                a = acc[f]
                a["n"] += got.size
                a["neq"] += int(np.sum(got == ref))
                a["maxabs"] = max(a["maxabs"], float(d.max()))
                scale = np.maximum(np.abs(ref), 1e-300)
                a["maxrel"] = max(a["maxrel"], float((d / scale).max()))
    return acc


def summarize_call(res):
    """Per-call exports (ws_can, shc_can, fv, height, albedo) and the exit Qf: max abs diff and bitwise-equal counts."""
    acc = {}
    for item in res:
        if item is None:
            continue
        k, key, o = item
        pairs = {}
        if o["call"] is not None:
            pairs.update(o["call"])
            pairs["qf_exit"] = o["qf_exit"]
        for name, (got, ref) in pairs.items():
            got = np.atleast_1d(got)
            ref = np.atleast_1d(ref)
            a = acc.setdefault(name, dict(n=0, neq=0, maxabs=0.0))
            a["n"] += got.size
            a["neq"] += int(np.sum(got == ref))
            a["maxabs"] = max(a["maxabs"], float(np.abs(got - ref).max()))
    return acc


def summarize_ghy(res):
    """GHY output scalars vs the record (meaningful in closed mode; in teacher mode they re-state D158)."""
    acc = {k: 0.0 for k in GHY_REFS}
    for item in res:
        if item is None:
            continue
        k, key, o = item
        for name in GHY_REFS:
            acc[name] = max(acc[name], abs(getattr(o["col"], name) - o["refs"][name]))
    return acc


def run_dataset(tag, files=None, mode="teacher", verbose=True, daily_update=True):
    """tag in nov26, dec01, jan01, nov26_day.  Ent state starts from the restart and is carried through the files in order."""
    import glob
    cells = E.load_restart_cells(RESTART[tag])
    paths = sorted(glob.glob(f"{FF}/{tag}/ffg_[0-9]*.bin"))
    if files is not None:
        paths = paths[:files]
    total = {f: dict(n=0, neq=0, maxabs=0.0, maxrel=0.0) for f in FIELDS}
    per_file = []
    callacc = {}
    lai_qty = None
    for ip, p in enumerate(paths):
        # ent_ff does not port the daily update; the prescribed LAI/albedo part (ent_daily_ff, validated bitwise at the nov26 -> nov27
        # boundary) is applied before the first step of a new day, i.e. when the end_of_day flag of the record is set
        if daily_update and ip > 0 and np.fromfile(p, ">f8", 450)[163] == 1.0:
            import ent_daily_ff as D
            if lai_qty is None:
                lai_qty = D.read_lai_file()
            itime = int(os.path.basename(p)[4:-4])
            jday = (itime // 48) % 365 + 1
            la = D.lai_linm2m(lai_qty, jday)
            for (ci, cj), cell in cells.items():
                D.daily_update(cell, la[:, cj - 1, ci - 1], D.hemi_of_j(cj), jday)
            if verbose:
                print("daily update (prescribed LAI + albedo) applied, jday", jday, flush=True)
        res = run_file(p, cells, mode)
        a = summarize(res)
        c = summarize_call(res)
        for f in FIELDS:
            for key in ("n", "neq"):
                total[f][key] += a[f][key]
            for key in ("maxabs", "maxrel"):
                total[f][key] = max(total[f][key], a[f][key])
        for name, v in c.items():
            t = callacc.setdefault(name, dict(n=0, neq=0, maxabs=0.0))
            t["n"] += v["n"]
            t["neq"] += v["neq"]
            t["maxabs"] = max(t["maxabs"], v["maxabs"])
        per_file.append((os.path.basename(p), a))
        if verbose:
            print(os.path.basename(p), {f: (a[f]["neq"], a[f]["n"], "%.2e" % a[f]["maxrel"]) for f in ("cnc", "betadl", "ci", "gpp")}, flush=True)
    return total, callacc, per_file


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "nov26"
    mode = sys.argv[2] if len(sys.argv) > 2 else "teacher"
    nf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    total, callacc, _ = run_dataset(tag, nf, mode)
    print("TOTAL per-iteration exports", tag, mode)
    for f, a in total.items():
        print("  %-9s n=%d bitwise_equal=%d maxabs=%.3e maxrel=%.3e" % (f, a["n"], a["neq"], a["maxabs"], a["maxrel"]))
    print("per-call exports and exit Qf")
    for f, a in callacc.items():
        print("  %-9s n=%d bitwise_equal=%d maxabs=%.3e" % (f, a["n"], a["neq"], a["maxabs"]))


def closed_report(tag="nov26", files=None):
    """Closed mode: GHY driven by ent_ff exports; reports export residuals and the max |GHY output - record| per output
    (scaled by the record's own max |value| over cells) for teacher vs closed."""
    import glob
    out = {}
    for mode in ("teacher", "closed"):
        cells = E.load_restart_cells(RESTART[tag])
        paths = sorted(glob.glob(f"{FF}/{tag}/ffg_[0-9]*.bin"))
        if files:
            paths = paths[:files]
        gmax = {k: 0.0 for k in GHY_REFS}
        gscale = {k: 1e-300 for k in GHY_REFS}
        wmax = 0.0
        for p in paths:
            res = run_file(p, cells, mode)
            g = summarize_ghy(res)
            for k in GHY_REFS:
                gmax[k] = max(gmax[k], g[k])
            for item in res:
                if item is None:
                    continue
                kk, key, o = item
                for k in GHY_REFS:
                    gscale[k] = max(gscale[k], abs(o["refs"][k]))
                wmax = max(wmax, float(np.abs(o["col"].w[:, :2] - o["refs"]["w_out"]).max()))
        out[mode] = (gmax, gscale, wmax)
        print(mode, "max|GHY out - record|/scale:", {k: "%.2e" % (gmax[k] / gscale[k]) for k in GHY_REFS}, "w_out max abs %.2e" % wmax, flush=True)
    return out
