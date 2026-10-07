"""D175: drop-in parallel variant of land_chain_ent.land_substep_ent (scalar GHY + Ent per land cell, D171).

Why: the per-cell scalar GHY + Ent loop is pure Python, about 1,500 independent cell calls per step (753 cells x 2 substeps), 6.5-10 s per
step on one core.  The cells are independent of each other (the Ent state, the carried Qf and the GHY state of a cell depend only on that
cell), so the loop is distributed over persistent worker processes; each worker owns a fixed subset of the cells (their Ent state and Qf
carry live in the worker for the whole run).  The numerics of one cell are exactly land_chain_ent.ghy_ent_call (same function, same
inputs, same order), so the results are bitwise those of the existing implementation by construction (and checked by the test).

Differences to land_chain_ent.land_substep_ent, all outside the numerics:
  * the PBL outputs (jax arrays) are converted to numpy once instead of indexing a jax array 1,500 x 8 times (float(v[n]) costs ~3.6 s per
    step through jax's eager gather); the float values are identical.
  * the per-call diagnostics log of EntLand (qf_dev_*) is not kept; the step log has wall, ncalls, nit_total, nit_max.

Cores: `nproc` worker processes (spawn) + the calling process, which waits while they work.  nproc=1 runs in-process (no workers).
API (same shape as land_chain_ent): ParEntLand(restart, nproc), install_par(A, ent) -> undo, land_substep_ent_par(...).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GHY_REFS = ["tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad"]
EOD = 163
DYN_KEYS = ("w", "ht", "nsn", "dzsn", "wsn", "hsn", "fr_snow")


def _compute(ent, g, fo, dyn, idx):
    """Run the cells idx (indices into g) with the EntLand `ent`; returns stacked outputs for those cells (same fields as
    land_chain_ent.land_substep_ent)."""
    m = len(idx)
    res = {k: np.zeros(m) for k in GHY_REFS}
    w = np.zeros((m, 7, 2)); ht = np.zeros((m, 7, 2)); nsn = np.zeros((m, 2), int)
    dzsn = np.zeros((m, 3, 2)); wsn = np.zeros((m, 3, 2)); hsn = np.zeros((m, 3, 2)); frs = np.zeros((m, 2))
    emax = np.zeros(m); frsat = np.zeros(m); nits = np.zeros(m, int)
    t0 = time.perf_counter()
    for a, n in enumerate(idx):
        fov = {k: float(v[n]) for k, v in fo.items()}
        dv = None
        if dyn is not None:
            dv = dict(w=dyn["w"][n][:7], ht=dyn["ht"][n][:7], nsn=dyn["nsn"][n], dzsn=dyn["dzsn"][n][:3], wsn=dyn["wsn"][n],
                      hsn=dyn["hsn"][n], fr_snow=dyn["fr_snow"][n])
        o = ent.call(g[n], fov, dv)
        c = o["col"]
        for k in GHY_REFS:
            res[k][a] = getattr(c, k)
        w[a], ht[a], nsn[a] = c.w[:7], c.ht[:7], c.nsn
        dzsn[a], wsn[a], hsn[a], frs[a] = c.dzsn[:3], c.wsn, c.hsn, c.fr_snow
        eo = c.evap_max_out
        emax[a] = 0.0 if np.isnan(eo) else eo
        frsat[a] = 0.0 if np.isnan(c.fr_sat) else c.fr_sat
        nits[a] = o["nit"]
    return dict(res=res, w=w, ht=ht, nsn=nsn, dzsn=dzsn, wsn=wsn, hsn=hsn, frs=frs, emax=emax, frsat=frsat, nits=nits,
                wall=time.perf_counter() - t0)


def _worker(conn, restart, keys, qf_mode, dts_mode, daily):
    import land_chain_ent as LE
    ent = LE.EntLand(restart, qf_mode=qf_mode, dts_mode=dts_mode, daily=daily, keys=set(keys))
    conn.send(("ready", len(ent.cells)))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        if msg[0] == "daily":
            ent.daily_update(msg[1])
            conn.send(("ok", None))
        elif msg[0] == "calls":
            _, g, fo, dyn, idx = msg
            conn.send(("res", _compute(ent, g, fo, dyn, idx)))


class ParEntLand:
    """Ent state of all cells distributed over `nproc` workers; step bookkeeping like land_chain_ent.EntLand."""

    def __init__(self, restart, nproc=2, qf_mode="carry", dts_mode="computed", daily=True):
        self.restart, self.nproc, self.qf_mode, self.dts_mode, self.daily = restart, nproc, qf_mode, dts_mode, daily
        self.itime = None
        self.sub = 0
        self.nsteps = 0
        self.log = []
        self.cur = None
        self._owner = None            # (i,j) -> worker index, fixed at the first call
        self._procs = None
        self._local = None

    # ---- worker management
    def _start(self, g):
        import multiprocessing as mp
        keys = [(int(r[0]), int(r[1])) for r in g]
        self._keys0 = keys
        if self.nproc <= 1:
            import land_chain_ent as LE
            self._local = LE.EntLand(self.restart, qf_mode=self.qf_mode, dts_mode=self.dts_mode, daily=self.daily, keys=set(keys))
            self._owner = {k: 0 for k in keys}
            return
        ctx = mp.get_context("spawn")
        self._owner = {k: i % self.nproc for i, k in enumerate(keys)}
        self._procs = []
        for w in range(self.nproc):
            kw = [k for k in keys if self._owner[k] == w]
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, self.restart, kw, self.qf_mode, self.dts_mode, self.daily), daemon=True)
            p.start()
            self._procs.append((p, a))
        for p, a in self._procs:
            assert a.recv()[0] == "ready"

    def close(self):
        if self._procs:
            for p, a in self._procs:
                try:
                    a.send(("stop",))
                except Exception:
                    pass
            for p, a in self._procs:
                p.join(timeout=10)
            self._procs = None

    # ---- EntLand-compatible step bookkeeping
    def begin_step(self, itime):
        self.itime = itime
        self.sub = 0
        self.cur = dict(itime=itime, ncalls=0, nit_total=0, nit_max=0, daily=False, wall=0.0)

    def end_step(self):
        if self.cur is not None:
            self.log.append(self.cur)
            self.cur = None
            self.nsteps += 1

    def maybe_daily(self, g):
        if self.daily and self.sub == 0 and self.nsteps > 0 and len(g) and g[0, EOD] == 1.0:
            jday = (self.itime // 48) % 365 + 1
            if self._local is not None:
                self._local.daily_update(jday)
            else:
                for p, a in self._procs:
                    a.send(("daily", jday))
                for p, a in self._procs:
                    a.recv()
            self.cur["daily"] = jday

    def run_cells(self, g, fo, dyn):
        if self._owner is None:
            self._start(g)
        N = len(g)
        t0 = time.perf_counter()
        if self._local is not None:
            outs = [(np.arange(N), _compute(self._local, g, fo, dyn, np.arange(N)))]
        else:
            own = np.array([self._owner[(int(r[0]), int(r[1]))] for r in g])
            outs = []
            for w, (p, a) in enumerate(self._procs):
                idx = np.flatnonzero(own == w)
                a.send(("calls", g, fo, dyn, idx))
            for w, (p, a) in enumerate(self._procs):
                idx = np.flatnonzero(own == w)
                tag, o = a.recv()
                outs.append((idx, o))
        res = {k: np.zeros(N) for k in GHY_REFS}
        w_ = np.zeros((N, 7, 2)); ht = np.zeros((N, 7, 2)); nsn = np.zeros((N, 2), int)
        dzsn = np.zeros((N, 3, 2)); wsn = np.zeros((N, 3, 2)); hsn = np.zeros((N, 3, 2)); frs = np.zeros((N, 2))
        emax = np.zeros(N); frsat = np.zeros(N); nits = np.zeros(N, int)
        for idx, o in outs:
            for k in GHY_REFS:
                res[k][idx] = o["res"][k]
            w_[idx], ht[idx], nsn[idx], dzsn[idx], wsn[idx], hsn[idx], frs[idx] = o["w"], o["ht"], o["nsn"], o["dzsn"], o["wsn"], o["hsn"], o["frs"]
            emax[idx], frsat[idx], nits[idx] = o["emax"], o["frsat"], o["nits"]
        c = self.cur
        if c is not None:
            c["wall"] += time.perf_counter() - t0
            c["ncalls"] += N
            c["nit_total"] += int(nits.sum())
            c["nit_max"] = max(c["nit_max"], int(nits.max()))
            c.setdefault("worker_wall_max", 0.0)
            c["worker_wall_max"] += max(o["wall"] for _, o in outs)
        return dict(res, w=w_, ht=ht, nsn=nsn, dzsn=dzsn, wsn=wsn, hsn=hsn, fr_snow=frs, evap_max_ij=emax, fr_sat_ij=frsat)


def land_substep_ent_par(p4, g, q1, trup, dtsurf=900.0, dyn=None, ent=None):
    """Same contract and results as land_chain_ent.land_substep_ent; the cell loop runs in ent's worker processes."""
    import land_chain as LC
    import pbl_compare as PC
    assert np.array_equal(p4[:, :2], g[:, :2])
    out = PC.run(p4)
    ps = p4[:, 16]
    tsv, qsrf = np.asarray(out["tsv"]), np.asarray(out["qsrf"])
    rho = 100.0 * ps / (LC.RGAS * tsv)
    ddml = p4[:, 23] > 0.5
    ma1 = g[:, 165]
    fo = dict(ts=tsv / (1.0 + qsrf * LC.XDELT), qs=qsrf, rho=rho, ch=np.asarray(out["ch"]), vs=np.asarray(out["ws"]),
              tprime=np.where(ddml, p4[:, 25] - p4[:, 7], 0.0), qprime=np.where(ddml, p4[:, 26] - p4[:, 39], 0.0),
              qm1=q1 * ma1)
    fo = {k: np.asarray(v, float) for k, v in fo.items()}
    ent.maybe_daily(g)
    ghy = ent.run_cells(g, fo, dyn)
    rcdmws = out["cm"] * out["ws"] * rho
    dlw = dtsurf * (trup - LC.STBO * (ghy["tbcs"] + LC.TF) ** 4)
    patch = dict(uflux1=rcdmws * out["us"], vflux1=rcdmws * out["vs"],
                 dth1=-(-ghy["ashg"] + dlw) / (LC.SHA * ma1), dq1=ghy["aevap"] / ma1, tsavg=tsv, qsavg=qsrf)
    ent.sub += 1
    return dict(patch=patch, pbl=out, ghy=ghy, rho=rho, dyn_next={k: ghy[k] for k in LC.DYN_KEYS},
                evap_max_ij=ghy["evap_max_ij"], fr_sat_ij=ghy["fr_sat_ij"])


def install_par(A, ent):
    """Like land_chain_ent.install but with land_substep_ent_par.  Returns an undo function."""
    import land_chain_ent as LE

    class Shim(LE._LCShim):
        def land_substep(self, p4, g, q1, trup, dtsurf=900.0, dyn=None):
            return land_substep_ent_par(p4, g, q1, trup, dtsurf, dyn, self._ent)

    M = A._surf_mods()
    old_lc, old_run = M["LC"], A.run_step
    M["LC"] = Shim(old_lc, ent)

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
        ent.close()
    return undo
