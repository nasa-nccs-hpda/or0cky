"""D178: model driver skeleton.  ONE object that holds the complete model state and advances it N steps with the EXISTING validated stages,
with a clearly separated boundary-provider interface, a per-stage timing log and bit-for-bit checkpoint/resume.

Nothing existing is edited: atm_step, atm_step_fast, surface_loop, surface_loop_v2, land_chain_ent, f3_diagnostics2, drv_rng,
atm_day_free_rad(_persist) are imported and their module attributes are rebound ONLY for the duration of a step (try/finally).

Main loop of the real model (MODELE.f:309-366) and where each part sits here
  startNewDay (diagnostics only)                      -
  atm_phase1: DYNAM+QDYNAM+energy fix, MELT_SI, CONDSE, RADIA    A.stage_dyn, Loop2.pre_cse (MELT_SI), A.stage_condse (batched), provider.stage_radia
  PRECIP_SI / PRECIP_OC / SURFACE (2 substeps) / ocean_driver  A.stage_surface = Loop2.stage_surface (tiles, GHY[+Ent], then surface_post_v2:
                                                          GROUND_LI/SI/LK, RIVERF, DYNSI, ocean step, ADVSI); 'replay' mode uses the
                                                          recorded surface (atm_step.stage_surface) instead
  atm_phase2: DISSIP, FILTER                           A.stage_dissip, A.stage_filter
  nextTick; dailyUpdates when the new step starts a day   drv_daily.day_boundary (DAILY_ATMDYN, ch4ox, SNOAGE, ...), Ent daily LAI update (EntLand)
  diagnostics                                          F3Acc2 sites: DIAGA (dynamics hook), accum_ma, RADIA, CONDSE, SURFACE substeps, flux sums
The ocean step is therefore INSIDE the surface stage here (as in surface_loop_v2), not a separate stage; the timing log reports the
surface stage split into 'surface_tiles' (PBL, tile fluxes, GHY[+Ent]) and 'surface_post' (ocean/ice/lake/ADVSI).

State held (all of it goes into a checkpoint): atmosphere S (incl. the CONDSE/RADIA carry `_carry`), the LSCOND module arrays `ms`, the surface
state SS (ocean, ice, lake, land ice, atm gtemp ...), the V2 state (USI/VSI of DYNSI, RSIX/RSIY of ADVSI), the carried land state (GHY prognostic
state + PBL results of the last substep), the Ent state of all cells (+Qf, step counter), the F3 accumulators (aij/aijl/idacc/s0), the random-seed
chain (SEEDS[0] of the next step), the provider state (frozen radiation heating rates, server carry), the step counter/timing log.

Boundary-provider interface (class BoundaryProvider): everything the driver still takes from outside its own computation goes through ONE object.
  real(itime)                 per-step record view R (A.Real-like; R.r carries SRHR/TRHR/COSZ1 of the step)
  stage_radia(S, R, ctx)      the radiation stage (record replay, or the radiation server)
  radiation_aij(itime)        RADIA's AIJ increment of a radiation step for the F3 accumulators (None on other steps)
  surface_records(R)          the per-step SURFACE record templates (ffp/ffs/ffl/ffg/fft rows) that the surface stage overwrites in part
  column_modifiers()          list of (name, fn): the plug-in point for COMPUTED replacements of recorded columns (D176 radiation-derived
                              columns, D177 COSZ1/PBL/ice-thermal columns).  fn(rec, ctx) -> rec' receives the template records
                              (dict pa,pb,ta,tb,la,lb,blk1,blk2,g1,g2 of numpy arrays) and a StepContext (itime, S, provider, driver);
                              it returns records with its columns replaced.  The driver runs them in order after surface_records() and logs, per
                              modifier, how many elements changed.  mode 'apply' uses the result, 'shadow' only logs the change (a
                              computed-vs-recorded comparison without changing the run).
  initial_seed(itime)         SEEDS[0] at CONDSE entry of the first step (the only recorded seed)
  daily_inputs(itime)         recorded constants/fields of the day boundary (MDRYA, ch4ox water mass, SNOAGE after aging) used where drv_daily has no computed form
  state() / set_state(d)      provider state for the checkpoint
Swapping a recorded input for a computed one is done by registering a column modifier or by subclassing the provider; the driver is not edited.
"""
import contextlib
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import atm_step as A  # noqa: E402
import atm_step_fast as F  # noqa: E402
import atm_day_open_loop as OL  # noqa: E402
import clouds_condse_io as cio  # noqa: E402
import clouds_condse_ff as cf  # noqa: E402
import drv_rng  # noqa: E402

FF = cio.FF_DEFAULT
NDAY = 48
_ORIG_SURFACE_RECORDS = A.surface_records
IM, JM, LM = A.IM, A.JM, A.LM


# ------------------------------------------------------------------------------------------------------------------ providers
class StepContext:
    """What a column modifier may look at."""

    def __init__(self, itime, S, provider, driver):
        self.itime, self.S, self.provider, self.driver = itime, S, provider, driver


class BoundaryProvider:
    """Abstract boundary provider (see the module docstring).  Subclass and override; unimplemented methods raise."""
    name = "abstract"

    def start(self, driver):
        self.driver = driver

    def real(self, itime):
        raise NotImplementedError

    def stage_radia(self, S, R, ctx):
        raise NotImplementedError

    def radiation_aij(self, itime):
        return None

    def surface_records(self, R):
        return _ORIG_SURFACE_RECORDS(R)

    def column_modifiers(self):
        return []

    def initial_seed(self, itime):
        raise NotImplementedError

    def daily_inputs(self, itime):
        return {}

    def state(self):
        return {}

    def set_state(self, d):
        pass

    def close(self):
        pass


class RecordProvider(BoundaryProvider):
    """Record-backed provider for the nov26 window (all radiation, surface record templates and day-boundary constants from the real dumps).
    Radiation: SRHR/TRHR of the last radiation step are held frozen over the next four steps exactly as RADIA does (atm_day_open_loop.RealRad),
    COSZ1 of every step is the recorded one."""
    name = "record"

    def __init__(self, date="nov26", daydir="nov26_day", ff=FF):
        self.date, self.daydir, self.ff = date, daydir, ff
        self.rad = None
        self.modifiers = []          # list of (name, fn, mode)

    def add_modifier(self, name, fn, mode="apply"):
        assert mode in ("apply", "shadow")
        self.modifiers.append((name, fn, mode))

    def column_modifiers(self):
        return list(self.modifiers)

    def real(self, itime):
        if self.rad is None or A.is_radiation_step(itime):
            rr = A.Real(self.daydir, itime, self.ff).site("r")
            self.rad = dict(SRHR=np.array(rr["SRHR"]), TRHR=np.array(rr["TRHR"]))
        return OL.RealRad(self.daydir, itime, self.ff, self.rad)

    def stage_radia(self, S, R, ctx):
        return A.__dict__["_driver_orig_stage_radia"](S, R, ctx)

    def radiation_aij(self, itime):
        if not A.is_radiation_step(itime):
            return None
        import radiation_server as rs
        return rs.read_packet(f"{self.ff}/{self.daydir}/rsv_n26_{itime}_out.bin")["AIJD"]

    def initial_seed(self, itime):
        return int(round(float(cio.read_cse(f"{self.ff}/{self.daydir}/ffc_cse_in_{itime}.bin")["SEEDS"][0]))) & 0xFFFFFFFF

    def daily_inputs(self, itime):
        mdrya, deltam, it_daily = OL.daily_mdrya(self.ff, self.date)
        dm, _ = OL.ch4ox_increment(self.ff, self.daydir, itime)
        sn = np.array(cio.read_cse(f"{self.ff}/{self.daydir}/ffc_cse_in_{itime}.bin")["SNOAGE"], copy=True)
        return dict(mdrya=mdrya, ch4ox_dm=dm, snoage_after=sn)

    def state(self):
        return dict(rad=self.rad)

    def set_state(self, d):
        self.rad = d["rad"]


class ServerRadiationProvider(RecordProvider):
    """Record-backed provider whose radiation stage is the REAL RADIA through the persistent radiation server (D159-D162): on every radiation step
    the 52-field packet is assembled from OUR state and the server returns SRHR/TRHR/COSZ1/... ; the RADIA seed is the one of OUR random chain.
    Only the nov26 restart has a server.  The server process is started lazily and is NOT part of the checkpoint (its carry RQT/KLIQ/SNOAGE is)."""
    name = "server"

    def __init__(self, date="nov26", daydir="nov26_day", ff=FF, out_dir=None, rundir=None):
        super().__init__(date, daydir, ff)
        import atm_day_free_rad as FRM
        import atm_day_free_rad_persist as FP
        self.FRM, self.FP = FRM, FP
        self.server_obj = FP.PersistentRadiation(date, ff, daydir, rundir=rundir, use_seed=True)
        self.FR = FRM.FreeRad(ff, daydir, "d178", out_dir, server=self._call)
        self._stage = FRM.make_stage_radia(self.FR)
        self.last_aij = None

    def _call(self, state, itime, rundir=None):
        srv = self.server_obj
        if srv.server is None:
            import radiation_server_persist as P
            srv.server = P.PersistentServer(srv.date, rundir=srv.rundir).start()
            srv.start_wall = srv.server.start_wall
        seed = drv_rng.radia_seed(self.driver.seed0)
        srv.seeds[itime] = seed
        out = srv.server.request(state, itime, seed)
        self.last_aij = np.array(out["AIJ"], copy=True)
        return out

    def real(self, itime):
        return self.FRM.FreeR(self.daydir, itime, self.ff, self.FR)

    def stage_radia(self, S, R, ctx):
        return self._stage(S, R, ctx)

    def radiation_aij(self, itime):
        return self.last_aij if A.is_radiation_step(itime) else None

    def set_snoage(self, sn):
        self.FR.snoage = np.array(sn, copy=True)

    def state(self):
        FR = self.FR
        return dict(rad=FR.rad, rqt=FR.rqt, kliq=FR.kliq, snoage=FR.snoage, calls=FR.calls)

    def set_state(self, d):
        FR = self.FR
        FR.rad, FR.rqt, FR.kliq, FR.snoage, FR.calls = d["rad"], d["rqt"], d["kliq"], d["snoage"], d["calls"]

    def close(self):
        self.server_obj.stop()


# ------------------------------------------------------------------------------------------------------------------ helpers
def _np(x):
    """jax / big-endian arrays -> native numpy (recursively through dict/list/tuple)."""
    if isinstance(x, dict):
        return {k: _np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_np(v) for v in x)
    if hasattr(x, "__array__") and not isinstance(x, np.ndarray) and not np.isscalar(x):
        return np.array(x)
    if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.dtype.byteorder not in ("=", "|"):
        return x.astype(np.float64)
    return x


def _obj_vars(o):
    d = dict(vars(o)) if hasattr(o, "__dict__") else {}
    for c in type(o).__mro__:
        for n in getattr(c, "__slots__", ()):
            if hasattr(o, n):
                d[n] = getattr(o, n)
    return d


def trees_equal(a, b, path="", out=None):
    """Exact (bitwise) comparison of two state trees; returns the list of differing paths."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing in one")
            else:
                trees_equal(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for k, (x, y) in enumerate(zip(a, b)):
                trees_equal(x, y, f"{path}[{k}]", out)
    elif type(a) is type(b) and (hasattr(a, "__dict__") or hasattr(a, "__slots__")):
        trees_equal(_obj_vars(a), _obj_vars(b), path + "." + type(a).__name__, out)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_, b_ = np.asarray(a), np.asarray(b)
        if a_.shape != b_.shape or a_.dtype != b_.dtype:
            out.append(f"{path}: shape/dtype {a_.shape}{a_.dtype} vs {b_.shape}{b_.dtype}")
        elif a_.dtype.kind == "f":
            ac, bc = np.ascontiguousarray(a_), np.ascontiguousarray(b_)
            if ac.tobytes() != bc.tobytes():
                u = np.uint64 if ac.dtype.itemsize == 8 else np.uint8
                out.append(f"{path}: {int((ac.view(u) != bc.view(u)).sum())} elements differ bitwise (max |diff| {np.nanmax(np.abs(ac - bc)):.3e})")
        elif not np.array_equal(a_, b_):
            out.append(f"{path}: integer/bool arrays differ")
    else:
        try:
            same = (a == b) or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))
        except Exception:
            same = False
        if not same:
            out.append(f"{path}: {a!r} vs {b!r}")
    return out


def tree_digest(x, h=None):
    """sha256 over the bytes of every array/scalar of a tree (key-sorted): a cheap per-step fingerprint for comparing runs."""
    h = h or hashlib.sha256()
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            h.update(str(k).encode())
            tree_digest(x[k], h)
    elif isinstance(x, (list, tuple)):
        for v in x:
            tree_digest(v, h)
    elif isinstance(x, np.ndarray):
        h.update(str(x.shape).encode() + str(x.dtype).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    elif x is not None:
        h.update(repr(x).encode())
    return h


# ------------------------------------------------------------------------------------------------------------------ the driver
class ModelDriver:
    """See the module docstring.

    surface   'closed' (surface_loop_v2.Loop2: ocean, ice, lake, land ice, GHY carried from OUR computation) | 'replay' (recorded surface
              records, atm_step.stage_surface, open-loop)
    ent       'record' (Ent exports from the ffg records, batched JAX GHY) | 'computed' (scalar GHY + Ent in the loop, land_chain_ent.install)
    rng       'chain' (RNDSS of CONDSE from the computed seed chain) | 'record' (recorded RNDSS; the chain is only checked)
    land_mode atm_step land_mode ('ghy' or 'recorded' (replay only))
    f3        accumulate the F3 diagnostics (F3Acc2) at their sites
    daily     'recorded' | 'computed' (drv_daily); see drv_daily.day_boundary | 'none' (no day boundary processing: the D172 chained window did none)
    """

    def __init__(self, provider=None, date="nov26", daydir="nov26_day", it0=33312, ff=FF, surface="closed", ent="record", rng="chain",
                 land_mode="ghy", f3=False, f3_ma1="own", daily="recorded", fast_condse=True, imf=True, out_dir=None, log_path=None,
                 ckpt_dir=None, ckpt_every=0, log=print, flags=None):
        self.provider = provider or RecordProvider(date, daydir, ff)
        self.date, self.daydir, self.it0, self.ff = date, daydir, it0, ff
        self.surface_mode, self.ent_mode, self.rng_mode, self.land_mode = surface, ent, rng, land_mode
        self.f3_on, self.f3_ma1, self.daily_mode, self.fast_condse, self.imf = f3, f3_ma1, daily, fast_condse, imf
        self.out_dir, self.log_path, self.ckpt_dir, self.ckpt_every, self.log = out_dir, log_path, ckpt_dir, ckpt_every, log
        self.flags = flags or {}
        self.itime = it0
        self.k = 0
        self.timing_log = []
        self.S = None                 # atmosphere state between steps (+ '_carry')
        self.ms = {}
        self.cur = None
        self.provider.start(self)
        t0 = time.perf_counter()
        self.ctx = A.make_ctx(date, imf=imf, ff=ff)
        F.ensure_backend(self.ctx)
        self.setup_time = {"ctx": time.perf_counter() - t0}
        self.loop = None
        self.ent = None
        self.acc = self.geo = None
        self.seed0 = self.provider.initial_seed(it0)
        if surface == "closed":
            import surface_loop as L
            import surface_loop_v2 as V2
            t0 = time.perf_counter()
            self.st = L.load_statics(date, ff)
            self.st["ctx"] = L.make_ocean_ctx(date, ff)
            SS = L.init_surface_state(date, ff, self.st)
            self.loop = V2.Loop2(date, daydir, ff, self.st, SS, flags=self.flags, it0=it0)
            self.setup_time["surface"] = time.perf_counter() - t0
        if ent == "computed":
            import land_chain_ent as LCE
            self.ent = LCE.EntLand(LCE.RESTART[daydir if daydir in LCE.RESTART else date], qf_mode="carry", dts_mode="computed")
        if f3:
            import f3_diagnostics as f3m
            import f3_diagnostics2 as f3b
            self.f3m, self.f3b = f3m, f3b
            _, self.geo = f3m.load_geo(ff, daydir)
            self.acc = f3b.F3Acc2(self.geo)

    # -------------------------------------------------------------------------------------------------- state / checkpoint
    def state_dict(self):
        """The complete model state as a tree of numpy arrays and plain Python values (what a checkpoint contains)."""
        d = dict(itime=self.itime, k=self.k, seed0=self.seed0, S=_np(self.S), ms=_np(self.ms), provider=_np(self.provider.state()),
                 timing_log=self.timing_log)
        if self.loop is not None:
            lp = self.loop
            V = lp.V
            d["surface"] = dict(SS=_np(lp.SS), land_prev=_np(lp.land_prev), usi=np.array(V.dyn.usi), vsi=np.array(V.dyn.vsi), adv=_np(V.adv),
                                uisurf=_np(V.uisurf), visurf=_np(V.visurf))
        if self.ent is not None:
            e = self.ent
            d["ent"] = dict(cells=e.cells, qf=dict(e.qf), itime=e.itime, sub=e.sub, nsteps=e.nsteps, log=e.log)
        if self.acc is not None:
            a = self.acc
            d["f3"] = dict(aij=_np(a.aij), aijl=_np(a.aijl), idacc=dict(a.idacc), s0=a.s0)
        return d

    def load_state_dict(self, d):
        self.itime, self.k, self.seed0 = d["itime"], d["k"], d["seed0"]
        self.S, self.ms = d["S"], d["ms"]
        self.provider.set_state(d["provider"])
        self.timing_log = d["timing_log"]
        if self.loop is not None:
            lp, s = self.loop, d["surface"]
            lp.SS, lp.land_prev = s["SS"], s["land_prev"]
            lp.V.dyn.usi, lp.V.dyn.vsi = s["usi"], s["vsi"]
            lp.V.adv, lp.V.uisurf, lp.V.visurf = s["adv"], s["uisurf"], s["visurf"]
        if self.ent is not None:
            e, s = self.ent, d["ent"]
            e.cells, e.qf, e.itime, e.sub, e.nsteps, e.log = s["cells"], s["qf"], s["itime"], s["sub"], s["nsteps"], s["log"]
        if self.acc is not None:
            a, s = self.acc, d["f3"]
            a.aij, a.aijl, a.idacc, a.s0 = s["aij"], s["aijl"], s["idacc"], s["s0"]

    def checkpoint(self, path=None):
        path = path or os.path.join(self.ckpt_dir, f"ckpt_{self.itime}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.perf_counter()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(dict(cfg=self.config(), state=self.state_dict()), f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path, time.perf_counter() - t0

    def resume(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["cfg"] != self.config():
            raise ValueError(f"checkpoint configuration differs: {d['cfg']} vs {self.config()}")
        self.load_state_dict(d["state"])
        return self

    def config(self):
        return dict(date=self.date, daydir=self.daydir, it0=self.it0, surface=self.surface_mode, ent=self.ent_mode, rng=self.rng_mode,
                    land_mode=self.land_mode, f3=self.f3_on, daily=self.daily_mode, provider=self.provider.name, imf=self.imf,
                    fast_condse=self.fast_condse, flags=self.flags)

    # -------------------------------------------------------------------------------------------------- patches for one step
    @contextlib.contextmanager
    def _patched(self, tm, R):
        saved = {n: getattr(A, n) for n in ("stage_surface", "stage_radia", "surface_records", "condse_inputs", "stage_dyn", "_substep")}
        A.__dict__["_driver_orig_stage_radia"] = saved["stage_radia"]
        undo_ent = None
        hk = None
        post_t = {"t": 0.0}
        import surface_loop_v2 as V2
        orig_post = V2.surface_post_v2
        try:
            if self.loop is not None:
                loop = self.loop

                def stage_surface(S, R_, ctx, rec=None, tm=None, land_mode="ghy"):
                    self.cur.S = S
                    return loop.stage_surface(S, R_, ctx, rec=rec, tm=tm, land_mode=land_mode)

                def timed_post(*a, **kw):
                    t0 = time.perf_counter()
                    try:
                        return orig_post(*a, **kw)
                    finally:
                        post_t["t"] += time.perf_counter() - t0
                V2.surface_post_v2 = timed_post
                A.stage_surface = stage_surface
            else:
                orig_surf = saved["stage_surface"]

                def stage_surface(S, R_, ctx, rec=None, tm=None, land_mode="ghy"):
                    self.cur.S = S
                    return orig_surf(S, R_, ctx, rec=rec, tm=tm, land_mode=land_mode)
                A.stage_surface = stage_surface
            A.stage_radia = self.provider.stage_radia
            A.surface_records = lambda R_: self._records(R_)
            A.condse_inputs = lambda S, R_: self._condse_inputs(saved["condse_inputs"], S, R_)
            if self.ent is not None:
                import land_chain_ent as LCE
                undo_ent = LCE.install(A, self.ent)
            if self.f3_on:
                import f3_diagnostics2 as f3b
                hk = f3b.ChainedHooks(A).install()
                self._wrap_substep_ma1(hk)
            ctxm = F.fast_condse() if self.fast_condse else contextlib.nullcontext()
            with ctxm:
                yield hk, post_t
        finally:
            V2.surface_post_v2 = orig_post
            if hk is not None:
                hk.uninstall()
            if undo_ent is not None:
                undo_ent()
            for n, v in saved.items():
                setattr(A, n, v)
            A.__dict__.pop("_driver_orig_stage_radia", None)

    def _wrap_substep_ma1(self, hk):
        inner = A._substep
        self._ma1 = []

        def sub(pbl12, tile, pbl3, li, blk, atm, dt, land):
            self._ma1.append(np.array(atm["MA1"], copy=True))
            return inner(pbl12, tile, pbl3, li, blk, atm, dt, land)
        A._substep = sub

    def _records(self, R):
        rec = self.provider.surface_records(R)
        for name, fn, mode in self.provider.column_modifiers():
            before = {k: np.array(v, copy=True) for k, v in rec.items()}
            out = fn({k: np.array(v, copy=True) for k, v in rec.items()}, StepContext(self.itime, self.cur.S, self.provider, self))
            nchg = {k: int((np.asarray(out[k]) != before[k]).sum()) for k in before if k in out and np.asarray(out[k]).shape == before[k].shape}
            self.cur.modifier_log[name] = dict(mode=mode, changed={k: v for k, v in nchg.items() if v})
            if mode == "apply":
                rec = {**rec, **out}
        return rec

    def _condse_inputs(self, orig, S, R):
        inp = orig(S, R)
        try:
            rec_rnd = np.array(inp["RNDSS"], copy=True)
        except KeyError:
            rec_rnd = None
        rnd, _ = cf.randu_stream(self.seed0, self.ctx.imaxj, self.ctx.cfg["lmcld"])
        lm = self.ctx.cfg["lmcld"]
        if rec_rnd is not None:
            vm = A.imaxj_mask(self.ctx)[None, None, :, :]                 # defined draws only (pole rows: i = 1)
            self.cur.rng_check = bool(np.array_equal(np.where(vm, rnd[:, :lm], 0.0), np.where(vm, rec_rnd[:, :lm], 0.0)))
        if self.rng_mode == "chain":
            inp["RNDSS"] = rnd
        return inp

    # -------------------------------------------------------------------------------------------------- day boundary
    def _day_boundary(self, it):
        import drv_daily
        return drv_daily.day_boundary(self, it)

    # -------------------------------------------------------------------------------------------------- one step
    def step(self):
        it = self.itime
        t_all = time.perf_counter()
        tm = {}
        self.cur = type("Cur", (), {})()
        self.cur.S = self.S
        self.cur.modifier_log = {}
        self.cur.rng_check = None
        row = dict(itime=it, k=self.k)
        t0 = time.perf_counter()
        if self.S is not None and it % NDAY == 0 and self.daily_mode != "none":
            row["daily"] = self._day_boundary(it)
        row["t_daily"] = time.perf_counter() - t0
        R = self.provider.real(it)
        t0 = time.perf_counter()
        if self.loop is not None:
            self.loop.pre_cse(R)                                    # MELT_SI (ATM_DRV.f:257), before CONDSE
        row["t_melt_si"] = time.perf_counter() - t0
        land_mode = self.land_mode
        with self._patched(tm, R) as (hk, post_t):
            if hk is not None:
                hk.calls.clear()
                self._ma1.clear()
            t0 = time.perf_counter()
            S, sn = A.run_step(self.date, it, self.ctx, R=R, S=self.S, ms=self.ms, land_mode=land_mode, timing=tm)
            row["t_run_step"] = time.perf_counter() - t0
            X = S.get("_condse_X")
            carry = {key: np.array(X[key], copy=True) for key in A.CARRY_KEYS if X is not None and key in X}
            if "_cloud_rad" in S:
                carry["CLDSS"], carry["CLDMC"] = (np.array(a, copy=True) for a in S["_cloud_rad"])
            if "_snoage_rad" in S:
                carry["SNOAGE"] = np.array(S["_snoage_rad"], copy=True)
            t0 = time.perf_counter()
            if self.f3_on:
                self._f3_sites(it, R, S, sn, X, hk)
            row["t_f3"] = time.perf_counter() - t0
            surf_post = post_t["t"]
        row["stages"] = {k: v for k, v in tm.items() if k.startswith("stage_")}
        row["surface_tiles"] = tm.get("surface", 0.0)
        row["surface_post"] = surf_post
        row["rng_chain_equals_record"] = self.cur.rng_check
        row["modifiers"] = self.cur.modifier_log
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            np.savez(f"{self.out_dir}/step_{it}.npz", **{f: np.asarray(S[f], float) for f in ("T", "Q", "U", "V", "QCL", "QCI", "P")})
        # --- carry to the next step; next seed (RADIA adds draws on radiation steps)
        self.seed0 = drv_rng.next_seed(self.seed0, A.is_radiation_step(it))
        S = {key: v for key, v in S.items() if not key.startswith("_")}
        if carry:
            S["_carry"] = carry
        self.S = _np(S)
        self.itime += 1
        self.k += 1
        row["wall"] = time.perf_counter() - t_all
        row["digest"] = tree_digest(dict(S=self.S, ms=_np(self.ms))).hexdigest()[:16]
        if self.ckpt_dir and self.ckpt_every and self.k % self.ckpt_every == 0:
            p, dt = self.checkpoint()
            row["checkpoint"] = dict(path=p, seconds=dt)
        self.timing_log.append(row)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    def run(self, nsteps):
        for _ in range(nsteps):
            r = self.step()
            self.log(f"step {r['k']:3d} it {r['itime']} wall {r['wall']:6.1f}s run_step {r['t_run_step']:.1f}s (tiles {r['surface_tiles']:.1f} post {r['surface_post']:.1f}"
                     f" f3 {r['t_f3']:.1f}) digest {r['digest']}" + (f" ckpt {r['checkpoint']['seconds']:.1f}s" if "checkpoint" in r else ""))
            sys.stdout.flush()
        return self

    # -------------------------------------------------------------------------------------------------- F3 sites (as f3_diagnostics2.run_chained_window)
    def _f3_sites(self, it, R, S, sn, X, hk):
        f3m, f3b, acc = self.f3m, self.f3b, self.acc
        if hk.diaga_w is not None:
            w = dict(hk.diaga_w)
            w["PEK1"] = None
            acc.diaga(w, self.geo, hk.start["TSAVG"], hk.start["QSAVG"], imf_pow=bool(self.ctx.dyn.imf_pow if hasattr(self.ctx, "dyn") else True))
        rad_step = A.is_radiation_step(it)
        acc.radia(np.asarray(S["COSZ1"]), rad_step, self.provider.radiation_aij(it) if rad_step else None)
        blks = [c["blk"] for c in hk.calls]
        pk = {ns: f3b.packet_chained(hk.calls[ns - 1], blks[ns - 1]) for ns in (1, 2)}
        sub = {}
        for ns in (1, 2):
            comp = hk.calls[ns - 1]["res"]["comp"]
            r_ = hk.calls[ns - 1]
            g = lambda v: A._grid(v, r_["res"], 0.0)          # noqa: E731
            ma1 = np.asarray(R.a["MA"])[0] if self.f3_ma1 == "record" else self._ma1[ns - 1].T
            dq1 = g(comp["dq1"])
            sub[ns] = dict(TSAVG=g(comp["tsavg"]), QSAVG=g(comp["qsavg"]), QFLUX1=-dq1 * ma1 / 900.0, UFLUX1=g(comp["uflux1"]),
                           VFLUX1=g(comp["vflux1"]))
        trhr0 = np.asarray(S["TRHR"])[0]
        acc.surface(it, sub, trhr0)
        ddms = S["DDMS"] if "DDMS" in S else R.cse_out["DDMS"]
        acc.surface_site(it, pk, ddms)
        acc.surface_flux([pk[1], pk[2]], trhr0)
        acc.prec(S["PREC"])
        acc.condse2({**{kk: S[kk] for kk in ("PREC", "EPREC", "PRECSS", "QCL", "QCI")}, "LMC": X["LMC"], "CLDMC": X["CLDMC"]}, S["PDSIG"],
                    np.asarray(sn["condse"]["PEDN"]))
        acc.dyn2(hk.w)
        acc.airmass(np.asarray(sn["filter"]["MA"]))

    def f3_layout(self):
        return self.acc.to_nc_layout() if self.acc is not None else None

    def save_f3(self, path):
        """The accumulated AIJ/AIJL in the format of f3_diagnostics2.run_chained_window(out_npz=...) (input of f3_chained_compare / f3_score)."""
        a, al = self.acc.to_nc_layout()
        np.savez(path, aij=a, aijl=al, idacc=np.array([self.acc.idacc[i] for i in (1, 2, 3, 4)]), cols=np.array(sorted(self.acc.aij)))


def f3_npz_from_checkpoint(ckpt, out_npz):
    """Write the F3 accumulators stored in a checkpoint in the run_chained_window npz format."""
    import f3_diagnostics2 as f3b
    with open(ckpt, "rb") as f:
        d = pickle.load(f)["state"]["f3"]
    acc = f3b.F3Acc2()
    acc.aij, acc.aijl, acc.idacc, acc.s0 = d["aij"], d["aijl"], d["idacc"], d["s0"]
    a, al = acc.to_nc_layout()
    np.savez(out_npz, aij=a, aijl=al, idacc=np.array([acc.idacc[i] for i in (1, 2, 3, 4)]), cols=np.array(sorted(acc.aij)))


# ------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--surface", default="closed", choices=("closed", "replay"))
    ap.add_argument("--ent", default="record", choices=("record", "computed"))
    ap.add_argument("--rng", default="chain", choices=("chain", "record"))
    ap.add_argument("--land-mode", default="ghy")
    ap.add_argument("--f3", action="store_true")
    ap.add_argument("--f3-ma1", default="own")
    ap.add_argument("--daily", default="recorded")
    ap.add_argument("--server", action="store_true", help="radiation stage through the persistent radiation server")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--stop-after", type=int, default=None, help="stop (and checkpoint) after this many steps in total")
    ap.add_argument("--save-final", default=None)
    a = ap.parse_args(argv)
    out_dir = f"{FF}/nov26_day/ours_d178_{a.tag}"
    prov = ServerRadiationProvider(out_dir=None) if a.server else RecordProvider()
    d = ModelDriver(prov, surface=a.surface, ent=a.ent, rng=a.rng, land_mode=a.land_mode, f3=a.f3, f3_ma1=a.f3_ma1, daily=a.daily,
                    out_dir=out_dir, log_path=f"{out_dir}.log.jsonl", ckpt_dir=a.ckpt_dir, ckpt_every=a.ckpt_every)
    if a.resume:
        d.resume(a.resume)
    n = (a.stop_after if a.stop_after is not None else a.steps) - d.k
    try:
        d.run(n)
    finally:
        prov.close()
    if a.save_final:
        d.checkpoint(a.save_final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
