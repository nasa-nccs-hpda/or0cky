"""D155-D157: one-model-day run of the chained atmosphere with FREE-RUNNING radiation (real RADIA through the radiation server).

Differs from atm_day_open_loop.py (D150) in exactly one thing: on every radiation step (itime-ITIMEI = 0 mod NRAD=5; k = 0, 5, ..., 50 of the
54-step window) the RADIA stage is not replayed from the recorded SRHR/TRHR but computed by the real, unmodified RADIA (SOCRATES as a black box)
through radiation_server.run_radiation, with the 52-field packet assembled from OUR chained state at that step:
  OUR state            : T, Q (after OUR CONDSE of this step), PK, PMID, PDSIG, PEDN, MA, BYMA (= 1/MA), LTROPO (our dynamics), the 15 cloud hand-off
                         arrays + TAUSS/TAUMC/CLDSS/CLDMC (OUR CONDSE exit), SNOAGE (carried: server output of the previous call, as RADIA carries it),
                         RQT and KLIQ (radiation-only memory, carried from the previous server output; the restart value of the live model at step 0).
  REPLAYED surface side: RSI ZSI SNOWI POND_MELT FLAG_DSWS FLAKE DLAKE FLICE FLAND FEARTH GTEMPR1-4 TSAVG WSAVG SNOWLI ZSNOWI BARESW FRSNOW SNOWD =
                         the live values of the real model at that step (dump-mode packets rsv_n26_<it>_in.bin, recorded by dump54, see below).
  Server outputs used where the open-loop code used the recorded ones: SRHR, TRHR (frozen over the next four steps exactly as RADIA does), COSZ1,
  the masked CLDSS/CLDMC (-> carried to the next CONDSE entry), Q (negative-Q reset), SNOAGE, RQT, KLIQ (carried).  T is advanced with the same
  radia_apply of atm_step.py (RAD_DRV.f:5474-5478); the server's own T is recorded as a consistency check.
Everything else (surface tile/PBL/land records, ocean/ice/lake state, Ent, day-boundary handling, daily_ch4ox, per-step COSZ1 on non-radiation
steps) is as in atm_day_open_loop.py.  The surface does NOT see our radiation (the tile SRHEAT/TRSURF records are the real ones): this is a free-running
ATMOSPHERE over a replayed surface.

Usage (fullfidelity/, conda python):
  python atm_day_free_rad.py run    --tag free_np [--nsteps 54]     # the day (11 server calls, sequential)
  python atm_day_free_rad.py report --tag free_np                   # D156/D157 tables (md + json in ff_data/nov26_day/ours_<tag>/)
  python atm_day_free_rad.py check0                                 # step-0 plumbing oracle through the assembler (one server call)
Outputs: ff_data/nov26_day/ours_<tag>/step_<it>.npz (T Q U V QCL QCI P), run.json, rad_in_<it>.npz (the assembled packets), rad_out_<it>.npz
(server outputs; AIJ reduced to the TOA columns).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import atm_step as A
import atm_step_fast as F
import atm_day_open_loop as OL
import atm_day_report as RP
import clouds_condse_io as cio
import dyn_glue_io as gio
import radiation_server as rs

DAYDIR = OL.DAYDIR
IT0 = OL.IT0
FF = cio.FF_DEFAULT
SCRATCH_RUNS = os.environ.get("FREERAD_RUNS", os.path.join(os.path.dirname(rs.SCRATCH), "mE_freeday"))
# AIJ columns (1-based accumulator numbers; kaij of this build is 1660). A static count of DEFACC.f ij_defs gave 225/239/224/233 (all conditional
# blocks of the source skipped), which hit empty columns; the real numbers are that count + 152 (an offset NOT traced to its source), identified and
# verified by value in verify_aij_columns: SRINCP0 gmean 349.6 W m-2 (= S0/RSDIST^2/4), SRNFP0 237.1, TRNFP0 -230.4, SRNFG gmean = gmean(SRHR(0)*COSZ1).
AIJ_COLS = {"SRNFP0": 377, "TRNFP0": 391, "SRINCP0": 376, "SRNFG": 385}

ATM_FROM_STATE = ("PK", "PMID", "PDSIG", "PEDN", "MA")
CLOUD15 = ("W_CLOUD", "FRAC_ST_WATER", "FRAC_ST_ICE", "FRAC_CNV_WATER", "FRAC_CNV_ICE", "MIX_ST_WATER", "MIX_ST_ICE", "MIX_CNV_WATER",
           "MIX_CNV_ICE", "DIM_ST_WATER", "DIM_ST_ICE", "DIM_CNV_WATER", "DIM_CNV_ICE", "FRAC_AREA_ST", "FRAC_AREA_CNV")
CLOUD_FROM_X = CLOUD15 + ("TAUSS", "TAUMC", "CLDSS", "CLDMC")
OURS = ("T", "Q", "PK", "PMID", "PDSIG", "MA", "BYMA", "PEDN") + CLOUD_FROM_X + ("SNOAGE", "RQT", "KLIQ", "LTROPO")
LIVE = tuple(k for k in rs.INPUT_FIELDS if k not in OURS)        # the replayed surface side (19 fields minus RQT/KLIQ/LTROPO/BYMA + ...)


# ------------------------------------------------------------------------------------------------ live (real) packets
def live_path(it, kind, ff=FF, daydir=DAYDIR):
    return f"{ff}/{daydir}/rsv_n26_{it}_{kind}.bin"


def load_live(it, kind="in", ff=FF, daydir=DAYDIR):
    return rs.read_packet(live_path(it, kind, ff, daydir))


def rad_steps(it0=IT0, nsteps=54):
    return [it0 + k for k in range(nsteps) if A.is_radiation_step(it0 + k)]


# ------------------------------------------------------------------------------------------------ the server with a day-long window
def run_radiation_day(state, itime, rundir=None, restart=None):
    """radiation_server.run_radiation with the model window of the day run (end 1950-11-27 03:00 = 54 steps).  radiation_server._prepare ends the
    window at 1950-11-26 03:00 (6 steps; enough for the D154 oracle at steps 0 and 5): a server asked for a later radiation step would run to the end
    of that window without ever reaching the hook (found the hard way: step 10 returned no packet)."""
    rundir = rundir or os.path.join(SCRATCH_RUNS, "run_day_%d" % os.getpid())
    restart = restart or rs.RESTARTS[33312]
    rs._prepare(rundir, restart)
    txt = open(os.path.join(rundir, "I")).read()
    assert "YEARE=1950,MONTHE=11,DATEE=26,HOURE=3," in txt
    open(os.path.join(rundir, "I"), "w").write(txt.replace("YEARE=1950,MONTHE=11,DATEE=26,HOURE=3,", "YEARE=1950,MONTHE=11,DATEE=27,HOURE=3,"))
    write = os.path.join(rundir, "rsv_in.bin")
    rs.write_packet(write, state)
    import subprocess
    env = rs._env({"RADSRV_MODE": "serve", "RADSRV_ITIME": str(itime), "RADSRV_IN": "rsv_in.bin", "RADSRV_OUT": "rsv_out.bin"})
    t0 = time.time()
    p = subprocess.Popen(["./P2SAoM40", "-i", "I"], cwd=rundir, env=env, stdout=open(os.path.join(rundir, "run.PRT"), "w"), stderr=subprocess.STDOUT)
    return rs.collect(p, rundir, t0)


# ------------------------------------------------------------------------------------------------ packet assembly
def assemble_packet(S, X, snoage, rqt, kliq, live):
    """52-field RADIA packet from a chained state S (post-CONDSE T, Q and the dynamics fields), the CONDSE exit X, the carried
    SNOAGE/RQT/KLIQ, and the live packet of the real model (replayed surface side).  Returns (state dict, source dict)."""
    st, src = {}, {}
    for k in rs.INPUT_FIELDS:
        if k in ("T", "Q") or k in ATM_FROM_STATE:
            st[k], src[k] = np.asarray(S[k], float), "our state"
        elif k == "BYMA":
            st[k], src[k] = 1.0 / np.asarray(S["MA"], float), "our state (1/MA)"
        elif k in CLOUD_FROM_X:
            st[k], src[k] = np.asarray(X[k], float), "our CONDSE exit"
        elif k == "SNOAGE":
            st[k], src[k] = np.asarray(snoage, float), "carried (server output)"
        elif k == "RQT":
            st[k], src[k] = np.asarray(rqt, float), "carried (server output)"
        elif k == "KLIQ":
            st[k], src[k] = np.asarray(kliq, float), "carried (server output)"
        elif k == "LTROPO" and "LTROPO" in S and np.asarray(S["LTROPO"]).shape == rs.INPUT_FIELDS[k]:
            st[k], src[k] = np.asarray(S["LTROPO"], float), "our dynamics"
        else:
            st[k], src[k] = np.asarray(live[k], float), "replayed (live real model)"
    return st, src


def packet_vs_live(st, live):
    out = {}
    for k in rs.INPUT_FIELDS:
        a, b = np.asarray(st[k], float), np.asarray(live[k], float)
        d = np.abs(a - b)
        out[k] = dict(equal=bool(np.array_equal(a, b)), maxabs=float(d.max()), ndiff=int((a != b).sum()), maxval=float(np.abs(b).max()))
    return out


# ------------------------------------------------------------------------------------------------ the free-radiation stage
class FreeRad:
    """State of the free-running radiation: frozen heating rates, carried SNOAGE/RQT/KLIQ, server call log."""

    def __init__(self, ff, daydir, tag, out_dir, runs_dir=SCRATCH_RUNS, keep_run=False, server=None):
        self.ff, self.daydir, self.tag, self.out_dir, self.runs_dir, self.keep_run = ff, daydir, tag, out_dir, runs_dir, keep_run
        self.server = server or run_radiation_day
        self.rad = None                     # dict(SRHR, TRHR)
        self.rqt = self.kliq = self.snoage = None
        self.calls = []
        self.last_out = None

    def call(self, it, S, X):
        live = load_live(it, "in", self.ff, self.daydir)
        rqt = live["RQT"] if self.rqt is None else self.rqt
        kliq = live["KLIQ"] if self.kliq is None else self.kliq
        sno = X["SNOAGE"] if self.snoage is None else self.snoage
        st, src = assemble_packet(S, X, sno, rqt, kliq, live)
        rundir = os.path.join(self.runs_dir, f"run_free_{self.tag}_{it}")
        t0 = time.time()
        out = self.server(st, it, rundir=rundir)
        wall = time.time() - t0
        if not self.keep_run:
            shutil.rmtree(rundir, ignore_errors=True)
        if self.out_dir:
            np.savez_compressed(f"{self.out_dir}/rad_in_{it}.npz", **st)
            red = {k: v for k, v in out.items() if k not in ("AIJ", "_wall_s")}
            red["AIJ_COLS"] = np.stack([out["AIJ"][:, :, c - 1] for c in AIJ_COLS.values()], axis=-1)
            np.savez_compressed(f"{self.out_dir}/rad_out_{it}.npz", **red)
        rec = dict(itime=it, wall=wall, server_wall=float(out["_wall_s"]),
                   input_diff_vs_live=dict(
                       carried_RQT=float(np.abs(rqt - live["RQT"]).max()), carried_KLIQ=float(np.abs(kliq - live["KLIQ"]).max()),
                       carried_SNOAGE=float(np.abs(sno - live["SNOAGE"]).max())
                       if "SNOAGE" in live else None,
                       **{k: float(np.abs(st[k] - live[k]).max()) for k in ("T", "Q", "CLDSS", "CLDMC", "TAUSS", "W_CLOUD", "PEDN", "MIX_ST_WATER")}))
        self.calls.append(rec)
        self.rqt, self.kliq, self.snoage = out["RQT"], out["KLIQ"], out["SNOAGE"]
        self.last_out = out
        return out


def make_stage_radia(FR):
    def stage_radia(S, R, ctx):
        it = R.itime
        if A.is_radiation_step(it):
            X = S.get("_condse_X")
            out = FR.call(it, S, X)
            FR.rad = dict(SRHR=np.array(out["SRHR"]), TRHR=np.array(out["TRHR"]))
            cosz1 = np.array(out["COSZ1"])
            S["SRHR"], S["TRHR"], S["COSZ1"] = FR.rad["SRHR"], FR.rad["TRHR"], cosz1
            t_new = A.radia_apply(S["T"], S["SRHR"], S["TRHR"], S["COSZ1"], S["MA"], S["PK"], ctx)
            FR.calls[-1]["T_apply_vs_server"] = float(np.abs(t_new - out["T"]).max())
            FR.calls[-1]["cosz1_vs_recorded"] = float(np.abs(cosz1 - np.array(R.site("r")["COSZ1"])).max())
            ref = A.radia_cloud_masking(X["CLDSS"], X["CLDMC"], X["TAUSS"], X["TAUMC"])
            FR.calls[-1]["cloudmask_vs_server"] = float(max(np.abs(ref[0] - out["CLDSS"]).max(), np.abs(ref[1] - out["CLDMC"]).max()))
            S["T"] = t_new
            S["Q"] = np.array(out["Q"], copy=True)               # RADIA's Q<0 -> 0 reset (state)
            S["_cloud_rad"] = (np.array(out["CLDSS"], copy=True), np.array(out["CLDMC"], copy=True))
            S["_snoage_rad"] = np.array(out["SNOAGE"], copy=True)
        else:
            S["SRHR"], S["TRHR"] = FR.rad["SRHR"], FR.rad["TRHR"]
            S["COSZ1"] = np.array(R.site("r")["COSZ1"])
            S["T"] = A.radia_apply(S["T"], S["SRHR"], S["TRHR"], S["COSZ1"], S["MA"], S["PK"], ctx)
        return S
    return stage_radia


class FreeR(A.Real):
    """A.Real whose radiation record carries the (free-running) frozen SRHR/TRHR of OUR last radiation step."""

    def __init__(self, date, itime, ff, FR):
        super().__init__(date, itime, ff)
        self.FR = FR

    @property
    def r(self):
        d = dict(self.site("r"))
        if self.FR.rad is not None:
            d["SRHR"], d["TRHR"] = self.FR.rad["SRHR"], self.FR.rad["TRHR"]
        return d


# ------------------------------------------------------------------------------------------------ the day
def run_day_free(date=OL.DATE, it0=IT0, nsteps=54, imf=True, land="recorded", daydir=DAYDIR, ff=FF, tag="free_np", save=True, log=print,
                 keep_run=False, server=None):
    ctx = A.make_ctx(date, imf=imf, ff=ff)
    F.ensure_backend(ctx)
    mdrya, deltam_real, it_daily = OL.daily_mdrya(ff, date)
    out_dir = f"{ff}/{daydir}/ours_{tag}"
    if save:
        os.makedirs(out_dir, exist_ok=True)
    FR = FreeRad(ff, daydir, tag, out_dir if save else None, keep_run=keep_run, server=server)
    axyp = ctx.gg["axyp"]
    imaxj = ctx.imaxj
    S, ms, rows = None, {}, []
    orig_radia = A.stage_radia
    A.stage_radia = make_stage_radia(FR)
    t_all = time.perf_counter()
    try:
        for k in range(nsteps):
            it = it0 + k
            rec_daily = None
            R = FreeR(daydir, it, ff, FR)
            if S is not None and (it % OL.NDAY) == 0 and mdrya is not None:
                rec_daily = OL.apply_daily(S, ctx, mdrya)
                dm, _ = OL.ch4ox_increment(ff, daydir, it)
                OL.apply_ch4ox(S, ctx, dm)
                carry = S.get("_carry", {})
                if "SNOAGE" in carry:                         # daily_RAD aging: recorded (as in D150); the server carry follows
                    sn_new = np.array(R.cse_in["SNOAGE"], copy=True)
                    rec_daily["snoage_change_recorded"] = float(np.abs(sn_new - carry["SNOAGE"]).max())
                    carry["SNOAGE"] = sn_new
                    FR.snoage = sn_new
            tm = {}
            t0 = time.perf_counter()
            with F.fast_condse():
                S, sn = A.run_step(date, it, ctx, R=R, S=S, ms=ms, land_mode=land, timing=tm)
            wall = time.perf_counter() - t0
            real_e = R.e
            w = RP.weights(real_e["MA"], axyp)
            end = {f: S[f] for f in RP.FIELDS + ("P",)}
            st = RP.state_metrics(end, real_e, w)
            X = S.get("_condse_X")
            cl = OL.cloud_stats(X, R.cse_out, imaxj) if X is not None else {}
            row = dict(step=k, itime=it, wall=wall, stages={a: b for a, b in tm.items() if a.startswith("stage_")}, stats=st, clouds=cl,
                       radiation_step=bool(A.is_radiation_step(it)), daily=rec_daily,
                       rad_call=FR.calls[-1] if (FR.calls and FR.calls[-1]["itime"] == it) else None)
            rows.append(row)
            if save:
                np.savez(f"{out_dir}/step_{it}.npz", **{f: np.asarray(end[f], float) for f in end})
            carry = {key: np.array(X[key], copy=True) for key in A.CARRY_KEYS if X is not None and key in X}
            if "_cloud_rad" in S:
                carry["CLDSS"], carry["CLDMC"] = (np.array(a, copy=True) for a in S["_cloud_rad"])
            if "_snoage_rad" in S:
                carry["SNOAGE"] = np.array(S["_snoage_rad"], copy=True)
            elif FR.snoage is not None:
                carry["SNOAGE"] = np.array(FR.snoage, copy=True)
            S = {key: v for key, v in S.items() if not key.startswith("_")}
            if carry:
                S["_carry"] = carry
            rc = row["rad_call"]
            log(f"step {k:2d} it {it} {wall:6.1f}s  rms T {st['T']['rms']:.2e} Q {st['Q']['rms']:.2e} U {st['U']['rms']:.2e} V {st['V']['rms']:.2e} "
                f"P {st['P']['rms']:.2e} | gmeanT {st['T']['gmean']:+.2e} gmeanQ {st['Q']['gmean']:+.2e}"
                + (f" | RADSERVER {rc['wall']:.0f}s dT_apply-vs-srv {rc.get('T_apply_vs_server', float('nan')):.1e}" if rc else ""))
            sys.stdout.flush()
    finally:
        A.stage_radia = orig_radia
    total = time.perf_counter() - t_all
    res = dict(date=date, it0=it0, nsteps=nsteps, imf=imf, dyn="numpy", land=land, tag=tag, total_wall=total, rows=rows, mdrya=mdrya,
               server_wall_total=float(sum(c["wall"] for c in FR.calls)), calls=FR.calls)
    if save:
        json.dump(RP.to_jsonable(res), open(f"{out_dir}/run.json", "w"))
    return res


# ------------------------------------------------------------------------------------------------ step-0 oracle through the assembler
def check_step0(ff=FF, daydir=DAYDIR, log=print, runs_dir=SCRATCH_RUNS):
    """Plumbing oracle: the assembler is applied to the REAL step-0 state (T,Q from ffc_cse_out, PK.. from ffc_cse_in, MA from ffa_step_r, the
    cloud arrays from ffc_cse_out) exactly as it is applied to our chained state; the server must return the recorded radiation bitwise."""
    it = IT0
    cin = cio.read_cse(f"{ff}/{daydir}/ffc_cse_in_{it}.bin")
    cout = cio.read_cse(f"{ff}/{daydir}/ffc_cse_out_{it}.bin")
    r = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it}_r.bin")
    live = load_live(it, "in", ff, daydir)
    live_out = load_live(it, "out", ff, daydir)
    S = dict(T=cout["T"], Q=cout["Q"], PK=cin["PK"], PMID=cin["PMID"], PDSIG=cin["PDSIG"], PEDN=cin["PEDN"], MA=r["MA"])
    st, src = assemble_packet(S, cout, cout["SNOAGE"], live["RQT"], live["KLIQ"], live)
    pv = packet_vs_live(st, live)
    out = run_radiation_day(st, it, rundir=os.path.join(runs_dir, "run_check0"))
    shutil.rmtree(os.path.join(runs_dir, "run_check0"), ignore_errors=True)
    res = dict(packet_vs_live={k: v["equal"] for k, v in pv.items()}, n_fields=len(pv), n_equal=sum(v["equal"] for v in pv.values()),
               sources={k: src[k] for k in src}, wall=out["_wall_s"], outputs={})
    for k in ("T", "Q", "SRHR", "TRHR", "COSZ1"):
        res["outputs"][k + " vs recorded ffa_step_r"] = bool(np.array_equal(out[k], r[k]))
    for k in rs.OUTPUT_FIELDS:
        if k != "AIJ":
            res["outputs"][k + " vs live out"] = bool(np.array_equal(out[k], live_out[k]))
    res["aij_cols_vs_live_aijd_maxabs"] = {n: float(np.abs(out["AIJ"][:, :, c - 1] - live_out["AIJD"][:, :, c - 1]).max()) for n, c in AIJ_COLS.items()}
    log(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ D156/D157 report
def _wmean(f, axyp, imaxj=None):
    """area-weighted global mean of an (IM,JM) field (pole rows: one cell each, weighted by the row area as one cell)."""
    m = RP.valid_mask("T")
    ax = np.asarray(axyp, float)
    ax = ax[None, :] * np.ones((RP.IM, 1)) if ax.ndim == 1 else ax
    return float((f * ax)[m].sum() / ax[m].sum())


def _rms2(f):
    m = RP.valid_mask("T")
    return float(np.sqrt(np.mean(f[m] ** 2)))


def verify_aij_columns(ff=FF, daydir=DAYDIR, axyp=None):
    """data check of the AIJ column numbers: SRNFP0 >= 0 and zero at night, TRNFP0 < 0, SRINCP0 > SRNFP0, SRNFG ~ surface net SW (SRHR(0)*COSZ1)."""
    lo = load_live(IT0, "out", ff, daydir)
    r = cio.read_cse(f"{ff}/{daydir}/ffa_step_{IT0}_r.bin")
    res = {}
    for n, c in AIJ_COLS.items():
        a = lo["AIJD"][:, :, c - 1]
        res[n] = dict(gmean=_wmean(a, axyp) if axyp is not None else None, min=float(a.min()), max=float(a.max()))
    sw0 = np.transpose(r["SRHR"], (1, 2, 0))[:, :, 0] * r["COSZ1"]
    res["SRHR0*COSZ1_gmean"] = _wmean(sw0, axyp) if axyp is not None else None
    res["SRNFG_vs_SRHR0*COSZ1_maxabs"] = float(np.abs(lo["AIJD"][:, :, AIJ_COLS["SRNFG"] - 1] - sw0).max())
    return res


def flux_comparison(ff, daydir, ours_dir, it0, nsteps, axyp):
    """per radiation step: ours (free, saved server output) vs the real run (recorded ffa_step_r for SRHR/TRHR; dump-mode out packet for the rest)."""
    rows = []
    for it in rad_steps(it0, nsteps):
        o = np.load(f"{ours_dir}/rad_out_{it}.npz")
        lv = load_live(it, "out", ff, daydir)
        r = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it}_r.bin")
        rr = dict(itime=it, k=it - it0)
        cz_o, cz_r = o["COSZ1"], r["COSZ1"]
        sr_o, sr_r = np.transpose(o["SRHR"], (1, 2, 0)), np.transpose(r["SRHR"], (1, 2, 0))
        tr_o, tr_r = np.transpose(o["TRHR"], (1, 2, 0)), np.transpose(r["TRHR"], (1, 2, 0))
        net_o = sr_o[:, :, 0] * cz_o + tr_o[:, :, 0]
        net_r = sr_r[:, :, 0] * cz_r + tr_r[:, :, 0]
        rr["surf_abs_gmean_real"], rr["surf_abs_gmean_ours"] = _wmean(net_r, axyp), _wmean(net_o, axyp)
        rr["surf_abs_gmean_diff"] = rr["surf_abs_gmean_ours"] - rr["surf_abs_gmean_real"]
        rr["surf_abs_rms_diff"] = _rms2(net_o - net_r)
        rr["surf_abs_rms_real"] = _rms2(net_r)
        # SRHR(0)*COSZ1 = net SW at the surface (SRNFLB(1)); TRHR(0) = STBO*sum(p TGR^4) - TRNFLB(1) = downward LW at the surface (RAD_DRV.f 4437-4551)
        rr["SWnet_sfc_gmean_diff"] = _wmean(sr_o[:, :, 0] * cz_o - sr_r[:, :, 0] * cz_r, axyp)
        rr["SWnet_sfc_rms_diff"] = _rms2(sr_o[:, :, 0] * cz_o - sr_r[:, :, 0] * cz_r)
        rr["LWdn_sfc_gmean_diff"] = _wmean(tr_o[:, :, 0] - tr_r[:, :, 0], axyp)
        rr["LWdn_sfc_rms_diff"] = _rms2(tr_o[:, :, 0] - tr_r[:, :, 0])
        # heating-rate columns: SRHR*COSZ1 and TRHR, layers 1..40 (K-equivalent W/m2 units as exported)
        hs_o, hs_r = sr_o[:, :, 1:] * cz_o[:, :, None], sr_r[:, :, 1:] * cz_r[:, :, None]
        m = RP.valid_mask("T")[:, :, None] & np.ones((1, 1, 40), bool)
        rr["SRHRcos_rms_diff"] = float(np.sqrt(np.mean(((hs_o - hs_r)[m]) ** 2)))
        rr["SRHRcos_rms_real"] = float(np.sqrt(np.mean(hs_r[m] ** 2)))
        rr["TRHR_rms_diff"] = float(np.sqrt(np.mean(((tr_o[:, :, 1:] - tr_r[:, :, 1:])[m]) ** 2)))
        rr["TRHR_rms_real"] = float(np.sqrt(np.mean(tr_r[:, :, 1:][m] ** 2)))
        # column-integrated (sum over layers) heating, a robust scalar
        for nm, a, b in (("SRHRcos_colsum", hs_o.sum(2), hs_r.sum(2)), ("TRHR_colsum", tr_o[:, :, 1:].sum(2), tr_r[:, :, 1:].sum(2))):
            rr[nm + "_gmean_diff"] = _wmean(a - b, axyp)
            rr[nm + "_rms_diff"] = _rms2(a - b)
            rr[nm + "_rms_real"] = _rms2(b)
        # TOA (AIJ delta columns of the server; the real is the dump-mode rounded delta): SRNFP0, TRNFP0, net
        ac = o["AIJ_COLS"]
        names = list(AIJ_COLS)
        for i, n in enumerate(names):
            a, b = ac[:, :, i], lv["AIJD"][:, :, AIJ_COLS[n] - 1]
            rr[f"{n}_gmean_real"], rr[f"{n}_gmean_ours"] = _wmean(b, axyp), _wmean(a, axyp)
            rr[f"{n}_gmean_diff"] = rr[f"{n}_gmean_ours"] - rr[f"{n}_gmean_real"]
            rr[f"{n}_rms_diff"] = _rms2(a - b)
            rr[f"{n}_rms_real"] = _rms2(b)
        net_a = ac[:, :, names.index("SRNFP0")] + ac[:, :, names.index("TRNFP0")]
        net_b = lv["AIJD"][:, :, AIJ_COLS["SRNFP0"] - 1] + lv["AIJD"][:, :, AIJ_COLS["TRNFP0"] - 1]
        rr["TOA_net_gmean_real"], rr["TOA_net_gmean_ours"] = _wmean(net_b, axyp), _wmean(net_a, axyp)
        rr["TOA_net_gmean_diff"] = rr["TOA_net_gmean_ours"] - rr["TOA_net_gmean_real"]
        rr["TOA_net_rms_diff"] = _rms2(net_a - net_b)
        # surface exports
        for k in ("FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS", "DIRNIR", "DIFNIR", "SRDN", "CFRAC"):
            a, b = o[k], lv[k]
            rr[k + "_gmean_diff"] = _wmean(a - b, axyp)
            rr[k + "_rms_diff"] = _rms2(a - b)
            rr[k + "_rms_real"] = _rms2(b)
        for t in range(4):
            a, b = o["FSF"][t], lv["FSF"][t]
            rr[f"FSF{t + 1}_rms_diff"], rr[f"FSF{t + 1}_rms_real"] = _rms2(a - b), _rms2(b)
            a, b = o["TRSURF"][t], lv["TRSURF"][t]
            rr[f"TRSURF{t + 1}_rms_diff"], rr[f"TRSURF{t + 1}_rms_real"] = _rms2(a - b), _rms2(b)
        a, b = o["ALB"][:, :, 0], lv["ALB"][:, :, 0]
        rr["ALB1_rms_diff"], rr["ALB1_gmean_diff"] = _rms2(a - b), _wmean(a - b, axyp)
        rows.append(rr)
    return rows


def report(tag, ff=FF, daydir=DAYDIR, it0=IT0, nsteps=54, open_tag="imf_np", members="p1,p2,p3,p4,p5", write=True):
    members = members.split(",")
    axyp = gio.load_g("nov26", ff)["axyp"]
    ours_dir = f"{ff}/{daydir}/ours_{tag}"
    open_dir = f"{ff}/{daydir}/ours_{open_tag}"
    res = RP.curves(ff, daydir, ours_dir, members, it0, nsteps, axyp)
    res_open = RP.curves(ff, daydir, open_dir, members, it0, nsteps, axyp)
    rows, first = RP.overlay(res, members)
    rows_o, first_o = RP.overlay(res_open, members)
    cc, cc_o = RP.class_counts(res, members, nsteps), RP.class_counts(res_open, members, nsteps)
    cross = RP.cross_curve(open_dir, ours_dir, ff, daydir, it0, nsteps, axyp)          # (free - open) per step
    gt = RP.growth_table(res, members)
    gt_o = RP.growth_table(res_open, members)
    fl = flux_comparison(ff, daydir, ours_dir, it0, nsteps, axyp)
    out = dict(res=res, rows=rows, first=first, class_counts=cc, class_counts_open=cc_o, first_open=first_o, cross=cross, growth=gt, growth_open=gt_o,
               flux=fl, aij_check=verify_aij_columns(ff, daydir, axyp))
    if write:
        json.dump(RP.to_jsonable(out), open(f"{ours_dir}/report.json", "w"))
        open(f"{ours_dir}/report.md", "w").write(markdown_report(out, res, res_open, rows, rows_o, members, tag, open_tag))
    return out, res, res_open


def markdown_report(out, res, res_open, rows, rows_o, members, tag, open_tag):
    L = []
    L.append(f"### free-radiation run `{tag}` vs open-loop `{open_tag}`: rms(ours - real) per step, floor = [min..max] over members of rms(member - control)\n")
    L.append("| step | T free / open [floor] | Q free / open [floor] | U free / open [floor] | V free / open [floor] | P free / open [floor] |")
    L.append("|---|---|---|---|---|---|")
    for k in range(res["nsteps"]):
        cells = []
        for f in ("T", "Q", "U", "V", "P"):
            a, b = rows[k].get(f), rows_o[k].get(f)
            if a is None:
                cells.append("-")
            else:
                cells.append(f"{a['ours']:.2e} {a['cls']} / {b['ours']:.2e} {b['cls']} [{a['lo']:.1e}..{a['hi']:.1e}]")
        L.append(f"| {k}{'*' if k % 5 == 0 else ''} | " + " | ".join(cells) + " |")
    L.append("\n(* = radiation step)\n")
    L.append("class counts free (rms): " + json.dumps({k: v["counts"] for k, v in out["class_counts"].items() if k.endswith(".rms")}))
    L.append("class counts open (rms): " + json.dumps({k: v["counts"] for k, v in out["class_counts_open"].items() if k.endswith(".rms")}))
    L.append("max ratio ours/max-member (steps>=3) free: " + json.dumps({k: [round(v["max_ratio"], 2), v["at_step"]] for k, v in out["class_counts"].items() if k.endswith(".rms")}))
    L.append("max ratio ours/max-member (steps>=3) open: " + json.dumps({k: [round(v["max_ratio"], 2), v["at_step"]] for k, v in out["class_counts_open"].items() if k.endswith(".rms")}))
    L.append("growth (e-fold/step from step 3) free: " + json.dumps({f: {k: round(v, 4) for k, v in r.items()} for f, r in out["growth"].items()}))
    L.append("growth open: " + json.dumps({f: {k: round(v, 4) for k, v in r.items()} for f, r in out["growth_open"].items()}))
    L.append("\n### mass-weighted global-mean difference to the real run (free / open), selected steps\n")
    L.append("| step | dT free | dT open | dQ free | dQ open |")
    L.append("|---|---|---|---|---|")
    for k in (0, 1, 2, 5, 11, 23, 35, 47, min(53, res["nsteps"] - 1)):
        a, b = res["ours"][k], res_open["ours"][k]
        L.append(f"| {k} | {a['T']['gmean']:+.2e} | {b['T']['gmean']:+.2e} | {a['Q']['gmean']:+.2e} | {b['Q']['gmean']:+.2e} |")
    L.append("\n### free minus open-loop run (rms), same dynamics/rounding path\n")
    L.append("| step | T | Q | U | V | P |")
    L.append("|---|---|---|---|---|---|")
    for k in (0, 1, 2, 5, 6, 11, 23, 35, 47, min(53, res["nsteps"] - 1)):
        c = out["cross"][k]
        L.append(f"| {k} | " + " | ".join(f"{c[f]['rms']:.2e}" for f in ("T", "Q", "U", "V", "P")) + " |")
    L.append("\n### radiative fluxes at the radiation steps: free-running (server) minus real run\n")
    keys = ["SWnet_sfc_gmean_diff", "SWnet_sfc_rms_diff", "LWdn_sfc_gmean_diff", "LWdn_sfc_rms_diff", "surf_abs_gmean_real", "TOA_net_gmean_real", "TOA_net_gmean_diff", "TOA_net_rms_diff",
            "SRNFP0_gmean_diff", "TRNFP0_gmean_diff", "SRHRcos_rms_diff", "SRHRcos_rms_real", "TRHR_rms_diff", "TRHR_rms_real", "SRDN_rms_diff", "SRDN_rms_real"]
    L.append("| k | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for r in out["flux"]:
        L.append(f"| {r['k']} | " + " | ".join(f"{r[x]:.3e}" for x in keys) + " |")
    return "\n".join(L) + "\n"


def snoage_sensitivity(it, tag="free_np", ff=FF, daydir=DAYDIR, runs_dir=SCRATCH_RUNS):
    """Side experiment (not part of the run): the saved packet of radiation step `it` with SNOAGE replaced by the live (real) value, one server call.
    Returns the max |difference| and the global-mean/rms differences of the main outputs versus the saved free-run server output."""
    axyp = gio.load_g("nov26", ff)["axyp"]
    od = f"{ff}/{daydir}/ours_{tag}"
    z = np.load(f"{od}/rad_in_{it}.npz")
    st = {k: z[k] for k in z.files}
    live = load_live(it, "in", ff, daydir)
    st["SNOAGE"] = np.array(live["SNOAGE"], float)
    out = run_radiation_day(st, it, rundir=os.path.join(runs_dir, f"run_snoage_{it}"))
    shutil.rmtree(os.path.join(runs_dir, f"run_snoage_{it}"), ignore_errors=True)
    base = np.load(f"{od}/rad_out_{it}.npz")
    res = dict(itime=it, snoage_maxdiff=float(np.abs(st["SNOAGE"] - z["SNOAGE"]).max()), snoage_ndiff_cells=int((st["SNOAGE"] != z["SNOAGE"]).sum()))
    for k in ("SRDN", "FSRDIR", "DIRVIS", "DIRNIR", "CFRAC"):
        res[k + "_gmean_diff"], res[k + "_rms_diff"] = _wmean(out[k] - base[k], axyp), _rms2(out[k] - base[k])
    sw = lambda o: np.transpose(o["SRHR"], (1, 2, 0))[:, :, 0] * o["COSZ1"]
    res["SWnet_sfc_gmean_diff"], res["SWnet_sfc_rms_diff"] = _wmean(sw(out) - sw(base), axyp), _rms2(sw(out) - sw(base))
    hs = lambda o: np.transpose(o["SRHR"], (1, 2, 0))[:, :, 1:] * o["COSZ1"][:, :, None]
    m = RP.valid_mask("T")[:, :, None] & np.ones((1, 1, 40), bool)
    res["SRHRcos_rms_diff"] = float(np.sqrt(np.mean(((hs(out) - hs(base))[m]) ** 2)))
    res["SRHRcos_rms_base"] = float(np.sqrt(np.mean(hs(base)[m] ** 2)))
    toa = lambda a: a[:, :, AIJ_COLS["SRNFP0"] - 1] + a[:, :, AIJ_COLS["TRNFP0"] - 1]
    res["TOAnet_gmean_diff"] = _wmean(toa(out["AIJ"]) - base["AIJ_COLS"][:, :, 0] - base["AIJ_COLS"][:, :, 1], axyp)
    res["TOAnet_rms_diff"] = _rms2(toa(out["AIJ"]) - base["AIJ_COLS"][:, :, 0] - base["AIJ_COLS"][:, :, 1])
    return res


def extra_summary(tag="free_np", open_tag="imf_np", ff=FF, daydir=DAYDIR, it0=IT0, nsteps=54, members=("p1", "p2", "p3", "p4", "p5")):
    """numbers quoted in the ledger: ratios to the member envelope at selected steps, global-mean drift statistics, cloud-field rms at selected steps."""
    axyp = gio.load_g("nov26", ff)["axyp"]
    res = RP.curves(ff, daydir, f"{ff}/{daydir}/ours_{tag}", list(members), it0, nsteps, axyp)
    reso = RP.curves(ff, daydir, f"{ff}/{daydir}/ours_{open_tag}", list(members), it0, nsteps, axyp)
    out = dict(ratio={}, drift={}, clouds={})
    for f in ("T", "Q", "U", "V", "P", "QCL", "QCI"):
        out["ratio"][f] = {}
        for k in (5, 11, 23, 35, 47, 53):
            lo, hi = RP.envelope(res, list(members), f, "rms", k)
            out["ratio"][f][k] = dict(free_over_min=res["ours"][k][f]["rms"] / lo, free_over_max=res["ours"][k][f]["rms"] / hi,
                                      open_over_min=reso["ours"][k][f]["rms"] / lo, open_over_max=reso["ours"][k][f]["rms"] / hi)
    for f in ("T", "Q"):
        for nm, rr in (("free", res), ("open", reso)):
            g = np.array([rr["ours"][k][f]["gmean"] for k in range(10, nsteps)])
            out["drift"][f"{f}_{nm}"] = dict(mean=float(g.mean()), n_negative=int((g < 0).sum()), n=int(g.size), abs_mean_of_abs=float(np.abs(g).mean()))
        mm = {}
        for m in members:
            g = np.array([res["mem"][m][k][f]["gmean"] for k in range(10, nsteps)])
            mm[m] = dict(mean=float(g.mean()), n_negative=int((g < 0).sum()))
        out["drift"][f + "_members"] = mm
    for nm, tg in (("free", tag), ("open", open_tag)):
        run = json.load(open(f"{ff}/{daydir}/ours_{tg}/run.json"))
        out["clouds"][nm] = {k: {c: run["rows"][k]["clouds"][c]["rms"] for c in ("CLDSS", "CLDMC", "TAUSS", "W_CLOUD", "QLSS", "QISS", "PREC") if c in run["rows"][k]["clouds"]}
                             for k in (5, 23, 47, 53)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "check0", "aijcheck", "extra", "snoage"))
    ap.add_argument("--it", type=int, default=33362)
    ap.add_argument("--tag", default="free_np")
    ap.add_argument("--nsteps", type=int, default=54)
    ap.add_argument("--open-tag", default="imf_np")
    ap.add_argument("--no-imf", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_day_free(nsteps=a.nsteps, imf=not a.no_imf, tag=a.tag)
    elif a.cmd == "check0":
        check_step0()
    elif a.cmd == "extra":
        print(json.dumps(RP.to_jsonable(extra_summary(a.tag, a.open_tag)), indent=1))
    elif a.cmd == "snoage":
        print(json.dumps(snoage_sensitivity(a.it, a.tag), indent=1))
    elif a.cmd == "aijcheck":
        axyp = gio.load_g("nov26", FF)["axyp"]
        print(json.dumps(verify_aij_columns(FF, DAYDIR, axyp), indent=1))
    else:
        out, res, res_o = report(a.tag, nsteps=a.nsteps, open_tag=a.open_tag)
        print(open(f"{FF}/{DAYDIR}/ours_{a.tag}/report.md").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
