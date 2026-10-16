"""D162: the free-radiation day (atm_day_free_rad.py, D155-D157) with the radiation stage served by the PERSISTENT radiation server
(radiation_server_persist.PersistentServer, D159-D161) instead of the one-shot server.

Nothing else changes: atm_day_free_rad.run_day_free is reused as is; only its `server` hook is replaced by PersistentRadiation, a callable
with the signature server(state, itime, rundir=None) -> output dict.  The random seed of RADIA at each radiation step is read from the real
model's record (ffc_cse_out_<it>.bin SEEDS[1], nov26_day); this is what the one-shot server gets implicitly because it re-runs the real model up
to that step.  One server process (start-up ~20 s, once); the model clock only moves forward, the radiation steps are requested in order.

Usage (fullfidelity/, conda python; RADSRVP_SCRATCH = directory holding mE2/model/P2SAoM40.bin, see D159):
  python atm_day_free_rad_persist.py run     --tag free_persist [--nsteps 54]
  python atm_day_free_rad_persist.py compare --tag free_persist [--ref free_np]    # bitwise comparison of the day against the one-shot run
"""
import argparse
import json
import os
import sys

import numpy as np

import atm_day_free_rad as FRM
import atm_day_open_loop as OL
import clouds_condse_io as cio
import radiation_server as rs
import radiation_server_persist as P
import atm_day_report as RP


def seed_at(it, ff=FRM.FF, daydir=FRM.DAYDIR):
    return int(round(float(cio.read_cse(f"{ff}/{daydir}/ffc_cse_out_{it}.bin")["SEEDS"][1])))


class PersistentRadiation:
    """Callable server for FreeRad: lazily starts one PersistentServer (nov26), passes the real seed of each step; stop() at the end."""

    def __init__(self, date="nov26", ff=FRM.FF, daydir=FRM.DAYDIR, rundir=None, use_seed=True):
        self.date, self.ff, self.daydir, self.rundir, self.use_seed = date, ff, daydir, rundir, use_seed
        self.server = None
        self.start_wall = None
        self.seeds = {}

    def __call__(self, state, itime, rundir=None):
        if self.server is None:
            self.server = P.PersistentServer(self.date, rundir=self.rundir).start()
            self.start_wall = self.server.start_wall
        seed = seed_at(itime, self.ff, self.daydir) if self.use_seed else None
        self.seeds[itime] = seed
        return self.server.request(state, itime, seed)

    def stop(self):
        if self.server is not None:
            self.server.stop()


def run_day_persist(tag="free_persist", nsteps=54, imf=True, log=print, **kw):
    srv = PersistentRadiation()
    try:
        res = FRM.run_day_free(nsteps=nsteps, imf=imf, tag=tag, server=srv, log=log, **kw)
    finally:
        srv.stop()
    res["persist"] = dict(start_wall=srv.start_wall, seeds=srv.seeds, per_call_server_s=[c[1] for c in srv.server.calls] if srv.server else [],
                          per_call_wall_s=[c[2] for c in srv.server.calls] if srv.server else [])
    out_dir = f"{FRM.FF}/{FRM.DAYDIR}/ours_{tag}"
    json.dump(RP.to_jsonable(res), open(f"{out_dir}/run.json", "w"))
    return res


def compare(tag, ref="free_np", ff=FRM.FF, daydir=FRM.DAYDIR, log=print):
    """Bitwise comparison of the two days: step_<it>.npz (T Q U V QCL QCI P, all 54 steps), rad_in (assembled packets) and rad_out (server outputs)."""
    a, b = f"{ff}/{daydir}/ours_{tag}", f"{ff}/{daydir}/ours_{ref}"
    out = dict(state={}, rad_in={}, rad_out={})
    for it in range(FRM.IT0, FRM.IT0 + 54):
        pa, pb = f"{a}/step_{it}.npz", f"{b}/step_{it}.npz"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        za, zb = np.load(pa), np.load(pb)
        out["state"][it] = {k: dict(equal=bool(np.array_equal(za[k], zb[k])), maxabs=float(np.abs(za[k] - zb[k]).max()),
                                    ndiff=int((za[k] != zb[k]).sum())) for k in za.files}
    for kind in ("rad_in", "rad_out"):
        for it in FRM.rad_steps():
            pa, pb = f"{a}/{kind}_{it}.npz", f"{b}/{kind}_{it}.npz"
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            za, zb = np.load(pa), np.load(pb)
            out[kind][it] = {k: dict(equal=bool(np.array_equal(za[k], zb[k])), maxabs=float(np.abs(za[k].astype(float) - zb[k].astype(float)).max()),
                                     ndiff=int((za[k] != zb[k]).sum())) for k in za.files if k in zb.files and not k.startswith("_")}
    summ = {}
    for sec in ("state", "rad_in", "rad_out"):
        n = sum(len(v) for v in out[sec].values())
        bad = [(it, k, d["maxabs"], d["ndiff"]) for it, v in out[sec].items() for k, d in v.items() if not d["equal"]]
        summ[sec] = dict(n_compared=n, n_unequal=len(bad), unequal=bad[:60])
    out["summary"] = summ
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "compare"))
    ap.add_argument("--tag", default="free_persist")
    ap.add_argument("--ref", default="free_np")
    ap.add_argument("--nsteps", type=int, default=54)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_day_persist(a.tag, a.nsteps)
    else:
        print(json.dumps(compare(a.tag, a.ref)["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
