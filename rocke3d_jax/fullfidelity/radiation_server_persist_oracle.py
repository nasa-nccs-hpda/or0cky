"""D160: multi-date oracle of the persistent radiation server.

For a date (nov26, dec01, jan01; restart in ff_data/_pristine_restarts) the real step-0 radiation packets are needed:
  dump_live(date): runs the (existing, unmodified-physics) one-shot binary in RADSRV_MODE=dump for the first step of the window; records
                   the live RADIA input packet (52 fields) and output packet into ff_data/<date>/rsv_n26_<itime>_{in,out}.bin
                   (same recipe as D152/D155; the file is not overwritten when present).
  oracle(date):    persistent server, live input packet + the real seed (ffc_cse_out SEEDS[1]) -> compares
                   - ffa_step_<it>_r (recorded real radiation): T Q SRHR TRHR COSZ1 bitwise
                   - the live output packet (dump mode): the other 16 outputs bitwise, AIJ against the rounded live delta
                   - ffc_cse_in_<it+1>: CLDSS CLDMC SNOAGE (post-RADIA masks = next step's CONDSE input) bitwise
CLI: python radiation_server_persist_oracle.py dump|oracle <date> [...]
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import clouds_condse_io as cio
import radiation_server as rs
import radiation_server_persist as P


def live_paths(date):
    it0 = P.DATES[date][3]          # first radiation step of the window
    d = f"{rs.FF}/{date}"
    return it0, f"{d}/rsv_n26_{it0}_in.bin", f"{d}/rsv_n26_{it0}_out.bin"


def dump_live(date, rundir=None, log=print):
    it0, pin, pout = live_paths(date)
    if os.path.exists(pin) and os.path.exists(pout):
        return it0, pin, pout, None
    s = P.PersistentServer(date, rundir=rundir or os.path.join(P.SCRATCH, "run_dump_%s" % date), binary=rs.BIN)
    s._prepare()
    rd = s.rundir
    for f in ("req.fifo", "rsp.fifo"):
        os.remove(os.path.join(rd, f))
    env = rs._env({"RADSRV_MODE": "dump", "RADSRV_ITIME": str(it0), "RADSRV_NSTEP": "1", "RADSRV_TAG": "n26"})
    t0 = time.time()
    subprocess.run(["./P2SAoM40", "-i", "I"], cwd=rd, env=env, stdout=open(os.path.join(rd, "run.PRT"), "w"), stderr=subprocess.STDOUT)
    wall = time.time() - t0
    for k, dst in (("in", pin), ("out", pout)):
        src = os.path.join(rd, f"rsv_n26_{it0}_{k}.bin")
        if not os.path.exists(src):
            raise RuntimeError("dump run produced no %s packet; log tail:\n%s" % (k, open(os.path.join(rd, "run.PRT"), errors="replace").read()[-1500:]))
        shutil.copy(src, dst)
    log("dump_live %s: %.0f s -> %s" % (date, wall, pin))
    return it0, pin, pout, wall


def seed_of(date, it):
    return int(round(float(cio.read_cse(f"{rs.FF}/{date}/ffc_cse_out_{it}.bin")["SEEDS"][1])))


def oracle(date, server=None, log=print):
    it0, pin, pout = live_paths(date)
    ff = f"{rs.FF}/{date}"
    live_in, live_out = rs.read_packet(pin), rs.read_packet(pout)
    seed = seed_of(date, it0)
    own = server is None
    s = server or P.PersistentServer(date).start()
    try:
        o = s.request(live_in, it0, seed)
        start = s.start_wall
    finally:
        if own:
            s.stop()
    r = cio.read_cse(f"{ff}/ffa_step_{it0}_r.bin")
    nxt = cio.read_cse(f"{ff}/ffc_cse_in_{it0 + 1}.bin")
    res = dict(date=date, itime=it0, seed=seed, start_wall=start, server_s=o["_server_s"], wall=o["_wall_s"], recorded={}, live={}, next_step={})
    for k in ("T", "Q", "SRHR", "TRHR", "COSZ1"):
        a, b = o[k], r[k]
        res["recorded"][k] = dict(equal=bool(np.array_equal(a, b)), maxabs=float(np.abs(a - b).max()), n=int(a.size))
    for k in rs.OUTPUT_FIELDS:
        if k == "AIJ":
            res["live"]["AIJ"] = dict(maxabs_vs_rounded_live_delta=float(np.abs(o["AIJ"] - live_out["AIJD"]).max()))
        else:
            res["live"][k] = bool(np.array_equal(o[k], live_out[k]))
    for k in ("CLDSS", "CLDMC", "SNOAGE"):
        res["next_step"][k] = bool(np.array_equal(o[k], nxt[k]))
    res["all_bitwise"] = bool(all(v["equal"] for v in res["recorded"].values()) and all(v for k, v in res["live"].items() if k != "AIJ")
                              and all(res["next_step"].values()))
    log(json.dumps(res, indent=1))
    return res


def main(argv=None):
    a = argv or sys.argv[1:]
    if a[0] == "dump":
        for d in a[1:]:
            dump_live(d)
    elif a[0] == "oracle":
        for d in a[1:]:
            oracle(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
