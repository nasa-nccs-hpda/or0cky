"""D154: bitwise oracle and input audit for the radiation server (radiation_server.py).

Oracle (per radiation step 33312 = step 0 and 33317 = step 5 of the nov26 window):
  inputs : atmosphere-side RADIA inputs taken from the EXISTING dumps (ffc_cse_out/in: T Q, PK PMID PDSIG PEDN, the 15
           cloud hand-off arrays, TAUSS/TAUMC/CLDSS/CLDMC, SNOAGE, RSI/FEARTH/FLAND/FLICE/FLAKE/TSAVG; ffa_step_r: MA);
           the remaining RADIA inputs (RQT, KLIQ, per-tile GTEMPR, ZSI, snow fields, ...) are NOT in any existing dump
           and come from the dump-mode packet (rsv_n26_<it>_in.bin, the live model state recorded by the new hook).
  server : real RADIA called inside the real model (restart nov26, run to the radiation step, packet overwrites inputs).
  truth  : ffa_step_<it>_r (T Q SRHR TRHR COSZ1), the dump-mode output packet (all other outputs, same trajectory),
           next-step ffc_cse_in (CLDSS CLDMC after the RADIA mask), tile dumps ffs (SRHEAT vs FSF*COSZ1).
Usage: python radiation_server_compare.py [oracle|audit|all] [--steps 33312,33317]
"""
import json
import os
import sys
import time

import numpy as np

import radiation_server as rs
from clouds_condse_io import read_cse

FFN = rs.FF + "/nov26"
DUMPRUN = os.path.join(rs.SCRATCH, "run_dump6")


def ensure_dump(nstep=6):
    if not os.path.exists(os.path.join(DUMPRUN, "rsv_n26_33312_in.bin")):
        rs.dump_state(DUMPRUN, 33312, nstep=nstep)
    return DUMPRUN


def oracle_state(it):
    """-> (state dict for the server, dict name -> 'existing dump'/'dump-mode', live inputs packet)"""
    live = rs.read_packet(os.path.join(DUMPRUN, "rsv_n26_%d_in.bin" % it))
    cin = read_cse(FFN + "/ffc_cse_in_%d.bin" % it)
    cout = read_cse(FFN + "/ffc_cse_out_%d.bin" % it)
    r = read_cse(FFN + "/ffa_step_%d_r.bin" % it)
    st, src = {}, {}
    from_out = ["T", "Q", "W_CLOUD", "FRAC_ST_WATER", "FRAC_ST_ICE", "FRAC_CNV_WATER", "FRAC_CNV_ICE", "MIX_ST_WATER",
                "MIX_ST_ICE", "MIX_CNV_WATER", "MIX_CNV_ICE", "DIM_ST_WATER", "DIM_ST_ICE", "DIM_CNV_WATER",
                "DIM_CNV_ICE", "FRAC_AREA_ST", "FRAC_AREA_CNV", "TAUSS", "TAUMC", "CLDSS", "CLDMC", "SNOAGE"]
    from_in = ["PK", "PMID", "PDSIG", "PEDN", "RSI", "FEARTH", "FLAND", "FLICE", "FLAKE", "TSAVG"]
    for k in rs.INPUT_FIELDS:
        if k in from_out:
            st[k], src[k] = cout[k], "existing ffc_cse_out"
        elif k in from_in:
            st[k], src[k] = cin[k], "existing ffc_cse_in"
        elif k == "MA":
            st[k], src[k] = r["MA"], "existing ffa_step_r"
        else:
            st[k], src[k] = live[k], "dump-mode packet (new)"
    return st, src, live


def cmp_arr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return dict(equal=False, note="shape %s vs %s" % (a.shape, b.shape))
    d = np.abs(a - b)
    return dict(equal=bool(np.array_equal(a, b)), maxabs=float(d.max()) if d.size else 0.0,
                ndiff=int((a != b).sum()), n=int(a.size), maxval=float(np.abs(b).max()))


def oracle(step, verbose=True):
    ensure_dump()
    st, src, live = oracle_state(step)
    res = {"step": step, "input_sources": {}, "input_vs_live": {}, "outputs": {}}
    for k in st:
        res["input_sources"][k] = src[k]
        if src[k].startswith("existing"):
            res["input_vs_live"][k] = cmp_arr(st[k], live[k])
    out = rs.run_radiation(st, step, rundir=os.path.join(rs.SCRATCH, "run_oracle_%d" % step))
    res["wall_s"] = out["_wall_s"]
    live_out = rs.read_packet(os.path.join(DUMPRUN, "rsv_n26_%d_out.bin" % step))
    r = read_cse(FFN + "/ffa_step_%d_r.bin" % step)
    for k in ("T", "Q", "SRHR", "TRHR", "COSZ1"):
        res["outputs"][k + " vs recorded ffa_step_r"] = cmp_arr(out[k], r[k])
    for k in rs.OUTPUT_FIELDS:
        if k == "AIJ":
            continue
        res["outputs"][k + " vs dump-mode (same run)"] = cmp_arr(out[k], live_out[k])
    # AIJ: server delta is exact (zeroed AIJ); dump-mode delta is AIJ_after - AIJ_before (rounded)
    a, b = out["AIJ"], live_out["AIJD"]
    scale = np.maximum(np.abs(a), 1.0)
    res["outputs"]["AIJ (exact delta) vs dump-mode AIJD (rounded delta)"] = dict(
        equal=bool(np.array_equal(a, b)), maxabs=float(np.abs(a - b).max()),
        max_rel_scaled=float((np.abs(a - b) / scale).max()), ndiff=int((a != b).sum()), n=int(a.size))
    # post-RADIA CLDSS/CLDMC recorded one step later at CONDSE entry
    nxt = FFN + "/ffc_cse_in_%d.bin" % (step + 1)
    if os.path.exists(nxt):
        c = read_cse(nxt)
        for k in ("CLDSS", "CLDMC", "SNOAGE"):
            res["outputs"][k + " vs recorded next-step ffc_cse_in"] = cmp_arr(out[k], c[k])
    # tile dumps: SRHEAT of ocean tiles (ffs col 16) vs FSF(1)*COSZ1
    p = FFN + "/ffs_%d.bin" % step
    if os.path.exists(p):
        rec = np.fromfile(p, ">f8").reshape(-1, 90)
        oc = rec[rec[:, 2] == 1]
        i, j = oc[:, 0].astype(int) - 1, oc[:, 1].astype(int) - 1
        exp = out["FSF"][0, i, j] * out["COSZ1"][i, j]
        res["outputs"]["FSF(1)*COSZ1 vs ffs SRHEAT (ocean tiles, %d recs)" % len(oc)] = cmp_arr(exp, oc[:, 15])
    res["_out"] = out
    if verbose:
        print("=== oracle step %d: server wall %.1f s" % (step, res["wall_s"]))
        bad = [k for k, v in res["input_vs_live"].items() if not v["equal"]]
        print("existing-dump inputs equal to live model state: %d of %d (differ: %s)" %
              (len(res["input_vs_live"]) - len(bad), len(res["input_vs_live"]), bad))
        for k, v in res["outputs"].items():
            print("  %-62s %s" % (k, ("BITWISE EQUAL" if v["equal"] else "DIFF " + json.dumps(v))))
    return res


def perturb(name, a):
    a = np.array(a, float)
    if name == "T":
        return a + 1e-6
    if name == "LTROPO":
        return a + 1
    if name == "FLAG_DSWS":
        return 1.0 - a
    if name == "KLIQ":
        return 1.0 - a
    if not a.any():
        return a + 1e-6
    return a * (1.0 + 1e-6)


def perturb_big(name, a):
    """Second-stage audit for fields that did not move at 1e-6 relative: a macroscopic change."""
    a = np.array(a, float)
    if name in ("TAUSS", "TAUMC"):
        return np.zeros_like(a)          # below taulim -> the corresponding CLDSS/CLDMC are masked to 0
    if name == "SNOWI":
        return a * 2.0 + 10.0
    if name == "DLAKE":
        return np.full_like(a, 0.1)
    if name == "FLAKE":
        return np.clip(a * 1.5 + 0.1, 0.0, 1.0)
    if name == "TSAVG":
        return a + 50.0
    return a * 1.5


def audit_big(fields, step=33312, base=None, max_parallel=10):
    ensure_dump()
    st, src, live = oracle_state(step)
    if base is None:
        base = rs.run_radiation(st, step, rundir=os.path.join(rs.SCRATCH, "run_audit_base"))
    jobs = []
    for k in fields:
        s2 = dict(st)
        s2[k] = perturb_big(k, st[k])
        jobs.append((k, s2, step, os.path.join(rs.SCRATCH, "run_auditbig_%s" % k)))
    outs = rs.run_many(jobs, max_parallel)
    res = {}
    for k in fields:
        moved = {o: float(np.abs(outs[k][o] - base[o]).max()) for o in rs.OUTPUT_FIELDS
                 if np.abs(outs[k][o] - base[o]).max() > 0}
        res[k] = moved
        print("BIG %-10s moves %2d/%d outputs: %s" % (k, len(moved), len(rs.OUTPUT_FIELDS),
                                                    ", ".join("%s(%.2e)" % kv for kv in list(moved.items())[:8])))
    return res


def audit(step=33312, fields=None, base=None, max_parallel=10):
    """Perturb each packet field in turn; report which outputs move."""
    ensure_dump()
    st, src, live = oracle_state(step)
    if base is None:
        base = rs.run_radiation(st, step, rundir=os.path.join(rs.SCRATCH, "run_audit_base"))
    fields = fields or list(rs.INPUT_FIELDS)
    jobs = []
    for k in fields:
        s2 = dict(st)
        s2[k] = perturb(k, st[k])
        jobs.append((k, s2, step, os.path.join(rs.SCRATCH, "run_audit_%s" % k)))
    t0 = time.time()
    outs = rs.run_many(jobs, max_parallel)
    res = {"wall_total_s": time.time() - t0, "fields": {}}
    for k in fields:
        moved = {}
        for o in rs.OUTPUT_FIELDS:
            d = np.abs(outs[k][o] - base[o])
            if d.max() > 0:
                moved[o] = float(d.max())
        res["fields"][k] = moved
        print("%-16s moves %2d/%d outputs: %s" % (k, len(moved), len(rs.OUTPUT_FIELDS),
                                                ", ".join("%s(%.2e)" % kv for kv in list(moved.items())[:8])))
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    what = args[0] if args else "oracle"
    steps = [33312, 33317]
    if "--steps" in args:
        steps = [int(x) for x in args[args.index("--steps") + 1].split(",")]
    if what in ("oracle", "all"):
        for s in steps:
            r = oracle(s)
            r.pop("_out")
            json.dump(r, open(os.path.join(rs.SCRATCH, "oracle_%d.json" % s), "w"), indent=1, default=str)
    if what == "auditbig":
        r = audit_big(["MA", "TAUSS", "TAUMC", "SNOWI", "FLAKE", "DLAKE", "TSAVG"], steps[0])
        json.dump(r, open(os.path.join(rs.SCRATCH, "auditbig_%d.json" % steps[0]), "w"), indent=1)
    if what in ("audit", "all"):
        r = audit(steps[0])
        json.dump(r, open(os.path.join(rs.SCRATCH, "audit_%d.json" % steps[0]), "w"), indent=1)
