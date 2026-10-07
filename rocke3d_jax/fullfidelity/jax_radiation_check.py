"""D185 validation runner for jax_radiation.py (prints/records every number quoted in scoping/D185_RADIATION_HANDOFF_ENTRY.md).

Usage (fullfidelity/, conda python, RADSRVP_SCRATCH = scratch dir holding mE2/model/P2SAoM40.bin, taskset to the allowed cores):
  OMP_NUM_THREADS=1 taskset -c 7 python jax_radiation_check.py [--json out.json] [--no-server]
Starts ONE persistent server (~20 s), makes 6 calls (~11 s each), stops it.
"""
import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import jax
import jax.numpy as jnp

import jax_radiation as JR
import atm_day_free_rad as FRM
import clouds_condse_io as cio
import radiation_server as rs
import radiation_server_persist as P

D = rs.FF + "/nov26_day"
STEPS = (33312, 33317)


def live(it, kind):
    return rs.read_packet(f"{D}/rsv_n26_{it}_{kind}.bin")


def record_state(it):
    """(S, X) as in atm_day_free_rad.check_step0: T,Q from ffc_cse_out, PK.. from ffc_cse_in, MA from ffa_step_r, clouds from ffc_cse_out."""
    cin = cio.read_cse(f"{D}/ffc_cse_in_{it}.bin")
    cout = cio.read_cse(f"{D}/ffc_cse_out_{it}.bin")
    r = cio.read_cse(f"{D}/ffa_step_{it}_r.bin")
    return dict(T=cout["T"], Q=cout["Q"], PK=cin["PK"], PMID=cin["PMID"], PDSIG=cin["PDSIG"], PEDN=cin["PEDN"], MA=r["MA"]), cout


def device_packet(it, carry_from=None):
    """Array-path packet from the recorded state of step `it` (device arrays).  Fields not derivable from the record are taken from the live
    packet and listed (source dict).  carry_from: a server/live OUTPUT dict to take RQT/KLIQ/SNOAGE from (the real carry) instead of the live input."""
    lv = live(it, "in")
    S, X = record_state(it)
    atm = {k: jnp.asarray(S[k]) for k in ("T", "Q", "PK", "PMID", "PDSIG", "PEDN", "MA")}
    atm["LTROPO"] = jnp.asarray(lv["LTROPO"])
    cloud = {k: jnp.asarray(X[k]) for k in JR.CLOUD_IN}
    cs = carry_from if carry_from is not None else dict(SNOAGE=X["SNOAGE"], RQT=lv["RQT"], KLIQ=lv["KLIQ"])
    carry = {k: jnp.asarray(cs[k]) for k in JR.CARRY_IN}
    surf = {k: jnp.asarray(lv[k]) for k in JR.SURF_IN}
    src = {}
    for k in rs.INPUT_FIELDS:
        src[k] = ("recorded state" if k in ("T", "Q", "PK", "PMID", "PDSIG", "PEDN", "MA", "BYMA") or k in JR.CLOUD_IN or (k == "SNOAGE" and carry_from is None)
                  else "server output of previous call" if carry_from is not None and k in JR.CARRY_IN
                  else "copied from live packet")
    return JR.assemble_packet_jax(atm, cloud, carry, surf), (atm, cloud, carry, surf), src


def packet_report():
    res = {}
    for it in STEPS:
        lv = live(it, "in")
        S, X = record_state(it)
        st, _ = FRM.assemble_packet(S, X, X["SNOAGE"], lv["RQT"], lv["KLIQ"], lv)          # the NumPy path
        pk, _, src = device_packet(it)
        pk = {k: np.asarray(v) for k, v in pk.items()}
        rows = {}
        for k in rs.INPUT_FIELDS:
            rows[k] = dict(array_vs_numpy_path=bool(np.array_equal(pk[k], st[k])), array_vs_live=bool(np.array_equal(pk[k], lv[k])),
                           ndiff_live=int((pk[k] != lv[k]).sum()), maxabs_live=float(np.abs(pk[k] - lv[k]).max()), source=src[k])
        res[it] = dict(fields=rows, n_array_eq_numpy=sum(r["array_vs_numpy_path"] for r in rows.values()),
                       n_array_eq_live=sum(r["array_vs_live"] for r in rows.values()),
                       independent_of_live=[k for k, r in rows.items() if r["source"] == "recorded state"],
                       copied_from_live=[k for k, r in rows.items() if r["source"] == "copied from live packet"])
    # real carry: RQT KLIQ SNOAGE of 33317 from the OUTPUT packet of 33312 (what the free run carries) versus the live input of 33317
    o0, l1 = live(33312, "out"), live(33317, "in")
    res["carry_33312out_vs_33317in"] = {k: dict(equal=bool(np.array_equal(o0[k], l1[k])), ndiff=int((o0[k] != l1[k]).sum()),
                                                maxabs=float(np.abs(o0[k] - l1[k]).max())) for k in JR.CARRY_IN}
    return res


def seed_report():
    out = {}
    s0 = JR.seed0_from_record(33312)
    seed0 = jnp.uint32(s0)
    for it in range(33312, 33318):
        rec1 = JR.seed_from_record(it)
        rec0 = JR.seed0_from_record(it)
        ch = int(JR.seed_chain_radia(seed0))
        out[it] = dict(seeds1_recorded=rec1, seeds1_chain_from_33312=ch, equal=ch == rec1, seeds0_recorded=rec0, seeds0_chain=int(seed0),
                       seeds0_equal=int(seed0) == rec0)
        seed0 = JR.seed_chain_next(seed0, (it - 33312) % 5 == 0)
    return out


def cmp_outputs(out, ref):
    """21 non-AIJ fields: bitwise; returns (list of unequal, dict)"""
    bad = {k: dict(ndiff=int((np.asarray(out[k]) != ref[k]).sum()), maxabs=float(np.abs(np.asarray(out[k]) - ref[k]).max()))
           for k in JR.OUT_FIELDS if not np.array_equal(np.asarray(out[k]), ref[k])}
    return bad


def server_report(log=print):
    res = {}
    with P.PersistentServer("nov26") as srv:
        res["start_wall_s"] = srv.start_wall
        hand = JR.RadiationHandoff(srv)
        hand.log.server_binary, hand.log.restart = srv.binary, srv.restart
        pid = srv.proc.pid

        @jax.jit
        def inside(packet, itime, seed):
            return hand.call(packet, itime, seed)

        calls = []

        def run_inside(it, pk, seed=None, tag=""):
            seed = JR.seed_from_record(it) if seed is None else seed
            n0 = len(hand.log.calls)
            t0 = time.perf_counter()
            out, aijc = inside(pk, jnp.int32(it), jnp.int32(seed))
            jax.block_until_ready((out, aijc))
            wall = time.perf_counter() - t0
            rec = hand.log.calls[n0]
            ref = live(it, "out")
            bad = cmp_outputs(out, ref)
            cols = [c - 1 for c in JR.AIJ_COLS.values()]
            d_aij = float(np.abs(np.asarray(aijc) - ref["AIJD"][:, :, cols]).max())
            ent = dict(itime=it, tag=tag, seed=seed, unequal_vs_recorded_out=bad, n_equal_of_21=21 - len(bad), aij_cols_maxabs_vs_AIJD=d_aij,
                       jit_call_wall_s=wall, callback_s=rec["callback_s"], request_s=rec["request_s"], server_s=rec["server_s"],
                       bytes_d2h=rec["bytes_d2h"], bytes_h2d=rec["bytes_h2d"], arrays_d2h=rec["arrays_d2h"], arrays_h2d=rec["arrays_h2d"])
            calls.append(ent)
            log(" call", tag, it, "equal %d/21" % ent["n_equal_of_21"], "AIJcols maxabs %.2e" % d_aij, "wall %.2f s (server %.2f s)" % (wall, rec["server_s"]))
            return out

        pk12, _, _ = device_packet(33312)
        pk17, _, _ = device_packet(33317)
        run_inside(33312, pk12, tag="in-jit")
        out17 = run_inside(33317, pk17, tag="in-jit")
        run_inside(33312, pk12, tag="in-jit repeat (order independence)")
        # eager (outside jit) reference for the overhead; the same packet, the same seed, the same server
        t0 = time.perf_counter()
        eager = hand.call_eager(pk12, 33312, JR.seed_from_record(33312))
        res["eager_33312"] = dict(wall_s=time.perf_counter() - t0, request_s=hand.log.calls[-1]["request_s"], server_s=hand.log.calls[-1]["server_s"],
                                  callback_s=hand.log.calls[-1]["callback_s"])
        ref_e = live(33312, "out")
        aij_right = np.array(hand.last_out["AIJ"])
        res["eager_33312"]["unequal_vs_recorded_out"] = {k: 1 for i, k in enumerate(JR.OUT_FIELDS) if not np.array_equal(np.asarray(eager[i]), ref_e[k])}
        # wrong seed: AIJ needs the right seed (the 21 other fields must not depend on it)
        n0 = len(hand.log.calls)
        out_ws = hand.call_eager(pk12, 33312, 12345)
        ref = live(33312, "out")
        cols = [c - 1 for c in JR.AIJ_COLS.values()]
        bad_ws = {k: int((np.asarray(out_ws[i]) != ref[k]).sum()) for i, k in enumerate(JR.OUT_FIELDS) if not np.array_equal(np.asarray(out_ws[i]), ref[k])}
        res["wrong_seed_33312"] = dict(seed=12345, unequal_fields_vs_recorded=bad_ws,
                                       aij_cols_maxabs_vs_AIJD=float(np.abs(np.asarray(out_ws[-1]) - ref["AIJD"][:, :, cols]).max()))
        aij_wrong = np.array(hand.last_out["AIJ"])
        d_r = np.abs(aij_right - ref["AIJD"]).max(axis=(0, 1))
        d_w = np.abs(aij_wrong - ref["AIJD"]).max(axis=(0, 1))
        res["wrong_seed_33312"]["aij_all_1660_columns"] = dict(
            right_seed_columns_over_1em9=int((d_r > 1e-9).sum()), right_seed_maxabs=float(d_r.max()),
            wrong_seed_columns_over_1em9=int((d_w > 1e-9).sum()), wrong_seed_maxabs=float(d_w.max()),
            right_vs_wrong_columns_differing=int((np.abs(aij_right - aij_wrong).max(axis=(0, 1)) > 0).sum()))
        res["calls"] = calls
        # ---- toy step: jit A -> callback -> jit B, device-resident state, itime and seed carried on device
        res["toy"] = toy_step(srv, log, use_guard="log")
        res["log_totals"] = hand.log.totals()
        res["sentence"] = hand.log.sentence()
        res["server_pid"] = pid
    time.sleep(1)
    res["server_process_alive_after_stop"] = os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    return res


def toy_step(srv, log=print, sync_after_handoff=False, use_guard=False):
    """Six steps 33312..33317 (radiation at 33312 and 33317).  state = T (and the rest of the packet) resident on the device;
    per step: jit A (stand-in for the dynamics; identity on T), jit hand-off (callback on radiation steps only), jit B (a reduction);
    itime and the seed (D174 chain) are device scalars advanced inside jit.  No host operation in the loop except the dispatch."""
    import dyn_step as ds
    hand = JR.RadiationHandoff(srv, keep_outputs=True)
    hand.log.server_binary, hand.log.restart = srv.binary, srv.restart
    S, X = record_state(33312)
    lv = live(33312, "in")
    _, (atm, cloud, carry, surf), _ = device_packet(33312)
    state = {**atm, **cloud}
    from atm_day_free_rad import FF  # noqa: F401
    mask = np.zeros((rs.IM, rs.JM), bool)
    imaxj = np.full(rs.JM, rs.IM)
    imaxj[0] = imaxj[-1] = 1
    for j in range(rs.JM):
        mask[:imaxj[j], j] = True
    const = JR.RadConst(1800.0, SHA_NOV26(), mask, ds.ITIMEI)
    step = JR.make_handoff_step(hand, const)
    hold = JR.empty_hold(carry)
    cosz = jnp.asarray(cio.read_cse(f"{D}/ffa_step_33312_r.bin")["COSZ1"])

    @jax.jit
    def jit_a(state):                                   # stand-in "dynamics": exact identity on T
        s = dict(state)
        s["T"] = state["T"] * 1.0
        return s

    @jax.jit
    def jit_b(T, itime, seed0, tsum):                   # stand-in "surface/other": reduction, advance the clock and the seed chain
        is_rad = ((itime - const.itimei) % JR.NRAD) == 0
        s0n = JR.seed_chain_next(seed0, is_rad)
        return itime + 1, s0n, tsum + jnp.sum(T), JR.seed_chain_radia(s0n)

    itime, seed0, tsum = jnp.int32(33312), jnp.uint32(JR.seed0_from_record(33312)), jnp.float64(0.0)
    seed = JR.seed_chain_radia(seed0)
    jax.block_until_ready((itime, seed0, tsum, seed, state, hold, surf, cosz))
    # warm-up compile with a throwaway call is NOT done (a call would hit the server): compile time is inside the first step (reported)
    hand.log.reset()
    hand.log.server_binary, hand.log.restart = srv.binary, srv.restart
    per_step = []
    guard_ok = True
    guard_err = None
    t_all = time.perf_counter()
    try:
        with (jax.transfer_guard(use_guard) if use_guard else contextlib.nullcontext()):
            for k in range(6):
                t0 = time.perf_counter()
                state = jit_a(state)
                if sync_after_handoff:
                    jax.block_until_ready(state)
                rad = bool(((33312 + k - const.itimei) % JR.NRAD) == 0)           # host-side knowledge only for the log (not used by the jit)
                T_new, Q_new, hold, cz = step(state, surf, hold, itime, seed, cosz)
                if sync_after_handoff:
                    jax.block_until_ready(T_new)
                state = {**state, "T": T_new, "Q": Q_new}
                itime, seed0, tsum, seed = jit_b(T_new, itime, seed0, tsum)
                if not rad:
                    hand.log.skipped_steps += 1
                per_step.append(dict(k=k, itime=33312 + k, radiation_step=rad, dispatch_s=time.perf_counter() - t0))
            jax.block_until_ready((state, tsum))
    except Exception as e:                                # the guard would raise on an implicit host->device transfer in the loop
        guard_ok, guard_err = False, repr(e)[:300]
    wall = time.perf_counter() - t_all
    tsum_host = float(jax.device_get(tsum))
    calls = hand.log.calls
    ref12 = live(33312, "out")
    res = dict(steps=6, wall_s=wall, n_callbacks=len(calls), itimes_called=[c["itime"] for c in calls], seeds_used=[c["seed"] for c in calls],
               seeds_recorded=[JR.seed_from_record(c["itime"]) for c in calls], transfer_guard_mode=use_guard, loop_raised=not guard_ok, loop_error=guard_err,
               per_call_arrays_d2h=[c["arrays_d2h"] for c in calls], per_call_arrays_h2d=[c["arrays_h2d"] for c in calls],
               per_call_bytes_d2h=[c["bytes_d2h"] for c in calls], per_call_bytes_h2d=[c["bytes_h2d"] for c in calls],
               final_device_get_arrays=1, tsum=tsum_host, per_step=per_step, log=hand.log.totals(), sentence=hand.log.sentence())
    if hand.history:
        o = hand.history[0][2]
        res["first_call_outputs_unequal_vs_recorded"] = [k for k in JR.OUT_FIELDS if not np.array_equal(o[k], ref12[k])]
    return res


def SHA_NOV26():
    import atm_step as A
    ctx = A.make_ctx("nov26", imf=False)
    return ctx.sha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-server", action="store_true")
    a = ap.parse_args(argv)
    res = {}
    t0 = time.time()
    res["packet"] = packet_report()
    for it in STEPS:
        p = res["packet"][it]
        print("[packet] %s step %d: array path == NumPy path %d/52, array path == live packet %d/52; copied from live: %s"
              % (JR.SENTENCE, it, p["n_array_eq_numpy"], p["n_array_eq_live"], p["copied_from_live"]))
    print("[carry] 33312 out -> 33317 in:", {k: v["equal"] for k, v in res["packet"]["carry_33312out_vs_33317in"].items()})
    res["seeds"] = seed_report()
    print("[seeds]", {it: (v["equal"], v["seeds0_equal"]) for it, v in res["seeds"].items()})
    if not a.no_server:
        res["server"] = server_report()
        print("[server]", res["server"]["sentence"])
        print("[toy]", res["server"]["toy"]["sentence"])
    res["wall_total_s"] = time.time() - t0
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
