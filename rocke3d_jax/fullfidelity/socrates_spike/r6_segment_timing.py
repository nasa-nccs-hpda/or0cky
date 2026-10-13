"""Timing of a segmented long run on one GPU (report 14.9): per segment of max_days model days, the host build (day products, zenith table), the
compilation of the hand-off step (first call) and the per-radiation-step time over every radiation step of the segment.  The state is the recorded
nov26 packet at every radiation step (timing only: the numbers are not compared with the Fortran here).
  python r6_segment_timing.py products.npz static0.npz specdir out.json
env R6_DAYS (run directory of the rundeck), R6_STATIC (day0_e.npz), R6_FIRST (33312), R6_NSEG (2), R6_MAXDAYS (30)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
import jax_radiation as JR
import jax_radiation_check as CK
import radiation_server as rs
import clouds_condse_io as cio
import dyn_step as ds
import atm_step as A
import radia_jax as RJ
import daily_products as DP
import segments as SG
import drv_zenith as Z

first = int(os.environ.get("R6_FIRST", 33312)); nseg = int(os.environ.get("R6_NSEG", 2)); maxd = int(os.environ.get("R6_MAXDAYS", 30))
last = first + nseg * maxd * 48 - 1
prod = RJ.Products(np.load(sys.argv[1]), np.load(os.environ["R6_STATIC"]))
dp = DP.DailyProducts(os.environ["R6_DAYS"], prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust,
                      solar_constant=prod.solar_constant, hlb0=prod.c60["hlb0"][0])
zen = Z.Zenith()
imaxj = np.full(rs.JM, rs.IM); imaxj[0] = imaxj[-1] = 1
mask = np.zeros((rs.IM, rs.JM), bool)
for j in range(rs.JM):
    mask[:imaxj[j], j] = True
ctx = A.make_ctx("nov26", imf=False)
const = JR.RadConst(1800.0, ctx.sha, mask, ds.ITIMEI)
_, (atm, cloud, carry, surf), _ = CK.device_packet(first)
state = {**atm, **cloud}
cosz = jnp.asarray(cio.read_cse(f"{rs.FF}/nov26_day/ffa_step_{first}_r.bin")["COSZ1"])
res = dict(device=str(jax.devices()[0]), jax=jax.__version__, segments=[])
rj = None
plan = SG.segment_plan(first, last, maxd)
for s in plan:
    t = time.perf_counter()
    nh, rj = SG.build_segment(s, dp, prod, sys.argv[3], sys.argv[2], zen, ds.ITIMEI, radia=rj)
    build = time.perf_counter() - t
    step = JR.make_handoff_step(nh, const)
    hold = JR.empty_hold(carry)
    its = list(range(SG.first_rad_itime(s["itime_first"], ds.ITIMEI), s["itime_last"] + 1, 5))
    walls = []
    for k, it in enumerate(its):
        t = time.perf_counter()
        T_new, Q_new, hold, cz = step(state, surf, hold, jnp.int32(it), jnp.int32(0), cosz)
        jax.block_until_ready(T_new)
        walls.append(time.perf_counter() - t)
        if k == 0:
            print("segment day0 %d: build %.1f s, first call (compile + run) %.1f s" % (s["day0"], build, walls[0]), flush=True)
    w = np.array(walls[1:])
    mem = {}
    try:
        st = jax.devices()[0].memory_stats() or {}
        mem = {k: int(v) for k, v in st.items() if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}
    except Exception:
        pass
    row = dict(day0=s["day0"], ndays=s["ndays"], rad_steps=len(its), build_s=build, first_call_s=walls[0], rad_step_median_s=float(np.median(w)),
               rad_step_mean_s=float(w.mean()), rad_step_max_s=float(w.max()), finite=bool(np.isfinite(np.asarray(T_new)).all()), mem=mem)
    res["segments"].append(row)
    print(json.dumps(row), flush=True)
    json.dump(res, open(sys.argv[4], "w"), indent=1)
