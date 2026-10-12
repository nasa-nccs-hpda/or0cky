"""segments.py against one long window: the per-segment day products and zenith rows are identical to those of a single long window at every
radiation step, and the whole radiation step (nov26 packet) is bitwise identical at steps in different segments (across midnight).
   JAX_PLATFORMS=cpu python r6_segments_check.py products_step0.npz static0.npz specdir RUNDIR [LAST_ITIME] [MAX_DAYS]
env R6_STATIC (day0_e.npz).  Prints one line per check; exit status 1 if any differs."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: F401
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import radiation_server as rs
import radia_jax as RJ
import radia_native as RN
import daily_products as DP
import segments as SG
import drv_zenith as Z

ITIMEI = 16032
first, last = 33312, int(sys.argv[5]) if len(sys.argv) > 5 else 33312 + 2 * 48 + 20
maxd = int(sys.argv[6]) if len(sys.argv) > 6 else 1
prod = RJ.Products(np.load(sys.argv[1]), np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
dp = DP.DailyProducts(sys.argv[4], prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust, solar_constant=prod.solar_constant)
zen = Z.Zenith()
# long window
dl, d0 = DP.day_products_for_run(dp, prod, first, last)
rj_long = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=zen, day_products=dl, day0=d0)
it0 = SG.first_rad_itime(first, ITIMEI)
tab_long = RN.zenith_table(zen, it0, (last - it0) // 5 + 1)
plan = SG.segment_plan(first, last, maxd)
print("segments:", [(s["day0"], s["ndays"], s["itime_first"], s["itime_last"]) for s in plan], flush=True)
bad = 0
rj_seg = None
rows = {}
for s in plan:
    nh, rj_seg = SG.build_segment(s, dp, prod, sys.argv[3], sys.argv[2], zen, ITIMEI, radia=rj_seg)
    if rj_seg is None:
        raise SystemExit("no radia")
    for it in range(SG.first_rad_itime(s["itime_first"], ITIMEI), s["itime_last"] + 1, 5):
        pa_l, pa_s = rj_long.pa_for_itime(it), rj_seg.pa_for_itime(it)
        same = all(np.array_equal(np.asarray(pa_l[k]), np.asarray(pa_s[k])) for k in pa_l)
        row_l = tab_long[(it - it0) // 5]
        row_s = np.asarray(nh.table[(it - nh.itimei) // 5])
        z_same = np.array_equal(row_l, row_s)
        bad += (not same) + (not z_same)
        rows[it] = (same, z_same)
print("radiation steps checked: %d, products identical: %d, zenith rows identical: %d" % (
    len(rows), sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())), flush=True)
# whole step at one radiation step per segment (the packet is the nov26 state)
pkt = rs.read_packet("%s/nov26_day/rsv_n26_33312_in.bin" % rs.FF)
rj_seg = None
for s in plan[:3]:
    it = SG.first_rad_itime(s["itime_first"], ITIMEI)
    nh, rj_seg = SG.build_segment(s, dp, prod, sys.argv[3], sys.argv[2], zen, ITIMEI, radia=rj_seg)
    a, b = rj_long(pkt, it), rj_seg(pkt, it)
    eq = all(np.array_equal(a[k], b[k]) for k in a)
    bad += not eq
    print("whole step itime %d (day %d): bitwise identical %s" % (it, it // 48, eq), flush=True)
sys.exit(1 if bad else 0)
