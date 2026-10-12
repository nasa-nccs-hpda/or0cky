"""R6: radia_jax with the day products of daily_products.py (no dump of the day) against the RECORDED real-server outputs of radiation steps of a
later day (nov26_day/rsv_n26_<it>_out.bin; the step-0 dump supplies only the static content).   Compares with the products of nov26 ('stale')
and with those of the day ('fresh').
   JAX_PLATFORMS=cpu python r6_days_radia_check.py products_step0.npz static0.npz specdir RUNDIR ITIME [ITIME ...]"""
import sys, os, time
import os as _os, sys as _sys
if _os.environ.get('R6_XLA_FLAGS', '1') == '1':
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    import clouds_jax_env  # noqa: F401  (must precede the jax import)
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import radiation_server as rs
import radia_jax as RJ
import daily_products as DP
import drv_zenith as Z

z = np.load(sys.argv[1])
prod = RJ.Products(z, np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
dp = DP.DailyProducts(sys.argv[4], prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust, solar_constant=prod.solar_constant)
its = [int(x) for x in sys.argv[5:]]
days = sorted({DP.jday_of_itime(it) for it in its})
day0 = min(it for it in its) // 48
nd = max(it for it in its) // 48 - day0 + 1
dd = [DP.day_dict(dp, DP.jday_of_itime((day0 + k) * 48), prod.ilon, prod.jlat, prod.g["fulgas"]) for k in range(nd)]
rj_fresh = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith(), day_products=dd, day0=day0)
rj_stale = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith())
valid = np.zeros((72, 46), bool)
valid[rj_fresh.ii - 1, rj_fresh.jj - 1] = True
mv = np.nonzero(valid)
FIELDS = ("T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB", "FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS", "DIRNIR",
          "DIFNIR", "SRDN", "CFRAC")
AX = {"T": (0, 1), "Q": (0, 1), "KLIQ": (2, 3), "ALB": (0, 1), "RQT": (1, 2)}
res = {}
for it in its:
    pkt = rs.read_packet("%s/nov26_day/rsv_n26_%d_in.bin" % (rs.FF, it))
    ref = rs.read_packet("%s/nov26_day/rsv_n26_%d_out.bin" % (rs.FF, it))
    for label, rj in (("stale", rj_stale), ("fresh", rj_fresh)):
        t0 = time.time()
        out = rj(pkt, it)
        row = {}
        for k in FIELDS:
            m, r = out[k], ref[k]
            ax = AX.get(k, (0, 1) if r.ndim == 2 else (1, 2))
            idx = [slice(None)] * r.ndim
            idx[ax[0]], idx[ax[1]] = mv
            mm, rr = m[tuple(idx)], r[tuple(idx)]
            d = np.abs(mm - rr)
            row[k] = dict(maxabs=float(d.max()), bitwise_frac=float((mm == rr).mean()))
        res["%d_%s" % (it, label)] = row
        print("itime %d jday %d %-5s (%.1f s)  " % (it, DP.jday_of_itime(it), label, time.time() - t0),
              " ".join("%s=%.1e" % (k, row[k]["maxabs"]) for k in ("SRHR", "TRHR", "T", "RQT", "FSF", "SRDN", "TRSURF", "ALB")), flush=True)
import json
out_json = os.environ.get("R6_OUT")
if out_json:
    json.dump(res, open(out_json, "w"), indent=1)
