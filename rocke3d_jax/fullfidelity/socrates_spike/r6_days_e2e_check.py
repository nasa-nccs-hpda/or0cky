"""R6: radia_jax with the day products of daily_products.py against the Fortran server at OTHER calendar dates (r6_days_oracle.py with
R6_SAVE_DIR: the recorded nov26 packet served at later itimes; the server steps its clock through the day boundaries and runs the daily
updates).  One RadiaJax (one compilation); the per-date products replace the traced product arrays.
   JAX_PLATFORMS=cpu python r6_days_e2e_check.py products_step0.npz static0.npz specdir RUNDIR OUTDIR RESULT.json
env R6_STATIC (day0_e.npz).  OUTDIR holds out_<itime>_0.npz."""
import glob, json, os, sys, time
import os as _os, sys as _sys
if _os.environ.get('R6_XLA_FLAGS', '1') == '1':
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    import clouds_jax_env  # noqa: F401  (must precede the jax import)
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import radiation_server as rs
import radia_jax as RJ
import daily_products as DP
import drv_zenith as Z

z = np.load(sys.argv[1])
prod = RJ.Products(z, np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
dp = DP.DailyProducts(sys.argv[4], prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust, solar_constant=prod.solar_constant)
rj = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith())
valid = np.zeros((72, 46), bool)
valid[rj.ii - 1, rj.jj - 1] = True
mv = np.nonzero(valid)
pkt = rs.read_packet("%s/nov26_day/rsv_n26_33312_in.bin" % rs.FF)
FIELDS = ("T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB", "FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS", "DIRNIR",
          "DIFNIR", "SRDN", "CFRAC")
AX = {"T": (0, 1), "Q": (0, 1), "KLIQ": (2, 3), "ALB": (0, 1), "RQT": (1, 2)}
res = {}
files = sorted(glob.glob(os.path.join(sys.argv[5], "out_*_0.npz")), key=lambda f: int(os.path.basename(f).split("_")[1]))
for f in files:
    it = int(os.path.basename(f).split("_")[1])
    ref = dict(np.load(f))
    jd = DP.jday_of_itime(it)
    dd = DP.day_dict(dp, jd, prod.ilon, prod.jlat, prod.g["fulgas"])
    rj.prod_arrays.update({k: jnp.asarray(v) for k, v in dd.items()})
    t0 = time.time()
    out = rj(pkt, it)
    row = {}
    for k in FIELDS:
        m, r = out[k], ref[k]
        ax = AX.get(k, (0, 1) if r.ndim == 2 else (1, 2))
        idx = [slice(None)] * r.ndim
        idx[ax[0]], idx[ax[1]] = mv
        mm, rr = m[tuple(idx)], r[tuple(idx)]
        row[k] = dict(maxabs=float(np.abs(mm - rr).max()), bitwise_frac=float((mm == rr).mean()))
    res[str(it)] = dict(jday=jd, fields=row)
    print("itime %d jday %3d (%.1f s)  " % (it, jd, time.time() - t0),
          " ".join("%s=%.1e" % (k, row[k]["maxabs"]) for k in ("SRHR", "TRHR", "T", "RQT", "FSF", "SRDN", "TRSURF", "ALB", "KLIQ")), flush=True)
    json.dump(res, open(sys.argv[6], "w"), indent=1)
