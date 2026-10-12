"""R6: whole radiation step (RADIA + SOCRATES in JAX, radia_jax.py) on the current device: correctness against the recorded real-server output
and timing.   python radia_bench.py products_step0.npz static0.npz specdir out.json ITIME [ITIME ...]
Timing: first call (compile + run), then the median of 5 calls of the jitted device function with block_until_ready (zenith grids and packet
transfer excluded, reported separately)."""
import sys, os, time, json
import os as _os, sys as _sys
if _os.environ.get('R6_XLA_FLAGS', '1') == '1':      # XLA:CPU without FMA contraction / HLO algebraic rewrites (as the other ports, D139/D145)
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    import clouds_jax_env  # noqa: F401  (must precede the jax import)
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import radiation_server as rs
import radia_jax as RJ
import drv_zenith as Z
import zenith_coszs as ZS

z = np.load(sys.argv[1])
prod = RJ.Products(z, np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
rj = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith())
valid = np.zeros((72, 46), bool)
valid[rj.ii - 1, rj.jj - 1] = True
res = dict(device=str(jax.devices()[0]), jax=jax.__version__, steps=[])
for it in [int(x) for x in sys.argv[5:]]:
    pkt = rs.read_packet("%s/nov26_day/rsv_n26_%d_in.bin" % (rs.FF, it))
    ref = rs.read_packet("%s/nov26_day/rsv_n26_%d_out.bin" % (rs.FF, it))
    t0 = time.time()
    cz2, cza = ZS.calc_zenith_angle(rj.zen, it, RJ.NRAD)
    cz1 = rj.zen.cosz1(it)
    t_zen = time.time() - t0
    t0 = time.time()
    out = rj(pkt, it, cosz=(cz2, cza, cz1), to_numpy=False)
    jax.block_until_ready(out["SRHR"])
    t_first = time.time() - t0
    ts = []
    for _ in range(5):
        t0 = time.time()
        o = rj(pkt, it, cosz=(cz2, cza, cz1), to_numpy=False)
        jax.block_until_ready(o["SRHR"])
        ts.append(time.time() - t0)
    st = dict(itime=it, zenith_host_s=t_zen, first_call_s=t_first, steady_median_s=float(np.median(ts)), steady_all_s=ts, fields={})
    for k in ("T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "SNOAGE", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB", "FSRDIR", "SRVISSURF", "FSRDIF",
              "DIRVIS", "DIRNIR", "DIFNIR", "SRDN", "CFRAC", "COSZ1"):
        m = np.asarray(out[k]); r = ref[k]
        ax = {"T": (0, 1), "Q": (0, 1), "KLIQ": (2, 3), "ALB": (0, 1)}.get(k, (1, 2) if r.ndim == 3 else (0, 1))
        idx = [slice(None)] * r.ndim
        idx[ax[0]], idx[ax[1]] = np.nonzero(valid)
        mm, rr = m[tuple(idx)], r[tuple(idx)]
        d = np.abs(mm - rr)
        st["fields"][k] = dict(n=int(mm.size), bitwise_pct=100.0 * float((mm == rr).mean()), max_abs=float(d.max()),
                               max_rel=float((d / np.maximum(np.abs(rr), 1e-300)).max()))
    res["steps"].append(st)
    print("itime %d: zenith host %.2f s, first call %.1f s, steady median %.3f s; worst max abs %s" % (
        it, t_zen, t_first, st["steady_median_s"], max((v["max_abs"], k) for k, v in st["fields"].items())), flush=True)
json.dump(res, open(sys.argv[4], "w"), indent=1)
