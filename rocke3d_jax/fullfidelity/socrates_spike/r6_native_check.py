"""R7: NativeHandoff (radia_native.py) as the replacement of the server callback of jax_radiation: the 21 output fields and the four AIJ
columns of the hand-off against the recorded real-server output, inside one jax.jit.
   python r6_native_check.py products_step0.npz static0.npz specdir ITIMEI ITIME [ITIME ...]"""
import os as _os, sys as _sys
if _os.environ.get('R6_XLA_FLAGS', '1') == '1':
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    import clouds_jax_env  # noqa: F401
import sys, os, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import radiation_server as rs
import radia_jax as RJ
import radia_native as RN
import drv_zenith as Z

z = np.load(sys.argv[1])
prod = RJ.Products(z, np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
rj = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith())
itimei = int(sys.argv[4])
its = [int(x) for x in sys.argv[5:]]
K = (max(its) - itimei) // 5 + 1
zen = Z.Zenith()
tab = RN.zenith_table(zen, itimei, K)
nh = RN.NativeHandoff(rj, tab, itimei)
valid = np.zeros((72, 46), bool); valid[rj.ii - 1, rj.jj - 1] = True
names = ("T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "SNOAGE", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB", "FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS",
         "DIRNIR", "DIFNIR", "SRDN", "CFRAC", "COSZ1")
f = jax.jit(lambda pkt, it: nh.call(pkt, it))
for it in its:
    pkt = rs.read_packet("%s/nov26_day/rsv_n26_%d_in.bin" % (rs.FF, it))
    ref = rs.read_packet("%s/nov26_day/rsv_n26_%d_out.bin" % (rs.FF, it))
    t0 = time.time()
    out, aij = f({k: jnp.asarray(v) for k, v in pkt.items()}, jnp.int32(it))
    jax.block_until_ready(aij)
    print("== itime %d (%.1f s)" % (it, time.time() - t0))
    for k in names:
        m, r = np.asarray(out[k]), ref[k]
        ax = {"T": (0, 1), "Q": (0, 1), "KLIQ": (2, 3), "ALB": (0, 1)}.get(k, (1, 2) if r.ndim == 3 else (0, 1))
        idx = [slice(None)] * r.ndim
        idx[ax[0]], idx[ax[1]] = np.nonzero(valid)
        mm, rr = m[tuple(idx)], r[tuple(idx)]
        d = np.abs(mm - rr)
        print("%-10s %9d %10.3f %12.3e %12.3e" % (k, mm.size, 100.0 * float((mm == rr).mean()), d.max(), (d / np.maximum(np.abs(rr), 1e-300)).max()))
    aijd = ref["AIJD"]
    for n, (nm, col) in enumerate(RN.AIJ_COLS.items()):
        m = np.asarray(aij)[:, :, n]; r = aijd[:, :, col - 1]
        d = np.abs(m - r)[valid]
        print("AIJ %-8s %9d %10.3f %12.3e %12.3e" % (nm, d.size, 100.0 * float((m[valid] == r[valid]).mean()), d.max(),
                                                      (d / np.maximum(np.abs(r[valid]), 1e-300)).max()))
