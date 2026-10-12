"""R6: radia_jax (RADIA + SOCRATES replaced by the JAX port) against the RECORDED outputs of the real server (nov26_day/rsv_n26_<it>_out.bin),
every field except AIJ, for the radiation steps given.
   JAX_PLATFORMS=cpu python r6_radia_check.py products_step0.npz static0.npz specdir ITIME [ITIME ...]"""
import sys, os, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import radiation_server as rs
import radia_jax as RJ
import drv_zenith as Z

z = np.load(sys.argv[1])
prod = RJ.Products(z)
rj = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith())
valid = np.zeros((72, 46), bool)
valid[rj.ii - 1, rj.jj - 1] = True
for it in [int(x) for x in sys.argv[4:]]:
    pkt = rs.read_packet("%s/nov26_day/rsv_n26_%d_in.bin" % (rs.FF, it))
    ref = rs.read_packet("%s/nov26_day/rsv_n26_%d_out.bin" % (rs.FF, it))
    t0 = time.time()
    out = rj(pkt, it)
    print("== itime %d  (%.1f s)" % (it, time.time() - t0))
    print("%-10s %9s %10s %12s %12s" % ("field", "entries", "bitwise %", "max abs", "max rel"))
    for k in ("T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "SNOAGE", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB", "FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS",
              "DIRNIR", "DIFNIR", "SRDN", "CFRAC", "COSZ1"):
        m, r = out[k], ref[k]
        ax = {"T": (0, 1), "Q": (0, 1), "KLIQ": (2, 3), "ALB": (0, 1)}.get(k)
        if ax is None:
            ax = (1, 2) if r.ndim == 3 else (0, 1)
            if r.ndim == 2:
                ax = (0, 1)
        sel = np.moveaxis(np.broadcast_to(valid, r.shape[ax[0]:ax[0] + 2] if r.ndim == 2 else valid.shape), 0, 0)
        # select the valid columns: build an index
        idx = [slice(None)] * r.ndim
        mv = np.nonzero(valid)
        idx[ax[0]], idx[ax[1]] = mv
        mm, rr = m[tuple(idx)], r[tuple(idx)]
        d = np.abs(mm - rr)
        rel = d / np.maximum(np.abs(rr), 1e-300)
        print("%-10s %9d %10.3f %12.3e %12.3e" % (k, mm.size, 100.0 * float((mm == rr).mean()), d.max(), rel.max()))
