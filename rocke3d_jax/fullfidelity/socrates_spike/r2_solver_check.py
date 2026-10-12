"""R2: compare solver_triple_hogan.py (JAX/NumPy port) with the REAL Fortran solver result in the dumps.
Inputs per triple_column call: tag 41 (trans, reflect: regions clear/strat/conv), 42 (s_down, s_up after IR weighting / solar source,
source_flux_ground), 44 (n_cloud_top ...), 46 (cloud_overlap(:, 0:n_layer, 1..18) = v11..v33, u11..u33; flux_inc_down; diffuse_albedo);
oracle: tag 43 flux_total.   JAX_PLATFORMS=cpu python r2_solver_check.py ffsoc_dump.bin"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R
from solver_triple_hogan import solver_triple_hogan


def groups(recs):
    cur = None
    for x in recs:
        if x["tag"] == 900:
            continue
        if x["tag"] == 40 and x["k"] == 1:
            cur = {}
        if cur is None:
            continue
        cur[(x["tag"], x["k"])] = x
        cur["iso"] = x["iso"]
        if x["tag"] == 43:
            yield cur
            cur = None


def run(g):
    a = lambda tag, k: g[(tag, k)]["a"]
    nct, nreg, nty, nl = [int(v) for v in a(44, 1).ravel()]
    tr, rf = a(41, 1), a(41, 2)
    sd, su, sg = a(42, 1), a(42, 2), a(42, 3)
    co, fin, alb = a(46, 1), a(46, 2), a(46, 3)
    assert co.shape[1] == nl + 1 and co.shape[2] == 18, co.shape
    v = co[:, :, 0:9].reshape(co.shape[0], nl + 1, 3, 3)
    u = co[:, :, 9:18].reshape(co.shape[0], nl + 1, 3, 3)
    reg = lambda arr, k: arr[:, :, k]
    out = solver_triple_hogan(nct, reg(tr, 0), reg(rf, 0), reg(sd, 0), reg(su, 0), reg(tr, 1), reg(rf, 1), reg(sd, 1), reg(su, 1),
                              reg(tr, 2), reg(rf, 2), reg(sd, 2), reg(su, 2), v, u, fin, sg[:, 0], sg[:, 1], sg[:, 2], alb)
    return np.asarray(out), a(43, 1), nct, nl


if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0, ncol=0, nonfin=0))
    ncall = 0
    for g in groups(recs):
        mine, orc, nct, nl = run(g)
        ncall += 1
        key = ("SW" if g["iso"] == 1 else "LW", "nct=1" if nct == 1 else "nct>1")
        s = st[key]
        ok = np.isfinite(orc) & np.isfinite(mine)
        s["nonfin"] += int((~ok).sum())
        d = np.abs(mine - orc)[ok]
        s["n"] += d.size
        s["bit"] += int((mine[ok] == orc[ok]).sum())
        s["ncol"] += mine.shape[0]
        if d.size:
            s["maxabs"] = max(s["maxabs"], float(d.max()))
            s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(orc[ok]), 1e-300)).max()))
    print("triple_column calls compared: %d" % ncall)
    print("%-3s %-6s %8s %9s %10s %9s %11s %11s" % ("", "", "columns", "entries", "bitwise %", "nonfinite", "max abs", "max rel"))
    for k in sorted(st):
        s = st[k]
        print("%-3s %-6s %8d %9d %10.3f %9d %11.3e %11.3e" % (k[0], k[1], s["ncol"], s["n"], 100.0 * s["bit"] / max(s["n"], 1),
                                                           s["nonfin"], s["maxabs"], s["maxrel"]))
