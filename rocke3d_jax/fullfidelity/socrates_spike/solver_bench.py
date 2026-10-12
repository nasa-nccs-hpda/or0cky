"""R2 follow-up: timing and accuracy of the JAX solver_triple_hogan port on the visible device (V100 via sbatch).
Takes real triple_column groups from a dump (tags 41/42/44/46, k-term 1 column each), checks the device result against the Fortran flux_total
(tag 43), then tiles the group to N columns (several n_cloud_top values, so the static-int jit is recompiled per value; reported per value) and
times the steady state (median of NREP, block_until_ready).   usage: python solver_bench.py DUMP [NREP]"""
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import r2_parse as R
import r2_solver_check as C
from solver_triple_hogan import solver_triple_hogan

NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print("device:", jax.devices()[0], "jax", jax.__version__, flush=True)
recs = R.read(sys.argv[1])
first = {}
for g in C.groups(recs):
    a = lambda t, k: g[(t, k)]["a"]
    nct = int(a(44, 1).ravel()[0])
    key = (g["iso"], nct)
    if key not in first:
        first[key] = g
    if len(first) >= 6:
        break
out = {"device": str(jax.devices()[0]), "cases": []}
for (iso, nct), g in sorted(first.items()):
    a = lambda t, k: g[(t, k)]["a"]
    nl = int(a(44, 1).ravel()[3])
    tr, rf, sd, su, sg, co, fin, alb, orc = (a(41, 1), a(41, 2), a(42, 1), a(42, 2), a(42, 3), a(46, 1), a(46, 2), a(46, 3), a(43, 1))
    def prep(n):
        rep = lambda x: np.repeat(x, n, axis=0)
        v = rep(co[:, :, 0:9]).reshape(n, nl + 1, 3, 3)
        u = rep(co[:, :, 9:18]).reshape(n, nl + 1, 3, 3)
        args = [jnp.asarray(rep(x[:, :, k])) for x in (tr, rf, sd, su) for k in (0,)]
        return None
    def make(n):
        rep = lambda x: jnp.asarray(np.repeat(x, n, axis=0))
        regs = lambda arr, k: rep(arr[:, :, k])
        v = rep(co[:, :, 0:9]).reshape(n, nl + 1, 3, 3)
        u = rep(co[:, :, 9:18]).reshape(n, nl + 1, 3, 3)
        return (regs(tr, 0), regs(rf, 0), regs(sd, 0), regs(su, 0), regs(tr, 1), regs(rf, 1), regs(sd, 1), regs(su, 1), regs(tr, 2),
                regs(rf, 2), regs(sd, 2), regs(su, 2), v, u, rep(fin), rep(sg[:, 0:1])[:, 0], rep(sg[:, 1:2])[:, 0], rep(sg[:, 2:3])[:, 0], rep(alb))
    f = jax.jit(lambda *x: solver_triple_hogan(nct, *x))
    res = f(*make(1)); res.block_until_ready()
    ok = np.isfinite(orc)
    case = {"iso": "SW" if iso == 1 else "LW", "n_cloud_top": nct, "n_layer": nl,
            "bitwise_vs_fortran": float((np.asarray(res)[ok] == orc[ok]).mean()),
            "max_abs_vs_fortran": float(np.abs(np.asarray(res) - orc)[ok].max()), "timing": {}}
    for n in (3000, 30000, 300000):
        args = make(n)
        t0 = time.perf_counter(); r = f(*args); r.block_until_ready(); t1 = time.perf_counter() - t0
        ts = []
        for _ in range(NREP):
            t0 = time.perf_counter(); r = f(*args); r.block_until_ready(); ts.append(time.perf_counter() - t0)
        case["timing"][str(n)] = {"first_s": t1, "median_s": float(np.median(ts)), "us_per_column": float(np.median(ts)) / n * 1e6}
        del args, r
    print(json.dumps(case), flush=True)
    out["cases"].append(case)
json.dump(out, open("solver_bench_result.json", "w"), indent=1)
