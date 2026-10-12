"""R2/R5: triple_column (ir_source / triple_solar_source / solver_triple_hogan, triple_column.py) against the dumped Fortran values.
Mode A (inputs = the Fortran's own two-stream coefficients, tag 41): sources (tag 42), flux_total (tag 43) and flux_direct (tag 48) are
expected to be bitwise.  Mode B (coefficients rebuilt by kernels.two_coeff from tags 40/44/45, r2_kernel_b_check.region_coeffs): the
differences are the rounding-level ones of section 8.   JAX_PLATFORMS=cpu python r2_triple_check.py ffsoc_dump.bin"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import r2_parse as R
import kernels as K
import r2_kernel_b_check as B
from triple_column import triple_column, SOLAR, IR


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
        if x["tag"] == 48:
            yield cur
            cur = None


def inputs(g):
    a = lambda t, k: g[(t, k)]["a"]
    nct, nreg, nty, nl = [int(v) for v in a(44, 1).ravel()]
    solar = g["iso"] == 1
    co = a(46, 1)
    kw = dict(diff_planck=a(47, 1), diff_planck_2=a(47, 2), w_free=a(47, 3), w_cloud=a(47, 4), frac_region=a(45, 2),
              d_planck_flux_surface=a(47, 8), flux_inc_direct=a(47, 6), adjust_solar_ke=a(47, 5),
              l_scale_solar=bool(a(47, 9).ravel()[0] > 0.5), direct_albedo=a(47, 7))
    return nct, nl, solar, co, a(46, 2), a(46, 3), kw


def coeffs_oracle(g, nl):
    tr, rf, t0, sc = g[(41, 1)]["a"], g[(41, 2)]["a"], g[(41, 4)]["a"], g[(41, 5)]["a"]
    return tr, rf, t0, sc


def coeffs_mine(g, solar):
    scheme = K.IP_PIFM80 if solar else K.IP_ELSASSER
    mine, (nct, nreg, nty, nl) = B.region_coeffs(g, solar, scheme)
    n = {k: v[None] for k, v in mine.items()}            # (1, n_layer, 3)
    if solar:
        sc = np.stack([n["s_up"], n["s_down"]], axis=2)   # (1, n_layer, 2, 3)
        t0 = n["trans_0"]
    else:
        sc = np.stack([n["ir_1d"], n["ir_2d"]], axis=2)
        t0 = np.zeros_like(n["trans"])
    return n["trans"], n["reflect"], t0, sc


def compare(st, key, m, o):
    s = st[key]
    ok = np.isfinite(o) & np.isfinite(m)
    s["n"] += int(ok.sum()); s["bit"] += int((m[ok] == o[ok]).sum()); s["nonfin"] += int((~ok).sum())
    d = np.abs(m - o)[ok]
    if d.size:
        s["maxabs"] = max(s["maxabs"], float(d.max()))


if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    stA = collections.defaultdict(lambda: dict(n=0, bit=0, nonfin=0, maxabs=0.0))
    stB = collections.defaultdict(lambda: dict(n=0, bit=0, nonfin=0, maxabs=0.0))
    ng = 0
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1     # use every step-th k-term group (eager JAX is slow)
    for gi, g in enumerate(groups(recs)):
        if gi % step:
            continue
        nct, nl, solar, co, fin, alb, kw = inputs(g)
        iso = SOLAR if solar else IR
        tag = "SW" if solar else "LW"
        valid_l = np.arange(nl) >= nct - 1                   # layers of regions 2,3 that the Fortran defines
        o42d, o42u = g[(42, 1)]["a"], g[(42, 2)]["a"]
        o43, o48 = g[(43, 1)]["a"], g[(48, 1)]["a"]
        ng += 1
        for mode, st, cf in (("A", stA, coeffs_oracle(g, nl)), ("B", stB, coeffs_mine(g, solar))):
            tr, rf, t0, sc = [jnp.asarray(x) for x in cf]
            r = triple_column(iso, nct, tr, rf, t0, sc, jnp.asarray(co), jnp.asarray(fin), jnp.asarray(alb),
                              **{k: (jnp.asarray(v) if not isinstance(v, bool) else v) for k, v in kw.items()})
            compare(st, (tag, "flux_total"), np.asarray(r["flux_total"]), o43)
            for rg in range(3):
                lo = 0 if rg == 0 else nct - 1
                compare(st, (tag, "s_down"), np.asarray(r["s_down"])[:, lo:, rg], o42d[:, lo:, rg])
                compare(st, (tag, "s_up"), np.asarray(r["s_up"])[:, lo:, rg], o42u[:, lo:, rg])
            compare(st, (tag, "source_flux_ground"), np.asarray(r["source_flux_ground"]), g[(42, 3)]["a"])
            if solar:
                compare(st, (tag, "flux_direct"), np.asarray(r["flux_direct"]), o48)
    print("triple_column groups: %d" % ng)
    for name, st in (("MODE A: Fortran coefficients (tag 41) -> sources, solver", stA), ("MODE B: coefficients rebuilt by kernels.two_coeff", stB)):
        print(name)
        for k in sorted(st):
            s = st[k]
            print("  %-3s %-20s %9d entries  bitwise %8.3f %%  max abs %10.3e  nonfinite skipped %d" %
                  (k[0], k[1], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["nonfin"]))
