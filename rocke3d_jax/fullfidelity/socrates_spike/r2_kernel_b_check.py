"""R2: compare kernels.two_coeff (spike kernel B, coefficient part) with the REAL Fortran two_coeff_region output dumped in tags 40-45 of
ffsoc_dump.bin.  Inputs of each k-term group: tag 40 (ss_prop arrays, sec_0), 44 (n_cloud_top, n_region, n_cloud_type, n_layer), 45 (frac_cloud,
frac_region, i_region_cloud); oracle: tag 41 (trans, reflect, trans_0_noscal, trans_0, source_coeff).  Region 1 (clear) is rebuilt for layers
1..n_cloud_top-1 from the *_clr arrays and for n_cloud_top..n_layer from the cloud-type-0 slice; regions 2-3 as the Fortran does: sum over cloud
types of frac_cloud * coefficients, divided by frac_region.   JAX_PLATFORMS=cpu python r2_kernel_b_check.py ffsoc_dump.bin"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R
import kernels as K
TINY = np.finfo(np.float64).tiny
def groups(recs):
    cur = None
    for x in recs:
        if x["tag"] == 900: continue
        if x["tag"] == 40 and x["k"] == 1:
            cur = {}
        if cur is None: continue
        cur[(x["tag"], x["k"])] = x
        cur["iso"] = x["iso"]
        if x["tag"] == 41 and x["k"] == 5:
            yield cur
            cur = None
def region_coeffs(g, solar, scheme):
    t = lambda k: g[(40, k)]["a"]
    tau_clr, om_clr, tau, om = t(1), t(2), t(3), t(4)
    tn_clr, tn, ph_clr, ph, sec0 = t(5), t(6), t(7), t(8), t(9)
    nct, nreg, nty, nl = [int(v) for v in g[(44, 1)]["a"].ravel()]
    fc, fr, irc = g[(45, 1)]["a"], g[(45, 2)]["a"], g[(45, 3)]["a"].ravel().astype(int)
    # NOTE: the dump arrays come through assumed-shape dummies, so the recorded lower bounds are always 1 and carry no information.
    # Index conventions (checked against the shapes): layer index = layer - 1 (id_cloud_top = 1: shape[1] == n_layer), cloud type
    # index in tau/omega/tau_noscal/phase_fnc = type (0 = clear part, 1..n_cloud_type), frac_cloud index = type - 1.
    assert tau.shape[1] == nl and tau.shape[2] == nty + 1 and fc.shape[1] == nl, (tau.shape, fc.shape, nl)
    lb_tau = lb_ph = lb_fc = (1, 1, 0, 0)
    s0 = sec0[0:1]
    names = ["trans", "reflect"] + (["trans_0", "trans_0_noscal", "s_up", "s_down"] if solar else ["ir_1d", "ir_2d"])
    def coef(tau_, tn_, om_, g_, lay_slice=None):
        return K.two_coeff(tau_, tn_, om_, g_, scheme, solar, s0 if solar else None, True)
    res = {n: np.zeros((nl, nreg)) for n in names}
    # clear region
    layers_clr = np.arange(1, nct)
    c1 = {k: np.asarray(v) for k, v in coef(tau_clr[0, :nct - 1], tn_clr[0, :nct - 1], om_clr[0, :nct - 1], ph_clr[0, :nct - 1, 0]).items()}
    i0 = nct - 1
    c2 = {k: np.asarray(v) for k, v in coef(tau[0, i0:i0 + nl - nct + 1, 0], tn[0, i0:i0 + nl - nct + 1, 0],
                                            om[0, i0:i0 + nl - nct + 1, 0],
                                            ph[0, i0:i0 + nl - nct + 1, 0, 0]).items()}
    for n in names:
        res[n][:nct - 1, 0] = c1[n]
        res[n][nct - 1:, 0] = c2[n]
    # cloudy regions
    tiny = TINY
    for ik in range(1, nty + 1):
        r = irc[ik - 1] - 1
        sl = slice(i0, i0 + nl - nct + 1)
        ck = {k: np.asarray(v) for k, v in coef(tau[0, sl, ik], tn[0, sl, ik], om[0, sl, ik],
                                                ph[0, sl, 0, ik]).items()}
        f = fc[0, nct - 1:nl, ik - 1]      # layers nct..nl of the dumped id_ct:n_layer slice
        use = f > tiny
        for n in names:
            res[n][nct - 1:, r] += np.where(use, f * ck[n], 0.0)
    for r in range(1, nreg):
        f = fr[0, nct - 1:nl, r]
        use = f > tiny
        for n in names:
            res[n][nct - 1:, r] = np.where(use, res[n][nct - 1:, r] / np.where(use, f, 1.0), 0.0)
    return res, (nct, nreg, nty, nl)
def oracle_of(g, solar, n_layer_total):
    tr = g[(41, 1)]["a"][0]; rf = g[(41, 2)]["a"][0]; t0n = g[(41, 3)]["a"][0]; t0 = g[(41, 4)]["a"][0]; sc = g[(41, 5)]["a"][0]
    if solar:
        return {"trans": tr, "reflect": rf, "trans_0": t0, "trans_0_noscal": t0n, "s_up": sc[:, 0, :], "s_down": sc[:, 1, :]}
    return {"trans": tr, "reflect": rf, "ir_1d": sc[:, 0, :], "ir_2d": sc[:, 1, :]}
if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    stat = collections.defaultdict(lambda: dict(n=0, bit=0, maxrel=0.0, maxabs=0.0, nonfin=0))
    ngr = 0
    for g in groups(recs):
        solar = g["iso"] == 1
        scheme = K.IP_PIFM80 if solar else K.IP_ELSASSER
        mine, (nct, nreg, nty, nl) = region_coeffs(g, solar, scheme)
        orc = oracle_of(g, solar, nl)
        ngr += 1
        for n, m in mine.items():
            for r in range(nreg):
                lo = 0 if r == 0 else nct - 1                # valid layers (Fortran leaves the others uninitialised)
                a, b = m[lo:, r], orc[n][lo:nl, r]
                key = ("SW" if solar else "LW", n, "clear" if r == 0 else "cloudy")
                s = stat[key]
                s["n"] += a.size
                ok = np.isfinite(b)
                s["nonfin"] += int((~ok).sum())
                a, b = a[ok], b[ok]
                s["bit"] += int((a == b).sum())
                d = np.abs(a - b)
                if d.size:
                    s["maxabs"] = max(s["maxabs"], float(d.max()))
                    s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(b), 1e-300)).max()))
    print("k-term groups compared: %d" % ngr)
    print("%-3s %-15s %-7s %10s %9s %11s %11s" % ("", "quantity", "region", "entries", "bitwise%", "max rel", "max abs"))
    for k in sorted(stat):
        s = stat[k]
        print("%-3s %-15s %-7s %10d %9.3f %11.3e %11.3e%s" % (k[0], k[1], k[2], s["n"], 100.0 * s["bit"] / max(s["n"] - s["nonfin"], 1), s["maxrel"], s["maxabs"],
              ("  (%d non-finite oracle entries skipped)" % s["nonfin"]) if s["nonfin"] else ""))
