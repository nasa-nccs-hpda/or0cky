"""R2/R3-R5: rebuild tau and omega of every k-term (kterm.py) from the band's grey properties (tag 10), the absorber coefficient
k_abs_layer (tag 25), d_mass (tag 20) and forward_scatter (tag 49), compare with tag 40 (tau_clr, omega_clr, tau, omega).
   JAX_PLATFORMS=cpu python r2_kterm_check.py ffsoc_dump.bin"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R
from kterm import kterm_props


def kterm_groups(recs):
    """yields (band-level dict, k-term dict, iex) in file order."""
    band, kt, iex = None, None, 0
    for x in recs:
        t = x["tag"]
        if t == 0 and x["k"] == 1:
            band = None
        if t == 10 and x["k"] == 1:
            band, iex = {"iso": x["iso"]}, 0
        if band is None:
            continue
        if t in (10, 20, 25):
            band[(t, x["k"])] = x
        if t == 40 and x["k"] == 1:
            kt = {}
            iex += 1
        if t in (40, 41, 42, 43, 44, 45, 46, 47, 48, 49):
            if kt is not None:
                kt[(t, x["k"])] = x
        if t == 48 and kt is not None:
            yield band, kt, iex
            kt = None


if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0))
    ng = 0
    for rescale in (True, False):
        pass
    for band, kt, iex in kterm_groups(recs):
        if (25, 1) not in band or (49, 1) not in kt:
            continue
        a = lambda d, t, k: d[(t, k)]["a"]
        nct, nreg, nty, nl = [int(v) for v in a(kt, 44, 1).ravel()]
        n_abs, idx = int(a(band, 25, 4)[0]), a(band, 25, 4).astype(int)[1:]
        i_abs = idx[0]
        kabs = a(band, 25, 1)                    # (n_prof, n_layer, nd_k_term, nd_abs)
        k_gas = kabs[:, :, iex - 1, i_abs - 1]
        gt_clr, es_clr, gt, es = [a(band, 10, k) for k in (1, 2, 3, 4)]
        dm = a(band, 20, 2)
        keq = a(band, 20, 1)                      # k_eqv: added to the grey absorption by solve_band_k_eqv_scl before the k-term loop
        gt_clr = gt_clr.copy(); gt = gt.copy()
        gt_clr[:, :nct - 1] += keq[:, :nct - 1]
        gt[:, nct - 1:, :] += keq[:, nct - 1:, None]
        fc, f = a(kt, 49, 1), a(kt, 49, 2)
        tc, oc, t, o = [np.asarray(x) for x in kterm_props(gt_clr, es_clr, gt, es, k_gas, dm, nct, fc, f[..., :] if f.ndim == 3 else f)]
        ref = {"tau_clr": a(kt, 40, 1), "omega_clr": a(kt, 40, 2), "tau": a(kt, 40, 3), "omega": a(kt, 40, 4)}
        tag = "SW" if band["iso"] == 1 else "LW"
        for name, m, orc, sl in (("tau_clr", tc, ref["tau_clr"], slice(0, nct - 1)), ("omega_clr", oc, ref["omega_clr"], slice(0, nct - 1)),
                                 ("tau", t, ref["tau"], slice(nct - 1, nl)), ("omega", o, ref["omega"], slice(nct - 1, nl))):
            mm, oo = m[:, sl], orc[:, sl]
            s = st[(tag, name)]
            ok = np.isfinite(oo)
            d = np.abs(mm - oo)[ok]
            s["n"] += d.size; s["bit"] += int((mm[ok] == oo[ok]).sum())
            if d.size:
                s["maxabs"] = max(s["maxabs"], float(d.max()))
                s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(oo[ok]), 1e-300)).max()))
        ng += 1
    print("k-term groups: %d" % ng)
    for k in sorted(st):
        s = st[k]
        print("%-3s %-10s %10d entries  bitwise %8.3f %%  max abs %10.3e  max rel %10.3e" %
              (k[0], k[1], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
