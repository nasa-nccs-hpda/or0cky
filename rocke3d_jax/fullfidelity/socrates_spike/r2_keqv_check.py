"""R3: rebuild k_eqv (tag 20/1) and, for the solar bands, adjust_solar_ke (tag 47/5) from the band inputs of tag 25 (k_abs_layer, weights,
index_abs, Planck terms) with k_eqv.py.   JAX_PLATFORMS=cpu python r2_keqv_check.py ffsoc_dump.bin"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import r2_parse as R
import k_eqv as KE
from r2_kterm_check import kterm_groups

if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0))
    done = set()
    for band, kt, iex in kterm_groups(recs):
        key = id(band)
        if key in done or (25, 1) not in band:
            continue
        done.add(key)
        a = lambda t, k: band[(t, k)]["a"]
        n_abs = int(a(25, 4)[0]); idx = a(25, 4).astype(int)[1:]
        kabs, w, nes = a(25, 1), a(25, 2), a(25, 3).astype(int)
        dm, zen = a(20, 2), a(20, 3)
        solar = band["iso"] == 1
        if solar:
            ke, adj = KE.k_eqv_solar(jnp.asarray(kabs), nes, w, idx, n_abs, jnp.asarray(dm), jnp.asarray(zen))
            comp = [("SW", "k_eqv", np.asarray(ke), a(20, 1)), ("SW", "adjust_solar_ke", np.asarray(adj), kt[(47, 5)]["a"])]
        else:
            # diffuse albedo: rho_alb index 2 (ip_surf_alb_diff) assumed; the check below shows whether it matches tag 46/3
            alb = kt[(46, 3)]["a"]
            ke = KE.k_eqv_ir(jnp.asarray(kabs), nes, w, idx, n_abs, jnp.asarray(dm), jnp.asarray(a(25, 5)), jnp.asarray(a(25, 6)),
                             jnp.asarray(a(25, 8)), jnp.asarray(alb))
            comp = [("LW", "k_eqv", np.asarray(ke), a(20, 1))]
        for tg, name, m, o in comp:
            s = st[(tg, name)]
            ok = np.isfinite(o)
            d = np.abs(m - o)[ok]
            s["n"] += d.size; s["bit"] += int((m[ok] == o[ok]).sum())
            if d.size:
                s["maxabs"] = max(s["maxabs"], float(d.max()))
                s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(o[ok]), 1e-300)).max()))
    print("band records compared: %d" % len(done))
    for k in sorted(st):
        s = st[k]
        print("%-3s %-16s %9d entries  bitwise %8.3f %%  max abs %10.3e  max rel %10.3e" %
              (k[0], k[1], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
