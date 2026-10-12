"""Batched radiance_calc (radiance_batch.py: all calls of the dump as ONE batch, n_cloud_top = 1, per-column major absorber, padded k-terms)
against the band-end fluxes of the Fortran (tag 30; LW cumulative in one channel).   usage: python r2_batch_check.py DUMP SPECDIR2 [max_calls]"""
import sys, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R, spec_data as SD
import radiance_batch as RB
from r2_e2e_check import calls

if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    spec = {1: SD.load("%s/spec_sw.npz" % sys.argv[2]), 2: SD.load("%s/spec_lw.npz" % sys.argv[2])}
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    cl = [c for c in calls(recs) if c["ref"]]
    for iso in (1, 2):
        cs = [c for c in cl if c["iso"] == iso][:mx]
        r = RB.columns_from_calls(cs)
        t0 = time.time()
        res = RB.radiance(spec[iso], iso, r)
        jax.block_until_ready(res[-1][0])
        print("%s: %d columns, %d bands in %.1f s (first call incl. compile)" % ("SW" if iso == 1 else "LW", len(cs), len(res), time.time() - t0), flush=True)
        cum_d = cum_u = None
        worst = {}
        for ib, (dn, up, dr) in enumerate(res, start=1):
            dn, up = np.asarray(dn), np.asarray(up)
            if iso == 2:
                cum_d = dn if cum_d is None else cum_d + dn
                cum_u = up if cum_u is None else cum_u + up
                dn, up = cum_d, cum_u
            for ci, c in enumerate(cs):
                if ib not in c["ref"]:
                    continue
                for name, m, ref in (("down", dn[ci], c["ref"][ib][1][0]), ("up", up[ci], c["ref"][ib][2][0])):
                    d = np.abs(m - ref)
                    w = worst.setdefault(name, dict(n=0, bit=0, abs=0.0, rel=0.0))
                    w["n"] += d.size; w["bit"] += int((m == ref).sum()); w["abs"] = max(w["abs"], float(d.max()))
                    w["rel"] = max(w["rel"], float((d / np.maximum(np.abs(ref), 1e-300)).max()))
        for name, w in worst.items():
            print("   %-4s flux_%-4s %6d entries  bitwise %6.2f %%  max abs %.3e  max rel %.3e" % ("SW" if iso == 1 else "LW", name, w["n"], 100.0 * w["bit"] / w["n"], w["abs"], w["rel"]))
