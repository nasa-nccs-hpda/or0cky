"""Device benchmark of the batched radiance_calc (radiance_batch.py) at production column counts: the dumped one-column calls are tiled to
3,170 LW and 1,826 SW columns (the model's per-step counts), all bands are compiled once (reported), then the whole isolir is timed
(median of NREP, block_until_ready).  The first dumped columns are compared with the Fortran band-end fluxes (tag 30) as a safety check.
usage: python radiance_bench.py DUMP SPECDIR2 [NREP]"""
import sys, time, json
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import r2_parse as R, spec_data as SD
import radiance_batch as RB
from r2_e2e_check import calls

recs = R.read(sys.argv[1])
spec = {1: SD.load("%s/spec_sw.npz" % sys.argv[2]), 2: SD.load("%s/spec_lw.npz" % sys.argv[2])}
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cl = [c for c in calls(recs) if c["ref"]]
print("device:", jax.devices()[0], flush=True)
out = {"device": str(jax.devices()[0])}
for iso, ncol in ((2, 3170), (1, 1826)):
    cs = [c for c in cl if c["iso"] == iso]
    r = RB.columns_from_calls(cs)
    n0 = len(cs)
    big = {}
    for k, v in r.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n0 and k not in (21, 22, 23, 24, 25, 26, 31, 20, 32, 33, 34, 35, 36):
            reps = int(np.ceil(ncol / n0))
            big[k] = np.concatenate([v] * reps, axis=0)[:ncol]
        else:
            big[k] = v
    # tiles: keep l_tile off for the tiled benchmark if the first call had it off; per-column tile arrays are not tiled here
    tag = "LW" if iso == 2 else "SW"
    t0 = time.time()
    X = RB.prepare(spec[iso], iso, big)
    scheme = RB.K.IP_PIFM80 if iso == 1 else RB.K.IP_ELSASSER
    nb = spec[iso]["n_band_absorb"].shape[0]
    fs = [jax.jit((lambda ib_: (lambda XX: RB.band_fluxes(spec[iso], iso, ib_, XX, scheme)))(ib)) for ib in range(1, nb + 1)]
    res = []
    for ib, f in enumerate(fs, start=1):
        t1 = time.time(); o = f(X); jax.block_until_ready(o); res.append(o)
        print("%s band %d: compile+first run %.1f s" % (tag, ib, time.time() - t1), flush=True)
    comp = time.time() - t0
    ts = []
    for _ in range(NREP):
        t1 = time.time()
        o = [f(X) for f in fs]
        jax.block_until_ready(o)
        ts.append(time.time() - t1)
    print("%s %d columns, %d bands: compile+first %.1f s, steady %.3f s (median of %d: %s)" %
          (tag, ncol, nb, comp, float(np.median(ts)), NREP, ["%.3f" % x for x in ts]), flush=True)
    out[tag] = dict(columns=ncol, compile_first_s=comp, steady_s=float(np.median(ts)), runs=ts)
    # whole pipeline in ONE jit (grey properties, Planck, all bands): this is the number to quote
    COL = (1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27, 28, 29, 30, "tile_flag", "tile_n", "tile_frac", "tile_t")
    percol = {str(k): jnp.asarray(big[k]) for k in COL if k in big}
    consts = {k: v for k, v in big.items() if str(k) not in percol}
    def full(pc):
        rr = dict(consts); rr.update({(int(k) if k.isdigit() else k): v for k, v in pc.items()})
        XX = RB.prepare(spec[iso], iso, rr)
        return [RB.band_fluxes(spec[iso], iso, ib_, XX, scheme) for ib_ in range(1, nb + 1)]
    ff = jax.jit(full)
    t1 = time.time(); o = ff(percol); jax.block_until_ready(o); first_full = time.time() - t1
    ts2 = []
    for _ in range(NREP):
        t1 = time.time(); o = ff(percol); jax.block_until_ready(o); ts2.append(time.time() - t1)
    print("%s whole pipeline in one jit: compile+first %.1f s, steady %.3f s (%s)" % (tag, first_full, float(np.median(ts2)),
                                                                                       ["%.3f" % x for x in ts2]), flush=True)
    out[tag]["one_jit_compile_first_s"] = first_full
    out[tag]["one_jit_steady_s"] = float(np.median(ts2))
    # safety check on the first n0 (real) columns
    cum_d = cum_u = None
    worst_abs = 0.0
    for ib, (dn, up, dr) in enumerate(res, start=1):
        dn, up = np.asarray(dn)[:n0], np.asarray(up)[:n0]
        if iso == 2:
            cum_d = dn if cum_d is None else cum_d + dn
            cum_u = up if cum_u is None else cum_u + up
            dn, up = cum_d, cum_u
        for ci, c in enumerate(cs):
            if ib in c["ref"]:
                worst_abs = max(worst_abs, float(np.abs(dn[ci] - c["ref"][ib][1][0]).max()), float(np.abs(up[ci] - c["ref"][ib][2][0]).max()))
    print("%s check vs Fortran band-end fluxes (first %d columns): max abs %.3e" % (tag, n0, worst_abs), flush=True)
    out[tag]["max_abs_vs_fortran"] = worst_abs
json.dump(out, open("radiance_bench_result.json", "w"), indent=1)
