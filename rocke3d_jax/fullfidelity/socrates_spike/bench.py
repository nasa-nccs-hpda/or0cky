"""SOCRATES spike S3: timing of the two prototype kernels on the visible JAX device (V100 via gpu/v100_run.sh).
Compile time is reported separately from steady-state; steady = median of NREP runs with block_until_ready.
Inputs: kernel A uses temperature/humidity/pressure of a RECORDED packet (rsv_n26_33317_in.bin) for a relative humidity (Tetens, an
approximation of ModelE's RHL; timing only) and the real aerosol tables; kernel B uses random but physically ranged optical depths
(timing only).  The term counts per band are from the spectral files (upper bound: sum over absorbers of k-terms).
usage: python bench.py [NREP]"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, jax, jax.numpy as jnp
import kernels as K
import radiation_server as rs

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SB = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/spectral_files/"
dev = jax.devices()[0]
print("device:", dev, "jax", jax.__version__, flush=True)


def timeit(fn, *a):
    t0 = time.perf_counter(); r = fn(*a); jax.block_until_ready(r); first = time.perf_counter() - t0
    ts = []
    for _ in range(NREP):
        t0 = time.perf_counter(); r = fn(*a); jax.block_until_ready(r); ts.append(time.perf_counter() - t0)
    return first, float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


res = {"device": str(dev)}
# ---------------- kernel A
pk = rs.read_packet("%s/nov26_day/rsv_n26_33317_in.bin" % rs.FF)
T, Q, P = pk["T"], pk["Q"], pk["PMID"]            # (IM,JM,LM)
IM, JM, LM = T.shape
N = IM * JM
Tc, Qc, Pc = [x.reshape(N, LM) for x in (T, Q, P)]
Tk = Tc if Tc.mean() > 100 else Tc + 273.15
es = 611.2 * np.exp(17.67 * (Tk - 273.15) / (Tk - 29.65))
qs = 0.622 * es / (Pc * 100.0 - 0.378 * es)
rh = np.clip(Qc / qs, 0.0, 1.2)
print("columns", N, "layers", LM, "RH min/mean/max %.3f %.3f %.3f" % (rh.min(), rh.mean(), rh.max()), flush=True)
mass = jnp.asarray(np.full((N, LM), 2000.0))
rng = np.random.default_rng(0)
mdd = jnp.asarray(rng.uniform(1e-12, 1e-8, (6, N, LM)))
fs = [0.3, 0.3, 0.3, 0.3, 0.0, 0.0]
comp = [0, 1, 2, 3, 4, 5]
for label, path in (("lw", SB + "sp_lw_ga7/aer_lw_ga7.nc"), ("sw", SB + "sp_sw_ga7/aer_sw_ga7.nc")):
    t = K.aer_table_to_arrays(path)
    tbl = {k: jnp.asarray(v) for k, v in t.items()}
    tbl.update(_rad_np=t["radius_eff_dry"], _nrad_np=t["n_radius"], _lhum_np=t["l_humidity"])
    radius = [float(0.5 * (t["radius_eff_dry"][c, 0] + t["radius_eff_dry"][c, int(t["n_radius"][c]) - 1])) for c in comp]
    f = jax.jit(lambda rh_, m_, d_: K.aer_optical(tbl, rh_, m_, comp, radius, d_, fs, [1.0] * 6, 6))
    r = timeit(f, jnp.asarray(rh), mass, mdd)
    print("kernel A %s (3312 cols x 40 layers x 6 aerosols, 4 with sulphate): compile+first %.3f s, steady median %.4f s (min %.4f max %.4f)" % ((label,) + r), flush=True)
    res["A_" + label] = r

# ---------------- kernel B
def bench_B(label, ncol, nterm, regions, solar, chunk_cols, scheme):
    L = LM
    chunks = ncol // chunk_cols
    Bn = chunk_cols * nterm * regions
    rr = np.random.default_rng(1)
    tau = jnp.asarray(10 ** rr.uniform(-4, 1, (L, Bn))); om = jnp.asarray(rr.uniform(0, 1, (L, Bn))); g = jnp.asarray(rr.uniform(0, 0.9, (L, Bn)))
    sec0 = jnp.asarray(rr.uniform(1.0, 5.0, Bn)); da = jnp.asarray(rr.uniform(0, 1, Bn)); dra = jnp.asarray(rr.uniform(0, 1, Bn))
    fin = jnp.asarray(rr.uniform(100, 1300, Bn)); su = jnp.asarray(rr.uniform(0, 1, (L, Bn))); sd = jnp.asarray(rr.uniform(0, 1, (L, Bn)))
    dpl = jnp.asarray(rr.uniform(1, 400, Bn))
    f = jax.jit(lambda *a: K.two_stream_homogen(a[0], a[1], a[2], scheme, solar, sec_0=a[3], diffuse_albedo=a[4], direct_albedo=a[5],
                                                flux_inc_down=a[6], s_up=a[7], s_down=a[8], d_planck_flux_surface=a[9]))
    args = (tau, om, g, sec0, da, dra, fin, su, sd, dpl)
    first, med, mn, mx = timeit(f, *args)
    tot = med * chunks
    print("kernel B %s: %d cols x %d terms x %d regions x %d layers, chunk %d cols (batch %d): compile+first %.3f s, steady %.4f s per chunk "
          "-> %.3f s for all %d chunks (min %.4f max %.4f)" % (label, ncol, nterm, regions, L, chunk_cols, Bn, first, med, tot, chunks, mn, mx), flush=True)
    return {"first": first, "chunk_median": med, "chunks": chunks, "total": tot, "min": mn, "max": mx, "batch": Bn}

for regions in (1, 3):
    res["B_lw_r%d" % regions] = bench_B("LW elsasser", 3312, 180, regions, False, 276, "elsasser")
    res["B_sw_r%d" % regions] = bench_B("SW pifm80 (half the columns lit)", 1656, 95, regions, True, 276, "pifm80")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_result_%s.json" % os.environ.get("SLURM_JOB_ID", "local")), "w"), indent=1)
