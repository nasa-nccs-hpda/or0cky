"""Group the flat perf profiles (data/perf_*.txt) into stages by routine name.  usage: python group_profile.py data/perf_*.txt"""
import re, sys, collections
G = [
 ("A  aerosol-table interpolation, ModelE planet_rad.F90 (interp_aer, calc_aer_opt_prop, set_aer)", r"^(interp_aer_|calc_aer_opt_prop_|set_aer_)$"),
 ("B  two-stream coefficients, transmission/source, flux solvers (triple/homogen, two_coeff*, trans_source_coeff, solar/ir/triple sources, diff_planck)",
  r"^(solver_|trans_source_coeff_|two_coeff|triple_|solar_|ir_source_|diff_planck|integ_planck|column_solver_|adjust_ir|augment_|calc_planck|calc_radiance|monochromatic_gas_flux_|single_scattering_all_|ir_source)"),
 ("C  optical properties (gas k-terms/continua, aerosol/cloud/grey, scattering rescale, ukca aerosol, cloud fields)",
  r"^(grey_opt_prop_|opt_prop_|single_scattering_|rescale_|set_major_abs_|scale_absorb_|inter_pt_lookup_|set_cloud_field_|aggregate_cloud_|calc_cont|cloud_|mcica|overlap_|set_atm_|repart_)"),
 ("D  band/k-term driver (radiance_calc self, solve_band_*, monochromatic_radiance*)", r"^(radiance_calc_|solve_band_|monochromatic_radiance)"),
 ("E  math library, memory, allocation, I/O (exp/pow/sqrt/log, malloc/free, memcpy/memset, for_*)", r"^(__libm|exp$|__powr8|vectlib_|_int_|_mm_|malloc|cfree|__intel|_intel|for_|for__|f_ldnint|unlink|__errno|__close|__libc|0x|__GI|def_.*(alloc|deallo))"),
 ("F  ModelE side other (interp1d, radpar_*, getdst/setaer, radia_, planet_rad*, planet_alb, ghg)", r"^(interp1d_|radpar_|radia_|planet_|ghgmod_|aerparam_|radsrvp|atm_)"),
]
rows = []
for fn in sys.argv[1:]:
    s = collections.Counter(); n = 0
    for l in open(fn):
        m = re.match(r"\s*([\d.]+)%\s+(\d+)\s+\[.\]\s+(\S+)", l)
        if m:
            s[m.group(3)] += int(m.group(2)); n += int(m.group(2))
    g = collections.Counter()
    for k, v in s.items():
        for name, rx in G:
            if re.search(rx, k):
                g[name] += v; break
        else:
            g["other / unclassified"] += v
    rows.append({k: 100.0 * v / n for k, v in g.items()} | {"_n": n})
print("samples per profile (999 Hz):", [r["_n"] for r in rows])
for name in [n for n, _ in G] + ["other / unclassified"]:
    v = [r.get(name, 0.0) for r in rows]
    print("%5.1f %% (min %.1f max %.1f)  %s" % (sum(v) / len(v), min(v), max(v), name))
