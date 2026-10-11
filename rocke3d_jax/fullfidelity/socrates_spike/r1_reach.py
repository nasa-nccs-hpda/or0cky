"""R1 (decision D3, ACCEPTANCE_CRITERIA s12): static reachability from radiance_calc over the SOCRATES source, READ-ONLY.
Edges: CALL statements, USE module (module -> its file), references to FUNCTIONs defined in the tree.  Comments and strings are stripped.
`--block a,b,..` removes routines (and everything reachable only through them) to model the options of this rundeck; the blocked set
used in R1_REACHABILITY.md is printed by `--rundeck`.   python r1_reach.py [--rundeck] [--block names] [--list]"""
import os, re, sys
CALLS_ONLY = "--calls-only" in sys.argv   # routines reached by CALL only (executable code); default also pulls USEd module files (types/constants)
SRC = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/socrates/src"
ROOT = "radiance_calc"
# Routines NOT selected by this rundeck (planet_rad.F90 set_solver_cloud_scheme + lw/sw_control defaults: csiw, cairns, max_rand,
# scatter_full, k_eqv_scl, ses2 only for some bands, l_clear false, McICA off).  Derived from the source, listed for review.
RUNDECK_BLOCK = """solve_band_k_eqv solve_band_random_overlap solve_band_random_overlap_resort_rebin
mcica_column mcica_sample
mix_column part_corr_column part_corr_cnv_column calc_flux_ipa
monochromatic_radiance_sph sph_solver build_sph_matrix sph_matrix_solver gauss_angle calc_radiance_ipa
solver_triple solver_triple_app_scat solver_no_scat band_solver column_solver""".split()
def strip(t):
    out = []
    for l in t.splitlines():
        l = re.sub(r"'[^']*'|\"[^\"]*\"", "''", l)
        l = l.split("!")[0]
        if l.strip(): out.append(l)
    return "\n".join(out)
files = {}
for d, _, fs in os.walk(SRC):
    for f in fs:
        if f.lower().endswith((".f90", ".f")):
            files[os.path.join(d, f)] = f
defs, mods = {}, {}           # routine/function -> file ; module -> file
txt = {}
for p in files:
    t = strip(open(p, errors="replace").read()); txt[p] = t
    for m in re.finditer(r"^\s*(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+)*(SUBROUTINE|FUNCTION)\s+(\w+)", t, re.I | re.M):
        defs.setdefault(m.group(2).lower(), p)
    for m in re.finditer(r"^\s*(?:\w+(?:\([^)]*\))?\s+)*FUNCTION\s+(\w+)", t, re.I | re.M):
        defs.setdefault(m.group(1).lower(), p)
    for m in re.finditer(r"^\s*MODULE\s+(\w+)\s*$", t, re.I | re.M):
        mods[m.group(1).lower()] = p
    base = os.path.splitext(files[p])[0].lower()
    defs.setdefault(base, p)
fnames = {n for n, p in defs.items()}
def edges(p):
    t = txt[p]; e = set()
    for m in re.finditer(r"\bCALL\s+(\w+)", t, re.I): e.add(m.group(1).lower())
    for m in re.finditer(r"^\s*USE\s+(\w+)", t, re.I | re.M):
        e.add("@" + m.group(1).lower())
    return e
def reach(block):
    seen_files, stack = {}, [defs[ROOT]]
    seen_names = set()
    while stack:
        p = stack.pop()
        if p in seen_files: continue
        seen_files[p] = True
        for n in edges(p):
            if n.startswith("@"):
                q = None if CALLS_ONLY else mods.get(n[1:])
            else:
                if n in block: continue
                q = defs.get(n)
            if q and q not in seen_files and files[q].lower().rsplit(".", 1)[0] not in block:
                stack.append(q)
    return list(seen_files)
def n_lines(p): return sum(1 for _ in open(p, errors="replace"))
def report(label, ps):
    by = {}
    for p in ps:
        k = os.path.relpath(p, SRC).split(os.sep)[0]
        by.setdefault(k, []).append(p)
    print("%s: %d files, %d lines" % (label, len(ps), sum(n_lines(p) for p in ps)))
    for k, v in sorted(by.items()): print("   %-18s %3d files %6d lines" % (k, len(v), sum(n_lines(p) for p in v)))
if __name__ == "__main__":
    a = sys.argv[1:]
    block = set(RUNDECK_BLOCK) if "--rundeck" in a else set()
    if "--block" in a: block |= set(a[a.index("--block") + 1].split(","))
    allr = reach(set()); report("all options", allr)
    if block:
        r = reach(block); report("rundeck (blocked: %d names)" % len(block), r)
        if "--list" in a:
            for p in sorted(r, key=n_lines, reverse=True): print("%6d %s" % (n_lines(p), os.path.relpath(p, SRC)))
