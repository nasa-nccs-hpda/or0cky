"""Lower bound of the executed SOCRATES source: routines that received at least one perf sample in data/perf_*.txt, mapped to files of
ModelE_Support/socrates/src (read only).  Routines never sampled but executed are missed (a lower bound, 999 Hz x 3 profiles).
usage: python executed_lines.py data/perf_*.txt"""
import os, re, sys
SRC = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/socrates/src"
idx = {}
for d, _, fs in os.walk(SRC):
    for f in fs:
        b, e = os.path.splitext(f)
        if e.lower() in (".f90", ".f"):
            idx.setdefault(b.lower(), os.path.join(d, f))
names = set()
for fn in sys.argv[1:]:
    for l in open(fn):
        m = re.match(r"\s*[\d.]+%\s+(\d+)\s+\[.\]\s+(\S+)", l)
        if m and int(m.group(1)) > 0:
            names.add(m.group(2))
hit, tot = {}, 0
for n in names:
    k = re.sub(r"^(\w+)_mp_(\w+?)_?$", r"\2", n)
    k = k.rstrip("_").lower()
    if k in idx and "/radiance_core/" in idx[k] or k in idx and "/modules_core/" in idx[k]:
        hit[k] = idx[k]
for k, p in sorted(hit.items()):
    tot += sum(1 for _ in open(p, errors="replace"))
print("sampled symbols: %d; mapped to radiance_core/modules_core files: %d; lines in those files: %d" % (len(names), len(hit), tot))
print("largest:", sorted(((sum(1 for _ in open(p, errors="replace")), k) for k, p in hit.items()), reverse=True)[:12])
