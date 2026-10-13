"""Summarise a jax.profiler trace dir from its chrome trace (trace.json.gz): per process/thread event counts and summed durations, top names.
    python prof_xplane_summary.py TRACE_DIR [topN]      (jax 0.5.3 has no ProfileData reader; the .trace.json.gz is written beside the .xplane.pb)"""
import glob
import gzip
import json
import sys
from collections import defaultdict

d, top = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15
f = glob.glob(d + '/**/*.trace.json.gz', recursive=True)[0]
ev = json.load(gzip.open(f))['traceEvents']
pn, tn = {}, {}
for e in ev:
    if e.get('ph') == 'M':
        if e['name'] == 'process_name': pn[e['pid']] = e['args']['name']
        if e['name'] == 'thread_name': tn[(e['pid'], e['tid'])] = e['args']['name']
X = [e for e in ev if e.get('ph') == 'X']
grp = defaultdict(list)
for e in X:
    grp[(e['pid'], e['tid'])].append(e)
print('file', f, 'events', len(X))
for (p, t), es in sorted(grp.items(), key=lambda kv: -sum(e['dur'] for e in kv[1])):
    tot = sum(e['dur'] for e in es) / 1e6
    if tot < 0.02: continue
    t0, t1 = min(e['ts'] for e in es), max(e['ts'] + e['dur'] for e in es)
    print(f'\n{pn.get(p, p)} / {tn.get((p, t), t)}: {len(es)} events, sum {tot:.3f} s, span {(t1 - t0) / 1e6:.3f} s')
    agg = defaultdict(lambda: [0, 0.0])
    for e in es:
        a = agg[e['name']]; a[0] += 1; a[1] += e['dur'] / 1e6
    for n, (c, s) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
        print(f'   {s:8.4f} s  x{c:<6d} {n[:100]}')
