"""Kernel launches per loop iteration in an XLA-dumped optimized HLO text (D214 optimisation tool; numpy-free).

usage: python hlo_loop_report.py <module>.after_optimizations.txt [more files]
For every `while` found, reports the body and condition computations and the number of top-level instructions in the body that launch work on a GPU (fusion, custom-call,
copy, dynamic-slice, dynamic-update-slice, scatter, gather, reduce, sort, select-and-scatter, ...), counting nested while / conditional bodies once and printing their
own counts separately.  Loop trip counts are not in the text (data dependent or known elsewhere): multiply by hand.  Also prints the total launch-type instruction count.
"""
import re
import sys
from collections import defaultdict

LAUNCH = ('fusion', 'custom-call', 'copy', 'dynamic-slice', 'dynamic-update-slice', 'scatter', 'gather', 'reduce', 'sort', 'select-and-scatter',
          'convolution', 'dot', 'reduce-window', 'all-reduce', 'cholesky', 'triangular-solve', 'fft', 'rng', 'rng-bit-generator', 'iota', 'pad', 'concatenate',
          'transpose', 'reverse', 'slice', 'broadcast', 'convert', 'add', 'multiply', 'select', 'compare')
SKIP = ('parameter', 'constant', 'tuple', 'get-tuple-element', 'bitcast', 'while', 'conditional', 'call', 'after-all', 'partition-id', 'replica-id')


def parse(path):
    comps, cur, name = {}, None, None
    for ln in open(path):
        m = re.match(r'^(?:ENTRY\s+)?%?([\w.\-]+)\s*(?:\(.*)?\{\s*$', ln.rstrip())
        if m and not ln.startswith(' '):
            name = m.group(1)
            cur = comps.setdefault(name, [])
            continue
        if ln.startswith('}'):
            cur = None
            continue
        if cur is not None and '=' in ln:
            cur.append(ln.strip())
    return comps


def opcode(ln):
    m = re.match(r'^(?:ROOT\s+)?%?[\w.\-]+\s*=\s*[^ ]+(?:\s[^ ]+)*?\s+([a-z][\w\-]*)\(', ln)
    if m:
        return m.group(1)
    m = re.search(r'=\s*(?:\([^)]*\)|[\w\[\],{}:]+)\s+([a-z][\w\-]*)\(', ln)
    return m.group(1) if m else None


def count(body):
    n = defaultdict(int)
    for ln in body:
        op = opcode(ln)
        if op and op not in SKIP:
            n[op] += 1
    return n


for path in sys.argv[1:]:
    comps = parse(path)
    print(f'=== {path}: {len(comps)} computations')
    tot = defaultdict(int)
    for nm, body in comps.items():
        if nm.startswith('fused') or 'fusion' in nm and nm.startswith('%fused'):
            continue
        for ln in body:
            op = opcode(ln)
            if op == 'while':
                mb = re.search(r'body=%?([\w.\-]+)', ln)
                mc = re.search(r'condition=%?([\w.\-]+)', ln)
                b = comps.get(mb.group(1), []) if mb else []
                c = comps.get(mc.group(1), []) if mc else []
                cb, cc = count(b), count(c)
                nb = sum(cb.values())
                nested = sum(1 for x in b if opcode(x) in ('while', 'conditional'))
                print(f'  while in {nm[:40]:40s}  body {mb.group(1)[:34] if mb else "?":34s} launches/iter {nb:5d} (nested while/cond: {nested})  cond launches {sum(cc.values())}   top ops: ' + ', '.join(f'{k}:{v}' for k, v in sorted(cb.items(), key=lambda kv: -kv[1])[:5]))
    for nm, body in comps.items():
        for k, v in count(body).items():
            tot[k] += v
    print('  total launch-type instructions (all computations, fusions counted once):', sum(v for k, v in tot.items() if k == 'fusion'), 'fusions;', sum(tot.values()), 'ops')
