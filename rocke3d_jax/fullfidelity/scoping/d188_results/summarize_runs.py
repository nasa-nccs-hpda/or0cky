import json,collections,sys
for d in ('nov26','dec01','jan01'):
    r=json.load(open(f'{sys.argv[1]}/run_{d}.json'))
    c1=r['c1']; c2=r['c2']
    print(d, r['slots'])
    print(' C1', dict(collections.Counter(v[0] for v in c1.values())), 'n', len(c1), ' non-A:', [k for k,v in c1.items() if v[0]!='A'][:10])
    c2f={k:v for k,v in c2.items()}
    print(' C2', dict(collections.Counter(v[0] for v in c2f.values())))
    print(' first-call units', {k:round(v,1) for k,v in r['first_call_unit_times_s'].items()}, 'stage', [round(x,3) for x in r['stage_times_s']], 'numpy', [round(x,3) for x in r['numpy_stage_times_s']])
    print(' host', {k:round(r[k],3) for k in r if k.startswith('t_')})
    g=collections.defaultdict(list)
    for k,v in c2.items():
        p=k.split('.')
        key='.'.join(p[:-1]) if p[0].startswith('ns') else p[0]
        g[key].append((v[2],v[0],p[-1]))
    for key,l in sorted(g.items()):
        w=max(l); print(f'   C2 {key:26s} n={len(l):2d} worst rel {w[0]:.2e} {w[1]} ({w[2]}) {dict(collections.Counter(x[1] for x in l))}')
