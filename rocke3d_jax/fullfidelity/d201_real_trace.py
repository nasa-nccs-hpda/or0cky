"""D201: the REAL record (ffg_<it>.bin) of land rows 110-112 ((40,21),(41,21),(42,21) 1-based record coords; row 112 = port land index 112 = grid (41,20) 0-based)
for the nov26 day, both substeps.  Output scoping/d201_results/real_rows.json."""
import json, numpy as np, ghy_compare as GC
FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
cols = dict(tbcs=245, tsns=246, ashg=247, alhg=248, aevap=249, ffnit=289, pr=143, ts_in=152, qs_in=153, pres=154, rho=155, ch=156, qm1=157, vs=158, srheat=149, trheat=150, tprime=161, qprime=162)
out = {}
for k in range(54):
    it = 33312 + k
    g = GC.load(f'{FF}/ffg_{it}.bin'); ng = len(g) // 2
    for row in (110, 111, 112):
        for ns in (0, 1):
            r = g[ns * ng + row]
            assert (int(r[0]), int(r[1])) == {110: (40, 21), 111: (41, 21), 112: (42, 21)}[row]
            out[f'{k}/{row}/{ns+1}'] = {n: float(r[c]) for n, c in cols.items()}
json.dump(out, open('scoping/d201_results/real_rows.json', 'w'), indent=0)
for k in range(0, 40):
    a = out[f'{k}/112/1']; b = out[f'{k}/112/2']
    print(k, 'tsns %.3f %.3f' % (a['tsns'], b['tsns']), 'aevap %.2e %.2e' % (a['aevap'], b['aevap']), 'nit', int(a['ffnit']), int(b['ffnit']), 'pr %.2e' % a['pr'], 'ts %.2f qs %.5f' % (a['ts_in'], a['qs_in']))
