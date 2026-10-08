"""D201: ON vs OFF column of land cell 112 (grid 41,20) per step (captures of d201_day.py) -> scoping/d201_results/cell112.md|json"""
import numpy as np, json
S='/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d201'
rows=[]
out=['| step | #fields differing in cell 112 | first/largest differing field | tsns OFF | tsns ON | tsns real | aevap OFF | aevap ON | aevap real | ws OFF | ws ON | ch OFF | ch ON | T(41,20,lev0) OFF | ON | rho ON |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
real=json.load(open('scoping/d201_results/real_rows.json'))
for k in range(41):
    a=np.load(f'{S}/on/cap/land_{k}.npz'); b=np.load(f'{S}/off/cap/land_{k}.npz')
    dif=[(x,float(np.nanmax(np.abs(a[x][112]-b[x][112])))) for x in b.files if x.startswith('land/') and x in a.files and np.any(a[x][112]!=b[x][112])]
    top=max(dif,key=lambda t:t[1]) if dif else ('-',0)
    r=real[f'{k}/112/2']
    rows.append(dict(k=k,ndiff=len(dif),top=top,tsns_off=float(b['land/ghy/tsns'][112]),tsns_on=float(a['land/ghy/tsns'][112]),tsns_real=r['tsns'],ae_off=float(b['land/ghy/aevap'][112]),ae_on=float(a['land/ghy/aevap'][112]),ae_real=r['aevap'],nit_real=[real[f'{k}/112/1']['ffnit'],r['ffnit']]))
    out.append(f"| {k} | {len(dif)} | {top[0].replace('land/','')} {top[1]:.2e} | {b['land/ghy/tsns'][112]:.4f} | {a['land/ghy/tsns'][112]:.4f} | {r['tsns']:.4f} | {b['land/ghy/aevap'][112]:.3e} | {a['land/ghy/aevap'][112]:.3e} | {r['aevap']:.3e} | {b['land/pbl/ws'][112]:.4f} | {a['land/pbl/ws'][112]:.4f} | {b['land/pbl/ch'][112]:.5f} | {a['land/pbl/ch'][112]:.5f} | {b['filter/T'][0]:.3f} | {a['filter/T'][0]:.3f} | {a['land/rho'][112]:.4f} |")
open('scoping/d201_results/cell112.md','w').write('\n'.join(out)+'\n'); json.dump(rows,open('scoping/d201_results/cell112.json','w'),indent=1)
print('\n'.join(out[:14])); print('...'); print('\n'.join(out[30:]))
