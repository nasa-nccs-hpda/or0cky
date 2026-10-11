import sys; sys.path.insert(0,'.')
import numpy as np, clouds_condse_io as cio
D='/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day2'
keys='DDM1 DDML DDMS EPREC PREC PRECSS QDN1 TDN1 QGAVG TGVAVG USAVG VSAVG RSI FEARTH FLAKE FLAND FLICE FOCEAN'.split()
res={k:[0,0,0.0] for k in keys}   # steps equal, steps compared, max abs diff
prev=None; nin=0
for it in range(33312,33312+108):
    ci=cio.read_cse(f'{D}/ffc_cse_in_{it}.bin')
    if prev is not None:
        for k in keys:
            if k in prev and k in ci:
                a,b=np.asarray(prev[k],float),np.asarray(ci[k],float)
                res[k][1]+=1; d=float(np.abs(a-b).max()); res[k][2]=max(res[k][2],d); res[k][0]+= d==0
    prev=cio.read_cse(f'{D}/ffc_cse_out_{it}.bin')
    if it==33312: print('co keys has', [k for k in keys if k in prev])
for k,v in res.items(): print(f'{k:8s} equal {v[0]:3d}/{v[1]:3d} maxdiff {v[2]:.3e}')
