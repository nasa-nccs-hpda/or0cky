import sys,numpy as np
sys.path.insert(0,'.')
import clouds_condse_io as cio
ff=cio.FF_DEFAULT
for k in [0,1,5,20,30,47]:
    it=33312+k
    i=cio.read_cse(f"{ff}/nov26_day/ffc_cse_in_{it}.bin"); o=cio.read_cse(f"{ff}/nov26_day/ffc_cse_out_{it}.bin")
    si=np.asarray(i['SNOAGE']); so=np.asarray(o['SNOAGE']); pr=np.asarray(o['PREC'])
    print(k, si.shape, pr.shape, 'PREC_in', np.asarray(i['PREC']).max())
    pred=si*np.exp(-pr[None])
    ch=(so!=si)
    # cells with change
    print(' changed cells',ch.any(0).sum(),' of which bitwise exp(-PREC):',(so==pred)[:, ch.any(0)].all(0).sum() if False else None)
    m=ch.any(0)
    d=np.abs(so-pred)[:,m]
    print(' maxabs so-pred over changed', d.max(), 'n nonbitwise', (d>0).any(0).sum(), 'PREC>0 & unchanged & si>0:', ((pr>0)&(~m)&(si[0]>0)).sum(), 'changed & PREC==0', (m&(pr<=0)).sum())
