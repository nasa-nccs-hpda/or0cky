"""D211: |dSNOAGE| (ours - record CONDSE exit) at step K restricted to the surface types that exist in the cell (type1 RSI>0, type2 FLICE>0, type3 FEARTH>0).  python present_stats.py OUTDIR K"""
import sys, numpy as np
sys.path.insert(0,'.')
import clouds_condse_io as cio
out=sys.argv[1]; k=int(sys.argv[2]); ff=cio.FF_DEFAULT
c=np.load(f'{out}/carry/carry_{k:02d}.npz'); ro=np.asarray(cio.read_cse(f"{ff}/nov26_day/ffc_cse_out_{33312+k}.bin")['SNOAGE']); ri=cio.read_cse(f"{ff}/nov26_day/ffc_cse_in_{33312+k}.bin")
pres=np.stack([np.asarray(ri['RSI'])>0,np.asarray(ri['FLICE'])>0,np.asarray(ri['FEARTH'])>0])
d=np.abs(c['SNOAGE']-ro)
for t,nm in enumerate(['ocean ice','land ice','land']):
    dd=d[t][pres[t]]
    print(nm,'cells present',int(pres[t].sum()),'max|d|',float(dd.max()),'n>1e-3',int((dd>1e-3).sum()),'n>0.1',int((dd>0.1).sum()),'n>1',int((dd>1).sum()),'| absent cells max|d|',float(d[t][~pres[t]].max()))
    i=np.unravel_index(np.argmax(np.where(pres[t],d[t],0)),d[t].shape); print('   argmax present',i,'ours',float(c['SNOAGE'][t][i]),'rec',float(ro[t][i]))
