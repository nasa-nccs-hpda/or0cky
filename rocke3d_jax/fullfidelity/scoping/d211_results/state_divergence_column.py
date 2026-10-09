import sys,numpy as np
sys.path.insert(0,'.')
import atm_day_report as RP, clouds_condse_io as cio
S=sys.argv[1]; i,j=int(sys.argv[2]),int(sys.argv[3])
for k in [0,1,2,10,20,28,29]:
    it=33312+k
    r=RP.load_real_e(cio.FF_DEFAULT,'nov26_day',it); o=np.load(f'{S}/run_base/ours_d193/step_{it}.npz')
    out=[]
    for f in ('T','Q'):
        a=np.asarray(r[f]);b=o[f]
        d=np.abs(a-b)
        out.append((f,'col maxrel',float((d[i,j]/np.abs(a[i,j]).max()).max()),'global maxrel',float((d/np.abs(a).max(axis=(0,1),keepdims=True)).max())))
    print(k,out)
