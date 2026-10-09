import numpy as np, sys, json
A='%s/d208/out/ours_d193'%sys.argv[1]; B='%s/d209/run_c/ours_d193'%sys.argv[1]
F=('T','U','V','Q','P','QCL','QCI'); out=[]
for it in range(33312,33366):
    a=np.load(f'{A}/step_{it}.npz'); b=np.load(f'{B}/step_{it}.npz'); row=dict(k=it-33312,it=it)
    for f in F:
        x,y=a[f],b[f]; d=np.abs(x-y)
        row[f]=dict(n_diff=int((x!=y).sum()),max_abs=float(d.max()),rms=float(np.sqrt(np.mean((x-y)**2))),scale=float(np.abs(x).max()),finite=bool(np.isfinite(y).all()))
    out.append(row)
json.dump(out,open(sys.argv[2],'w'),indent=1)
for r in out:
    if r['k']>=46 or r['k']==0: print(r['k'],{f:(r[f]['n_diff'],'%.1e'%r[f]['max_abs'],'%.1e'%r[f]['rms']) for f in F}, all(r[f]['finite'] for f in F))
print('first step with any diff:',next((r['k'] for r in out if any(r[f]['n_diff'] for f in F)),None))
