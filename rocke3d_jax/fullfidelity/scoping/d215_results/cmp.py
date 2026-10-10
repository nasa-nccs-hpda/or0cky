import netCDF4 as nc, numpy as np, json, sys
runs=["run_ens1","run_ens2","run_ens3","run_pris1","run_pris2","run_pris3","../real54"]
D={r:nc.Dataset(r+"/fort.2.nc") for r in runs}
ref=D["run_ens1"]
out={}
for r in runs:
    d=D[r]; neq=[];nan=0;tot=0
    for k,v in ref.variables.items():
        if k not in d.variables: neq.append(k+":missing"); continue
        a=np.asarray(v[:]); b=np.asarray(d[k][:])
        if a.dtype.kind in "fiu":
            tot+=1
            if a.dtype.kind=="f":
                nan+=int((~np.isfinite(b)).sum())
            if a.shape!=b.shape or not np.array_equal(a,b,equal_nan=a.dtype.kind=="f"): neq.append(k)
    out[r]=dict(n_numeric_vars=tot,nonfinite=nan,vars_differing_from_ens1=neq[:30],n_differ=len(neq))
    for k in ("itime","itimee"):
        if k in d.variables: out[r][k]=int(np.asarray(d[k][:]).ravel()[0])
# scalar: area-weighted mean of tsurf-like? list candidate names
print([k for k in ref.variables if k.lower() in ("tsurf","tgrnd","gtemp","tocean","p","t","q","u","v","axyp")][:20])
json.dump(out,open("../../../../../../../../../dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity/scoping/d215_results/state_compare.json","w") if False else open("state_compare.json","w"),indent=1)
print(json.dumps(out,indent=1))
