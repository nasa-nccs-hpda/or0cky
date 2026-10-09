import clouds_jax_env, glob, os, json, numpy as np
FF="/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day"
its=sorted(int(os.path.basename(p)[4:-4]) for p in glob.glob(f'{FF}/ffp_[0-9]*.bin'))
def keys(it):
    p=np.fromfile(f'{FF}/ffp_{it}.bin','>f8').reshape(-1,154); h=len(p)//2
    return set(map(tuple,p[:h,:3].astype(int)))   # first substep block
prev=keys(its[0]); out={}
ev=[]
for it in its[1:]:
    cur=keys(it)
    for t in (1,2,3,4):
        new=[k for k in cur-prev if k[2]==t]; van=[k for k in prev-cur if k[2]==t]
        if new or van: ev.append(dict(step=it-its[0],it=it,type=t,new=len(new),vanished=len(van),new_cells=[(int(k[0]),int(k[1])) for k in new][:8]))
    prev=cur
tot={t:[sum(e['new'] for e in ev if e['type']==t),sum(e['vanished'] for e in ev if e['type']==t)] for t in (1,2,3,4)}
print(json.dumps(tot)); json.dump(dict(totals_new_vanished_by_type=tot,events=ev),open('/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity/scoping/d209_results/tile_appearance_census.json','w'),indent=1)
for e in ev: print(e)
