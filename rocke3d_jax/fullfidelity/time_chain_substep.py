"""Warm per-stage timing of one full NIsurf substep of the Track B surface chain (D23). Usage: python time_chain_substep.py [ff_data_dir/nov26 itime]"""
import sys, time, numpy as np
sys.path.insert(0,".")
import chain_two_substeps as T2, land_chain as LC, chain_aggregate_aturb as C, surface_chain_ff as CH
import pbl_compare as PC, surface_tile_ff as S, landice_tile_ff as LI, tile_aggregate_ff as TA, ghy_compare as GC
T={}
def wrap(mod,name,key):
    f=getattr(mod,name)
    def g(*a,**k):
        t=time.perf_counter(); r=f(*a,**k); T[key]=T.get(key,0)+time.perf_counter()-t; return r
    setattr(mod,name,g)
wrap(CH,"run_chain","ocean+ice PBL+tile"); wrap(C,"landice_chain","land-ice PBL+tile")
wrap(LC,"land_substep","land PBL+GHY"); wrap(T2,"run_aturb","ATURB+UV")
dd=sys.argv[1] if len(sys.argv)>1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26"; it=int(sys.argv[2]) if len(sys.argv)>2 else 33312
p=PC.load(f"{dd}/ffp_{it}.bin"); t=S.load(f"{dd}/ffs_{it}.bin"); l=LI.load(f"{dd}/ffl_{it}.bin"); fft=TA.load(f"{dd}/fft_{it}.bin"); g=GC.load(f"{dd}/ffg_{it}.bin")
n=len(p)//2; nt=len(t)//2; nl=len(l)//2; B=len(fft)//2; ng=len(g)//2
pa=p[:n]; a12,a3,a4=pa[pa[:,2]<=2],pa[pa[:,2]==3],pa[pa[:,2]==4]
blk=fft[:B]; ft,patch,_=TA.unpack(blk)
lut=C._cell_lookup(blk); g1=g[:ng]; idx=lut[g1[:,0].astype(int),g1[:,1].astype(int)]
trup=LC.infer_trup(g1,patch["dth1"][idx,3],900.0)
atm,dt=T2.load_atm(f"{dd}/ffa_{it}_c1_in.bin")
land=dict(p4=a4,g=g1,trup=trup)
for rep in range(3):
    T.clear(); t0=time.perf_counter()
    T2.substep(a12,t[:nt],a3,l[:nl],blk,atm,dt,blk,land)
    tot=time.perf_counter()-t0
    print(f"run {rep}: total {tot:.2f}s  "+"  ".join(f"{k} {v:.2f}" for k,v in T.items())+f"  other {tot-sum(T.values()):.2f}",flush=True)
print("cells:",len(blk),"tiles ocean+ice",nt,"landice",nl,"land",len(a4))
