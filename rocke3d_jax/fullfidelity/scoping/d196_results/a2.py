import sys; sys.path.insert(0,'.')
import numpy as np, clouds_condse_io as cio, pbl_compare as PC
FF='/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
def rows(it):
    p=PC.load(f'{FF}/ffp_{it}.bin'); return {(int(r[0]),int(r[1]),int(r[2])):r for r in p}
def cse(it): return cio.read_cse(f'{FF}/ffc_cse_in_{it}.bin')
R47,R48=rows(33359),rows(33360)
print('ncols',len(next(iter(R48.values()))), 'n rows',len(R47),len(R48))
new=[k for k in R48 if k[2]==1 and k not in R47]
gone=[k for k in R47 if k[2]==1 and k not in R48]
print('new type1',len(new),'gone type1',len(gone))
c47,c48=cse(33359),cse(33360)
cells=sorted(set((k[0],k[1]) for k in new))
print(cells[:5], sorted(set(j for i,j in cells)))
for nm in ('FOCEAN','FLAKE','FEARTH','FLICE','RSI'):
    print(nm,'changed cells all:',int((c47[nm]!=c48[nm]).sum()))
# 1-based assumption
for (i,j) in cells[:60]:
    ii,jj=i-1,j-1
    print((i,j),'foc %.3g flake47 %.6g flake48 %.6g rsi47 %.10g rsi48 %.10g fearth47 %.4g fearth48 %.4g'%(c47['FOCEAN'][ii,jj],c47['FLAKE'][ii,jj],c48['FLAKE'][ii,jj],c47['RSI'][ii,jj],c48['RSI'][ii,jj],c47['FEARTH'][ii,jj],c48['FEARTH'][ii,jj]), 'tile fr', [R48[(i,j,t)][3:6] if (i,j,t) in R48 else None for t in (1,2)][:1])
