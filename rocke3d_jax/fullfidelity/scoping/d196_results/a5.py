import sys; sys.path.insert(0,'.')
import numpy as np, clouds_condse_io as cio
FF='/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
SP='/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d196'
def cse(it): return cio.read_cse(f'{FF}/ffc_cse_in_{it}.bin')
c0=cse(33312); foc=c0['FOCEAN']; fl=c0['FLAKE']; ice=foc>0; lake=(foc==0)&(fl>0)
for k in list(range(0,48,4))+[46,47]:
    o=np.load(f'{SP}/c1/nov26_day_step{k}.npz')['surf/ice/rsi']; r=cse(33312+k+1)['RSI']; d=np.abs(o-r)
    io=np.unravel_index(np.argmax(np.where(ice,d,0)),d.shape); il=np.unravel_index(np.argmax(np.where(lake,d,0)),d.shape)
    print(k,'ocean max %.3e at %s (n>1e-6: %d, n>1e-4: %d)  lake max %.3e at %s (n>1e-6: %d)'%(d[ice].max(),io,(d[ice]>1e-6).sum(),(d[ice]>1e-4).sum(),d[lake].max(),il,(d[lake]>1e-6).sum()))
    pure=(r==1)&(o==1); 
print('=====')
c47,c48,c49=cse(33359),cse(33360),cse(33361)
o47=np.load(f'{SP}/c1/nov26_day_step47.npz')['surf/ice/rsi']; o48=np.load(f'{SP}/c1/nov26_day_step48.npz')['surf/ice/rsi']
ml=lambda a:a
dl=np.abs(o48-c49['RSI']); 
fch=(c48['FLAKE']!=c47['FLAKE'])&lake
print('lake cells flake changed',fch.sum(),' lake cells total',lake.sum())
print('end of step 48 lake |ours-record(49)|>1e-3 :',(dl[lake]>1e-3).sum(),' of which flake-changed',((dl>1e-3)&lake&fch).sum(),' max',dl[lake].max())
dl47=np.abs(o47-c48['RSI']);print('end of 47 vs record 48: lake >1e-3:',(dl47[lake]>1e-3).sum(),'; at end of 46 vs record 47: ', (np.abs(np.load(f'{SP}/c1/nov26_day_step46.npz')['surf/ice/rsi']-c47['RSI'])[lake]>1e-3).sum())
# crunch cells: rsi48==1 after rsi47<1 or flake decrease
crunch=lake&fch&(c48['RSI']==1)&(c47['RSI']<1); print('lake cells rsi->1.0 (crunch/shrink) at boundary',crunch.sum(),'flake down',(lake&(c48['FLAKE']<c47['FLAKE'])).sum())
i,j=np.unravel_index(np.argmax(np.where(lake,dl47,0)),dl47.shape); print('max cell',i,j,'rsi47 rec',c47['RSI'][i,j],'rsi48 rec',c48['RSI'][i,j],'ours47',o47[i,j],'flake',c47['FLAKE'][i,j],c48['FLAKE'][i,j])
# does 54-cell mismatch exist because ours rsi==1 exactly in these: yes
# ocean cells at boundary: ours-vs-record diff change 47->48
d47=np.abs(o47-c48['RSI'])[ice]; d46=np.abs(np.load(f'{SP}/c1/nov26_day_step46.npz')['surf/ice/rsi']-c47['RSI'])[ice]
print('ocean: max diff end46-vs-rec47 %.3e ; end47-vs-rec48 %.3e'%(d46.max(),d47.max()))
