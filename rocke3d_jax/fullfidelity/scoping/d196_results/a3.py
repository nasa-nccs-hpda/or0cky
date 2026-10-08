import sys; sys.path.insert(0,'.')
import numpy as np, clouds_condse_io as cio, pbl_compare as PC
FF='/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
def cse(it): return cio.read_cse(f'{FF}/ffc_cse_in_{it}.bin')
c47,c48=cse(33359),cse(33360)
foc=c47['FOCEAN']; fl47,fl48=c47['FLAKE'],c48['FLAKE']; r47,r48=c47['RSI'],c48['RSI']
chg=r47!=r48
print('RSI changed',chg.sum(),'of which focean>0:',(chg&(foc>0)).sum(),' flake>0:',(chg&(fl48>0)).sum())
lake=(foc==0)&(fl47>0)
print('lake cells',lake.sum(),'rsi47 stats among lake cells: ==1',(lake&(r47==1)).sum(),'0<r<1',(lake&(r47>0)&(r47<1)).sum(),'==0',(lake&(r47==0)).sum())
pred=r47*fl47/fl48
m=chg&lake
print('changed lake cells',m.sum(),'max|rsi48-rsi47*fl47/fl48| (changed, non-crunch)', np.abs(pred-r48)[m&(r47*fl47<=fl48)].max(), 'count nonCrunch',(m&(r47*fl47<=fl48)).sum())
# bitwise-ish
d=np.abs(pred-r48)[m]; print('rel diff', (d/np.maximum(r48[m],1e-300)).max())
# flake increased/decreased
print('flake up',(m&(fl48>fl47)).sum(),'down',(m&(fl48<fl47)).sum())
# pure-ice at 47 (rsi==1) lake cells where flake grows
pure=lake&(r47==1)
print('pure-ice lake cells at 47',pure.sum(),'with flake growth',(pure&(fl48>fl47)).sum(),'rsi48<1',(pure&(r48<1)).sum())
# the 54 cells == pure & rsi48<1 ?
R48={ (int(r[0]),int(r[1]),int(r[2])) for r in PC.load(f'{FF}/ffp_33360.bin')}
R47={ (int(r[0]),int(r[1]),int(r[2])) for r in PC.load(f'{FF}/ffp_33359.bin')}
new=sorted(k[:2] for k in R48 if k[2]==1 and k not in R47)
mask=np.zeros_like(pure); 
for i,j in new: mask[i-1,j-1]=True
print('54 cells == pure&rsi48<1 :', (mask==(pure&(r48<1))).all(), mask.sum(), (pure&(r48<1)).sum())
# type1 fraction in other tile cols? print header cols of a type1 row
import pickle
p=PC.load(f'{FF}/ffp_33360.bin')
print('----')
c46=cse(33358); c49=cse(33361); c50=cse(33362)
cells=np.argwhere(mask)
# prediction on the 54
e=np.abs(pred-r48)[mask]; print('54 cells: max abs err of rsi48 vs rsi47*fl47/fl48 = %.3e ; max (1-rsi48) = %.3e min = %.3e'%(e.max(),(1-r48[mask]).max(),(1-r48[mask]).min()))
print('54 cells: rsi at 46,47 all ==1:',(c46['RSI'][mask]==1).all(),(r47[mask]==1).all(), ' flake46==flake47:',(c46['FLAKE'][mask]==fl47[mask]).all())
for nm,c in (('49',c49),('50',c50)):
    n=(c['RSI'][mask]<1).sum(); print('step',nm,'cells (of 54) with rsi<1:',n)
# normal-step lake rsi change count 46->47 among lake cells
l=lake
print('lake cells whose RSI changed 46->47:',((c46['RSI']!=r47)&l).sum(),' 47->48:',((r47!=r48)&l).sum(),' 48->49:',((r48!=c49['RSI'])&l).sum())
print('lake cells FLAKE changed 46->47:',((c46['FLAKE']!=fl47)&l).sum(),' 47->48:',((fl47!=fl48)&l).sum(),' 48->49:',((fl48!=c49['FLAKE'])&l).sum())
print('ocean cells RSI changed 46->47 / 47->48:',((c46['RSI']!=r47)&(foc>0)).sum(),((r47!=r48)&(foc>0)).sum())
for (i,j) in [(54,41),(64,39),(71,39)]:
    print((i,j),[ '%.8f'%cse(it)['RSI'][i-1,j-1] for it in (33359,33360,33361,33362,33363,33364,33365)])
# flake growth of the 54 vs other lake cells
print('flake growth/flake of 54 cells: min %.2e max %.2e'%(((fl48-fl47)/fl47)[mask].min(),((fl48-fl47)/fl47)[mask].max()))
print('lake cells with flake growth but rsi47<1 : ', (lake&(fl48>fl47)&(r47<1)).sum())
print('====')
a=(1-r48)[mask]; b=(1-fl47/fl48)[mask]
print('(1-rsi48) vs (1-fl47/fl48): max rel diff %.3e, median rel %.3e, max abs %.3e'%(np.abs(a-b).max()/1, np.median(np.abs(a-b)/b), np.abs(a-b).max()))
print('rel diff >1e-3 count',(np.abs(a-b)/b>1e-3).sum(),'of 54')
tf=(fl48*(1-r48))[mask]; print('tile fraction flake48*(1-rsi48): min %.3e max %.3e'%(tf.min(),tf.max()))
