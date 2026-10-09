"""D211: real-Fortran CONDSE check on one latitude row of a step of nov26_day (record entry -> our port -> record exit)."""
import sys, numpy as np
sys.path.insert(0,'.')
import clouds_condse_ff as cf, clouds_condse_io as cio, clouds_mstcnv_ff as mc
DATE='nov26_day'; k=int(sys.argv[1]); j=int(sys.argv[2]); imf=int(sys.argv[3]) if len(sys.argv)>3 else 1
it=33312+k
inp,ref=cio.load_step(DATE,it)
cfg=cf.make_cfg(DATE)
G=cfg['geom']
cols=[(i,j) for i in range(int(G['IMAXJ'][j]))]
cf.set_backend('imf' if imf else 'numpy')
X,cnt=cf.condse_step(inp,cfg,cols=cols,with_momentum=False)
for n in ['SNOAGE','PREC','EPREC','PRECSS','T','Q']:
    a=np.asarray(X[n]);b=np.asarray(ref[n])
    ax=[q for q in range(a.ndim) if a.shape[q]==46][0] if n!='T' and n!='Q' else 1
    a=np.take(a,j,axis=ax);b=np.take(b,j,axis=ax)
    d=np.abs(a-b)
    print(n,'max abs diff',d.max(),'n!=', (a!=b).sum())

a=np.asarray(X['SNOAGE'])[:,32,j];b=np.asarray(ref['SNOAGE'])[:,32,j]
print('cell(32,%d) SNOAGE port'%j,a,'rec',b,'PREC port',X['PREC'][32,j],'rec',ref['PREC'][32,j],'EPREC',X['EPREC'][32,j],ref['EPREC'][32,j], 'rec entry SNOAGE', np.asarray(inp['SNOAGE'])[:,32,j])
