"""D211: sensitivity of the rain/snow decision and SNOAGE at one cell to a relative perturbation of the REAL CONDSE entry T and Q of that column (NumPy port, libimf backend, row j)."""
import sys, copy, numpy as np
sys.path.insert(0,'.')
import clouds_condse_ff as cf, clouds_condse_io as cio
DATE='nov26_day'; k=int(sys.argv[1]); ci=int(sys.argv[2]); cj=int(sys.argv[3])
inp,ref=cio.load_step(DATE,33312+k)
cfg=cf.make_cfg(DATE); G=cfg['geom']
cols=[(i,cj) for i in range(int(G['IMAXJ'][cj]))]
cf.set_backend('imf')
rng=np.random.default_rng(1)
print('record: PREC',ref['PREC'][ci,cj],'EPREC',ref['EPREC'][ci,cj],'SNOAGE',ref['SNOAGE'][:,ci,cj],'entry',inp['SNOAGE'][:,ci,cj])
for eps in [0,1e-12,1e-9,1e-7,1e-6,1e-5,1e-4,1e-3]:
  res=[]
  for seed in range(4 if eps else 1):
    rng=np.random.default_rng(seed)
    p=copy.deepcopy(inp)
    p['T'][ci,cj,:]*= (1+eps*rng.standard_normal(p['T'].shape[2]))
    p['Q'][ci,cj,:]*= (1+eps*rng.standard_normal(p['Q'].shape[2]))
    X,cnt=cf.condse_step(p,cfg,cols=cols,with_momentum=False)
    res.append((float(X['PREC'][ci,cj]),float(X['EPREC'][ci,cj]),float(X['SNOAGE'][0,ci,cj])))
  print('eps',eps,res,flush=True)
