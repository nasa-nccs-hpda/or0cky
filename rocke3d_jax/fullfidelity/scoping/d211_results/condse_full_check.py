"""D211: whole-population CONDSE (NumPy port, libimf backend) from the REAL entry record of step k of nov26_day vs the REAL exit record: SNOAGE PREC EPREC PRECSS bitwise?  python condse_full_check.py K..."""
import sys, time, numpy as np
sys.path.insert(0,'.')
import clouds_condse_ff as cf, clouds_condse_io as cio
cfg=cf.make_cfg('nov26_day'); cf.set_backend('imf')
for k in map(int, sys.argv[1:]):
    t0=time.time(); inp,ref=cio.load_step('nov26_day',33312+k)
    X,cnt=cf.condse_step(inp,cfg,with_momentum=False)
    m=np.asarray(X['PREC'])!=np.asarray(ref['PREC']); print('  PREC mismatch cells',[(int(a),int(b),float(np.asarray(X['PREC'])[a,b]),float(np.asarray(ref['PREC'])[a,b])) for a,b in zip(*np.nonzero(m))][:5])
    print('step',k,{n:int((np.asarray(X[n])!=np.asarray(ref[n])).sum()) for n in ('SNOAGE','PREC','EPREC','PRECSS','T','Q')},'snow cells(rec)',int((np.asarray(ref['EPREC'])<0).sum()),'sec',round(time.time()-t0),flush=True)
