import sys,numpy as np
sys.path.insert(0,'.')
import clouds_condse_io as cio
ff=cio.FF_DEFAULT; S=sys.argv[1]; i=int(sys.argv[2]); j=int(sys.argv[3])
for k in range(27,34):
    ri=cio.read_cse(f"{ff}/nov26_day/ffc_cse_in_{33312+k}.bin"); ro=cio.read_cse(f"{ff}/nov26_day/ffc_cse_out_{33312+k}.bin")
    PK=np.asarray(ri['PK'])[0,i,j]
    Tin=np.asarray(ri['T'])[i,j,0]; Tout=np.asarray(ro['T'])[i,j,0]
    o=np.load(f'{S}/run_base/ours_d193/step_{33312+k}.npz')
    sn=np.asarray(ri['SNOAGE'])[0,i,j]; sno=np.asarray(ro['SNOAGE'])[0,i,j]
    print(k,'PK',PK,'rec Tin-t',Tin*PK-273.16,'rec Tout',Tout*PK-273.16,'ours end-of-step T0 (units?)',o['T'][i,j,0],'rec snow',sn,sno,'PREC',np.asarray(ro['PREC'])[i,j],'SVLHX0 in',np.asarray(ri['SVLHX'])[0,i,j],'out',np.asarray(ro['SVLHX'])[0,i,j],'foc/fland',np.asarray(ri['FOCEAN'])[i,j],np.asarray(ri['FLAND'])[i,j],'RSI',np.asarray(ri['RSI'])[i,j])
