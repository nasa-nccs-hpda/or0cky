"""D138: one full ocean step (ported ODIFF AND computed OPFIL2 coefficients: neither recorded) from the real entry.
usage: python ocean_step_opcoef_compare.py <date> [itime-index ...]"""
import sys, os, tempfile
import numpy as np
import ocean_chain_io as C
import ocean_step_odiff as OD
from ocean_opfil2_coeffs import calc_opfil2_coeffs
from ocean_step_compare import ctx_for, fx_of
from ocean_step_chain_compare import errs

if __name__ == '__main__':
    date = sys.argv[1]; idx = [int(x) for x in sys.argv[2:]] or [0]
    d = C.FF_DEFAULT + '/' + date; ctx = ctx_for(d)
    v, _ = calc_opfil2_coeffs(ctx['lmu'])
    # the unassigned SMOOTH entries are never read by OPFIL2 (n >= NMIN only); take the recorded file's values there for the byte layout
    ref = np.fromfile(d + '/ffo_opcoef.bin', dtype='>f8').astype(np.float64)
    nsm = int(v[4]); sm = slice(5, 5 + nsm)
    v[sm] = np.where(v[sm] == 0, ref[sm], v[sm])
    tmp = tempfile.mkdtemp() + '/opcoef_computed.bin'
    v.astype('>f8').tofile(tmp)
    ctx['opcoef'] = tmp
    steps = C.list_steps(d)
    for k in idx:
        it = steps[k]; sn = C.load_step(d, it); fx = fx_of(sn); fx['itime'] = it
        out = OD.ocean_step_odiff(sn[0], fx, ctx)
        print(date, it, 'ODIFF' if it % 6 == 0 else '', ' '.join(f'{a}:{b:.1e}' for a, b in errs(out, sn[14]).items() if a in ('uo', 'vo', 'vonp', 'g0m', 's0m', 'mo', 'uod', 'vod')), flush=True)
