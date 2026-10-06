"""D137: full-step exit error with the ported ODIFF. usage: python ocean_step_odiff_compare.py <date> [nsteps]
Per step from the real tag-0 entry (no recorded ODIFF is supplied), error at exit (tag 14); then a free-running chain."""
import sys
import time
import ocean_chain_io as C
import ocean_step as O
import ocean_step_odiff as OD
from ocean_step_compare import ctx_for, fx_of
from ocean_step_chain_compare import errs

if __name__ == '__main__':
    date = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    d = C.FF_DEFAULT + '/' + date; ctx = ctx_for(d)
    steps = C.list_steps(d)[:n]
    print('== per step from real entry, ported ODIFF (no recorded ODIFF) ==')
    for it in steps:
        sn = C.load_step(d, it); fx = fx_of(sn); fx['itime'] = it
        t0 = time.time(); out = OD.ocean_step_odiff(sn[0], fx, ctx)
        e = errs(out, sn[14])
        print(it, 'ODIFF' if it % 6 == 0 else '     ', ' '.join(f'{k}:{v:.1e}' for k, v in e.items() if k in ('uo', 'vo', 'vonp', 'g0m', 's0m', 'mo', 'uod', 'vod')), f'[{time.time()-t0:.0f}s]', flush=True)
    print('== free-running chain ==')
    s = C.load_step(d, steps[0])[0]
    for it in steps:
        sn = C.load_step(d, it); fx = fx_of(sn); fx['itime'] = it
        s = OD.ocean_step_odiff(s, fx, ctx)
        e = errs(s, sn[14])
        print(it, 'ODIFF' if it % 6 == 0 else '     ', ' '.join(f'{k}:{v:.1e}' for k, v in e.items() if k in ('uo', 'vo', 'vonp', 'g0m', 's0m', 'mo', 'uod', 'vod')), flush=True)
