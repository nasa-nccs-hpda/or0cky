"""D137: ODIFF port vs the ffo_state tag 12 (pre_odiff) -> 13 (post_odiff) dumps. usage: python odiff_compare.py [date ...]"""
import sys
import numpy as np
import ocean_chain_io as C
import ocean_step as O
import ocean_odiff as D
from ocean_step_compare import ctx_for


def dh_of(sn12, ctx):
    return O.run_odhorz0(sn12, ctx)['dh3d']


def compare(date, itime, ctx, **kw):
    d = C.FF_DEFAULT + '/' + date
    sn = C.load_step(d, itime)
    if 12 not in sn or 13 not in sn:
        return None
    a, b = sn[12], sn[13]
    dh = dh_of(a, ctx)
    uo, vo, vonp = D.odiff(a['mo'], a['uo'], a['vo'], dh, ctx['lmu'], ctx['lmv'], **kw)
    out = {}
    for k, g, r in (('uo', uo, b['uo']), ('vo', vo, b['vo']), ('vonp', vonp, b['vonp'])):
        dd = np.abs(g - r)
        out[k] = (float(dd.max()), float(dd.max() / max(np.abs(r).max(), 1e-300)), int((g != r).sum()), g.size)
    out['fired'] = bool(itime % 6 == 0)
    out['moved'] = float(np.abs(a['uo'] - b['uo']).max())
    return out


if __name__ == '__main__':
    for date in (sys.argv[1:] or ['jan01', 'nov26', 'dec01']):
        d = C.FF_DEFAULT + '/' + date
        ctx = ctx_for(d)
        for it in C.list_steps(d)[:6]:
            r = compare(date, it, ctx)
            if r is None:
                continue
            print(date, it, 'fired' if r['fired'] else 'idle ', f"input->output change {r['moved']:.2e}",
                  ' '.join(f"{k}: maxabs {r[k][0]:.2e} rel {r[k][1]:.2e} nbit {r[k][2]}/{r[k][3]}" for k in ('uo', 'vo', 'vonp')), flush=True)
