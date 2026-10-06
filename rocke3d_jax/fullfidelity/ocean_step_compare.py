"""Stage-boundary comparison of ocean_step against the ffo_state dumps (D119/D120).
usage: python ocean_step_compare.py <date> [itime]   (replay mode: each stage starts from the real previous snapshot)"""
import sys
import numpy as np
import ocean_chain_io as C
import ocean_step as O

FIELDS = O.F3 + ['opress', 'smw', 'smu', 'smv', 'mmi']


def ctx_for(d):
    g = C.load_geom(d)
    g['opcoef'] = d + '/ffo_opcoef.bin'
    return g


def fx_of(snaps):
    fx = {}
    for t in (0, 1):
        fx.update({k: v for k, v in snaps[t].items() if k.startswith('o')})
    return fx


def diff(a, b):
    if np.isnan(a).any():
        return float('nan'), float('nan')
    d = np.abs(a - b); sc = max(np.abs(b).max(), 1e-300)
    return float(d.max()), float(d.max() / sc)


def replay(d, itime, ctx, stages=None):
    sn = C.load_step(d, itime)
    fx = fx_of(sn)
    fx['itime'] = itime
    if 13 in sn:
        fx['odiff'] = dict(uo=sn[13]['uo'], vo=sn[13]['vo'], vonp=sn[13]['vonp'])
    rows = []
    # tag chain for replay: input snapshot tag -> output snapshot tag
    for name, fn, out_tag in O.STAGES:
        in_tag = {'precip': 0, 'ground': 1, 'ostres': 2, 'oconv': 3, 'drag': 4, 'polar': 5, 'dynamics': 6, 'straits': 10, 'post': 11, 'odiff': 12, 'meso': 13}[name]
        if in_tag not in sn or out_tag not in sn:
            continue
        if name == 'ground' or name == 'ostres' or name == 'drag' or name == 'polar':
            pass
        inp = {k: v for k, v in sn[1].items() if k in C.KPP1 or k == 'g0m1'}
        inp.update(sn[in_tag])
        got = fn(inp, fx, ctx)
        ref = sn[out_tag]
        if name == 'dynamics' and 9 in sn:      # SMW is read right after OFLUXV (tag 9); at tag 10 the module array holds other data
            ref = dict(ref); ref['smw'] = sn[9]['smw']
        res = {}
        for k in FIELDS:
            if k in got and k in ref:
                g_, r_ = got[k], ref[k]
                if k == 'smw':      # compare only where OFLUXV writes (lmm >= l+1); elsewhere the Fortran array is stale
                    m_ = (np.arange(2, 15)[None, None, :] <= ctx['lmm'][:, :, None]); g_, r_ = np.where(m_, g_, 0), np.where(m_, r_, 0)
                res[k] = diff(g_, r_)
        rows.append((name, res))
    return rows


if __name__ == '__main__':
    d = C.FF_DEFAULT + '/' + sys.argv[1]
    ctx = ctx_for(d)
    its = [int(sys.argv[2])] if len(sys.argv) > 2 else C.list_steps(d)[:6]
    for it in its:
        for name, r in replay(d, it, ctx):
            print(it, name, ' '.join(f'{k}:{v[1]:.1e}' for k, v in r.items() if not (v[0] == 0)) or 'exact')
