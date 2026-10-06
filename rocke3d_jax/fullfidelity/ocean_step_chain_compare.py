"""Chained whole-ocean step vs the ffo_state dumps (D119/D120).

usage: python ocean_step_chain_compare.py <date> [nsteps]
 (1) each step run from the REAL pre-PRECIP_OC snapshot (tag 0) through all stages, compared at every stage boundary
     that has a snapshot (chained inside the step: port output of stage k feeds stage k+1) and at exit (tag 14);
 (2) a free-running chain over all steps (port exit state carried to the next step; recorded: fluxes, ODIFF).
Prints max relative error (max|diff| / max|ref|) per field. NaN is reported as nan."""
import sys
import time
import numpy as np
import ocean_chain_io as C
import ocean_step as O
from ocean_step_compare import ctx_for, fx_of, diff

FIELDS = O.F3 + ['opress', 'mmi', 'opbot', 'ogeoz', 'kpl', 'vonp', 'must', 'g0mst', 's0mst']
TAG_OF = {'precip': 1, 'ground': 2, 'ostres': 3, 'oconv': 4, 'drag': 5, 'polar': 6, 'dynamics': 10, 'straits': 11,
          'post': 12, 'odiff': 13, 'meso': 14}


def fx_step(sn, itime):
    fx = fx_of(sn)
    fx['itime'] = itime
    if 13 in sn:
        fx['odiff'] = dict(uo=sn[13]['uo'], vo=sn[13]['vo'], vonp=sn[13]['vonp'])
    return fx


def errs(got, ref, fields=FIELDS):
    out = {}
    for k in fields:
        if k in got and k in ref:
            out[k] = diff(got[k], ref[k])[1]
    return out


def one_step(d, itime, ctx, sn=None, tags=True):
    sn = sn or C.load_step(d, itime)
    fx = fx_step(sn, itime)
    rows = []
    def trace(name, s):
        t = TAG_OF[name]
        if tags and t in sn and name not in ('dynamics',):
            rows.append((name, errs(s, sn[t])))
    t0 = time.time()
    out = O.ocean_step(sn[0], fx, ctx, trace=trace)
    return out, rows, time.time() - t0, sn


def main(date, nsteps=12):
    d = C.FF_DEFAULT + '/' + date
    ctx = ctx_for(d)
    steps = C.list_steps(d)[:nsteps]
    print('== single steps from real entry, error at each stage boundary (chained inside the step) ==')
    for it in steps:
        out, rows, el, sn = one_step(d, it, ctx)
        for name, e in rows:
            print(it, name, ' '.join(f'{k}:{v:.1e}' for k, v in e.items() if v != 0) or 'exact')
        print(it, 'EXIT', ' '.join(f'{k}:{v:.1e}' for k, v in errs(out, sn[14]).items()), f'[{el:.1f}s]')
    print('== free-running chain (port state carried between steps) ==')
    sn0 = C.load_step(d, steps[0])
    s = sn0[0]
    for it in steps:
        sn = C.load_step(d, it)
        fx = fx_step(sn, it)
        s = O.ocean_step(s, fx, ctx)
        e = errs(s, sn[14])
        print(it, ' '.join(f'{k}:{v:.1e}' for k, v in e.items()))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
