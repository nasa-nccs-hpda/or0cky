"""Validate stconv_jax.stconv_jax against the real STCONV dumps (D68).

ffz_stin (straits state at STCONV entry), ffz_stkpp (KPPMIX inputs and raw outputs per strait,
half-box and ITER: used here only for the EOS values, which are external), ffz_stout (exit state).
Per itime (one model step): run the batched port on the entry state and compare the exit state.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kppmix_ff as K  # noqa: E402
from stconv_jax import stconv_jax  # noqa: E402
from eos_jax import load_vgsp  # noqa: E402

VGSP = load_vgsp()

LMO = 13
NM = 12
RIN = 1356
RKPP = 246
TOL = 1e-9


def load_in(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64)
    assert raw.size == RIN, raw.size
    n = int(round(raw[0]))
    lmst = np.round(raw[1:13]).astype(np.int64)
    blk = lambda off: raw[off:off + 156].reshape(NM, LMO).T  # noqa: E731  (13,12) -> [L-1,n]
    def pad(a):
        out = np.zeros((NM, LMO + 1)); out[:, 1:] = a.T; return out
    mmst = pad(blk(13)); must = pad(blk(169)); g0 = pad(blk(325)); gx = pad(blk(481))
    gz = pad(blk(637)); s0 = pad(blk(793)); sx = pad(blk(949)); sz = pad(blk(1105))
    dist = raw[1261:1273]; wist = raw[1273:1285]
    jst = np.stack([np.round(raw[1285:1297]), np.round(raw[1297:1309])], axis=1).astype(np.int64)
    sinpo = raw[1309:1355]; dts = raw[1355]
    assert n == NM
    return dict(lmst=lmst, mmst=mmst, must=must, g0=g0, gx=gx, gz=gz, s0=s0, sx=sx, sz=sz,
                dist=dist, wist=wist, jst=jst, sinpo=sinpo, dts=dts)


def load_out(path):
    d = load_in(path)
    return {k: d[k] for k in ['must', 'g0', 'gx', 'gz', 's0', 'sx', 'sz']}


def load_eos(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64).reshape(-1, RKPP)
    # layout: n, iq, iter, lmij, g(13), s(13), byrho(13), dbloc(13), ritop(13), dbsfc(13),...
    n = raw[:, 0].astype(int); iq = raw[:, 1].astype(int); it = raw[:, 2].astype(int)
    byrho = raw[:, 4 + 26:4 + 39]; dbloc = raw[:, 4 + 39:4 + 52]
    ritop = raw[:, 4 + 52:4 + 65]; dbsfc = raw[:, 4 + 65:4 + 78]
    return n, iq, it, byrho, dbloc, ritop, dbsfc


def tabs_and_grid(ze):
    tab = K.kmixinit(ze)
    lsrpd, fsr, dz, dzb = K.init_solar(ze)
    pad = lambda a: np.concatenate([a, np.zeros(LMO + 1 - len(a))])  # noqa: E731
    return dict(wmt=tab['wmt'], wst=tab['wst'], fz500=tab['fz500'], vtc=tab['vtc'], cg=tab['cg'],
                difmiw=tab['difmiw'], difsiw=tab['difsiw'], lsrpd=lsrpd,
                fsr=pad(fsr), dfsrdz=pad(dz), dfsrdzb=pad(dzb))


def run(d, itime, ze, grav=9.80665):
    s = load_in(f'{d}/ffz_stin_{itime}.bin')
    out_ref = load_out(f'{d}/ffz_stout_{itime}.bin')
    n, iq, it, byrho, dbloc, ritop, dbsfc = load_eos(f'{d}/ffz_stkpp_{itime}.bin')
    eos = {}
    for key, arr in [('byrho', byrho), ('dbloc', dbloc), ('dbsfc', dbsfc), ('ritop', ritop)]:
        stack = np.zeros((4, 2 * NM, LMO + 1))
        for k in range(1, 5):
            for h in range(2 * NM):
                rows = np.where((n == h // 2 + 1) & (iq == h % 2 + 1) & (it == k))[0]
                if rows.size == 0:  # keep the last available iteration
                    rows = np.where((n == h // 2 + 1) & (iq == h % 2 + 1))[0][-1:]
                stack[k - 1, h, 1:] = arr[rows[0]]
        eos[key] = stack
    tabs = tabs_and_grid(ze)
    res = stconv_jax(ze, grav, s['dts'], NM, s['lmst'], s['mmst'], s['dist'], s['wist'], s['jst'],
                     s['sinpo'], VGSP, s['must'], s['g0'], s['gx'], s['gz'], s['s0'], s['sx'], s['sz'],
                     tabs)
    got = {k: np.asarray(v) for k, v in res.items() if k in
           ('must', 'g0mst', 'gxmst', 'gzmst', 's0mst', 'sxmst', 'szmst')}
    ref_map = {'must': 'must', 'g0mst': 'g0', 'gxmst': 'gx', 'gzmst': 'gz', 's0mst': 's0',
               'sxmst': 'sx', 'szmst': 'sz'}
    err = {}
    for gk, rk in ref_map.items():
        worst = 0.0
        for nn in range(NM):
            m = s['lmst'][nn]
            ref = out_ref[rk][nn, 1:m + 1]; g = got[gk][nn, 1:m + 1]
            sc = max(np.max(np.abs(ref)), 1e-300)
            worst = max(worst, float(np.max(np.abs(g - ref)) / sc))
        err[gk] = worst
    return err


if __name__ == '__main__':
    d = sys.argv[1]
    from kppmix_compare import load_kppmix_records
    worst_all = {}
    for f in sorted(glob.glob(f'{d}/ffz_stin_*.bin')):
        itime = f.split('_')[-1][:-4]
        ze = np.asarray(load_kppmix_records(f'{d}/ffz_kppmix_{itime}.bin')[0]['ze'], dtype=np.float64)
        err = run(d, itime, ze)
        print(itime, {k: f'{v:.1e}' for k, v in err.items()})
        for k, v in err.items():
            worst_all[k] = max(worst_all.get(k, 0.0), v)
    print('WORST', {k: f'{v:.1e}' for k, v in worst_all.items()},
          'PASS' if max(worst_all.values()) <= TOL else 'FAIL', 'tol', TOL)
