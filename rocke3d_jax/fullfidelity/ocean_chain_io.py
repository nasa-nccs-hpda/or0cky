"""Readers for the D119 ocean-chain dumps (ffo_geom.bin, ffo_state_<itime>.bin).

Written by the `ffo_snap` hook (instrumentation/OCNDYN2_oceanchain.f.patch, units 1270-1271).
All arrays are big-endian float64 streams in Fortran (column-major) order. Canonical shapes returned here are
0-based numpy arrays: 3-D fields (IM, JM, LMO), 2-D (IM, JM), straits (LMO, NMST) / (2, NMST, LMO).

Snapshot tags (ktag): 0 pre_precip, 1 entry (OCEANS entry, after GROUND_OC inputs are ready), 2 post_ground,
3 post_ostres, 4 post_oconv, 5 post_drag (after OBDRAG2+OCOAST), 6 post_polar (after the UOD/VOD relax),
7 post_odhorz0 (first ODHORZ0 in the NO loop), 8 post_odhorz (after the leapfrog loop), 9 post_ofluxv,
10 post_oadvt, 11 post_straits (after scatter), 12 pre_odiff (after the second ODHORZ0 + ocnstate_derived),
13 post_odiff, 14 exit (after ocnmeso_drv).
Steps in the 12-step run: the first 6 carry all tags, steps 7-12 only tags 0, 1 and 14.
"""
import glob
import os

import numpy as np

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
TAGS = {0: 'pre_precip', 1: 'entry', 2: 'post_ground', 3: 'post_ostres', 4: 'post_oconv', 5: 'post_drag',
        6: 'post_polar', 7: 'post_odhorz0', 8: 'post_odhorz', 9: 'post_ofluxv', 10: 'post_oadvt',
        11: 'post_straits', 12: 'pre_odiff', 13: 'post_odiff', 14: 'exit'}
F3A = ['g0m', 's0m', 'gxmo', 'gymo', 'gzmo', 'sxmo', 'symo', 'szmo', 'mo', 'uo', 'vo', 'uod', 'vod']
F3B = ['mmi', 'smu', 'smv', 'smw']
F2 = ['opbot', 'opress', 'ogeoz', 'kpl']
STR2 = ['must', 'mmst', 'g0mst', 'gxmst', 'gzmst', 's0mst', 'sxmst', 'szmst']
STR3 = ['moe', 'g0me', 'gxme', 'gyme', 'gzme', 's0me', 'sxme', 'syme', 'szme']
FLUX_P = ['oprec', 'oeprec', 'orsi', 'orunpsi', 'oerunpsi', 'osrunpsi']
FLUX_E = ['orsi', 'oflowo', 'omelti', 'oevapor', 'orunosi', 'oeflowo', 'oemelti', 'oe0', 'oerunosi', 'osmelti',
          'osrunosi', 'osolarw', 'osolari', 'oapress', 'odmua', 'odmva', 'odmui', 'odmvi']
KPP1 = ['mo1', 'gxm1', 'gym1', 'sxm1', 'sym1', 'uo1', 'vo1', 'uod1', 'vod1', 's0m1']
IM, JM, LMO = 72, 46, 13


def _f(path):
    return np.memmap(path, dtype='>f8', mode='r')


def load_geom(d):
    """Static geometry/straits tables from ffo_geom.bin. Fortran (i,j) -> numpy [i-1,j-1]."""
    a = np.asarray(_f(os.path.join(d, 'ffo_geom.bin'))).astype(np.float64)
    im, jm, lmo, nmst = [int(round(x)) for x in a[:4]]
    o = 4
    def take(n, shape=None):
        nonlocal o
        v = a[o:o + n]; o += n
        return v.reshape(shape, order='F') if shape else v
    g = dict(im=im, jm=jm, lmo=lmo, nmst=nmst)
    for k in ('focean', 'hocean', 'fgeotherm'):
        g[k] = take(im * jm, (im, jm))
    for k in ('lmm', 'lmu', 'lmv'):
        g[k] = np.rint(take(im * jm, (im, jm))).astype(np.int64)
    g['lmst'] = np.rint(take(nmst)).astype(np.int64)
    g['ist'] = np.rint(take(nmst * 2, (nmst, 2), )).astype(np.int64)
    g['jst'] = np.rint(take(nmst * 2, (nmst, 2))).astype(np.int64)
    g['wist'] = take(nmst); g['dist'] = take(nmst); g['distpg'] = take(nmst)
    g['kn2'] = np.rint(take(2 * 2 * nmst, (2, 2, nmst))).astype(np.int64)
    g['xst'] = take(nmst * 2, (nmst, 2)); g['yst'] = take(nmst * 2, (nmst, 2))
    assert o == len(a), (o, len(a))
    return g


def _read_record(a, o):
    k, itime, im, jm, lmo, nmst, lsrpd, sz = [int(round(x)) for x in a[o:o + 8]]
    o += 8
    n3 = im * jm * lmo
    rec = dict(ktag=k, itime=itime, lsrpd=lsrpd)
    def take(n, shape):
        nonlocal o
        v = np.array(a[o:o + n], dtype=np.float64); o += n
        return v.reshape(shape, order='F')
    for f in F3A + F3B:
        rec[f] = take(n3, (im, jm, lmo))
    for f in F2:
        rec[f] = take(im * jm, (im, jm))
    rec['kpl'] = np.rint(rec['kpl']).astype(np.int64)
    rec['vonp'] = take(lmo, (lmo,))
    for f in STR2:
        rec[f] = take(lmo * nmst, (lmo, nmst))
    for f in STR3:
        rec[f] = take(2 * nmst * lmo, (2, nmst, lmo))
    if k == 0:
        for f in FLUX_P:
            rec[f] = take(im * jm, (im, jm))
    if k == 1:
        for f in FLUX_E:
            rec[f] = take(im * jm, (im, jm))
        rec['g0m1'] = take(im * jm * lsrpd, (im, jm, lsrpd))
        for f in KPP1:
            rec[f] = take(im * jm, (im, jm))
    if k == 2:
        for f in ('odmsi', 'odhsi', 'odssi'):
            rec[f] = take(2 * im * jm, (2, im, jm))
    return rec, o


def load_step(d, itime):
    """All snapshots of one step: dict tag -> record dict."""
    a = _f(os.path.join(d, f'ffo_state_{itime}.bin'))
    out = {}
    o = 0
    while o < len(a):
        rec, o = _read_record(a, o)
        out[rec['ktag']] = rec
    return out


def list_steps(d):
    return sorted(int(os.path.basename(p)[10:-4]) for p in glob.glob(os.path.join(d, 'ffo_state_*.bin')))
