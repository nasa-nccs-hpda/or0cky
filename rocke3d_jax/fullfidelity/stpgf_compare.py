"""Validate straits_jax.stpgf_jax against the real STPGF dumps (D72): ffz_pgf_in (entry) and
ffz_pgf_out (exit, MUST only is compared). JST from the run directory's OSTRAITS namelist;
DXYPO from odhorz_ff.geomo_dyn_arrays; the seawater EOS from the OFTAB table (eos_jax)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos_jax import load_vgsp  # noqa: E402
from straits_jax import stpgf_jax, parse_straits_nml  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402

LMO = 13
NM = 12
GRAV = 9.80665
TOL = 1e-9


def load_pgf(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64)
    o = 0
    def take(n):
        nonlocal o
        v = raw[o:o + n]; o += n
        return v
    must = take(156).reshape((LMO, NM), order='F').T            # (N, L-1)
    oprese = take(24).reshape((2, NM), order='F').T             # (N, 2)
    hoceane = take(24).reshape((2, NM), order='F').T
    lmme = np.round(take(24)).reshape((2, NM), order='F').T.astype(np.int64)
    def tri(n):
        return take(312).reshape((2, NM, LMO), order='F').transpose(1, 0, 2)  # (N, 2, L-1)
    moe, g0me, gzme, s0me, szme = tri(0), tri(0), tri(0), tri(0), tri(0)
    # tri consumed in order moe, g0me, gzme, s0me, szme (the file order); re-read correctly below
    o -= 5 * 312
    moe = tri(0); g0me = tri(0); gzme = tri(0); s0me = tri(0); szme = tri(0)
    distpg = take(NM); wist = take(NM)
    lmst = np.round(take(NM)).astype(np.int64)
    dts = take(1)[0]
    def pad1(a):
        out = np.zeros(a.shape[:-1] + (LMO + 1,)); out[..., 1:] = a; return out
    def pad_must(a):
        out = np.zeros((NM, LMO + 1)); out[:, 1:] = a; return out
    return dict(must=pad_must(must), oprese=oprese, hoceane=hoceane, lmme=lmme,
                moe=pad1(moe), g0me=pad1(g0me), gzme=pad1(gzme), s0me=pad1(s0me), szme=pad1(szme),
                distpg=distpg, wist=wist, lmst=lmst, dts=dts)


def load_pgf_out_must(path):
    return load_pgf(path)['must']


if __name__ == '__main__':
    vgsp = load_vgsp()
    worst = 0.0
    for d in sys.argv[1:]:
        geo = parse_straits_nml(f'{d}/OSTRAITS')
        dxypo = geomo_dyn_arrays()[-1]
        dxyp = dxypo[geo['jst']]
        for f in sorted(glob.glob(f'{d}/ffz_pgf_in_*.bin')):
            it = f.split('_')[-1][:-4]
            s = load_pgf(f)
            ref = load_pgf_out_must(f'{d}/ffz_pgf_out_{it}.bin')
            got = np.asarray(stpgf_jax(vgsp, GRAV, s['dts'], s['lmst'], s['lmme'], s['oprese'],
                                       s['hoceane'], s['moe'], s['g0me'], s['gzme'], s['s0me'],
                                       s['szme'], dxyp, s['must'], s['distpg'], s['wist']))
            for n in range(NM):
                m = s['lmst'][n]
                sc = max(np.max(np.abs(ref[n, 1:m + 1])), 1e-300)
                worst = max(worst, float(np.max(np.abs(got[n, 1:m + 1] - ref[n, 1:m + 1])) / sc))
    print('STPGF worst rel error MUST', f'{worst:.1e}', 'PASS' if worst <= TOL else 'FAIL', 'tol', TOL)
