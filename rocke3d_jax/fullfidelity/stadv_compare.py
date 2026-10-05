"""Validate straits_jax.stadv_jax against the real STADV dumps (D71).

ffz_stadv_in / ffz_stadv_out: straits state at STADV entry and exit (layout as ffz_stin).
ffz_me_in / ffz_me_out: end-point arrays (MOE, G0ME, GXME, GYME, GZME, S0ME, SXME, SYME, SZME),
each (2, NMST, LMO) in Fortran order. Geometry (IST, JST, XST, YST) from the run directory's
OSTRAITS namelist. DXYPO(J1), DXYPO(J2) from odhorz_ff.geomo_dyn_arrays.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stconv_compare as C  # noqa: E402
from straits_jax import stadv_seq, parse_straits_nml  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402

LMO = 13
NM = 12
ME_NAMES = ['moe', 'g0me', 'gxme', 'gyme', 'gzme', 's0me', 'sxme', 'syme', 'szme']
TOL = 1e-9


def load_me(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64)
    out = {}
    for k, name in enumerate(ME_NAMES):
        a = raw[k * 312:(k + 1) * 312].reshape((2, NM, LMO), order='F')   # (end, n, L-1)
        b = np.zeros((NM, 2, LMO + 1))
        b[:, :, 1:] = a.transpose(1, 0, 2)
        out[name] = b
    return out


def run(d, itime, dxypo, geo):
    s = C.load_in(f'{d}/ffz_stadv_in_{itime}.bin')
    s_out = C.load_in(f'{d}/ffz_stadv_out_{itime}.bin')
    me_in = load_me(f'{d}/ffz_me_in_{itime}.bin')
    me_out = load_me(f'{d}/ffz_me_out_{itime}.bin')
    lm = s['lmst']
    j1 = s['jst'][:, 0]; j2 = s['jst'][:, 1]
    dxyp1 = dxypo[j1]; dxyp2 = dxypo[j2]
    # geometry from the namelist (IST/JST must agree with the dump's JST)
    assert np.array_equal(geo['jst'], s['jst']), 'namelist JST disagrees with the dump'
    me_new, mst_new = stadv_seq(s['dts'], lm, s['mmst'], s['must'], dxyp1, dxyp2, geo['xst'],
                                geo['yst'], geo['ist'], geo['jst'], me_in,
                                {'g0': s['g0'], 'gx': s['gx'], 'gz': s['gz'], 's0': s['s0'],
                                 'sx': s['sx'], 'sz': s['sz']})
    out = dict(me_new)
    out.update({'g0mst': mst_new['g0'], 'gxmst': mst_new['gx'], 'gzmst': mst_new['gz'],
                's0mst': mst_new['s0'], 'sxmst': mst_new['sx'], 'szmst': mst_new['sz']})
    err = {}
    st_map = {'g0mst': 'g0', 'gxmst': 'gx', 'gzmst': 'gz', 's0mst': 's0', 'sxmst': 'sx',
              'szmst': 'sz'}
    for gk, rk in st_map.items():
        worst = 0.0
        for n in range(NM):
            m = lm[n]
            ref = s_out[rk][n, 1:m + 1]
            sc = max(np.max(np.abs(ref)), 1e-300)
            worst = max(worst, float(np.max(np.abs(out[gk][n, 1:m + 1] - ref)) / sc))
        err[gk] = worst
    for name in ME_NAMES:
        worst = 0.0
        for n in range(NM):
            m = lm[n]
            ref = me_out[name][n, :, 1:m + 1]
            got = out[name][n][:, 1:m + 1]
            sc = max(np.max(np.abs(ref)), 1e-300)
            worst = max(worst, float(np.max(np.abs(got - ref)) / sc))
        err[name] = worst
    return err


if __name__ == '__main__':
    worst_all = {}
    for d in sys.argv[1:]:
        geo = parse_straits_nml(f'{d}/OSTRAITS')
        dxypo = geomo_dyn_arrays()[-1]
        for f in sorted(glob.glob(f'{d}/ffz_stadv_in_*.bin')):
            itime = f.split('_')[-1][:-4]
            err = run(d, itime, dxypo, geo)
            for k, v in err.items():
                worst_all[k] = max(worst_all.get(k, 0.0), v)
    print('WORST', {k: f'{v:.1e}' for k, v in worst_all.items()},
          'PASS' if max(worst_all.values()) <= TOL else 'FAIL', 'tol', TOL)
