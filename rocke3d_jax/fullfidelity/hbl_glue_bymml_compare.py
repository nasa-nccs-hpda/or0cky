"""Bitwise check of hbl_glue_ff.ghat_scalar_fluxes GHATS with BYMML(1) recorded (D60 follow-up).

Reads the per-ITER `ffz_bymml_<itime>.bin` records (ATM_DRV.f ffdump_bymml, unit 1011) written
at the S-OVDIFFS call, pairs them with the matching KPPMIX and S-OVDIFFS records, and compares
the port's GHATS against the S record's ghat bitwise on every level.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbl_glue_ff import ghat_scalar_fluxes  # noqa: E402
from kppmix_compare import load_kppmix_records  # noqa: E402
from ovdiffs_compare import load_ovdiffs_records  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402

RECLEN = 9


def load_bymml(path):
    raw = np.fromfile(path, dtype='>f8').reshape(-1, RECLEN)
    return [dict(i=int(round(r[0])), j=int(round(r[1])), iter=int(round(r[2])),
                 lmij=int(round(r[3])), bymml=r[4], mml=r[5], s0ml0=r[6],
                 deltam=r[7], deltas=r[8]) for r in raw]


def check_itime(kpp_path, ovd_path, byml_path, dxypo):
    kr_list = load_kppmix_records(kpp_path)
    ov = load_ovdiffs_records(ovd_path)
    bm = load_bymml(byml_path)
    assert len(kr_list) == len(bm) == len(ov) // 2, (len(kr_list), len(bm), len(ov))
    exact = total = 0
    maxabs = 0.0
    for p, (kr, b) in enumerate(zip(kr_list, bm)):
        assert (kr['i'], kr['j'], kr['iter'], kr['lmij']) == (b['i'], b['j'], b['iter'], b['lmij'])
        g, s = ov[2 * p], ov[2 * p + 1]
        assert s['tag'] == 1 and s['i'] == kr['i'] and s['j'] == kr['j']
        lm = kr['lmij']
        _, ghats = ghat_scalar_fluxes(kr['akvg'], kr['akvs'], kr['ghat'], 0.0, b['deltas'],
                                      b['deltam'], b['s0ml0'], b['bymml'], dxypo[kr['j']], lm)
        ref = s['ghat'][1:lm]
        exact += int(np.sum(ghats[1:lm] == ref))
        total += lm - 1
        maxabs = max(maxabs, float(np.max(np.abs(ghats[1:lm] - ref))))
    return exact, total, maxabs


if __name__ == '__main__':
    root = sys.argv[1]
    dxypo = geomo_dyn_arrays()[-1]
    E = T = 0
    M = 0.0
    for name in sorted(os.listdir(root)):
        if not name.startswith('ffz_bymml_'):
            continue
        itime = name[len('ffz_bymml_'):-4]
        e, t, m = check_itime(f'{root}/ffz_kppmix_{itime}.bin', f'{root}/ffz_ovdiffs_{itime}.bin',
                              f'{root}/{name}', dxypo)
        print(itime, 'exact levels', e, 'of', t, 'max abs err', m)
        E += e; T += t; M = max(M, m)
    print('TOTAL exact levels', E, 'of', T, 'max abs err', M)
