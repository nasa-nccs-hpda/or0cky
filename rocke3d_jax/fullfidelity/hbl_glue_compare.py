"""Validate hbl_glue_ff.py against real OCONV dumps (D60).

Each real KPPMIX call (ffz_kppmix_<itime>.bin) is followed by exactly one G OVDIFFS call and
one S OVDIFFS call (ffz_ovdiffs_<itime>.bin, tag 0 then tag 1, same i/j). The OVDIFFS records
carry the diffusivity `k` and flux `ghat` that OCNKPP.f passed in, so they are the ground
truth for the scaling in hbl_glue_ff.py.

BYMML(1) is not recorded by any existing dump. It is recovered per call from the S record's
GHATS (one scalar, from the level with the largest |GHAT|), then the S-flux check is made at
all levels with that value. The G-flux, the k values, and everything that does not use BYMML
are checked bitwise. The S-flux check is a tolerance check because it uses the recovered
BYMML. The production glue takes BYMML as an input (see hbl_glue_ff docstring).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbl_glue_ff import ghat_scalar_fluxes, scale_akv  # noqa: E402
from kppmix_compare import load_kppmix_records  # noqa: E402
from ovdiffs_compare import load_ovdiffs_records  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402
from odhorz0_compare import FF_DEFAULT  # noqa: E402

LMO = 13
RS_SETUP = 1218
KMUV_DUMP = 74
# header(6) + g,s,po(39) + mo1(1) + ul(13*74) + ravm(74) + lmuv(74) + ogeoz..u2rho(7)
BYRHO_OFF = 6 + 3 * LMO + 1 + LMO * KMUV_DUMP + 2 * KMUV_DUMP + 7


def load_setup_byrho_deltas(path):
    """Minimal reader for ffz_setup records (layout from ATM_DRV.f ffdump_setup, D59).
    Returns per-record (byrho, deltae, deltas, deltam) with byrho 1-indexed."""
    raw = np.fromfile(path, dtype='>f8').reshape(-1, RS_SETUP)
    out = []
    for row in raw:
        byrho_off = BYRHO_OFF
        byrho = np.zeros(LMO + 1)
        byrho[1:] = row[byrho_off:byrho_off + LMO]
        deltae = row[1158]
        deltas = row[1159]
        deltam = row[1160]
        out.append(dict(i=int(round(row[0])), j=int(round(row[1])), iter=int(round(row[2])),
                        lmij=int(round(row[3])), byrho=byrho, deltae=deltae, deltas=deltas,
                        deltam=deltam))
    return out


def pair_check(kr, gr, sr, setup, dxypo):
    lmij = kr['lmij']
    j = kr['j']
    assert gr['tag'] == 0 and sr['tag'] == 1, (gr['tag'], sr['tag'])
    assert (gr['i'], gr['j'], sr['i'], sr['j']) == (kr['i'], j, kr['i'], j)
    assert gr['lmij'] == lmij and sr['lmij'] == lmij
    byrho = setup['byrho']
    deltae, deltas, deltam = setup['deltae'], setup['deltas'], setup['deltam']
    s0ml0_1 = sr['u0'][1]
    dxy = dxypo[j]

    k_g = scale_akv(kr['akvg'], byrho, lmij)
    k_s = scale_akv(kr['akvs'], byrho, lmij)
    res = {}
    res['k_g_bad'] = int(np.sum(k_g[1:lmij] != gr['k'][1:lmij]))
    res['k_s_bad'] = int(np.sum(k_s[1:lmij] != sr['k'][1:lmij]))

    # GHATG does not use BYMML: bitwise check with bymml=0 (unused by the G expression).
    ghatg, _ = ghat_scalar_fluxes(kr['akvg'], kr['akvs'], kr['ghat'], deltae, deltas, deltam,
                                  s0ml0_1, 0.0, dxy, lmij)
    res['ghat_g_bad'] = int(np.sum(ghatg[1:lmij] != gr['ghat'][1:lmij]))

    # Recover BYMML(1) from the S flux at the level with the largest |GHAT|.
    lev = max(range(1, lmij), key=lambda L: abs(kr['ghat'][L]))
    denom = kr['akvs'][lev] * kr['ghat'][lev] * dxy
    if denom == 0.0 or s0ml0_1 * deltam == 0.0:
        res['s_undetermined'] = True  # no GHATS signal at this call (zero GHAT or zero mass)
        res['ghat_s_maxrel'] = 0.0
        res['ghat_s_exact_levels'] = 0
        res['ghat_s_nlev'] = 0
        res['bymml'] = float('nan')
        return res
    res['s_undetermined'] = False
    bymml = (deltas - sr['ghat'][lev] / denom) / (s0ml0_1 * deltam)
    _, ghats = ghat_scalar_fluxes(kr['akvg'], kr['akvs'], kr['ghat'], deltae, deltas, deltam,
                                  s0ml0_1, bymml, dxy, lmij)
    ref = sr['ghat'][1:lmij]
    scale = np.maximum(np.abs(ref), 1e-300)
    res['ghat_s_maxrel'] = float(np.max(np.abs(ghats[1:lmij] - ref) / scale))
    res['ghat_s_exact_levels'] = int(np.sum(ghats[1:lmij] == ref))
    res['ghat_s_nlev'] = lmij - 1
    res['bymml'] = bymml
    return res


def main():
    dxypo = geomo_dyn_arrays()[-1]
    tot = dict(calls=0, k_g_bad=0, k_s_bad=0, ghat_g_bad=0, ghat_s_exact=0, ghat_s_levels=0,
               s_undetermined=0, nonfinite=0)
    maxrel = 0.0
    for date in ['nov26', 'dec01', 'jan01']:
        for sf in sorted(glob.glob(f'{FF_DEFAULT}/{date}/ffz_setup_*.bin')):
            itime = sf.split('_')[-1][:-4]
            kf = f'{FF_DEFAULT}/{date}/ffz_kppmix_{itime}.bin'
            of = f'{FF_DEFAULT}/{date}/ffz_ovdiffs_{itime}.bin'
            if not (os.path.exists(kf) and os.path.exists(of)):
                print('skip (missing dump)', date, itime)
                continue
            kr_list = load_kppmix_records(kf)
            ov = load_ovdiffs_records(of)
            setups = load_setup_byrho_deltas(sf)
            assert len(kr_list) == len(setups) == len(ov) // 2, (len(kr_list), len(setups), len(ov))
            for p, (kr, st) in enumerate(zip(kr_list, setups)):
                assert (kr['i'], kr['j'], kr['iter']) == (st['i'], st['j'], st['iter'])
                res = pair_check(kr, ov[2 * p], ov[2 * p + 1], st, dxypo)
                tot['calls'] += 1
                tot['k_g_bad'] += res['k_g_bad']
                tot['k_s_bad'] += res['k_s_bad']
                tot['ghat_g_bad'] += res['ghat_g_bad']
                tot['ghat_s_exact'] += res['ghat_s_exact_levels']
                tot['ghat_s_levels'] += res['ghat_s_nlev']
                tot['s_undetermined'] += int(res['s_undetermined'])
                if not np.isfinite(res['ghat_s_maxrel']):
                    tot['nonfinite'] += 1
                else:
                    maxrel = max(maxrel, res['ghat_s_maxrel'])
    print('calls', tot['calls'])
    print('k (AKVG, AKVS after R2 scaling): mismatched values', tot['k_g_bad'], tot['k_s_bad'])
    print('GHATG mismatched values', tot['ghat_g_bad'])
    print('GHATS exact levels', tot['ghat_s_exact'], 'of', tot['ghat_s_levels'],
          'max rel err', maxrel)
    print('S-flux undetermined (zero GHAT or zero mass; not checked):', tot['s_undetermined'])
    print('non-finite S-flux comparisons:', tot['nonfinite'])


if __name__ == '__main__':
    main()
