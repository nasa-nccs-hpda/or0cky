"""Check the batched HBL loop's momentum state against ffz_momi (D66).

ffz_momi (ffdump_momi, unit 1015) holds the solved UL(:,K) after each momentum OVDIFF call, with
the column index I, the ITER and K. The loop is run capped at 1 and at 2 ITERs, and its UL is
compared with the matching records (active K only: LMUV(K) > 1).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocnhbl_jax_compare as C  # noqa: E402
import kppmix_ff as K  # noqa: E402
from kppmix_compare import load_kppmix_records  # noqa: E402
from oconv_save_compare import column_groups  # noqa: E402
from setup_jax_compare import parse_batch, RS  # noqa: E402
from ocnhbl_jax import hbl_loop  # noqa: E402

LMO = 13
MI = 31  # i,j,iter,k,lmuv (5) + u (13) + u0 (13)


def build(d, itime):
    h, mo, g0ml, g0ml0, s0ml, s0ml0, ul0, ulm, uld0, uld, ravm, lmuv, dtbydz, bydz2 = \
        C.load_hblin(f'{d}/ffz_hblin_{itime}.bin')
    kr = load_kppmix_records(f'{d}/ffz_kppmix_{itime}.bin')
    groups = column_groups(kr)
    raw = np.fromfile(f'{d}/ffz_setup_{itime}.bin', dtype='>f8').astype(np.float64).reshape(-1, RS)
    sp = parse_batch(raw)
    ze = np.asarray(kr[0]['ze'], dtype=np.float64)
    pos = [[kr.index(g[k]) for k in range(len(g))] for g in groups]
    N = len(groups)
    def E(f, k):
        return np.stack([sp[f][pos[c][min(k, len(pos[c]) - 1)]] for c in range(N)])
    def Es(f, k):
        return np.array([sp[f][pos[c][min(k, len(pos[c]) - 1)]] for c in range(N)])
    eos = dict(byrho=np.stack([E('byrho', k) for k in range(4)]), rhom=np.stack([E('rhom', k) for k in range(4)]),
               rho1=np.stack([E('rho1', k) for k in range(4)]), alpha=np.stack([Es('alpha1', k) for k in range(4)]),
               beta=np.stack([Es('beta1', k) for k in range(4)]), shc=np.stack([Es('shc1', k) for k in range(4)]))
    tab = K.kmixinit(ze); ls, fsr, dz, dzb = K.init_solar(ze)
    pad = lambda a: np.concatenate([a, np.zeros(LMO + 1 - len(a))])  # noqa: E731
    tabs = dict(wmt=tab['wmt'], wst=tab['wst'], fz500=tab['fz500'], vtc=tab['vtc'], cg=tab['cg'],
                difmiw=tab['difmiw'], difsiw=tab['difsiw'], lsrpd=ls, fsr=pad(fsr), dfsrdz=pad(dz), dfsrdzb=pad(dzb))
    post = np.fromfile(f'{d}/ffz_post_{itime}.bin', dtype='>f8').astype(np.float64).reshape(-1, 34)
    args = dict(ze=ze, grav=9.80665, lmij=h['lmij'], kmuv=h['kmuv'], pole=h['j'] == 46, dts=h['dts'][0],
                dxypo=h['dxypo'], mo=mo, mo1=h['mo1'], deltae=h['deltae'], deltas=h['deltas'],
                deltam=h['deltam'], deltasr=h['deltasr'], u2rho=h['u2rho'], ogeoz=h['ogeoz'],
                hocean=h['hocean'], s0m1=post[:, 4], ravm=ravm, lmuv=lmuv, dtbydz=dtbydz, bydz2=bydz2,
                ul0=ul0, ulm=ulm, uld0=uld0, uld=uld, g0ml0=g0ml0, s0ml0=s0ml0, g0ml=g0ml, s0ml=s0ml,
                eos=eos, tabs=tabs)
    return args, groups, h


def momi_records(d, itime):
    raw = np.fromfile(f'{d}/ffz_momi_{itime}.bin', dtype='>f8').astype(np.float64).reshape(-1, MI)
    return raw


def main(d):
    for mf in sorted(__import__('glob').glob(f'{d}/ffz_momi_*.bin')):
        itime = mf.split('_')[-1][:-4]
        args, groups, h = build(d, itime)
        rec = momi_records(d, itime)
        col_index = {}
        for c in range(len(groups)):
            col_index[(int(h['i'][c]), int(h['j'][c]))] = c
        for itmax in (1, 2):
            out = hbl_loop(**args, itmax=itmax)
            ul = np.asarray(out['ul'])
            worst = 0.0; n = 0; bad_cols = set()
            for r in rec:
                i, j, it, k, lmuv_k = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])
                if it != itmax or lmuv_k <= 1:
                    continue
                c = col_index[(i, j)]
                ref = r[5:5 + LMO]
                got = ul[c, 1:LMO + 1, k]
                sc = max(np.max(np.abs(ref)), 1e-300)
                e = float(np.max(np.abs(got[:lmuv_k] - ref[:lmuv_k])) / sc)
                worst = max(worst, e); n += 1
                if e > 1e-6:
                    bad_cols.add(c)
            print(itime, 'itmax', itmax, 'records', n, 'worst rel UL', f'{worst:.2e}', 'columns > 1e-6:', len(bad_cols))

if __name__ == '__main__':
    main(sys.argv[1])


def check_u0(d, itime):
    """Compare the UL0(:,K) recorded at each momentum call with the loop's entry UL0 for the same
    column (ITER=1 records only). Reports the worst relative difference per level."""
    args, groups, h = build(d, itime)
    rec = momi_records(d, itime)
    ul0 = np.asarray(args['ul0'])
    col_index = {(int(h['i'][c]), int(h['j'][c])): c for c in range(len(groups))}
    worst = np.zeros(LMO)
    for r in rec:
        if int(r[2]) != 1:
            continue
        c = col_index[(int(r[0]), int(r[1]))]
        k = int(r[3])
        ref = r[18:18 + LMO]
        got = ul0[c, 1:LMO + 1, k]
        sc = np.maximum(np.abs(ref), 1e-300)
        worst = np.maximum(worst, np.abs(got - ref) / sc)
    print('UL0 at momentum call vs entry dump, worst rel per level:', np.array2string(worst, precision=2))
