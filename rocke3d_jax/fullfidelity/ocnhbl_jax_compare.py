"""Validate ocnhbl_jax.hbl_loop against the real OCONV loop-state dumps (D66).

Per itime: ffz_hblin (ITER=1 entry state, one record per column), ffz_setup and ffz_kppmix (EOS per
ITER, from the setup record of each call), ffz_hblout (exit state: UL, ULD, G0ML, S0ML, HBL, KBL),
ffz_post (FLG3D/FLS3D/DM). All columns of one itime run as one batch.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kppmix_ff as K  # noqa: E402
from kppmix_compare import load_kppmix_records  # noqa: E402
from oconv_save_compare import column_groups  # noqa: E402
from setup_jax_compare import parse_batch, RS  # noqa: E402
from ocnhbl_jax import hbl_loop  # noqa: E402

LMO = 13
KMAX = 74
JM = 46
HIN = 4101
HOUT = 1957
POST = 34


def load_hblin(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64).reshape(-1, HIN)
    N = raw.shape[0]
    def arr13(off):
        out = np.zeros((N, LMO + 1)); out[:, 1:] = raw[:, off:off + LMO]; return out, off + LMO
    off = 14
    mo, off = arr13(off); g0ml, off = arr13(off); g0ml0, off = arr13(off)
    s0ml, off = arr13(off); s0ml0, off = arr13(off)
    def ul(off):
        flat = raw[:, off:off + LMO * KMAX].reshape(N, KMAX, LMO).transpose(0, 2, 1)
        out = np.zeros((N, LMO + 1, KMAX + 1)); out[:, 1:, 1:] = flat
        return out, off + LMO * KMAX
    # dump order (ffdump_hblin): ul, ul0, uld, uld0
    ulm, off = ul(off); ul0, off = ul(off); uld, off = ul(off); uld0, off = ul(off)
    ravm = np.zeros((N, KMAX + 1)); ravm[:, 1:] = raw[:, off:off + KMAX]; off += KMAX
    lmuv = np.zeros((N, KMAX + 1), dtype=np.int64); lmuv[:, 1:] = np.round(raw[:, off:off + KMAX]).astype(np.int64)
    off += KMAX
    dtbydz, off = arr13(off); bydz2, off = arr13(off)
    assert off == HIN
    h = dict(i=raw[:, 0].astype(int), j=raw[:, 1].astype(int), lmij=raw[:, 2].astype(np.int64),
             kmuv=raw[:, 3].astype(np.int64), dts=raw[:, 4], dxypo=raw[:, 5], mo1=raw[:, 6],
             deltae=raw[:, 7], deltas=raw[:, 8], deltam=raw[:, 9], deltasr=raw[:, 10],
             u2rho=raw[:, 11], ogeoz=raw[:, 12], hocean=raw[:, 13])
    return h, mo, g0ml, g0ml0, s0ml, s0ml0, ul0, ulm, uld0, uld, ravm, lmuv, dtbydz, bydz2


def load_hblout(path):
    raw = np.fromfile(path, dtype='>f8').astype(np.float64).reshape(-1, HOUT)
    N = raw.shape[0]
    ul = raw[:, 7:7 + LMO * KMAX].reshape(N, KMAX, LMO).transpose(0, 2, 1)
    uld = raw[:, 7 + LMO * KMAX:7 + 2 * LMO * KMAX].reshape(N, KMAX, LMO).transpose(0, 2, 1)
    off = 7 + 2 * LMO * KMAX
    g = np.zeros((N, LMO + 1)); g[:, 1:] = raw[:, off:off + LMO]
    s = np.zeros((N, LMO + 1)); s[:, 1:] = raw[:, off + LMO:off + 2 * LMO]
    return dict(hbl=raw[:, 5], kbl=np.round(raw[:, 6]).astype(np.int64), ul=ul, uld=uld, g0ml=g, s0ml=s)


def run(d):
    hin = os.path.join(d, 'ffz_hblin_*.bin')
    for hf in sorted(glob.glob(hin)):
        itime = hf.split('_')[-1][:-4]
        h, mo, g0ml, g0ml0, s0ml, s0ml0, ul0, ulm, uld0, uld, ravm, lmuv, dtbydz, bydz2 = load_hblin(hf)
        kr = load_kppmix_records(f'{d}/ffz_kppmix_{itime}.bin')
        groups = column_groups(kr)
        assert len(groups) == len(h['i']), (len(groups), len(h['i']))
        raw_setup = np.fromfile(f'{d}/ffz_setup_{itime}.bin', dtype='>f8').astype(np.float64).reshape(-1, RS)
        sp = parse_batch(raw_setup)
        ze = np.asarray(kr[0]['ze'], dtype=np.float64)
        # EOS per (column, ITER): setup record index = position of that call in kr
        pos = {}
        for c, grp in enumerate(groups):
            base = kr.index(grp[0])
            for k, r in enumerate(grp):
                pos[(c, k)] = base + k
        def eos_of(field, k):
            out = np.zeros((len(groups), LMO + 1))
            for c, grp in enumerate(groups):
                kk = min(k, len(grp) - 1)
                out[c] = sp[field][pos[(c, kk)]]
            return out
        def eos_s(field, k):
            return np.array([sp[field][pos[(c, min(k, len(g) - 1))]] for c, g in enumerate(groups)])
        eos = dict(byrho=np.stack([eos_of('byrho', k) for k in range(4)]),
                   rhom=np.stack([eos_of('rhom', k) for k in range(4)]),
                   rho1=np.stack([eos_of('rho1', k) for k in range(4)]),
                   alpha=np.stack([eos_s('alpha1', k) for k in range(4)]),
                   beta=np.stack([eos_s('beta1', k) for k in range(4)]),
                   shc=np.stack([eos_s('shc1', k) for k in range(4)]))
        tab = K.kmixinit(ze)
        lsrpd, fsr, dfsrdz, dfsrdzb = K.init_solar(ze)
        pad = lambda a: np.concatenate([a, np.zeros(LMO + 1 - len(a))]) if len(a) < LMO + 1 else a  # noqa: E731
        tabs = dict(wmt=tab['wmt'], wst=tab['wst'], fz500=tab['fz500'], vtc=tab['vtc'], cg=tab['cg'],
                    difmiw=tab['difmiw'], difsiw=tab['difsiw'], lsrpd=lsrpd,
                    fsr=pad(fsr), dfsrdz=pad(dfsrdz), dfsrdzb=pad(dfsrdzb))
        pole = h['j'] == JM
        post = np.fromfile(f'{d}/ffz_post_{itime}.bin', dtype='>f8').astype(np.float64).reshape(-1, POST)
        s0m1 = post[:, 4]
        t0 = time.time()
        # setup reads the entry UL (ulm) at ITER=1; UL0 is the momentum input
        out = hbl_loop(ze, 9.80665, h['lmij'], h['kmuv'], pole, h['dts'][0], h['dxypo'], mo, h['mo1'],
                       h['deltae'], h['deltas'], h['deltam'], h['deltasr'], h['u2rho'], h['ogeoz'],
                       h['hocean'], s0m1, ravm, lmuv, dtbydz, bydz2, ul0, ulm, uld0, uld,
                       g0ml0, s0ml0, g0ml, s0ml, eos, tabs)
        out = {k: np.asarray(v) for k, v in out.items()}
        el = time.time() - t0
        ref = load_hblout(f'{d}/ffz_hblout_{itime}.bin')
        last = [grp[-1] for grp in groups]
        hbl_ref = np.array([r['hbl'] for r in last]); kbl_ref = np.array([r['kbl'] for r in last])
        ref_iter = np.array([len(g) for g in groups])
        if os.environ.get('HBL_DEBUG'):
            bad = np.where(np.abs(out['hbl'] - hbl_ref) / np.abs(hbl_ref) > 1e-6)[0]
            print('  mismatched cols', len(bad), 'first', bad[:5])
            for c in bad[:3]:
                print('  col', c, 'lmij', h['lmij'][c], 'out iters', out['iters'][c], 'ref iter', int(ref_iter[c]),
                      'out hbl', out['hbl'][c], 'ref', hbl_ref[c], 'out kbl', out['kbl'][c], 'ref', kbl_ref[c])
                print('    out g0ml', out['g0ml'][c, 1:4], 'ref', ref['g0ml'][c, 1:4])
        lm = h['lmij']
        Lm = (np.arange(LMO + 1)[None, :] >= 1) & (np.arange(LMO + 1)[None, :] <= lm[:, None])
        Km = (np.arange(KMAX + 1)[None, :] >= 1) & (np.arange(KMAX + 1)[None, :] <= h['kmuv'][:, None])
        def cerr(a, b, mask):
            sc = np.maximum(np.max(np.abs(np.where(mask, b, 0.0)), axis=tuple(range(1, b.ndim))), 1e-300)
            ex = np.where(mask, np.abs(a - b), 0.0)
            return float(np.max(ex / sc.reshape((-1,) + (1,) * (b.ndim - 1))))
        ul_mask = Lm[:, :, None] & Km[:, None, :]
        ul_err = cerr(out['ul'][:, 1:LMO + 1, 1:KMAX + 1], ref['ul'], ul_mask[:, 1:LMO + 1, 1:KMAX + 1])
        g_err = cerr(out['g0ml'][:, 1:LMO + 1], ref['g0ml'][:, 1:LMO + 1], Lm[:, 1:LMO + 1])
        s_err = cerr(out['s0ml'][:, 1:LMO + 1], ref['s0ml'][:, 1:LMO + 1], Lm[:, 1:LMO + 1])
        flg_ref = post[:, 6:20]; fls_ref = post[:, 20:34]
        flm = np.arange(LMO + 1)[None, :] <= lm[:, None]
        flg_err = cerr(out['flg3d'][:, :LMO + 1], flg_ref, flm)
        fls_err = cerr(out['fls3d'][:, :LMO + 1], fls_ref, flm)
        print(itime, 'cols', len(groups), 'kbl mismatch', int(np.sum(out['kbl'] != kbl_ref)),
              'hbl rel', f"{np.max(np.abs(out['hbl'] - hbl_ref) / np.abs(hbl_ref)):.1e}",
              'ul', f"{ul_err:.1e}", 'g0ml', f"{g_err:.1e}", 's0ml', f"{s_err:.1e}",
              'flg', f"{flg_err:.1e}", 'fls', f"{fls_err:.1e}", 'sec', round(el, 2))


if __name__ == '__main__':
    for d in sys.argv[1:]:
        run(d)
