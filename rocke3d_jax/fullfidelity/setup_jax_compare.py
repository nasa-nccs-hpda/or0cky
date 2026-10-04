"""Validate setup_jax.setup_jax against the recorded setup inputs and KPPMIX inputs (D64).

Per itime: parse the ffz_setup records (D59 layout) into batched arrays, run setup_jax on all
columns at once, and compare with the recorded KPPMIX inputs (ffz_kppmix, D54), the same
reference D59 used. Speed-first tolerance 1e-9 relative (per-array scale).
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kppmix_compare import load_kppmix_records  # noqa: E402
from setup_jax import setup_jax, KMAX  # noqa: E402

LMO = 13
RS = 1218
JM = 46
TOL = 1e-9


def parse_batch(raw):
    N = raw.shape[0]
    off = 6
    def take(n):
        nonlocal off
        v = raw[:, off:off + n]; off += n
        return v
    grav = raw[0, 5]
    pad = lambda x: np.concatenate([np.zeros((N, 1)), x], axis=1)  # noqa: E731
    g = pad(take(LMO)); s = pad(take(LMO)); po = pad(take(LMO))
    mo1 = take(1)[:, 0]
    ul_flat = take(LMO * KMAX)                                    # (N, LMO*KMAX), F order
    ul = np.zeros((N, LMO + 2, KMAX + 1))
    ul[:, 1:LMO + 1, 1:KMAX + 1] = ul_flat.reshape((N, KMAX, LMO)).transpose(0, 2, 1)
    ravm = np.concatenate([np.zeros((N, 1)), take(KMAX)], axis=1)
    take(KMAX)                                                    # lmuv (unused here)
    ogeoz, hocean, deltae, deltas, deltam, deltasr, u2rho = [take(1)[:, 0] for _ in range(7)]
    byrho = pad(take(LMO)); rhom = pad(take(LMO)); rho1 = pad(take(LMO))
    take(LMO)                                                     # ptd
    alpha1, beta1, shc1 = [take(1)[:, 0] for _ in range(3)]
    assert off == RS
    return dict(grav=grav, g=g, s=s, mo1=mo1, ul=ul, ravm=ravm, ogeoz=ogeoz, hocean=hocean,
                deltae=deltae, deltas=deltas, deltam=deltam, deltasr=deltasr, u2rho=u2rho,
                byrho=byrho, rhom=rhom, rho1=rho1, alpha1=alpha1, beta1=beta1, shc1=shc1)


def run(setup_path, kpp_path):
    raw = np.fromfile(setup_path, dtype='>f8').astype(np.float64).reshape(-1, RS)
    recs = load_kppmix_records(kpp_path)
    assert len(recs) == raw.shape[0]
    p = parse_batch(raw)
    rows = raw[:, :4].astype(int)
    j = rows[:, 1]
    kmuv = np.where(j == JM, KMAX, 4).astype(np.int64)
    ze = np.asarray(recs[0]['ze'], dtype=np.float64)
    lmij = np.array([r['lmij'] for r in recs], dtype=np.int64)
    t0 = time.time()
    out = setup_jax(ze, lmij, kmuv, p['ogeoz'], p['hocean'], p['grav'], p['ul'], p['ravm'],
                    p['g'], p['s'], p['byrho'], p['rhom'], p['rho1'], p['alpha1'], p['beta1'],
                    p['shc1'], p['u2rho'], p['deltae'], p['deltas'], p['deltam'], p['deltasr'])
    out = {k: np.asarray(v) for k, v in out.items()}
    el = time.time() - t0
    worst = {}
    for name, key, ref_key in [('zgrid', 'zgrid', 'zgrid'), ('hwide', 'hwide', 'hwide'),
                               ('byhwide', 'byhwide', 'byhwide'), ('shsq', 'shsq', 'shsq'),
                               ('dvsq', 'dvsq', 'dvsq'), ('dbloc', 'dbloc', 'dbloc'),
                               ('ritop', 'ritop', 'ritop')]:
        ref = np.stack([r[ref_key] for r in recs])
        if name in ('shsq', 'dvsq', 'dbloc', 'ritop'):
            m = np.arange(LMO + 1)[None, :] <= lmij[:, None]
            diff = np.where(m, np.abs(out[key] - ref), 0.0)
            sc = np.maximum(np.max(np.abs(np.where(m, ref, 0.0)), axis=1), 1e-300)
            worst[name] = float(np.max(diff / sc[:, None]))
        else:
            worst[name] = float(np.max(np.abs(out[key] - ref) / np.maximum(np.max(np.abs(ref), axis=1), 1e-300)[:, None]))
    for name in ('ustar', 'bo', 'bosol'):
        ref = np.array([r[name] for r in recs])
        worst[name] = float(np.max(np.abs(out[name] - ref) / np.maximum(np.abs(ref), 1e-300)))
    return len(recs), worst, el


if __name__ == '__main__':
    root = sys.argv[1]
    tot = 0; W = {}; T = 0.0
    for sf in sorted(glob.glob(f'{root}/ffz_setup_*.bin')):
        it = sf.split('_')[-1][:-4]
        n, w, el = run(sf, f'{root}/ffz_kppmix_{it}.bin')
        tot += n; T += el
        for k, v in w.items():
            W[k] = max(W.get(k, 0.0), v)
    print('TOTAL calls', tot, 'worst rel', {k: f'{v:.1e}' for k, v in W.items()}, 'sec', round(T, 2),
          'PASS' if max(W.values()) <= TOL else 'FAIL', 'tol', TOL)
