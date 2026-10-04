"""Validate kppmix_jax.kppmix_jax against the recorded real KPPMIX calls (D63).

Each ffz_kppmix_<itime>.bin record (D54) is one KPPMIX call. All calls of one itime run as one
batch. Checks: kbl exact, hbl and the four outputs to a stated relative tolerance (speed-first
port, not bitwise). Reports the worst relative error per output and the batch runtime.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kppmix_ff as K  # noqa: E402
from kppmix_compare import load_kppmix_records  # noqa: E402
from kppmix_jax import kppmix_jax  # noqa: E402

TOL = 1e-9


def batch_inputs(recs):
    ze = np.asarray(recs[0]['ze'], dtype=np.float64)  # dump is big-endian; JAX needs native
    tabs = K.kmixinit(ze)
    lsrpd, fsr, dfsrdz, dfsrdzb = K.init_solar(ze)

    def padn(a, n=K.LMO + 1):
        out = np.zeros(n)
        out[:len(a)] = a[:n]
        return out

    fsr_p, dz_p, dzb_p = padn(fsr), padn(dfsrdz), padn(dfsrdzb)
    f = lambda key: np.stack([r[key] for r in recs])  # noqa: E731
    args = dict(
        ze=ze,
        zgrid=f('zgrid'), hwide=f('hwide'), byhwide=f('byhwide'),
        lmij=np.array([r['lmij'] for r in recs], dtype=np.int64),
        shsq=f('shsq'), dvsq=f('dvsq'),
        ustar=np.array([r['ustar'] for r in recs]),
        bo=np.array([r['bo'] for r in recs]),
        bosol=np.array([r['bosol'] for r in recs]),
        dbloc=f('dbloc'), ritop=f('ritop'),
        wmt=tabs['wmt'], wst=tabs['wst'], fz500=tabs['fz500'], vtc=tabs['vtc'],
        cg=tabs['cg'], difmiw=tabs['difmiw'], difsiw=tabs['difsiw'],
        lsrpd=lsrpd, fsr=fsr_p, dfsrdz=dz_p, dfsrdzb=dzb_p,
    )
    return args


def numpy_reference(recs, a):
    """kppmix_ff.kppmix (D54) on every record: the reference the JAX port must match."""
    out = dict(visc=[], difs=[], dift=[], ghat=[], hbl=[], kbl=[])
    for r in recs:
        v, d, t, g, h, kb = K.kppmix(r['ze'], r['zgrid'], r['hwide'], r['byhwide'], r['lmij'],
                                     r['shsq'], r['dvsq'], r['ustar'], r['bo'], r['bosol'],
                                     r['dbloc'], r['ritop'], a['wmt'], a['wst'], a['fz500'],
                                     a['vtc'], a['cg'], a['difmiw'], a['difsiw'], a['lsrpd'],
                                     a['fsr'], a['dfsrdz'], a['dfsrdzb'])
        out['visc'].append(v); out['difs'].append(d); out['dift'].append(t)
        out['ghat'].append(g); out['hbl'].append(h); out['kbl'].append(kb)
    return {k: np.array(v) for k, v in out.items()}


def run(path):
    recs = load_kppmix_records(path)
    a = batch_inputs(recs)
    t0 = time.time()
    visc, difs, dift, ghats, hbl, kbl = kppmix_jax(**a)
    visc, difs, dift, ghats = map(np.asarray, (visc, difs, dift, ghats))
    hbl, kbl = np.asarray(hbl), np.asarray(kbl)
    el = time.time() - t0
    ref = numpy_reference(recs, a)
    vs_numpy = dict(visc=0.0, difs=0.0, dift=0.0, ghat=0.0, hbl=0.0)
    for n in range(len(recs)):
        m = recs[n]['lmij']
        vs_numpy['hbl'] = max(vs_numpy['hbl'], abs(hbl[n] - ref['hbl'][n]) / max(abs(ref['hbl'][n]), 1e-300))
        for name, got, want in [('visc', visc[n], ref['visc'][n]), ('difs', difs[n], ref['difs'][n]),
                                ('dift', dift[n], ref['dift'][n]), ('ghat', ghats[n], ref['ghat'][n])]:
            sc = max(np.max(np.abs(want[1:m + 1])), 1e-300)
            vs_numpy[name] = max(vs_numpy[name], float(np.max(np.abs(got[1:m + 1] - want[1:m + 1])) / sc))
    kbl_vs_numpy = int(np.sum(np.asarray(kbl) != ref['kbl']))
    worst = dict(visc=0.0, difs=0.0, dift=0.0, ghat=0.0, hbl=0.0)
    kbl_bad = 0
    for n, r in enumerate(recs):
        m = r['lmij']
        if int(kbl[n]) != int(r['kbl']):
            kbl_bad += 1
        worst['hbl'] = max(worst['hbl'], abs(hbl[n] - r['hbl']) / max(abs(r['hbl']), 1e-300))
        for name, got, ref, lo, hi in [('visc', visc[n], r['akvm'], 1, m),
                                       ('difs', difs[n], r['akvs'], 1, m),
                                       ('dift', dift[n], r['akvg'], 1, m),
                                       ('ghat', ghats[n], r['ghat'], 1, m)]:
            ref_s = ref[lo:hi + 1]
            scale = max(np.max(np.abs(ref_s)), 1e-300)
            worst[name] = max(worst[name], float(np.max(np.abs(got[lo:hi + 1] - ref_s)) / scale))
    return len(recs), kbl_bad, worst, el, kbl_vs_numpy, vs_numpy


if __name__ == '__main__':
    tot_calls = tot_kbl = 0
    tot_worst = {}
    tot_rec = {}
    tot_t = 0.0
    for path in sys.argv[1:]:
        for f in sorted(glob.glob(f'{path}/ffz_kppmix_*.bin')):
            n, kb, w, el, kbn, vn = run(f)
            tot_calls += n; tot_kbl += kbn; tot_t += el
            for k, v in vn.items():
                tot_worst[k] = max(tot_worst.get(k, 0.0), v)
            for k, v in w.items():
                tot_rec[k] = max(tot_rec.get(k, 0.0), v)
            print(os.path.basename(f), 'calls', n, 'vs numpy', {k: f'{v:.1e}' for k, v in vn.items()}, 'kbl mismatches vs numpy', kbn, 'sec', round(el, 2))
    ok = tot_kbl == 0 and max(tot_worst.values()) <= TOL
    print('TOTAL calls', tot_calls, 'kbl mismatches vs numpy', tot_kbl, 'worst vs numpy', {k: f'{v:.2e}' for k, v in tot_worst.items()},
          'worst vs recorded (D54 residual)', {k: f'{v:.2e}' for k, v in tot_rec.items()},
          'sec', round(tot_t, 2), 'PASS' if ok else 'FAIL', 'tol', TOL)
