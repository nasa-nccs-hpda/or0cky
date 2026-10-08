"""Prognostic-state difference between two saved step files (numpy only; runs on the Discover login node).

usage: python gpu_state_diff.py CAND.npz REF.npz [CAND2.npz REF2.npz ...]

For each pair prints, for the end-of-step state fields (filter/T U V Q QCL QCI P), the non-finite counts and the RMS difference as a
fraction of the field's own RMS (and the max absolute difference). Compare the step-5 number with the step-0 number to see growth, and with
a CPU-vs-CPU pair (e.g. libimf vs libm) for the size of a 1-ulp exp/pow perturbation. This is a diagnostic, not an acceptance verdict.
"""
import sys

import numpy as np

KEYS = ['filter/T', 'filter/U', 'filter/V', 'filter/Q', 'filter/QCL', 'filter/QCI', 'filter/P']


def diff(cand, ref):
    print(f'candidate {cand}\nreference {ref}')
    c, r = np.load(cand), np.load(ref)
    print(f"{'field':<12}{'nonfinite c/r':>16}{'rms diff / rms field':>24}{'max |diff|':>14}")
    for k in KEYS:
        if k not in c.files or k not in r.files:
            print(f'{k:<12}  missing')
            continue
        a, b = np.asarray(c[k], float), np.asarray(r[k], float)
        nf = f'{int((~np.isfinite(a)).sum())}/{int((~np.isfinite(b)).sum())}'
        ok = np.isfinite(a) & np.isfinite(b)
        d = (a - b)[ok]
        rms_f = float(np.sqrt(np.mean(b[ok] ** 2))) if ok.any() else float('nan')
        rms_d = float(np.sqrt(np.mean(d ** 2))) if ok.any() else float('nan')
        print(f'{k:<12}{nf:>16}{rms_d / rms_f if rms_f else float("nan"):>24.3e}{float(np.max(np.abs(d))) if d.size else float("nan"):>14.3e}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2 or len(a) % 2:
        sys.exit(__doc__)
    for i in range(0, len(a), 2):
        diff(a[i], a[i + 1])
        print()
