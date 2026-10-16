"""D132: validate clouds_mstcnv_batch.py against the per-column port clouds_mstcnv_ff.mstcnv_column on the real CONDSE columns.

Input: pickle of clouds_condse_profile.py --dump (per column: MSTCNV input r and per-column output o of the real step).  The batch is run on
all non-polar columns (or the first --max-cols) at once and every output field is compared with the stored per-column output:
columns with any value difference, columns with any bit difference, worst abs and scale-relative difference.  Pure-diagnostic outputs
(see clouds_mstcnv_batch docstring) are not compared.  The decision outputs (lmcmin, lmcmax, mccont) are reported separately.
Usage: python3 clouds_mstcnv_batch_compare.py PKL [--imf] [--max-cols N] [--convecting-only]
"""
import argparse
import pickle
import sys
import time

import numpy as np

import clouds_condse_ff as cf
import clouds_mstcnv_batch as mb
import clouds_mstcnv_ff as mc


def load(pkl, max_cols=None):
    d = pickle.load(open(pkl, "rb"))
    tr = d["trace"]
    keys = [k for k in sorted(tr, key=lambda k: (k[1], k[0])) if tr[k] is not None and len(tr[k]["r"]["ra"]) == 4]
    if max_cols:
        keys = keys[:max_cols]
    return d, keys, [tr[k]["r"] for k in keys], [tr[k]["o"] for k in keys]


def compare(o_ref, o_b):
    n = len(o_ref)
    res = {}
    for k in mb.OUT_SCAL + mb.OUT_ARR + ["lhp", "precnvl", "smom", "qmom", "um", "vm"]:
        ref = np.array([np.asarray(o[k], float) for o in o_ref])
        got = np.asarray(o_b[k], float).reshape(ref.shape)
        ne = ref != got
        ax = tuple(range(1, ref.ndim))
        nb = ref.view(np.int64) != got.view(np.int64) if ref.dtype == np.float64 and got.dtype == np.float64 else ne
        d = np.where(ne, np.abs(ref - got), 0.0)
        sc = max(float(np.abs(ref).max()), 1e-300)
        res[k] = (int(ne.any(axis=ax).sum()) if ax else int(ne.sum()), int(np.ascontiguousarray(nb).any(axis=ax).sum()) if ax else int(nb.sum()),
                  float(d.max()), float(d.max()) / sc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--imf", action="store_true")
    ap.add_argument("--max-cols", type=int, default=None)
    ap.add_argument("--rerun", action="store_true", help="re-run the per-column port in the current libm mode as reference (automatic with --imf: the stored outputs are libm)")
    a = ap.parse_args()
    if a.imf:
        cf.set_backend("imf")
    d, keys, rl, ol = load(a.pkl, a.max_cols)
    tr0 = time.perf_counter()
    if a.imf or a.rerun:
        ol = [mc.mstcnv_column(r, cf.RUN_TUNE) for r in rl]
        print(f"per-column reference re-run: {time.perf_counter() - tr0:.1f} s")
    R = mb.stack_r(rl)
    t = time.perf_counter()
    o = mb.mstcnv_batch(R, cf.RUN_TUNE)
    tb = time.perf_counter() - t
    res = compare(ol, o)
    nconv = sum(1 for x in ol if x["lmcmin"] > 0)
    print(f"{d['date']} step {d['step']} ({'imf' if a.imf else 'libm'}): {len(keys)} non-polar columns ({nconv} convecting), batch {tb:.2f} s "
          f"({1e3 * tb / len(keys):.2f} ms/col)")
    bad = 0
    for k, (nv, nb, ma, mr) in res.items():
        flag = "" if nv == 0 and nb == 0 else "   <-- DIFFERS"
        bad += bool(nv)
        print(f"  {k:10s} cols differing {nv:5d} (bit {nb:5d})  max abs {ma:.3e}  max rel-to-scale {mr:.3e}{flag}")
    print(f"fields with any difference: {bad} of {len(res)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
