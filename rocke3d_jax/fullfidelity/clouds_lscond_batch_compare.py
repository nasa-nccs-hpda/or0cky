"""D131: validate clouds_lscond_batch.py against the per-column port clouds_lscond_ff.lscond on the real CONDSE columns.

Input: the pickle written by `clouds_condse_profile.py --dump` (for every column of one real step: LSCOND input state S0 and parameters P0 as
built by clouds_condse_ff from the real entry arrays and the MSTCNV output of that column).  The per-column reference is re-run from S0
(in the same libm mode), the batch is run on all non-polar columns (KMAX<=4) at once, and every output is compared:
  - number of columns with ANY value difference (==, i.e. -0.0 equals 0.0), per field and in total,
  - number of columns/fields that differ in the BIT pattern (int64 view; catches -0.0 vs 0.0),
  - worst absolute difference and worst scale-relative difference.
The CSIZELIP carry is resolved by the batch (forward fill from the carry of the column before the first one).
Usage: python3 clouds_lscond_batch_compare.py PKL [--imf] [--max-cols N]
"""
import argparse
import copy
import pickle
import sys
import time

import numpy as np

import clouds_condse_ff as cf
import clouds_lscond_batch as lb
import clouds_lscond_ff as ls0

LM = 40
STATE_KEYS = ("tl ql th rh qclx qcix svlhxl cldsavl cldssl taussl tausslip csizel csizelip cldsal cldsv1 qlss qiss sshr dctei sm qm "
              "rh1 wmpr prebar1 lhp").split()
W_KEYS = ("cleara rhf rh00 er ec prep qheatl qheati qheat prebar preice").split()
SCAL_W = ("prcpss", "hcndss", "ierr", "lerr", "wmerr", "ckij", "wmsum")


def load(pkl):
    d = pickle.load(open(pkl, "rb"))
    tr = d["trace"]
    cols = [k for k in sorted(tr, key=lambda k: (k[1], k[0])) if tr[k] is not None and tr[k]["P0"]["kmax"] <= 4]
    return d, [tr[k]["S0"] for k in cols], [tr[k]["P0"] for k in cols], cols


def run_ref(S_list, P_list, max_cols=None):
    out, Ws = [], []
    t = time.perf_counter()
    n = len(S_list) if max_cols is None else max_cols
    prev = None
    for S0, P in zip(S_list[:n], P_list[:n]):
        S0 = copy.deepcopy(S0)
        if prev is not None:                         # chain the module-array carry from the reference output of the previous column
            for k in ("tausslip", "csizelip", "cldsal", "cldsv1", "dqlsc"):
                S0[k] = prev[k]
        S, W, _ = ls0.lscond(S0, P)
        prev = S
        out.append(S)
        Ws.append(W)
    return out, Ws, time.perf_counter() - t


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


def compare(Sref, Wref, S, W, n):
    """-> dict field -> (ncols_value_diff, ncols_bit_diff, maxabs, maxrel)."""
    res = {}

    def add(name, ref, got):                         # ref, got: (n, ...) arrays
        ref, got = np.asarray(ref, float), np.asarray(got, float)
        ne = (ref != got) & ~(np.isnan(ref) & np.isnan(got))
        nb = bits(ref) != bits(got)
        ax = tuple(range(1, ref.ndim))
        d = np.abs(ref - got)
        d = np.where(ne, d, 0.0)
        sc = max(float(np.abs(ref).max()), 1e-300)
        res[name] = (int(ne.any(axis=ax).sum()) if ax else int(ne.sum()), int(nb.any(axis=ax).sum()) if ax else int(nb.sum()),
                     float(d.max()) if d.size else 0.0, float(d.max()) / sc if d.size else 0.0)
    for k in STATE_KEYS:
        ref = np.array([s[k] for s in Sref[:n]])
        got = S[k][:, :n].T
        add(k, ref, got)
    for k in ("qmom", "smom"):
        add(k, np.array([s[k] for s in Sref[:n]]), S[k][:, :, :n].transpose(2, 0, 1))
    for k in ("um", "vm"):
        add(k, np.array([s[k][:][:] for s in Sref[:n]])[:, :, :4], S[k][:, :, :n].transpose(2, 0, 1))
    for k in W_KEYS:
        add("W." + k, np.array([w[k] for w in Wref[:n]]), W[k][:, :n].T)
    for k in SCAL_W:
        add("W." + k, np.array([w[k] for w in Wref[:n]], float), np.asarray(W[k][:n], float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--imf", action="store_true")
    ap.add_argument("--max-cols", type=int, default=None)
    a = ap.parse_args()
    if a.imf:
        cf.set_backend("imf")
    d, S_list, P_list, cols = load(a.pkl)
    n = len(S_list) if a.max_cols is None else a.max_cols
    carry = {k: S_list[0][k] for k in ("tausslip", "csizelip", "cldsal", "cldsv1")}
    Sref, Wref, t_ref = run_ref(S_list, P_list, n)
    S, P = lb.pack(S_list[:n], P_list[:n], carry)
    t = time.perf_counter()
    S, W = lb.lscond_batch(S, P)
    t_b = time.perf_counter() - t
    res = compare(Sref, Wref, S, W, n)
    print(f"{d['date']} step {d['step']} ({'imf' if a.imf else 'libm'}): {n} non-polar columns; per-column {t_ref:.2f} s "
          f"({1e3 * t_ref / n:.2f} ms/col), batch {t_b:.3f} s ({1e3 * t_b / n:.3f} ms/col), speedup {t_ref / t_b:.1f}x")
    anyc = np.zeros(n, bool)
    for k, (nv, nb, ma, mr) in res.items():
        flag = "" if nv == 0 and nb == 0 else "   <-- DIFFERS"
        print(f"  {k:12s} cols differing {nv:5d} (bit {nb:5d})  max abs {ma:.3e}  max rel-to-scale {mr:.3e}{flag}")
    tot_v = sum(1 for v in res.values() if v[0])
    print(f"fields with any difference: {tot_v} of {len(res)}")
    return 0 if tot_v == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
