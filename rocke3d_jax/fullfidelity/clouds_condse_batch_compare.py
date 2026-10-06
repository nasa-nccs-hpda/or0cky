"""D132: validate clouds_condse_batch.py (batched CONDSE chain) on real steps.

For one step of a date (default step index 1) it runs the batched chain on the real entry state and reports
  (a) batch vs the per-column port clouds_condse_ff output (pickle written by `clouds_condse_profile.py --dump`, same libm mode, no
      momentum back-transfer): per field the number of differing cells and the worst difference (bitwise comparison with ==);
  (b) batch vs the REAL exit state of the instrumented model, with the same per-field status (exact / bound 1e-12 of field scale / FAIL)
      and the column attribution of clouds_condse_compare (columns inexact, columns beyond the bound);
  (c) wall time of the batched chain (the per-column time is in the profile pickle's text).
Usage: python3 clouds_condse_batch_compare.py PKL [--imf] [--step 1]
"""
import argparse
import pickle
import sys
import time

import numpy as np

import clouds_condse_batch as cb
import clouds_condse_compare as cc
import clouds_condse_ff as cf
import clouds_condse_io as cio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--imf", action="store_true")
    a = ap.parse_args()
    d = pickle.load(open(a.pkl, "rb"))
    date, step = d["date"], d["step"]
    if a.imf:
        cf.set_backend("imf")
    it0 = dict(cio.DATES)[date]
    inp, ref = cio.load_step(date, it0 + step)
    cfg = cf.make_cfg(date)
    Xp = d["X"]
    nd = 0
    if not a.imf:
        # (a) fresh LSCOND carry (as in the profile run), no momentum back-transfer: the per-column pickle has none either
        Xa, _ = cb.condse_step_batch(inp, cfg, with_momentum=False)
        print("(a) batch vs per-column port (same libm mode, fresh module-state carry, before the momentum back-transfer):")
        for k in sorted(Xp):
            if k in ("U", "V", "UALIJ", "VALIJ") or k not in Xa:
                continue
            ne = int((Xa[k] != Xp[k]).sum())
            if ne:
                nd += 1
                print(f"   {k:14s} differing cells {ne}/{Xp[k].size} max abs {np.abs(Xa[k] - Xp[k]).max():.3e}")
        print(f"   fields differing from the per-column port: {nd} of {len(Xp) - 4}")
    # (b) steps 0..step chained (module-state carry passes from step to step as in clouds_condse_compare), with momentum
    ms = {}
    for s_ in range(step + 1):
        inp_s, ref_s = cio.load_step(date, it0 + s_)
        t = time.perf_counter()
        Xb, cnt = cb.condse_step_batch(inp_s, cfg, ms=ms)
        tb = time.perf_counter() - t
        print(f"{date} step {s_} ({'imf' if a.imf else 'libm'}): batched chain {tb:.1f} s for {cnt['columns']} columns ({1e3 * tb / cnt['columns']:.2f} ms/col)")
    ref = ref_s
    G = cfg["geom"]
    for tag, X in (("batch", Xb),):
        nst = {"exact": 0, "bound": 0, "FAIL": 0}
        worst = []
        for n in cc.ALL_FIELDS:
            if n not in X or n not in ref:
                continue
            mask = None
            if n in ("UALIJ", "VALIJ"):
                mask = np.zeros(X[n].shape, bool)
                for j in range(cc.JM):
                    mask[:, :int(G["IMAXJ"][j]), j] = True
            s = cc.stat(X[n], ref[n], mask)
            nst[s["status"]] += 1
            if s["status"] != "exact":
                worst.append((n, s))
        inex, flip = cc.colmaps(X, ref, G)
        tot = np.zeros((cc.IM, cc.JM), bool)
        for j in range(cc.JM):
            tot[:int(G["IMAXJ"][j]), j] = True
        print(f"(b) {tag} vs REAL exit state: fields exact {nst['exact']}, bound {nst['bound']}, FAIL {nst['FAIL']}; columns inexact "
              f"{int((inex & tot).sum())}/{int(tot.sum())}, beyond 1e-12 bound {int((flip & tot).sum())}")
        for n, s in worst:
            print(f"     {n:14s} {s['status']:5s} inexact {s['ninexact']}/{s['n']} maxabs {s['maxabs']:.3g} (scale {s['scale']:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
