"""D147: JAX CONDSE vs the numpy batched CONDSE (bitwise, every exit field) and vs the REAL exit state, with per-step timing and compile time.
Usage: python3 clouds_condse_jax_compare.py [--dates nov26,dec01,jan01] [--steps 2] [--ls-mode xla|libm] [--only-ls]
Steps 0..steps-1 are chained per date (module-state carry passes from step to step in each variant separately).
"""
import argparse
import sys
import time

import clouds_jax_env  # noqa: F401
import numpy as np

import clouds_condse_batch as cb
import clouds_condse_compare as cc
import clouds_condse_ff as cf
import clouds_condse_io as cio
import clouds_condse_jax as cj


def bitdiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        return int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum()), int((np.ascontiguousarray(a).view(np.int64) != np.ascontiguousarray(b).view(np.int64)).sum())
    return int((a != b).sum()), int((a != b).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="nov26,dec01,jan01")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--ls-mode", default="xla")
    ap.add_argument("--only-ls", action="store_true")
    a = ap.parse_args(argv)
    bad_total = 0
    for date, it0 in cio.DATES:
        if date not in a.dates.split(","):
            continue
        cfg = cf.make_cfg(date)
        G = cfg["geom"]
        msn, msj = {}, {}
        for s in range(a.steps):
            inp, ref = cio.load_step(date, it0 + s)
            t = time.perf_counter()
            Xn, _ = cb.condse_step_batch(inp, cfg, ms=msn)
            tn = time.perf_counter() - t
            t = time.perf_counter()
            Xj, _ = cj.condse_step_jax(inp, cfg, ms=msj, ls_mode=a.ls_mode, use_mc=not a.only_ls)
            tj = time.perf_counter() - t
            nv = nb = 0
            worst = {}
            for k in sorted(Xn):
                if k not in Xj:
                    continue
                v, b = bitdiff(Xn[k], Xj[k])
                if v or b:
                    nv += 1
                    worst[k] = (v, b, float(np.abs(np.asarray(Xn[k], float) - np.asarray(Xj[k], float)).max()))
            nst = {"exact": 0, "bound": 0, "FAIL": 0}
            for n in cc.ALL_FIELDS:
                if n in Xj and n in ref:
                    mask = None
                    if n in ("UALIJ", "VALIJ"):
                        mask = np.zeros(Xj[n].shape, bool)
                        for j in range(cc.JM):
                            mask[:, :int(G["IMAXJ"][j]), j] = True
                    nst[cc.stat(Xj[n], ref[n], mask)["status"]] += 1
            nstn = {"exact": 0, "bound": 0, "FAIL": 0}
            for n in cc.ALL_FIELDS:
                if n in Xn and n in ref:
                    mask = None
                    if n in ("UALIJ", "VALIJ"):
                        mask = np.zeros(Xn[n].shape, bool)
                        for j in range(cc.JM):
                            mask[:, :int(G["IMAXJ"][j]), j] = True
                    nstn[cc.stat(Xn[n], ref[n], mask)["status"]] += 1
            print(f"{date} step {s}: numpy batch {tn:.2f} s, JAX {tj:.2f} s{' (includes compile)' if not (s or date != a.dates.split(',')[0]) else ''}; "
                  f"fields JAX!=numpy: {nv}; vs REAL exit: numpy {nstn}, JAX {nst}", flush=True)
            for k, v in worst.items():
                print(f"    {k}: differing cells {v[0]} (bit {v[1]}) max abs {v[2]:.3e}")
            bad_total += nv
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
