"""D130: where the time goes in the CONDSE per-column chain (clouds_condse_ff.py) on one real step.

Two measurements on the same real step (default nov26 step index 1 = itime 33313, all 3,170 columns, libm mode):
  1. a wall-clock split with light timers wrapped around the validated column ports (mstcnv_column, lscond_main / lscond_ctei /
     lscond_tail, the radiation hand-off, the remaining glue = column total minus those), no profiler overhead;
  2. a cProfile of a slice of latitude rows (--prof-rows J0:J1) for the top functions.
With --dump PATH the per-column inputs/outputs (trace: MSTCNV inputs r, outputs o, LSCOND inputs S0, P0 for every column) and the
per-column exit state X are pickled for the batch compare scripts (they are the per-column reference of D131/D132).
Usage: python3 clouds_condse_profile.py [--date nov26] [--step 1] [--imf] [--prof-rows 20:24] [--dump out.pkl]
"""
import argparse
import cProfile
import io
import pickle
import pstats
import time

import numpy as np

import clouds_condse_ff as cf
import clouds_condse_io as cio
import clouds_lscond_ff as ls0
import clouds_mstcnv_ff as mc

T = {}


def _wrap(mod, name, key):
    f = getattr(mod, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[key] = T.get(key, 0.0) + time.perf_counter() - t
    setattr(mod, name, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="nov26")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--imf", action="store_true")
    ap.add_argument("--prof-rows", default="20:24")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    if a.imf:
        cf.set_backend("imf")
    it0 = dict(cio.DATES)[a.date]
    inp, ref = cio.load_step(a.date, it0 + a.step)
    cfg = cf.make_cfg(a.date)
    G = cfg["geom"]
    # one allk module for the poles must exist before wrapping
    allk = cf._lscond_allk()
    _wrap(mc, "mstcnv_column", "MSTCNV")
    for m in (ls0, allk):
        _wrap(m, "lscond_main", "LSCOND_main")
        _wrap(m, "lscond_ctei", "LSCOND_ctei")
        _wrap(m, "lscond_tail", "LSCOND_tail")
    _wrap(cf, "_hand_off", "hand_off")
    trace = {(i, j): None for j in range(cf.JM) for i in range(int(G["IMAXJ"][j]))} if a.dump else {}
    cfg2 = dict(cfg)
    cfg2["trace"] = trace
    t = time.perf_counter()
    X, cnt = cf.condse_step(inp, cfg2, with_momentum=False)
    tot = time.perf_counter() - t
    n = cnt["columns"]
    print(f"{a.date} step {a.step} ({'imf' if a.imf else 'libm'}): {n} columns, column loop {tot:.1f} s = {1e3 * tot / n:.1f} ms/column")
    other = tot - sum(T.values())
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:13s} {v:8.2f} s  {100 * v / tot:5.1f} %  {1e3 * v / n:6.2f} ms/column")
    print(f"  {'other glue':13s} {other:8.2f} s  {100 * other / tot:5.1f} %  {1e3 * other / n:6.2f} ms/column  (column set-up, LSCOND set-up lists, stores, merge)")
    print(f"  convecting columns (lmcmin>0): {cnt['convecting']} of {n}")
    # restore the unwrapped functions for the cProfile slice is not needed: the wrappers only add two perf_counter calls
    j0, j1 = (int(x) for x in a.prof_rows.split(":"))
    order = [(i, j) for j in range(j0, j1) for i in range(int(G["IMAXJ"][j]))]
    pr = cProfile.Profile()
    pr.enable()
    cf.condse_step(inp, dict(cfg), cols=order)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    print(f"cProfile of rows {j0}:{j1} ({len(order)} columns), sorted by own time:")
    print("\n".join(s.getvalue().splitlines()[:34]))
    if a.dump:
        pickle.dump(dict(trace=trace, X=X, cnt=cnt, date=a.date, step=a.step), open(a.dump, "wb"), protocol=4)
        print("dumped", a.dump)


if __name__ == "__main__":
    main()
