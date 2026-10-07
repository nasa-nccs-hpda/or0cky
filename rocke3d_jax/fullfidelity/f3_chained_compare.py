"""D172: compare a saved chained-window accumulation (f3_diagnostics2.run_chained_window(..., out_npz=...)) with the real 54-step accumulation,
column by column.  usage: python f3_chained_compare.py chained54.npz [out.json]
Columns: every column the chained run accumulated.  Per column: max|ours-real|/max|real| (scale of the real increment), global-mean
(area weighted, cos-lat proxy) relative difference of the accumulated field, and pass/fail at the D163 tolerance 1e-12 (never loosened)."""
import json
import sys

import numpy as np

import f3_diagnostics as f3
import f3_diagnostics2 as d2

TOL = 1e-12


def compare_chained(npz, real_path=f3.REAL_ACC):
    z = np.load(npz)
    a, cols = z["aij"], [int(c) for c in z["cols"]]
    real = np.load(real_path)["daij"]
    meta = f3.aij_names_from_nc("/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc")
    w = np.cos(np.deg2rad(np.linspace(-90, 90, 46)))[:, None] * np.ones((1, 72))
    rows = []
    for c in cols:
        r, o = real[c - 1], a[c - 1]
        sc = np.abs(r).max()
        d = np.abs(o - r).max()
        gr, go = (r * w).sum() / w.sum(), (o * w).sum() / w.sum()
        rows.append(dict(col=c, name=meta[c]["name"].strip(), real_scale=float(sc), maxabs=float(d), rel=float(d / sc) if sc > 0 else (0.0 if d == 0 else float("inf")),
                         gmean_real=float(gr), gmean_ours=float(go), gmean_rel=float((go - gr) / abs(gr)) if gr != 0 else float(go - gr),
                         within_tol=bool(d <= TOL * sc if sc > 0 else d == 0)))
    return rows


if __name__ == "__main__":
    rows = compare_chained(sys.argv[1])
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], "w"), indent=1)
    n = len(rows)
    ok = sum(r["within_tol"] for r in rows)
    print(f"{n} columns, {ok} within {TOL}")
    for r in sorted(rows, key=lambda r: r["rel"]):
        print(f"{r['col']:5d} {r['name']:16s} rel {r['rel']:.3g}  gmean rel {r['gmean_rel']:.3g}")
