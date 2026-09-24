"""Field-by-field comparison of two ModelE restart (fort.N.nc) files.

Usage: python compare_restarts.py A.nc B.nc [--json out.json] [--top N]

For every shared numeric variable reports max|diff|, RMS diff, RMS of the
reference (A), relative RMS diff, and where the max occurs. Also flags
*vacuous* fields (reference is all one value) so a "match" on them is not
counted as evidence -- guards against the rigged-validation failure mode
recorded in STATUS.md.
"""
import argparse, json, sys
import numpy as np
import netCDF4 as nc


def compare(pa, pb):
    a, b = nc.Dataset(pa), nc.Dataset(pb)
    rows = []
    for name, va in a.variables.items():
        if name not in b.variables or va.dtype.kind not in "fi":
            continue
        x = np.ma.filled(va[:], np.nan).astype(np.float64)
        y = np.ma.filled(b.variables[name][:], np.nan).astype(np.float64)
        if x.shape != y.shape:
            rows.append(dict(name=name, error=f"shape {x.shape} vs {y.shape}"))
            continue
        d = y - x
        fin = np.isfinite(d)
        n_nonfinite_mismatch = int((~fin & ~(np.isnan(x) & np.isnan(y))).sum())
        dd = np.where(fin, d, 0.0)
        xx = np.where(np.isfinite(x), x, 0.0)
        rms_ref = float(np.sqrt(np.mean(xx ** 2))) if x.size else 0.0
        rms_d = float(np.sqrt(np.mean(dd ** 2))) if x.size else 0.0
        amax = float(np.abs(dd).max()) if x.size else 0.0
        loc = tuple(int(i) for i in np.unravel_index(np.abs(dd).argmax(), dd.shape)) if x.size and dd.ndim else ()
        rows.append(dict(
            name=name, shape=list(x.shape), max_abs=amax, rms_diff=rms_d, rms_ref=rms_ref,
            rel_rms=(rms_d / rms_ref if rms_ref > 0 else (0.0 if rms_d == 0 else float("inf"))),
            at=loc, identical=bool(amax == 0 and n_nonfinite_mismatch == 0),
            nonfinite_mismatch=n_nonfinite_mismatch,
            vacuous=bool(x.size > 1 and np.nanmax(x) == np.nanmin(x)),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--json"); ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    rows = compare(args.a, args.b)
    ok = [r for r in rows if "error" not in r]
    ident = [r for r in ok if r["identical"]]
    vac = [r for r in ok if r["vacuous"]]
    print(f"{len(ok)} numeric variables compared; {len(ident)} bitwise identical; "
          f"{len(vac)} vacuous (constant reference, uninformative)")
    print(f"{'variable':22s} {'max|d|':>12s} {'rms d':>12s} {'rel rms':>10s}  at")
    for r in sorted(ok, key=lambda r: -r["rel_rms"])[: args.top]:
        print(f"{r['name']:22s} {r['max_abs']:12.4e} {r['rms_diff']:12.4e} {r['rel_rms']:10.2e}  {r['at']}")
    for r in rows:
        if "error" in r:
            print("ERROR", r)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
