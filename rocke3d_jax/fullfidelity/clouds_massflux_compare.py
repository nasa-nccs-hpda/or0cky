"""Validate clouds_massflux_ff.py (MASS_FLUX, D93) against the instrumented real model.

Dump: ff_data/<date>/ffc_mf_<itime>.bin, big-endian f8, record = itime,site(=1),ncall, then
  LMIN,LHX,QMO1,QMO2,SMO1,SMO2,SLH,WMDN,WMUP,WMEDG | AIRM(LMIN),AIRM(LMIN+1),BYAM(LMIN),BYAM(LMIN+1),BYAM(LMIN+2),
  SM(LMIN+2),QM(LMIN+2),PLK(LMIN),PLK(LMIN+1),PL(LMIN),PL(LMIN+1) | outputs FPLUME,FMP2,DQSUM, final ITER (9 = loop
  completed without exit), final DMSE1, TNX, QNX.   One call site (CLOUDS2.F90:1056, in MSTCNV).
Every 3rd call per (itime) plus the first two (ffc_h_consts.txt).
"""
import glob
import sys
import numpy as np
import clouds_massflux_ff as mf

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
COLS = ("itime site ncall lmin lhx qmo1 qmo2 smo1 smo2 slh wmdn wmup wmedg airm0 airm1 byam0 byam1 byam2 sm2 qm2 "
        "plk0 plk1 pl0 pl1 fplume fmp2 dqsum iters dmse1 tnx qnx").split()
INPUTS = "lmin lhx qmo1 qmo2 smo1 smo2 slh wmdn wmup wmedg airm0 airm1 byam0 byam1 byam2 sm2 qm2 plk0 plk1 pl0 pl1".split()
OUTS = "fplume fmp2 dqsum iters dmse1 tnx qnx".split()


def load_date(date, ff=FF_DEFAULT):
    files = sorted(glob.glob(f"{ff}/{date}/ffc_mf_[0-9]*.bin"))
    if not files:
        return None
    parts = []
    for f in files:
        raw = np.fromfile(f, dtype=">f8")
        assert raw.size % len(COLS) == 0, (f, raw.size)
        parts.append(raw.reshape(-1, len(COLS)).astype(np.float64))
    r = np.concatenate(parts)
    return {c: r[:, i] for i, c in enumerate(COLS)}


def run_port(d, **kw):
    return mf.mass_flux(*[d[k] for k in INPUTS], **kw)


def compare(d, res=None):
    res = res if res is not None else run_port(d)
    out = {}
    for k in OUTS:
        a, b = np.asarray(res[k], float), d[k]
        if k in ("tnx", "qnx"):
            # TNX/QNX are module scalars: they hold a STALE value from an earlier call when no iteration of this
            # call executed the DQSUM>0 block (the port returns NaN there); compare only where the port set them.
            m = np.isfinite(a)
            a, b = a[m], b[m]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        fin = np.isfinite(a) & np.isfinite(b)
        out[k] = dict(n=a.size, bitwise=int(same.sum()), max_abs=float(np.max(np.abs(a - b)[fin])) if fin.any() else 0.0,
                      max_rel=float(np.max((np.abs(a - b) / np.maximum(np.abs(b), 1e-300))[fin])) if fin.any() else 0.0)
    return out


def branches(d):
    it = d["iters"].astype(int)
    return dict(n=it.size, exit_iter={int(k): int((it == k).sum()) for k in range(1, 10)},
                dqsum_pos_final=int((d["dqsum"] > 0).sum()), fplume_le_001=int((d["fplume"] <= 0.001).sum()),
                fplume_min=float(d["fplume"].min()), fplume_max=float(d["fplume"].max()),
                lhx_is_lhe=int((d["lhx"] == 2.5e6).sum()), no_exit_8=int((it == 9).sum()))


if __name__ == "__main__":
    ok_all = True
    for date, _ in DATES:
        d = load_date(date)
        if d is None:
            print(f"{date}: no dumps")
            continue
        res = run_port(d)
        st = compare(d, res)
        # exact decision-flow check: iteration count of exit and final-branch flag must agree for every record
        it_mis = int((res["iters"] != d["iters"]).sum())
        pos_mis = int(((res["dqsum"] > 0) != (d["dqsum"] > 0)).sum())
        fpl_mis = int(((res["fplume"] <= 0.001) != (d["fplume"] <= 0.001)).sum())
        print(f"== {date}: {d['lmin'].size} records; iteration-count mismatches={it_mis}; dqsum>0 flips={pos_mis}; "
              f"FPLUME<=.001 (caller's cycle test) flips={fpl_mis}")
        for k, r in st.items():
            good = (r["max_rel"] < 1e-9) if k not in ("iters", "dmse1") else (r["bitwise"] == r["n"] if k == "iters" else r["max_abs"] < 1e-12)
            ok_all &= good
            print(f"    {k:7s} n={r['n']} bitwise={r['bitwise']} ({100.0*r['bitwise']/r['n']:.2f}%) max_abs={r['max_abs']:.2e} "
                  f"max_rel={r['max_rel']:.2e} [{'OK' if good else 'FAIL'}]")
        ok_all &= (it_mis == 0 and pos_mis == 0 and fpl_mis == 0)
        print("  branches:", branches(d))
    print("ALL MATCH" if ok_all else "MISMATCH FOUND")
    sys.exit(0 if ok_all else 1)
