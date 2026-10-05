"""Validate clouds_dq_ff.py (get_dq_cond / get_dq_evap) against the instrumented real model (D89).

Dumps: ff_data/<date>/ffc_dq_<itime>.bin, 13-double big-endian records
  itime, site, kind, ncall, sm, qm, plk, mass, lhx, pl, cond, dqsum, f
site 1:MSTCNV updraft cond (CLOUDS2.F90:1364) 2:MSTCNV downdraft evap (2067) 3:MSTCNV precip evap
(2716) 4:LSCOND evap liquid (4002) 5:LSCOND evap ice (4013) 6:LSCOND cond (4425); kind 1=cond 2=evap.
Every FFC_DQ_STRIDE-th call of each (itime,site) is recorded (stride in ffc_dq_consts.txt).

Usage: python3 clouds_dq_compare.py
"""
import glob
import os
import sys
import numpy as np
import clouds_dq_ff as dq

FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
NREC = 13
COLS = ["itime", "site", "kind", "ncall", "sm", "qm", "plk", "mass", "lhx", "pl", "cond", "dqsum", "f"]
SITE_NAMES = {1: "MSTCNV updraft cond", 2: "MSTCNV downdraft evap", 3: "MSTCNV precip evap",
              4: "LSCOND evap liquid", 5: "LSCOND evap ice", 6: "LSCOND cond"}
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]


def read_consts(path):
    out = {}
    for line in open(path):
        k, v = line.split()
        out[k] = float(v)
    return out


def load_step(path):
    raw = np.fromfile(path, dtype=">f8")
    assert raw.size % NREC == 0, raw.size
    r = raw.reshape(-1, NREC).astype(np.float64)
    return {c: r[:, i] for i, c in enumerate(COLS)}


def load_date(date, ff=FF_DEFAULT):
    files = sorted(glob.glob(f"{ff}/{date}/ffc_dq_[0-9]*.bin"))
    parts = [load_step(f) for f in files]
    if not parts:
        return None
    return {c: np.concatenate([p[c] for p in parts]) for c in COLS}


def run_port(d):
    """Apply the port to every record; returns (dqsum, f) arrays."""
    kind = d["kind"]
    out_dq = np.empty_like(d["sm"])
    out_f = np.empty_like(d["sm"])
    c = kind == 1
    e = kind == 2
    out_dq[c], out_f[c] = dq.get_dq_cond(d["sm"][c], d["qm"][c], d["plk"][c], d["mass"][c],
                                         d["lhx"][c], d["pl"][c])
    out_dq[e], out_f[e] = dq.get_dq_evap(d["sm"][e], d["qm"][e], d["plk"][e], d["mass"][e],
                                         d["lhx"][e], d["pl"][e], d["cond"][e])
    return out_dq, out_f


def stats(d):
    """Per-site comparison statistics."""
    mdq, mf = run_port(d)
    rows = {}
    for s in sorted(set(d["site"].astype(int))):
        m = d["site"] == s
        a, b = mdq[m], d["dqsum"][m]
        fa, fb = mf[m], d["f"][m]
        absd = np.abs(a - b)
        # relative to the active scale of the record (dqsum is clamped to qm or cond)
        scale = np.where(m, np.where(d["kind"] == 1, d["qm"], d["cond"]), 1.0)[m]
        rel = absd / np.maximum(scale, 1e-300)
        fabs = np.abs(fa - fb)
        rows[s] = dict(n=int(m.sum()), nonzero=int((b > 0).sum()), exact=int((a == b).sum()),
                       f_exact=int((fa == fb).sum()),
                       max_abs=float(absd.max()), max_rel_scale=float(rel.max()),
                       max_f_abs=float(fabs.max()),
                       n_branch_mismatch=int(((a > 0) != (b > 0)).sum()),
                       n_active=int((scale > 0).sum()),
                       n_clamped_hi=int((b == scale)[scale > 0].sum()))
    return rows


if __name__ == "__main__":
    ok_all = True
    try:
        c = read_consts(f"{FF_DEFAULT}/nov26/ffc_dq_consts.txt")
        print("constants (port vs real): bysha %.17e %.17e | mrat %.17e %.17e | rvap %.17e %.17e | tf %.17e %.17e"
              % (dq.BYSHA, c["bysha"], dq.MRAT, c["mrat"], dq.RVAP, c["rvap"], dq.TF, c["tf"]))
    except OSError:
        pass
    for date, _ in DATES:
        d = load_date(date)
        if d is None:
            print(f"{date}: no dumps")
            continue
        print(f"== {date}: {d['sm'].size} records ==")
        for s, r in stats(d).items():
            ok = r["max_rel_scale"] < 1e-12 and r["n_branch_mismatch"] == 0
            ok_all &= ok
            print(f"  site {s} {SITE_NAMES[s]:<22} n={r['n']:6d} active={r['n_active']:6d} dq>0={r['nonzero']:6d} "
                  f"bitwise={r['exact']:6d} max_abs={r['max_abs']:.2e} max_rel={r['max_rel_scale']:.2e} "
                  f"f_bitwise={r['f_exact']:6d} max_f_abs={r['max_f_abs']:.2e} branch_mismatch={r['n_branch_mismatch']} "
                  f"[{'OK' if ok else 'FAIL'}]")
    print("ALL MATCH" if ok_all else "MISMATCH FOUND")
    sys.exit(0 if ok_all else 1)
