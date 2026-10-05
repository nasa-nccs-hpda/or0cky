"""Validate clouds_lscond_size_ff.py (D107, LSCOND particle size / optical thickness tail block) against the
instrumented real model: ff_data/<date>/ffc_ls_tail_<itime>.bin (entry and exit of the block, one record per sampled
column call, every 4th LSCOND call of the step; layout in clouds_lscond_io.py).

Usage: python3 clouds_lscond_size_compare.py [--imf]      (--imf: route exp/pow through Intel libimf if present)
"""
import sys
import numpy as np
import clouds_lscond_io as io
import clouds_lscond_size_ff as sz

OUT_NAMES = io.TAIL_OUT_LM


def run_port(d, use_vmp=True, mutate=None, nmax=None):
    """Run the port on every record of a loaded tail dict -> (outputs dict of (N,LM) arrays, wmsum (N,), counters)."""
    n = d["i"].size if nmax is None else min(nmax, d["i"].size)
    out = {k: np.zeros((n, io.LM)) for k in OUT_NAMES}
    wm = np.zeros(n)
    tot = {}
    for r in range(n):
        a = {k: d[k][r].tolist() for k in io.TAIL_IN_LM}
        a["lhp"] = d["lhp"][r].tolist()
        par = {k: float(d[k][r]) for k in io.TAIL_SC}
        if mutate is not None:
            mutate(a, par)
        s, w, cnt = sz.size_tail(a, par, use_vmp=use_vmp)
        for k in OUT_NAMES:
            out[k][r] = s[k]
        wm[r] = w
        for k, v in cnt.items():
            tot[k] = tot.get(k, 0) + v
    return out, wm, tot


def _stat(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    eq = a == b
    bad = ~eq
    ad = np.abs(a - b)
    with np.errstate(all="ignore"):
        rel = np.where(bad, ad / np.maximum(np.abs(b), 1e-300), 0.0)
    return dict(n=int(a.size), bitwise=int(eq.sum()), max_abs=float(ad.max()) if a.size else 0.0,
                max_rel=float(rel.max()) if a.size else 0.0)


def compare(d, res):
    out, wm, _ = res
    n = wm.size
    st = {k: _stat(out[k], d["out_" + k][:n]) for k in OUT_NAMES}
    st["wmsum"] = _stat(wm, d["out_wmsum"][:n])
    return st


def mismatch_records(d, res):
    out, wm, _ = res
    n = wm.size
    bad = np.zeros(n, bool)
    for k in OUT_NAMES:
        bad |= (out[k] != d["out_" + k][:n]).any(axis=1)
    bad |= wm != d["out_wmsum"][:n]
    return bad


def branch_flips(d, res):
    """Records where a threshold-test outcome visible in the outputs differs between port and real."""
    out, _, _ = res
    n = out["svlhxl"].shape[0]
    f = {}
    f["svlhxl_reset"] = int(((out["svlhxl"] == 0) != (d["out_svlhxl"][:n] == 0)).sum())
    f["taussl_zero"] = int(((out["taussl"] == 0) != (d["out_taussl"][:n] == 0)).sum())
    f["taussl_cap100"] = int(((out["taussl"] == 100.0) != (d["out_taussl"][:n] == 100.0)).sum())
    f["tausslip_zero"] = int(((out["tausslip"] == 0) != (d["out_tausslip"][:n] == 0)).sum())
    f["cldssl_zero"] = int(((out["cldssl"] == 0) != (d["out_cldssl"][:n] == 0)).sum())
    f["qclx_zero"] = int(((out["qclx"] == 0) != (d["out_qclx"][:n] == 0)).sum())
    return f


def const_check(date, ff=io.FF_DEFAULT):
    c = io.read_consts(f"{ff}/{date}/ffc_ls_consts.txt")
    return dict(bygrav=sz.BYGRAV == c["bygrav"], teeny=sz.TEENY == c["teeny"], by3=sz.BY3 == c["by3"],
                twopi=sz.TWOPI == c["twopi"], rgas=sz.RGAS == c["rgas"], lhe=sz.LHE == c["lhe"], lhs=sz.LHS == c["lhs"])


if __name__ == "__main__":
    imf = "--imf" in sys.argv
    mode = sz.use_imf(imf) if imf else False
    print("libimf mode:", mode, "(requested)" if imf else "")
    ok_all = True
    for date, _ in io.DATES:
        d = io.load_tail(date)
        if d is None:
            print(f"{date}: no dumps")
            continue
        res = run_port(d)
        st = compare(d, res)
        bad = mismatch_records(d, res)
        print(f"== {date}: {d['i'].size} tail records (columns); constants equal: {const_check(date)}")
        for k, r in st.items():
            ok = r["max_rel"] < 1e-12
            ok_all &= ok
            print(f"    {k:9s} n={r['n']} bitwise={r['bitwise']} ({100.0 * r['bitwise'] / r['n']:.3f}%) "
                  f"max_abs={r['max_abs']:.2e} max_rel={r['max_rel']:.2e} [{'OK' if ok else 'FAIL'}]")
        print(f"  records with any mismatch: {int(bad.sum())}; branch flips: {branch_flips(d, res)}")
        print(f"  branch counters (layer visits): {res[2]}")
    print("ALL MATCH" if ok_all else "MISMATCH FOUND")
    sys.exit(0 if ok_all else 1)
