"""Validate clouds_lscond_ff.py (D108 LSCOND main layer loop, D109 CTEI + remainder = whole LSCOND column call) against the
instrumented real model.  Dumps (layouts in clouds_lscond_io.py):
  ffc_ls_bnd_<itime>.bin  LSCOND entry (all inputs + module state) and exit (state + scalars), every 10th column call
  ffc_ls_mid_<itime>.bin  state and locals after the main L loop, before CTEI, same calls (row-aligned with bnd)
  ffc_ls_tail_<itime>.bin entry of the particle size / optical thickness block (= state after CTEI), every 4th call

Checks:
  main  : port(main loop) from the call-entry state  vs the mid checkpoint (state + locals + PRCPSS, HCNDSS)
  full  : port(main + CTEI + tail) from the entry state  vs the call exit state
  ctei  : port(CTEI) from the mid checkpoint  vs the tail-block entry, on the calls present in both files
  chain : port(CTEI + tail) from the mid checkpoint  vs the call exit state
Usage: python3 clouds_lscond_compare.py [--imf] [--max N]
"""
import sys
import numpy as np
import clouds_lscond_io as io
import clouds_lscond_ff as ls
import clouds_lscond_size_ff as sz

LM = io.LM
PAR_KEYS = "bybr rimax rwmax rwcldox rcldlx rcldix wmui cmx u00a u00b rtemp use_vmp do_blu00 wconst scdncw scdnci bydtsrc dtsrc " \
           "pearth dcl lmcld kmax".split()
INPUT_LM = io.IN_LM
STATE_LM = io.STATE_LM


def column_state(b, r, mid=None, m=None):
    """Build (S, P) lists for row r of the bnd dict `b`; if `mid` (row m) is given, the state is the post-main-loop checkpoint
    (and W is returned as the locals)."""
    S = {k: b[k][r].tolist() for k in INPUT_LM if k not in ("pl",)}
    S["pl"] = b["pl"][r].tolist()
    src, pre = (b, "in_") if mid is None else (mid, "")
    row = r if mid is None else m
    for k in STATE_LM:
        S[k] = src[pre + k][row].tolist()
    S["lhp"] = src[pre + "lhp"][row].tolist()
    S["prebar1"] = src[pre + "prebar1"][row].tolist()
    S["qmom"] = src[pre + "qmom"][row].tolist()
    S["smom"] = src[pre + "smom"][row].tolist()
    S["um"] = src[pre + "um"][row].tolist()
    S["vm"] = src[pre + "vm"][row].tolist()
    S["precnvl"] = b["precnvl"][r].tolist()
    S["rndssl"] = b["rndssl"][r].tolist()
    P = {k: float(b[k][r]) for k in PAR_KEYS}
    P["use_vmp"] = bool(P["use_vmp"])
    P["ra"] = b["ra"][r].tolist()
    P["dcl"], P["lmcld"], P["kmax"] = int(P["dcl"]), int(P["lmcld"]), int(P["kmax"])
    return S, P


def locals_from_mid(mid, m):
    W = {k: mid[k][m].tolist() for k in io.LOC_LM}
    W["prebar"] = mid["prebar"][m].tolist()
    W["preice"] = mid["preice"][m].tolist()
    W["prcpss"] = float(mid["prcpss"][m])
    W["hcndss"] = float(mid["hcndss"][m])
    W["ckij"] = 1.0
    return W


SQRT_AMP = ("cldsavl", "cldssl", "loc_cleara", "loc_rhf", "cldsal", "cldsv1", "taussl", "tausslip", "csizel", "csizelip",
            "wmsum")   # fields downstream of CLEARA = DSQRT(1-RH ...): see cleara_sqrt_explained


def stat(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    eq = a == b
    ad = np.abs(a - b)
    sc = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    return dict(n=int(a.size), bitwise=int(eq.sum()), max_abs=float(ad.max()) if a.size else 0.0,
                max_rel=float((ad / sc).max()) if a.size else 0.0)


def _arr(S, k):
    return np.array(S[k], dtype=float)


def compare_state(res, ref, pre="", names=None, lmcld=None):
    """res: list of port S dicts; ref: dict of (N,...) arrays with prefix `pre`.  -> {name: stat}"""
    out = {}
    for k in (names or STATE_LM):
        a = np.array([_arr(S, k) for S in res])
        out[k] = stat(a, ref[pre + k][:len(res)])
    for k in ("lhp", "prebar1"):
        a = np.array([_arr(S, k) for S in res])
        out[k] = stat(a, ref[pre + k][:len(res)])
    for k in ("qmom", "smom", "um", "vm"):
        a = np.array([np.array(S[k]) for S in res])
        out[k] = stat(a, ref[pre + k][:len(res)])
    return out


def run_main(b, mid, n):
    Ss, Ws, cnt = [], [], ls.new_counters()
    for r in range(n):
        S, P = column_state(b, r)
        W = ls.lscond_main(S, P, cnt)
        Ss.append(S)
        Ws.append(W)
    return Ss, Ws, cnt


def run_full(b, n):
    Ss, Ws, cnt = [], [], ls.new_counters()
    for r in range(n):
        S, P = column_state(b, r)
        W = ls.lscond_main(S, P, cnt)
        ls.lscond_ctei(S, W, P, cnt)
        ls.lscond_tail(S, W, P, cnt)
        Ss.append(S)
        Ws.append(W)
    return Ss, Ws, cnt


def run_ctei_from_mid(b, mid, n, tail=True):
    Ss, Ws, cnt = [], [], ls.new_counters()
    for r in range(n):
        S, P = column_state(b, r, mid, r)
        W = locals_from_mid(mid, r)
        ls.lscond_ctei(S, W, P, cnt)
        Ss.append(S)
        Ws.append(W)
        if tail:
            ls.lscond_tail(S, W, P, cnt)
    return Ss, Ws, cnt


def compare_main(b, mid, Ss, Ws):
    st = compare_state(Ss, mid)
    lm = b["lmcld"][:len(Ss)].astype(int)
    for k in io.LOC_LM:
        a = np.array([_arr(W, k) for W in Ws])
        ref = mid[k][:len(Ss)]
        if k in ("cleara", "rhf", "rh00"):          # only set for L<=LMCLD (the rest is uninitialised Fortran stack)
            a, ref = a[:, :29], ref[:, :29]
        st["loc_" + k] = stat(a, ref)
    for k in ("prebar", "preice"):
        st["loc_" + k] = stat(np.array([W[k] for W in Ws]), mid[k][:len(Ss)])
    st["prcpss"] = stat([W["prcpss"] for W in Ws], mid["prcpss"][:len(Ss)])
    st["hcndss"] = stat([W["hcndss"] for W in Ws], mid["hcndss"][:len(Ss)])
    return st


def compare_exit(b, Ss, Ws):
    st = compare_state(Ss, b, "out_")
    n = len(Ss)
    st["prcpss"] = stat([W["prcpss"] for W in Ws], b["prcpss"][:n])
    st["hcndss"] = stat([W["hcndss"] for W in Ws], b["hcndss"][:n])
    st["wmsum"] = stat([W.get("wmsum", np.nan) for W in Ws], b["wmsum"][:n])
    if all("ierr" in W for W in Ws):
        st["ierr"] = stat([W["ierr"] for W in Ws], b["ierr"][:n])
    return st


def align_mid_tail(mid, tail):
    """index pairs (m, t) of calls (itime,i,j) present in both the mid and tail files."""
    key_t = {(int(a), int(i), int(j)): k for k, (a, i, j) in enumerate(zip(tail["itime"], tail["i"], tail["j"]))}
    pairs = [(m, key_t[k]) for m, k in enumerate(zip(mid["itime"].astype(int), mid["i"].astype(int), mid["j"].astype(int)))
             if k in key_t]
    return pairs


def compare_ctei_vs_tail(b, mid, tail, pairs, Ss, Ws):
    """Ss/Ws: CTEI output for mid rows (tail=False run) restricted to pairs[:,0]."""
    mi = np.array([p[0] for p in pairs])
    ti = np.array([p[1] for p in pairs])
    res = {}
    for k in ("cldssl", "qclx", "qcix", "svlhxl", "tl", "cldsal", "cldsv1", "taussl", "tausslip", "csizel", "csizelip", "qlss", "qiss", "wmpr"):
        a = np.array([_arr(Ss[m], k) for m in mi])
        res[k] = stat(a, tail[k][ti])
    a = np.array([_arr(Ws[m], "cleara")[:29] for m in mi])
    res["cleara"] = stat(a, tail["cleara"][ti][:, :29])
    res["ckij"] = stat([Ws[m]["ckij"] for m in mi], tail["ckij"][ti])
    res["hcndss"] = stat([Ws[m]["hcndss"] for m in mi], tail["hcndss"][ti])
    return res


def cleara_sqrt_explained(mid, Ss, Ws):
    """CLEARA = DSQRT((1-RH)/((1-RH00)+teeny)) amplifies a last-bit difference of RH when RH is close to 1: d(CLEARA) =
    d(RH) / (2 sqrt((1-RH)(1-RH00))).  Every layer (L<=LMCLD) whose post-main-loop CLDSAVL (=1-CLEARA) differs between port and
    real must (a) have a differing RH or RH1 (an upstream last-bit difference, here from exp/pow in QSAT) or sit on the
    RH<=1 / RH>1 edge, and (b) the difference must be within 4x the sqrt-amplified bound.  -> (n_mismatch, n_explained)."""
    n = len(Ss)
    a = np.array([_arr(S, "cldsavl") for S in Ss])
    rhp = np.array([_arr(S, "rh") for S in Ss])
    d = mid["cldsavl"][:n]
    rh = mid["rh"][:n]
    rh00 = mid["rh00"][:n]
    bad = a != d
    drh = np.abs(rhp - rh)
    with np.errstate(all="ignore"):
        bound = 4.0 * drh / (2.0 * np.sqrt(np.maximum(1.0 - rh, 1e-300) * np.maximum(1.0 - rh00, 1e-300)))
    edge = (rh <= 1.0) != (rhp <= 1.0)
    expl = bad & (((drh > 0) & (np.abs(a - d) <= bound + 1e-15)) | edge | (np.abs(1.0 - rh) < 1e-12))
    return int(bad.sum()), int(expl.sum())


def fmt(st, tol):
    bad = []
    lines = []
    for k, r in st.items():
        ok = r["max_rel"] <= tol or (k in SQRT_AMP and r["max_abs"] <= 5e-7)
        if not ok:
            bad.append(k)
        lines.append(f"    {k:14s} n={r['n']:8d} bitwise={r['bitwise']:8d} ({100.0 * r['bitwise'] / max(r['n'], 1):7.3f}%) "
                     f"max_abs={r['max_abs']:.2e} max_rel={r['max_rel']:.2e} [{'OK' if ok else 'FAIL'}]")
    return lines, bad


def flips(b, Ss, Ws):
    """Discrete-decision mismatches between port and real at the call exit (branch flips caused by 1-ulp differences):
    per layer: SVLHXL, LHP, QCLX==0, QCIX==0, DCTEI!=0 (CTEI mixed this layer pair), TAUSSL==0, CLDSSL==0; per call: IERR."""
    n = len(Ss)
    f = {}
    g = lambda k: np.array([_arr(S, k) for S in Ss])  # noqa: E731
    f["svlhxl"] = int((g("svlhxl") != b["out_svlhxl"][:n]).sum())
    f["lhp"] = int((g("lhp") != b["out_lhp"][:n]).sum())
    f["qclx_zero"] = int(((g("qclx") == 0) != (b["out_qclx"][:n] == 0)).sum())
    f["qcix_zero"] = int(((g("qcix") == 0) != (b["out_qcix"][:n] == 0)).sum())
    f["ctei_mixed_layer"] = int(((g("dctei") != 0) != (b["out_dctei"][:n] != 0)).sum())
    f["taussl_zero"] = int(((g("taussl") == 0) != (b["out_taussl"][:n] == 0)).sum())
    f["cldssl_zero"] = int(((g("cldssl") == 0) != (b["out_cldssl"][:n] == 0)).sum())
    f["ierr"] = int((np.array([W["ierr"] for W in Ws]) != b["ierr"][:n]).sum())
    return f


if __name__ == "__main__":
    imf = "--imf" in sys.argv
    nmax = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None
    print("libimf mode:", sz.use_imf(imf) if imf else False)
    ok_all = True
    TOL = 1e-6
    quiet = "--quiet" in sys.argv

    def show(title, st):
        global ok_all
        lines, bad = fmt(st, TOL)
        tb = sum(r["bitwise"] for r in st.values())
        tn = sum(r["n"] for r in st.values())
        print(f"  {title}: {'FAIL ' + str(bad) if bad else 'OK'}  (all values bitwise {tb}/{tn} = {100.0 * tb / tn:.4f}%; "
              f"worst max_rel {max(r['max_rel'] for r in st.values()):.2e})")
        if not quiet or bad:
            print("\n".join(lines))
        ok_all &= not bad

    for date, _ in io.DATES:
        b, mid, tail = io.load_bnd(date), io.load_mid(date), io.load_tail(date)
        n = b["i"].size if nmax is None else min(nmax, b["i"].size)
        assert np.array_equal(b["itime"], mid["itime"]) and np.array_equal(b["i"], mid["i"]) and np.array_equal(b["j"], mid["j"])
        print(f"== {date}: {n} bnd/mid column calls, {tail['i'].size} tail records")
        Ss, Ws, c1 = run_main(b, mid, n)
        show("D108 main loop (entry -> post-main checkpoint)", compare_main(b, mid, Ss, Ws))
        nb, ne = cleara_sqrt_explained(mid, Ss, Ws)
        print(f"   CLDSAVL mismatches {nb}, of which explained by sqrt amplification of an RH last-bit difference: {ne}")
        ok_all &= (nb == ne)
        Ss, Ws, c2 = run_full(b, n)
        show("D108+D109 whole call (entry -> exit)", compare_exit(b, Ss, Ws))
        print("   branch flips at exit:", flips(b, Ss, Ws))
        Ss, Ws, c3 = run_ctei_from_mid(b, mid, n, tail=False)
        pairs = [p for p in align_mid_tail(mid, tail) if p[0] < n]
        # tail-entry check needs the tail-entry fields only (cleara/ckij/hcndss/tl/qclx/...), run on CTEI output
        res = compare_ctei_vs_tail(b, mid, tail, pairs, Ss, Ws)
        res = {k: v for k, v in res.items() if k in ("cldssl", "qclx", "qcix", "svlhxl", "tl", "cleara", "ckij", "hcndss")}
        show(f"D109 CTEI (mid checkpoint -> tail entry), {len(pairs)} calls present in both files", res)
        Ss, Ws, c4 = run_ctei_from_mid(b, mid, n, tail=True)
        show("D109 chain (mid checkpoint -> CTEI -> tail -> exit)", compare_exit(b, Ss, Ws))
        print("   counters (layer visits, whole call):", dict(c2))
    print("ALL MATCH" if ok_all else "MISMATCH FOUND")
    sys.exit(0 if ok_all else 1)
