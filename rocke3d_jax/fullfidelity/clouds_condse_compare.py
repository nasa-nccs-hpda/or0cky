"""D126: validate clouds_condse_ff.py (CONDSE column driver + the imported MSTCNV and LSCOND ports) against the instrumented real model.

Dumps (clouds_condse_io.py): ffc_cse_in_<itime>.bin (CONDSE entry) + old ffd_<itime>_pre_condse.bin, and ffc_cse_out_<itime>.bin (exit)
+ old ffd_<itime>_post_condse.bin, per date 6 steps (nov26 33312, dec01 33552, jan01 17520), all 3170 columns per step.

Per step the port starts from the entry arrays, runs every column in the Fortran order and the momentum back-transfer, and every exit field is
compared with the real one: `exact` (bitwise), `bound` (every cell |d| <= BOUND_REL * field scale, field scale = max|ref| of the field), or
`FAIL` (first failing cell is printed, 1-based Fortran index order of the dumped array).  Columns are attributed: a column is `inexact` if any
column-local field differs bitwise and `flipped` if any field of the column exceeds the bound (a threshold/branch flip, see the D107-D113
libm discussion).  The reference build uses Intel libimf: `--imf` routes exp/pow through libimf (needs the Intel runtime) and is expected to
be bitwise; without it numpy/glibc libm is used and threshold flips are reported, not hidden.

Checks (besides the exit fields):
  random : RANDU stream from the recorded seed vs RNDSS (L<=LMCLD) and the post-draw seed
  setup  : MSTCNV inputs built by the port vs the recorded MSTCNV boundary records (ffc_mc_cols) of the same step, exact
  lsin   : LSCOND inputs/parameters built by the port vs the recorded LSCOND boundary records (ffc_ls_bnd) of the same step, exact
Usage: python3 clouds_condse_compare.py [--imf] [--dates nov26,dec01,jan01] [--steps 0,1,..] [--cols-j J0:J1] [--json out.json]
"""
import argparse
import json
import sys
import time

import numpy as np

import clouds_condse_ff as cf
import clouds_condse_io as cio
import clouds_lscond_io as lio
import clouds_mstcnv_io as mio

BOUND_REL = 1e-12
LM, IM, JM = 40, 72, 46
FIELDS_LIJ = ("TTOLD QTOLD SVLHX SVLAT RHSAV CLDSAV CLDSAV1 FSS TAUSS TAUSSIP TAUMC CLDSS CLDMC CSIZMC CSIZSS CSIZSSIP QLSS QISS QLMC QIMC "
              "W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE DIM_ST_WATER "
              "DIM_ST_ICE DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV").split()
FIELDS_IJL = "T Q QCL QCI U V TLS QLS TMC QMC".split()
FIELDS_IJ = "P_ACC PM_ACC PREC EPREC PRECSS DDM1 DDMS TDN1 QDN1 DDML AIRX".split()
FIELDS_OTHER = "TMOM QMOM SNOAGE LMC UKM VKM UKMSP VKMSP UKMNP VKMNP".split()
ALL_FIELDS = FIELDS_IJL + FIELDS_OTHER[:2] + FIELDS_LIJ + FIELDS_IJ + FIELDS_OTHER[2:] + ["UALIJ", "VALIJ"]


def stat(a, b, mask=None):
    """-> dict(n, nexact, maxabs, scale, status, first, nbad). mask: boolean array same shape (cells to compare)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if mask is None:
        mask = np.ones(a.shape, bool)
    d = np.abs(a - b)
    ok_exact = (a == b) | ~mask
    scale = float(np.max(np.abs(b[mask]))) if mask.any() else 0.0
    tol = BOUND_REL * max(scale, 1e-300)
    ok_bound = (d <= tol) | ~mask
    nex = int(ok_exact.sum() - (~mask).sum())
    out = dict(n=int(mask.sum()), nexact=nex, maxabs=float(d[mask].max()) if mask.any() else 0.0, scale=scale)
    if ok_exact.all():
        out.update(status="exact", first=None, nbad=0)
    elif ok_bound.all():
        out.update(status="bound", first=None, nbad=0)
    else:
        idx = np.argwhere(~ok_bound)
        first = tuple(int(x) + 1 for x in idx[0])
        out.update(status="FAIL", first=first, nbad=int(len(idx)))
    out["ninexact"] = int((~ok_exact).sum())
    return out


def colmaps(X, ref, G):
    """(IM,JM) booleans: columns with any inexact cell / any cell beyond the bound, over the column-local cloud/thermo fields."""
    inex = np.zeros((IM, JM), bool)
    flip = np.zeros((IM, JM), bool)
    for n in FIELDS_IJL[:0] + ["T", "Q", "QCL", "QCI", "TMC", "QMC"]:
        a, b = X[n], ref[n]
        d = np.abs(a - b)
        tol = BOUND_REL * max(float(np.abs(b).max()), 1e-300)
        inex |= (a != b).any(axis=2)
        flip |= (d > tol).any(axis=2)
    for n in FIELDS_LIJ + ["TMOM", "QMOM"]:
        a, b = X[n], ref[n]
        d = np.abs(a - b)
        tol = BOUND_REL * max(float(np.abs(b).max()), 1e-300)
        ax = tuple(k for k in range(a.ndim) if k not in ((1, 2) if n not in ("TMOM", "QMOM") else (1, 2)))
        inex |= (a != b).any(axis=ax)
        flip |= (d > tol).any(axis=ax)
    for n in ("PREC", "EPREC", "PRECSS", "AIRX", "DDML", "DDM1", "DDMS", "TDN1", "QDN1"):
        a, b = X[n], ref[n]
        tol = BOUND_REL * max(float(np.abs(b).max()), 1e-300)
        inex |= a != b
        flip |= np.abs(a - b) > tol
    return inex, flip


def check_random(inp, cfg):
    G = cfg["geom"]
    rn, seed = cf.randu_stream(inp["SEEDS"][0], [int(x) for x in G["IMAXJ"]], cfg["lmcld"])
    L = cfg["lmcld"]
    ref = inp["RNDSS"]
    ok = True
    for j in range(JM):
        n = int(G["IMAXJ"][j])
        ok &= bool(np.array_equal(rn[:, :L, :n, j], ref[:, :L, :n, j]))
    return dict(rndss_exact=ok, seed_exact=(seed == int(inp["SEEDS"][1])))


def check_setup(trace, cols, itime):
    """MSTCNV inputs of traced columns vs the recorded ffc_mc_cols records of this step."""
    res = dict(n=0, bad=0, fields={})
    if cols is None:
        return res
    h, I = cols["hdr"], cols["inp"]
    rows = np.where(h["itime"] == itime)[0]
    for k in rows:
        key = (int(h["i"][k]) - 1, int(h["j"][k]) - 1)
        if key not in trace:
            continue
        r = trace[key]["r"]
        res["n"] += 1
        for name in I:
            a = np.asarray(r[name], float)
            b = np.asarray(I[name][k], float)
            if name in ("ra", "um", "vm", "u0", "v0"):
                continue
            if not np.array_equal(a.reshape(b.shape), b):
                res["bad"] += 1
                res["fields"][name] = res["fields"].get(name, 0) + 1
    return res


def check_lsin(trace, bnd, itime):
    """LSCOND inputs/parameters of traced columns vs recorded ffc_ls_bnd records (entry arrays)."""
    res = dict(n=0, bad=0, fields={})
    if bnd is None:
        return res
    rows = np.where(bnd["itime"] == itime)[0]
    names_in = lio.IN_LM
    for k in rows:
        key = (int(bnd["i"][k]) - 1, int(bnd["j"][k]) - 1)
        if key not in trace or "S0" not in trace[key]:
            continue
        S0, P0 = trace[key]["S0"], trace[key]["P0"]
        res["n"] += 1
        for nm in names_in:
            if not np.array_equal(np.asarray(S0[nm], float), bnd[nm][k]):
                res["bad"] += 1
                res["fields"][nm] = res["fields"].get(nm, 0) + 1
        for nm in ("tl", "ql", "th", "rh", "qclx", "qcix", "svlhxl", "cldsavl"):
            if not np.array_equal(np.asarray(S0[nm], float), bnd["in_" + nm][k]):
                res["bad"] += 1
                res["fields"]["in_" + nm] = res["fields"].get("in_" + nm, 0) + 1
        for nm in ("wconst", "scdncw", "scdnci", "wmui", "bybr", "pearth", "dcl", "lmcld", "kmax", "rimax", "rwmax", "rwcldox", "rcldlx", "rcldix",
                   "cmx", "u00a", "u00b", "rtemp", "bydtsrc", "dtsrc"):
            if float(P0[nm]) != float(bnd[nm][k]):
                res["bad"] += 1
                res["fields"]["P_" + nm] = res["fields"].get("P_" + nm, 0) + 1
        if not np.array_equal(np.asarray(S0["rndssl"], float)[:int(P0["lmcld"])], bnd["rndssl"][k][:int(P0["lmcld"])]):
            res["bad"] += 1
            res["fields"]["rndssl"] = res["fields"].get("rndssl", 0) + 1
    return res


def run_step(date, itime, imf, jrange=None, ff=cio.FF_DEFAULT, cols_rec=None, bnd=None, cfg=None, ms=None):
    inp, ref = cio.load_step(date, itime, ff)
    if inp is None:
        return None
    if cfg is None:
        cfg = cf.make_cfg(date, ff)
    cfg = dict(cfg)
    trace = {}
    if cols_rec is not None:
        h = cols_rec["hdr"]
        for k in np.where(h["itime"] == itime)[0]:
            trace[(int(h["i"][k]) - 1, int(h["j"][k]) - 1)] = None
    if bnd is not None:
        for k in np.where(bnd["itime"] == itime)[0]:
            trace[(int(bnd["i"][k]) - 1, int(bnd["j"][k]) - 1)] = None
    cfg["trace"] = trace
    G = cfg["geom"]
    order = None
    if jrange is not None:
        order = [(i, j) for j in range(jrange[0], jrange[1]) for i in range(int(G["IMAXJ"][j]))]
    t = time.time()
    X, cnt = cf.condse_step(inp, cfg, cols=order, ms=ms)
    dt = time.time() - t
    ncols = cnt["columns"]
    res = dict(date=date, itime=itime, imf=imf, ncols=ncols, seconds=dt, ms_per_col=1e3 * dt / max(ncols, 1), counts=cnt, fields={})
    res["random"] = check_random(inp, cfg)
    res["setup"] = check_setup({k: v for k, v in trace.items() if v}, cols_rec, itime)
    res["lsin"] = check_lsin({k: v for k, v in trace.items() if v}, bnd, itime)
    imaxj = G["IMAXJ"]
    for n in ALL_FIELDS:
        if n not in X or n not in ref:
            continue
        a, b = X[n], ref[n]
        mask = None
        if n in ("UALIJ", "VALIJ"):
            mask = np.zeros(a.shape, bool)
            for j in range(JM):
                mask[:, :int(imaxj[j]), j] = True
        if jrange is not None:
            m2 = np.zeros(a.shape, bool)
            jax = {"T": 1, "Q": 1, "QCL": 1, "QCI": 1, "TLS": 1, "QLS": 1, "TMC": 1, "QMC": 1, "TMOM": 2, "QMOM": 2, "U": 1, "V": 1}.get(n, a.ndim - 1)
            sl = [slice(None)] * a.ndim
            sl[jax] = slice(*jrange)
            m2[tuple(sl)] = True
            mask = m2 if mask is None else (mask & m2)
        res["fields"][n] = stat(a, b, mask)
    if jrange is None:
        inex, flip = colmaps(X, ref, G)
        total = np.zeros((IM, JM), bool)
        for j in range(JM):
            total[:int(imaxj[j]), j] = True
        res["cols_inexact"] = int((inex & total).sum())
        res["cols_flipped"] = int((flip & total).sum())
        res["cols_total"] = int(total.sum())
        res["cols_flipped_list"] = [(int(i) + 1, int(j) + 1) for i, j in np.argwhere(flip & total)][:50]
    return res


def print_res(r):
    mode = "imf" if r["imf"] else "libm"
    print(f"== {r['date']} itime {r['itime']} [{mode}] columns {r['ncols']}  {r['seconds']:.1f} s ({r['ms_per_col']:.1f} ms/column)")
    rd = r["random"]
    print(f"   random: RNDSS exact {rd['rndss_exact']}, seed exact {rd['seed_exact']}")
    print(f"   setup vs MSTCNV records: {r['setup']['n']} columns, {r['setup']['bad']} field mismatches {r['setup']['fields']}")
    print(f"   lsin vs LSCOND records : {r['lsin']['n']} columns, {r['lsin']['bad']} field mismatches {r['lsin']['fields']}")
    cnt = r["counts"]
    print("   counts: " + " ".join(f"{k}={v}" for k, v in cnt.items()))
    nst = {"exact": 0, "bound": 0, "FAIL": 0}
    for n, s in r["fields"].items():
        nst[s["status"]] += 1
    print(f"   fields: exact {nst['exact']}, bound {nst['bound']}, FAIL {nst['FAIL']}  (of {len(r['fields'])})")
    for n, s in r["fields"].items():
        if s["status"] != "exact":
            extra = f" first failing cell {s['first']} nbad {s['nbad']}" if s["status"] == "FAIL" else ""
            print(f"     {n:15s} {s['status']:5s} inexact {s['ninexact']}/{s['n']} maxabs {s['maxabs']:.3g} (scale {s['scale']:.3g}){extra}")
    if "cols_total" in r:
        print(f"   columns: inexact {r['cols_inexact']}/{r['cols_total']}, flipped (beyond bound) {r['cols_flipped']}"
              f"{'  first: ' + str(r['cols_flipped_list'][:8]) if r['cols_flipped'] else ''}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf", action="store_true")
    ap.add_argument("--dates", default="nov26,dec01,jan01")
    ap.add_argument("--steps", default="0,1,2,3,4,5")
    ap.add_argument("--cols-j", default=None, help="restrict to rows J0:J1 (0-based, python slice); module-state carry is then approximate")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.imf:
        cf.set_backend("imf")
    jr = tuple(int(x) for x in a.cols_j.split(":")) if a.cols_j else None
    allres = []
    for date, it0 in cio.DATES:
        if date not in a.dates.split(","):
            continue
        if not cio.have_dumps(date):
            print(f"{date}: dumps missing, skipped")
            continue
        cols_rec = mio.load_cols(date)
        bnd = lio.load_bnd(date)
        cfg = cf.make_cfg(date)
        ms = {}                     # LSCOND module-array carry from step to step (starts from zeros at the first step of the window)
        for s in (int(x) for x in a.steps.split(",")):
            r = run_step(date, it0 + s, a.imf, jr, cols_rec=cols_rec, bnd=bnd, cfg=cfg, ms=ms)
            if r is None:
                print(f"{date} itime {it0 + s}: dump missing, skipped")
                continue
            print_res(r)
            allres.append(r)
    if a.json:
        json.dump(allres, open(a.json, "w"), default=str)
    bad = sum(1 for r in allres for s in r["fields"].values() if s["status"] == "FAIL")
    print(f"TOTAL steps {len(allres)} failing fields {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
