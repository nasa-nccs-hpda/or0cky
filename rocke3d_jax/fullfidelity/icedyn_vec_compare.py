"""Validate icedyn_vec (D87) against the scalar icedyn_dynsi_ff port on real recorded ice-dynamics
inputs (ffy_<itime>_in.bin) and against the real Fortran outputs (ffy_<itime>_out.bin, same comparison
as dynsi_compare.compare_one). Prints worst differences and scalar -> batched timings.

Usage: python icedyn_vec_compare.py [date itime]   (no args: first record of every date)
"""
import sys
import time
import numpy as np

import icedyn_dynsi_ff as D
import icedyn_vec as V
import dynsi_compare as C
from icedyn_geom_ff import geomicdyn, icdyn_masks
from icedyn_geom_compare import read_geom

FF_DATA = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = {"nov26": 33312, "dec01": 33552, "jan01": 17520}
DTS = 1800.0
OIPHI = np.deg2rad(25.0)
SINWAT, COSWAT = np.sin(OIPHI), np.cos(OIPHI)
NX1, NY1, IMICDYN = C.NX1, C.NY1, C.IMICDYN


def setup_geometry(date):
    gd = read_geom(f"{FF_DATA}/{date}/ffz_geom.bin")
    radius = gd["dxt"][1] / (2.0 * np.pi / gd["imicdyn"])
    D.RADIUS = radius
    D.BYRAD2 = 1.0 / (radius * radius)
    geom = geomicdyn(gd["imicdyn"], gd["jmicdyn"], radius)
    heffm, uvm = icdyn_masks(gd["gfocean"], gd["nx1"], gd["jmicdyn"])
    D.init_geometry(geom, heffm, uvm)


def rel(a, b):
    """max|a-b| / max|b| (scale-relative worst difference)."""
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)


def usi_vsi(din):
    usi0 = np.zeros((IMICDYN, NY1))
    vsi0 = np.zeros((IMICDYN, NY1))
    usi0[:, :] = din["uice0"][2:NX1, 1:NY1 + 1]
    vsi0[:, :] = din["vice0"][2:NX1, 1:NY1 + 1]
    return usi0, vsi0


def stage_checks(date, itime, verbose=True):
    """Per-function worst differences vs the scalar port; returns dict name -> (worst, t_scalar, t_vec)."""
    setup_geometry(date)
    din = C.read_dynsi_in(f"{FF_DATA}/{date}/ffy_{itime}_in.bin")
    out = {}
    u1, v1 = din["uice0"], din["vice0"]
    args = (din["gairx"], din["gairy"], din["gwatx"], din["gwaty"], din["heff"], din["area"],
            din["amass"], din["cor"], (SINWAT, COSWAT), 1, din["pgfub"], din["pgfvb"])

    t0 = time.time(); fs = D.form(NX1, NY1, u1, v1, *args); ts = time.time() - t0
    t0 = time.time(); fv = V.form(NX1, NY1, u1, v1, *args); tv = time.time() - t0
    out["form"] = (max(rel(fv[k], fs[k]) for k in fs), ts, tv)
    t0 = time.time(); es, zs = D.plast(NX1, NY1, u1, v1, fs["press"]); ts = time.time() - t0
    t0 = time.time(); ev, zv = V.plast(NX1, NY1, u1, v1, fs["press"]); tv = time.time() - t0
    out["plast"] = (max(rel(ev, es), rel(zv, zs)), ts, tv)

    def mk():
        uice = {1: u1.copy(), 2: D._pad(NX1, NY1), 3: u1.copy()}
        vice = {1: v1.copy(), 2: D._pad(NX1, NY1), 3: v1.copy()}
        return uice, vice
    bydts = 1.0 / DTS
    res = {}
    for tag, fn in (("s", D.relax), ("v", V.relax)):
        uice, vice = mk()
        uc, vc = u1.copy(), v1.copy()
        t0 = time.time()
        fn(NX1, NY1, uice, vice, uc, vc, fs["forcex"].copy(), fs["forcey"].copy(), fs["draga"],
           fs["drags"], fs["eta"], fs["zeta"], din["amass"], din["cor"], bydts)
        res[tag] = (uice, vice, uc, vc, time.time() - t0)
    us, vs, ucs, vcs, ts = res["s"]
    uv, vv, ucv, vcv, tv = res["v"]
    w = max(rel(uv[k], us[k]) for k in (1, 2, 3))
    w = max(w, max(rel(vv[k], vs[k]) for k in (1, 2, 3)), rel(ucv, ucs), rel(vcv, vcs))
    out["relax"] = (w, ts, tv)

    # tridiagonal solves on a realistic diagonally dominant batch (the real relax coefficients are
    # exercised inside relax above)
    rng = np.random.default_rng(0)
    n, L = NX1 - 2, NY1 - 2
    a, b, c, r = (rng.random((n, L)) + 2, rng.random((n, L)) + 10, rng.random((n, L)) + 2,
                  rng.random((n, L)))
    b[:, 3] = 1.0; a[:, 3] = 0.0; c[:, 3] = 0.0; r[:, 3] = 0.0   # masked line (b0==1 branch)
    for nm, fs_, fv_ in (("thomas", D.tridiag_thomas, V.tridiag_thomas_batch),
                         ("cyclic", D.tridiag_cyclic, V.tridiag_cyclic_batch)):
        t0 = time.time(); ref = np.stack([fs_(a[:, k], b[:, k], c[:, k], r[:, k]) for k in range(L)], 1)
        ts = time.time() - t0
        t0 = time.time(); got = fv_(a, b, c, r); tv = time.time() - t0
        out[nm] = (rel(got, ref), ts, tv)
    if verbose:
        for k, (wv, ts, tv) in out.items():
            print(f"{date}/{itime} {k:7s} vs scalar {wv:.2e}   {ts:.4f}s -> {tv:.4f}s")
    return out


def full_checks(date, itime, verbose=True):
    """Full vpicedyn: vec vs scalar (to ulp level) and vec vs real Fortran (existing test tolerance)."""
    setup_geometry(date)
    din = C.read_dynsi_in(f"{FF_DATA}/{date}/ffy_{itime}_in.bin")
    usi0, vsi0 = usi_vsi(din)
    args = (NX1, NY1, usi0, vsi0, din["gairx"], din["gairy"], din["gwatx"], din["gwaty"], din["heff"],
            din["area"], din["amass"], din["cor"], SINWAT, COSWAT, 1.0 / DTS, 1, din["pgfub"], din["pgfvb"])
    t0 = time.time(); us, vs, ks, dws = D.vpicedyn(*args); ts = time.time() - t0
    t0 = time.time(); uv, vv, kv, dwv = V.vpicedyn(*args); tv = time.time() - t0
    vs_scalar = max(rel(uv, us), rel(vv, vs), rel(dwv, dws))
    base = f"{FF_DATA}/{date}"
    orig = D.vpicedyn
    try:
        D.vpicedyn = V.vpicedyn
        res, kki = C.compare_one(f"{base}/ffz_geom.bin", f"{base}/ffy_{itime}_in.bin",
                                 f"{base}/ffy_{itime}_out.bin", SINWAT, COSWAT, DTS, verbose=False)
    finally:
        D.vpicedyn = orig
    if verbose:
        print(f"{date}/{itime} vpicedyn: kki scalar={ks} vec={kv}; vec vs scalar {vs_scalar:.2e}; "
              f"{ts:.2f}s -> {tv:.3f}s ({ts / tv:.0f}x)")
        print("   vs real Fortran (max_rel, mean_rel): " +
              ", ".join(f"{k} {m:.1e}/{a:.1e}" for k, (m, a) in res.items()))
    return dict(kki_s=ks, kki_v=kv, vs_scalar=vs_scalar, t_s=ts, t_v=tv, real=res)


if __name__ == "__main__":
    todo = [(sys.argv[1], int(sys.argv[2]))] if len(sys.argv) > 2 else list(DATES.items())
    for date, itime in todo:
        stage_checks(date, itime)
        full_checks(date, itime)
