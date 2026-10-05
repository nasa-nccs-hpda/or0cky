"""Validate icedyn_jax (D88) against icedyn_vec on real ice-dynamics inputs (per-stage and full
VPICEDYN) and against the real Fortran outputs; prints worst scale-relative differences and timings.
Usage: python icedyn_jax_compare.py [date itime]"""
import sys
import time
import numpy as np
import icedyn_dynsi_ff as D
import icedyn_vec as V
import icedyn_jax as J
import icedyn_vec_compare as K
import dynsi_compare as C

NX1, NY1 = K.NX1, K.NY1


def stage_checks(date, itime, verbose=True):
    K.setup_geometry(date)
    din = C.read_dynsi_in(f"{K.FF_DATA}/{date}/ffy_{itime}_in.bin")
    u1, v1 = din["uice0"], din["vice0"]
    args = (din["gairx"], din["gairy"], din["gwatx"], din["gwaty"], din["heff"], din["area"],
            din["amass"], din["cor"], (K.SINWAT, K.COSWAT), 1, din["pgfub"], din["pgfvb"])
    out = {}
    fv = V.form(NX1, NY1, u1, v1, *args)
    J.form(NX1, NY1, u1, v1, *args)  # warm-up/compile
    t0 = time.time(); fj = J.form(NX1, NY1, u1, v1, *args); tj = time.time() - t0
    out["form"] = max(K.rel(fj[k], fv[k]) for k in fv)
    ev, zv = V.plast(NX1, NY1, u1, v1, fv["press"])
    ej, zj = J.plast(NX1, NY1, u1, v1, fv["press"])
    out["plast"] = max(K.rel(ej, ev), K.rel(zj, zv))
    mk = lambda: ({1: u1.copy(), 2: D._pad(NX1, NY1), 3: u1.copy()},
                  {1: v1.copy(), 2: D._pad(NX1, NY1), 3: v1.copy()})
    bydts = 1.0 / K.DTS
    uv, vv = mk()
    V.relax(NX1, NY1, uv, vv, u1.copy(), v1.copy(), fv["forcex"].copy(), fv["forcey"].copy(),
            fv["draga"], fv["drags"], fv["eta"], fv["zeta"], din["amass"], din["cor"], bydts)
    ujd, vjd = mk()
    rargs = (NX1, NY1, ujd, vjd, u1.copy(), v1.copy(), fv["forcex"].copy(), fv["forcey"].copy(),
             fv["draga"], fv["drags"], fv["eta"], fv["zeta"], din["amass"], din["cor"], bydts)
    J.relax(*rargs)
    t0 = time.time(); uj, vj = J.relax(*rargs); tjr = time.time() - t0
    out["relax"] = max(max(K.rel(uj[k], uv[k]), K.rel(vj[k], vv[k])) for k in (1, 2, 3))
    rng = np.random.default_rng(0)
    n, L = NX1 - 2, NY1 - 2
    a, b, c, r = (rng.random((n, L)) + 2, rng.random((n, L)) + 10, rng.random((n, L)) + 2,
                  rng.random((n, L)))
    b[:, 3] = 1.0; a[:, 3] = 0.0; c[:, 3] = 0.0; r[:, 3] = 0.0
    out["thomas"] = K.rel(np.asarray(J.tridiag_thomas_batch(a, b, c, r)), V.tridiag_thomas_batch(a, b, c, r))
    out["cyclic"] = K.rel(np.asarray(J.tridiag_cyclic_batch(a, b, c, r)), V.tridiag_cyclic_batch(a, b, c, r))
    if verbose:
        print(f"{date}/{itime} jax vs vec: " + ", ".join(f"{k} {v:.1e}" for k, v in out.items())
              + f"   (jit form {tj*1e3:.1f} ms, relax {tjr*1e3:.1f} ms)")
    return out


def full_checks(date, itime, verbose=True):
    K.setup_geometry(date)
    din = C.read_dynsi_in(f"{K.FF_DATA}/{date}/ffy_{itime}_in.bin")
    usi0, vsi0 = K.usi_vsi(din)
    args = (NX1, NY1, usi0, vsi0, din["gairx"], din["gairy"], din["gwatx"], din["gwaty"], din["heff"],
            din["area"], din["amass"], din["cor"], K.SINWAT, K.COSWAT, 1.0 / K.DTS, 1, din["pgfub"],
            din["pgfvb"])
    t0 = time.time(); uv, vv, kv, dwv = V.vpicedyn(*args); tv = time.time() - t0
    J.vpicedyn(*args)  # compile
    t0 = time.time(); uj, vj, kj, dwj = J.vpicedyn(*args); tj = time.time() - t0
    wj = max(K.rel(uj, uv), K.rel(vj, vv), K.rel(dwj, dwv))
    base = f"{K.FF_DATA}/{date}"
    orig = D.vpicedyn
    try:
        D.vpicedyn = J.vpicedyn
        res, kki = C.compare_one(f"{base}/ffz_geom.bin", f"{base}/ffy_{itime}_in.bin",
                                 f"{base}/ffy_{itime}_out.bin", K.SINWAT, K.COSWAT, K.DTS, verbose=False)
    finally:
        D.vpicedyn = orig
    if verbose:
        print(f"{date}/{itime} vpicedyn: kki vec={kv} jax={kj}; jax vs vec {wj:.2e}; vec {tv:.3f}s, jax {tj:.3f}s")
        print("   vs real Fortran (max_rel/mean_rel): " +
              ", ".join(f"{k} {m:.1e}/{a:.1e}" for k, (m, a) in res.items()))
    return dict(kki_v=kv, kki_j=kj, vs_vec=wj, real=res, t_v=tv, t_j=tj)


if __name__ == "__main__":
    todo = [(sys.argv[1], int(sys.argv[2]))] if len(sys.argv) > 2 else list(K.DATES.items())
    for d, it in todo:
        stage_checks(d, it)
        full_checks(d, it)
