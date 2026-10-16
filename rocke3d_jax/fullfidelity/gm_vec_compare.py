"""Validate gm_vec (D83) against the scalar gmredi_ff ports on the real dumps, as
gmredi_compare.py / gmkdif_compare.py / gmfexp_compare.py load them.
Usage: python gm_vec_compare.py <date> <itime>"""
import sys
import time
import numpy as np
from gmredi_ff import isoslope4, gmkdif, gmfexp, LMO
from gm_vec import (isoslope4_vec, gmkdif_vec, gmfexp_vec)
from gmredi_compare import load_isoslope4, ISOSLOPE4_FIELDS
from gmkdif_compare import load_gmkdif, GMKDIF_FIELDS
from gmfexp_compare import load_gmfexp
from ocnmeso_compare import load_densgrad, FF_DEFAULT
from odhorz_compare import load_lmm, load_lmv, load_lmu

ISO_ORDER = ISOSLOPE4_FIELDS


def load_all(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    isos = load_isoslope4(f"{ff}/ffz_isoslope4_{itime}.bin")
    gmk = load_gmkdif(f"{ff}/ffz_gmkdif_{itime}.bin")
    recs = load_gmfexp(f"{ff}/ffz_gmfexp_{itime}.bin")
    k3d = np.where(np.arange(LMO + 1)[None, None, :] <= lmm[:, :, None], 800.0, 0.0)
    k3d[:, :, 0] = 0.0
    return lmm, lmu, lmv, dg, isos, gmk, recs, k3d


def iso_args(lmm, dg, k3d):
    return (lmm, dg["rhox"], dg["rhoy"], dg["rhomz"], dg["byrhoz"], dg["bydh"], dg["dzv"], k3d)


def kdif_args(lmm, gmk, isos):
    return (lmm, gmk["kpl"]) + tuple(isos[n] for n in ISO_ORDER)


def fexp_args(lmm, lmu, lmv, rec, gmk, dg):
    return (lmm, lmu, lmv, rec["mo0"], rec["trm0"], rec["txm0"], rec["tym0"], rec["tzm0"],
            bool(rec["qlimit"]),
            gmk["bxx"], gmk["byy"], gmk["bzz"], gmk["azx"], gmk["bzx"], gmk["czx"],
            gmk["aezx"], gmk["ezx"], gmk["cezx"], gmk["azy"], gmk["bzy"], gmk["czy"],
            gmk["aezy"], gmk["ezy"], gmk["cezy"], gmk["kpl"], dg["bydh"], dg["bydzv"])


def diffs(a, b):
    d = np.abs(a - b)
    with np.errstate(all="ignore"):
        rel = np.where(b != 0.0, d / np.abs(b), np.where(d == 0.0, 0.0, np.inf))
    return float(d.max()), float(rel.max())


def timed(f, *a):
    t0 = time.time(); r = f(*a); return r, time.time() - t0


def main(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    print(f"== {date} (itime={itime}) ==")
    a = iso_args(lmm, dg, k3d)
    rs, ts = timed(isoslope4, *a); rv, tv = timed(isoslope4_vec, *a)
    worst = max((diffs(rv[n], rs[n]) for n in ISO_ORDER), key=lambda x: x[0])
    wrel = max(diffs(rv[n], rs[n])[1] for n in ISO_ORDER)
    print(f"  isoslope4: max abs {worst[0]:.2e} max rel {wrel:.2e} ({ts:.2f}s -> {tv:.3f}s)")
    a = kdif_args(lmm, gmk, isos)
    rs, ts = timed(gmkdif, *a); rv, tv = timed(gmkdif_vec, *a)
    worst = max(diffs(rv[n], rs[n])[0] for n in GMKDIF_FIELDS)
    wrel = max(diffs(rv[n], rs[n])[1] for n in GMKDIF_FIELDS)
    print(f"  gmkdif:    max abs {worst:.2e} max rel {wrel:.2e} ({ts:.2f}s -> {tv:.3f}s)")
    for k, rec in enumerate(recs):
        a = fexp_args(lmm, lmu, lmv, rec, gmk, dg)
        rs, ts = timed(gmfexp, *a); rv, tv = timed(gmfexp_vec, *a)
        parts = " ".join(f"{n} {diffs(g, r)[0]:.1e}/{diffs(g, r)[1]:.1e}"
                         for n, g, r in zip(("TRM", "TXM", "TYM", "TZM"), rv, rs))
        print(f"  gmfexp call {k} (qlimit={bool(rec['qlimit'])}): abs/rel {parts} ({ts:.2f}s -> {tv:.3f}s)")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
