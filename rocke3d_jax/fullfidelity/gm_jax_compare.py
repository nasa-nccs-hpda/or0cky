"""Validate gm_jax (D85) against gm_vec (numpy) on the real dumps, loaded as gm_vec_compare does,
and time warm numpy vs jax calls. Usage: python gm_jax_compare.py <date> <itime>"""
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from gm_vec import isoslope4_vec, gmkdif_vec, gmfexp_vec
from gm_jax import isoslope4_jax, gmkdif_jax, gmfexp_jax
from gm_vec_compare import load_all, iso_args, kdif_args, fexp_args, ISO_ORDER, diffs
from gmkdif_compare import GMKDIF_FIELDS

NCALL = 5


def to_jax(args):
    """numpy -> jnp arrays; the qlimit bool stays a Python bool (static argument of gmfexp_jax)."""
    return tuple(a if isinstance(a, bool) else jnp.asarray(a) for a in args)


def warm(fn, n=NCALL):
    """Mean wall time of n calls after one warm-up (compile) call; jax results are blocked on."""
    jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); jax.block_until_ready(fn()); ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


def worst(rj, rv, names):
    d = [diffs(np.asarray(rj[n]), rv[n]) for n in names]
    return max(x[0] for x in d), max(x[1] for x in d)


def main(date, itime):
    lmm, lmu, lmv, dg, isos, gmk, recs, k3d = load_all(date, itime)
    print(f"== {date} (itime={itime}) jax {jax.__version__} on {jax.devices()[0].platform} ==")
    a = iso_args(lmm, dg, k3d); ja = to_jax(a)
    wabs, wrel = worst(isoslope4_jax(*ja), isoslope4_vec(*a), ISO_ORDER)
    tn = warm(lambda: isoslope4_vec(*a)); tj = warm(lambda: isoslope4_jax(*ja))
    print(f"  isoslope4: max abs {wabs:.2e} max rel {wrel:.2e} (warm numpy {tn:.4f}s, jax {tj:.4f}s)")
    a = kdif_args(lmm, gmk, isos); ja = to_jax(a)
    wabs, wrel = worst(gmkdif_jax(*ja), gmkdif_vec(*a), GMKDIF_FIELDS)
    tn = warm(lambda: gmkdif_vec(*a)); tj = warm(lambda: gmkdif_jax(*ja))
    print(f"  gmkdif:    max abs {wabs:.2e} max rel {wrel:.2e} (warm numpy {tn:.4f}s, jax {tj:.4f}s)")
    for k, rec in enumerate(recs):
        a = fexp_args(lmm, lmu, lmv, rec, gmk, dg); ja = to_jax(a)
        rj, rv = gmfexp_jax(*ja), gmfexp_vec(*a)
        parts = " ".join(f"{n} {diffs(np.asarray(g), r)[0]:.1e}/{diffs(np.asarray(g), r)[1]:.1e}"
                         for n, g, r in zip(("TRM", "TXM", "TYM", "TZM"), rj, rv))
        tn = warm(lambda: gmfexp_vec(*a)); tj = warm(lambda: gmfexp_jax(*ja))
        print(f"  gmfexp call {k} (qlimit={a[8]}): abs/rel {parts} (warm numpy {tn:.4f}s, jax {tj:.4f}s)")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
