"""D139: FFT72 / AVRX in JAX (dyn_fft72_ff.fft / ffti traced with jnp arrays).

The statement-by-statement port of FFT72.f in dyn_fft72_ff.py works on dicts of per-row vectors with elementwise
operations only, so it is traced unchanged with jax arrays (the operation order, and so the IEEE rounding of every
elementwise operation, is the numpy one; XLA does not reassociate, whether it contracts a*b+c into an FMA on a
given backend is measured in tests/test_dyn_jax.py, not assumed).  The module-level scratch dicts of dyn_fft72_ff
are rewritten completely by every call before they are read, so interleaving numpy and traced calls is safe in a
single thread.

avrx_field_jax: AVRX on every layer of a (IM,JM,LM) field for a static list of rows, batched over (row, layer)
exactly as dyn_aflux_ff.avrx_field batches over layers.  The per-harmonic multiplier (BYSN(N)*DRAT(J)) is computed
on the host with the same double product, so A(N) <- (bysn*drat)*A(N) is the same operation.
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dyn_fft72_ff as f72

KM = f72.KM
IMH = KM // 2
IM = KM


def fft_rows_jax(x):
    """x (KM, n) -> A, B (KM/2+1, n)."""
    A, B = f72.fft({k + 1: x[k] for k in range(KM)})
    sh = x.shape[1:]
    return (jnp.stack([jnp.broadcast_to(jnp.asarray(A[n]), sh) for n in range(IMH + 1)]),
            jnp.stack([jnp.broadcast_to(jnp.asarray(B[n]), sh) for n in range(IMH + 1)]))


def ffti_rows_jax(A, B):
    F = f72.ffti({n: A[n] for n in range(IMH + 1)}, {n: B[n] for n in range(IMH + 1)})
    sh = A.shape[1:]
    return jnp.stack([jnp.broadcast_to(jnp.asarray(F[k + 1]), sh) for k in range(KM)])


def avrx_plan(rows, g, tab):
    """Host-side static data for avrx_field_jax: active rows (DRAT<=1) and the (n_act, IMH+1) multiplier table with
    mask.  Same indices as dyn_aflux_ff.avrx_field (including the Python negative-index quirk bysn[-1] for N=0, which
    only matters if NMIN(J)=0)."""
    bysn, drat, nmin = tab
    act = [j for j in rows if not drat[j] > 1]
    coef = np.zeros((len(act), IMH + 1))
    mask = np.zeros((len(act), IMH + 1), bool)
    for k, j in enumerate(act):
        for n in range(nmin[j], IMH + 1):               # N=NMIN..IMH-1, then IMH (A and B both, as in avrx_field)
            coef[k, n] = bysn[n - 1] * drat[j]
            mask[k, n] = True
    return np.array(act, dtype=int), coef, mask


def avrx_field_jax(x, act, coef, mask):
    """x (IM,JM,LM) -> copy with rows `act` (0-based j) replaced by their AVRX-truncated values."""
    LMx = x.shape[2]
    nj = len(act)
    if nj == 0:
        return x
    xs = x[:, act, :].reshape(IM, nj * LMx)
    A, B = fft_rows_jax(xs)
    cf = jnp.repeat(jnp.asarray(coef).T, LMx, axis=1)           # (IMH+1, nj*LM)
    mk = jnp.repeat(jnp.asarray(mask).T, LMx, axis=1)
    A = jnp.where(mk, cf * A, A)
    B = jnp.where(mk, cf * B, B)
    y = ffti_rows_jax(A, B).reshape(IM, nj, LMx)
    return x.at[:, act, :].set(y)
