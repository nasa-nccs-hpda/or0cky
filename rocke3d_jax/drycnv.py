"""
JAX Implementation of ROCKE-3D Dry Convection (DRYCNV.f)
========================================================

Dry convection mixing: a single upward pass over layer pairs (L, L+1); if the
virtual temperature of layer L exceeds that of L+1 the two layers are mixed
(both set to the pressure-weighted mean). Layer L is final after its own step
(later steps only touch L+1 and above), so the algorithm needs only a tiny
carry -- the current layer's (T, Q) -- not a rewrite of the whole array.

Two entry points, same physics:

  dry_convection_mixing_lf(T, Q, PK, PDSIG)  -- LAYER-FIRST arrays, shape
      (L, ...). Native/fast path. C-order (L, J, I) is byte-for-byte the memory
      layout of Fortran's (I, J, L) arrays, so each layer is a contiguous
      slab and the port can share ModelE's layout with no copies.
  dry_convection_mixing(T, Q, PK, PDSIG)     -- LEVEL-LAST arrays, shape
      (..., L) (the original API). Transposes to layer-first internally.

Optimization history (2026-09, "Round 2"): the original version was a
lax.scan that carried and dynamic-update-sliced the *entire* (I, J, L)
T/Q/TV arrays at each of the LM-1 steps (O(L^2) memory traffic; ~12-16 ms
CPU). This version scans layer slabs with a two-array carry: same arithmetic,
same association order, results equal to the original to float32 rounding
(<= ~1e-4 K absolute on T ~ 250 K; identical distance from Fortran). Unrolling
the layer loop at trace time was tried and is *slower* on CPU (XLA fuses the
serial chain with heavy producer duplication), so the loop stays a scan.

Usage:
    from drycnv import dry_convection_mixing_jit, dry_convection_mixing_lf_jit
"""

import jax.numpy as jnp
from jax import jit, lax


def _mix_layers(T, Q, PK, PDSIG, deltx):
    """Layer-first core: T, Q, PK, PDSIG all (L, ...)."""

    def body(carry, xs):
        t, q = carry                      # current layer L (possibly already mixed)
        t_n, q_n, pk0, pk1, dp0, dp1 = xs  # layer L+1 and the (L, L+1) PK / PDSIG
        tv = t * (1 + q * deltx)
        tv_n = t_n * (1 + q_n * deltx)
        unstable = tv > tv_n
        pkms = pk0 * dp0 + pk1 * dp1
        tvms = tv * pk0 * dp0 + tv_n * pk1 * dp1
        qms = q * dp0 + q_n * dp1
        rdp = 1.0 / (dp0 + dp1)
        thm = tvms / (pkms * (1 + qms * rdp * deltx))
        qm = qms * rdp
        # layer L is final after this step; L+1 carries into the next.
        return ((jnp.where(unstable, thm, t_n), jnp.where(unstable, qm, q_n)),
                (jnp.where(unstable, thm, t), jnp.where(unstable, qm, q)))

    xs = (T[1:], Q[1:], PK[:-1], PK[1:], PDSIG[:-1], PDSIG[1:])
    (t_last, q_last), (t_fin, q_fin) = lax.scan(body, (T[0], Q[0]), xs)
    return (jnp.concatenate([t_fin, t_last[None]], axis=0),
            jnp.concatenate([q_fin, q_last[None]], axis=0))


@jit
def dry_convection_mixing_lf(
    T: jnp.ndarray,
    Q: jnp.ndarray,
    PK: jnp.ndarray,
    PDSIG: jnp.ndarray,
    deltx: float = 0.608,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dry convection mixing on layer-first arrays (L, ...)."""
    return _mix_layers(T, Q, PK, PDSIG, deltx)


@jit
def dry_convection_mixing(
    T: jnp.ndarray,
    Q: jnp.ndarray,
    PK: jnp.ndarray,
    PDSIG: jnp.ndarray,
    deltx: float = 0.608,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Dry convection mixing on level-last arrays (..., L) (original API).

    Args:
        T: Temperature (I, J, L)  [GISS convention: T_actual / PK]
        Q: Moisture (I, J, L)
        PK: Exner-like factor (I, J, L)
        PDSIG: Layer thickness (I, J, L)
        deltx: Virtual temperature factor (default: 0.608)

    Returns:
        Updated T, Q after mixing unstable layers (level-last).
    """
    lf = lambda x: jnp.moveaxis(x, -1, 0)
    t, q = _mix_layers(lf(T), lf(Q), lf(PK), lf(PDSIG), deltx)
    return jnp.moveaxis(t, 0, -1), jnp.moveaxis(q, 0, -1)


# JIT-compiled versions (already jitted; names kept for API compatibility)
dry_convection_mixing_jit = dry_convection_mixing
dry_convection_mixing_lf_jit = dry_convection_mixing_lf
