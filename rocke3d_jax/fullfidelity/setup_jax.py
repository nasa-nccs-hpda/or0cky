"""JAX (batched) port of the OCONV per-column setup block (D59, ocnsetup_ff.setup_inputs) -- D64.

Batched over N columns. Each column has its own lmij, kmuv (IM+2 at the pole, 4 elsewhere) and
ZSCALE. Shear sums run over K with a mask (K <= kmuv per column). Same formulas as D59, reordered
sums (tolerance-checked, not bitwise). Index conventions as ocnsetup_ff: ze length LMO+1; zgrid,
hwide, byhwide length LMO+2; shsq, dvsq, dbloc, ritop length LMO+1 (Fortran index 0..LMO).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

LMO = 13
KMAX = 74
EPSLN = 1e-20


def setup_jax(ze, lmij, kmuv, ogeoz, hocean, grav, ul, ravm, g, s, byrho, rhom, rho1,
              alpha1, beta1, shc1, u2rho, deltae, deltas, deltam, deltasr):
    """Batched setup. ul (N, LMO+1, KMAX+1) 1-indexed in both L and K. ravm (N, KMAX+1).
    g, s (N, LMO+1) 1-indexed. byrho, rhom, rho1 (N, LMO+1). Scalars per column (N,).
    Returns dict: zgrid, hwide, byhwide, shsq, dvsq, dbloc, ritop, ustar, bo, bosol."""
    N = ogeoz.shape[0]
    zscale = (ogeoz + grav * hocean) / (ze[lmij] * grav)
    zg = jnp.zeros((N, LMO + 2))
    mid = -0.5 * zscale[:, None] * (ze[None, :LMO] + ze[None, 1:LMO + 1])   # L = 1..LMO
    zg = zg.at[:, 1:LMO + 1].set(mid)
    zg = zg.at[:, LMO + 1].set(-ze[LMO] * zscale)
    zg = zg.at[:, 0].set(EPSLN)
    hw = jnp.zeros((N, LMO + 2))
    hw = hw.at[:, 0].set(EPSLN)
    hw = hw.at[:, 1:LMO + 1].set(zg[:, 0:LMO] - zg[:, 1:LMO + 1])
    hw = hw.at[:, LMO + 1].set(EPSLN)
    byhw = jnp.where(hw > 0, 1.0 / jnp.where(hw == 0, 1.0, hw), 0.0)
    byhw = byhw.at[:, 0].set(0.0).at[:, LMO + 1].set(0.0)

    # --- shears: sum over K <= kmuv (mask), L = 1..lmij-1 (interface) and 1..lmij (tracer) ---
    Kidx = jnp.arange(KMAX + 1)
    kmask = (Kidx[None, :] >= 1) & (Kidx[None, :] <= kmuv[:, None])            # (N, KMAX+1)
    rav = jnp.where(kmask, ravm, 0.0)                                           # (N, KMAX+1)
    d_int = ul[:, 1:LMO + 1, :] - ul[:, 2:LMO + 2, :]                           # ul(L)-ul(L+1)
    shsq_core = jnp.sum(rav[:, None, :] * d_int * d_int, axis=2)                 # (N, LMO)
    d_tr = ul[:, 0:1, :] - ul[:, 1:LMO + 1, :]                                  # ul(1)-ul(L)
    dvsq_core = jnp.sum(rav[:, None, :] * d_tr * d_tr, axis=2)                  # (N, LMO)
    Lidx = jnp.arange(1, LMO + 1)[None, :]
    shsq = jnp.zeros((N, LMO + 1)).at[:, 1:LMO + 1].set(
        jnp.where(Lidx < lmij[:, None], shsq_core, 0.0))
    dvsq = jnp.zeros((N, LMO + 1)).at[:, 1:LMO + 1].set(
        jnp.where(Lidx <= lmij[:, None], dvsq_core, 0.0))

    # --- density gradients (uses recorded EOS values) ---
    Lall = jnp.arange(0, LMO + 1)[None, :]
    dbsfc = jnp.where((Lall >= 2) & (Lall <= lmij[:, None]),
                      grav * (1.0 - rho1 * byrho), 0.0)
    dbloc = jnp.zeros((N, LMO + 1))
    dbloc = dbloc.at[:, 1:LMO].set(grav * (1.0 - rhom[:, 2:LMO + 1] * byrho[:, 2:LMO + 1]))
    dbloc = jnp.where((Lall >= 1) & (Lall <= lmij[:, None] - 1), dbloc, 0.0)
    ritop = jnp.where((Lall >= 2) & (Lall <= lmij[:, None]),
                      (zg[:, 1:2] - zg[:, :LMO + 1]) * dbsfc, 0.0)

    # --- surface forcing ---
    talpha1, sbeta1, byshc = alpha1, beta1, 1.0 / shc1
    ustar = jnp.sqrt(u2rho * byrho[:, 1])
    grav_b2 = grav * (byrho[:, 1] * byrho[:, 1])
    bo = -(grav_b2 * (sbeta1 * deltas + talpha1 * byshc * deltae
                      - (sbeta1 * s[:, 1] + talpha1 * byshc * g[:, 1]) * deltam))
    bosol = -(((grav_b2 * talpha1) * byshc) * deltasr)
    return dict(zgrid=zg, hwide=hw, byhwide=byhw, shsq=shsq, dvsq=dvsq, dbloc=dbloc,
                ritop=ritop, ustar=ustar, bo=bo, bosol=bosol)
