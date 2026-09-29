"""Batched JAX port of precip_oc_ff.py (OCNDYN.f PRECIP_OC) -- Stage 2 (first item) of the
DYNSI/ocean port, D33. Pure per-cell arithmetic, no branches -- nothing to merge with jnp.where."""
import jax
jax.config.update("jax_enable_x64", True)


def precip_oc_cell(focean, oprec, orsi, orunpsi, oeprec, oerunpsi, osrunpsi, dxypo, mo0, g0m0, s0m0):
    """Batched version of precip_oc_ff.precip_oc_cell."""
    mo = mo0 + ((1.0 - orsi) * oprec + orsi * orunpsi) * focean
    g0m = g0m0 + ((1.0 - orsi) * (oeprec * dxypo) + orsi * (oerunpsi * dxypo)) * focean
    s0m = s0m0 + orsi * (osrunpsi * dxypo) * focean
    return dict(mo=mo, g0m=g0m, s0m=s0m)
