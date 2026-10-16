"""Per-substep links of SURFACE.f's `DO NS=1,NIsurf` loop that sit between one substep's ATURB exit and the next
substep's PBL/tile inputs -- Track B.

Real chain (SURFACE.f:385-1178): loadbl (no-op for surface types that persist), recalc_agrid_uv, atm_exports_phasesrf
(=get_atm_layer1: layer-1 T/Q/U/V copied to the coupling arrays), get_dbl (PBL depth + geostrophic wind from the ATURB
exit state and the composite ustar/lmonin), then the tile loops. Everything here is transcribed from the Fortran and
checked bit-for-bit against the recorded next-substep PBL inputs (tests/test_substep_chain.py).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

RGAS = 287.048730386149032
GRAV = 9.80664999999999942
OMEGA2 = 1.45842581331033012e-4
OMEGA = 0.5 * OMEGA2
ZGS = 10.0
XDELT = 0.0            # P2SAoM40: TS/THV use xdelt=0 (pbl_ff.py header)
DBLS0, SLOPE0 = 10.0, 0.5
DBL_MAX_STABLE = DBLS0 + SLOPE0 * 500.0

_A, _B, _C, _D = 113.4977618974100, 438.5012518098521, 88.49964112645850, -11.50111432385882
_E, _F, _G = 30.00033943846368, 299.9975118132485, 299.9994728900967


def thbar(x, y):
    """shared/Utilities.F90 THBAR: mean temperature for vertical differencing (rational approximation)."""
    q = x / y
    return x * (_A + q * (_B + q * (_C + q * (_D + q)))) / (_E + q * (_F + _G * q))


def layer1_exports(T, Q, UA, VA, MA, PEK, PMID):
    """get_atm_layer1 + the surface-layer scalars SURFACE.f derives from them. T,Q,UA,VA are (..., L) (aturb layout
    (J,I,L)); MA,PEK,PMID are the layer-1 (...) arrays. Returns utop, vtop, qtop, tkv (=T1*PEK1, xdelt=0), zs1 and
    ztop (height of the first layer mid-point)."""
    q1 = Q[..., 0]
    thv1 = T[..., 0] * (1.0 + q1 * XDELT)
    tkv = thv1 * PEK
    zs1 = 0.5e-2 * RGAS * tkv * MA / PMID
    return dict(utop=UA[..., 0], vtop=VA[..., 0], qtop=q1, tkv=tkv, zs1=zs1, ztop=ZGS + zs1)


def get_dbl(ustar, lmonin, coriol, pblht, dclev, T, Q, UA, VA, PMID, PK, ztop):
    """PBL_DRV.f get_dbl for arrays shaped (...) with atmosphere columns (..., L) / (L, ...) already moved to
    (..., L). Returns ugeo, vgeo, bldep. The Fortran loop `do l=2,lm ... if (zpbl>=dbls) exit` is a fixed-length
    scan with a `done` freeze (a completed loop leaves l=lm+1, reproduced by the initial value)."""
    lm = T.shape[-1]
    ldbl = jnp.maximum(jnp.floor(dclev + 0.5).astype(jnp.int32), 1)
    dbl = jnp.maximum(pblht, DBLS0)
    stable = lmonin > 0.0
    tmp = jnp.maximum(jnp.abs(coriol), OMEGA)
    dbls = DBLS0 + SLOPE0 * jnp.sqrt(jnp.abs(lmonin * ustar / tmp))
    dbls = jnp.maximum(jnp.minimum(dbls, DBL_MAX_STABLE), ZGS)
    tv = T * (1.0 + XDELT * Q) * PK

    zpbl = ztop
    pl1 = PMID[..., 0]
    tl1 = tv[..., 0]
    ldbls = jnp.full(ztop.shape, lm + 1, jnp.int32)
    done = dbls <= ztop
    ldbls = jnp.where(done, 1, ldbls)
    for l in range(1, lm):
        pl = PMID[..., l]
        tl = tv[..., l]
        tbar = thbar(tl1, tl)
        zn = zpbl - (RGAS / GRAV) * tbar * (pl - pl1) / (pl1 + pl) * 2.0
        zpbl = jnp.where(done, zpbl, zn)
        hit = (~done) & (zpbl >= dbls)
        ldbls = jnp.where(hit, l + 1, ldbls)
        done = done | hit
        pl1 = jnp.where(done, pl1, pl)
        tl1 = jnp.where(done, tl1, tl)
    dbl = jnp.where(stable, jnp.minimum(dbl, dbls), dbl)
    ldbl = jnp.where(stable, jnp.minimum(ldbl, ldbls), ldbl)
    idx = jnp.clip(ldbl - 1, 0, lm - 1)
    ug = jnp.take_along_axis(UA, idx[..., None], axis=-1)[..., 0]
    vg = jnp.take_along_axis(VA, idx[..., None], axis=-1)[..., 0]
    return ug, vg, dbl
