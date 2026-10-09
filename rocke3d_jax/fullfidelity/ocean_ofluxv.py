"""D119: copy of ofluxv_jax (D43) that also returns SMW (the vertical mass flux OADVT2 consumes). Arithmetic unchanged."""
"""JAX (batched, vectorized) port of OCNDYN2.f's OFLUXV + OADVUZ -- Stage 2, D43.

Vectorized over (IM,JM) per layer via broadcast masks; the layer-shifted gates (`m_active(i,j,2)`
for layer 1 and the bottom layer, `m_active(i,j,l+1)` for interior layers -- the genuine
single-layer-column edge case verified from source in D43) are encoded directly as mask
comparisons against a shifted LMM. `OADVUZ`'s per-(i,j) running Courant-style update is
irreducibly sequential in L (each layer depends on the previous layer's CMUP/FMUP) but is fully
vectorized over (I,J) via a `jax.lax.scan` over layers. See ofluxv_ff.py for the reference
derivation, physical documentation, and the DXYPO/DTOLF bookkeeping note.

Arrays here are 0-indexed (IM,JM,LMO) -- NOT the 1-indexed convention ofluxv_ff.py uses.
ofluxv_jax_compare.py handles the translation.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from osourc_ff import DZO_L13

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665

_ze_np = np.zeros(LMO + 1)
for _l in range(1, LMO + 1):
    _ze_np[_l] = _ze_np[_l - 1] + DZO_L13[_l - 1]
# This module is first imported lazily from inside jit traces (jax_ocean.stage_dynamics); a plain jnp.asarray at import time then yields a
# TRACER that escapes the trace (UnexpectedTracerError on the next trace; seen on the Discover GPU, jax 0.6.1, job 58799715). Force concrete arrays.
with jax.ensure_compile_time_eval():
    ZE = jnp.asarray(_ze_np)                                  # 0..LMO
    DZO = jnp.asarray(np.concatenate([[0.0], DZO_L13]))       # 0..LMO, DZO[0] unused


def _m_active_mask(lmm, l_idx):
    """0-indexed (IM,JM,L) boolean mask matching m_active(i,j,l) for a batch of 1-indexed layer
    numbers l_idx (shape (L,)): ordinary LMM>=l everywhere except row JM-1 (Fortran J=JM),
    restricted to column 0 only."""
    ordinary = l_idx[None, None, :] <= lmm[:, :, None]
    pole_active = l_idx <= lmm[0, JM - 1]
    pole_row = jnp.zeros((IM, l_idx.shape[0]), dtype=bool).at[0, :].set(pole_active)
    return ordinary.at[:, JM - 1, :].set(pole_row)


def _oadvuz_jax(r0, m0, mw, dt, active_mask):
    """0-indexed OADVUZ via lax.scan over layers (sequential in L, vectorized over I,J).
    `active_mask` (IM,JM,LMO) marks which (i,j,l) cells the Fortran's nbyz-gated loop actually
    touches -- at inactive cells R is left UNCHANGED and CMUP/FMUP are NOT updated (frozen at
    their last-active value), exactly matching the Fortran's per-cell `continue`/skip semantics.
    Computing every cell densely without this mask produces 0/0 NaNs at genuinely-inactive
    columns (M0=0 there) -- caught by a real NaN in the first validation run."""
    def step(carry, layer):
        cmup, fmup = carry
        r_l, m_l, mw_l, r_next, mask_l = layer
        cm = dt * mw_l
        fm = jnp.where(cm >= 0.0, cm * r_l, cm * r_next)
        mnew = jnp.where(mask_l, m_l + cmup - cm, 1.0)  # avoid 0/0 at inactive cells
        r_new = jnp.where(mask_l, (r_l * m_l + (fmup - fm)) / mnew, r_l)
        cmup_new = jnp.where(mask_l, cm, cmup)
        fmup_new = jnp.where(mask_l, fm, fmup)
        return (cmup_new, fmup_new), r_new

    r_shift = jnp.concatenate([r0[:, :, 1:], jnp.zeros((IM, JM, 1))], axis=2)
    cmup0 = jnp.zeros((IM, JM))
    fmup0 = jnp.zeros((IM, JM))
    r_l = jnp.moveaxis(r0, 2, 0)
    m_l = jnp.moveaxis(m0, 2, 0)
    mw_l = jnp.moveaxis(mw, 2, 0)
    r_next_l = jnp.moveaxis(r_shift, 2, 0)
    mask_l_all = jnp.moveaxis(active_mask, 2, 0)
    _, r_out = jax.lax.scan(step, (cmup0, fmup0), (r_l, m_l, mw_l, r_next_l, mask_l_all))
    return jnp.moveaxis(r_out, 0, 2)


@jax.jit
def ofluxv_jax(lmm, lmu, lmv, dtolf, opbot0, opress0, mo0, uo0, vo0):
    """Batched OFLUXV. 2D fields (lmm,lmu,lmv,opbot0,opress0) shape (IM,JM); 3D fields shape
    (IM,JM,LMO), 0-indexed. Returns (mo, uo, vo), each (IM,JM,LMO)."""
    l_idx = jnp.arange(1, LMO + 1, dtype=jnp.float64)

    # ---- MB: snapshot MO, uniform at the pole row (mirrors the pole-copy in ofluxv_ff.py) ----
    mb = mo0
    pole_mask = l_idx <= lmm[0, JM - 1]
    mb_pole = jnp.where(pole_mask, mo0[0:1, JM - 1, :], mo0[:, JM - 1, :])
    mb = mb.at[:, JM - 1, :].set(mb_pole)

    # ---- MSUM (2D, computed once) ----
    lm_int = lmm.astype(jnp.int32)
    lm_clip = jnp.clip(lm_int, 1, LMO)
    ze_lm = ZE[lm_clip]
    msum = (opbot0 - opress0) / GRAV

    mask2 = _m_active_mask(lmm, jnp.array([2.0]))[:, :, 0]  # "reaches layer 2"

    # ---- Layer 1 ----
    mfinal1 = msum * DZO[1] / ze_lm
    smw1 = jnp.where(mask2, mo0[:, :, 0] - mfinal1, 0.0)
    mo1 = jnp.where(mask2, mfinal1, mo0[:, :, 0])

    # ---- Layers 2..LMO-1 (interior; l+1<=LMO-1 range, 0-idx 1..LMO-2) ----
    mo_layers = [mo1]
    smw_layers = [smw1]
    smw_prev = smw1
    for l in range(2, LMO):  # Fortran l=2..LMO-1
        mask_l = _m_active_mask(lmm, jnp.array([float(l + 1)]))[:, :, 0]
        mfinal = msum * DZO[l] / ze_lm
        smw_l = jnp.where(mask_l, smw_prev + (mo0[:, :, l - 1] - mfinal), smw_prev)
        mo_l = jnp.where(mask_l, mfinal, mo0[:, :, l - 1])
        mo_layers.append(mo_l)
        smw_layers.append(smw_l)
        smw_prev = smw_l

    # ---- Bottom layer LMO (0-idx LMO-1): overwrite mo at index lm-1 per column ----
    mfinal_bot = msum * DZO[lm_clip] / ze_lm
    # Stack layers 1..LMO-1 plus a same-shape filler at index LMO -- every column's real bottom
    # layer (wherever its own LM actually falls) gets overwritten below by `is_bottom`, so this
    # filler value is never read.
    mo_stack = jnp.stack(mo_layers + [mo0[:, :, LMO - 1]], axis=2)
    smw_stack = jnp.stack(smw_layers + [smw_layers[-1]], axis=2)

    # Build final MO by layer index, applying the bottom-layer overwrite at each column's own LM
    l_idx_all = jnp.arange(1, LMO + 1, dtype=jnp.float64)
    is_bottom = (l_idx_all[None, None, :] == lm_clip[:, :, None].astype(jnp.float64)) & mask2[:, :, None]
    mo_final = jnp.where(is_bottom, mfinal_bot[:, :, None], mo_stack)

    # ---- Fill pole (uniform across longitudes) ----
    mo_pole = jnp.where(pole_mask, mo_final[0:1, JM - 1, :], mo_final[:, JM - 1, :])
    mo_final = mo_final.at[:, JM - 1, :].set(mo_pole)
    smw_pole = jnp.where(pole_mask, smw_stack[0:1, JM - 1, :], smw_stack[:, JM - 1, :])
    smw_final = smw_stack.at[:, JM - 1, :].set(smw_pole)

    # ---- Vertical advection of UO ----
    lmu_i = lmu.astype(jnp.float64)
    lmv_i = lmv.astype(jnp.float64)
    j_mask = jnp.zeros(JM, dtype=bool).at[1:JM - 1].set(True)  # Fortran J=2..JM-1 -> 0-idx 1..JM-2
    mask_u = (l_idx_all[None, None, :] <= lmu_i[:, :, None]) & j_mask[None, :, None]
    mask_u = mask_u.at[:, JM - 1, :].set(False)

    mb_ip1 = jnp.roll(mb, shift=-1, axis=0)
    smw_ip1 = jnp.roll(smw_final, shift=-1, axis=0)
    mtmp_u = jnp.where(mask_u, 0.5 * (mb + mb_ip1), 0.0)
    mwtmp_u = jnp.where(mask_u, 0.5 * (smw_final + smw_ip1) / dtolf, 0.0)
    is_bottom_u = (l_idx_all[None, None, :] == lmu_i[:, :, None]) & (lmu_i[:, :, None] >= 1) & j_mask[None, :, None]
    mwtmp_u = jnp.where(is_bottom_u, 0.0, mwtmp_u)
    uo = _oadvuz_jax(uo0, mtmp_u, mwtmp_u, dtolf, mask_u)

    # ---- Vertical advection of VO ----
    mask_v = (l_idx_all[None, None, :] <= lmv_i[:, :, None]) & j_mask[None, :, None]
    mask_v = mask_v.at[:, JM - 1, :].set(False)
    mb_jp1 = jnp.concatenate([mb[:, 1:, :], jnp.zeros((IM, 1, LMO))], axis=1)
    smw_jp1 = jnp.concatenate([smw_final[:, 1:, :], jnp.zeros((IM, 1, LMO))], axis=1)
    mtmp_v = jnp.where(mask_v, 0.5 * (mb + mb_jp1), 0.0)
    mwtmp_v = jnp.where(mask_v, 0.5 * (smw_final + smw_jp1) / dtolf, 0.0)
    is_bottom_v = (l_idx_all[None, None, :] == lmv_i[:, :, None]) & (lmv_i[:, :, None] >= 1) & j_mask[None, :, None]
    mwtmp_v = jnp.where(is_bottom_v, 0.0, mwtmp_v)
    vo = _oadvuz_jax(vo0, mtmp_v, mwtmp_v, dtolf, mask_v)

    return mo_final, uo, vo, smw_final
