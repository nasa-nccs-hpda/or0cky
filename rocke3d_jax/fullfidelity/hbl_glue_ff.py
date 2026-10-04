"""OCONV HBL-iteration scalar diffusion inputs (D60, piece 2a of the OCONV port).

Between KPPMIX (ported in D54) and the G/S OVDIFFS calls (ported in D55) the OCNKPP.f
ITER loop does two things, both reproduced here:

1. GHAT flux scaling, before the diffusivities are rescaled (OCNKPP.f:2243-2251, live
   branch with OCN_GISS_TURB undefined):
     GHATG(L) = AKVG(L)*GHAT(L)*DELTAE*DXYPO(J)
     GHATS(L) = AKVS(L)*GHAT(L)*(DELTAS - S0ML0(1)*BYMML(1)*DELTAM)*DXYPO(J)
   for L = 1..LMIJ-1. GHATM is zeroed (OCNKPP.f:2244), so there is no momentum flux term.

2. Diffusivity rescaling by the density factor (OCNKPP.f:2240-2270). KVTDISS is zero for
   this build (use_tdiss=0), so (AKV+KVTDISS)*R2 reduces to AKV*R2:
     R  = 0.5*(RHO(L)+RHO(L+1)),  RHO = 1/BYRHO   (OCNKPP.f:2020, 2240)
     K  = AKV*R**2
   Only AKVG and AKVS feed OVDIFFS here; AKVM (momentum) and AKVC are rescaled the same way
   but are consumed by the momentum OVDIFF (D56) and TRACERS (dead) respectively.

Inputs not produced by an already-ported piece: BYMML(1) and S0ML0(1), which come from the
OCONV mass bookkeeping (MML/MML0 at OCNKPP.f:1963-1983). They are passed in explicitly here.
BYMML is switched from pre-source to post-source mass at ITER=2 (OCNKPP.f:1981-1983), so the
caller must supply the value for the current iteration.

DXYPO(J) comes from odhorz_ff.geomo_dyn_arrays (validated in D36/D40).
"""
import numpy as np

LMO = 13


def scale_akv(akv, byrho, lmij):
    """OCNKPP.f:2240-2270 density rescaling, KVTDISS=0. akv and byrho 1-indexed, length LMO+1
    (or longer); returns a new length-(LMO+1) array with entries 1..lmij-1 set. byrho[L] for
    L in 1..lmij is needed (RHO(L+1) is used for L=lmij-1)."""
    out = np.zeros(LMO + 1)
    for L in range(1, lmij):
        rho_l = 1.0 / byrho[L]
        rho_lp1 = 1.0 / byrho[L + 1]
        R = 5e-1 * (rho_l + rho_lp1)
        R2 = R * R
        out[L] = (akv[L] + 0.0) * R2
    return out


def ghat_scalar_fluxes(akvg, akvs, ghat, deltae, deltas, deltam, s0ml0_1, bymml_1, dxypo_j, lmij):
    """OCNKPP.f:2250-2252. akvg/akvs are the UNSCALED diffusivities (before scale_akv), ghat
    is KPPMIX's GHAT. Returns (ghatg, ghats), 1-indexed, entries 1..lmij-1 set.

    Operation order follows the Fortran left-to-right evaluation:
      AKVG*GHAT*DELTAE*DXYPO(J)
      AKVS*GHAT*(DELTAS-S0ML0(1)*BYMML(1)*DELTAM)*DXYPO(J)
    """
    ghatg = np.zeros(LMO + 1)
    ghats = np.zeros(LMO + 1)
    for L in range(1, lmij):
        ghatg[L] = akvg[L] * ghat[L] * deltae * dxypo_j
        ghats[L] = akvs[L] * ghat[L] * (deltas - s0ml0_1 * bymml_1 * deltam) * dxypo_j
    return ghatg, ghats
