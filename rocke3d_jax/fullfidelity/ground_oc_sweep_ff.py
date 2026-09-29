"""Full-fidelity port of OCNDYN.f's GROUND_OC below-freezing layer sweep (the L=2..LMM(I,J) loop
at the end of GROUND_OC, after OSOURC has already updated layer 1) -- Stage 2 of the DYNSI/ocean
port, D35. Checks each lower layer for below-freezing conditions (pressure-corrected via the
`SHCGS`/`OFTAB` specific-heat table) and converts any below-freezing water to frazil ice mass/
salt/heat, mirroring OSOURC's own open-ocean/under-ice freezing check but per-layer, with an
explicit pressure correction to the freezing point.

`SHCGS(GF00,S0L)` is the one genuine external-data dependency in `GROUND_OC` (a specific-heat
lookup table read from the `OFTAB` binary file at `init_OCEAN`, `OCNFUNTAB.f`'s `OCFUNC` module --
unlike D34's `FSR`/`FSRZ`/`LSRPD`, this is NOT re-derivable from hardcoded PARAMETERs). Rather than
extracting/replicating the whole multi-dimensional table, this delta records the one place it's
used -- the pressure-correction term `PCORR = SHCGS(GF00,S0L)*8.19d-8*P0L` -- as a recorded real
input, the same "record what's not yet ported" pattern used for D29's GAIRX/GWATX and D33's
oPREC/oRSI/etc. `P0L` itself (accumulated column pressure) is real Fortran state, not a table
lookup, but is recorded too rather than re-derived, since re-deriving it would require replaying
the whole water column from layer 1 in the exact same accumulation order -- fragile to get right
across dump-file boundaries for no real benefit (P0L is trivial arithmetic, not physics insight).
"""
from seaice_core_ff import Ei, FSSS
from osourc_ff import gfrezs, tfrezs


def ground_oc_sweep_layer(mo_l0, g0m_l0, s0m_l0, dxypj, pcorr, p0l):
    """One layer of GROUND_OC's below-freezing sweep (OCNDYN.f:4801-4824's DO L=2,LMM(I,J) body).
    `pcorr`/`p0l` are recorded real inputs (see module docstring). Returns dict: mo, g0m, s0m
    (updated), dm0, de0, ds0 (frazil-ice mass/energy/salt removed from this layer)."""
    g0l = g0m_l0 / (mo_l0 * dxypj)
    s0l = s0m_l0 / (mo_l0 * dxypj)
    gf00 = gfrezs(s0l)
    gf0 = gf00 - pcorr

    if g0l < gf0:
        tf0 = tfrezs(s0l) - 7.53e-8 * p0l
        si0 = FSSS * s0l
        ei0 = Ei(tf0, si0 * 1e3)
        dm0 = mo_l0 * (g0l - gf0) / (ei0 - gf0)
        de0 = ei0 * dm0
        ds0 = si0 * dm0
        return dict(mo=mo_l0 - dm0, g0m=g0m_l0 - de0 * dxypj, s0m=s0m_l0 - ds0 * dxypj,
                    dm0=dm0, de0=de0, ds0=ds0)

    return dict(mo=mo_l0, g0m=g0m_l0, s0m=s0m_l0, dm0=0.0, de0=0.0, ds0=0.0)
