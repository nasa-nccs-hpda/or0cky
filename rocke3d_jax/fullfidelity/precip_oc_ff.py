"""Full-fidelity port of OCNDYN.f's PRECIP_OC -- Stage 2 of the DYNSI/ocean port, D33 (the FIRST
Stage 2 deliverable: the ocean numerical core proper, previously only scoped at the file level).

Applies precipitation (direct + through-ice runoff) to the ocean's own top-layer mass/enthalpy/
salt (MO/G0M/S0M, on the ocean's OWN prognostic grid -- IMO=72,JMO=46, confirmed the SAME
resolution as the atmosphere for this rundeck via ORES_5x4.F90, a first real de-risking finding
for Stage 2 mirroring D29's ice-dyn-grid finding). TRACERS_OCEAN/TRACERS_WATER are both undefined
for this rundeck (confirmed throughout this project), so the tracer branch is dead code, not
ported. `oPREC`/`oRSI`/`oRUNPSI`/`oEPREC`/`oERUNPSI`/`oSRUNPSI` are recorded, real inputs: they are
the OUTPUT of `AG2OG_precip`'s atm-grid-to-ocean-grid regrid (a general HNTR8 area-weighted
interpolation utility, not specific to this physics), so -- same pattern as D29's GAIRX/GWATX --
this delta records them post-regrid rather than re-deriving the interpolation itself.
"""


def precip_oc_cell(focean, oprec, orsi, orunpsi, oeprec, oerunpsi, osrunpsi, dxypo, mo0, g0m0, s0m0):
    """OCNDYN.f PRECIP_OC's per-cell update, gated by the caller on (focean>0 and oprec>0) --
    matching the real subroutine's own IF test. Returns dict: mo, g0m, s0m (layer-1 mass/enthalpy*
    mass/salt*mass, updated)."""
    mo = mo0 + ((1.0 - orsi) * oprec + orsi * orunpsi) * focean
    g0m = g0m0 + ((1.0 - orsi) * (oeprec * dxypo) + orsi * (oerunpsi * dxypo)) * focean
    s0m = s0m0 + orsi * (osrunpsi * dxypo) * focean
    return dict(mo=mo, g0m=g0m, s0m=s0m)
