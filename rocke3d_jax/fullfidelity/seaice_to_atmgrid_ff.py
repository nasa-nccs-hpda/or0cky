"""Full-fidelity port of SEAICE_DRV.f seaice_to_atmgrid -- Stage 1 of the DYNSI/ocean port, D31.
Reconciles the ocean-grid sea-ice state (si_ocn, just updated by GROUND_SI/FORM_SI on the ocean
grid) onto the atm-grid copy (si_atm) that SURFACE/PBL/radiation read from -- the state-copy half
of this project's earlier "GROUND_SI/FORM_SI runs twice" finding (D24). This module ports the
second loop only (deriving GTEMP/GTEMP2/GTEMPR/ZSNOWI/ZSI/FWSIM from the just-copied state); the
first loop (plain field copy: RSI/SNOWI/MSI/HSI/SSI/pond_melt/flag_dsws) is a direct assignment,
not modeled here since it has no arithmetic to get wrong.

Not ported (radiation-adjacent, out of scope per this project's SOCRATES-is-third-party
constraint): the third loop's conditional RESET_SURF_FLUXES call, which only redistributes
RAD_COM's FSF/TRSURF diagnostic accumulators across a changed ice fraction -- it does not affect
sea-ice prognostic state and feeds directly into the radiation scheme's own bookkeeping.
"""
from ice_props_ff import RHOI, RHOS, ACE1I
from seaice_core_ff import Ti, Ti2b, XSI

TF = 273.15


def seaice_to_atmgrid_cell(rsi, snowi, msi, hsi1, hsi2, ssi1, ssi2, ssi3, ssi4):
    """SEAICE_DRV.f seaice_to_atmgrid's second loop, one atm-grid cell. Returns dict: gtemp,
    gtemp2, gtempr, zsnowi, zsi, fwsim."""
    msi1 = snowi + ACE1I

    if ACE1I > XSI[1] * msi1:  # some ice in first layer
        mice1 = ACE1I - XSI[1] * msi1
        mice2 = XSI[1] * msi1
        snowl1 = msi1 - ACE1I
        snowl2 = 0.0
    else:  # some snow in second layer
        mice1 = 0.0
        mice2 = ACE1I
        snowl1 = XSI[0] * msi1
        snowl2 = XSI[1] * msi1 - ACE1I

    if mice1 != 0.0:
        gtemp = Ti2b(hsi1 / (XSI[0] * msi1), 1e3 * ssi1 / mice1, snowl1, mice1)
    else:
        gtemp = Ti(hsi1 / (XSI[0] * msi1), 0.0)
    gtemp2 = Ti2b(hsi2 / (XSI[1] * msi1), 1e3 * ssi2 / mice2, snowl2, mice2)

    gtempr = gtemp + TF
    zsnowi = snowi / RHOS
    zsi = (ACE1I + msi) / RHOI
    fwsim = rsi * (msi1 + msi - (ssi1 + ssi2 + ssi3 + ssi4))

    return dict(gtemp=gtemp, gtemp2=gtemp2, gtempr=gtempr, zsnowi=zsnowi, zsi=zsi, fwsim=fwsim)
