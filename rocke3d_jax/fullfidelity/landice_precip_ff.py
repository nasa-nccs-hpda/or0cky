"""Full-fidelity port of LANDICE_DRV.f PRECIP_LI / LANDICE.f PRECLI -- Track B, float64 (Stage 1 of the
DYNSI/ocean port, D28). Applies precipitation to a land-ice tile: rain either heats/melts the top layer
(possibly through to the second layer, moving ice up) or snow accumulates (possibly compacting into ice,
moving down). No tracers, no SCM. PRECLI itself "uses nothing, no globals involved" per the real source's
own comment -- pure per-cell arithmetic, matching the trivial-height-class (NHC=1) finding of D22.
"""
from ice_props_ff import RHOI

LHM = 3.34e5
SHI = 2060.0
Z1E, Z2LI = 0.1, 2.9
ACE1LI = Z1E * RHOI
ACE2LI = Z2LI * RHOI
HC1LI = ACE1LI * SHI


def precli(snow0, tg10, tg20, prcp, enrgp):
    """LANDICE.f PRECLI. Returns dict: snow, tg1, tg2, run0, edifs, difs, erun2."""
    edifs = difs = erun2 = run0 = 0.0
    snow, tg1, tg2 = snow0, tg10, tg20
    hc1 = HC1LI + snow * SHI
    if enrgp >= 0.0:
        if enrgp >= -tg1 * hc1:
            dwater = (tg1 * hc1 + enrgp) / LHM
            tg1 = 0.0
            run0 = dwater + prcp
            if dwater < snow:
                snow = snow - dwater
            else:
                difs = snow - dwater
                snow = 0.0
                tg1 = -tg2 * difs / ACE1LI
                edifs = difs * (tg2 * SHI - LHM)
                erun2 = edifs
        else:
            tg1 = tg1 + enrgp / hc1
            run0 = prcp
    else:
        tg1 = (tg1 * hc1 + enrgp + LHM * prcp) / (hc1 + prcp * SHI)
        snow = snow + prcp
        if snow > ACE1LI:
            difs = snow - 0.9 * ACE1LI
            edifs = difs * (tg1 * SHI - LHM)
            erun2 = difs * (tg2 * SHI - LHM)
            tg2 = tg2 + (tg1 - tg2) * difs / ACE2LI
            snow = 0.9 * ACE1LI
    return dict(snow=snow, tg1=tg1, tg2=tg2, run0=run0, edifs=edifs, difs=difs, erun2=erun2)


def precip_li(ftype, prcp, enrgp, snow0, tg10, tg20):
    """LANDICE_DRV.f PRECIP_LI: gate (ftype>0 and prcp>0), else pass state through / zero the fluxes."""
    if ftype > 0.0 and prcp > 0.0:
        out = precli(snow0, tg10, tg20, prcp, enrgp)
        return dict(snow=out["snow"], tg1=out["tg1"], tg2=out["tg2"], runo=out["run0"], implm=out["difs"],
                    implh=out["erun2"], e1=out["edifs"])
    return dict(snow=snow0, tg1=tg10, tg2=tg20, runo=0.0, implm=0.0, implh=0.0, e1=0.0)
