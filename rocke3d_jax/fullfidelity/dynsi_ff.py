"""D168: DYNSI input assembly and post-processing (ICEDYN_DRV.f:328-877, GET_UISURF 2125-2205, OCN_Interp.f IG2OG_oceans 1605-1690,
OCNDYN.f get_exports_layer1 5667-5683) around the already-validated VPICEDYN (icedyn_vec.vpicedyn / icedyn_dynsi_ff.vpicedyn).

NEW module; nothing existing is modified.  Non-cubed-sphere branch (this rundeck: ice grid == atmosphere grid == ocean grid, 72 x 46,
OCEAN_IMPORTEXPORT_ON_BGRID not defined; both checked against decks/P2SAoM40.R and the dump geometry).

Atmosphere/ocean-grid arrays are numpy [i, j], 0-based (i = longitude 0..71, j = latitude 0..45).  Ice-dynamics arrays follow the existing
convention of icedyn_dynsi_ff (1-padded, Fortran (I, J) -> [I, J], shape (NX1+1, NY1+1)).

What is computed here (everything DYNSI does outside VPICEDYN):
  * polar replication of RSI/MSI/SNOWI/DMUA/DMVA, masking of DMUA/DMVA where FOCEAN*RSI <= 0
  * GAIRX/GAIRY (4-point average of the atmosphere stress, / DTS), aPtmp (OGEOZA + ice load), PGFU/PGFV, HEFF, AREA, AMASS, COR,
    GWATX/GWATY (4-point average of UOSURF/VOSURF), PGFUB/PGFVB, ghost columns, north-pole rows
  * after VPICEDYN: DMU/DMV (all rows), USI/VSI, DMUI/DMVI (net momentum into the ocean) and the north-pole means, UI2rho
    (4-point B->A average of the stress magnitude, / DTS), the UNDERICE ustar = max(5e-4, sqrt(UI2rho/RHOWS)), GET_UISURF
  * IG2OG (identity on the same grid, polar row set to zero) -> ODMUI/ODMVI
  * UOSURF/VOSURF of the atmosphere grid from the ocean UO/VO of layer 1 (get_exports_layer1 + OG2AG identity + polar vector)

Libm mode: numpy/glibc (sin, cos, sqrt); the real run used the model's compiler libm.  Constants: RHOI = 916.6, RHOWS = 1030, GRAV = 9.80665,
OMEGA = 2*pi/(86400*365/366) (D119), OIPHI = 25 degrees, DTS = 1800 s, RADIUS = 6371000 (inferred from the dump DXT by the older ports).
Sources named in the ledger entry scoping/D168_DYNSI_ENTRY.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icedyn_dynsi_ff as D  # noqa: E402
import icedyn_vec as V  # noqa: E402
from icedyn_geom_ff import geomicdyn, icdyn_masks  # noqa: E402

IM, JM = 72, 46
NX1, NY1 = IM + 2, JM
RADIUS = 6371000.0
GRAV = 9.80665
RHOI = 916.6
RHOWS = 1030.0
ACE1I = 0.1 * RHOI
BYRHOI = 1.0 / RHOI
OMEGA = 2.0 * np.pi / (86400.0 * 365.0 / 366.0)
OIPHI = 25.0 * (np.pi / 180.0)
SINWAT, COSWAT = np.sin(OIPHI), np.cos(OIPHI)
DTS = 1800.0
BYDTS = 1.0 / DTS


def _pad():
    return np.zeros((NX1 + 1, NY1 + 1))


class Geom:
    """ICEDYN geometry for one FOCEAN: GEOMICDYN + ICDYN_MASKS (existing ports) + the DYNSI-only arrays DXP, DYV, DXYP, DXYN, DXYS, DXYV, BYDXYP
    (ICEDYN.f:1022-1086).  1-padded 1-D arrays (index = Fortran J)."""

    def __init__(self, focean, radius=RADIUS):
        self.focean = np.asarray(focean, float)              # (IM, JM) ice-grid ocean fraction (= atmosphere FOCEAN)
        g = geomicdyn(IM, JM, radius)
        heffm, uvm = icdyn_masks(self.focean, NX1, NY1)
        D.RADIUS = radius
        D.BYRAD2 = 1.0 / (radius * radius)
        D.init_geometry(g, heffm, uvm)
        self.g, self.heffm, self.uvm, self.radius = g, heffm, uvm, radius
        dlon, dlat = g['dlon'], g['dlat']
        fjeq = 0.5 * (JM + 1)
        lat = np.zeros(JM + 2)
        for j in range(2, JM):
            lat[j] = dlat * (j - fjeq)
        lat[1] = -0.25 * 2.0 * np.pi
        lat[JM] = -lat[1]
        dxp = np.zeros(JM + 2)
        for j in range(2, JM):
            dxp[j] = radius * dlon * np.cos(lat[j])
        dyv = np.zeros(JM + 2)
        for j in range(2, JM + 1):
            dyv[j] = radius * (lat[j] - lat[j - 1])
        dxyp = np.zeros(JM + 2)
        dxys = np.zeros(JM + 2)
        dxyn = np.zeros(JM + 2)
        sinv = np.sin(dlat * (1 + .5 - fjeq))
        dxyp[1] = radius * radius * dlon * (sinv + 1)
        sinvm1 = np.sin(dlat * (JM - .5 - fjeq))
        dxyp[JM] = radius * radius * dlon * (1 - sinvm1)
        dxys[1], dxys[JM], dxyn[1], dxyn[JM] = 0.0, dxyp[JM], dxyp[1], 0.0
        for j in range(2, JM):
            sinvm1 = np.sin(dlat * (j - .5 - fjeq))
            sinv = np.sin(dlat * (j + .5 - fjeq))
            dxyp[j] = radius * radius * dlon * (sinv - sinvm1)
            dxys[j] = .5 * dxyp[j]
            dxyn[j] = .5 * dxyp[j]
        dxyv = np.zeros(JM + 2)
        for j in range(2, JM + 1):
            dxyv[j] = dxyn[j - 1] + dxys[j]
        self.dxp, self.dyv, self.dxyp, self.dxys, self.dxyn, self.dxyv = dxp, dyv, dxyp, dxys, dxyn, dxyv
        self.bydxyp = np.zeros(JM + 2)
        self.bydxyp[1:JM + 1] = 1.0 / dxyp[1:JM + 1]
        # SINIU/COSIU (ICEDYN.f:1108-1112), 1-padded
        ii = np.arange(1, IM + 1)
        self.siniu = np.zeros(IM + 1)
        self.cosiu = np.zeros(IM + 1)
        self.siniu[1:] = np.sin(ii * (2.0 * np.pi) / IM)
        self.cosiu[1:] = np.cos(ii * (2.0 * np.pi) / IM)
        self.siniu[IM], self.cosiu[IM] = 0.0, 1.0


# ------------------------------------------------------------------------------------------------ ocean exports to the ice grid
def ocean_polar_trig():
    """OGEOM.f:157 COSIC/SINIC (ocean, (I-.5)*TWOPI/IM) and GEOM_B.f:312 COSIP/SINIP (atmosphere, LON = DLON*(I-.5)), I = 1..IM."""
    i = np.arange(1, IM + 1)
    cos_o, sin_o = np.cos((i - .5) * (2.0 * np.pi) / IM), np.sin((i - .5) * (2.0 * np.pi) / IM)
    dlon = 2.0 * np.pi / IM
    return cos_o, sin_o, np.cos(dlon * (i - .5)), np.sin(dlon * (i - .5))


def uosurf_from_ocean(uo1, vo1, sinpo, sinvo, ivnpo=IM // 4):
    """OCNDYN.f get_exports_layer1 (5667-5683): ocean C-grid layer-1 UO,VO (IM, JM) -> A-grid UOSURF/VOSURF.  The same-grid OG2AG is the
    identity; the atmosphere latlon polar-vector step of OG2AG_TOC2SST (OCN_Interp.f:662-670) then sets atm UOSURF(1,JM) / VOSURF(1,JM) from the
    polar row.  sinpo, sinvo: 1-padded arrays (index = Fortran J) from odhorz_ff.geomo_dyn_arrays.  IVNP = IM/4 (OCEAN_COM.f:22).
    Returns (uosurf, vosurf) (IM, JM): the ATMOSPHERE exports.  The south polar row is zero (hasSouthPole)."""
    cos_o, sin_o, cos_a, sin_a = ocean_polar_trig()
    u = np.zeros((IM, JM))
    v = np.zeros((IM, JM))
    for j in range(2, JM):                      # Fortran J = 2..JM-1
        awt1 = (sinpo[j] - sinvo[j - 1]) / (sinvo[j] - sinvo[j - 1])
        awt2 = 1.0 - awt1
        jj = j - 1
        u[0, jj] = .5 * (uo1[0, jj] + uo1[IM - 1, jj])
        v[0, jj] = vo1[0, jj - 1] * awt1 + vo1[0, jj] * awt2
        for i in range(1, IM):
            u[i, jj] = .5 * (uo1[i, jj] + uo1[i - 1, jj])
            v[i, jj] = vo1[i, jj - 1] * awt1 + vo1[i, jj] * awt2
    u[:, JM - 1] = uo1[IM - 1, JM - 1] * cos_o + uo1[ivnpo - 1, JM - 1] * sin_o
    v[:, JM - 1] = uo1[ivnpo - 1, JM - 1] * cos_o - uo1[IM - 1, JM - 1] * sin_o
    unp = np.sum(u[:, JM - 1] * cos_a) * 2 / IM
    vnp = np.sum(u[:, JM - 1] * sin_a) * 2 / IM
    # The atmosphere polar box is a single point: the recorded restart exports (uosurf/vosurf of the real run, checked on nov26) are
    # constant along the polar row and equal to this polar vector, so the row is filled with (UNP, VNP).
    u[:, JM - 1], v[:, JM - 1] = unp, vnp
    return u, v


# ------------------------------------------------------------------------------------------------ DYNSI input assembly
def assemble_inputs(G, dmua, dmva, rsi, msi, snowi, ogeoza, uosurf, vosurf, bydts=BYDTS):
    """DYNSI up to (excluding) VPICEDYN.  All atmosphere-grid inputs (IM, JM) numpy [i, j].  rsi/msi/snowi must be the ocean-domain
    ice (zero in lake cells); ogeoza, uosurf, vosurf the ocean exports (zero over land as TOC2SST leaves them).
    Returns dict: padded ice-grid arrays (gairx, gairy, gwatx, gwaty, pgfub, pgfvb, heff, area, amass, cor) and irsi, imsi (IM, JM) plus the
    replicated rsi/msi/snowi/dmua/dmva used."""
    foc = G.focean
    rsi, msi, snowi = rsi.copy(), msi.copy(), snowi.copy()
    dmua, dmva = dmua.copy(), dmva.copy()
    for a in (rsi, msi, snowi, dmua, dmva):             # replicate polar boxes
        a[1:, JM - 1] = a[0, JM - 1]
    nodata = foc * rsi <= 0
    dmua[nodata] = 0.0
    dmva[nodata] = 0.0

    def avg4(a, bydts_):
        out = np.zeros((IM, JM - 1))
        am1 = np.roll(a, 1, axis=0)
        out[:] = 0.25 * (a[:, :JM - 1] + am1[:, :JM - 1] + am1[:, 1:JM] + a[:, 1:JM]) * bydts_
        return out

    gairx, gairy = _pad(), _pad()
    for gg, a in ((gairx, dmua), (gairy, dmva)):
        gg[1:IM + 1, 1:JM] = avg4(a, bydts)             # J = 1..JM-1 (GAIR(i,j), not i+1: non-cubed branch)
        gg[IM + 1, 1:JM] = gg[1, 1:JM]
        gg[IM + 2, 1:JM] = gg[2, 1:JM]
        gg[1:NX1 + 1, JM] = a[0, JM - 1] * bydts        # north pole
    # pressure on the atmosphere grid and its gradients on the ice grid
    ptmp = ogeoza + (rsi * (msi + snowi + ACE1I)) * GRAV / RHOWS
    pgfu = _pad()
    pgfv = _pad()
    ip1 = np.roll(np.arange(IM), -1)
    for j0 in range(1, JM - 1):                         # J = 2..JM-1 (0-based j0 = J-1)
        c = (foc[:, j0] > 0) & (foc[ip1, j0] > 0) & (rsi[:, j0] + rsi[ip1, j0] > 0)
        val = -(ptmp[ip1, j0] - ptmp[:, j0]) / G.dxp[j0 + 1]
        pgfu[1:IM + 1, j0 + 1] = np.where(c, val, 0.0)
    for j0 in range(0, JM - 1):                         # J = 1..JM-1
        c = (foc[:, j0 + 1] > 0) & (foc[:, j0] > 0) & (rsi[:, j0] + rsi[:, j0 + 1] > 0)
        val = -(ptmp[:, j0 + 1] - ptmp[:, j0]) / G.dyv[j0 + 2]
        pgfv[1:IM + 1, j0 + 1] = np.where(c, val, 0.0)
    # HEFF, AREA
    heff, area = _pad(), _pad()
    heff[2:NX1, 1:JM + 1] = (rsi * (ACE1I + msi)) * BYRHOI
    heff[2:NX1, 1:JM + 1] = heff[2:NX1, 1:JM + 1] * D.HEFFM[2:NX1, 1:JM + 1]
    area[2:NX1, 1:JM + 1] = rsi
    heff[1, 1:JM + 1] = heff[NX1 - 1, 1:JM + 1]
    area[1, 1:JM + 1] = area[NX1 - 1, 1:JM + 1]
    heff[NX1, 1:JM + 1] = heff[2, 1:JM + 1]
    area[NX1, 1:JM + 1] = area[2, 1:JM + 1]
    amass, cor = _pad(), _pad()
    sinen = D.SINEN
    for j in range(1, JM):                              # J = 1..JM-1
        a = RHOI * 0.25 * (heff[1:NX1, j] + heff[2:NX1 + 1, j] + heff[1:NX1, j + 1] + heff[2:NX1 + 1, j + 1])
        amass[1:NX1, j] = a
        cor[1:NX1, j] = a * 2.0 * OMEGA * sinen[1:NX1, j]
    amass[1:NX1 + 1, JM] = RHOI * heff[1, JM]
    cor[1:NX1 + 1, JM] = amass[1:NX1 + 1, JM] * 2.0 * OMEGA * sinen[1, JM]
    # currents on the ice B grid, pressure-gradient force on B grid
    gwatx, gwaty, pgfub, pgfvb = _pad(), _pad(), _pad(), _pad()
    um1 = np.roll(uosurf, 1, axis=0)
    vm1 = np.roll(vosurf, 1, axis=0)
    gwatx[1:IM + 1, 1:JM] = 0.25 * (um1[:, :JM - 1] + um1[:, 1:JM] + uosurf[:, :JM - 1] + uosurf[:, 1:JM])
    gwaty[1:IM + 1, 1:JM] = 0.25 * (vm1[:, :JM - 1] + vm1[:, 1:JM] + vosurf[:, :JM - 1] + vosurf[:, 1:JM])
    pu = pgfu[1:IM + 1, :]
    pum1 = np.roll(pu, 1, axis=0)
    pgfub[1:IM + 1, 1:JM] = 0.5 * (pum1[:, 1:JM] + pum1[:, 2:JM + 1])
    pv = pgfv[1:IM + 1, :]
    pvm1 = np.roll(pv, 1, axis=0)
    pgfvb[1:IM + 1, 1:JM] = 0.5 * (pvm1[:, 1:JM] + pv[:, 1:JM])
    for a in (gwatx, gwaty, pgfub, pgfvb):              # north-pole row zero (set above by construction), ghost columns
        a[IM + 1, 1:JM + 1] = a[1, 1:JM + 1]
        a[IM + 2, 1:JM + 1] = a[2, 1:JM + 1]
    return dict(gairx=gairx, gairy=gairy, gwatx=gwatx, gwaty=gwaty, pgfub=pgfub, pgfvb=pgfvb, heff=heff, area=area, amass=amass, cor=cor,
                irsi=rsi, imsi=msi, rsi=rsi, msi=msi, snowi=snowi)


def post_vpicedyn(G, uice1, vice1, dwatn, gwatx, gwaty, irsi, usi_old, vsi_old, bydts=BYDTS):
    """DYNSI after VPICEDYN: DMU/DMV, USI/VSI, DMUI/DMVI, UI2rho.  Returns dict (ice-grid (IM, JM) numpy [i, j] for usi, vsi, dmui, dmvi;
    padded dmu, dmv; ui2rho (IM, JM))."""
    foc = G.focean
    dmu, dmv = _pad(), _pad()
    half = NY1 // 2
    for j in range(1, JM + 1):
        hemi = -1.0 if j <= half else 1.0
        dmu[1:NX1, j] = DTS * dwatn[1:NX1, j] * (COSWAT * (uice1[1:NX1, j] - gwatx[1:NX1, j]) - hemi * SINWAT * (vice1[1:NX1, j] - gwaty[1:NX1, j]))
        dmv[1:NX1, j] = DTS * dwatn[1:NX1, j] * (hemi * SINWAT * (uice1[1:NX1, j] - gwatx[1:NX1, j]) + COSWAT * (vice1[1:NX1, j] - gwaty[1:NX1, j]))
    dmu[1, :] = dmu[NX1 - 1, :]
    dmu[NX1, :] = dmu[2, :]
    usi = usi_old.copy()
    vsi = vsi_old.copy()
    for j in range(2, JM):                              # J_0S..J_1S = 2..JM-1
        u = uice1[2:IM + 2, j]
        v = vice1[2:IM + 2, j]
        usi[:, j - 1] = np.where(np.abs(u) < 1e-10, 0.0, u)
        vsi[:, j - 1] = np.where(np.abs(v) < 1e-10, 0.0, v)
    usi[:, JM - 1] = 0.0
    vsi[:, JM - 1] = 0.0
    dmui = np.zeros((IM, JM))
    dmvi = np.zeros((IM, JM))
    Iidx = np.arange(1, IM + 1)                          # Fortran I (the Fortran loop visits I = IM, 1, 2, ..., IM-1; each cell is independent)
    ip1 = Iidx % IM + 1
    fr = foc * irsi
    for J in range(2, JM):
        d = 0.5 * (dmu[Iidx + 1, J - 1] + dmu[Iidx + 1, J])
        dmui[:, J - 1] = 0.5 * d * (fr[Iidx - 1, J - 1] + fr[ip1 - 1, J - 1])
        e = 0.5 * (dmv[Iidx, J] + dmv[Iidx + 1, J])
        dmvi[:, J - 1] = 0.5 * e * (fr[Iidx - 1, J - 1] * G.dxyn[J] + fr[Iidx - 1, J] * G.dxys[J + 1]) / G.dxyv[J + 1]
    dmui[:, 0] = 0.0
    # north pole
    J = JM
    d = 0.5 * (dmu[Iidx + 1, J - 1] + dmu[Iidx + 1, J])
    dmui[:, J - 1] = 0.5 * d * (fr[Iidx - 1, J - 1] + fr[ip1 - 1, J - 1])
    dmuinp = 0.0
    for i in range(IM):
        dmuinp = dmuinp + dmui[i, J - 1]
    dmuinp = dmuinp / IM
    dmui[:, J - 1] = dmuinp
    dmvi[:, J - 1] = 0.0
    # UI2rho on the atmosphere grid: 4-point average of the B-grid stress, magnitude / DTS
    ui2rho = np.zeros((IM, JM))
    dxyn, dxys, byd = G.dxyn, G.dxys, G.bydxyp
    for j in range(1, JM + 1):
        for_ = foc[:, j - 1] * irsi[:, j - 1] > 0
        if not for_.any():
            continue
        i1 = np.arange(1, IM + 1)
        if j > 1:
            tu = dxys[j] * (dmu[i1 + 1, j - 1] + dmu[i1, j - 1])
            tv = dxys[j] * (dmv[i1 + 1, j - 1] + dmv[i1, j - 1])
        else:
            tu = tv = 0.0
        dua = 0.5 * (dxyn[j] * (dmu[i1 + 1, j] + dmu[i1, j]) + tu) * byd[j]
        dva = 0.5 * (dxyn[j] * (dmv[i1 + 1, j] + dmv[i1, j]) + tv) * byd[j]
        ui2rho[:, j - 1] = np.where(for_, np.sqrt(dua ** 2 + dva ** 2) * bydts, 0.0)
    return dict(usi=usi, vsi=vsi, dmui=dmui, dmvi=dmvi, dmu=dmu, dmv=dmv, ui2rho=ui2rho)


def ustar_from_ui2rho(ui2rho):
    """SEAICE_DRV.f:313 UNDERICE: Ustar = MAX(5d-4, SQRT(UI2rho/RHOWS))."""
    return np.maximum(5e-4, np.sqrt(ui2rho / RHOWS))


def to_ocean(dmui, dmvi):
    """IG2OG_oceans, same grid: copy, north polar row zero."""
    o_u, o_v = dmui.copy(), dmvi.copy()
    o_u[:, JM - 1] = 0.0
    o_v[:, JM - 1] = 0.0
    return o_u, o_v


def get_uisurf(G, usi, vsi):
    """ICEDYN_DRV.f GET_UISURF (non-cubed): B -> A 4-point average of the ice velocity; polar rows from the polar vector sums.  (IM, JM)."""
    u = np.zeros((IM, JM))
    v = np.zeros((IM, JM))
    for j in range(2, JM):
        for a, o in ((usi, u), (vsi, v)):
            am1 = np.roll(a, 1, axis=0)
            o[:, j - 1] = 0.25 * (a[:, j - 1] + a[:, j - 2] + am1[:, j - 1] + am1[:, j - 2])
    # south pole
    s_u = 0.0
    s_v = 0.0
    for i in range(1, IM + 1):
        s_u = s_u + (vsi[i - 1, 0] * G.siniu[i]) * 2. / IM
        s_v = s_v + (vsi[i - 1, 0] * G.cosiu[i]) * 2. / IM
    u[:, 0], v[:, 0] = s_u, s_v
    n_u = 0.0
    n_v = 0.0
    for i in range(1, IM + 1):
        n_u = n_u - (vsi[i - 1, JM - 2] * G.siniu[i]) * 2. / IM
        n_v = n_v + (vsi[i - 1, JM - 2] * G.cosiu[i]) * 2. / IM
    u[:, JM - 1], v[:, JM - 1] = n_u, n_v
    return u, v


def vpicedyn(inp, usi, vsi, vec=True):
    """Run the validated VPICEDYN (batched numpy by default, scalar with vec=False) on assembled inputs.  Returns (uice1, vice1, kki, dwatn)."""
    f = V.vpicedyn if vec else D.vpicedyn
    return f(NX1, NY1, usi, vsi, inp['gairx'], inp['gairy'], inp['gwatx'], inp['gwaty'], inp['heff'], inp['area'], inp['amass'], inp['cor'],
             SINWAT, COSWAT, BYDTS, 1, inp['pgfub'], inp['pgfvb'])


def dynsi(G, dmua, dmva, rsi, msi, snowi, ogeoza, uosurf, vosurf, usi, vsi, vec=True):
    """Whole DYNSI.  usi, vsi: ice velocity state of the previous step (IM, JM).  Returns dict with everything the day loop needs:
    odmui, odmvi (ocean grid), ui2rho, ustar, usi, vsi (new state), uisurf, visurf, plus intermediates (inputs, uice1, ...)."""
    inp = assemble_inputs(G, dmua, dmva, rsi, msi, snowi, ogeoza, uosurf, vosurf)
    uice1, vice1, kki, dwatn = vpicedyn(inp, usi, vsi, vec=vec)
    post = post_vpicedyn(G, uice1, vice1, dwatn, inp['gwatx'], inp['gwaty'], inp['irsi'], usi, vsi)
    odmui, odmvi = to_ocean(post['dmui'], post['dmvi'])
    uis, vis = get_uisurf(G, post['usi'], post['vsi'])
    return dict(odmui=odmui, odmvi=odmvi, ui2rho=post['ui2rho'], ustar=ustar_from_ui2rho(post['ui2rho']), usi=post['usi'], vsi=post['vsi'],
                uisurf=uis, visurf=vis, kki=kki, inputs=inp, uice1=uice1, vice1=vice1, dwatn=dwatn, post=post)
