"""D94/D95: analytic lat-lon geometry used by AVRX / isotropuv / SDRAG (GEOM_B.f, 4x5 half-polar-box
case IM=72, JM=46), as a numpy transcription.  Arrays have length JM (index = Fortran J - 1) or IM.

Source lines (model/GEOM_B.f of modelE2_planet_2.0): DLON 30, FJEQ 36, DLAT 170-173, COSP/DXP/COSV/DXV/DYP
180-232, DXYP/DXYN/DXYS 215-243, BYDYP 250, DXYV/RAPVS/RAPVN 255-262, SINIV/COSIV 308-309.
RADIUS is a runtime planet parameter (USE_PLANET_RAD) and must be supplied (the dumps record it).
Operation order follows the Fortran statements; sin/cos come from numpy (libm may differ from ifort's by
1 ulp, which the comparison scripts quantify against the recorded geometry).
"""
import numpy as np

IM, JM = 72, 46
PI = 3.1415926535897932
TWOPI = 2.0 * PI
RADIAN = PI / 180.0
FIM = float(IM)
BYIM = 1.0 / FIM
DLON = TWOPI * BYIM
FJEQ = .5 * (1 + JM)


def geometry(radius, im=IM, jm=JM):
    dlat_dg = 180.0 / float(jm - 1)
    dlat = dlat_dg * RADIAN
    lat = np.zeros(jm)
    cosp = np.zeros(jm)
    dxp = np.zeros(jm)
    lat[0] = -.25 * TWOPI
    lat[jm - 1] = -lat[0]
    for j in range(2, jm):
        lat[j - 1] = dlat * (j - FJEQ)
        cosp[j - 1] = np.cos(lat[j - 1])
        dxp[j - 1] = radius * DLON * cosp[j - 1]
    lat1 = dlat * (1. - FJEQ)
    cosp1 = np.cos(lat1)
    dxp1 = radius * DLON * cosp1
    cosv = np.zeros(jm)
    dxv = np.zeros(jm)
    dyv = np.zeros(jm)
    for j in range(2, jm + 1):
        cosv[j - 1] = .5 * (cosp[j - 2] + cosp[j - 1])
        dxv[j - 1] = .5 * (dxp[j - 2] + dxp[j - 1])
        dyv[j - 1] = radius * (lat[j - 1] - lat[j - 2])
        if j == 2:
            cosv[j - 1] = .5 * (cosp1 + cosp[j - 1])
            dxv[j - 1] = .5 * (dxp1 + dxp[j - 1])
        if j == jm:
            cosv[j - 1] = .5 * (cosp[j - 2] + cosp1)
            dxv[j - 1] = .5 * (dxp[j - 2] + dxp1)
    dyp = np.zeros(jm)
    dyp[0] = radius * (lat[1] - lat[0] - 0.5 * dlat)
    dyp[jm - 1] = radius * (lat[jm - 1] - lat[jm - 2] - 0.5 * dlat)
    dxyp = np.zeros(jm)
    sinv = np.sin(dlat * (1 + .5 - FJEQ))
    dxyp[0] = radius * radius * DLON * (sinv + 1)
    sinvm1 = np.sin(dlat * (jm - .5 - FJEQ))
    dxyp[jm - 1] = radius * radius * DLON * (1 - sinvm1)
    dxys = np.zeros(jm)
    dxyn = np.zeros(jm)
    dxys[jm - 1] = dxyp[jm - 1]
    dxyn[0] = dxyp[0]
    for j in range(2, jm):
        dyp[j - 1] = radius * dlat
        sinvm1 = np.sin(dlat * (j - .5 - FJEQ))
        sinv = np.sin(dlat * (j + .5 - FJEQ))
        dxyp[j - 1] = radius * radius * DLON * (sinv - sinvm1)
        dxys[j - 1] = .5 * dxyp[j - 1]
        dxyn[j - 1] = .5 * dxyp[j - 1]
    bydyp = 1.0 / dyp
    dxyv = np.zeros(jm)
    rapvs = np.zeros(jm)
    rapvn = np.zeros(jm)
    for j in range(2, jm + 1):
        dxyv[j - 1] = dxyn[j - 2] + dxys[j - 1]
        rapvs[j - 1] = .5 * dxys[j - 1] / dxyv[j - 1]
        rapvn[j - 2] = .5 * dxyn[j - 2] / dxyv[j - 1]
    siniv = np.array([np.sin((i - 1) * DLON) for i in range(1, im + 1)])
    cosiv = np.array([np.cos((i - 1) * TWOPI * BYIM) for i in range(1, im + 1)])
    return dict(radius=radius, dlon=DLON, fjeq=FJEQ, lat=lat, cosp=cosp, dxp=dxp, dyp=dyp, bydyp=bydyp,
                cosv=cosv, dxv=dxv, dyv=dyv, dxyp=dxyp, dxys=dxys, dxyn=dxyn, dxyv=dxyv, rapvs=rapvs,
                rapvn=rapvn, siniv=siniv, cosiv=cosiv)
