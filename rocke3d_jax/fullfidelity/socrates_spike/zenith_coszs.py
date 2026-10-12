"""COSZS of GEOM_B.f (RAD_COSZ0): time mean cosine of the zenith angle COSZ and the sunlit-weighted mean COSZA over the interval of NRAD steps, as
used by RADIA (calc_zenith_angle(modelEclock, NRAD, cosz2, cosza)).  Built on the D177 port of the orbit and of COSZT (drv_zenith.py)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import drv_zenith as Z

PI, TWOPI = Z.PI, Z.TWOPI


def coszs_grid(grid, rot1, rot2, sind, cosd):
    """grid: Z.CoszGrid.  Returns (cosz, cosza), each (IM, JM)."""
    m = grid.m
    sin, cos, acos, sqrt = m['sin'], m['cos'], m['acos'], m['sqrt']
    IM, JM = Z.IM, Z.JM
    zero1 = 1.0e-2
    drot = rot2 - rot1
    sr1, cr1, sr2, cr2 = sin(rot1), cos(rot1), sin(rot2), cos(rot2)
    lt1 = rot1 + grid.lon
    lt2 = rot2 + grid.lon
    slt1 = sr1 * grid.cosip + cr1 * grid.sinip
    clt1 = cr1 * grid.cosip - sr1 * grid.sinip
    s2lt1 = 2.0 * slt1 * clt1
    slt2 = sr2 * grid.cosip + cr2 * grid.sinip
    clt2 = cr2 * grid.cosip - sr2 * grid.sinip
    s2lt2 = 2.0 * slt2 * clt2
    cz = np.zeros((IM, JM + 1))
    cza = np.zeros((IM, JM + 1))
    for j in (1, JM):
        sjsd = grid.sinj[j] * sind
        cjcd = grid.cosj[j] * cosd
        if sjsd + cjcd > zero1:
            if sjsd - cjcd < 0.0:
                cdusk = -sjsd / cjcd
                dusk = acos(cdusk)
                sdusk = sqrt(cjcd * cjcd - sjsd * sjsd) / cjcd
                s2dusk = 2.0 * sdusk * cdusk
                dawn, sdawn, s2dawn = -dusk, -sdusk, -s2dusk
                ecosz = sjsd * (dusk - dawn) + cjcd * (sdusk - sdawn)
                ecosqz = sjsd * ecosz + cjcd * (sjsd * (sdusk - sdawn) + .5 * cjcd * (dusk - dawn + .5 * (s2dusk - s2dawn)))
                cz[0, j] = ecosz / TWOPI
                cza[0, j] = ecosqz / ecosz
            else:
                ecosz = sjsd * TWOPI
                ecosqz = sjsd * ecosz + .5 * cjcd * cjcd * TWOPI
                cz[0, j] = ecosz / TWOPI
                cza[0, j] = ecosqz / ecosz
    for j in range(2, JM):
        sjsd = grid.sinj[j] * sind
        cjcd = grid.cosj[j] * cosd
        if not (sjsd + cjcd > zero1):
            continue
        if sjsd - cjcd < 0.0:
            cdusk = -sjsd / cjcd
            dusk = acos(cdusk)
            sdusk = sqrt(cjcd * cjcd - sjsd * sjsd) / cjcd
            s2dusk = 2.0 * sdusk * cdusk
            dawn, sdawn, s2dawn = -dusk, -sdusk, -s2dusk
            zero2 = zero1 / cjcd
            for i in range(IM):
                if not (dusk > lt1[i] + zero2):
                    dusk = dusk + TWOPI
                    dawn = dawn + TWOPI
                if not (dawn < lt2[i] - zero2):
                    continue                                         # cz = cza = 0
                if not (dawn >= lt1[i]):                              # label 640 -> not 700
                    if not (dusk < lt2[i]):
                        ecosz = sjsd * drot + cjcd * (slt2[i] - slt1[i])
                        ecosqz = sjsd * ecosz + cjcd * (sjsd * (slt2[i] - slt1[i]) + .5 * cjcd * (drot + .5 * (s2lt2[i] - s2lt1[i])))
                    elif not (dawn + TWOPI < lt2[i] - zero2):         # 660
                        ecosz = sjsd * (dusk - lt1[i]) + cjcd * (sdusk - slt1[i])
                        ecosqz = sjsd * ecosz + cjcd * (sjsd * (sdusk - slt1[i]) + .5 * cjcd * (dusk - lt1[i] + .5 * (s2dusk - s2lt1[i])))
                    else:                                             # 680
                        ecosz = sjsd * (drot - dawn - TWOPI + dusk) + cjcd * (slt2[i] - sdawn + sdusk - slt1[i])
                        ecosqz = sjsd * ecosz + cjcd * (sjsd * (sdusk - slt1[i] + slt2[i] - sdawn) + .5 * cjcd * (
                            dusk + drot - dawn - TWOPI + .5 * (s2dusk - s2lt1[i] + s2lt2[i] - s2dawn)))
                else:                                                 # 700
                    if dusk >= lt2[i]:
                        ecosz = sjsd * (lt2[i] - dawn) + cjcd * (slt2[i] - sdawn)
                        ecosqz = sjsd * ecosz + cjcd * (sjsd * (slt2[i] - sdawn) + .5 * cjcd * (lt2[i] - dawn + .5 * (s2lt2[i] - s2dawn)))
                    else:
                        ecosz = sjsd * (dusk - dawn) + cjcd * (sdusk - sdawn)
                        ecosqz = sjsd * ecosz + cjcd * (sjsd * (sdusk - sdawn) + .5 * cjcd * (dusk - dawn + .5 * (s2dusk - s2dawn)))
                cz[i, j] = ecosz / drot
                cza[i, j] = ecosqz / ecosz
        else:
            for i in range(IM):
                ecosz = sjsd * drot + cjcd * (slt2[i] - slt1[i])
                ecosqz = sjsd * ecosz + cjcd * (sjsd * (slt2[i] - slt1[i]) + .5 * cjcd * (drot + .5 * (s2lt2[i] - s2lt1[i])))
                cz[i, j] = ecosz / drot
                cza[i, j] = ecosqz / ecosz
    return cz[:, 1:], cza[:, 1:]


def calc_zenith_angle(zen, itime, nstep):
    """CALC_ZENITH_ANGLE(modelEclock, nstep, cosz, cosza) at the clock reading itime.  zen: drv_zenith.Zenith."""
    t1 = Z.time_of_itime(itime, zen.iyear1)
    sind, cosd = zen.daily(t1)
    rot1 = zen.orbit.hour_angle(t1)
    rot2 = zen.orbit.hour_angle(t1 + nstep * Z.DTSRC)
    drot = rot2 - rot1
    drot = -PI + Z._mod(drot + PI, 2.0 * PI)
    if abs(drot) > PI / 4:
        raise RuntimeError("Time step is too large for accurate radiation.")
    if drot < 0:
        rot2 = rot1
        rot1 = rot1 + drot
    elif drot > 0:
        rot2 = rot1 + drot
    else:
        rot2 = rot2 + 1.0e-10
    return coszs_grid(zen.grid, rot1, rot2, sind, cosd)
