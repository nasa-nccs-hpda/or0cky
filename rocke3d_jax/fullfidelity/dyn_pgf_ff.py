"""D97: PGF (pressure gradient force, live non-V2_PGF version) of the atmospheric dynamics ported to numpy.

Fortran reference: model/ATMDYN.f 1107-1324 (modelE2_planet_2.0, rundeck P2SAoM40; V2_PGF undefined).
  1. per-column top-down integration of pressure and layer geopotential (**KAPA powers), AdM=SPA scratch,
     GZ (= PHI) exported to the physics;
  2. polar columns copied from I=1;
  3. N-S derivative -> DVT (sequential accumulate, two updates per cell in the Fortran I order);
  4. smoothed E-W derivative PGFU -> AVRX (rows 2..JM-1) -> DUT;
  5. do_polefix=1: DUT,DVT at the polar velocity rows scaled by ACOR, ACOR2;
  6. UT,VT += DUT,DVT / VMASS.

Array conventions as in dyn_aflux_ff.py: U-like (IM,JM,LM), MAM/MAFTER (LM,IM,JM).  DUT,DVT are the DYNAMICS
module accumulators; at PGF entry the real run has them zero in rows 2..JM (ADVECV zeroes them on exit); the
dump records them anyway and the port takes them as inputs.

PK-type powers: x**KAPA in ifort calls the Intel libimf scalar pow, which numpy's pow does not reproduce
bit-for-bit (D96 finding, intel_libm_ff.py).  `imf_pow=True` uses libimf (bitwise); the default is numpy pow
(a few 1e-16-relative differences that propagate to GZ).
"""
import numpy as np

import intel_libm_ff
from dyn_aflux_ff import IM, JM, LM, avrx_field


def _pw(x, kapa, imf_pow):
    return intel_libm_ff.pow_imf(x, kapa) if imf_pow else x ** kapa


def pgf(dt1, mam, ut, vt, mafter, s0, sz, dut, dvt, g, tab=None, imf_pow=False, polefix=True,
        avrx=True, stats=None):
    """PGF.  Returns dict(ut, vt, dut, dvt, gz, phi, adm(=SPA), pgfu0, pgfu).  Inputs are not modified."""
    kapa, grav, rgas = g['kapa'], g['grav'], g['rgas']
    zK, zKp1, zKp2, byGRAV = g['bykapa'], g['bykapap1'], g['bykapap2'], g['bygrav']
    mtop = g['mtop']
    ut = ut.copy(); vt = vt.copy(); dut = dut.copy(); dvt = dvt.copy()
    dt4 = dt1 / 4.
    hk = _pw(np.array([.01]), kapa, imf_pow)[0] if imf_pow else .01 ** kapa   # HUNDREDTHeKAPA = .01d0**KAPA
    mam_t = mam.transpose(1, 2, 0)                                   # (IM,JM,LM) view of MAM(L,I,J)
    shp = (IM, JM)
    adm = np.zeros((IM, JM, LM)); p = np.zeros((IM, JM, LM)); gz = np.zeros((IM, JM, LM))
    dgzu = np.zeros((IM, JM, LM)); dgza = np.zeros((IM, JM, LM))
    m = np.full(shp, mtop)
    pu = m * grav
    pku = _pw(pu, kapa, imf_pow); pkpu = pku * pu; pkppu = pkpu * pu
    for l in range(LM - 1, -1, -1):                                  # integrate pressures from the top down
        dp = mam_t[:, :, l] * grav
        zdp = 1 / dp
        y = sz[:, :, l] * 2 * zdp * hk
        x = s0[:, :, l] * hk + y * (pu + .5 * dp)
        pd = pu + dp
        pkd = _pw(pd, kapa, imf_pow); pkpd = pkd * pd; pkppd = pkpd * pd
        dgzu[:, :, l] = rgas * (x * (pkd - pku) * zK - y * (pkpd - pkpu) * zKp1)
        dgza[:, :, l] = rgas * (x * (dp * pkd - (pkpd - pkpu) * zKp1) * zK
                                - y * (dp * pkpd - (pkppd - pkppu) * zKp2) * zKp1) * zdp
        adm[:, :, l] = dgzu[:, :, l] * byGRAV
        p[:, :, l] = grav * (m + .5 * mam_t[:, :, l])
        m = m + mam_t[:, :, l]
        pu = pd
        pku = pkd; pkpu = pkpd; pkppu = pkppd
    gzd = g['zatmo'].copy()                                          # integrate altitude from the bottom up
    for l in range(LM):
        gz[:, :, l] = gzd + dgza[:, :, l]
        gzd = gzd + dgzu[:, :, l]
    # polar values from I=1 (Fortran computes only I<=IMAXJ(J) = 1 there)
    for arr in (adm, p, gz):
        arr[1:, 0, :] = arr[0, 0, :][None, :]
        arr[1:, JM - 1, :] = arr[0, JM - 1, :][None, :]
    phi = gz.copy()
    # N-S derivative -> DVT, J=2..JM
    fac = (dt4 * g['dxv'])[None, 1:JM, None]                         # FACTOR=DT4*DXV(J)
    flux = (((adm[:, 1:JM, :] + adm[:, 0:JM - 1, :]) * (p[:, 1:JM, :] - p[:, 0:JM - 1, :])
             + (mam_t[:, 1:JM, :] + mam_t[:, 0:JM - 1, :]) * (gz[:, 1:JM, :] - gz[:, 0:JM - 1, :])) * fac)
    d0 = dvt[:, 1:JM, :]
    # iteration i: DVT(i)-=F_i then DVT(i-1)-=F_i (i=1 -> IM): cell k receives F_k then F_{k+1}, except
    # cell IM which receives F_1 (as IM1) before its own F_IM
    f_next = np.roll(flux, -1, axis=0)
    dv = (d0 - flux) - f_next
    dv[IM - 1] = (d0[IM - 1] - flux[0]) - flux[IM - 1]
    dvt[:, 1:JM, :] = dv
    # smoothed E-W derivative
    pgfu = np.zeros((IM, JM, LM))
    ip1 = np.roll(np.arange(IM), -1)
    pgfu[:, 1:JM, :] = ((adm[ip1][:, 1:JM, :] + adm[:, 1:JM, :]) * (p[ip1][:, 1:JM, :] - p[:, 1:JM, :])
                        + (mam_t[ip1][:, 1:JM, :] + mam_t[:, 1:JM, :]) * (gz[ip1][:, 1:JM, :] - gz[:, 1:JM, :]))
    pgfu0 = pgfu.copy()
    if avrx:
        pgfu = avrx_field(pgfu, range(1, JM - 1), g, tab)            # JRANGE=(2, JM-1)
    facu = (-dt4 * g['dyv'])[None, 1:JM, None]                       # FACTOR=-DT4*DYV(J)
    dut[:, 1:JM, :] = dut[:, 1:JM, :] + facu * (pgfu[:, 1:JM, :] + pgfu[:, 0:JM - 1, :])
    if g['do_polefix'] == 1 and polefix:
        for j in (1, JM - 1):
            dut[:, j, :] = dut[:, j, :] * g['acor']
            dvt[:, j, :] = dvt[:, j, :] * g['acor2']
    # undo the scaling performed at the beginning of ADVECV, J=2..JM
    mn = mafter.transpose(1, 2, 0)
    vmass = .5 * (((mn[:, 0:JM - 1, :] + mn[ip1][:, 0:JM - 1, :]) * g['dxyn'][None, 0:JM - 1, None])
                  + ((mn[:, 1:JM, :] + mn[ip1][:, 1:JM, :]) * g['dxys'][None, 1:JM, None]))
    ut[:, 1:JM, :] = ut[:, 1:JM, :] + dut[:, 1:JM, :] / vmass
    vt[:, 1:JM, :] = vt[:, 1:JM, :] + dvt[:, 1:JM, :] / vmass
    return dict(ut=ut, vt=vt, dut=dut, dvt=dvt, gz=gz, phi=phi, adm=adm, pgfu0=pgfu0, pgfu=pgfu)
