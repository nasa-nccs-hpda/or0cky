"""D98: ADVECV (momentum advection + Coriolis) of the atmospheric dynamics ported to numpy.

Fortran reference: model/MOMEN2ND.f 31-498 (modelE2_planet_2.0, rundeck P2SAoM40).  Stages:
  1. scale UT,VT by VMASS(MBEFOR);
  2. horizontal advection: W-E, S-N, SW-NE and SE-NW mass-flux contributions, accumulated into DUT,DVT in the
     exact Fortran sequence (i outer from I=IM, J inner with carried "previous J" fluxes);
  3. do_polefix=1: the two polar velocity rows (J=2, J=JM) are recomputed from x-y advection of rotated polar
     velocities (corner fluxes dropped, ACOR correction) and replace the rows from stage 2;
  4. vertical advection with SD=MW (ASDU, RAVPN/RAVPS), UT,VT += DUT,DVT, DUT,DVT cleared;
  5. Coriolis with the metric term FD (from SPA of this pass's AFLUX), polar override (2*FCOR(jpo)+FCOR(jns));
  6. UT,VT = (UT,VT + DUT,DVT)/VMASS(MAFTER).

Floating-point order.  DUT,DVT are accumulated sequentially in Fortran, and each cell receives several updates
whose order matters for the last bit.  The port reproduces the per-cell order exactly while vectorising over
(J,L) and over cells: a cell k is touched in the "IP1 role" of loop iteration k and the "I role" of iteration
k+1, i.e. IP1-role updates first then I-role updates, except cell IM whose I-role (iteration 1, I=IM) precedes its
IP1-role (iteration IM).  The same pattern holds for the Coriolis updates (I, then IM1) and the polar fix.
Fluxes use the Fortran left-to-right sums/products.  Geometry comes from the recorded dump (ffd_aflux_geom.bin);
inputs MU,MV,MW,SPA are the module arrays as left by this pass's AFLUX (recorded in the ADVECV input dump).

Array conventions as in dyn_aflux_ff.py: U,V,UT,VT,PU(MU),PV(MV),SPA (IM,JM,LM); SD(MW) (IM,JM,LM-1);
MMEAN,MBEFOR,MAFTER (LM,IM,JM); numpy index = Fortran index - 1.
"""
import numpy as np

from dyn_aflux_ff import IM, JM, LM


def _roll_p(a):
    """Align the pair index: result[k] = a[k-1] (pair (I=k-1, IP1=k), cyclic)."""
    return np.roll(a, 1, axis=0)


def _hadv_component(wE, nS, sW, sE, d0):
    """Accumulate one velocity component (U or V) over rows J=2..JM exactly as the Fortran loop 230.
    wE (IM,JM-1,LM): W-E flux of pair p at rows j=1..JM-1 (0-based);
    nS,sW,sE (IM,JM-2,LM): new fluxes of pair p at rows j=1..JM-2, "carry" of row j is the flux of row j-1;
    d0 (IM,JM-1,LM): starting DUT/DVT (zero in the real calls). Returns accumulated array (IM,JM-1,LM)."""
    shp = wE.shape
    zero = np.zeros((IM, 1, shp[2]))
    # carries for rows j=1..JM-1: zeros at the first row (south pole boundary), then flux of previous row
    carry_ns = np.concatenate([zero, nS], axis=1)
    carry_sw = np.concatenate([zero, sW], axis=1)
    carry_se = np.concatenate([zero, sE], axis=1)
    # "new" terms exist for rows 1..JM-2 only (north pole row JM-1 has no new terms)
    last = np.zeros((IM, 1, shp[2]))
    has_new = np.concatenate([np.ones((1, JM - 2, 1)), np.zeros((1, 1, 1))], axis=1)
    nS_new = np.concatenate([nS, last], axis=1)
    sW_new = np.concatenate([sW, last], axis=1)
    sE_new = np.concatenate([sE, last], axis=1)

    def ip1_seq(r):                                          # cell k as IP1 of pair k-1
        r = r + _roll_p(wE)
        r = r + _roll_p(carry_sw)
        # -SE_NW(new) only where a new term exists; subtracting an exact 0.0 elsewhere changes nothing
        r = r - _roll_p(sE_new)
        return r

    def i_seq(r):                                            # cell k as I of pair k
        r = r - wE
        r = r + carry_ns
        r = r + carry_se
        r = r - nS_new
        r = r - sW_new
        return r
    a = i_seq(ip1_seq(d0))
    b = ip1_seq(i_seq(d0))
    a[IM - 1] = b[IM - 1]                                    # cell IM: I-role (iteration 1) precedes IP1-role
    return a


def advecv(dt1, u, v, mmean, mbefor, ut, vt, mafter, pu, pv, sd, spa, g, polefix=True, stages=None):
    """ADVECV.  pu=MU, pv=MV (kg/s, IM,JM,LM), sd=MW (IM,JM,LM-1), spa (IM,JM,LM) as left by AFLUX of the same
    pass.  Returns (ut, vt); inputs are not modified.  `stages` (dict) collects intermediates."""
    polwt, acor = g['polwt'], g['acor']
    dxyn, dxys, dxv = g['dxyn'], g['dxys'], g['dxv']
    fcor, ravpn, ravps = g['fcor'], g['ravpn'], g['ravps']
    sini, cosi = g['siniv'], g['cosiv']
    dt2 = dt1 / 2.
    dt8 = dt1 / 8.
    dt12 = dt1 / 12.
    dt24 = dt1 / 24.
    ut = ut.copy(); vt = vt.copy()
    ip1 = np.roll(np.arange(IM), -1)                        # IP1 index of each I
    im1 = np.roll(np.arange(IM), 1)

    def vmass_of(m):                                        # J=2..JM  -> (IM,JM-1,LM)
        mt = m.transpose(1, 2, 0)
        return .5 * (((mt[:, 0:JM - 1, :] + mt[ip1][:, 0:JM - 1, :]) * dxyn[None, 0:JM - 1, None])
                     + ((mt[:, 1:JM, :] + mt[ip1][:, 1:JM, :]) * dxys[None, 1:JM, None]))
    # 1. scale UT,VT
    vm0 = vmass_of(mbefor)
    ut[:, 1:JM, :] = ut[:, 1:JM, :] * vm0
    vt[:, 1:JM, :] = vt[:, 1:JM, :] * vm0
    # interpolated polar velocities used by the horizontal advection
    uc = u.copy(); vc = v.copy()
    uc[:, 1, :] = polwt * u[:, 1, :] + (1 - polwt) * u[:, 2, :]
    vc[:, 1, :] = polwt * v[:, 1, :] + (1 - polwt) * v[:, 2, :]
    uc[:, JM - 1, :] = polwt * u[:, JM - 1, :] + (1 - polwt) * u[:, JM - 2, :]
    vc[:, JM - 1, :] = polwt * v[:, JM - 1, :] + (1 - polwt) * v[:, JM - 2, :]
    # 2. horizontal advection fluxes (pair p = I, partner IP1)
    j = slice(1, JM)                                        # J=2..JM
    jm1 = slice(0, JM - 1)
    flux = dt12 * (((pu[ip1][:, j, :] + pu[ip1][:, jm1, :]) + pu[:, j, :]) + pu[:, jm1, :])
    wE_u = flux * (uc[ip1][:, j, :] + uc[:, j, :])
    wE_v = flux * (vc[ip1][:, j, :] + vc[:, j, :])
    jr = slice(1, JM - 1)                                   # J=2..JM-1 (edge between J and J+1)
    jr1 = slice(2, JM)
    f_ns = dt12 * (((pv[:, jr, :] + pv[ip1][:, jr, :]) + pv[:, jr1, :]) + pv[ip1][:, jr1, :])
    nS_u = f_ns * (uc[:, jr, :] + uc[:, jr1, :])
    nS_v = f_ns * (vc[:, jr, :] + vc[:, jr1, :])
    f_sw = dt24 * (((pu[ip1][:, jr, :] + pu[:, jr, :]) + pv[ip1][:, jr, :]) + pv[ip1][:, jr1, :])
    sW_u = f_sw * (uc[ip1][:, jr1, :] + uc[:, jr, :])
    sW_v = f_sw * (vc[ip1][:, jr1, :] + vc[:, jr, :])
    f_se = dt24 * ((((-pu[ip1][:, jr, :]) - pu[:, jr, :]) + pv[ip1][:, jr, :]) + pv[ip1][:, jr1, :])
    sE_u = f_se * (uc[:, jr1, :] + uc[ip1][:, jr, :])
    sE_v = f_se * (vc[:, jr1, :] + vc[ip1][:, jr, :])
    z = np.zeros((IM, JM - 1, LM))
    dut = np.zeros((IM, JM, LM)); dvt = np.zeros((IM, JM, LM))
    dut[:, 1:JM, :] = _hadv_component(wE_u, nS_u, sW_u, sE_u, z)
    dvt[:, 1:JM, :] = _hadv_component(wE_v, nS_v, sW_v, sE_v, z)
    if stages is not None:
        stages['dut_hadv'] = dut.copy(); stages['dvt_hadv'] = dvt.copy()
    # 3. polar rows from x-y advection (do_polefix=1); U,V are the restored (uninterpolated) values
    if g['do_polefix'] == 1 and polefix:
        for pole in (0, 1):
            if pole == 0:
                hemi, jpo, jns, jv, jvs, jvn, wts = -1, 0, 1, 1, 1, 2, polwt
            else:
                hemi, jpo, jns, jv, jvs, jvn, wts = 1, JM - 1, JM - 2, JM - 1, JM - 2, JM - 1, 1. - polwt
            c = cosi[:, None]; s = (hemi * sini)[:, None]
            UP = {}; VP = {}
            for jj in (jvs, jvn):
                UP[jj] = c * u[:, jj, :] - s * v[:, jj, :]
                VP[jj] = c * v[:, jj, :] + s * u[:, jj, :]
            upv = wts * UP[jvs] + (1. - wts) * UP[jvn]       # UP(:,jv) = interpolation
            vpv = wts * VP[jvs] + (1. - wts) * VP[jvn]
            UP[jv] = upv; VP[jv] = vpv
            fl = dt8 * (((pu[ip1][:, jpo, :] + pu[:, jpo, :]) + pu[ip1][:, jns, :]) + pu[:, jns, :])   # W-E, pair p
            fu = fl * (UP[jv][ip1] + UP[jv])
            fv = fl * (VP[jv][ip1] + VP[jv])
            sn = dt8 * (((pv[:, jvs, :] + pv[ip1][:, jvs, :]) + pv[:, jvn, :]) + pv[ip1][:, jvn, :])
            sn = sn * hemi
            snu = sn * (UP[jvs] + UP[jvn])
            snv = sn * (VP[jvs] + VP[jvn])
            zz = np.zeros((IM, LM))

            def acc(wf, snf):
                """cell k: IP1 role (+W-E of pair k-1) then I role (-W-E of pair k, +S-N of pair k); cell IM reversed."""
                def ip1_seq(r):
                    return r + _roll_p(wf)
                def i_seq(r):
                    r = r - wf
                    return r + snf
                a = i_seq(ip1_seq(zz)); b = ip1_seq(i_seq(zz))
                a[IM - 1] = b[IM - 1]
                return a
            du = acc(fu, snu)
            dv = acc(fv, snv)
            # DMT(IP1)=DMT+FLUX+FLUX; DMT(I)=DMT-FLUX-FLUX; DMT(I)=DMT+SN+SN
            def acc_m(wf, snf):
                def ip1_seq(r):
                    return (r + _roll_p(wf)) + _roll_p(wf)
                def i_seq(r):
                    r = (r - wf) - wf
                    return (r + snf) + snf
                a = i_seq(ip1_seq(zz)); b = ip1_seq(i_seq(zz))
                a[IM - 1] = b[IM - 1]
                return a
            dmt = acc_m(fl, sn)
            du = du + (acor - 1.) * (du - dmt * UP[jv])
            dv = dv + (acor - 1.) * (dv - dmt * VP[jv])
            utmp, vtmp = du, dv
            dut[:, jv, :] = c * utmp + s * vtmp
            dvt[:, jv, :] = c * vtmp - s * utmp
    if stages is not None:
        stages['dut_pole'] = dut.copy(); stages['dvt_pole'] = dvt.copy()
    # 4. vertical advection (U,V restored values)
    sdt = sd.transpose(2, 0, 1)                              # (LM-1,IM,JM)
    asdu = np.zeros((IM, JM, LM - 1))
    sdi = np.roll(sd, -1, axis=0)                            # SD(I+1) (I=IM -> SD(1))
    asdu[:, 1:JM, :] = dt2 * (((sd[:, 0:JM - 1, :] + sdi[:, 0:JM - 1, :]) * ravpn[None, 0:JM - 1, None])
                              + ((sd[:, 1:JM, :] + sdi[:, 1:JM, :]) * ravps[None, 1:JM, None]))
    for (dd, xx) in ((dut, u), (dvt, v)):
        rows = slice(1, JM)
        dd[:, rows, 0] = dd[:, rows, 0] + asdu[:, rows, 0] * (xx[:, rows, 0] + xx[:, rows, 1])
        for l in range(1, LM - 1):
            dd[:, rows, l] = dd[:, rows, l] - asdu[:, rows, l - 1] * (xx[:, rows, l - 1] + xx[:, rows, l])
            dd[:, rows, l] = dd[:, rows, l] + asdu[:, rows, l] * (xx[:, rows, l] + xx[:, rows, l + 1])
        l = LM - 1
        dd[:, rows, l] = dd[:, rows, l] - asdu[:, rows, l - 1] * (xx[:, rows, l - 1] + xx[:, rows, l])
    ut[:, 1:JM, :] = ut[:, 1:JM, :] + dut[:, 1:JM, :]
    vt[:, 1:JM, :] = vt[:, 1:JM, :] + dvt[:, 1:JM, :]
    if stages is not None:
        stages['ut_adv'] = ut.copy(); stages['vt_adv'] = vt.copy()
    # 5. Coriolis (DUT,DVT restart from zero)
    mm = mmean.transpose(1, 2, 0)
    fd = np.zeros((IM, JM, LM))
    spa_m = np.roll(spa, 1, axis=0)                          # SPA(IM1,..)
    fd[:, 0, :] = (-.5 * (spa_m[:, 0, :] + spa[:, 0, :])) * dxv[1]
    fd[:, JM - 1, :] = (.5 * (spa_m[:, JM - 1, :] + spa[:, JM - 1, :])) * dxv[JM - 1]
    jj = slice(1, JM - 1)
    fd[:, jj, :] = fcor[None, jj, None] + (.25 * (spa_m[:, jj, :] + spa[:, jj, :])) * (dxv[None, 1:JM - 1, None] - dxv[None, 2:JM, None])
    pdt4 = dt8 * (mm[:, 0:JM - 1, :] + mm[:, 1:JM, :])      # DT8*(MMEAN(L,I,J-1)+MMEAN(L,I,J))
    alph = pdt4 * (fd[:, 1:JM, :] + fd[:, 0:JM - 1, :])      # (IM,JM-1,LM)
    ud, vd = _coriolis_acc(alph, u[:, 1:JM, :], v[:, 1:JM, :])
    dut[:, 1:JM, :] = ud; dvt[:, 1:JM, :] = vd
    if g['do_polefix'] == 1 and polefix:
        for jpo, jns, jrow in ((0, 1, 1), (JM - 1, JM - 2, JM - 1)):
            pdt4p = dt8 * (mm[:, jpo, :] + mm[:, jns, :])
            alp = pdt4p * (2 * fcor[jpo] + fcor[jns])
            udp, vdp = _coriolis_acc(alp[:, None, :], u[:, jrow:jrow + 1, :], v[:, jrow:jrow + 1, :])
            dut[:, jrow, :] = udp[:, 0, :]; dvt[:, jrow, :] = vdp[:, 0, :]
    if stages is not None:
        stages['dut_cor'] = dut.copy(); stages['dvt_cor'] = dvt.copy()
    # 6. final: (UT+DUT)/VMASS(MAFTER)
    vm1 = vmass_of(mafter)
    ut[:, 1:JM, :] = (ut[:, 1:JM, :] + dut[:, 1:JM, :]) / vm1
    vt[:, 1:JM, :] = (vt[:, 1:JM, :] + dvt[:, 1:JM, :]) / vm1
    return ut, vt


def _coriolis_acc(alph, ur, vr):
    """DUT(I)+=ALPH(I)*V(I); DUT(IM1)+=ALPH(I)*V(IM1); DVT(I)-=ALPH(I)*U(I); DVT(IM1)-=ALPH(I)*U(IM1), I=1..IM.
    Cell k receives ALPH_k first (I role), then ALPH_{k+1} (IM1 role of iteration k+1), except cell IM whose IM1
    role (iteration 1, ALPH_1) comes first.  Start value zero."""
    an = np.roll(alph, -1, axis=0)                           # ALPH(k+1)
    ud = (0. + alph * vr) + an * vr
    vd = (0. - alph * ur) - an * ur
    ud[IM - 1] = (0. + alph[0] * vr[IM - 1]) + alph[IM - 1] * vr[IM - 1]
    vd[IM - 1] = (0. - alph[0] * ur[IM - 1]) - alph[IM - 1] * ur[IM - 1]
    return ud, vd
