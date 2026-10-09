"""D205: NumPy reference port of the Fortran `daily_LAKE` (LAKES.f:2492-3010, the file the P2SAoM40 build links: decks/P2SAoM40.mk `OBJ_LIST += LAKES_COM LAKES`),
called once per model day from `daily_atm` (ATM_DRV.f:1031, after daily_RAD and before daily_EARTH / daily_LI / UPDTYPE), plus the helpers it needs:
  cubicroot   shared/CubicEquation_mod.F90 (Cardano; conical lakes with soil saturation)
  newton      LAKES.f:4059 (only reached with Power_law_lakes > 0; ported, NOT exercised: the rundeck leaves Power_law_lakes = 0)
  water_deficit  GHY_DRV.f:4054-4129 `compute_water_deficit` (DMWLDF, the soil saturation deficit that daily_LAKE reads; computed by GHY at the END OF EVERY STEP)
  tanlk       LAKES.f:770-784 (TANLK from FLAKE0 and HLAKE, Cael_lakes = 0)

Statement order follows the Fortran (one scalar cell at a time, np.float64 arithmetic, the integer powers as products, the real powers through the Intel libimf pow when
`intel_libm_ff` is available and numpy's pow otherwise: `POW` reports which).  Parameters (LAKES.f:60-105, LAKES.f:656-661; the rundeck P2SAoM40.R sets only variable_lk=1 and
init_flake=1): variable_lk = 1, MINMLD = 1 m, lake_ice_max = 5 m (< 1e10: lake ice is not melted away, the "implicit" (MDWNIMP/EDWNIMP) branch), Power_law_lakes = 0 (conical lakes),
small_lake_evap = 0, Cael_lakes = 0, C_lake/E_lake unused.

NOT ported (and why):
  * TRACERS_WATER branches (the build has no water tracers: nothing in the restarts/records);
  * the diagnostics (INC_AJ, INC_AREG, AIJ): not part of the model state;
  * RESET_SURF_FLUXES (RAD_DRV.f:5503): it edits FSF/TRSURF (the radiation-flux surface arrays used to restart the radiation); radiation is a replayed record in this
    port, so the calls are only COUNTED/LISTED in `out['reset_surf_fluxes']` (itype_old, itype_new, ftype_orig, ftype_now);
  * IRRIGATION_ON read_irrig (not compiled in this rundeck: no IRRIGATION_ON in decks/P2SAoM40.mk, not checked beyond that);
  * the HALO_UPDATEs (single domain).
The inputs MDWNIMP/EDWNIMP/DGML/DMWLDF are returned as increments/arrays so the caller can feed daily_LI / GHY (`dfrac` branch at GHY_DRV.f:4531) if those are ported.
"""
import math

import numpy as np

IM, JM = 72, 46
PI = math.pi
BY3 = 1.0 / 3.0                      # MathematicalConstants_mod by3 = 1d0/3d0
RHOW = 1000.0
LHM = 3.34e5
SHW = 4185.0
TEENY = 1e-30
TF = 273.15
MINMLD = 1.0                         # LAKES.f:61
HLAKE_MIN = 1.0                      # LAKES.f:63
VARIABLE_LK = 1                      # P2SAoM40.R:176
LAKE_ICE_MAX = 5.0                   # LAKES.f:96 (default)
POWER_LAW_LAKES = 0                  # LAKES.f:98 (default)
SMALL_LAKE_EVAP = 0                  # LAKES.f:105 (default)
C_LAKE, E_LAKE = 0.235, 1.204
RHOI = 916.6                         # SEAICE.f RHOI (seaice_core_jax.RHOI)
ACE1I = 0.1 * RHOI                   # SEAICE.f:34 Z1I*RHOI
AC2OIM = 0.1 * RHOI                  # SEAICE.f:38 Z2OIM*RHOI
XSI = (0.5, 0.5, 0.5, 0.5)

try:                                 # Intel libimf pow for the REAL exponents (x**BY3, abs(arg)**one3rd), as the real ifort build
    import intel_libm_ff as _IMF
    _pow_f = _IMF._load() if _IMF.available() else None
except Exception:                    # pragma: no cover
    _pow_f = None
POW = 'libimf' if _pow_f is not None else 'numpy'


def _pow(x, y):
    if _pow_f is not None:
        return _pow_f(float(x), float(y))
    return float(np.power(np.float64(x), np.float64(y)))


# ------------------------------------------------------------------------------------------------------ helpers
def cubicroot(a, b, c, d):
    """shared/CubicEquation_mod.F90 cubicroot -> (roots list, n).  Same branches and statement order."""
    EPS0 = 1e-8
    one3rd = 1.0 / 3.0
    if abs(a) < (abs(b) + abs(c) + abs(d)) * EPS0:
        if abs(b) < (abs(c) + abs(d)) * EPS0:
            if abs(c) < abs(d) * EPS0:
                raise RuntimeError('Internal Error in Cardano: no solution.')
            x0 = -d / c
            n = 1
            x1 = 0.0
        else:
            D1 = c * c - 4.0 * b * d
            if D1 > 0.0:
                Q1 = math.sqrt(D1)
                x0 = (-c + Q1) / (2.0 * b)
                x1 = (-c - Q1) / (2.0 * b)
                n = 2
            elif D1 == 0.0:
                x0 = -c / (2.0 * b)
                x1 = x0
                n = 1
            else:
                x0 = -c / (2.0 * b)
                x1 = math.sqrt(-D1) / (2.0 * b)
                n = 0
        return [x0, x1], n
    a2 = b / a
    a1 = c / a
    a0 = d / a
    Q1 = (3.0 * a1 - a2 * a2) / 9.0
    R1 = (9.0 * a2 * a1 - 27.0 * a0 - 2.0 * a2 * a2 * a2) / 54.0
    D1 = Q1 * Q1 * Q1 + R1 * R1
    if D1 > 0.0:
        arg = R1 + math.sqrt(D1)
        S = math.copysign(1.0, arg) * _pow(abs(arg), one3rd)
        arg = R1 - math.sqrt(D1)
        T = math.copysign(1.0, arg) * _pow(abs(arg), one3rd)
        x0 = -a2 / 3.0 + S + T
        x1 = -a2 / 3.0 - (S + T) * 0.5
        x2 = math.sqrt(3.0) * (S - T) * 0.5
        n = 1
    elif D1 == 0.0:
        S = math.copysign(1.0, R1) * _pow(abs(R1), one3rd)
        x0 = -a2 / 3.0 + 2.0 * S
        x1 = -a2 / 3.0 - S
        x2 = x1
        n = 2
    else:
        ST = complex(R1, math.sqrt(-D1)) ** one3rd
        S, T = ST.real, ST.imag
        x0 = -a2 / 3.0 + 2.0 * S
        x1 = -a2 / 3.0 - S + math.sqrt(3.0) * T
        x2 = -a2 / 3.0 - S - math.sqrt(3.0) * T
        n = 3
    return [x0, x1, x2], n


def newton(b, c, d, errmax):
    """LAKES.f:4059 newton(a,b,c,d,errmax,n) -> (a, n).  Power_law_lakes > 0 only (not exercised)."""
    a = 0.5 * c / b
    n = 0
    Foff = _pow(a, d) + a * b - c
    while n < 10:
        a = a - Foff / (d * _pow(a, d - 1) + b)
        alow = (1 - errmax) * a
        Foff = _pow(alow, d) + alow * b - c
        n += 1
        if Foff < 0:
            return a, n
        a = alow
    return a, n


def tanlk(flake0, hlake, axyp):
    """LAKES.f:770-784 (Power_law_lakes < 1, Cael_lakes = 0): TANLK = sqrt(FLAKE0*AXYP/PI)/(3*HLAKE) where FLAKE0 > 0, else 2e3.  Arrays (IM, JM)."""
    out = np.full(np.shape(flake0), 2e3)
    m = np.asarray(flake0) > 0
    out[m] = np.sqrt(flake0[m] * axyp[m] / PI) / (3.0 * hlake[m])
    return out


def lake_statics(topo, flake, axyp):
    """HLAKE (DLAKE0) and TANLK as the Fortran init_LAKES sets them (LAKES.f:670-690, 770-784): HLAKE from the TOPO file, at least HLAKE_MIN = 1 m where FLAKE0 + FLAKE > 0;
    TANLK from FLAKE0 (TOPO 'flake') and that HLAKE.  topo: jax_static.topography() dict; flake: the restart/current FLAKE.  Returns (hlake, tanlk)."""
    hl = np.where(topo['flake'] + flake > 0, np.maximum(topo['hlake'], HLAKE_MIN), topo['hlake'])
    return hl, tanlk(topo['flake'], hl, axyp)


def water_deficit(g_rows, w, ecells, fearth, thm0, im=IM, jm=JM):
    """GHY_DRV.f:4054-4129 compute_water_deficit on the recorded GHY rows.  g_rows: (Ne, 450) ffg rows of the LAST step (dz = cols 75-80, q = 81-110 (5x6),
    Ent fv = col 169 with the get_fb_fv round-off rule), w: (Ne, >=7, 2) soil water after the step (w(0:6, 1:2) of GHY: index 0 = canopy), ecells = j*IM+i flat cell of
    each row, fearth (IM, JM), thm0 = thm(0, 1:4) (saturated relative water content of the 4 textures).  Returns DMWLDF (IM, JM) kg/m2 (0 where fearth <= 0)."""
    out = np.zeros((im, jm))
    ngm, imt = 6, 5
    for r in range(len(ecells)):
        i, j = int(ecells[r]) % im, int(ecells[r]) // im
        if fearth[i, j] <= 0.0:
            continue
        row = g_rows[r]
        dz = row[74:80]
        q = row[80:110].reshape(5, 6, order='F')
        fv = float(row[169])
        if fv > 1.0 - 1e-6:
            fv = 1.0
        if fv < 1e-6:
            fv = 0.0
        fb = 1.0 - fv
        wr = w[r]
        w_stor = [0.0, 0.0]
        w_tot = [0.0, 0.0]
        for ibv in range(2):
            for k in range(ngm):
                for m in range(imt - 1):
                    w_stor[ibv] = w_stor[ibv] + q[m, k] * thm0[m] * dz[k]
                w_tot[ibv] = w_tot[ibv] + wr[k + 1, ibv]
        w_tot[1] = w_tot[1] + wr[0, 1]
        d = RHOW * (fb * (w_stor[0] - w_tot[0]) + fv * (w_stor[1] - w_tot[1]))
        out[i, j] = max(d, 0.0)
    return out


# ------------------------------------------------------------------------------------------------------ daily_LAKE
FIELDS_IN = ('flake', 'fearth', 'fland', 'rsi', 'msi', 'snowi', 'hsi', 'mwl', 'gml', 'tlake', 'mldlk')


def daily_lake(S, flice, focean, tanlk_, hlake, axyp, dmwldf, valid=None, lake_ice_max=LAKE_ICE_MAX, variable_lk=VARIABLE_LK):
    """daily_LAKE.  S: dict of (IM, JM) arrays flake fearth fland rsi msi snowi mwl gml tlake mldlk and hsi (IM, JM, 4) (copied; inputs untouched).
    flice, focean, tanlk_, hlake, axyp: statics (IM, JM).  dmwldf: (IM, JM) from `water_deficit` (copied: the Fortran zeroes DMWLDF/DGML in some cells).
    valid: (IM, JM) bool, the IMAXJ domain (default: all cells except i > 0 on the pole rows).
    Returns dict: the updated fields (names of FIELDS_IN), svflake (FLAKE before), dgml, dmwldf (as modified), dlake, glake, gtemp, gtempr, mlhc (the final loop; gtemp/mlhc are NaN where FLAKE <= 0 after the update = not set by the Fortran; gtempr = TF in a removed lake), mdwnimp, edwnimp (increments), counters, events (per changed cell), reset_surf_fluxes."""
    A = {k: np.array(S[k], dtype=np.float64, copy=True) for k in FIELDS_IN}
    flake, fearth, fland = A['flake'], A['fearth'], A['fland']
    rsi, msi, snowi, hsi = A['rsi'], A['msi'], A['snowi'], A['hsi']
    mwl, gml, tlake, mldlk = A['mwl'], A['gml'], A['tlake'], A['mldlk']
    dmw = np.array(dmwldf, dtype=np.float64, copy=True)
    dgml = np.zeros((IM, JM))
    mdwn = np.zeros((IM, JM))
    edwn = np.zeros((IM, JM))
    svflake = flake.copy()
    if valid is None:
        valid = np.ones((IM, JM), bool)
        valid[1:, 0] = False
        valid[1:, JM - 1] = False
    cnt = dict(cells_visited=0, new_flake_ne_flake=0, expand=0, shrink=0, have_lake=0, no_lake=0, crunch=0, layer_entrain=0, layer_new=0, layer_scale=0,
               dont_saturate=0, saturate_solve=0, ice_dump=0)
    events, rsf = [], []
    gtempr = np.full((IM, JM), np.nan)
    if variable_lk != 0:
        for j in range(JM):
            for i in range(IM):
                if not valid[i, j]:
                    continue
                if not ((flake[i, j] + fearth[i, j] > 0) and focean[i, j] == 0):
                    continue
                cnt['cells_visited'] += 1
                axy = axyp[i, j]
                tn = tanlk_[i, j]
                flake_old, fearth_old, rsi_old = flake[i, j], fearth[i, j], rsi[i, j]
                plake = flake[i, j] * (1. - rsi[i, j])
                plkic = flake[i, j] * rsi[i, j]
                if POWER_LAW_LAKES < 1:
                    a_swamp = PI * (MINMLD * tn * 3) ** 2
                else:
                    a_swamp = _pow(MINMLD / C_LAKE, 1 / (E_LAKE - 1))
                mw_swamp = a_swamp * MINMLD * RHOW
                mwtot = mwl[i, j] + plkic * (msi[i, j] + snowi[i, j] + ACE1I) * axy
                if mwtot <= mw_swamp and SMALL_LAKE_EVAP == 1:
                    alake = mwtot / (RHOW * MINMLD)
                else:
                    if POWER_LAW_LAKES < 1:
                        alake = _pow(9.0 * PI * (tn * mwtot / RHOW) ** 2, BY3)
                    else:
                        alake = _pow(mwtot / (RHOW * C_LAKE), 1 / E_LAKE)
                new_flake = alake / axy
                if new_flake < 1e-10:
                    dmw[i, j] = 0.0
                mwsat = 0.0
                if new_flake > flake[i, j] and dmw[i, j] > 0.0:
                    cnt['saturate_solve'] += 1
                    mwtot1 = mwtot + flake[i, j] * axy * dmw[i, j]
                    mw_swamp1 = mw_swamp + a_swamp * dmw[i, j]
                    if mwtot1 <= mw_swamp1 and SMALL_LAKE_EVAP == 1:
                        new_flake = mwtot1 / ((RHOW * MINMLD + dmw[i, j]) * axy)
                    else:
                        if POWER_LAW_LAKES > 0:
                            b = dmw[i, j] / (RHOW * C_LAKE)
                            c = mwtot1 / (RHOW * C_LAKE)
                            Aa, _ = newton(b, c, E_LAKE, 1e-12)
                            new_flake = Aa / axy
                        else:
                            a = math.sqrt(axy) ** 3 / math.sqrt(PI)
                            b = 3.0 * dmw[i, j] / RHOW * tn * axy
                            c = 0.0
                            d = - 3.0 * tn * mwtot1 / RHOW
                            x, n_roots = cubicroot(a, b, c, d)
                            if n_roots < 1:
                                raise RuntimeError('lakes: no solution')
                            y = max(x[:n_roots])
                            new_flake = y ** 2
                    new_flake = max(new_flake, flake[i, j])
                    mwsat = (new_flake - flake[i, j]) * axy * dmw[i, j]
                new_flake = min(new_flake, .95 * (flake[i, j] + fearth[i, j]))
                new_flake = min(new_flake, flake[i, j] + .049 * fearth[i, j])
                if new_flake < 1e-10:
                    new_flake = 0.0
                hlk = 0.0
                hlkic = 0.0
                if new_flake > 0:
                    hlk = (mwl[i, j] - mwsat) / (RHOW * new_flake * axy)
                    hlkic = (mwtot - mwsat) / (RHOW * new_flake * axy)
                if new_flake != flake[i, j]:
                    cnt['new_flake_ne_flake'] += 1
                    if lake_ice_max < 1e10:
                        have_lake = (new_flake > 0 and (hlk > 1. or (hlk > 0.5 and hlkic > 1.)))
                    else:
                        have_lake = (new_flake > 0.)
                    if have_lake:
                        cnt['have_lake'] += 1
                        frsat = 0.
                        if new_flake > flake[i, j]:
                            cnt['expand'] += 1
                            if mwl[i, j] > dmw[i, j] * (new_flake - flake[i, j]) * axy:
                                frsat = dmw[i, j] * (new_flake - flake[i, j]) * axy / mwl[i, j]
                                mwl[i, j] = mwl[i, j] * (1. - frsat)
                                dgml[i, j] = frsat * gml[i, j]
                                gml[i, j] = gml[i, j] * (1. - frsat)
                                mldlk[i, j] = mldlk[i, j] * (1. - frsat)
                            else:
                                cnt['dont_saturate'] += 1
                                dmw[i, j] = 0.0
                                dgml[i, j] = 0.0
                        # conserve lake ice
                        if rsi[i, j] * flake[i, j] > new_flake:
                            cnt['crunch'] += 1
                            frac = plkic / new_flake
                            snownew = snowi[i, j] * frac
                            msinew = (msi[i, j] + ACE1I) * frac - ACE1I
                            rsi[i, j] = 1.
                            fmsi3 = ACE1I * (frac - 1.0)
                            fmsi2 = fmsi3 * XSI[0]
                            fmsi4 = fmsi3 * XSI[3]
                            h1, h2, h3, h4 = hsi[i, j, 0], hsi[i, j, 1], hsi[i, j, 2], hsi[i, j, 3]
                            fhsi2 = fmsi2 * h1 / (XSI[0] * (ACE1I + snowi[i, j]))
                            if fmsi3 < frac * XSI[1] * (ACE1I + snowi[i, j]):
                                fhsi3 = fmsi3 * h2 / (XSI[1] * (ACE1I + snowi[i, j]))
                            else:
                                fhsi3 = h2 * frac + (fmsi3 - frac * XSI[1] * (ACE1I + snowi[i, j])) * h1 / (XSI[0] * (ACE1I + snowi[i, j]))
                            if fmsi4 < frac * XSI[2] * msi[i, j]:
                                fhsi4 = fmsi4 * h3 / (XSI[2] * msi[i, j])
                            else:
                                fhsi4 = h3 * frac + (fmsi4 - frac * XSI[2] * msi[i, j]) * fhsi3 / fmsi3
                            hsi[i, j, 0] = h1 * (ACE1I + snownew) / (ACE1I + snowi[i, j])
                            hsi[i, j, 1] = h2 * frac + fhsi2 - fhsi3
                            hsi[i, j, 2] = h3 * frac + fhsi3 - fhsi4
                            hsi[i, j, 3] = h4 * frac + fhsi4
                            msi[i, j] = msinew
                            snowi[i, j] = snownew
                        else:
                            rsi[i, j] = plkic / new_flake
                        # adjust layering if necessary
                        hlk = mwl[i, j] / (RHOW * new_flake * axy)
                        new_mld = min(max(MINMLD, hlk - hlake[i, j]), hlk)
                        if mldlk[i, j] * flake[i, j] < new_flake * new_mld:
                            if flake[i, j] == 0 or hlk <= new_mld:
                                cnt['layer_new'] += 1
                                mldlk[i, j] = new_mld
                            else:
                                cnt['layer_entrain'] += 1
                                f_entr = (new_flake * new_mld - mldlk[i, j] * flake[i, j]) / (new_flake * hlk - mldlk[i, j] * flake[i, j])
                                m1 = mldlk[i, j] * RHOW
                                m2 = max(mwl[i, j] / (flake[i, j] * axy) - m1, 0.0)
                                m1t1 = m1 * tlake[i, j]
                                m2t2 = gml[i, j] / (SHW * flake[i, j] * axy) - m1t1
                                new_tlake = (m1t1 + f_entr * m2t2) / (m1 + f_entr * m2)
                                tlake[i, j] = new_tlake
                                mldlk[i, j] = new_mld
                        else:
                            cnt['layer_scale'] += 1
                            mldlk[i, j] = mldlk[i, j] * flake[i, j] / new_flake
                        # adjust land surface fractions
                        flake[i, j] = new_flake
                        fland[i, j] = 1. - flake[i, j]
                        fearth[i, j] = fland[i, j] - flice[i, j]
                    else:
                        cnt['no_lake'] += 1
                        if flake[i, j] > 0:
                            imlt = ACE1I + msi[i, j] + snowi[i, j]
                            hmlt = hsi[i, j, 0] + hsi[i, j, 1] + hsi[i, j, 2] + hsi[i, j, 3]
                            if lake_ice_max < 1e10:
                                mdwn[i, j] = mdwn[i, j] + plkic * imlt * axy
                                edwn[i, j] = edwn[i, j] + plkic * hmlt * axy
                            else:
                                mwl[i, j] = mwl[i, j] + plkic * imlt * axy
                                gml[i, j] = gml[i, j] + plkic * hmlt * axy
                            rsi[i, j] = 0.
                            snowi[i, j] = 0.
                            hsi[i, j, 0] = -LHM * XSI[0] * ACE1I
                            hsi[i, j, 1] = -LHM * XSI[1] * ACE1I
                            hsi[i, j, 2] = -LHM * XSI[2] * AC2OIM
                            hsi[i, j, 3] = -LHM * XSI[3] * AC2OIM
                            msi[i, j] = AC2OIM
                            tlake[i, j] = gml[i, j] / (SHW * mwl[i, j] + TEENY)
                            gtempr[i, j] = TF                    # GTEMPR(I,J)=TF (the removed lake; the SCM branch is not compiled)
                            mldlk[i, j] = MINMLD
                            flake[i, j] = 0.
                            fland[i, j] = 1.
                            fearth[i, j] = fland[i, j] - flice[i, j]
                    events.append(dict(i=i, j=j, flake_old=float(flake_old), flake_new=float(flake[i, j]), have_lake=bool(have_lake)))
                # radiative flux accounting (RESET_SURF_FLUXES): listed only
                if flake[i, j] > flake_old:
                    rsf.append((i, j, 4, 1, float(flake_old), float(flake[i, j])))
                if flake_old > flake[i, j]:
                    if plake > 0:
                        rsf.append((i, j, 1, 4, float(fearth_old), float(fearth_old + plake - flake[i, j] * (1 - rsi[i, j]))))
                    if plkic > 0 and plkic != flake[i, j] * rsi[i, j]:
                        rsf.append((i, j, 2, 4, float(fearth_old + plake - flake[i, j] * (1 - rsi[i, j])), float(fearth[i, j])))
    # --------------------------------------------------------------------------- set GTEMP array for lakes (final loop)
    dlake = np.zeros((IM, JM))
    glake = np.zeros((IM, JM))
    gtemp = np.full((IM, JM), np.nan)
    mlhc = np.full((IM, JM), np.nan)
    for j in range(JM):
        for i in range(IM):
            if not valid[i, j]:
                continue
            if flake[i, j] > 0:
                axy = axyp[i, j]
                dlake[i, j] = mwl[i, j] / (RHOW * flake[i, j] * axy)
                glake[i, j] = gml[i, j] / (flake[i, j] * axy)
                gtemp[i, j] = tlake[i, j]
                gtempr[i, j] = tlake[i, j] + TF
                mlhc[i, j] = SHW * mldlk[i, j] * RHOW
                if msi[i, j] > lake_ice_max * RHOW:
                    cnt['ice_dump'] += 1
                    imlt = msi[i, j] - lake_ice_max * RHOW
                    fraci = imlt / msi[i, j]
                    hmlt = (hsi[i, j, 2] + hsi[i, j, 3]) * fraci
                    plkic = flake[i, j] * rsi[i, j]
                    mdwn[i, j] = mdwn[i, j] + plkic * imlt * axy
                    edwn[i, j] = edwn[i, j] + plkic * hmlt * axy
                    msi[i, j] = (1 - fraci) * msi[i, j]
                    hsi[i, j, 2] = (1 - fraci) * hsi[i, j, 2]
                    hsi[i, j, 3] = (1 - fraci) * hsi[i, j, 3]
    out = dict(A)
    out.update(svflake=svflake, dgml=dgml, dmwldf=dmw, dlake=dlake, glake=glake, gtemp=gtemp, gtempr=gtempr, mlhc=mlhc, mdwnimp=mdwn, edwnimp=edwn,
               counters=cnt, events=events, reset_surf_fluxes=rsf, pow=POW)
    return out
