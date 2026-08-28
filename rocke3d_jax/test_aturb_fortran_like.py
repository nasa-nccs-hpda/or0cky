"""
Fortran-Like Reference Implementation for ATURB.f
==================================================

This script provides a Python/NumPy implementation of the ATURB.f subroutine
to serve as a reference for validating the JAX implementation.

Usage:
    from test_aturb_fortran_like import atm_diffus_fortran_like
    u_out, v_out, t_out, q_out = atm_diffus_fortran_like(u, v, t, q, pmid, pedn, pk, tvsurf, uflux1, vflux1, tflux1, qflux1, qsavg, tsavg)
"""

import numpy as np

# Constants (from ROCKE-3D's CONSTANT module)
GRAV = 9.80665  # Gravitational acceleration (m/s^2)
DELTX = 0.608   # Virtual temperature factor
LHE = 2.501e6   # Latent heat of vaporization (J/kg)
SHA = 1004.6    # Specific heat of dry air (J/kg/K)
RGAS = 287.0    # Specific gas constant for dry air (J/kg/K)
KAPPA = 0.4     # Von Karman constant
ZGS = 10.0      # Surface layer height (m)
USTAR_MIN = 0.01  # Minimum friction velocity (m/s)
PRT = 0.95      # Prandtl number for turbulence
TEENY = 1e-12   # Tiny value to avoid division by zero

# Constants from SOCPBL module (used in k_gcm and e_gcm)
B1 = 19.0       # Turbulence constant
RIMAX = 10.0    # Maximum Richardson number
C1 = 1.0       # Stability function coefficients (placeholders)
C2 = 2.0
C3 = 3.0
C4 = 4.0
C5 = 5.0
EMAX = 1e6      # Maximum TKE


def getdz_fortran(tv, pmid, pedn, pk, tvsurf):
    """
    Fortran-like implementation of getdz.
    """
    I, J, L = tv.shape
    dz = np.zeros((I, J, L))
    dze = np.zeros((I, J, L))
    rho = np.zeros((I, J, L))
    rhoe = np.zeros((I, J, L))
    dz0 = np.zeros((I, J))
    
    for i in range(I):
        for j in range(J):
            for l in range(L - 1):
                temp0 = tv[i, j, l] * pk[i, j, l]
                temp1 = tv[i, j, l + 1] * pk[i, j, l + 1]
                temp1e = 0.5 * (temp0 + temp1)
                dz[i, j, l] = - (RGAS / GRAV) * temp1e * np.log(pmid[i, j, l + 1] / pmid[i, j, l])
                dze[i, j, l] = - (RGAS / GRAV) * temp0 * np.log(pedn[i, j, l + 1] / pedn[i, j, l])
                rhoe[i, j, l + 1] = 100.0 * (pmid[i, j, l + 1] - pmid[i, j, l]) / (GRAV * dz[i, j, l])
                rho[i, j, l] = 100.0 * (pedn[i, j, l + 1] - pedn[i, j, l]) / (GRAV * dze[i, j, l])
            
            # Handle the first layer (l=0)
            temp0_l1 = tv[i, j, 0] * pk[i, j, 0]
            dz0[i, j] = - (RGAS / GRAV) * 0.5 * (temp0_l1 + tvsurf[i, j]) * np.log(pmid[i, j, 0] / pedn[i, j, 0])
            rhoe[i, j, 0] = 100.0 * pedn[i, j, 0] / (tvsurf[i, j] * RGAS)
            
            # Handle the last layer (l=L-1)
            temp1_lL = tv[i, j, L - 1] * pk[i, j, L - 1]
            dze[i, j, L - 1] = - (RGAS / GRAV) * temp1_lL * np.log(pedn[i, j, L] / pedn[i, j, L - 1])
            dz[i, j, L - 1] = 0.0
            rho[i, j, L - 1] = 100.0 * (pedn[i, j, L - 1] - pedn[i, j, L]) / (GRAV * dze[i, j, L - 1])
    
    return dz, dze, rho, rhoe, dz0


def zze_fortran(dz, dze, dz0, lm):
    """
    Fortran-like implementation of zze.
    """
    z = np.cumsum(dz, axis=-1)
    ze = np.concatenate([np.expand_dims(dz0, axis=-1), z], axis=-1)
    return z, ze


def find_phim0_fortran(zet):
    """
    Fortran-like implementation of find_phim0.
    """
    # Clip zet to avoid invalid powers
    zet_clipped = np.clip(zet, -0.99, 10.0)
    phim = np.where(
        zet_clipped >= 0.0,  # Stable or neutral
        np.where(
            zet_clipped <= 0.5,
            1.0 + 5.3 * zet_clipped,
            1.0 + 5.3 * 0.5 + 0.1 * (zet_clipped - 0.5)
        ),
        (1.0 - 19.0 * zet_clipped) ** (-0.25)  # Unstable
    )
    return phim


def find_phih_fortran(zet):
    """
    Fortran-like implementation of find_phih.
    """
    # Clip zet to avoid invalid powers
    zet_clipped = np.clip(zet, -0.99, 10.0)
    phih = np.where(
        zet_clipped >= 0.0,  # Stable or neutral
        np.where(
            zet_clipped <= 0.5,
            0.95 * (1.0 + 8.0 * zet_clipped),
            0.95 * (1.0 + 8.0 * 0.5 + 0.1 * (zet_clipped - 0.5))
        ),
        np.where(
            zet_clipped >= -1.072,
            0.95 * (1.0 - 11.6 * zet_clipped) ** (-0.5),
            0.9 * KAPPA ** (4.0 / 3.0) * (-zet_clipped) ** (-1.0 / 3.0)
        )
    )
    return phih


def l_gcm_fortran(ze, dbl, lmonin, ustar, qturb, an2, lm):
    """
    Fortran-like implementation of l_gcm.
    """
    zet = ze[1:] / lmonin
    phim = find_phim0_fortran(zet)
    phih = find_phih_fortran(zet)
    
    lscale = np.where(
        (ze[1:] <= dbl) & (zet < 0.0),  # Within unstable PBL
        KAPPA * ze[1:] * (1.0 - ze[1:] / dbl),
        KAPPA * ze[1:]  # Stable or above PBL
    )
    
    return lscale


def k_gcm_fortran(tvflx, qflx, uflx, vflx, ustar, wstar, dbl, lmonin, ze, lscale, e, qturb, an2, as2, dtdz, dqdz, dudz, dvdz, lm):
    """
    Fortran-like implementation of k_gcm.
    """
    # Initialize outputs
    kh = np.zeros(lm)
    kq = np.zeros(lm)
    km = np.zeros(lm)
    ke = np.zeros(lm)
    wt = np.zeros(lm)
    wq = np.zeros(lm)
    w2 = np.zeros(lm)
    uw = np.zeros(lm)
    vw = np.zeros(lm)
    wt_nl = np.zeros(lm)
    wq_nl = np.zeros(lm)
    
    # Compute stability parameter (zet = z / L)
    zet_surf = 0.1 * dbl / lmonin
    if zet_surf < 0.0:
        phim_surf = (1.0 - 19.0 * zet_surf) ** (-0.25)
        phih_surf = 0.95 * (1.0 - 11.6 * zet_surf) ** (-0.5)
        by_phim_surf = 1.0 / phim_surf
        wm_surf = ustar * by_phim_surf
        pr_surf = phih_surf * by_phim_surf + 0.72 * KAPPA * wstar / wm_surf
        cgh_surf = 7.2 * wstar * (-tvflx) / (wm_surf ** 2 * dbl)
        cgu_surf = 0.0
        cgv_surf = 0.0
    else:
        phih_surf = 0.0
        by_phim_surf = 0.0
        wm_surf = 0.0
        pr_surf = 0.0
        cgh_surf = 0.0
        cgu_surf = 0.0
        cgv_surf = 0.0
    
    # Loop over layers
    for j in range(lm):
        zet_j = ze[j + 1] / lmonin
        
        # Compute turbulence parameters
        tau = B1 * lscale[j] / (qturb[j] + TEENY)
        gh = tau ** 2 * an2[j]
        gm = tau ** 2 * as2[j]
        
        # Clip stability parameters
        gh = np.clip(gh, -10.0, 10.0)
        
        # Use placeholder coefficients for sm and sh (since SOCPBL coefficients are not available)
        sm = 1.0
        sh = 1.0
        
        # Compute diffusivities
        km_j = np.clip(tau * e[j] * sm, 0.0, 1e6)
        kh_j = np.clip(tau * e[j] * sh, 0.0, 1e6)
        kq_j = kh_j
        
        # Initialize non-local fluxes
        wt_nl_j = 0.0
        wq_nl_j = 0.0
        uw_nl_j = 0.0
        vw_nl_j = 0.0
        
        # Check if within unstable PBL
        zzi = ze[j + 1] / dbl
        if (zzi <= 1.0) and (zet_j < 0.0):
            if zzi < 0.1:  # Surface layer
                km_j = KAPPA * ze[j + 1] * wm_surf * (1.0 - zzi) ** 2
                kh_j = km_j / pr_surf if abs(pr_surf) > TEENY else km_j
            else:  # Outer layer of PBL
                km_j = KAPPA * ze[j + 1] * wm_surf * (1.0 - zzi) ** 2
                kh_j = km_j / pr_surf if abs(pr_surf) > TEENY else km_j
                wt_nl_j = kh_j * cgh_surf
                uw_nl_j = km_j * cgu_surf
                vw_nl_j = km_j * cgv_surf
            
            # Compute TKE
            tmp = (1.6 * ustar ** 2 * (1.0 - zzi) + TEENY) ** 1.5 + 1.2 * wstar ** 3 * zzi * (1.0 - 0.9 * zzi) ** 1.5
            w2_j = tmp ** (2.0 / 3.0)
            km_j = np.clip(km_j, 0.0, 1e6)
            kh_j = np.clip(kh_j, 0.0, 1e6)
            kq_j = kh_j
        else:  # Above PBL
            w2_j = (2.0 / 3.0) * (2.0 * e[j] - tau * (0.0 * km_j * as2[j] + 0.0 * kh_j * an2[j]))
        
        # Compute fluxes
        ke_j = 5.0 * km_j
        wt_j = -kh_j * dtdz[j] + wt_nl_j
        wq_j = -kq_j * dqdz[j]
        w2_j = np.clip(w2_j, 0.24 * e[j], 2.0 * e[j])
        uw_j = -km_j * dudz[j] + uw_nl_j
        vw_j = -km_j * dvdz[j] + vw_nl_j
        
        # Store outputs
        kh[j] = kh_j
        kq[j] = kq_j
        km[j] = km_j
        ke[j] = ke_j
        wt[j] = wt_j
        wq[j] = wq_j
        w2[j] = w2_j
        uw[j] = uw_j
        vw[j] = vw_j
        wt_nl[j] = wt_nl_j
        wq_nl[j] = wq_nl_j
    
    return kh, kq, km, ke, wt, wq, w2, uw, vw, wt_nl, wq_nl


def tridiag_fortran(sub, dia, sup, rhs, n):
    """
    Fortran-like implementation of tridiag (Thomas algorithm).
    """
    # Forward sweep
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    c_prime[0] = sup[0] / dia[0] if abs(dia[0]) > TEENY else 0.0
    d_prime[0] = rhs[0] / dia[0] if abs(dia[0]) > TEENY else 0.0
    
    for j in range(1, n):
        denom = dia[j] - sub[j] * c_prime[j - 1]
        c_prime[j] = sup[j] / denom if abs(denom) > TEENY else 0.0
        d_prime[j] = (rhs[j] - sub[j] * d_prime[j - 1]) / denom if abs(denom) > TEENY else 0.0
    
    # Backward substitution
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]
    
    for j in range(n - 2, -1, -1):
        x[j] = d_prime[j] - c_prime[j] * x[j + 1]
    
    return x


def de_solver_main_fortran(x0, p1, p4, rhoebydz, bydzerho, flux_bot, flux_top, dtime, n, qlimit=False):
    """
    Fortran-like implementation of de_solver_main.
    """
    # Ensure rhoebydz and bydzerho have size n (pad if necessary)
    if rhoebydz.shape[0] == n - 1:
        rhoebydz_padded = np.concatenate([[0.0], rhoebydz])
    else:
        rhoebydz_padded = rhoebydz
    
    if bydzerho.shape[0] == n - 1:
        bydzerho_padded = np.concatenate([[0.0], bydzerho])
    else:
        bydzerho_padded = bydzerho
    
    # Initialize sub, dia, sup, rhs
    sub = np.zeros(n)
    dia = np.zeros(n)
    sup = np.zeros(n)
    rhs = np.zeros(n)
    
    # Fill sub, dia, sup, rhs for j=2 to n-1
    for j in range(1, n - 1):
        sub[j] = -dtime * p1[j] * rhoebydz_padded[j] * bydzerho_padded[j]
        sup[j] = -dtime * p1[j] * rhoebydz_padded[j] * bydzerho_padded[j]
        dia[j] = 1.0 - (sub[j] + sup[j])
        rhs[j] = x0[j] + dtime * p4[j]
        if qlimit and rhs[j] < 0:
            rhs[j] = 0.0
    
    # Lower boundary condition (j=1)
    alpha = dtime * p1[0] * rhoebydz_padded[0] * bydzerho_padded[0]
    dia[0] = 1.0 + alpha
    sup[0] = -alpha
    rhs[0] = x0[0] - dtime * bydzerho_padded[0] * flux_bot
    
    # Upper boundary condition (j=n)
    alpha = dtime * p1[n - 1] * rhoebydz_padded[n - 1] * bydzerho_padded[n - 1]
    sub[n - 1] = -alpha
    dia[n - 1] = 1.0 + alpha
    rhs[n - 1] = x0[n - 1] + dtime * bydzerho_padded[n - 1] * flux_top
    
    # Solve the tridiagonal system
    x = tridiag_fortran(sub, dia, sup, rhs, n)
    
    return x


def e_gcm_fortran(tvflx, wstar, ustar, dbl, lmonin, ze, g_alpha, an2, as2, lscale, e, lm):
    """
    Fortran-like implementation of e_gcm.
    """
    # Recompute lmonin (as in Fortran e_gcm)
    ustar3 = ustar ** 3
    wstar3 = wstar ** 3
    lmonin_out = ustar3 / (KAPPA * g_alpha[0] * tvflx)
    if abs(lmonin_out) < 1e-6:
        lmonin_out = np.sign(lmonin_out) * 1e-6
    if abs(lmonin_out) > 1e6:
        lmonin_out = np.sign(lmonin_out) * 1e6
    
    # Compute e for all layers
    e_out = np.zeros_like(e)
    for j in range(lm):
        zj = ze[j]
        if zj <= dbl:
            zeta = zj / lmonin_out
            if zeta >= 0.0:  # Stable or neutral
                if zeta <= 0.5:
                    phi_m = 1.0 + 5.0 * zeta
                else:
                    phi_m = 5.0 + zeta
            else:  # Unstable
                phi_m = (1.0 - 15.0 * zeta) ** (-0.25)
            tmp = 0.4 * wstar3 + ustar3 * (dbl - zj) * phi_m / (KAPPA * zj)
            e_out[j] = np.clip(tmp ** (2.0 / 3.0), TEENY, EMAX)
        else:
            ri = an2[j] / max(as2[j], TEENY)
            if ri < RIMAX:
                aa = C1 * ri * ri - C2 * ri + C3
                bb = C4 * ri + C5
                cc = 2.0
                if abs(aa) < 1e-8:
                    gm = -cc / bb
                else:
                    tmp = bb ** 2 - 4.0 * aa * cc
                    gm = (-bb - np.sqrt(tmp)) / (2.0 * aa)
                tmp = 0.5 * (B1 * lscale[j]) ** 2 * as2[j] / max(gm, TEENY)
                e_out[j] = np.clip(tmp, TEENY, EMAX)
            else:
                e_out[j] = TEENY
    
    return e_out


def atm_diffus_fortran_like(
    u, v, t, q, pmid, pedn, pk, tvsurf, uflux1, vflux1, tflux1, qflux1, qsavg, tsavg, dtime=0.0
):
    """
    Fortran-like implementation of atm_diffus (ATURB.f).
    """
    I, J, L = u.shape
    
    # Initialize outputs
    u_out = np.copy(u)
    v_out = np.copy(v)
    t_out = np.copy(t)
    q_out = np.copy(q)
    
    # Initialize turbulent kinetic energy (e)
    e = np.ones_like(t) * 0.1
    
    # Convert temperature to virtual temperature
    t_virtual = t_out * (1.0 + DELTX * q_out)
    
    # Compute dz, dze, rho, rhoe, and dz0
    dz, dze, rho, rhoe, dz0 = getdz_fortran(t_virtual, pmid, pedn, pk, tvsurf)
    
    # Compute height (z) and height at edges (ze)
    z, ze = zze_fortran(dz[0, 0, :], dze[0, 0, :], dz0[0, 0], L)
    
    # Compute surface virtual temperature and fluxes
    tvs = tvsurf[0, 0] / pk[0, 0, 0]
    uflx = uflux1[0, 0] / rhoe[0, 0, 0]
    vflx = vflux1[0, 0] / rhoe[0, 0, 0]
    qflx = qflux1[0, 0] / rhoe[0, 0, 0]
    tvflx = (tflux1[0, 0] * (1.0 + DELTX * qsavg[0, 0]) / (rhoe[0, 0, 0] * pk[0, 0, 0])) + (DELTX * tsavg[0, 0] / pk[0, 0, 0] * qflx)
    
    # Compute friction velocity (ustar)
    ustar = np.sqrt(np.sqrt(uflx**2 + vflx**2))
    ustar = max(ustar, USTAR_MIN)
    ustar2 = ustar**2
    
    # Compute Monin-Obukhov length (lmonin)
    tmp = 1.0 / (ustar * KAPPA * ZGS)
    dudz = uflx * tmp
    dvdz = vflx * tmp
    dtdz = tvflx * PRT * tmp
    dqdz = qflx * PRT * tmp
    g_alpha = GRAV / tvs
    den = KAPPA * g_alpha * tvflx
    if abs(den) < TEENY:
        den = TEENY
    lmonin = ustar**3 / den
    if abs(lmonin) < 1e-6:
        lmonin = np.sign(lmonin) * 1e-6
    if abs(lmonin) > 1e6:
        lmonin = np.sign(lmonin) * 1e6
    
    # Compute PBL depth and top layer
    dbl = 1000.0  # Placeholder
    ldbl = min(L - 1, 10)  # Placeholder
    ldbl_max = min(L - 1, 20)  # Placeholder
    
    # Compute convective velocity scale (wstar)
    wstar = (-g_alpha * tvflx * dbl) ** (1.0 / 3.0) if tvflx < 0.0 else TEENY
    
    # Compute z-derivatives for all layers
    dudz_full = np.zeros_like(u_out)
    dvdz_full = np.zeros_like(v_out)
    dtdz_full = np.zeros_like(t_out)
    dqdz_full = np.zeros_like(q_out)
    
    for l in range(1, L):
        dudz_full[:, :, l] = (u_out[:, :, l] - u_out[:, :, l - 1]) / dz[:, :, l - 1]
        dvdz_full[:, :, l] = (v_out[:, :, l] - v_out[:, :, l - 1]) / dz[:, :, l - 1]
        dtdz_full[:, :, l] = (t_out[:, :, l] - t_out[:, :, l - 1]) / dz[:, :, l - 1]
        dqdz_full[:, :, l] = (q_out[:, :, l] - q_out[:, :, l - 1]) / dz[:, :, l - 1]
    
    # Compute Brunt-Väisälä frequency squared (an2) and shear number squared (as2)
    g_alpha_full = GRAV / t_out
    an2_full = g_alpha_full * dtdz_full
    as2_full = np.maximum(dudz_full ** 2 + dvdz_full ** 2, TEENY)
    
    # Compute turbulence length scale (lscale)
    lscale = l_gcm_fortran(ze, dbl, lmonin, ustar, np.sqrt(2.0 * e[0, 0, :]), an2_full[0, 0, :], L)
    
    # Compute turbulent diffusivities and fluxes
    kh, kq, km, ke, wt, wq, w2, uw, vw, wt_nl, wq_nl = k_gcm_fortran(
        tvflx, qflx, uflx, vflx, ustar, wstar, dbl, lmonin, ze,
        lscale, e[0, 0, :], np.sqrt(2.0 * e[0, 0, :]),
        an2_full[0, 0, :], as2_full[0, 0, :],
        dtdz_full[0, 0, :], dqdz_full[0, 0, :],
        dudz_full[0, 0, :], dvdz_full[0, 0, :], L
    )
    
    # Compute rhoebydz and bydzerho for de_solver_main
    rhoebydz = rhoe[0, 0, 1:L] / dz[0, 0, :L-1]
    bydzerho = 1.0 / (dze[0, 0, :L] * rho[0, 0, :L])
    
    # Solve for temperature (t)
    p4_t = np.zeros(L)
    for l in range(1, L - 1):
        p4_t[l] = -(rhoe[0, 0, l + 1] * wt_nl[l + 1] - rhoe[0, 0, l] * wt_nl[l]) * bydzerho[l]
    flux_bot_t = rhoe[0, 0, 0] * tvflx + rhoe[0, 0, 1] * wt_nl[1]
    flux_top_t = rhoe[0, 0, L - 1] * wt_nl[L - 1]
    t_out[0, 0, :] = de_solver_main_fortran(
        t_out[0, 0, :], kh, p4_t, rhoebydz, bydzerho, flux_bot_t, flux_top_t, dtime, L, False
    )
    
    # Solve for moisture (q)
    p4_q = np.zeros(L)
    for l in range(1, L - 1):
        p4_q[l] = -(rhoe[0, 0, l + 1] * wq_nl[l + 1] - rhoe[0, 0, l] * wq_nl[l]) * bydzerho[l]
    flux_bot_q = rhoe[0, 0, 0] * qflx + rhoe[0, 0, 1] * wq_nl[1]
    flux_top_q = rhoe[0, 0, L - 1] * wq_nl[L - 1]
    q_out[0, 0, :] = de_solver_main_fortran(
        q_out[0, 0, :], kq, p4_q, rhoebydz, bydzerho, flux_bot_q, flux_top_q, dtime, L, True
    )
    
    # Update turbulent kinetic energy (e)
    e_out = e_gcm_fortran(tvflx, wstar, ustar, dbl, lmonin, ze, g_alpha_full[0, 0, :], 
                          an2_full[0, 0, :], as2_full[0, 0, :], lscale, e[0, 0, :], L)
    
    # Solve for u and v (velocity diffusion)
    rhoebydz_uv = rhoe[0, 0, 1:L] / dz[0, 0, :L-1]
    bydzerho_uv = 1.0 / (dze[0, 0, :L] * rho[0, 0, :L])
    
    # Solve for u
    p4_u = np.zeros(L)
    for l in range(1, L - 1):
        p4_u[l] = -(rhoe[0, 0, l + 1] * uw[l + 1] - rhoe[0, 0, l] * uw[l]) * bydzerho_uv[l]
    flux_bot_u = uflux1[0, 0] + rhoe[0, 0, 1] * uw[1]
    flux_top_u = 0.0
    u_out[0, 0, :] = de_solver_main_fortran(
        u_out[0, 0, :], km, p4_u, rhoebydz_uv, bydzerho_uv, flux_bot_u, flux_top_u, dtime, L, False
    )
    
    # Solve for v
    p4_v = np.zeros(L)
    for l in range(1, L - 1):
        p4_v[l] = -(rhoe[0, 0, l + 1] * vw[l + 1] - rhoe[0, 0, l] * vw[l]) * bydzerho_uv[l]
    flux_bot_v = vflux1[0, 0] + rhoe[0, 0, 1] * vw[1]
    flux_top_v = 0.0
    v_out[0, 0, :] = de_solver_main_fortran(
        v_out[0, 0, :], km, p4_v, rhoebydz_uv, bydzerho_uv, flux_bot_v, flux_top_v, dtime, L, False
    )
    
    return u_out, v_out, t_out, q_out


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Define test dimensions
    I, J, L = 2, 2, 5
    
    # Generate random inputs
    u = np.random.uniform(-10.0, 10.0, (I, J, L))
    v = np.random.uniform(-10.0, 10.0, (I, J, L))
    t = np.random.uniform(200.0, 300.0, (I, J, L))
    q = np.random.uniform(0.0, 0.02, (I, J, L))
    pmid = np.random.uniform(500.0, 1000.0, (I, J, L))
    pedn = np.random.uniform(500.0, 1000.0, (I, J, L + 1))
    pk = np.random.uniform(0.9, 1.1, (I, J, L))
    tvsurf = np.random.uniform(280.0, 300.0, (I, J))
    uflux1 = np.random.uniform(-0.1, 0.1, (I, J))
    vflux1 = np.random.uniform(-0.1, 0.1, (I, J))
    tflux1 = np.random.uniform(-0.1, 0.1, (I, J))
    qflux1 = np.random.uniform(-0.01, 0.01, (I, J))
    qsavg = np.random.uniform(0.0, 0.02, (I, J))
    tsavg = np.random.uniform(280.0, 300.0, (I, J))
    
    # Run Fortran-like implementation
    u_out, v_out, t_out, q_out = atm_diffus_fortran_like(
        u, v, t, q, pmid, pedn, pk, tvsurf, uflux1, vflux1, tflux1, qflux1, qsavg, tsavg
    )
    
    print("Fortran-like outputs:")
    print(f"U: {u_out[0, 0, :]}")
    print(f"V: {v_out[0, 0, :]}")
    print(f"T: {t_out[0, 0, :]}")
    print(f"Q: {q_out[0, 0, :]}")
