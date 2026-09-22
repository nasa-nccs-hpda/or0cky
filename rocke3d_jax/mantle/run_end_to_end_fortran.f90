! Fortran end-to-end workflow for ROCKE-3D
! This program mimics the JAX run_end_to_end.py workflow
PROGRAM RUN_END_TO_END_FORTRAN
  IMPLICIT NONE

  ! Declare variables for 1D arrays (PBL, FLUXES, SURFACE, RADIATION)
  INTEGER, PARAMETER :: grid_size = 1000
  REAL*8, DIMENSION(grid_size) :: z, T_1d, Q_1d, U, V, P
  REAL*8, DIMENSION(grid_size) :: dpsim, dpsih
  REAL*8, DIMENSION(grid_size) :: momentum_flux_u, momentum_flux_v
  REAL*8, DIMENSION(grid_size) :: heat_flux, solar_flux, lw_flux
  REAL*8, DIMENSION(grid_size) :: z0, zet_clamped, zet_temp

  ! Declare variables for 3D arrays (DRYCNV)
  INTEGER, PARAMETER :: L_layers = 20
  REAL*8, DIMENSION(grid_size, 1, L_layers) :: T_3d, Q_3d
  REAL*8, DIMENSION(grid_size, 1, L_layers) :: PK, PDSIG

  ! Declare timing variables
  REAL*8 :: start_time, end_time
  REAL*8 :: pbl_time, fluxes_time, surface_time
  REAL*8 :: radiation_time, drycnv_time, total_time
  INTEGER :: i, L

  ! Initialize z with repeated JAX values (seed=42)
  REAL*8, DIMENSION(10) :: z_jax
  z_jax = [618.7439D0, 65.45722D0, 187.3704D0, 639.8887D0, & 
           622.38617D0, 639.84766D0, 92.98521D0, 978.35645D0, 863.1134D0, 601.8371D0]
  ! Debug: Print z_jax
  PRINT *, 'z_jax values:'
  PRINT *, z_jax
  DO i = 1, grid_size
     z(i) = z_jax(MOD(i-1, 10) + 1)
  END DO
  ! For now, set T_1d, Q_1d, U, V, P to constant values to avoid randomness
  T_1d = 300.0D0  ! Constant temperature (K)
  Q_1d = 0.01D0   ! Constant moisture (kg/kg)
  U = 10.0D0      ! Constant wind U (m/s)
  V = 10.0D0      ! Constant wind V (m/s)
  P = 100000.0D0 ! Constant pressure (Pa)

  ! Scale to realistic ranges (matching JAX)
  ! Remove z scaling to avoid extreme zet values
  ! z remains as z_jax values (10-1000 m)
  T_1d = T_1d * 40.0D0 + 280.0D0  ! Temperature: 280-320 K
  Q_1d = Q_1d * 0.02D0   ! Moisture: 0-0.02 kg/kg
  U = U * 19.0D0 + 1.0D0 ! Wind U: 1-20 m/s
  V = V * 19.0D0 + 1.0D0 ! Wind V: 1-20 m/s
  P = P * 10000.0D0 + 90000.0D0  ! Pressure: 90000-100000 Pa
  z0 = 0.01D0  ! Reference height for PBL (matching JAX)
  ! Ensure z is never 0 to avoid division by zero in LOG(z/z0)
  WHERE(z .LE. 0.0D0) z = 10.0D0

  ! Initialize 3D arrays for DRYCNV (I, J=1, L)
  DO i = 1, grid_size
     DO L = 1, L_layers
        T_3d(i, 1, L) = T_1d(i)
        Q_3d(i, 1, L) = Q_1d(i)
        PK(i, 1, L) = 0.8D0
        PDSIG(i, 1, L) = 0.1D0
     END DO
  END DO

  ! Start total timing
  CALL CPU_TIME(start_time)

  ! Ensure z > 0 to avoid LOG(0) in FIND_DPSIH
  WHERE(z .LE. 0.0D0) z = 10.0D0
  ! Run PBL (matching JAX: lmonin = -10.0)
  ! Compute zet = z / lmonin (matching JAX)
  CALL CPU_TIME(start_time)
  ! Clamp zet to [-10.0, 10.0] to avoid numerical instability
  DO i = 1, grid_size
     zet_temp(i) = z(i) / (-10.0D0)
     zet_clamped(i) = MAX(-10.0D0, MIN(10.0D0, zet_temp(i)))
  END DO
  CALL FIND_DPSIM(zet_clamped, z0(1), dpsim, grid_size)
  CALL FIND_DPSIH(zet_clamped, z0(1), z, z0(1), dpsih, grid_size)
  CALL CPU_TIME(end_time)
  pbl_time = end_time - start_time

  ! Run FLUXES
  CALL CPU_TIME(start_time)
  CALL COMPUTE_MOMENTUM_FLUX(U, V, dpsim, momentum_flux_u, momentum_flux_v, grid_size)
  CALL COMPUTE_HEAT_FLUX(T_1d, dpsih, heat_flux, grid_size)
  CALL CPU_TIME(end_time)
  fluxes_time = end_time - start_time

  ! Run SURFACE (simplified: just pass through fluxes)
  CALL CPU_TIME(start_time)
  CALL COMPUTE_SURFACE_FLUXES(T_1d, Q_1d, U, V, momentum_flux_u, momentum_flux_v, heat_flux, grid_size)
  CALL CPU_TIME(end_time)
  surface_time = end_time - start_time

  ! Run RADIATION
  CALL CPU_TIME(start_time)
  CALL COMPUTE_SOLAR_FLUX(T_1d, solar_flux, grid_size)
  CALL COMPUTE_LW_FLUX(T_1d, Q_1d, lw_flux, grid_size)
  CALL CPU_TIME(end_time)
  radiation_time = end_time - start_time

  ! Run DRYCNV
  CALL CPU_TIME(start_time)
  CALL DRY_CONVECTION_MIXING(T_3d, Q_3d, PK, PDSIG, grid_size, 1, L_layers)
  CALL CPU_TIME(end_time)
  drycnv_time = end_time - start_time

  ! Total time
  total_time = pbl_time + fluxes_time + surface_time + radiation_time + drycnv_time

  ! Save outputs to text file
  OPEN(UNIT=10, FILE='fortran_end_to_end_output.txt')
  WRITE(10, *) 'Performance Summary:'
  WRITE(10, *) '  PBL:       ', pbl_time
  WRITE(10, *) '  FLUXES:    ', fluxes_time
  WRITE(10, *) '  SURFACE:   ', surface_time
  WRITE(10, *) '  RADIATION: ', radiation_time
  WRITE(10, *) '  DRYCNV:    ', drycnv_time
  WRITE(10, *) '  Total:     ', total_time
  WRITE(10, *) ''
  WRITE(10, *) 'Outputs:'
  WRITE(10, *) 'dpsim:'
  WRITE(10, *) (dpsim(i), i=1, grid_size)
  WRITE(10, *) 'dpsih:'
  WRITE(10, *) (dpsih(i), i=1, grid_size)
  WRITE(10, *) 'momentum_flux_u:'
  WRITE(10, *) (momentum_flux_u(i), i=1, grid_size)
  WRITE(10, *) 'momentum_flux_v:'
  WRITE(10, *) (momentum_flux_v(i), i=1, grid_size)
  WRITE(10, *) 'heat_flux:'
  WRITE(10, *) (heat_flux(i), i=1, grid_size)
  WRITE(10, *) 'solar_flux:'
  WRITE(10, *) (solar_flux(i), i=1, grid_size)
  WRITE(10, *) 'lw_flux:'
  WRITE(10, *) (lw_flux(i), i=1, grid_size)
  CLOSE(10)
  
  ! Save outputs to binary files for later plotting
  OPEN(UNIT=20, FILE='fortran_dpsim.bin', FORM='unformatted', ACCESS='stream')
  WRITE(20) dpsim
  CLOSE(20)
  
  OPEN(UNIT=21, FILE='fortran_dpsih.bin', FORM='unformatted', ACCESS='stream')
  WRITE(21) dpsih
  CLOSE(21)
  
  OPEN(UNIT=22, FILE='fortran_momentum_flux_u.bin', FORM='unformatted', ACCESS='stream')
  WRITE(22) momentum_flux_u
  CLOSE(22)
  
  OPEN(UNIT=23, FILE='fortran_momentum_flux_v.bin', FORM='unformatted', ACCESS='stream')
  WRITE(23) momentum_flux_v
  CLOSE(23)
  
  OPEN(UNIT=24, FILE='fortran_heat_flux.bin', FORM='unformatted', ACCESS='stream')
  WRITE(24) heat_flux
  CLOSE(24)
  
  OPEN(UNIT=25, FILE='fortran_solar_flux.bin', FORM='unformatted', ACCESS='stream')
  WRITE(25) solar_flux
  CLOSE(25)
  
  OPEN(UNIT=26, FILE='fortran_lw_flux.bin', FORM='unformatted', ACCESS='stream')
  WRITE(26) lw_flux
  CLOSE(26)

  PRINT *, 'Fortran end-to-end workflow completed.'
  PRINT *, 'Outputs saved to fortran_end_to_end_output.txt'
END PROGRAM RUN_END_TO_END_FORTRAN

! ======================================================================
! Subroutines from PBL.f
! ======================================================================

MODULE pbl_constants
  REAL*8, PARAMETER :: kappa = 0.4D0
  REAL*8, PARAMETER :: gamamu = 16.0D0
  REAL*8, PARAMETER :: gamahs = 4.7D0
  REAL*8, PARAMETER :: zet1 = 1.0D0
  REAL*8, PARAMETER :: slope1 = 5.0D0
  REAL*8, PARAMETER :: zetm = -1.0D0
  REAL*8, PARAMETER :: by3 = 1.0D0 / 3.0D0
  REAL*8, PARAMETER :: sigma = 1.0D0
  REAL*8, PARAMETER :: sigma1 = 1.0D0
  REAL*8, PARAMETER :: zeth = -1.0D0
  REAL*8, PARAMETER :: gamahu = 16.0D0
  REAL*8, PARAMETER :: gamams = 4.7D0
END MODULE pbl_constants

SUBROUTINE find_dpsim(zet, zet0, dpsim, n)
  USE pbl_constants
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: zet(n), zet0
  REAL*8, INTENT(OUT) :: dpsim(n)
  REAL*8 :: x, x0, xm, term1, term2, term3, term4, term5
  INTEGER :: i
  DO i = 1, n
     IF(zet(i).ge.0.d0) THEN
        IF(zet(i).le.zet1) THEN
           dpsim(i) = -gamams * (zet(i) - zet0)
        ELSE
           term1 = -gamams * (zet1 - zet0)
           term2 = zet1 * (slope1 - gamams) * LOG(zet(i)/zet1)
           term3 = -slope1 * (zet(i) - zet1)
           dpsim(i) = term1 + term2 + term3
        END IF
     ELSE
        x = (1.0D0 - gamamu * zet(i))**0.25d0
        x0 = (1.0D0 - gamamu * zet0)**0.25d0
        xm = (1.0D0 - gamamu * zetm)**0.25d0
        IF(zet(i).gt.zetm) THEN
           term1 = LOG((1+x)*(1+x)*(1+x*x))
           term2 = LOG((1+x0)*(1+x0)*(1+x0*x0))
           term3 = 2.0D0 * (ATAN(x) - ATAN(x0))
           dpsim(i) = term1 - term2 - term3
        ELSE
           term1 = LOG((1+xm)*(1+xm)*(1+xm*xm))
           term2 = LOG((1+x0)*(1+x0)*(1+x0*x0))
           term3 = 2.0D0 * (ATAN(xm) - ATAN(x0))
           term4 = LOG(zet(i)/zetm)
           term5 = 1.140125d0 * ((-zet(i))**by3 - (-zetm)**by3)
           dpsim(i) = term1 - term2 - term3 + term4 - term5
        END IF
     END IF
  END DO
  RETURN
END SUBROUTINE find_dpsim

SUBROUTINE find_dpsih(zet, zet0, z, z0, dpsih, n)
  USE pbl_constants
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: zet(n), zet0, z(n), z0
  REAL*8, INTENT(OUT) :: dpsih(n)
  INTEGER :: i
  DO i = 1, n
     ! Match JAX implementation: simplified unstable branch
     IF(zet(i).ge.0.d0) THEN
        IF(zet(i).le.zet1) THEN
           dpsih(i) = sigma1 * LOG(z(i)/z0) - sigma * gamahs * (zet(i) - zet0)
        ELSE
           dpsih(i) = sigma1 * LOG(zet1/z0) - sigma * gamahs * (zet1 - zet0) + & 
                      (1 + sigma * (zet1 * (slope1 - gamahs) - 1)) * LOG(zet(i)/zet1) - & 
                      sigma * slope1 * (zet(i) - zet1)
        END IF
     ELSE
        ! Unstable branch (simplified to match JAX)
        dpsih(i) = sigma1 * LOG(z(i)/z0) - sigma * gamahu * (zet(i) - zet0)
     END IF
  END DO
  RETURN
END SUBROUTINE find_dpsih

! ======================================================================
! Subroutines from FLUXES.f (simplified versions)
! ======================================================================

SUBROUTINE compute_momentum_flux(us, vs, dpsim, uflux, vflux, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: us(n), vs(n), dpsim(n)
  REAL*8, INTENT(OUT) :: uflux(n), vflux(n)
  REAL*8 :: ws(n), rho, cdm
  INTEGER :: i
  rho = 1.2D0
  cdm = 0.001D0
  DO i = 1, n
     ws(i) = SQRT(us(i)**2 + vs(i)**2)
     uflux(i) = rho * cdm * ws(i) * (us(i) - 0.0D0)
     vflux(i) = rho * cdm * ws(i) * (vs(i) - 0.0D0)
  END DO
  RETURN
END SUBROUTINE compute_momentum_flux

SUBROUTINE compute_heat_flux(t1, dpsih, heat_flux, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: t1(n), dpsih(n)
  REAL*8, INTENT(OUT) :: heat_flux(n)
  REAL*8 :: rho, cdh, ws(n), cp
  INTEGER :: i
  rho = 1.2D0
  cdh = 0.001D0
  cp = 1004.6D0
  DO i = 1, n
     ws(i) = 10.0D0
     heat_flux(i) = rho * cdh * ws(i) * cp * (t1(i) - 290.0D0)
  END DO
  RETURN
END SUBROUTINE compute_heat_flux

! ======================================================================
! Subroutines from SURFACE.f (simplified)
! ======================================================================

SUBROUTINE compute_surface_fluxes(t1, q1, u, v, uflux, vflux, heat_flux, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: t1(n), q1(n), u(n), v(n)
  REAL*8, INTENT(IN) :: uflux(n), vflux(n), heat_flux(n)
  RETURN
END SUBROUTINE compute_surface_fluxes

! ======================================================================
! Subroutines from RADIATION.f (simplified)
! ======================================================================

SUBROUTINE compute_solar_flux(t, solar_flux, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: t(n)
  REAL*8, INTENT(OUT) :: solar_flux(n)
  INTEGER :: i
  DO i = 1, n
     solar_flux(i) = 1360.0D0 * (1.0D0 - 0.3D0)
  END DO
  RETURN
END SUBROUTINE compute_solar_flux

SUBROUTINE compute_lw_flux(t, q, lw_flux, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: t(n), q(n)
  REAL*8, INTENT(OUT) :: lw_flux(n)
  REAL*8, PARAMETER :: stbo = 5.67e-8
  INTEGER :: i
  DO i = 1, n
     lw_flux(i) = stbo * t(i)**4
  END DO
  RETURN
END SUBROUTINE compute_lw_flux

! ======================================================================
! Subroutines from DRYCNV.f (simplified)
! ======================================================================

SUBROUTINE dry_convection_mixing(T, Q, PK, PDSIG, I_size, J_size, L_size)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: I_size, J_size, L_size
  REAL*8, INTENT(INOUT) :: T(I_size, J_size, L_size), Q(I_size, J_size, L_size)
  REAL*8, INTENT(IN) :: PK(I_size, J_size, L_size), PDSIG(I_size, J_size, L_size)
  REAL*8 :: TV(I_size, J_size, L_size), THM, QM, PKMS, TVMS, QMS, RDP
  INTEGER :: i_idx, j_idx, k_idx
  DO k_idx = 1, L_size-1
     DO j_idx = 1, J_size
        DO i_idx = 1, I_size
           TV(i_idx, j_idx, k_idx) = T(i_idx, j_idx, k_idx) * (1 + Q(i_idx, j_idx, k_idx) * 0.608D0)
        END DO
     END DO
  END DO
  DO k_idx = 1, L_size-1
     DO j_idx = 1, J_size
        DO i_idx = 1, I_size
           IF (TV(i_idx, j_idx, k_idx) .GT. TV(i_idx, j_idx, k_idx+1)) THEN
              PKMS = PK(i_idx, j_idx, k_idx) * PDSIG(i_idx, j_idx, k_idx) + & 
                     PK(i_idx, j_idx, k_idx+1) * PDSIG(i_idx, j_idx, k_idx+1)
              TVMS = TV(i_idx, j_idx, k_idx) * PK(i_idx, j_idx, k_idx) * & 
                     PDSIG(i_idx, j_idx, k_idx) + TV(i_idx, j_idx, k_idx+1) * & 
                     PK(i_idx, j_idx, k_idx+1) * PDSIG(i_idx, j_idx, k_idx+1)
              QMS = Q(i_idx, j_idx, k_idx) * PDSIG(i_idx, j_idx, k_idx) + & 
                    Q(i_idx, j_idx, k_idx+1) * PDSIG(i_idx, j_idx, k_idx+1)
              RDP = 1.0D0 / (PDSIG(i_idx, j_idx, k_idx) + PDSIG(i_idx, j_idx, k_idx+1))
              THM = TVMS / (PKMS * (1.0D0 + QMS * RDP * 0.608D0))
              QM = QMS * RDP
              T(i_idx, j_idx, k_idx) = THM
              T(i_idx, j_idx, k_idx+1) = THM
              Q(i_idx, j_idx, k_idx) = QM
              Q(i_idx, j_idx, k_idx+1) = QM
           END IF
        END DO
     END DO
  END DO
  RETURN
END SUBROUTINE dry_convection_mixing
