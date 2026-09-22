! Fortran end-to-end workflow for ROCKE-3D
! This program mimics the JAX run_end_to_end.py workflow
PROGRAM RUN_END_TO_END_FORTRAN
IMPLICIT NONE

! Declare variables for 1D arrays (PBL, FLUXES, SURFACE, RADIATION)
INTEGER, PARAMETER :: grid_size = 1000
REAL*8, DIMENSION(grid_size) :: z, T_1d, Q_1d, U, V, P
REAL*8, DIMENSION(grid_size) :: dpsim, dpsih
REAL*8, DIMENSION(grid_size) :: momentum_flux_u, momentum_flux_v, heat_flux
REAL*8, DIMENSION(grid_size) :: solar_flux, lw_flux
REAL*8, DIMENSION(grid_size) :: z0

! Declare variables for 3D arrays (DRYCNV)
INTEGER, PARAMETER :: L_layers = 20
REAL*8, DIMENSION(grid_size, 1, L_layers) :: T_3d, Q_3d
REAL*8, DIMENSION(grid_size, 1, L_layers) :: PK, PDSIG

! Declare timing variables
REAL*8 :: start_time, end_time
REAL*8 :: pbl_time, fluxes_time, surface_time, radiation_time, drycnv_time, total_time
INTEGER :: i, L

! Initialize random seed
CALL RANDOM_SEED()

! Initialize atmospheric state (same as JAX)
CALL RANDOM_NUMBER(z)
CALL RANDOM_NUMBER(T_1d)
CALL RANDOM_NUMBER(Q_1d)
CALL RANDOM_NUMBER(U)
CALL RANDOM_NUMBER(V)
CALL RANDOM_NUMBER(P)

! Scale to realistic ranges
z = z * 1000.0D0       ! Height: 0-1000 m
T_1d = T_1d * 40.0D0 + 280.0D0  ! Temperature: 280-320 K
Q_1d = Q_1d * 0.02D0   ! Moisture: 0-0.02 kg/kg
U = U * 20.0D0        ! Wind U: 0-20 m/s
V = V * 20.0D0        ! Wind V: 0-20 m/s
P = P * 10000.0D0 + 90000.0D0  ! Pressure: 90000-100000 Pa
z0 = 0.0D0  ! Reference height for PBL

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

! Run PBL
CALL CPU_TIME(start_time)
CALL FIND_DPSIM(z, z0, dpsim)
CALL FIND_DPSIH(z, z0, z, z0, dpsih)
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
CALL DRY_CONVECTION_MIXING(T_3d, Q_3d, PK, PDSIG, grid_size, L_layers)
CALL CPU_TIME(end_time)
drycnv_time = end_time - start_time

! Total time
total_time = pbl_time + fluxes_time + surface_time + radiation_time + drycnv_time

! Save outputs
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
WRITE(10, *) 'T_3d (first layer):'
WRITE(10, *) (T_3d(i, 1, 1), i=1, grid_size)
WRITE(10, *) 'Q_3d (first layer):'
WRITE(10, *) (Q_3d(i, 1, 1), i=1, grid_size)
CLOSE(10)

PRINT *, 'Fortran end-to-end workflow completed.'
PRINT *, 'Outputs saved to fortran_end_to_end_output.txt'
END PROGRAM RUN_END_TO_END_FORTRAN

! ======================================================================
! Subroutines from PBL.f
! ======================================================================

! Constants from PBL.f
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

SUBROUTINE find_dpsim(zet, zet0, dpsim)
USE pbl_constants
IMPLICIT NONE
REAL*8, INTENT(IN) :: zet, zet0
REAL*8, INTENT(OUT) :: dpsim
REAL*8 :: x, x0, xm
IF(zet.ge.0.d0) THEN
    IF(zet.le.zet1) THEN
        dpsim = -gamams * (zet - zet0)
    ELSE
        dpsim = -gamams * (zet1 - zet0) + zet1 * (slope1 - gamams) * LOG(zet/zet1) - slope1 * (zet - zet1)
    END IF
ELSE
    x = (1.0D0 - gamamu * zet)**0.25d0
    x0 = (1.0D0 - gamamu * zet0)**0.25d0
    xm = (1.0D0 - gamamu * zetm)**0.25d0
    IF(zet.gt.zetm) THEN
        dpsim = LOG((1+x)*(1+x)*(1+x*x) / ((1+x0)*(1+x0)*(1+x0*x0))) - 2.0D0 * (ATAN(x) - ATAN(x0))
    ELSE
        dpsim = LOG((1+xm)*(1+xm)*(1+xm*xm) / ((1+x0)*(1+x0)*(1+x0*x0))) - 2.0D0 * (ATAN(xm) - ATAN(x0)) + LOG(zet/zetm) - 1.140125d0 * ((-zet)**by3 - (-zetm)**by3)
    END IF
END IF
RETURN
END SUBROUTINE find_dpsim

SUBROUTINE find_dpsih(zet, zet0, z, z0, dpsih)
USE pbl_constants
IMPLICIT NONE
REAL*8, INTENT(IN) :: zet, zet0, z, z0
REAL*8, INTENT(OUT) :: dpsih
REAL*8 :: x, x0, xh, rat
IF(zet.ge.0.d0) THEN
    IF(zet.le.zet1) THEN
        dpsih = sigma1 * LOG(z/z0) - sigma * gamahs * (zet - zet0)
    ELSE
        dpsih = sigma1 * LOG(zet1/z0) - sigma * gamahs * (zet1 - zet0) + (1 + sigma * (zet1 * (slope1 - gamahs) - 1)) * LOG(zet/zet1) - sigma * slope1 * (zet - zet1)
    END IF
ELSE
    x = (1.0D0 - gamahu * zet)**0.5d0
    x0 = (1.0D0 - gamahu * zet0)**0.5d0
    xh = (1.0D0 - gamahu * zeth)**0.5d0
    IF(zet.gt.zeth) THEN
        IF(-gamahu * zet.lt.1d-5) THEN
            rat = z0/z * (1.0D0 - 0.5d0 * gamahu * (zet - zet0))
        ELSE
            rat = (1+x) * (1-x0) / ((1-x) * (1+x0))
        END IF
        dpsih = LOG(z/z0) + sigma * LOG(rat)
    ELSE
        dpsih = LOG(z/z0) + sigma * LOG((1+xh) * (1-x0) / ((1-xh) * (1+x0))) - 0.7957508d0 * ((-zeth)**(-by3) - (-zet)**(-by3))
    END IF
END IF
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
REAL*8, INTENT(IN) :: t1(n), q1(n), u(n), v(n), uflux(n), vflux(n), heat_flux(n)
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
    solar_flux(i) = 1360.0D0 * (1.0D0 - 0.3D0)  ! Simplified solar flux
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
    lw_flux(i) = stbo * t(i)**4  ! Simplified LW flux
END DO
RETURN
END SUBROUTINE compute_lw_flux

! ======================================================================
! Subroutines from DRYCNV.f (simplified)
! ======================================================================

SUBROUTINE dry_convection_mixing(T, Q, PK, PDSIG, I, J, L)
IMPLICIT NONE
INTEGER, INTENT(IN) :: I, J, L
REAL*8, INTENT(INOUT) :: T(I,J,L), Q(I,J,L)
REAL*8, INTENT(IN) :: PK(I,J,L), PDSIG(I,J,L)
REAL*8 :: TV(I,J,L), THM, QM, PKMS, TVMS, QMS, RDP
INTEGER :: i, j, k
DO k = 1, L-1
    DO j = 1, J
        DO i = 1, I
            TV(i,j,k) = T(i,j,k) * (1 + Q(i,j,k) * 0.608D0)
        END DO
    END DO
END DO
DO k = 1, L-1
    DO j = 1, J
        DO i = 1, I
            IF (TV(i,j,k) .GT. TV(i,j,k+1)) THEN
                PKMS = PK(i,j,k) * PDSIG(i,j,k) + PK(i,j,k+1) * PDSIG(i,j,k+1)
                TVMS = TV(i,j,k) * PK(i,j,k) * PDSIG(i,j,k) + TV(i,j,k+1) * PK(i,j,k+1) * PDSIG(i,j,k+1)
                QMS = Q(i,j,k) * PDSIG(i,j,k) + Q(i,j,k+1) * PDSIG(i,j,k+1)
                RDP = 1.0D0 / (PDSIG(i,j,k) + PDSIG(i,j,k+1))
                THM = TVMS / (PKMS * (1 + QMS * RDP * 0.608D0))
                QM = QMS * RDP
                T(i,j,k) = THM
                T(i,j,k+1) = THM
                Q(i,j,k) = QM
                Q(i,j,k+1) = QM
            END IF
        END DO
    END DO
END DO
RETURN
END SUBROUTINE dry_convection_mixing
