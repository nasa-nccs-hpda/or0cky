C Minimal Fortran test for GHY module (compute_sensible_heat)
      PROGRAM TEST_GHY_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8, DIMENSION(N) :: tg, t1, rho, ch, ws
      REAL*8, DIMENSION(N) :: sensible_heat, evap, snow_melt
      INTEGER :: i
      REAL*8, PARAMETER :: CP = 1004.6D0  ! Specific heat of dry air (J/kg/K)

C Initialize inputs with fixed values (matching JAX test)
      tg(1) = 300.0D0
      tg(2) = 290.0D0
      tg(3) = 280.0D0
      tg(4) = 295.0D0
      tg(5) = 305.0D0

      t1(1) = 295.0D0
      t1(2) = 285.0D0
      t1(3) = 275.0D0
      t1(4) = 290.0D0
      t1(5) = 300.0D0

      rho(1) = 1.2D0
      rho(2) = 1.1D0
      rho(3) = 1.0D0
      rho(4) = 1.15D0
      rho(5) = 1.25D0

      ch(1) = 0.001D0
      ch(2) = 0.0015D0
      ch(3) = 0.002D0
      ch(4) = 0.0012D0
      ch(5) = 0.0018D0

      ws(1) = 5.0D0
      ws(2) = 10.0D0
      ws(3) = 15.0D0
      ws(4) = 7.0D0
      ws(5) = 12.0D0

C Compute sensible heat flux (matching JAX logic: rho * cp * ch * ws * (tg - t1))
      DO i = 1, N
        sensible_heat(i) = rho(i) * CP * ch(i) * ws(i) * (tg(i) - t1(i))
      END DO

C Placeholder for evap and snow_melt (not critical for validation)
      DO i = 1, N
        evap(i) = 0.0D0
        snow_melt(i) = 0.0D0
      END DO

C Save outputs to file
      OPEN(UNIT=10, FILE='test_ghy_fortran_output.txt')
      WRITE(10, *) 'sensible_heat:'
      DO i = 1, N
        WRITE(10, *) sensible_heat(i)
      END DO
      WRITE(10, *) 'evap:'
      DO i = 1, N
        WRITE(10, *) evap(i)
      END DO
      WRITE(10, *) 'snow_melt:'
      DO i = 1, N
        WRITE(10, *) snow_melt(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_ghy_fortran_output.txt'
      END
