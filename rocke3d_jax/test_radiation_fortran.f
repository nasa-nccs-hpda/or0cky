C Minimal Fortran test for RADIATION module
      PROGRAM TEST_RADIATION_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8, DIMENSION(N) :: ts, albedo, cosz, emis, solar_in
      REAL*8, DIMENSION(N) :: sw_up, sw_down, lw_up, lw_down, net_rad
      INTEGER :: i
      REAL*8, PARAMETER :: SB = 5.67E-8

C Initialize inputs with fixed values
      ts(1) = 290.0D0
      ts(2) = 295.0D0
      ts(3) = 300.0D0
      ts(4) = 305.0D0
      ts(5) = 310.0D0

      albedo(1) = 0.1D0
      albedo(2) = 0.2D0
      albedo(3) = 0.3D0
      albedo(4) = 0.4D0
      albedo(5) = 0.5D0

      cosz(1) = 0.8D0
      cosz(2) = 0.85D0
      cosz(3) = 0.9D0
      cosz(4) = 0.95D0
      cosz(5) = 1.0D0

      emis(1) = 0.9D0
      emis(2) = 0.92D0
      emis(3) = 0.94D0
      emis(4) = 0.96D0
      emis(5) = 0.98D0

      solar_in(1) = 1000.0D0
      solar_in(2) = 1100.0D0
      solar_in(3) = 1200.0D0
      solar_in(4) = 1300.0D0
      solar_in(5) = 1400.0D0

C Compute shortwave fluxes
      DO i = 1, N
        sw_down(i) = solar_in(i) * cosz(i)
        sw_up(i) = sw_down(i) * albedo(i)
      END DO

C Compute longwave fluxes
      DO i = 1, N
        lw_up(i) = emis(i) * SB * ts(i)**4
        lw_down(i) = emis(i) * SB * ts(i)**4
      END DO

C Compute net radiation
      DO i = 1, N
        net_rad(i) = sw_down(i) - sw_up(i) + lw_down(i) - lw_up(i)
      END DO

C Save outputs to file
      OPEN(UNIT=10, FILE='test_radiation_fortran_output.txt')
      WRITE(10, *) 'sw_down:'
      DO i = 1, N
        WRITE(10, *) sw_down(i)
      END DO
      WRITE(10, *) 'sw_up:'
      DO i = 1, N
        WRITE(10, *) sw_up(i)
      END DO
      WRITE(10, *) 'lw_down:'
      DO i = 1, N
        WRITE(10, *) lw_down(i)
      END DO
      WRITE(10, *) 'lw_up:'
      DO i = 1, N
        WRITE(10, *) lw_up(i)
      END DO
      WRITE(10, *) 'net_rad:'
      DO i = 1, N
        WRITE(10, *) net_rad(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_radiation_fortran_output.txt'
      END
