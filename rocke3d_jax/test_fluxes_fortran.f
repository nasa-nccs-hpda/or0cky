C Minimal Fortran test for FLUXES module
      PROGRAM TEST_FLUXES_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8, DIMENSION(N) :: us, vs, tsv, qsrf, rho
      REAL*8, DIMENSION(N) :: cdm, cdh, cq, u1, v1, t1, q1
      REAL*8, DIMENSION(N) :: uflux, vflux, tflux, qflux
      INTEGER :: i
      REAL*8 :: wind_speed
      REAL*8, PARAMETER :: CP = 1004.6D0, LH = 2.501D6

C Initialize inputs with fixed values
      us(1) = 5.0D0
      us(2) = 7.0D0
      us(3) = 9.0D0
      us(4) = 11.0D0
      us(5) = 13.0D0

      vs(1) = 3.0D0
      vs(2) = 4.0D0
      vs(3) = 5.0D0
      vs(4) = 6.0D0
      vs(5) = 7.0D0

      tsv(1) = 290.0D0
      tsv(2) = 295.0D0
      tsv(3) = 300.0D0
      tsv(4) = 305.0D0
      tsv(5) = 310.0D0

      qsrf(1) = 0.01D0
      qsrf(2) = 0.012D0
      qsrf(3) = 0.014D0
      qsrf(4) = 0.016D0
      qsrf(5) = 0.018D0

      rho(1) = 1.0D0
      rho(2) = 1.1D0
      rho(3) = 1.2D0
      rho(4) = 1.3D0
      rho(5) = 1.4D0

      cdm(1) = 0.001D0
      cdm(2) = 0.0012D0
      cdm(3) = 0.0014D0
      cdm(4) = 0.0016D0
      cdm(5) = 0.0018D0

      cdh(1) = 0.0008D0
      cdh(2) = 0.0009D0
      cdh(3) = 0.001D0
      cdh(4) = 0.0011D0
      cdh(5) = 0.0012D0

      cq(1) = 0.0007D0
      cq(2) = 0.0008D0
      cq(3) = 0.0009D0
      cq(4) = 0.001D0
      cq(5) = 0.0011D0

      u1(1) = 6.0D0
      u1(2) = 8.0D0
      u1(3) = 10.0D0
      u1(4) = 12.0D0
      u1(5) = 14.0D0

      v1(1) = 4.0D0
      v1(2) = 5.0D0
      v1(3) = 6.0D0
      v1(4) = 7.0D0
      v1(5) = 8.0D0

      t1(1) = 285.0D0
      t1(2) = 290.0D0
      t1(3) = 295.0D0
      t1(4) = 300.0D0
      t1(5) = 305.0D0

      q1(1) = 0.008D0
      q1(2) = 0.01D0
      q1(3) = 0.012D0
      q1(4) = 0.014D0
      q1(5) = 0.016D0

C Compute wind speed and momentum flux (matching JAX logic)
      DO i = 1, N
        wind_speed = SQRT(us(i) * us(i) + vs(i) * vs(i))
        uflux(i) = rho(i) * cdm(i) * wind_speed * (us(i) - u1(i))
        vflux(i) = rho(i) * cdm(i) * wind_speed * (vs(i) - v1(i))
        tflux(i) = rho(i) * cdh(i) * wind_speed * CP * (tsv(i) - t1(i))
        qflux(i) = rho(i) * cq(i) * wind_speed * LH * (qsrf(i) - q1(i))
      END DO

C Save outputs to file
      OPEN(UNIT=10, FILE='test_fluxes_fortran_output.txt')
      WRITE(10, *) 'uflux:'
      DO i = 1, N
        WRITE(10, *) uflux(i)
      END DO
      WRITE(10, *) 'vflux:'
      DO i = 1, N
        WRITE(10, *) vflux(i)
      END DO
      WRITE(10, *) 'tflux:'
      DO i = 1, N
        WRITE(10, *) tflux(i)
      END DO
      WRITE(10, *) 'qflux:'
      DO i = 1, N
        WRITE(10, *) qflux(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_fluxes_fortran_output.txt'
      END
