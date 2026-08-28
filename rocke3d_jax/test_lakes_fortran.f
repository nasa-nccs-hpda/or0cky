C Minimal Fortran test for LAKES module (compute_lake_temperature)
      PROGRAM TEST_LAKES_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8, DIMENSION(N) :: gml, mwl
      REAL*8, DIMENSION(N) :: tlake
      INTEGER :: i
      REAL*8, PARAMETER :: SHW = 4185.0D0, TMAXRHO = 4.0D0

C Initialize inputs with fixed values
      gml(1) = 1.0e7
      gml(2) = 2.0e7
      gml(3) = 3.0e7
      gml(4) = 4.0e7
      gml(5) = 5.0e7

      mwl(1) = 1000.0D0
      mwl(2) = 2000.0D0
      mwl(3) = 3000.0D0
      mwl(4) = 4000.0D0
      mwl(5) = 5000.0D0

C Compute lake temperature (matching JAX logic: tlake = gml / (mwl * shw))
      DO i = 1, N
        IF (mwl(i) > 0.0D0) THEN
          tlake(i) = gml(i) / (mwl(i) * SHW)
        ELSE
          tlake(i) = TMAXRHO
        END IF
      END DO

C Save outputs to file
      OPEN(UNIT=10, FILE='test_lakes_fortran_output.txt')
      WRITE(10, *) 'tlake:'
      DO i = 1, N
        WRITE(10, *) tlake(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_lakes_fortran_output.txt'
      END
