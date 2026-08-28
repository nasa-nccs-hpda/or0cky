C Minimal Fortran test for SEAICE module (compute_sea_ice_temperature)
      PROGRAM TEST_SEAICE_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8, DIMENSION(N) :: ssil, hsil
      REAL*8, DIMENSION(N) :: tsil
      INTEGER :: i
      REAL*8, PARAMETER :: MU = 0.054D0

C Initialize inputs with fixed values
      ssil(1) = 0.0032D0
      ssil(2) = 0.0034D0
      ssil(3) = 0.0036D0
      ssil(4) = 0.0038D0
      ssil(5) = 0.0040D0

      hsil(1) = 0.5D0
      hsil(2) = 1.0D0
      hsil(3) = 1.5D0
      hsil(4) = 2.0D0
      hsil(5) = 2.5D0

C Compute sea ice temperature (matching JAX logic: tsil = -MU * ssil * 1000.0)
      DO i = 1, N
        tsil(i) = -MU * ssil(i) * 1000.0D0
      END DO

C Save outputs to file
      OPEN(UNIT=10, FILE='test_seaice_fortran_output.txt')
      WRITE(10, *) 'tsil:'
      DO i = 1, N
        WRITE(10, *) tsil(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_seaice_fortran_output.txt'
      END
