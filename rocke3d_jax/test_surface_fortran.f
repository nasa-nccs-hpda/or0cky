C Minimal Fortran test for SURFACE module (compute_surface_properties)
      PROGRAM TEST_SURFACE_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8, DIMENSION(N) :: t1, q1, ps
      REAL*8, DIMENSION(N) :: thv1, rho, qsat
      INTEGER :: i
      REAL*8, PARAMETER :: RGAS = 287.0D0
      REAL*8, PARAMETER :: DELTX = 0.608D0
      REAL*8, PARAMETER :: TF = 273.15D0
      REAL*8 :: es

C Initialize inputs with fixed values (matching JAX test)
      t1(1) = 295.0D0
      t1(2) = 285.0D0
      t1(3) = 275.0D0
      t1(4) = 290.0D0
      t1(5) = 300.0D0

      q1(1) = 0.008D0
      q1(2) = 0.01D0
      q1(3) = 0.012D0
      q1(4) = 0.014D0
      q1(5) = 0.016D0

      ps(1) = 1.0D5
      ps(2) = 1.01D5
      ps(3) = 1.02D5
      ps(4) = 1.03D5
      ps(5) = 1.04D5

C Compute surface properties (matching JAX logic)
      DO i = 1, N
        thv1(i) = t1(i) * (1.0D0 + q1(i) * DELTX)
        rho(i) = ps(i) / (RGAS * thv1(i))
        es = 611.2D0 * EXP(17.67D0 * (t1(i) - TF) / (t1(i) - 29.65D0))
        qsat(i) = 0.622D0 * es / (ps(i) - es)
      END DO

C Save outputs to file
      OPEN(UNIT=10, FILE='test_surface_fortran_output.txt')
      WRITE(10, *) 'thv1:'
      DO i = 1, N
        WRITE(10, *) thv1(i)
      END DO
      WRITE(10, *) 'rho:'
      DO i = 1, N
        WRITE(10, *) rho(i)
      END DO
      WRITE(10, *) 'qsat:'
      DO i = 1, N
        WRITE(10, *) qsat(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_surface_fortran_output.txt'
      END
