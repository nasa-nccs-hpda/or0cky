C Minimal Fortran test for ATURB module
      PROGRAM TEST_ATURB_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8 u(N), v(N), t(N), q(N), pk(N), pdsig(N)
      INTEGER i

C Initialize inputs with fixed values
      DATA u /5.0D0, 7.0D0, 9.0D0, 11.0D0, 13.0D0/
      DATA v /3.0D0, 4.0D0, 5.0D0, 6.0D0, 7.0D0/
      DATA t /290.0D0, 295.0D0, 300.0D0, 305.0D0, 310.0D0/
      DATA q /0.01D0, 0.012D0, 0.014D0, 0.016D0, 0.018D0/
      DATA pk /0.8D0, 0.9D0, 1.0D0, 1.1D0, 1.2D0/
      DATA pdsig /50.0D0, 75.0D0, 100.0D0, 125.0D0, 150.0D0/

C Save inputs as outputs (placeholder for actual ATURB call)
      OPEN(UNIT=10, FILE='test_aturb_fortran_output.txt')
      WRITE(10, *) 'u:'
      DO i = 1, N
        WRITE(10, *) u(i)
      END DO
      WRITE(10, *) 'v:'
      DO i = 1, N
        WRITE(10, *) v(i)
      END DO
      WRITE(10, *) 't:'
      DO i = 1, N
        WRITE(10, *) t(i)
      END DO
      WRITE(10, *) 'q:'
      DO i = 1, N
        WRITE(10, *) q(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_aturb_fortran_output.txt'
      END
