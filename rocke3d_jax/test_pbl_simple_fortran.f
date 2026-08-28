C Minimal Fortran test for PBL_SIMPLE module
      PROGRAM TEST_PBL_SIMPLE_FORTRAN
      INTEGER, PARAMETER :: N = 5
      REAL*8 us(N), vs(N), tsv(N), qsrf(N)
      INTEGER i

C Initialize inputs with fixed values
      DATA us /5.0D0, 7.0D0, 9.0D0, 11.0D0, 13.0D0/
      DATA vs /3.0D0, 4.0D0, 5.0D0, 6.0D0, 7.0D0/
      DATA tsv /290.0D0, 295.0D0, 300.0D0, 305.0D0, 310.0D0/
      DATA qsrf /0.01D0, 0.012D0, 0.014D0, 0.016D0, 0.018D0/

C Save inputs as outputs (placeholder for actual PBL_SIMPLE call)
      OPEN(UNIT=10, FILE='test_pbl_simple_fortran_output.txt')
      WRITE(10, *) 'us:'
      DO i = 1, N
        WRITE(10, *) us(i)
      END DO
      WRITE(10, *) 'vs:'
      DO i = 1, N
        WRITE(10, *) vs(i)
      END DO
      WRITE(10, *) 'tsv:'
      DO i = 1, N
        WRITE(10, *) tsv(i)
      END DO
      WRITE(10, *) 'qsrf:'
      DO i = 1, N
        WRITE(10, *) qsrf(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_pbl_simple_fortran_output.txt'
      END
