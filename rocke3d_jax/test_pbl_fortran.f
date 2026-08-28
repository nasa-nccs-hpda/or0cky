C Minimal Fortran test for PBL module (find_dpsim)
      PROGRAM TEST_PBL_FORTRAN
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 5
      REAL*8 :: find_dpsim
      REAL*8, DIMENSION(N) :: zet, zet0, dpsim_out
      INTEGER :: i
      INTRINSIC :: LOG, ATAN
      DO i = 1, N
        zet(i) = -0.5d0 + (i - 1) * 0.2d0
        zet0(i) = -0.3d0 + (i - 1) * 0.1d0
      END DO
      DO i = 1, N
        dpsim_out(i) = find_dpsim(zet(i), zet0(i))
      END DO
      OPEN(UNIT=10, FILE='test_pbl_fortran_output.txt')
      WRITE(10, *) 'dpsim:'
      DO i = 1, N
        WRITE(10, *) dpsim_out(i)
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_pbl_fortran_output.txt'
      END PROGRAM TEST_PBL_FORTRAN

      REAL*8 FUNCTION find_dpsim(zet, zet0)
      IMPLICIT NONE
      REAL*8, INTENT(IN) :: zet, zet0
      REAL*8, PARAMETER :: gamams = 4.7d0
      REAL*8, PARAMETER :: gamamu = 16.0d0
      REAL*8, PARAMETER :: zet1 = 1.0d0
      REAL*8, PARAMETER :: slope1 = 5.0d0
      REAL*8, PARAMETER :: by3 = 1.0d0 / 3.0d0
      REAL*8, PARAMETER :: zetm = -1.0d0
      REAL*8 :: x, x0, xm
      INTRINSIC :: LOG, ATAN
      if(zet.ge.0.d0) then
        if(zet.le.zet1) then
          find_dpsim = -gamams * (zet - zet0)
        else
          find_dpsim = -gamams * (zet1 - zet0)
     &          + zet1 * (slope1 - gamams) * LOG(zet / zet1)
     &          - slope1 * (zet - zet1)
        endif
      else
        x = (1.d0 - gamamu * zet)**0.25d0
        x0 = (1.d0 - gamamu * zet0)**0.25d0
        xm = (1.d0 - gamamu * zetm)**0.25d0
        if(zet.gt.zetm) then
          find_dpsim = LOG((1+x)*(1+x)*(1+x*x)/
     1          ((1+x0)*(1+x0)*(1+x0*x0)))-
     2          2.d0 * (ATAN(x) - ATAN(x0))
        else
          find_dpsim = LOG((1+xm)*(1+xm)*(1+xm*xm)/
     1          ((1+x0)*(1+x0)*(1+x0*x0)))-
     2          2.d0 * (ATAN(xm) - ATAN(x0)) +
     3          LOG(zet / zetm) -
     4          1.140125d0 * ((-zet)**by3 - (-zetm)**by3)
        endif
      endif
      END FUNCTION find_dpsim
