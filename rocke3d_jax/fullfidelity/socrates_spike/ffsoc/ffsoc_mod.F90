! D3 / R2 instrumentation module (scratch copy of SOCRATES only; the original tree is untouched).
! Adds NO arithmetic to the library: only stage timers (SYSTEM_CLOCK) and stream dumps of arrays already computed.
! Active only if env FFSOC is set.  FFSOC_STRIDE (default 331): arrays are dumped for every STRIDE-th radiance_calc call
! (call counter 0-based over LW and SW calls, in call order); timers are recorded for EVERY call.
! Output: ffsoc_dump.bin in the run directory, big-endian stream.  Record = 8 int32 header
!   (tag, isolir, ncall, i_band, k, rank, nd1, nd2) + for rank>=1 extents/lbounds as 2*rank int32 + data real*8 (or int32 for tag 900 times in ns as real*8).
! Tags: 0 inputs of radiance_calc; 10 grey optical properties after grey_opt_prop; 20 k_eqv etc. of solve_band_k_eqv_scl;
!       21 band fluxes after monochromatic_radiance (per k-term); 30 band-end cumulative fluxes; 900 stage times [s] of one call.
MODULE ffsoc_mod
  IMPLICIT NONE
  INTEGER, PARAMETER :: ffsoc_unit = 1700, ffsoc_nst = 12
  LOGICAL, SAVE :: ffsoc_on = .FALSE., ffsoc_init_done = .FALSE., ffsoc_dump_now = .FALSE.
  INTEGER, SAVE :: ffsoc_stride = 331, ffsoc_ncall = 0
  INTEGER(8), SAVE :: ffsoc_t0(ffsoc_nst) = 0, ffsoc_rate = 1
  REAL(8), SAVE :: ffsoc_acc(ffsoc_nst) = 0.0d0
  INTEGER, SAVE :: ffsoc_nacc(ffsoc_nst) = 0
  INTERFACE ffsoc_arr
    MODULE PROCEDURE ffsoc_a1, ffsoc_a2, ffsoc_a3, ffsoc_a4
  END INTERFACE
CONTAINS
  SUBROUTINE ffsoc_init
    CHARACTER(LEN=32) :: s
    INTEGER :: st, ios
    ffsoc_init_done = .TRUE.
    CALL GET_ENVIRONMENT_VARIABLE('FFSOC', s, status=st)
    IF (st /= 0) RETURN
    ffsoc_on = .TRUE.
    CALL GET_ENVIRONMENT_VARIABLE('FFSOC_STRIDE', s, status=st)
    IF (st == 0) READ(s, *, IOSTAT=ios) ffsoc_stride
    IF (ffsoc_stride < 1) ffsoc_stride = 1
    CALL SYSTEM_CLOCK(COUNT_RATE=ffsoc_rate)
    OPEN(ffsoc_unit, FILE='ffsoc_dump.bin', ACCESS='stream', FORM='unformatted', STATUS='REPLACE')
  END SUBROUTINE ffsoc_init

  SUBROUTINE ffsoc_begin_call
    IF (.NOT. ffsoc_init_done) CALL ffsoc_init
    IF (.NOT. ffsoc_on) RETURN
    ffsoc_dump_now = (MOD(ffsoc_ncall, ffsoc_stride) == 0)
    ffsoc_acc = 0.0d0
    ffsoc_nacc = 0
  END SUBROUTINE ffsoc_begin_call

  SUBROUTINE ffsoc_tic(k)
    INTEGER, INTENT(IN) :: k
    INTEGER(8) :: c
    IF (.NOT. ffsoc_on) RETURN
    CALL SYSTEM_CLOCK(c)
    ffsoc_t0(k) = c
  END SUBROUTINE ffsoc_tic

  SUBROUTINE ffsoc_toc(k)
    INTEGER, INTENT(IN) :: k
    INTEGER(8) :: c
    IF (.NOT. ffsoc_on) RETURN
    CALL SYSTEM_CLOCK(c)
    ffsoc_acc(k) = ffsoc_acc(k) + REAL(c - ffsoc_t0(k), 8) / REAL(ffsoc_rate, 8)
    ffsoc_nacc(k) = ffsoc_nacc(k) + 1
  END SUBROUTINE ffsoc_toc

  SUBROUTINE ffsoc_end_call(isolir, n_band)
    INTEGER, INTENT(IN) :: isolir, n_band
    IF (.NOT. ffsoc_on) RETURN
    WRITE(ffsoc_unit) 900, isolir, ffsoc_ncall, n_band, 0, 1, ffsoc_nst, 0
    WRITE(ffsoc_unit) 1, ffsoc_nst
    WRITE(ffsoc_unit) ffsoc_acc
    WRITE(ffsoc_unit) ffsoc_nacc
    ffsoc_ncall = ffsoc_ncall + 1
  END SUBROUTINE ffsoc_end_call

  SUBROUTINE ffsoc_hdr(tag, isolir, i_band, k, rank, nd1, nd2)
    INTEGER, INTENT(IN) :: tag, isolir, i_band, k, rank, nd1, nd2
    WRITE(ffsoc_unit) tag, isolir, ffsoc_ncall, i_band, k, rank, nd1, nd2
  END SUBROUTINE ffsoc_hdr

  SUBROUTINE ffsoc_a1(tag, isolir, i_band, k, a)
    INTEGER, INTENT(IN) :: tag, isolir, i_band, k
    REAL(8), INTENT(IN) :: a(:)
    IF (.NOT. (ffsoc_on .AND. ffsoc_dump_now)) RETURN
    CALL ffsoc_hdr(tag, isolir, i_band, k, 1, SIZE(a, 1), 0)
    WRITE(ffsoc_unit) SIZE(a, 1), LBOUND(a, 1)
    WRITE(ffsoc_unit) a
  END SUBROUTINE ffsoc_a1
  SUBROUTINE ffsoc_a2(tag, isolir, i_band, k, a)
    INTEGER, INTENT(IN) :: tag, isolir, i_band, k
    REAL(8), INTENT(IN) :: a(:, :)
    IF (.NOT. (ffsoc_on .AND. ffsoc_dump_now)) RETURN
    CALL ffsoc_hdr(tag, isolir, i_band, k, 2, SIZE(a, 1), SIZE(a, 2))
    WRITE(ffsoc_unit) SIZE(a, 1), SIZE(a, 2), LBOUND(a, 1), LBOUND(a, 2)
    WRITE(ffsoc_unit) a
  END SUBROUTINE ffsoc_a2
  SUBROUTINE ffsoc_a3(tag, isolir, i_band, k, a)
    INTEGER, INTENT(IN) :: tag, isolir, i_band, k
    REAL(8), INTENT(IN) :: a(:, :, :)
    IF (.NOT. (ffsoc_on .AND. ffsoc_dump_now)) RETURN
    CALL ffsoc_hdr(tag, isolir, i_band, k, 3, SIZE(a, 1), SIZE(a, 2))
    WRITE(ffsoc_unit) SIZE(a, 1), SIZE(a, 2), SIZE(a, 3), LBOUND(a, 1), LBOUND(a, 2), LBOUND(a, 3)
    WRITE(ffsoc_unit) a
  END SUBROUTINE ffsoc_a3
  SUBROUTINE ffsoc_a4(tag, isolir, i_band, k, a)
    INTEGER, INTENT(IN) :: tag, isolir, i_band, k
    REAL(8), INTENT(IN) :: a(:, :, :, :)
    IF (.NOT. (ffsoc_on .AND. ffsoc_dump_now)) RETURN
    CALL ffsoc_hdr(tag, isolir, i_band, k, 4, SIZE(a, 1), SIZE(a, 2))
    WRITE(ffsoc_unit) SIZE(a, 1), SIZE(a, 2), SIZE(a, 3), SIZE(a, 4), LBOUND(a, 1), LBOUND(a, 2), LBOUND(a, 3), LBOUND(a, 4)
    WRITE(ffsoc_unit) a
  END SUBROUTINE ffsoc_a4
END MODULE ffsoc_mod
