! Timing driver for DRYCNV (dry convection mixing), sized to P2SAoM40's
! real grid (im=72, jm=46, lm=40 -- confirmed from this project's own
! P2SAoM40 restart file, see benchmark_all_p2saom40.py).
!
! Same algorithm as drycnv.py's dry_convection_mixing / benchmark_all*.py's
! dry_convection_numpy, translated to Fortran, compiled with the same ifort
! toolchain used to build the real ModelE2/P2SAoM40 model (-O2), so the
! comparison against JAX is apples-to-apples: same shapes, same iteration
! count, real compiled Fortran vs real JAX, both on this node's CPU.
!
! Build:
!   module load intel/2020Update4
!   ifort -O2 -o timing_drycnv_p2saom40 timing_drycnv_p2saom40.f90
! Run:
!   ./timing_drycnv_p2saom40
! Prints a single line: "MEAN_SECONDS <value>" (mean wall time per call,
! averaged over ITER repeated calls, matching the Python side's iterations).

program timing_drycnv_p2saom40
  implicit none
  integer, parameter :: im = 72, jm = 46, lm = 40
  integer, parameter :: iter = 100
  real(8), parameter :: deltx = 0.608d0

  real(8), dimension(im, jm, lm) :: T, Q, PK, PDSIG
  real(8), dimension(im, jm, lm) :: T_out, Q_out, TV
  real(8) :: PKMS, TVMS, QMS, RDP, THM, QM
  integer :: i, j, l, it
  integer(8) :: count_start, count_end, count_rate
  real(8) :: elapsed, mean_seconds
  real(8) :: r

  ! Deterministic pseudo-random fill (matches the value RANGES used in
  ! drycnv_input_generator in benchmark_all_p2saom40.py: T in [200,300],
  ! Q in [0,0.02], PK in [0.5,1.0], PDSIG in [0.1,0.2]). Not the same RNG
  ! stream as JAX -- this is a timing test, not a bit-for-bit cross-check
  ! (see compare_fortran_jax.py / FINDINGS.md for the existing numerical
  ! validation work on find_dpsim/find_dpsih).
  call random_seed()
  do l = 1, lm
    do j = 1, jm
      do i = 1, im
        call random_number(r); T(i, j, l) = 200.0d0 + r * 100.0d0
        call random_number(r); Q(i, j, l) = r * 0.02d0
        call random_number(r); PK(i, j, l) = 0.5d0 + r * 0.5d0
        call random_number(r); PDSIG(i, j, l) = 0.1d0 + r * 0.1d0
      end do
    end do
  end do

  call system_clock(count_rate=count_rate)

  ! Warm-up (parity with the Python side calling once before timing)
  call run_drycnv(T, Q, PK, PDSIG, T_out, Q_out, im, jm, lm, deltx)

  call system_clock(count_start)
  do it = 1, iter
    call run_drycnv(T, Q, PK, PDSIG, T_out, Q_out, im, jm, lm, deltx)
  end do
  call system_clock(count_end)

  elapsed = real(count_end - count_start, 8) / real(count_rate, 8)
  mean_seconds = elapsed / real(iter, 8)

  print '(A,ES14.6)', 'MEAN_SECONDS ', mean_seconds

contains

  subroutine run_drycnv(T, Q, PK, PDSIG, T_out, Q_out, im, jm, lm, deltx)
    integer, intent(in) :: im, jm, lm
    real(8), intent(in) :: deltx
    real(8), dimension(im, jm, lm), intent(in) :: T, Q, PK, PDSIG
    real(8), dimension(im, jm, lm), intent(out) :: T_out, Q_out
    real(8), dimension(im, jm, lm) :: TV
    real(8) :: PKMS, TVMS, QMS, RDP, THM, QM
    integer :: i, j, l

    T_out = T
    Q_out = Q
    TV = T_out * (1.0d0 + Q_out * deltx)

    do l = 1, lm - 1
      do j = 1, jm
        do i = 1, im
          if (TV(i, j, l) > TV(i, j, l + 1)) then
            PKMS = PK(i, j, l) * PDSIG(i, j, l) + PK(i, j, l + 1) * PDSIG(i, j, l + 1)
            TVMS = TV(i, j, l) * PK(i, j, l) * PDSIG(i, j, l) &
                 + TV(i, j, l + 1) * PK(i, j, l + 1) * PDSIG(i, j, l + 1)
            QMS = Q_out(i, j, l) * PDSIG(i, j, l) + Q_out(i, j, l + 1) * PDSIG(i, j, l + 1)
            RDP = 1.0d0 / (PDSIG(i, j, l) + PDSIG(i, j, l + 1))
            THM = TVMS / (PKMS * (1.0d0 + QMS * RDP * deltx))
            QM = QMS * RDP
            T_out(i, j, l) = THM
            T_out(i, j, l + 1) = THM
            Q_out(i, j, l) = QM
            Q_out(i, j, l + 1) = QM
          end if
        end do
      end do
      TV(:, :, l) = T_out(:, :, l) * (1.0d0 + Q_out(:, :, l) * deltx)
      TV(:, :, l + 1) = T_out(:, :, l + 1) * (1.0d0 + Q_out(:, :, l + 1) * deltx)
    end do
  end subroutine run_drycnv

end program timing_drycnv_p2saom40
