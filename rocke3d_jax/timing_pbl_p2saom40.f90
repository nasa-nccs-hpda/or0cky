! Timing driver for PBL similarity (simil), sized to P2SAoM40's real
! horizontal grid (im=72, jm=46 -> n=3312 points -- confirmed from this
! project's own P2SAoM40 restart file, see benchmark_all_p2saom40.py).
!
! Translates simil_numpy from benchmark_all.py/benchmark_all_cpu.py (the
! same reference implementation those scripts already benchmark JAX
! against) directly into Fortran array syntax, compiled with the same
! ifort toolchain used to build the real ModelE2/P2SAoM40 model (-O2).
!
! Build:
!   module load intel/2020Update4
!   ifort -O2 -o timing_pbl_p2saom40 timing_pbl_p2saom40.f90
! Run:
!   ./timing_pbl_p2saom40
! Prints: "MEAN_SECONDS <value>" (mean wall time per call over ITER calls).

program timing_pbl_p2saom40
  implicit none
  integer, parameter :: im = 72, jm = 46
  integer, parameter :: n = im * jm
  integer, parameter :: iter = 100
  real(8), parameter :: kappa = 0.4d0

  real(8), dimension(n) :: z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
  real(8), dimension(n) :: u, t, q, dpsim, dpsih, dpsiq
  integer :: i, it
  integer(8) :: count_start, count_end, count_rate
  real(8) :: elapsed, mean_seconds, r

  call random_seed()
  do i = 1, n
    call random_number(r); z(i) = 1.0d0 + r * 99.0d0
    call random_number(r); z0m(i) = 0.01d0 + r * 0.09d0
    call random_number(r); z0h(i) = 0.01d0 + r * 0.09d0
    call random_number(r); z0q(i) = 0.01d0 + r * 0.09d0
    call random_number(r); lmonin(i) = -100.0d0 + r * 200.0d0
    call random_number(r); ustar(i) = 0.1d0 + r * 0.9d0
    call random_number(r); tstar(i) = 0.1d0 + r * 0.9d0
    call random_number(r); qstar(i) = 0.01d0 + r * 0.09d0
    call random_number(r); tg(i) = 280.0d0 + r * 40.0d0
    call random_number(r); qg(i) = 0.01d0 + r * 0.04d0
  end do

  call system_clock(count_rate=count_rate)

  call run_simil(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg, &
                  u, t, q, dpsim, dpsih, dpsiq, n)

  call system_clock(count_start)
  do it = 1, iter
    call run_simil(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg, &
                    u, t, q, dpsim, dpsih, dpsiq, n)
  end do
  call system_clock(count_end)

  elapsed = real(count_end - count_start, 8) / real(count_rate, 8)
  mean_seconds = elapsed / real(iter, 8)

  print '(A,ES14.6)', 'MEAN_SECONDS ', mean_seconds

contains

  subroutine run_simil(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg, &
                        u, t, q, dpsim, dpsih, dpsiq, n)
    integer, intent(in) :: n
    real(8), dimension(n), intent(in) :: z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
    real(8), dimension(n), intent(out) :: u, t, q, dpsim, dpsih, dpsiq
    real(8), dimension(n) :: zet, zet0_m, zet0_h, x, x0_m, dm, dh
    real(8), parameter :: gamamu = 16.0d0, by3 = 1.0d0 / 3.0d0, zetm = -1.0d0
    logical, dimension(n) :: mask_unstable_gt, mask_unstable_le

    zet = z / lmonin
    zet0_m = z0m / lmonin
    zet0_h = z0h / lmonin

    where (1.0d0 - gamamu * zet >= 0.0d0)
      x = (1.0d0 - gamamu * zet) ** 0.25d0
    elsewhere
      x = 0.0d0
    end where

    where (1.0d0 - gamamu * zet0_m >= 0.0d0)
      x0_m = (1.0d0 - gamamu * zet0_m) ** 0.25d0
    elsewhere
      x0_m = 0.0d0
    end where

    mask_unstable_gt = (zet < 0.0d0) .and. (zet > zetm)
    mask_unstable_le = (zet < 0.0d0) .and. (zet <= zetm)

    where (mask_unstable_gt)
      dpsim = log(((1.0d0 + x) * (1.0d0 + x) * (1.0d0 + x * x)) / &
                   ((1.0d0 + x0_m) * (1.0d0 + x0_m) * (1.0d0 + x0_m * x0_m))) &
              - 2.0d0 * (atan(x) - atan(x0_m))
    elsewhere (mask_unstable_le)
      dpsim = log(((1.0d0 + (1.0d0 - gamamu * zetm) ** 0.25d0) * &
                    (1.0d0 + (1.0d0 - gamamu * zetm) ** 0.25d0) * &
                    (1.0d0 + (1.0d0 - gamamu * zetm) ** 0.5d0)) / &
                   ((1.0d0 + x0_m) * (1.0d0 + x0_m) * (1.0d0 + x0_m * x0_m))) &
              - 2.0d0 * (atan((1.0d0 - gamamu * zetm) ** 0.25d0) - atan(x0_m)) &
              + log(zet / zetm) &
              - 1.140125d0 * ((-zet) ** by3 - (-zetm) ** by3)
    elsewhere (zet <= 1.0d0)
      dpsim = -4.7d0 * (zet - zet0_m)
    elsewhere
      dpsim = -4.7d0 * (1.0d0 - zet0_m) + 1.0d0 * (5.0d0 - 4.7d0) * log(zet / 1.0d0) &
              - 5.0d0 * (zet - 1.0d0)
    end where

    dm = max(log(z / z0m) - dpsim, 1.0d-3)

    dpsih = log(z / z0h) - gamamu * (zet - zet0_h)
    dh = max(log(z / z0h) - dpsih, 1.0d-3)
    dpsiq = dpsih

    u = (ustar / kappa) * (log(z / z0m) - dpsim)
    t = tg + (tstar / kappa) * (log(z / z0h) - dpsih)
    q = qg + (qstar / kappa) * (log(z / z0q) - dpsiq)
  end subroutine run_simil

end program timing_pbl_p2saom40
