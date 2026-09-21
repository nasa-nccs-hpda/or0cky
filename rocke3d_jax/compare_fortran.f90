! P2SAoM40 Fortran reference leg of the JAX-vs-Fortran comparison.
!
! Reads the shared input arrays written by compare_generate_inputs.py
! (compare_data/*.dat, raw float64, Fortran/column-major order), computes
! DRYCNV and PBL-similarity on them, writes the outputs back out (for the
! accuracy check against JAX) and reports mean per-call wall time over 100
! calls (for the timing comparison, same iteration count as the JAX side).
!
! Build:
!   module load intel/2020Update4
!   ifort -O2 -o compare_fortran compare_fortran.f90
! Run (from this directory, after compare_generate_inputs.py has run):
!   ./compare_fortran
! Writes compare_data/drycnv_fortran_{T,Q}_out.dat,
!        compare_data/pbl_fortran_{u,t,q,dpsim,dpsih,dpsiq}_out.dat
! Prints:
!   DRYCNV_MEAN_SECONDS <value>
!   PBL_MEAN_SECONDS <value>

program compare_fortran
  implicit none
  integer, parameter :: im = 72, jm = 46, lm = 40
  integer, parameter :: n = im * jm
  integer, parameter :: iter = 100
  real(8), parameter :: deltx = 0.608d0
  real(8), parameter :: kappa = 0.4d0
  character(len=*), parameter :: data_dir = "compare_data/"

  real(8), dimension(im, jm, lm) :: T, Q, PK, PDSIG, T_out, Q_out
  real(8), dimension(n) :: z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
  real(8), dimension(n) :: u, t_pbl, q_pbl, dpsim, dpsih, dpsiq
  integer(8) :: count_start, count_end, count_rate
  real(8) :: elapsed
  integer :: it

  call system_clock(count_rate=count_rate)

  ! --- DRYCNV ---
  call read_3d(data_dir // "drycnv_T.dat", T, im, jm, lm)
  call read_3d(data_dir // "drycnv_Q.dat", Q, im, jm, lm)
  call read_3d(data_dir // "drycnv_PK.dat", PK, im, jm, lm)
  call read_3d(data_dir // "drycnv_PDSIG.dat", PDSIG, im, jm, lm)

  call run_drycnv(T, Q, PK, PDSIG, T_out, Q_out, im, jm, lm, deltx)  ! warm-up
  call system_clock(count_start)
  do it = 1, iter
    call run_drycnv(T, Q, PK, PDSIG, T_out, Q_out, im, jm, lm, deltx)
  end do
  call system_clock(count_end)
  elapsed = real(count_end - count_start, 8) / real(count_rate, 8)
  print '(A,ES14.6)', 'DRYCNV_MEAN_SECONDS ', elapsed / real(iter, 8)

  call write_3d(data_dir // "drycnv_fortran_T_out.dat", T_out, im, jm, lm)
  call write_3d(data_dir // "drycnv_fortran_Q_out.dat", Q_out, im, jm, lm)

  ! --- PBL ---
  call read_1d(data_dir // "pbl_z.dat", z, n)
  call read_1d(data_dir // "pbl_z0m.dat", z0m, n)
  call read_1d(data_dir // "pbl_z0h.dat", z0h, n)
  call read_1d(data_dir // "pbl_z0q.dat", z0q, n)
  call read_1d(data_dir // "pbl_lmonin.dat", lmonin, n)
  call read_1d(data_dir // "pbl_ustar.dat", ustar, n)
  call read_1d(data_dir // "pbl_tstar.dat", tstar, n)
  call read_1d(data_dir // "pbl_qstar.dat", qstar, n)
  call read_1d(data_dir // "pbl_tg.dat", tg, n)
  call read_1d(data_dir // "pbl_qg.dat", qg, n)

  call run_simil(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg, &
                  u, t_pbl, q_pbl, dpsim, dpsih, dpsiq, n, kappa)  ! warm-up
  call system_clock(count_start)
  do it = 1, iter
    call run_simil(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg, &
                    u, t_pbl, q_pbl, dpsim, dpsih, dpsiq, n, kappa)
  end do
  call system_clock(count_end)
  elapsed = real(count_end - count_start, 8) / real(count_rate, 8)
  print '(A,ES14.6)', 'PBL_MEAN_SECONDS ', elapsed / real(iter, 8)

  call write_1d(data_dir // "pbl_fortran_u_out.dat", u, n)
  call write_1d(data_dir // "pbl_fortran_t_out.dat", t_pbl, n)
  call write_1d(data_dir // "pbl_fortran_q_out.dat", q_pbl, n)
  call write_1d(data_dir // "pbl_fortran_dpsim_out.dat", dpsim, n)
  call write_1d(data_dir // "pbl_fortran_dpsih_out.dat", dpsih, n)
  call write_1d(data_dir // "pbl_fortran_dpsiq_out.dat", dpsiq, n)

contains

  subroutine read_3d(path, arr, im, jm, lm)
    character(len=*), intent(in) :: path
    integer, intent(in) :: im, jm, lm
    real(8), dimension(im, jm, lm), intent(out) :: arr
    integer :: unit
    open(newunit=unit, file=path, form='unformatted', access='stream', status='old')
    read(unit) arr
    close(unit)
  end subroutine read_3d

  subroutine write_3d(path, arr, im, jm, lm)
    character(len=*), intent(in) :: path
    integer, intent(in) :: im, jm, lm
    real(8), dimension(im, jm, lm), intent(in) :: arr
    integer :: unit
    open(newunit=unit, file=path, form='unformatted', access='stream', status='replace')
    write(unit) arr
    close(unit)
  end subroutine write_3d

  subroutine read_1d(path, arr, n)
    character(len=*), intent(in) :: path
    integer, intent(in) :: n
    real(8), dimension(n), intent(out) :: arr
    integer :: unit
    open(newunit=unit, file=path, form='unformatted', access='stream', status='old')
    read(unit) arr
    close(unit)
  end subroutine read_1d

  subroutine write_1d(path, arr, n)
    character(len=*), intent(in) :: path
    integer, intent(in) :: n
    real(8), dimension(n), intent(in) :: arr
    integer :: unit
    open(newunit=unit, file=path, form='unformatted', access='stream', status='replace')
    write(unit) arr
    close(unit)
  end subroutine write_1d

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

  ! Faithful translation of pbl.py's find_dpsim (all 4 branches: stable
  ! zet<=zet1, stable zet>zet1, unstable zet>zetm, unstable zet<=zetm).
  elemental real(8) function find_dpsim_f(zet, zet0) result(dpsim)
    real(8), intent(in) :: zet, zet0
    real(8), parameter :: gamamu = 16.0d0, gamams = 4.7d0, zet1 = 1.0d0
    real(8), parameter :: slope1 = 5.0d0, by3 = 1.0d0 / 3.0d0, zetm = -1.0d0
    real(8) :: x, x0, xm

    if (zet >= 0.0d0) then
      if (zet <= zet1) then
        dpsim = -gamams * (zet - zet0)
      else
        dpsim = -gamams * (zet1 - zet0) + zet1 * (slope1 - gamams) * log(zet / zet1) &
                - slope1 * (zet - zet1)
      end if
    else
      x = (1.0d0 - gamamu * zet) ** 0.25d0
      x0 = (1.0d0 - gamamu * zet0) ** 0.25d0
      xm = (1.0d0 - gamamu * zetm) ** 0.25d0
      if (zet > zetm) then
        dpsim = log(((1 + x) * (1 + x) * (1 + x * x)) / ((1 + x0) * (1 + x0) * (1 + x0 * x0))) &
                - 2.0d0 * (atan(x) - atan(x0))
      else
        dpsim = log(((1 + xm) * (1 + xm) * (1 + xm * xm)) / ((1 + x0) * (1 + x0) * (1 + x0 * x0))) &
                - 2.0d0 * (atan(xm) - atan(x0)) + log(zet / zetm) &
                - 1.140125d0 * ((-zet) ** by3 - (-zetm) ** by3)
      end if
    end if
  end function find_dpsim_f

  ! Faithful translation of pbl.py's find_dpsih (all 3 branches: stable
  ! zet<=zet1, stable zet>zet1, unstable). Used for BOTH dpsih (with z0h)
  ! and dpsiq (with z0q, its own call -- NOT dpsiq=dpsih).
  elemental real(8) function find_dpsih_f(zet, zet0, z, z0) result(dpsih)
    real(8), intent(in) :: zet, zet0, z, z0
    real(8), parameter :: gamahs = 4.7d0, gamahu = 16.0d0, zet1 = 1.0d0
    real(8), parameter :: slope1 = 5.0d0, sigma = 1.0d0, sigma1 = 1.0d0

    if (zet >= 0.0d0) then
      if (zet <= zet1) then
        dpsih = sigma1 * log(z / z0) - sigma * gamahs * (zet - zet0)
      else
        dpsih = sigma1 * log(zet1 / z0) - sigma * gamahs * (zet1 - zet0) &
                + (1.0d0 + sigma * (zet1 * (slope1 - gamahs) - 1.0d0)) * log(zet / zet1) &
                - sigma * slope1 * (zet - zet1)
      end if
    else
      dpsih = sigma1 * log(z / z0) - sigma * gamahu * (zet - zet0)
    end if
  end function find_dpsih_f

  subroutine run_simil(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg, &
                        u, t, q, dpsim, dpsih, dpsiq, n, kappa)
    integer, intent(in) :: n
    real(8), intent(in) :: kappa
    real(8), dimension(n), intent(in) :: z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg
    real(8), dimension(n), intent(out) :: u, t, q, dpsim, dpsih, dpsiq
    real(8), dimension(n) :: zet, zet0_m, zet0_h, zet0_q

    zet = z / lmonin
    zet0_m = z0m / lmonin
    zet0_h = z0h / lmonin
    zet0_q = z0q / lmonin

    dpsim = find_dpsim_f(zet, zet0_m)
    dpsih = find_dpsih_f(zet, zet0_h, z, z0h)
    dpsiq = find_dpsih_f(zet, zet0_q, z, z0q)  ! separate call with z0q, matching simil()

    u = (ustar / kappa) * (log(z / z0m) - dpsim)
    t = tg + (tstar / kappa) * (log(z / z0h) - dpsih)
    q = qg + (qstar / kappa) * (log(z / z0q) - dpsiq)
  end subroutine run_simil

end program compare_fortran
