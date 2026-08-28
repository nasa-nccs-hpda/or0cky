program drycnv_benchmark
  implicit none
  integer, parameter :: I = 32, J = 32, L = 20
  real*8, dimension(I, J, L) :: T, Q, PK, PDSIG, T_out, Q_out
  real*8 :: deltx = 0.608d0
  integer :: i, j, l, n
  real*8 :: start_time, end_time, total_time
  integer :: iterations = 1000

  ! Initialize random inputs (same as JAX test)
  call random_seed()
  call random_number(T)
  call random_number(Q)
  call random_number(PK)
  call random_number(PDSIG)
  T = T * 100.0d0 + 200.0d0  ! Scale to 200-300 K
  Q = Q * 0.02d0             ! Scale to 0-0.02 kg/kg
  PK = PK * 0.5d0 + 0.5d0    ! Scale to 0.5-1.0
  PDSIG = PDSIG * 0.1d0 + 0.1d0  ! Scale to 0.1-0.2

  ! Warm-up run
  call dry_convection_mixing(T, Q, PK, PDSIG, T_out, Q_out, deltx, I, J, L)

  ! Benchmark
  call cpu_time(start_time)
  do n = 1, iterations
     call dry_convection_mixing(T, Q, PK, PDSIG, T_out, Q_out, deltx, I, J, L)
  end do
  call cpu_time(end_time)
  total_time = (end_time - start_time) / real(iterations, 8)
  print *, "Fortran execution time (per call): ", total_time, " seconds"

contains
  subroutine dry_convection_mixing(T, Q, PK, PDSIG, T_out, Q_out, deltx, I, J, L)
    implicit none
    integer, intent(in) :: I, J, L
    real*8, intent(in) :: T(I, J, L), Q(I, J, L), PK(I, J, L), PDSIG(I, J, L), deltx
    real*8, intent(out) :: T_out(I, J, L), Q_out(I, J, L)
    real*8 :: TV(I, J, L), PKMS, TVMS, QMS, RDP, THM, QM
    integer :: i, j, l

    T_out = T
    Q_out = Q
    TV = T * (1.0d0 + Q * deltx)

    do l = 1, L - 1
       do j = 1, J
          do i = 1, I
             if (TV(i, j, l) > TV(i, j, l + 1)) then
                PKMS = PK(i, j, l) * PDSIG(i, j, l) + PK(i, j, l + 1) * PDSIG(i, j, l + 1)
                TVMS = TV(i, j, l) * PK(i, j, l) * PDSIG(i, j, l) + &
                       TV(i, j, l + 1) * PK(i, j, l + 1) * PDSIG(i, j, l + 1)
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
       TV = T_out * (1.0d0 + Q_out * deltx)  ! Update TV for next iteration
    end do
  end subroutine dry_convection_mixing
end program drycnv_benchmark
