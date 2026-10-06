! D103-D106 driver: reads synthetic/recorded cases from in.bin, runs the REAL AADVQ0 + AADVQ(qlimit=.true.) from
! QUS3D.f and writes q0.bin / out.bin per case in the layouts of dyn_qdynam_io.py (ffd_qdyn_*_q0 / _out, big endian).
! in.bin: [ncase], per case: [id], RM (im,jm,lm), RMOM (9,im,jm,lm), MB, MU, MV, MW (im,jm,lm), IMAXJ(jm) as doubles.
program drv
  use resolution, only: im, jm, lm
  use atm_com
  use geom, only: imaxj
  use domain_decomp_atm, only: grid
  use tracer_adv
  implicit none
  real*8 :: hdr(1), idd(1)
  integer :: nc, c, j, l, k, n, fu
  real*8, allocatable :: rm(:,:,:), rmom(:,:,:,:), ijm(:)
  character(len=64) :: fname
  real*8 :: dimaxj(jm)
  integer :: idz
  allocate(rm(im,jm,lm), rmom(9,im,jm,lm))
  allocate(mus(im,jm,lm), mvs(im,jm,lm), mws(im,jm,lm), mb(im,jm,lm), mma(im,jm,lm))
  call alloc_tracer_adv(grid)
  open(10, file='in.bin', form='unformatted', access='stream', status='old')
  read(10) hdr
  nc = int(hdr(1))
  do c = 1, nc
    read(10) idd
    read(10) rm
    read(10) rmom
    read(10) mb
    read(10) mus
    read(10) mvs
    read(10) mws
    read(10) dimaxj
    imaxj = nint(dimaxj)
    write(fname,'(A,I0,A)') 'q0_', c, '.bin'
    call aadvq0
    fu = 20
    open(fu, file=trim(fname), form='unformatted', access='stream', status='replace')
    idz = 0
    if (do_z_extra) idz = 1
    write(fu) idd(1), dble(ncyc), dble(idz), 0d0
    write(fu) dble(ncycxy)
    write(fu) dble(nstepx)
    write(fu) dble(nstepz_extra)
    write(fu) dble(lminzij)
    write(fu) dble(lmaxzij)
    write(fu) mw_extra
    write(fu) mus
    write(fu) mvs
    write(fu) mws
    write(fu) pv_south
    write(fu) dble(ni_checkfobs_y)
    write(fu) dble(ni_checkfobs_z)
    n = 0
    do l=1,lm ; do j=1,jm ; n=n+ni_checkfobs_y(j,l) ; enddo ; enddo
    write(fu) dble(n)
    do l=1,lm ; do j=1,jm ; do k=1,ni_checkfobs_y(j,l)
      write(fu) dble(j),dble(l),dble(i_checkfobs_y(k,j,l))
    enddo ; enddo ; enddo
    n = 0
    do l=1,lm ; do j=1,jm ; n=n+ni_checkfobs_z(j,l) ; enddo ; enddo
    write(fu) dble(n)
    do l=1,lm ; do j=1,jm ; do k=1,ni_checkfobs_z(j,l)
      write(fu) dble(j),dble(l),dble(i_checkfobs_z(k,j,l))
    enddo ; enddo ; enddo
    close(fu)
    sfbm = 0.; sbm = 0.; sbf = 0.
    sfcm = 0.; scm = 0.; scf = 0.
    call aadvq(rm, rmom, .true., 'q       ')
    write(fname,'(A,I0,A)') 'out_', c, '.bin'
    open(fu, file=trim(fname), form='unformatted', access='stream', status='replace')
    write(fu) idd(1)
    write(fu) rm
    write(fu) rmom
    write(fu) mma
    write(fu) sbf
    write(fu) sbm
    write(fu) sfbm
    write(fu) scf
    write(fu) scm
    write(fu) sfcm
    write(fu) scf3d
    close(fu)
  enddo
end program drv
