program drv
  use QUSDEF
  implicit none
  integer, parameter :: nx=72
  integer :: nb, b, ql, dtyp, ierr, nerr, u, ios
  real*8 :: s(nx), smom(nmom,nx), mass(nx), dm(nx), f(nx), fmom(nmom,nx)
  integer :: dir(nmom)
  real*8 :: hdr(2)
  open(10,file='in.bin',form='unformatted',access='stream',status='old')
  open(11,file='out.bin',form='unformatted',access='stream',status='replace')
  read(10) hdr
  nb = int(hdr(1))
  do b=1,nb
    read(10) hdr
    ql = int(hdr(1)); dtyp = int(hdr(2))
    read(10) s
    read(10) smom
    read(10) mass
    read(10) dm
    if (dtyp.eq.0) dir = xdir
    if (dtyp.eq.1) dir = ydir
    if (dtyp.eq.2) dir = zdir
    f = 0 ; fmom = 0
    call adv1d(s,smom,f,fmom,mass,dm,nx,ql.eq.1,1,dir,ierr,nerr)
    write(11) dble(ierr), dble(nerr)
    write(11) s
    write(11) smom
    write(11) mass
    write(11) f
    write(11) fmom
  enddo
end program
