! D205: driver of the standalone REAL daily_LAKE: reads in.bin (big-endian f8, order below), calls daily_LAKE once, writes out.bin and resets.txt.
! in : flake fearth fland rsi msi snowi mwl gml tlake mldlk (im,jm each), hsi (4,im,jm), tanlk dlake0 dmwldf flice focean axyp (im,jm each)
! out: flake fearth fland rsi msi snowi mwl gml tlake mldlk (im,jm each), hsi (4,im,jm), dmwldf dgml svflake dlake glake gtemp gtempr mlhc mdwnimp edwnimp (im,jm each)
program drv
  use RESOLUTION, only : im, jm
  use LAKES_COM
  use SEAICE_COM
  use GEOM
  use GHY_COM
  use FLUXES
  use LANDICE_COM
  use DIAG_COM
  implicit none
  real*8, dimension(:,:), pointer :: rsi, msi, snowi, gt, gtr, mlhc
  real*8, dimension(:,:,:), pointer :: hsi
  integer :: j
  allocate(si_atm%RSI(im,jm), si_atm%MSI(im,jm), si_atm%SNOWI(im,jm), si_atm%HSI(4,im,jm))
  allocate(atmocn%GTEMP(im,jm), atmocn%GTEMPR(im,jm), atmocn%MLHC(im,jm))
  atmocn%GTEMP = -1d300; atmocn%GTEMPR = -1d300; atmocn%MLHC = -1d300
  dgml = 0d0; mdwnimp = 0d0; edwnimp = 0d0; dlake = -1d300; glake = -1d300; svflake = -1d300
  imaxj = im; imaxj(1) = 1; imaxj(jm) = 1
  jreg = 1
  byaxyp = 0d0
  open(10, file='in.bin', form='unformatted', access='stream', convert='big_endian')
  read(10) flake, fearth, fland, si_atm%RSI, si_atm%MSI, si_atm%SNOWI, mwl, gml, tlake, mldlk, si_atm%HSI, tanlk, dlake0, dmwldf, flice, focean, axyp
  close(10)
  where (axyp > 0) byaxyp = 1d0/axyp
  open(11, file='resets.txt')
  call daily_LAKE
  close(11)
  open(12, file='out.bin', form='unformatted', access='stream', convert='big_endian')
  write(12) flake, fearth, fland, si_atm%RSI, si_atm%MSI, si_atm%SNOWI, mwl, gml, tlake, mldlk, si_atm%HSI, dmwldf, dgml, svflake, dlake, glake, &
       atmocn%GTEMP, atmocn%GTEMPR, atmocn%MLHC, mdwnimp, edwnimp
  close(12)
end program drv
subroutine INC_AJ(i, j, it, jx, v)
  integer :: i, j, it, jx
  real*8 :: v
end subroutine INC_AJ
subroutine INC_AREG(i, j, jr, jx, v)
  integer :: i, j, jr, jx
  real*8 :: v
end subroutine INC_AREG
subroutine PRINTLK(c)
  character(*) :: c
end subroutine PRINTLK
subroutine stop_model(c, n)
  character(*) :: c
  integer :: n
  print *, 'STOP_MODEL ', c, n
  stop 1
end subroutine stop_model
subroutine RESET_SURF_FLUXES(i, j, ito, itn, fo, fn)
  integer :: i, j, ito, itn
  real*8 :: fo, fn
  write(11,'(4i6,2es26.17)') i, j, ito, itn, fo, fn
end subroutine RESET_SURF_FLUXES
