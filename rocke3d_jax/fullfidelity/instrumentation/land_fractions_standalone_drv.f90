! D208: driver of the standalone REAL update_land_fractions: reads in.bin (big-endian f8 stream), calls it once, writes out.bin.
! in : n, thm(0,1:4), then (cells i=1..n, j=1): w_ij(0:6,3,n), ht_ij(0:6,3,n), fr_snow_ij(2,n), dz_ij(n,6), q_ij(n,5,6), fv(n), fearth(n), flake(n), svflake(n),
!      focean(n), DMWLDF(n), DGML(n), BYAXYP(n)
! out: w_ij, ht_ij, fr_snow_ij
program drv
  use ghy_com
  use LAKES_COM
  use sle001
  use fluxes
  use GEOM
  use ent_com
  use DOMAIN_DECOMP_ATM
  implicit none
  real*8 :: nn
  integer :: n
  open(10, file='in.bin', form='unformatted', access='stream', convert='big_endian')
  read(10) nn
  n = nint(nn)
  GRID%I_STRT_HALO = 1; GRID%I_STOP_HALO = n; GRID%J_STRT_HALO = 1; GRID%J_STOP_HALO = 1
  allocate(w_ij(0:ngm,3,n,1), ht_ij(0:ngm,3,n,1), fr_snow_ij(2,n,1), dz_ij(n,1,ngm), q_ij(n,1,imt,ngm), fearth(n,1), snowbv(2,n,1), top_dev_ij(n,1))
  allocate(flake(n,1), svflake(n,1), focean(n,1), DMWLDF(n,1), DGML(n,1), BYAXYP(n,1), entcells(n,1))
  allocate(atmlnd%fr_snow_rad(2,n,1))
  thm = 0d0; snowbv = 0d0; top_dev_ij = 0d0
  read(10) thm(0,1:4)
  read(10) w_ij, ht_ij, fr_snow_ij, dz_ij, q_ij, entcells, fearth, flake, svflake, focean, DMWLDF, DGML, BYAXYP
  close(10)
  call update_land_fractions
  open(12, file='out.bin', form='unformatted', access='stream', convert='big_endian')
  write(12) w_ij, ht_ij, fr_snow_ij
  close(12)
end program drv
