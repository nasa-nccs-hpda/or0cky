! D103-D106 standalone harness for the REAL QUS3D.f (module TRACER_ADV: AADVQ0, AADVQ, XSTEP, ZSTEP, aadvqx/y/z,
! checkflux, aadvqz_column).  Only the infrastructure modules that QUS3D.f uses are replaced by serial stubs
! (single rank: both poles local, halo updates are no-ops).  QUS3D.f and QUSDEF (lines 1-41 of QUSDEF.f) are the
! unmodified real sources.  Build recipe: see instrumentation/build_and_run.md (D103-D106 section).
module resolution
  implicit none
  integer, parameter :: im=72, jm=46, lm=40
end module resolution

module quscom
  implicit none
  integer, parameter :: im=72, jm=46, lm=40
  real*8, parameter :: byim = 1d0/72d0
end module quscom

module geom
  implicit none
  integer, parameter :: imxx=72, jmxx=46
  real*8, parameter :: fim = 72d0, byim = 1./fim
  integer :: imaxj(jmxx)
end module geom

module domain_decomp_1d
  implicit none
  integer, parameter :: north=2, south=1
  type dist_grid
    integer :: j_strt=1, j_stop=46, j_strt_halo=1, j_stop_halo=46
    integer :: j_strt_skp=2, j_stop_skp=45, j_strt_stgr=2, j_stop_stgr=46
  end type dist_grid
  interface halo_update
    module procedure hu2, hu3
  end interface
  interface halo_update_column
    module procedure huc3, huc4
  end interface
  interface globalsum
    module procedure gs_i
  end interface
  interface globalmax
    module procedure gm_i1
  end interface
contains
  subroutine hu2(g, a, from)
    type(dist_grid) :: g
    real*8 :: a(:,:)
    integer, optional :: from
  end subroutine hu2
  subroutine hu3(g, a, from)
    type(dist_grid) :: g
    real*8 :: a(:,:,:)
    integer, optional :: from
  end subroutine hu3
  subroutine huc3(g, a, from)
    type(dist_grid) :: g
    real*8 :: a(:,:,:)
    integer, optional :: from
  end subroutine huc3
  subroutine huc4(g, a, from)
    type(dist_grid) :: g
    real*8 :: a(:,:,:,:)
    integer, optional :: from
  end subroutine huc4
  subroutine halo_update_mask(g, sbufs, sbufn, rbufs, rbufn)
    type(dist_grid) :: g
    real*8 :: sbufs(:), sbufn(:), rbufs(:), rbufn(:)
  end subroutine halo_update_mask
  subroutine gs_i(g, loc, glob, all)
    type(dist_grid) :: g
    integer :: loc, glob
    logical, optional :: all
    glob = loc
  end subroutine gs_i
  subroutine gm_i1(g, loc, glob)
    type(dist_grid) :: g
    integer :: loc(:), glob(:)
    glob = loc
  end subroutine gm_i1
  logical function am_i_root()
    am_i_root = .true.
  end function am_i_root
  subroutine getdomainbounds(g, j_strt, j_stop, j_strt_stgr, j_stop_stgr, j_strt_skp, j_stop_skp, &
       j_strt_halo, j_stop_halo, have_south_pole, have_north_pole)
    type(dist_grid) :: g
    integer, optional :: j_strt, j_stop, j_strt_stgr, j_stop_stgr, j_strt_skp, j_stop_skp, j_strt_halo, j_stop_halo
    logical, optional :: have_south_pole, have_north_pole
    if (present(j_strt)) j_strt = g%j_strt
    if (present(j_stop)) j_stop = g%j_stop
    if (present(j_strt_stgr)) j_strt_stgr = g%j_strt_stgr
    if (present(j_stop_stgr)) j_stop_stgr = g%j_stop_stgr
    if (present(j_strt_skp)) j_strt_skp = g%j_strt_skp
    if (present(j_stop_skp)) j_stop_skp = g%j_stop_skp
    if (present(j_strt_halo)) j_strt_halo = g%j_strt_halo
    if (present(j_stop_halo)) j_stop_halo = g%j_stop_halo
    if (present(have_south_pole)) have_south_pole = .true.
    if (present(have_north_pole)) have_north_pole = .true.
  end subroutine getdomainbounds
end module domain_decomp_1d

module domain_decomp_atm
  use domain_decomp_1d, only: dist_grid, getdomainbounds
  implicit none
  type(dist_grid), save :: grid
end module domain_decomp_atm

module atm_com
  implicit none
  real*8, allocatable, dimension(:,:,:) :: mus, mvs, mws, mb, mma
end module atm_com

subroutine stop_model(msg, code)
  character(*) :: msg
  integer :: code
  write(6,*) 'STOP_MODEL: ', msg
  open(99, file='stop.txt', status='replace')
  write(99,*) msg
  close(99)
  stop 3
end subroutine stop_model
