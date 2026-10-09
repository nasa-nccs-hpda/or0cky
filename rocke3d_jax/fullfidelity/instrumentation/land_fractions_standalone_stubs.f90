! D208: stub modules for compiling the REAL update_land_fractions (GHY_DRV.f:4367-4645) and get_fb_fv (GHY_DRV.f:4881-4896), extracted verbatim, standalone.
! No water tracers (TRACERS_WATER undefined as in the P2SAoM40 build).  The domain is a list of N cells stored as i = 1..N, j = 1 (cells are independent in the routine).
! Provided only the names USEd by the routine.  snow_cover (radiation snow fraction, atmlnd%fr_snow_rad) is a no-op stub and snow_cover_same_as_rad = 0 (the model
! default, SNOW_DRV.f:9), so the branch that calls it is the one compiled; its output is not part of the land state.
module constant
  implicit none
  real*8, parameter :: twopi = 6.283185307179586d0, rhow = 1d3, shw = 4185.d0
end module constant
module TimeConstants_mod
  implicit none
  real*8, parameter :: EARTH_DAYS_PER_YEAR = 365.d0
end module TimeConstants_mod
module ghy_com
  implicit none
  integer, parameter :: ngm = 6, imt = 5
  real*8, allocatable :: dz_ij(:,:,:), q_ij(:,:,:,:), w_ij(:,:,:,:), ht_ij(:,:,:,:), fr_snow_ij(:,:,:), fearth(:,:), snowbv(:,:,:), top_dev_ij(:,:)
end module ghy_com
module LAKES_COM
  implicit none
  real*8, allocatable :: flake(:,:), svflake(:,:)
end module LAKES_COM
module sle001
  implicit none
  real*8 :: thm(0:64,4)
end module sle001
module fluxes
  implicit none
  type t_atmlnd
    real*8, allocatable :: fr_snow_rad(:,:,:)
  end type t_atmlnd
  type(t_atmlnd) :: atmlnd
  real*8, allocatable :: focean(:,:), DMWLDF(:,:), DGML(:,:)
end module fluxes
module GEOM
  implicit none
  real*8, allocatable :: BYAXYP(:,:)
end module GEOM
module DOMAIN_DECOMP_ATM
  implicit none
  type t_grid
    integer :: I_STRT_HALO, I_STOP_HALO, J_STRT_HALO, J_STOP_HALO
  end type t_grid
  type(t_grid) :: GRID
contains
  subroutine getDomainBounds(g, I_STRT, I_STOP, J_STRT, J_STOP)
    type(t_grid) :: g
    integer, intent(out) :: I_STRT, I_STOP, J_STRT, J_STOP
    I_STRT = g%I_STRT_HALO; I_STOP = g%I_STOP_HALO; J_STRT = g%J_STRT_HALO; J_STOP = g%J_STOP_HALO
  end subroutine getDomainBounds
end module DOMAIN_DECOMP_ATM
module soil_drv
contains
  subroutine snow_cover(a, b, c)
    real*8 :: a, b, c
  end subroutine snow_cover
end module soil_drv
module snow_drvm
  implicit none
  integer :: snow_cover_same_as_rad = 0
end module snow_drvm
module ent_com
  implicit none
  real*8, allocatable :: entcells(:,:)
end module ent_com
module ent_mod
  implicit none
  real*8, allocatable :: fv_store(:,:)
contains
  subroutine ent_get_exports(cell, fraction_of_vegetated_soil)
    real*8 :: cell
    real*8, intent(out) :: fraction_of_vegetated_soil
    fraction_of_vegetated_soil = cell        ! entcells(i,j) carries the Ent vegetated fraction of the cell
  end subroutine ent_get_exports
end module ent_mod
subroutine stop_model(c, n)
  character(*) :: c
  integer :: n
  print *, 'STOP_MODEL ', c, n
  stop 1
end subroutine stop_model
subroutine set_new_ghy_cells_outputs
  use ghy_com
  use LAKES_COM
  ! stub: the real routine (GHY_DRV.f:4648) only sets output diagnostics; the call count is not needed, the Python port lists the rows
end subroutine set_new_ghy_cells_outputs
