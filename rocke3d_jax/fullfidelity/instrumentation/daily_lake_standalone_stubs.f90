! D205: stub modules for compiling the REAL daily_LAKE (LAKES.f:2492-3010, extracted verbatim) standalone, no water tracers, no SCM, no IRRIGATION_ON.
! Values of the parameters as in LAKES.f:60-105 and decks/P2SAoM40.R (variable_lk=1); only the names USEd by daily_LAKE are provided.
module CONSTANT
  implicit none
  real*8, parameter :: rhow = 1d3, by3 = 1.d0/3d0, pi = 3.1415926535897932d0, lhm = 3.34d5, shi = 2060., shw = 4185., teeny = 1.d-30, tf = 273.15d0
end module CONSTANT
module RESOLUTION
  implicit none
  integer, parameter :: im = 72, jm = 46
end module RESOLUTION
module LAKES
  implicit none
  real*8, parameter :: minmld = 1.d0
  integer :: variable_lk = 1
  real*8 :: lake_ice_max = 5.d0
  integer :: Power_law_lakes = 0
  real*8, parameter :: C_lake = 0.235d0, E_lake = 1.204d0
  integer :: small_lake_evap = 0
end module LAKES
module LAKES_COM
  use RESOLUTION, only : im, jm
  implicit none
  real*8, dimension(im,jm) :: mwl, flake, tanlk, mldlk, tlake, gml, svflake, dlake0, dlake, glake
end module LAKES_COM
module SEAICE_COM
  use RESOLUTION, only : im, jm
  implicit none
  type t_si
    real*8, dimension(:,:), pointer :: RSI, MSI, SNOWI
    real*8, dimension(:,:,:), pointer :: HSI
  end type t_si
  type(t_si) :: si_atm
end module SEAICE_COM
module SEAICE
  implicit none
  real*8, parameter :: ace1i = .1d0*916.6d0, ac2oim = .1d0*916.6d0
  real*8, parameter, dimension(4) :: xsi = (/0.5d0, 0.5d0, 0.5d0, 0.5d0/)
end module SEAICE
module GEOM
  use RESOLUTION, only : im, jm
  implicit none
  real*8, dimension(im,jm) :: axyp, byaxyp
  integer, dimension(jm) :: imaxj
end module GEOM
module GHY_COM
  use RESOLUTION, only : im, jm
  implicit none
  real*8, dimension(im,jm) :: fearth
end module GHY_COM
module FLUXES
  use RESOLUTION, only : im, jm
  implicit none
  type t_atmice
    integer :: J_imelt = 1, J_hmelt = 2
  end type t_atmice
  type t_atmocn
    real*8, dimension(:,:), pointer :: GTEMP, GTEMPR, MLHC
  end type t_atmocn
  type(t_atmice) :: atmice
  type(t_atmocn) :: atmocn
  real*8, dimension(im,jm) :: dmwldf, dgml, fland, flice, focean
end module FLUXES
module LANDICE_COM
  use RESOLUTION, only : im, jm
  implicit none
  real*8, dimension(im,jm) :: mdwnimp, edwnimp
end module LANDICE_COM
module DIAG_COM
  use RESOLUTION, only : im, jm
  implicit none
  integer, parameter :: j_run = 1, j_erun = 2, j_implm = 3, J_IMPLH = 4, itlkice = 1, itlake = 2
  integer, parameter :: IJ_MLKtoGR = 1, IJ_HLKtoGR = 2, IJ_IMPMKI = 3, IJ_IMPHKI = 4
  integer, dimension(im,jm) :: jreg
  real*8, dimension(im,jm,4), target :: AIJ_LOC
end module DIAG_COM
module DOMAIN_DECOMP_ATM
  use RESOLUTION, only : im, jm
  implicit none
  type t_grid
    integer :: I_STRT = 1, I_STOP = im, J_STRT = 1, J_STOP = jm
  end type t_grid
  type(t_grid) :: GRID
contains
  subroutine getDomainBounds(g, J_STRT, J_STOP)
    type(t_grid) :: g
    integer, intent(out) :: J_STRT, J_STOP
    J_STRT = g%J_STRT
    J_STOP = g%J_STOP
  end subroutine getDomainBounds
  subroutine HALO_UPDATE(g, a)
    type(t_grid) :: g
    real*8 :: a(:,:)
  end subroutine HALO_UPDATE
end module DOMAIN_DECOMP_ATM
module model_com
  implicit none
  type t_cal
  contains
    procedure :: getSecondsPerDay
  end type t_cal
  type(t_cal) :: calendar
contains
  function getSecondsPerDay(this) result(s)
    class(t_cal) :: this
    integer :: s
    s = 86400
  end function getSecondsPerDay
end module model_com
module Rational_mod
end module Rational_mod
