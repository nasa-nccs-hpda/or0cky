#include "rundeck_opts.h"
module VerticalRes
!@sum Vertical Resolution file, 40 layers, top at .1 mb
!@+   but recoded to be able to import PLANET_PARAMS info
!@auth Original Development Team
  use constant, only : grav,mb2kg
#ifdef PLANET_PARAMS
  use PlanetParams_mod, only : PlanetParams
#endif
  Implicit None
!@var LM    = number of dynamical layers
!@var LS1   = lowest layer of strtosphere
  Integer*4,Parameter :: LM=40

!@var PSF,PMTOP global mean surface, model top pressure  (mb)
!@var PTOP pressure at interface level sigma/const press coord syst (mb)
!@var PSFMPT,PSTRAT pressure due to troposhere,stratosphere
!@var PLbot pressure levels at bottom of layers (mb)

#ifdef PLANET_PARAMS
  real*8, parameter :: PSF = PlanetParams%psf
#else
  real*8, parameter :: PSF = 984d0
#endif
  real*8, parameter, private :: pratio = PSF/984d0

  real*8, parameter :: &
    PLBOT(1:LM+1) = pratio*(/ &
      984d0, 964d0, 942d0, 917d0, 890d0, 860d0, 825d0, &  !  L=1,..
      785d0, 740d0, 692d0, 642d0, 591d0, 539d0, 489d0, &  !  L=...
      441d0, 396d0, 354d0, 316d0, 282d0, 251d0, 223d0, &
      197d0, 173d0,                                    &
      150d0,                                           &  !  L=LS1
      128d0, 108d0,  90d0,  73d0,  57d0,  43d0,  31d0, &  !  L=...
      20d0,  10d0,5.62d0,3.16d0,1.78d0,  1.d0,         &
      .562d0,.316d0,.178d0, .1d0 /)                       !  L=..,LM+1

  integer, private :: iii ! iterator

!@var delp nominal pressure thicknesses of layers (mb)
  real*8, dimension(lm), parameter, private :: delp = plbot(1:lm)-plbot(2:lm+1)

#ifndef PLANET_PARAMS
  integer, parameter :: ls1=24
#else
! Calculate ls1 from PlanetParams%ptop using allowed compile-time operations.
! This clunky coding will be enthusiastically removed once LS1 disappears from
! the model
  real*8, parameter, private :: ptop0 = PlanetParams%ptop
  integer, dimension(lm), parameter, private :: lev=(/ (iii,iii=1,lm) /)
! factor 1d6 is simply to make all pdiff magnitudes greater than 1
  real*8, parameter, dimension(lm+1), private :: pdiff = 1d6*(plbot-ptop0)
  integer, parameter, dimension(lm), private :: &
                                ! lprod is nonzero if ptop0 lies between two elements of plbot
    lprod = lev*int(min(1d0,max(0d0,-pdiff(1:lm)*pdiff(2:lm+1)))), &
                                ! ldiff is nonzero if ptop0 is exactly equal to an element of plbot
    ldiff = lev*(1-int(min(1d0,abs(pdiff(1:lm)))))
! summation to get the smaller candiate for ls1
  integer, parameter, private :: ls1_lower = &
    lprod( 1)+lprod( 2)+lprod( 3)+lprod( 4)+lprod( 5) &
    +lprod( 6)+lprod( 7)+lprod( 8)+lprod( 9)+lprod(10) &
    +lprod(11)+lprod(12)+lprod(13)+lprod(14)+lprod(15) &
    +lprod(16)+lprod(17)+lprod(18)+lprod(19)+lprod(20) &
    +lprod(21)+lprod(22)+lprod(23)+lprod(24)+lprod(25) &
    +lprod(26)+lprod(27)+lprod(28)+lprod(29)+lprod(30) &
    +lprod(31)+lprod(32)+lprod(33)+lprod(34)+lprod(35) &
    +lprod(36)+lprod(37)+lprod(38)+lprod(39)+lprod(40) &
                                !
    +ldiff( 1)+ldiff( 2)+ldiff( 3)+ldiff( 4)+ldiff( 5) &
    +ldiff( 6)+ldiff( 7)+ldiff( 8)+ldiff( 9)+ldiff(10) &
    +ldiff(11)+ldiff(12)+ldiff(13)+ldiff(14)+ldiff(15) &
    +ldiff(16)+ldiff(17)+ldiff(18)+ldiff(19)+ldiff(20) &
    +ldiff(21)+ldiff(22)+ldiff(23)+ldiff(24)+ldiff(25) &
    +ldiff(26)+ldiff(27)+ldiff(28)+ldiff(29)+ldiff(30) &
    +ldiff(31)+ldiff(32)+ldiff(33)+ldiff(34)+ldiff(35) &
    +ldiff(36)+ldiff(37)+ldiff(38)+ldiff(39)+ldiff(40)

! choose ls1 by rounding ptop to the closest element of plbot
  integer, parameter, private :: &
    wt = 1-nint((plbot(ls1_lower)-ptop0)/delp(ls1_lower))
  integer, parameter :: ls1 = wt*ls1_lower + (1-wt)*(ls1_lower+1)
#endif

  integer, parameter :: ls1_nominal=ls1

  Real*8,Parameter :: &
    PTOP = plbot(ls1), &
    PMTOP = plbot(lm+1), &
    PSFMPT = PSF-PTOP, &
    PSTRAT = PTOP-PMTOP

!@var tropomask unity from layers 1-ls1-1, zero above
!@var stratmask zero from layers 1-ls1-1, unity above
  real*8, dimension(lm), parameter, private :: &
    tropomask = (/ (1d0, iii=1,ls1-1), (0d0, iii=ls1,lm) /), &
    stratmask = 1d0-tropomask

!@var MDRYA = dry atmospheric mass (kg/m^2) = 100*PSF/GRAV
!@var MTOP  = mass above dynamical top (kg/m^2) = 100*PMTOP/GRAV
!@var MFIXs = summation of MFIX (kg/m^2) = 100*(PTOP-PMTOP)/GRAV
!@var MVAR  = spatially and temporally varying column mass (kg/m^2)
!@var MSURF = MTOP + MFIXs + MVAR (kg/m^2)
!@var AM(L) = MFIX(L) + MVAR*MFRAC(L) (kg/m^2)
!@var MFIX(L)  = fixed mass in each layer (kg/m^2) = 100*[PLBOT(L)-PLBOT(L+1)]/GRAV
!@var MFRAC(L) = fraction of variable mass in each layer = DSIG(L)
  Real*8,Parameter :: MDRYA = psf*mb2kg, MTOP = pmtop*mb2kg, MFIXs = pstrat*mb2kg, &
    MFIX(LM)  = delp*stratmask*mb2kg, &
    MFRAC(LM) = delp*tropomask/psfmpt

End Module VerticalRes
