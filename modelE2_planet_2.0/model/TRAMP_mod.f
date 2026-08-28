#include "rundeck_opts.h"
      MODULE AMP_AEROSOL
!@sum Driver for Aerosol Microphysics
!@auth Susanne Bauer
#ifdef TRACERS_AMP
      USE AERO_CONFIG, ONLY: NMODES
      USE AERO_PARAM,  ONLY: NEMIS_SPCS
#endif
      USE RESOLUTION,   ONLY: LM
#ifdef TRACERS_AMP_M9
      use TRACERS_VBS, only: vbs_tracers
#endif  /* TRACERS_AMP_M9 */

      IMPLICIT NONE
      SAVE
C**************  Latitude-Dependant (allocatable) *******************
      ! Mie lookup tables
      REAL*8, DIMENSION(15,17,23,6)      :: AMP_EXT, AMP_ASY, AMP_SCA   !(15,17,23,6) (RE,IM,size,lambda)
      REAL*8, DIMENSION(15,17,23)        :: AMP_Q55
      REAL*8, DIMENSION(23,26,26,26,6)   :: AMP_EXT_CS,AMP_ASY_CS,AMP_SCA_CS !(23,26,26,26,6) (radius,OC,SO4,H2O,lambda)
      REAL*8, DIMENSION(23,26,26,26)     :: AMP_Q55_CS
      INTEGER                            :: LWaerosolCalcs = 1 ! 0 = LW aerosol optics calcs made at 1st time step only
c                                                                1 = all timesteps (rad. response to aerosol size changes)
      INTEGER                            :: separate_h2so4p = 1 ! 0 = all SO4 is amm sulf for optics calcs
c                                                                 1 = separate amm sulf from h2so4 for optics calcs
c                                                                     (in OMA, requires TRACERS_NITRATE is enabled)
#ifdef TRACERS_AMP
      REAL*8, DIMENSION(LM,nmodes)       :: Reff_LEV, NUMB_LEV
      REAL*8, DIMENSION(LM,nmodes)       :: MIX_OC, MIX_SU, MIX_AQ
      COMPLEX*8, DIMENSION(LM,nmodes,6)  :: RindexAMP
      REAL*8, DIMENSION(LM,nmodes,7)     :: dry_Vf_LEV
      INTEGER                            :: AMP_RAD_KEY = 1 ! 1=Volume Mixing || 2=Core - Shell || 3=Maxwell Garnett

      REAL*8, ALLOCATABLE, DIMENSION(:,:,:)       :: AQsulfRATE !(l,i,j)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:)     :: DIAM       ![m](i,j,l,nmodes)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:)     :: DIAM_dry   ![m](i,j,l,nmodes)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:)     :: NACTV      != 1.0D-30  ![#/m^3](i,j,l,nmodes)
      REAL*8, ALLOCATABLE, DIMENSION(:,:)         :: ampPM2p5, ampPM10  ! [kg/kg air]
#endif
#ifdef OMA_TRAMPRAD
      REAL*8, DIMENSION(LM,150)       :: Reff_LEV, NUMB_LEV
      REAL*8, DIMENSION(LM,150)       :: MIX_OC, MIX_SU, MIX_AQ
      COMPLEX*8, DIMENSION(LM,150,6)  :: RindexAMP
      REAL*8, DIMENSION(LM,150,7)     :: dry_Vf_LEV
#endif

!-------------------------------------------------------------------------------------------------------------------------
!-------------------------------------------------------------------------------------------------------------------------
!     The array NACTV(X,Y,Z,I) contains current values of the number of aerosol particles 
!     activated in clouds for each mode I for use outside of the MATRIX microphysical module.
!-------------------------------------------------------------------------------------------------------------------------
!-------------------------------------------------------------------------------------------------------------------------      
!     1 - BC  2-BCmix 3-OC 4-OCmix 5-SS1  6-SS2 7-D1 8-D2
!-------------------------------------------------------------------------------------------------------------------------      
!-------------------------------------------------------------------------------------------------------------------------
!     The array DIAM(x,y,z,n) contains current values of some measure of average ambient mode diameter for each mode  
!       for use outside of the MATRIX microphysical module where it is calculated. 
!
!     The current measure of particle diameter is the diameter of average mass:
! 
!        DIAM(x,y,z,n) = [ (6/pi) * (Mi/Ni) * (1/D) ]^(1/3)
!
!     with Mi the total mass concentration (including water) in mode i, Ni the number concentration in mode i, and
!     D a constant ambient particle density, currently set to D = DENSP = 1.4 g/cm^3. 
!-------------------------------------------------------------------------------------------------------------------------      

#ifdef TRACERS_AMP_M9
      integer, parameter :: vbs_sets=nmodes
      type(vbs_tracers), dimension(vbs_sets) :: vbs_conc ! only used for storing tracer indices
#endif  /* TRACERS_AMP_M9 */

      END MODULE AMP_AEROSOL

