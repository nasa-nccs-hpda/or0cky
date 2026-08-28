#include "rundeck_opts.h"
      MODULE OFFLINE_AEROSOL

      USE RESOLUTION,   ONLY: LM

      IMPLICIT NONE
      SAVE

      INTEGER, PARAMETER :: NMODES=16
      integer :: I_0,I_1,J_0,J_1
C**************  Latitude-Dependant (allocatable) *******************
      ! Mie lookup tables
      REAL*8, DIMENSION(15,17,23,6)    :: AMP_EXT, AMP_ASY, AMP_SCA   !(15,17,23,6) (RE,IM,size,lambda)
      REAL*8, DIMENSION(15,17,23)      :: AMP_Q55
      REAL*8, DIMENSION(23,26,26,26,6) :: AMP_EXT_CS, AMP_ASY_CS, AMP_SCA_CS !(23,26,26,26,6) (radius,OC,SO4,H2O,lambda)
      REAL*8, DIMENSION(23,26,26,26)   :: AMP_Q55_CS
      ! 1 Dim arrays for Radiation
      REAL*8, DIMENSION(LM,nmodes)     :: Reff_LEV, NUMB_LEV
      REAL*8, DIMENSION(LM,nmodes)     :: MIX_OC, MIX_SU, MIX_AQ
      COMPLEX*8, DIMENSION(LM,nmodes,6):: RindexAMP
      REAL*8, DIMENSION(LM,nmodes,7)   :: dry_Vf_LEV
      ! FALSE : one Radiation call
      ! TRUE  : nmodes Radiation calls
      INTEGER                            :: AMP_RAD_KEY = 1 ! 1=Volume Mixing || 2=Core - Shell || 3=Maxwell Garnett


      REAL*8, DIMENSION(144,90,LM,nmodes):: DIAM       ![m](i,j,l,nmodes)
c c     REAL*8, ALLOCATABLE, DIMENSION(I_0:I_1,J_0:J_1,LM,nmodes):: DIAM_dry   ![m](i,j,l,nmodes)

      END MODULE OFFLINE_AEROSOL

