#include "rundeck_opts.h"

      MODULE AERO_CONFIG
      IMPLICIT NONE
!-------------------------------------------------------------------------------------------------------------------------
!
!     MATRIX CONFIGURATION MODULE.
!
!-------------------------------------------------------------------------------------------------------------------------
#ifdef TRACERS_AMP_M9
      INTEGER, PARAMETER :: NGASES     = 13     ! number of gas-phase species
#else
      INTEGER, PARAMETER :: NGASES     = 3      ! number of gas-phase species
#endif  /* TRACERS_AMP_M9 */
      INTEGER, PARAMETER :: GAS_H2SO4  = 1      !-
      INTEGER, PARAMETER :: GAS_HNO3   = 2      !-indices in the GAS array
      INTEGER, PARAMETER :: GAS_NH3    = 3      !-
#ifdef TRACERS_AMP_M9
      INTEGER, PARAMETER :: GAS_OCM2 = 4        !-
      INTEGER, PARAMETER :: GAS_OCM1 = 5        !-
      INTEGER, PARAMETER :: GAS_OCM0 = 6        !-
      INTEGER, PARAMETER :: GAS_OCP1 = 7        !-
      INTEGER, PARAMETER :: GAS_OCP2 = 8        !-
      INTEGER, PARAMETER :: GAS_OCP3 = 9        !-
      INTEGER, PARAMETER :: GAS_OCP4 = 10       !-
      INTEGER, PARAMETER :: GAS_OCP5 = 11       !-
      INTEGER, PARAMETER :: GAS_OCP6 = 12       !-
      INTEGER, PARAMETER :: GAS_OH   = 13       !-indices in the GAS array
#endif  /* TRACERS_AMP_M9 */
      INTEGER, PARAMETER :: PROD_INDEX_SULF = 1 ! SULF index in PROD_INDEX(:,:)
#ifndef TRACERS_AMP_M11 /* not */
      INTEGER, PARAMETER :: PROD_INDEX_BCAR = 2 ! BCAR index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCAR = 3 ! OCAR index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_DUST = 4 ! DUST index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_SEAS = 5 ! SEAS index in PROD_INDEX(:,:)
#ifdef TRACERS_AMP_M9
      INTEGER, PARAMETER :: PROD_INDEX_OCM2 = 6 ! OCM2 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCM1 = 7 ! OCM1 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCM0 = 8 ! OCM0 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCP1 = 9 ! OCP1 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCP2 = 10! OCP2 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCP3 = 11! OCP3 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCP4 = 12! OCP4 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCP5 = 13! OCP5 index in PROD_INDEX(:,:)
      INTEGER, PARAMETER :: PROD_INDEX_OCP6 = 14! OCP6 index in PROD_INDEX(:,:)
#endif  /* TRACERS_AMP_M9 */
#endif  /* not TRACERS_AMP_M11 */
!-------------------------------------------------------------------------------------------------------------------------
!@param NMODES_MAX Maximum number of modes possible
!@param MNAME Aerosol mode names (and numbers) that might appear in one or more
!@+           mechanisms. These mode number only pertain to this set of all
!@+           possible modes, and are not the mode numbers used for any specific
!@+           mechanism.
      INTEGER, PARAMETER :: NMODES_MAX=18 
      CHARACTER(LEN=3), PARAMETER, DIMENSION(NMODES_MAX) :: &
        MNAME=(/'AKK','ACC','DD1','DS1','DD2','DS2','SSA','SSC','SSS', &
                'OCC','BC1','BC2','BC3','OCS','DBC','BOC','BCS','MXX'/)
      ! Mode #    1     2     3     4     5     6     7     8     9    ! # for IMODES below
      ! Mode #   10    11    12    13    14    15    16    17    18    ! # for IMODES below
!-------------------------------------------------------------------------------------------------------------------------
!@param NEXTRA_MAX Maximum number of mass tracers that belong to no mode.
!@param NMASS_SPCS_MAX Maximum number of mass tracers per mode.
!@param CHEM_SPC_NAME_EXTRA_ALL Names of all possible species outside modes.
!@param CHEM_SPC_NAME_ALL Names of all possible species inside modes.
!@var MSPCS Map of species that can be present per mode (NMASS_SPCS_MAX,NMODES_MAX).
!@+         =0 species not present in mode, =1 species present in mode.
      INTEGER, PARAMETER :: NEXTRA_MAX=3
      CHARACTER(LEN=3), PARAMETER, DIMENSION(NEXTRA_MAX) :: &
        CHEM_SPC_NAME_EXTRA_ALL=(/'NO3','NH4','H2O'/)
      INTEGER, PARAMETER :: NMASS_SPCS_MAX=14
      CHARACTER(LEN=4), PARAMETER, DIMENSION(NMASS_SPCS_MAX) :: &
        CHEM_SPC_NAME_ALL=(/'SULF','BCAR','OCAR','DUST','SEAS','OCM2','OCM1', &
                            'OCM0','OCP1','OCP2','OCP3','OCP4','OCP5','OCP6'/)
      INTEGER, SAVE :: MSPCS(NMASS_SPCS_MAX,NMODES_MAX)
      DATA MSPCS(PROD_INDEX_SULF,1:NMODES_MAX)/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/
#ifndef TRACERS_AMP_M11 /* not */
      DATA MSPCS(PROD_INDEX_BCAR,1:NMODES_MAX)/0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1/
      DATA MSPCS(PROD_INDEX_OCAR,1:NMODES_MAX)/0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1/
      DATA MSPCS(PROD_INDEX_DUST,1:NMODES_MAX)/0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,1/
      DATA MSPCS(PROD_INDEX_SEAS,1:NMODES_MAX)/0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1/
#ifdef TRACERS_AMP_M9
      DATA MSPCS(PROD_INDEX_OCM2,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCM1,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCM0,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCP1,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCP2,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCP3,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCP4,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCP5,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
      DATA MSPCS(PROD_INDEX_OCP6,1:NMODES_MAX)/0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1/
#endif  /* TRACERS_AMP_M9 */
#endif  /* not TRACERS_AMP_M11 */
!-------------------------------------------------------------------------------------------------------------------------
!@param MECH MATRIX mechanism selected.
!@param NAEROVARS Number of aerosol tracers in MATRIX with microphysics.
!@param NEXTRA Number of aerosol tracers in MATRIX without microphysics.
!@param NMODES Number of modes active in selected mechanism.
!@param NMASS_SPCS Number of mass species active in selected mechanism.
!@param IMODES Indices of modes currently active, selected from MNAME.
#ifdef TRACERS_AMP_M1
      INTEGER, PARAMETER :: MECH=1,NAEROVARS=51,NEXTRA=3,NMODES=16,NMASS_SPCS=5 ! Mechanism 1
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 5, 6, 7, 8,10,11,12,13,15,16,17,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M2
      INTEGER, PARAMETER :: MECH=2,NAEROVARS=51,NEXTRA=3,NMODES=16,NMASS_SPCS=5 ! Mechanism 2
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 5, 6, 7, 8,10,11,12,14,15,16,17,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M3
      INTEGER, PARAMETER :: MECH=3,NAEROVARS=41,NEXTRA=3,NMODES=13,NMASS_SPCS=5 ! Mechanism 3  
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 5, 6, 7, 8,10,11,12,16,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M4
      INTEGER, PARAMETER :: MECH=4,NAEROVARS=34,NEXTRA=1,NMODES=10,NMASS_SPCS=5 ! Mechanism 4 
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 2, 3, 4, 5, 6, 9,10,11,12,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M5
      INTEGER, PARAMETER :: MECH=5,NAEROVARS=45,NEXTRA=3,NMODES=14,NMASS_SPCS=5 ! Mechanism 5
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 7, 8,10,11,12,13,15,16,17,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M6
      INTEGER, PARAMETER :: MECH=6,NAEROVARS=45,NEXTRA=3,NMODES=14,NMASS_SPCS=5 ! Mechanism 6 
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 7, 8,10,11,12,14,15,16,17,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M7
      INTEGER, PARAMETER :: MECH=7,NAEROVARS=35,NEXTRA=3,NMODES=11,NMASS_SPCS=5 ! Mechanism 7  
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 7, 8,10,11,12,16,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M8
      INTEGER, PARAMETER :: MECH=8,NAEROVARS=28,NEXTRA=1,NMODES= 8,NMASS_SPCS=5 ! Mechanism 8 
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 2, 3, 4, 9,10,11,12,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M9
      INTEGER, PARAMETER :: MECH=9,NAEROVARS=173,NEXTRA=3,NMODES=15,NMASS_SPCS=14 ! Mechanism 9
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 5, 6, 7, 8,10,11,12,14,16,17,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M10
      INTEGER, PARAMETER :: MECH=10,NAEROVARS=47,NEXTRA=3,NMODES=15,NMASS_SPCS=5 ! Mechanism 10
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2, 3, 4, 5, 6, 7, 8,10,11,12,14,16,17,18/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1, 2, 3, 4, 5/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 1, 2, 3/)
#elif defined TRACERS_AMP_M11
      INTEGER, PARAMETER :: MECH=11,NAEROVARS=4,NEXTRA=1,NMODES=2,NMASS_SPCS=1 ! Mechanism 11
      INTEGER, PARAMETER, DIMENSION(NMODES) :: &
        IMODES=(/ 1, 2/)
      INTEGER, PARAMETER, DIMENSION(NMASS_SPCS) :: &
        ISPCS=(/ 1/)
      INTEGER, PARAMETER, DIMENSION(NEXTRA) :: &
        ISPCS_EXTRA=(/ 3/)
#endif
!-------------------------------------------------------------------------------------------------------------------------
!     1. Set the number of quadrature points per mode (1-2); must use NPOINTS=1 for the present.
!-------------------------------------------------------------------------------------------------------------------------
      INTEGER, PARAMETER :: NPOINTS=1
!-------------------------------------------------------------------------------------------------------------------------
!     2. Select modes to undergo condensational growth for the desired mechanism (1-8). (Ignore other mechanisms.)
!        ICONDn(I)=1, condensational growth done; ICONDn(I)=0, condensational growth not done. 
!        Ordinarily, all modes would undergo condenational growth.
!-------------------------------------------------------------------------------------------------------------------------
#ifdef TRACERS_AMP_M1
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/)  ! Mechanism 1
#elif defined TRACERS_AMP_M2
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/)  ! Mechanism 2
#elif defined TRACERS_AMP_M3
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1/)        ! Mechanism 3
#elif defined TRACERS_AMP_M4
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1/)              ! Mechanism 4
#elif defined TRACERS_AMP_M5
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1,1/)      ! Mechanism 5
#elif defined TRACERS_AMP_M6
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1,1/)      ! Mechanism 6
#elif defined TRACERS_AMP_M7
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1/)            ! Mechanism 7
#elif defined TRACERS_AMP_M8
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1/)                  ! Mechanism 8
#elif defined TRACERS_AMP_M9
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/)    ! Mechanism 9
#elif defined TRACERS_AMP_M10
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/)    ! Mechanism 10
#elif defined TRACERS_AMP_M11
      INTEGER, PARAMETER, DIMENSION(NMODES) :: ICOND=(/1,1/)    ! Mechanism 11
#endif
!-------------------------------------------------------------------------------------------------------------------------
!     These require no editing.
!-------------------------------------------------------------------------------------------------------------------------
!@param MODE_NAME Names of modes for current mechanism
!@param CHEM_SPC_NAME Names of mass species for current mechanism
      CHARACTER(LEN=3), PARAMETER, DIMENSION(NMODES)     :: MODE_NAME=MNAME(IMODES)
      CHARACTER(LEN=4), PARAMETER, DIMENSION(NMASS_SPCS) :: CHEM_SPC_NAME=CHEM_SPC_NAME_ALL(ISPCS)
      CHARACTER(LEN=3), PARAMETER, DIMENSION(NEXTRA)     :: CHEM_SPC_NAME_EXTRA=CHEM_SPC_NAME_EXTRA_ALL(ISPCS_EXTRA)
      INTEGER, PARAMETER :: NAEROBOX=NAEROVARS+NEXTRA 
      INTEGER, PARAMETER :: NWEIGHTS=NMODES*NPOINTS   
      INTEGER, PARAMETER :: NBINS = 30
!-------------------------------------------------------------------------------------------------------------------------------------
!     3. Optionally edit the table of coagulation interactions.
!        The donor modes may not be modified. Each receptor mode must contain all species present in either donor mode.
!        Entering 'OFF' for the receptor mode name disables coagulation between the two donor modes.
!-------------------------------------------------------------------------------------------------------------------------------------
#ifdef TRACERS_AMP_M1
!     Mechanism 1.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   DD2   DS2   SSA   SSC   OCC   BC1   BC2   BC3   DBC   BOC   BCS   MXX    SECOND
!                                                                                                                                MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BC1','BC2','BC3','DBC','BOC','BCS','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BCS','BCS','BCS','DBC','BOC','BCS','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','DD2','DD2','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','DD2','DS2','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 5)/'DD2','DD2','DD2','DD2','DD2','DD2','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DD2
      DATA CITABLE(1:NMODES, 6)/'DS2','DS2','DD2','DS2','DD2','DS2','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DS2
      DATA CITABLE(1:NMODES, 7)/'SSA','SSA','MXX','MXX','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 8)/'SSC','SSC','MXX','MXX','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSC
      DATA CITABLE(1:NMODES, 9)/'OCC','OCC','MXX','MXX','MXX','MXX','MXX','MXX','OCC','BOC','BOC','BOC','MXX','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES,10)/'BC1','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BC1','BC1','BC1','DBC','BOC','BCS','MXX'/ ! BC1
      DATA CITABLE(1:NMODES,11)/'BC2','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BC1','BC2','BC2','DBC','BOC','BCS','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,12)/'BC3','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BC1','BC2','BC3','DBC','BOC','BCS','MXX'/ ! BC3
      DATA CITABLE(1:NMODES,13)/'DBC','DBC','DBC','DBC','DBC','DBC','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DBC
      DATA CITABLE(1:NMODES,14)/'BOC','BOC','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','MXX','BOC','BOC','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,15)/'BCS','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BCS','BCS','BCS','DBC','BOC','BCS','MXX'/ ! BCS
      DATA CITABLE(1:NMODES,16)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M2
!
!     Mechanism 2.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   DD2   DS2   SSA   SSC   OCC   BC1   BC2   OCS   DBC   BOC   BCS   MXX    SECOND
!                                                                                                                                MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BC1','BC2','OCS','DBC','BOC','BCS','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCS','BCS','BCS','OCS','DBC','BOC','BCS','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','DD2','DD2','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','DD2','DS2','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 5)/'DD2','DD2','DD2','DD2','DD2','DD2','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DD2
      DATA CITABLE(1:NMODES, 6)/'DS2','DS2','DD2','DS2','DD2','DS2','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DS2
      DATA CITABLE(1:NMODES, 7)/'SSA','SSA','MXX','MXX','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 8)/'SSC','SSC','MXX','MXX','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSC
      DATA CITABLE(1:NMODES, 9)/'OCC','OCS','MXX','MXX','MXX','MXX','MXX','MXX','OCC','BOC','BOC','OCS','MXX','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES,10)/'BC1','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BC1','BC1','BOC','DBC','BOC','BCS','MXX'/ ! BC1
      DATA CITABLE(1:NMODES,11)/'BC2','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BC1','BC2','BOC','DBC','BOC','BCS','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,12)/'OCS','OCS','MXX','MXX','MXX','MXX','MXX','MXX','OCS','BOC','BOC','OCS','MXX','BOC','BOC','MXX'/ ! OCS 
      DATA CITABLE(1:NMODES,13)/'DBC','DBC','DBC','DBC','DBC','DBC','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DBC
      DATA CITABLE(1:NMODES,14)/'BOC','BOC','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','MXX','BOC','BOC','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,15)/'BCS','BCS','DBC','DBC','DBC','DBC','MXX','MXX','BOC','BCS','BCS','BOC','DBC','BOC','BCS','MXX'/ ! BCS
      DATA CITABLE(1:NMODES,16)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M3
!
!     Mechanism 3.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   DD2   DS2   SSA   SSC   OCC   BC1   BC2   BOC   MXX    SECOND
!                                                                                                              MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BC1','BC2','BOC','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BC1','BC2','BOC','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','DD2','DD2','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','DD2','DS2','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 5)/'DD2','DD2','DD2','DD2','DD2','DD2','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DD2
      DATA CITABLE(1:NMODES, 6)/'DS2','DS2','DD2','DS2','DD2','DS2','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DS2
      DATA CITABLE(1:NMODES, 7)/'SSA','SSA','MXX','MXX','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 8)/'SSC','SSC','MXX','MXX','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX'/ ! SSC
      DATA CITABLE(1:NMODES, 9)/'OCC','OCC','MXX','MXX','MXX','MXX','MXX','MXX','OCC','BOC','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES,10)/'BC1','BC1','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BC1','BC1','BOC','MXX'/ ! BC1
      DATA CITABLE(1:NMODES,11)/'BC2','BC2','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BC1','BC2','BOC','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,12)/'BOC','BOC','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,13)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M4
!
!     Mechanism 4.
!
!     FIRST MODE               ACC   DD1   DS1   DD2   DS2   SSS   OCC   BC1   BC2   MXX    SECOND
!                                                                                            MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'ACC','DD1','DS1','DD2','DS2','SSS','OCC','BC1','BC2','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 2)/'DD1','DD1','DD1','DD2','DD2','MXX','MXX','MXX','MXX','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 3)/'DS1','DD1','DS1','DD2','DS2','MXX','MXX','MXX','MXX','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 4)/'DD2','DD2','DD2','DD2','DD2','MXX','MXX','MXX','MXX','MXX'/ ! DD2
      DATA CITABLE(1:NMODES, 5)/'DS2','DD2','DS2','DD2','DS2','MXX','MXX','MXX','MXX','MXX'/ ! DS2
      DATA CITABLE(1:NMODES, 6)/'SSS','MXX','MXX','MXX','MXX','SSS','MXX','MXX','MXX','MXX'/ ! SSS
      DATA CITABLE(1:NMODES, 7)/'OCC','MXX','MXX','MXX','MXX','MXX','OCC','MXX','MXX','MXX'/ ! OCC
      DATA CITABLE(1:NMODES, 8)/'BC1','MXX','MXX','MXX','MXX','MXX','MXX','BC1','BC1','MXX'/ ! BC1
      DATA CITABLE(1:NMODES, 9)/'BC2','MXX','MXX','MXX','MXX','MXX','MXX','BC1','BC2','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,10)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M5
!
!     Mechanism 5.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   SSA   SSC   OCC   BC1   BC2   BC3   DBC   BOC   BCS   MXX    SECOND
!                                                                                                                    MODE
!-------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','SSA','SSC','OCC','BC1','BC2','BC3','DBC','BOC','BCS','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','SSA','SSC','OCC','BCS','BCS','BCS','DBC','BOC','BCS','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 5)/'SSA','SSA','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 6)/'SSC','SSC','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSC
      DATA CITABLE(1:NMODES, 7)/'OCC','OCC','MXX','MXX','MXX','MXX','OCC','BOC','BOC','BOC','MXX','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES, 8)/'BC1','BCS','DBC','DBC','MXX','MXX','BOC','BC1','BC1','BC1','DBC','BOC','BCS','MXX'/ ! BC1
      DATA CITABLE(1:NMODES, 9)/'BC2','BCS','DBC','DBC','MXX','MXX','BOC','BC1','BC2','BC2','DBC','BOC','BCS','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,10)/'BC3','BCS','DBC','DBC','MXX','MXX','BOC','BC1','BC2','BC3','DBC','BOC','BCS','MXX'/ ! BC3
      DATA CITABLE(1:NMODES,11)/'DBC','DBC','DBC','DBC','MXX','MXX','MXX','DBC','DBC','DBC','DBC','MXX','DBC','MXX'/ ! DBC
      DATA CITABLE(1:NMODES,12)/'BOC','BOC','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','MXX','BOC','BOC','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,13)/'BCS','BCS','DBC','DBC','MXX','MXX','BOC','BCS','BCS','BCS','DBC','BOC','BCS','MXX'/ ! BCS
      DATA CITABLE(1:NMODES,14)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M6
!
!     Mechanism 6.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   SSA   SSC   OCC   BC1   BC2   OCS   DBC   BOC   BCS   MXX    SECOND
!                                                                                                                    MODE
!-------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','SSA','SSC','OCC','BC1','BC2','OCS','DBC','BOC','BCS','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','SSA','SSC','OCS','BCS','BCS','OCS','DBC','BOC','BCS','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','DBC','MXX'/ ! DS2
      DATA CITABLE(1:NMODES, 5)/'SSA','SSA','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 6)/'SSC','SSC','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSC
      DATA CITABLE(1:NMODES, 7)/'OCC','OCS','MXX','MXX','MXX','MXX','OCC','BOC','BOC','OCS','MXX','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES, 8)/'BC1','BCS','DBC','DBC','MXX','MXX','BOC','BC1','BC1','BOC','DBC','BOC','BCS','MXX'/ ! BC1
      DATA CITABLE(1:NMODES, 9)/'BC2','BCS','DBC','DBC','MXX','MXX','BOC','BC1','BC2','BOC','DBC','BOC','BCS','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,10)/'OCS','OCS','MXX','MXX','MXX','MXX','OCS','BOC','BOC','OCS','MXX','MXX','MXX','MXX'/ ! OCS 
      DATA CITABLE(1:NMODES,11)/'DBC','DBC','DBC','DBC','MXX','MXX','MXX','DBC','DBC','MXX','DBC','MXX','MXX','MXX'/ ! DBC
      DATA CITABLE(1:NMODES,12)/'BOC','BOC','MXX','MXX','MXX','MXX','BOC','BOC','BOC','MXX','MXX','BOC','MXX','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,13)/'BCS','BCS','DBC','DBC','MXX','MXX','BOC','BCS','BCS','MXX','MXX','MXX','BCS','MXX'/ ! BCS
      DATA CITABLE(1:NMODES,14)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M7
!
!     Mechanism 7.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   SSA   SSC   OCC   BC1   BC2   BOC  MXX     SECOND
!                                                                                                  MODE
!-------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','SSA','SSC','OCC','BC1','BC2','BOC','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','SSA','SSC','OCC','BC1','BC2','BOC','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 5)/'SSA','SSA','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 6)/'SSC','SSC','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX'/ ! SSC
      DATA CITABLE(1:NMODES, 7)/'OCC','OCC','MXX','MXX','MXX','MXX','OCC','BOC','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES, 8)/'BC1','BC1','MXX','MXX','MXX','MXX','BOC','BC1','BC1','BOC','MXX'/ ! BC1
      DATA CITABLE(1:NMODES, 9)/'BC2','BC2','MXX','MXX','MXX','MXX','BOC','BC1','BC2','BOC','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,10)/'BOC','BOC','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,11)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M8
!
!     Mechanism 8.
!
!     FIRST MODE               ACC   DD1   DS1   SSS   OCC   BC1   BC2   MXX    SECOND
!                                                                                MODE
!-------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'ACC','DD1','DS1','SSS','OCC','BC1','BC2','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 2)/'DD1','DD1','DD1','MXX','MXX','MXX','MXX','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 3)/'DS1','DD1','DS1','MXX','MXX','MXX','MXX','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 4)/'SSS','MXX','MXX','SSS','MXX','MXX','MXX','MXX'/ ! SSS
      DATA CITABLE(1:NMODES, 5)/'OCC','MXX','MXX','MXX','OCC','MXX','MXX','MXX'/ ! OCC
      DATA CITABLE(1:NMODES, 6)/'BC1','MXX','MXX','MXX','MXX','BC1','BC1','MXX'/ ! BC1
      DATA CITABLE(1:NMODES, 7)/'BC2','MXX','MXX','MXX','MXX','BC1','BC2','MXX'/ ! BC2
      DATA CITABLE(1:NMODES, 8)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M9
!     Mechanism 9
!
!     FIRST MODE               AKK   ACC   DD1   DS1   DD2   DS2   SSA   SSC   OCC   BC1   BC2   OCS   BOC   BCS   MXX    SECOND
!                                                                                                                          MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BC1','BC2','OCS','BOC','BCS','MXX'/ ! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCS','BCS','BCS','OCS','BOC','BCS','MXX'/ ! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','DD2','DD2','MXX','MXX','DD1','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','DD2','DS2','MXX','MXX','DS1','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DS1
      DATA CITABLE(1:NMODES, 5)/'DD2','DD2','DD2','DD2','DD2','DD2','MXX','MXX','DD2','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DD2
      DATA CITABLE(1:NMODES, 6)/'DS2','DS2','DD2','DS2','DD2','DS2','MXX','MXX','DS2','MXX','MXX','MXX','MXX','MXX','MXX'/ ! DS2
      DATA CITABLE(1:NMODES, 7)/'SSA','SSA','MXX','MXX','MXX','MXX','SSA','SSC','SSA','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSA
      DATA CITABLE(1:NMODES, 8)/'SSC','SSC','MXX','MXX','MXX','MXX','SSC','SSC','SSC','MXX','MXX','MXX','MXX','MXX','MXX'/ ! SSC
!      DATA CITABLE(1:NMODES, 9)/'OCC','OCS','MXX','MXX','MXX','MXX','MXX','MXX','OCC','BOC','BOC','OCS','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES, 9)/'OCC','OCS','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BOC','BOC','OCS','BOC','BOC','MXX'/ ! OCC
      DATA CITABLE(1:NMODES,10)/'BC1','BCS','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BC1','BC1','BOC','BOC','BCS','MXX'/ ! BC1
      DATA CITABLE(1:NMODES,11)/'BC2','BCS','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BC1','BC2','BOC','BOC','BCS','MXX'/ ! BC2
      DATA CITABLE(1:NMODES,12)/'OCS','OCS','MXX','MXX','MXX','MXX','MXX','MXX','OCS','BOC','BOC','OCS','BOC','BOC','MXX'/ ! OCS
      DATA CITABLE(1:NMODES,13)/'BOC','BOC','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','BOC','BOC','MXX'/ ! BOC
      DATA CITABLE(1:NMODES,14)/'BCS','BCS','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BCS','BCS','BOC','BOC','BCS','MXX'/ ! BCS
      DATA CITABLE(1:NMODES,15)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/ ! MXX
!-------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M10
!     Mechanism 10.
!
!     FIRST MODE               AKK   ACC   DD1   DS1   DD2   DS2   SSA   SSC   OCC   BC1   BC2   OCS    BOC   BCS   MXX    SECOND
!                                                                                                                           MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCC','BC1','BC2','OCS','BOC','BCS','MXX'/! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC','DD1','DS1','DD2','DS2','SSA','SSC','OCS','BCS','BCS','OCS','BOC','BCS','MXX'/! ACC
      DATA CITABLE(1:NMODES, 3)/'DD1','DD1','DD1','DD1','DD2','DD2','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! DD1
      DATA CITABLE(1:NMODES, 4)/'DS1','DS1','DD1','DS1','DD2','DS2','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! DS1
      DATA CITABLE(1:NMODES, 5)/'DD2','DD2','DD2','DD2','DD2','DD2','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! DD2
      DATA CITABLE(1:NMODES, 6)/'DS2','DS2','DD2','DS2','DD2','DS2','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! DS2
      DATA CITABLE(1:NMODES, 7)/'SSA','SSA','MXX','MXX','MXX','MXX','SSA','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! SSA
      DATA CITABLE(1:NMODES, 8)/'SSC','SSC','MXX','MXX','MXX','MXX','SSC','SSC','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! SSC
      DATA CITABLE(1:NMODES, 9)/'OCC','OCS','MXX','MXX','MXX','MXX','MXX','MXX','OCC','BOC','BOC','OCS','BOC','BOC','MXX'/! OCC
      DATA CITABLE(1:NMODES,10)/'BC1','BCS','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BC1','BC1','BOC','BOC','BCS','MXX'/! BC1
      DATA CITABLE(1:NMODES,11)/'BC2','BCS','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BC1','BC2','BOC','BOC','BCS','MXX'/! BC2
      DATA CITABLE(1:NMODES,12)/'OCS','OCS','MXX','MXX','MXX','MXX','MXX','MXX','OCS','BOC','BOC','OCS','BOC','BOC','MXX'/! OCS
      DATA CITABLE(1:NMODES,13)/'BOC','BOC','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BOC','BOC','BOC','BOC','BOC','MXX'/! BOC
      DATA CITABLE(1:NMODES,14)/'BCS','BCS','MXX','MXX','MXX','MXX','MXX','MXX','BOC','BCS','BCS','BOC','BOC','BCS','MXX'/! BCS
      DATA CITABLE(1:NMODES,15)/'MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX','MXX'/! MXX
!-------------------------------------------------------------------------------------------------------------------------------------
#elif defined TRACERS_AMP_M11
!     Mechanism 11.
!
!     FIRST MODE               AKK   ACC    SECOND
!                                           MODE
!-------------------------------------------------------------------------------------------------------------------------------------
      CHARACTER(LEN=3) :: CITABLE(NMODES,NMODES)
      DATA CITABLE(1:NMODES, 1)/'AKK','ACC'/! AKK
      DATA CITABLE(1:NMODES, 2)/'ACC','ACC'/! ACC
!-------------------------------------------------------------------------------------------------------------------------------------
#endif
      END MODULE AERO_CONFIG
!
! Information for the transported species for each mechanism. 
!
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 1
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2   
!     0         H2O            0            AH2O                  3     MASS_H2O                     3   
!     1         AKK            1            SULF                  4     MASS_AKK_SULF                4   
!     1         AKK            1            NUMB                  5     NUMB_AKK_1                   5   
!     2         ACC            1            SULF                  6     MASS_ACC_SULF                6   
!     2         ACC            1            NUMB                  7     NUMB_ACC_1                   7   
!     3         DD1            1            SULF                  8     MASS_DD1_SULF                8   
!     3         DD1            4            DUST                  9     MASS_DD1_DUST                9   
!     3         DD1            1            NUMB                 10     NUMB_DD1_1                  10   
!     4         DS1            1            SULF                 11     MASS_DS1_SULF               11   
!     4         DS1            4            DUST                 12     MASS_DS1_DUST               12   
!     4         DS1            1            NUMB                 13     NUMB_DS1_1                  13   
!     5         DD2            1            SULF                 14     MASS_DD2_SULF               14   
!     5         DD2            4            DUST                 15     MASS_DD2_DUST               15   
!     5         DD2            1            NUMB                 16     NUMB_DD2_1                  16   
!     6         DS2            1            SULF                 17     MASS_DS2_SULF               17   
!     6         DS2            4            DUST                 18     MASS_DS2_DUST               18   
!     6         DS2            1            NUMB                 19     NUMB_DS2_1                  19   
!     7         SSA            1            SULF                 20     MASS_SSA_SULF               20   
!     7         SSA            5            SEAS                 21     MASS_SSA_SEAS               21   
!     7         SSA            1            NUMB                 22     NUMB_SSA_1                  22   
!     8         SSC            1            SULF                 23     MASS_SSC_SULF                    
!     8         SSC            5            SEAS                 24     MASS_SSC_SEAS               23   
!     8         SSC            1            NUMB                 25     NUMB_SSC_1                  24   
!     9         OCC            1            SULF                 26     MASS_OCC_SULF               25   
!     9         OCC            3            OCAR                 27     MASS_OCC_OCAR               26   
!     9         OCC            1            NUMB                 28     NUMB_OCC_1                  27   
!    10         BC1            1            SULF                 29     MASS_BC1_SULF               28   
!    10         BC1            2            BCAR                 30     MASS_BC1_BCAR               29   
!    10         BC1            1            NUMB                 31     NUMB_BC1_1                  30   
!    11         BC2            1            SULF                 32     MASS_BC2_SULF               31   
!    11         BC2            2            BCAR                 33     MASS_BC2_BCAR               32   
!    11         BC2            1            NUMB                 34     NUMB_BC2_1                  33   
!    12         BC3            1            SULF                 35     MASS_BC3_SULF               34   
!    12         BC3            2            BCAR                 36     MASS_BC3_BCAR               35   
!    12         BC3            1            NUMB                 37     NUMB_BC3_1                  36   
!    13         DBC            1            SULF                 38     MASS_DBC_SULF               37   
!    13         DBC            2            BCAR                 39     MASS_DBC_BCAR               38   
!    13         DBC            4            DUST                 40     MASS_DBC_DUST               39   
!    13         DBC            1            NUMB                 41     NUMB_DBC_1                  40   
!    14         BOC            1            SULF                 42     MASS_BOC_SULF               41   
!    14         BOC            2            BCAR                 43     MASS_BOC_BCAR               42   
!    14         BOC            3            OCAR                 44     MASS_BOC_OCAR               43   
!    14         BOC            1            NUMB                 45     NUMB_BOC_1                  44   
!    15         BCS            1            SULF                 46     MASS_BCS_SULF               45   
!    15         BCS            2            BCAR                 47     MASS_BCS_BCAR               46   
!    15         BCS            1            NUMB                 48     NUMB_BCS_1                  47   
!    16         MXX            1            SULF                 49     MASS_MXX_SULF               48   
!    16         MXX            2            BCAR                 50     MASS_MXX_BCAR               49   
!    16         MXX            3            OCAR                 51     MASS_MXX_OCAR               50   
!    16         MXX            4            DUST                 52     MASS_MXX_DUST               51   
!    16         MXX            5            SEAS                 53     MASS_MXX_SEAS               52   
!    16         MXX            1            NUMB                 54     NUMB_MXX_1                  53
!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   DD2      5
!   DS2      6
!   SSA      7
!   SSC      8
!   OCC      9
!   BC1     10
!   BC2     11
!   BC3     12
!   DBC     13
!   BOC     14
!   BCS     15
!   MXX     16
! 
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 2
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         AKK            1            SULF                  4     MASS_AKK_SULF                4
!     1         AKK            1            NUMB                  5     NUMB_AKK_1                   5
!     2         ACC            1            SULF                  6     MASS_ACC_SULF                6
!     2         ACC            1            NUMB                  7     NUMB_ACC_1                   7
!     3         DD1            1            SULF                  8     MASS_DD1_SULF                8
!     3         DD1            4            DUST                  9     MASS_DD1_DUST                9
!     3         DD1            1            NUMB                 10     NUMB_DD1_1                  10
!     4         DS1            1            SULF                 11     MASS_DS1_SULF               11
!     4         DS1            4            DUST                 12     MASS_DS1_DUST               12
!     4         DS1            1            NUMB                 13     NUMB_DS1_1                  13 
!     5         DD2            1            SULF                 14     MASS_DD2_SULF               14
!     5         DD2            4            DUST                 15     MASS_DD2_DUST               15
!     5         DD2            1            NUMB                 16     NUMB_DD2_1                  16
!     6         DS2            1            SULF                 17     MASS_DS2_SULF               17
!     6         DS2            4            DUST                 18     MASS_DS2_DUST               18
!     6         DS2            1            NUMB                 19     NUMB_DS2_1                  19
!     7         SSA            1            SULF                 20     MASS_SSA_SULF               20
!     7         SSA            5            SEAS                 21     MASS_SSA_SEAS               21
!     7         SSA            1            NUMB                 22     NUMB_SSA_1                    
!     8         SSC            1            SULF                 23     MASS_SSC_SULF                 
!     8         SSC            5            SEAS                 24     MASS_SSC_SEAS               22 
!     8         SSC            1            NUMB                 25     NUMB_SSC_1                    
!     9         OCC            1            SULF                 26     MASS_OCC_SULF               23
!     9         OCC            3            OCAR                 27     MASS_OCC_OCAR               24
!     9         OCC            1            NUMB                 28     NUMB_OCC_1                  25  
!    10         BC1            1            SULF                 29     MASS_BC1_SULF               26
!    10         BC1            2            BCAR                 30     MASS_BC1_BCAR               27
!    10         BC1            1            NUMB                 31     NUMB_BC1_1                  28
!    11         BC2            1            SULF                 32     MASS_BC2_SULF               29
!    11         BC2            2            BCAR                 33     MASS_BC2_BCAR               30
!    11         BC2            1            NUMB                 34     NUMB_BC2_1                  31
!    12         OCS            1            SULF                 35     MASS_OCS_SULF               32 
!    12         OCS            3            OCAR                 36     MASS_OCS_OCAR               33
!    12         OCS            1            NUMB                 37     NUMB_OCS_1                  34
!    13         DBC            1            SULF                 38     MASS_DBC_SULF               35
!    13         DBC            2            BCAR                 39     MASS_DBC_BCAR               36
!    13         DBC            4            DUST                 40     MASS_DBC_DUST               37
!    13         DBC            1            NUMB                 41     NUMB_DBC_1                  38
!    14         BOC            1            SULF                 42     MASS_BOC_SULF               39
!    14         BOC            2            BCAR                 43     MASS_BOC_BCAR               40
!    14         BOC            3            OCAR                 44     MASS_BOC_OCAR               41
!    14         BOC            1            NUMB                 45     NUMB_BOC_1                  42
!    15         BCS            1            SULF                 46     MASS_BCS_SULF               43
!    15         BCS            2            BCAR                 47     MASS_BCS_BCAR               44
!    15         BCS            1            NUMB                 48     NUMB_BCS_1                  45
!    16         MXX            1            SULF                 49     MASS_MXX_SULF               46
!    16         MXX            2            BCAR                 50     MASS_MXX_BCAR               47
!    16         MXX            3            OCAR                 51     MASS_MXX_OCAR               48
!    16         MXX            4            DUST                 52     MASS_MXX_DUST               49
!    16         MXX            5            SEAS                 53     MASS_MXX_SEAS               50
!    16         MXX            1            NUMB                 54     NUMB_MXX_1                  51
!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   DD2      5
!   DS2      6
!   SSA      7
!   SSC      8
!   OCC      9
!   BC1     10
!   BC2     11
!   DBC     13
!   BOC     14
!   BCS     15
!   OCS     12
!   MXX     16
! 
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 3
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         AKK            1            SULF                  4     MASS_AKK_SULF                4
!     1         AKK            1            NUMB                  5     NUMB_AKK_1                   5
!     2         ACC            1            SULF                  6     MASS_ACC_SULF                6
!     2         ACC            1            NUMB                  7     NUMB_ACC_1                   7
!     3         DD1            1            SULF                  8     MASS_DD1_SULF                8
!     3         DD1            4            DUST                  9     MASS_DD1_DUST                9
!     3         DD1            1            NUMB                 10     NUMB_DD1_1                  10
!     4         DS1            1            SULF                 11     MASS_DS1_SULF               11
!     4         DS1            4            DUST                 12     MASS_DS1_DUST               12
!     4         DS1            1            NUMB                 13     NUMB_DS1_1                  13
!     5         DD2            1            SULF                 14     MASS_DD2_SULF               14
!     5         DD2            4            DUST                 15     MASS_DD2_DUST               15
!     5         DD2            1            NUMB                 16     NUMB_DD2_1                  16
!     6         DS2            1            SULF                 17     MASS_DS2_SULF               17
!     6         DS2            4            DUST                 18     MASS_DS2_DUST               18
!     6         DS2            1            NUMB                 19     NUMB_DS2_1                  19
!     7         SSA            1            SULF                 20     MASS_SSA_SULF               20
!     7         SSA            5            SEAS                 21     MASS_SSA_SEAS               21
!     7         SSA            1            NUMB                 22     NUMB_SSA_1                    
!     8         SSC            1            SULF                 23     MASS_SSC_SULF                 
!     8         SSC            5            SEAS                 24     MASS_SSC_SEAS               22
!     8         SSC            1            NUMB                 25     NUMB_SSC_1                    
!     9         OCC            1            SULF                 26     MASS_OCC_SULF               23
!     9         OCC            3            OCAR                 27     MASS_OCC_OCAR               24
!     9         OCC            1            NUMB                 28     NUMB_OCC_1                  25
!    10         BC1            1            SULF                 29     MASS_BC1_SULF               26
!    10         BC1            2            BCAR                 30     MASS_BC1_BCAR               27
!    10         BC1            1            NUMB                 31     NUMB_BC1_1                  28
!    11         BC2            1            SULF                 32     MASS_BC2_SULF               29
!    11         BC2            2            BCAR                 33     MASS_BC2_BCAR               30
!    11         BC2            1            NUMB                 34     NUMB_BC2_1                  31
!    12         BOC            1            SULF                 35     MASS_BOC_SULF               32
!    12         BOC            2            BCAR                 36     MASS_BOC_BCAR               33
!    12         BOC            3            OCAR                 37     MASS_BOC_OCAR               34
!    12         BOC            1            NUMB                 38     NUMB_BOC_1                  35
!    13         MXX            1            SULF                 39     MASS_MXX_SULF               36
!    13         MXX            2            BCAR                 40     MASS_MXX_BCAR               37
!    13         MXX            3            OCAR                 41     MASS_MXX_OCAR               38
!    13         MXX            4            DUST                 42     MASS_MXX_DUST               39
!    13         MXX            5            SEAS                 43     MASS_MXX_SEAS               40
!    13         MXX            1            NUMB                 44     NUMB_MXX_1                  41
!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   DD2      5
!   DS2      6
!   SSA      7
!   SSC      8
!   OCC      9
!   BC1     10
!   BC2     11
!   BOC     12
!   MXX     13
! 
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 4
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         ACC            1            SULF                  4     MASS_ACC_SULF                4
!     1         ACC            1            NUMB                  5     NUMB_ACC_1                   5
!     2         DD1            1            SULF                  6     MASS_DD1_SULF                6
!     2         DD1            4            DUST                  7     MASS_DD1_DUST                7
!     2         DD1            1            NUMB                  8     NUMB_DD1_1                   8
!     3         DS1            1            SULF                  9     MASS_DS1_SULF                9
!     3         DS1            4            DUST                 10     MASS_DS1_DUST               10
!     3         DS1            1            NUMB                 11     NUMB_DS1_1                  11
!     4         DD2            1            SULF                 12     MASS_DD2_SULF               12
!     4         DD2            4            DUST                 13     MASS_DD2_DUST               13
!     4         DD2            1            NUMB                 14     NUMB_DD2_1                  14
!     5         DS2            1            SULF                 15     MASS_DS2_SULF               15
!     5         DS2            4            DUST                 16     MASS_DS2_DUST               16
!     5         DS2            1            NUMB                 17     NUMB_DS2_1                  17
!     6         SSS            1            SULF                 18     MASS_SSS_SULF               18
!     6         SSS            5            SEAS                 19     MASS_SSS_SEAS               19
!     6         SSS            1            NUMB                 20     NUMB_SSS_1                    
!     7         OCC            1            SULF                 21     MASS_OCC_SULF               20
!     7         OCC            3            OCAR                 22     MASS_OCC_OCAR               21
!     7         OCC            1            NUMB                 23     NUMB_OCC_1                  22
!     8         BC1            1            SULF                 24     MASS_BC1_SULF               23
!     8         BC1            2            BCAR                 25     MASS_BC1_BCAR               24
!     8         BC1            1            NUMB                 26     NUMB_BC1_1                  25
!     9         BC2            1            SULF                 27     MASS_BC2_SULF               26
!     9         BC2            2            BCAR                 28     MASS_BC2_BCAR               27
!     9         BC2            1            NUMB                 29     NUMB_BC2_1                  28
!    10         MXX            1            SULF                 30     MASS_MXX_SULF               29
!    10         MXX            2            BCAR                 31     MASS_MXX_BCAR               30
!    10         MXX            3            OCAR                 32     MASS_MXX_OCAR               31
!    10         MXX            4            DUST                 33     MASS_MXX_DUST               32
!    10         MXX            5            SEAS                 34     MASS_MXX_SEAS               33
!    10         MXX            1            NUMB                 35     NUMB_MXX_1                  34
!
!MODE_NAME  MODE NUMBER
!
!   ACC      1
!   DD1      2
!   DS1      3
!   DD2      4
!   DS2      5
!   SSS      6
!   OCC      7
!   BC1      8
!   BC2      9
!   MXX     10
!
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 5
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         AKK            1            SULF                  4     MASS_AKK_SULF                4
!     1         AKK            1            NUMB                  5     NUMB_AKK_1                   5
!     2         ACC            1            SULF                  6     MASS_ACC_SULF                6
!     2         ACC            1            NUMB                  7     NUMB_ACC_1                   7
!     3         DD1            1            SULF                  8     MASS_DD1_SULF                8
!     3         DD1            4            DUST                  9     MASS_DD1_DUST                9
!     3         DD1            1            NUMB                 10     NUMB_DD1_1                  10
!     4         DS1            1            SULF                 11     MASS_DS1_SULF               11
!     4         DS1            4            DUST                 12     MASS_DS1_DUST               12
!     4         DS1            1            NUMB                 13     NUMB_DS1_1                  13
!     5         SSA            1            SULF                 14     MASS_SSA_SULF               14
!     5         SSA            5            SEAS                 15     MASS_SSA_SEAS               15
!     5         SSA            1            NUMB                 16     NUMB_SSA_1                    
!     6         SSC            1            SULF                 17     MASS_SSC_SULF                 
!     6         SSC            5            SEAS                 18     MASS_SSC_SEAS               16
!     6         SSC            1            NUMB                 19     NUMB_SSC_1                    
!     7         OCC            1            SULF                 20     MASS_OCC_SULF               17
!     7         OCC            3            OCAR                 21     MASS_OCC_OCAR               18
!     7         OCC            1            NUMB                 22     NUMB_OCC_1                  19
!     8         BC1            1            SULF                 23     MASS_BC1_SULF               20
!     8         BC1            2            BCAR                 24     MASS_BC1_BCAR               21
!     8         BC1            1            NUMB                 25     NUMB_BC1_1                  22
!     9         BC2            1            SULF                 26     MASS_BC2_SULF               23
!     9         BC2            2            BCAR                 27     MASS_BC2_BCAR               24
!     9         BC2            1            NUMB                 28     NUMB_BC2_1                  25
!    10         BC3            1            SULF                 29     MASS_BC3_SULF               26
!    10         BC3            2            BCAR                 30     MASS_BC3_BCAR               27
!    10         BC3            1            NUMB                 31     NUMB_BC3_1                  28
!    11         DBC            1            SULF                 32     MASS_DBC_SULF               29
!    11         DBC            2            BCAR                 33     MASS_DBC_BCAR               30
!    11         DBC            4            DUST                 34     MASS_DBC_DUST               31
!    11         DBC            1            NUMB                 35     NUMB_DBC_1                  32
!    12         BOC            1            SULF                 36     MASS_BOC_SULF               33
!    12         BOC            2            BCAR                 37     MASS_BOC_BCAR               34
!    12         BOC            3            OCAR                 38     MASS_BOC_OCAR               35
!    12         BOC            1            NUMB                 39     NUMB_BOC_1                  36
!    13         BCS            1            SULF                 40     MASS_BCS_SULF               37
!    13         BCS            2            BCAR                 41     MASS_BCS_BCAR               38
!    13         BCS            1            NUMB                 42     NUMB_BCS_1                  39
!    14         MXX            1            SULF                 43     MASS_MXX_SULF               40
!    14         MXX            2            BCAR                 44     MASS_MXX_BCAR               41
!    14         MXX            3            OCAR                 45     MASS_MXX_OCAR               42 
!    14         MXX            4            DUST                 46     MASS_MXX_DUST               43
!    14         MXX            5            SEAS                 47     MASS_MXX_SEAS               44
!    14         MXX            1            NUMB                 48     NUMB_MXX_1                  45
!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   SSA      5
!   SSC      6
!   OCC      7
!   BC1      8
!   BC2      9
!   BC3     10
!   DBC     11
!   BOC     12
!   BCS     13
!   MXX     14
!
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 6
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         AKK            1            SULF                  4     MASS_AKK_SULF                4
!     1         AKK            1            NUMB                  5     NUMB_AKK_1                   5
!     2         ACC            1            SULF                  6     MASS_ACC_SULF                6
!     2         ACC            1            NUMB                  7     NUMB_ACC_1                   7
!     3         DD1            1            SULF                  8     MASS_DD1_SULF                8
!     3         DD1            4            DUST                  9     MASS_DD1_DUST                9
!     3         DD1            1            NUMB                 10     NUMB_DD1_1                  10
!     4         DS1            1            SULF                 11     MASS_DS1_SULF               11
!     4         DS1            4            DUST                 12     MASS_DS1_DUST               12
!     4         DS1            1            NUMB                 13     NUMB_DS1_1                  13
!     5         SSA            1            SULF                 14     MASS_SSA_SULF               14
!     5         SSA            5            SEAS                 15     MASS_SSA_SEAS               15
!     5         SSA            1            NUMB                 16     NUMB_SSA_1                     
!     6         SSC            1            SULF                 17     MASS_SSC_SULF                 
!     6         SSC            5            SEAS                 18     MASS_SSC_SEAS               16
!     6         SSC            1            NUMB                 19     NUMB_SSC_1                    
!     7         OCC            1            SULF                 20     MASS_OCC_SULF               17
!     7         OCC            3            OCAR                 21     MASS_OCC_OCAR               18
!     7         OCC            1            NUMB                 22     NUMB_OCC_1                  19
!     8         BC1            1            SULF                 23     MASS_BC1_SULF               20
!     8         BC1            2            BCAR                 24     MASS_BC1_BCAR               21
!     8         BC1            1            NUMB                 25     NUMB_BC1_1                  22
!     9         BC2            1            SULF                 26     MASS_BC2_SULF               23
!     9         BC2            2            BCAR                 27     MASS_BC2_BCAR               24
!     9         BC2            1            NUMB                 28     NUMB_BC2_1                  25
!    10         OCS            1            SULF                 29     MASS_OCS_SULF               26
!    10         OCS            3            OCAR                 30     MASS_OCS_OCAR               27
!    10         OCS            1            NUMB                 31     NUMB_OCS_1                  28
!    11         DBC            1            SULF                 32     MASS_DBC_SULF               29
!    11         DBC            2            BCAR                 33     MASS_DBC_BCAR               30
!    11         DBC            4            DUST                 34     MASS_DBC_DUST               31
!    11         DBC            1            NUMB                 35     NUMB_DBC_1                  32
!    12         BOC            1            SULF                 36     MASS_BOC_SULF               33
!    12         BOC            2            BCAR                 37     MASS_BOC_BCAR               34
!    12         BOC            3            OCAR                 38     MASS_BOC_OCAR               35
!    12         BOC            1            NUMB                 39     NUMB_BOC_1                  36
!    13         BCS            1            SULF                 40     MASS_BCS_SULF               37
!    13         BCS            2            BCAR                 41     MASS_BCS_BCAR               38
!    13         BCS            1            NUMB                 42     NUMB_BCS_1                  39
!    14         MXX            1            SULF                 43     MASS_MXX_SULF               40
!    14         MXX            2            BCAR                 44     MASS_MXX_BCAR               41
!    14         MXX            3            OCAR                 45     MASS_MXX_OCAR               42
!    14         MXX            4            DUST                 46     MASS_MXX_DUST               43
!    14         MXX            5            SEAS                 47     MASS_MXX_SEAS               44
!    14         MXX            1            NUMB                 48     NUMB_MXX_1                  45
!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   SSA      5
!   SSC      6
!   OCC      7
!   BC1      8
!   BC2      9
!   DBC     11
!   BOC     12
!   BCS     13
!   OCS     10
!   MXX     14
!
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 7
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         AKK            1            SULF                  4     MASS_AKK_SULF                4
!     1         AKK            1            NUMB                  5     NUMB_AKK_1                   5
!     2         ACC            1            SULF                  6     MASS_ACC_SULF                6
!     2         ACC            1            NUMB                  7     NUMB_ACC_1                   7
!     3         DD1            1            SULF                  8     MASS_DD1_SULF                8
!     3         DD1            4            DUST                  9     MASS_DD1_DUST                9
!     3         DD1            1            NUMB                 10     NUMB_DD1_1                  10
!     4         DS1            1            SULF                 11     MASS_DS1_SULF               11
!     4         DS1            4            DUST                 12     MASS_DS1_DUST               12
!     4         DS1            1            NUMB                 13     NUMB_DS1_1                  13
!     5         SSA            1            SULF                 14     MASS_SSA_SULF               14
!     5         SSA            5            SEAS                 15     MASS_SSA_SEAS               15
!     5         SSA            1            NUMB                 16     NUMB_SSA_1                    
!     6         SSC            1            SULF                 17     MASS_SSC_SULF                 
!     6         SSC            5            SEAS                 18     MASS_SSC_SEAS               16
!     6         SSC            1            NUMB                 19     NUMB_SSC_1                    
!     7         OCC            1            SULF                 20     MASS_OCC_SULF               17
!     7         OCC            3            OCAR                 21     MASS_OCC_OCAR               18
!     7         OCC            1            NUMB                 22     NUMB_OCC_1                  19
!     8         BC1            1            SULF                 23     MASS_BC1_SULF               20
!     8         BC1            2            BCAR                 24     MASS_BC1_BCAR               21
!     8         BC1            1            NUMB                 25     NUMB_BC1_1                  22
!     9         BC2            1            SULF                 26     MASS_BC2_SULF               23
!     9         BC2            2            BCAR                 27     MASS_BC2_BCAR               24
!     9         BC2            1            NUMB                 28     NUMB_BC2_1                  25
!    10         BOC            1            SULF                 29     MASS_BOC_SULF               26
!    10         BOC            2            BCAR                 30     MASS_BOC_BCAR               27
!    10         BOC            3            OCAR                 31     MASS_BOC_OCAR               28
!    10         BOC            1            NUMB                 32     NUMB_BOC_1                  29
!    11         MXX            1            SULF                 33     MASS_MXX_SULF               30
!    11         MXX            2            BCAR                 34     MASS_MXX_BCAR               31
!    11         MXX            3            OCAR                 35     MASS_MXX_OCAR               32
!    11         MXX            4            DUST                 36     MASS_MXX_DUST               33
!    11         MXX            5            SEAS                 37     MASS_MXX_SEAS               34
!    11         MXX            1            NUMB                 38     NUMB_MXX_1                  35
!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   SSA      5
!   SSC      6
!   OCC      7
!   BC1      8
!   BC2      9
!   BOC     10
!   MXX     11
!
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 8
!-------------------------------------------------------------------------------------------------------------------------
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!
!     0         NO3            0            ANO3                  1     MASS_NO3                     1
!     0         NH4            0            ANH4                  2     MASS_NH4                     2
!     0         H2O            0            AH2O                  3     MASS_H2O                     3
!     1         ACC            1            SULF                  4     MASS_ACC_SULF                4
!     1         ACC            1            NUMB                  5     NUMB_ACC_1                   5
!     2         DD1            1            SULF                  6     MASS_DD1_SULF                6
!     2         DD1            4            DUST                  7     MASS_DD1_DUST                7
!     2         DD1            1            NUMB                  8     NUMB_DD1_1                   8
!     3         DS1            1            SULF                  9     MASS_DS1_SULF                9
!     3         DS1            4            DUST                 10     MASS_DS1_DUST               10
!     3         DS1            1            NUMB                 11     NUMB_DS1_1                  11
!     4         SSS            1            SULF                 12     MASS_SSS_SULF               12
!     4         SSS            5            SEAS                 13     MASS_SSS_SEAS               13
!     4         SSS            1            NUMB                 14     NUMB_SSS_1                    
!     5         OCC            1            SULF                 15     MASS_OCC_SULF               14
!     5         OCC            3            OCAR                 16     MASS_OCC_OCAR               15 
!     5         OCC            1            NUMB                 17     NUMB_OCC_1                  16
!     6         BC1            1            SULF                 18     MASS_BC1_SULF               17
!     6         BC1            2            BCAR                 19     MASS_BC1_BCAR               18
!     6         BC1            1            NUMB                 20     NUMB_BC1_1                  19
!     7         BC2            1            SULF                 21     MASS_BC2_SULF               20
!     7         BC2            2            BCAR                 22     MASS_BC2_BCAR               21
!     7         BC2            1            NUMB                 23     NUMB_BC2_1                  22
!     8         MXX            1            SULF                 24     MASS_MXX_SULF               23
!     8         MXX            2            BCAR                 25     MASS_MXX_BCAR               24
!     8         MXX            3            OCAR                 26     MASS_MXX_OCAR               25
!     8         MXX            4            DUST                 27     MASS_MXX_DUST               26
!     8         MXX            5            SEAS                 28     MASS_MXX_SEAS               27
!     8         MXX            1            NUMB                 29     NUMB_MXX_1                  28
!
!MODE_NAME  MODE NUMBER
!
!   ACC      1
!   DD1      2
!   DS1      3
!   SSS      4
!   OCC      5
!   BC1      6
!   BC2      7
!   MXX      8
!-------------------------------------------------------------------------------------------------------------------------
!MECHANISM 10
!-------------------------------------------------------------------------------------------------------------------------
!     0         NO3            0            ANO3                  1      MASS_NO3                     1
!     0         NH4            0            ANH4                  2      MASS_NH4                     2
!     0         H2O            0            AH2O                  3      MASS_H2O                     3
!     1         AKK            1            SULF                  4      MASS_AKK_SULF                4
!     1         AKK            1            NUMB                  5      NUMB_AKK_1                   5
!     2         ACC            1            SULF                  6      MASS_ACC_SULF                6
!     2         ACC            1            NUMB                  7      NUMB_ACC_1                   7
!     3         DD1            1            SULF                  8      MASS_DD1_SULF                8
!     3         DD1            4            DUST                  9      MASS_DD1_DUST                9
!     3         DD1            1            NUMB                 10      NUMB_DD1_1                  10
!     4         DS1            1            SULF                 11      MASS_DS1_SULF               11
!     4         DS1            4            DUST                 12      MASS_DS1_DUST               12
!     4         DS1            1            NUMB                 13      NUMB_DS1_1                  13
!     5         DD2            1            SULF                 14      MASS_DD2_SULF               14
!     5         DD2            4            DUST                 15      MASS_DD2_DUST               15
!     5         DD2            1            NUMB                 16      NUMB_DD2_1                  16
!     6         DS2            1            SULF                 17      MASS_DS2_SULF               17
!     6         DS2            4            DUST                 18      MASS_DS2_DUST               18
!     6         DS2            1            NUMB                 19      NUMB_DS2_1                  19
!     7         SSA            1            SULF                 20      MASS_SSA_SULF               20
!     7         SSA            5            SEAS                 21      MASS_SSA_SEAS               21
!     7         SSA            1            NUMB                 22      NUMB_SSA_1                  22
!     8         SSC            1            SULF                 23      MASS_SSC_SULF
!     8         SSC            5            SEAS                 24      MASS_SSC_SEAS               23
!     8         SSC            1            NUMB                 25      NUMB_SSC_1                  24
!     9         OCC            1            SULF                 26      MASS_OCC_SULF               25
!     9         OCC            3            OCAR                 27      MASS_OCC_OCAR               26
!     9         OCC            1            NUMB                 28      NUMB_OCC_1                  27
!    10         BC1            1            SULF                 29      MASS_BC1_SULF               28
!    10         BC1            2            BCAR                 30      MASS_BC1_BCAR               29
!    10         BC1            1            NUMB                 31      NUMB_BC1_1                  30
!    11         BC2            1            SULF                 32      MASS_BC2_SULF               31
!    11         BC2            2            BCAR                 33      MASS_BC2_BCAR               32
!    11         BC2            1            NUMB                 34      NUMB_BC2_1                  33
!    12         OCS            1            SULF                 35      MASS_OCS_SULF               34
!    12         OCS            3            OCAR                 36      MASS_OCS_OCAR               35
!    12         OCS            1            NUMB                 37      NUMB_OCS_1                  36
!    13         BOC            1            SULF                 38      MASS_BOC_SULF               37
!    13         BOC            2            BCAR                 39      MASS_BOC_BCAR               38
!    13         BOC            3            OCAR                 40      MASS_BOC_OCAR               39
!    13         BOC            1            NUMB                 41      NUMB_BOC_1                  40
!    14         BCS            1            SULF                 42      MASS_BCS_SULF               41
!    14         BCS            2            BCAR                 43      MASS_BCS_BCAR               42
!    14         BCS            1            NUMB                 44      NUMB_BCS_1                  43
!    15         MXX            1            SULF                 45      MASS_MXX_SULF               44
!    15         MXX            2            BCAR                 46      MASS_MXX_BCAR               45
!    15         MXX            3            OCAR                 47      MASS_MXX_OCAR               46
!    15         MXX            4            DUST                 48      MASS_MXX_DUST               47
!    15         MXX            5            SEAS                 49      MASS_MXX_SEAS               48
!    15         MXX            1            NUMB                 50      NUMB_MXX_1                  49
!
!MODE #   MODE_NAME   CHEM_SPC #   CHEM_SPC_NAME   location in AERO         AERO_SPCS    TRACER NUMBER
!

!
!MODE_NAME  MODE NUMBER
!
!   AKK      1
!   ACC      2
!   DD1      3
!   DS1      4
!   DD2      5
!   DS2      6
!   SSA      7
!   SSC      8
!   OCC      9
!   BC1     10
!   BC2     11
!   OCS     12
!   BOC     13
!   BCS     14
!   MXX     15
! 
!-------------------------------------------------------------------------------------------------------------------------

