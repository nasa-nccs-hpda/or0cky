#include "rundeck_opts.h"
!------------------------------------------------------------------------------
module AmpTracersMetadata_mod
!------------------------------------------------------------------------------
!@sum  AMPTracersMetadata_mod encapsulates the TRACERS_AMP metadata
!@auth NCCS ASTG
  use sharedTracersMetadata_mod, only: DMS_setspec, &
    SO2_setspec, H2O2_s_setspec, NH3_setspec
  USE CONSTANT, only: mwat
  USE AERO_CONFIG, only: nmodes
  USE AERO_PARAM, only: &
    SOLU_DD1, SOLU_DD2, SOLU_AKK, SOLU_ACC, &
    SOLU_DS1, SOLU_DS2, SOLU_SSA, SOLU_SSC, &
    SOLU_SSS, SOLU_OCC, SOLU_BC1, SOLU_BC2, SOLU_BC3, &
    SOLU_DBC, SOLU_BOC, SOLU_BCS, SOLU_OCS, SOLU_MXX
  USE AERO_PARAM, only: &
    DG_DD1, DG_DD2, DG_AKK,                    &
    DG_DS1, DG_DS2, DG_SSA, DG_SSC, DG_ACC,    &
    DG_SSS, DG_OCC, DG_BC1, DG_BC2, DG_BC3,    &
    DG_DBC, DG_BOC, DG_BCS, DG_OCS, DG_MXX
  USE AERO_ACTV, only: &
    DENS_SULF, DENS_DUST, DENS_SEAS, DENS_BCAR, DENS_OCAR
  use Tracer_com, only: &
    n_M_NO3,   n_M_NH4,   n_M_H2O,   n_M_AKK_SU, &
    n_N_AKK_1, n_M_ACC_SU,n_N_ACC_1, n_M_DD1_SU, &
    n_M_DD1_DU,n_N_DD1_1, n_M_DS1_SU,n_M_DS1_DU, &
    n_N_DS1_1 ,n_M_DD2_SU,n_M_DD2_DU,n_N_DD2_1 , &
    n_M_DS2_SU,n_M_DS2_DU,n_N_DS2_1 ,n_M_SSA_SU, &
    n_M_SSA_SS,n_M_SSC_SS,                       &
    n_M_OCC_SU,n_M_OCC_OC,n_N_OCC_1 ,            &
    n_M_BC1_SU,n_M_BC1_BC,n_N_BC1_1 ,n_M_BC2_SU, &
    n_M_BC2_BC,n_N_BC2_1 ,n_M_BC3_SU,n_M_BC3_BC, &
    n_N_BC3_1 ,n_M_DBC_SU,n_M_DBC_BC,n_M_DBC_DU, &
    n_N_DBC_1 ,n_M_BOC_SU,n_M_BOC_BC,n_M_BOC_OC, &
    n_N_BOC_1, n_M_BCS_SU,n_M_BCS_BC,n_N_BCS_1 , &
    n_M_MXX_SU,n_M_MXX_BC,n_M_MXX_OC,n_M_MXX_DU, &
    n_M_MXX_SS,n_N_MXX_1 ,n_M_OCS_SU,n_M_OCS_OC, &
    n_N_OCS_1,n_M_SSS_SS,n_M_SSS_SU,             &
    n_H2SO4, n_N_SSA_1, n_N_SSC_1
  use TRACER_COM, only: &
    n_M_ACC_OCM2, n_M_ACC_OCM1, n_M_ACC_OCM0, &
    n_M_ACC_OCP1, n_M_ACC_OCP2, n_M_ACC_OCP3, &
    n_M_ACC_OCP4, n_M_ACC_OCP5, n_M_ACC_OCP6
  use TRACER_COM, only: &
    n_M_DD1_OCM2, n_M_DD1_OCM1, n_M_DD1_OCM0, &
    n_M_DD1_OCP1, n_M_DD1_OCP2, n_M_DD1_OCP3, &
    n_M_DD1_OCP4, n_M_DD1_OCP5, n_M_DD1_OCP6
  use TRACER_COM, only: &
    n_M_DS1_OCM2, n_M_DS1_OCM1, n_M_DS1_OCM0, &
    n_M_DS1_OCP1, n_M_DS1_OCP2, n_M_DS1_OCP3, &
    n_M_DS1_OCP4, n_M_DS1_OCP5, n_M_DS1_OCP6
  use TRACER_COM, only: &
    n_M_DD2_OCM2, n_M_DD2_OCM1, n_M_DD2_OCM0, &
    n_M_DD2_OCP1, n_M_DD2_OCP2, n_M_DD2_OCP3, &
    n_M_DD2_OCP4, n_M_DD2_OCP5, n_M_DD2_OCP6
  use TRACER_COM, only: &
    n_M_DS2_OCM2, n_M_DS2_OCM1, n_M_DS2_OCM0, &
    n_M_DS2_OCP1, n_M_DS2_OCP2, n_M_DS2_OCP3, &
    n_M_DS2_OCP4, n_M_DS2_OCP5, n_M_DS2_OCP6
  use TRACER_COM, only: &
    n_M_SSA_OCM2, n_M_SSA_OCM1, n_M_SSA_OCM0, &
    n_M_SSA_OCP1, n_M_SSA_OCP2, n_M_SSA_OCP3, &
    n_M_SSA_OCP4, n_M_SSA_OCP5, n_M_SSA_OCP6
  use TRACER_COM, only: &
    n_M_SSC_OCM2, n_M_SSC_OCM1, n_M_SSC_OCM0, &
    n_M_SSC_OCP1, n_M_SSC_OCP2, n_M_SSC_OCP3, &
    n_M_SSC_OCP4, n_M_SSC_OCP5, n_M_SSC_OCP6
  use TRACER_COM, only: &
    n_M_OCC_OCM2, n_M_OCC_OCM1, n_M_OCC_OCM0, &
    n_M_OCC_OCP1, n_M_OCC_OCP2, n_M_OCC_OCP3, &
    n_M_OCC_OCP4, n_M_OCC_OCP5, n_M_OCC_OCP6
  use TRACER_COM, only: &
    n_M_BC1_OCM2, n_M_BC1_OCM1, n_M_BC1_OCM0, &
    n_M_BC1_OCP1, n_M_BC1_OCP2, n_M_BC1_OCP3, &
    n_M_BC1_OCP4, n_M_BC1_OCP5, n_M_BC1_OCP6
  use TRACER_COM, only: &
    n_M_BC2_OCM2, n_M_BC2_OCM1, n_M_BC2_OCM0, &
    n_M_BC2_OCP1, n_M_BC2_OCP2, n_M_BC2_OCP3, &
    n_M_BC2_OCP4, n_M_BC2_OCP5, n_M_BC2_OCP6
  use TRACER_COM, only: &
    n_M_OCS_OCM2, n_M_OCS_OCM1, n_M_OCS_OCM0, &
    n_M_OCS_OCP1, n_M_OCS_OCP2, n_M_OCS_OCP3, &
    n_M_OCS_OCP4, n_M_OCS_OCP5, n_M_OCS_OCP6
  use TRACER_COM, only: &
    n_M_BOC_OCM2, n_M_BOC_OCM1, n_M_BOC_OCM0, &
    n_M_BOC_OCP1, n_M_BOC_OCP2, n_M_BOC_OCP3, &
    n_M_BOC_OCP4, n_M_BOC_OCP5, n_M_BOC_OCP6
  use TRACER_COM, only: &
    n_M_BCS_OCM2, n_M_BCS_OCM1, n_M_BCS_OCM0, &
    n_M_BCS_OCP1, n_M_BCS_OCP2, n_M_BCS_OCP3, &
    n_M_BCS_OCP4, n_M_BCS_OCP5, n_M_BCS_OCP6
  use TRACER_COM, only: &
    n_M_MXX_OCM2, n_M_MXX_OCM1, n_M_MXX_OCM0, &
    n_M_MXX_OCP1, n_M_MXX_OCP2, n_M_MXX_OCP3, &
    n_M_MXX_OCP4, n_M_MXX_OCP5, n_M_MXX_OCP6
  use TRACER_COM, only: tracers
  use RunTimeControls_mod, only: &
    tracers_nitrate, tracers_aerosols_koch, tracers_aerosols_seasalt, &
    tracers_amp_m1, tracers_amp_m2,         &
    tracers_amp_m3, tracers_amp_m4,         &
    tracers_amp_m5, tracers_amp_m6,         &
    tracers_amp_m7, tracers_amp_m8,         &
    tracers_amp_m9, tracers_amp_m10,        &
    tracers_amp_m11,                        &
    tracers_special_shindell
  use Tracer_com, only: ntmAMPi, ntmAMPe, ntmAMP, ntm_chem, coupled_chem
  use OldTracer_mod, only: set_needtrs
  use OldTracer_mod, only: nPart
  use OldTracer_mod, only: set_tr_mm
  use OldTracer_mod, only: set_ntm_power
  use OldTracer_mod, only: set_trpdens
  use OldTracer_mod, only: set_trradius
  use OldTracer_mod, only: set_fq_aer
  use OldTracer_mod, only: set_tr_wd_type
  use OldTracer_mod, only: set_has_chemistry
  use OldTracer_mod, only: oldAddTracer
  use OldTracer_mod, only: trname, MAX_LEN_NAME
  use Tracer_mod, only: Tracer

  implicit none
  private

  public AMP_initMetadata

!@var AMP_AERO_MAP Map of indices of transported tracers in AMP (?) (1-based)
  integer, allocatable, dimension(:), public :: AMP_AERO_MAP
!@var AMP_NUMB_MAP Enumeration of number concentrations in populations, zero elsewhere (1-based)
  integer, allocatable, dimension(:), public :: AMP_NUMB_MAP
!@var AMP_MODES_MAP Enumeration of all tracers in a population, zero elsewhere (1-based)
  integer, allocatable, dimension(:), public :: AMP_MODES_MAP
!@var AMP_trm_nm1 Index of first mass tracer in a population (ntm-based)
  integer, allocatable, dimension(:), public :: AMP_trm_nm1
!@var AMP_trm_nm2 Index of last mass tracer in a population (ntm-based)
  integer, allocatable, dimension(:), public :: AMP_trm_nm2
!@var iNamp indices of number concentration tracers
  integer, dimension(nmodes), public :: iNamp=0

  real(8), parameter :: microns2meters = 1.0d-6
  REAL(8), PARAMETER :: RG_AKK = microns2meters*DG_AKK/2.0d0 
  REAL(8), PARAMETER :: RG_ACC = microns2meters*DG_ACC/2.0d0
  REAL(8), PARAMETER :: RG_DD1 = microns2meters*DG_DD1/2.0d0
  REAL(8), PARAMETER :: RG_DD2 = microns2meters*DG_DD2/2.0d0
  REAL(8), PARAMETER :: RG_DS1 = microns2meters*DG_DS1/2.0d0
  REAL(8), PARAMETER :: RG_DS2 = microns2meters*DG_DS2/2.0d0
  REAL(8), PARAMETER :: RG_SSA = microns2meters*DG_SSA/2.0d0
  REAL(8), PARAMETER :: RG_SSC = microns2meters*DG_SSC/2.0d0
  REAL(8), PARAMETER :: RG_SSS = microns2meters*DG_SSS/2.0d0
  REAL(8), PARAMETER :: RG_OCC = microns2meters*DG_OCC/2.0d0
  REAL(8), PARAMETER :: RG_BC1 = microns2meters*DG_BC1/2.0d0
  REAL(8), PARAMETER :: RG_BC2 = microns2meters*DG_BC2/2.0d0
  REAL(8), PARAMETER :: RG_BC3 = microns2meters*DG_BC3/2.0d0
  REAL(8), PARAMETER :: RG_DBC = microns2meters*DG_DBC/2.0d0
  REAL(8), PARAMETER :: RG_BOC = microns2meters*DG_BOC/2.0d0
  REAL(8), PARAMETER :: RG_BCS = microns2meters*DG_BCS/2.0d0
  REAL(8), PARAMETER :: RG_OCS = microns2meters*DG_OCS/2.0d0
  REAL(8), PARAMETER :: RG_MXX = microns2meters*DG_MXX/2.0d0

  real(8), parameter :: SULF_MolecMass = 96.0d0
  real(8), parameter :: DUST_MolecMass = 1.0d0
  real(8), parameter :: SEAS_MolecMass = 75.0d0
  real(8), parameter :: BCAR_MolecMass = 12.0d0
  real(8), parameter :: OCAR_MolecMass = 12.0d0

  character(len=MAX_LEN_NAME) :: trname_curr
  integer :: n,nn,nAMP,imode,itr

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
  subroutine AMP_initMetadata(pTracer)
!------------------------------------------------------------------------------
    implicit none
    class (Tracer), pointer :: pTracer

    !**** Tracers for Scheme AMP: Aerosol Microphysics (Mechanism M1 - M8)

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      call  M_NO3_setSpec('M_NO3')
      call  M_NH4_setSpec('M_NH4')
    endif
    call  M_H2O_setSpec('M_H2O')

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10.or. &
        tracers_amp_m11) then
      n_M_AKK_SU = AMP_SetSpec('AKK', 'SU')
      n_N_AKK_1  = AMP_SetSpec('AKK', '1' )
    end if

    n_M_ACC_SU = AMP_SetSpec('ACC', 'SU')
    if (tracers_amp_m9) then
      n_M_ACC_OCM2 = AMP_SetSpec('ACC', 'OCM2')
      n_M_ACC_OCM1 = AMP_SetSpec('ACC', 'OCM1')
      n_M_ACC_OCM0 = AMP_SetSpec('ACC', 'OCM0')
      n_M_ACC_OCP1 = AMP_SetSpec('ACC', 'OCP1')
      n_M_ACC_OCP2 = AMP_SetSpec('ACC', 'OCP2')
      n_M_ACC_OCP3 = AMP_SetSpec('ACC', 'OCP3')
      n_M_ACC_OCP4 = AMP_SetSpec('ACC', 'OCP4')
      n_M_ACC_OCP5 = AMP_SetSpec('ACC', 'OCP5')
      n_M_ACC_OCP6 = AMP_SetSpec('ACC', 'OCP6')
    endif
    n_N_ACC_1  = AMP_SetSpec('ACC', '1' )

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_DD1_SU = AMP_SetSpec('DD1', 'SU')
      n_M_DD1_DU = AMP_SetSpec('DD1', 'DU')
      if (tracers_amp_m9) then
        n_M_DD1_OCM2 = AMP_SetSpec('DD1', 'OCM2')
        n_M_DD1_OCM1 = AMP_SetSpec('DD1', 'OCM1')
        n_M_DD1_OCM0 = AMP_SetSpec('DD1', 'OCM0')
        n_M_DD1_OCP1 = AMP_SetSpec('DD1', 'OCP1')
        n_M_DD1_OCP2 = AMP_SetSpec('DD1', 'OCP2')
        n_M_DD1_OCP3 = AMP_SetSpec('DD1', 'OCP3')
        n_M_DD1_OCP4 = AMP_SetSpec('DD1', 'OCP4')
        n_M_DD1_OCP5 = AMP_SetSpec('DD1', 'OCP5')
        n_M_DD1_OCP6 = AMP_SetSpec('DD1', 'OCP6')
      endif
      n_N_DD1_1  = AMP_SetSpec('DD1', '1' )
    endif

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_DS1_SU = AMP_SetSpec('DS1', 'SU')
      n_M_DS1_DU = AMP_SetSpec('DS1', 'DU')
      if (tracers_amp_m9) then
        n_M_DS1_OCM2 = AMP_SetSpec('DS1', 'OCM2')
        n_M_DS1_OCM1 = AMP_SetSpec('DS1', 'OCM1')
        n_M_DS1_OCM0 = AMP_SetSpec('DS1', 'OCM0')
        n_M_DS1_OCP1 = AMP_SetSpec('DS1', 'OCP1')
        n_M_DS1_OCP2 = AMP_SetSpec('DS1', 'OCP2')
        n_M_DS1_OCP3 = AMP_SetSpec('DS1', 'OCP3')
        n_M_DS1_OCP4 = AMP_SetSpec('DS1', 'OCP4')
        n_M_DS1_OCP5 = AMP_SetSpec('DS1', 'OCP5')
        n_M_DS1_OCP6 = AMP_SetSpec('DS1', 'OCP6')
      endif
      n_N_DS1_1  = AMP_SetSpec('DS1', '1' )
    endif

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_DD2_SU = AMP_SetSpec('DD2','SU')
      n_M_DD2_DU = AMP_SetSpec('DD2','DU')
      if (tracers_amp_m9) then
        n_M_DD2_OCM2 = AMP_SetSpec('DD2', 'OCM2')
        n_M_DD2_OCM1 = AMP_SetSpec('DD2', 'OCM1')
        n_M_DD2_OCM0 = AMP_SetSpec('DD2', 'OCM0')
        n_M_DD2_OCP1 = AMP_SetSpec('DD2', 'OCP1')
        n_M_DD2_OCP2 = AMP_SetSpec('DD2', 'OCP2')
        n_M_DD2_OCP3 = AMP_SetSpec('DD2', 'OCP3')
        n_M_DD2_OCP4 = AMP_SetSpec('DD2', 'OCP4')
        n_M_DD2_OCP5 = AMP_SetSpec('DD2', 'OCP5')
        n_M_DD2_OCP6 = AMP_SetSpec('DD2', 'OCP6')
      endif
      n_N_DD2_1  = AMP_SetSpec('DD2','1' )

      n_M_DS2_SU = AMP_SetSpec('DS2','SU')
      n_M_DS2_DU = AMP_SetSpec('DS2','DU')
      if (tracers_amp_m9) then
        n_M_DS2_OCM2 = AMP_SetSpec('DS2', 'OCM2')
        n_M_DS2_OCM1 = AMP_SetSpec('DS2', 'OCM1')
        n_M_DS2_OCM0 = AMP_SetSpec('DS2', 'OCM0')
        n_M_DS2_OCP1 = AMP_SetSpec('DS2', 'OCP1')
        n_M_DS2_OCP2 = AMP_SetSpec('DS2', 'OCP2')
        n_M_DS2_OCP3 = AMP_SetSpec('DS2', 'OCP3')
        n_M_DS2_OCP4 = AMP_SetSpec('DS2', 'OCP4')
        n_M_DS2_OCP5 = AMP_SetSpec('DS2', 'OCP5')
        n_M_DS2_OCP6 = AMP_SetSpec('DS2', 'OCP6')
      endif
      n_N_DS2_1  = AMP_SetSpec('DS2','1' )
    end if

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_SSA_SU = AMP_SetSpec('SSA','SU')
      n_M_SSA_SS = AMP_SetSpec('SSA','SS')
      if (tracers_amp_m9) then
        n_M_SSA_OCM2 = AMP_SetSpec('SSA', 'OCM2')
        n_M_SSA_OCM1 = AMP_SetSpec('SSA', 'OCM1')
        n_M_SSA_OCM0 = AMP_SetSpec('SSA', 'OCM0')
        n_M_SSA_OCP1 = AMP_SetSpec('SSA', 'OCP1')
        n_M_SSA_OCP2 = AMP_SetSpec('SSA', 'OCP2')
        n_M_SSA_OCP3 = AMP_SetSpec('SSA', 'OCP3')
        n_M_SSA_OCP4 = AMP_SetSpec('SSA', 'OCP4')
        n_M_SSA_OCP5 = AMP_SetSpec('SSA', 'OCP5')
        n_M_SSA_OCP6 = AMP_SetSpec('SSA', 'OCP6')
      endif
      n_N_SSA_1  = AMP_SetSpec('SSA','1' )

      n_M_SSC_SS = AMP_SetSpec('SSC','SS')
      if (tracers_amp_m9) then
        n_M_SSC_OCM2 = AMP_SetSpec('SSC', 'OCM2')
        n_M_SSC_OCM1 = AMP_SetSpec('SSC', 'OCM1')
        n_M_SSC_OCM0 = AMP_SetSpec('SSC', 'OCM0')
        n_M_SSC_OCP1 = AMP_SetSpec('SSC', 'OCP1')
        n_M_SSC_OCP2 = AMP_SetSpec('SSC', 'OCP2')
        n_M_SSC_OCP3 = AMP_SetSpec('SSC', 'OCP3')
        n_M_SSC_OCP4 = AMP_SetSpec('SSC', 'OCP4')
        n_M_SSC_OCP5 = AMP_SetSpec('SSC', 'OCP5')
        n_M_SSC_OCP6 = AMP_SetSpec('SSC', 'OCP6')
      endif
      n_N_SSC_1  = AMP_SetSpec('SSC','1' )
    end if

    if (tracers_amp_m4 .or. &
        tracers_amp_m8) then ! cases not tested
      n_M_SSS_SU = AMP_SetSpec('SSS','SU') ! need rundeck with these
      n_M_SSS_SS = AMP_SetSpec('SSS','SS') ! settings
    end if

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_OCC_SU = AMP_SetSpec('OCC','SU')
      n_M_OCC_OC = AMP_SetSpec('OCC','OC')
      if (tracers_amp_m9) then
        n_M_OCC_OCM2 = AMP_SetSpec('OCC', 'OCM2')
        n_M_OCC_OCM1 = AMP_SetSpec('OCC', 'OCM1')
        n_M_OCC_OCM0 = AMP_SetSpec('OCC', 'OCM0')
        n_M_OCC_OCP1 = AMP_SetSpec('OCC', 'OCP1')
        n_M_OCC_OCP2 = AMP_SetSpec('OCC', 'OCP2')
        n_M_OCC_OCP3 = AMP_SetSpec('OCC', 'OCP3')
        n_M_OCC_OCP4 = AMP_SetSpec('OCC', 'OCP4')
        n_M_OCC_OCP5 = AMP_SetSpec('OCC', 'OCP5')
        n_M_OCC_OCP6 = AMP_SetSpec('OCC', 'OCP6')
      endif
      n_N_OCC_1  = AMP_SetSpec('OCC','1' )
    endif

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_BC1_SU = AMP_SetSpec('BC1','SU')
      n_M_BC1_BC = AMP_SetSpec('BC1','BC')
      if (tracers_amp_m9) then
        n_M_BC1_OCM2 = AMP_SetSpec('BC1', 'OCM2')
        n_M_BC1_OCM1 = AMP_SetSpec('BC1', 'OCM1')
        n_M_BC1_OCM0 = AMP_SetSpec('BC1', 'OCM0')
        n_M_BC1_OCP1 = AMP_SetSpec('BC1', 'OCP1')
        n_M_BC1_OCP2 = AMP_SetSpec('BC1', 'OCP2')
        n_M_BC1_OCP3 = AMP_SetSpec('BC1', 'OCP3')
        n_M_BC1_OCP4 = AMP_SetSpec('BC1', 'OCP4')
        n_M_BC1_OCP5 = AMP_SetSpec('BC1', 'OCP5')
        n_M_BC1_OCP6 = AMP_SetSpec('BC1', 'OCP6')
      endif
      n_N_BC1_1  = AMP_SetSpec('BC1','1' )
    endif

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_BC2_SU = AMP_SetSpec('BC2','SU')
      n_M_BC2_BC = AMP_SetSpec('BC2','BC')
      if (tracers_amp_m9) then
        n_M_BC2_OCM2 = AMP_SetSpec('BC2', 'OCM2')
        n_M_BC2_OCM1 = AMP_SetSpec('BC2', 'OCM1')
        n_M_BC2_OCM0 = AMP_SetSpec('BC2', 'OCM0')
        n_M_BC2_OCP1 = AMP_SetSpec('BC2', 'OCP1')
        n_M_BC2_OCP2 = AMP_SetSpec('BC2', 'OCP2')
        n_M_BC2_OCP3 = AMP_SetSpec('BC2', 'OCP3')
        n_M_BC2_OCP4 = AMP_SetSpec('BC2', 'OCP4')
        n_M_BC2_OCP5 = AMP_SetSpec('BC2', 'OCP5')
        n_M_BC2_OCP6 = AMP_SetSpec('BC2', 'OCP6')
      endif
      n_N_BC2_1  = AMP_SetSpec('BC2','1' )
    endif

    if (tracers_amp_m1 .or. &
        tracers_amp_m5) then
      n_M_BC3_SU = AMP_SetSpec('BC3','SU')
      n_M_BC3_BC = AMP_SetSpec('BC3','BC')
      n_N_BC3_1  = AMP_SetSpec('BC3','1' )
    end if

    if (tracers_amp_m2 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_OCS_SU = AMP_SetSpec('OCS','SU')
      n_M_OCS_OC = AMP_SetSpec('OCS','OC')
      if (tracers_amp_m9) then
        n_M_OCS_OCM2 = AMP_SetSpec('OCS', 'OCM2')
        n_M_OCS_OCM1 = AMP_SetSpec('OCS', 'OCM1')
        n_M_OCS_OCM0 = AMP_SetSpec('OCS', 'OCM0')
        n_M_OCS_OCP1 = AMP_SetSpec('OCS', 'OCP1')
        n_M_OCS_OCP2 = AMP_SetSpec('OCS', 'OCP2')
        n_M_OCS_OCP3 = AMP_SetSpec('OCS', 'OCP3')
        n_M_OCS_OCP4 = AMP_SetSpec('OCS', 'OCP4')
        n_M_OCS_OCP5 = AMP_SetSpec('OCS', 'OCP5')
        n_M_OCS_OCP6 = AMP_SetSpec('OCS', 'OCP6')
      endif
      n_N_OCS_1  = AMP_SetSpec('OCS','1' )
    end if

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m6) then
      n_M_DBC_SU = AMP_SetSpec('DBC','SU') 
      n_M_DBC_BC = AMP_SetSpec('DBC','BC')
      n_M_DBC_DU = AMP_SetSpec('DBC','DU') 
      n_N_DBC_1  = AMP_SetSpec('DBC','1' ) 
    end if

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_BOC_SU = AMP_SetSpec('BOC','SU')
      n_M_BOC_BC = AMP_SetSpec('BOC','BC')
      n_M_BOC_OC = AMP_SetSpec('BOC','OC')
      if (tracers_amp_m9) then
        n_M_BOC_OCM2 = AMP_SetSpec('BOC', 'OCM2')
        n_M_BOC_OCM1 = AMP_SetSpec('BOC', 'OCM1')
        n_M_BOC_OCM0 = AMP_SetSpec('BOC', 'OCM0')
        n_M_BOC_OCP1 = AMP_SetSpec('BOC', 'OCP1')
        n_M_BOC_OCP2 = AMP_SetSpec('BOC', 'OCP2')
        n_M_BOC_OCP3 = AMP_SetSpec('BOC', 'OCP3')
        n_M_BOC_OCP4 = AMP_SetSpec('BOC', 'OCP4')
        n_M_BOC_OCP5 = AMP_SetSpec('BOC', 'OCP5')
        n_M_BOC_OCP6 = AMP_SetSpec('BOC', 'OCP6')
      endif
      n_N_BOC_1  = AMP_SetSpec('BOC','1' )
    end if

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_BCS_SU = AMP_SetSpec('BCS','SU')
      n_M_BCS_BC = AMP_SetSpec('BCS','BC')
      if (tracers_amp_m9) then
        n_M_BCS_OCM2 = AMP_SetSpec('BCS', 'OCM2')
        n_M_BCS_OCM1 = AMP_SetSpec('BCS', 'OCM1')
        n_M_BCS_OCM0 = AMP_SetSpec('BCS', 'OCM0')
        n_M_BCS_OCP1 = AMP_SetSpec('BCS', 'OCP1')
        n_M_BCS_OCP2 = AMP_SetSpec('BCS', 'OCP2')
        n_M_BCS_OCP3 = AMP_SetSpec('BCS', 'OCP3')
        n_M_BCS_OCP4 = AMP_SetSpec('BCS', 'OCP4')
        n_M_BCS_OCP5 = AMP_SetSpec('BCS', 'OCP5')
        n_M_BCS_OCP6 = AMP_SetSpec('BCS', 'OCP6')
      endif
      n_N_BCS_1  = AMP_SetSpec('BCS','1' )
    end if

    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      n_M_MXX_SU = AMP_SetSpec('MXX','SU')
      n_M_MXX_BC = AMP_SetSpec('MXX','BC')
      n_M_MXX_OC = AMP_SetSpec('MXX','OC')
      n_M_MXX_DU = AMP_SetSpec('MXX','DU')
      n_M_MXX_SS = AMP_SetSpec('MXX','SS')
      if (tracers_amp_m9) then
        n_M_MXX_OCM2 = AMP_SetSpec('MXX', 'OCM2')
        n_M_MXX_OCM1 = AMP_SetSpec('MXX', 'OCM1')
        n_M_MXX_OCM0 = AMP_SetSpec('MXX', 'OCM0')
        n_M_MXX_OCP1 = AMP_SetSpec('MXX', 'OCP1')
        n_M_MXX_OCP2 = AMP_SetSpec('MXX', 'OCP2')
        n_M_MXX_OCP3 = AMP_SetSpec('MXX', 'OCP3')
        n_M_MXX_OCP4 = AMP_SetSpec('MXX', 'OCP4')
        n_M_MXX_OCP5 = AMP_SetSpec('MXX', 'OCP5')
        n_M_MXX_OCP6 = AMP_SetSpec('MXX', 'OCP6')
      endif
      n_N_MXX_1  = AMP_SetSpec('MXX','1' )
    endif

    call  H2SO4_setSpec('H2SO4')
    call  DMS_setSpec('DMS')  ! duplicate with Koch
    call  SO2_setSpec('SO2')  ! duplicate with Koch
    if (.not. tracers_special_shindell .or. coupled_chem.le.0) then
      call  H2O2_s_setSpec('H2O2_s') ! duplicate with Koch
    endif
    if (tracers_amp_m1 .or. &
        tracers_amp_m2 .or. &
        tracers_amp_m3 .or. &
        tracers_amp_m4 .or. &
        tracers_amp_m5 .or. &
        tracers_amp_m6 .or. &
        tracers_amp_m7 .or. &
        tracers_amp_m8 .or. &
        tracers_amp_m9 .or. &
        tracers_amp_m10) then
      call  NH3_setSpec('NH3')  ! duplicate with nitrate
      if (tracers_aerosols_koch.or.tracers_aerosols_seasalt) then
        call stop_model('contradictory tracer specs', 255)
      end if
      if (tracers_nitrate) then
        call stop_model('contradictory tracer specs', 255)
      end if
    endif

! create AMP maps
    ntmAMP = ntmAMPe-ntmAMPi+1
    allocate(AMP_AERO_MAP(ntmAMP))
    allocate(AMP_NUMB_MAP(ntmAMP))
    allocate(AMP_MODES_MAP(ntmAMP))
    allocate(AMP_trm_nm1(ntmAMP))
    allocate(AMP_trm_nm2(ntmAMP))
    AMP_NUMB_MAP=0 ! only this needs to be zeroed out

#ifdef TRACERS_AMP_M1
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,16,17,18,19,20, &
      21,22,   24,25,26,27,28,29,30, & ! skip M_SSC_SU
      31,32,33,34,35,36,37,38,39,40, &
      41,42,43,44,45,46,47,48,49,50, &
      51,52,53,54                  /)
#elif defined TRACERS_AMP_M2
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,16,17,18,19,20, &
      21,      24,   26,27,28,29,30, & ! skip N_SSA_1,M_SSC_SU,N_SSC_1
      31,32,33,34,35,36,37,38,39,40, &
      41,42,43,44,45,46,47,48,49,50, &
      51,52,53,54                  /)
#elif defined TRACERS_AMP_M3
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15      ,18   ,20, & ! skip N_SSA_1,M_SSC_SU,N_SSC_1
      21,22,23,24,25,26,27,28,29,30, &
      31,32,33,34,35,36,37,38,39,40, &
      41,42,43,44                  /)
#elif defined TRACERS_AMP_M4
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,16,17,18,19,    & ! skip N_SSS_1
      21,22,23,24,25,26,27,28,29,30, &
      31,32,33,34,35               /)
#elif defined TRACERS_AMP_M5
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,      18,   20, & ! skip N_SSA_1,M_SSC_SU,N_SSC_1
      21,22,23,24,25,26,27,28,29,30, &
      31,32,33,34,35,36,37,38,39,40, &
      41,42,43,44,45,46,47,48      /)
#elif defined TRACERS_AMP_M6
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,      18,   20, & ! skip N_SSA_1,M_SSC_SU,N_SSC_1
      21,22,23,24,25,26,27,28,29,30, &
      31,32,33,34,35,36,37,38,39,40, &
      41,42,43,44,45,46,47,48      /)
#elif defined TRACERS_AMP_M7
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,      18,   20, & ! skip N_SSA_1,M_SSC_SU,N_SSC_1
      21,22,23,24,25,26,27,28,29,30, &
      31,32,33,34,35,36,37,38      /)
#elif defined TRACERS_AMP_M8
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,   15,16,17,18,19,20, & ! skip N_SSS_1
      21,22,23,24,25,26,27,28,29   /)
#elif defined TRACERS_AMP_M9
    AMP_AERO_MAP=(/ &
        1,  2,  3,  4,  5,  6,  7,  8,  9, 10, &
       11, 12, 13, 14, 15, 16, 17, 18, 19, 20, &
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, &
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, &
       41, 42, 43, 44, 45, 46, 47, 48, 49, 50, &
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, &
       61, 62, 63, 64, 65, 66, 67, 68, 69, 70, &
       71, 72, 73, 74, 75, 76,     78, 79, 80, & ! skip M_SSC_SU
       81, 82, 83, 84, 85, 86, 87, 88, 89, 90, &
       91, 92, 93, 94, 95, 96, 97, 98, 99,100, &
      101,102,103,104,105,106,107,108,109,110, &
      111,112,113,114,115,116,117,118,119,120, &
      121,122,123,124,125,126,127,128,129,130, &
      131,132,133,134,135,136,137,138,139,140, &
      141,142,143,144,145,146,147,148,149,150, &
      151,152,153,154,155,156,157,158,159,160, &
      161,162,163,164,165,166,167,168,169,170, &
      171,172,173,174,175,176                /)
#elif defined TRACERS_AMP_M10
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, &
      11,12,13,14,15,16,17,18,19,20, &
      21,22,   24,25,26,27,28,29,30, & ! skip M_SSC_SU
      31,32,33,34,35,36,37,38,39,40, &
      41,42,43,44,45,46,47,48,49,50 /)
#elif defined TRACERS_AMP_M11
    AMP_AERO_MAP=(/ &
      1 ,2 ,3 ,4, 5/)
#else
    call stop_model('AMP_AERO_MAP needs to be defined.', 255)
#endif

    imode=1
    itr=0 ! this will become the index of the first mass in a population
    do n=ntmAMPi,ntmAMPe
      nAMP=n-ntmAMPi+1
      trname_curr=trim(trname(n))

      select case (trname_curr(1:2))
      case ('N_')
        AMP_NUMB_MAP(nAMP)=maxval(AMP_NUMB_MAP)+1
      end select

      select case (trname_curr)
      case ('M_NO3','M_NH4','M_H2O')
        AMP_MODES_MAP(nAMP)=0
      case default
        AMP_MODES_MAP(nAMP)=imode
        select case (trname_curr(1:2))
        case ('N_')
          imode=imode+1 ! increase index for next mode
        end select
      end select

      select case (trname_curr)
      case ('M_NO3','M_NH4','M_H2O')
        AMP_trm_nm1(nAMP)=0
        AMP_trm_nm2(nAMP)=0
      case default
        if (itr==0) then
          itr=n
        endif
        AMP_trm_nm1(nAMP)=itr
        select case (trname_curr(1:2))
        case ('N_')
          itr=0 ! reset for next mode
          do nn=1,nAMP
            if (AMP_trm_nm1(nn)==AMP_trm_nm1(nAMP)) then
              AMP_trm_nm2(nn)=n-1
            endif
          enddo
        end select
      end select
    enddo

!------------------------------------------------------------------------------
  contains
!------------------------------------------------------------------------------

    subroutine H2SO4_setSpec(name)
      character(len=*), intent(in) :: name
      n = oldAddTracer(name)
      n_H2SO4 = n
      call set_ntm_power(n, -11)
      call set_tr_mm(n, 98.d0)
      call set_trpdens(n, DENS_SULF)
      call set_trradius(n, DG_ACC * .5d-6)
      call set_fq_aer(n, SOLU_ACC)
      call set_tr_wd_type(n, npart)
      call set_has_chemistry(n, .true.)
    end subroutine H2SO4_setSpec

    subroutine M_NO3_setSpec(name)
      character(len=*), intent(in) :: name
      n = oldAddTracer(name)
      n_M_NO3 = n
      if (ntmAMPi==0) ntmAMPi=n
      ntmAMPe=n
      call set_ntm_power(n, -11)
      call set_tr_mm(n, 62.d0)
      call set_trpdens(n, 1.7d3)
      call set_trradius(n, 3.d-7 ) !m
      call set_fq_aer(n, 1.d0)  !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_has_chemistry(n, .true.)
    end subroutine M_NO3_setSpec

    subroutine M_NH4_setSpec(name)
      character(len=*), intent(in) :: name
      n = oldAddTracer(name)
      n_M_NH4 = n
      if (ntmAMPi==0) ntmAMPi=n
      ntmAMPe=n
      call set_ntm_power(n, -11)
      call set_tr_mm(n, 18.d0)
      call set_trpdens(n, 1.7d3)
      call set_trradius(n, 3.d-7)
      call set_fq_aer(n, 1.d+0)
      call set_tr_wd_type(n, npart)
      call set_has_chemistry(n, .true.)
    end subroutine M_NH4_setSpec

    subroutine M_H2O_setSpec(name)
      character(len=*), intent(in) :: name
      n = oldAddTracer(name)
      n_M_H2O = n
      if (ntmAMPi==0) ntmAMPi=n
      ntmAMPe=n
      call set_ntm_power(n, -11)
      call set_tr_mm(n, mwat)
      call set_trpdens(n, 1.d3)
      call set_trradius(n, 3.d-7)
      call set_fq_aer(n, 1.d+0)
      call set_tr_wd_type(n, npart) !nWater
    end subroutine M_H2O_setSpec

!------------------------------------------------------------------------------
    function AMP_SetSpec(mode, component) result (tracerIndex)
!------------------------------------------------------------------------------
      use OldTracer_mod, only: om2oc, set_om2oc
      use Dictionary_mod, only: sync_param
      implicit none
      character(len=*), intent(in) :: mode
      character(len=*), intent(in) :: component
      type (Tracer), pointer :: t
      ! local variables
      character(len=1) :: prefix
      character(len=64) :: tracerName
      integer :: tracerIndex
      real*8 :: tmp
      integer :: lc ! length of component string or 2, whichever is smaller
      integer :: i

      prefix = getTracerPrefix(component)
      tracerName = prefix // "_" // mode // "_" // component
      tracerIndex = oldAddTracer(trim(tracerName))

! create array with number concentration indices, based on ntm
      if (trim(component) == '1') then
        do i=1,nmodes ! brute force, but only happens once
          if (iNamp(i)==0) then
            iNamp(i)=tracerIndex
            exit
          endif
        enddo
      endif

      ntmAMPe = tracerIndex

      lc=min(2,len(component)) ! this is to convert OCM2,OCM1 etc. to OC for M9
      if (trim(component) == 'SU') then
        t => tracers%getReference(trim(tracerName))
        call t%insert('SO4',.true.)
      else if (component(1:lc) == 'OC') then
        call set_om2oc(tracerIndex, 1.4d0)
        tmp = om2oc(tracerIndex)
        call sync_param(trim(tracerName)//"_om2oc",tmp)
        call set_om2oc(tracerIndex, tmp)
      else if (trim(component) == 'BC') then
        t => tracers%getReference(trim(tracerName))
        call t%insert('BC',.true.)
      endif
      call set_ntm_power(tracerIndex, -11)
      if (component(1:lc) == 'OC') then
        tmp = getMolecularMass(component(1:lc)) * om2oc(tracerIndex)
      else
        tmp = getMolecularMass(component(1:lc))
      endif
      call set_tr_mm(tracerIndex, tmp)
      call set_trpdens(tracerIndex, getDensity(component(1:lc)))
      call set_trradius(tracerIndex, getRadius(mode))
      call set_fq_aer(tracerIndex, getSolubility(mode))
      call set_tr_wd_type(tracerIndex, nPART)

    end function AMP_SetSpec

!------------------------------------------------------------------------------
    function getTracerPrefix(component) result (unitPrefix)
!------------------------------------------------------------------------------
      implicit none
      character(len=*), intent(in) :: component
      character(len=1) :: unitPrefix

      select case (trim(component))
      case ('1')
        unitPrefix = 'N'        ! number density
      case default
        unitPrefix = 'M'        ! mixing ratio
      end select

    end function getTracerPrefix

!------------------------------------------------------------------------------
    function getMolecularMass(component) result (molecularMass)
!------------------------------------------------------------------------------
      implicit none
      character(len=*), intent(in) :: component
      ! local variables
      real(8) :: molecularMass

      select case (trim(component))
      case ('SU')
        molecularMass = SULF_MolecMass
      case ('DU')
        molecularMass = DUST_MolecMass
      case ('SS')
        molecularMass = SEAS_MolecMass
      case ('BC')
        molecularMass = BCAR_MolecMass
      case ('OC')
        molecularMass = OCAR_MolecMass
      case ('1')
        molecularMass = 1.0d0
      case default
        call stop_model('Incorrect molecular mass choice', 255)
      end select

    end function getMolecularMass

!------------------------------------------------------------------------------
    function getDensity(component)  result (density)
!------------------------------------------------------------------------------
      implicit none
      character(len=*), intent(in) :: component
      real(8) :: density

      select case (trim(component))
      case ('SU')
        density = DENS_SULF
      case ('DU')
        density = DENS_DUST
      case ('SS')
        density = DENS_SEAS
      case ('BC')
        density = DENS_BCAR
      case ('OC')
        density = DENS_OCAR
      case ('1')
        density = 1.0d0
      case default
        call stop_model('Incorrect density choice', 255)
      end select

    end function getDensity

!------------------------------------------------------------------------------
    function getRadius(mode)  result (radius)
!------------------------------------------------------------------------------
      implicit none
      character(len=*), intent(in) :: mode
      real(8) :: radius

      select case (trim(mode))
      case ('AKK')
        radius = RG_AKK
      case ('ACC')
        radius = RG_ACC
      case ('DD1')
        radius = RG_DD1
      case ('DS1')
        radius = RG_DS1
      case ('DD2')
        radius = RG_DD2
      case ('DS2')
        radius = RG_DS2
      case ('SSA')
        radius = RG_SSA
      case ('SSC')
        radius = RG_SSC
      case ('SSS')
        radius = RG_SSS
      case ('OCC')
        radius = RG_OCC
      case ('BC1')
        radius = RG_BC1
      case ('BC2')
        radius = RG_BC2
      case ('BC3')
        radius = RG_BC3
      case ('DBC')
        radius = RG_DBC
      case ('BOC')
        radius = RG_BOC
      case ('BCS')
        radius = RG_BCS
      case ('MXX')
        radius = RG_MXX
      case ('OCS')
        radius = RG_OCS
      case default
        call stop_model('Incorrect radius choice', 255)
      end select

    end function getRadius

!------------------------------------------------------------------------------
    function getSolubility(mode)  result (solubility)
!------------------------------------------------------------------------------
      implicit none
      character(len=*), intent(in) :: mode
      real(8) :: solubility

      select case (trim(mode))
      case ('AKK')
        solubility = SOLU_AKK
      case ('ACC')
        solubility = SOLU_ACC
      case ('DD1')
        solubility = SOLU_DD1
      case ('DS1')
        solubility = SOLU_DS1
      case ('DD2')
        solubility = SOLU_DD2
      case ('DS2')
        solubility = SOLU_DS2
      case ('SSA')
        solubility = SOLU_SSA
      case ('SSC')
        solubility = SOLU_SSC
      case ('SSS')
        solubility = SOLU_SSS
      case ('OCC')
        solubility = SOLU_OCC
      case ('BC1')
        solubility = SOLU_BC1
      case ('BC2')
        solubility = SOLU_BC2
      case ('BC3')
        solubility = SOLU_BC3
      case ('DBC')
        solubility = SOLU_DBC
      case ('BOC')
        solubility = SOLU_BOC
      case ('BCS')
        solubility = SOLU_BCS
      case ('MXX')
        solubility = SOLU_MXX
      case ('OCS')
        solubility = SOLU_OCS
      case default
        call stop_model('Incorrect solubility choice', 255)
      end select

    end function getSolubility

  end subroutine AMP_initMetadata

end module AmpTracersMetadata_mod
