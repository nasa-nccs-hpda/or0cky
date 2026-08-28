! SOCRATES Configuration File. It is used to couple the quantities
! used by SOCRATES to the corresponding parameters in ModelE.

MODULE rad_ccf

! Description:
!   Module containing settings of standard constants used within the
!   radiation scheme. This replaces a number of separate constant
!   configuration files. The original file names are indicated at the
!   head of each section.

  USE realtype_rd, ONLY: RealK
  USE mathematicalconstants_mod, ONLY: pi
  USE constant, ONLY: mair, grav, pN2, gasc, bymrat, k_boltzmann, avog, &
    modelE_h_planck=>h_planck, modelE_c_light=>c_light
  USE gas_list_pcf, ONLY: ip_n2, molar_weight

! Diffusivity for elsasser's scheme.
  REAL (RealK), PARAMETER :: elsasser_factor = 1.66e+00_RealK
!   Diffusivity factor for elsasser's scheme

! Diffusivity factor for use with equivalent extinction
  REAL (RealK), PARAMETER :: diffusivity_factor_minor = 1.66e+00_RealK
!   Minor diffusivity factor

! Mathematical constants.
  PUBLIC :: pi
!   Value of pi from ModelE mathematicalconstants_mod module

! Physical constants for the Earth.
  REAL (RealK), PARAMETER :: mol_weight_air  = mair*1.0E-03_RealK
!   Molar weight of dry air
  REAL (RealK), PARAMETER :: n2_mass_frac = pN2* &
      molar_weight(ip_n2)*1.0E-03_RealK/mol_weight_air
!   Mass fraction of nitrogen from ModelE constant module
  REAL (RealK), PARAMETER :: grav_acc           = grav
!   Acceleration due to gravity from ModelE constant module
  REAL (RealK), PARAMETER :: r_gas              = gasc
!   Universal gas constant from ModelE constant module
  REAL (RealK), PARAMETER :: r_gas_dry          = r_gas/mol_weight_air
!   Gas constant for dry air
  REAL (RealK), PARAMETER :: ratio_molar_weight = bymrat
!   Molecular weight of dry air/molecular weight of water from ModelE constant
!   module
  REAL (RealK), PARAMETER :: r = r_gas_dry
  REAL (RealK), PARAMETER :: c_virtual = ratio_molar_weight - 1.0_RealK
  REAL (RealK), PARAMETER :: repsilon = 1.0_RealK / ratio_molar_weight

! Physical constants.
  PUBLIC :: k_boltzmann
!   Boltzmann's constant from ModelE constant module
  REAL (RealK), PARAMETER :: n_avogadro  = avog
!   Avogadro's number from ModelE constant module

! added for compatibility with SOCRATES 2111
! ------------------------------------------------------------------
! physical_constants_pp_ccf
! ------------------------------------------------------------------
! Module setting physical constants.
  REAL (RealK), PARAMETER :: h_planck         = modelE_h_planck
!   Planck's constant (J s)
  REAL (RealK), PARAMETER :: c_light          = modelE_c_light
!   Speed of light in a vacuum (m s-1)


END MODULE rad_ccf
