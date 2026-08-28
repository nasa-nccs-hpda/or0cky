! Module containing derive type definitions for use in planet radiation.

MODULE def_planetstr

  USE realtype_rd, ONLY: RealK
  USE constant,    ONLY: solar_t_effective, solar_radius

  IMPLICIT NONE

! Define derived type to keep high resolution solar spectrum
  TYPE StrSolarSpec

    INTEGER :: &
        n_points = 0
!         Number of points in the spectrum
    REAL (RealK), ALLOCATABLE :: &
        wavelength(:), &
!         Wavelength at which the spectral irradiance is specified
        irrad(:)
!         Solar spectral irradiance in units of Wm-2.m-1
    REAL (RealK) :: &
        t_effective = solar_t_effective, &
!         Effective solar temperature
        radius = solar_radius
!         Radius at the photosphere

  END TYPE StrSolarSpec

! Define derived type to keep tabulated aerosol optical properties
  TYPE StrAerData

!   Dimension sizes
    INTEGER :: &
        n_component, &
!         Number of aerosol components
        n_radius_max, &
!         Number of radii
        n_humidity, &
!         Number of humidities
        n_band
!         Number of bands

!   Dimensions
    INTEGER, ALLOCATABLE :: &
        i_component(:), &
!         Aerosol component index defined in rad_pcf
        band(:), &
!         Band index
        n_radius(:)
!         Number of radii in table
    REAL(RealK), ALLOCATABLE :: &
        radius_eff(:, :, :), &
!         Effective radius
        humidity(:)
!         Humidities

!   Data
    LOGICAL, ALLOCATABLE :: &
        l_humidity(:)
!         Logical for humidity dependence of optical properties
    INTEGER, ALLOCATABLE :: &
        i_component_map(:)
!         Maps component index in i_component (as defined in rad_pcf) into
!         component index used in data arrays
    REAL(RealK), ALLOCATABLE :: &
        radius_eff_dry(:, :), &
!         Dry effective radius
        k_abs(:, :, :, :), &
!         Absorption coefficient
        k_scat(:, :, :, :), &
!         Scattering coefficient
        g_asym(:, :, :, :), &
!         Asymmetry parameter
        density(:), &
!         Bulk density of aerosol components
        wavelength(:)
!         Wavelength of diagnostic aerosol optical properties

    CHARACTER(LEN=256) :: &
        spectral_file
!         Name of spectral file to be used with aerosol data
    CHARACTER(LEN=4) :: &
        spectral_region
!         Indicates if data is for short-wave (also contains diagnostic
!         quantities) or long-wave. Either 'sw' or 'lw'.
    LOGICAL :: &
        l_diag
!         Flag for data in table being for diagnostics

  END TYPE StrAerData

! Define derived type to keep diagnostic aerosol optical properties
  TYPE StrAerDiag

    REAL(RealK), ALLOCATABLE :: &
        tau_ext(:, :), &
!         Extinction optical depth
        tau_scat(:, :), &
!         Scattering optical depth
        g_asym(:, :)
!         Asymmetry parameter

  END TYPE StrAerDiag

CONTAINS

! Subroutine to allocate arrays in StrAerDiag derived type
SUBROUTINE allocate_aer_diag(aer_diag, nd_layer, nd_band)

! Input arguments
  TYPE(StrAerDiag), INTENT(INOUT) :: &
      aer_diag
!       Aerosol diagnostics
  INTEGER :: &
      nd_layer, &
!       Size allocated for layers
      nd_band
!       Size allocated for bands

  IF (.NOT. ALLOCATED(aer_diag%tau_ext)) &
      ALLOCATE(aer_diag%tau_ext(nd_layer, nd_band))
  IF (.NOT. ALLOCATED(aer_diag%tau_scat)) &
      ALLOCATE(aer_diag%tau_scat(nd_layer, nd_band))
  IF (.NOT. ALLOCATED(aer_diag%g_asym)) &
      ALLOCATE(aer_diag%g_asym(nd_layer, nd_band))

END SUBROUTINE allocate_aer_diag

! Subroutine to deallocate arrays in StrAerDiag derived type
SUBROUTINE deallocate_aer_diag(aer_diag)

! Input arguments
  TYPE(StrAerDiag), INTENT(INOUT) :: &
      aer_diag
!       Aerosol diagnostics

  IF (ALLOCATED(aer_diag%tau_ext)) &
      DEALLOCATE(aer_diag%tau_ext)
  IF (ALLOCATED(aer_diag%tau_scat)) &
      DEALLOCATE(aer_diag%tau_scat)
  IF (ALLOCATED(aer_diag%g_asym)) &
      DEALLOCATE(aer_diag%g_asym)

END SUBROUTINE deallocate_aer_diag

END MODULE def_planetstr