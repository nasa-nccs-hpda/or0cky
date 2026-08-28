#include "rundeck_opts.h"

MODULE planet_rad

! This is the module containing the interface between ModelE and SOCRATES. All
! internal and SOCRATES variables are in SI units.

! SOCRATES modules
  USE realtype_rd,      ONLY: RealK
  USE rad_pcf
  USE gas_list_pcf
  USE def_spectrum,     ONLY: StrSpecData
  USE def_atm,          ONLY: StrAtm
  USE def_dimen,        ONLY: StrDim
  USE def_bound,        ONLY: StrBound
  USE def_control,      ONLY: StrCtrl
  USE def_cld,          ONLY: StrCld
  USE def_aer,          ONLY: StrAer
  USE def_out,          ONLY: StrOut

! Planet radiation modules
  USE def_planetstr,    ONLY: StrSolarSpec, StrAerData, StrAerDiag

! GCM modules
  USE atm_com,          ONLY: lm_req
  USE resolution,       ONLY: lm_gcm => lm

! Albedo module
  USE planet_alb,       ONLY: n_veg

  IMPLICIT NONE

!-------------------------------------------------------------------------------
! User determined control variables
!-------------------------------------------------------------------------------

! Solar constant (no variability yet)
  REAL(RealK) :: &
      solar_constant = 1360.67_RealK
!       Solar constant in W/m**2 at 1 AU

! Cloud treatment
  LOGICAL :: &
#if defined(DEBUG_RADIATION) || defined(DEBUG_RADIATION_KEEP)
      l_cloud = .FALSE.
#else
      l_cloud = .TRUE.
#endif
!       Flag for including radiative properties of clouds
  INTEGER :: &
      i_cloud_ice_size_scheme = 1
!       Scheme for calculating effective ice particle size
!       (GISS == 1, UM == 2)

! Scaling of mass mixing ratios with control%i_inhom == ip_scaling
  REAL(RealK), PARAMETER :: &
      inhom_cloud_adj_st  = 1.0_RealK, &
!       Cloud inhomogeneity adjustment for stratiform clouds
      inhom_cloud_adj_cnv = 1.0_RealK
!       Cloud inhomogeneity adjustment for convective clouds

! Gaseous absorption
  LOGICAL, PARAMETER :: &
      l_set_major_abs = .TRUE., &
!       Set major absorber in each band automatically (for use with equivalent
!       extinction)
      l_allow_cont_major = .FALSE.
!       Allow continua to become the major absorber

!-------------------------------------------------------------------------------
! End of user determined control variables
!-------------------------------------------------------------------------------

! Solar spectrum:
  TYPE (StrSolarSpec) :: solar_spec

! Control options:
  TYPE(StrCtrl) :: &
      control_lw, &
!       Long-wave control options
      control_sw
!       Short-wave control options

! Dimensions:
  TYPE(StrDim) :: dimen

! Spectral information:
  TYPE(StrSpecData) :: &
      spectrum_lw, &
!       Long-wave spectral data
      spectrum_sw
!       Short-wave spectral data

! Ease access to number of bands
  INTEGER :: &
      n_band_lw, &
!       Number of long-wave bands
      n_band_sw, &
!       Number of short-wave bands
      n_band_diag
!       Number of bands for aerosol diagnostics

! Atmospheric properties:
  TYPE(StrAtm) :: &
      atm_lw, &
!       Atmospheric properties for long-wave call
      atm_sw
!       Atmospheric properties for short-wave call

! Cloud properties:
  TYPE(StrCld) :: &
      cld_lw, &
!       Cloud properties for long-wave call
      cld_sw
!       Cloud properties for short-wave call

! Aerosol properties:
  TYPE(StrAer) :: &
      aer_lw, &
!       Aerosol properties for long-wave call
      aer_sw
!       Aerosol properties for short-wave call

! Boundary conditions:
  TYPE(StrBound) :: &
      bound_lw, &
!       Boundary condition for long-wave call
      bound_sw
!       Boundary conditions for short-wave call

! Output from radiation code
  TYPE(StrOut) :: &
      radout_lw, &
!       Output from long-wave call
      radout_sw
!       Output from short-wave call

! Input files
  CHARACTER(LEN=128) :: &
      dir_solar_spec = '/discover/nobackup/projects/giss/' // &
          'socrates/stellar_spectra', &
!       Directory of solar/stellar spectrum
      file_solar_spec = 'sun', &
!       Solar/stellar spectrum file
      dir_spectral = '/discover/nobackup/projects/giss/' // &
          'socrates/spectral_files', &
!       Directory of spectral files
      file_spectral_lw = '', &
!       Long-wave spectral file
      file_spectral_sw = '', &
!       Short-wave spectral file
      file_aer_opt_prop_lw = '', &
!       File with long-wave optical properties of aerosols
      file_aer_opt_prop_sw = '', &
!       File with short-wave optical properties of aerosols
      file_aer_opt_prop_diag = ''
!       File with diagnostic optical properties of aerosols

! Dimensions
  INTEGER, PARAMETER :: &
      npd_layer = lm_gcm + lm_req, &
!       Number of layers in radiation (LX)
      npd_profile = 1
!       Number of profiles

! Variables for interaction with GCM
  REAL(RealK) :: &
      p_level(0:npd_layer), &
!       Pressures at model levels
      p_layer(npd_layer), &
!       Pressures in model layers
      t_level(0:npd_layer), &
!       Temperatures at model levels
      t_layer(npd_layer)
!       Temperatures in model layers

! Cloud properties in each layer. The layering in these arrays are as in
! SOCRATES, i.e. with the numbering starting at the top with the topmost layer
! indexed as 1. The number of layers in these arrays are as in radiaiton, i.e.
! potentially more layers than are dynamically active.
  REAL(RealK), DIMENSION(npd_layer) :: &
      w_cloud_l, &
!       Total cloud area fraction in layers
      frac_st_water_l, &
!       In-cloud fractions of stratiform water cloud
      frac_st_ice_l, &
!       In-cloud fractions of stratiform ice cloud
      frac_cnv_water_l, &
!       In-cloud fractions of convective water cloud
      frac_cnv_ice_l, &
!       In-cloud fractions of stratiform ice cloud
      mix_ratio_st_water_l, &
!       Mass mixing ratios of stratiform water cloud
      mix_ratio_st_ice_l, &
!       Mass mixing ratios of stratiform ice cloud
      mix_ratio_cnv_water_l, &
!       Mass mixing ratios of convective water cloud
      mix_ratio_cnv_ice_l, &
!       Mass mixing ratios of convective ice cloud
      dim_char_st_water_l, &
!       Characteristic dimensions of stratiform water cloud
      dim_char_st_ice_l, &
!       Characteristic dimensions of stratiform ice cloud
      dim_char_cnv_water_l, &
!       Characteristic dimensions of convective water cloud
      dim_char_cnv_ice_l, &
!       Characteristic dimensions of convective ice cloud
      frac_area_st_l, &
!       Fractional stratiform cloud area used to derive in-cloud MMR
      frac_area_cnv_l
!       Fractional convective cloud area used to derive in-cloud MMR
  REAL(RealK) :: &
      cld_scaling
!       Scaling of cloud mass

! Parameters used to calculate albedo
  INTEGER, PARAMETER :: &
      n_snow_land = 2
!       Number of elements in snow depth array (SNOWD)
  REAL(RealK) :: &
      wind_speed_mag, &
!       Magnitude of wind speed in m/s (WMAG)
      frac_sea, &
!       Sea fraction in grid box (POCEAN)
      frac_land, &
!       Land (soil and vegetation) fraction in grid box (PEARTH)
      frac_sea_ice, &
!       Sea ice fraction in grid box (POICE)
      frac_land_ice, &
!       Land ice fraction in grid box (PLICE)
      t_rad_sea, &
!       Sea effective temperature (TGO)
      t_rad_land, &
!       Land (soil and vegetation) effective temperature (TGE)
      t_rad_sea_ice, &
!       Sea ice effective temperature (TGOI)
      t_rad_land_ice, &
!       Land ice effective temperature (TGLI)
      soil_wetness, &
!       Soil wetness (WEARTH)
      frac_veg_tbl(n_veg), &
!       Vegetation fraction table (PVT)
      frac_land_snow_tbl(n_snow_land), &
!       Fraction of snow over soil (1) and vegetation (2) (snow_frac)
      snow_depth_land_tbl(n_snow_land), &
!       Snow depth over land in m (SNOWD)
      snow_depth_sea_ice, &
!       Snow depth over sea ice in m (ZSNWOI)
      snow_amnt_sea_lake
!       Snow amount over sea/lake ice in kg/m**2 (SNOWOI)
  INTEGER, ALLOCATABLE :: &
      index_tile_lw(:), &
!       The index of tiles of given types used by long-wave radiation
      index_tile_sw(:)
!       The index of tiles of given types used by long-wave radiation

! Geometric parameters for short-wave radiation
  REAL(RealK) :: &
      cos_zenith, &
!       Cosine of zenith angle
      solar_irrad
!       Current incoming solar radiation at substellar point
  LOGICAL :: &
      l_lit
!       Flag for calculating short-wave fluxes

! Set wavelengths
  REAL(RealK), PARAMETER :: &
      wl_vis_nir_div = 7.7E-07_RealK
!       Wavelength of division between visible and near-IR

! General parameters
  REAL(RealK), PARAMETER :: &
      tol_cos_zenith = 0.001_RealK
!       Tolerance for value of cosize of zenith angle above which to calculate
!       sort-wave fluxes

! Aerosol optical property tables
  TYPE(StrAerData) :: &
      aer_tbl_lw, &
!       Long-wave aerosol optical properties structure
      aer_tbl_sw, &
!       Short-wave aerosol optical properties structure
      aer_tbl_diag
!       Diagnostic aerosol optical properties structure

! Current aerosol data
  INTEGER, PARAMETER :: &
      npd_aer = 50
!       Maximum number of aerosols
  INTEGER :: &
      n_aer, &
!       Number of aerosols
      n_aer_tracer, &
!       Number of aerosols that are tracers (number of valid entries in
!       i_aer_tracer)
      aer_component(npd_aer), &
!       Component index of aerosol (as defined in rad_ccf)
      i_aer_tracer(npd_aer)
!       Indices of aerosol tracers
  REAL(RealK) :: &
      aer_radius_eff_dry(npd_aer), &
!       Effective radii
      aer_mass_div_dens(npd_aer, npd_layer), &
!       Aerosol mass divided by aerosol bulk density
      aer_frac_sulph(npd_aer), &
!       Sulfate fraction of basic aerosol composition
      rel_humidity(npd_layer), &
!       Relative humidity
      aer_scaling_sw(npd_aer), &
!       Scaling of aerosol mass for short-wave
      aer_scaling_lw(npd_aer), &
!       Scaling of aerosol mass for long-wave
      aer_scaling_diag(npd_aer)
!       Scaling of aerosol mass for diagnostics

! Maps aerosol index in GISS radiation onto aerosol index in planet radiation
  INTEGER, PARAMETER :: i_aer_planet(8) = (/ &
      ip_ammonium_sulphate, & ! 1
      ip_sodium_chloride, &   ! 2
      ip_nitrate, &           ! 3
      ip_biogenic, &          ! 4
      ip_soot, &              ! 5
      ip_soot, &              ! 6
      ip_dust_like, &         ! 7
      ip_sulphuric /)         ! 8

! Aerosol diagnostics
  LOGICAL :: &
      l_aerosol_diag
!       Compute aerosol diagnostics
  logical :: l_aerosol_diag_onoff = .false.
!       Flag for switching diagnostics on or off internally

! Aerosol diagnostics
  TYPE(StrAerDiag), ALLOCATABLE :: &
      aer_diag(:), &
!       Normal aerosol diagnostics
      aer_diag_dry(:)
!       Dry aerosol diagnostics

! Saved output from planet_rad, to be used in RAD_DRV.f
  REAL(RealK) :: &
      tot_cloud_cover
!       Total cloud cover

CONTAINS

! Initialise radiation calculation
SUBROUTINE init_planet_rad(NTRACE)

  USE rad_pcf
  USE dictionary_mod, ONLY: sync_param
  USE lw_control,     ONLY: l_aerosol_lw
  USE sw_control,     ONLY: l_aerosol_sw
  USE planet_alb,     ONLY: init_planet_alb
  USE def_planetstr,  ONLY: allocate_aer_diag

! Input variables
  INTEGER, INTENT(IN) :: &
      NTRACE
!       Number of tracers

! Local variables
  INTEGER :: &
    i_begin, &
!     Index of beginning of substring
    i_end, &
!     Index of end of substring
    i_trc
!     Tracer loop index
  LOGICAL :: &
    l_exists
!     Flag indicating if file exists
  CHARACTER(LEN=256) :: &
    path_solar_spec, &
!     Full path to solar spectrum file
    path_spectral_lw, &
!     Full path to long-wave spectral file
    path_spectral_sw, &
!     Full path to short-wave spectral file
    path_aer_opt_prop_lw, &
!     Full path to long-wave aerosol optical properties file
    path_aer_opt_prop_sw, &
!     Full path to short-wave aerosol optical properties file
    path_aer_opt_prop_diag, &
!     Full path to diagnostic aerosol optical properties file
    name_solar
!     Name of solar spectrum

! Read solar spectrum
  CALL get_path_file(dir_solar_spec, file_solar_spec, path_solar_spec)
  CALL read_solar_spectrum(path_solar_spec, solar_spec)

! Construct full paths to spectral files
  CALL get_path_file(dir_spectral, file_spectral_lw, path_spectral_lw)
  CALL get_path_file(dir_spectral, file_spectral_sw, path_spectral_sw)
  i_begin = SCAN(path_solar_spec, '/', .TRUE.) + 1
  i_end = LEN_TRIM(path_solar_spec)
  name_solar = path_solar_spec(i_begin:i_end)
  IF (TRIM(name_solar) == 'sun') THEN
    INQUIRE(FILE=path_spectral_sw, EXIST=l_exists)
    IF (.NOT.l_exists) THEN
      path_spectral_sw = TRIM(path_spectral_sw) // '_' // TRIM(name_solar)
    END IF
  ELSE
    path_spectral_sw = TRIM(path_spectral_sw) // '_' // TRIM(name_solar)
  END IF

! Read SOCRATES spectral files
  CALL read_spectrum(path_spectral_lw, spectrum_lw)
  CALL read_spectrum(path_spectral_sw, spectrum_sw)

! Set number of bands
  n_band_lw = spectrum_lw%basic%n_band
  n_band_sw = spectrum_sw%basic%n_band

! Turn on N2 and Ar if necessary
  CALL set_control_n2_ar(spectrum_lw, control_lw)
  CALL set_control_n2_ar(spectrum_sw, control_sw)

! Search through the spectrum and control structure to locate gases to be kept
  CALL compress_spectrum(control_lw, spectrum_lw)
  CALL compress_spectrum(control_sw, spectrum_sw)

! Read optical properties of aerosols
  IF (l_aerosol_lw) THEN
    CALL get_path_file(dir_spectral, file_aer_opt_prop_lw, &
        path_aer_opt_prop_lw)
    CALL read_aer_opt_prop(path_aer_opt_prop_lw, aer_tbl_lw)
  END IF
  IF (l_aerosol_sw) THEN
    CALL get_path_file(dir_spectral, file_aer_opt_prop_sw, &
        path_aer_opt_prop_sw)
    i_begin = SCAN(path_solar_spec, '.', .TRUE.)
    i_end = LEN_TRIM(path_solar_spec)
    IF (TRIM(name_solar) == 'sun') THEN
      INQUIRE(FILE=path_aer_opt_prop_sw, EXIST=l_exists)
      IF (.NOT.l_exists) THEN
        path_aer_opt_prop_sw = path_aer_opt_prop_sw(1:i_begin-1) // &
            '_' // TRIM(name_solar) // path_aer_opt_prop_sw(i_begin:i_end)
      END IF
    ELSE
      path_aer_opt_prop_sw = path_aer_opt_prop_sw(1:i_begin-1) // &
          '_' // TRIM(name_solar) // path_aer_opt_prop_sw(i_begin:i_end)
    END IF
    CALL read_aer_opt_prop(path_aer_opt_prop_sw, aer_tbl_sw)
  END IF
  IF (l_aerosol_diag) THEN
    CALL get_path_file(dir_spectral, file_aer_opt_prop_diag, &
        path_aer_opt_prop_diag)
    CALL read_aer_opt_prop(path_aer_opt_prop_diag, aer_tbl_diag)
  END IF

! Initialise albedo calculation
  CALL init_planet_alb(spectrum_sw, solar_spec)

! Allocate arrays for aerosol diagnostics
  IF (l_aerosol_diag) THEN
    n_band_diag = aer_tbl_diag%n_band
    ALLOCATE(aer_diag(NTRACE))
    ALLOCATE(aer_diag_dry(NTRACE))
    DO i_trc = 1, NTRACE
      CALL allocate_aer_diag(aer_diag(i_trc), npd_layer, &
          aer_tbl_diag%n_band)
      CALL allocate_aer_diag(aer_diag_dry(i_trc), npd_layer, &
          aer_tbl_diag%n_band)
    END DO
  ELSE
    n_band_diag = 0
  END IF

END SUBROUTINE init_planet_rad


! Create full paths to input files
SUBROUTINE get_path_file(dir_file, name_file, path_file)

! Input variables
  CHARACTER(LEN=*), INTENT(IN) :: &
    dir_file, &
!     File directory
    name_file
!     Name of file

! Output variables
  CHARACTER(LEN=*), INTENT(OUT) :: &
    path_file
!     Full path to file

  IF (name_file(1:1) == '/') THEN
!   Filename is already full path to the file
    path_file = name_file
  ELSE
!   Concatenate directory and filename to get full path
    path_file = TRIM(dir_file) // '/' // TRIM(name_file)
  END IF

END SUBROUTINE get_path_file


! Wrapper routine for running SOCRATES
SUBROUTINE run_planet_rad(iup_h2o, iup_co2, iup_o3, &
    iup_o2, iup_no2, iup_n2o, iup_ch4, iup_cfc11, iup_cfc12, &
    iup_xghg, iup_yghg, iup_so2, &
    ULGAS, CLDEPS, PRNB, PRNX &
#if defined(DEBUG_RADIATION) || defined(DEBUG_RADIATION_KEEP)
  , l_dbrad_print &
#endif
    )

  USE def_out, ONLY: allocate_out

! Input variables
  INTEGER, INTENT(IN) :: &
      iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
      iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2
!     Identifiers for gases in ULGAS array
  REAL(RealK), INTENT(IN) :: &
      ULGAS(npd_layer,13), &
!       Column densities of absorbers
      CLDEPS(npd_layer), &
!       Cloud heterogeneity
      PRNB(6,4), &
!       Diffuse short-wave albedos for each surface type
      PRNX(6,4)
!       Direct short-wave albedos for each surface type
#if defined(DEBUG_RADIATION) || defined(DEBUG_RADIATION_KEEP)
  LOGICAL, INTENT(IN) :: &
      l_dbrad_print
!       Flag for printing fluxes

! Local variables
  INTEGER :: &
      i
!       Loop index
#endif

! Set pressures in radiation layers. Should probably be a log-mean instead,
! but this is how it is implemented in SETGAS.
  p_layer = &
      (p_level(0:npd_layer-1) + p_level(1:npd_layer))*0.5_RealK

! Determine if current column is lit
  l_lit = cos_zenith > tol_cos_zenith

! Set temperatures on radiation levels
#ifndef DEBUG_RADIATION
  CALL set_t_level
#endif

! Set up for long-wave call
  CALL set_control_lw
  CALL set_dimen(control_lw, spectrum_lw, dimen)
  CALL set_atm(control_lw, spectrum_lw, dimen, ULGAS, &
      iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
      iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2, atm_lw)
  CALL set_cld(control_lw, spectrum_lw, atm_lw, dimen, &
      inhom_cloud_adj_st, inhom_cloud_adj_cnv, CLDEPS, cld_lw)
  CALL set_aer(control_lw, spectrum_lw, dimen, atm_lw, aer_tbl_lw, aer_lw)
  CALL set_bound_lw
  IF (l_set_major_abs) CALL set_major_abs(spectrum_lw, atm_lw)

! Calculate long-wave fluxes
  CALL radiance_calc(control_lw, dimen, spectrum_lw, atm_lw, cld_lw, &
      aer_lw, bound_lw, radout_lw)

! Only perform short-wave calculation if gridpoint is lit
  IF (l_lit) THEN
!   Gridpoint is lit

!   Set up for short-wave call
    CALL set_control_sw
    CALL set_dimen(control_sw, spectrum_sw, dimen)
    CALL set_atm(control_sw, spectrum_sw, dimen, ULGAS, &
        iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
        iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2, atm_sw)
    CALL set_cld(control_sw, spectrum_sw, atm_sw, dimen, &
        inhom_cloud_adj_st, inhom_cloud_adj_cnv, CLDEPS, cld_sw)
    CALL set_aer(control_sw, spectrum_sw, dimen, atm_sw, aer_tbl_sw, aer_sw)
    CALL set_bound_sw(PRNB, PRNX)
    IF (l_set_major_abs) CALL set_major_abs(spectrum_sw, atm_sw)

!   Calculate short-wave fluxes
    CALL radiance_calc(control_sw, dimen, spectrum_sw, atm_sw, cld_sw, &
        aer_sw, bound_sw, radout_sw)

  ELSE
!   Gridpoint is not lit, all short-wave fluxes are zero

    CALL set_control_sw
    CALL set_dimen(control_sw, spectrum_sw, dimen)
    CALL allocate_out(radout_sw, control_sw, dimen, spectrum_sw)

    radout_sw%flux_direct = 0.0_RealK
    radout_sw%flux_down = 0.0_RealK
    radout_sw%flux_up = 0.0_RealK
    radout_sw%flux_direct_clear = 0.0_RealK
    radout_sw%flux_down_clear = 0.0_RealK
    radout_sw%flux_up_clear = 0.0_RealK
    radout_sw%flux_up_tile = 0.0_RealK
    radout_sw%flux_up_band = 0.0_RealK
    radout_sw%flux_down_band = 0.0_RealK

  END IF

! Calculate diagnostic aerosol optical properties
  IF (l_aerosol_diag .and. l_aerosol_diag_onoff) &
       CALL calc_aerosol_diag(atm_lw%mass(npd_profile, :))

#if (defined(DEBUG_RADIATION) || defined(DEBUG_RADIATION_KEEP))
    
  IF (l_dbrad_print) THEN
    WRITE(0,*) 'SOCRATES long-wave fluxes'
    WRITE(0,'(A15,A15,A15,A15,A15)') 'Pressure [Pa]', 'T [K]', 'Flux down [W/m2]', &
         'Flux up [W/m2]', 'Flux net [W/m2]'
    DO i=0,npd_layer
      WRITE(0,'(F15.4,F15.4,F15.4,F15.4,F15.4)') &
          atm_lw%p_level(npd_profile,i)*1.0D-02, &
          atm_lw%t_level(npd_profile,i), &
          radout_lw%flux_down(npd_profile,i,1), &
          radout_lw%flux_up(npd_profile,i,1), &
          radout_lw%flux_up(npd_profile,i,1) - &
          radout_lw%flux_down(npd_profile,i,1)
    END DO

    IF (l_lit) THEN
      WRITE(0,*) 'SOCRATES short-wave fluxes'
      WRITE(0,'(A15,A15,A15,A15,A15)') 'Pressure [Pa]', 'T [K]', 'Flux down [W/m2]', &
          'Flux up [W/m2]', 'Flux net [W/m2]'
      DO i=0,npd_layer
        WRITE(0,'(F15.4,F15.4,F15.4,F15.4,F15.4)') &
            atm_lw%p_level(npd_profile,i)*1.0D-02, &
            atm_lw%t_level(npd_profile,i), &
            SUM(radout_sw%flux_down(npd_profile,i, &
              1:control_sw%n_channel))/cos_zenith, &
            SUM(radout_sw%flux_up(npd_profile,i, &
              1:control_sw%n_channel))/cos_zenith, &
            SUM(radout_sw%flux_down(npd_profile,i,1:control_sw%n_channel) - &
              radout_sw%flux_up(npd_profile,i,1:control_sw%n_channel))/cos_zenith
      END DO
    END IF

    WRITE(0,*) 'Cloud fractions'
    WRITE(0,'(A15,A15,A15,A15,A15,A15)') 'Pressure [Pa]', 'w_cloud', &
        'frac_st_water_l', 'frac_st_ice_l', 'frac_cnv_water_l', 'frac_cnv_ice_l'
    DO i=1,npd_layer
      WRITE(0,'(F15.4,F15.4,F15.4,F15.4,F15.4,F15.4)') &
          atm_lw%p(npd_profile,i)*1.0D-02, &
          w_cloud_l(i), &
          frac_st_water_l(i), &
          frac_st_ice_l(i), &
          frac_cnv_water_l(i), &
          frac_cnv_ice_l(i)
    END DO
  END IF
#endif

END SUBROUTINE run_planet_rad


! Set variables in control structure for long-wave call. Control
! variables for individual gases are set in set_control_gas.
SUBROUTINE set_control_lw

  USE def_control, ONLY: allocate_control
  USE lw_control
  USE constant, ONLY: pN2, pAr

! Local variables
  INTEGER :: &
      i
!       Loop index

  CALL allocate_control(control_lw, spectrum_lw)

! Spectral region and bands
  control_lw%isolir      = ip_infra_red
  control_lw%first_band  = 1
  control_lw%last_band   = spectrum_lw%basic%n_band

! Physical processes
  control_lw%l_microphysics    = l_microphysics_lw
  control_lw%l_gas             = l_gas_lw
  control_lw%l_drop            = l_drop_lw
  control_lw%l_ice             = l_ice_lw
  control_lw%l_aerosol         = l_aerosol_lw
  control_lw%l_aerosol_mode    = control_lw%l_aerosol
  control_lw%l_solar_tail_flux = l_solar_tail_flux_lw

! Turn on standard and/or genereralised continuum absorption if requested and
! present in spectral file
  control_lw%l_continuum = l_continuum_lw .AND. spectrum_lw%basic%l_present(9)
  control_lw%l_cont_gen  = l_cont_gen_lw  .AND. spectrum_lw%basic%l_present(19)

! Gaseous absorption
  control_lw%i_gas_overlap = i_gas_overlap_lw

! Properties of clouds
  control_lw%l_cloud = l_cloud .AND. l_cloud_lw
  IF (control_lw%l_cloud) THEN
    control_lw%i_cloud_representation = i_cloud_representation_lw
    control_lw%i_st_water             = i_st_water_lw
    control_lw%i_cnv_water            = i_cnv_water_lw
    control_lw%i_st_ice               = i_st_ice_lw
    control_lw%i_cnv_ice              = i_cnv_ice_lw
    control_lw%l_global_cloud_top     = l_global_cloud_top_lw
    control_lw%i_inhom                = i_inhom_lw
    control_lw%i_overlap              = i_overlap_lw
  ELSE
    control_lw%i_cloud_representation = ip_cloud_off
  END IF

! Angular integration and algorithmic options
  control_lw%i_angular_integration = i_angular_integration_lw
  control_lw%i_2stream             = i_2stream_lw
  control_lw%l_ir_source_quad      = l_ir_source_quad_lw
  control_lw%i_scatter_method      = i_scatter_method_lw
  control_lw%l_rescale             = l_rescale_lw

! Miscallaneous options
  control_lw%l_tile         = l_tile_lw
  control_lw%l_mixing_ratio = .TRUE.

! Calculate clear-sky fluxes if requested
  control_lw%l_clear = l_clear_lw

! Set properties for individual bands
  control_sw%n_channel = 1
  DO i = 1, spectrum_lw%basic%n_band
    control_lw%map_channel(i)           = 1
    control_lw%weight_band(i)           = 1.0
    control_lw%i_scatter_method_band(i) = control_lw%i_scatter_method
    control_lw%i_gas_overlap_band(i)    = control_lw%i_gas_overlap
    IF (ANY(spectrum_lw%gas%i_scale_fnc(i,:) == ip_scale_ses2)) THEN
!     If SES2 scaling is used in this band then the overlap must also use SES2
      control_lw%i_gas_overlap_band(i)  = ip_overlap_mix_ses2
    END IF
  END DO

! Set order used in forming the forward scattering parameter
  IF (control_lw%l_rescale) control_lw%n_order_forward = 2

! Set solver and cloud scheme
  CALL set_solver_cloud_scheme(control_lw)

! Diagnostics
  control_lw%l_flux_up_band = .TRUE.

END SUBROUTINE set_control_lw


! Set variables in control structure for short-wave call. Control
! variables for individual gases are set in set_control_gas.
SUBROUTINE set_control_sw

  USE def_control, ONLY: allocate_control
  USE sw_control
  USE constant, ONLY: pN2, pAr

! Local variables
  INTEGER :: &
      i
!       Loop index

  CALL allocate_control(control_sw, spectrum_sw)

! Spectral region and bands
  control_sw%isolir      = ip_solar
  control_sw%first_band  = 1
  control_sw%last_band   = spectrum_sw%basic%n_band

! Physical processes
  control_sw%l_microphysics    = l_microphysics_sw
  control_sw%l_gas             = l_gas_sw
  control_sw%l_rayleigh        = l_rayleigh_sw
  control_sw%l_drop            = l_drop_sw
  control_sw%l_ice             = l_ice_sw
  control_sw%l_aerosol         = l_aerosol_sw
  control_sw%l_aerosol_mode    = control_sw%l_aerosol
  control_sw%l_orog            = l_orog_sw
  control_sw%l_solvar          = l_solvar_sw

! Turn on standard and/or genereralised continuum absorption if requested and
! present in spectral file
  control_sw%l_continuum = l_continuum_sw .AND. spectrum_sw%basic%l_present(9)
  control_sw%l_cont_gen  = l_cont_gen_sw  .AND. spectrum_sw%basic%l_present(19)

! Gaseous absorption
  control_sw%i_gas_overlap = i_gas_overlap_sw

! Properties of clouds
  control_sw%l_cloud = l_cloud .AND. l_cloud_sw
  IF (control_sw%l_cloud) THEN
    control_sw%i_cloud_representation = i_cloud_representation_sw
    control_sw%i_st_water             = i_st_water_sw
    control_sw%i_cnv_water            = i_cnv_water_sw
    control_sw%i_st_ice               = i_st_ice_sw
    control_sw%i_cnv_ice              = i_cnv_ice_sw
    control_sw%l_global_cloud_top     = l_global_cloud_top_sw
    control_sw%i_inhom                = i_inhom_sw
    control_sw%i_overlap              = i_overlap_sw
  ELSE
    control_sw%i_cloud_representation = ip_cloud_off
  END IF

! Angular integration and algorithmic options
  control_sw%i_angular_integration = i_angular_integration_sw
  control_sw%i_2stream             = i_2stream_sw
  control_sw%i_scatter_method      = i_scatter_method_sw
  control_sw%l_rescale             = l_rescale_sw

! Miscallaneous options
  control_sw%l_tile         = l_tile_sw
  control_sw%l_mixing_ratio = .TRUE.

! Calculate clear-sky fluxes if requested
  control_sw%l_clear = l_clear_sw

! Set properties for individual bands
  control_sw%n_channel = spectrum_sw%basic%n_band
  DO i = 1, spectrum_sw%basic%n_band
    control_sw%map_channel(i)           = i
    control_sw%weight_band(i)           = 1.0
    control_sw%i_scatter_method_band(i) = control_sw%i_scatter_method
    control_sw%i_gas_overlap_band(i)    = control_sw%i_gas_overlap
    IF (ANY(spectrum_sw%gas%i_scale_fnc(i,:) == ip_scale_ses2)) THEN
!     If SES2 scaling is used in this band then the overlap must also use SES2
      control_sw%i_gas_overlap_band(i)  = ip_overlap_mix_ses2
    END IF
  END DO

! Set order used in forming the forward scattering parameter
  IF (control_sw%l_rescale) control_sw%n_order_forward = 2

! Set solver and cloud scheme
  CALL set_solver_cloud_scheme(control_sw)

! Diagnostics
  control_sw%l_flux_up_band = .TRUE.
  control_sw%l_flux_down_band = .TRUE.

END SUBROUTINE set_control_sw


! Set long-wave boundary condition
SUBROUTINE set_bound_lw

  USE lw_control, ONLY: i_sea_ice_alb
  USE planet_alb, ONLY: set_surf_prop_lw
  USE def_bound,  ONLY: allocate_bound

  CALL allocate_bound(bound_lw, dimen, spectrum_lw)
  IF (.NOT. ALLOCATED(index_tile_lw)) &
      ALLOCATE(index_tile_lw(dimen%nd_tile_type))

! Set the surface basis functions. By defining f_brdf{1,0,0,0} to be 4, rho_alb
! becomes equal to the diffuse albedo.
  bound_lw%n_brdf_basis_fnc = 1
  bound_lw%f_brdf(1, 0, 0, 0) = 4.0_RealK

! Set long-wave surface properties
  CALL set_surf_prop_lw(i_sea_ice_alb, &
      wind_speed_mag, frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
      t_rad_sea, t_rad_land, t_rad_sea_ice, t_rad_land_ice, &
      soil_wetness, frac_veg_tbl, frac_land_snow_tbl, snow_depth_land_tbl, &
      snow_depth_sea_ice, snow_amnt_sea_lake, &
      spectrum_lw, solar_spec, control_lw%l_tile, index_tile_lw, bound_lw, &
      atm_lw%n_profile, n_snow_land, dimen%nd_tile_type)

END SUBROUTINE set_bound_lw


! Set short-wave boundary condition
SUBROUTINE set_bound_sw(PRNB, PRNX)

  USE def_bound,  ONLY: allocate_bound
  USE planet_alb, ONLY: set_surf_prop_sw
  USE def_bound,  ONLY: allocate_bound

! Input variables
  REAL(RealK), INTENT(IN) :: &
      PRNB(6,4), &
!       Diffuse short-wave albedos
      PRNX(6,4)
!       Direct short-wave albedos

  CALL allocate_bound(bound_sw, dimen, spectrum_sw)
  IF (.NOT. ALLOCATED(index_tile_sw)) &
      ALLOCATE(index_tile_sw(dimen%nd_tile_type))

  bound_sw%zen_0(atm_sw%n_profile) = 1.0_RealK/cos_zenith
  bound_sw%solar_irrad(atm_sw%n_profile) = solar_irrad

! Set the surface basis functions. By defining f_brdf{1,0,0,0} to be 4, rho_alb
! becomes equal to the diffuse albedo.
  bound_sw%n_brdf_basis_fnc = 1
  bound_sw%f_brdf(1, 0, 0, 0) = 4.0_RealK
  bound_sw%rho_alb = 0.0_RealK

! Set short-wave surface properties
  CALL set_surf_prop_sw(frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
      PRNB, PRNX, control_sw%l_tile, index_tile_sw, &
      bound_sw, atm_sw%n_profile, dimen%nd_tile_type, spectrum_sw%basic%n_band)

END SUBROUTINE set_bound_sw


! Set temperatures at radiation levels using linear interpolation in log
! pressure
SUBROUTINE set_t_level

! Local variables
  INTEGER :: &
      i
!       Layer/level loop index
  REAL(RealK) :: &
      wgt_up, &
!       Interpolation weight for layer temperature above level
      wgt_dwn
!       Interpolation weight for layer temperature below level

! For the top level use isothermal extrapolation
  t_level(0) = t_layer(1)

! For the bottom level use isothermal extrapolation
  t_level(npd_layer) = t_layer(npd_layer)

! For all interior layer boundaries use linear interpolation in log pressure
  DO i = 1, npd_layer - 1
    wgt_up = (LOG10(p_layer(i + 1)) - LOG10(p_level(i)))/ &
        (LOG10(p_layer(i + 1)) - LOG10(p_layer(i)))
    wgt_dwn = 1.0_RealK - wgt_up
    t_level(i) = wgt_up*t_layer(i) + wgt_dwn*t_layer(i + 1)
  END DO

END SUBROUTINE set_t_level


! Set temperatures at radiation levels. Algorithm from THERML in RADIATION.f,
! not currently used.
SUBROUTINE set_t_level_giss

! This routine calculates level temperatures from the layer temperatures
! set by the GCM. It is identical to the old routine used in the
! subroutine THERML.
!
! Should be updated to a be more understandable.
!
! TLGRAD=1.0  (Default)
!             Layer-mean temperatures (TLM) supplied by GCM are used
!             to define the layer edge temperature TLT (top) and TLB
!             (bottom) using overall atmospheric temperature profile
!             to establish temperature gradient within each layer so
!             as to minimize the temperature discontinuities between
!             layer edges and to conserve layer thermal energy.
!
! TLGRAD=0.0  This results in isothermal layers with TLT = TLB = TLM
!
! TLGRAD<0.0  TLT and TLB are used as specified, without any further
!             adjustments.  This is mainly for off-line use when the
!             temperature profile (TLM,TLT,TLB) can be fully defined
!             from a continuous temperature profile.
!
! NOTE:       TLGRAD can also accommodate values between 0.0 and 1.0
!
! PTLISO      (Default PTLISO=2.5mb)
!             Pressure level above which model layers are defined to
!             be isothermal.  This is appropriate for optically thin
!             layers where emitted flux depends on mean temperature.

  USE constant, ONLY: kapa

! Control parameters
  REAL(RealK), PARAMETER :: &
      TLGRAD = 1.0_RealK, &
!       Control parameter for temperature gradient across layer. If >= 0
!       tlt = tlm+ d T*TLGRAD, tlb = tlm - dT*TLGRAD, where dT is chosen to
!       try to minimize discontinuities if TLGRAD = 1. If TLGRAD < 0 tlt, tlm
!       and tlb are all inputs (OFFLINE use)
      PTLISO = 2.5_RealK
!       Pressure above which the temerature is taken to be isothermal across
!       layers tlt = tlb = tlm above PTLISO [mb] independent of TLGRAD

! Local variables
  REAL(RealK) :: &
      PLB(npd_layer+1), &
!       Pressure in mbar at radiation levels with reversed indexing
      PL(npd_layer), &
!       Pressure in model layers with reversed indexing
      TLM(npd_layer), &
!       Temperature in radiation levels with reversed indexing
      TLB(npd_layer+1), &
!       Calculated temperatures at radiation levels with reversed indexing
      TLT(npd_layer), &
!       Calculated temperatures at the top of radiation layers
      TA, TB, TC, P1, P2, P3, P4, DT1CPT, DTHALF
!       Temporary work variables
  INTEGER :: &
      L, &
!       Layer loop index
      L1 = 1, &
!       Lowest layer above ground
      NL = npd_layer
!       Number of radiation layers

! Set pressures and temperatures
  PLB = p_level(npd_layer:0:-1)*1.0E-02_RealK
  PL  = p_layer(npd_layer:1:-1)*1.0E-02_RealK
  TLM = t_layer(npd_layer:1:-1)

! Perform calculation of level temperatures.
! Author: Andrew Lacis
  TA = TLM(L1)
  TB = TLM(L1+1)
  P1 = PLB(L1)
  P2 = PLB(L1+1)
  P3 = PLB(L1+2)
  DT1CPT = .5*TA*(P1**kapa-P2**kapa) / PL(L1)**kapa
  DTHALF = (TA-TB)*(P1-P2)/(P1-P3)
  if (DTHALF > DT1CPT) DTHALF = DT1CPT
  TLB(L1) = TA+DTHALF*TLGRAD
  TLT(L1) = TA-DTHALF*TLGRAD
  DO L = L1+1,NL-1
    TC = TLM(L+1)
    P4 = PLB(L+2)
    DTHALF = .5*((TA-TB)/(P1-P3)+(TB-TC)/(P2-P4))*(P2-P3)*TLGRAD
    TLB(L) = TB+DTHALF
    TLT(L) = TB-DTHALF
    TA = TB
    TB = TC
    P1 = P2
    P2 = P3
    P3 = P4
  END DO
  DTHALF = (TA-TB)*(P2-P3)/(P1-P3)*TLGRAD
  TLB(NL) = TC+DTHALF
  TLT(NL) = TC-DTHALF
  DO L = NL,L1,-1
    if (PLB(L) > PTLISO) EXIT
    TLT(L) = TLM(L)
    TLB(L) = TLM(L)
  END DO
  TLB(NL+1) = TLT(NL)

! Set t_level array
  t_level = TLB(npd_layer+1:1:-1)

END SUBROUTINE set_t_level_giss


! Set the GISS radiation output arrays from the fluxes calculated by the
! planetary radiation scheme.
SUBROUTINE get_planet_radout(TLB, TLT, NORMS0, &
    TRUFLB, TRDFLB, TRNFLB, TRFCRL, &
    TRUFLB_CLEAR, TRDFLB_CLEAR, TRNFLB_CLEAR, &
    SRUFLB, SRDFLB, SRNFLB, SRFHRL, &
    SRUFLB_CLEAR, SRDFLB_CLEAR, SRNFLB_CLEAR, FSRNFG, &
    SRIVIS, SROVIS, PLAVIS, SRINIR, SRONIR, PLANIR, &
    SRDVIS, SRUVIS, ALBVIS, SRDNIR, SRUNIR, ALBNIR, &
    SRTVIS, SRRVIS, SRAVIS, SRTNIR, SRRNIR, SRANIR, &
    SRXVIS, SRXNIR, S0, &
    TRUPTOA, SRUPTOA, SRDNTOA, &
    aesqex, aesqsc, aesqcb &
#if defined DEBUG_RADIATION || defined DEBUG_RADIATION_KEEP
  , l_dbrad_print &
#endif
  )

  USE rad_pcf

! Input variables
  INTEGER, INTENT(IN) :: &
      NORMS0
!       Determines if short-wave fluxes should be normalized to solar constant

! Output from THERML
  REAL(RealK), INTENT(OUT) :: &
      TLB(npd_layer+1), &
!       Temperatures at lower bounds of layers
      TLT(npd_layer), &
!       Temperatures at top bounds of layers
      TRUFLB(npd_layer+1), &
!       Upward flux on radiation levels
      TRDFLB(npd_layer+1), &
!       Downward flux on radiation levels
      TRNFLB(npd_layer+1), &
!       Net flux on radiation levels
      TRFCRL(npd_layer), &
!       Difference of net flux between layers to be used to calculate heating
!       rate
      TRUFLB_CLEAR(npd_layer+1), &
!       Upward clear-sky flux on radiation levels
      TRDFLB_CLEAR(npd_layer+1), &
!       Downward clear-sky flux on radiation levels
      TRNFLB_CLEAR(npd_layer+1)
!       Net clear-sky flux on radiation levels

! Output from SOLARM
  REAL(RealK), INTENT(OUT) :: &
      SRUFLB(npd_layer+1), &
!       Upward flux on radiation levels
      SRDFLB(npd_layer+1), &
!       Downward flux on radiation levels
      SRNFLB(npd_layer+1), &
!       Net flux on radiation levels
      SRFHRL(npd_layer), &
!       Difference of net flux between layers to be used to calculate heating
!       rate
      SRUFLB_CLEAR(npd_layer+1), &
!       Upward clear-sky flux on radiation levels
      SRDFLB_CLEAR(npd_layer+1), &
!       Downward clear-sky flux on radiation levels
      SRNFLB_CLEAR(npd_layer+1), &
!       Net clear-sky flux on radiation levels
      FSRNFG(4), &
!       Flux absorbed at ground by the four different surface types
      SRIVIS, &
!       Incident visible flux at TOA
      SROVIS, &
!       Outgoing visible flux at TOA
      PLAVIS, &
!       Visible planet albedo (SROVIS/SRIVIS)
      SRINIR, &
!       Incident near-IR flux at TOA
      SRONIR, &
!       Outgoing near-IR flux at TOA
      PLANIR, &
!       Near-IR planet albedo (SRONIR/SRINIR)
      SRDVIS, &
!       Downward visible flux at BOA
      SRUVIS, &
!       Upward visible flux at BOA
      ALBVIS, &
!       Visible surface albedo (RUVIS/SRDVIS)
      SRDNIR, &
!       Downward near-IR flux at BOA
      SRUNIR, &
!       Upward near-IR flux at BOA
      ALBNIR, &
!       Near-IR surface albedo (SRUNIR/SRDNIR)
      SRTVIS, &
!       Fractional amount of visible flux transmitted
      SRRVIS, &
!       Fractional amount of visible flux reflected
      SRAVIS, &
!       Fractional amount of visible flux absorbed (1 - SRRVIS - SRTVIS)
      SRTNIR, &
!       Fractional amount of near-IR flux transmitted
      SRRNIR, &
!       Fractional amount of near-IR flux reflected
      SRANIR, &
!       Fractional amount of near-IR flux absorbed (1 - SRTNIR - SRRNIR)
      SRXVIS, &
!       Fraction of visible solar flux in direct beam at surface
      SRXNIR, &
!       Fraction of near-IR solar flux in direct beam at surface
      S0
!       Total solar TOA flux at the substellar point

! Diagnostics
  REAL(RealK), INTENT(OUT) :: &
    TRUPTOA(spectrum_lw%basic%n_band), &
!     Outgoing thermal radiation at TOA in each band
    SRUPTOA(spectrum_sw%basic%n_band), &
!     Outgoing solar radiation at TOA in each band
    SRDNTOA(spectrum_sw%basic%n_band), &
!     Downwelling solar radiation at TOA in each band
    aesqex(npd_layer, MAX(1, n_band_diag), n_aer_tracer), &
!     Aerosol extinction
    aesqsc(npd_layer, MAX(1, n_band_diag), n_aer_tracer), &
!     Aerosol scattering
    aesqcb(npd_layer, MAX(1, n_band_diag), n_aer_tracer)
!     Aerosol asymmetry

! Local variables
  INTEGER :: &
    i_trc, &
!     Tracer loop index
    i_layer, i_inverse
!     Layer loop indices

#if defined DEBUG_RADIATION || defined DEBUG_RADIATION_KEEP
  logical, intent(in) :: l_dbrad_print
  integer :: ib, n, i_aer
  character( len=20 ) :: cform
#endif

! Remember that the indexing is reversed in GISS radiation arrays.

! Set level temperatures
  TLB(1:npd_layer+1) = t_level(npd_layer:0:-1)
  TLT(1:npd_layer) = t_level(npd_layer-1:0:-1)

! Set THERML fluxes
  TRUFLB(1:npd_layer+1) = radout_lw%flux_up(npd_profile, npd_layer:0:-1, 1)
  TRDFLB(1:npd_layer+1) = radout_lw%flux_down(npd_profile, npd_layer:0:-1, 1)
  TRNFLB(1:npd_layer+1) = TRUFLB(1:npd_layer+1) - TRDFLB(1:npd_layer+1)
  TRFCRL(1:npd_layer) = TRNFLB(2:npd_layer+1) - TRNFLB(1:npd_layer)
  IF (control_lw%l_clear) THEN
    TRUFLB_CLEAR(1:npd_layer+1) = &
        radout_lw%flux_up_clear(npd_profile, npd_layer:0:-1, 1)
    TRDFLB_CLEAR(1:npd_layer+1) = &
        radout_lw%flux_down_clear(npd_profile, npd_layer:0:-1, 1)
    TRNFLB_CLEAR(1:npd_layer+1) = &
        TRUFLB_CLEAR(1:npd_layer+1) - TRDFLB_CLEAR(1:npd_layer+1)
  END IF

! Normalize fluxes to solar constant if required
  IF (l_lit .AND. NORMS0 > 0) THEN
    radout_sw%flux_up(npd_profile,:,:) = &
        radout_sw%flux_up(npd_profile,:,:)/cos_zenith
    radout_sw%flux_down(npd_profile,:,:) = &
        radout_sw%flux_down(npd_profile,:,:)/cos_zenith
    IF (control_sw%l_tile) THEN
      radout_sw%flux_up_tile(npd_profile,:,:) = &
          radout_sw%flux_up_tile(npd_profile,:,:)/cos_zenith
    END IF
    IF (control_sw%l_clear) THEN
      radout_sw%flux_up_clear(npd_profile,:,:) = &
          radout_sw%flux_up_clear(npd_profile,:,:)/cos_zenith
      radout_sw%flux_down_clear(npd_profile,:,:) = &
          radout_sw%flux_down_clear(npd_profile,:,:)/cos_zenith
    END IF
    IF (control_sw%l_flux_up_band) THEN
      radout_sw%flux_up_band(npd_profile,:,:) = &
          radout_sw%flux_up_band(npd_profile,:,:)/cos_zenith
    END IF
    IF (control_sw%l_flux_down_band) THEN
      radout_sw%flux_down_band(npd_profile,:,:) = &
          radout_sw%flux_down_band(npd_profile,:,:)/cos_zenith
    END IF
  END IF

! Set SOLARM fluxes
  SRUFLB(1:npd_layer+1) = SUM(radout_sw%flux_up(npd_profile, &
      npd_layer:0:-1, 1:control_sw%n_channel), 2)
  SRDFLB(1:npd_layer+1) = SUM(radout_sw%flux_down(npd_profile, &
      npd_layer:0:-1, 1:control_sw%n_channel), 2)
  SRNFLB(1:npd_layer+1) = SRDFLB(1:npd_layer+1) - SRUFLB(1:npd_layer+1)
  SRFHRL(1:npd_layer) = SRNFLB(2:npd_layer+1) - SRNFLB(1:npd_layer)
  IF (control_sw%l_clear) THEN
    SRUFLB_CLEAR(1:npd_layer+1) = &
        SUM(radout_sw%flux_up_clear(npd_profile, npd_layer:0:-1, &
        1:control_sw%n_channel), 2)
    SRDFLB_CLEAR(1:npd_layer+1) = &
        SUM(radout_sw%flux_down_clear(npd_profile, npd_layer:0:-1, &
        1:control_sw%n_channel), 2)
    SRNFLB_CLEAR(1:npd_layer+1) = &
        SRDFLB_CLEAR(1:npd_layer+1) - SRUFLB_CLEAR(1:npd_layer+1)
  END IF

! Set solar flux absorbed by ground for different surface types
  IF (l_lit .AND. control_sw%l_tile) THEN
!   Tiling is on, get fluxes for each surface type
    IF (frac_sea > 0.0_RealK) THEN
      FSRNFG(1) = SRDFLB(1) - &
          SUM(radout_sw%flux_up_tile(npd_profile, &
          index_tile_sw(ip_ocean_tile), 1:control_sw%n_channel))
    ELSE
      FSRNFG(1) = SRDFLB(1)
    END IF
    IF (frac_land > 0.0_RealK) THEN
      FSRNFG(2) = SRDFLB(1) - &
          SUM(radout_sw%flux_up_tile(npd_profile, &
          index_tile_sw(ip_land_tile), 1:control_sw%n_channel))
    ELSE
      FSRNFG(2) = SRDFLB(1)
    END IF
    IF (frac_sea_ice > 0.0_RealK) THEN
      FSRNFG(3) = SRDFLB(1) - &
          SUM(radout_sw%flux_up_tile(npd_profile, &
          index_tile_sw(ip_seaice_tile), 1:control_sw%n_channel))
    ELSE
      FSRNFG(3) = SRDFLB(1)
    END IF
    IF (frac_land_ice > 0.0_RealK) THEN
      FSRNFG(4) = SRDFLB(1) - &
          SUM(radout_sw%flux_up_tile(npd_profile, &
          index_tile_sw(ip_landice_tile), 1:control_sw%n_channel))
    ELSE
      FSRNFG(4) = SRDFLB(1)
    END IF

  ELSE
!   Tiling is off or column is not lit
    IF (frac_sea > 0.0_RealK) THEN
      FSRNFG(1) = SRNFLB(1)
      FSRNFG(2) = SRDFLB(1)
      FSRNFG(3) = SRDFLB(1)
      FSRNFG(4) = SRDFLB(1)
    ELSE IF (frac_land > 0.0_RealK) THEN
      FSRNFG(1) = SRDFLB(1)
      FSRNFG(2) = SRNFLB(1)
      FSRNFG(3) = SRDFLB(1)
      FSRNFG(4) = SRDFLB(1)
    ELSE IF (frac_sea_ice > 0.0_RealK) THEN
      FSRNFG(1) = SRDFLB(1)
      FSRNFG(2) = SRDFLB(1)
      FSRNFG(3) = SRNFLB(1)
      FSRNFG(4) = SRDFLB(1)
    ELSE IF (frac_land_ice > 0.0_RealK) THEN
      FSRNFG(1) = SRDFLB(1)
      FSRNFG(2) = SRDFLB(1)
      FSRNFG(3) = SRDFLB(1)
      FSRNFG(4) = SRNFLB(1)
    ELSE
      CALL stop_model('get_planet_radout: All surface types are zero.', 255)
    END IF
  END IF

! Visible fluxes at TOA
  SRIVIS = SUM(radout_sw%flux_down(npd_profile, 0, 1:control_sw%n_channel)* &
      spectrum_sw%solar%weight_blue(1:control_sw%n_channel))
  SROVIS = SUM(radout_sw%flux_up(npd_profile, 0, 1:control_sw%n_channel)* &
      spectrum_sw%solar%weight_blue(1:control_sw%n_channel))
  IF (l_lit) THEN
    PLAVIS = SROVIS/SRIVIS
  ELSE
    PLAVIS = 1.0_RealK
  END IF

! Near-IR fluxes at TOA
  SRINIR = SUM(radout_sw%flux_down(npd_profile, 0, 1:control_sw%n_channel)* &
      (1.0_RealK - spectrum_sw%solar%weight_blue(1:control_sw%n_channel)))
  SRONIR = SUM(radout_sw%flux_up(npd_profile, 0, 1:control_sw%n_channel)* &
      (1.0_RealK - spectrum_sw%solar%weight_blue(1:control_sw%n_channel)))
  IF (l_lit) THEN
    PLANIR = SRONIR/SRINIR
  ELSE
    PLANIR = 1.0_RealK
  END IF

! Visible fluxes at BOA
  SRDVIS = &
      SUM(radout_sw%flux_down(npd_profile, npd_layer, 1:control_sw%n_channel)* &
      spectrum_sw%solar%weight_blue(1:control_sw%n_channel))
  SRUVIS = &
      SUM(radout_sw%flux_up(npd_profile, npd_layer, 1:control_sw%n_channel)* &
      spectrum_sw%solar%weight_blue(1:control_sw%n_channel))
  IF (l_lit) THEN
    ALBVIS = SRUVIS/(SRDVIS + TINY(SRDVIS))
  ELSE
    ALBVIS = 1.0_RealK
  END IF

! Near-IR fluxes at BOA
  SRDNIR = &
      SUM(radout_sw%flux_down(npd_profile, npd_layer, 1:control_sw%n_channel)* &
      (1.0_RealK - spectrum_sw%solar%weight_blue(1:control_sw%n_channel)))
  SRUNIR = &
      SUM(radout_sw%flux_up(npd_profile, npd_layer, 1:control_sw%n_channel)* &
      (1.0_RealK - spectrum_sw%solar%weight_blue(1:control_sw%n_channel)))
  IF (l_lit) THEN
    ALBNIR = SRUNIR/(SRDNIR + TINY(SRDNIR))
  ELSE
    ALBNIR = 1.0_RealK
  END IF

! Fractional visible fluxes
  SRTVIS = 0.0_RealK ! Not available
  SRRVIS = 0.0_RealK ! Not available
  SRAVIS = 0.0_RealK ! Not available

! Fractional near-IR fluxes
  SRTNIR = 0.0_RealK ! Not available
  SRRNIR = 0.0_RealK ! Not available
  SRANIR = 0.0_RealK ! Not available

! Direct fractional fluxes
  IF (l_lit) THEN
    SRXVIS = SUM(radout_sw%flux_direct(npd_profile, &
        npd_layer, 1:control_sw%n_channel)* &
        spectrum_sw%solar%weight_blue(1:control_sw%n_channel))/ &
        SUM(radout_sw%flux_direct(npd_profile, 0, 1:control_sw%n_channel)* &
        spectrum_sw%solar%weight_blue(1:control_sw%n_channel))
    SRXNIR = SUM(radout_sw%flux_direct(npd_profile, &
        npd_layer, 1:control_sw%n_channel)* &
        (1.0_RealK - spectrum_sw%solar%weight_blue(1:control_sw%n_channel)))/ &
        SUM(radout_sw%flux_direct(npd_profile, 0, 1:control_sw%n_channel)* &
        (1.0_RealK - spectrum_sw%solar%weight_blue(1:control_sw%n_channel)))
  ELSE
    SRXVIS = 0.0_RealK
    SRXNIR = 0.0_RealK
  END IF

! Total solar TOA flux at substellar point
  S0 = solar_irrad

! Clouds
  IF (control_lw%l_cloud) THEN
    tot_cloud_cover = radout_lw%tot_cloud_cover(npd_profile)
  ELSE
    tot_cloud_cover = 0.0_RealK
  END IF

! Band-by-band flux diagnosics
  TRUPTOA = radout_lw%flux_up_band(npd_profile, 0, 1:spectrum_lw%basic%n_band)
  SRUPTOA = radout_sw%flux_up_band(npd_profile, 0, 1:spectrum_sw%basic%n_band)
  SRDNTOA = radout_sw%flux_down_band(npd_profile, 0, 1:spectrum_sw%basic%n_band)

! Aerosol diagnostics
  IF (l_aerosol_diag) THEN
    do i_layer = 1,npd_layer
      i_inverse = npd_layer - i_layer + 1
      DO i_trc = 1, n_aer_tracer
        aesqex(i_inverse, :, i_trc) = aer_diag(i_trc)%tau_ext(i_layer, :)
        aesqsc(i_inverse, :, i_trc) = aer_diag(i_trc)%tau_scat(i_layer, :)
        aesqcb(i_inverse, :, i_trc) = aer_diag(i_trc)%g_asym(i_layer, :)* &
          aer_diag(i_trc)%tau_scat(i_layer, :)
      END DO
    end do
  ELSE
    aesqex(:, :, :) = 0.0_RealK
    aesqsc(:, :, :) = 0.0_RealK
    aesqcb(:, :, :) = 0.0_RealK
  END IF

#if defined DEBUG_RADIATION || defined DEBUG_RADIATION_KEEP
  if ( l_dbrad_print .and. l_aerosol_diag ) then
    write( 0, * ) 'In planet_rad.F90 in subroutine get_planet_radout'
    write( 0, * ) 'l_aerosol_diag: ', l_aerosol_diag, '; Number of diagnostic bands: ', &
         n_band_diag
    do i_trc = 1,n_aer_tracer
      i_aer = i_aer_tracer( i_trc )
      write( 0, * ) 'i_trc', i_trc, ' i_aer_tracer(i_trc): ', i_aer
      write( cform, '(a9,i3,a8)' ) '(a5,2a15,', n_band_diag, '(a6,i9))'
      write( 0, cform ) 'level', 'Pressure [Pa]', 'Aerosol [m-3]', &
           ( '  Band', ib, ib=1,n_band_diag )
      write( cform, '(a4,i3,a6)' ) '(i5, ', n_band_diag+2, 'd15.4)'
      do n = 1,npd_layer
        write( 0, cform ) ( n, atm_lw%p_level( npd_profile, npd_layer-n+1 )*1.0D-02, &
             aer_mass_div_dens( i_aer, npd_layer-n+1 ), aesqex( n, ib, i_trc ), &
             ib=1,n_band_diag )
      end do
    end do
  end if
#endif

END SUBROUTINE get_planet_radout


! Calculates aerosol optical properties for diagnostic purposes
SUBROUTINE calc_aerosol_diag(mass)

! Input variables
  REAL(RealK), INTENT(IN) :: &
      mass(npd_layer)
!       Mass column density of each layer

! Local variables
  INTEGER :: &
      i_layer, &
!       Layer loop index
      i_band, &
!       Diagnostic band loop index
      i_trc, &
!       Tracer loop index
      i_aer
!       Current aerosol index
  REAL(RealK) :: &
    rel_humidity_dry(npd_layer) = 0.0_RealK, &
!     Relative humidity for calculation of dry aerosol diagnostics
    aer_mix_ratio_diag(npd_layer, npd_aer), &
!     Mixing ratio
    aer_absorption_diag(npd_layer, npd_aer, aer_tbl_diag%n_band), &
!     Absorption coefficient
    aer_scattering_diag(npd_layer, npd_aer, aer_tbl_diag%n_band), &
!     Scattering coefficient
    aer_asymmetry_diag(npd_layer, npd_aer, aer_tbl_diag%n_band)
!     Asymmetry parameter

! Calculate mixing ratio, absorption, scattering and asymmetry, use short-
! wave scaling
  CALL calc_aer_opt_prop(npd_layer, n_aer, mass, &
      aer_component, aer_radius_eff_dry, &
      aer_mass_div_dens, aer_frac_sulph, &
      aer_scaling_diag, rel_humidity, aer_tbl_diag, &
      aer_mix_ratio_diag, aer_absorption_diag, &
      aer_scattering_diag, aer_asymmetry_diag, &
      npd_layer, npd_aer, aer_tbl_diag%n_band)

! Calculate the diagnostics
  DO i_trc = 1, n_aer_tracer
    i_aer = i_aer_tracer(i_trc)
    DO i_band = 1, aer_tbl_diag%n_band
      DO i_layer = 1, npd_layer
        aer_diag(i_trc)%tau_ext(i_layer, i_band) = &
            aer_mix_ratio_diag(i_layer, i_aer)*mass(i_layer)* &
            (aer_absorption_diag(i_layer, i_aer, i_band) + &
            aer_scattering_diag(i_layer, i_aer, i_band))
        aer_diag(i_trc)%tau_scat(i_layer, i_band) = &
            aer_mix_ratio_diag(i_layer, i_aer)*mass(i_layer)* &
            aer_scattering_diag(i_layer, i_aer, i_band)
        aer_diag(i_trc)%g_asym(i_layer, i_band) = &
            aer_asymmetry_diag(i_layer, i_aer, i_band)
      END DO
    END DO
  END DO

! Calculate mixing ratio, absorption, scattering and asymmetry, use short-
! wave scaling for dry diagnostics
  CALL calc_aer_opt_prop(npd_layer, n_aer, mass, &
      aer_component, aer_radius_eff_dry, &
      aer_mass_div_dens, aer_frac_sulph, &
      aer_scaling_sw, rel_humidity_dry, aer_tbl_diag, &
      aer_mix_ratio_diag, aer_absorption_diag, &
      aer_scattering_diag, aer_asymmetry_diag, &
      npd_layer, npd_aer, aer_tbl_diag%n_band)

! Calculate the dry diagnostics
  DO i_trc = 1, n_aer_tracer
    i_aer = i_aer_tracer(i_trc)
    DO i_band = 1, aer_tbl_diag%n_band
      DO i_layer = 1, npd_layer
        aer_diag_dry(i_trc)%tau_ext(i_layer, i_band) = &
            aer_mix_ratio_diag(i_layer, i_aer)*mass(i_layer)* &
            (aer_absorption_diag(i_layer, i_aer, i_band) + &
            aer_scattering_diag(i_layer, i_aer, i_band))
        aer_diag_dry(i_trc)%tau_scat(i_layer, i_band) = &
            aer_mix_ratio_diag(i_layer, i_aer)*mass(i_layer)* &
            aer_scattering_diag(i_layer, i_aer, i_band)
        aer_diag_dry(i_trc)%g_asym(i_layer, i_band) = &
            aer_asymmetry_diag(i_layer, i_aer, i_band)
      END DO
    END DO
  END DO

END SUBROUTINE calc_aerosol_diag


! Deallocates everything allocated by planetary radiation schemes
SUBROUTINE deallocate_planet_rad

  USE def_atm,        ONLY: deallocate_atm
  USE def_bound,      ONLY: deallocate_bound
  USE def_control,    ONLY: deallocate_control
  USE def_cld,        ONLY: deallocate_cld, deallocate_cld_prsc, &
                            deallocate_cld_mcica
  USE def_aer,        ONLY: deallocate_aer, deallocate_aer_prsc
  USE def_out,        ONLY: deallocate_out

! Deallocate long-wave call structures
  CALL deallocate_control(control_lw)
  CALL deallocate_atm(atm_lw)
  CALL deallocate_cld(cld_lw)
  CALL deallocate_cld_prsc(cld_lw)
  CALL deallocate_cld_mcica(cld_lw)
  CALL deallocate_aer(aer_lw)
  CALL deallocate_aer_prsc(aer_lw)
  CALL deallocate_bound(bound_lw)
  CALL deallocate_out(radout_lw)
  IF (ALLOCATED(index_tile_lw)) DEALLOCATE(index_tile_lw)

! Deallocate short-wave call structures
  CALL deallocate_control(control_sw)
  CALL deallocate_atm(atm_sw)
  CALL deallocate_cld(cld_sw)
  CALL deallocate_cld_prsc(cld_sw)
  CALL deallocate_cld_mcica(cld_sw)
  CALL deallocate_aer(aer_sw)
  CALL deallocate_aer_prsc(aer_sw)
  CALL deallocate_bound(bound_sw)
  CALL deallocate_out(radout_sw)
  IF (ALLOCATED(index_tile_sw)) DEALLOCATE(index_tile_sw)

END SUBROUTINE deallocate_planet_rad

END MODULE planet_rad

!-------------------------------------------------------------------------------
! Subroutines not part of the planet_rad module. These subroutines are used
! by both short-wave and long-wave radiation calls.
!-------------------------------------------------------------------------------

! Reads a high resolution solar spectrum. Based on the SOCRATES routine
! read_solar_sectrum in src/general. Any changes to the format of the SOCRATES
! solar spectrum file should be reflected here.
SUBROUTINE read_solar_spectrum(path_solar_spec, solar_spec)

  USE realtype_rd, ONLY: ReaLK
  USE filemanager, ONLY: findunit
  USE planet_rad,  ONLY: StrSolarSpec

  IMPLICIT NONE

! Input variables
  CHARACTER(LEN=*), INTENT(IN) :: &
      path_solar_spec
!       Full path to file with solar spectrum

! Output variables
  TYPE(StrSolarSpec), INTENT(OUT) :: &
      solar_spec
!       Solar spectrum

! Local variables
  INTEGER :: &
      iu_solar, &
!       Unit number for solar spectrum
      ios, &
!       I/O error flag
      i
!       Loop variable
  REAL(RealK) :: &
      scale_wv, &
!       Scaling for wavelength to correct units
      scale_irr
!       Scaling for irradiance to correct units
  LOGICAL :: &
      l_exists, &
!       True if solar spectrum file exists
      l_count
!       Flag for counting points
  CHARACTER(LEN=256) :: &
      iomessage, &
!       I/O error message
      line
!       Line of input data

! Check that file exists
  INQUIRE(FILE=path_solar_spec, EXIST=l_exists)
  IF (.NOT. l_exists) THEN
    CALL stop_model('Solar spectrum file ' // TRIM(path_solar_spec) // &
        ' does not exist.', 255)
  END IF

! Open file
  CALL findunit(iu_solar)
  OPEN(UNIT=iu_solar, FILE=path_solar_spec, IOSTAT=ios, STATUS='OLD', &
      ACTION='READ', IOMSG=iomessage)
  IF (ios /= 0) THEN
    CALL stop_model('Solar spectrum file ' // TRIM(path_solar_spec) // &
        ' could not be opened: ' // TRIM(iomessage))
  END IF

! Read first to find the number of points in the spectrum.
  solar_spec%n_points = 0
  l_count=.FALSE.
  DO
    READ(iu_solar, '(A)', IOSTAT=ios) line
    IF (ios /= 0) THEN
      EXIT
    ELSE IF (line(1:11) == '*BEGIN_DATA') THEN
      l_count = .TRUE.
    ELSE IF (line(1:4) == '*END') THEN
      l_count = .FALSE.
    ELSE IF (l_count) THEN
      solar_spec%n_points = solar_spec%n_points + 1
    ENDIF
  ENDDO

  ALLOCATE(solar_spec%wavelength(solar_spec%n_points))
  ALLOCATE(solar_spec%irrad(solar_spec%n_points))

! Read in the file.
  scale_wv=1.0_RealK
  scale_irr=1.0_RealK
  REWIND(iu_solar)
  DO
    READ(iu_solar, '(A)', IOSTAT=ios) line
    IF (line(1:11) == '*BEGIN_DATA') THEN
      DO i = 1, solar_spec%n_points
        READ(iu_solar, *, IOSTAT=ios) &
          solar_spec%wavelength(i), &
          solar_spec%irrad(i)
        IF (ios /= 0) THEN
          CALL stop_model('Solar spectrum ' // TRIM(path_solar_spec) // &
              ' is corrupt.', 255)
        ENDIF
      ENDDO
      EXIT
    ENDIF
    IF (line(1:11) == '*SCALE_DATA') THEN
      READ(iu_solar, *, IOSTAT=ios) scale_wv, scale_irr
    ENDIF
    IF (line(1:7) == '*RADIUS') THEN
      READ(iu_solar, *, IOSTAT=ios) solar_spec%radius
    END IF
    IF (line(1:12) == '*TEMPERATURE') THEN
      READ(iu_solar, *, IOSTAT=ios) solar_spec%t_effective
    END IF
  ENDDO

! Check that reading was carried out: failure to read the data
! will cause n_solar_points to be 0.
  IF (solar_spec%n_points == 0) THEN
    CALL stop_model('No solar spectrum data were read. Check format ' // &
        'of file of irradiances: ' // TRIM(path_solar_spec), 255)
  ENDIF

! Scale the values to the correct units:
  solar_spec%wavelength = solar_spec%wavelength * scale_wv
  solar_spec%irrad      = solar_spec%irrad      * scale_irr

END SUBROUTINE read_solar_spectrum


! Reads aerosol optical properties from file
SUBROUTINE read_aer_opt_prop(file_aer_opt_prop, aer_tbl)

  USE filemanager,   ONLY: nf
  USE def_planetstr, ONLY: StrAerData

  IMPLICIT NONE

  INCLUDE 'netcdf.inc'

! Input variables
  CHARACTER(LEN=*), INTENT(IN) :: &
      file_aer_opt_prop
!       File with aerosol optical properties
  TYPE(StrAerData), INTENT(INOUT) :: &
      aer_tbl
!       Aerosol optical properties

! Local variables
  INTEGER :: &
      ncidin_aer, &
!       NetCDF file identifier
      status, &
!       Status of NetCDF I/O
      dimid, &
!       Dimension identifier
      varid, &
!       Variable identifier
      i, &
!       Loop index
      i_component_max
!       Maximum rad_ccf aerosol component index
  INTEGER, ALLOCATABLE :: &
      i_humidity(:)
!       Aerosol humidity dependence logical (as integer)
  LOGICAL :: &
      l_diag
!       Flag indicating if file contains diagnostic quantities

! Open file with optical properties
  status = nf_open(file_aer_opt_prop, NF_NOWRITE, ncidin_aer)
  IF (status /= NF_NOERR) THEN
    CALL stop_model('read_aer_opt_prop: Could not open file with aerosol ' // &
        'optical properties', 255)
  END IF

! Get global attributes
  CALL nf(nf_get_att_text(ncidin_aer, NF_GLOBAL, 'spectral_file', &
      aer_tbl%spectral_file))
  CALL nf(nf_get_att_text(ncidin_aer, NF_GLOBAL, 'spectral_region', &
      aer_tbl%spectral_region))

! Get length of dimensions
  CALL nf(nf_inq_dimid(ncidin_aer, 'component', dimid))
  CALL nf(nf_inq_dimlen(ncidin_aer, dimid, aer_tbl%n_component))
  CALL nf(nf_inq_dimid(ncidin_aer, 'radius', dimid))
  CALL nf(nf_inq_dimlen(ncidin_aer, dimid, aer_tbl%n_radius_max))
  CALL nf(nf_inq_dimid(ncidin_aer, 'humidity', dimid))
  CALL nf(nf_inq_dimlen(ncidin_aer, dimid, aer_tbl%n_humidity))
  CALL nf(nf_inq_dimid(ncidin_aer, 'band', dimid))
  CALL nf(nf_inq_dimlen(ncidin_aer, dimid, aer_tbl%n_band))

! Allocate arrays
  ALLOCATE( &
      aer_tbl%i_component(aer_tbl%n_component), &
      aer_tbl%radius_eff(aer_tbl%n_humidity, aer_tbl%n_radius_max, aer_tbl%n_component &
                         ), &
      aer_tbl%humidity(aer_tbl%n_humidity), &
      aer_tbl%band(aer_tbl%n_band), &
      aer_tbl%n_radius(aer_tbl%n_component), &
      aer_tbl%radius_eff_dry(aer_tbl%n_radius_max, aer_tbl%n_component), &
      aer_tbl%l_humidity(aer_tbl%n_component), &
      aer_tbl%k_abs(aer_tbl%n_band, aer_tbl%n_humidity, &
                    aer_tbl%n_radius_max, aer_tbl%n_component), &
      aer_tbl%k_scat(aer_tbl%n_band, aer_tbl%n_humidity, &
                     aer_tbl%n_radius_max, aer_tbl%n_component), &
      aer_tbl%g_asym(aer_tbl%n_band, aer_tbl%n_humidity, &
                     aer_tbl%n_radius_max, aer_tbl%n_component), &
      aer_tbl%density(aer_tbl%n_component))
  ALLOCATE(i_humidity(aer_tbl%n_component))

! Check if aerosol data are for diagnostic purposes and allocate wavelength
! array
  IF (aer_tbl%spectral_region(1:4) == 'diag') THEN
    aer_tbl%l_diag = .TRUE.
    ALLOCATE(aer_tbl%wavelength(aer_tbl%n_band))
  ELSE
    aer_tbl%l_diag = .FALSE.
  END IF

! Read data
  CALL nf(nf_inq_varid(ncidin_aer, 'component', varid))
  CALL nf(nf_get_var_int(ncidin_aer, varid, aer_tbl%i_component))
  CALL nf(nf_inq_varid(ncidin_aer, 'radius_eff', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%radius_eff))
  CALL nf(nf_inq_varid(ncidin_aer, 'humidity', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%humidity))
  CALL nf(nf_inq_varid(ncidin_aer, 'band', varid))
  CALL nf(nf_get_var_int(ncidin_aer, varid, aer_tbl%band))
  CALL nf(nf_inq_varid(ncidin_aer, 'n_radius', varid))
  CALL nf(nf_get_var_int(ncidin_aer, varid, aer_tbl%n_radius))
  CALL nf(nf_inq_varid(ncidin_aer, 'radius_eff_dry', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%radius_eff_dry))
  CALL nf(nf_inq_varid(ncidin_aer, 'l_humidity', varid))
  CALL nf(nf_get_var_int(ncidin_aer, varid, i_humidity))
  CALL nf(nf_inq_varid(ncidin_aer, 'k_abs', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%k_abs))
  CALL nf(nf_inq_varid(ncidin_aer, 'k_scat', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%k_scat))
  CALL nf(nf_inq_varid(ncidin_aer, 'g_asym', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%g_asym))
  CALL nf(nf_inq_varid(ncidin_aer, 'density', varid))
  CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%density))
  IF (aer_tbl%l_diag) THEN
    CALL nf(nf_inq_varid(ncidin_aer, 'wavelength', varid))
    CALL nf(nf_get_var_double(ncidin_aer, varid, aer_tbl%wavelength))
  END IF

! Done reading, close file
  CALL nf(nf_close(ncidin_aer))

! Convert i_humidity to l_humidity
  aer_tbl%l_humidity = i_humidity > 0

! Create array mapping component index in rad_ccf to component index is above
! arrays
  i_component_max = MAXVAL(aer_tbl%i_component)
  ALLOCATE(aer_tbl%i_component_map(i_component_max))
  aer_tbl%i_component_map = -1
  DO i = 1, aer_tbl%n_component
    aer_tbl%i_component_map(aer_tbl%i_component(i)) = i
  END DO

! Deallocate temporary arrays
  DEALLOCATE(i_humidity)

END SUBROUTINE read_aer_opt_prop


! This subroutine sets the logical control variables for each gas using
! provided the logical switch and the scalings in FULGAS (includes the.
SUBROUTINE set_control_gas(l_co2, l_o2, l_no2, l_n2o, l_ch4, &
    l_cfc11, l_cfc12, l_so2, l_o3, fulgas, &
    CO2X, O2X, NO2X, N2OX, CH4X, CFC11X, CFC12X, XGHGX, YGHGX, &
    N2CX, SO2X, O3X, control)

  USE realtype_rd, ONLY: RealK
  USE def_control, ONLY: StrCtrl
  USE constant,    ONLY: pH2

  IMPLICIT NONE

! Input variables
  LOGICAL, INTENT(IN) :: &
      l_co2, &
!         Flag for absorption by carbon dioxide
      l_o2, &
!         Flag for absorption by oxygen
      l_no2, &
!         Flag for absorption by nitrogen dioxide
      l_n2o, &
!         Flag for absorption by nitrous oxide
      l_ch4, &
!         Flag for absorption by methane
      l_cfc11, &
!         Flag for absorption by CFC11
      l_cfc12, &
!         Flag for absorption by CFC12
      l_so2, &
!         Flag for absorption by sulfur dioxide
      l_o3
!         Flag for absorption by ozone
  REAL(RealK), INTENT(IN) :: &
      fulgas(13), &
!       Array with scaling factors for gases
      CO2X, &
!       Abundance scaling factor for carbon dioxide
      O2X, &
!       Abundance scaling factor for oxygen
      NO2X, &
!       Abundance scaling factor for nitrogen dioxide
      N2OX, &
!       Abundance scaling factor for nitrous oxide
      CH4X, &
!       Abundance scaling factor for methane
      CFC11X, &
!       Abundance scaling factor for CFC11
      CFC12X, &
!       Abundance scaling factor for CFC12
      XGHGX, &
!       Abundance scaling factor for GHGX (added to CFC11)
      YGHGX, &
!       Abundance scaling factor for GHGY (added to CFC12)
      N2CX, &
!       Abundance scaling factor for nitrogen
      SO2X, &
!       Abundance scaling factor for sulfur dioxide
      O3X
!       Abundance scaling factor for ozone

! Output variables
  TYPE(StrCtrl), INTENT(INOUT) :: &
      control
!       Control options

! Gases supported by both GISS radiation and SOCRATES
  control%l_co2 = l_co2 .AND. &
      CO2X > 0.0_RealK .AND. fulgas(2) > 0.0_RealK
  control%l_o3 = l_o3 .AND. &
      O3X > 0.0_RealK .AND. fulgas(3) > 0.0_RealK
  control%l_o2  = l_o2 .AND. &
      O2X > 0.0_RealK .AND. fulgas(4) > 0.0_RealK
  control%l_no2 = l_no2 .AND. &
      NO2X > 0.0_RealK .AND. fulgas(5) > 0.0_RealK
  control%l_n2o = l_n2o .AND. &
      N2OX > 0.0_RealK .AND. fulgas(6) > 0.0_RealK
  control%l_ch4 = l_ch4 .AND. &
      CH4X > 0.0_RealK .AND. fulgas(7) > 0.0_RealK
  control%l_cfc11 = l_cfc11  .AND. &
      ((CFC11X > 0.0_RealK .AND. fulgas(8) > 0.0_RealK) .OR. &
       (XGHGX > 0.0_RealK .AND. fulgas(11) > 0.0_RealK))
  control%l_cfc12 = l_cfc12 .AND. &
      ((CFC12X > 0.0_RealK .AND. fulgas(9) > 0.0_RealK) .OR. &
       (YGHGX > 0.0_RealK .AND. fulgas(12) > 0.0_RealK))
  control%l_so2 = &
      l_so2 .AND. SO2X > 0.0_RealK .AND. fulgas(13) > 0.0_RealK

! Gases only supported by SOCRATES
  control%l_h2 = pH2 > 0.0_RealK

END SUBROUTINE set_control_gas


! Set logical control variables for inclusion of N2 and Ar. These are treated
! as a special case as they may have a non-zero abundance, but still do not
! need to be included in the spectral files. The exception being if
! the generalised Rayleigh scattering formulation is used or a continuum
! involving these species is included. N2 and Ar flags are therefore
! only switched on if they are included in the spectral file and their
! abundances is non-zero.
SUBROUTINE set_control_n2_ar(spectrum, control)

  USE realtype_rd,  ONLY: RealK
  USE def_spectrum, ONLY: StrSpecData
  USE def_control,  ONLY: StrCtrl
  USE gas_list_pcf, ONLY: ip_n2, ip_ar
  USE constant,     ONLY: pN2, pAr

  IMPLICIT NONE

! Input variables
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information

! Input/output variables
  TYPE(StrCtrl), INTENT(INOUT) :: &
      control
!       Control options

  IF (ANY(spectrum%gas%type_absorb(1:spectrum%gas%n_absorb) == ip_n2) &
      .AND. pN2 > 0.0_RealK) THEN
    control%l_n2 = .TRUE.
  END IF
  IF (ANY(spectrum%gas%type_absorb(1:spectrum%gas%n_absorb) == ip_ar) &
      .AND. pAr > 0.0_RealK) THEN
    control%l_ar = .TRUE.
  END IF

END SUBROUTINE set_control_n2_ar


! Set dimensions structure
SUBROUTINE set_dimen(control, spectrum, dimen)

  USE def_control,  ONLY: StrCtrl
  USE def_spectrum, ONLY: StrSpecData
  USE def_dimen,    ONLY: StrDim
  USE planet_rad,   ONLY: npd_profile, npd_layer, npd_aer
  USE rad_pcf

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information

! Output variables
  TYPE(StrDim), INTENT(INOUT) :: &
      dimen
!       Dimensions

! Set the dimensions in the dimen structure. For explanations see comments in
! structure type definition in def_dimen.
  dimen%nd_profile                = npd_profile
  dimen%nd_layer                  = npd_layer
  dimen%nd_layer_clr              = npd_layer
  dimen%id_cloud_top              = 1
  dimen%nd_2sg_profile            = npd_profile
  dimen%nd_flux_profile           = npd_profile
  dimen%nd_radiance_profile       = 1
  dimen%nd_j_profile              = 1
  dimen%nd_channel                = spectrum%basic%n_band
  dimen%nd_max_order              = 1
  dimen%nd_direction              = 1
  dimen%nd_viewing_level          = 1
  dimen%nd_sph_coeff              = 1
  dimen%nd_brdf_basis_fnc         = 2
  dimen%nd_brdf_trunc             = 1
  dimen%nd_tile_type              = 4
  dimen%nd_aerosol_mode           = npd_aer
  dimen%nd_profile_aerosol_prsc   = 1
  dimen%nd_profile_cloud_prsc     = 1
  dimen%nd_opt_level_aerosol_prsc = 1
  dimen%nd_opt_level_cloud_prsc   = 1
  dimen%nd_cloud_component        = 4
  dimen%nd_cloud_type             = 4
  dimen%nd_cloud_representation   = 4
  SELECT CASE(control%i_cloud)
  CASE (ip_cloud_column_max)
    dimen%nd_column = 3*dimen%nd_layer + 2
  CASE DEFAULT
    dimen%nd_column = 1
  END SELECT
  dimen%nd_subcol_gen             = 1
  dimen%nd_subcol_req             = 1
  SELECT CASE(control%i_solver)
  CASE(ip_solver_mix_app_scat, ip_solver_mix_direct, ip_solver_mix_direct_hogan)
    dimen%nd_overlap_coeff=8
    dimen%nd_region=2
  CASE(ip_solver_triple_app_scat, ip_solver_triple, ip_solver_triple_hogan)
    dimen%nd_overlap_coeff=18
    dimen%nd_region=3
  CASE DEFAULT
    dimen%nd_overlap_coeff=1
    dimen%nd_region=2
  END SELECT
  dimen%nd_source_coeff           = 2
  dimen%nd_point_tile             = npd_profile
  dimen%nd_tile                   = 4

END SUBROUTINE set_dimen


! Set cloud scheme and solver from selected cloud vertical overlap treatment
! and cloud representation
SUBROUTINE set_solver_cloud_scheme(control)

  USE def_control, ONLY: StrCtrl
  USE rad_pcf

  IMPLICIT NONE

! Input/output variables
  TYPE(StrCtrl), INTENT(INOUT) :: &
      control
!       Control options

  IF (control%i_cloud_representation == ip_cloud_homogen) THEN
    control%i_solver = ip_solver_homogen_direct
    IF (control%i_inhom == ip_mcica) THEN
      control%i_cloud = ip_cloud_mcica
    ELSE
      IF (control%i_overlap == ip_max_rand) THEN
        control%i_cloud = ip_cloud_mix_max
      ELSE IF (control%i_overlap == ip_exponential_rand) THEN
        control%i_cloud = ip_cloud_part_corr
      ELSE IF (control%i_overlap == ip_rand) THEN
        control%i_cloud = ip_cloud_mix_random
      ELSE
        CALL stop_model('set_solver_cloud_scheme: Selected cloud overlap ' // &
            'scheme is not available.', 255)
      END IF
    END IF

  ELSE IF (control%i_cloud_representation == ip_cloud_ice_water) THEN

    IF (control%i_inhom == ip_mcica) THEN
      control%i_cloud = ip_cloud_mcica
      IF (control%i_scatter_method == ip_no_scatter_abs .OR. &
          control%i_scatter_method == ip_no_scatter_ext) THEN
        control%i_solver = ip_solver_no_scat
        control%i_solver_clear = ip_solver_no_scat
      ELSE
        control%i_solver = ip_solver_homogen_direct
      END IF
    ELSE
      IF (control%i_scatter_method == ip_scatter_approx) THEN
          control%i_solver = ip_solver_mix_app_scat
      ELSE
        control%i_solver = ip_solver_mix_direct_hogan
      END IF
      IF (control%i_overlap == ip_max_rand) THEN
        control%i_cloud = ip_cloud_mix_max
      ELSE IF (control%i_overlap == ip_exponential_rand) THEN
        control%i_cloud = ip_cloud_part_corr
      ELSE IF (control%i_overlap == ip_rand) THEN
        control%i_cloud = ip_cloud_mix_random
      ELSE
        CALL stop_model('set_solver_cloud_scheme: Selected cloud overlap ' // &
            'scheme is not available.', 255)
      END IF
    END IF

  ELSE IF (control%i_cloud_representation == ip_cloud_conv_strat .OR. &
           control%i_cloud_representation == ip_cloud_csiw) THEN

    IF (control%i_inhom == ip_mcica) THEN
      CALL stop_model('set_control: McICA is not compatible with the ' // &
          'selected cloud representation.', 255)
    ELSE
      IF (control%i_scatter_method == ip_scatter_approx) THEN
         control%i_solver = ip_solver_triple_app_scat
      ELSE
        control%i_solver = ip_solver_triple_hogan
      END IF
      IF (control%i_overlap == ip_max_rand) THEN
        control%i_cloud = ip_cloud_triple
      ELSE IF (control%i_overlap == ip_exponential_rand) THEN
        control%i_cloud = ip_cloud_part_corr_cnv
      ELSE IF (control%i_overlap == ip_rand) THEN
        control%i_cloud = ip_cloud_mix_random
      ELSE
        CALL stop_model('set_solver_cloud_scheme: Selected cloud overlap ' // &
            'scheme is not available.', 255)
      END IF
    END IF

  ELSE IF (control%i_cloud_representation == ip_cloud_off) THEN

    ! No treatment of cloud for LW radiation
    control%l_cloud = .FALSE.
    control%i_cloud = ip_cloud_clear
    IF (control%i_scatter_method == ip_no_scatter_abs .OR. &
        control%i_scatter_method == ip_no_scatter_ext) THEN
      control%i_solver = ip_solver_no_scat
      control%i_solver_clear = ip_solver_no_scat
    ELSE
      control%i_solver = ip_solver_homogen_direct
    END IF
    control%l_microphysics = .FALSE.

  ELSE
    CALL stop_model('set_solver_cloud_scheme: Selected cloud ' // &
        'representation not available.', 255)

  END IF ! i_cloud_representation

  IF (control%l_clear) THEN
    control%i_solver_clear = ip_solver_homogen_direct
  END IF

END SUBROUTINE set_solver_cloud_scheme


! Remove gases from spectrum structure that are not to be included
SUBROUTINE compress_spectrum(control, spectrum)

  USE def_spectrum,     ONLY: StrSpecData
  USE def_control,      ONLY: StrCtrl
  USE missing_data_mod, ONLY: rmdi

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
! Input/output variables
  TYPE(StrSpecData), INTENT(INOUT) :: &
      spectrum
!       Spectral information

! Local variables
  INTEGER :: &
      i, j, n_band_absorb, n_band_cont, &
!       Loop indices
      i_gas_1, i_gas_2
!       Continuum gas indices
  LOGICAL :: &
      l_keep_absorb(spectrum%gas%n_absorb), &
!       Flags for the retention of gases in the spectral file
      l_keep_cont(spectrum%contgen%n_cont)
!       Flags for the retention of continua in the spectral file

! Search through the spectrum and find gases to be kept
  l_keep_absorb = .FALSE.
  DO i=1,spectrum%gas%n_absorb
    l_keep_absorb(i) = l_retain_gas(control, spectrum%gas%type_absorb(i))
  END DO

! Search through the spectrum and find continua to be kept
  l_keep_cont = .FALSE.
  DO i=1,spectrum%contgen%n_cont
    i_gas_1 = spectrum%contgen%index_cont_gas_1(i)
    i_gas_2 = spectrum%contgen%index_cont_gas_2(i)
    l_keep_cont(i) = &
        l_retain_gas(control, spectrum%gas%type_absorb(i_gas_1)) .AND. &
        l_retain_gas(control, spectrum%gas%type_absorb(i_gas_2))
  END DO

! For all bands, loop through gases and continua, and remove those that are
! not to be kept
  DO i=1, spectrum%basic%n_band

!   Gases
    n_band_absorb=0
    DO j=1, spectrum%gas%n_band_absorb(i)
      IF (l_keep_absorb(spectrum%gas%index_absorb(j, i))) THEN
        n_band_absorb = n_band_absorb + 1
        spectrum%gas%index_absorb(n_band_absorb, i) = &
            spectrum%gas%index_absorb(j, i)
      END IF
    END DO
    spectrum%gas%n_band_absorb(i)=n_band_absorb

!   Continua
    n_band_cont=0
    DO j=1, spectrum%contgen%n_band_cont(i)
      IF (l_keep_cont(spectrum%contgen%index_cont(j, i))) THEN
        n_band_cont = n_band_cont + 1
        spectrum%contgen%index_cont(n_band_cont, i) = &
            spectrum%contgen%index_cont(j, i)
      END IF
    END DO
    spectrum%contgen%n_band_cont(i)=n_band_cont

  END DO

! Set "blue" (visible) weight for each band if not read from spectral file
  IF (spectrum%solar%weight_blue(1) == rmdi) THEN
    CALL set_blue_weight(spectrum)
  END IF

CONTAINS

! Function to check whether a gas with a certain type index should be
! retained
  LOGICAL FUNCTION l_retain_gas(control, i_type_gas)

    USE gas_list_pcf

    IMPLICIT NONE

!   Input variables
    TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
    INTEGER, INTENT(IN) :: &
        i_type_gas
!     Type index of gas

    IF ((i_type_gas == ip_h2o) .OR. &
        ((i_type_gas == ip_co2) .AND. control%l_co2) .OR. &
        ((i_type_gas == ip_o3) .AND. control%l_o3).OR. &
        ((i_type_gas == ip_no2) .AND. control%l_no2).OR. &
        ((i_type_gas == ip_n2o) .AND. control%l_n2o).OR. &
        ((i_type_gas == ip_ch4) .AND. control%l_ch4).OR. &
        ((i_type_gas == ip_so2) .AND. control%l_so2).OR. &
        ((i_type_gas == ip_cfc11) .AND. control%l_cfc11).OR. &
        ((i_type_gas == ip_cfc12) .AND. control%l_cfc12).OR. &
        ((i_type_gas == ip_cfc113) .AND. control%l_cfc113).OR. &
        ((i_type_gas == ip_cfc114) .AND. control%l_cfc114).OR. &
        ((i_type_gas == ip_hcfc22) .AND. control%l_hcfc22).OR. &
        ((i_type_gas == ip_hfc125) .AND. control%l_hfc125).OR. &
        ((i_type_gas == ip_hfc134a) .AND. control%l_hfc134a) .OR. &
        ((i_type_gas == ip_n2) .AND. control%l_n2) .OR. &
        ((i_type_gas == ip_ar) .AND. control%l_ar) .OR. &
        ((i_type_gas == ip_h2) .AND. control%l_h2)) THEN
      l_retain_gas = .TRUE.
    ELSE
      l_retain_gas = .FALSE.
    END IF

  END FUNCTION l_retain_gas

END SUBROUTINE compress_spectrum


! Sets the weight of each band to compute the visible (blue) flux
SUBROUTINE set_blue_weight(spectrum)

  USE realtype_rd,  ONLY: RealK
  USE def_spectrum, ONLY: StrSpecData
  USE planet_rad,   ONLY: wl_vis_nir_div

  IMPLICIT NONE

! Input/output variables
  TYPE(StrSpecData), INTENT(INOUT) :: &
      spectrum
!       Spectral information

! Local variables
  INTEGER :: &
      i, i_band, i_band_ex
!       Loop index
  REAL(RealK) :: &
      energy_range_blue, &
!       Energy range of a band within the blue region
      total_energy_range
!       Total energy range of a band

  DO i_band=1,spectrum%basic%n_band
    IF (spectrum%basic%wavelength_long(i_band) < wl_vis_nir_div) THEN
!     The whole band is in the blue region
      spectrum%solar%weight_blue(i_band) = 1.0_RealK
    ELSE IF (spectrum%basic%wavelength_short(i_band) > wl_vis_nir_div) THEN
!     The whole band is in the blue region
      spectrum%solar%weight_blue(i_band) = 0.0_RealK

    ELSE
!     Part of the band is in the blue region

!     Compute the energy range in the blue region and total band energy range
      energy_range_blue = 1.0_RealK/spectrum%basic%wavelength_short(i_band) - &
          1.0_RealK/wl_vis_nir_div
      total_energy_range = 1.0_RealK/spectrum%basic%wavelength_short(i_band) - &
          1.0_RealK/spectrum%basic%wavelength_long(i_band)

!     Remove any excluded regions from the band from the energy range
      IF (spectrum%basic%l_present(14)) THEN
!       Exclusion block present in spectral file
        DO i=1,spectrum%basic%n_band_exclude(i_band)
          i_band_ex = spectrum%basic%index_exclude(i, i_band)
          energy_range_blue = energy_range_blue - ( &
              1.0_RealK/spectrum%basic%wavelength_short(i_band_ex) - &
              MAX(1.0_RealK/spectrum%basic%wavelength_long(i_band_ex), &
              wl_vis_nir_div))
          total_energy_range = total_energy_range - ( &
              1.0_RealK/spectrum%basic%wavelength_short(i_band_ex) - &
              1.0_RealK/spectrum%basic%wavelength_long(i_band_ex))
        END DO
      END IF

!     Set the blue weight of the band
      spectrum%solar%weight_blue(i_band) = energy_range_blue/total_energy_range
    END IF
  END DO

END SUBROUTINE set_blue_weight


! Set atmosphere structure
SUBROUTINE set_atm(control, spectrum, dimen, ULGAS, &
    iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
    iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2, atm)

  USE realtype_rd,  ONLY: RealK
  USE def_spectrum, ONLY: StrSpecData
  USE def_control,  ONLY: StrCtrl
  USE def_dimen,    ONLY: StrDim
  USE def_atm,      ONLY: StrAtm, allocate_atm
  USE constant,     ONLY: grav, avog, loschmidt_constant, rgas
  USE def_atm,      ONLY: allocate_atm
  USE planet_rad,   ONLY: p_layer, t_layer, p_level, t_level
  USE gas_list_pcf

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information
  TYPE(StrDim), INTENT(IN) :: &
      dimen
!       Dimensions
  REAL(RealK), INTENT(IN) :: &
      ULGAS(dimen%nd_layer,13)
!       Column densities of absorbers
  INTEGER, INTENT(IN) :: &
      iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
      iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2
!     Identifiers for gases in ULGAS array

! Output variables
  TYPE(StrAtm), INTENT(INOUT) :: &
      atm
!       Atmospheric properties

! This routine sets the atmospheric properties in the atmosphere structure from
! the modelE variables.
! Note that units are converted to SI, and indexing is reversed so that
! layer 1 is the top atmospheric layer.

  CALL allocate_atm(atm, dimen, spectrum)

! Set atmosphere grid
  atm%n_profile = dimen%nd_profile
  atm%n_layer = dimen%nd_layer

! Set pressures and temperatures in atmosphere structure
  atm%p(atm%n_profile,:)        = p_layer
  atm%t(atm%n_profile,:)        = t_layer
  atm%p_level(atm%n_profile,:)  = p_level
  atm%t_level(atm%n_profile,:)  = t_level

! Calculate the dry mass in kg/m**2 in each layer assuming hydrostatic
! equilibrium
  atm%mass(atm%n_profile,:) = (atm%p_level(atm%n_profile, 1:atm%n_layer) - &
      atm%p_level(atm%n_profile,0:atm%n_layer-1))/grav

! Calculate the dry density in each layer using the ideal gas equation
  atm%density(atm%n_profile, :) = atm%p(atm%n_profile, :)/ &
      (rgas*atm%t(atm%n_profile, :))

! Set gas mixing ratios
  CALL set_gas_mix_ratio(control, spectrum, dimen, ULGAS, &
      iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
      iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2, atm)

END SUBROUTINE set_atm


! Set gas mixing ratios
SUBROUTINE set_gas_mix_ratio(control, spectrum, dimen, ULGAS, &
    iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
    iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2, atm)

  USE realtype_rd,  ONLY: RealK
  USE gas_list_pcf
  USE def_spectrum, ONLY: StrSpecData
  USE def_control,  ONLY: StrCtrl
  USE def_dimen,    ONLY: StrDim
  USE def_atm,      ONLY: StrAtm
  USE constant,     ONLY: avog, loschmidt_constant, mair, pN2, pAr, pH2

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information
  TYPE(StrDim), INTENT(IN) :: &
      dimen
!       Dimensions
  REAL(RealK), INTENT(IN) :: &
      ULGAS(dimen%nd_layer,13)
!       Column densities of absorbers
  INTEGER, INTENT(IN) :: &
      iup_h2o, iup_co2, iup_o3, iup_o2, iup_no2, iup_n2o, iup_ch4, &
      iup_cfc11, iup_cfc12, iup_xghg, iup_yghg, iup_so2
!     Identifiers for gases in ULGAS array

! Input/output variables
  TYPE(StrAtm), INTENT(INOUT) :: &
      atm
!       Atmospheric properties


! Column densities are stored in ULGAS in units of cm STP, SOCRATES requires
! gas mixing ratios in kg/kg. This routine converts the ULGAS column densities
! from SETGAS into mass mixing ratios.

  INTEGER :: &
      i, &
!       Loop index
      imep_h2o, &
!       Pointer to H2O
      imep_co2, &
!       Pointer to CO2
      imep_o3, &
!       Pointer to O3
      imep_o2, &
!       Pointer to O2
      imep_no2, &
!       Pointer to NO2
      imep_n2o, &
!       Pointer to N2O
      imep_ch4, &
!       Pointer to CH4
      imep_so2, &
!       Pointer to SO2
      imep_n2, &
!       Pointer to N2
      imep_cfc11, &
!       Pointer to CFC11
      imep_cfc12, &
!       Pointer to CFC12
      imep_cfc113, &
!       Pointer to CFC113
      imep_cfc114, &
!       Pointer to CFC114
      imep_hcfc22, &
!       Pointer to HCFC22
      imep_hfc125, &
!       Pointer to HFC125
      imep_hfc134a, &
!       Pointer to HFC134A
      imep_ar, &
!       Pointer to Ar
      imep_h2
!       Pointer to H2

  atm%gas_mix_ratio(:, :, :) = 0.0_RealK

! Set all gas pointers to zero initially
  imep_h2o = 0
  imep_co2 = 0
  imep_o3 = 0
  imep_o2 = 0
  imep_no2 = 0
  imep_n2o = 0
  imep_ch4 = 0
  imep_so2 = 0
  imep_n2 = 0
  imep_cfc11 = 0
  imep_cfc12 = 0
  imep_cfc113 = 0
  imep_cfc114 = 0
  imep_hcfc22 = 0
  imep_hfc125 = 0
  imep_hfc134a = 0
  imep_ar = 0
  imep_h2 = 0

! Get indices of gases in gas_mix_ratio
  DO i=1,spectrum%gas%n_absorb
    SELECT CASE (spectrum%gas%type_absorb(i))
    CASE (ip_h2o)
      imep_h2o = i
    CASE (ip_co2)
      imep_co2 = i
    CASE (ip_o3)
      imep_o3 = i
    CASE (ip_o2)
      imep_o2 = i
    CASE (ip_no2)
      imep_no2 = i
    CASE (ip_n2o)
      imep_n2o = i
    CASE (ip_ch4)
      imep_ch4 = i
    CASE (ip_so2)
      imep_so2 = i
    CASE (ip_n2)
      imep_n2 = i
    CASE (ip_cfc11)
      imep_cfc11 = i
    CASE (ip_cfc12)
      imep_cfc12 = i
    CASE (ip_cfc113)
      imep_cfc113 = i
    CASE (ip_cfc114)
      imep_cfc114 = i
    CASE (ip_hcfc22)
      imep_hcfc22 = i
    CASE (ip_hfc125)
      imep_hfc125 = i
    CASE (ip_hfc134a)
      imep_hfc134a = i
    CASE (ip_ar)
      imep_ar = i
    CASE (ip_h2)
      imep_h2 = i
    END SELECT
  END DO

! Set gas mixing ratios using the column densities ULGAS, converting from
! number densities in CGS at STP to mass mixing ratios assuming
! hydrostatic equilibrium.

! Check if gas is should be included (l_gasname) and is included in spectral
! file (imep_gasname > 0).

! H2O
  IF (imep_h2o > 0) THEN
    atm%gas_mix_ratio(atm%n_profile,:,imep_h2o) = &
        ULGAS(atm%n_layer:1:-1,iup_h2o)/ &
        (avog/(molar_weight(ip_h2o)*loschmidt_constant*10.0_RealK))/ &
        atm%mass(atm%n_profile,:)
  END IF

! CO2
  IF (control%l_co2) THEN
    IF (imep_co2 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_co2) = &
          ULGAS(atm%n_layer:1:-1,iup_co2)/ &
          (avog/(molar_weight(ip_co2)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: CO2 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! O3
  IF (control%l_o3) THEN
    IF (imep_o3 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_o3) = &
          ULGAS(atm%n_layer:1:-1,iup_o3)/ &
          (avog/(molar_weight(ip_o3)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: O3 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! O2
  IF (control%l_o2) THEN
    IF (imep_o2 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_o2) = &
          ULGAS(atm%n_layer:1:-1,iup_o2)/ &
          (avog/(molar_weight(ip_o2)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: O2 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! NO2
  IF (control%l_no2) THEN
    IF (imep_no2 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_no2) = &
          ULGAS(atm%n_layer:1:-1,iup_no2)/ &
          (avog/(molar_weight(ip_no2)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: NO2 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! N2O
  IF (control%l_n2o) THEN
    IF (imep_n2o > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_n2o) = &
          ULGAS(atm%n_layer:1:-1,iup_n2o)/ &
          (avog/(molar_weight(ip_n2o)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: N2O not found in spectral ' // &
          'file',255)
    END IF
  END IF

! CH4
  IF (control%l_ch4) THEN
    IF (imep_ch4 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_ch4) = &
          ULGAS(atm%n_layer:1:-1,iup_ch4)/ &
          (avog/(molar_weight(ip_ch4)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: CH4 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! SO2
  IF (control%l_so2) THEN
    IF (imep_so2 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_so2) = &
          ULGAS(atm%n_layer:1:-1,iup_so2)/ &
          (avog/(molar_weight(ip_so2)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: SO2 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! N2
  IF (control%l_n2) THEN
    IF (imep_n2 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_n2) = &
          pN2*molar_weight(ip_n2)/mair
    ELSE
      CALL stop_model('set_gas_mix_ratio: N2 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! CFC11
  IF (control%l_cfc11) THEN
    IF (imep_cfc11 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_cfc11) = &
          (ULGAS(atm%n_layer:1:-1,iup_cfc11) + &
          ULGAS(atm%n_layer:1:-1,iup_xghg))/ &
          (avog/(molar_weight(ip_cfc11)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: CFC11 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! CFC12
  IF (control%l_cfc12) THEN
    IF (imep_cfc12 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_cfc12) = &
          ULGAS(atm%n_layer:1:-1,iup_cfc12)/ &
          (avog/(molar_weight(ip_cfc12)*loschmidt_constant*10.0_RealK))/ &
          atm%mass(atm%n_profile,:)
    ELSE
      CALL stop_model('set_gas_mix_ratio: CFC12 not found in spectral ' // &
          'file',255)
    END IF
  END IF

! Ar
  IF (control%l_ar) THEN
    IF (imep_ar > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_ar) = &
          pAr*molar_weight(ip_ar)/mair
    ELSE
      CALL stop_model('set_gas_mix_ratio: Ar not found in spectral ' // &
          'file',255)
    END IF
  END IF

! H2
  IF (control%l_h2) THEN
    IF (imep_h2 > 0) THEN
      atm%gas_mix_ratio(atm%n_profile,:,imep_h2) = &
          pH2*molar_weight(ip_h2)/mair
    ELSE
      CALL stop_model('set_gas_mix_ratio: H2 not found in spectral ' // &
          'file',255)
    END IF
  END IF

END SUBROUTINE set_gas_mix_ratio


! Set the major gas in each band for use with equivalent extinction
SUBROUTINE set_major_abs(spectrum, atm)

  USE realtype_rd,  ONLY: RealK
  USE def_spectrum, ONLY: StrSpecData
  USE def_atm,      ONLY: StrAtm
  USE planet_rad,   ONLY: l_allow_cont_major

  IMPLICIT NONE

! Input variables
  TYPE(StrAtm), INTENT(IN) :: &
      atm
!       Atmospheric properties

! Input/output variables
  TYPE(StrSpecData), INTENT(INOUT) :: &
      spectrum
!       Spectral information

! Local variables
  INTEGER :: &
      i, j, i_band, i_gas, i_cont_gas_1, i_cont_gas_2, i_cont, i_layer, &
!       Loop variables
      major_gas, &
!       Index number of major gas
      n_major_gas, &
!       Order number of major gas
      major_cont, &
!       Index number of major continuum
      n_major_cont, &
!       Order number of major continuum
      n_k
!       Number of k-terms
  REAL(RealK) :: &
      column_mass_gas(MAX(spectrum%gas%n_absorb, 1)), &
!       Column mass for gas
      trans_column_gas(MAX(spectrum%gas%n_absorb, 1)), &
!       Transmission of column for all gases and perfectly correlated
!       continua
      column_mass_cont(MAX(spectrum%contgen%n_cont, 1)), &
!       Column mass for continua
      trans_column_cont(MAX(spectrum%contgen%n_cont, 1)), &
!       Transmission of column for continua
      trans_column_tot, &
!       Total transmission of all gases
      trans_def_major = EXP(-1.0_RealK)
!       Total transmission at which to define major absorber

  DO i_band=1,spectrum%basic%n_band

!   Attempt to define major absorber from where total optical depth reaches 1
    column_mass_gas = 0.0_RealK
    column_mass_cont = 0.0_RealK
    DO i_layer=1,atm%n_layer

!     Gases
      trans_column_gas = 1.0_RealK
      DO i=1,spectrum%gas%n_band_absorb(i_band)
        i_gas = spectrum%gas%index_absorb(i, i_band)
        column_mass_gas(i) = column_mass_gas(i) + &
            atm%gas_mix_ratio(atm%n_profile,i_layer,i_gas)* &
            atm%mass(atm%n_profile,i_layer)
        n_k = spectrum%gas%i_band_k(i_band, i_gas)
        trans_column_gas(i) = SUM(spectrum%gas%w(1:n_k, i_band, i_gas)* &
            EXP(-spectrum%gas%k(1:n_k, i_band, i_gas)*column_mass_gas(i)))
      END DO

!     Continua
      trans_column_cont = 1.0_RealK
      DO j=1,spectrum%contgen%n_band_cont(i_band)
        i_cont = spectrum%contgen%index_cont(j, i_band)
        IF (spectrum%contgen%i_cont_overlap_band(i_band, i_cont) == 0) THEN
!         The overlap treatment for the continuum is the same as that
!         for gases.
          i_cont_gas_1 = spectrum%contgen%index_cont_gas_1(i_cont)
          i_cont_gas_2 = spectrum%contgen%index_cont_gas_2(i_cont)
          column_mass_cont(j) = column_mass_cont(j) + &
              atm%gas_mix_ratio(atm%n_profile,i_layer,i_cont_gas_1)* &
              atm%gas_mix_ratio(atm%n_profile,i_layer,i_cont_gas_2)* &
              atm%density(atm%n_profile,i_layer)* &
              atm%mass(atm%n_profile,i_layer)
          n_k = spectrum%contgen%i_band_k_cont(i_band, i_cont)
          trans_column_cont(j) = &
              SUM(spectrum%contgen%w_cont(1:n_k, i_band, i_cont)* &
              EXP(-spectrum%contgen%k_cont(1:n_k, i_band, i_cont)* &
              column_mass_cont(j)))
        END IF
      END DO

!     Assume random overlap to calculate total band transmission and check if
!     an optical depth of 1 has been reached
      trans_column_tot = PRODUCT(trans_column_gas)*PRODUCT(trans_column_cont)
      IF (trans_column_tot <= trans_def_major) EXIT
    END DO

!   Set major gas. If an optical depth of 1 was not reached the total column
!   mass down to the surface is used.
    IF (spectrum%gas%n_band_absorb(i_band) > 0) THEN
      n_major_gas = MINLOC(trans_column_gas, 1)
      major_gas = spectrum%gas%index_absorb(n_major_gas, i_band)

!     Swap index of major gas with the first gas listed
      spectrum%gas%index_absorb(n_major_gas, i_band) = &
        spectrum%gas%index_absorb(1, i_band)
      spectrum%gas%index_absorb(1, i_band) = major_gas
    ELSE
      n_major_gas = 1
    END IF

!   Set major continuum. If an optical depth of 1 was not reached the total
!   column mass down to the surface is used.
    IF (spectrum%contgen%n_band_cont(i_band) > 0) THEN
      n_major_cont = MINLOC(trans_column_cont, 1)
      major_cont = spectrum%contgen%index_cont(n_major_cont, i_band)

!     Swap index of major continuum with the first continuum listed
      spectrum%contgen%index_cont(n_major_cont, i_band) = &
        spectrum%contgen%index_cont(1, i_band)
      spectrum%contgen%index_cont(1, i_band) = major_cont

!     Determine if the a continuum is the major absorber
      IF (l_allow_cont_major .AND. &
          trans_column_cont(n_major_cont) < trans_column_gas(n_major_gas)) THEN
        spectrum%contgen%l_cont_major(i_band) = .TRUE.
      ELSE
        spectrum%contgen%l_cont_major(i_band) = .FALSE.
      END IF
    END IF

  END DO

END SUBROUTINE set_major_abs


! Set structure with cloud data
SUBROUTINE set_cld(control, spectrum, atm, dimen, &
    inhom_cloud_adj_st, inhom_cloud_adj_cnv, CLDEPS, cld)

  USE realtype_rd,  ONLY: RealK
  USE rad_pcf
  USE def_spectrum, ONLY: StrSpecData
  USE def_atm,      ONLY: StrAtm
  USE def_control,  ONLY: StrCtrl
  USE def_dimen,    ONLY: StrDim
  USE def_cld,      ONLY: StrCld, allocate_cld, allocate_cld_prsc, &
                          allocate_cld_mcica

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information
  TYPE(StrAtm), INTENT(IN) :: &
      atm
!       Atmospheric properties
  TYPE(StrDim), INTENT(IN) :: &
      dimen
!       Dimensions
  REAL(RealK), INTENT(IN) :: &
      inhom_cloud_adj_st, &
!       Scalar cloud inhomogeneity adjustment for stratiform clouds
      inhom_cloud_adj_cnv, &
!       Scalar cloud inhomogeneity adjustment for convective clouds
      CLDEPS(dimen%nd_layer)
!       Cloud heterogeneity

! Input/output variables
  TYPE(StrCld), INTENT(INOUT) :: &
      cld
!       Cloud properties

! Local variables
  REAL(RealK) :: &
      condensed_min_dim(dimen%nd_cloud_component), &
!       Minimum dimensions of condensed components
      condensed_max_dim(dimen%nd_cloud_component)
!       Maximum dimensions of condensed components
  INTEGER :: &
      k
!       Loop index

  CALL allocate_cld(cld, dimen, spectrum)
  CALL allocate_cld_prsc(cld, dimen, spectrum)
  CALL allocate_cld_mcica(cld, dimen, spectrum)

  IF (control%l_cloud) THEN

!   Check if cloud scheme is using the experimental version, which
!   requires dp_corr_strat and dp_corr_conv in control structure to be set
!   (not included here).
    IF (control%i_cloud == ip_cloud_part_corr .OR. &
        control%i_cloud == ip_cloud_part_corr_cnv) THEN
      CALL stop_model('set_cld: Selected cloud scheme not supported', 255)
    END IF

    CALL set_cloud_parametrization(control, spectrum, dimen, cld, &
        condensed_min_dim, condensed_max_dim)

    CALL set_cloud_field(control, atm, dimen, &
        condensed_min_dim, condensed_max_dim, cld)

!   Set correction for cloud inhomogeneities
    SELECT CASE(control%i_inhom)

    CASE (ip_scaling)
      cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_water) = &
          cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_water)* &
          inhom_cloud_adj_st
      cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_ice) = &
          cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_ice)* &
          inhom_cloud_adj_st
      cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_water) = &
          cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_water)* &
          inhom_cloud_adj_cnv
      cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_ice) = &
          cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_ice)* &
          inhom_cloud_adj_cnv

    CASE (ip_cairns)

      DO k=1,cld%n_cloud_type
        cld%condensed_rel_var_dens(atm%n_profile,:,k) = &
            CLDEPS(atm%n_layer:1:-1)/ &
            (1.0_RealK - CLDEPS(atm%n_layer:1:-1))
      END DO

    CASE DEFAULT

      CALL stop_model('set_cld: Select cloud inhomogeneity scheme not ' // &
          'supported.', 255)

    END SELECT

  END IF

END SUBROUTINE set_cld


! Set cloud parameterization
SUBROUTINE set_cloud_parametrization(control, spectrum, dimen, cld, &
    condensed_min_dim, condensed_max_dim)

  USE realtype_rd,  ONLY: RealK
  USE rad_pcf
  USE def_spectrum, ONLY: StrSpecData
  USE def_control,  ONLY: StrCtrl
  USE def_dimen,    ONLY: StrDim
  USE def_cld,      ONLY: StrCld

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information
  TYPE(StrDim), INTENT(IN) :: &
      dimen
!       Dimensions

! Input/output variables
  TYPE(StrCld), INTENT(INOUT) :: &
      cld
!       Cloud properties

! Output variables
  REAL(RealK), INTENT(OUT) :: &
      condensed_min_dim(dimen%nd_cloud_component), &
!       Minimum dimensions of condensed components
      condensed_max_dim(dimen%nd_cloud_component)
!       Maximum dimensions of condensed components

! Local variables
  INTEGER :: &
      i_scheme, &
!       Parametrization scheme
      i, j
!       Loop indices

! External functions
  INTEGER, EXTERNAL :: &
      set_n_cloud_parameter
!       Function to set number of cloudy parameters

! Stratiform water clouds
  IF (control%i_st_water <= spectrum%dim%nd_drop_type .AND. &
      spectrum%drop%l_drop_type(control%i_st_water)) THEN
    i_scheme = spectrum%drop%i_drop_parm(control%i_st_water)
    cld%i_condensed_param(ip_clcmp_st_water) = i_scheme
    cld%condensed_n_phf(ip_clcmp_st_water) = &
        spectrum%drop%n_phf(control%i_st_water)
    condensed_min_dim(ip_clcmp_st_water) = &
        spectrum%drop%parm_min_dim(control%i_st_water)
    condensed_max_dim(ip_clcmp_st_water) = &
        spectrum%drop%parm_max_dim(control%i_st_water)

  ELSE
    CALL stop_model('set_cld: No data for selected droplet type for ' // &
        'stratiform water clouds in spectral file.', 255)
  END IF

  DO i=1, spectrum%basic%n_band
    DO j=1, set_n_cloud_parameter(i_scheme, ip_clcmp_st_water, &
        cld%condensed_n_phf(ip_clcmp_st_water))
      cld%condensed_param_list(j, ip_clcmp_st_water, i) = &
          spectrum%drop%parm_list(j, i, control%i_st_water)
    END DO
  END DO

! Convective water clouds
  IF (control%i_cnv_water <= spectrum%dim%nd_drop_type .AND. &
      spectrum%drop%l_drop_type(control%i_cnv_water)) THEN
    i_scheme = spectrum%drop%i_drop_parm(control%i_cnv_water)
    cld%i_condensed_param(ip_clcmp_cnv_water) = i_scheme
    cld%condensed_n_phf(ip_clcmp_cnv_water) = &
        spectrum%drop%n_phf(control%i_cnv_water)
    condensed_min_dim(ip_clcmp_cnv_water) = &
        spectrum%drop%parm_min_dim(control%i_cnv_water)
    condensed_max_dim(ip_clcmp_cnv_water) = &
        spectrum%drop%parm_max_dim(control%i_cnv_water)
  ELSE
    CALL stop_model('set_cld: No data for selected droplet type for ' // &
        'convective water clouds in spectral file.', 255)
  END IF

  DO i=1, spectrum%basic%n_band
    DO j=1, set_n_cloud_parameter(i_scheme, ip_clcmp_cnv_water, &
        cld%condensed_n_phf(ip_clcmp_cnv_water))
      cld%condensed_param_list(j, ip_clcmp_cnv_water, i) = &
          spectrum%drop%parm_list(j, i, control%i_cnv_water)
    END DO
  END DO

! Stratiform ice clouds
  IF (control%i_st_ice <= spectrum%dim%nd_ice_type .AND. &
      spectrum%ice%l_ice_type(control%i_st_ice)) THEN
    i_scheme = spectrum%ice%i_ice_parm(control%i_st_ice)
    cld%i_condensed_param(ip_clcmp_st_ice) = i_scheme
    cld%condensed_n_phf(ip_clcmp_st_ice) = spectrum%ice%n_phf(control%i_st_ice)
    condensed_min_dim(ip_clcmp_st_ice) = &
        spectrum%ice%parm_min_dim(control%i_st_ice)
    condensed_max_dim(ip_clcmp_st_ice) = &
        spectrum%ice%parm_max_dim(control%i_st_ice)
  ELSE
    CALL stop_model('set_cld: No data for selected crystal type for ' // &
        'stratiform ice clouds in spectral file.', 255)
  END IF

  DO i=1, spectrum%basic%n_band
    DO j=1, set_n_cloud_parameter(i_scheme, ip_clcmp_st_ice, &
        cld%condensed_n_phf(ip_clcmp_st_ice))
      cld%condensed_param_list(j, ip_clcmp_st_ice, i) = &
          spectrum%ice%parm_list(j, i, control%i_st_ice)
    END DO
  END DO

! Convective ice clouds
  IF (control%i_cnv_ice <= spectrum%dim%nd_ice_type .AND. &
      spectrum%ice%l_ice_type(control%i_cnv_ice)) THEN
    i_scheme = spectrum%ice%i_ice_parm(control%i_cnv_ice)
    cld%i_condensed_param(ip_clcmp_cnv_ice) = i_scheme
    cld%condensed_n_phf(ip_clcmp_cnv_ice) = &
        spectrum%ice%n_phf(control%i_cnv_ice)
    condensed_min_dim(ip_clcmp_cnv_ice) = &
       spectrum%ice%parm_min_dim(control%i_cnv_ice)
    condensed_max_dim(ip_clcmp_cnv_ice) = &
        spectrum%ice%parm_max_dim(control%i_cnv_ice)
  ELSE
    CALL stop_model('set_cld: No data for selected crystal type for ' // &
        'convective ice clouds in spectral file.', 255)
  END IF

  DO i=1, spectrum%basic%n_band
    DO j=1, set_n_cloud_parameter(i_scheme, ip_clcmp_cnv_ice, &
        cld%condensed_n_phf(ip_clcmp_cnv_ice))
      cld%condensed_param_list(j, ip_clcmp_cnv_ice, i) = &
          spectrum%ice%parm_list(j, i, control%i_cnv_ice)
    END DO
  END DO

END SUBROUTINE set_cloud_parametrization


! Set cloud fields
SUBROUTINE set_cloud_field(control, atm, dimen, &
    condensed_min_dim, condensed_max_dim, cld)

  USE realtype_rd, ONLY: RealK
  USE rad_pcf
  USE def_control, ONLY: StrCtrl
  USE def_atm,     ONLY: StrAtm
  USE def_dimen,   ONLY: StrDim
  USE def_cld,     ONLY: StrCld
  USE planet_rad,  ONLY: w_cloud_l, frac_st_water_l, frac_st_ice_l, &
                         frac_cnv_water_l, frac_cnv_ice_l, &
                         mix_ratio_st_water_l, mix_ratio_st_ice_l, &
                         mix_ratio_cnv_water_l, mix_ratio_cnv_ice_l, &
                         dim_char_st_water_l, dim_char_st_ice_l, &
                         dim_char_cnv_water_l, dim_char_cnv_ice_l, &
                         frac_area_st_l, frac_area_cnv_l, &
                         cld_scaling, i_cloud_ice_size_scheme

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrAtm), INTENT(IN) :: &
      atm
!       Atmospheric properties
  TYPE(StrDim), INTENT(IN) :: &
      dimen
!       Dimensions
  REAL(RealK), INTENT(IN) :: &
      condensed_min_dim(dimen%nd_cloud_component), &
!       Minimum dimensions of condensed components
      condensed_max_dim(dimen%nd_cloud_component)
!       Maximum dimensions of condensed components

! Input/output variables
  TYPE(StrCld), INTENT(INOUT) :: &
      cld
!       Cloud properties

! Local variables
  REAL(RealK), PARAMETER :: &
      tol_frac_cloud = 1.0E-10_RealK
!       Tolerance on cloud fraction where cloud is not included in radiation
! Schemes for calculating effective ice particle size
  INTEGER, PARAMETER :: &
      ip_cloud_ice_size_giss = 1, &
!       Effective ice cloud particle size used is 2*SIZEIC
      ip_cloud_ice_size_um = 2
!       Effective ice cloud particle size based on UM parametrization
  INTEGER :: &
    i_layer
!       Layer loop index

! Parameters for the aggregate parametrization
  REAL(RealK), PARAMETER :: a0_agg_cold = 7.5094588E-04_RealK
  REAL(RealK), PARAMETER :: b0_agg_cold = 5.0830326E-07_RealK
  REAL(RealK), PARAMETER :: a0_agg_warm = 1.3505403E-04_RealK
  REAL(RealK), PARAMETER :: b0_agg_warm = 2.6517429E-05_RealK
  REAL(RealK), PARAMETER :: t_switch    = 216.208_RealK
  REAL(RealK), PARAMETER :: t0_agg      = 279.5_RealK
  REAL(RealK), PARAMETER :: s0_agg      = 0.05_RealK

! Initialise all clouds fields to zero
  cld%w_cloud = 0.0_RealK
  cld%frac_cloud = 0.0_RealK
  cld%condensed_mix_ratio = 0.0_RealK
  cld%condensed_dim_char = 0.0_RealK

  SELECT CASE (control%i_cloud_representation)

  CASE (ip_cloud_csiw)

!   Four different cloud components: stratiform ice and water, and convective
!   ice and water
    cld%n_cloud_type = 4
    cld%n_condensed = 4
    cld%type_condensed(1) = ip_clcmp_st_water
    cld%type_condensed(2) = ip_clcmp_st_ice
    cld%type_condensed(3) = ip_clcmp_cnv_water
    cld%type_condensed(4) = ip_clcmp_cnv_ice

!   Now we set the cloud properties needed by SOCRATES using the variables
!   from CLOUDS_COM, set in CLOUDS2_DRV.

!   Total cloud cover
    cld%w_cloud(atm%n_profile,:) = w_cloud_l

!   In-cloud fractions of different types of clouds
    cld%frac_cloud(atm%n_profile,:,ip_cloud_type_sw) = &
        frac_st_water_l
    cld%frac_cloud(atm%n_profile,:,ip_cloud_type_si) = &
        frac_st_ice_l
    cld%frac_cloud(atm%n_profile,:,ip_cloud_type_cw) = &
        frac_cnv_water_l
    cld%frac_cloud(atm%n_profile,:,ip_cloud_type_ci) = &
        frac_cnv_ice_l

!   Mass mixing ratios of different types of clouds
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_water) = &
        cld_scaling*mix_ratio_st_water_l/ &
        (frac_area_st_l + TINY(frac_area_st_l))
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_ice) = &
        cld_scaling*mix_ratio_st_ice_l/ &
        (frac_area_st_l + TINY(frac_area_st_l))
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_water) = &
        cld_scaling*mix_ratio_cnv_water_l/ &
        (frac_area_cnv_l + TINY(frac_area_cnv_l))
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_ice) = &
        cld_scaling*mix_ratio_cnv_ice_l/ &
        (frac_area_cnv_l + TINY(frac_area_cnv_l))

  CASE DEFAULT

    CALL stop_model('set_cld: Selected cloud representation not ' // &
        'implemented with ModelE.', 255)

  END SELECT ! i_cloud_representation

! Effective radius of water cloud particles
  IF (control%l_microphysics) THEN
!   Effective cloud droplet radius is provided by cloud scheme
    cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_st_water) = &
        dim_char_st_water_l
    cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_water) = &
        dim_char_cnv_water_l
  ELSE
!   Effective radii are set to standard values
    cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_st_water) = &
        7.0E-06_RealK
    cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_water) = &
        7.0E-06_RealK
  END IF

! Characteristic dimension of stratiform ice cloud particles determined by
! parametrization
  IF (i_cloud_ice_size_scheme == ip_cloud_ice_size_giss) THEN
    cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_st_ice) = &
        2.0_RealK*dim_char_st_ice_l
  ELSE IF (i_cloud_ice_size_scheme == ip_cloud_ice_size_um) THEN
    SELECT CASE (cld%i_condensed_param(ip_clcmp_st_ice))

    CASE (ip_ice_agg_de, ip_ice_agg_de_sun)
!     Remember ip_ice_agg_de == ip_ice_fu_phf

!     Aggregate parametrization based on effective dimension.
!     In the initial form, the same approach is used for stratiform
!     and convective cloud.

!     The fit provided here is based on Stephan Havemann's fit of
!     Dge with temperature, consistent with David Mitchell's treatment
!     of the variation of the size distribution with temperature. The
!     parametrization of the optical properties is based on De
!     (=(3/2)volume/projected area), whereas Stephan's fit gives Dge
!     (=(2*SQRT(3)/3)*volume/projected area), which explains the
!     conversion factor. The fit to Dge is in two sections, because
!     Mitchell's relationship predicts a cusp at 216.208 K. Limits
!     of 8 and 124 microns are imposed on Dge: these are based on this
!     relationship and should be reviewed if it is changed. Note also
!     that the relationship given here is for polycrystals only.

!     Preliminary calculation of Dge
      WHERE (atm%t(atm%n_profile,:) < t_switch)
        cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_st_ice) = &
            a0_agg_cold*EXP(s0_agg*(atm%t(atm%n_profile,:) - t0_agg)) + &
            b0_agg_cold
      ELSE WHERE
        cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_st_ice) = &
            a0_agg_warm*EXP(s0_agg*(atm%t(atm%n_profile,:) - t0_agg)) + &
            b0_agg_warm
      END WHERE

!     Limit and convert to De
      cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_st_ice) = &
          (3.0_RealK/2.0_RealK)*(3.0_RealK/(2.0_RealK*SQRT(3.0_RealK)))* &
          MIN(1.24E-04_RealK, MAX(8.0E-06_RealK, &
          cld%condensed_dim_char(atm%n_profile, :, ip_clcmp_st_ice)))

    CASE DEFAULT

      CALL stop_model('set_cld: Selected convective ice cloud particle ' // &
        'parametrization not implemented with ModelE.', 255)

    END SELECT

  ELSE

    CALL stop_model('set_cld: Selected ice cloud particle size scheme ' // &
        'not found.', 255)

  END IF

! Characteristic dimension of convective ice cloud particles determined by
! parametrization
  IF (i_cloud_ice_size_scheme == ip_cloud_ice_size_giss) THEN
    cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_ice) = &
        2.0_RealK*dim_char_cnv_ice_l
  ELSE IF (i_cloud_ice_size_scheme == ip_cloud_ice_size_um) THEN
    SELECT CASE (cld%i_condensed_param(ip_clcmp_cnv_ice))

    CASE (ip_ice_agg_de, ip_ice_agg_de_sun)
!     Remember ip_ice_agg_de == ip_ice_fu_phf

!     Aggregate parametrization based on effective dimension.
!     In the initial form, the same approach is used for stratiform
!     and convective cloud.

!     The fit provided here is based on Stephan Havemann's fit of
!     Dge with temperature, consistent with David Mitchell's treatment
!     of the variation of the size distribution with temperature. The
!     parametrization of the optical properties is based on De
!     (=(3/2)volume/projected area), whereas Stephan's fit gives Dge
!     (=(2*SQRT(3)/3)*volume/projected area), which explains the
!     conversion factor. The fit to Dge is in two sections, because
!     Mitchell's relationship predicts a cusp at 216.208 K. Limits
!     of 8 and 124 microns are imposed on Dge: these are based on this
!     relationship and should be reviewed if it is changed. Note also
!     that the relationship given here is for polycrystals only.

!     Preliminary calculation of Dge
      WHERE (atm%t(atm%n_profile,:) < t_switch)
        cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_ice) = &
            a0_agg_cold*EXP(s0_agg*(atm%t(atm%n_profile,:) - t0_agg)) + &
            b0_agg_cold
      ELSE WHERE
        cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_ice) = &
            a0_agg_warm*EXP(s0_agg*(atm%t(atm%n_profile,:) - t0_agg)) + &
            b0_agg_warm
      END WHERE

!     Limit and convert to De
      cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_ice) = &
          (3.0_RealK/2.0_RealK)*(3.0_RealK/(2.0_RealK*SQRT(3.0_RealK)))* &
          MIN(1.24E-04_RealK, MAX(8.0E-06_RealK, &
          cld%condensed_dim_char(atm%n_profile,:,ip_clcmp_cnv_ice)))

    CASE DEFAULT

      CALL stop_model('set_cld: Selected convective ice cloud particle ' // &
        'parametrization not implemented with ModelE.', 255)

    END SELECT

  ELSE

    CALL stop_model('set_cld: Selected ice cloud particle size scheme ' // &
        'not found.', 255)

  END IF

! Cloud fraction/mixing ratio consistency check
  WHERE (cld%frac_cloud(atm%n_profile,:,ip_cloud_type_sw) <= tol_frac_cloud)
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_water) = 0.0_RealK
  END WHERE
  WHERE (cld%frac_cloud(atm%n_profile,:,ip_cloud_type_si) <= tol_frac_cloud)
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_st_ice) = 0.0_RealK
  END WHERE
  WHERE (cld%frac_cloud(atm%n_profile,:,ip_cloud_type_cw) <= tol_frac_cloud)
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_water) = 0.0_RealK
  END WHERE
  WHERE (cld%frac_cloud(atm%n_profile,:,ip_cloud_type_ci) <= tol_frac_cloud)
    cld%condensed_mix_ratio(atm%n_profile,:,ip_clcmp_cnv_ice) = 0.0_RealK
  END WHERE

! Constrain the sizes of cloud particles to lie within the range of validity of
! the parametrization scheme
  DO i_layer = 1, atm%n_layer
    cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_st_water) = &
        MAX(condensed_min_dim(ip_clcmp_st_water), &
        MIN(condensed_max_dim(ip_clcmp_st_water), &
        cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_st_water)))
    cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_st_ice) = &
        MAX(condensed_min_dim(ip_clcmp_st_ice), &
        MIN(condensed_max_dim(ip_clcmp_st_ice), &
        cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_st_ice)))
    cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_cnv_water) = &
        MAX(condensed_min_dim(ip_clcmp_cnv_water), &
        MIN(condensed_max_dim(ip_clcmp_cnv_water), &
        cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_cnv_water)))
    cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_cnv_ice) = &
        MAX(condensed_min_dim(ip_clcmp_cnv_ice), &
        MIN(condensed_max_dim(ip_clcmp_cnv_ice), &
        cld%condensed_dim_char(atm%n_profile, i_layer, ip_clcmp_cnv_ice)))
  END DO

END SUBROUTINE set_cloud_field


! Set aerosol data
SUBROUTINE set_aer(control, spectrum, dimen, atm, aer_tbl, aer)

  USE realtype_rd,   ONLY: RealK
  USE rad_pcf
  USE def_spectrum,  ONLY: StrSpecData
  USE def_control,   ONLY: StrCtrl
  USE def_dimen,     ONLY: StrDim
  USE def_atm,       ONLY: StrAtm
  USE def_aer,       ONLY: StrAer, allocate_aer, allocate_aer_prsc
  USE def_planetstr, ONLY: StrAerData
  USE planet_rad,    ONLY: n_aer, aer_component, aer_radius_eff_dry, &
                           aer_mass_div_dens, aer_frac_sulph, &
                           aer_scaling_sw, aer_scaling_lw, rel_humidity

  IMPLICIT NONE

! Input variables
  TYPE(StrCtrl), INTENT(IN) :: &
      control
!       Control options
  TYPE(StrSpecData), INTENT(IN) :: &
      spectrum
!       Spectral information
  TYPE(StrDim), INTENT(IN) :: &
      dimen
!       Dimensions
  TYPE(StrAtm), INTENT(IN) :: &
      atm
!       Atmospheric properties
  TYPE(StrAerData), INTENT(IN) :: &
      aer_tbl
!       Table of aerosol optical properties

! Input/output variables
  TYPE(StrAer), INTENT(INOUT) :: &
      aer
!       Aerosol properties

! Local variables
  INTEGER :: &
      n_profile, &
!       Number of profiles (= 1)
      n_layer, &
!       Number of atmospheric layers
      n_band
!       Number of bands

! Ease access to array sizes
  n_profile = atm%n_profile
  n_layer = atm%n_layer
  n_band = aer_tbl%n_band

  CALL allocate_aer(aer, dimen, spectrum)
  CALL allocate_aer_prsc(aer, dimen, spectrum)

! CLASSIC aerosols are off
  aer%mr_type_index = 0
  aer%mr_source = ip_aersrc_classic_roff
  aer%mix_ratio = 0.0_RealK
  WHERE (rel_humidity <= 1.0_RealK)
    aer%mean_rel_humidity(n_profile, :) = rel_humidity
  ELSE WHERE
    aer%mean_rel_humidity(n_profile, :) = 1.0_RealK
  END WHERE

  IF (control%l_aerosol) THEN
!   Set aerosol optical properties

!   Check that number of aerosols does not exceed the allocated space
    IF (n_aer > dimen%nd_aerosol_mode) THEN
      CALL stop_model('set_aer: Number of aerosols exceeds maximum number ' // &
          'allowed. Increase npd_aer in planet_rad.', 255)
    END IF

!   Calculate the aerosol optical properties
    aer%n_mode = n_aer
    IF (control%isolir == ip_solar) THEN
      CALL calc_aer_opt_prop(n_layer, n_aer, atm%mass(n_profile, :), &
          aer_component, aer_radius_eff_dry, &
          aer_mass_div_dens, aer_frac_sulph, &
          aer_scaling_sw, rel_humidity, aer_tbl, &
          aer%mode_mix_ratio(n_profile, :, :), &
          aer%mode_absorption(n_profile, :, :, :), &
          aer%mode_scattering(n_profile, :, :, :), &
          aer%mode_asymmetry(n_profile, :, :, :), &
          dimen%nd_layer, dimen%nd_aerosol_mode, n_band)
    ELSE IF (control%isolir == ip_infra_red) THEN
      CALL calc_aer_opt_prop(n_layer, n_aer, atm%mass(n_profile, :), &
          aer_component, aer_radius_eff_dry, &
          aer_mass_div_dens, aer_frac_sulph, &
          aer_scaling_lw, rel_humidity, aer_tbl, &
          aer%mode_mix_ratio(n_profile, :, :), &
          aer%mode_absorption(n_profile, :, :, :), &
          aer%mode_scattering(n_profile, :, :, :), &
          aer%mode_asymmetry(n_profile, :, :, :), &
          dimen%nd_layer, dimen%nd_aerosol_mode, n_band)
    END IF

  ELSE
!   Aerosols are off
    aer%n_mode = 0
  ENDIF

END SUBROUTINE set_aer

! Calculates aerosol mixing ratio, absorption, scattering and asymmety
SUBROUTINE calc_aer_opt_prop(n_layer, n_aer, atm_mass, aer_component, &
    aer_radius_eff_dry, aer_mass_div_dens, aer_frac_sulph, aer_scaling, &
    rel_humidity, aer_tbl, &
    aer_mix_ratio, aer_absorption, aer_scattering, aer_asymmetry, &
    nd_layer, nd_aer, nd_band)

  USE realtype_rd,   ONLY: RealK
  USE def_planetstr, ONLY: StrAerData
  USE rad_pcf

  IMPLICIT NONE

! Sizes of dummy arrays
  INTEGER, INTENT(IN) :: &
      nd_layer, &
!       Size allocated for layers
      nd_aer, &
!       Size allocated for aerosols
      nd_band
!       Size allocated for bands

! Input variables
  INTEGER, INTENT(IN) :: &
      n_aer, &
!       Number of aerosols
      n_layer, &
!       Number of atmospheric layers
      aer_component(nd_aer)
!       Component index of aerosol (as defined in rad_ccf)
  REAL(RealK) :: &
      atm_mass(nd_layer), &
!       Column density of each atmospheric layer
      aer_radius_eff_dry(nd_aer), &
!       Effective radii
      aer_mass_div_dens(nd_aer, nd_layer), &
!       Aerosol mass divided by aerosol bulk density
      aer_frac_sulph(nd_aer), &
!       Sulfate fraction of basic aerosol composition
      aer_scaling(nd_aer), &
!       Scaling of aerosol mass for short-wave
      rel_humidity(nd_layer)
!       Relative humidity
  TYPE(StrAerData), INTENT(IN) :: &
      aer_tbl
!       Table of aerosol optical properties

! Output variables
  REAL(RealK), INTENT(OUT) :: &
      aer_mix_ratio(nd_layer, nd_aer), &
!       Aerosol mass mixing ratio
      aer_absorption(nd_layer, nd_aer, nd_band), &
!       Aerosol absorption coefficient
      aer_scattering(nd_layer, nd_aer, nd_band), &
!       Aerosol scattering coefficient
      aer_asymmetry(nd_layer, nd_aer, nd_band)
!       Aerosol asymmetry

! Local variables
  INTEGER :: &
      n_band, &
!       Number of bands
      i_aer_tbl, &
!       Aerosol component index in aer_tbl
      i_aer
!       Aerosol loop index
  REAL(RealK) :: &
      k_abs(n_layer, aer_tbl%n_band), &
!       Interpolated absorption coefficients
      k_scat(n_layer, aer_tbl%n_band), &
!       Interpolated scattering coefficients
      g_asym(n_layer, aer_tbl%n_band), &
!       Interpolated asymmetry parameters
      k_abs_sulph(n_layer, aer_tbl%n_band), &
!       Interpolated absorption coefficients for sulphate
      k_scat_sulph(n_layer, aer_tbl%n_band), &
!       Interpolated scattering coefficients for sulphate
      g_asym_sulph(n_layer, aer_tbl%n_band)
!       Interpolated asymmetry parameters for sulphate

! Ease access to array sizes
  n_band = aer_tbl%n_band

  DO i_aer = 1, n_aer

!   Find component index in aer_tbl
    i_aer_tbl = aer_tbl%i_component_map(aer_component(i_aer))

!   Calculate and set mass mixing ratio
    aer_mix_ratio(1:n_layer, i_aer) = &
        aer_mass_div_dens(i_aer, 1:n_layer)*aer_tbl%density(i_aer_tbl)/ &
        atm_mass(1:n_layer)

!   Scale mixing ratio
    aer_mix_ratio(1:n_layer, i_aer) = &
        aer_mix_ratio(1:n_layer, i_aer)*aer_scaling(i_aer)

!   Interpolate aerosol optical properties to correct humidity and
!   effective radius
    CALL interp_aer(aer_component(i_aer), aer_radius_eff_dry(i_aer), &
        aer_tbl, n_layer, n_band, &
        k_abs(1:n_layer, 1:n_band), &
        k_scat(1:n_layer, 1:n_band), &
        g_asym(1:n_layer, 1:n_band))

!   Add optical properties to SOCRATES input arrays. Remember to convert to
!   correct bulk density and weight the asymmetry parameter by the scattering
!   coefficient.
    IF (aer_frac_sulph(i_aer) > 0.0_RealK) THEN
!     Sulphate fraction is non-zero, include sulphate in the optical properties

!     Interpolate sulphate optical properties to correct humidity and radius
      CALL interp_aer(ip_ammonium_sulphate, aer_radius_eff_dry(i_aer), &
          aer_tbl, n_layer, n_band, &
          k_abs_sulph(1:n_layer, 1:n_band), &
          k_scat_sulph(1:n_layer, 1:n_band), &
          g_asym_sulph(1:n_layer, 1:n_band))

!     Add sulphate optical properties to the optical properties of current
!     aerosol
      aer_absorption(1:n_layer, i_aer, 1:n_band) = &
          (1.0_RealK - aer_frac_sulph(i_aer))* &
          k_abs(1:n_layer, 1:n_band) + &
          aer_frac_sulph(i_aer)*k_abs_sulph(1:n_layer, 1:n_band)* &
          aer_tbl%density(aer_tbl%i_component_map(ip_ammonium_sulphate))/ &
          aer_tbl%density(i_aer_tbl)
      aer_scattering(1:n_layer, i_aer, 1:n_band) = &
          (1.0_RealK - aer_frac_sulph(i_aer))* &
          k_scat(1:n_layer, 1:n_band) + &
          aer_frac_sulph(i_aer)*k_scat_sulph(1:n_layer, 1:n_band)* &
          aer_tbl%density(aer_tbl%i_component_map(ip_ammonium_sulphate))/ &
          aer_tbl%density(i_aer_tbl)
      aer_asymmetry(1:n_layer, i_aer, 1:n_band) = &
          ((1.0_RealK - aer_frac_sulph(i_aer))*k_scat(1:n_layer, 1:n_band)* &
          g_asym(1:n_layer, 1:n_band) + &
          aer_frac_sulph(i_aer)*k_scat_sulph(1:n_layer, 1:n_band)* &
          g_asym_sulph(1:n_layer, 1:n_band))/ &
          ((1.0_RealK - aer_frac_sulph(i_aer))*k_scat(1:n_layer, 1:n_band) + &
          aer_frac_sulph(i_aer)*k_scat_sulph(1:n_layer, 1:n_band))
    ELSE
!     No sulphate

      aer_absorption(1:n_layer, i_aer, 1:n_band) = k_abs(1:n_layer, 1:n_band)
      aer_scattering(1:n_layer, i_aer, 1:n_band) = k_scat(1:n_layer, 1:n_band)
      aer_asymmetry(1:n_layer, i_aer, 1:n_band) = g_asym(1:n_layer, 1:n_band)

    END IF

  END DO

END SUBROUTINE


! Interpolates aerosol properties to correct effective radius and relative
! humidity
SUBROUTINE interp_aer(i_aer_component, aer_radius_eff_dry, aer_tbl, &
    n_layer, n_band, k_abs, k_scat, g_asym)

  USE realtype_rd,   ONLY: RealK
  USE rad_pcf
  USE def_planetstr, ONLY: StrAerData
  USE planet_rad,    ONLY: rel_humidity

  IMPLICIT NONE

! Input variables
  INTEGER, INTENT(IN) :: &
      i_aer_component, &
!       Index of aerosol component as defined in rad_pcf
      n_layer, &
!       Number of atmospheric layers
      n_band
!       Number of spectral bands
  REAL(RealK), INTENT(IN) :: &
      aer_radius_eff_dry
!       Dry effective radius
  TYPE(StrAerData), INTENT(IN) :: &
      aer_tbl
!       Table of aerosol optical properties

! Output variables
  REAL(RealK), INTENT(OUT) :: &
      k_abs(n_layer, n_band), &
!       Interpolated absorption coefficients
      k_scat(n_layer, n_band), &
!       Interpolated scattering coefficients
      g_asym(n_layer, n_band)
!       Interpolated asymmetry parameters

! Local variables
  INTEGER :: &
      n_humidity, &
!       Number of humidities in table
      i_aer_tbl, &
!       Index of aerosol component in aerosol table
      n_radius, &
!       Number of radii in table
      i_layer, &
!       Layer loop index
      i_radius, &
!       Radius loop index
      i_humidity_pointer(n_layer), &
!       Pointer to lower humidity index for interpolation
      i_radius_pointer
!       Pointer to lower radius index for interpolation
  REAL(RealK) :: &
      delta_humidity, &
!       Step in humidity array
      weight_humidity_upper(n_layer), &
!       Weight for upper humidity point for interpolation
      weight_radius_upper, &
!       Weight for upper humidity point for interpolation
      k_abs_tmp(n_layer, aer_tbl%n_radius_max, n_band), &
!       Absorption coefficients interpolated to correct humidity
      k_scat_tmp(n_layer, aer_tbl%n_radius_max, n_band), &
!       Scattering coefficients interpolated to correct humidity
      g_asym_tmp(n_layer, aer_tbl%n_radius_max, n_band)
!       Asymmetry parameter interpolated to correct humidity

! Ease access to number of humidities
  n_humidity = aer_tbl%n_humidity

! Find component index in aer_tbl
  i_aer_tbl = aer_tbl%i_component_map(i_aer_component)

! Number of radii in table of this component
  n_radius = aer_tbl%n_radius(i_aer_tbl)

! Find weights for interpolation in humidities. Assume equal spacing in
! humidities (this is done in the SOCRATES code as well).
  IF (aer_tbl%l_humidity(i_aer_tbl)) THEN
    delta_humidity = 1.0e+00_RealK/REAL(n_humidity - 1, RealK)
    DO i_layer = 1, n_layer
      i_humidity_pointer(i_layer) = MIN(1 + &
          INT(rel_humidity(i_layer)*(n_humidity - 1)), &
          n_humidity - 1)
      weight_humidity_upper(i_layer) = ( &
          MIN(1.0_RealK, rel_humidity(i_layer)) - &
          aer_tbl%humidity(i_humidity_pointer(i_layer)))/ &
          delta_humidity
    END DO
  END IF

! Find weights for interpolation in dry effective radius
  IF (aer_radius_eff_dry <= &
      aer_tbl%radius_eff_dry(1, i_aer_tbl)) THEN
!   Effective radius is below the smallest effective radius in table
    i_radius_pointer = 1
    weight_radius_upper = 0.0_RealK
    
!   Print warning
    WRITE(6,'(A,ES7.1,A,ES7.1)') &
        'Warning: Aerosol effective radius below lower limit of table for ' // &
        TRIM(name_aerosol_component(aer_tbl%i_component(i_aer_tbl))) // &
        ': ', aer_radius_eff_dry, ' <= ', &
        aer_tbl%radius_eff_dry(1, i_aer_tbl)

  ELSE IF (aer_radius_eff_dry >= &
      aer_tbl%radius_eff_dry(n_radius, i_aer_tbl)) THEN
!   Effective radius is above the largest effective radius in table
    i_radius_pointer = n_radius - 1
    weight_radius_upper = 1.0_RealK

!   Print warning
    WRITE(6,'(A,ES7.1,A,ES7.1)') &
        'Warning: Aerosol effective radius above upper limit of table for ' // &
        TRIM(name_aerosol_component(aer_tbl%i_component(i_aer_tbl))) // &
        ': ', aer_radius_eff_dry, ' >= ', &
        aer_tbl%radius_eff_dry(n_radius, i_aer_tbl)

  ELSE
!   Effective radius is within limits of table, locate closest index
    DO i_radius = 1, n_radius
      IF (aer_radius_eff_dry <= &
          aer_tbl%radius_eff_dry(i_radius, i_aer_tbl)) THEN
        i_radius_pointer = i_radius - 1
        weight_radius_upper = (aer_radius_eff_dry - &
            aer_tbl%radius_eff_dry(i_radius_pointer, i_aer_tbl))/ &
            (aer_tbl%radius_eff_dry(i_radius_pointer + 1, i_aer_tbl) - &
            aer_tbl%radius_eff_dry(i_radius_pointer, i_aer_tbl))
        EXIT
      END IF
    END DO
  END IF

! Interpolate in humidity if needed
  IF (aer_tbl%l_humidity(i_aer_tbl)) THEN
!   Aerosol data has humidity dependence

    DO i_layer = 1, n_layer
      k_abs_tmp(i_layer, 1:n_radius, 1:n_band) = TRANSPOSE( &
          (1.0_RealK - weight_humidity_upper(i_layer))* &
          aer_tbl%k_abs(1:n_band, i_humidity_pointer(i_layer), &
          1:n_radius, i_aer_tbl) + &
          weight_humidity_upper(i_layer)* &
          aer_tbl%k_abs(1:n_band, i_humidity_pointer(i_layer) + 1, &
          1:n_radius, i_aer_tbl))
      k_scat_tmp(i_layer, 1:n_radius, 1:n_band) = TRANSPOSE( &
          (1.0_RealK - weight_humidity_upper(i_layer))* &
          aer_tbl%k_scat(1:n_band, i_humidity_pointer(i_layer), &
          1:n_radius, i_aer_tbl) + &
          weight_humidity_upper(i_layer)* &
          aer_tbl%k_scat(1:n_band, i_humidity_pointer(i_layer) + 1, &
          1:n_radius, i_aer_tbl))
      g_asym_tmp(i_layer, 1:n_radius, 1:n_band) = TRANSPOSE( &
          (1.0_RealK - weight_humidity_upper(i_layer))* &
          aer_tbl%g_asym(1:n_band, i_humidity_pointer(i_layer), &
          1:n_radius, i_aer_tbl) + &
          weight_humidity_upper(i_layer)* &
          aer_tbl%g_asym(1:n_band, i_humidity_pointer(i_layer) + 1, &
          1:n_radius, i_aer_tbl))
    END DO

  ELSE
!   No humidity dependence, data is the same for all humidities

    DO i_layer = 1, n_layer
      k_abs_tmp(i_layer, 1:n_radius, 1:n_band) = TRANSPOSE( &
          aer_tbl%k_abs(1:n_band, 1, 1:n_radius, i_aer_tbl))
      k_scat_tmp(i_layer, 1:n_radius, 1:n_band) = TRANSPOSE( &
          aer_tbl%k_scat(1:n_band, 1, 1:n_radius, i_aer_tbl))
      g_asym_tmp(i_layer, 1:n_radius, 1:n_band) = TRANSPOSE( &
          aer_tbl%g_asym(1:n_band, 1, 1:n_radius, i_aer_tbl))
    END DO

  END IF

! Interpolate in dry effective radius
  k_abs(1:n_layer, 1:n_band) = &
      (1.0_RealK - weight_radius_upper)* &
      k_abs_tmp(1:n_layer, i_radius_pointer, 1:n_band) + &
      weight_radius_upper* &
      k_abs_tmp(1:n_layer, i_radius_pointer + 1, 1:n_band)
  k_scat(1:n_layer, 1:n_band) = &
      (1.0_RealK - weight_radius_upper)* &
      k_scat_tmp(1:n_layer, i_radius_pointer, 1:n_band) + &
      weight_radius_upper* &
      k_scat_tmp(1:n_layer, i_radius_pointer + 1, 1:n_band)
  g_asym(1:n_layer, 1:n_band) = &
      (1.0_RealK - weight_radius_upper)* &
      g_asym_tmp(1:n_layer, i_radius_pointer, 1:n_band) + &
      weight_radius_upper* &
      g_asym_tmp(1:n_layer, i_radius_pointer + 1, 1:n_band)

END SUBROUTINE interp_aer
