#include "rundeck_opts.h"

MODULE planet_alb

  USE realtype_rd, ONLY: RealK

  IMPLICIT NONE

!-------------------------------------------------------------------------------
! Long-wave albedo data
!-------------------------------------------------------------------------------

! Albedo bands
  INTEGER, PARAMETER :: n_band_alb_lw = 30
  REAL(RealK), PARAMETER :: &
      alb_band_min_lw(n_band_alb_lw) = 1.0E+02_RealK*(/ &
           0.0_RealK,  100.0_RealK,  230.0_RealK,  400.0_RealK,  570.0_RealK, &
         610.0_RealK,  650.0_RealK,  690.0_RealK,  730.0_RealK,  760.0_RealK, &
         790.0_RealK,  850.0_RealK,  900.0_RealK,  970.0_RealK, 1010.0_RealK, &
        1070.0_RealK, 1210.0_RealK, 1330.0_RealK, 1450.0_RealK, 1590.0_RealK, &
        1720.0_RealK, 1900.0_RealK, 2020.0_RealK, 2080.0_RealK, 2130.0_RealK, &
        2210.0_RealK, 2250.0_RealK, 2300.0_RealK, 2380.0_RealK, 2390.0_RealK  &
        /), &
!       Lower band limits [m-1] used to define long-wave albedos
      alb_band_max_lw(n_band_alb_lw) = 1.0E+02_RealK*(/ &
         100.0_RealK,  230.0_RealK,  400.0_RealK,  570.0_RealK,  610.0_RealK, &
         650.0_RealK,  690.0_RealK,  730.0_RealK,  760.0_RealK,  790.0_RealK, &
         850.0_RealK,  900.0_RealK,  970.0_RealK, 1010.0_RealK, 1070.0_RealK, &
        1210.0_RealK, 1330.0_RealK, 1450.0_RealK, 1590.0_RealK, 1720.0_RealK, &
        1900.0_RealK, 2020.0_RealK, 2080.0_RealK, 2130.0_RealK, 2210.0_RealK, &
        2250.0_RealK, 2300.0_RealK, 2380.0_RealK, 2390.0_RealK, 2500.0_RealK /)
!       Upper band limits [m-1] used to define long-wave albedos

! Tabulated long-wave albedos. The data is from Valdar Oinas' gcmtable.f code,
! which was used to calculate the thermal albedo arrays AOCEAN and AGSIDV
! above. The albedos below are interpolated onto the bands in radiation.
  REAL(RealK), PARAMETER :: &
      alb_sea_lw(n_band_alb_lw) = (/ &
        0.150_RealK, 0.140_RealK, 0.120_RealK, 0.110_RealK, 0.100_RealK, &
        0.100_RealK, 0.097_RealK, 0.091_RealK, 0.081_RealK, 0.070_RealK, &
        0.053_RealK, 0.040_RealK, 0.036_RealK, 0.041_RealK, 0.045_RealK, &
        0.052_RealK, 0.056_RealK, 0.058_RealK, 0.059_RealK, 0.061_RealK, &
        0.062_RealK, 0.063_RealK, 0.063_RealK, 0.064_RealK, 0.064_RealK, &
        0.064_RealK, 0.064_RealK, 0.065_RealK, 0.065_RealK, 0.065_RealK /), &
!       Long-wave ocean albedos
      alb_soil_lw(n_band_alb_lw) = (/ &
        0.002_RealK, 0.006_RealK, 0.036_RealK, 0.130_RealK, 0.063_RealK, &
        0.042_RealK, 0.030_RealK, 0.033_RealK, 0.041_RealK, 0.045_RealK, &
        0.040_RealK, 0.045_RealK, 0.076_RealK, 0.120_RealK, 0.180_RealK, &
        0.190_RealK, 0.025_RealK, 0.021_RealK, 0.033_RealK, 0.039_RealK, &
        0.043_RealK, 0.030_RealK, 0.040_RealK, 0.051_RealK, 0.070_RealK, &
        0.086_RealK, 0.095_RealK, 0.110_RealK, 0.120_RealK, 0.130_RealK /), &
!       Long-wave soil albedos
      alb_snow_lw(n_band_alb_lw) = (/ &
        0.280_RealK, 0.059_RealK, 0.082_RealK, 0.120_RealK, 0.068_RealK, &
        0.047_RealK, 0.031_RealK, 0.025_RealK, 0.022_RealK, 0.020_RealK, &
        0.017_RealK, 0.014_RealK, 0.013_RealK, 0.014_RealK, 0.021_RealK, &
        0.020_RealK, 0.017_RealK, 0.012_RealK, 0.009_RealK, 0.008_RealK, &
        0.017_RealK, 0.026_RealK, 0.025_RealK, 0.022_RealK, 0.016_RealK, &
        0.011_RealK, 0.009_RealK, 0.006_RealK, 0.004_RealK, 0.0006_RealK /), &
!       Long-wave snow albedos
      alb_ice_lw(n_band_alb_lw) = (/ &
        0.280_RealK, 0.059_RealK, 0.082_RealK, 0.120_RealK, 0.068_RealK, &
        0.047_RealK, 0.031_RealK, 0.025_RealK, 0.022_RealK, 0.020_RealK, &
        0.017_RealK, 0.014_RealK, 0.013_RealK, 0.014_RealK, 0.021_RealK, &
        0.020_RealK, 0.017_RealK, 0.012_RealK, 0.009_RealK, 0.008_RealK, &
        0.017_RealK, 0.026_RealK, 0.025_RealK, 0.022_RealK, 0.016_RealK, &
        0.011_RealK, 0.009_RealK, 0.006_RealK, 0.004_RealK, 0.0006_RealK /), &
!       Long-wave ice albedos
      alb_vegetation_lw(n_band_alb_lw) = (/ &
        0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, &
        0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, &
        0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, &
        0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, &
        0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK, 0.0_RealK /)
!       Long-wave vegetation albedos

! Number of vegetation types (NV)
  INTEGER, PARAMETER :: n_veg = 12

! Vegetation depth mask by type in m (VTMASK)
  REAL(RealK), PARAMETER :: veg_depth(n_veg) = (/ &
          0.1_RealK, & ! BSAND
          0.2_RealK, & ! TNDRA
          0.2_RealK, & ! GRASS
          0.5_RealK, & ! SHRUB
          2.0_RealK, & ! TREES
          5.0_RealK, & ! DECID
         10.0_RealK, & ! EVERG
         25.0_RealK, & ! RAINF
          0.2_RealK, & ! CROPS
          0.1_RealK, & ! BDIRT
      1.0E-05_RealK, & ! ALGAE
          0.2_RealK /) ! GRAC4

! Sea ice albedo scheme identifiers
  INTEGER, PARAMETER, PUBLIC :: &
      ip_sea_ice_alb_Hansen = 0, &
!       Scheme by Schramm and J. Hansen's (6 spectral bands)
      ip_sea_ice_alb_Lacis  = 1
!       Scheme by Lacis (original version, 6-band)

! Characteristic amount of snow on sea ice (tuning parameter for Lacis sea ice
! albedo scheme (DMOICE)
  REAL(RealK), PARAMETER :: char_amount_snow = 10.0_RealK

!-------------------------------------------------------------------------------
! Short-wave albedo data
!-------------------------------------------------------------------------------

! Albedo bands
  INTEGER, PARAMETER :: n_band_alb_sw = 6
  REAL(RealK), PARAMETER :: &
      alb_band_min_sw(n_band_alb_sw) = 1.0E+09_RealK/(/ &
        4000.0_RealK, 2200.0_RealK, 1500.0_RealK, &
        1250.0_RealK,  860.0_RealK,  770.0_RealK /), &
!       Lower band limits [m-1] used to define short-wave albedos
      alb_band_max_sw(n_band_alb_sw) = 1.0E+09_RealK/(/ &
        2200.0_RealK, 1500.0_RealK, 1250.0_RealK, &
         860.0_RealK,  770.0_RealK, 300.0_RealK /)
!       Upper band limits [m-1] used to define short-wave albedos

  REAL(RealK), ALLOCATABLE :: &
    alb_wgt_sw(:,:)
!     Weight of GISS-band short-wave albedos in each SOCRATES band

!-------------------------------------------------------------------------------
! End albedo data
!-------------------------------------------------------------------------------

! Pointers to albedo weightings
  INTEGER, PARAMETER :: &
      ip_weight_planck  = 1, &
!       Planck function at surface temperature
      ip_weight_solar   = 2, &
!       Solar irradiance at TOA
      ip_weight_uniform = 3
!       Uniform weighting

! Select weighting schemes. To use the old weighting scheme set the
! respective control variable to ip_weight_uniform.
  INTEGER :: &
      i_alb_weight_lw = ip_weight_planck, &
!       Long-wave weighting scheme
      i_alb_weight_sw = ip_weight_solar
!       Short-wave weighting scheme

CONTAINS

! Initialisation routine for albedos
SUBROUTINE init_planet_alb(sp_sw, sol)

  USE def_spectrum,  ONLY: StrSpecData
  USE def_planetstr, ONLY: StrSolarSpec

! Input variables
  TYPE(StrSpecData), INTENT(IN) :: &
      sp_sw
!       Short-wave information
  TYPE(StrSolarSpec), INTENT(IN) :: &
      sol
!       Solar spectrum

! Allocate and calculate short-wave albedo weights
  ALLOCATE(alb_wgt_sw(n_band_alb_sw, sp_sw%basic%n_band))
  CALL calc_alb_wgt(n_band_alb_sw, alb_band_min_sw, alb_band_max_sw, &
    sp_sw, i_alb_weight_sw, 0.0_RealK, sol, alb_wgt_sw)

END SUBROUTINE init_planet_alb


! Subroutine to set long-wave surface albedos and effective temperature
! for use with SOCRATES
SUBROUTINE set_surf_prop_lw(i_sea_ice_alb_scheme, &
    wind_speed_mag, frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
    t_rad_sea, t_rad_land, t_rad_sea_ice, t_rad_land_ice, &
    soil_wetness, frac_veg_tbl, frac_land_snow_tbl, snow_depth_land_tbl, &
    snow_depth_sea_ice, snow_amnt_sea_lake, sp, sol, &
    l_tile, index_tile, bound, n_profile, n_snow_land, n_tile_type)

  USE rad_pcf
  USE def_spectrum,  ONLY: StrSpecData
  USE def_planetstr, ONLY: StrSolarSpec
  USE def_bound,     ONLY: StrBound

! Input variables
  INTEGER, INTENT(IN) :: &
      n_profile, &
!       Number of profiles in SOCRATES (= 1)
      n_snow_land, &
!       Number of elements in snow depth array (SNOWD)
      n_tile_type, &
!       Number of tile types allocated
      i_sea_ice_alb_scheme
!       Sea ice albedo scheme (KSIALB)
  REAL(RealK), INTENT(IN) :: &
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
  TYPE(StrSpecData), INTENT(IN) :: &
      sp
!       Spectral information
  TYPE(StrSolarSpec), INTENT(IN) :: &
      sol
!       Solar spectrum

! Input/output variables
  LOGICAL, INTENT(INOUT) :: &
      l_tile
!       Logical for tiling of current grid box
  TYPE(StrBound), INTENT(INOUT) :: &
      bound
!       Boundary structure

! Output variables
  INTEGER, INTENT(OUT) :: &
      index_tile(n_tile_type)
!       The index of tiles of given types

! Local variables
  INTEGER :: &
      iv
!       Loop variable
  REAL(RealK) :: &
      x, av, bv, &
!       Dummy variables used to calculate sea albedo
      frac_land_soil, &
!       Fraction of land consisting of soil (DSFRAC)
      frac_land_veg, &
!       Fraction of land consisting of vegetation (VGFRAC)
      frac_land_snow, &
!       Fraction of land that is covered by snow corrected for masking
!       (1.0 - EXPSNE)
      frac_snow_cover_sea_ice, &
!       Snow cover fraction on sea ice (patchy = 1.0 - EXPSNO)
      frac_bare_sea_ice
!       Bare sea ice fraction (EXPSNO = 1.0 - patchy)
  REAL(RealK), ALLOCATABLE :: &
      alb_rad_band(:)
!       Calculated albedos in radiation bands

  ALLOCATE(alb_rad_band(sp%basic%n_band))

! Turn tiling off if not requested or needed
  CALL set_tiling(frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
      l_tile, bound, index_tile, n_tile_type)

! Zero the irrelevant direct albedo
  bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = 0.0_RealK
  IF (l_tile) THEN
    bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_dir, &
        1:bound%n_tile, 1:sp%basic%n_band) = 0.0_RealK
  END IF

! Zero all bound arrays initially
  bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = 0.0_RealK
  IF (l_tile) THEN
    bound%frac_tile(bound%n_point_tile, 1:bound%n_tile) = 0.0_RealK
    bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
        1:bound%n_tile, 1:sp%basic%n_band) = 0.0_RealK
    bound%t_tile(bound%n_point_tile, 1:bound%n_tile) = 0.0_RealK
  END IF
  bound%t_ground = 0.0_RealK

!-------------------------------------------------------------------------------
! Sea
!-------------------------------------------------------------------------------

  IF (frac_sea > 0.0_RealK) THEN

!   Calculate sea albedo from parametrization
#ifdef DEBUG_RADIATION
!   Sea albedo manually set to zero
    alb_rad_band = 0.0_RealK
#else
    x = 1.0_RealK/(1.0_RealK + wind_speed_mag)
    av = -0.0147087_RealK*x**2 + 0.0292266_RealK*x - 0.0081079_RealK
    bv = (1.01673_RealK - 0.0083652_RealK*wind_speed_mag)
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_sea_lw, sp, &
        i_alb_weight_lw, t_rad_sea, sol, alb_rad_band)
    alb_rad_band = av + bv*alb_rad_band
#endif
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*frac_sea

  ! Add sea temperature to ground temperature
    bound%t_ground = bound%t_ground + frac_sea*t_rad_sea

  ! Add sea tile data
    IF (l_tile) THEN
      bound%frac_tile(bound%n_point_tile, index_tile(ip_ocean_tile)) = frac_sea
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_ocean_tile), 1:sp%basic%n_band) = alb_rad_band
      bound%t_tile(bound%n_point_tile, index_tile(ip_ocean_tile)) = t_rad_sea
    END IF
  
  END IF

!-------------------------------------------------------------------------------
! Land (soil and vegetation)
!-------------------------------------------------------------------------------

  IF (frac_land > 0.0_RealK) THEN

!   In the following code when computing albedo for snow covered land we use
!   frac_land_snow_tbl(1:2) which is the snow fraction cover for bare/vegetated
!   soil. It is computed in GHY_DRV.f in accordance with the surface topography.
!   The final snow cover is minimum of frac_land_snow_tbl and the snow fraction
!   obtained using the vegetation masking.
    IF (snow_depth_land_tbl(1) + snow_depth_land_tbl(2) > 1.0E-03_RealK) THEN

!     Correct soil fraction
      frac_land_soil = frac_veg_tbl(1)*(1.0_RealK - frac_land_snow_tbl(1)* &
          (1.0_RealK - EXP(-snow_depth_land_tbl(1)/veg_depth(1)))) + &
          frac_veg_tbl(10)*(1.0_RealK - frac_land_snow_tbl(1)* &
          (1.0_RealK - EXP(-snow_depth_land_tbl(1)/veg_depth(10))))

!     Correct vegetation fraction
      frac_land_veg = 0.0_RealK
      DO iv = 2,n_veg
        IF (iv == 10 .OR. iv == 11) CYCLE ! Skip dirt (soil) and algae
        frac_land_veg = frac_land_veg + &
            frac_veg_tbl(iv)*(1.0_RealK - frac_land_snow_tbl(2)* &
            (1.0_RealK - EXP(-snow_depth_land_tbl(2)/veg_depth(iv))))
      END DO

!     The remaning land is covered by snow
      frac_land_snow = 1.0 - (frac_land_soil + frac_land_veg)

    ELSE

!     No correction due to snow necessary
      frac_land_soil = frac_veg_tbl(1) + frac_veg_tbl(10)
      frac_land_veg = 1.0_RealK - frac_land_soil
      frac_land_snow = 0.0_RealK

    END IF

!   Calculate and add soil albedos at radiation bands
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_soil_lw, sp, &
        i_alb_weight_lw, t_rad_land, sol, alb_rad_band)
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*(1.0_RealK - soil_wetness)*frac_land_soil*frac_land

!   Add albedo data for soil to land tile
    IF (l_tile) THEN
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
        index_tile(ip_land_tile), 1:sp%basic%n_band) = &
        alb_rad_band*(1.0_RealK - soil_wetness)*frac_land_soil
    END IF

!   Calculate and add vegetation albedos at radiation bands
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_vegetation_lw, sp, &
        i_alb_weight_lw, t_rad_land, sol, alb_rad_band)
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*frac_land_veg*frac_land

!   Add albedo data for vegetation to land tile
    IF (l_tile) THEN
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_land_tile), 1:sp%basic%n_band) = &
          bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_land_tile), 1:sp%basic%n_band) + &
          alb_rad_band*frac_land_veg
    END IF

!   Calculate and add snow albedos at radiation bands
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_snow_lw, sp, &
        i_alb_weight_lw, t_rad_land, sol, alb_rad_band)
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*frac_land_snow*frac_land

!   Add soil and vegetation temperature to ground temperature
    bound%t_ground = bound%t_ground + frac_land*t_rad_land

!   Add remaining land tile data
    IF (l_tile) THEN
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_land_tile), 1:sp%basic%n_band) = &
          bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_land_tile), 1:sp%basic%n_band) + &
          alb_rad_band*frac_land_snow
      bound%frac_tile(bound%n_point_tile, index_tile(ip_land_tile)) = frac_land
      bound%t_tile(bound%n_point_tile, index_tile(ip_land_tile)) = t_rad_land
    END IF
  
  END IF

!-------------------------------------------------------------------------------
! Sea ice
!-------------------------------------------------------------------------------

  IF (frac_sea_ice > 0.0_RealK) THEN

!   Calculate snow cover fraction
    IF (i_sea_ice_alb_scheme == ip_sea_ice_alb_Hansen) THEN

      IF (snow_depth_sea_ice > 0.0_RealK) THEN
!       Snow cover fraction is < 1.0 if snow depth < 0.1 m)
        IF (snow_depth_sea_ice < 0.1_RealK) THEN
          frac_snow_cover_sea_ice = snow_depth_sea_ice/0.1_RealK
        ELSE
          frac_snow_cover_sea_ice = 1.0_RealK
        END IF
      ELSE
!       Ice is bare with no snow
        frac_snow_cover_sea_ice = 0.0_RealK
      END IF
      frac_bare_sea_ice = 1.0_RealK - frac_snow_cover_sea_ice

    ELSE IF (i_sea_ice_alb_scheme == ip_sea_ice_alb_Lacis) THEN

!     Use exponential decay using characteristic snow amount
      frac_bare_sea_ice = EXP(-snow_amnt_sea_lake/char_amount_snow)
      frac_snow_cover_sea_ice = 1.0_RealK - frac_bare_sea_ice

    ELSE
      CALL stop_model('set_surf_prop_lw: Selected sea ice albedo scheme ' // &
          'not defined', 255)
    END IF

!   Calculate and add snow covered ice albedos at radiation bands
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_snow_lw, sp, &
        i_alb_weight_lw, t_rad_sea_ice, sol, alb_rad_band)
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*frac_snow_cover_sea_ice*frac_sea_ice

!   Add albedo data for soil to land tile
    IF (l_tile) THEN
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_seaice_tile), 1:sp%basic%n_band) = &
          alb_rad_band*frac_snow_cover_sea_ice
    END IF

!   Calculate and add bare ice albedos at radiation bands
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_ice_lw, sp, &
        i_alb_weight_lw, t_rad_sea_ice, sol, alb_rad_band)
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*frac_bare_sea_ice*frac_sea_ice

!   Add sea ice temperature to ground temperature
    bound%t_ground = bound%t_ground + frac_sea_ice*t_rad_sea_ice

!   Add remaining sea ice tile data
    IF (l_tile) THEN
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_seaice_tile), 1:sp%basic%n_band) = &
          bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_seaice_tile), 1:sp%basic%n_band) + &
          alb_rad_band*frac_bare_sea_ice
      bound%frac_tile(bound%n_point_tile, index_tile(ip_seaice_tile)) = &
          frac_sea_ice
      bound%t_tile(bound%n_point_tile, index_tile(ip_seaice_tile)) = &
          t_rad_sea_ice
    END IF

  END IF

!-------------------------------------------------------------------------------
! Land ice
!-------------------------------------------------------------------------------

  IF (frac_land_ice > 0.0_RealK) THEN

!   Calculate and add land ice albedos at radiation bands
    CALL get_surf_alb(n_band_alb_lw, alb_band_min_lw, &
        alb_band_max_lw, alb_ice_lw, sp, &
        i_alb_weight_lw, t_rad_land_ice, sol, alb_rad_band)
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:sp%basic%n_band) + &
        alb_rad_band*frac_land_ice

!   Add sea ice temperature to ground temperature
    bound%t_ground = bound%t_ground + frac_land_ice*t_rad_land_ice

!   Add land ice tile data
    IF (l_tile) THEN
      bound%frac_tile(bound%n_point_tile, index_tile(ip_landice_tile)) = &
          frac_land_ice
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_landice_tile), 1:sp%basic%n_band) = alb_rad_band
      bound%t_tile(bound%n_point_tile, index_tile(ip_landice_tile)) = &
          t_rad_land_ice
    END IF

  END IF

  DEALLOCATE(alb_rad_band)

END SUBROUTINE set_surf_prop_lw


SUBROUTINE set_surf_prop_sw(frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
    PRNB, PRNX, l_tile, index_tile, bound, n_profile, n_tile_type, n_rad_band)

  USE rad_pcf
  USE def_spectrum, ONLY: StrSpecData
  USE def_bound,    ONLY: StrBound

! Input variables
  INTEGER, INTENT(IN) :: &
      n_profile, &
!       Number of profiles in SOCRATES (= 1)
      n_tile_type, &
!       Number of tile types allocated
      n_rad_band
!       Number of radiation bands
  REAL(RealK), INTENT(IN) :: &
      frac_sea, &
!       Sea fraction in grid box (POCEAN)
      frac_land, &
!       Land (soil and vegetation) fraction in grid box (PEARTH)
      frac_sea_ice, &
!       Sea ice fraction in grid box (POICE)
      frac_land_ice, &
!       Land ice fraction in grid box (PLICE)
      PRNB(6,4), &
!       Diffuse short-wave albedos
      PRNX(6,4)
!       Direct short-wave albedos

! Input/output variables
  LOGICAL, INTENT(INOUT) :: &
      l_tile
!       Logical for tiling of current grid box
  TYPE(StrBound), INTENT(INOUT) :: &
      bound
!       Boundary structure

! Output variables
  INTEGER, INTENT(OUT) :: &
      index_tile(n_tile_type)
!       The index of tiles of given types

! Local variables
  REAL(RealK), ALLOCATABLE :: &
      alb_diff_rad_band(:), &
!       Calculated diffuse albedos in radiation bands
      alb_dir_rad_band(:)
!       Calculated direct albedos in radiation bands

  ALLOCATE( &
      alb_diff_rad_band(n_rad_band), &
      alb_dir_rad_band(n_rad_band))

! Turn tiling off if not requested or needed
  CALL set_tiling(frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
      l_tile, bound, index_tile, n_tile_type)

! Zero the albedos initially
  bound%rho_alb(n_profile, ip_surf_alb_dir,  1:n_rad_band) = 0.0_RealK
  bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) = 0.0_RealK
  IF (l_tile) THEN
    bound%frac_tile(bound%n_point_tile, 1:n_tile_type) = 0.0_RealK
    bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          1:n_tile_type, 1:n_rad_band) = 0.0_RealK
    bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_dir, &
          1:n_tile_type, 1:n_rad_band) = 0.0_RealK
  END IF

!-------------------------------------------------------------------------------
! Sea
!-------------------------------------------------------------------------------

  IF (frac_sea > 0.0_RealK) THEN

!   Calculate short-wave diffuse and direct albedos at radiation bands
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNB(:,1), alb_wgt_sw, &
        alb_diff_rad_band)
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNX(:,1), alb_wgt_sw, &
        alb_dir_rad_band)

!   Add contribution to total albedos
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) + &
        alb_diff_rad_band*frac_sea
    bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) + &
        alb_dir_rad_band*frac_sea

!   Add sea tile data
    IF (l_tile) THEN
      bound%frac_tile(bound%n_point_tile, index_tile(ip_ocean_tile)) = frac_sea
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_ocean_tile), 1:n_rad_band) = alb_diff_rad_band
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_dir, &
          index_tile(ip_ocean_tile), 1:n_rad_band) = alb_dir_rad_band
    END IF

  END IF

!-------------------------------------------------------------------------------
! Land (soil and vegetation)
!-------------------------------------------------------------------------------

  IF (frac_land > 0.0_RealK) THEN

!   Calculate short-wave diffuse and direct albedos at radiation bands
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNB(:,2), alb_wgt_sw, &
        alb_diff_rad_band)
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNX(:,2), alb_wgt_sw, &
        alb_dir_rad_band)

!   Add contribution to total albedos
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) + &
        alb_diff_rad_band*frac_land
    bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) + &
        alb_dir_rad_band*frac_land

!   Add sea tile data
    IF (l_tile) THEN
      bound%frac_tile(bound%n_point_tile, index_tile(ip_land_tile)) = frac_land
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_land_tile), 1:n_rad_band) = alb_diff_rad_band
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_dir, &
          index_tile(ip_land_tile), 1:n_rad_band) = alb_dir_rad_band
    END IF
  
  END IF

!-------------------------------------------------------------------------------
! Sea ice
!-------------------------------------------------------------------------------

  IF (frac_sea_ice > 0.0_RealK) THEN

!   Calculate short-wave diffuse and direct albedos at radiation bands
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNB(:,3), alb_wgt_sw, &
        alb_diff_rad_band)
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNX(:,3), alb_wgt_sw, &
        alb_dir_rad_band)

!   Add contribution to total albedos
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) + &
        alb_diff_rad_band*frac_sea_ice
    bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) + &
        alb_dir_rad_band*frac_sea_ice

!   Add sea tile data
    IF (l_tile) THEN
      bound%frac_tile(bound%n_point_tile, index_tile(ip_seaice_tile)) = &
          frac_sea_ice
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_seaice_tile), 1:n_rad_band) = alb_diff_rad_band
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_dir, &
          index_tile(ip_seaice_tile), 1:n_rad_band) = alb_dir_rad_band
    END IF

  END IF

!-------------------------------------------------------------------------------
! Land ice
!-------------------------------------------------------------------------------

  IF (frac_land_ice > 0.0_RealK) THEN

!   Calculate short-wave diffuse and direct albedos at radiation bands
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNB(:,4), alb_wgt_sw, &
        alb_diff_rad_band)
    CALL upd_surf_alb(n_band_alb_sw, n_rad_band, PRNX(:,4), alb_wgt_sw, &
        alb_dir_rad_band)

!   Add contribution to total albedos
    bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_diff, 1:n_rad_band) + &
        alb_diff_rad_band*frac_land_ice
    bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) = &
        bound%rho_alb(n_profile, ip_surf_alb_dir, 1:n_rad_band) + &
        alb_dir_rad_band*frac_land_ice

!   Add sea tile data
    IF (l_tile) THEN
      bound%frac_tile(bound%n_point_tile, index_tile(ip_landice_tile)) = &
          frac_land_ice
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_diff, &
          index_tile(ip_landice_tile), 1:n_rad_band) = alb_diff_rad_band
      bound%rho_alb_tile(bound%n_point_tile, ip_surf_alb_dir, &
          index_tile(ip_landice_tile), 1:n_rad_band) = alb_dir_rad_band
    END IF

  END IF

  DEALLOCATE(alb_diff_rad_band, alb_dir_rad_band)

END SUBROUTINE set_surf_prop_sw

END MODULE planet_alb

!-------------------------------------------------------------------------------
! General routines and function used in albedo calculation
!-------------------------------------------------------------------------------

SUBROUTINE set_tiling(frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
      l_tile, bound, index_tile, n_tile_type)

  USE realtype_rd, ONLY: RealK
  USE def_bound,   ONLY: StrBound
  USE rad_pcf

  IMPLICIT NONE

! Input variables
  INTEGER, INTENT(IN) :: &
      n_tile_type
!       Number of tile types allocated
  REAL(RealK), INTENT(IN) :: &
      frac_sea, &
!       Sea fraction in grid box (POCEAN)
      frac_land, &
!       Land (soil and vegetation) fraction in grid box (PEARTH)
      frac_sea_ice, &
!       Sea ice fraction in grid box (POICE)
      frac_land_ice
!       Land ice fraction in grid box (PLICE)

! Input/output variables
  LOGICAL, INTENT(INOUT) :: &
      l_tile
!       Logical for tiling of current grid box
  TYPE(StrBound), INTENT(INOUT) :: &
      bound
!       Boundary properties

! Output variables
  INTEGER, INTENT(OUT) :: &
      index_tile(n_tile_type)
!       The index of tiles of given types

! Switch on tiling in current grid box if tiling is on and it contains more than
! one surface type
  IF (l_tile .AND. &
      ((frac_sea > 0.0_RealK .AND. frac_sea < 1.0_RealK) .OR. &
       (frac_land > 0.0_RealK .AND. frac_land < 1.0_RealK) .OR. &
       (frac_sea_ice > 0.0_RealK .AND. frac_sea_ice < 1.0_RealK) .OR. &
       (frac_land_ice > 0.0_RealK .AND. frac_land_ice < 1.0_RealK))) THEN

!   Only one profile to tile
    bound%n_point_tile = 1
    bound%list_tile(bound%n_point_tile) = 1

!   Four different tile types
    bound%n_tile = 4

!   Initialise indices for tiles
    index_tile(ip_ocean_tile) = 1
    index_tile(ip_seaice_tile) = 2
    index_tile(ip_land_tile) = 3
    index_tile(ip_landice_tile) = 4

! Otherwise tiling is off for the current gridbox
  ELSE
    l_tile = .FALSE.
    bound%n_point_tile = 0
  END IF

END SUBROUTINE set_tiling


! Get surface albedo in all radiation bands. Calculates new albedo weightings.
SUBROUTINE get_surf_alb(n_alb_band, alb_band_min, alb_band_max, &
    alb, sp, i_weight, temp, sol, alb_rad_band)

  USE realtype_rd,   ONLY: RealK
  USE def_spectrum,  ONLY: StrSpecData
  USE def_planetstr, ONLY: StrSolarSpec

  IMPLICIT NONE

  INTEGER, INTENT(IN) :: &
      n_alb_band, &
!       Number of albedo bands
      i_weight
!       Weighting scheme
  REAL(RealK), INTENT(IN) :: &
      alb_band_min(n_alb_band), &
!       Lower limit of albedo bands
      alb_band_max(n_alb_band), &
!       Upper limit of albedo bands
      alb(n_alb_band), &
!       Albedos for each albedo band
      temp
!       Weighting temperature
  TYPE(StrSpecData), INTENT(IN) :: sp
!       Spectral information
  TYPE(StrSolarSpec), INTENT(IN) :: sol
!       Solar spectrum
  REAL(RealK), INTENT(OUT) :: alb_rad_band(sp%basic%n_band)
!       Calculated average albedos in each radiation band

! Local variables
  INTEGER :: &
      ib
!       Loop index
  REAL(RealK) :: &
      alb_wgt(n_alb_band, sp%basic%n_band)
!       Weight of each albedo band for each radiation band

! Calculate new albedo weights
  CALL calc_alb_wgt(n_alb_band, alb_band_min, alb_band_max, &
      sp, i_weight, temp, sol, alb_wgt)

! Update the albedos in each radiation band
  CALL upd_surf_alb(n_alb_band, sp%basic%n_band, alb, alb_wgt, alb_rad_band)

END SUBROUTINE get_surf_alb


! Calculate surface albedo using provided albedo weightings.
SUBROUTINE upd_surf_alb(n_alb_band, n_rad_band, alb, alb_wgt, alb_rad_band)

  USE realtype_rd,  ONLY: RealK
  USE def_spectrum, ONLY: StrSpecData

  IMPLICIT NONE

  INTEGER, INTENT(IN) :: &
      n_alb_band, &
!       Number of albedo bands
      n_rad_band
!       Number of radiation bands
  REAL(RealK), INTENT(IN) :: &
      alb(n_alb_band), &
!       Albedos for each albedo band
      alb_wgt(n_alb_band, n_rad_band)
!       Weight of each albedo band for each radiation band
  REAL(RealK), INTENT(OUT) :: alb_rad_band(n_rad_band)
!       Calculated average albedos in each radiation band

! Local variables
  INTEGER :: &
      ib
!       Loop index

! Calculate the albedo in each radiation band
  DO ib=1,n_rad_band
    alb_rad_band(ib) = SUM(alb*alb_wgt(:,ib))
  END DO

END SUBROUTINE upd_surf_alb


! Calculate weights of surface albedos in each albedo band for all
! radiation bands.
SUBROUTINE calc_alb_wgt(n_alb_band, alb_band_min, alb_band_max, &
    sp, i_weight, temp, sol, alb_wgt)

  USE realtype_rd,   ONLY: RealK
  USE def_spectrum,  ONLY: StrSpecData
  USE def_planetstr, ONLY: StrSolarSpec

  IMPLICIT NONE

  INTEGER, INTENT(IN) :: &
      n_alb_band, &
!       Number of albedo bands
      i_weight
!       Weighting scheme
  REAL(RealK), INTENT(IN) :: &
      alb_band_min(n_alb_band), &
!       Lower limit of albedo bands
      alb_band_max(n_alb_band), &
!       Upper limit of albedo bands
      temp
!       Temperature for Planckian weighting
  TYPE(StrSpecData), INTENT(IN) :: sp
!       Spectral information
  TYPE(StrSolarSpec), INTENT(IN) :: sol
!       Solar spectrum
  REAL(RealK), INTENT(OUT) :: &
      alb_wgt(n_alb_band, sp%basic%n_band)
!       Weights for each albedo band for all radiation bands

! Local variables
  INTEGER :: &
      ib, ibx, &
!       Loop indices
      i_band_exclude
!       Index of band to exclude
  REAL(RealK) :: &
      alb_wgt_ex(n_alb_band), &
!       Albedo weights for excluded bands
      alb_wgt_norm
!       Norm of albedo weights
  LOGICAL :: &
      l_short_edge, &
!       Flag for current band beging the short wavelength edge band
      l_long_edge
!       Flag for current band being the long wavelength edge band

! Calculate albedo weights of all radiation bands
  DO ib=1,sp%basic%n_band
!   Check if current band is an edge band
    l_short_edge = MINVAL(sp%basic%wavelength_short) == &
        sp%basic%wavelength_short(ib)
    l_long_edge = MAXVAL(sp%basic%wavelength_long) == &
        sp%basic%wavelength_long(ib)

!   Calculate albedo weights in this band
    CALL calc_alb_band_wgt(n_alb_band, alb_band_min, alb_band_max, &
        1.0_RealK/sp%basic%wavelength_long(ib), &
        1.0_RealK/sp%basic%wavelength_short(ib), &
        l_short_edge, l_long_edge, &
        i_weight, temp, sol, alb_wgt(:,ib))
  END DO

! Subtract weights of excluded bands
  DO ib=1,sp%basic%n_band
    DO ibx=1,sp%basic%n_band_exclude(ib)
      i_band_exclude = sp%basic%index_exclude(ibx,ib)
      alb_wgt(:,ib) = alb_wgt(:,ib) -  alb_wgt(:,i_band_exclude)
    END DO
  END DO

! Normalise weights
  DO ib=1,sp%basic%n_band
    alb_wgt_norm = SUM(alb_wgt(:,ib))
    IF (alb_wgt_norm < 1.0E+10*TINY(alb_wgt_norm)) THEN
!     If the norm is near zero there is a negligible amount of solar radiation
!     in this band the albedo can be set to zero.
      alb_wgt(:,ib) = 0.0_RealK
    ELSE
      alb_wgt(:,ib) = alb_wgt(:,ib)/SUM(alb_wgt(:,ib))
    END IF
  END DO

END SUBROUTINE calc_alb_wgt


! This function calculates the weight for the albedo in each GISS radiation
! band for a given wavenumber range.
SUBROUTINE calc_alb_band_wgt(n_alb_band, alb_band_min, alb_band_max, &
    rad_band_min, rad_band_max, l_short_edge, l_long_edge, &
    i_weight, temp, sol, alb_band_wgt)

  USE realtype_rd,   ONLY: RealK
  USE def_planetstr, ONLY: StrSolarSpec

  IMPLICIT NONE

! Input variables
  INTEGER, INTENT(IN) :: &
      n_alb_band, &
!     Number of albedo bands
      i_weight
!       Weighting to be used
  REAL(RealK), INTENT(IN) :: &
      alb_band_min(n_alb_band), &
!       Lower limit of albedo bands
      alb_band_max(n_alb_band), &
!       Upper limit of albedo bands
      rad_band_min, &
!       Lower limit for radiation band
      rad_band_max, &
!       Upper limit for radiation band
      temp
!       Temperature for Planckian weighting
  LOGICAL, INTENT(IN) :: &
      l_short_edge, &
!       Flag for current band beging the short wavelength edge band
      l_long_edge
!       Flag for current band being the long wavelength edge band
  TYPE(StrSolarSpec), INTENT(IN) :: &
      sol
!       Solar spectrum

! Output variables
  REAL(RealK), INTENT(OUT) :: &
      alb_band_wgt(n_alb_band)
!       Weight of each albedo band for current radiation band

! Local variables
  INTEGER :: &
      ind_1, &
!       Index of lower albedo band encapsulating the radiation band
      ind_2, &
!       Index of the upper albedo band encapsulating the radiation band
      i
!       Loop index

! External functions
  REAL(RealK), EXTERNAL :: &
      integ_wgt
!       Calculates integral of weighting function

  alb_band_wgt = 0.0_RealK

! Check if both band limits are outside albedo range
  IF (rad_band_min <= alb_band_min(1) .AND. &
      rad_band_max <= alb_band_min(1)) THEN
    alb_band_wgt(1) = integ_wgt(i_weight, temp, sol, &
        rad_band_min, rad_band_max, l_short_edge, l_long_edge)
    RETURN
  ELSE IF (rad_band_min >= alb_band_max(n_alb_band) .AND. &
           rad_band_max >= alb_band_max(n_alb_band)) THEN
    alb_band_wgt(n_alb_band) = integ_wgt(i_weight, temp, sol, &
        rad_band_min, rad_band_max, l_short_edge, l_long_edge)
    RETURN
  END IF

! Find the albedo band indices encapsulating the radiation band. Explicitly
! check if radiation band is outside range as MINLOC cannot be trusted to
! return zero in these cases with ifort.
  IF (ANY(rad_band_min >= alb_band_min)) THEN
    ind_1 = MINLOC(ABS(rad_band_min - alb_band_min), 1, &
        rad_band_min >= alb_band_min)
  ELSE
    ind_1 = 0
  END IF
  IF (ANY(rad_band_max <= alb_band_max)) THEN
    ind_2 = MINLOC(ABS(rad_band_max - alb_band_max), 1, &
        rad_band_max <= alb_band_max)
  ELSE
    ind_2 = 0
  END IF

! Check if the radiation band is within one albedo band
  IF (ind_1 /= 0 .AND. ind_1 == ind_2) THEN
!   Calculate albedo area within the radiation band
    alb_band_wgt(ind_1) = integ_wgt(i_weight, temp, sol, &
        rad_band_min, rad_band_max, l_short_edge, l_long_edge)
    RETURN
  END IF

! Check if either albedo band index is outside range (both will not be)
  alb_band_wgt = 0.0_RealK
  IF (ind_1 == 0) THEN
!   Add weight outside albedo bands to corresponding edge albedo band
    alb_band_wgt(1) = alb_band_wgt(1) + &
        integ_wgt(i_weight, temp, sol, rad_band_min, alb_band_min(1), &
        .FALSE., l_long_edge)
  ELSE
!   Add weight contribution from first albedo band
    alb_band_wgt(ind_1) = alb_band_wgt(ind_1) + &
        integ_wgt(i_weight, temp, sol, rad_band_min, alb_band_max(ind_1), &
        .FALSE., l_long_edge)
  END IF
  IF (ind_2 == 0) THEN
!   Add weight outside albedo bands to corresponding edge albedo band
    alb_band_wgt(n_alb_band) = alb_band_wgt(n_alb_band) + &
        integ_wgt(i_weight, temp, sol, alb_band_max(n_alb_band), rad_band_max, &
        l_short_edge, .FALSE.)
    ind_2 = n_alb_band + 1
  ELSE
!   Add weight contribution from last albedo band
    alb_band_wgt(ind_2) = alb_band_wgt(ind_2) + &
        integ_wgt(i_weight, temp, sol, alb_band_min(ind_2), rad_band_max, &
        l_short_edge, .FALSE.)
  END IF

! Sum up remaining albedo bands
  DO i = ind_1+1,ind_2-1
    alb_band_wgt(i) = alb_band_wgt(i) + &
        integ_wgt(i_weight, temp, sol, alb_band_min(i), alb_band_max(i), &
        .FALSE., .FALSE.)
  END DO

END SUBROUTINE calc_alb_band_wgt


! Function to integrate weighting function across a band
REAL(RealK) FUNCTION integ_wgt(i_weight, temp, sol, &
    nu_min, nu_max, l_short_edge, l_long_edge)

  USE realtype_rd,   ONLY: RealK
  USE def_planetstr, ONLY: StrSolarSpec
  USE planet_alb,    ONLY: ip_weight_planck, ip_weight_solar, ip_weight_uniform

  IMPLICIT NONE

! Input variables
  INTEGER, INTENT(IN) :: &
      i_weight
!       Weighting scheme
  REAL(RealK), INTENT(IN) :: &
      temp, &
!       Temperature for Planckian weighting
      nu_min, &
!       Lower integration limit
      nu_max
!       Upper integration limit
  LOGICAL, INTENT(IN) :: &
      l_short_edge, &
!       Flag for adding flux from shorter wavelengths beyond nu_max
      l_long_edge
!       Flag for adding flux from longer wavelengths lower than nu_min
  TYPE(StrSolarSpec), INTENT(IN) :: &
      sol
!       Solar spectrum

! External functions
  REAL(RealK), EXTERNAL :: &
      integ_planck, &
!       Calculates integral of Planck function
      integ_solar
!       Calculates integral of solar irradiance

  SELECT CASE (i_weight)

  CASE (ip_weight_planck)
!   Planck function evaluated at surface temperature.
    integ_wgt = integ_planck(temp, nu_min, nu_max)

  CASE (ip_weight_solar)
!   Solar irradiance at TOA
    integ_wgt = integ_solar(sol, nu_min, nu_max, l_short_edge, l_long_edge)

  CASE (ip_weight_uniform)
!   Uniform weight
    integ_wgt = nu_max - nu_min
    
  CASE DEFAULT
    CALL stop_model('Error in planet_alb: unknown weighting scheme.', 255)

  END SELECT

END FUNCTION integ_wgt


! Calculate integral of Planck function
REAL(RealK) FUNCTION integ_planck(temp, nu_min, nu_max)

  USE realtype_rd, ONLY: RealK
  USE constant,    ONLY: pi, k_boltzmann, h_planck, c_light

  IMPLICIT NONE

! Input variables
  REAL(RealK), INTENT(IN) :: &
      temp, &
!       Temperature at which to evaluate Planck function
      nu_min, &
!       Lower integration limit
      nu_max
!       Upper integration limit

! Local variables
  REAL(RealK), PARAMETER :: &
      alpha = k_boltzmann/(h_planck*c_light), &
!       Combinatino of physical constants
      q = 2.0_RealK*pi*h_planck*alpha**4*c_light**2, &
!       Combinatino of physical constants
      delta_u = 1.0E-01_RealK
!       Integration step
  INTEGER :: &
      i, &
!       Loop index
      n_point
!       Number of integration points
  REAL(RealK) :: &
      u_min, &
!       Lower integration limit
      u_max, &
!       Upper integration limit
      delta_u_use, &
!       Actual integration step
      u
!       Integration coordinate
  REAL(RealK), ALLOCATABLE :: &
      dummy(:)
!       Dummy integrand

! Find minimum and maximum values for U at given THETA
  u_min = nu_min/(alpha*temp)
  u_max = nu_max/(alpha*temp)
  n_point = MAX(NINT((u_max - u_min)/delta_u) + 1, 2)

! Ensure an integer number of integration steps in interval
  delta_u_use = (u_max - u_min)/(n_point - 1)

! Calculate integrand
  ALLOCATE(dummy(n_point))
  DO i=1,n_point 
    u = u_min + delta_u_use*(i - 1)
    dummy(i) = u**3*EXP(-u)/(1.0_RealK - EXP(-u))
  ENDDO

! Integration over the transformed variable u=(h_planck*nu)/(k_blotzmann*temp)
  integ_planck = q*temp**4* &
      0.5_RealK*delta_u_use*SUM(dummy(1:n_point-1) + dummy(2:n_point))

  DEALLOCATE(dummy)

END FUNCTION integ_planck


! Calculate integral of solar spectrum
REAL(RealK) FUNCTION integ_solar(sol, nu_min, nu_max, l_short_edge, l_long_edge)

  USE realtype_rd,   ONLY: RealK
  USE def_planetstr, ONLY: StrSolarSpec

  IMPLICIT NONE

! Input variables
  TYPE(StrSolarSpec), INTENT(IN) :: &
        sol
!     Solar spectrum
  REAL(RealK), INTENT(IN) :: &
      nu_min, &
!       Lower integration limit
      nu_max
!       Upper integration limit
  LOGICAL, INTENT(IN) :: &
      l_short_edge, &
!       Flag for adding flux from shorter wavelengths beyond nu_max
      l_long_edge
!       Flag for adding flux from longer wavelengths lower than nu_min

! Local variables
  INTEGER :: &
      i_begin, &
!       Index at which to begin integration
      i_end
!       Index at which to end integraiton
  REAL(RealK) :: &
      wavelength_long, &
!       Long wavelength integration limit
      wavelength_short
!       Short wavelength integration limit

! External functions
  REAL(RealK), EXTERNAL :: &
      rayleigh_jeans_tail
!       Calculates Rayleigh-Jeans tail of Planck function

! Calculate wavelength integration limits
  wavelength_long = 1.0_RealK/nu_min
  wavelength_short = 1.0_RealK/nu_max

! Check if entire band is outside range of solar spectrum
  IF (wavelength_long < sol%wavelength(1)) THEN
!   No flux at shorter wavelengths than defined by solar spectrum
    integ_solar = 0.0_RealK
    RETURN
  ELSE IF (wavelength_short > sol%wavelength(sol%n_points)) THEN
!   Entire band is in the Rayleigh-Jeans tail
    integ_solar = rayleigh_jeans_tail(sol, wavelength_short) - &
         rayleigh_jeans_tail(sol, wavelength_long)
    IF (l_long_edge) THEN
      integ_solar = integ_solar + rayleigh_jeans_tail(sol, wavelength_long)
    END IF
    RETURN
  END IF

! Find the indices encapsulating the integration limits
  i_begin = MINLOC(sol%wavelength - wavelength_short, 1, &
      sol%wavelength >= wavelength_short)
  i_end = MAXLOC(sol%wavelength - wavelength_long, 1, &
      sol%wavelength <= wavelength_long)

! Check if either albedo band index is outside range (both will not be)
  integ_solar = 0.0_RealK
  IF (.NOT.l_short_edge .AND. i_begin > 1) THEN
!   Add contribution between lower band limit and first integration point using
!   linear interpolation and trapezoidal rule
    integ_solar = integ_solar + &
        0.5_RealK*(sol%wavelength(i_begin) - wavelength_short)* &
        (sol%irrad(i_begin-1) + &
        (sol%irrad(i_begin) - sol%irrad(i_begin-1))/ &
        (sol%wavelength(i_begin) - sol%wavelength(i_begin-1))* &
        (wavelength_short - sol%wavelength(i_begin-1)) + &
        sol%irrad(i_begin))
  END IF
  IF (.NOT.l_long_edge) THEN
    IF (i_end == sol%n_points) THEN
!     Add contribution between end of solar data and end of band
      integ_solar = integ_solar + &
          rayleigh_jeans_tail(sol, sol%wavelength(sol%n_points)) - &
          rayleigh_jeans_tail(sol, wavelength_long)
    ELSE
!     Add contribution between upper band limit and last integration point
!     using linear interpolation and trapezoidal rule
      integ_solar = integ_solar + &
          0.5_RealK*(sol%wavelength(i_end+1) - wavelength_long)* &
          (sol%irrad(i_end) + &
          (sol%irrad(i_end+1) - sol%irrad(i_end))/ &
          (sol%wavelength(i_end+1) - sol%wavelength(i_end))* &
          (wavelength_long - sol%wavelength(i_end)) + &
          sol%irrad(i_end))
    END IF
  END IF

! Integrate over remaining wavelengths using the trapezoidal rule
  integ_solar = integ_solar + 0.5_RealK* &
      SUM((sol%irrad(i_begin:i_end-1) + sol%irrad(i_begin+1:i_end))* &
          (sol%wavelength(i_begin+1:i_end) - sol%wavelength(i_begin:i_end-1)))

! Add edge fluxes if required
  IF (l_short_edge) THEN
    integ_solar = integ_solar + 0.5_RealK* &
        SUM((sol%irrad(1:i_begin-1) + sol%irrad(2:i_begin))* &
            (sol%wavelength(2:i_begin) - sol%wavelength(1:i_begin-1)))
  END IF
  IF (l_long_edge) THEN
    integ_solar = integ_solar + 0.5_RealK* &
        SUM((sol%irrad(i_end:sol%n_points-1) + &
             sol%irrad(i_end+1:sol%n_points))* &
            (sol%wavelength(i_end+1:sol%n_points) - &
             sol%wavelength(i_end:sol%n_points-1))) + &
        rayleigh_jeans_tail(sol, sol%wavelength(sol%n_points))
  END IF

END FUNCTION integ_solar


! Calculate Rayleigh jeans tail
REAL(RealK) FUNCTION rayleigh_jeans_tail(sol, wavelength)

  USE realtype_rd,   ONLY: RealK
  USE constant,      ONLY: astronomical_unit, c_light, k_boltzmann
  USE def_planetstr, ONLY: StrSolarSpec

  IMPLICIT NONE

  TYPE(StrSolarSpec), INTENT(IN) :: &
      sol
!       Solar spectrum
  REAL(RealK), INTENT(IN) :: &
      wavelength
!       Wavelength at which to evaluate Rayleigh-Jeans tail

! Evaluate black-body flux
  rayleigh_jeans_tail = 2.0_RealK*c_light*k_boltzmann* &
     sol%t_effective/(3.0_RealK*wavelength**3)

! Scale to get the flux at 1 AU
  rayleigh_jeans_tail = rayleigh_jeans_tail* &
     (sol%radius/astronomical_unit)**2

END FUNCTION rayleigh_jeans_tail
